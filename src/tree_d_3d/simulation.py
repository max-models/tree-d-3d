"""Time integration of charged particles in a plane, with fields from the flat tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from methodcard import ConvergenceProblem, Rule, method

from tree_d_3d._profiling import _PROFILE_TAGS
from tree_d_3d._utils import _check, _device, _finite, _host
from tree_d_3d.bodies import ChargedBody
from tree_d_3d.constants import K_E
from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.kernels import _kernels, kernels_compiled
from tree_d_3d.quad import Quad

if TYPE_CHECKING:
    from simgeometry.shapes import Shape

#: ``method="auto"`` sums pairs directly up to this many charged particles.
DIRECT_THRESHOLD: int = 1000


_METHODS = ("auto", "tree", "direct", "fmm")

# name -> (axis, sign): a particle is beyond the wall when
# sign * (coordinate - wall) > 0.
_WALLS = {"x_min": (0, -1), "x_max": (0, 1), "y_min": (1, -1), "y_max": (1, 1)}


def _parse_walls(walls: Mapping[str, float]) -> list[tuple[str, int, int, float]]:
    unknown = set(walls) - set(_WALLS)
    if unknown:
        raise ValueError(
            f"Unknown absorbing walls {sorted(unknown)}; use {list(_WALLS)}"
        )
    parsed = [
        (name, *_WALLS[name], _finite(name, walls[name]))
        for name in _WALLS
        if name in walls
    ]
    for low, high in (("x_min", "x_max"), ("y_min", "y_max")):
        if low in walls and high in walls and not walls[low] < walls[high]:
            raise ValueError(f"{low} must be below {high}")
    return parsed


def _parse_objects(objects: Mapping[str, Shape]) -> dict[str, Shape]:
    for name, shape in objects.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Absorbing objects need nonempty string names")
        if name in _WALLS:
            raise ValueError(f"{name!r} is a wall name; name the object differently")
        if getattr(shape, "ndim", None) != 2 or not hasattr(shape, "contains_points"):
            raise ValueError(
                f"Absorbing object {name!r} must be a 2D simgeometry shape"
            )
    return dict(objects)


class AbsorbedParticles:
    """Particles removed at absorbing walls and objects, in removal order.

    Positions and velocities are taken just after the drift that crossed the
    wall or entered the object, so positions lie up to one step beyond the
    boundary; ``wall`` names the wall or object; ``time`` is the end of
    that step. ``position`` and ``velocity`` are ``(2, M)`` (component-major).
    The bodies themselves are kept in `bodies`, detached from the
    simulation.
    """

    def __init__(self) -> None:
        self.bodies: list[ChargedBody] = []
        self.time = np.zeros(0)
        self.wall = np.zeros(0, dtype=str)
        self.position = np.zeros((2, 0))
        self.velocity = np.zeros((2, 0))
        self.mass = np.zeros(0)
        self.charge = np.zeros(0)

    def record(self, bodies, *, time, walls, positions, velocities, masses, charges):
        self.bodies.extend(bodies)
        self.time = np.append(self.time, np.full(len(bodies), time))
        self.wall = np.append(self.wall, np.asarray(walls, dtype=str))
        self.position = np.hstack((self.position, positions))
        self.velocity = np.hstack((self.velocity, velocities))
        self.mass = np.append(self.mass, masses)
        self.charge = np.append(self.charge, charges)

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.charge))

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy carried out through the walls."""
        return float(0.5 * np.sum(self.mass * np.sum(self.velocity**2, axis=0)))

    def counts(self) -> dict[str, int]:
        """Number of particles absorbed at each wall and object."""
        names, counts = np.unique(self.wall, return_counts=True)
        return {str(name): int(count) for name, count in zip(names, counts)}

    def charges(self) -> dict[str, float]:
        """Charge absorbed at each wall and object (per electrode, its collected current times the run time)."""
        return {
            str(name): float(np.sum(self.charge[self.wall == name]))
            for name in np.unique(self.wall)
        }

    def __len__(self) -> int:
        return len(self.bodies)


def _drift_rules() -> dict:
    """The SymPy rule of the exact gyration drift: one particle, no field (for the tests)."""
    import sympy

    vx, vy, b_z, dt = sympy.symbols("vx vy B_z dt", real=True)
    angle = b_z * dt
    rule = dt * (sympy.sin(angle) / angle * vx + (1 - sympy.cos(angle)) / angle * vy)

    def call(vx, vy, B_z, dt):
        body = ChargedBody(0.0, 0.0, vx, vy)
        simulation = ElectrostaticSimulation(
            [body], Quad(0.0, 0.0, 1.0), dt=dt, B_z=B_z
        )
        simulation.step()
        return float(body.pos[0])

    ranges = {"vx": (-1.0, 1.0), "vy": (-1.0, 1.0), "B_z": (0.1, 3.0)}
    return {"drift": Rule(rule, call, ranges | {"dt": (0.1, 1.0)})}


def _orbit_problem() -> ConvergenceProblem:
    """A circular orbit about a fixed softened charge in a magnetic field.

    An electron (``q = -1``, ``m = 1``) at radius 1 about a fixed ion, ``k_e = 1``,
    ``epsilon = 0.1``, ``B_z = 0.7``: the electric and magnetic forces both point inward,
    so the orbit is circular for ``v^2 = F_e + v B_z``, ``F_e = (1 + epsilon^2)^{-3/2}``.
    The error is the distance to the exact position after ``t = 2``.
    """
    softening, b_z, time = 0.1, 0.7, 2.0
    attraction = (1 + softening**2) ** (-1.5)
    speed = 0.5 * (b_z + np.sqrt(b_z**2 + 4 * attraction))
    exact = np.array([np.cos(speed * time), np.sin(speed * time)])

    def error(dt: float) -> float:
        electron = ChargedBody(1.0, 0.0, vy=speed, charge=-1.0)
        ion = ChargedBody(0.0, 0.0, fixed=True)
        simulation = ElectrostaticSimulation(
            [electron, ion],
            Quad(0.0, 0.0, 2.0),
            theta=0.0,
            k_e=1.0,
            softening=softening,
            dt=dt,
            B_z=b_z,
        )
        for _ in range(round(time / dt)):
            simulation.step()
        return float(np.linalg.norm(electron.pos - exact))

    return ConvergenceProblem(error, steps=[0.04, 0.02, 0.01, 0.005])


def _pair_energy_rules() -> dict:
    """The SymPy rules of the exact pair energy of two charges (for the tests)."""
    import sympy

    x, q1, q2, k_e, eps, r_0 = sympy.symbols("x q1 q2 k_e eps r_0", positive=True)
    coulomb = k_e * q1 * q2 / sympy.sqrt(x**2 + eps**2)
    log = -k_e * q1 * q2 * sympy.log((x**2 + eps**2) / r_0**2)

    def energy(kernel: str):
        def call(x, q1, q2, k_e, eps, r_0=1.0):
            bodies = [ChargedBody(0.0, 0.0, charge=q1), ChargedBody(x, 0.0, charge=q2)]
            simulation = ElectrostaticSimulation(
                bodies,
                Quad(0.0, 0.0, 4.0),
                k_e=k_e,
                softening=eps,
                kernel=kernel,
                reference_radius=r_0,
            )
            return simulation.get_potential_energy(exact=True)

        return call

    ranges = {"x": (0.2, 3.0), "eps": (0.0, 0.5), "r_0": (0.5, 2.0)}
    return {
        "coulomb_pairs": Rule(coulomb, energy("coulomb"), ranges),
        "log_pairs": Rule(log, energy("log"), ranges),
    }


@method(
    equations={
        "motion": r"m_i \dot{\mathbf v}_i = q_i\big(\mathbf E(\mathbf r_i) + \mathbf v_i \times B_z\hat{\mathbf z}\big), \qquad \dot{\mathbf r}_i = \mathbf v_i, \qquad \mathbf E(\mathbf r_i) = k_e\, \mathbf E_\mathrm{raw}(\mathbf r_i;\, j \ne i)",
        "energy": r"K = \frac12\sum_i m_i v_i^2, \qquad U = \frac12\sum_i q_i \phi_i",
        "momentum": r"\mathbf P = \sum_i m_i \mathbf v_i, \qquad L_z = \sum_i m_i (x_i v_{y,i} - y_i v_{x,i})",
        "absorption": r"\text{removed after a drift if } s\,(x_\mathrm{axis} - x_\mathrm{wall}) > 0, \text{ or if the chord } \mathbf r^n \to \mathbf r^{n+1} \text{ enters an object}",
        "method": r"\texttt{auto}: \text{direct if } \theta = 0 \text{ or } N_\mathrm{charged} \le N_\mathrm{direct} \text{ (or no compiled kernels)}; \text{ else the FMM (log, } \theta < 1\text{), else the tree walk}",
    },
    assumptions=(
        r"free space, nonrelativistic, a uniform static $B_z$; the self-field is excluded ($j \ne i$); fixed sources never move",
        r"finite-$\theta$ forces are not pairwise symmetric (except the FMM with `symmetric=True`): momentum and energy are conserved to the tree error",
        r"absorbing walls and objects only remove particles: no image charges, no wall potential",
    ),
    components=(
        "tree_d_3d.simulation.ElectrostaticSimulation.step",
        "tree_d_3d.simulation.ElectrostaticSimulation.get_potential_energy",
        "tree_d_3d.bodies.ChargedBody",
        "tree_d_3d.flat_tree.direct_field",
        "tree_d_3d.flat_tree.FlatQuadtree.evaluate",
        "tree_d_3d.flat_tree.FlatQuadtree.fmm",
    ),
)
class ElectrostaticSimulation:
    """Isolated, nonrelativistic electrostatic N-body dynamics in a plane.

    Bounds are the initial tree extent, not physical walls. Without
    ``absorbing_walls`` all sources are retained. Symmetric electric kicks around an exact uniform-B drift give a
    second-order scheme, reducing to velocity Verlet when B=0. Positions and
    velocities are synchronized. Resolve close encounters and oscillations
    by reducing dt; softening changes the interaction below that length scale.

    ``method`` selects the field evaluation: ``"tree"`` (Barnes-Hut at
    ``theta``), ``"direct"`` (exact all-pairs sum), or ``"auto"``, which sums
    directly for ``theta == 0``, at most ``direct_threshold`` charged
    particles, or without the compiled kernels, where that is exact and
    faster. Fields are cached and reused while positions, charges and
    parameters are unchanged, so one step costs one field evaluation.

    ``absorbing_walls`` maps any of ``x_min``, ``x_max``, ``y_min`` and
    ``y_max`` to a wall coordinate. A particle that ends a drift beyond a wall
    is removed before the end-of-step field evaluation and stops acting as a
    source; see `absorbed`. Walls only remove particles: the field
    stays free-space, with no image charges or wall potential.

    ``absorbing_objects`` maps names to 2D `simgeometry` shapes (a
    ``Circle``, ``Polygon``, ``Aperture2D``, ...). A particle that ends a drift
    inside one is removed the same way and recorded under its name; an
    inverted shape absorbs everything outside it, such as a round chamber
    wall. The whole step is tested, not just its end, so a particle cannot
    jump over a plate thinner than ``|v| dt``: it is absorbed by the object
    whose boundary its step crosses first (``Polygon`` and ``Aperture2D``
    exactly, by simgeometry's compiled swept polygon kernel; other shapes by
    their ``crossings``). The step is the straight chord from the old to the
    new position: with ``B_z`` it lies within ``|v| dt theta / 8`` of the
    gyration arc, ``theta`` the gyration angle per step. Like walls,
    objects do not change the field: for electrodes, pair them with
    `tree_d_3d.BoundaryIntegralElectrodes`.

    ``kernel="log"`` makes every particle an infinite line charge per unit
    length along z, the interaction of 2D Cartesian PIC, instead of a point
    charge in the plane. Its pair energy is ``-2 k_e q_i q_j ln(r / r_0)``
    with ``r_0 = reference_radius``, which shifts the potential energy but
    not the forces. Line charges add ``method="fmm"``, the O(N) fast
    multipole method of `FlatQuadtree.fmm`, which ``"auto"`` selects
    above ``direct_threshold``. ``order`` is the number of expansion terms
    for line charges; ``leaf_capacity`` defaults to 32 for the FMM and 8
    otherwise.

    Finite-theta forces are not pairwise symmetric: check energy and momentum
    convergence.
    """

    def _adopt(self) -> None:
        """Move body state into ``(2, N)`` arrays and point each body at its column."""
        n = len(self.bodies)

        def columns(name):
            rows = np.array([getattr(b, name) for b in self.bodies], dtype=float)
            return np.ascontiguousarray(rows.reshape(n, 2).T)

        self.positions = columns("pos")
        self.velocities = columns("vel")
        self.forces = columns("force")
        self.masses = np.array([b.mass for b in self.bodies], dtype=float)
        self.charges = np.array([b.charge for b in self.bodies], dtype=float)
        self.fixed = np.array([b.fixed for b in self.bodies], dtype=bool)
        for i, body in enumerate(self.bodies):
            body._bind(
                self.positions,
                self.velocities,
                self.forces,
                self.masses,
                self.charges,
                self.fixed,
                i,
            )
        self._electric = np.zeros((2, n))
        self._potential = np.zeros(n)
        self._field_state = None
        self._fixed_fields_done = False
        self._tree = None

    def _entered_objects(self, start: np.ndarray | None) -> np.ndarray:
        """The name of the object each particle entered first along its step ("": none).

        ``Polygon`` and ``Aperture2D`` objects are tested together by
        simgeometry's swept polygon kernel (exact along the step), the other
        shapes by their ``crossings`` (exact for circles, axis-aligned boxes and
        polygonal shapes, sampled otherwise) and ``contains_points``. Where a
        step meets both kinds, the earlier crossing wins.
        """
        from simgeometry.kernels.polygon_hits import object_hit_times
        from simgeometry.polygon_objects import polygon_objects

        end = self.positions
        n = end.shape[1]
        names = list(self._objects)
        shapes = list(self._objects.values())
        moved = (
            np.zeros(0, dtype=int)
            if start is None
            else np.flatnonzero(np.any(start != end, axis=0))
        )
        # the other shapes: parameter of the first crossing along the step, 0 for
        # an end inside without one, 2 for no hit (the polygon kernel's convention)
        packed = polygon_objects(shapes)
        polygon_index = () if packed is None else packed.index
        other_time = np.full(n, 2.0)
        other = np.full(n, -1)
        for index, shape in enumerate(shapes):
            if index in polygon_index:
                continue
            time = np.where(_host(shape.contains_points(end)).astype(bool), 0.0, 2.0)
            if start is not None and moved.size:
                hits = shape.crossings(start[:, moved], end[:, moved])
                crossed = _host(hits.hit).astype(bool)
                time[moved[crossed]] = _host(hits.fraction)[crossed]
            earlier = time < other_time
            other_time[earlier] = time[earlier]
            other[earlier] = index
        first = other
        if packed is not None:
            alive = xp.ones(n, dtype=bool)
            slot = _host(
                packed.absorb(
                    _device(end),
                    alive,
                    start=None if start is None else _device(start),
                )
            )
            polygon = np.full(n, -1)
            polygon[slot >= 0] = np.asarray(packed.index)[slot[slot >= 0]]
            # steps that met both a polygon object and another shape
            both = np.flatnonzero((slot >= 0) & (other >= 0))
            polygon_time = np.zeros(n)
            if both.size:
                x1, y1 = end[0, both], end[1, both]
                x0, y0 = (x1, y1) if start is None else (start[0, both], start[1, both])
                vertices, offsets = _host(packed.vertices), _host(packed.offsets)
                for k in np.unique(slot[both]):
                    rows = slot[both] == k
                    polygon_time[both[rows]] = object_hit_times(
                        vertices,
                        offsets,
                        int(packed.first_host[k]),
                        int(packed.last_host[k]),
                        int(packed.inverted_host[k]),
                        x0[rows],
                        y0[rows],
                        x1[rows],
                        y1[rows],
                        np.full(int(rows.sum()), start is not None),
                    )
            take = (polygon >= 0) & (
                (other < 0)
                | (polygon_time < other_time)
                | ((polygon_time == other_time) & (polygon < other))
            )
            first = np.where(take, polygon, other)
        result = np.full(n, "", dtype=object)
        result[first >= 0] = np.asarray(names, dtype=object)[first[first >= 0]]
        return result

    def _outside_walls(
        self, start: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mask of absorbed particles, and the name of the deepest wall crossed or object entered.

        With ``start`` (the positions before the drift) a particle also enters
        an object when its step crosses the object's boundary. Walls take
        precedence over objects.
        """
        n = self.num_particles
        depth = np.zeros(n)
        name = np.full(n, "", dtype=object)
        for wall, axis, sign, value in self._walls:
            beyond = sign * (self.positions[axis] - value)
            deeper = beyond > depth
            depth[deeper] = beyond[deeper]
            name[deeper] = wall
        if self._objects:
            entered = self._entered_objects(start)
            free = name == ""
            name[free] = entered[free]
        return name != "", name

    def __init__(
        self,
        bodies: Sequence[ChargedBody],
        bounds: Quad,
        theta: float = 0.5,
        k_e: float = K_E,
        softening: float = 0.0,
        dt: float = 1e-12,
        B_z: float = 0.0,
        *,
        leaf_capacity: int | None = None,
        max_depth: int = 48,
        method: str = "auto",
        direct_threshold: int = DIRECT_THRESHOLD,
        absorbing_walls: Mapping[str, float] | None = None,
        absorbing_objects: Mapping[str, Shape] | None = None,
        kernel: str = "coulomb",
        order: int = 18,
        reference_radius: float = 1.0,
        symmetric: bool = False,
    ) -> None:
        self.bodies = list(bodies)
        if len({id(b) for b in self.bodies}) != len(self.bodies):
            raise ValueError("Each particle must be a distinct object")
        FlatQuadtree(
            np.zeros((2, 0)),
            np.zeros(0),
            bounds,
            leaf_capacity=8 if leaf_capacity is None else leaf_capacity,
            max_depth=max_depth,
            kernel=kernel,
            order=order,
            symmetric=symmetric,
        )
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if method == "fmm" and kernel != "log":
            raise ValueError("method='fmm' needs kernel='log'")
        if method == "fmm" and not 0 < theta < 1:
            raise ValueError("method='fmm' needs 0 < theta < 1")
        self.kernel = kernel
        self.order = int(order)
        #: symmetric FMM truncation: exactly antisymmetric pair forces (``kernel="log"``).
        self.symmetric = bool(symmetric)
        self.reference_radius = _finite(
            "reference_radius", reference_radius, positive=True
        )
        if (
            isinstance(direct_threshold, bool)
            or int(direct_threshold) != direct_threshold
            or direct_threshold < 0
        ):
            raise ValueError("direct_threshold must be a nonnegative integer")
        self.bounds = bounds
        self.theta = _finite("theta", theta, minimum=0)
        self.k_e = _finite("k_e", k_e, minimum=0)
        self.softening = _finite("softening", softening, minimum=0)
        self.dt = _finite("dt", dt, positive=True)
        self.B_z = _finite("B_z", B_z)
        self.leaf_capacity = leaf_capacity
        self.max_depth = max_depth
        self.method = method
        self.direct_threshold = int(direct_threshold)
        self.current_time = 0.0
        self.iteration = 0
        self._walls = _parse_walls(absorbing_walls or {})
        self._objects = _parse_objects(absorbing_objects or {})
        self._adopt()
        #: Removed particles with their state just after crossing a wall.
        self.absorbed = AbsorbedParticles()
        outside, _ = self._outside_walls()
        if np.any(outside):
            raise ValueError(
                "Particles must start inside the absorbing walls and outside the "
                "absorbing objects"
            )

    @property
    def num_particles(self) -> int:
        return self.positions.shape[1]

    @property
    def absorbing_walls(self) -> dict[str, float]:
        return {name: value for name, _, _, value in self._walls}

    @property
    def absorbing_objects(self) -> dict[str, Shape]:
        return dict(self._objects)

    def _absorb(self, start: np.ndarray | None = None) -> None:
        """Remove particles beyond a wall or inside an object and record them.

        ``start`` holds the positions before the drift, for the swept object test.
        """
        if not self._walls and not self._objects:
            return
        if len(self.bodies) != self.num_particles:
            self._adopt()
            start = None
        outside, name = self._outside_walls(start)
        if not np.any(outside):
            return
        removed = np.flatnonzero(outside)
        self.absorbed.record(
            [self.bodies[i] for i in removed],
            time=self.current_time + self.dt,
            walls=list(name[removed]),
            positions=self.positions[:, removed],
            velocities=self.velocities[:, removed],
            masses=self.masses[removed],
            charges=self.charges[removed],
        )
        for i in removed:
            # Detach, so the body keeps its state at absorption.
            body = self.bodies[i]
            body._bind(
                self.positions[:, i : i + 1].copy(),
                self.velocities[:, i : i + 1].copy(),
                self.forces[:, i : i + 1].copy(),
                self.masses[i : i + 1].copy(),
                self.charges[i : i + 1].copy(),
                self.fixed[i : i + 1].copy(),
                0,
            )
        keep = ~outside
        self.bodies = [b for b, k in zip(self.bodies, keep) if k]
        self.positions = np.ascontiguousarray(self.positions[:, keep])
        self.velocities = np.ascontiguousarray(self.velocities[:, keep])
        self.forces = np.ascontiguousarray(self.forces[:, keep])
        self.masses = self.masses[keep]
        self.charges = self.charges[keep]
        self.fixed = self.fixed[keep]
        for i, body in enumerate(self.bodies):
            body._bind(
                self.positions,
                self.velocities,
                self.forces,
                self.masses,
                self.charges,
                self.fixed,
                i,
            )
        self._electric = np.ascontiguousarray(self._electric[:, keep])
        self._potential = self._potential[keep]
        self._field_state = None
        self._tree = None

    def _resolve_method(self) -> str:
        if self.method != "auto":
            return self.method
        charged = int(np.count_nonzero(self.charges))
        if self.theta == 0 or charged <= self.direct_threshold:
            return "direct"
        if not kernels_compiled():
            return "direct"
        if self.kernel == "log" and self.theta < 1:
            return "fmm"
        return "tree"

    def _parameters(self) -> tuple:
        return (
            self.theta,
            self.softening,
            self.k_e,
            self.method,
            self.direct_threshold,
            self.leaf_capacity,
            self.max_depth,
            self.kernel,
            self.order,
            self.reference_radius,
        )

    def _fields_current(self) -> bool:
        if self._field_state is None:
            return False
        pos, charge, fixed, params = self._field_state
        return (
            params == self._parameters()
            and np.array_equal(pos, self.positions)
            and np.array_equal(charge, self.charges)
            and np.array_equal(fixed, self.fixed)
        )

    def _expand_bounds(self) -> None:
        if not self.num_particles:
            return
        if not np.all(np.isfinite(self.positions)):
            raise ValueError("Particle positions must be finite; reduce dt")
        b = self.bounds
        lo = np.minimum(self.positions.min(axis=1), [b.xmid - b.half, b.ymid - b.half])
        hi = np.maximum(self.positions.max(axis=1), [b.xmid + b.half, b.ymid + b.half])
        if np.any(lo < (b.xmid - b.half, b.ymid - b.half)) or np.any(
            hi > (b.xmid + b.half, b.ymid + b.half)
        ):
            center = lo / 2 + hi / 2
            half = np.max(hi / 2 - lo / 2) * (1 + 1e-12)
            self.bounds = Quad(float(center[0]), float(center[1]), float(half))

    def build_tree(self) -> FlatQuadtree:
        """Build a tree, enlarging its extent without dropping particles."""
        if len(self.bodies) != self.num_particles:
            self._adopt()
        self._expand_bounds()
        leaf = self.leaf_capacity
        if leaf is None:
            leaf = 32 if self._resolve_method() == "fmm" else 8
        return FlatQuadtree(
            self.positions,
            self.charges,
            self.bounds,
            leaf_capacity=leaf,
            max_depth=self.max_depth,
            kernel=self.kernel,
            order=self.order,
            symmetric=self.symmetric,
        )

    def _line_offset(self, targets: np.ndarray) -> np.ndarray:
        """Raw-potential shift for the line-charge reference radius r_0."""
        others = np.sum(self.charges) - self.charges[targets]
        return 2 * np.log(self.reference_radius) * others

    def _evaluate_targets(self, targets: np.ndarray) -> None:
        """Fill the cached force and potential at ``targets``."""
        if self.k_e == 0 or not len(targets):
            return
        if self._tree is not None:
            phi, field = self._tree.evaluate(
                self.positions[:, targets],
                exclude=targets,
                theta=self.theta,
                softening=self.softening,
            )
        else:
            phi, field = direct_field(
                self.positions[:, targets],
                self.positions,
                self.charges,
                softening=self.softening,
                exclude=targets,
                kernel=self.kernel,
            )
        phi, field = _host(phi), _host(field)
        if self.kernel == "log":
            phi = phi + self._line_offset(targets)
        self._potential[targets] = self.k_e * phi
        self._electric[:, targets] = self.k_e * self.charges[None, targets] * field

    @sp.profile("tree_d_3d.simulation.compute_fields", tags=_PROFILE_TAGS)
    def compute_fields(self, include_fixed: bool = True) -> FlatQuadtree | None:
        """Electric force and potential on every particle, reusing a cache.

        Fixed particles never move, so stepping skips them as targets
        (``include_fixed=False``) and their force and potential stay zero
        until they are asked for; they always act as sources. A later call
        with ``include_fixed=True`` fills them in from the same tree.
        Returns the tree used, or None for direct summation.
        """
        if len(self.bodies) != self.num_particles:
            self._adopt()
        if self._fields_current():
            if include_fixed and not self._fixed_fields_done:
                self._evaluate_targets(np.flatnonzero(self.charges * self.fixed))
                self._fixed_fields_done = True
            return self._tree
        self._expand_bounds()
        n = self.num_particles
        self._electric = np.zeros((2, n))
        self._potential = np.zeros(n)
        self._tree = None
        method = self._resolve_method()
        if method in ("tree", "fmm"):
            self._tree = self.build_tree()
        charged = self.charges != 0
        if method == "fmm":
            # The FMM yields every particle's field at once, fixed ones too.
            if self.k_e != 0 and np.any(charged):
                assert self._tree is not None
                phi, field = self._tree.fmm(theta=self.theta, softening=self.softening)
                phi, field = _host(phi), _host(field)
                everyone = np.arange(n)
                self._potential[:] = self.k_e * (phi + self._line_offset(everyone))
                self._electric[:] = self.k_e * self.charges[None, :] * field
            self._fixed_fields_done = True
            self._field_state = (
                self.positions.copy(),
                self.charges.copy(),
                self.fixed.copy(),
                self._parameters(),
            )
            return self._tree
        mobile = charged & (include_fixed | ~self.fixed)
        self._evaluate_targets(np.flatnonzero(mobile))
        self._fixed_fields_done = bool(
            include_fixed or not np.any(charged & self.fixed)
        )
        self._field_state = (
            self.positions.copy(),
            self.charges.copy(),
            self.fixed.copy(),
            self._parameters(),
        )
        return self._tree

    @method(
        equations={
            "half_kick": r"\mathbf v^- = \mathbf v^n + \frac{\Delta t}{2m}\, q\mathbf E^n, \qquad \mathbf v^{n+1} = \mathbf v^+ + \frac{\Delta t}{2m}\, q\mathbf E^{n+1}",
            "drift": r"\mathbf r^{n+1} = \mathbf r^n + \Delta t\left[\frac{\sin\alpha}{\alpha}\, \mathbf v^- + \frac{1 - \cos\alpha}{\alpha}\, (v^-_y, -v^-_x)\right], \qquad \alpha = \frac{q}{m} B_z \Delta t",
            "rotation": r"\mathbf v^+ = R(-\alpha)\, \mathbf v^- = \big(v^-_x\cos\alpha + v^-_y\sin\alpha,\ v^-_y\cos\alpha - v^-_x\sin\alpha\big)",
        },
        order=2,
        assumptions=(
            r"the drift is the exact solution of $\dot{\mathbf v} = (qB_z/m)(v_y, -v_x)$ over $\Delta t$ and its time integral, written with $\operatorname{sinc}\alpha$ and $\frac{\alpha}{2}\operatorname{sinc}^2\frac{\alpha}{2}$ so the $B_z \to 0$ limit is exact: velocity Verlet at $B_z = 0$",
            r"explicit: $\Delta t$ must resolve close encounters and the plasma period; the magnetic rotation is exact for any $\alpha$",
            r"one field evaluation per step (cached); particles are absorbed after the drift, before the end-of-step fields",
        ),
        symbolic=_drift_rules,
        problem=_orbit_problem,
    )
    @sp.profile("tree_d_3d.simulation.step", tags=_PROFILE_TAGS)
    def step(self, dt: float | None = None) -> FlatQuadtree | None:
        """Electric half-kick, exact cyclotron drift, electric half-kick.

        Returns the tree used for the end-of-step forces, or None when the
        fields were summed directly. Forces on fixed particles are not
        evaluated here and read as zero; call `compute_fields` for them.
        """
        self.dt = _finite("dt", self.dt if dt is None else dt, positive=True)
        self.compute_fields(include_fixed=False)
        mobile = ~self.fixed
        inv_mass = 1 / self.masses[mobile]
        vel = self.velocities
        vel[:, mobile] += 0.5 * self.dt * self._electric[:, mobile] * inv_mass
        angle = self.charges[mobile] / self.masses[mobile] * self.B_z * self.dt
        c, s = np.cos(angle), np.sin(angle)
        # Stable integral of the rotation, including the B=0 limit.
        a = np.sinc(angle / np.pi)
        b = 0.5 * angle * np.sinc(angle / (2 * np.pi)) ** 2
        vx, vy = vel[0, mobile], vel[1, mobile]
        start = self.positions.copy() if self._objects else None
        self.positions[:, mobile] += self.dt * np.stack(
            (a * vx + b * vy, a * vy - b * vx)
        )
        vel[:, mobile] = np.stack((c * vx + s * vy, c * vy - s * vx))
        self._absorb(start)
        tree = self.compute_fields(include_fixed=False)
        vel = self.velocities
        mobile = ~self.fixed
        inv_mass = 1 / self.masses[mobile]
        vel[:, mobile] += 0.5 * self.dt * self._electric[:, mobile] * inv_mass
        self.forces[:] = self._electric + self.charges * self.B_z * np.stack(
            (vel[1], -vel[0])
        )
        self.current_time += self.dt
        self.iteration += 1
        return tree

    def get_kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=0)))

    @method(
        equations={
            "field_energy": r"U = \frac12\sum_i q_i \phi_i \qquad \text{(the field evaluation of the forces)}",
            "coulomb_pairs": r"U = k_e \sum_{i<j} \frac{q_i q_j}{\sqrt{r_{ij}^2 + \varepsilon^2}}",
            "log_pairs": r"U = -k_e \sum_{i<j} q_i q_j \ln\frac{r_{ij}^2 + \varepsilon^2}{r_0^2}",
        },
        assumptions=(
            r"without the self-energy; `exact=True` sums all pairs, $O(N^2)$; otherwise exact for direct sums and accurate to the tree error",
            r"$r_0$ = `reference_radius` shifts the log energy, not the forces",
        ),
        symbolic=_pair_energy_rules,
    )
    def get_potential_energy(self, exact: bool = False) -> float:
        """Pair energy for the force kernel, excluding self-energy.

        By default this is ``0.5 * sum(q_i * phi_i)`` from the field
        evaluation used for the forces, at no extra cost: exact when the
        fields are summed directly, and accurate to the tree error otherwise.
        ``exact=True`` sums all pairs, which costs O(N squared).
        """
        if self.k_e == 0 or not len(self.bodies):
            return 0.0
        if not exact:
            self.compute_fields()
            return float(0.5 * self.charges @ self._potential)
        if self.kernel == "log":
            everyone = np.arange(self.num_particles)
            phi, _ = direct_field(
                self.positions,
                self.positions,
                self.charges,
                softening=self.softening,
                exclude=everyone,
                kernel="log",
            )
            phi = _host(phi) + self._line_offset(everyone)
            return float(0.5 * self.k_e * self.charges @ phi)
        status = xp.zeros(1, dtype=np.int64)
        energy = xp.zeros(1)
        _kernels().pair_energy(
            _device(self.positions),
            _device(self.charges),
            self.softening,
            status,
            energy,
        )
        _check(status)
        return float(self.k_e * float(energy[0]))

    def get_total_energy(self, exact: bool = False) -> float:
        return self.get_kinetic_energy() + self.get_potential_energy(exact)

    def get_total_charge(self) -> float:
        return float(np.sum(self.charges))

    def get_center_of_mass(self) -> np.ndarray:
        mass = np.sum(self.masses)
        if not mass:
            return np.zeros(2)
        return self.positions @ self.masses / mass

    def get_momentum(self) -> np.ndarray:
        return self.velocities @ self.masses if len(self.masses) else np.zeros(2)

    def get_angular_momentum(self) -> float:
        """Mechanical angular momentum about the origin (z component)."""
        x, y = self.positions
        vx, vy = self.velocities
        return float(np.sum(self.masses * (x * vy - y * vx)))
