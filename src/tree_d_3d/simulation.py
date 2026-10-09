"""Time integration of charged particles in a plane, with fields from the flat tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cunumpy as xp
import numpy as np

from tree_d_3d._utils import _check, _device, _finite, _host
from tree_d_3d.bodies import ChargedBody
from tree_d_3d.constants import K_E
from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.kernels import _kernels, kernels_compiled
from tree_d_3d.quad import Quad

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


class AbsorbedParticles:
    """Particles removed at absorbing walls, in removal order.

    Positions and velocities are taken just after the drift that crossed the
    wall, so positions lie up to one step beyond it; ``time`` is the end of
    that step. ``position`` and ``velocity`` are ``(2, M)`` (component-major).
    The bodies themselves are kept in :attr:`bodies`, detached from the
    simulation.
    """

    def __init__(self) -> None:
        self.bodies: list[ChargedBody] = []
        self.time = np.zeros(0)
        self.wall = np.zeros(0, dtype="<U5")
        self.position = np.zeros((2, 0))
        self.velocity = np.zeros((2, 0))
        self.mass = np.zeros(0)
        self.charge = np.zeros(0)

    def record(self, bodies, *, time, walls, positions, velocities, masses, charges):
        self.bodies.extend(bodies)
        self.time = np.append(self.time, np.full(len(bodies), time))
        self.wall = np.append(self.wall, np.asarray(walls, dtype="<U5"))
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
        """Number of particles absorbed at each wall."""
        names, counts = np.unique(self.wall, return_counts=True)
        return {str(name): int(count) for name, count in zip(names, counts)}

    def __len__(self) -> int:
        return len(self.bodies)


class ElectrostaticSimulation:
    """Isolated, nonrelativistic electrostatic N-body dynamics in a plane.

    Bounds are the initial tree extent, not physical walls. Without
    ``absorbing_walls`` all sources are retained. Symmetric electric kicks around an exact uniform-B drift give a
    second-order scheme, reducing to velocity Verlet when B=0. Positions and
    velocities are synchronized. Resolve close encounters and oscillations
    by reducing dt; softening changes the interaction below that length scale.

    ``method`` selects the field evaluation: ``"tree"`` (Barnes-Hut at
    ``theta``), ``"direct"`` (exact all-pairs sum), or ``"auto"``, which sums
    directly for ``theta == 0`` or at most ``direct_threshold`` charged
    particles, where that is exact and faster. Fields are cached and reused
    while positions, charges and parameters are unchanged, so one step costs
    one field evaluation.

    ``absorbing_walls`` maps any of ``x_min``, ``x_max``, ``y_min`` and
    ``y_max`` to a wall coordinate. A particle that ends a drift beyond a wall
    is removed before the end-of-step field evaluation and stops acting as a
    source; see :attr:`absorbed`. Walls only remove particles: the field
    stays free-space, with no image charges or wall potential.

    ``kernel="log"`` makes every particle an infinite line charge per unit
    length along z, the interaction of 2D Cartesian PIC, instead of a point
    charge in the plane. Its pair energy is ``-2 k_e q_i q_j ln(r / r_0)``
    with ``r_0 = reference_radius``, which shifts the potential energy but
    not the forces. Line charges add ``method="fmm"``, the O(N) fast
    multipole method of :meth:`FlatQuadtree.fmm`, which ``"auto"`` selects
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

    def _outside_walls(self) -> tuple[np.ndarray, np.ndarray]:
        """Mask of particles beyond a wall, and the deepest wall crossed."""
        n = self.num_particles
        depth = np.zeros(n)
        wall = np.full(n, -1)
        for k, (_, axis, sign, value) in enumerate(self._walls):
            beyond = sign * (self.positions[axis] - value)
            deeper = beyond > depth
            depth[deeper] = beyond[deeper]
            wall[deeper] = k
        return wall >= 0, wall

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
        kernel: str = "coulomb",
        order: int = 18,
        reference_radius: float = 1.0,
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
        )
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if method == "fmm" and kernel != "log":
            raise ValueError("method='fmm' needs kernel='log'")
        if method == "fmm" and not 0 < theta < 1:
            raise ValueError("method='fmm' needs 0 < theta < 1")
        self.kernel = kernel
        self.order = int(order)
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
        self._adopt()
        #: Removed particles with their state just after crossing a wall.
        self.absorbed = AbsorbedParticles()
        outside, _ = self._outside_walls()
        if np.any(outside):
            raise ValueError("Particles must start inside the absorbing walls")

    @property
    def num_particles(self) -> int:
        return self.positions.shape[1]

    @property
    def absorbing_walls(self) -> dict[str, float]:
        return {name: value for name, _, _, value in self._walls}

    def _absorb(self) -> None:
        """Remove particles beyond a wall and record them."""
        if not self._walls:
            return
        if len(self.bodies) != self.num_particles:
            self._adopt()
        outside, wall = self._outside_walls()
        if not np.any(outside):
            return
        removed = np.flatnonzero(outside)
        self.absorbed.record(
            [self.bodies[i] for i in removed],
            time=self.current_time + self.dt,
            walls=[self._walls[k][0] for k in wall[removed]],
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

    def step(self, dt: float | None = None) -> FlatQuadtree | None:
        """Electric half-kick, exact cyclotron drift, electric half-kick.

        Returns the tree used for the end-of-step forces, or None when the
        fields were summed directly. Forces on fixed particles are not
        evaluated here and read as zero; call :meth:`compute_fields` for them.
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
        self.positions[:, mobile] += self.dt * np.stack(
            (a * vx + b * vy, a * vy - b * vx)
        )
        vel[:, mobile] = np.stack((c * vx + s * vy, c * vy - s * vx))
        self._absorb()
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
