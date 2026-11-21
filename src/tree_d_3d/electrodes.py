"""Conductors at prescribed potentials, by the boundary-integral (panel) method.

A conductor's surface carries an induced line-charge density whose potential, added to
the free charges' potential, makes the conductor an equipotential: the single-layer
formulation of `tree_d_3d`'s research note "10 Boundary-integral electrodes"
(``obsidian-notes/notes/tree-plasma-ideas``). Discretising a conductor into straight
panels and collocating at panel midpoints turns this into a small dense linear system
``S Q = g``: ``S`` couples panel charges to panel potentials, a panel's own column uses
the closed-form potential of a uniformly charged straight segment at its own midpoint
(finite, unlike a point charge's self-potential). A panel close to its target -- within
``near_field`` of its own lengths, another panel's midpoint or a probe point -- also
enters with that exact segment potential and field (`segment_field`); farther
panels are ``far_points`` (2) Gauss--Legendre point line charges each, whose error falls
as ``(L / r)**(2 far_points)``, summed fast by `tree_d_3d.direct_field`; one point at
the midpoint (``far_points=1``) is the usual flat-panel approximation, with error
``(L / r)**2``. Without the near-field correction the potential between collocation points is off by
about a tenth of the source scale next to a panel and the method converges at first
order; with it the boundary is an equipotential to within the piecewise-constant charge
density's own error.

A period along y (`periodic_log_kernel`: a row of line charges in closed form) and a
plane domain face (`Face`, Neumann or Dirichlet, by mirror images) take the place of the
stacked and mirrored copies of the electrodes an extraction cell would otherwise need.

Up to `DIRECT_LIMIT` panels the system is solved directly, by an LU factorisation done
once per geometry and reused by every `BoundaryIntegralElectrodes.solve`. Above it
``solver="auto"`` iterates instead: GMRES, preconditioned by the exact panel matrix of each
run of consecutive panels (block Jacobi), with each product ``S Q`` a fast multipole sum
over the Gauss nodes (`tree_d_3d.FlatQuadtree.fmm`) plus the sparse exact near
corrections, so memory and time grow almost linearly with the panels instead of as
``P**2`` and ``P**3``. Large sums at probes and over free charges walk a multipole tree
(above `TREE_PAIRS` pairs). With a period the fast sums run in the plane mapped by
``z -> exp(2 pi z / period)``, where the row of copies of a charge becomes one charge
(`BoundaryIntegralElectrodes._periodic_tree_sum`), for points within `MAP_REACH` periods
of the conductors along x.

The log kernel is the one `tree_d_3d.FlatQuadtree` and `tree_d_3d.direct_field`
use with ``kernel="log"``: potential ``-2 q ln r`` and field ``2 q d / r**2`` per unit
charge, without ``k_e`` (applied here, as in `tree_d_3d.GroundedStripField`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scope_profiler as sp
from maybempi import collectives
from methodcard import ConvergenceProblem, Rule, method
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.spatial import KDTree

from tree_d_3d._profiling import _PROFILE_TAGS
from tree_d_3d.constants import K_E
from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.quad import Quad

#: Panels closer to a target than this many of their own lengths enter exactly.
NEAR_FIELD: float = 4.0
#: Gauss--Legendre points per panel in the fast far-field sum.
FAR_POINTS: int = 2
#: Above this many panels ``solver="auto"`` iterates (GMRES with multipole sums) instead of LU.
DIRECT_LIMIT: int = 3000
#: Above this many target-source pairs a free-space sum walks a multipole tree.
TREE_PAIRS: int = 4_000_000
#: The opening angle and leaf size of the panels' multipole sums. With ``fmm_order`` 32 the
#: error is about 2e-12 of the field, at 1.3x the speed of 0.5 with order 24 (measured on
#: panel nodes along curves, where leaves of 16 beat 32).
FMM_THETA: float = 0.7
FMM_LEAF: int = 16
#: Consecutive panels of a conductor per block of the GMRES preconditioner.
PRECONDITIONER_BLOCK: int = 256
#: With a period, the fast sums map the plane by ``exp(2 pi z / period)``; they take points
#: within this many periods of the conductors along x (the map's range is ``exp(2 pi`` this),
#: farther ones fall back to the direct periodic sum.
MAP_REACH: float = 8.0


def _segment_rules() -> dict:
    """The SymPy rules of `segment_field`: the kernel integrated along the segment.

    For the tests. The segment runs from the origin along x; SymPy integrates
    ``-ln((u - s)^2 + v^2) / L`` over ``0 < s < L`` (``v > 0``), and the field is minus
    the gradient of the result.
    """
    import sympy

    u, v, length, s = sympy.symbols("u v length s", positive=True)
    phi = -sympy.integrate(sympy.log((u - s) ** 2 + v**2), (s, 0, length)) / length
    ranges = {"u": (-1.0, 2.0), "v": (0.05, 1.0), "length": (0.2, 1.5)}

    def component(index: int):
        def call(u, v, length):
            phi, field = segment_field(
                np.array([[u], [v]]), np.zeros((2, 1)), np.array([[length], [0.0]])
            )
            return float(phi[0]) if index < 0 else float(field[index, 0])

        return call

    return {
        "potential": Rule(phi, component(-1), ranges),
        "field_along": Rule(-sympy.diff(phi, u), component(0), ranges),
        "field_across": Rule(-sympy.diff(phi, v), component(1), ranges),
    }


@method(
    equations={
        "potential": r"\phi_\mathrm{seg} = -\frac{2}{L}\int_0^L \ln\sqrt{(u - s)^2 + v^2}\, ds = -\frac{2}{L}\Big[u\ln r_a - (u - L)\ln r_b - L + |v|\vartheta\Big]",
        "field_along": r"\mathbf E_\mathrm{seg}\cdot\hat{\mathbf t} = \frac{2}{L}\ln\frac{r_a}{r_b}",
        "field_across": r"\mathbf E_\mathrm{seg}\cdot\hat{\mathbf n} = \frac{2}{L}\operatorname{sign}(v)\,\vartheta, \qquad \vartheta = \operatorname{atan2}(u, |v|) - \operatorname{atan2}(u - L, |v|)",
    },
    assumptions=(
        r"unit total charge spread evenly along a straight segment of length $L$, in the log kernel, without $k_e$",
        r"$u$, $v$ the target's coordinates along and across the segment from its start; $r_a$, $r_b$ the distances to its ends; $\hat{\mathbf n} = (-t_y, t_x)$",
        r"on the segment the normal field (the jump of the surface charge) is taken as zero; at an end the potential is finite and the field diverges logarithmically",
    ),
    symbolic=_segment_rules,
)
def segment_field(points, start, end) -> tuple[np.ndarray, np.ndarray]:
    """Potential ``(K,)`` and field ``(2, K)`` of uniformly charged straight segments.

    Pairs column by column: ``points``, ``start`` and ``end`` are ``(2, K)``, and segment
    ``k`` from ``start[:, k]`` to ``end[:, k]`` carries unit total charge spread evenly
    along it, in the log kernel of `tree_d_3d.direct_field` (a point charge ``q``
    has potential ``-2 q ln r``), without ``k_e``. With ``u`` and ``v`` the target's
    coordinates along and across the segment of length ``L`` from its start,

    ``phi = -(2 / L) [u ln r_a - (u - L) ln r_b - L + |v| theta]``,
    ``E = (2 / L) [ln(r_a / r_b) t + sign(v) theta n]``,

    where ``r_a`` and ``r_b`` are the distances to the two ends, ``theta`` the angle the
    segment subtends at the target, and ``t`` and ``n`` the unit tangent and normal. On
    the segment itself the normal field (a jump of the surface charge) is taken as zero;
    at an end point the potential is finite but the field diverges logarithmically.
    """
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    d = end - start
    length = np.hypot(d[0], d[1])
    t = d / length
    n = np.stack((-t[1], t[0]))
    rel = points - start
    u = np.sum(rel * t, axis=0)
    v = np.sum(rel * n, axis=0)
    av = np.abs(v)
    ra2 = u * u + v * v
    rb2 = (u - length) ** 2 + v * v
    theta = np.arctan2(u, av) - np.arctan2(u - length, av)
    # at an end point the potential stays finite and the field diverges (log)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_a = np.log(ra2)
        log_b = np.log(rb2)
        ends = np.where(ra2 > 0, u * log_a, 0.0) - np.where(
            rb2 > 0, (u - length) * log_b, 0.0
        )
        along = 0.5 * (log_a - log_b)
        field = (2.0 / length) * (along * t + np.sign(v) * theta * n)
    integral = 0.5 * ends - length + av * theta
    phi = -2.0 * integral / length
    return phi, field


def _point_field(points, sources) -> tuple[np.ndarray, np.ndarray]:
    """The point line-charge kernel, pairwise: ``-2 ln r`` and ``2 d / r**2``.

    Zero for coincident pairs, which the far sums leave out.
    """
    d = points - sources
    r2 = np.sum(d * d, axis=0)
    at = r2 == 0.0
    r2 = np.where(at, 1.0, r2)
    return np.where(at, 0.0, -np.log(r2)), np.where(at, 0.0, 2.0 * d / r2)


#: Above this ``2 pi |x| / period`` the periodic kernel takes its exponential asymptote.
_PERIODIC_ASYMPTOTE = 30.0
#: Rows of the pairwise arrays held at once by the periodic sums.
_PAIRS_PER_CHUNK = 2_000_000
# the search radius of exact-coincidence queries (a k-d tree prunes beyond it at once)
_TINY = 1e-150


def _periodic_rules() -> dict:
    """The SymPy rules of `periodic_log_kernel`: the closed form and minus its gradient."""
    import sympy

    x, y = sympy.symbols("x y", real=True)
    period = sympy.Integer(2)
    k = 2 * sympy.pi / period
    phi = -sympy.log(2 * (sympy.cosh(k * x) - sympy.cos(k * y))) + 2 * sympy.log(k)
    ranges = {"x": (-3.0, 3.0), "y": (0.05, 0.95)}

    def component(index: int):
        def call(x, y):
            phi, field = periodic_log_kernel(np.array([[x], [y]]), float(period))
            return float(phi[0]) if index < 0 else float(field[index, 0])

        return call

    return {
        "potential": Rule(phi, component(-1), ranges),
        "field_x": Rule(-sympy.diff(phi, x), component(0), ranges),
        "field_y": Rule(-sympy.diff(phi, y), component(1), ranges),
    }


@method(
    equations={
        "potential": r"G_P(\mathbf d) = \sum_{m=-\infty}^{\infty} -\ln|\mathbf d - m P \hat{\mathbf y}|^2 = -\ln\Big[2\Big(\cosh\frac{2\pi d_x}{P} - \cos\frac{2\pi d_y}{P}\Big)\Big] + 2\ln\frac{2\pi}{P}",
        "field_x": r"-\partial_x G_P = \frac{2\pi}{P}\,\frac{\sinh(2\pi d_x/P)}{\cosh(2\pi d_x/P) - \cos(2\pi d_y/P)}",
        "field_y": r"-\partial_y G_P = \frac{2\pi}{P}\,\frac{\sin(2\pi d_y/P)}{\cosh(2\pi d_x/P) - \cos(2\pi d_y/P)}",
        "stable": r"\cosh a - \cos b = 2\sinh^2\frac a2 + 2\sin^2\frac b2",
    },
    assumptions=(
        r"a row of unit line charges a period $P$ apart along $y$, in the log kernel, without $k_e$; the divergent sum is renormalised so that $G_P + \ln|\mathbf d|^2 \to 0$ as $\mathbf d \to 0$",
        r"far from the row ($|d_x| \gg P$) $G_P \to -2\pi|d_x|/P + 2\ln(2\pi/P)$: the potential of a charged sheet, $2\pi/P$ per unit charge on either side",
        r"$\cosh a - \cos b$ is evaluated as $2\sinh^2(a/2) + 2\sin^2(b/2)$ (no cancellation near a charge) and by its exponential asymptote above $|a| = 30$",
    ),
    symbolic=_periodic_rules,
)
def periodic_log_kernel(d, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Potential ``(K,)`` and field ``(2, K)`` of a row of unit line charges, period ``period`` along y.

    ``d`` ``(2, K)`` are the targets relative to one charge of the row; the others sit
    at ``d = (0, m period)``. In the log kernel of `tree_d_3d.direct_field`, without
    ``k_e``, the sum of ``-ln r**2`` over the row is renormalised to

    ``G = -ln(2 (cosh(2 pi x / P) - cos(2 pi y / P))) + 2 ln(2 pi / P)``,

    which approaches the single charge's ``-ln r**2`` next to it. The field is
    ``-grad G``. At a charge itself the potential is infinite.
    """
    d = np.asarray(d, dtype=float)
    k = 2.0 * np.pi / float(period)
    a = k * d[0]
    b = k * d[1]
    large = np.abs(a) > _PERIODIC_ASYMPTOTE
    near_a = np.where(large, 0.0, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        # cosh a - cos b without cancellation near a charge
        c = 2.0 * np.sinh(0.5 * near_a) ** 2 + 2.0 * np.sin(0.5 * b) ** 2
        e = np.exp(-np.abs(np.where(large, a, 0.0)))
        tail = 1.0 - 2.0 * np.cos(b) * e + e * e
        log_c = np.where(
            large,
            np.abs(a) - np.log(2.0) + np.log(np.where(large, tail, 1.0)),
            np.log(c),
        )
        phi = -np.log(2.0) - log_c + 2.0 * np.log(k)
        ex = np.where(large, np.sign(a) * (1.0 - e * e) / tail, np.sinh(near_a) / c)
        ey = np.where(large, 2.0 * np.sin(b) * e / tail, np.sin(b) / c)
    return phi, k * np.stack((ex, ey))


class _Operator(LinearOperator):
    """A square SciPy linear operator applying ``function``."""

    def __init__(self, size: int, function: Callable[[np.ndarray], np.ndarray]):
        super().__init__(np.dtype(float), (size, size))
        self._function = function

    def _matvec(self, x):
        return self._function(np.ravel(x))


@dataclass(frozen=True)
class Face:
    """A plane domain face, ``x = position`` (``axis="x"``) or ``y = position``.

    ``kind="neumann"``: no normal field on the face (a symmetry plane); every charge,
    induced or free, has a mirror image of the same sign. ``kind="dirichlet"``: the
    face is held at ``potential``; the images carry the opposite sign. Only the side
    of the face holding the conductors is physical.
    """

    axis: str
    position: float
    kind: str = "neumann"
    potential: float = 0.0

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError(f"a face's axis is 'x' or 'y', not {self.axis!r}")
        if self.kind not in ("neumann", "dirichlet"):
            raise ValueError(f"a face is 'neumann' or 'dirichlet', not {self.kind!r}")
        if self.kind == "neumann" and self.potential != 0.0:
            raise ValueError("a Neumann face has no potential")

    @property
    def sign(self) -> float:
        """The sign of the image charges: +1 for Neumann, -1 for Dirichlet."""
        return 1.0 if self.kind == "neumann" else -1.0

    def mirror(self, points: np.ndarray) -> np.ndarray:
        """``points`` ``(2, n)`` mirrored about the face."""
        axis = 0 if self.axis == "x" else 1
        mirrored = np.array(points, dtype=float, copy=True)
        mirrored[axis] = 2.0 * self.position - mirrored[axis]
        return mirrored


def _table_problem() -> ConvergenceProblem:
    """Bilinear interpolation of a smooth potential and field against the grid spacing."""
    rng = np.random.default_rng(3)
    points = rng.uniform(0.0, 1.0, (2, 200))

    def sample(x, y):
        return np.sin(3.0 * x) * np.cos(2.0 * y), np.stack(
            (np.cos(2.0 * x + y), np.sin(x - y))
        )

    exact_phi, exact_field = sample(*points)

    def error(spacing: float) -> dict[str, float]:
        x = np.linspace(0.0, 1.0, round(1.0 / spacing) + 1)
        phi, field = sample(*np.meshgrid(x, x, indexing="ij"))
        phi, field = TabulatedField(x, x, phi, field)(points)
        return {
            "potential": float(np.max(np.abs(phi - exact_phi))),
            "field": float(np.max(np.abs(field - exact_field))),
        }

    return ConvergenceProblem(error, steps=[1 / 10, 1 / 20, 1 / 40, 1 / 80])


@method(
    equations={
        "bilinear": r"f(x, y) = (1 - f_x)(1 - f_y)\, f_{i,j} + f_x (1 - f_y)\, f_{i+1,j} + (1 - f_x) f_y\, f_{i,j+1} + f_x f_y\, f_{i+1,j+1}",
        "cell": r"f_x = \frac{x - x_0}{\Delta x} - i \in [0, 1], \qquad i = \min\!\left(\left\lfloor \frac{x - x_0}{\Delta x} \right\rfloor, n_x - 2\right)",
    },
    order={"potential": 2, "field": 2},
    assumptions=(
        r"a regular grid; potential and field interpolated separately, so $\mathbf E = -\nabla\phi$ holds only to the interpolation error",
        r"points outside the grid take the value at the nearest edge",
    ),
    problem=_table_problem,
)
@dataclass(frozen=True)
class TabulatedField:
    """Potential and field sampled on a regular grid, interpolated bilinearly.

    ``x`` ``(nx,)`` and ``y`` ``(ny,)`` are the grid lines, ``potential`` ``(nx, ny)``
    and ``field`` ``(2, nx, ny)`` the samples, as `BoundaryIntegralElectrodes.tabulate`
    makes them. Calling it with ``(2, M)`` points returns the interpolated potential
    ``(M,)`` and field ``(2, M)``; points outside the grid take the value at the
    nearest edge (see `contains`).
    """

    x: np.ndarray
    y: np.ndarray
    potential: np.ndarray
    field: np.ndarray

    def contains(self, points) -> np.ndarray:
        """Mask of the ``(2, M)`` points inside the tabulated rectangle."""
        points = np.asarray(points, dtype=float).reshape(2, -1)
        return (
            (points[0] >= self.x[0])
            & (points[0] <= self.x[-1])
            & (points[1] >= self.y[0])
            & (points[1] <= self.y[-1])
        )

    @staticmethod
    def _cell(grid: np.ndarray, coordinate: np.ndarray):
        """Index of the cell holding each coordinate (clamped) and the fraction across it."""
        f = np.clip((coordinate - grid[0]) / (grid[1] - grid[0]), 0.0, grid.size - 1.0)
        index = np.asarray(np.minimum(f.astype(np.int64), grid.size - 2))
        return index, np.asarray(f - index)

    def __call__(self, points) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float).reshape(2, -1)
        i, fx = self._cell(self.x, points[0])
        j, fy = self._cell(self.y, points[1])
        weights = (
            ((1 - fx) * (1 - fy), i, j),
            (fx * (1 - fy), i + 1, j),
            ((1 - fx) * fy, i, j + 1),
            (fx * fy, i + 1, j + 1),
        )
        phi = np.zeros(points.shape[1])
        field = np.zeros((2, points.shape[1]))
        for w, a, b in weights:
            phi += w * self.potential[a, b]
            field += w * self.field[:, a, b]
        return phi, field


class OutlinedShape(Protocol):
    """A 2D shape with boundary loops, such as any 2D `simgeometry` shape."""

    def outlines(self, max_spacing=None) -> list[np.ndarray]: ...


@dataclass
class Conductor:
    """A conductor's boundary: a polyline of vertices ``(2, n)`` held at one potential.

    Consecutive vertices are panel endpoints; ``closed`` connects the last vertex back
    to the first (a loop), giving ``n`` panels instead of ``n - 1``.
    """

    vertices: np.ndarray
    potential: float
    closed: bool = True


def conductors_from_shape(
    shape: OutlinedShape,
    potential: float,
    max_panel_length: float | Callable[[np.ndarray], np.ndarray],
) -> list[Conductor]:
    """Closed conductors on the boundary of a 2D `simgeometry` shape.

    One `Conductor` per boundary loop of ``shape.outlines(max_panel_length)``
    (an ``Aperture2D`` gives two, one per plate), all held at ``potential``; an open
    curve such as a ``Polyline`` (a plate of zero thickness) gives an open conductor. Panels are
    at most ``max_panel_length`` long; it can also be a function of ``(2, m)`` points
    returning ``(m,)`` lengths, to grade the panels (short where the field varies
    fastest, such as an aperture tip). Keep panels shorter than the gaps between
    surfaces, a plate's thickness included, so that the charge on one face does not
    have to resolve the field of the facing one.
    """
    closed = bool(getattr(shape, "closed", True))
    return [
        Conductor(loop, potential, closed=closed)
        for loop in shape.outlines(max_panel_length)
    ]


def _grounded_circle_problem() -> ConvergenceProblem:
    """A line charge outside a grounded circle, against its three-image solution.

    Circle of radius ``a = 1.5``, unit charge at ``d = 5``: the image ``-q`` at
    ``a^2 / d`` and ``q_c = -q ln(d / a) / ln(a)`` at the centre (no conductor at
    infinity, unit reference length). The errors of the potential at seven points on a
    circle of radius 3 and of the force on the charge, against the panel length.
    """
    from simgeometry import Circle

    a, d = 1.5, 5.0
    z_image, q_centre = a * a / d, -np.log(d / a) / np.log(a)
    angle = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False) + 0.1
    points = 3.0 * np.stack((np.cos(angle), np.sin(angle)))
    distance = np.hypot(points[0][:, None] - [d, z_image, 0.0], points[1][:, None]).T
    exact_phi = -2.0 * (
        np.log(distance[0]) - np.log(distance[1]) + q_centre * np.log(distance[2])
    )
    exact_force = -2.0 / (d - z_image) + 2.0 * q_centre / d

    def error(panel: float) -> dict[str, float]:
        conductors = conductors_from_shape(
            Circle(midpoint=(0, 0), radius=a), 0.0, panel
        )
        electrodes = BoundaryIntegralElectrodes(conductors, k_e=1.0)
        electrodes.solve(np.array([[d], [0.0]]), np.array([1.0]))
        phi = electrodes.potential(points)
        force = electrodes.force_on_sources()[1][0, 0]
        return {
            "potential": float(np.max(np.abs(phi - exact_phi))),
            "force": float(abs(force - exact_force)),
        }

    return ConvergenceProblem(
        error, steps=[2.0 * np.pi * a / n for n in (50, 100, 200, 400)]
    )


@method(
    equations={
        "boundary_integral": r"-2k_e\int_{\Gamma}\sigma(\mathbf y)\ln|\mathbf x - \mathbf y|\,ds_y \;-\; 2k_e\sum_j q_j\ln|\mathbf x - \mathbf r_j| = V_k, \qquad \mathbf x \in \Gamma_k",
        "collocation": r"\mathsf S\,\mathbf Q = \mathbf V / k_e - \boldsymbol\phi_\mathrm{free}^\mathrm{raw}, \qquad \phi_\mathrm{free}^\mathrm{raw}(\mathbf m_i) = -\sum_j q_j \ln|\mathbf m_i - \mathbf r_j|^2",
        "self_term": r"S_{kk} = -\frac{2}{L_k}\int_{-L_k/2}^{L_k/2}\ln|t|\,dt = 2\left(1 - \ln\frac{L_k}{2}\right)",
        "panels": r"S_{ik} = \begin{cases} \phi_\mathrm{seg}(\mathbf m_i;\, k) & |\mathbf m_i - \mathbf m_k| < n_\mathrm{near} L_k \\ -\sum_{g=1}^{G} w_g \ln|\mathbf m_i - \mathbf y_{k,g}|^2 & \text{otherwise} \end{cases}",
        "far_points": r"\mathbf y_{k,g} = \mathbf a_k + \xi_g (\mathbf b_k - \mathbf a_k), \qquad \xi_g = \frac{x_g + 1}{2}, \quad w_g = \frac{\omega_g}{2}, \qquad \text{error } O\big((L/r)^{2G}\big)",
        "field": r"\phi(\mathbf x) = k_e\Big[\sum_{k\ \mathrm{near}} Q_k\, \phi_\mathrm{seg}(\mathbf x; k) + \sum_{k\ \mathrm{far}} \sum_g Q_k w_g\, \phi_\mathrm{pt}(\mathbf x - \mathbf y_{k,g}) - \sum_j q_j \ln|\mathbf x - \mathbf r_j|^2\Big]",
        "images": r"\phi_\mathrm{pt}(\mathbf x - \mathbf y) \to \sum_{\iota \in \{1, \mathsf M\}} s_\iota\, G_P(\mathbf x - \iota\,\mathbf y), \qquad s_{\mathsf M} = \pm 1 \text{ (Neumann, Dirichlet)}, \qquad \phi \to \phi + V_\mathrm{face}",
    },
    order={"potential": 2, "force": 2},
    assumptions=(
        r"a first-kind single-layer equation; straight panels of uniform charge $Q_k$ (piecewise-constant density), collocated at their midpoints $\mathbf m_k$",
        r"$G$ = `far_points` = 2 Gauss-Legendre nodes $x_g$, weights $\omega_g$, per far panel; panels within $n_\mathrm{near}$ = `near_field` = 4 of their lengths enter exactly (`segment_field`)",
        r"$\mathsf S$ is not symmetric: up to `DIRECT_LIMIT` panels dense and LU-factorised once, $O(P^3)$, and $O(P^2)$ per `solve`; above it GMRES with a block-Jacobi preconditioner (runs of `PRECONDITIONER_BLOCK` consecutive panels) and multipole-tree products, $O(P)$ per iteration, the iterations growing slowly with $P$",
        r"no conductor at infinity and no constraint on the total charge: the reference length of the logarithm is the unit of length, and a lone conductor of logarithmic capacity 1 (a circle of radius 1) makes $\mathsf S$ singular",
        r"keep panels shorter than the gaps between surfaces; the order is in the panel length",
        r"with a period $P$ along $y$ the point kernel is the row sum $G_P$ (`periodic_log_kernel`) and the near panel copies, shifted by $\pm P$, enter exactly; a face $\mathsf M$ mirrors every charge about it, and the conductors' potentials are measured from a Dirichlet face's $V_\mathrm{face}$",
    ),
    components=(
        segment_field,
        periodic_log_kernel,
        "tree_d_3d.flat_tree.direct_field",
    ),
    problem=_grounded_circle_problem,
)
class BoundaryIntegralElectrodes:
    """Conductors held at prescribed potentials, by the single-layer boundary integral.

    ``conductors`` is a sequence of `Conductor`. Call `solve` once per
    configuration of free charges (or with none, for a pure capacitance problem); then
    `probe` (and `potential`, `field`) give the total field --
    electrodes plus free charges -- at arbitrary points, and `force_on_sources`
    the force the electrodes and the other free charges exert back on the free charges.

    Panels within ``near_field`` of their own lengths of a target use the exact field of
    a uniformly charged segment (`segment_field`); farther ones are summed fast as
    ``far_points`` Gauss--Legendre point charges each, whose error falls as
    ``(L / r) ** (2 * far_points)``. ``near_field=0, far_points=1`` gives the plain
    midpoint rule.

    ``solver`` is ``"direct"`` (dense LU), ``"gmres"`` (fast multipole products,
    converged to ``tolerance`` in at most ``max_iterations``, with ``fmm_order``
    expansion terms) or ``"auto"``, which iterates above `DIRECT_LIMIT` panels.
    ``iterations`` counts the last solve's GMRES steps. Sums over more than
    `TREE_PAIRS` pairs (probes, free charges) walk a multipole tree too.

    ``period`` makes the geometry periodic along y: every charge, induced or free,
    stands for a row of copies ``period`` apart, summed in closed form by
    `periodic_log_kernel` (the panel copies near a target still enter exactly).
    ``face`` adds a plane domain face (`Face`): a Neumann face mirrors every charge
    with its sign, a Dirichlet face with the opposite sign and holds the face at its
    potential. A face normal to x combines with a period; a second, parallel face
    is a conductor, such as an open plate spanning one period at the face's
    potential (an infinite plane under the period). The conductors must lie on one
    side of the face, the physical one. With a period the potential far from the
    conductors grows linearly in ``|x|`` unless the total charge per period
    vanishes; a domain closed by conductors and faces is unaffected.

    Conductors at one potential may touch along an edge, also across the period
    seam: the panel pairs on the shared edge are dropped (``num_shared_panels``), so
    the conductors act as their union. `probe` returns the exact potential and zero
    field inside a solid conductor (`inside`), and `surface_field` the normal field
    just outside each panel.

    ``comm``, an MPI communicator, shares the work out over its ranks: every rank
    passes the same conductors, free charges and points, and gets the whole result.
    """

    def _near_pairs(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The pairs of targets and panel copies closer than ``near_field`` panel lengths.

        A panel copy is an image (``0`` the panel itself, ``1`` its mirror in the
        face) shifted by a whole number of periods along y (``0`` without a
        period). Returns the target, panel and image indices and the y shift of
        each pair.
        """
        empty = np.zeros(0, dtype=np.int64)
        if not self.near_field or not points.shape[1]:
            return empty, empty, empty, np.zeros(0)
        radius = self.near_field * float(self.lengths.max())
        found = []
        for image, tree in enumerate(self._image_trees):
            for shift in self._shifts:
                shifted = points - np.array([[0.0], [shift]])
                pairs = KDTree(shifted.T).sparse_distance_matrix(
                    tree, radius, output_type="ndarray"
                )
                if not pairs.size:
                    continue
                target = pairs["i"].astype(np.int64)
                panel = pairs["j"].astype(np.int64)
                keep = pairs["v"] < self.near_field * self.lengths[panel]
                found.append(
                    (
                        target[keep],
                        panel[keep],
                        np.full(int(keep.sum()), image),
                        np.full(int(keep.sum()), shift),
                    )
                )
        if not found:
            return empty, empty, empty, np.zeros(0)
        target, panel, image, shift = (np.concatenate(c) for c in zip(*found))
        return target, panel, image, shift

    def _near_correction(
        self,
        points: np.ndarray,
        target: np.ndarray,
        panel: np.ndarray,
        image: np.ndarray,
        shift: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per pair and unit panel charge: the exact segment less the panel copy's point terms.

        Point terms at zero distance are left out of the far sums (`_far_sum`), so
        they are left out here too.
        """
        shifted = points[:, target] - np.stack((np.zeros_like(shift), shift))
        phi, field = segment_field(
            shifted,
            self._image_starts[image, :, panel].T,
            self._image_ends[image, :, panel].T,
        )
        k = self.far_points
        for g in range(k):
            node = self._image_nodes[image, :, panel * k + g].T
            pt_phi, pt_field = _point_field(shifted, node)
            phi = phi - self._weights[g] * pt_phi
            field = field - self._weights[g] * pt_field
        sign = self._image_signs[image]
        return sign * phi, sign * field

    def _pairwise_periodic(self, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`periodic_log_kernel` on ``(2, M, N)`` separations, zero where they vanish."""
        assert self.period is not None
        shape = d.shape[1:]
        flat = d.reshape(2, -1)
        at = (flat[0] == 0.0) & (np.remainder(flat[1], self.period) == 0.0)
        safe = np.where(at, 1.0, flat)
        phi, field = periodic_log_kernel(safe, self.period)
        phi = np.where(at, 0.0, phi)
        field = np.where(at, 0.0, field)
        return phi.reshape(shape), field.reshape((2, *shape))

    def _pairwise_potential(self, points: np.ndarray, sources: np.ndarray):
        """The ``(M, N)`` raw potentials of unit charges, zero for coincident pairs."""
        if self.period is not None:
            return self._pairwise_periodic(points[:, :, None] - sources[:, None, :])[0]
        r2 = np.sum((points[:, :, None] - sources[:, None, :]) ** 2, axis=0)
        with np.errstate(divide="ignore"):
            return np.where(r2 > 0.0, -np.log(r2), 0.0)

    def _near_entries(self):
        """Every panel's own pair and the near pairs, as ``(target, panel, image, shift)``."""
        target, panel, image, shift = self._near_pairs(self.midpoints)
        own = (target == panel) & (image == 0) & (shift == 0.0)
        index = np.arange(self.num_panels)
        return (
            np.concatenate((target[~own], index)),
            np.concatenate((panel[~own], index)),
            np.concatenate((image[~own], np.zeros_like(index))),
            np.concatenate((shift[~own], np.zeros(self.num_panels))),
        )

    def _build_matrix(self) -> np.ndarray:
        """The dense ``S``: Gauss nodes for far pairs, exact segments for near ones.

        Each rank builds its share of the rows; every rank gets the whole matrix.
        """
        k = self.far_points
        rows = collectives.block(self.num_panels, self.comm)
        targets = self.midpoints[:, rows]
        matrix = np.zeros((targets.shape[1], self.num_panels))
        for image, sign in enumerate(self._image_signs):
            nodes = self._image_nodes[image]
            for g in range(k):
                matrix += (
                    sign
                    * self._weights[g]
                    * self._pairwise_potential(targets, nodes[:, g::k])
                )
        # every panel's own segment, also with near_field = 0
        target, panel, image, shift = self._near_entries()
        mine = (target >= rows.start) & (target < rows.stop)
        target, panel, image, shift = (
            target[mine],
            panel[mine],
            image[mine],
            shift[mine],
        )
        correction, _ = self._near_correction(
            self.midpoints, target, panel, image, shift
        )
        np.add.at(matrix, (target - rows.start, panel), correction)
        if collectives.is_parallel(self.comm):
            matrix = collectives.gather_blocks(
                self.comm, np.ascontiguousarray(matrix.T), self.num_panels
            ).T
        return np.ascontiguousarray(matrix)

    def _drop_shared_edges(self) -> None:
        """Remove the panel pairs where two conductors touch along an edge.

        Two loops sharing an edge (also across a period) would put two panels with
        opposite orientation on the same segment, which makes the panel system
        singular. When both panels belong to conductors at the same potential, the
        pair is interior to the union of the two and is dropped; at different
        potentials the touching conductors are an error. A shared edge must be
        panelled the same way from both sides, as `conductors_from_shape` does.
        """
        lengths = self.lengths
        scale = float(np.max(np.abs(np.concatenate((self.starts, self.ends), axis=1))))
        tolerance = 1e-9 * max(scale, float(lengths.max()))
        tangent = (self.ends - self.starts) / lengths
        tree = KDTree(self.midpoints.T)
        drop = np.zeros(lengths.size, dtype=bool)
        shifts = (0.0,) if self.period is None else (-self.period, 0.0, self.period)
        for shift in shifts:
            offset = np.array([[0.0], [shift]])
            for i, others in enumerate(
                tree.query_ball_point((self.midpoints + offset).T, float(lengths.max()))
            ):
                for j in others:
                    if j <= i and shift == 0.0:
                        continue
                    if np.dot(tangent[:, i], tangent[:, j]) > -1.0 + 1e-9:
                        continue
                    # antiparallel: on the same line and overlapping?
                    rel = self.midpoints[:, j] - self.midpoints[:, i] - offset[:, 0]
                    across = abs(rel[0] * tangent[1, i] - rel[1] * tangent[0, i])
                    along = abs(rel @ tangent[:, i])
                    if (
                        across > tolerance
                        or along >= 0.5 * (lengths[i] + lengths[j]) - tolerance
                    ):
                        continue
                    same = np.allclose(
                        self.starts[:, i] + offset[:, 0],
                        self.ends[:, j],
                        atol=tolerance,
                    ) and np.allclose(
                        self.ends[:, i] + offset[:, 0],
                        self.starts[:, j],
                        atol=tolerance,
                    )
                    if not same:
                        raise ValueError(
                            "conductors touch along an edge panelled differently on "
                            "its two sides; panel the shared edge the same way"
                        )
                    if self.target_potential[i] != self.target_potential[j]:
                        raise ValueError(
                            "conductors at different potentials touch along an edge"
                        )
                    drop[[i, j]] = True
        keep = ~drop
        #: number of panels dropped on edges shared by touching conductors
        self.num_shared_panels = int(drop.sum())
        if not keep.any():
            raise ValueError("every panel lies on a shared edge")
        self.midpoints = self.midpoints[:, keep]
        self.lengths = self.lengths[keep]
        self.starts = self.starts[:, keep]
        self.ends = self.ends[:, keep]
        self.group = self.group[keep]
        self.target_potential = self.target_potential[keep]

    def _check_one_side(self, face: Face) -> None:
        """Raise unless every conductor lies on one side of ``face`` (touching it is allowed)."""
        axis = 0 if face.axis == "x" else 1
        offset = np.concatenate((self.starts[axis], self.ends[axis])) - face.position
        tolerance = 1e-12 * max(1.0, float(np.abs(offset).max()))
        if np.any(offset > tolerance) and np.any(offset < -tolerance):
            raise ValueError("the conductors must all lie on one side of the face")

    def _loop_polygon(self, conductor: Conductor):
        """The even-odd test of a closed conductor's loop, by simgeometry's compiled kernel."""
        cache = self.__dict__.setdefault("_polygons", {})
        test = cache.get(id(conductor))
        if test is None:
            from simgeometry import Polygon

            polygon = Polygon(
                vertices=np.ascontiguousarray(np.asarray(conductor.vertices, float).T)
            )

            def test(points, polygon=polygon):
                return np.asarray(polygon.contains_points(points), dtype=bool)

            cache[id(conductor)] = test
        return test

    def _loop_holds(self, conductor: Conductor, points: np.ndarray) -> np.ndarray:
        """Mask of the ``(2, M)`` points inside a closed conductor's loop, over the periods."""
        vertices = np.asarray(conductor.vertices, dtype=float)
        low, high = vertices[1].min(), vertices[1].max()
        left, right = vertices[0].min(), vertices[0].max()
        held = np.zeros(points.shape[1], dtype=bool)
        if not points.shape[1]:
            return held
        if self.period is None:
            shifts = np.zeros(1)
        else:
            first = np.floor((points[1].min() - high) / self.period)
            last = np.ceil((points[1].max() - low) / self.period)
            shifts = np.arange(first, last + 1) * self.period
        for shift in shifts:
            candidates = np.flatnonzero(
                ~held
                & (points[1] - shift >= low)
                & (points[1] - shift <= high)
                & (points[0] >= left)
                & (points[0] <= right)
            )
            if candidates.size:
                shifted = points[:, candidates] - np.array([[0.0], [shift]])
                held[candidates[self._loop_polygon(conductor)(shifted)]] = True
        return held

    def _exact_block(self, block: np.ndarray) -> np.ndarray:
        """The dense ``S`` among the panels ``block``: their point terms and near corrections."""
        k = self.far_points
        matrix = np.zeros((block.size, block.size))
        for image, sign in enumerate(self._image_signs):
            nodes = self._image_nodes[image]
            for g in range(k):
                matrix += (
                    sign
                    * self._weights[g]
                    * self._pairwise_potential(
                        self.midpoints[:, block], nodes[:, block * k + g]
                    )
                )
        return matrix + self._correction[block][:, block].toarray()

    def _mapped(self, points: np.ndarray):
        """``exp(2 pi (x - x0 + i y) / period)``: complex ``(M,)`` and as ``(2, M)`` points."""
        assert self.period is not None
        z = np.exp(
            (2.0 * np.pi / self.period) * ((points[0] - self._map_x0) + 1j * points[1])
        )
        return z, np.stack((z.real, z.imag))

    def _prepare_iterative(self) -> None:
        """The sparse near corrections and the block preconditioner of the GMRES solve.

        ``S Q`` is a multipole-tree sum over every panel's Gauss nodes (and their
        mirror images) at the panel midpoints, plus the sparse
        near correction ``C`` (exact segment less point terms). The preconditioner is
        block Jacobi: the exact ``S`` among each run of `PRECONDITIONER_BLOCK`
        consecutive panels of a conductor, LU-factorised once. It takes in the
        high-frequency modes of this first-kind equation, which are local along a
        conductor: 51 iterations instead of 150 with the self terms alone at 21k panels.
        A sparse LU of all near entries does worse, missing their smooth far part.
        """
        target, panel, image, shift = self._near_entries()
        correction, _ = self._near_correction(
            self.midpoints, target, panel, image, shift
        )
        shape = (self.num_panels, self.num_panels)
        self._correction = csc_matrix((correction, (target, panel)), shape=shape)
        blocks = []
        for conductor in np.unique(self.group):
            panels = np.flatnonzero(self.group == conductor)
            for first in range(0, panels.size, PRECONDITIONER_BLOCK):
                blocks.append(panels[first : first + PRECONDITIONER_BLOCK])
        # each rank factorises its share of the blocks; `_precondition` gathers
        mine = blocks[collectives.block(len(blocks), self.comm)]
        self._blocks = [(block, lu_factor(self._exact_block(block))) for block in mine]
        self._block_order = np.concatenate(blocks)
        # the tree's sources (every image's nodes) and its charge-free targets
        nodes = np.concatenate(list(self._image_nodes), axis=1)
        k = self.far_points
        if k % 2:
            # the middle node is the midpoint: it is the target, excluded from itself
            self._fmm_points = nodes
            self._fmm_targets = (np.arange(self.num_panels) * k + k // 2).astype(
                np.int64
            )
        else:
            self._fmm_points = np.concatenate((nodes, self.midpoints), axis=1)
            self._fmm_targets = nodes.shape[1] + np.arange(
                self.num_panels, dtype=np.int64
            )
        # with a period the products are free-space sums over the mapped points
        self._fmm_plane = (
            self._fmm_points
            if self.period is None
            else self._mapped(self._fmm_points)[1]
        )
        low = self._fmm_plane.min(axis=1)
        high = self._fmm_plane.max(axis=1)
        half = 0.5 * float(np.max(high - low)) * (1.0 + 1e-6) + 1e-300
        centre = 0.5 * (low + high)
        self._fmm_quad = Quad(float(centre[0]), float(centre[1]), half)

    def _mappable(self, *point_sets: np.ndarray) -> bool:
        """Whether every point lies within `MAP_REACH` periods of the conductors along x."""
        assert self.period is not None
        reach = MAP_REACH * self.period
        return all(
            not p.shape[1] or float(np.abs(p[0] - self._map_x0).max()) <= reach
            for p in point_sets
        )

    def __init__(
        self,
        conductors: Sequence[Conductor],
        *,
        period: float | None = None,
        face: Face | None = None,
        k_e: float = K_E,
        near_field: float = NEAR_FIELD,
        far_points: int = FAR_POINTS,
        solver: str = "auto",
        tolerance: float = 1e-10,
        fmm_order: int = 32,
        max_iterations: int = 2000,
        comm=None,
    ) -> None:
        if solver not in ("auto", "direct", "gmres"):
            raise ValueError("solver must be 'auto', 'direct' or 'gmres'")
        if not 0 < tolerance < 1:
            raise ValueError("tolerance must lie in (0, 1)")
        if not conductors:
            raise ValueError("need at least one conductor")
        if not near_field >= 0:
            raise ValueError("near_field must be >= 0")
        if int(far_points) != far_points or far_points < 1:
            raise ValueError("far_points must be a positive integer")
        if period is not None and not (np.isfinite(period) and period > 0):
            raise ValueError("period must be positive and finite")
        if period is not None and face is not None and face.axis == "y":
            raise ValueError("a face normal to the periodic y axis is not supported")
        self.k_e = float(k_e)
        #: the MPI communicator whose ranks share the work (``None``: one process)
        self.comm = comm
        self.near_field = float(near_field)
        self.far_points = int(far_points)
        #: the period along y, ``None`` in free space
        self.period = None if period is None else float(period)
        #: the domain face, mirroring every charge, or ``None``
        self.face = face
        self.conductors = list(conductors)
        midpoints, lengths, group, target, starts, ends = [], [], [], [], [], []
        for i, c in enumerate(self.conductors):
            v = np.asarray(c.vertices, dtype=float)
            if v.ndim != 2 or v.shape[0] != 2 or v.shape[1] < 2:
                raise ValueError("a conductor needs vertices of shape (2, n >= 2)")
            a = v if c.closed else v[:, :-1]
            b = np.roll(v, -1, axis=1) if c.closed else v[:, 1:]
            seg_length = np.linalg.norm(b - a, axis=0)
            if np.any(seg_length <= 0):
                raise ValueError("a panel has zero length")
            midpoints.append(0.5 * (a + b))
            starts.append(a)
            ends.append(b)
            lengths.append(seg_length)
            group.append(np.full(a.shape[1], i))
            target.append(np.full(a.shape[1], float(c.potential)))
        #: panel midpoints, ``(2, num_panels)``
        self.midpoints = np.concatenate(midpoints, axis=1)
        #: panel lengths, ``(num_panels,)``
        self.lengths = np.concatenate(lengths)
        #: panel end points, ``(2, num_panels)`` each
        self.starts = np.concatenate(starts, axis=1)
        self.ends = np.concatenate(ends, axis=1)
        #: index of the conductor each panel belongs to, ``(num_panels,)``
        self.group = np.concatenate(group)
        #: the panel's conductor's potential, ``(num_panels,)``
        self.target_potential = np.concatenate(target)
        self._drop_shared_edges()
        self.num_panels = self.midpoints.shape[1]
        if face is not None:
            self._check_one_side(face)
        # far panels as far_points Gauss--Legendre point charges each, panel-major
        x, w = np.polynomial.legendre.leggauss(self.far_points)
        self._weights = 0.5 * w
        along = 0.5 * (x + 1.0)
        self._nodes = (
            self.starts[:, :, None]
            + (self.ends - self.starts)[:, :, None] * along[None, None, :]
        ).reshape(2, -1)
        if self.far_points % 2:
            # the middle node is the midpoint, exactly (a self pair must coincide)
            self._nodes[:, self.far_points // 2 :: self.far_points] = self.midpoints
        self._node_weights = np.tile(self._weights, self.num_panels)
        # the panels and their mirror image in the face; shifts by whole periods
        images = [lambda points: points]
        if face is not None:
            images.append(face.mirror)
        self._image_signs = np.array([1.0] + ([face.sign] if face else []))
        self._image_starts = np.stack([mirror(self.starts) for mirror in images])
        self._image_ends = np.stack([mirror(self.ends) for mirror in images])
        self._image_nodes = np.stack([mirror(self._nodes) for mirror in images])
        self._image_trees = [
            KDTree((0.5 * (a + b)).T)
            for a, b in zip(self._image_starts, self._image_ends)
        ]
        self._image_node_trees = [KDTree(nodes.T) for nodes in self._image_nodes]
        self._panel_tree = self._image_trees[0]
        # closed loops around no other conductor: their inside is metal
        self._solid = np.array(
            [
                c.closed
                and not any(
                    other is not c
                    and self._loop_holds(c, np.asarray(other.vertices, float)[:, :1])
                    for other in self.conductors
                )
                for c in self.conductors
            ]
        )
        self._shifts = (
            (0.0,) if self.period is None else (-self.period, 0.0, self.period)
        )
        # the centre of the conductors along x, where the periodic map is centred
        self._map_x0 = 0.5 * float(
            np.min(np.minimum(self.starts[0], self.ends[0]))
            + np.max(np.maximum(self.starts[0], self.ends[0]))
        )
        #: the potential the conductors' potentials are measured from: a Dirichlet face's
        self.offset = face.potential if face is not None else 0.0
        mappable = self.period is None or self._mappable(self.starts, self.ends)
        if solver == "gmres" and not mappable:
            raise ValueError(
                f"with a period the iterative solver needs the conductors within "
                f"{MAP_REACH} periods along x; use solver='direct'"
            )
        if solver == "auto":
            solver = (
                "gmres" if mappable and self.num_panels > DIRECT_LIMIT else "direct"
            )
        #: ``"direct"`` (dense LU) or ``"gmres"`` (multipole-tree matrix-vector products)
        self.solver = solver
        self.tolerance = float(tolerance)
        self.fmm_order = int(fmm_order)
        self.max_iterations = int(max_iterations)
        # multipole trees over sources that never move (the panels' nodes), by key
        self._trees: dict = {}
        #: GMRES iterations of the last `solve` (0 for the direct solver)
        self.iterations = 0
        with sp.region("tree_d_3d.electrodes.factorize", tags=_PROFILE_TAGS):
            if solver == "direct":
                self._matrix = self._build_matrix()
                self._lu = lu_factor(self._matrix)
            else:
                self._prepare_iterative()
        #: solved panel charges, set by `solve`
        self.charge: np.ndarray | None = None
        self._source_positions = np.zeros((2, 0))
        self._source_charges = np.zeros(0)

    def _shared(self, function: Callable[[slice], tuple], n: int) -> tuple:
        """``function`` of a share of ``range(n)`` per rank, its arrays gathered along the last axis.

        Serially ``function(slice(0, n))``; on several ranks each evaluates its own
        share (`maybempi.collectives.block`) and every rank gets the whole result.
        """
        if not collectives.is_parallel(self.comm):
            return function(slice(0, n))
        return tuple(
            collectives.gather_blocks(self.comm, part, n)
            for part in function(collectives.block(n, self.comm))
        )

    def _periodic_tree_sum(self, points, sources, charges, key):
        """A periodic sum as a free-space one over the mapped points.

        With ``z = exp(2 pi (x - x0 + i y) / P)`` the row kernel is
        ``G_P = -ln|z_t - z_s|^2 + 2 pi (x_t + x_s - 2 x0) / P + 2 ln(2 pi / P)``: a log sum
        over the mapped sources (a multipole tree, kept for sources that never move)
        and terms linear in ``x``. The field follows from ``dz/dx = (2 pi / P) z``. A
        target on a source drops that pair with its linear terms, which leaves the
        other periods only (the remainder, zero there), as `_periodic_sum` does.
        """
        assert self.period is not None
        k = 2.0 * np.pi / self.period
        z_t, mapped_t = self._mapped(points)
        tree = self._trees.get(("periodic", key)) if key is not None else None
        if tree is None:
            _, mapped_s = self._mapped(sources)
            low, high = mapped_s.min(axis=1), mapped_s.max(axis=1)
            centre = 0.5 * (low + high)
            half = 0.5 * float(np.max(high - low)) * (1.0 + 1e-9) + 1e-300
            tree = FlatQuadtree(
                mapped_s,
                charges,
                Quad(float(centre[0]), float(centre[1]), half),
                kernel="log",
                order=self.fmm_order,
                leaf_capacity=FMM_LEAF,
            )
            if key is not None:
                self._trees[("periodic", key)] = tree
        else:
            tree.set_charges(charges)
        distance, nearest = KDTree(sources.T).query(
            points.T, distance_upper_bound=_TINY
        )
        exclude = np.where(distance == 0.0, nearest, -1).astype(np.int64)
        phi_z, field_z = tree.evaluate(mapped_t, exclude=exclude, theta=FMM_THETA)
        phi_z, field_z = np.asarray(phi_z), np.asarray(field_z)
        total = float(np.sum(charges))
        x_t = points[0] - self._map_x0
        phi = (
            phi_z
            + k * total * x_t
            + k * float(np.sum(charges * (sources[0] - self._map_x0)))
            + 2.0 * total * np.log(k)
        )
        complex_field = (field_z[0] - 1j * field_z[1]) * k * z_t - k * total
        dropped = np.flatnonzero(exclude >= 0)
        if dropped.size:
            q = charges[exclude[dropped]]
            phi[dropped] -= q * (2.0 * k * x_t[dropped] + 2.0 * np.log(k))
            complex_field[dropped] += k * q
        return phi, np.stack((complex_field.real, -complex_field.imag))

    def _periodic_sum(
        self, points: np.ndarray, sources: np.ndarray, charges: np.ndarray, key=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """`periodic_log_kernel` summed over ``sources``; coincident pairs give zero.

        Above `TREE_PAIRS` pairs, and within `MAP_REACH` periods, by the mapped tree
        (`_periodic_tree_sum`), otherwise directly, in chunks.
        """
        m = points.shape[1]
        phi = np.zeros(m)
        field = np.zeros((2, m))
        if not sources.shape[1]:
            return phi, field
        if m * sources.shape[1] > TREE_PAIRS and self._mappable(points, sources):
            return self._periodic_tree_sum(points, sources, charges, key)
        rows = max(1, _PAIRS_PER_CHUNK // sources.shape[1])
        for first in range(0, m, rows):
            block = slice(first, first + rows)
            d = points[:, block, None] - sources[:, None, :]
            pair_phi, pair_field = self._pairwise_periodic(d)
            phi[block] = pair_phi @ charges
            field[:, block] = pair_field @ charges
        return phi, field

    def _tree_sum(self, points, sources, charges, exclude, key):
        """A large free-space sum by a Barnes-Hut walk with ``fmm_order`` log expansions.

        The tree over sources that never move (the panels' nodes, ``key``) is built
        once and only recharged; others are built for the call. At `FMM_THETA` and the
        default order 32 the error is about 1e-12 of the field.
        """
        tree = self._trees.get(key) if key is not None else None
        if tree is None:
            low, high = sources.min(axis=1), sources.max(axis=1)
            centre = 0.5 * (low + high)
            half = 0.5 * float(np.max(high - low)) * (1.0 + 1e-9) + 1e-300
            tree = FlatQuadtree(
                sources,
                charges,
                Quad(float(centre[0]), float(centre[1]), half),
                kernel="log",
                order=self.fmm_order,
                leaf_capacity=FMM_LEAF,
            )
            if key is not None:
                self._trees[key] = tree
        else:
            tree.set_charges(charges)
        phi, field = tree.evaluate(points, exclude=exclude, theta=FMM_THETA)
        return np.array(phi), np.array(field)

    def _far_sum(
        self,
        points: np.ndarray,
        sources: np.ndarray,
        charges: np.ndarray,
        image: int,
        exclude: np.ndarray | None = None,
        key=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw potential and field at ``points`` of one image of point line charges.

        Free space: `tree_d_3d.direct_field`, skipping ``exclude``, or above
        `TREE_PAIRS` pairs a multipole tree (`_tree_sum`; ``key`` names sources that
        never move, whose tree is kept); with a period,
        `periodic_log_kernel` summed in chunks, where a target on a source keeps
        only the other periods (the remainder, zero there).
        """
        sign = self._image_signs[image]
        if image and self.face is not None:
            sources = self.face.mirror(sources)
        if self.period is None:
            if points.shape[1] * sources.shape[1] > TREE_PAIRS:
                return self._tree_sum(points, sources, sign * charges, exclude, key)
            phi, field = direct_field(
                points, sources, sign * charges, kernel="log", exclude=exclude
            )
            return np.array(phi), np.array(field)
        return self._periodic_sum(points, sources, sign * charges, key=key)

    def _precondition(self, residual: np.ndarray) -> np.ndarray:
        """The block-Jacobi solve (`_prepare_iterative`), each rank solving its blocks."""
        local = [lu_solve(lu, residual[block]) for block, lu in self._blocks]
        solved, _ = collectives.gather_columns(
            self.comm, np.concatenate(local) if local else np.zeros(0)
        )
        out = np.empty_like(residual)
        out[self._block_order] = solved
        return out

    def _apply(self, charge: np.ndarray) -> np.ndarray:
        """``S @ charge`` by a multipole tree and the sparse near corrections."""
        node_charge = np.repeat(charge, self.far_points) * self._node_weights
        sources = np.concatenate(
            [sign * node_charge for sign in self._image_signs]
            + [
                np.zeros(
                    self._fmm_points.shape[1]
                    - node_charge.size * len(self._image_signs)
                )
            ]
        )
        # the points never move: one tree per geometry, new charges per product
        tree = getattr(self, "_fmm_tree", None)
        if tree is None:
            tree = self._fmm_tree = FlatQuadtree(
                self._fmm_plane,
                sources,
                self._fmm_quad,
                kernel="log",
                order=self.fmm_order,
                leaf_capacity=FMM_LEAF,
                comm=self.comm,
            )
        else:
            tree.set_charges(sources)

        def walk(share: slice):
            # a walk at the midpoints beats the FMM over every node: 2x at 21k panels
            targets = self._fmm_targets[share]
            phi, _ = tree.evaluate(
                self._fmm_plane[:, targets], exclude=targets, theta=FMM_THETA
            )
            return (np.asarray(phi),)

        (phi,) = self._shared(walk, self.num_panels)
        if self.period is not None:
            # the terms of the row kernel linear in x (`_periodic_tree_sum`); a target's
            # own node (odd far_points) is left out with its linear terms
            k = 2.0 * np.pi / self.period
            x = self._fmm_points[0] - self._map_x0
            total = float(np.sum(sources))
            x_t = x[self._fmm_targets]
            own = sources[self._fmm_targets]
            phi = (
                phi
                + k * total * x_t
                + k * float(np.sum(sources * x))
                + 2.0 * total * np.log(k)
                - own * (2.0 * k * x_t + 2.0 * np.log(k))
            )
        return phi + self._correction @ charge

    @sp.profile("tree_d_3d.electrodes.gmres", tags=_PROFILE_TAGS)
    def _solve_iterative(self, rhs: np.ndarray) -> np.ndarray:
        """GMRES on `_apply`, preconditioned block by block (`_precondition`)."""
        operator = _Operator(self.num_panels, self._apply)
        preconditioner = _Operator(self.num_panels, self._precondition)
        count = [0]

        def callback(_):
            count[0] += 1

        charge, info = gmres(
            operator,
            rhs,
            rtol=self.tolerance,
            atol=0.0,
            restart=min(200, self.max_iterations),
            maxiter=max(1, -(-self.max_iterations // 200)),
            M=preconditioner,
            callback=callback,
            callback_type="pr_norm",
        )
        self.iterations = count[0]
        if info != 0:
            raise RuntimeError(
                f"GMRES did not converge to {self.tolerance} in {count[0]} iterations"
            )
        return charge

    def _node_exclude(self, points: np.ndarray, image: int) -> np.ndarray:
        """Per target, a node of the image it sits on exactly (-1: none)."""
        # exact hits only: a tiny search radius prunes the k-d tree walk at once
        distance, nearest = self._image_node_trees[image].query(
            points.T, distance_upper_bound=_TINY
        )
        return np.where(distance == 0.0, nearest, -1).astype(np.int64)

    @sp.profile("tree_d_3d.electrodes.panel_field", tags=_PROFILE_TAGS)
    def _panel_field(self, points: np.ndarray, charge: np.ndarray):
        """Raw potential and field at ``points`` from panels carrying ``charge``.

        Far panel copies are summed fast as ``far_points`` Gauss--Legendre point
        charges each (all periods at once with a period); for the near copies those
        point terms are taken back out and the exact segment terms added.
        """
        node_charge = np.repeat(charge, self.far_points) * self._node_weights
        m = points.shape[1]
        phi = np.zeros(m)
        e = np.zeros((2, m))
        for image in range(len(self._image_signs)):
            exclude = None if self.period else self._node_exclude(points, image)
            p, f = self._far_sum(
                points, self._nodes, node_charge, image, exclude, key=("nodes", image)
            )
            phi += p
            e += f
        target, panel, image, shift = self._near_pairs(points)
        if target.size:
            pair_phi, pair_e = self._near_correction(
                points, target, panel, image, shift
            )
            q = charge[panel]
            phi += np.bincount(target, q * pair_phi, m)
            for axis in (0, 1):
                e[axis] += np.bincount(target, q * pair_e[axis], m)
        return phi, e

    def _source_sum(
        self, points: np.ndarray, first_source: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw potential and field at ``points`` of the free charges and their images.

        ``first_source`` says the points are the free charges from that index on,
        each skipping its own singular term (not its images).
        """
        m = points.shape[1]
        phi = np.zeros(m)
        e = np.zeros((2, m))
        if not self._source_charges.size:
            return phi, e
        for image in range(len(self._image_signs)):
            exclude = (
                first_source + np.arange(m)
                if first_source is not None and image == 0
                else None
            )
            p, f = self._far_sum(
                points, self._source_positions, self._source_charges, image, exclude
            )
            phi += p
            e += f
        return phi, e

    def _sources_alone(self, points: np.ndarray, exclude_self: bool = False):
        """Raw potential and field of the free charges alone, in free space."""
        if not self._source_charges.size:
            return np.zeros(points.shape[1]), np.zeros((2, points.shape[1]))
        exclude = np.arange(self._source_charges.size) if exclude_self else None
        phi, e = direct_field(
            points,
            self._source_positions,
            self._source_charges,
            kernel="log",
            exclude=exclude,
        )
        return np.array(phi), np.array(e)

    @sp.profile("tree_d_3d.electrodes.solve", tags=_PROFILE_TAGS)
    def solve(self, source_positions=None, source_charges=None) -> np.ndarray:
        """Solve for the panel charges that make every conductor an equipotential.

        ``source_positions`` ``(2, N)`` and ``source_charges`` ``(N,)`` are free line
        charges (omit both for a pure capacitance problem); with a period or a face
        their images enter too. Returns and caches the panel charges, used by
        `probe` and `force_on_sources`.
        """
        if source_positions is None:
            source_positions = np.zeros((2, 0))
            source_charges = np.zeros(0)
        self._source_positions = np.asarray(source_positions, dtype=float).reshape(
            2, -1
        )
        self._source_charges = np.asarray(source_charges, dtype=float).ravel()
        free_raw, _ = self._shared(
            lambda share: self._source_sum(self.midpoints[:, share]), self.num_panels
        )
        rhs = (self.target_potential - self.offset) / self.k_e - free_raw
        if self.solver == "direct":
            charge = lu_solve(self._lu, rhs)
        else:
            charge = self._solve_iterative(rhs)
        self.charge = charge
        return charge

    def inside(self, points) -> np.ndarray:
        """Per point ``(2, M)``, the index of the solid conductor holding it (-1: none).

        A closed conductor is solid when its loop holds no other conductor and none
        of the free charges passed to `solve`; a loop around them is a shell, whose
        inside is not metal (the outer wall of a capacitor, a round chamber). Even-odd
        rule on the vertex loop, repeated over the periods with a period; open
        conductors hold nothing. A point on a boundary may count either way.
        """
        points = np.asarray(points, dtype=float).reshape(2, -1)
        index = np.full(points.shape[1], -1, dtype=np.int64)
        for k, conductor in enumerate(self.conductors):
            if (
                not self._solid[k]
                or self._loop_holds(conductor, self._source_positions).any()
            ):
                continue
            free = np.flatnonzero(index < 0)
            index[free[self._loop_holds(conductor, points[:, free])]] = k
        return index

    def surface_field(self) -> np.ndarray:
        """The normal field (V/m) just outside each panel, ``(num_panels,)``.

        ``4 pi k_e sigma`` with ``sigma = Q / L`` the panel's charge density: by
        Gauss's law the field leaving a closed conductor along its outward normal
        (the field inside is zero), as needed for the force on an electrode or
        field emission. For a panel of an open conductor (a plate of zero
        thickness) it is the jump of the normal field across the plate. Call
        `solve` first.
        """
        if self.charge is None:
            raise RuntimeError("call solve() before surface_field()")
        return 4.0 * np.pi * self.k_e * self.charge / self.lengths

    def _probe(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`probe` at ``points``, on this rank alone."""
        assert self.charge is not None
        holder = self.inside(points)
        free = holder < 0
        phi = np.empty(points.shape[1])
        e = np.zeros((2, points.shape[1]))
        phi[~free] = [self.conductors[k].potential for k in holder[~free]]
        if free.any():
            p, f = self._panel_field(points[:, free], self.charge)
            p_s, f_s = self._source_sum(points[:, free])
            phi[free] = self.k_e * (p + p_s) + self.offset
            e[:, free] = self.k_e * (f + f_s)
        return phi, e

    def probe(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Potential (V) and field (V/m, shape ``(2, M)``) at ``points`` ``(2, M)``.

        Includes the solved panel charges and the free charges passed to `solve`,
        with their images; ``points`` are probes, not sources, so no self term
        applies. A point inside a closed conductor (`inside`) gets the conductor's
        potential and zero field, the exact values the panel sums approach there.
        On a panel the potential is the conductor's to the discretisation error and
        the field is the mean of the two sides; see `surface_field` for the field
        just outside. Call `solve` first.
        """
        if self.charge is None:
            raise RuntimeError("call solve() before probe()")
        points = np.asarray(points, dtype=float).reshape(2, -1)
        return self._shared(
            lambda share: self._probe(points[:, share]), points.shape[1]
        )

    def panel_field(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Potential (V) and field (V/m) at ``points`` ``(2, M)`` of all but the free charges.

        The induced charge from the last `solve` and the images of the free charges
        passed to it (with a period or a face), without the free charges themselves
        in free space: for codes that add those by a tree code, say. Near panels
        enter exactly, as in `probe`.
        """
        if self.charge is None:
            raise RuntimeError("call solve() before panel_field()")
        points = np.asarray(points, dtype=float).reshape(2, -1)
        charge = self.charge

        def local(share: slice):
            here = points[:, share]
            phi, e = self._panel_field(here, charge)
            phi_s, e_s = self._source_sum(here)
            phi_a, e_a = self._sources_alone(here)
            return (
                self.k_e * (phi + phi_s - phi_a) + self.offset,
                self.k_e * (e + e_s - e_a),
            )

        return self._shared(local, points.shape[1])

    def tabulate(self, lower, upper, shape) -> TabulatedField:
        """`probe` on a regular grid, for fast bilinear interpolation afterwards.

        ``lower`` and ``upper`` are the corners ``(x, y)`` of the rectangle and
        ``shape`` ``(nx, ny)`` the number of grid lines along each axis (at least 2).
        Like `probe` it includes the free charges passed to `solve`, so
        tabulate a static configuration and scale it, for example by an RF voltage.
        """
        nx, ny = (int(n) for n in shape)
        if nx < 2 or ny < 2:
            raise ValueError("a table needs at least 2 grid lines along each axis")
        x = np.linspace(float(lower[0]), float(upper[0]), nx)
        y = np.linspace(float(lower[1]), float(upper[1]), ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        phi, field = self.probe(np.stack((xx.ravel(), yy.ravel())))
        return TabulatedField(
            x=x,
            y=y,
            potential=np.asarray(phi).reshape(nx, ny),
            field=np.asarray(field).reshape(2, nx, ny),
        )

    def potential(self, points) -> np.ndarray:
        """Potential (V) at ``points`` ``(2, M)``; see `probe`."""
        return self.probe(points)[0]

    def field(self, points) -> np.ndarray:
        """Field (V/m, shape ``(2, M)``) at ``points`` ``(2, M)``; see `probe`."""
        return self.probe(points)[1]

    def force_on_sources(self) -> tuple[np.ndarray, np.ndarray]:
        """Potential and field (V/m, shape ``(2, N)``) at the free charges from everything else.

        Excludes each source's own singular self-term, but not its images (with a
        period or a face), so multiplying the field by a source's charge gives the
        physical force on it from the panels, the other sources and the images.
        Call `solve` first.
        """
        if self.charge is None:
            raise RuntimeError("call solve() before force_on_sources()")
        charge = self.charge

        def local(share: slice):
            here = self._source_positions[:, share]
            phi_p, e_p = self._panel_field(here, charge)
            phi_s, e_s = self._source_sum(here, first_source=share.start)
            return self.k_e * (phi_p + phi_s) + self.offset, self.k_e * (e_p + e_s)

        return self._shared(local, self._source_charges.size)

    def total_charge(self, conductor_index: int | None = None):
        """Induced charge of one conductor, or of every conductor if ``index`` is ``None``."""
        if self.charge is None:
            raise RuntimeError("call solve() before total_charge()")
        if conductor_index is None:
            return np.bincount(
                self.group, weights=self.charge, minlength=len(self.conductors)
            )
        return float(self.charge[self.group == conductor_index].sum())
