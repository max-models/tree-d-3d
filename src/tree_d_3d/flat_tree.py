"""The flat quadtree and direct sums: free-space electrostatics for charges constrained to the xy plane.

Two kernels: the three-dimensional softened Coulomb kernel of point charges in the
plane (``kernel="coulomb"``) and the logarithmic Green's function of 2D Poisson, of
line charges along z (``kernel="log"``). No periodic images, conducting walls,
collisions, radiation, or relativistic effects are included.

The simulation stores particles as arrays and evaluates fields with the
kernels ``tree_d_3d.kernels.<name>`` (``quadtree_quadrants``,
``evaluate_quadtree``, ``fmm_m2l``, ...): a flat quadtree (`FlatQuadtree`) or, for small systems,
an exact direct sum. The object tree `ReferenceQuadtree` is the readable
reference implementation of the same algorithm and is kept for inspection and
tests.

Each kernel has a Pyccel, a NumPy and a CUDA version with the same arguments, and
runs the one matching its arrays. `FlatQuadtree` and `direct_field`
work on arrays of cunumpy's active backend: on NumPy the Pyccel versions run (the
NumPy versions without Pyccel), on CuPy the CUDA versions, with the launch sizes
given here; their results are arrays of that backend. The simulation keeps its
particles on the host (each `ChargedBody` is a view of a column) and hands them
to the field solver, which returns to the host.

Per-particle arrays with a component axis are component-major: positions,
targets and the fields at them have shape ``(2, N)`` (``array[0]`` is x of every
particle); per-particle scalars (``charge``, ``phi``, ``exclude``) are ``(N,)``;
the tree-node arrays (``node_center`` ``(nodes, 2)``, ``node_p``, moments...)
are node-major.
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from maybempi import collectives
from methodcard import ConvergenceProblem, Rule, method

from tree_d_3d import _cartesian
from tree_d_3d._profiling import _PROFILE_TAGS
from tree_d_3d._utils import _check, _device, _finite, _host, _points
from tree_d_3d.constants import K_E
from tree_d_3d.kernels import _kernels
from tree_d_3d.quad import Quad

#: Highest expansion order of the CUDA FMM kernels (TREE_D_3D_MAX_ORDER).
MAX_ORDER = 64

#: Default expansion order of the log kernel (complex terms) and of the Coulomb FMM
#: (Cartesian terms up to this total degree), when ``order`` is not given.
LOG_ORDER = 12
COULOMB_ORDER = 4


_KERNELS = ("coulomb", "log")

#: build the tree in one ``quadtree_build`` call on the CPU (off: the GPU's level kernels)
_HOST_BUILD = True


def _compiled_host_build(kernels) -> bool:
    """Whether ``quadtree_build`` runs compiled: else the level kernels (NumPy) are faster.

    Without Pyccel (e.g. in Pyodide) its NumPy version is the Pyccel loop run by the
    interpreter, while the level kernels are vectorized; the uncompiled reference kernels
    (``compiled=False``) keep the single call.
    """
    host = getattr(kernels.quadtree_build, "host_kernel", None)
    return host is None or host.kernel.selected() != "numpy"


#: targets per group of the CPU walk (`FlatQuadtree.evaluate`), at most 62
GROUP_SIZE = 32

#: small tables shared by every tree (M2L matrices), by key
_CACHE: dict = {}


def _direct_rules() -> dict:
    """The SymPy rules of `direct_field`: one source, one target (for the tests)."""
    import sympy

    x, y, x1, y1, q1, eps = sympy.symbols("x y x1 y1 q1 eps", real=True)
    r2 = (x - x1) ** 2 + (y - y1) ** 2 + eps**2
    coulomb = q1 / sympy.sqrt(r2)
    log = -q1 * sympy.log(r2)
    ranges = {"x": (1.0, 2.0), "y": (-1.0, 1.0), "x1": (-1.0, 0.5)}
    ranges |= {"y1": (-1.0, 1.0), "q1": (-2.0, 2.0), "eps": (0.0, 0.5)}

    def field(kernel: str, potential: bool):
        def call(x, y, x1, y1, q1, eps):
            phi, e = direct_field(
                np.array([[x], [y]]),
                np.array([[x1], [y1]]),
                np.array([q1]),
                softening=eps,
                kernel=kernel,
            )
            return float(phi[0]) if potential else float(e[0, 0])

        return call

    return {
        "coulomb_potential": Rule(coulomb, field("coulomb", True), ranges),
        "coulomb_field": Rule(-sympy.diff(coulomb, x), field("coulomb", False), ranges),
        "log_potential": Rule(log, field("log", True), ranges),
        "log_field": Rule(-sympy.diff(log, x), field("log", False), ranges),
    }


@method(
    equations={
        "coulomb_potential": r"\phi(\mathbf r) = \sum_j \frac{q_j}{\sqrt{|\mathbf r - \mathbf r_j|^2 + \varepsilon^2}}",
        "coulomb_field": r"\mathbf E(\mathbf r) = -\nabla\phi = \sum_j \frac{q_j\,(\mathbf r - \mathbf r_j)}{\left(|\mathbf r - \mathbf r_j|^2 + \varepsilon^2\right)^{3/2}}",
        "log_potential": r"\phi(\mathbf r) = -\sum_j q_j \ln\left(|\mathbf r - \mathbf r_j|^2 + \varepsilon^2\right)",
        "log_field": r"\mathbf E(\mathbf r) = 2\sum_j \frac{q_j\,(\mathbf r - \mathbf r_j)}{|\mathbf r - \mathbf r_j|^2 + \varepsilon^2}",
    },
    assumptions=(
        r"raw kernels: the callers multiply by $k_e = 1/(4\pi\varepsilon_0)$",
        r"`coulomb`: point charges confined to the plane; `log`: line charges along $z$ per unit length, $\phi = -(\lambda/2\pi\varepsilon_0)\ln r$",
        r"Plummer softening $\varepsilon$; the excluded source and zero charges are skipped; a charged pair at $r^2 + \varepsilon^2 = 0$ raises",
    ),
    symbolic=_direct_rules,
)
@sp.profile("tree_d_3d.flat_tree.direct_field", tags=_PROFILE_TAGS)
def direct_field(
    targets,
    pos,
    charge,
    *,
    softening=0.0,
    exclude=None,
    compiled=True,
    kernel="coulomb",
) -> tuple[np.ndarray, np.ndarray]:
    """Exact potential ``(M,)`` and field ``(2, M)`` at ``targets`` ``(2, M)``, without ``k_e``.

    ``pos`` ``(2, N)`` and ``charge`` ``(N,)`` are the sources. ``exclude[i]``
    is the index of a source skipped for target ``i`` (-1 for none). ``kernel``
    is ``"coulomb"`` or ``"log"`` (line charges), as for `FlatQuadtree`.
    """
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}")
    module = xp if compiled else np
    targets = _points(targets, "targets", compiled=compiled)
    pos = _points(pos, "pos", compiled=compiled)
    charge = _device(charge, compiled=compiled)
    n_targets = targets.shape[1]
    if exclude is None:
        exclude = module.full(n_targets, -1, dtype=np.int64)
    exclude = _device(exclude, np.int64, compiled=compiled)
    if not n_targets or not pos.shape[1]:
        return module.zeros(n_targets), module.zeros((2, n_targets))
    kernels = _kernels(compiled)
    phi = module.zeros(n_targets)
    field = module.zeros((2, n_targets))
    status = module.zeros(1, dtype=np.int64)
    direct = kernels.direct_log_sum if kernel == "log" else kernels.direct_sum
    direct(targets, exclude, pos, charge, softening, phi, field, status)
    _check(status)
    return phi, field


class FlatNode:
    """Read-only view of one node of a `FlatQuadtree`."""

    def __init__(self, tree: FlatQuadtree, index: int) -> None:
        self.tree = tree
        self.index = index

    @property
    def quad(self) -> Quad:
        x, y = self.tree.node_center[self.index]
        return Quad(float(x), float(y), float(self.tree.node_half[self.index]))

    @property
    def count(self) -> int:
        return int(self.tree.node_end[self.index] - self.tree.node_start[self.index])

    @property
    def total_charge(self) -> float:
        return float(self.tree.node_q[self.index])

    @property
    def total_abs_charge(self) -> float:
        return float(self.tree.node_qabs[self.index])

    @property
    def dipole(self) -> np.ndarray:
        return _host(self.tree.node_p[self.index]).copy()

    @property
    def second_moment(self) -> np.ndarray:
        sxx, sxy, syy = (float(v) for v in _host(self.tree.node_s[self.index]))
        return np.array([[sxx, sxy], [sxy, syy]])

    @property
    def particles(self) -> np.ndarray:
        """Indices of the particles inside this node."""
        tree = self.tree
        start, end = int(tree.node_start[self.index]), int(tree.node_end[self.index])
        return _host(tree.perm[start:end])

    def is_external(self) -> bool:
        return int(self.tree.node_child[self.index]) < 0

    @property
    def children(self) -> tuple[FlatNode | None, ...]:
        first = int(self.tree.node_child[self.index])
        if first < 0:
            return (None, None, None, None)
        return tuple(FlatNode(self.tree, first + k) for k in range(4))


_NODE_ARRAYS = (
    "node_center",
    "node_half",
    "node_start",
    "node_end",
    "node_child",
    "node_depth",
)


def _binomial_table(size: int) -> np.ndarray:
    """C(n, k) for 0 <= n, k < size, as floats."""
    table = np.zeros((size, size))
    for row in range(size):
        table[row, 0] = 1.0
        for col in range(1, row + 1):
            table[row, col] = table[row - 1, col - 1] + table[row - 1, col]
    return table


def _m2l_coefficients(order: int, symmetric: bool) -> np.ndarray:
    """The real M2L matrix of ``fmm_m2l_host``: ``coeff[k, l]``, see its docstring."""
    binomial = _binomial_table(2 * order + 1)
    coeff = np.zeros((order + 1, order + 1))
    for m in range(1, order + 1):
        coeff[0, m] = 1.0 / m
    for k in range(1, order + 1):
        coeff[k, 0] = 1.0 / k
        for m in range(1, order + 1):
            if k <= (order - m if symmetric else order):
                coeff[k, m] = binomial[k + m - 1, m] / k
    return coeff


def _moment_rules() -> dict:
    """The SymPy rules of the cell moments: two charges in a root cell (for the tests)."""
    import sympy

    cx, cy, x1, y1, x2, y2 = sympy.symbols("cx cy x1 y1 x2 y2", real=True)
    q1, q2 = sympy.symbols("q1 q2", real=True)
    terms = ((q1, x1, y1), (q2, x2, y2))
    s_xy = sum(q * (x - cx) * (y - cy) for q, x, y in terms)
    a_3 = sum(q * (x - cx + sympy.I * (y - cy)) ** 3 for q, x, y in terms)
    # absolute positions, inside the cell of half-width 1 about (cx, cy)
    ranges = {name: (-0.85, 0.85) for name in ("x1", "y1", "x2", "y2")}
    ranges |= {"cx": (-0.1, 0.1), "cy": (-0.1, 0.1), "q1": (-2.0, 2.0)}
    ranges |= {"q2": (-2.0, 2.0)}

    def root(kernel: str):
        def call(cx, cy, x1, y1, x2, y2, q1, q2):
            tree = FlatQuadtree(
                np.array([[x1, x2], [y1, y2]]),
                np.array([q1, q2]),
                Quad(cx, cy, 1.0),
                leaf_capacity=1,
                kernel=kernel,
                order=3,
            )
            if kernel == "log":
                return float(tree.moment_im[0, 3])
            return float(tree.node_s[0, 1])

        return call

    return {
        "coulomb_moments": Rule(s_xy, root("coulomb"), ranges),
        "log_moments": Rule(sympy.im(sympy.expand(a_3)), root("log"), ranges),
    }


def _multipole_rules() -> dict:
    """The SymPy rules of the far-field expansions: an accepted root cell (for the tests).

    Two charges in the cell ``[-1, 1]^2``, split into leaves, and a target beyond
    ``2 h / theta``, so the walk takes the root's expansion: through the quadrupole for
    the Coulomb kernel, to order ``p = 3`` for the log kernel.
    """
    import sympy

    x, y, eps = sympy.symbols("x y eps", real=True)
    x1, y1, x2, y2 = sympy.symbols("x1 y1 x2 y2", real=True)
    q1, q2 = sympy.symbols("q1 q2", real=True)
    terms = ((q1, x1, y1), (q2, x2, y2))
    charge = q1 + q2
    p_x = sum(q * xj for q, xj, _ in terms)
    p_y = sum(q * yj for q, _, yj in terms)
    s_xx = sum(q * xj * xj for q, xj, _ in terms)
    s_xy = sum(q * xj * yj for q, xj, yj in terms)
    s_yy = sum(q * yj * yj for q, _, yj in terms)
    rho = sympy.sqrt(x**2 + y**2 + eps**2)
    quadratic = s_xx * x**2 + 2 * s_xy * x * y + s_yy * y**2
    coulomb = (
        charge / rho
        + (p_x * x + p_y * y) / rho**3
        + (3 * quadratic / rho**5 - (s_xx + s_yy) / rho**3) / 2
    )
    w = 1 / (x + sympy.I * y)
    series = sum(
        sum(q * (xj + sympy.I * yj) ** n for q, xj, yj in terms) * w**n / n
        for n in range(1, 4)
    )
    log = -charge * sympy.log(x**2 + y**2) + 2 * sympy.re(sympy.expand_complex(series))
    ranges = {name: (-0.9, 0.9) for name in ("x1", "y1", "x2", "y2")}
    ranges |= {"x": (2.5, 4.0), "y": (2.5, 4.0), "eps": (0.0, 0.5)}
    ranges |= {"q1": (-2.0, 2.0), "q2": (-2.0, 2.0)}

    def walk(kernel: str):
        def call(x, y, x1, y1, x2, y2, q1, q2, eps=0.0):
            tree = FlatQuadtree(
                np.array([[x1, x2], [y1, y2]]),
                np.array([q1, q2]),
                Quad(0.0, 0.0, 1.0),
                leaf_capacity=1,
                kernel=kernel,
                order=3,
            )
            phi, _ = tree.evaluate(np.array([[x], [y]]), theta=1.0, softening=eps)
            return float(phi[0])

        return call

    return {
        "coulomb_multipole": Rule(coulomb, walk("coulomb"), ranges),
        "log_multipole": Rule(log, walk("log"), ranges),
    }


def _walk_problem() -> ConvergenceProblem:
    """The far-field error of one accepted cell against the size of its charges.

    Twelve charges at ``s`` times fixed offsets in the root cell, sixteen targets on a
    circle of radius 4: the walk takes the root's expansion, so the error is its
    truncation error, ``O(s^3)`` through the quadrupole and ``O(s^{p+1})`` for the log
    kernel at ``p = 2``.
    """
    rng = np.random.default_rng(1)
    offsets = rng.uniform(-0.5, 0.5, (2, 12))
    charges = rng.normal(0.3, 1.0, 12)
    angle = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    targets = 4.0 * np.stack((np.cos(angle), np.sin(angle)))

    def error(size: float) -> dict[str, float]:
        errors = {}
        for kernel in _KERNELS:
            pos = size * offsets
            tree = FlatQuadtree(
                pos,
                charges,
                Quad(0.0, 0.0, 1.0),
                leaf_capacity=1,
                kernel=kernel,
                order=2,
            )
            _, field = tree.evaluate(targets, theta=0.9)
            _, exact = direct_field(targets, pos, charges, kernel=kernel)
            errors[kernel] = float(np.max(np.abs(_host(field) - _host(exact))))
        return errors

    return ConvergenceProblem(error, steps=[1.0, 0.5, 0.25, 0.125])


def _fmm_problem() -> ConvergenceProblem:
    """The FMM error against the size of the charge clusters, at ``p = 2``.

    Four clusters of three charges at the centres of depth-2 cells, two in opposite
    corners of the root: those two interact by one M2L, the rest by P2P. Shrinking the
    clusters by ``s`` leaves the tree as it is, so the error is the truncation error of
    that M2L and the L2P, ``O(s^{p+1})`` in the potential and ``O(s^p)`` in the field.
    """
    rng = np.random.default_rng(2)
    centres = np.array([[-0.75, -0.25, 0.75, 0.25], [0.75, 0.25, -0.75, -0.25]])
    offsets = rng.uniform(-0.2, 0.2, (2, 4, 3))
    charges = rng.normal(0.3, 1.0, 12)
    everyone = np.arange(12)

    def error(size: float) -> dict[str, float]:
        pos = (centres[:, :, None] + size * offsets).reshape(2, -1)
        tree = FlatQuadtree(
            pos, charges, Quad(0.0, 0.0, 1.0), leaf_capacity=3, kernel="log", order=2
        )
        phi, field = tree.fmm(theta=0.5)
        phi_exact, exact = direct_field(
            pos, pos, charges, exclude=everyone, kernel="log"
        )
        return {
            "potential": float(np.max(np.abs(_host(phi) - _host(phi_exact)))),
            "field": float(np.max(np.abs(_host(field) - _host(exact)))),
        }

    return ConvergenceProblem(error, steps=[0.5, 0.25, 0.125, 0.0625])


def _reference_rules() -> dict:
    """The SymPy rule of `FlatQuadtree.field_at` for one line charge (for the tests)."""
    import sympy

    x, y, q, k_e, r_0 = sympy.symbols("x y q k_e r_0", positive=True)
    rule = -2 * k_e * q * sympy.log(sympy.sqrt(x**2 + y**2) / r_0)

    def call(x, y, q, k_e, r_0):
        tree = FlatQuadtree(
            np.zeros((2, 1)), np.array([q]), Quad(0.0, 0.0, 1.0), kernel="log"
        )
        phi, _ = tree.field_at(np.array([[x], [y]]), k_e=k_e, reference_radius=r_0)
        return float(phi[0])

    return {"log_potential": Rule(rule, call, {"r_0": (0.5, 3.0)})}


@method(
    equations={
        "split": r"N_\mathrm{cell} > N_\mathrm{leaf},\ \ \mathrm{depth} < d_\mathrm{max} \ \Rightarrow\ \text{four children of half-width } h/2,\ \mathbf c \pm h/2",
        "quadrant": r"k = [x > x_\mathrm{mid}] + 2\,[y < y_\mathrm{mid}] \qquad (0\text{ upper left},\ 1\text{ upper right},\ 2\text{ lower left},\ 3\text{ lower right})",
        "coulomb_moments": r"Q = \sum_j q_j, \qquad \mathbf p = \sum_j q_j \mathbf d_j, \qquad \mathsf S = \sum_j q_j\, \mathbf d_j \mathbf d_j^{\mathsf T}, \qquad \mathbf d_j = \mathbf r_j - \mathbf c",
        "log_moments": r"a_n = \sum_j q_j (z_j - c)^n, \qquad n = 0, \dots, p, \qquad z = x + iy",
    },
    assumptions=(
        r"a square root cell; built level by level, breadth first, with a stable partition: the order inside a cell is the input order",
        r"moments about the geometric cell centre $\mathbf c$, summed directly from each node's particles (no M2M); $\mathsf S$ is raw, not traceless",
        r"cells with $\sum_j |q_j| = 0$ are skipped by the walks; a particle on a dividing line goes left or up",
    ),
    symbolic=_moment_rules,
)
class FlatQuadtree:
    """Array quadtree for point charges in a plane or for line charges.

    Same tree and opening rule as `ReferenceQuadtree`: nodes holding
    more than ``leaf_capacity`` particles split into four until
    ``max_depth``, cells containing the target are always opened, and the
    opening test uses unsoftened distance. ``positions`` is ``(2, N)``
    (component-major) and ``charges`` ``(N,)``; both are copied, so rebuild
    after moving or changing a source.

    ``kernel="coulomb"`` (the default) is the 3D point-charge interaction,
    with signed moments through quadrupole order. ``kernel="log"`` treats
    each charge as an infinite line charge per unit length along z, the
    interaction of 2D Cartesian PIC: potential ``-2 k_e q ln r`` and field
    ``2 k_e q r / r**2``. It uses complex multipole expansions of ``order``
    terms, so accuracy can be raised by raising ``order`` as well as by
    lowering ``theta``.

    ``comm`` (an MPI communicator) shares the cells' moments out over its ranks,
    each summing a run of cells with an even share of the particles; every rank
    passes the same particles and gets the whole tree.
    """

    def _empty(self, root_quad: Quad) -> None:
        # Pyccel rejects zero-length arrays, so build the lone root here.
        self._traversals: dict = {}
        self._sorted = None
        self._cart_moments = None
        self.perm = self._xp.empty(0, dtype=np.int64)
        self.node_count = 1
        self.node_center = self._xp.array(
            [[root_quad.xmid, root_quad.ymid]], dtype=float
        )
        self.node_half = self._xp.array([root_quad.half], dtype=float)
        self.node_start = self._xp.zeros(1, dtype=np.int64)
        self.node_end = self._xp.zeros(1, dtype=np.int64)
        self.node_child = self._xp.full(1, -1, dtype=np.int64)
        self.node_depth = self._xp.zeros(1, dtype=np.int64)
        self.levels = np.array([0, 1])
        self.node_q = self._xp.zeros(1)
        self.node_qabs = self._xp.zeros(1)
        self.node_p = self._xp.zeros((1, 2))
        self.node_s = self._xp.zeros((1, 3))
        self.moment_re = self._xp.zeros((1, self.order + 1))
        self.moment_im = self._xp.zeros((1, self.order + 1))

    def _build_host(self, root_quad: Quad, capacity: int) -> int:
        """`_build` on the CPU: the whole build in one ``quadtree_build`` call."""
        n = self.pos.shape[1]
        int_scratch = [np.empty(n, dtype=np.int64) for _ in range(5)]
        perm, slot_node, quadrant, new_perm, new_slot_node = int_scratch
        # one spare column, as Pyccel takes a (2, 1) array for a Fortran-ordered one
        new_sorted_pos = np.empty((2, n + 1))
        sizes = np.zeros(3, dtype=np.int64)
        while True:
            sorted_pos = np.zeros((2, n + 1))
            sorted_pos[:, :n] = self.pos
            self.node_center = np.empty((capacity, 2))
            self.node_half = np.empty(capacity)
            self.node_start = np.empty(capacity, dtype=np.int64)
            self.node_end = np.empty(capacity, dtype=np.int64)
            self.node_child = np.empty(capacity, dtype=np.int64)
            self.node_depth = np.empty(capacity, dtype=np.int64)
            self.node_center[0] = (root_quad.xmid, root_quad.ymid)
            self.node_half[0] = root_quad.half
            levels = np.empty(capacity + 1, dtype=np.int64)
            self._kernels.quadtree_build(
                sorted_pos,
                self.leaf_capacity,
                self.max_depth,
                self.node_center,
                self.node_half,
                self.node_start,
                self.node_end,
                self.node_child,
                self.node_depth,
                perm,
                levels,
                slot_node,
                quadrant,
                np.empty(capacity, dtype=np.int64),
                new_sorted_pos,
                new_perm,
                new_slot_node,
                sizes,
            )
            if not sizes[2]:
                break
            capacity *= 2
        self.perm = perm
        self.levels = levels[: sizes[1] + 1].copy()
        # the build leaves the positions in tree order
        self._sorted_pos = sorted_pos
        return int(sizes[0])

    def _build(self, root_quad: Quad) -> int:
        """Split the nodes level by level; returns the node count.

        On the CPU the whole build is one ``quadtree_build`` call (`_build_host`); on
        the GPU (and with ``_HOST_BUILD`` off or without Pyccel) every level is one ``quadtree_quadrants``
        and one ``quadtree_scatter`` launch over the particles with prefix sums in
        between. Either way the nodes are numbered breadth first and each node's
        particles keep their input order, as in a serial build.
        ``self.levels[d]:self.levels[d + 1]`` are the nodes at depth ``d``.
        """
        x, k = self._xp, self._kernels
        n = self.pos.shape[1]
        capacity = 4 * (2 * n // self.leaf_capacity + 16)
        if _HOST_BUILD and isinstance(self.pos, np.ndarray) and _compiled_host_build(k):
            return self._build_host(root_quad, capacity)
        self.node_center = x.empty((capacity, 2))
        self.node_half = x.empty(capacity)
        self.node_start = x.empty(capacity, dtype=np.int64)
        self.node_end = x.empty(capacity, dtype=np.int64)
        self.node_child = x.empty(capacity, dtype=np.int64)
        self.node_depth = x.empty(capacity, dtype=np.int64)
        self.node_center[0, 0] = root_quad.xmid
        self.node_center[0, 1] = root_quad.ymid
        self.node_half[0] = root_quad.half
        self.node_start[0] = 0
        self.node_end[0] = n
        self.node_depth[0] = 0
        perm = x.arange(n, dtype=np.int64)
        slot_node = x.zeros(n, dtype=np.int64)
        new_perm = x.empty(n, dtype=np.int64)
        new_slot_node = x.empty(n, dtype=np.int64)
        quadrant = x.empty(n, dtype=np.int64)
        quadrants = x.arange(4)
        x_sign = x.asarray([-1.0, 1.0, -1.0, 1.0])
        y_sign = x.asarray([1.0, 1.0, -1.0, -1.0])
        levels = [0, 1]
        while True:
            lo, hi = levels[-2], levels[-1]
            size = self.node_end[lo:hi] - self.node_start[lo:hi]
            split = (size > self.leaf_capacity) & (
                self.node_depth[lo:hi] < self.max_depth
            )
            n_split = int(split.sum())
            if not n_split:
                self.node_child[lo:hi] = -1
                break
            count = hi + 4 * n_split
            if count > capacity:
                capacity = max(2 * capacity, count)
                for name in _NODE_ARRAYS:
                    old = getattr(self, name)
                    new = x.empty((capacity, *old.shape[1:]), dtype=old.dtype)
                    new[:hi] = old[:hi]
                    setattr(self, name, new)
            rank = x.cumsum(split, dtype=np.int64) - split
            self.node_child[lo:hi] = x.where(split, hi + 4 * rank, -1)
            parents = lo + x.flatnonzero(split)
            children = self.node_child[parents][:, None] + quadrants
            half = 0.5 * self.node_half[parents][:, None]
            self.node_half[children] = half
            self.node_depth[children] = self.node_depth[parents][:, None] + 1
            self.node_center[children, 0] = (
                self.node_center[parents, 0][:, None] + x_sign * half
            )
            self.node_center[children, 1] = (
                self.node_center[parents, 1][:, None] + y_sign * half
            )
            k.quadtree_quadrants(
                self.pos, perm, slot_node, self.node_center, self.node_child, quadrant
            )
            # below[k, q]: slots up to k in quadrant q
            below = x.cumsum(quadrant[:, None] == quadrants, axis=0, dtype=np.int64)
            start = self.node_start[parents]
            before = below[x.maximum(start - 1, 0)] * (start > 0)[:, None]
            in_child = below[self.node_end[parents] - 1] - before
            child_start = start[:, None] + x.cumsum(in_child, axis=1) - in_child
            self.node_start[children] = child_start
            self.node_end[children] = child_start + in_child
            k.quadtree_scatter(
                perm,
                slot_node,
                quadrant,
                below,
                self.node_start,
                self.node_child,
                new_perm,
                new_slot_node,
            )
            perm, new_perm = new_perm, perm
            slot_node, new_slot_node = new_slot_node, slot_node
            levels.append(count)
        self.perm = perm
        self.levels = np.asarray(levels)
        return hi

    def _share(self, work) -> slice:
        """This rank's contiguous run of items with about an even share of ``work``."""
        work = np.asarray(_host(work), dtype=float)
        if not collectives.is_parallel(self.comm):
            return slice(0, work.size)
        ranks, me = collectives.size(self.comm), collectives.rank(self.comm)
        cumulative = np.cumsum(work)
        total = cumulative[-1] if work.size else 0.0
        cuts = np.searchsorted(cumulative, total * np.arange(1, ranks) / ranks)
        bounds = np.concatenate(([0], cuts, [work.size]))
        return slice(int(bounds[me]), int(bounds[me + 1]))

    def _moment_share(self) -> slice:
        """This rank's run of cells, holding about an even share of the particle visits."""
        return self._share(self.node_end - self.node_start)

    def _gather_moments(self, share: slice, outputs: tuple) -> None:
        """Fill ``outputs`` (per-cell arrays) on every rank from each rank's ``share``."""
        if not collectives.is_parallel(self.comm):
            return
        columns = [np.asarray(out).reshape(self.node_count, -1) for out in outputs]
        mine = np.concatenate([c[share] for c in columns], axis=1)
        gathered, _ = collectives.gather_columns(self.comm, mine.T)
        first = 0
        for out, column in zip(outputs, columns):
            width = column.shape[1]
            out[...] = gathered[first : first + width].T.reshape(out.shape)
            first += width

    def _binomial(self) -> np.ndarray:
        """``C(n, k)`` for ``n, k <= 2 p``, as floats (shared by all trees)."""
        key = ("binomial", self.order)
        if key not in _CACHE:
            _CACHE[key] = _binomial_table(2 * self.order + 1)
        return _CACHE[key]

    def _moments(self) -> None:
        """The cells' charges and multipole moments from ``self.charge``.

        Each cell sums its own particles, so with ``comm`` every rank sums its run
        of cells (`_moment_share`) and the ranks gather the others'.
        """
        count = self.node_count
        kernel = self.kernel
        share = self._moment_share()
        with sp.region("tree_d_3d.flat_tree.moments", tags=_PROFILE_TAGS):
            self.node_q = self._xp.empty(count)
            self.node_qabs = self._xp.empty(count)
            self.node_p = self._xp.empty((count, 2))
            self.node_s = self._xp.empty((count, 3))
            self._stack = self._xp.empty(
                3 * int(self.node_depth.max()) + 8, dtype=np.int64
            )
            # on the CPU of one process: one upward pass (leaves, then M2M)
            upward = isinstance(self.pos, np.ndarray) and not collectives.is_parallel(
                self.comm
            )
            levels = np.asarray(self.levels, dtype=np.int64)
            if kernel == "log":
                self.moment_re = self._xp.empty((count, self.order + 1))
                self.moment_im = self._xp.empty((count, self.order + 1))
                outputs = (self.node_qabs, self.moment_re, self.moment_im)
                if upward:
                    self._kernels.log_moments_upward(
                        self.pos,
                        self.charge,
                        self.perm,
                        levels,
                        self.node_center,
                        self.node_start,
                        self.node_end,
                        self.node_child,
                        self._binomial(),
                        *outputs,
                    )
                elif share.stop > share.start:
                    self._kernels.log_quadtree_moments(
                        self.pos,
                        self.charge,
                        self.perm,
                        share.stop - share.start,
                        self.node_center[share],
                        self.node_start[share],
                        self.node_end[share],
                        *(out[share] for out in outputs),
                        n_threads=share.stop - share.start,
                    )
                self._gather_moments(share, outputs)
                self.node_q[:] = self.moment_re[:, 0]
                self.node_p[:, 0] = self.moment_re[:, 1]
                self.node_p[:, 1] = self.moment_im[:, 1]
                return
            outputs = (self.node_q, self.node_qabs, self.node_p, self.node_s)
            if upward:
                self._kernels.quadtree_moments_upward(
                    self.pos,
                    self.charge,
                    self.perm,
                    levels,
                    self.node_center,
                    self.node_start,
                    self.node_end,
                    self.node_child,
                    *outputs,
                )
            elif share.stop > share.start:
                self._kernels.quadtree_moments(
                    self.pos,
                    self.charge,
                    self.perm,
                    share.stop - share.start,
                    self.node_center[share],
                    self.node_start[share],
                    self.node_end[share],
                    *(out[share] for out in outputs),
                    n_threads=share.stop - share.start,
                )
            self._gather_moments(share, outputs)

    def __init__(
        self,
        positions,
        charges,
        root_quad: Quad,
        *,
        leaf_capacity: int = 8,
        max_depth: int = 48,
        compiled: bool = True,
        kernel: str = "coulomb",
        order: int | None = None,
        symmetric: bool = False,
        comm=None,
    ) -> None:
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {_KERNELS}")
        if order is None:
            order = LOG_ORDER if kernel == "log" else COULOMB_ORDER
        if isinstance(order, bool) or int(order) != order or order < 1:
            raise ValueError("order must be a positive integer")
        if order > MAX_ORDER:
            raise ValueError(f"order must be at most {MAX_ORDER}, got {order}")
        self.kernel = kernel
        self.order = int(order)
        #: the MPI communicator whose ranks share the moments (``None``: one process)
        self.comm = comm if compiled else None
        #: FMM translations truncated by total degree (exact pair antisymmetry).
        self.symmetric = bool(symmetric)
        self._compiled = bool(compiled)
        # the active backend's arrays; NumPy for the uncompiled (Python) kernels
        self._xp = xp if compiled else np
        _finite("half_size", root_quad.half, positive=True)
        _finite("xmid", root_quad.xmid)
        _finite("ymid", root_quad.ymid)
        for name, value in (("leaf_capacity", leaf_capacity), ("max_depth", max_depth)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.pos = _points(positions, "positions", compiled=compiled).copy()
        self.charge = _device(charges, compiled=compiled).reshape(-1).copy()
        n = self.pos.shape[1]
        if len(self.charge) != n:
            raise ValueError("positions and charges must have the same length")
        if not bool(self._xp.all(self._xp.isfinite(self.pos))):
            raise ValueError("Particle position must be finite")
        lo = [[root_quad.xmid - root_quad.half], [root_quad.ymid - root_quad.half]]
        hi = [[root_quad.xmid + root_quad.half], [root_quad.ymid + root_quad.half]]
        lo, hi = self._xp.asarray(lo), self._xp.asarray(hi)
        if bool(self._xp.any(self.pos < lo)) or bool(self._xp.any(self.pos > hi)):
            raise ValueError("Particle lies outside tree bounds")
        self.leaf_capacity = int(leaf_capacity)
        self.max_depth = int(max_depth)
        self._kernels = _kernels(compiled)
        if not n:
            self._empty(root_quad)
            return
        with sp.region("tree_d_3d.flat_tree.build", tags=_PROFILE_TAGS):
            count = self._build(root_quad)
            self.node_count = int(count)
            for name in _NODE_ARRAYS:
                setattr(self, name, getattr(self, name)[:count].copy())
        self._traversals: dict = {}
        self._sorted = None
        self._cart_moments = None
        self._moments()

    def _m2l_coefficients(self) -> np.ndarray:
        """The real M2L matrix of ``fmm_m2l_host`` for this order and truncation."""
        key = ("m2l_coefficients", self.order, self.symmetric)
        if key not in _CACHE:
            _CACHE[key] = _m2l_coefficients(self.order, self.symmetric)
        return _CACHE[key]

    def _m2l_rows(self, theta: float, m2l, share: slice):
        """The M2L partners of every cell (both directions of ``m2l[share]``), as rows."""
        key = ("m2l_rows", theta, share.start, share.stop)
        if key not in self._traversals:
            rows = np.zeros(self.node_count + 1, dtype=np.int64)
            count = share.stop - share.start
            partners = np.empty(max(2 * count, 1), dtype=np.int64)
            cursor = np.empty(self.node_count, dtype=np.int64)
            self._kernels.pair_csr(m2l[share], count, 1, rows, partners, cursor)
            self._traversals[key] = (rows, partners)
        return self._traversals[key]

    def _fmm_shares(self, theta: float) -> tuple[slice, slice, slice]:
        """This rank's M2L pairs, leaves (for L2P) and P2P pairs, each an even share."""
        m2l, p2p, sizes = self._traversals[theta]
        if not collectives.is_parallel(self.comm):
            return (
                slice(0, int(sizes[0])),
                slice(0, self.node_count),
                slice(0, int(sizes[1])),
            )
        key = ("shares", theta)
        if key not in self._traversals:
            counts = _host(self.node_end - self.node_start)
            leaves = np.where(_host(self.node_child) < 0, counts, 0)
            pairs = _host(p2p[: int(sizes[1])])
            a, b = counts[pairs[:, 0]], counts[pairs[:, 1]]
            near = np.where(pairs[:, 0] == pairs[:, 1], 0.5 * a * a, a * b)
            self._traversals[key] = (
                self._share(np.ones(int(sizes[0]))),
                self._share(leaves),
                self._share(near),
            )
        return self._traversals[key]

    def set_charges(self, charges) -> None:
        """New charges ``(N,)`` for the same particles, without rebuilding the tree.

        The cells, their particles and the FMM's interaction lists depend only on the
        positions; this recomputes the cells' charges and moments, which costs a
        fraction of a build. For repeated sums over fixed points, such as the products
        of an iterative solver.
        """
        charges = _device(charges, compiled=self._compiled).reshape(-1)
        if charges.shape[0] != self.pos.shape[1]:
            raise ValueError("charges must have one entry per particle")
        self.charge[...] = charges
        self._sorted = None
        self._cart_moments = None
        if self.pos.shape[1]:
            self._moments()

    @property
    def root(self) -> FlatNode:
        return FlatNode(self, 0)

    def collect_nodes(self) -> list[FlatNode]:
        """Views of every node that holds at least one particle."""
        occupied = _host(self._xp.flatnonzero(self.node_end > self.node_start))
        return [FlatNode(self, int(i)) for i in occupied]

    def _sorted_sources(self):
        """The sources' positions and charges in tree order (``perm``), kept until they change."""
        if self._sorted is None:
            spos = getattr(self, "_sorted_pos", None)
            if spos is None:
                # one spare column: Pyccel takes a (2, 1) array for a Fortran-ordered one
                n = self.pos.shape[1]
                spos = np.zeros((2, n + 1))
                spos[:, :n] = self.pos[:, self.perm]
            self._sorted = (spos, np.ascontiguousarray(self.charge[self.perm]))
        return self._sorted

    def _target_order(self, targets: np.ndarray) -> np.ndarray:
        """An order of the targets in which consecutive ones are close.

        The tree order for the sources themselves, else the Morton (Z) order of their
        positions.
        """
        if targets.shape == self.pos.shape and np.array_equal(targets, self.pos):
            return self.perm
        lo = targets.min(axis=1, keepdims=True)
        span = float(np.max(targets.max(axis=1, keepdims=True) - lo))
        cells = ((targets - lo) * ((65535.0 / span) if span > 0 else 0.0)).astype(
            np.uint64
        )
        key = np.zeros(targets.shape[1], dtype=np.uint64)
        for axis, shift in ((0, 1), (1, 0)):
            v = cells[axis]
            v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
            v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
            v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
            v = (v | (v << np.uint64(1))) & np.uint64(0x55555555)
            key |= v << np.uint64(shift)
        return np.argsort(key, kind="stable")

    def _evaluate_groups(
        self, targets, exclude, theta, softening, potential, phi, field, status
    ):
        """`evaluate` on the CPU: the walk of ``evaluate_groups`` by groups of targets.

        The groups are runs of nearby targets; each target gets the same terms as in its
        own walk.
        """
        order = self._target_order(targets)
        n_targets = targets.shape[1]
        spos, scharge = self._sorted_sources()
        log_kernel = self.kernel == "log"
        empty = np.zeros((1, 1))
        # one spare column: Pyccel takes a (2, 1) array for a Fortran-ordered one
        sorted_targets = np.empty((2, n_targets + 1))
        sorted_targets[:, :n_targets] = targets[:, order]
        phi_sorted = np.empty(n_targets)
        field_sorted = np.empty((2, n_targets + 1))
        self._kernels.evaluate_groups(
            sorted_targets,
            np.ascontiguousarray(exclude[order]),
            np.append(np.arange(0, n_targets, GROUP_SIZE), n_targets),
            1 if log_kernel else 0,
            1 if potential else 0,
            spos,
            scharge,
            self.perm,
            self.node_center,
            self.node_half,
            self.node_start,
            self.node_end,
            self.node_child,
            self.node_q,
            self.node_qabs,
            self.node_p,
            self.node_s,
            self.moment_re if log_kernel else empty,
            self.moment_im if log_kernel else empty,
            float(theta),
            float(softening),
            self._stack.shape[0],
            phi_sorted,
            field_sorted,
        )
        phi[order] = phi_sorted
        field[:, order] = field_sorted[:, :n_targets]
        bad = np.flatnonzero(phi == 1.0e300)
        if bad.size:
            phi[bad[0]] = 0.0
            status[0] = bad[0] + 1

    @method(
        equations={
            "acceptance": r"\text{accept the cell if } \mathbf r \notin \text{cell} \ \text{ and }\ 2h < \theta D, \qquad D = |\mathbf r - \mathbf c|\ \text{(unsoftened)}",
            "coulomb_multipole": r"\phi \mathrel{+}= \frac{Q}{\rho} + \frac{\mathbf p\cdot\mathbf R}{\rho^3} + \frac12\left(\frac{3\,\mathbf R^{\mathsf T}\mathsf S\mathbf R}{\rho^5} - \frac{\operatorname{tr}\mathsf S}{\rho^3}\right), \qquad \mathbf R = \mathbf r - \mathbf c, \quad \rho = \sqrt{D^2 + \varepsilon^2}",
            "coulomb_field": r"\mathbf E \mathrel{+}= \frac{Q\mathbf R}{\rho^3} + \frac{3(\mathbf p\cdot\mathbf R)\mathbf R}{\rho^5} - \frac{\mathbf p}{\rho^3} + \frac{15}{2}\frac{(\mathbf R^{\mathsf T}\mathsf S\mathbf R)\mathbf R}{\rho^7} - \frac{3\,\mathsf S\mathbf R}{\rho^5} - \frac32\frac{\operatorname{tr}\mathsf S\,\mathbf R}{\rho^5}",
            "log_multipole": r"\phi \mathrel{+}= -a_0 \ln|z - c|^2 + 2\sum_{n=1}^{p} \frac{\operatorname{Re}(a_n w^n)}{n}, \qquad E_x - iE_y \mathrel{+}= 2\sum_{n=0}^{p} a_n w^{n+1}, \qquad w = \frac{1}{z - c}",
            "leaves": r"\text{leaves: the softened direct sum of } \texttt{direct\_field}",
        },
        order={"coulomb": 3, "log": 3},
        assumptions=(
            r"one target at a time, an explicit stack; the cell holding the target is always opened, whatever $\theta$ and $\varepsilon$; $\theta = 0$ is the exact sum",
            r"the Coulomb expansion through the quadrupole, from the Taylor series of $1/|\mathbf R - \mathbf d|$ with $R \to \rho$; $\mathbf E = -\nabla\phi$ holds exactly for this softened form",
            r"accepted log-kernel cells are unsoftened: softening acts on the direct leaf pairs only, an $O(\varepsilon^2/D^2)$ difference in the far field",
            r"the order is in the size of an accepted cell's charges over its distance, so in $\theta$: 3 through the quadrupole, $p + 1$ for the log kernel (3 at the $p = 2$ of the test problem)",
        ),
        references=(
            "[Barnes and Hut, Nature 324, 446 (1986)](https://doi.org/10.1038/324446a0)",
        ),
        components=("tree_d_3d.flat_tree.direct_field",),
        symbolic=_multipole_rules,
        problem=_walk_problem,
    )
    @sp.profile("tree_d_3d.flat_tree.evaluate", tags=_PROFILE_TAGS)
    def evaluate(
        self, targets, *, exclude=None, theta=0.5, softening=0.0, potential=True
    ):
        """Potential ``(M,)`` and field ``(2, M)`` at ``targets`` ``(2, M)``, without ``k_e``.

        ``exclude[i]`` is a source index skipped for target ``i`` (-1: none).
        ``potential=False`` returns the potential as zeros, which saves the logarithms
        of the line-charge kernel when only the field is needed.
        """
        targets = _points(targets, "targets", compiled=self._compiled)
        n_targets = targets.shape[1]
        if exclude is None:
            exclude = self._xp.full(n_targets, -1, dtype=np.int64)
        exclude = _device(exclude, np.int64, compiled=self._compiled)
        phi = self._xp.zeros(n_targets)
        field = self._xp.zeros((2, n_targets))
        status = self._xp.zeros(1, dtype=np.int64)
        n_sources = self.pos.shape[1]
        if isinstance(self.pos, np.ndarray) and n_sources and n_targets:
            self._evaluate_groups(
                targets, exclude, theta, softening, potential, phi, field, status
            )
            _check(status)
            return phi, field
        elif self.kernel == "log" and n_sources and n_targets:
            self._kernels.evaluate_log_quadtree(
                targets,
                exclude,
                self.pos,
                self.charge,
                self.perm,
                self.node_center,
                self.node_half,
                self.node_start,
                self.node_end,
                self.node_child,
                self.node_qabs,
                self.moment_re,
                self.moment_im,
                float(theta),
                float(softening),
                self._stack,
                phi,
                field,
                status,
            )
            _check(status)
        elif n_sources and n_targets:
            self._kernels.evaluate_quadtree(
                targets,
                exclude,
                self.pos,
                self.charge,
                self.perm,
                self.node_center,
                self.node_half,
                self.node_start,
                self.node_end,
                self.node_child,
                self.node_q,
                self.node_qabs,
                self.node_p,
                self.node_s,
                float(theta),
                float(softening),
                self._stack,
                phi,
                field,
                status,
            )
            _check(status)
        if not potential:
            phi[...] = 0.0
        return phi, field

    def _cartesian_moments(self) -> np.ndarray:
        """The Cartesian multipole moments of every cell (`tree_d_3d._cartesian`), kept."""
        if self._cart_moments is None:
            tables = _cartesian.tables(self.order)
            moments = np.empty((self.node_count, _cartesian.size(self.order)))
            self._kernels.cart_moments_upward(
                self.pos,
                self.charge,
                self.perm,
                np.asarray(self.levels, dtype=np.int64),
                self.node_center,
                self.node_start,
                self.node_end,
                self.node_child,
                tables["mono_parent"],
                tables["mono_dir"],
                tables["shift"],
                tables["shift_coef"],
                moments,
            )
            self._cart_moments = moments
        return self._cart_moments

    def _dual_lists(self, tree: FlatQuadtree, theta: float):
        """The M2L and P2P pairs (target cell of ``tree``, source cell), as rows by target."""
        depth = int(tree.node_depth.max()) + int(self.node_depth.max())
        stack = np.empty((3 * depth + 8, 2), dtype=np.int64)
        capacity = np.array([16, 8]) * (tree.node_count + self.node_count) + 64
        counts = np.zeros(3, dtype=np.int64)
        while True:
            m2l = np.empty((capacity[0], 2), dtype=np.int64)
            p2p = np.empty((capacity[1], 2), dtype=np.int64)
            self._kernels.dual_traverse(
                tree.node_center,
                tree.node_half,
                tree.node_start,
                tree.node_end,
                tree.node_child,
                self.node_center,
                self.node_half,
                self.node_start,
                self.node_end,
                self.node_child,
                theta,
                stack,
                m2l,
                p2p,
                counts,
            )
            if not counts[2]:
                break
            capacity = np.maximum(2 * capacity, counts[:2] + 64)
        rows = []
        for pairs, size in ((m2l, int(counts[0])), (p2p, int(counts[1]))):
            start = np.zeros(tree.node_count + 1, dtype=np.int64)
            partners = np.empty(max(size, 1), dtype=np.int64)
            cursor = np.empty(tree.node_count, dtype=np.int64)
            self._kernels.pair_csr(pairs, size, 0, start, partners, cursor)
            rows.append((start, partners, size))
        return rows

    def _target_plan(self, targets: np.ndarray, theta: float, leaf: int):
        """The target tree and the dual lists of `fmm_at`, kept for the same targets.

        Repeated sums at the same points (the products of an iterative solver, probes
        after `set_charges`) reuse them; any other targets replace them.
        """
        key = ("targets", theta, leaf)
        cached = self._traversals.get(key)
        if cached is not None and np.array_equal(cached[0], targets):
            return cached[1], cached[2]
        m = targets.shape[1]
        lo, hi = targets.min(axis=1), targets.max(axis=1)
        centre = 0.5 * (lo + hi)
        half = 0.5 * float(np.max(hi - lo)) * (1 + 1e-12) + 1e-300
        tree = FlatQuadtree(
            targets,
            np.zeros(m),
            Quad(float(centre[0]), float(centre[1]), half),
            leaf_capacity=leaf,
            max_depth=self.max_depth,
            kernel="coulomb",
            compiled=self._compiled,
        )
        lists = self._dual_lists(tree, theta)
        self._traversals[key] = (targets.copy(), tree, lists)
        return tree, lists

    @sp.profile("tree_d_3d.flat_tree.fmm_at", tags=_PROFILE_TAGS)
    def fmm_at(
        self,
        targets,
        *,
        exclude=None,
        theta: float = 0.5,
        softening: float = 0.0,
        potential: bool = True,
        leaf_capacity: int | None = None,
    ):
        """Potential ``(M,)`` and field ``(2, M)`` at ``targets`` ``(2, M)`` by the FMM, without ``k_e``.

        The fast multipole method of `fmm` at points other than the sources: the
        targets get a tree of their own (``leaf_capacity`` per leaf, by default the
        sources'), a dual traversal pairs its cells with the source cells (M2L where
        ``sqrt(2) (h_t + h_s) < theta d``, exact sums between leaves otherwise), and the
        local expansions run down the target tree to the targets. ``exclude[i]`` is a
        source index skipped for target ``i`` (-1: none) in the exact sums;
        ``potential=False`` returns zeros for the potential and, for line charges,
        skips its logarithms. The conventions of `evaluate` otherwise (the log
        kernel's expansions unsoftened), at the FMM's cost: about one M2L per target
        cell and partner instead of one multipole term per target and cell. On the
        CPU only.
        """
        theta = _finite("theta", theta, minimum=0)
        if not 0 < theta < 1:
            raise ValueError("FMM theta must lie in (0, 1)")
        softening = _finite("softening", softening, minimum=0)
        targets = _points(targets, "targets", compiled=self._compiled)
        if not isinstance(targets, np.ndarray) or not isinstance(self.pos, np.ndarray):
            raise NotImplementedError("fmm_at runs on the CPU (NumPy arrays)")
        if not bool(np.all(np.isfinite(targets))):
            raise ValueError("targets must be finite")
        m, n = targets.shape[1], self.pos.shape[1]
        phi, field = np.zeros(m), np.zeros((2, m))
        if not m or not n:
            return phi, field
        exclude = (
            np.full(m, -1, dtype=np.int64)
            if exclude is None
            else np.asarray(exclude, dtype=np.int64).reshape(-1)
        )
        leaf = self.leaf_capacity if leaf_capacity is None else int(leaf_capacity)
        tree, lists = self._target_plan(targets, theta, leaf)
        (m2l_rows, m2l_partners, n_m2l), (p2p_rows, p2p_partners, _) = lists
        levels = np.asarray(tree.levels, dtype=np.int64)
        k = self._kernels
        tpos, _ = tree._sorted_sources()
        texclude = np.ascontiguousarray(exclude[tree.perm])
        spos, scharge = self._sorted_sources()
        phi_sorted = np.empty(m)
        field_sorted = np.empty((2, m + 1))
        if self.kernel == "log":
            order = self.order
            local_re = np.zeros((tree.node_count, order + 1))
            local_im = np.zeros((tree.node_count, order + 1))
            if n_m2l:
                k.fmm_m2l_host(
                    m2l_rows,
                    m2l_partners,
                    tree.node_center,
                    self.node_center,
                    self.moment_re,
                    self.moment_im,
                    self._m2l_coefficients(),
                    local_re,
                    local_im,
                )
            scratch = np.empty(2 * order + 2)
            for first, last in zip(levels[:-2], levels[1:-1]):
                k.fmm_downward(
                    int(first),
                    int(last),
                    tree.node_center,
                    tree.node_child,
                    self._binomial(),
                    scratch,
                    scratch.copy(),
                    local_re,
                    local_im,
                )
            k.log_targets_host(
                tpos,
                texclude,
                tree.node_count,
                tree.node_center,
                tree.node_start,
                tree.node_end,
                tree.node_child,
                local_re,
                local_im,
                p2p_rows,
                p2p_partners,
                spos,
                scharge,
                self.perm,
                self.node_start,
                self.node_end,
                softening,
                1 if potential else 0,
                phi_sorted,
                field_sorted,
            )
        else:
            tables = _cartesian.tables(self.order)
            local = np.zeros((tree.node_count, _cartesian.size(self.order)))
            if n_m2l:
                k.cart_m2l_host(
                    m2l_rows,
                    m2l_partners,
                    tree.node_center,
                    self.node_center,
                    self._cartesian_moments(),
                    softening,
                    tables["ex"],
                    tables["ey"],
                    tables["rec"],
                    tables["m2l"],
                    tables["m2l_coef"],
                    local,
                )
            k.cart_downward(
                levels,
                tree.node_center,
                tree.node_child,
                tables["mono_parent"],
                tables["mono_dir"],
                tables["shift"],
                tables["shift_coef"],
                local,
            )
            k.cart_targets_host(
                tpos,
                texclude,
                tree.node_count,
                tree.node_center,
                tree.node_start,
                tree.node_end,
                tree.node_child,
                local,
                tables["mono_parent"],
                tables["mono_dir"],
                tables["ex"],
                tables["ey"],
                tables["grad"],
                p2p_rows,
                p2p_partners,
                spos,
                scharge,
                self.perm,
                self.node_start,
                self.node_end,
                softening,
                phi_sorted,
                field_sorted,
            )
        phi[tree.perm] = phi_sorted
        field[:, tree.perm] = field_sorted[:, :m]
        bad = np.flatnonzero(phi == 1.0e300)
        if bad.size:
            status = np.array([bad[0] + 1], dtype=np.int64)
            _check(status)
        if not potential:
            phi[:] = 0.0
        return phi, field

    def _traverse(self, theta: float):
        """The M2L and P2P node-pair lists of the dual-tree traversal, and their sizes.

        Runs ``fmm_traverse`` generation by generation from the pair (root, root),
        restarting with larger buffers if one overflows.
        """
        x = self._xp
        capacity = np.array([16, 16, 10]) * self.node_count + 64
        while True:
            m2l = x.empty((capacity[0], 2), dtype=np.int64)
            p2p = x.empty((capacity[1], 2), dtype=np.int64)
            frontier = x.zeros((capacity[2], 2), dtype=np.int64)
            next_frontier = x.empty((capacity[2], 2), dtype=np.int64)
            counts = x.zeros(4, dtype=np.int64)
            n_frontier = 1
            sizes = np.zeros(4, dtype=np.int64)
            while n_frontier:
                counts[2] = 0
                self._kernels.fmm_traverse(
                    self.node_center,
                    self.node_half,
                    self.node_start,
                    self.node_end,
                    self.node_child,
                    theta,
                    frontier,
                    n_frontier,
                    next_frontier,
                    m2l,
                    p2p,
                    counts,
                    n_threads=n_frontier,
                )
                sizes = _host(counts)
                if sizes[3]:
                    break
                n_frontier = int(sizes[2])
                frontier, next_frontier = next_frontier, frontier
            if not sizes[3]:
                return m2l, p2p, sizes
            capacity = np.maximum(2 * capacity, sizes[:3])

    def _coulomb_fmm(self, theta, softening, phi, field):
        """`fmm` for ``kernel="coulomb"``: Cartesian expansions, on the CPU."""
        if not isinstance(self.pos, np.ndarray):
            raise NotImplementedError("the Coulomb FMM runs on the CPU (NumPy arrays)")
        k = self._kernels
        n = self.pos.shape[1]
        if theta not in self._traversals:
            self._traversals[theta] = self._traverse(theta)
        m2l, p2p, sizes = self._traversals[theta]
        tables = _cartesian.tables(self.order)
        local = np.zeros((self.node_count, _cartesian.size(self.order)))
        if sizes[0]:
            rows, partners = self._m2l_rows(theta, m2l, slice(0, int(sizes[0])))
            k.cart_m2l_host(
                rows,
                partners,
                self.node_center,
                self.node_center,
                self._cartesian_moments(),
                softening,
                tables["ex"],
                tables["ey"],
                tables["rec"],
                tables["m2l"],
                tables["m2l_coef"],
                local,
            )
        k.cart_downward(
            np.asarray(self.levels, dtype=np.int64),
            self.node_center,
            self.node_child,
            tables["mono_parent"],
            tables["mono_dir"],
            tables["shift"],
            tables["shift_coef"],
            local,
        )
        spos, scharge = self._sorted_sources()
        phi_sorted = np.zeros(n)
        field_sorted = np.zeros((2, n + 1))
        status = np.zeros(1, dtype=np.int64)
        k.cart_evaluate_host(
            spos,
            scharge,
            self.node_count,
            self.node_center,
            self.node_start,
            self.node_end,
            self.node_child,
            local,
            tables["mono_parent"],
            tables["mono_dir"],
            tables["ex"],
            tables["ey"],
            tables["grad"],
            p2p,
            int(sizes[1]),
            softening,
            phi_sorted,
            field_sorted,
            status,
        )
        if int(status[0]):
            status[0] = self.perm[int(status[0]) - 1] + 1
        _check(status)
        phi[self.perm] = phi_sorted
        field[:, self.perm] = field_sorted[:, :n]
        self.fmm_pairs = (int(sizes[0]), int(sizes[1]))
        return phi, field

    @method(
        equations={
            "traversal": r"(a, b) \to \begin{cases} \text{M2L} & \sqrt2\,(h_a + h_b) < \theta\, d_{ab} \\ \text{P2P} & a = b \text{ a leaf, or two leaves} \\ \text{split } a \text{ or } b & \text{otherwise (the larger, not a leaf)} \end{cases}",
            "m2l": r"b_0 \mathrel{+}= a_0\ln t - \sum_{k=1}^{p}\frac{a_k}{k\,t^k}, \qquad b_l \mathrel{+}= \frac{(-1)^{l+1}a_0}{l\,t^l} - \frac{(-1)^l}{t^l}\sum_{k=1}^{k_{\max}}\frac{a_k}{k}\binom{k+l-1}{l}\frac{1}{t^k}, \qquad t = c_\mathrm{dst} - c_\mathrm{src}",
            "truncation": r"k_{\max} = p \ \text{(rectangle)}, \qquad k_{\max} = p - l \ \text{(triangle, } \texttt{symmetric=True}\text{)}",
            "l2l": r"b^\mathrm{child}_m \mathrel{+}= \sum_{l=m}^{p} \binom{l}{m} b_l\, s^{\,l-m}, \qquad s = c_\mathrm{child} - c_\mathrm{parent}",
            "l2p": r"W = \sum_{l=0}^{p} b_l u^l, \qquad \phi \mathrel{+}= -2\operatorname{Re} W, \qquad E_x - iE_y \mathrel{+}= 2W', \qquad u = z_i - c_\mathrm{leaf}",
            "p2p": r"\phi_i \mathrel{+}= -\sum_j q_j \ln\left(r_{ij}^2 + \varepsilon^2\right), \qquad \mathbf E_i \mathrel{+}= 2\sum_j \frac{q_j\, \mathbf d_{ij}}{r_{ij}^2 + \varepsilon^2}",
            "coulomb_moments": r"m_{\boldsymbol\alpha} = \sum_j q_j (\mathbf r_j - \mathbf c)^{\boldsymbol\alpha}, \quad |\boldsymbol\alpha| \le p \qquad \text{(Cartesian, } \mathbf r^{\boldsymbol\alpha} = x^{\alpha_x} y^{\alpha_y}\text{; leaves, then M2M)}",
            "coulomb_taylor": r"n R^2 A_{\boldsymbol\gamma} = -(2n - 1)\left(t_x A_{\boldsymbol\gamma - \mathbf e_x} + t_y A_{\boldsymbol\gamma - \mathbf e_y}\right) - (n - 1)\left(A_{\boldsymbol\gamma - 2\mathbf e_x} + A_{\boldsymbol\gamma - 2\mathbf e_y}\right), \quad A_{\mathbf 0} = \frac1R, \quad R^2 = |\mathbf t|^2 + \varepsilon^2, \quad n = |\boldsymbol\gamma|",
            "coulomb_m2l": r"T_{\boldsymbol\beta} \mathrel{+}= \sum_{|\boldsymbol\alpha| + |\boldsymbol\beta| \le p} (-1)^{|\boldsymbol\alpha|} \binom{\boldsymbol\alpha + \boldsymbol\beta}{\boldsymbol\alpha} m_{\boldsymbol\alpha}\, A_{\boldsymbol\alpha + \boldsymbol\beta}(\mathbf t), \qquad \phi(\mathbf c + \mathbf u) = \sum_{\boldsymbol\beta} T_{\boldsymbol\beta}\, \mathbf u^{\boldsymbol\beta}, \quad \mathbf E = -\nabla\phi",
        },
        order={"potential": 3, "field": 2},
        assumptions=(
            r"$0 < \theta < 1$; every particle is a target; each pair is listed once and used in both directions",
            r"`kernel=\"coulomb\"` (CPU only): Cartesian expansions of the softened kernel $f = (|\mathbf r|^2 + \varepsilon^2)^{-1/2}$ to total degree $p$ (`order`, default 4), $A_{\boldsymbol\gamma} = \partial^{\boldsymbol\gamma} f / \boldsymbol\gamma!$ by the Taylor recurrence; the total-degree truncation makes pair forces exactly antisymmetric; softened throughout, far field included",
            r"`fmm_at`: the same traversal between a tree of the targets and the sources' tree, one way, with exact sums between their leaves",
            r"the local expansion $\Phi_L(z) = \sum_l b_l (z - c)^l$ of degree $p$ about each cell centre; $\ln t$ on the principal branch, which shifts $b_0$ by $2\pi i a_0$ and drops out of $\operatorname{Re}$",
            r"the triangle truncation makes the mutual pair forces exactly antisymmetric (same error order, larger constant)",
            r"log kernel: softening acts on the P2P pairs only; no automatic error control: $\theta$ and $p$ set the accuracy",
            r"the error of an M2L is $O((r/d)^{p+1})$ in the potential and $O((r/d)^{p})$ in the field, $r/d$ the cluster size over the cell distance; the declared orders are those at the $p = 2$ of the test problem",
        ),
        references=(
            "[Greengard and Rokhlin, J. Comput. Phys. 73, 325 (1987)](https://doi.org/10.1016/0021-9991(87)90140-9)",
            "[Carrier, Greengard and Rokhlin, SIAM J. Sci. Stat. Comput. 9, 669 (1988)](https://doi.org/10.1137/0909044)",
            "[Dehnen, J. Comput. Phys. 179, 27 (2002)](https://doi.org/10.1006/jcph.2002.7026)",
        ),
        components=("tree_d_3d.flat_tree.FlatQuadtree",),
        problem=_fmm_problem,
    )
    @sp.profile("tree_d_3d.flat_tree.fmm", tags=_PROFILE_TAGS)
    def fmm(
        self, *, theta: float = 0.5, softening: float = 0.0, potential: bool = True
    ):
        """Potential and field at every source, by the FMM.

        Fast multipole method, for ``kernel="log"`` in complex form
        (Greengard-Rokhlin), for ``kernel="coulomb"`` with Cartesian expansions of the
        softened kernel to total degree ``order`` (on the CPU only, softened in the far
        field too; `tree_d_3d._cartesian`): a dual-tree traversal pairs well-separated cells, whose
        multipole expansions are translated into each other's local
        expansions (M2L, both ways at once); local expansions are shifted down
        the tree (L2L) and evaluated at the particles (L2P); leaves that are
        not well separated interact exactly (P2P). Cost is O(N) rather than the O(N log N) of
        `evaluate` with every particle as a target. Returns the raw
        potential ``(N,)`` and field ``(2, N)`` in input order, without ``k_e``,
        excluding each particle's own charge; for line charges softening applies to
        the exact near field only.
        ``potential=False`` skips the potential (returned as zeros), which saves
        a logarithm per near-field pair of line charges when only forces are needed.
        `fmm_at` is the same at points other than the sources.
        """
        theta = _finite("theta", theta, minimum=0)
        if not 0 < theta < 1:
            raise ValueError("FMM theta must lie in (0, 1)")
        softening = _finite("softening", softening, minimum=0)
        n = self.pos.shape[1]
        phi = self._xp.zeros(n)
        field = self._xp.zeros((2, n))
        if not n:
            return phi, field
        if self.kernel != "log":
            return self._coulomb_fmm(theta, softening, phi, field)
        k = self._kernels
        count = self.node_count
        # the interaction lists depend on the cells only: kept for set_charges
        if theta not in self._traversals:
            self._traversals[theta] = self._traverse(theta)
        m2l, p2p, sizes = self._traversals[theta]
        order = self.order
        binomial = self._xp.asarray(_binomial_table(2 * order + 1))
        local_re = self._xp.zeros((count, order + 1))
        local_im = self._xp.zeros((count, order + 1))
        scratch_re = self._xp.empty(2 * order + 2)
        scratch_im = self._xp.empty(2 * order + 2)
        # each rank takes a share of the M2L pairs, the leaves and the P2P pairs; the
        # expansions and fields are sums, completed by an allreduce
        m2l_share, leaf_share, p2p_share = self._fmm_shares(theta)
        host = isinstance(self.pos, np.ndarray)
        if host and m2l_share.stop > m2l_share.start:
            rows, partners = self._m2l_rows(theta, m2l, m2l_share)
            k.fmm_m2l_host(
                rows,
                partners,
                self.node_center,
                self.node_center,
                self.moment_re,
                self.moment_im,
                self._m2l_coefficients(),
                local_re,
                local_im,
            )
        if not host and m2l_share.stop > m2l_share.start:
            k.fmm_m2l(
                m2l[m2l_share],
                m2l_share.stop - m2l_share.start,
                self.node_center,
                self.moment_re,
                self.moment_im,
                binomial,
                1 if self.symmetric else 0,
                scratch_re,
                scratch_im,
                local_re,
                local_im,
                n_threads=max(m2l_share.stop - m2l_share.start, 1),
            )
        if collectives.is_parallel(self.comm):
            local_re = collectives.allsum(self.comm, local_re)
            local_im = collectives.allsum(self.comm, local_im)
        for lo, hi in zip(self.levels[:-2], self.levels[1:-1]):
            k.fmm_downward(
                int(lo),
                int(hi),
                self.node_center,
                self.node_child,
                binomial,
                scratch_re,
                scratch_im,
                local_re,
                local_im,
                n_threads=int(hi - lo),
            )
        status = self._xp.zeros(1, dtype=np.int64)
        if host and not collectives.is_parallel(self.comm):
            spos, scharge = self._sorted_sources()
            phi_sorted = np.zeros(n)
            field_sorted = np.zeros((2, n + 1))
            k.fmm_evaluate_host(
                spos,
                scharge,
                count,
                self.node_center,
                self.node_start,
                self.node_end,
                self.node_child,
                local_re,
                local_im,
                p2p,
                int(sizes[1]),
                softening,
                1 if potential else 0,
                phi_sorted,
                field_sorted,
                status,
            )
            if int(status[0]):
                status[0] = self.perm[int(status[0]) - 1] + 1
            phi[self.perm] = phi_sorted
            field[:, self.perm] = field_sorted[:, :n]
        elif not collectives.is_parallel(self.comm):
            k.fmm_evaluate(
                self.pos,
                self.charge,
                self.perm,
                count,
                self.node_center,
                self.node_start,
                self.node_end,
                self.node_child,
                local_re,
                local_im,
                p2p,
                int(sizes[1]),
                softening,
                1 if potential else 0,
                phi,
                field,
                status,
                n_threads=max(count, int(sizes[1])),
            )
        else:
            leaves, near = leaf_share, p2p_share
            # this rank's leaves (L2P, no pairs), then its pairs (P2P, no leaves)
            for nodes, pairs in ((leaves, slice(0, 0)), (slice(0, 0), near)):
                if nodes.stop == nodes.start and pairs.stop == pairs.start:
                    continue  # Pyccel rejects zero-length arrays
                k.fmm_evaluate(
                    self.pos,
                    self.charge,
                    self.perm,
                    nodes.stop - nodes.start,
                    self.node_center[nodes.start :],
                    self.node_start[nodes.start :],
                    self.node_end[nodes.start :],
                    self.node_child[nodes.start :],
                    local_re[nodes.start :],
                    local_im[nodes.start :],
                    p2p[pairs.start :],
                    pairs.stop - pairs.start,
                    softening,
                    1 if potential else 0,
                    phi,
                    field,
                    status,
                    n_threads=max(
                        nodes.stop - nodes.start, pairs.stop - pairs.start, 1
                    ),
                )
            phi = collectives.allsum(self.comm, phi)
            field = collectives.allsum(self.comm, field)
            status[0] = collectives.allmax(self.comm, int(status[0]))
        _check(status)
        self.fmm_pairs = (int(sizes[0]), int(sizes[1]))
        return phi, field

    @method(
        equations={
            "coulomb_potential": r"\phi = k_e\, \phi_\mathrm{raw}, \qquad \mathbf E = k_e\, \mathbf E_\mathrm{raw}",
            "log_potential": r"\phi = k_e\left(\phi_\mathrm{raw} + 2\ln r_0 \sum_j q_j\right) = -2k_e \sum_j q_j \ln\frac{r_j}{r_0}",
        },
        assumptions=(
            r"the log potential is defined up to a constant: $r_0$ (`reference_radius`) shifts energies, not forces",
            r"$k_e = 1/(4\pi\varepsilon_0)$ by default; SI inputs give SI outputs",
        ),
        components=("tree_d_3d.flat_tree.FlatQuadtree.evaluate",),
        symbolic=_reference_rules,
    )
    def field_at(
        self,
        positions,
        *,
        theta: float = 0.5,
        k_e: float = K_E,
        softening: float = 0.0,
        reference_radius: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Potential, shape (n,), and electric field, shape (2, n), at probes (2, n).

        For line charges the potential is only defined up to a constant; it
        is ``-2 k_e sum q ln(r / reference_radius)``, so the potential of the
        net charge vanishes at ``reference_radius`` from it.
        """
        theta = _finite("theta", theta, minimum=0)
        reference_radius = _finite("reference_radius", reference_radius, positive=True)
        k_e = _finite("k_e", k_e, minimum=0)
        softening = _finite("softening", softening, minimum=0)
        positions = _points(positions, "positions", compiled=self._compiled)
        if not bool(self._xp.all(self._xp.isfinite(positions))):
            raise ValueError("positions must be finite")
        if k_e == 0:
            n = positions.shape[1]
            return self._xp.zeros(n), self._xp.zeros((2, n))
        phi, field = self.evaluate(positions, theta=theta, softening=softening)
        if self.kernel == "log":
            phi = phi + 2 * self._xp.sum(self.charge) * float(np.log(reference_radius))
        return k_e * phi, k_e * field
