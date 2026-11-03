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
from methodcard import ConvergenceProblem, Rule, method

from tree_d_3d._profiling import _PROFILE_TAGS
from tree_d_3d._utils import _check, _device, _finite, _host, _points
from tree_d_3d.constants import K_E
from tree_d_3d.kernels import _kernels
from tree_d_3d.quad import Quad

#: Highest expansion order of the CUDA FMM kernels (TREE_D_3D_MAX_ORDER).
MAX_ORDER = 64


_KERNELS = ("coulomb", "log")


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
    """

    def _empty(self, root_quad: Quad) -> None:
        # Pyccel rejects zero-length arrays, so build the lone root here.
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

    def _build(self, root_quad: Quad) -> int:
        """Split the nodes level by level; returns the node count.

        Every level is one ``quadtree_quadrants`` and one ``quadtree_scatter``
        launch over the particles with prefix sums in between, so the nodes are
        numbered breadth first and each node's particles keep their input order,
        as in a serial build. ``self.levels[d]:self.levels[d + 1]`` are the nodes
        at depth ``d``.
        """
        x, k = self._xp, self._kernels
        n = self.pos.shape[1]
        capacity = 4 * (2 * n // self.leaf_capacity + 16)
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
            k.quadtree_quadrants(
                self.pos, perm, slot_node, self.node_center, self.node_child, quadrant
            )
            # below[k, q]: slots up to k in quadrant q
            below = x.cumsum(quadrant[:, None] == quadrants, axis=0, dtype=np.int64)
            parents = lo + x.flatnonzero(split)
            start = self.node_start[parents]
            before = below[x.maximum(start - 1, 0)] * (start > 0)[:, None]
            in_child = below[self.node_end[parents] - 1] - before
            child_start = start[:, None] + x.cumsum(in_child, axis=1) - in_child
            children = self.node_child[parents][:, None] + quadrants
            half = 0.5 * self.node_half[parents][:, None]
            self.node_start[children] = child_start
            self.node_end[children] = child_start + in_child
            self.node_half[children] = half
            self.node_depth[children] = self.node_depth[parents][:, None] + 1
            self.node_center[children, 0] = (
                self.node_center[parents, 0][:, None] + x_sign * half
            )
            self.node_center[children, 1] = (
                self.node_center[parents, 1][:, None] + y_sign * half
            )
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
        order: int = 12,
        symmetric: bool = False,
    ) -> None:
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {_KERNELS}")
        if isinstance(order, bool) or int(order) != order or order < 1:
            raise ValueError("order must be a positive integer")
        if order > MAX_ORDER:
            raise ValueError(f"order must be at most {MAX_ORDER}, got {order}")
        self.kernel = kernel
        self.order = int(order)
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
        with sp.region("tree_d_3d.flat_tree.moments", tags=_PROFILE_TAGS):
            self.node_q = self._xp.empty(count)
            self.node_qabs = self._xp.empty(count)
            self.node_p = self._xp.empty((count, 2))
            self.node_s = self._xp.empty((count, 3))
            if kernel == "log":
                self.moment_re = self._xp.empty((count, self.order + 1))
                self.moment_im = self._xp.empty((count, self.order + 1))
                self._kernels.log_quadtree_moments(
                    self.pos,
                    self.charge,
                    self.perm,
                    count,
                    self.node_center,
                    self.node_start,
                    self.node_end,
                    self.node_qabs,
                    self.moment_re,
                    self.moment_im,
                    n_threads=count,
                )
                self.node_q[:] = self.moment_re[:, 0]
                self.node_p[:, 0] = self.moment_re[:, 1]
                self.node_p[:, 1] = self.moment_im[:, 1]
                self._stack = self._xp.empty(
                    3 * int(self.node_depth.max()) + 8, dtype=np.int64
                )
                return
            self._kernels.quadtree_moments(
                self.pos,
                self.charge,
                self.perm,
                count,
                self.node_center,
                self.node_start,
                self.node_end,
                self.node_q,
                self.node_qabs,
                self.node_p,
                self.node_s,
                n_threads=count,
            )
            self._stack = self._xp.empty(
                3 * int(self.node_depth.max()) + 8, dtype=np.int64
            )

    @property
    def root(self) -> FlatNode:
        return FlatNode(self, 0)

    def collect_nodes(self) -> list[FlatNode]:
        """Views of every node that holds at least one particle."""
        occupied = _host(self._xp.flatnonzero(self.node_end > self.node_start))
        return [FlatNode(self, int(i)) for i in occupied]

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
    def evaluate(self, targets, *, exclude=None, theta=0.5, softening=0.0):
        """Potential ``(M,)`` and field ``(2, M)`` at ``targets`` ``(2, M)``, without ``k_e``.

        ``exclude[i]`` is a source index skipped for target ``i`` (-1: none).
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
        if self.kernel == "log" and n_sources and n_targets:
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

    @method(
        equations={
            "traversal": r"(a, b) \to \begin{cases} \text{M2L} & \sqrt2\,(h_a + h_b) < \theta\, d_{ab} \\ \text{P2P} & a = b \text{ a leaf, or two leaves} \\ \text{split } a \text{ or } b & \text{otherwise (the larger, not a leaf)} \end{cases}",
            "m2l": r"b_0 \mathrel{+}= a_0\ln t - \sum_{k=1}^{p}\frac{a_k}{k\,t^k}, \qquad b_l \mathrel{+}= \frac{(-1)^{l+1}a_0}{l\,t^l} - \frac{(-1)^l}{t^l}\sum_{k=1}^{k_{\max}}\frac{a_k}{k}\binom{k+l-1}{l}\frac{1}{t^k}, \qquad t = c_\mathrm{dst} - c_\mathrm{src}",
            "truncation": r"k_{\max} = p \ \text{(rectangle)}, \qquad k_{\max} = p - l \ \text{(triangle, } \texttt{symmetric=True}\text{)}",
            "l2l": r"b^\mathrm{child}_m \mathrel{+}= \sum_{l=m}^{p} \binom{l}{m} b_l\, s^{\,l-m}, \qquad s = c_\mathrm{child} - c_\mathrm{parent}",
            "l2p": r"W = \sum_{l=0}^{p} b_l u^l, \qquad \phi \mathrel{+}= -2\operatorname{Re} W, \qquad E_x - iE_y \mathrel{+}= 2W', \qquad u = z_i - c_\mathrm{leaf}",
            "p2p": r"\phi_i \mathrel{+}= -\sum_j q_j \ln\left(r_{ij}^2 + \varepsilon^2\right), \qquad \mathbf E_i \mathrel{+}= 2\sum_j \frac{q_j\, \mathbf d_{ij}}{r_{ij}^2 + \varepsilon^2}",
        },
        order={"potential": 3, "field": 2},
        assumptions=(
            r"the log kernel only, $0 < \theta < 1$; every particle is a target; each pair is listed once and used in both directions",
            r"the local expansion $\Phi_L(z) = \sum_l b_l (z - c)^l$ of degree $p$ about each cell centre; $\ln t$ on the principal branch, which shifts $b_0$ by $2\pi i a_0$ and drops out of $\operatorname{Re}$",
            r"the triangle truncation makes the mutual pair forces exactly antisymmetric (same error order, larger constant)",
            r"softening acts on the P2P pairs only; no automatic error control: $\theta$ and $p$ set the accuracy",
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
        """Line-charge potential and field at every source, by the FMM.

        Fast multipole method for ``kernel="log"`` (Greengard-Rokhlin, complex
        form): a dual-tree traversal pairs well-separated cells, whose
        multipole expansions are translated into each other's local
        expansions (M2L, both ways at once); local expansions are shifted down
        the tree (L2L) and evaluated at the particles (L2P); leaves that are
        not well separated interact exactly (P2P). Cost is O(N) rather than the O(N log N) of
        `evaluate` with every particle as a target. Returns the raw
        potential ``(N,)`` and field ``(2, N)`` in input order, without ``k_e``,
        excluding each particle's own charge; softening applies to the exact
        near field only.
        ``potential=False`` skips the potential (returned as zeros), which saves
        a logarithm per near-field pair when only forces are needed.
        """
        if self.kernel != "log":
            raise ValueError("The FMM is implemented for kernel='log'")
        theta = _finite("theta", theta, minimum=0)
        if not 0 < theta < 1:
            raise ValueError("FMM theta must lie in (0, 1)")
        softening = _finite("softening", softening, minimum=0)
        n = self.pos.shape[1]
        phi = self._xp.zeros(n)
        field = self._xp.zeros((2, n))
        if not n:
            return phi, field
        k = self._kernels
        count = self.node_count
        m2l, p2p, sizes = self._traverse(theta)
        order = self.order
        binomial = self._xp.asarray(_binomial_table(2 * order + 1))
        local_re = self._xp.zeros((count, order + 1))
        local_im = self._xp.zeros((count, order + 1))
        scratch_re = self._xp.empty(2 * order + 2)
        scratch_im = self._xp.empty(2 * order + 2)
        k.fmm_m2l(
            m2l,
            int(sizes[0]),
            self.node_center,
            self.moment_re,
            self.moment_im,
            binomial,
            1 if self.symmetric else 0,
            scratch_re,
            scratch_im,
            local_re,
            local_im,
            n_threads=max(int(sizes[0]), 1),
        )
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
