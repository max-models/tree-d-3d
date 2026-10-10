"""The flat quadtree and direct sums: free-space electrostatics for charges constrained to the xy plane.

Uses the three-dimensional softened Coulomb kernel, not the logarithmic
Green's function of 2D Poisson. No periodic images, conducting walls,
collisions, radiation, or relativistic effects are included.

The simulation stores particles as arrays and evaluates fields with the
kernels ``tree_d_3d.kernels.<name>`` (``quadtree_quadrants``,
``evaluate_quadtree``, ``fmm_m2l``, ...): a flat quadtree (:class:`FlatQuadtree`) or, for small systems,
an exact direct sum. The object tree :class:`ReferenceQuadtree` is the readable
reference implementation of the same algorithm and is kept for inspection and
tests.

Each kernel has a Pyccel, a NumPy and a CUDA version with the same arguments, and
runs the one matching its arrays. :class:`FlatQuadtree` and :func:`direct_field`
work on arrays of cunumpy's active backend: on NumPy the Pyccel versions run (the
NumPy versions without Pyccel), on CuPy the CUDA versions, with the launch sizes
given here; their results are arrays of that backend. The simulation keeps its
particles on the host (each :class:`ChargedBody` is a view of a column) and hands them
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

from tree_d_3d._utils import _check, _device, _finite, _host, _points
from tree_d_3d.constants import K_E
from tree_d_3d.kernels import _kernels
from tree_d_3d.quad import Quad

#: Highest expansion order of the CUDA FMM kernels (TREE_D_3D_MAX_ORDER).
MAX_ORDER = 64


_KERNELS = ("coulomb", "log")


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
    is ``"coulomb"`` or ``"log"`` (line charges), as for :class:`FlatQuadtree`.
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
    """Read-only view of one node of a :class:`FlatQuadtree`."""

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


class FlatQuadtree:
    """Array quadtree for point charges in a plane or for line charges.

    Same tree and opening rule as :class:`ReferenceQuadtree`: nodes holding
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
        count = self._build(root_quad)
        self.node_count = int(count)
        for name in _NODE_ARRAYS:
            setattr(self, name, getattr(self, name)[:count].copy())
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
        self._stack = self._xp.empty(3 * int(self.node_depth.max()) + 8, dtype=np.int64)

    @property
    def root(self) -> FlatNode:
        return FlatNode(self, 0)

    def collect_nodes(self) -> list[FlatNode]:
        """Views of every node that holds at least one particle."""
        occupied = _host(self._xp.flatnonzero(self.node_end > self.node_start))
        return [FlatNode(self, int(i)) for i in occupied]

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

    def fmm(
        self, *, theta: float = 0.5, softening: float = 0.0, potential: bool = True
    ):
        """Line-charge potential and field at every source, by the FMM.

        Fast multipole method for ``kernel="log"`` (Greengard-Rokhlin, complex
        form): a dual-tree traversal pairs well-separated cells, whose
        multipole expansions are translated into each other's local
        expansions (M2L, both ways at once); local expansions are shifted down
        the tree (L2L) and evaluated at the particles (L2P); adjacent leaves
        interact exactly (P2P). Cost is O(N) rather than the O(N log N) of
        :meth:`evaluate` with every particle as a target. Returns the raw
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
