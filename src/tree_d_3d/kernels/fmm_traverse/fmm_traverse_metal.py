"""The Metal version (Apple-silicon GPU, float32) of the ``fmm_traverse`` kernel.

``fmm_traverse_pyccel.py`` next to this file is the CPU version with the same
arguments. The GPU classifies every pair of the frontier in float32 (one thread per
pair: skip, near-field pair, far-field pair, split a, split b, or split a node with
itself). The Apple GPU has no 64-bit atomics for ``counts``, so the host appends the
pairs to ``m2l``, ``p2p`` and ``next_frontier`` from those codes, with the same
semantics as the CPU: appended in pair order (so the lists are identical to the CPU's
unless a pair is within float32 rounding of the ``theta`` criterion), ``counts``
holding the sizes needed, ``counts[3] = -1`` if a buffer is too small.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels import KERNEL_PACKAGE

SKIP, P2P, M2L, SPLIT_SELF, SPLIT_A, SPLIT_B = range(6)

_SOURCE = """
    uint k = thread_position_in_grid.x;
    if (k >= frontier_shape[0]) return;
    int64_t a = frontier[2 * k], b = frontier[2 * k + 1];
    if (node_end[a] == node_start[a] || node_end[b] == node_start[b]) {
        code[k] = 0;
        return;
    }
    bool leaf_a = node_child[a] < 0, leaf_b = node_child[b] < 0;
    if (a == b) {
        code[k] = leaf_a ? 1 : 3;
        return;
    }
    float dx = node_center[2 * a] - node_center[2 * b];
    float dy = node_center[2 * a + 1] - node_center[2 * b + 1];
    float distance = metal::precise::sqrt(dx * dx + dy * dy);
    float radii = metal::precise::sqrt(2.0f) * (node_hw[a] + node_hw[b]);
    if (radii < theta[0] * distance) {
        code[k] = 2;
        return;
    }
    if (leaf_a && leaf_b) {
        code[k] = 1;
        return;
    }
    bool split_a = leaf_b || (!leaf_a && node_hw[a] >= node_hw[b]);
    code[k] = split_a ? 4 : 5;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "tree_d_3d_fmm_traverse",
    _SOURCE,
    inputs=[
        "frontier",
        "node_center",
        "node_hw",
        "node_start",
        "node_end",
        "node_child",
        "theta",
    ],
    outputs=["code"],
)


def _append_rows(rows: np.ndarray, buffer: Array, start: int) -> bool:
    """Write the rows that fit at ``start``, ``start + 1``, ...; whether all fitted."""
    index = start + np.arange(len(rows))
    fits = index < buffer.shape[0]
    buffer[index[fits]] = rows[fits]
    return bool(fits.all())


def fmm_traverse(
    node_center: Array,
    node_half: Array,
    node_start: Array,
    node_end: Array,
    node_child: Array,
    theta: float,
    frontier: Array,
    n_frontier: int,
    next_frontier: Array,
    m2l: Array,
    p2p: Array,
    counts: Array,
) -> None:
    """One generation of the dual-tree traversal, which fills mutual M2L and P2P lists."""
    n = int(n_frontier)
    if not n:
        return
    code = np.zeros(n, dtype=np.int32)
    _KERNEL(
        frontier[:n],
        node_center,
        node_half,
        node_start,
        node_end,
        node_child,
        float(theta),
        out=code,
    )
    pairs = np.asarray(frontier[:n])
    first_a = np.asarray(node_child)[pairs[:, 0]]
    first_b = np.asarray(node_child)[pairs[:, 1]]
    ok = True

    for kind, buffer, slot in ((M2L, m2l, 0), (P2P, p2p, 1)):
        rows = pairs[code == kind]
        ok &= _append_rows(rows, buffer, int(counts[slot]))
        counts[slot] += len(rows)

    # the child pairs of every splitting pair, in frontier order
    split = np.flatnonzero(code >= SPLIT_SELF)
    size = np.where(code[split] == SPLIT_SELF, 10, 4)
    offset = np.cumsum(size) - size
    total = int(size.sum())
    rows = np.empty((total, 2), dtype=pairs.dtype)
    quad = np.arange(4)
    self_pairs = np.array([(c, d) for c in range(4) for d in range(c, 4)])
    for kind in (SPLIT_SELF, SPLIT_A, SPLIT_B):
        chosen = code[split] == kind
        k, where = split[chosen], offset[chosen]
        if kind == SPLIT_SELF:
            at = where[:, None] + np.arange(10)
            rows[at, 0] = first_a[k][:, None] + self_pairs[:, 0]
            rows[at, 1] = first_a[k][:, None] + self_pairs[:, 1]
        else:
            at = where[:, None] + quad
            if kind == SPLIT_A:
                rows[at, 0] = first_a[k][:, None] + quad
                rows[at, 1] = pairs[k, 1][:, None]
            else:
                rows[at, 0] = pairs[k, 0][:, None]
                rows[at, 1] = first_b[k][:, None] + quad
    base = int(counts[2])
    fits = base + offset + size <= next_frontier.shape[0]
    ok &= bool(fits.all())
    for width in (10, 4):
        blocks = np.flatnonzero(fits & (size == width))
        at = (base + offset[blocks])[:, None] + np.arange(width)
        next_frontier[at] = rows[offset[blocks][:, None] + np.arange(width)]
    base += total
    counts[2] = base
    if not ok:
        counts[3] = -1
