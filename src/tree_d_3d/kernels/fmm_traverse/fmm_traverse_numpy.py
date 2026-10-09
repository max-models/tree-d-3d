"""The NumPy version of the ``fmm_traverse`` kernel, for machines without Pyccel.

Vectorized over the frontier; the same arguments and results (in the same order) as
``fmm_traverse_pyccel.py``.
"""

from __future__ import annotations

import numpy as np

# the child pairs (c, d), c <= d, of a node paired with itself
_SELF_C = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
_SELF_D = np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3])


def _append(out, pairs, counts, slot):
    """Append ``pairs`` to ``out`` at ``counts[slot]``, as far as they fit."""
    i = int(counts[slot])
    fit = max(min(len(pairs), out.shape[0] - i), 0)
    out[i : i + fit] = pairs[:fit]
    counts[slot] = i + len(pairs)
    if fit < len(pairs):
        counts[3] = -1


def fmm_traverse(
    node_center: np.ndarray,
    node_half: np.ndarray,
    node_start: np.ndarray,
    node_end: np.ndarray,
    node_child: np.ndarray,
    theta: float,
    frontier: np.ndarray,
    n_frontier: int,
    next_frontier: np.ndarray,
    m2l: np.ndarray,
    p2p: np.ndarray,
    counts: np.ndarray,
) -> None:
    """One generation of the dual-tree traversal, which fills mutual M2L and P2P lists."""
    a, b = frontier[:n_frontier, 0], frontier[:n_frontier, 1]
    live = (node_end[a] != node_start[a]) & (node_end[b] != node_start[b])
    leaf_a, leaf_b = node_child[a] < 0, node_child[b] < 0
    same = live & (a == b)
    other = live & (a != b)
    d = node_center[a] - node_center[b]
    radii = np.sqrt(2.0) * (node_half[a] + node_half[b])
    far = other & (radii < theta * np.hypot(d[:, 0], d[:, 1]))
    near = other & ~far
    leaves = (same & leaf_a) | (near & leaf_a & leaf_b)
    split_self = same & ~leaf_a
    split = near & ~(leaf_a & leaf_b)
    split_a = split & (leaf_b | (~leaf_a & (node_half[a] >= node_half[b])))
    split_b = split & ~split_a
    _append(m2l, np.column_stack((a[far], b[far])), counts, 0)
    _append(p2p, np.column_stack((a[leaves], b[leaves])), counts, 1)
    # children pairs, in frontier order: 10 for a split node with itself, else 4
    size = np.where(split_self, 10, np.where(split, 4, 0))
    offset = np.cumsum(size) - size
    pairs = np.empty((int(size.sum()), 2), dtype=np.int64)
    k = np.flatnonzero(split_self)
    rows = offset[k, None] + np.arange(10)
    first = node_child[a[k], None]
    pairs[rows, 0] = first + _SELF_C
    pairs[rows, 1] = first + _SELF_D
    for mask, child_side in ((split_a, 0), (split_b, 1)):
        k = np.flatnonzero(mask)
        rows = offset[k, None] + np.arange(4)
        ends = (a[k], b[k])
        pairs[rows, child_side] = node_child[ends[child_side], None] + np.arange(4)
        pairs[rows, 1 - child_side] = ends[1 - child_side][:, None]
    i = int(counts[2])
    counts[2] = i + len(pairs)
    # whole groups only, like the per-pair versions
    whole = np.repeat(offset + size <= next_frontier.shape[0] - i, size)
    if not whole.all():
        counts[3] = -1
    rows = np.flatnonzero(whole)
    next_frontier[i + rows] = pairs[rows]
