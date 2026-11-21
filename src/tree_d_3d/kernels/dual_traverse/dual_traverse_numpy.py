"""The version of the ``dual_traverse`` kernel for machines without Pyccel.

The traversal of ``dual_traverse_pyccel.py`` breadth first instead of depth first,
vectorized over the pairs of a generation: the same M2L and P2P pairs, in another
order (which only changes the order of the sums that use them).
"""

from __future__ import annotations

import numpy as np


def _store(out, pairs, counts, slot):
    """Write ``pairs`` into ``out`` from ``counts[slot]`` as far as they fit; count them all."""
    i = int(counts[slot])
    fit = max(min(len(pairs), out.shape[0] - i), 0)
    out[i : i + fit] = pairs[:fit]
    counts[slot] = i + len(pairs)
    if fit < len(pairs):
        counts[2] = 1


def dual_traverse(
    t_center,
    t_half,
    t_start,
    t_end,
    t_child,
    s_center,
    s_half,
    s_start,
    s_end,
    s_child,
    theta,
    stack,
    m2l,
    p2p,
    counts,
):
    """Pairs (target cell, source cell) from (root, root): see ``dual_traverse_pyccel``.

    ``stack`` is unused.
    """
    counts[:3] = 0
    a = np.zeros(1, dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    while len(a):
        live = (t_end[a] != t_start[a]) & (s_end[b] != s_start[b])
        a, b = a[live], b[live]
        d = t_center[a] - s_center[b]
        far = np.sqrt(2.0) * (t_half[a] + s_half[b]) < theta * np.hypot(
            d[:, 0], d[:, 1]
        )
        _store(m2l, np.column_stack((a[far], b[far])), counts, 0)
        a, b = a[~far], b[~far]
        leaf_a, leaf_b = t_child[a] < 0, s_child[b] < 0
        leaves = leaf_a & leaf_b
        _store(p2p, np.column_stack((a[leaves], b[leaves])), counts, 1)
        a, b, leaf_a, leaf_b = a[~leaves], b[~leaves], leaf_a[~leaves], leaf_b[~leaves]
        split_a = leaf_b | (~leaf_a & (t_half[a] >= s_half[b]))
        quad = np.arange(4)
        a = np.where(split_a[:, None], t_child[a][:, None] + quad, a[:, None]).reshape(
            -1
        )
        b = np.where(split_a[:, None], b[:, None], s_child[b][:, None] + quad).reshape(
            -1
        )
