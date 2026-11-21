"""The version of the ``quadtree_moments_upward`` kernel for machines without Pyccel.

``quadtree_moments_upward_pyccel.py`` vectorized over the cells of a level: the leaves
sum all their particles at once, the parents shift all their children at once.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import expand


def quadtree_moments_upward(
    pos,
    charge,
    perm,
    levels,
    node_center,
    node_start,
    node_end,
    node_child,
    node_q,
    node_qabs,
    node_p,
    node_s,
):
    """Signed moments of every node about its centre, from the deepest level up.

    See ``quadtree_moments_upward_pyccel``.
    """
    for level in range(len(levels) - 2, -1, -1):
        nodes = np.arange(levels[level], levels[level + 1], dtype=np.intp)
        n = len(nodes)
        # q, |q|, px, py, sxx, sxy, syy of every node of the level
        out = np.zeros((7, n))
        first = node_child[nodes]
        leaf = first < 0
        if leaf.any():
            owner, k = expand(node_start[nodes[leaf]], node_end[nodes[leaf]])
            rows = np.flatnonzero(leaf)[owner]
            j = perm[k]
            c = charge[j]
            dx = pos[0, j] - node_center[nodes[rows], 0]
            dy = pos[1, j] - node_center[nodes[rows], 1]
            for i, w in enumerate(
                (c, np.abs(c), c * dx, c * dy, c * dx * dx, c * dx * dy, c * dy * dy)
            ):
                out[i] += np.bincount(rows, w, n)
        if (~leaf).any():
            parent = np.flatnonzero(~leaf)
            rows = np.repeat(parent, 4)
            child = (first[parent][:, None] + np.arange(4)).reshape(-1)
            dx = node_center[child, 0] - node_center[nodes[rows], 0]
            dy = node_center[child, 1] - node_center[nodes[rows], 1]
            cq, cpx, cpy = node_q[child], node_p[child, 0], node_p[child, 1]
            for i, w in enumerate(
                (
                    cq,
                    node_qabs[child],
                    cpx + cq * dx,
                    cpy + cq * dy,
                    node_s[child, 0] + 2.0 * cpx * dx + cq * dx * dx,
                    node_s[child, 1] + cpx * dy + cpy * dx + cq * dx * dy,
                    node_s[child, 2] + 2.0 * cpy * dy + cq * dy * dy,
                )
            ):
                out[i] += np.bincount(rows, w, n)
        node_q[nodes], node_qabs[nodes] = out[0], out[1]
        node_p[nodes] = out[2:4].T
        node_s[nodes] = out[4:7].T
