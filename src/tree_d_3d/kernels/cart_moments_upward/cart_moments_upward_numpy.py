"""The version of the ``cart_moments_upward`` kernel for machines without Pyccel.

``cart_moments_upward_pyccel.py`` vectorized over the cells of a level: the leaves sum
the monomials of all their particles at once, the parents shift the moments of all
their children at once (the rows of ``shift`` as one matrix product).
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import expand, monomials, term_matrix


def cart_moments_upward(
    pos,
    charge,
    perm,
    levels,
    node_center,
    node_start,
    node_end,
    node_child,
    mono_parent,
    mono_dir,
    shift,
    shift_coef,
    moments,
):
    """Moments ``m_g = sum_j q_j (r_j - c)^g`` of every node, from the deepest level up.

    See ``cart_moments_upward_pyccel``.
    """
    n_coef = moments.shape[1]
    scatter = term_matrix(shift[:, 0], n_coef)
    for level in range(len(levels) - 2, -1, -1):
        nodes = np.arange(levels[level], levels[level + 1], dtype=np.intp)
        out = np.zeros((len(nodes), n_coef))
        first = node_child[nodes]
        leaf = first < 0
        if leaf.any():
            owner, k = expand(node_start[nodes[leaf]], node_end[nodes[leaf]])
            rows = np.flatnonzero(leaf)[owner]
            j = perm[k]
            mono = monomials(
                pos[0, j] - node_center[nodes[rows], 0],
                pos[1, j] - node_center[nodes[rows], 1],
                mono_parent,
                mono_dir,
                first=charge[j],
            )
            np.add.at(out, rows, mono)
        if (~leaf).any():
            parent = np.flatnonzero(~leaf)
            rows = np.repeat(parent, 4)
            child = (first[parent][:, None] + np.arange(4)).reshape(-1)
            mono = monomials(
                node_center[child, 0] - node_center[nodes[rows], 0],
                node_center[child, 1] - node_center[nodes[rows], 1],
                mono_parent,
                mono_dir,
            )
            terms = shift_coef * moments[child][:, shift[:, 1]] * mono[:, shift[:, 2]]
            np.add.at(out, rows, terms @ scatter)
        moments[nodes] = out
