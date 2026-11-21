"""The version of the ``cart_downward`` kernel for machines without Pyccel.

``cart_downward_pyccel.py`` vectorized over the cells of a level: every parent shifts
its local expansion into its four children at once (the rows of ``shift`` as one
matrix product).
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import monomials, term_matrix


def cart_downward(
    levels, node_center, node_child, mono_parent, mono_dir, shift, shift_coef, local
):
    """Shift every cell's local expansion into its children, from the root down.

    See ``cart_downward_pyccel``.
    """
    n_coef = local.shape[1]
    scatter = term_matrix(shift[:, 1], n_coef)
    for level in range(len(levels) - 2):
        nodes = np.arange(levels[level], levels[level + 1], dtype=np.intp)
        parent = nodes[node_child[nodes] >= 0]
        if not len(parent):
            continue
        rows = np.repeat(parent, 4)
        child = (node_child[parent][:, None] + np.arange(4)).reshape(-1)
        mono = monomials(
            node_center[child, 0] - node_center[rows, 0],
            node_center[child, 1] - node_center[rows, 1],
            mono_parent,
            mono_dir,
        )
        terms = shift_coef * local[rows][:, shift[:, 0]] * mono[:, shift[:, 2]]
        local[child] += terms @ scatter
