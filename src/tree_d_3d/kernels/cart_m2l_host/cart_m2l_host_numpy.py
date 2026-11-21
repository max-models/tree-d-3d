"""The version of the ``cart_m2l_host`` kernel for machines without Pyccel.

The translations of ``cart_m2l_host_pyccel.py``, vectorized over all (destination,
source) cell pairs: the Taylor coefficients of every pair by the recurrence (one column
at a time), then the terms of ``m2l`` as one matrix product, summed into the
destination cells.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import chunks, expand, term_matrix


def cart_m2l_host(
    row_start,
    sources,
    dst_center,
    src_center,
    moments,
    softening,
    ex,
    ey,
    rec,
    m2l,
    m2l_coef,
    local,
):
    """Add the expansions of cells ``sources[row_start[n]:row_start[n + 1]]`` to cell ``n``.

    See ``cart_m2l_host_pyccel``.
    """
    n_coef = moments.shape[1]
    n_nodes = row_start.shape[0] - 1
    eps2 = softening * softening
    scatter = term_matrix(m2l[:, 0], n_coef)
    dst_all, edge_all = expand(row_start[:-1], row_start[1:])
    for part in chunks(np.full(len(edge_all), max(len(m2l), 1))):
        dst, src = dst_all[part], np.asarray(sources[edge_all[part]], dtype=np.intp)
        tx = dst_center[dst, 0] - src_center[src, 0]
        ty = dst_center[dst, 1] - src_center[src, 1]
        r2 = tx * tx + ty * ty + eps2
        coef = np.empty((len(dst), n_coef))
        coef[:, 0] = 1.0 / np.sqrt(r2)
        for i in range(1, n_coef):
            n = ex[i] + ey[i]
            s = np.zeros(len(dst))
            if rec[i, 0] >= 0:
                s += (2 * n - 1) * tx * coef[:, rec[i, 0]]
            if rec[i, 1] >= 0:
                s += (2 * n - 1) * ty * coef[:, rec[i, 1]]
            if rec[i, 2] >= 0:
                s += (n - 1) * coef[:, rec[i, 2]]
            if rec[i, 3] >= 0:
                s += (n - 1) * coef[:, rec[i, 3]]
            coef[:, i] = -s / (n * r2)
        terms = m2l_coef * moments[src][:, m2l[:, 1]] * coef[:, m2l[:, 2]]
        total = np.zeros((n_nodes, n_coef))
        np.add.at(total, dst, terms @ scatter)
        local += total
