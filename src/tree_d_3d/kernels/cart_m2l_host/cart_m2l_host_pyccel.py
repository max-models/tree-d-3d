"""The Pyccel version (CPU) of the ``cart_m2l_host`` kernel of the Coulomb FMM.

``cart_m2l_host_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`).
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import sqrt

import numpy as np


def cart_m2l_host(
    row_start: "int64[:]",
    sources: "int64[:]",
    dst_center: "float[:,:]",
    src_center: "float[:,:]",
    moments: "float[:,:]",
    softening: float,
    ex: "int64[:]",
    ey: "int64[:]",
    rec: "int64[:,:]",
    m2l: "int64[:,:]",
    m2l_coef: "float[:]",
    local: "float[:,:]",
) -> None:
    """Add the expansions of cells ``sources[row_start[n]:row_start[n + 1]]`` to cell ``n``.

    For ``t = c_dst - c_src`` the Taylor coefficients ``A_g(t)`` of the softened kernel
    follow from the recurrence on ``rec``; then ``T_b += m2l_coef * m_a * A_g`` over the
    rows ``(b, a, g)`` of ``m2l``. Each destination cell is written by its own row only
    (chunks of 64 cells share their scratch).
    """
    n_coef = moments.shape[1]
    n_terms = m2l.shape[0]
    eps2 = softening * softening
    n_nodes = row_start.shape[0] - 1
    n_chunks = (n_nodes + 63) // 64
    # fmt: off
    #$ omp parallel for firstprivate(acc, coef) private(dst, cx, cy, e, src, tx, ty, r2, i, n, s, t, acc_ptr, coef_ptr) schedule(dynamic, 1)
    # fmt: on
    for chunk in range(n_chunks):
        acc = np.empty(n_coef, dtype=float)
        coef = np.empty(n_coef, dtype=float)
        for dst in range(64 * chunk, min(64 * chunk + 64, n_nodes)):
            if row_start[dst + 1] == row_start[dst]:
                continue
            for i in range(n_coef):
                acc[i] = 0.0
            cx = dst_center[dst, 0]
            cy = dst_center[dst, 1]
            for e in range(row_start[dst], row_start[dst + 1]):
                src = sources[e]
                tx = cx - src_center[src, 0]
                ty = cy - src_center[src, 1]
                r2 = tx * tx + ty * ty + eps2
                coef[0] = 1.0 / sqrt(r2)
                for i in range(1, n_coef):
                    n = ex[i] + ey[i]
                    s = 0.0
                    if rec[i, 0] >= 0:
                        s += (2 * n - 1) * tx * coef[rec[i, 0]]
                    if rec[i, 1] >= 0:
                        s += (2 * n - 1) * ty * coef[rec[i, 1]]
                    if rec[i, 2] >= 0:
                        s += (n - 1) * coef[rec[i, 2]]
                    if rec[i, 3] >= 0:
                        s += (n - 1) * coef[rec[i, 3]]
                    coef[i] = -s / (n * r2)
                for t in range(n_terms):
                    acc[m2l[t, 0]] += (
                        m2l_coef[t] * moments[src, m2l[t, 1]] * coef[m2l[t, 2]]
                    )
            for i in range(n_coef):
                local[dst, i] += acc[i]
        del acc
        del coef
