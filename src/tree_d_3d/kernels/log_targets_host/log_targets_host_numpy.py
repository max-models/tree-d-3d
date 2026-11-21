"""The version of the ``log_targets_host`` kernel for machines without Pyccel.

``log_targets_host_pyccel.py`` vectorized: the local expansions at every target of the
target leaves at once, and the near field over all (target, source) particle pairs of
their partner leaves, accumulated with ``np.bincount``.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import leaf_particles, target_pairs


def log_targets_host(
    tpos,
    texclude,
    t_count,
    t_center,
    t_start,
    t_end,
    t_child,
    local_re,
    local_im,
    row_start,
    partners,
    spos,
    scharge,
    sindex,
    s_start,
    s_end,
    softening,
    with_potential,
    phi,
    field,
):
    """Line-charge potential and field at the targets, in target-tree order.

    See ``log_targets_host_pyccel``.
    """
    order = local_re.shape[1] - 1
    eps2 = softening * softening
    n = len(phi)
    leaf, k = leaf_particles(t_count, t_start, t_end, t_child)
    u = (tpos[0, k] - t_center[leaf, 0]) + 1j * (tpos[1, k] - t_center[leaf, 1])
    local = local_re[leaf] + 1j * local_im[leaf]
    w = local[:, order].copy()
    d = np.zeros_like(w)
    for l in range(order - 1, -1, -1):
        d = d * u + w
        w = w * u + local[:, l]
    pot = np.zeros(n)
    ex = np.zeros(n)
    ey = np.zeros(n)
    bad = np.zeros(n, dtype=bool)
    pot[k] = -2.0 * w.real
    ex[k] = 2.0 * d.real
    ey[k] = -2.0 * d.imag
    for i, j in target_pairs(
        t_start, t_end, t_child, t_count, row_start, partners, s_start, s_end
    ):
        c = scharge[j]
        use = (c != 0.0) & (sindex[j] != texclude[i])
        i, j, c = i[use], j[use], c[use]
        dx = tpos[0, i] - spos[0, j]
        dy = tpos[1, i] - spos[1, j]
        r2 = dx * dx + dy * dy + eps2
        coincident = r2 == 0.0
        if coincident.any():
            bad[i[coincident]] = True
            keep = ~coincident
            i, c, dx, dy, r2 = i[keep], c[keep], dx[keep], dy[keep], r2[keep]
        if with_potential != 0:
            pot -= np.bincount(i, c * np.log(r2), n)
        ex += np.bincount(i, 2.0 * c * dx / r2, n)
        ey += np.bincount(i, 2.0 * c * dy / r2, n)
    if with_potential == 0:
        pot[:] = 0.0
    phi[k] = np.where(bad[k], 1.0e300, pot[k])
    field[0, k] = ex[k]
    field[1, k] = ey[k]
