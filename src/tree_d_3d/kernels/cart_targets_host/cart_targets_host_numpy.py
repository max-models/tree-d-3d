"""The version of the ``cart_targets_host`` kernel for machines without Pyccel.

``cart_targets_host_pyccel.py`` vectorized: the Cartesian local expansions at every
target of the target leaves at once, and the near field over all (target, source)
particle pairs of their partner leaves, accumulated with ``np.bincount``.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import (
    cartesian_local,
    leaf_particles,
    monomials,
    target_pairs,
)


def cart_targets_host(
    tpos,
    texclude,
    t_count,
    t_center,
    t_start,
    t_end,
    t_child,
    local,
    mono_parent,
    mono_dir,
    ex_pow,
    ey_pow,
    grad,
    row_start,
    partners,
    spos,
    scharge,
    sindex,
    s_start,
    s_end,
    softening,
    phi,
    field,
):
    """Point-charge potential and field at the targets, in target-tree order.

    See ``cart_targets_host_pyccel``.
    """
    eps2 = softening * softening
    n = len(phi)
    leaf, k = leaf_particles(t_count, t_start, t_end, t_child)
    mono = monomials(
        tpos[0, k] - t_center[leaf, 0],
        tpos[1, k] - t_center[leaf, 1],
        mono_parent,
        mono_dir,
    )
    far, gx, gy = cartesian_local(local[leaf], mono, ex_pow, ey_pow, grad)
    pot = np.zeros(n)
    ex = np.zeros(n)
    ey = np.zeros(n)
    bad = np.zeros(n, dtype=bool)
    pot[k], ex[k], ey[k] = far, -gx, -gy
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
        inv_r = 1.0 / np.sqrt(r2)
        cr3 = c * inv_r * inv_r * inv_r
        pot += np.bincount(i, c * inv_r, n)
        ex += np.bincount(i, cr3 * dx, n)
        ey += np.bincount(i, cr3 * dy, n)
    phi[k] = np.where(bad[k], 1.0e300, pot[k])
    field[0, k] = ex[k]
    field[1, k] = ey[k]
