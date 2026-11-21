"""The version of the ``cart_evaluate_host`` kernel for machines without Pyccel.

``cart_evaluate_host_pyccel.py`` vectorized: the Cartesian local expansions at every
leaf particle at once, and the mutual near field over all particle pairs of the leaf
pairs, accumulated with ``np.bincount``.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import (
    cartesian_local,
    leaf_particles,
    monomials,
    near_pairs,
)


def cart_evaluate_host(
    spos,
    scharge,
    node_count,
    node_center,
    node_start,
    node_end,
    node_child,
    local,
    mono_parent,
    mono_dir,
    ex,
    ey,
    grad,
    pairs,
    count,
    softening,
    phi,
    field,
    status,
):
    """Far field from the leaves' local expansions plus the exact mutual near field.

    See ``cart_evaluate_host_pyccel``.
    """
    eps2 = softening * softening
    n = len(phi)
    nf = field.shape[1]  # Pyccel's layout may keep a spare column
    leaf, k = leaf_particles(node_count, node_start, node_end, node_child)
    mono = monomials(
        spos[0, k] - node_center[leaf, 0],
        spos[1, k] - node_center[leaf, 1],
        mono_parent,
        mono_dir,
    )
    pot, gx, gy = cartesian_local(local[leaf], mono, ex, ey, grad)
    phi[k] += pot
    field[0, k] -= gx
    field[1, k] -= gy
    for ka, kb in near_pairs(node_start, node_end, pairs, count):
        qi, qj = scharge[ka], scharge[kb]
        dx = spos[0, ka] - spos[0, kb]
        dy = spos[1, ka] - spos[1, kb]
        r2 = dx * dx + dy * dy + eps2
        coincident = r2 == 0.0
        if coincident.any():
            flagged = coincident & (qi != 0.0) & (qj != 0.0)
            if flagged.any():
                status[0] = ka[np.argmax(flagged)] + 1
                return
            keep = ~coincident
            ka, kb, qi, qj = ka[keep], kb[keep], qi[keep], qj[keep]
            dx, dy, r2 = dx[keep], dy[keep], r2[keep]
        inv_r = 1.0 / np.sqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r
        phi += np.bincount(ka, qj * inv_r, n) + np.bincount(kb, qi * inv_r, n)
        fx, fy = inv_r3 * dx, inv_r3 * dy
        field[0] += np.bincount(ka, qj * fx, nf) - np.bincount(kb, qi * fx, nf)
        field[1] += np.bincount(ka, qj * fy, nf) - np.bincount(kb, qi * fy, nf)
