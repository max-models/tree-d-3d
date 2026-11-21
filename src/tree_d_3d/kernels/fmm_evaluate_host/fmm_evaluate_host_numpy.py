"""The version of the ``fmm_evaluate_host`` kernel for machines without Pyccel.

``fmm_evaluate_host_pyccel.py`` vectorized: the local expansions at every leaf particle
at once (Horner over the terms), and the mutual near field over all particle pairs of
the leaf pairs, accumulated with ``np.bincount``.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import leaf_particles, near_pairs


def fmm_evaluate_host(
    spos,
    scharge,
    node_count,
    node_center,
    node_start,
    node_end,
    node_child,
    local_re,
    local_im,
    pairs,
    count,
    softening,
    with_potential,
    phi,
    field,
    status,
):
    """Far field from leaf local expansions plus exact mutual near field, in tree order.

    See ``fmm_evaluate_host_pyccel``.
    """
    order = local_re.shape[1] - 1
    eps2 = softening * softening
    n = len(phi)
    nf = field.shape[1]  # Pyccel's layout may keep a spare column
    leaf, k = leaf_particles(node_count, node_start, node_end, node_child)
    u = (spos[0, k] - node_center[leaf, 0]) + 1j * (spos[1, k] - node_center[leaf, 1])
    local = local_re[leaf] + 1j * local_im[leaf]
    w = local[:, order].copy()
    d = np.zeros_like(w)
    for l in range(order - 1, -1, -1):
        d = d * u + w
        w = w * u + local[:, l]
    phi[k] += -2.0 * w.real
    field[0, k] += 2.0 * d.real
    field[1, k] -= 2.0 * d.imag
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
        if with_potential != 0:
            log_r2 = np.log(r2)
            phi -= np.bincount(ka, qj * log_r2, n) + np.bincount(kb, qi * log_r2, n)
        fx = 2.0 * dx / r2
        fy = 2.0 * dy / r2
        field[0] += np.bincount(ka, qj * fx, nf) - np.bincount(kb, qi * fx, nf)
        field[1] += np.bincount(ka, qj * fy, nf) - np.bincount(kb, qi * fy, nf)
