"""The version of the ``fmm_m2l_host`` kernel for machines without Pyccel.

The translations of ``fmm_m2l_host_pyccel.py``, vectorized over all (destination,
source) cell pairs: the scaled moments of every pair times the matrix ``coeff`` at once,
summed into the destination cells.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import chunks, complex_powers, expand


def fmm_m2l_host(
    row_start,
    sources,
    dst_center,
    src_center,
    moment_re,
    moment_im,
    coeff,
    local_re,
    local_im,
):
    """Add the multipoles of cells ``sources[row_start[n]:row_start[n + 1]]`` to cell ``n``.

    See ``fmm_m2l_host_pyccel``.
    """
    order = moment_re.shape[1] - 1
    n_nodes = row_start.shape[0] - 1
    dst_all, edge_all = expand(row_start[:-1], row_start[1:])
    sign = -((-1.0) ** np.arange(order + 1))
    for part in chunks(np.full(len(edge_all), order + 1)):
        dst, src = dst_all[part], np.asarray(sources[edge_all[part]], dtype=np.intp)
        t = (dst_center[dst, 0] - src_center[src, 0]) + 1j * (
            dst_center[dst, 1] - src_center[src, 1]
        )
        powers = complex_powers(1.0 / t, order)
        a = moment_re[src] + 1j * moment_im[src]
        s = (a * powers) @ coeff
        b = sign * s * powers
        b[:, 0] = a[:, 0] * np.log(t) - s[:, 0]
        total = np.zeros((n_nodes, order + 1), dtype=complex)
        np.add.at(total, dst, b)
        local_re += total.real
        local_im += total.imag
