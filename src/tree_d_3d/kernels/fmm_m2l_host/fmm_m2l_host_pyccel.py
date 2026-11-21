"""The Pyccel version (CPU) of the ``fmm_m2l_host`` kernel of the FMM.

``fmm_m2l_host_numpy.py`` next to this file is the version for machines without Pyccel,
with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`). The flat quadtree is described in `tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import atan2, log

import numpy as np


def fmm_m2l_host(
    row_start: "int64[:]",
    sources: "int64[:]",
    dst_center: "float[:,:]",
    src_center: "float[:,:]",
    moment_re: "float[:,:]",
    moment_im: "float[:,:]",
    coeff: "float[:,:]",
    local_re: "float[:,:]",
    local_im: "float[:,:]",
) -> None:
    """Add the multipoles of cells ``sources[row_start[n]:row_start[n + 1]]`` to cell ``n``.

    The destination cells' centres are ``dst_center``, the sources' ``src_center``
    (the same array within one tree).

    With ``t = c_dst - c_src``, ``w = 1 / t`` and the scaled moments
    ``a'_k = a_k w**k`` (``a'_0 = a_0``), the translation is a real matrix product,
    ``s_l = sum_k coeff[k, l] a'_k``, with ``coeff[0, l] = 1 / l``,
    ``coeff[k, 0] = 1 / k`` and ``coeff[k, l] = C(k + l - 1, l) / k`` (zero beyond the
    truncation): ``b_0 += a_0 ln t - s_0`` and ``b_l += -(-1)**l s_l w**l``. Each cell
    is written by its own row only, so the rows are independent (chunks of 64 cells
    share their scratch).
    """
    order = moment_re.shape[1] - 1
    n_nodes = row_start.shape[0] - 1
    n_chunks = (n_nodes + 63) // 64
    # fmt: off
    #$ omp parallel for firstprivate(b_re, b_im, s_re, s_im, pw_re, pw_im) private(dst, cx, cy, e, src, t_re, t_im, t2, w_re, w_im, k, l, a_re, a_im, ar, ai, c, log_re, log_im, sign, b_re_ptr, b_im_ptr, s_re_ptr, s_im_ptr, pw_re_ptr, pw_im_ptr) schedule(dynamic, 1)
    # fmt: on
    for chunk in range(n_chunks):
        b_re = np.empty(order + 1, dtype=float)
        b_im = np.empty(order + 1, dtype=float)
        s_re = np.empty(order + 1, dtype=float)
        s_im = np.empty(order + 1, dtype=float)
        pw_re = np.empty(order + 1, dtype=float)
        pw_im = np.empty(order + 1, dtype=float)
        for dst in range(64 * chunk, min(64 * chunk + 64, n_nodes)):
            if row_start[dst + 1] == row_start[dst]:
                continue
            for l in range(order + 1):
                b_re[l] = 0.0
                b_im[l] = 0.0
            cx = dst_center[dst, 0]
            cy = dst_center[dst, 1]
            for e in range(row_start[dst], row_start[dst + 1]):
                src = sources[e]
                t_re = cx - src_center[src, 0]
                t_im = cy - src_center[src, 1]
                t2 = t_re * t_re + t_im * t_im
                w_re = t_re / t2
                w_im = -t_im / t2
                pw_re[0] = 1.0
                pw_im[0] = 0.0
                for k in range(1, order + 1):
                    pw_re[k] = pw_re[k - 1] * w_re - pw_im[k - 1] * w_im
                    pw_im[k] = pw_re[k - 1] * w_im + pw_im[k - 1] * w_re
                for l in range(order + 1):
                    s_re[l] = 0.0
                    s_im[l] = 0.0
                for k in range(order + 1):
                    a_re = moment_re[src, k]
                    a_im = moment_im[src, k]
                    ar = a_re * pw_re[k] - a_im * pw_im[k]
                    ai = a_re * pw_im[k] + a_im * pw_re[k]
                    for l in range(order + 1):
                        c = coeff[k, l]
                        s_re[l] += c * ar
                        s_im[l] += c * ai
                a_re = moment_re[src, 0]
                a_im = moment_im[src, 0]
                log_re = 0.5 * log(t2)
                log_im = atan2(t_im, t_re)
                b_re[0] += a_re * log_re - a_im * log_im - s_re[0]
                b_im[0] += a_re * log_im + a_im * log_re - s_im[0]
                sign = 1.0
                for l in range(1, order + 1):
                    # -(-1)**l
                    b_re[l] += sign * (s_re[l] * pw_re[l] - s_im[l] * pw_im[l])
                    b_im[l] += sign * (s_re[l] * pw_im[l] + s_im[l] * pw_re[l])
                    sign = -sign
            for l in range(order + 1):
                local_re[dst, l] += b_re[l]
                local_im[dst, l] += b_im[l]
        del b_re
        del b_im
        del s_re
        del s_im
        del pw_re
        del pw_im
