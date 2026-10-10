"""The Pyccel version (CPU) of the ``fmm_m2l`` kernel of the Barnes-Hut and FMM solvers.

``fmm_m2l_cuda.cu`` next to this file is the CUDA version (GPU) and ``fmm_m2l_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import atan2, log

import numpy as np


def fmm_m2l(
    pairs: "int64[:,:]",
    count: int,
    node_center: "float[:,:]",
    moment_re: "float[:,:]",
    moment_im: "float[:,:]",
    binomial: "float[:,:]",
    symmetric: int,
    inv_re: "float[:]",
    inv_im: "float[:]",
    local_re: "float[:,:]",
    local_im: "float[:,:]",
) -> None:
    """Add each pair's multipole of one cell to the other's local expansion.

    ``symmetric`` nonzero truncates the translated series by total degree
    (``k + l <= p``, Dehnen's symmetric truncation) instead of ``k <= p`` and
    ``l <= p``: the pair interaction then depends on ``z_i - z_j`` only, so the
    forces of a mutual pair are exactly antisymmetric. ``inv_re``/``inv_im`` are
    scratch buffers of length 2 p + 2.
    """
    order = moment_re.shape[1] - 1
    # pairs are independent; the local expansions are shared, so they are updated atomically
    # fmt: off
    #$ omp parallel for firstprivate(pw_re, pw_im) private(direction, t_re, t_im, t2, w_re, w_im, k, kmax, a0_re, a0_im, log_re, log_im, b_re, b_im, sign, l, src, dst, ak_re, ak_im, s_re, s_im, term_re, term_im, c, pw_re_ptr, pw_im_ptr) schedule(dynamic, 64)
    # fmt: on
    for pair in range(count):
        pw_re = np.empty(2 * order + 2, dtype=float)
        pw_im = np.empty(2 * order + 2, dtype=float)
        for direction in range(2):
            if direction == 0:
                src = pairs[pair, 0]
                dst = pairs[pair, 1]
            else:
                src = pairs[pair, 1]
                dst = pairs[pair, 0]
            # t = c_dst - c_src; powers of 1/t.
            t_re = node_center[dst, 0] - node_center[src, 0]
            t_im = node_center[dst, 1] - node_center[src, 1]
            t2 = t_re * t_re + t_im * t_im
            w_re = t_re / t2
            w_im = -t_im / t2
            pw_re[0] = 1.0
            pw_im[0] = 0.0
            for k in range(1, 2 * order + 2):
                pw_re[k] = pw_re[k - 1] * w_re - pw_im[k - 1] * w_im
                pw_im[k] = pw_re[k - 1] * w_im + pw_im[k - 1] * w_re
            a0_re = moment_re[src, 0]
            a0_im = moment_im[src, 0]
            # b_0 = a_0 log t - sum_k a_k / (k t**k)
            log_re = 0.5 * log(t2)
            log_im = atan2(t_im, t_re)
            b_re = a0_re * log_re - a0_im * log_im
            b_im = a0_re * log_im + a0_im * log_re
            for k in range(1, order + 1):
                ak_re = moment_re[src, k] / k
                ak_im = moment_im[src, k] / k
                b_re -= ak_re * pw_re[k] - ak_im * pw_im[k]
                b_im -= ak_re * pw_im[k] + ak_im * pw_re[k]
            # fmt: off
            #$ omp atomic
            # fmt: on
            local_re[dst, 0] += b_re
            # fmt: off
            #$ omp atomic
            # fmt: on
            local_im[dst, 0] += b_im
            # b_l = (-1)**(l+1) a_0 / (l t**l)
            #       - (-1)**l / t**l sum_k (a_k/k) C(k+l-1, l) / t**k
            sign = 1.0
            for l in range(1, order + 1):
                sign = -sign
                s_re = 0.0
                s_im = 0.0
                kmax = order
                if symmetric != 0:
                    kmax = order - l
                for k in range(1, kmax + 1):
                    c = binomial[k + l - 1, l] / k
                    ak_re = moment_re[src, k] * c
                    ak_im = moment_im[src, k] * c
                    s_re += ak_re * pw_re[k] - ak_im * pw_im[k]
                    s_im += ak_re * pw_im[k] + ak_im * pw_re[k]
                # sign is (-1)**l here.
                term_re = -sign * a0_re / l - sign * s_re
                term_im = -sign * a0_im / l - sign * s_im
                # fmt: off
                #$ omp atomic
                # fmt: on
                local_re[dst, l] += term_re * pw_re[l] - term_im * pw_im[l]
                # fmt: off
                #$ omp atomic
                # fmt: on
                local_im[dst, l] += term_re * pw_im[l] + term_im * pw_re[l]
        del pw_re
        del pw_im
