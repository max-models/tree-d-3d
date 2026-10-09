"""The Metal version (Apple-silicon GPU, float32) of the ``fmm_m2l`` kernel.

``fmm_m2l_pyccel.py`` next to this file is the CPU version with the same arguments.
One thread per pair adds into zeroed atomic float32 delta arrays (see
:mod:`tree_d_3d.kernels._metal_atomic`); the host adds those to the local expansions in
float64. The shifts sum alternating terms with binomials up to about C(2p, p), so in
float32 the accuracy degrades with the order p (about 1e-4 relative at p = 12); the
per-thread scratch holds orders up to ``TREE_D_3D_MAX_ORDER`` = 64, as on CUDA. The
``inv_re`` and ``inv_im`` scratch arguments of the CPU version are unused.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels._metal_atomic import make_atomic_kernel

_HEADER = "#define TREE_D_3D_MAX_ORDER 64\n"

_SOURCE = """
    uint pair = thread_position_in_grid.x;
    if (pair >= pairs_shape[0]) return;
    int order = (int)moment_re_shape[1] - 1;
    int width = order + 1;
    float powers_re[2 * TREE_D_3D_MAX_ORDER + 2];
    float powers_im[2 * TREE_D_3D_MAX_ORDER + 2];
    for (int direction = 0; direction < 2; ++direction) {
        int64_t src = direction == 0 ? pairs[2 * pair] : pairs[2 * pair + 1];
        int64_t dst = direction == 0 ? pairs[2 * pair + 1] : pairs[2 * pair];
        float t_re = node_center[2 * dst] - node_center[2 * src];
        float t_im = node_center[2 * dst + 1] - node_center[2 * src + 1];
        float t2 = t_re * t_re + t_im * t_im;
        float w_re = t_re / t2, w_im = -t_im / t2;
        powers_re[0] = 1.0f;
        powers_im[0] = 0.0f;
        for (int k = 1; k < 2 * order + 2; ++k) {
            powers_re[k] = powers_re[k - 1] * w_re - powers_im[k - 1] * w_im;
            powers_im[k] = powers_re[k - 1] * w_im + powers_im[k - 1] * w_re;
        }
        float a0_re = moment_re[src * width], a0_im = moment_im[src * width];
        float log_re = 0.5f * metal::precise::log(t2);
        float log_im = metal::precise::atan2(t_im, t_re);
        float b_re = a0_re * log_re - a0_im * log_im;
        float b_im = a0_re * log_im + a0_im * log_re;
        for (int k = 1; k <= order; ++k) {
            float ak_re = moment_re[src * width + k] / (float)k;
            float ak_im = moment_im[src * width + k] / (float)k;
            b_re -= ak_re * powers_re[k] - ak_im * powers_im[k];
            b_im -= ak_re * powers_im[k] + ak_im * powers_re[k];
        }
        atomic_fetch_add_explicit(&local_re_delta[dst * width], b_re, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_im_delta[dst * width], b_im, memory_order_relaxed);
        float sign = 1.0f;
        for (int l = 1; l <= order; ++l) {
            sign = -sign;
            float s_re = 0.0f, s_im = 0.0f;
            for (int k = 1; k <= order; ++k) {
                float c = binomial[(k + l - 1) * binomial_shape[1] + l] / (float)k;
                float ak_re = moment_re[src * width + k] * c;
                float ak_im = moment_im[src * width + k] * c;
                s_re += ak_re * powers_re[k] - ak_im * powers_im[k];
                s_im += ak_re * powers_im[k] + ak_im * powers_re[k];
            }
            float term_re = -sign * a0_re / (float)l - sign * s_re;
            float term_im = -sign * a0_im / (float)l - sign * s_im;
            atomic_fetch_add_explicit(&local_re_delta[dst * width + l],
                term_re * powers_re[l] - term_im * powers_im[l], memory_order_relaxed);
            atomic_fetch_add_explicit(&local_im_delta[dst * width + l],
                term_re * powers_im[l] + term_im * powers_re[l], memory_order_relaxed);
        }
    }
"""

_KERNEL = make_atomic_kernel(
    "tree_d_3d_fmm_m2l",
    _SOURCE,
    inputs=["pairs", "node_center", "moment_re", "moment_im", "binomial"],
    outputs=["local_re_delta", "local_im_delta"],
    header=_HEADER,
)


def fmm_m2l(
    pairs: Array,
    count: int,
    node_center: Array,
    moment_re: Array,
    moment_im: Array,
    binomial: Array,
    inv_re: Array,
    inv_im: Array,
    local_re: Array,
    local_im: Array,
) -> None:
    """Add each pair's multipole of one cell to the other's local expansion."""
    if not count:
        return
    local_re_delta = np.zeros(local_re.shape)
    local_im_delta = np.zeros(local_im.shape)
    _KERNEL(
        pairs[: int(count)],
        node_center,
        moment_re,
        moment_im,
        binomial,
        out=(local_re_delta, local_im_delta),
        n_threads=int(count),
    )
    local_re += local_re_delta
    local_im += local_im_delta
