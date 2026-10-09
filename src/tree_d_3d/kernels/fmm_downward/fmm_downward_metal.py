"""The Metal version (Apple-silicon GPU, float32) of the ``fmm_downward`` kernel.

``fmm_downward_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per node of ``first_node:last_node``; a child has one parent, so
each thread writes its own children's rows of a zeroed delta (no atomics), which the
host adds to the local expansions in float64. The ``power_re`` and ``power_im``
scratch arguments of the CPU version are unused. See :mod:`tree_d_3d.kernels._metal`.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels._metal import make_kernel

_HEADER = "#define TREE_D_3D_MAX_ORDER 64\n"

_SOURCE = """
    uint t = thread_position_in_grid.x;
    int64_t node = (int64_t)range[0] + t;
    if (node >= (int64_t)range[1]) return;
    int64_t first = node_child[node];
    if (first < 0) return;
    int order = (int)local_re_shape[1] - 1;
    int width = order + 1;
    float s_pow_re[TREE_D_3D_MAX_ORDER + 1];
    float s_pow_im[TREE_D_3D_MAX_ORDER + 1];
    for (int64_t child = first; child < first + 4; ++child) {
        float s_re = node_center[2 * child] - node_center[2 * node];
        float s_im = node_center[2 * child + 1] - node_center[2 * node + 1];
        s_pow_re[0] = 1.0f;
        s_pow_im[0] = 0.0f;
        for (int k = 1; k <= order; ++k) {
            s_pow_re[k] = s_pow_re[k - 1] * s_re - s_pow_im[k - 1] * s_im;
            s_pow_im[k] = s_pow_re[k - 1] * s_im + s_pow_im[k - 1] * s_re;
        }
        for (int m = 0; m <= order; ++m) {
            float acc_re = 0.0f, acc_im = 0.0f;
            for (int l = m; l <= order; ++l) {
                float c = binomial[l * binomial_shape[1] + m];
                float b_re = local_re[node * width + l] * c;
                float b_im = local_im[node * width + l] * c;
                acc_re += b_re * s_pow_re[l - m] - b_im * s_pow_im[l - m];
                acc_im += b_re * s_pow_im[l - m] + b_im * s_pow_re[l - m];
            }
            delta_re[child * width + m] = acc_re;
            delta_im[child * width + m] = acc_im;
        }
    }
"""

_KERNEL = make_kernel(
    "tree_d_3d_fmm_downward",
    _SOURCE,
    inputs=["range", "node_center", "node_child", "binomial", "local_re", "local_im"],
    outputs=["delta_re", "delta_im"],
    header=_HEADER,
)


def fmm_downward(
    first_node: int,
    last_node: int,
    node_center: Array,
    node_child: Array,
    binomial: Array,
    power_re: Array,
    power_im: Array,
    local_re: Array,
    local_im: Array,
) -> None:
    """Shift the local expansions of nodes ``first_node:last_node`` into their children."""
    if last_node <= first_node:
        return
    delta_re = np.zeros(local_re.shape)
    delta_im = np.zeros(local_im.shape)
    node_range = np.array([first_node, last_node], dtype=np.int32)
    _KERNEL(
        node_range,
        node_center,
        node_child,
        binomial,
        local_re,
        local_im,
        out=(delta_re, delta_im),
        n_threads=int(last_node - first_node),
    )
    local_re += delta_re
    local_im += delta_im
