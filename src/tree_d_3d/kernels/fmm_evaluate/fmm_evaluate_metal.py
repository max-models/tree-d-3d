"""The Metal version (Apple-silicon GPU, float32) of the ``fmm_evaluate`` kernel.

``fmm_evaluate_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per leaf (far field) and per pair (exact near field) adds into
zeroed atomic float32 delta arrays (see :mod:`tree_d_3d.kernels._metal_atomic`); the host
adds those to ``phi`` and ``field`` in float64. An unsoftened coincident charge sets
``status[0]`` to a failing particle + 1 (the first failing thread's; the other
threads still run). See :mod:`tree_d_3d.kernels._metal` for the float32 caveats.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels._metal import merge_status
from tree_d_3d.kernels._metal_atomic import make_atomic_kernel

_SOURCE = """
    uint t = thread_position_in_grid.x;
    int64_t n_leaf_nodes = node_count[0];
    int64_t n_pairs = count[0];
    bool potential = with_potential[0] != 0;
    int order = (int)local_re_shape[1] - 1;
    int width = order + 1;
    float eps2 = softening[0] * softening[0];
    uint n = pos_shape[1];
    if ((int64_t)t < n_leaf_nodes && node_child[t] < 0) {
        int64_t node = t;
        for (int64_t k = node_start[node]; k < node_end[node]; ++k) {
            int64_t i = perm[k];
            float u_re = pos[i] - node_center[2 * node];
            float u_im = pos[n + i] - node_center[2 * node + 1];
            float w_re = local_re[node * width + order], w_im = local_im[node * width + order];
            float d_re = 0.0f, d_im = 0.0f;
            for (int l = order - 1; l >= 0; --l) {
                float nd_re = d_re * u_re - d_im * u_im + w_re;
                d_im = d_re * u_im + d_im * u_re + w_im;
                d_re = nd_re;
                float nw_re = w_re * u_re - w_im * u_im + local_re[node * width + l];
                w_im = w_re * u_im + w_im * u_re + local_im[node * width + l];
                w_re = nw_re;
            }
            atomic_fetch_add_explicit(&phi_delta[i], -2.0f * w_re, memory_order_relaxed);
            atomic_fetch_add_explicit(&field_delta[i], 2.0f * d_re, memory_order_relaxed);
            atomic_fetch_add_explicit(&field_delta[n + i], -2.0f * d_im, memory_order_relaxed);
        }
    }
    if ((int64_t)t >= n_pairs) return;
    int64_t a = pairs[2 * t], b = pairs[2 * t + 1];
    for (int64_t ka = node_start[a]; ka < node_end[a]; ++ka) {
        int64_t i = perm[ka];
        int64_t start_b = a == b ? ka + 1 : node_start[b];
        for (int64_t kb = start_b; kb < node_end[b]; ++kb) {
            int64_t j = perm[kb];
            float dx = pos[i] - pos[j], dy = pos[n + i] - pos[n + j];
            float r2 = dx * dx + dy * dy + eps2;
            if (r2 == 0.0f) {
                if (charge[i] != 0.0f && charge[j] != 0.0f) {
                    atomic_store_explicit(&flags[t], (int)(i + 1), memory_order_relaxed);
                    return;
                }
                continue;
            }
            if (potential) {
                float log_r2 = metal::precise::log(r2);
                atomic_fetch_add_explicit(&phi_delta[i], -charge[j] * log_r2, memory_order_relaxed);
                atomic_fetch_add_explicit(&phi_delta[j], -charge[i] * log_r2, memory_order_relaxed);
            }
            float fx = 2.0f * dx / r2, fy = 2.0f * dy / r2;
            atomic_fetch_add_explicit(&field_delta[i], charge[j] * fx, memory_order_relaxed);
            atomic_fetch_add_explicit(&field_delta[n + i], charge[j] * fy, memory_order_relaxed);
            atomic_fetch_add_explicit(&field_delta[j], -charge[i] * fx, memory_order_relaxed);
            atomic_fetch_add_explicit(&field_delta[n + j], -charge[i] * fy, memory_order_relaxed);
        }
    }
"""

_KERNEL = make_atomic_kernel(
    "tree_d_3d_fmm_evaluate",
    _SOURCE,
    inputs=[
        "pos",
        "charge",
        "perm",
        "node_center",
        "node_start",
        "node_end",
        "node_child",
        "local_re",
        "local_im",
        "pairs",
        "node_count",
        "count",
        "softening",
        "with_potential",
    ],
    outputs=["phi_delta", "field_delta", "flags"],
)


def fmm_evaluate(
    pos: Array,
    charge: Array,
    perm: Array,
    node_count: int,
    node_center: Array,
    node_start: Array,
    node_end: Array,
    node_child: Array,
    local_re: Array,
    local_im: Array,
    pairs: Array,
    count: int,
    softening: float,
    with_potential: int,
    phi: Array,
    field: Array,
    status: Array,
) -> None:
    """Far field from the leaf local expansions plus the exact mutual near field."""
    n_threads = max(int(node_count), int(count))
    if not n_threads:
        return
    phi_delta = np.zeros(phi.shape)
    field_delta = np.zeros(field.shape)
    flags = np.zeros(n_threads, dtype=np.int32)
    _KERNEL(
        pos,
        charge,
        perm,
        node_center,
        node_start,
        node_end,
        node_child,
        local_re,
        local_im,
        pairs[: int(count)],
        int(node_count),
        int(count),
        float(softening),
        int(with_potential),
        out=(phi_delta, field_delta, flags),
        n_threads=n_threads,
    )
    phi += phi_delta
    field += field_delta
    merge_status(flags, status)
