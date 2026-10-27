"""The Metal version (Apple-silicon GPU, float32) of the ``log_quadtree_moments`` kernel.

``log_quadtree_moments_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per node; only the first ``node_count`` rows of the outputs are
written. See :mod:`tree_d_3d.kernels._metal`.
"""

from __future__ import annotations

from tree_d_3d._types import Array
from tree_d_3d.kernels import KERNEL_PACKAGE

_SOURCE = """
    uint node = thread_position_in_grid.x;
    if (node >= node_center_shape[0]) return;
    int order = order_in[0];
    float xmid = node_center[2 * node], ymid = node_center[2 * node + 1];
    float qabs = 0.0f;
    uint n_pos = pos_shape[1];
    for (int64_t k = node_start[node]; k < node_end[node]; ++k) {
        int64_t j = perm[k];
        float q = charge[j];
        qabs += metal::abs(q);
        float dx = pos[j] - xmid, dy = pos[n_pos + j] - ymid;
        float power_re = q, power_im = 0.0f;
        for (int n = 0; n <= order; ++n) {
            moment_re[node * (order + 1) + n] += power_re;
            moment_im[node * (order + 1) + n] += power_im;
            float next_re = power_re * dx - power_im * dy;
            power_im = power_re * dy + power_im * dx;
            power_re = next_re;
        }
    }
    node_qabs[node] = qabs;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "tree_d_3d_log_quadtree_moments",
    _SOURCE,
    inputs=[
        "pos",
        "charge",
        "perm",
        "node_center",
        "node_start",
        "node_end",
        "order_in",
    ],
    outputs=["node_qabs", "moment_re", "moment_im"],
)


def log_quadtree_moments(
    pos: Array,
    charge: Array,
    perm: Array,
    node_count: int,
    node_center: Array,
    node_start: Array,
    node_end: Array,
    node_qabs: Array,
    moment_re: Array,
    moment_im: Array,
) -> None:
    """Complex multipole moments a_0 ... a_p of every node."""
    if node_count:
        n = int(node_count)
        _KERNEL(
            pos,
            charge,
            perm,
            node_center[:n],
            node_start[:n],
            node_end[:n],
            int(moment_re.shape[1] - 1),
            out=(node_qabs[:n], moment_re[:n], moment_im[:n]),
        )
