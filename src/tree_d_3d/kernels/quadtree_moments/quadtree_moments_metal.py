"""The Metal version (Apple-silicon GPU, float32) of the ``quadtree_moments`` kernel.

``quadtree_moments_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per node, each summing its own particles; only the first
``node_count`` rows of the outputs are written. See :mod:`tree_d_3d.kernels._metal`.
"""

from __future__ import annotations

from tree_d_3d._types import Array
from tree_d_3d.kernels import KERNEL_PACKAGE

_SOURCE = """
    uint node = thread_position_in_grid.x;
    if (node >= node_center_shape[0]) return;
    float q = 0.0f, qabs = 0.0f, px = 0.0f, py = 0.0f, sxx = 0.0f, sxy = 0.0f, syy = 0.0f;
    float xmid = node_center[2 * node], ymid = node_center[2 * node + 1];
    uint n = pos_shape[1];
    for (int64_t k = node_start[node]; k < node_end[node]; ++k) {
        int64_t j = perm[k];
        float c = charge[j];
        float dx = pos[j] - xmid, dy = pos[n + j] - ymid;
        q += c;
        qabs += metal::abs(c);
        px += c * dx;
        py += c * dy;
        sxx += c * dx * dx;
        sxy += c * dx * dy;
        syy += c * dy * dy;
    }
    node_q[node] = q;
    node_qabs[node] = qabs;
    node_p[2 * node] = px;
    node_p[2 * node + 1] = py;
    node_s[3 * node] = sxx;
    node_s[3 * node + 1] = sxy;
    node_s[3 * node + 2] = syy;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "tree_d_3d_quadtree_moments",
    _SOURCE,
    inputs=["pos", "charge", "perm", "node_center", "node_start", "node_end"],
    outputs=["node_q", "node_qabs", "node_p", "node_s"],
)


def quadtree_moments(
    pos: Array,
    charge: Array,
    perm: Array,
    node_count: int,
    node_center: Array,
    node_start: Array,
    node_end: Array,
    node_q: Array,
    node_qabs: Array,
    node_p: Array,
    node_s: Array,
) -> None:
    """Signed moments of every node about its centre; ``node_s`` is xx, xy, yy."""
    if node_count:
        n = int(node_count)
        _KERNEL(
            pos,
            charge,
            perm,
            node_center[:n],
            node_start[:n],
            node_end[:n],
            out=(node_q[:n], node_qabs[:n], node_p[:n], node_s[:n]),
        )
