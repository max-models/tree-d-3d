"""The Metal version (Apple-silicon GPU, float32) of the ``evaluate_log_quadtree`` kernel.

``evaluate_log_quadtree_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per target, each walking the tree with a stack in thread
memory (the ``stack`` argument is the CPU's scratch and unused here); a tree too
deep for it reports status -1. See :mod:`tree_d_3d.kernels._metal`.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels import KERNEL_PACKAGE
from tree_d_3d.kernels._metal import BH_HEADER, merge_status

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n_targets = targets_shape[1];
    if (i >= n_targets) return;
    uint n_pos = pos_shape[1];
    int order = (int)moment_re_shape[1] - 1;
    float eps2 = softening[0] * softening[0];
    float th = theta[0];
    float tx = targets[i], ty = targets[n_targets + i];
    int64_t skip = exclude[i];
    float potential = 0.0f, ex = 0.0f, ey = 0.0f;
    int cells[TREE_D_3D_MAX_STACK];
    cells[0] = 0;
    int top = 1;
    while (top > 0) {
        int node = cells[--top];
        if (node_qabs[node] == 0.0f) continue;
        int first = (int)node_child[node];
        if (first < 0) {
            for (int64_t k = node_start[node]; k < node_end[node]; ++k) {
                int64_t j = perm[k];
                float c = charge[j];
                if (j == skip || c == 0.0f) continue;
                float dx = tx - pos[j], dy = ty - pos[n_pos + j];
                float r2 = dx * dx + dy * dy + eps2;
                if (r2 == 0.0f) {
                    flags[i] = (int)(i + 1);
                    return;
                }
                potential -= c * metal::precise::log(r2);
                float s = 2.0f * c / r2;
                ex += s * dx;
                ey += s * dy;
            }
            continue;
        }
        float xmid = node_center[2 * node], ymid = node_center[2 * node + 1];
        float hw = node_half[node];
        float rx = tx - xmid, ry = ty - ymid;
        float d2 = rx * rx + ry * ry;
        bool inside = tx >= xmid - hw && tx <= xmid + hw &&
                      ty >= ymid - hw && ty <= ymid + hw;
        if (!inside && 4.0f * hw * hw < th * th * d2) {
            float w_re = rx / d2, w_im = -ry / d2;
            // prev = w**n, cur = w**(n+1)
            float prev_re = 1.0f, prev_im = 0.0f, cur_re = w_re, cur_im = w_im;
            float f_re = 0.0f, f_im = 0.0f, series = 0.0f;
            for (int n = 0; n <= order; ++n) {
                float a_re = moment_re[node * (order + 1) + n];
                float a_im = moment_im[node * (order + 1) + n];
                f_re += a_re * cur_re - a_im * cur_im;
                f_im += a_re * cur_im + a_im * cur_re;
                if (n > 0) series += (a_re * prev_re - a_im * prev_im) / (float)n;
                prev_re = cur_re;
                prev_im = cur_im;
                float next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
            }
            potential += -moment_re[node * (order + 1)] * metal::precise::log(d2) + 2.0f * series;
            ex += 2.0f * f_re;
            ey -= 2.0f * f_im;
        } else {
            if (top + 4 > TREE_D_3D_MAX_STACK) {
                flags[i] = -1;
                return;
            }
            for (int child = 0; child < 4; ++child) cells[top++] = first + child;
        }
    }
    phi[i] = potential;
    field[i] = ex;
    field[n_targets + i] = ey;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "tree_d_3d_evaluate_log_quadtree",
    _SOURCE,
    inputs=[
        "targets",
        "exclude",
        "pos",
        "charge",
        "perm",
        "node_center",
        "node_half",
        "node_start",
        "node_end",
        "node_child",
        "node_qabs",
        "moment_re",
        "moment_im",
        "theta",
        "softening",
    ],
    outputs=["phi", "field", "flags"],
    header=BH_HEADER,
)


def evaluate_log_quadtree(
    targets: Array,
    exclude: Array,
    pos: Array,
    charge: Array,
    perm: Array,
    node_center: Array,
    node_half: Array,
    node_start: Array,
    node_end: Array,
    node_child: Array,
    node_qabs: Array,
    moment_re: Array,
    moment_im: Array,
    theta: float,
    softening: float,
    stack: Array,
    phi: Array,
    field: Array,
    status: Array,
) -> None:
    """Barnes-Hut line-charge potential and field with multipole cells on the GPU."""
    flags = np.zeros(targets.shape[1], dtype=np.int32)
    _KERNEL(
        targets,
        exclude,
        pos,
        charge,
        perm,
        node_center,
        node_half,
        node_start,
        node_end,
        node_child,
        node_qabs,
        moment_re,
        moment_im,
        float(theta),
        float(softening),
        out=(phi, field, flags),
    )
    merge_status(flags, status)
