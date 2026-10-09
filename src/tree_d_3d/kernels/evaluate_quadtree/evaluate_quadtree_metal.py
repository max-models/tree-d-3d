"""The Metal version (Apple-silicon GPU, float32) of the ``evaluate_quadtree`` kernel.

``evaluate_quadtree_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per target, each walking the tree with a stack in thread
memory (the ``stack`` argument is the CPU's scratch and unused here); a tree too
deep for it reports status -1. See :mod:`tree_d_3d.kernels._metal`.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels._metal import BH_HEADER, make_kernel, merge_status

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n_targets = targets_shape[1];
    if (i >= n_targets) return;
    uint n_pos = pos_shape[1];
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
                float inv_r = 1.0f / metal::precise::sqrt(r2);
                float inv_r3 = inv_r * inv_r * inv_r;
                potential += c * inv_r;
                ex += c * inv_r3 * dx;
                ey += c * inv_r3 * dy;
            }
            continue;
        }
        float xmid = node_center[2 * node], ymid = node_center[2 * node + 1];
        float hw = node_half[node];
        float rx = tx - xmid, ry = ty - ymid;
        float distance = metal::precise::sqrt(rx * rx + ry * ry);
        bool inside = tx >= xmid - hw && tx <= xmid + hw &&
                      ty >= ymid - hw && ty <= ymid + hw;
        if (!inside && 2.0f * hw < th * distance) {
            float inv_r = 1.0f / metal::precise::sqrt(distance * distance + eps2);
            float inv_r2 = inv_r * inv_r;
            float inv_r3 = inv_r2 * inv_r;
            float inv_r5 = inv_r3 * inv_r2;
            float inv_r7 = inv_r5 * inv_r2;
            float q = node_q[node];
            float px = node_p[2 * node], py = node_p[2 * node + 1];
            float sxx = node_s[3 * node], sxy = node_s[3 * node + 1], syy = node_s[3 * node + 2];
            float pr = px * rx + py * ry;
            float srx = sxx * rx + sxy * ry;
            float sry = sxy * rx + syy * ry;
            float rsr = rx * srx + ry * sry;
            float trace = sxx + syy;
            potential += q * inv_r + pr * inv_r3 + 0.5f * (3.0f * rsr * inv_r5 - trace * inv_r3);
            float radial = q * inv_r3 + 3.0f * pr * inv_r5 + 7.5f * rsr * inv_r7 - 1.5f * trace * inv_r5;
            ex += radial * rx - px * inv_r3 - 3.0f * srx * inv_r5;
            ey += radial * ry - py * inv_r3 - 3.0f * sry * inv_r5;
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

_KERNEL = make_kernel(
    "tree_d_3d_evaluate_quadtree",
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
        "node_q",
        "node_qabs",
        "node_p",
        "node_s",
        "theta",
        "softening",
    ],
    outputs=["phi", "field", "flags"],
    header=BH_HEADER,
)


def evaluate_quadtree(
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
    node_q: Array,
    node_qabs: Array,
    node_p: Array,
    node_s: Array,
    theta: float,
    softening: float,
    stack: Array,
    phi: Array,
    field: Array,
    status: Array,
) -> None:
    """Barnes-Hut potential and field with dipole and quadrupole cells on the GPU."""
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
        node_q,
        node_qabs,
        node_p,
        node_s,
        float(theta),
        float(softening),
        out=(phi, field, flags),
    )
    merge_status(flags, status)
