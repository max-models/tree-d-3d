"""The Metal version (Apple-silicon GPU, float32) of the ``direct_log_sum`` kernel.

``direct_log_sum_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per target; see :mod:`tree_d_3d.kernels._metal`.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels._metal import make_kernel, merge_status

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n_targets = targets_shape[1];
    if (i >= n_targets) return;
    uint n_sources = pos_shape[1];
    float eps2 = softening[0] * softening[0];
    float tx = targets[i], ty = targets[n_targets + i];
    int64_t skip = exclude[i];
    float p = 0.0f, fx = 0.0f, fy = 0.0f;
    for (uint j = 0; j < n_sources; ++j) {
        float c = charge[j];
        if ((int64_t)j == skip || c == 0.0f) continue;
        float dx = tx - pos[j], dy = ty - pos[n_sources + j];
        float r2 = dx * dx + dy * dy + eps2;
        if (r2 == 0.0f) {
            flags[i] = (int)(i + 1);
            return;
        }
        p -= c * metal::precise::log(r2);
        float s = 2.0f * c / r2;
        fx += s * dx;
        fy += s * dy;
    }
    phi[i] = p;
    field[i] = fx;
    field[n_targets + i] = fy;
"""

_KERNEL = make_kernel(
    "tree_d_3d_direct_log_sum",
    _SOURCE,
    inputs=["targets", "exclude", "pos", "charge", "softening"],
    outputs=["phi", "field", "flags"],
)


def direct_log_sum(
    targets: Array,
    exclude: Array,
    pos: Array,
    charge: Array,
    softening: float,
    phi: Array,
    field: Array,
    status: Array,
) -> None:
    """All-pairs line-charge potential and field on the GPU; same conventions as the CPU."""
    flags = np.zeros(targets.shape[1], dtype=np.int32)
    _KERNEL(targets, exclude, pos, charge, float(softening), out=(phi, field, flags))
    merge_status(flags, status)
