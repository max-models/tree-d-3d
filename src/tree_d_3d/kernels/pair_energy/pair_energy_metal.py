"""The Metal version (Apple-silicon GPU, float32) of the ``pair_energy`` kernel.

``pair_energy_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per particle ``i`` sums its pairs with ``j > i`` into its own
entry of ``partial`` (float32); the host adds the sum of ``partial``, in float64,
to ``energy[0]``. See :mod:`tree_d_3d.kernels._metal`.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels._metal import make_kernel

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n = pos_shape[1];
    if (i >= n) return;
    float ci = charge[i];
    if (ci == 0.0f) return;
    float eps2 = softening[0] * softening[0];
    float sum = 0.0f;
    for (uint j = i + 1; j < n; ++j) {
        float cj = charge[j];
        if (cj == 0.0f) continue;
        float dx = pos[i] - pos[j], dy = pos[n + i] - pos[n + j];
        float r2 = dx * dx + dy * dy + eps2;
        if (r2 == 0.0f) {
            flags[i] = 1;
            return;
        }
        sum += cj / metal::precise::sqrt(r2);
    }
    partial[i] = ci * sum;
"""

_KERNEL = make_kernel(
    "tree_d_3d_pair_energy",
    _SOURCE,
    inputs=["pos", "charge", "softening"],
    outputs=["partial", "flags"],
)


def pair_energy(
    pos: Array,
    charge: Array,
    softening: float,
    status: Array,
    energy: Array,
) -> None:
    """Add the pair sum of q_i q_j / sqrt(r**2 + eps**2) to ``energy[0]`` on the GPU."""
    partial = np.zeros(pos.shape[1], dtype=np.float64)
    flags = np.zeros(pos.shape[1], dtype=np.int32)
    _KERNEL(pos, charge, float(softening), out=(partial, flags))
    if flags.any():
        status[0] = 1
    energy[0] += float(partial.sum())
