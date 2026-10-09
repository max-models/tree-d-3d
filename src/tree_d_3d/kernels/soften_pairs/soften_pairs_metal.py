"""The Metal version (Apple-silicon GPU, float32) of the ``soften_pairs`` kernel.

``soften_pairs_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per pair adds into zeroed atomic float32 delta arrays (see
:mod:`tree_d_3d.kernels._metal_atomic`); the host adds those to ``phi`` and ``field`` in
float64. A pair at zero separation sets ``status[0]`` to its index + 1 (the first
such pair; the other pairs are still applied).
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array
from tree_d_3d.kernels._metal import merge_status
from tree_d_3d.kernels._metal_atomic import make_atomic_kernel

_SOURCE = """
    uint k = thread_position_in_grid.x;
    if (k >= pairs_shape[0]) return;
    int64_t i = pairs[2 * k], j = pairs[2 * k + 1];
    uint n = pos_shape[1];
    float dx = pos[i] - pos[j], dy = pos[n + i] - pos[n + j];
    float r2 = dx * dx + dy * dy;
    if (r2 == 0.0f) {
        atomic_store_explicit(&flags[k], (int)(k + 1), memory_order_relaxed);
        return;
    }
    float soft = r2 + params[0];
    if (params[1] != 0.0f) {
        float dphi = -metal::precise::log(soft / r2);
        atomic_fetch_add_explicit(&phi_delta[i], charge[j] * dphi, memory_order_relaxed);
        atomic_fetch_add_explicit(&phi_delta[j], charge[i] * dphi, memory_order_relaxed);
    }
    float scale = 2.0f / soft - 2.0f / r2;
    float fx = scale * dx, fy = scale * dy;
    atomic_fetch_add_explicit(&field_delta[i], charge[j] * fx, memory_order_relaxed);
    atomic_fetch_add_explicit(&field_delta[n + i], charge[j] * fy, memory_order_relaxed);
    atomic_fetch_add_explicit(&field_delta[j], -charge[i] * fx, memory_order_relaxed);
    atomic_fetch_add_explicit(&field_delta[n + j], -charge[i] * fy, memory_order_relaxed);
"""

_KERNEL = make_atomic_kernel(
    "tree_d_3d_soften_pairs",
    _SOURCE,
    inputs=["pos", "charge", "pairs", "params"],
    outputs=["phi_delta", "field_delta", "flags"],
)


def soften_pairs(
    pos: Array,
    charge: Array,
    pairs: Array,
    eps2: float,
    with_potential: int,
    phi: Array,
    field: Array,
    status: Array,
) -> None:
    """Add the softened-minus-bare kernel of the listed pairs to both partners."""
    if not pairs.shape[0]:
        return
    params = np.array([eps2, float(with_potential != 0)], dtype=np.float32)
    phi_delta = np.zeros(phi.shape)
    field_delta = np.zeros(field.shape)
    flags = np.zeros(pairs.shape[0], dtype=np.int32)
    _KERNEL(
        pos,
        charge,
        pairs,
        params,
        out=(phi_delta, field_delta, flags),
        n_threads=pairs.shape[0],
    )
    phi += phi_delta
    field += field_delta
    merge_status(flags, status)
