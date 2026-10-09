"""A Metal kernel whose outputs are atomics, for the FMM kernels that accumulate.

The accumulating kernels (``fmm_m2l``, ``fmm_evaluate``, ``soften_pairs``) cannot
add into their caller's arrays in place: MLX arrays are immutable and the Apple GPU
has only 32-bit atomics. They add into zeroed float32 (``atomic<float>``) delta
arrays instead, and the host adds those to the caller's arrays in float64, so the
sums differ from the CPU in the last float32 bits and in their order.
"""

from __future__ import annotations

import cunumpy as xp

from tree_d_3d.kernels._metal import THREADGROUP


def make_atomic_kernel(
    name: str,
    source: str,
    inputs: list[str],
    outputs: list[str],
    header: str = "",
) -> xp.kernels.MetalKernel:
    """A float32 :class:`~cunumpy.kernels.MetalKernel` with ``atomic<T>`` outputs, zeroed."""
    return xp.kernels.MetalKernel(
        source,
        inputs=inputs,
        outputs=outputs,
        name=name,
        header=header,
        threadgroup=THREADGROUP,
        float64="cast",
        atomic_outputs=True,
        init_value=0,
    )
