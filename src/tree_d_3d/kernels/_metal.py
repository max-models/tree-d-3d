"""Helpers of the Metal versions (``<name>/<name>_metal.py``) of the kernels.

A Metal version is the body of a Metal Shading Language kernel run by
:class:`cunumpy.kernels.MetalKernel` on the GPU of an Apple-silicon Mac, wrapped in
a function with the arguments of the Pyccel version: NumPy arrays in, results
written into the output arrays. The GPU has no float64, so the kernels compute in
float32 (``float64="cast"``): expect a relative error of about 1e-5 against the
CPU versions. Import a version only where :func:`metal_available` is true.
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np

from tree_d_3d._types import Array

#: Threads per threadgroup of every Metal kernel.
THREADGROUP = 256

#: Depth of the per-thread traversal stack of the tree evaluations (see
#: ``TREE_D_3D_MAX_STACK`` in ``barnes_hut.cuh``).
BH_MAX_STACK = 512

#: ``int`` stack entries (cell numbers) are enough on the GPU.
BH_HEADER = f"#define TREE_D_3D_MAX_STACK {BH_MAX_STACK}\n"


def metal_available() -> bool:
    """Whether the Metal versions can run here (MLX installed, Metal GPU present)."""
    return bool(xp.kernels.metal_available())


def make_kernel(
    name: str,
    source: str,
    inputs: list[str],
    outputs: list[str],
    header: str = "",
) -> xp.kernels.MetalKernel:
    """A :class:`~cunumpy.kernels.MetalKernel` computing in float32, outputs zeroed."""
    return xp.kernels.MetalKernel(
        source,
        inputs=inputs,
        outputs=outputs,
        name=name,
        header=header,
        threadgroup=THREADGROUP,
        float64="cast",
        init_value=0,
    )


def merge_status(flags: Array, status: Array) -> None:
    """Fold per-thread failure flags into ``status[0]`` like the CPU versions.

    A thread writes ``i + 1`` (or -1 for a too deep tree) to its own entry of
    ``flags``; ``status[0]`` becomes the first such value unless it is already set.
    """
    failed = np.flatnonzero(flags)
    if failed.size and status[0] == 0:
        status[0] = flags[failed[0]]
