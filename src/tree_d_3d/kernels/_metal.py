"""Helpers of the Metal versions (``<name>/<name>_metal.py``) of the kernels.

A Metal version is the body of a Metal Shading Language kernel built by
:meth:`KERNEL_PACKAGE.metal_kernel <cunumpy.kernels.KernelPackage.metal_kernel>` and run on the GPU of an Apple-silicon Mac, wrapped in
a function with the arguments of the Pyccel version: NumPy arrays in, results
written into the output arrays. The GPU has no float64, so the kernels compute in
float32 (``float64="cast"``): expect a relative error of about 1e-5 against the
CPU versions. Import a version only where :func:`cunumpy.kernels.metal_available` is true.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d._types import Array

#: Depth of the per-thread traversal stack of the tree evaluations (see
#: ``TREE_D_3D_MAX_STACK`` in ``barnes_hut.cuh``).
BH_MAX_STACK = 512

#: ``int`` stack entries (cell numbers) are enough on the GPU.
BH_HEADER = f"#define TREE_D_3D_MAX_STACK {BH_MAX_STACK}\n"


def merge_status(flags: Array, status: Array) -> None:
    """Fold per-thread failure flags into ``status[0]`` like the CPU versions.

    A thread writes ``i + 1`` (or -1 for a too deep tree) to its own entry of
    ``flags``; ``status[0]`` becomes the first such value unless it is already set.
    """
    failed = np.flatnonzero(flags)
    if failed.size and status[0] == 0:
        status[0] = flags[failed[0]]
