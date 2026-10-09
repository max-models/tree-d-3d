"""The quadtree_moments kernel of the Barnes-Hut and FMM solvers: Pyccel, NumPy and CUDA versions
with the same arguments.

``kernel(...)`` runs the CUDA version when the arrays are CuPy arrays, and otherwise the
Pyccel version, or the NumPy version if Pyccel is unavailable (see
``cunumpy.kernels.set_host_kernel_implementation`` to choose).
"""

import cunumpy as xp

from tree_d_3d.kernels._build import KERNEL_OPTIONS

kernel = xp.kernels.Kernel.from_folder(__name__, **KERNEL_OPTIONS)
