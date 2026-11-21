"""The fmm_evaluate_host kernel of the FMM on the CPU: Pyccel and NumPy versions with the
same arguments.

``fmm_evaluate`` (local expansions at the leaves' particles, then the mutual near-field
pairs) on the particles in tree order, so the leaves are contiguous; the results are in
tree order too. The GPU runs ``fmm_evaluate``.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
