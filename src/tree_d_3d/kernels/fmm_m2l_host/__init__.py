"""The fmm_m2l_host kernel of the FMM on the CPU: Pyccel and NumPy versions with the same
arguments.

The M2L translations grouped by destination cell: each cell sums the translated
expansions of all its partners in registers and writes its local expansion once, with
no atomics. The GPU runs ``fmm_m2l`` over the mutual pairs instead.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
