"""The pair_csr kernel of the FMM on the CPU: Pyccel and NumPy versions with the same
arguments.

Turns a list of mutual node pairs into rows by node (compressed sparse rows): for every
node, the partners of all its pairs, so that a kernel can sum into each node without
atomics.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
