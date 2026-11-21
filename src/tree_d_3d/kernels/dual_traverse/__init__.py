"""The dual_traverse kernel of the FMM at arbitrary targets, on the CPU: Pyccel and NumPy versions
with the same arguments.

The dual-tree traversal of a target tree against a source tree: the M2L pairs
(well-separated cells) and the P2P pairs (leaves that are not), in one call. There is no GPU version.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
