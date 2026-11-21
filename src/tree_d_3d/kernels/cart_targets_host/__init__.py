"""The cart_targets_host kernel of the FMM at arbitrary targets, on the CPU: Pyccel and NumPy versions
with the same arguments.

The point-charge field at the targets of each target leaf: its Cartesian local
expansion (L2P) plus the exact softened sum over its P2P source leaves. There is no GPU version.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
