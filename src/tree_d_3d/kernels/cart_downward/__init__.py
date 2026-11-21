"""The cart_downward kernel of the Coulomb FMM on the CPU: Pyccel and NumPy versions with the same
arguments.

The L2L shifts of the local expansions from every cell into its children, level by
level from the root. The Cartesian expansions and their index tables are described in
`tree_d_3d._cartesian`. There is no GPU version.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
