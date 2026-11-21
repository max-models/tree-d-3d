"""The cart_moments_upward kernel of the Coulomb FMM on the CPU: Pyccel and NumPy versions with the same
arguments.

The Cartesian multipole moments of every cell in one upward pass: leaves from their
particles, the other cells by shifting their children's moments (M2M). The Cartesian expansions and their index tables are described in
`tree_d_3d._cartesian`. There is no GPU version.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
