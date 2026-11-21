"""The cart_m2l_host kernel of the Coulomb FMM on the CPU: Pyccel and NumPy versions with the same
arguments.

The M2L translations grouped by destination cell: Taylor coefficients of the softened
kernel by recurrence, contracted with the source's moments. The Cartesian expansions and their index tables are described in
`tree_d_3d._cartesian`. There is no GPU version.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
