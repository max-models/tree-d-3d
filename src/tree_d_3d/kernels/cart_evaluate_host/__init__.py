"""The cart_evaluate_host kernel of the Coulomb FMM on the CPU: Pyccel and NumPy versions with the same
arguments.

The local expansions at the leaves' particles (L2P) and the mutual softened near field
(P2P), on the particles in tree order. The Cartesian expansions and their index tables are described in
`tree_d_3d._cartesian`. There is no GPU version.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
