"""The log_moments_upward kernel of the tree on the CPU: Pyccel and NumPy versions with the
same arguments.

The complex multipole moments of every cell in one upward pass: the leaves sum their
particles, the other cells shift their children's moments (M2M), so each particle is
read once instead of once per level. The GPU sums each cell's particles directly
(``log_quadtree_moments``).
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
