"""The quadtree_split kernel of the tree build on the CPU: Pyccel and NumPy versions with the
same arguments.

One level of the build in one call: the child quadrant of every particle, the particle
count and range of every child, and the stable partition of the particles into the children.
The GPU builds a level with ``quadtree_quadrants``, prefix sums and ``quadtree_scatter``.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
