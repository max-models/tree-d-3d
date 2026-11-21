"""The quadtree_build kernel of the tree build on the CPU: Pyccel and NumPy versions with
the same arguments.

The whole build in one call: level by level, the nodes that split get their four
children, and the particles are partitioned stably into them. The GPU builds a level with ``quadtree_quadrants``, prefix sums and
``quadtree_scatter``.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
