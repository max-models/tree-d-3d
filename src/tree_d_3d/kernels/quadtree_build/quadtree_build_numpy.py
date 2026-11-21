"""The version of the ``quadtree_build`` kernel for machines without Pyccel.

The algorithm of ``quadtree_build_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.quadtree_build.quadtree_build_pyccel import quadtree_build

__all__ = ["quadtree_build"]
