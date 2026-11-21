"""The version of the ``quadtree_moments_upward`` kernel for machines without Pyccel.

The algorithm of ``quadtree_moments_upward_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.quadtree_moments_upward.quadtree_moments_upward_pyccel import (
    quadtree_moments_upward,
)

__all__ = ["quadtree_moments_upward"]
