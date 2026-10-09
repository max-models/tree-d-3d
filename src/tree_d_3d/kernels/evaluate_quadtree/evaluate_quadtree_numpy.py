"""The version of the ``evaluate_quadtree`` kernel for machines without Pyccel.

There is no vectorized form (each target walks the tree with its own stack): this is the algorithm of
``evaluate_quadtree_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.evaluate_quadtree.evaluate_quadtree_pyccel import (
    evaluate_quadtree,
)

__all__ = ["evaluate_quadtree"]
