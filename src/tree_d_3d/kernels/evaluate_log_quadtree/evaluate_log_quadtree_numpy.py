"""The version of the ``evaluate_log_quadtree`` kernel for machines without Pyccel.

There is no vectorized form (each target walks the tree with its own stack): this is the algorithm of
``evaluate_log_quadtree_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.evaluate_log_quadtree.evaluate_log_quadtree_pyccel import (
    evaluate_log_quadtree,
)

__all__ = ["evaluate_log_quadtree"]
