"""The version of the ``evaluate_groups`` kernel for machines without Pyccel.

The algorithm of ``evaluate_groups_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.evaluate_groups.evaluate_groups_pyccel import evaluate_groups

__all__ = ["evaluate_groups"]
