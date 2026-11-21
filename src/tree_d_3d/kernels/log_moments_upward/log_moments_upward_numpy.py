"""The version of the ``log_moments_upward`` kernel for machines without Pyccel.

The algorithm of ``log_moments_upward_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.log_moments_upward.log_moments_upward_pyccel import (
    log_moments_upward,
)

__all__ = ["log_moments_upward"]
