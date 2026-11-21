"""The version of the ``cart_moments_upward`` kernel for machines without Pyccel.

The algorithm of ``cart_moments_upward_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.cart_moments_upward.cart_moments_upward_pyccel import (
    cart_moments_upward,
)

__all__ = ["cart_moments_upward"]
