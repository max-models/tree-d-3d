"""The version of the ``cart_downward`` kernel for machines without Pyccel.

The algorithm of ``cart_downward_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.cart_downward.cart_downward_pyccel import cart_downward

__all__ = ["cart_downward"]
