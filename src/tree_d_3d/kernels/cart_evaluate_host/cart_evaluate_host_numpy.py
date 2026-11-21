"""The version of the ``cart_evaluate_host`` kernel for machines without Pyccel.

The algorithm of ``cart_evaluate_host_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.cart_evaluate_host.cart_evaluate_host_pyccel import (
    cart_evaluate_host,
)

__all__ = ["cart_evaluate_host"]
