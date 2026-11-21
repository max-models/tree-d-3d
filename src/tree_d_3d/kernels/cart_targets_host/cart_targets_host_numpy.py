"""The version of the ``cart_targets_host`` kernel for machines without Pyccel.

The algorithm of ``cart_targets_host_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.cart_targets_host.cart_targets_host_pyccel import (
    cart_targets_host,
)

__all__ = ["cart_targets_host"]
