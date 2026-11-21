"""The version of the ``cart_m2l_host`` kernel for machines without Pyccel.

The algorithm of ``cart_m2l_host_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.cart_m2l_host.cart_m2l_host_pyccel import cart_m2l_host

__all__ = ["cart_m2l_host"]
