"""The version of the ``pair_csr`` kernel for machines without Pyccel.

The algorithm of ``pair_csr_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.pair_csr.pair_csr_pyccel import pair_csr

__all__ = ["pair_csr"]
