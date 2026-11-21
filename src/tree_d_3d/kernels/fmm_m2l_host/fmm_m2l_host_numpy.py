"""The version of the ``fmm_m2l_host`` kernel for machines without Pyccel.

The algorithm of ``fmm_m2l_host_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.fmm_m2l_host.fmm_m2l_host_pyccel import fmm_m2l_host

__all__ = ["fmm_m2l_host"]
