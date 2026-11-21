"""The version of the ``fmm_evaluate_host`` kernel for machines without Pyccel.

The algorithm of ``fmm_evaluate_host_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.fmm_evaluate_host.fmm_evaluate_host_pyccel import (
    fmm_evaluate_host,
)

__all__ = ["fmm_evaluate_host"]
