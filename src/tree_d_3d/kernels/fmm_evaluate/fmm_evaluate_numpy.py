"""The version of the ``fmm_evaluate`` kernel for machines without Pyccel.

There is no vectorized form (the near field walks the leaf pairs): this is the algorithm of
``fmm_evaluate_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.fmm_evaluate.fmm_evaluate_pyccel import fmm_evaluate

__all__ = ["fmm_evaluate"]
