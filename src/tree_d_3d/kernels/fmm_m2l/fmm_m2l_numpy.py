"""The version of the ``fmm_m2l`` kernel for machines without Pyccel.

There is no vectorized form (the translation of each pair is a short loop over the expansion order): this is the algorithm of
``fmm_m2l_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.fmm_m2l.fmm_m2l_pyccel import fmm_m2l

__all__ = ["fmm_m2l"]
