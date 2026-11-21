"""The version of the ``dual_traverse`` kernel for machines without Pyccel.

The algorithm of ``dual_traverse_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.dual_traverse.dual_traverse_pyccel import dual_traverse

__all__ = ["dual_traverse"]
