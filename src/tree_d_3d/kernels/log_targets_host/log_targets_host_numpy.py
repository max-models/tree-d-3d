"""The version of the ``log_targets_host`` kernel for machines without Pyccel.

The algorithm of ``log_targets_host_pyccel.py``, run by the Python interpreter.
"""

from tree_d_3d.kernels.log_targets_host.log_targets_host_pyccel import log_targets_host

__all__ = ["log_targets_host"]
