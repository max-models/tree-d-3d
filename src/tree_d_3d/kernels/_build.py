"""The options every kernel folder passes to :meth:`cunumpy.kernels.Kernel.from_folder`.

The ``__init__.py`` of a kernel folder ``tree_d_3d/kernels/<name>/`` declares its
kernel with these::

    kernel = xp.kernels.Kernel.from_folder(__name__, **KERNEL_OPTIONS)

which finds ``<name>_pyccel.py``, ``<name>_numpy.py`` and ``<name>_cuda.cu`` in
the folder.
Nothing is compiled here: the Pyccel version is compiled on its first host call
(cached on disk by :func:`~tree_d_3d.kernels.compile.compile_cached`), the CUDA
version on its first launch. Importing this module needs neither Pyccel nor CuPy.
"""

from __future__ import annotations

from typing import Any

#: CUDA threads per block of every kernel.
BLOCK_SIZE = 256

#: Default dynamic shared memory per block of a CUDA device, in bytes.
SHARED_MEMORY_PER_BLOCK = 48 * 1024


def shared_memory_fits(num_values: int, itemsize: int = 8) -> bool:
    """Whether ``num_values`` values fit the default shared memory of a block."""
    return num_values * itemsize <= SHARED_MEMORY_PER_BLOCK


def _compile(module: Any) -> Any:
    from tree_d_3d.kernels.compile import compile_cached

    return compile_cached(module)


#: ``<name>_pyccel.py`` compiled with Pyccel on first use; CUDA for CuPy
#: arguments, the host version for NumPy ones (also while CuPy is active);
#: without ``n_threads`` one CUDA thread per entry of the last axis of the
#: first array (``"last_axis"``: the particle count of a component-major
#: ``(2, N)`` or ``(N,)`` particle array). A kernel whose first array is not
#: a per-particle array passes ``n_threads`` at its call sites. CUDA sources
#: include tree_d_3d headers relative to the source root (cunumpy's default), e.g.
#: ``#include "tree_d_3d/kernels/barnes_hut.cuh"``.
KERNEL_OPTIONS: dict[str, Any] = {
    "host_suffix": "_pyccel",
    "compile_host": _compile,
    "dispatch": "arrays",
    "block_size": BLOCK_SIZE,
    "n_threads_from": "last_axis",
}
