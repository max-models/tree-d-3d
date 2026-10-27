"""The Barnes-Hut and FMM kernels: one folder per kernel.

Every kernel ``<name>`` lives in its own folder::

    tree_d_3d/kernels/<name>/__init__.py        kernel = KERNEL_PACKAGE.from_folder(__name__)
    tree_d_3d/kernels/<name>/<name>_pyccel.py   CPU, compiled with Pyccel on first use
    tree_d_3d/kernels/<name>/<name>_numpy.py    CPU, when Pyccel is unavailable
    tree_d_3d/kernels/<name>/<name>_cuda.cu     GPU (CuPy), compiled with NVRTC
    tree_d_3d/kernels/<name>/<name>_metal.py    Apple GPU (MLX), float32

All versions take the same arguments; a kernel runs the CUDA version on CuPy arrays
and the Pyccel version (or NumPy without Pyccel) on NumPy arrays. Importing this
package needs neither Pyccel nor CuPy.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import cunumpy as xp
from cunumpy.kernels import metal_available

#: The build cache and launch options of every kernel folder (``TREE_D_3D_KERNEL_CACHE``,
#: ``TREE_D_3D_OPENMP``); see :class:`cunumpy.kernels.KernelPackage`.
KERNEL_PACKAGE = xp.kernels.KernelPackage("tree-d-3d")

#: The kernels, by name; each has a folder ``tree_d_3d.kernels.<name>``.
KERNEL_NAMES = (
    "quadtree_quadrants",
    "quadtree_scatter",
    "quadtree_moments",
    "evaluate_quadtree",
    "direct_sum",
    "pair_energy",
    "log_quadtree_moments",
    "evaluate_log_quadtree",
    "direct_log_sum",
    "fmm_traverse",
    "fmm_m2l",
    "fmm_downward",
    "fmm_evaluate",
    "soften_pairs",
)

#: The kernels with a Metal version (all of them).
METAL_KERNEL_NAMES = KERNEL_NAMES

_COMPILED: bool | None = None


class _Uncompiled:
    """A kernel's plain-Python version, called like the kernel (launch sizes ignored)."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self._function = function

    def __call__(self, *args: Any, n_threads: Any = None) -> Any:
        return self._function(*args)


def _kernel_objects() -> dict[str, Any]:
    return {
        name: importlib.import_module(f"{__name__}.{name}").kernel
        for name in KERNEL_NAMES
    }


def __getattr__(name: str) -> Any:
    """``KERNELS``: every kernel as a :class:`cunumpy.kernels.KernelCatalog`, loaded on first use."""
    if name == "KERNELS":
        catalog = xp.kernels.KernelCatalog(_kernel_objects())
        globals()["KERNELS"] = catalog
        return catalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _kernels(compiled: bool = True) -> SimpleNamespace:
    """The kernels by name, as a namespace.

    Each runs its Pyccel version on NumPy arrays (its NumPy version without Pyccel)
    and its CUDA version on CuPy arrays. With ``compiled=False``, the plain-Python
    versions (reference results, tests).
    """
    kernels = _kernel_objects()
    if not compiled:
        return SimpleNamespace(
            **{
                name: _Uncompiled(kernel.host_kernel.kernel.python)
                for name, kernel in kernels.items()
            }
        )
    return SimpleNamespace(**kernels)


def kernels_compiled() -> bool:
    """Whether the kernels run compiled.

    On the CuPy backend they always do (CUDA); otherwise this checks once per
    process whether the Pyccel versions compile. Without them the tree kernels
    run as Python loops, so ``method="auto"`` sums directly.
    """
    global _COMPILED
    if xp.get_backend() == "cupy":
        return True
    if _COMPILED is None:
        failed = [
            name
            for name, kernel in _kernel_objects().items()
            if not kernel.host_kernel.kernel.available("pyccel")
        ]
        _COMPILED = not failed
        if failed:
            warnings.warn(
                f"Barnes-Hut kernels {failed} could not be compiled; their NumPy "
                "versions run instead (the tree kernels as Python loops)",
                RuntimeWarning,
                stacklevel=2,
            )
    return _COMPILED


def metal_kernel(name: str) -> Callable[..., None]:
    """Return the Metal version of kernel ``name``: a function with the Pyccel arguments.

    Needs MLX and a Metal GPU (``pip install 'tree-d-3d[metal]'``); the arrays are NumPy
    arrays, computed in float32 and written back into the output arrays.

    Raises:
        KeyError: If the kernel has no Metal version.
        RuntimeError: If there is no usable Metal GPU.
    """
    if name not in METAL_KERNEL_NAMES:
        raise KeyError(f"{name!r} has no Metal version; have {METAL_KERNEL_NAMES}")
    if not metal_available():
        raise RuntimeError("Metal kernels need MLX and a Metal GPU (tree-d-3d[metal])")
    module = importlib.import_module(f"{__name__}.{name}.{name}_metal")
    return getattr(module, name)
