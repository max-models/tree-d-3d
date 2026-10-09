"""Argument checks and array conversions shared by the solvers."""

from __future__ import annotations

from typing import Any

import cunumpy as xp
import numpy as np


def _finite(name: str, value: float, *, minimum=None, positive=False) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _coincident() -> ValueError:
    return ValueError("Coincident charges require positive softening")


def _check(status) -> None:
    """Raise for the failure a kernel reported in ``status[0]``."""
    code = int(status[0])
    if code > 0:
        raise _coincident()
    if code < 0:
        raise RuntimeError(
            "the quadtree is too deep for the traversal stack of the CUDA kernels "
            "(TREE_D_3D_MAX_STACK in tree_d_3d/kernels/barnes_hut.cuh); lower max_depth"
        )


def _device(array, dtype: Any = float, *, compiled: bool = True):
    """A contiguous array of the active backend (NumPy for the uncompiled kernels)."""
    module = xp if compiled else np
    if not compiled:
        array = xp.to_numpy(array) if not isinstance(array, np.ndarray) else array
    return module.ascontiguousarray(module.asarray(array, dtype=dtype))


def _host(array) -> np.ndarray:
    """A NumPy array (from either backend)."""
    return np.asarray(xp.to_numpy(array))


def _points(array, name: str, *, compiled: bool = True):
    """``array`` as a contiguous component-major ``(2, N)`` array of the active backend."""
    points = _device(array, compiled=compiled)
    if points.ndim != 2 or points.shape[0] != 2:
        raise ValueError(
            f"{name} must have shape (2, N) (component-major), got {points.shape}"
        )
    # A (2, 1) column picked by fancy indexing keeps F-order strides that NumPy still
    # flags as C-contiguous; Pyccel checks the strides, so canonicalize them.
    itemsize = points.dtype.itemsize
    if tuple(points.strides) != (points.shape[1] * itemsize, itemsize):
        points = points.copy(order="C")
    return points
