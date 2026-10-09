"""Type aliases."""

from __future__ import annotations

from typing import TypeAlias

from numpy import ndarray

#: An array of either backend (NumPy or CuPy); annotated as a NumPy array.
Array: TypeAlias = ndarray
