"""The NumPy version of the ``direct_sum`` kernel, for machines without Pyccel.

Vectorized over blocks of targets and all sources; the same arguments and results as ``direct_sum_pyccel.py``.
"""

from __future__ import annotations

import numpy as np

#: Targets per block of the all-pairs arrays (bounds the memory to CHUNK x sources).
CHUNK = 256


def _pairs(targets, exclude, pos, charge, softening, start, status):
    """Separations, squared distances and charges of one block of targets; None after
    a coincidence (``status[0]`` is then set)."""
    block = targets[:, start : start + CHUNK]
    # d[k, i, j]: component k of target i minus source j
    d = block[:, :, None] - pos[:, None, :]
    r2 = np.einsum("kij,kij->ij", d, d) + softening * softening
    c = np.broadcast_to(charge, r2.shape).copy()
    skip = exclude[start : start + CHUNK]
    rows = np.flatnonzero(skip >= 0)
    c[rows, skip[rows]] = 0.0
    bad = (c != 0.0) & (r2 == 0.0)
    if bad.any():
        status[0] = start + int(np.argmax(bad.any(axis=1))) + 1
        return None
    return d, np.where(c != 0.0, r2, 1.0), c


def direct_sum(
    targets: np.ndarray,
    exclude: np.ndarray,
    pos: np.ndarray,
    charge: np.ndarray,
    softening: float,
    phi: np.ndarray,
    field: np.ndarray,
    status: np.ndarray,
) -> None:
    """All-pairs potential and field at targets; same conventions as the tree.

    ``status[0]`` becomes 1 + a target that met an unsoftened coincident charge.
    """
    for start in range(0, targets.shape[1], CHUNK):
        result = _pairs(targets, exclude, pos, charge, softening, start, status)
        if result is None:
            return
        d, r2, c = result
        inv_r = 1.0 / np.sqrt(r2)
        stop = start + r2.shape[0]
        phi[start:stop] = np.sum(c * inv_r, axis=1)
        field[:, start:stop] = np.einsum("ij,kij->ki", c * inv_r**3, d)
