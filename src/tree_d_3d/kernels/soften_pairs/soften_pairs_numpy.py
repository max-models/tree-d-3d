"""The NumPy version of the ``soften_pairs`` kernel, for machines without Pyccel.

Vectorized over the listed pairs; the same arguments and results as ``soften_pairs_pyccel.py``.
"""

from __future__ import annotations

import numpy as np

#: Targets per block of the all-pairs arrays (bounds the memory to CHUNK x sources).
CHUNK = 256


def soften_pairs(
    pos: np.ndarray,
    charge: np.ndarray,
    pairs: np.ndarray,
    eps2: float,
    with_potential: int,
    phi: np.ndarray,
    field: np.ndarray,
    status: np.ndarray,
) -> None:
    """Replace the bare line-charge kernel by the softened one for listed pairs.

    Adds, for each pair, the difference between the softened
    (-q ln(r**2 + eps**2), 2 q d / (r**2 + eps**2)) and the bare kernel, to
    both partners. ``status[0]`` becomes 1 + a pair index at zero separation.
    """
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[:, i] - pos[:, j]  # (2, pairs)
    r2 = np.einsum("ki,ki->i", d, d)
    zero = np.flatnonzero(r2 == 0.0)
    if zero.size:
        status[0] = int(zero[0]) + 1
        return
    soft = r2 + eps2
    if with_potential != 0:
        dphi = -np.log(soft / r2)
        np.add.at(phi, i, charge[j] * dphi)
        np.add.at(phi, j, charge[i] * dphi)
    f = (2.0 / soft - 2.0 / r2)[None, :] * d
    for k in range(field.shape[0]):
        np.add.at(field[k], i, charge[j] * f[k])
        np.add.at(field[k], j, -charge[i] * f[k])
