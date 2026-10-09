"""The NumPy version of the ``pair_energy`` kernel, for machines without Pyccel.

Vectorized over blocks of particles against the later ones; the same arguments and results as ``pair_energy_pyccel.py``.
"""

from __future__ import annotations

import numpy as np

#: Targets per block of the all-pairs arrays (bounds the memory to CHUNK x sources).
CHUNK = 256


def pair_energy(
    pos: np.ndarray,
    charge: np.ndarray,
    softening: float,
    status: np.ndarray,
    energy: np.ndarray,
) -> None:
    """Add the exact sum of q_i q_j / sqrt(r**2 + eps**2) over pairs, without
    ``k_e``, to ``energy[0]``; ``status[0]`` is 1 for coincident unsoftened charges."""
    n = pos.shape[1]
    total = 0.0
    for start in range(0, n, CHUNK):
        rows = np.arange(start, min(start + CHUNK, n))
        d = pos[:, rows, None] - pos[:, None, :]
        r2 = np.einsum("kij,kij->ij", d, d) + softening * softening
        later = np.arange(n)[None, :] > rows[:, None]
        cc = charge[rows, None] * charge[None, :]
        live = later & (cc != 0.0)
        if np.any(live & (r2 == 0.0)):
            status[0] = 1
            return
        total += float(
            np.sum(np.where(live, cc / np.sqrt(np.where(live, r2, 1.0)), 0.0))
        )
    energy[0] += total
