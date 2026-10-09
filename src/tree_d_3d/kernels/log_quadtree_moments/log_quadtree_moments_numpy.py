"""The NumPy version of the ``log_quadtree_moments`` kernel, for machines without Pyccel.

Vectorized over the particles of each node; the same arguments and results as ``log_quadtree_moments_pyccel.py``.
"""

from __future__ import annotations

import numpy as np


def log_quadtree_moments(
    pos: np.ndarray,
    charge: np.ndarray,
    perm: np.ndarray,
    node_count: int,
    node_center: np.ndarray,
    node_start: np.ndarray,
    node_end: np.ndarray,
    node_qabs: np.ndarray,
    moment_re: np.ndarray,
    moment_im: np.ndarray,
) -> None:
    """Complex multipole moments a_0 ... a_p of every node."""
    powers = np.arange(moment_re.shape[1])
    for node in range(node_count):
        j = perm[node_start[node] : node_end[node]]
        q = charge[j]
        z = (pos[0, j] - node_center[node, 0]) + 1j * (pos[1, j] - node_center[node, 1])
        moments = np.sum(q[:, None] * z[:, None] ** powers[None, :], axis=0)
        moment_re[node] = moments.real
        moment_im[node] = moments.imag
        node_qabs[node] = np.sum(np.abs(q))
