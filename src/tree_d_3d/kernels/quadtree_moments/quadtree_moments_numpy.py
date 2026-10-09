"""The NumPy version of the ``quadtree_moments`` kernel, for machines without Pyccel.

Vectorized over the particles of each node; the same arguments and results as ``quadtree_moments_pyccel.py``.
"""

from __future__ import annotations

import numpy as np


def quadtree_moments(
    pos: np.ndarray,
    charge: np.ndarray,
    perm: np.ndarray,
    node_count: int,
    node_center: np.ndarray,
    node_start: np.ndarray,
    node_end: np.ndarray,
    node_q: np.ndarray,
    node_qabs: np.ndarray,
    node_p: np.ndarray,
    node_s: np.ndarray,
) -> None:
    """Signed moments of every node about its centre; ``node_s`` is xx, xy, yy."""
    for node in range(node_count):
        j = perm[node_start[node] : node_end[node]]
        c = charge[j]
        dx = pos[0, j] - node_center[node, 0]
        dy = pos[1, j] - node_center[node, 1]
        node_q[node] = np.sum(c)
        node_qabs[node] = np.sum(np.abs(c))
        node_p[node, 0] = np.sum(c * dx)
        node_p[node, 1] = np.sum(c * dy)
        node_s[node, 0] = np.sum(c * dx * dx)
        node_s[node, 1] = np.sum(c * dx * dy)
        node_s[node, 2] = np.sum(c * dy * dy)
