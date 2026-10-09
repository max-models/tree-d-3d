"""The NumPy version of the ``quadtree_quadrants`` kernel, for machines without Pyccel.

Vectorized over the particles; the same arguments and results as
``quadtree_quadrants_pyccel.py``.
"""

from __future__ import annotations

import numpy as np


def quadtree_quadrants(
    pos: np.ndarray,
    perm: np.ndarray,
    slot_node: np.ndarray,
    node_center: np.ndarray,
    node_child: np.ndarray,
    quadrant: np.ndarray,
) -> None:
    """One level of the tree build: the child quadrant of every sorted particle."""
    center = node_center[slot_node]
    px, py = pos[0, perm], pos[1, perm]
    q = (px > center[:, 0]).astype(np.int64) + 2 * (py < center[:, 1])
    quadrant[:] = np.where(node_child[slot_node] < 0, -1, q)
