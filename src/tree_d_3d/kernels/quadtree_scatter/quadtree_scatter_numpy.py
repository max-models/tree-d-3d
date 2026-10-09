"""The NumPy version of the ``quadtree_scatter`` kernel, for machines without Pyccel.

Vectorized over the particles; the same arguments and results as
``quadtree_scatter_pyccel.py``.
"""

from __future__ import annotations

import numpy as np


def quadtree_scatter(
    perm: np.ndarray,
    slot_node: np.ndarray,
    quadrant: np.ndarray,
    below: np.ndarray,
    node_start: np.ndarray,
    node_child: np.ndarray,
    new_perm: np.ndarray,
    new_slot_node: np.ndarray,
) -> None:
    """One level of the tree build: move every particle of a split node into its child."""
    stay = quadrant < 0
    new_perm[stay] = perm[stay]
    new_slot_node[stay] = slot_node[stay]
    k = np.flatnonzero(~stay)
    q = quadrant[k]
    node = slot_node[k]
    start = node_start[node]
    before = np.where(start > 0, below[np.maximum(start - 1, 0), q], 0)
    child = node_child[node] + q
    slot = node_start[child] + below[k, q] - 1 - before
    new_perm[slot] = perm[k]
    new_slot_node[slot] = child
