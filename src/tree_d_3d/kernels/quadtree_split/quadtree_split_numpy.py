"""The NumPy version of the ``quadtree_split`` kernel, for machines without Pyccel.

Vectorized over the particles; the same arguments and results as
``quadtree_split_pyccel.py``.
"""

from __future__ import annotations

import numpy as np


def quadtree_split(
    sorted_pos: np.ndarray,
    perm: np.ndarray,
    slot_node: np.ndarray,
    node_center: np.ndarray,
    node_child: np.ndarray,
    node_start: np.ndarray,
    node_end: np.ndarray,
    lo: int,
    hi: int,
    cursor: np.ndarray,
    quadrant: np.ndarray,
    new_sorted_pos: np.ndarray,
    new_perm: np.ndarray,
    new_slot_node: np.ndarray,
) -> None:
    """One level of the tree build; see ``quadtree_split_pyccel.py``."""
    child = node_child[slot_node]
    split = child >= 0
    node = slot_node[split]
    q = (sorted_pos[0, split] > node_center[node, 0]).astype(np.int64) + 2 * (
        sorted_pos[1, split] < node_center[node, 1]
    ).astype(np.int64)
    quadrant[:] = -1
    quadrant[split] = q
    target = child[split] + q
    np.add.at(cursor, target, 1)
    parents = lo + np.flatnonzero(node_child[lo:hi] >= 0)
    first = node_child[parents][:, None] + np.arange(4)
    sizes = cursor[first]
    starts = node_start[parents][:, None] + np.cumsum(sizes, axis=1) - sizes
    node_start[first] = starts
    node_end[first] = starts + sizes
    keep = ~split
    new_perm[keep] = perm[keep]
    new_slot_node[keep] = slot_node[keep]
    new_sorted_pos[:, keep] = sorted_pos[:, keep]
    # stable within each child: order by child, keeping the slot order
    slots = np.flatnonzero(split)
    order = np.argsort(target, kind="stable")
    counts = np.bincount(target, minlength=cursor.size)
    first_of = np.cumsum(counts) - counts
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size) - first_of[target[order]]
    destination = node_start[target] + rank
    new_perm[destination] = perm[slots]
    new_slot_node[destination] = target
    new_sorted_pos[:, destination] = sorted_pos[:, slots]
