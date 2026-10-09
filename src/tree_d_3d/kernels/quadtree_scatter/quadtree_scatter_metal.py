"""The Metal version (Apple-silicon GPU) of the ``quadtree_scatter`` kernel.

``quadtree_scatter_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per sorted slot, each writing its own target slot (the targets
are a permutation); integer only, so exact. See :mod:`tree_d_3d.kernels._metal`.
"""

from __future__ import annotations

from tree_d_3d._types import Array
from tree_d_3d.kernels._metal import make_kernel

_SOURCE = """
    uint k = thread_position_in_grid.x;
    if (k >= perm_shape[0]) return;
    int64_t q = quadrant[k];
    int64_t node = slot_node[k];
    if (q < 0) {
        new_perm[k] = perm[k];
        new_slot_node[k] = node;
        return;
    }
    int64_t start = node_start[node];
    int64_t before = start > 0 ? below[(start - 1) * 4 + q] : 0;
    int64_t child = node_child[node] + q;
    int64_t slot = node_start[child] + below[k * 4 + q] - 1 - before;
    new_perm[slot] = perm[k];
    new_slot_node[slot] = child;
"""

_KERNEL = make_kernel(
    "tree_d_3d_quadtree_scatter",
    _SOURCE,
    inputs=["perm", "slot_node", "quadrant", "below", "node_start", "node_child"],
    outputs=["new_perm", "new_slot_node"],
)


def quadtree_scatter(
    perm: Array,
    slot_node: Array,
    quadrant: Array,
    below: Array,
    node_start: Array,
    node_child: Array,
    new_perm: Array,
    new_slot_node: Array,
) -> None:
    """Move every particle of a split node into its child, stably."""
    if perm.shape[0]:
        _KERNEL(
            perm,
            slot_node,
            quadrant,
            below,
            node_start,
            node_child,
            out=(new_perm, new_slot_node),
        )
