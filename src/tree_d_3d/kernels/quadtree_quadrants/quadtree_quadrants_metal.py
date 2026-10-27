"""The Metal version (Apple-silicon GPU, float32) of the ``quadtree_quadrants`` kernel.

``quadtree_quadrants_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per sorted slot; see :mod:`tree_d_3d.kernels._metal`. ``pos`` is
``(2, N)`` (component-major). Positions are compared with the node centres in float32, so a particle within about 1e-7 of a
centre line can fall into the other quadrant than on the CPU.
"""

from __future__ import annotations

from tree_d_3d._types import Array
from tree_d_3d.kernels import KERNEL_PACKAGE

_SOURCE = """
    uint k = thread_position_in_grid.x;
    if (k >= perm_shape[0]) return;
    int64_t node = slot_node[k];
    if (node_child[node] < 0) {
        quadrant[k] = -1;
        return;
    }
    int64_t j = perm[k];
    uint n = pos_shape[1];
    quadrant[k] = (pos[j] > node_center[2 * node] ? 1 : 0)
        + (pos[n + j] < node_center[2 * node + 1] ? 2 : 0);
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "tree_d_3d_quadtree_quadrants",
    _SOURCE,
    inputs=["pos", "perm", "slot_node", "node_center", "node_child"],
    outputs=["quadrant"],
)


def quadtree_quadrants(
    pos: Array,
    perm: Array,
    slot_node: Array,
    node_center: Array,
    node_child: Array,
    quadrant: Array,
) -> None:
    """The child quadrant of every sorted particle (-1: its node does not split)."""
    if perm.shape[0]:
        _KERNEL(pos, perm, slot_node, node_center, node_child, out=quadrant)
