"""The Pyccel version (CPU) of the ``quadtree_split`` kernel of the tree build.

``quadtree_split_numpy.py`` next to this file is the version for machines without Pyccel,
with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`). The flat quadtree is described in `tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors


def quadtree_split(
    sorted_pos: "float[:,:]",
    perm: "int64[:]",
    slot_node: "int64[:]",
    node_center: "float[:,:]",
    node_child: "int64[:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    lo: "int64",
    hi: "int64",
    cursor: "int64[:]",
    quadrant: "int64[:]",
    new_sorted_pos: "float[:,:]",
    new_perm: "int64[:]",
    new_slot_node: "int64[:]",
) -> None:
    """One level of the tree build: split the nodes ``lo:hi`` into their children.

    Slot ``k`` holds particle ``perm[k]``, at ``sorted_pos[:, k]``, in node
    ``slot_node[k]``: the positions travel with the slots, so they are read in order.
    A node with ``node_child`` >= 0 splits into the four children from there on
    (0..3: upper left, upper right, lower left, lower right). ``cursor`` must be zero
    for those children; ``quadrant`` is scratch. Sets the children's ``node_start``
    and ``node_end`` and writes the partitioned slots to ``new_sorted_pos``,
    ``new_perm`` and ``new_slot_node``; the particles of a child keep their order, and
    those of an unsplit node stay in their slots.
    """
    n = perm.shape[0]
    for k in range(n):
        node = slot_node[k]
        child = node_child[node]
        if child < 0:
            quadrant[k] = -1
            continue
        q = 0
        if sorted_pos[0, k] > node_center[node, 0]:
            q += 1
        if sorted_pos[1, k] < node_center[node, 1]:
            q += 2
        quadrant[k] = q
        cursor[child + q] += 1
    for node in range(lo, hi):
        child = node_child[node]
        if child < 0:
            continue
        start = node_start[node]
        for q in range(4):
            size = cursor[child + q]
            node_start[child + q] = start
            node_end[child + q] = start + size
            cursor[child + q] = start
            start += size
    for k in range(n):
        q = quadrant[k]
        if q < 0:
            slot = k
            new_slot_node[slot] = slot_node[k]
        else:
            child = node_child[slot_node[k]] + q
            slot = cursor[child]
            cursor[child] += 1
            new_slot_node[slot] = child
        new_perm[slot] = perm[k]
        new_sorted_pos[0, slot] = sorted_pos[0, k]
        new_sorted_pos[1, slot] = sorted_pos[1, k]
