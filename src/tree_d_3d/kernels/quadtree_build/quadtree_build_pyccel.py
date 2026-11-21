"""The Pyccel version (CPU) of the ``quadtree_build`` kernel of the tree build.

``quadtree_build_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`). The flat quadtree is described in `tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors


def quadtree_build(
    sorted_pos: "float[:,:]",
    leaf_capacity: int,
    max_depth: int,
    node_center: "float[:,:]",
    node_half: "float[:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    node_depth: "int64[:]",
    perm: "int64[:]",
    levels: "int64[:]",
    slot_node: "int64[:]",
    quadrant: "int64[:]",
    cursor: "int64[:]",
    new_sorted_pos: "float[:,:]",
    new_perm: "int64[:]",
    new_slot_node: "int64[:]",
    sizes: "int64[:]",
) -> None:
    """Build the tree of the ``n`` particles at ``sorted_pos[:, :n]`` (``n = perm.shape[0]``).

    On entry node 0 holds the root (``node_center[0]``, ``node_half[0]``) and
    ``sorted_pos`` the positions in input order. A node holding more than
    ``leaf_capacity`` particles above ``max_depth`` splits into four children (0..3:
    upper left, upper right, lower left, lower right), numbered breadth first in the
    order of their parents; a particle on a dividing line goes left or up, and the
    particles of a child keep their order. On exit ``perm`` lists the particles in tree
    order, ``sorted_pos`` holds their positions in that order (``new_sorted_pos``,
    ``new_perm``, ``new_slot_node``, ``slot_node``, ``quadrant`` and ``cursor`` are
    scratch), ``levels[:n_levels + 1]`` bounds the nodes of each depth, ``sizes[0]`` is
    the node count and ``sizes[1]`` the number of levels. If the node arrays (length
    ``cursor.shape[0]``) are too small, ``sizes[2]`` becomes 1 and the tree is unfinished.
    """
    n = perm.shape[0]
    capacity = cursor.shape[0]
    for k in range(n):
        perm[k] = k
        slot_node[k] = 0
    node_start[0] = 0
    node_end[0] = n
    node_depth[0] = 0
    levels[0] = 0
    levels[1] = 1
    n_levels = 1
    lo = 0
    hi = 1
    sizes[2] = 0
    while True:
        count = hi
        for node in range(lo, hi):
            if (
                node_end[node] - node_start[node] > leaf_capacity
                and node_depth[node] < max_depth
            ):
                if count + 4 > capacity:
                    sizes[2] = 1
                    return
                node_child[node] = count
                half = 0.5 * node_half[node]
                for c in range(4):
                    node_half[count + c] = half
                    node_depth[count + c] = node_depth[node] + 1
                    cursor[count + c] = 0
                node_center[count, 0] = node_center[node, 0] - half
                node_center[count, 1] = node_center[node, 1] + half
                node_center[count + 1, 0] = node_center[node, 0] + half
                node_center[count + 1, 1] = node_center[node, 1] + half
                node_center[count + 2, 0] = node_center[node, 0] - half
                node_center[count + 2, 1] = node_center[node, 1] - half
                node_center[count + 3, 0] = node_center[node, 0] + half
                node_center[count + 3, 1] = node_center[node, 1] - half
                count += 4
            else:
                node_child[node] = -1
        if count == hi:
            break
        # the quadrant of every particle in a splitting node, and the children's sizes
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
        for k in range(n):
            perm[k] = new_perm[k]
            slot_node[k] = new_slot_node[k]
            sorted_pos[0, k] = new_sorted_pos[0, k]
            sorted_pos[1, k] = new_sorted_pos[1, k]
        lo = hi
        hi = count
        n_levels += 1
        levels[n_levels] = hi
    sizes[0] = hi
    sizes[1] = n_levels
