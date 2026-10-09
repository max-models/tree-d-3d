"""The Pyccel version (CPU) of the ``quadtree_scatter`` kernel of the Barnes-Hut and FMM solvers.

``quadtree_scatter_cuda.cu`` next to this file is the CUDA version (GPU) and
``quadtree_scatter_numpy.py`` the version for machines without Pyccel, all with the
same arguments; the folder's ``__init__.py`` declares the kernel (see
:mod:`tree_d_3d.kernels`). The flat quadtree is described in
:mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors


def quadtree_scatter(
    perm: "int64[:]",
    slot_node: "int64[:]",
    quadrant: "int64[:]",
    below: "int64[:,:]",
    node_start: "int64[:]",
    node_child: "int64[:]",
    new_perm: "int64[:]",
    new_slot_node: "int64[:]",
) -> None:
    """One level of the tree build: move every particle of a split node into its child.

    ``below[k, q]`` counts the slots up to and including ``k`` in quadrant ``q``
    (the cumulative sum of the one-hot ``quadrant``), so a particle's rank among
    its node's particles of the same quadrant is a difference of two entries: the
    partition is stable, as a serial one. ``node_start`` of the children must be
    set. Particles of unsplit nodes (``quadrant`` -1) stay in their slot.
    """
    # fmt: off
    #$ omp parallel for private(q, node, start, before, child, slot) schedule(dynamic, 4096)
    # fmt: on
    for k in range(perm.shape[0]):
        q = quadrant[k]
        node = slot_node[k]
        if q < 0:
            new_perm[k] = perm[k]
            new_slot_node[k] = node
            continue
        start = node_start[node]
        before = 0
        if start > 0:
            before = below[start - 1, q]
        child = node_child[node] + q
        slot = node_start[child] + below[k, q] - 1 - before
        new_perm[slot] = perm[k]
        new_slot_node[slot] = child
