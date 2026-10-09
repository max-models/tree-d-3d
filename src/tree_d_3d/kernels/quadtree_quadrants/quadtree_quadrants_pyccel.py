"""The Pyccel version (CPU) of the ``quadtree_quadrants`` kernel of the Barnes-Hut and FMM solvers.

``quadtree_quadrants_cuda.cu`` next to this file is the CUDA version (GPU) and
``quadtree_quadrants_numpy.py`` the version for machines without Pyccel, all with the
same arguments; the folder's ``__init__.py`` declares the kernel (see
:mod:`tree_d_3d.kernels`). The flat quadtree is described in
:mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors


def quadtree_quadrants(
    pos: "float[:,:]",
    perm: "int64[:]",
    slot_node: "int64[:]",
    node_center: "float[:,:]",
    node_child: "int64[:]",
    quadrant: "int64[:]",
) -> None:
    """One level of the tree build: the child quadrant of every sorted particle.

    Slot ``k`` holds particle ``perm[k]`` in node ``slot_node[k]``. ``quadrant[k]``
    becomes -1 if that node does not split (``node_child`` < 0), and otherwise
    0..3 for the upper left, upper right, lower left and lower right child.
    """
    # fmt: off
    #$ omp parallel for private(node, j, q) schedule(dynamic, 4096)
    # fmt: on
    for k in range(perm.shape[0]):
        node = slot_node[k]
        if node_child[node] < 0:
            quadrant[k] = -1
            continue
        j = perm[k]
        q = 0
        if pos[0, j] > node_center[node, 0]:
            q += 1
        if pos[1, j] < node_center[node, 1]:
            q += 2
        quadrant[k] = q
