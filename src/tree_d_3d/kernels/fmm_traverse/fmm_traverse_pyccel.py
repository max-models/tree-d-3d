"""The Pyccel version (CPU) of the ``fmm_traverse`` kernel of the Barnes-Hut and FMM solvers.

``fmm_traverse_cuda.cu`` next to this file is the CUDA version (GPU) and ``fmm_traverse_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import sqrt


def fmm_traverse(
    node_center: "float[:,:]",
    node_half: "float[:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    theta: float,
    frontier: "int64[:,:]",
    n_frontier: int,
    next_frontier: "int64[:,:]",
    m2l: "int64[:,:]",
    p2p: "int64[:,:]",
    counts: "int64[:]",
) -> None:
    """One generation of the dual-tree traversal, which fills mutual M2L and P2P lists.

    Each node pair of ``frontier[:n_frontier]`` becomes an M2L pair (well
    separated: (r_a + r_b) < theta d, with r the half-diagonal and d the centre
    distance), a P2P pair (adjacent leaves) or the pairs of its children, appended
    to ``next_frontier``. Start from the single pair (0, 0) and repeat with the new
    frontier until it is empty. ``counts`` holds the numbers of M2L, P2P and next
    frontier pairs, and pairs are appended at those positions; ``counts[3]``
    becomes -1 if a buffer is too small (pairs that do not fit are counted, not
    written, so the counts are the sizes needed).
    """
    root2 = sqrt(2.0)
    for k in range(n_frontier):
        a = frontier[k, 0]
        b = frontier[k, 1]
        if node_end[a] == node_start[a] or node_end[b] == node_start[b]:
            continue
        leaf_a = node_child[a] < 0
        leaf_b = node_child[b] < 0
        if a == b:
            if leaf_a:
                i = counts[1]
                counts[1] = i + 1
                if i < p2p.shape[0]:
                    p2p[i, 0] = a
                    p2p[i, 1] = a
                else:
                    counts[3] = -1
                continue
            first = node_child[a]
            i = counts[2]
            counts[2] = i + 10
            if i + 10 > next_frontier.shape[0]:
                counts[3] = -1
                continue
            for c in range(4):
                for d in range(c, 4):
                    next_frontier[i, 0] = first + c
                    next_frontier[i, 1] = first + d
                    i += 1
            continue
        dx = node_center[a, 0] - node_center[b, 0]
        dy = node_center[a, 1] - node_center[b, 1]
        distance = sqrt(dx * dx + dy * dy)
        radii = root2 * (node_half[a] + node_half[b])
        if radii < theta * distance:
            i = counts[0]
            counts[0] = i + 1
            if i < m2l.shape[0]:
                m2l[i, 0] = a
                m2l[i, 1] = b
            else:
                counts[3] = -1
            continue
        if leaf_a and leaf_b:
            i = counts[1]
            counts[1] = i + 1
            if i < p2p.shape[0]:
                p2p[i, 0] = a
                p2p[i, 1] = b
            else:
                counts[3] = -1
            continue
        i = counts[2]
        counts[2] = i + 4
        if i + 4 > next_frontier.shape[0]:
            counts[3] = -1
            continue
        if leaf_b or (not leaf_a and node_half[a] >= node_half[b]):
            first = node_child[a]
            for c in range(4):
                next_frontier[i + c, 0] = first + c
                next_frontier[i + c, 1] = b
        else:
            first = node_child[b]
            for c in range(4):
                next_frontier[i + c, 0] = a
                next_frontier[i + c, 1] = first + c
