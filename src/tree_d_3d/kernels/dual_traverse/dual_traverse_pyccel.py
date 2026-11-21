"""The Pyccel version (CPU) of the ``dual_traverse`` kernel of the FMM at targets.

``dual_traverse_numpy.py`` next to this file is the version for machines without Pyccel,
with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`).
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import sqrt


def dual_traverse(
    t_center: "float[:,:]",
    t_half: "float[:]",
    t_start: "int64[:]",
    t_end: "int64[:]",
    t_child: "int64[:]",
    s_center: "float[:,:]",
    s_half: "float[:]",
    s_start: "int64[:]",
    s_end: "int64[:]",
    s_child: "int64[:]",
    theta: float,
    stack: "int64[:,:]",
    m2l: "int64[:,:]",
    p2p: "int64[:,:]",
    counts: "int64[:]",
) -> None:
    """Pairs (target cell, source cell) from (root, root), depth first.

    A pair is an M2L pair if ``sqrt(2) (h_t + h_s) < theta d`` (``d`` the centre
    distance), a P2P pair if both are leaves, and otherwise splits the larger cell
    (the one that is not a leaf). Empty cells are dropped (the lists depend on the cells only,
    not on the charges). ``counts`` holds the numbers of M2L and P2P pairs; a
    full list or stack sets ``counts[2] = 1`` and the lists are incomplete (``counts``
    still count every pair met, a hint for the sizes).
    """
    root2 = sqrt(2.0)
    counts[0] = 0
    counts[1] = 0
    counts[2] = 0
    stack[0, 0] = 0
    stack[0, 1] = 0
    top = 1
    while top > 0:
        top -= 1
        a = stack[top, 0]
        b = stack[top, 1]
        if t_end[a] == t_start[a] or s_end[b] == s_start[b]:
            continue
        dx = t_center[a, 0] - s_center[b, 0]
        dy = t_center[a, 1] - s_center[b, 1]
        if root2 * (t_half[a] + s_half[b]) < theta * sqrt(dx * dx + dy * dy):
            i = counts[0]
            counts[0] = i + 1
            if i < m2l.shape[0]:
                m2l[i, 0] = a
                m2l[i, 1] = b
            else:
                counts[2] = 1
            continue
        leaf_a = t_child[a] < 0
        leaf_b = s_child[b] < 0
        if leaf_a and leaf_b:
            i = counts[1]
            counts[1] = i + 1
            if i < p2p.shape[0]:
                p2p[i, 0] = a
                p2p[i, 1] = b
            else:
                counts[2] = 1
            continue
        if top + 4 > stack.shape[0]:
            counts[2] = 1
            return
        if leaf_b or (not leaf_a and t_half[a] >= s_half[b]):
            for c in range(4):
                stack[top + c, 0] = t_child[a] + c
                stack[top + c, 1] = b
        else:
            for c in range(4):
                stack[top + c, 0] = a
                stack[top + c, 1] = s_child[b] + c
        top += 4
