"""The Pyccel version (CPU) of the ``quadtree_moments_upward`` kernel of the tree.

``quadtree_moments_upward_numpy.py`` next to this file is the version for machines
without Pyccel, with the same arguments; the folder's ``__init__.py`` declares the
kernel (see `tree_d_3d.kernels`). The flat quadtree is described in
`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors


def quadtree_moments_upward(
    pos: "float[:,:]",
    charge: "float[:]",
    perm: "int64[:]",
    levels: "int64[:]",
    node_center: "float[:,:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    node_q: "float[:]",
    node_qabs: "float[:]",
    node_p: "float[:,:]",
    node_s: "float[:,:]",
) -> None:
    """Signed moments of every node about its centre; ``node_s`` is xx, xy, yy.

    ``levels[d]:levels[d + 1]`` are the nodes at depth ``d``. Leaves sum their
    particles; a parent adds each child's moments shifted by ``s = c_child - c``:
    ``p += p_c + q_c s`` and ``S += S_c + p_c s^T + s p_c^T + q_c s s^T``. Levels are
    done from the deepest up; the nodes of a level are independent.
    """
    n_levels = levels.shape[0] - 1
    for level in range(n_levels - 1, -1, -1):
        # fmt: off
        #$ omp parallel for private(q, qabs, px, py, sxx, sxy, syy, xmid, ymid, k, j, c, dx, dy, first, child, cq, cpx, cpy) schedule(dynamic, 64)
        # fmt: on
        for node in range(levels[level], levels[level + 1]):
            q = 0.0
            qabs = 0.0
            px = 0.0
            py = 0.0
            sxx = 0.0
            sxy = 0.0
            syy = 0.0
            xmid = node_center[node, 0]
            ymid = node_center[node, 1]
            first = node_child[node]
            if first < 0:
                for k in range(node_start[node], node_end[node]):
                    j = perm[k]
                    c = charge[j]
                    dx = pos[0, j] - xmid
                    dy = pos[1, j] - ymid
                    q += c
                    qabs += abs(c)
                    px += c * dx
                    py += c * dy
                    sxx += c * dx * dx
                    sxy += c * dx * dy
                    syy += c * dy * dy
            else:
                for child in range(first, first + 4):
                    dx = node_center[child, 0] - xmid
                    dy = node_center[child, 1] - ymid
                    cq = node_q[child]
                    cpx = node_p[child, 0]
                    cpy = node_p[child, 1]
                    q += cq
                    qabs += node_qabs[child]
                    px += cpx + cq * dx
                    py += cpy + cq * dy
                    sxx += node_s[child, 0] + 2.0 * cpx * dx + cq * dx * dx
                    sxy += node_s[child, 1] + cpx * dy + cpy * dx + cq * dx * dy
                    syy += node_s[child, 2] + 2.0 * cpy * dy + cq * dy * dy
            node_q[node] = q
            node_qabs[node] = qabs
            node_p[node, 0] = px
            node_p[node, 1] = py
            node_s[node, 0] = sxx
            node_s[node, 1] = sxy
            node_s[node, 2] = syy
