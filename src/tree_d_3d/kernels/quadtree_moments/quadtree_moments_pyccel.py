"""The Pyccel version (CPU) of the ``quadtree_moments`` kernel of the Barnes-Hut and FMM solvers.

``quadtree_moments_cuda.cu`` next to this file is the CUDA version (GPU) and ``quadtree_moments_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors


def quadtree_moments(
    pos: "float[:,:]",
    charge: "float[:]",
    perm: "int64[:]",
    node_count: int,
    node_center: "float[:,:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_q: "float[:]",
    node_qabs: "float[:]",
    node_p: "float[:,:]",
    node_s: "float[:,:]",
) -> None:
    """Signed moments of every node about its centre; ``node_s`` is xx, xy, yy."""
    # fmt: off
    #$ omp parallel for private(q, qabs, px, py, sxx, sxy, syy, xmid, ymid, k, j, c, dx, dy) schedule(dynamic, 64)
    # fmt: on
    for node in range(node_count):
        q = 0.0
        qabs = 0.0
        px = 0.0
        py = 0.0
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        xmid = node_center[node, 0]
        ymid = node_center[node, 1]
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
        node_q[node] = q
        node_qabs[node] = qabs
        node_p[node, 0] = px
        node_p[node, 1] = py
        node_s[node, 0] = sxx
        node_s[node, 1] = sxy
        node_s[node, 2] = syy
