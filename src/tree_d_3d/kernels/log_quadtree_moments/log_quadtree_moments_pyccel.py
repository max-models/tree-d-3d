"""The Pyccel version (CPU) of the ``log_quadtree_moments`` kernel of the Barnes-Hut and FMM solvers.

``log_quadtree_moments_cuda.cu`` next to this file is the CUDA version (GPU) and ``log_quadtree_moments_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors


def log_quadtree_moments(
    pos: "float[:,:]",
    charge: "float[:]",
    perm: "int64[:]",
    node_count: int,
    node_center: "float[:,:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_qabs: "float[:]",
    moment_re: "float[:,:]",
    moment_im: "float[:,:]",
) -> None:
    """Complex multipole moments a_0 ... a_p of every node."""
    order = moment_re.shape[1] - 1
    # fmt: off
    #$ omp parallel for private(xmid, ymid, qabs, n, k, j, q, dx, dy, power_re, power_im, next_re) schedule(dynamic, 64)
    # fmt: on
    for node in range(node_count):
        xmid = node_center[node, 0]
        ymid = node_center[node, 1]
        qabs = 0.0
        for n in range(order + 1):
            moment_re[node, n] = 0.0
            moment_im[node, n] = 0.0
        for k in range(node_start[node], node_end[node]):
            j = perm[k]
            q = charge[j]
            qabs += abs(q)
            dx = pos[0, j] - xmid
            dy = pos[1, j] - ymid
            power_re = q
            power_im = 0.0
            for n in range(order + 1):
                moment_re[node, n] += power_re
                moment_im[node, n] += power_im
                next_re = power_re * dx - power_im * dy
                power_im = power_re * dy + power_im * dx
                power_re = next_re
        node_qabs[node] = qabs
