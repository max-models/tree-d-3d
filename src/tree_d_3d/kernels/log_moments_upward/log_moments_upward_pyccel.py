"""The Pyccel version (CPU) of the ``log_moments_upward`` kernel of the tree.

``log_moments_upward_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`). The flat quadtree is described in `tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

import numpy as np


def log_moments_upward(
    pos: "float[:,:]",
    charge: "float[:]",
    perm: "int64[:]",
    levels: "int64[:]",
    node_center: "float[:,:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    binomial: "float[:,:]",
    node_qabs: "float[:]",
    moment_re: "float[:,:]",
    moment_im: "float[:,:]",
) -> None:
    """Complex multipole moments ``a_n = sum_j q_j (z_j - c)**n``, n = 0 ... p, of every node.

    ``levels[d]:levels[d + 1]`` are the nodes at depth ``d``. Leaves sum their
    particles; a parent adds each child's moments shifted by ``s = c_child - c``,
    ``a_n += sum_m C(n, m) a_m(child) s**(n - m)``. Levels are done from the deepest
    up; the nodes of a level are independent.
    """
    order = moment_re.shape[1] - 1
    n_levels = levels.shape[0] - 1
    for level in range(n_levels - 1, -1, -1):
        # fmt: off
        #$ omp parallel for firstprivate(sh_re, sh_im) private(xmid, ymid, qabs, n, m, k, j, q, dx, dy, power_re, power_im, next_re, first, child, s_re, s_im, acc_re, acc_im, c, a_re, a_im, sh_re_ptr, sh_im_ptr) schedule(dynamic, 64)
        # fmt: on
        for node in range(levels[level], levels[level + 1]):
            sh_re = np.empty(order + 1, dtype=float)
            sh_im = np.empty(order + 1, dtype=float)
            xmid = node_center[node, 0]
            ymid = node_center[node, 1]
            for n in range(order + 1):
                moment_re[node, n] = 0.0
                moment_im[node, n] = 0.0
            first = node_child[node]
            if first < 0:
                qabs = 0.0
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
            else:
                qabs = 0.0
                for child in range(first, first + 4):
                    qabs += node_qabs[child]
                    if node_end[child] == node_start[child]:
                        continue
                    s_re = node_center[child, 0] - xmid
                    s_im = node_center[child, 1] - ymid
                    sh_re[0] = 1.0
                    sh_im[0] = 0.0
                    for n in range(1, order + 1):
                        sh_re[n] = sh_re[n - 1] * s_re - sh_im[n - 1] * s_im
                        sh_im[n] = sh_re[n - 1] * s_im + sh_im[n - 1] * s_re
                    for n in range(order + 1):
                        acc_re = 0.0
                        acc_im = 0.0
                        for m in range(n + 1):
                            c = binomial[n, m]
                            a_re = moment_re[child, m] * c
                            a_im = moment_im[child, m] * c
                            acc_re += a_re * sh_re[n - m] - a_im * sh_im[n - m]
                            acc_im += a_re * sh_im[n - m] + a_im * sh_re[n - m]
                        moment_re[node, n] += acc_re
                        moment_im[node, n] += acc_im
                node_qabs[node] = qabs
            del sh_re
            del sh_im
