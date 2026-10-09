"""The Pyccel version (CPU) of the ``fmm_downward`` kernel of the Barnes-Hut and FMM solvers.

``fmm_downward_cuda.cu`` next to this file is the CUDA version (GPU) and ``fmm_downward_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

import numpy as np


def fmm_downward(
    first_node: int,
    last_node: int,
    node_center: "float[:,:]",
    node_child: "int64[:]",
    binomial: "float[:,:]",
    power_re: "float[:]",
    power_im: "float[:]",
    local_re: "float[:,:]",
    local_im: "float[:,:]",
) -> None:
    """Shift the local expansions of nodes ``first_node:last_node`` into their children.

    Call it level by level from the root: a node's expansion is complete only after
    its parent's was shifted into it (the nodes of a level are contiguous).
    """
    order = local_re.shape[1] - 1
    # the nodes of a level shift into disjoint children, so they are independent
    # fmt: off
    #$ omp parallel for firstprivate(sh_re, sh_im) private(first, child, s_re, s_im, k, m, acc_re, acc_im, l, c, b_re, b_im, sh_re_ptr, sh_im_ptr) schedule(dynamic, 16)
    # fmt: on
    for node in range(first_node, last_node):
        sh_re = np.empty(order + 1, dtype=float)
        sh_im = np.empty(order + 1, dtype=float)
        first = node_child[node]
        if first < 0:
            del sh_re
            del sh_im
            continue
        for child in range(first, first + 4):
            s_re = node_center[child, 0] - node_center[node, 0]
            s_im = node_center[child, 1] - node_center[node, 1]
            sh_re[0] = 1.0
            sh_im[0] = 0.0
            for k in range(1, order + 1):
                sh_re[k] = sh_re[k - 1] * s_re - sh_im[k - 1] * s_im
                sh_im[k] = sh_re[k - 1] * s_im + sh_im[k - 1] * s_re
            for m in range(order + 1):
                acc_re = 0.0
                acc_im = 0.0
                for l in range(m, order + 1):
                    c = binomial[l, m]
                    b_re = local_re[node, l] * c
                    b_im = local_im[node, l] * c
                    acc_re += b_re * sh_re[l - m] - b_im * sh_im[l - m]
                    acc_im += b_re * sh_im[l - m] + b_im * sh_re[l - m]
                local_re[child, m] += acc_re
                local_im[child, m] += acc_im
        del sh_re
        del sh_im
