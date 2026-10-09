"""The Pyccel version (CPU) of the ``evaluate_log_quadtree`` kernel of the Barnes-Hut and FMM solvers.

``evaluate_log_quadtree_cuda.cu`` next to this file is the CUDA version (GPU) and ``evaluate_log_quadtree_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import log

import numpy as np


def evaluate_log_quadtree(
    targets: "float[:,:]",
    exclude: "int64[:]",
    pos: "float[:,:]",
    charge: "float[:]",
    perm: "int64[:]",
    node_center: "float[:,:]",
    node_half: "float[:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    node_qabs: "float[:]",
    moment_re: "float[:,:]",
    moment_im: "float[:,:]",
    theta: float,
    softening: float,
    stack: "int64[:]",
    phi: "float[:]",
    field: "float[:,:]",
    status: "int64[:]",
) -> None:
    """Line-charge potential and field at targets, without k_e.

    Same traversal and opening rule as :func:`evaluate_quadtree`. Accepted
    cells use the order-p expansion, unsoftened: for a target at z,
    ``sum q/(z - z_j) = sum_n a_n w**(n+1)`` and
    ``sum q ln(z - z_j) = a_0 ln(z - c) - sum_{n>=1} a_n w**n / n`` with
    ``w = 1 / (z - c)``. ``status[0]`` becomes 1 + a target that met an unsoftened coincident charge
    (the first, on the CPU).
    """
    order = moment_re.shape[1] - 1
    eps2 = softening * softening
    num_targets = targets.shape[1]
    stack_size = stack.shape[0]
    num_bad = 0
    # targets are independent and each walks the tree with its own stack ``walk`` (the
    # ``stack`` argument is the serial scratch); walks differ in length, hence dynamic
    # scheduling; a failing target only flags itself and the first is reported below
    # fmt: off
    #$ omp parallel for firstprivate(walk) private(tx, ty, skip, potential, ex, ey, bad, top, node, first, k, j, c, dx, dy, r2, xmid, ymid, half, rx, ry, d2, inside, w_re, w_im, prev_re, prev_im, cur_re, cur_im, f_re, f_im, series, n, a_re, a_im, next_re, child, walk_ptr) reduction(+:num_bad) schedule(dynamic, 64)
    # fmt: on
    for i in range(num_targets):
        tx = targets[0, i]
        ty = targets[1, i]
        skip = exclude[i]
        bad = False
        potential = 0.0
        ex = 0.0
        ey = 0.0
        walk = np.empty(stack_size, dtype=np.int64)
        walk[0] = 0
        top = 1
        while top > 0:
            top -= 1
            node = walk[top]
            if node_qabs[node] == 0.0:
                continue
            first = node_child[node]
            if first < 0:
                for k in range(node_start[node], node_end[node]):
                    j = perm[k]
                    c = charge[j]
                    if j == skip or c == 0.0:
                        continue
                    dx = tx - pos[0, j]
                    dy = ty - pos[1, j]
                    r2 = dx * dx + dy * dy + eps2
                    if r2 == 0.0:
                        bad = True
                        break
                    potential -= c * log(r2)
                    ex += 2.0 * c * dx / r2
                    ey += 2.0 * c * dy / r2
                if bad:
                    top = 0  # leave the walk
                continue
            xmid = node_center[node, 0]
            ymid = node_center[node, 1]
            half = node_half[node]
            rx = tx - xmid
            ry = ty - ymid
            d2 = rx * rx + ry * ry
            inside = (
                tx >= xmid - half
                and tx <= xmid + half
                and ty >= ymid - half
                and ty <= ymid + half
            )
            if not inside and 4.0 * half * half < theta * theta * d2:
                w_re = rx / d2
                w_im = -ry / d2
                # prev = w**n, cur = w**(n+1)
                prev_re = 1.0
                prev_im = 0.0
                cur_re = w_re
                cur_im = w_im
                f_re = 0.0
                f_im = 0.0
                series = 0.0
                for n in range(order + 1):
                    a_re = moment_re[node, n]
                    a_im = moment_im[node, n]
                    f_re += a_re * cur_re - a_im * cur_im
                    f_im += a_re * cur_im + a_im * cur_re
                    if n > 0:
                        series += (a_re * prev_re - a_im * prev_im) / n
                    prev_re = cur_re
                    prev_im = cur_im
                    next_re = cur_re * w_re - cur_im * w_im
                    cur_im = cur_re * w_im + cur_im * w_re
                    cur_re = next_re
                potential += -moment_re[node, 0] * log(d2) + 2.0 * series
                ex += 2.0 * f_re
                ey -= 2.0 * f_im
            else:
                for child in range(4):
                    walk[top] = first + child
                    top += 1
        del walk
        if bad:
            num_bad += 1
            phi[i] = 1.0e300  # marks the failure for the scan below
        else:
            phi[i] = potential
            field[0, i] = ex
            field[1, i] = ey
    if num_bad > 0:
        for i in range(num_targets):
            if phi[i] == 1.0e300:
                phi[i] = 0.0
                status[0] = i + 1
                return
