"""The Pyccel version (CPU) of the ``evaluate_quadtree`` kernel of the Barnes-Hut and FMM solvers.

``evaluate_quadtree_cuda.cu`` next to this file is the CUDA version (GPU) and ``evaluate_quadtree_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import sqrt

import numpy as np


def evaluate_quadtree(
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
    node_q: "float[:]",
    node_qabs: "float[:]",
    node_p: "float[:,:]",
    node_s: "float[:,:]",
    theta: float,
    softening: float,
    stack: "int64[:]",
    phi: "float[:]",
    field: "float[:,:]",
    status: "int64[:]",
) -> None:
    """Potential and field at targets, without ``k_e``.

    Source ``exclude[i]`` is skipped for target ``i`` (-1 skips nothing).
    Cells containing the target are always opened, and the opening test uses
    unsoftened distance. ``status[0]`` becomes 1 + a target that met an unsoftened coincident charge
    (the first, on the CPU).
    """
    eps2 = softening * softening
    num_targets = targets.shape[1]
    stack_size = stack.shape[0]
    num_bad = 0
    # targets are independent and each walks the tree with its own stack ``walk`` (the
    # ``stack`` argument is the serial scratch); walks differ in length, hence dynamic
    # scheduling; a failing target only flags itself and the first is reported below
    # fmt: off
    #$ omp parallel for firstprivate(walk) private(tx, ty, skip, potential, ex, ey, bad, top, node, first, k, j, c, dx, dy, r2, inv_r, inv_r2, inv_r3, inv_r5, inv_r7, xmid, ymid, half, rx, ry, distance, inside, q, px, py, sxx, sxy, syy, pr, srx, sry, rsr, trace, radial, child, walk_ptr) reduction(+:num_bad) schedule(dynamic, 64)
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
                    inv_r = 1.0 / sqrt(r2)
                    inv_r3 = inv_r * inv_r * inv_r
                    potential += c * inv_r
                    ex += c * inv_r3 * dx
                    ey += c * inv_r3 * dy
                if bad:
                    top = 0  # leave the walk
                continue
            xmid = node_center[node, 0]
            ymid = node_center[node, 1]
            half = node_half[node]
            rx = tx - xmid
            ry = ty - ymid
            distance = sqrt(rx * rx + ry * ry)
            inside = (
                tx >= xmid - half
                and tx <= xmid + half
                and ty >= ymid - half
                and ty <= ymid + half
            )
            if not inside and 2.0 * half < theta * distance:
                inv_r = 1.0 / sqrt(distance * distance + eps2)
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2
                inv_r7 = inv_r5 * inv_r2
                q = node_q[node]
                px = node_p[node, 0]
                py = node_p[node, 1]
                sxx = node_s[node, 0]
                sxy = node_s[node, 1]
                syy = node_s[node, 2]
                pr = px * rx + py * ry
                srx = sxx * rx + sxy * ry
                sry = sxy * rx + syy * ry
                rsr = rx * srx + ry * sry
                trace = sxx + syy
                potential += (
                    q * inv_r
                    + pr * inv_r3
                    + 0.5 * (3.0 * rsr * inv_r5 - trace * inv_r3)
                )
                radial = (
                    q * inv_r3
                    + 3.0 * pr * inv_r5
                    + 7.5 * rsr * inv_r7
                    - 1.5 * trace * inv_r5
                )
                ex += radial * rx - px * inv_r3 - 3.0 * srx * inv_r5
                ey += radial * ry - py * inv_r3 - 3.0 * sry * inv_r5
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
