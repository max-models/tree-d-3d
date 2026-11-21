"""The Pyccel version (CPU) of the ``evaluate_groups`` kernel of the Barnes-Hut walk.

``evaluate_groups_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`). The flat quadtree is described in `tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import log, sqrt

import numpy as np


def evaluate_groups(
    targets: "float[:,:]",
    exclude: "int64[:]",
    group_start: "int64[:]",
    log_kernel: int,
    with_potential: int,
    spos: "float[:,:]",
    scharge: "float[:]",
    sindex: "int64[:]",
    node_center: "float[:,:]",
    node_half: "float[:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    node_q: "float[:]",
    node_qabs: "float[:]",
    node_p: "float[:,:]",
    node_s: "float[:,:]",
    moment_re: "float[:,:]",
    moment_im: "float[:,:]",
    theta: float,
    softening: float,
    stack_size: int,
    phi: "float[:]",
    field: "float[:,:]",
) -> None:
    """Potential and field at the targets, without ``k_e``, walking the tree by groups.

    The targets of group ``g`` are ``group_start[g]:group_start[g + 1]`` (at most 62).
    ``spos``, ``scharge`` and ``sindex`` are the sources' positions, charges and
    indices in tree order (``perm``); ``exclude[i]`` is a source index skipped for
    target ``i``. ``log_kernel`` 0 is the Coulomb walk of ``evaluate_quadtree``
    (moments ``node_q``, ``node_p``, ``node_s``), 1 the line-charge walk of
    ``evaluate_log_quadtree`` (``moment_re``, ``moment_im``); ``with_potential`` 0
    skips the line-charge potential (its logarithms), leaving ``phi`` zero. A target that meets an
    unsoftened coincident charge gets ``phi = 1e300``.
    """
    eps2 = softening * softening
    n_groups = group_start.shape[0] - 1
    order = moment_re.shape[1] - 1
    # fmt: off
    #$ omp parallel for firstprivate(stack_node, stack_mask, pot, ex, ey, bad, gx_t, gy_t, gskip) private(sx, sy, sid, cb, t0, nt, b, bit, i, xlo, xhi, ylo, yhi, full, top, node, mask, first, k, c, dx, dy, r2, inv_r, inv_r2, inv_r3, inv_r5, inv_r7, xmid, ymid, half, rx, ry, distance, d2, inside, q, px, py, sxx, sxy, syy, pr, srx, sry, rsr, trace, radial, accept, reject, gx, gy, dmin, dmax, tx, ty, skip, w_re, w_im, prev_re, prev_im, cur_re, cur_im, f_re, f_im, series, n, a_re, a_im, next_re, child, stack_node_ptr, stack_mask_ptr, pot_ptr, ex_ptr, ey_ptr, bad_ptr, gx_t_ptr, gy_t_ptr, gskip_ptr) schedule(dynamic, 4)
    # fmt: on
    for g in range(n_groups):
        stack_node = np.empty(stack_size, dtype=np.int64)
        stack_mask = np.empty(stack_size, dtype=np.int64)
        pot = np.zeros(64, dtype=float)
        ex = np.zeros(64, dtype=float)
        ey = np.zeros(64, dtype=float)
        bad = np.zeros(64, dtype=np.int64)
        gx_t = np.empty(64, dtype=float)
        gy_t = np.empty(64, dtype=float)
        gskip = np.empty(64, dtype=np.int64)
        t0 = group_start[g]
        nt = group_start[g + 1] - t0
        for b in range(nt):
            gx_t[b] = targets[0, t0 + b]
            gy_t[b] = targets[1, t0 + b]
            gskip[b] = exclude[t0 + b]
        xlo = targets[0, t0]
        xhi = xlo
        ylo = targets[1, t0]
        yhi = ylo
        for b in range(1, nt):
            xlo = min(xlo, targets[0, t0 + b])
            xhi = max(xhi, targets[0, t0 + b])
            ylo = min(ylo, targets[1, t0 + b])
            yhi = max(yhi, targets[1, t0 + b])
        full = (1 << nt) - 1
        stack_node[0] = 0
        stack_mask[0] = full
        top = 1
        while top > 0:
            top -= 1
            node = stack_node[top]
            mask = stack_mask[top]
            if node_qabs[node] == 0.0:
                continue
            first = node_child[node]
            if first < 0:
                if mask == full:
                    # every target: loop over them innermost (vectorizable); an
                    # excluded pair adds an exact zero, a coincident one flags its target
                    for k in range(node_start[node], node_end[node]):
                        c = scharge[k]
                        if c == 0.0:
                            continue
                        sx = spos[0, k]
                        sy = spos[1, k]
                        sid = sindex[k]
                        if log_kernel != 0:
                            for b in range(nt):
                                dx = gx_t[b] - sx
                                dy = gy_t[b] - sy
                                r2 = dx * dx + dy * dy + eps2
                                cb = 0.0 if gskip[b] == sid else c
                                r2 = 1.0 if gskip[b] == sid else r2
                                bad[b] += 1 if r2 == 0.0 else 0
                                if with_potential != 0:
                                    pot[b] -= cb * log(r2)
                                ex[b] += 2.0 * cb * dx / r2
                                ey[b] += 2.0 * cb * dy / r2
                        else:
                            for b in range(nt):
                                dx = gx_t[b] - sx
                                dy = gy_t[b] - sy
                                r2 = dx * dx + dy * dy + eps2
                                cb = 0.0 if gskip[b] == sid else c
                                r2 = 1.0 if gskip[b] == sid else r2
                                bad[b] += 1 if r2 == 0.0 else 0
                                inv_r = 1.0 / sqrt(r2)
                                inv_r3 = inv_r * inv_r * inv_r
                                pot[b] += cb * inv_r
                                ex[b] += cb * inv_r3 * dx
                                ey[b] += cb * inv_r3 * dy
                    continue
                for b in range(nt):
                    bit = (mask >> b) & 1  # a temporary: Pyccel drops the brackets
                    if bit == 0 or bad[b] != 0:
                        continue
                    i = t0 + b
                    tx = gx_t[b]
                    ty = gy_t[b]
                    skip = gskip[b]
                    for k in range(node_start[node], node_end[node]):
                        c = scharge[k]
                        if sindex[k] == skip or c == 0.0:
                            continue
                        dx = tx - spos[0, k]
                        dy = ty - spos[1, k]
                        r2 = dx * dx + dy * dy + eps2
                        if r2 == 0.0:
                            bad[b] = 1
                            break
                        if log_kernel != 0:
                            if with_potential != 0:
                                pot[b] -= c * log(r2)
                            ex[b] += 2.0 * c * dx / r2
                            ey[b] += 2.0 * c * dy / r2
                        else:
                            inv_r = 1.0 / sqrt(r2)
                            inv_r3 = inv_r * inv_r * inv_r
                            pot[b] += c * inv_r
                            ex[b] += c * inv_r3 * dx
                            ey[b] += c * inv_r3 * dy
                continue
            xmid = node_center[node, 0]
            ymid = node_center[node, 1]
            half = node_half[node]
            # the group's box against the cell: all take it, none does, or decide each
            gx = max(max(xlo - xmid, xmid - xhi), 0.0)
            gy = max(max(ylo - ymid, ymid - yhi), 0.0)
            dmin = sqrt(gx * gx + gy * gy)
            gx = max(abs(xlo - xmid), abs(xhi - xmid))
            gy = max(abs(ylo - ymid), abs(yhi - ymid))
            dmax = sqrt(gx * gx + gy * gy)
            if (
                xhi < xmid - half
                or xlo > xmid + half
                or yhi < ymid - half
                or ylo > ymid + half
            ) and 2.0 * half < theta * dmin * (1.0 - 1.0e-12):
                accept = mask
                reject = 0
            elif 2.0 * half >= theta * dmax * (1.0 + 1.0e-12):
                accept = 0
                reject = mask
            else:
                accept = 0
                reject = 0
                for b in range(nt):
                    bit = (mask >> b) & 1
                    if bit == 0:
                        continue
                    tx = targets[0, t0 + b]
                    ty = targets[1, t0 + b]
                    rx = tx - xmid
                    ry = ty - ymid
                    inside = (
                        tx >= xmid - half
                        and tx <= xmid + half
                        and ty >= ymid - half
                        and ty <= ymid + half
                    )
                    if log_kernel != 0:
                        d2 = rx * rx + ry * ry
                        if not inside and 4.0 * half * half < theta * theta * d2:
                            accept = accept | (1 << b)
                        else:
                            reject = reject | (1 << b)
                    else:
                        distance = sqrt(rx * rx + ry * ry)
                        if not inside and 2.0 * half < theta * distance:
                            accept = accept | (1 << b)
                        else:
                            reject = reject | (1 << b)
            if accept != 0:
                for b in range(nt):
                    bit = (accept >> b) & 1
                    if bit == 0 or bad[b] != 0:
                        continue
                    rx = targets[0, t0 + b] - xmid
                    ry = targets[1, t0 + b] - ymid
                    if log_kernel != 0:
                        d2 = rx * rx + ry * ry
                        w_re = rx / d2
                        w_im = -ry / d2
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
                        if with_potential != 0:
                            pot[b] += -moment_re[node, 0] * log(d2) + 2.0 * series
                        ex[b] += 2.0 * f_re
                        ey[b] -= 2.0 * f_im
                    else:
                        inv_r = 1.0 / sqrt(rx * rx + ry * ry + eps2)
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
                        pot[b] += (
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
                        ex[b] += radial * rx - px * inv_r3 - 3.0 * srx * inv_r5
                        ey[b] += radial * ry - py * inv_r3 - 3.0 * sry * inv_r5
            if reject != 0:
                for child in range(4):
                    stack_node[top] = first + child
                    stack_mask[top] = reject
                    top += 1
        for b in range(nt):
            i = t0 + b
            if bad[b] != 0:
                phi[i] = 1.0e300
            else:
                phi[i] = pot[b]
                field[0, i] = ex[b]
                field[1, i] = ey[b]
        del stack_node
        del stack_mask
        del pot
        del ex
        del ey
        del bad
        del gx_t
        del gy_t
        del gskip
