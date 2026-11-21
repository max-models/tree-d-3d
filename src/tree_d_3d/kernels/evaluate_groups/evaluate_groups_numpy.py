"""The version of the ``evaluate_groups`` kernel for machines without Pyccel.

The walk of ``evaluate_groups_pyccel.py`` with the same arguments and the same terms for
every target, vectorized over (target, cell) pairs instead of walking one group at a time:
all targets descend the tree together, level by level, each pair is accepted as a
multipole, summed directly (a leaf) or replaced by the cell's four children. The sums
run in another order, so results agree with the Pyccel version to rounding. This is the
version Pyodide (the browser) runs.
"""

from __future__ import annotations

import numpy as np

#: The most (target, source) pairs of the direct sums held at once.
_CHUNK = 1 << 20


def _direct(
    t, cells, targets, exclude, spos, scharge, sindex, node_start, node_end, out
):
    """Add the direct sums of the targets ``t`` over the leaves ``cells``."""
    log_kernel, with_potential, eps2, pot, ex, ey, bad = out
    # np.repeat takes the platform's index type (32 bits in the browser) as counts
    counts = (node_end[cells] - node_start[cells]).astype(np.intp)
    keep = counts > 0
    t, cells, counts = t[keep], cells[keep], counts[keep]
    ends = np.cumsum(counts)
    begin = 0
    while begin < len(t):
        # whole leaves per chunk, at least one
        done = ends[begin] - counts[begin]
        stop = int(np.searchsorted(ends, done + _CHUNK, side="right"))
        stop = max(stop, begin + 1)
        tt, cc, nn = t[begin:stop], cells[begin:stop], counts[begin:stop]
        offset = np.repeat(np.cumsum(nn) - nn, nn)
        k = np.repeat(node_start[cc], nn) + np.arange(int(nn.sum())) - offset
        i = np.repeat(tt, nn)
        c = scharge[k]
        use = (c != 0.0) & (sindex[k] != exclude[i])
        i, k, c = i[use], k[use], c[use]
        dx = targets[0, i] - spos[0, k]
        dy = targets[1, i] - spos[1, k]
        r2 = dx * dx + dy * dy + eps2
        coincident = r2 == 0.0
        if coincident.any():
            bad[i[coincident]] = 1
            r2 = np.where(coincident, 1.0, r2)
            c = np.where(coincident, 0.0, c)
        n = len(pot)
        if log_kernel:
            if with_potential:
                pot -= np.bincount(i, c * np.log(r2), n)
            ex += np.bincount(i, 2.0 * c * dx / r2, n)
            ey += np.bincount(i, 2.0 * c * dy / r2, n)
        else:
            inv_r = 1.0 / np.sqrt(r2)
            cr3 = c * inv_r * inv_r * inv_r
            pot += np.bincount(i, c * inv_r, n)
            ex += np.bincount(i, cr3 * dx, n)
            ey += np.bincount(i, cr3 * dy, n)
        begin = stop


def _multipoles(t, cells, targets, node_center, node_q, node_p, node_s, re, im, out):
    """Add the multipole expansions of the accepted ``cells`` at the targets ``t``."""
    log_kernel, with_potential, eps2, pot, ex, ey, _ = out
    n = len(pot)
    rx = targets[0, t] - node_center[cells, 0]
    ry = targets[1, t] - node_center[cells, 1]
    if log_kernel:
        d2 = rx * rx + ry * ry
        w = (rx - 1j * ry) / d2
        a = re[cells] + 1j * im[cells]
        prev = np.ones_like(w)
        cur = w
        f = np.zeros_like(w)
        series = np.zeros(len(t))
        for k in range(a.shape[1]):
            f += a[:, k] * cur
            if k > 0:
                series += (a[:, k] * prev).real / k
            prev = cur
            cur = cur * w
        if with_potential:
            pot += np.bincount(t, -re[cells, 0] * np.log(d2) + 2.0 * series, n)
        ex += np.bincount(t, 2.0 * f.real, n)
        ey -= np.bincount(t, 2.0 * f.imag, n)
        return
    inv_r = 1.0 / np.sqrt(rx * rx + ry * ry + eps2)
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r2 * inv_r
    inv_r5 = inv_r3 * inv_r2
    inv_r7 = inv_r5 * inv_r2
    q = node_q[cells]
    px, py = node_p[cells, 0], node_p[cells, 1]
    sxx, sxy, syy = node_s[cells, 0], node_s[cells, 1], node_s[cells, 2]
    pr = px * rx + py * ry
    srx = sxx * rx + sxy * ry
    sry = sxy * rx + syy * ry
    rsr = rx * srx + ry * sry
    trace = sxx + syy
    pot += np.bincount(
        t, q * inv_r + pr * inv_r3 + 0.5 * (3.0 * rsr * inv_r5 - trace * inv_r3), n
    )
    radial = q * inv_r3 + 3.0 * pr * inv_r5 + 7.5 * rsr * inv_r7 - 1.5 * trace * inv_r5
    ex += np.bincount(t, radial * rx - px * inv_r3 - 3.0 * srx * inv_r5, n)
    ey += np.bincount(t, radial * ry - py * inv_r3 - 3.0 * sry * inv_r5, n)


def evaluate_groups(
    targets,
    exclude,
    group_start,
    log_kernel,
    with_potential,
    spos,
    scharge,
    sindex,
    node_center,
    node_half,
    node_start,
    node_end,
    node_child,
    node_q,
    node_qabs,
    node_p,
    node_s,
    moment_re,
    moment_im,
    theta,
    softening,
    stack_size,
    phi,
    field,
):
    """Potential and field at the targets, without ``k_e``: see ``evaluate_groups_pyccel``.

    The groups (``group_start``) and ``stack_size`` are not needed here: every target walks
    the tree on its own, all of them at once.
    """
    n = int(group_start[-1])
    pot, ex, ey = np.zeros(n), np.zeros(n), np.zeros(n)
    bad = np.zeros(n, dtype=np.int64)
    out = (
        log_kernel != 0,
        with_potential != 0,
        softening * softening,
        pot,
        ex,
        ey,
        bad,
    )
    t = np.arange(n)
    cells = np.zeros(n, dtype=np.int64)
    while len(t):
        live = node_qabs[cells] != 0.0
        t, cells = t[live], cells[live]
        first = node_child[cells]
        leaf = first < 0
        if leaf.any():
            _direct(
                t[leaf],
                cells[leaf],
                targets,
                exclude,
                spos,
                scharge,
                sindex,
                node_start,
                node_end,
                out,
            )
        inner = ~leaf
        t, cells, first = t[inner], cells[inner], first[inner]
        tx, ty = targets[0, t], targets[1, t]
        xmid, ymid = node_center[cells, 0], node_center[cells, 1]
        half = node_half[cells]
        rx, ry = tx - xmid, ty - ymid
        inside = (np.abs(rx) <= half) & (np.abs(ry) <= half)
        if log_kernel != 0:
            far = 4.0 * half * half < theta * theta * (rx * rx + ry * ry)
        else:
            far = 2.0 * half < theta * np.sqrt(rx * rx + ry * ry)
        accept = far & ~inside
        if accept.any():
            _multipoles(
                t[accept],
                cells[accept],
                targets,
                node_center,
                node_q,
                node_p,
                node_s,
                moment_re,
                moment_im,
                out,
            )
        t, first = t[~accept], first[~accept]
        t = np.repeat(t, 4)
        cells = (first[:, None] + np.arange(4)).reshape(-1)
    good = bad == 0
    phi[:n] = np.where(good, pot, 1.0e300)
    field[0, :n] = np.where(good, ex, field[0, :n])
    field[1, :n] = np.where(good, ey, field[1, :n])
