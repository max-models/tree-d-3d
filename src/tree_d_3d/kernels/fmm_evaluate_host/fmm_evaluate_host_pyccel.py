"""The Pyccel version (CPU) of the ``fmm_evaluate_host`` kernel of the FMM.

``fmm_evaluate_host_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`). The flat quadtree is described in `tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import log


def fmm_evaluate_host(
    spos: "float[:,:]",
    scharge: "float[:]",
    node_count: int,
    node_center: "float[:,:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    local_re: "float[:,:]",
    local_im: "float[:,:]",
    pairs: "int64[:,:]",
    count: int,
    softening: float,
    with_potential: int,
    phi: "float[:]",
    field: "float[:,:]",
    status: "int64[:]",
) -> None:
    """Far field from leaf local expansions plus exact mutual near field, in tree order.

    ``spos`` and ``scharge`` are the particles in tree order (``pos[:, perm]``), and
    ``phi`` and ``field`` are filled in that order (zero on entry): the arguments and
    conventions of ``fmm_evaluate`` otherwise. ``status[0]`` becomes 1 + the tree slot
    of a particle that met an unsoftened coincident charge.
    """
    order = local_re.shape[1] - 1
    eps2 = softening * softening
    # fmt: off
    #$ omp parallel for private(k, u_re, u_im, w_re, w_im, d_re, d_im, l, nd_re, nw_re) schedule(dynamic, 16)
    # fmt: on
    for node in range(node_count):
        if node_child[node] >= 0:
            continue
        for k in range(node_start[node], node_end[node]):
            u_re = spos[0, k] - node_center[node, 0]
            u_im = spos[1, k] - node_center[node, 1]
            w_re = local_re[node, order]
            w_im = local_im[node, order]
            d_re = 0.0
            d_im = 0.0
            for l in range(order - 1, -1, -1):
                nd_re = d_re * u_re - d_im * u_im + w_re
                d_im = d_re * u_im + d_im * u_re + w_im
                d_re = nd_re
                nw_re = w_re * u_re - w_im * u_im + local_re[node, l]
                w_im = w_re * u_im + w_im * u_re + local_im[node, l]
                w_re = nw_re
            phi[k] += -2.0 * w_re
            field[0, k] += 2.0 * d_re
            field[1, k] -= 2.0 * d_im
    for pair in range(count):
        a = pairs[pair, 0]
        b = pairs[pair, 1]
        for ka in range(node_start[a], node_end[a]):
            xi = spos[0, ka]
            yi = spos[1, ka]
            qi = scharge[ka]
            pi = 0.0
            fxi = 0.0
            fyi = 0.0
            start_b = node_start[b]
            if a == b:
                start_b = ka + 1
            for kb in range(start_b, node_end[b]):
                qj = scharge[kb]
                dx = xi - spos[0, kb]
                dy = yi - spos[1, kb]
                r2 = dx * dx + dy * dy + eps2
                if r2 == 0.0:
                    if qi != 0.0 and qj != 0.0:
                        status[0] = ka + 1
                        return
                    continue
                if with_potential != 0:
                    log_r2 = log(r2)
                    pi -= qj * log_r2
                    phi[kb] -= qi * log_r2
                inv = 2.0 / r2
                fx = dx * inv
                fy = dy * inv
                fxi += qj * fx
                fyi += qj * fy
                field[0, kb] -= qi * fx
                field[1, kb] -= qi * fy
            phi[ka] += pi
            field[0, ka] += fxi
            field[1, ka] += fyi
