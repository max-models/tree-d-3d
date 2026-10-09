"""The Pyccel version (CPU) of the ``fmm_evaluate`` kernel of the Barnes-Hut and FMM solvers.

``fmm_evaluate_cuda.cu`` next to this file is the CUDA version (GPU) and ``fmm_evaluate_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import log


def fmm_evaluate(
    pos: "float[:,:]",
    charge: "float[:]",
    perm: "int64[:]",
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
    """Far field from leaf local expansions plus exact mutual near field.

    Every particle is a target; self-interaction is excluded. With
    ``with_potential = 0`` the near-field potential is skipped, which saves
    a logarithm per pair when only forces are needed. ``status[0]`` becomes
    1 + a particle that met an unsoftened coincident charge.
    """
    order = local_re.shape[1] - 1
    eps2 = softening * softening
    # fmt: off
    #$ omp parallel for private(k, i, u_re, u_im, w_re, w_im, d_re, d_im, l, nd_re, nw_re) schedule(dynamic, 16)
    # fmt: on
    for node in range(node_count):
        if node_child[node] >= 0:
            continue
        for k in range(node_start[node], node_end[node]):
            i = perm[k]
            u_re = pos[0, i] - node_center[node, 0]
            u_im = pos[1, i] - node_center[node, 1]
            # Horner for W = sum b_l u**l and W' = sum l b_l u**(l-1).
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
            phi[i] += -2.0 * w_re
            field[0, i] += 2.0 * d_re
            field[1, i] -= 2.0 * d_im
    for pair in range(count):
        a = pairs[pair, 0]
        b = pairs[pair, 1]
        for ka in range(node_start[a], node_end[a]):
            i = perm[ka]
            start_b = node_start[b]
            if a == b:
                start_b = ka + 1
            for kb in range(start_b, node_end[b]):
                j = perm[kb]
                dx = pos[0, i] - pos[0, j]
                dy = pos[1, i] - pos[1, j]
                r2 = dx * dx + dy * dy + eps2
                if r2 == 0.0:
                    if charge[i] != 0.0 and charge[j] != 0.0:
                        status[0] = i + 1
                        return
                    continue
                if with_potential != 0:
                    log_r2 = log(r2)
                    phi[i] -= charge[j] * log_r2
                    phi[j] -= charge[i] * log_r2
                fx = 2.0 * dx / r2
                fy = 2.0 * dy / r2
                field[0, i] += charge[j] * fx
                field[1, i] += charge[j] * fy
                field[0, j] -= charge[i] * fx
                field[1, j] -= charge[i] * fy
