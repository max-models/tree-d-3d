"""The Pyccel version (CPU) of the ``log_targets_host`` kernel of the FMM at targets.

``log_targets_host_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`).
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import log


def log_targets_host(
    tpos: "float[:,:]",
    texclude: "int64[:]",
    t_count: int,
    t_center: "float[:,:]",
    t_start: "int64[:]",
    t_end: "int64[:]",
    t_child: "int64[:]",
    local_re: "float[:,:]",
    local_im: "float[:,:]",
    row_start: "int64[:]",
    partners: "int64[:]",
    spos: "float[:,:]",
    scharge: "float[:]",
    sindex: "int64[:]",
    s_start: "int64[:]",
    s_end: "int64[:]",
    softening: float,
    with_potential: int,
    phi: "float[:]",
    field: "float[:,:]",
) -> None:
    """Line-charge potential and field at the targets, in target-tree order.

    A target leaf ``n`` evaluates its local expansion (``phi = -2 Re W``,
    ``E_x - i E_y = 2 W'``) and sums the source leaves
    ``partners[row_start[n]:row_start[n + 1]]`` exactly, skipping source
    ``texclude[i]`` (an index of ``sindex``). ``with_potential`` 0 leaves ``phi``
    zero. A target that meets an unsoftened coincident charge gets ``phi = 1e300``.
    """
    order = local_re.shape[1] - 1
    eps2 = softening * softening
    # fmt: off
    #$ omp parallel for private(k, u_re, u_im, w_re, w_im, d_re, d_im, l, nd_re, nw_re, pot, ex, ey, bad, skip, e, src, j, c, dx, dy, r2) schedule(dynamic, 16)
    # fmt: on
    for node in range(t_count):
        if t_child[node] >= 0:
            continue
        for k in range(t_start[node], t_end[node]):
            u_re = tpos[0, k] - t_center[node, 0]
            u_im = tpos[1, k] - t_center[node, 1]
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
            pot = -2.0 * w_re
            ex = 2.0 * d_re
            ey = -2.0 * d_im
            bad = 0
            skip = texclude[k]
            for e in range(row_start[node], row_start[node + 1]):
                src = partners[e]
                for j in range(s_start[src], s_end[src]):
                    c = scharge[j]
                    if c == 0.0 or sindex[j] == skip:
                        continue
                    dx = tpos[0, k] - spos[0, j]
                    dy = tpos[1, k] - spos[1, j]
                    r2 = dx * dx + dy * dy + eps2
                    if r2 == 0.0:
                        bad = 1
                        continue
                    if with_potential != 0:
                        pot -= c * log(r2)
                    ex += 2.0 * c * dx / r2
                    ey += 2.0 * c * dy / r2
            if bad != 0:
                phi[k] = 1.0e300
            elif with_potential != 0:
                phi[k] = pot
            else:
                phi[k] = 0.0
            field[0, k] = ex
            field[1, k] = ey
