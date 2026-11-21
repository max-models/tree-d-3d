"""The Pyccel version (CPU) of the ``cart_targets_host`` kernel of the FMM at targets.

``cart_targets_host_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`).
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import sqrt

import numpy as np


def cart_targets_host(
    tpos: "float[:,:]",
    texclude: "int64[:]",
    t_count: int,
    t_center: "float[:,:]",
    t_start: "int64[:]",
    t_end: "int64[:]",
    t_child: "int64[:]",
    local: "float[:,:]",
    mono_parent: "int64[:]",
    mono_dir: "int64[:]",
    ex_pow: "int64[:]",
    ey_pow: "int64[:]",
    grad: "int64[:,:]",
    row_start: "int64[:]",
    partners: "int64[:]",
    spos: "float[:,:]",
    scharge: "float[:]",
    sindex: "int64[:]",
    s_start: "int64[:]",
    s_end: "int64[:]",
    softening: float,
    phi: "float[:]",
    field: "float[:,:]",
) -> None:
    """Point-charge potential and field at the targets, in target-tree order.

    A target leaf ``n`` evaluates its Cartesian local expansion and sums the source
    leaves ``partners[row_start[n]:row_start[n + 1]]`` exactly with the softened
    kernel, skipping source ``texclude[i]`` (an index of ``sindex``). A target that
    meets an unsoftened coincident charge gets ``phi = 1e300``.
    """
    n_coef = local.shape[1]
    eps2 = softening * softening
    # fmt: off
    #$ omp parallel for firstprivate(mono) private(k, i, pot, ex, ey, bad, skip, e, src, j, c, dx, dy, r2, inv_r, inv_r3, mono_ptr) schedule(dynamic, 16)
    # fmt: on
    for node in range(t_count):
        mono = np.empty(n_coef, dtype=float)
        if t_child[node] < 0:
            for k in range(t_start[node], t_end[node]):
                mono[0] = 1.0
                for i in range(1, n_coef):
                    if mono_dir[i] == 0:
                        mono[i] = mono[mono_parent[i]] * (
                            tpos[0, k] - t_center[node, 0]
                        )
                    else:
                        mono[i] = mono[mono_parent[i]] * (
                            tpos[1, k] - t_center[node, 1]
                        )
                pot = 0.0
                ex = 0.0
                ey = 0.0
                for i in range(n_coef):
                    pot += local[node, i] * mono[i]
                    if grad[i, 0] >= 0:
                        ex -= ex_pow[i] * local[node, i] * mono[grad[i, 0]]
                    if grad[i, 1] >= 0:
                        ey -= ey_pow[i] * local[node, i] * mono[grad[i, 1]]
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
                        inv_r = 1.0 / sqrt(r2)
                        inv_r3 = inv_r * inv_r * inv_r
                        pot += c * inv_r
                        ex += c * inv_r3 * dx
                        ey += c * inv_r3 * dy
                phi[k] = 1.0e300 if bad != 0 else pot
                field[0, k] = ex
                field[1, k] = ey
        del mono
