"""The Pyccel version (CPU) of the ``direct_log_sum`` kernel of the Barnes-Hut and FMM solvers.

``direct_log_sum_cuda.cu`` next to this file is the CUDA version (GPU) and ``direct_log_sum_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import log


def direct_log_sum(
    targets: "float[:,:]",
    exclude: "int64[:]",
    pos: "float[:,:]",
    charge: "float[:]",
    softening: float,
    phi: "float[:]",
    field: "float[:,:]",
    status: "int64[:]",
) -> None:
    """All-pairs line-charge potential and field; same conventions.

    ``status[0]`` becomes 1 + a target that met an unsoftened coincident charge.
    """
    eps2 = softening * softening
    num_targets = targets.shape[1]
    num_sources = pos.shape[1]
    num_bad = 0
    # targets are independent; a failing one only flags itself and the first is reported below
    # fmt: off
    #$ omp parallel for private(tx, ty, skip, potential, ex, ey, bad, j, c, dx, dy, r2) reduction(+:num_bad) schedule(dynamic, 8)
    # fmt: on
    for i in range(num_targets):
        tx = targets[0, i]
        ty = targets[1, i]
        skip = exclude[i]
        potential = 0.0
        ex = 0.0
        ey = 0.0
        bad = False
        for j in range(num_sources):
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
