"""The Pyccel version (CPU) of the ``pair_energy`` kernel of the Barnes-Hut and FMM solvers.

``pair_energy_cuda.cu`` next to this file is the CUDA version (GPU) and ``pair_energy_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import sqrt


def pair_energy(
    pos: "float[:,:]",
    charge: "float[:]",
    softening: float,
    status: "int64[:]",
    energy: "float[:]",
) -> None:
    """Add the exact sum of q_i q_j / sqrt(r**2 + eps**2) over pairs, without
    ``k_e``, to ``energy[0]``; ``status[0]`` is 1 for coincident unsoftened charges."""
    eps2 = softening * softening
    total = 0.0
    num_bad = 0
    n = pos.shape[1]
    # row i has n - i - 1 pairs, so hand out the rows dynamically
    # fmt: off
    #$ omp parallel for private(ci, partial, j, cj, dx, dy, r2, bad) reduction(+:total, num_bad) schedule(dynamic, 16)
    # fmt: on
    for i in range(n):
        ci = charge[i]
        if ci == 0.0:
            continue
        partial = 0.0
        bad = False
        for j in range(i + 1, n):
            cj = charge[j]
            if cj == 0.0:
                continue
            dx = pos[0, i] - pos[0, j]
            dy = pos[1, i] - pos[1, j]
            r2 = dx * dx + dy * dy + eps2
            if r2 == 0.0:
                bad = True
                break
            partial += cj / sqrt(r2)
        if bad:
            num_bad += 1
        else:
            total += ci * partial
    if num_bad > 0:
        status[0] = 1
        return
    energy[0] += total
