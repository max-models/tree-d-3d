"""The Pyccel version (CPU) of the ``soften_pairs`` kernel of the Barnes-Hut and FMM solvers.

``soften_pairs_cuda.cu`` next to this file is the CUDA version (GPU) and ``soften_pairs_numpy.py`` the
version for machines without Pyccel, all with the same arguments; the folder's
``__init__.py`` declares the kernel (see :mod:`tree_d_3d.kernels`). The flat
quadtree is described in :mod:`tree_d_3d.flat_tree`.
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import log


def soften_pairs(
    pos: "float[:,:]",
    charge: "float[:]",
    pairs: "int64[:,:]",
    eps2: float,
    with_potential: int,
    phi: "float[:]",
    field: "float[:,:]",
    status: "int64[:]",
) -> None:
    """Replace the bare line-charge kernel by the softened one for listed pairs.

    Adds, for each pair, the difference between the softened
    (-q ln(r**2 + eps**2), 2 q d / (r**2 + eps**2)) and the bare kernel, to
    both partners. ``status[0]`` becomes 1 + a pair index at zero separation.
    """
    for k in range(pairs.shape[0]):
        i = pairs[k, 0]
        j = pairs[k, 1]
        dx = pos[0, i] - pos[0, j]
        dy = pos[1, i] - pos[1, j]
        r2 = dx * dx + dy * dy
        if r2 == 0.0:
            status[0] = k + 1
            return
        soft = r2 + eps2
        if with_potential != 0:
            dphi = -log(soft / r2)
            phi[i] += charge[j] * dphi
            phi[j] += charge[i] * dphi
        scale = 2.0 / soft - 2.0 / r2
        fx = scale * dx
        fy = scale * dy
        field[0, i] += charge[j] * fx
        field[1, i] += charge[j] * fy
        field[0, j] -= charge[i] * fx
        field[1, j] -= charge[i] * fy
