"""The Pyccel version (CPU) of the ``cart_evaluate_host`` kernel of the Coulomb FMM.

``cart_evaluate_host_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`).
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

from math import sqrt

import numpy as np


def cart_evaluate_host(
    spos: "float[:,:]",
    scharge: "float[:]",
    node_count: int,
    node_center: "float[:,:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    local: "float[:,:]",
    mono_parent: "int64[:]",
    mono_dir: "int64[:]",
    ex: "int64[:]",
    ey: "int64[:]",
    grad: "int64[:,:]",
    pairs: "int64[:,:]",
    count: int,
    softening: float,
    phi: "float[:]",
    field: "float[:,:]",
    status: "int64[:]",
) -> None:
    """Far field from the leaves' local expansions plus the exact mutual near field.

    The particles are in tree order (``spos``, ``scharge``), and ``phi`` and ``field``
    (zero on entry) are filled in that order. The near field is the softened Coulomb
    kernel over the leaf pairs ``pairs[:count]`` (a leaf with itself once); every
    particle is a target and self-interaction is excluded. ``status[0]`` becomes 1 +
    the tree slot of a particle that met an unsoftened coincident charge.
    """
    n_coef = local.shape[1]
    eps2 = softening * softening
    # fmt: off
    #$ omp parallel for firstprivate(mono) private(k, i, w, gx, gy, mono_ptr) schedule(dynamic, 16)
    # fmt: on
    for node in range(node_count):
        mono = np.empty(n_coef, dtype=float)
        if node_child[node] < 0:
            for k in range(node_start[node], node_end[node]):
                mono[0] = 1.0
                for i in range(1, n_coef):
                    if mono_dir[i] == 0:
                        mono[i] = mono[mono_parent[i]] * (
                            spos[0, k] - node_center[node, 0]
                        )
                    else:
                        mono[i] = mono[mono_parent[i]] * (
                            spos[1, k] - node_center[node, 1]
                        )
                w = 0.0
                gx = 0.0
                gy = 0.0
                for i in range(n_coef):
                    w += local[node, i] * mono[i]
                    if grad[i, 0] >= 0:
                        gx += ex[i] * local[node, i] * mono[grad[i, 0]]
                    if grad[i, 1] >= 0:
                        gy += ey[i] * local[node, i] * mono[grad[i, 1]]
                phi[k] += w
                field[0, k] -= gx
                field[1, k] -= gy
        del mono
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
                inv_r = 1.0 / sqrt(r2)
                inv_r3 = inv_r * inv_r * inv_r
                pi += qj * inv_r
                phi[kb] += qi * inv_r
                fxi += qj * inv_r3 * dx
                fyi += qj * inv_r3 * dy
                field[0, kb] -= qi * inv_r3 * dx
                field[1, kb] -= qi * inv_r3 * dy
            phi[ka] += pi
            field[0, ka] += fxi
            field[1, ka] += fyi
