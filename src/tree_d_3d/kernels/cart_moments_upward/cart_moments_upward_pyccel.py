"""The Pyccel version (CPU) of the ``cart_moments_upward`` kernel of the Coulomb FMM.

``cart_moments_upward_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`).
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

import numpy as np


def cart_moments_upward(
    pos: "float[:,:]",
    charge: "float[:]",
    perm: "int64[:]",
    levels: "int64[:]",
    node_center: "float[:,:]",
    node_start: "int64[:]",
    node_end: "int64[:]",
    node_child: "int64[:]",
    mono_parent: "int64[:]",
    mono_dir: "int64[:]",
    shift: "int64[:,:]",
    shift_coef: "float[:]",
    moments: "float[:,:]",
) -> None:
    """Moments ``m_g = sum_j q_j (r_j - c)^g`` of every node, ``|g| <= p``.

    ``levels[d]:levels[d + 1]`` are the nodes at depth ``d``. Leaves sum their
    particles; a parent adds each child's moments shifted by ``s = c_child - c``,
    ``m_g += sum_b C(g, b) m_b(child) s^(g - b)`` (rows ``shift``). Levels are done from
    the deepest up; the nodes of a level are independent.
    """
    n_coef = moments.shape[1]
    n_shift = shift.shape[0]
    n_levels = levels.shape[0] - 1
    for level in range(n_levels - 1, -1, -1):
        # fmt: off
        #$ omp parallel for firstprivate(mono) private(xmid, ymid, i, k, j, q, first, child, t, mono_ptr) schedule(dynamic, 64)
        # fmt: on
        for node in range(levels[level], levels[level + 1]):
            mono = np.empty(n_coef, dtype=float)
            xmid = node_center[node, 0]
            ymid = node_center[node, 1]
            for i in range(n_coef):
                moments[node, i] = 0.0
            first = node_child[node]
            if first < 0:
                for k in range(node_start[node], node_end[node]):
                    j = perm[k]
                    q = charge[j]
                    mono[0] = q
                    for i in range(1, n_coef):
                        if mono_dir[i] == 0:
                            mono[i] = mono[mono_parent[i]] * (pos[0, j] - xmid)
                        else:
                            mono[i] = mono[mono_parent[i]] * (pos[1, j] - ymid)
                    for i in range(n_coef):
                        moments[node, i] += mono[i]
            else:
                for child in range(first, first + 4):
                    if node_end[child] == node_start[child]:
                        continue
                    mono[0] = 1.0
                    for i in range(1, n_coef):
                        if mono_dir[i] == 0:
                            mono[i] = mono[mono_parent[i]] * (
                                node_center[child, 0] - xmid
                            )
                        else:
                            mono[i] = mono[mono_parent[i]] * (
                                node_center[child, 1] - ymid
                            )
                    for t in range(n_shift):
                        moments[node, shift[t, 0]] += (
                            shift_coef[t]
                            * moments[child, shift[t, 1]]
                            * mono[shift[t, 2]]
                        )
            del mono
