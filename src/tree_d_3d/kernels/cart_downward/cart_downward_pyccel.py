"""The Pyccel version (CPU) of the ``cart_downward`` kernel of the Coulomb FMM.

``cart_downward_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`).
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors

import numpy as np


def cart_downward(
    levels: "int64[:]",
    node_center: "float[:,:]",
    node_child: "int64[:]",
    mono_parent: "int64[:]",
    mono_dir: "int64[:]",
    shift: "int64[:,:]",
    shift_coef: "float[:]",
    local: "float[:,:]",
) -> None:
    """Shift every cell's local expansion into its children, from the root down.

    ``T_b(child) += sum_g C(g, b) T_g s^(g - b)`` over the rows ``(g, b, g - b)`` of
    ``shift``, ``s = c_child - c``. The nodes of a level shift into disjoint children.
    """
    n_coef = local.shape[1]
    n_shift = shift.shape[0]
    n_levels = levels.shape[0] - 1
    for level in range(n_levels - 1):
        # fmt: off
        #$ omp parallel for firstprivate(mono) private(first, child, i, t, mono_ptr) schedule(dynamic, 16)
        # fmt: on
        for node in range(levels[level], levels[level + 1]):
            mono = np.empty(n_coef, dtype=float)
            first = node_child[node]
            if first >= 0:
                for child in range(first, first + 4):
                    mono[0] = 1.0
                    for i in range(1, n_coef):
                        if mono_dir[i] == 0:
                            mono[i] = mono[mono_parent[i]] * (
                                node_center[child, 0] - node_center[node, 0]
                            )
                        else:
                            mono[i] = mono[mono_parent[i]] * (
                                node_center[child, 1] - node_center[node, 1]
                            )
                    for t in range(n_shift):
                        local[child, shift[t, 1]] += (
                            shift_coef[t] * local[node, shift[t, 0]] * mono[shift[t, 2]]
                        )
            del mono
