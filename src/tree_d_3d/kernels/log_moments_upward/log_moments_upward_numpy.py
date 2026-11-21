"""The version of the ``log_moments_upward`` kernel for machines without Pyccel.

``log_moments_upward_pyccel.py`` vectorized over the cells of a level: the leaves sum
the powers of all their particles at once, the parents shift the moments of all their
children at once.
"""

from __future__ import annotations

import numpy as np

from tree_d_3d.kernels._vectorized import complex_powers, expand


def log_moments_upward(
    pos,
    charge,
    perm,
    levels,
    node_center,
    node_start,
    node_end,
    node_child,
    binomial,
    node_qabs,
    moment_re,
    moment_im,
):
    """Complex multipole moments of every node, from the deepest level up.

    See ``log_moments_upward_pyccel``.
    """
    order = moment_re.shape[1] - 1
    for level in range(len(levels) - 2, -1, -1):
        nodes = np.arange(levels[level], levels[level + 1], dtype=np.intp)
        moments = np.zeros((len(nodes), order + 1), dtype=np.complex128)
        qabs = np.zeros(len(nodes))
        first = node_child[nodes]
        leaf = first < 0
        if leaf.any():
            owner, k = expand(node_start[nodes[leaf]], node_end[nodes[leaf]])
            rows = np.flatnonzero(leaf)[owner]
            j = perm[k]
            q = charge[j]
            dz = (pos[0, j] - node_center[nodes[rows], 0]) + 1j * (
                pos[1, j] - node_center[nodes[rows], 1]
            )
            np.add.at(moments, rows, q[:, None] * complex_powers(dz, order))
            qabs += np.bincount(rows, np.abs(q), len(nodes))
        if (~leaf).any():
            parent = np.flatnonzero(~leaf)
            rows = np.repeat(parent, 4)
            child = (first[parent][:, None] + np.arange(4)).reshape(-1)
            s = (node_center[child, 0] - node_center[nodes[rows], 0]) + 1j * (
                node_center[child, 1] - node_center[nodes[rows], 1]
            )
            shift = complex_powers(s, order)
            a = moment_re[child] + 1j * moment_im[child]
            shifted = np.zeros_like(a)
            for n in range(order + 1):
                for m in range(n + 1):
                    shifted[:, n] += binomial[n, m] * a[:, m] * shift[:, n - m]
            np.add.at(moments, rows, shifted)
            qabs += np.bincount(rows, node_qabs[child], len(nodes))
        moment_re[nodes] = moments.real
        moment_im[nodes] = moments.imag
        node_qabs[nodes] = qabs
