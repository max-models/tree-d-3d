"""The NumPy version of the ``fmm_downward`` kernel, for machines without Pyccel.

Vectorized over the children of the level; the same arguments and results as
``fmm_downward_pyccel.py``.
"""

from __future__ import annotations

import numpy as np


def fmm_downward(
    first_node: int,
    last_node: int,
    node_center: np.ndarray,
    node_child: np.ndarray,
    binomial: np.ndarray,
    power_re: np.ndarray,
    power_im: np.ndarray,
    local_re: np.ndarray,
    local_im: np.ndarray,
) -> None:
    """Shift the local expansions of nodes ``first_node:last_node`` into their children."""
    order = local_re.shape[1] - 1
    nodes = np.arange(first_node, last_node)
    nodes = nodes[node_child[nodes] >= 0]
    if not nodes.size:
        return
    parent = np.repeat(nodes, 4)
    child = (node_child[nodes][:, None] + np.arange(4)).ravel()
    shift = node_center[child] - node_center[parent]
    s = shift[:, 0] + 1j * shift[:, 1]
    powers = s[:, None] ** np.arange(order + 1)
    local = local_re[parent] + 1j * local_im[parent]
    for m in range(order + 1):
        terms = local[:, m:] * binomial[m : order + 1, m] * powers[:, : order + 1 - m]
        acc = terms.sum(axis=1)
        local_re[child, m] += acc.real
        local_im[child, m] += acc.imag
