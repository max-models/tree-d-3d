"""Shared pieces of the vectorized NumPy kernels (the versions that run without Pyccel).

The NumPy kernels loop over a few small dimensions (expansion terms, tree levels) and do
everything else as whole-array operations over cells, particles or pairs of them. Index
arrays passed to ``np.repeat`` are cast to ``np.intp``, which is 32 bits in Pyodide.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

#: The most items (pairs of cells or particles) a vectorized step holds at once.
CHUNK = 1 << 18


def expand(starts, ends) -> tuple[np.ndarray, np.ndarray]:
    """For segments ``starts[i]:ends[i]``: the segment of every item, and the item.

    Returns ``(owner, item)``, with ``owner[j]`` the segment index of item ``item[j]``,
    in segment order.
    """
    counts = np.asarray(ends - starts, dtype=np.intp)
    counts = np.maximum(counts, 0)
    total = int(counts.sum())
    owner = np.repeat(np.arange(len(counts), dtype=np.intp), counts)
    offset = np.repeat(np.cumsum(counts) - counts, counts)
    item = np.repeat(np.asarray(starts, dtype=np.intp), counts)
    item += np.arange(total, dtype=np.intp) - offset
    return owner, item


def chunks(weights, limit: int = CHUNK) -> Iterator[slice]:
    """Runs of consecutive entries whose ``weights`` add up to about ``limit``, at least one."""
    weights = np.asarray(weights, dtype=np.int64)
    ends = np.cumsum(weights)
    begin = 0
    while begin < len(weights):
        done = ends[begin] - weights[begin]
        stop = max(int(np.searchsorted(ends, done + limit, side="right")), begin + 1)
        yield slice(begin, stop)
        begin = stop


def monomials(dx, dy, mono_parent, mono_dir, first=None) -> np.ndarray:
    """``(len(dx), n_coef)`` monomials ``first * dx**a * dy**b`` in the order of ``mono_parent``.

    Monomial ``i > 0`` is monomial ``mono_parent[i]`` times ``dx`` (``mono_dir[i] == 0``)
    or ``dy``; monomial 0 is ``first`` (default 1).
    """
    n_coef = len(mono_parent)
    out = np.empty((len(dx), n_coef))
    out[:, 0] = 1.0 if first is None else first
    for i in range(1, n_coef):
        out[:, i] = out[:, mono_parent[i]] * (dx if mono_dir[i] == 0 else dy)
    return out


def cartesian_local(local, mono, ex, ey, grad):
    """Potential and the two gradient components of Cartesian local expansions."""
    pot = np.sum(local * mono, axis=1)
    gx = np.zeros(len(mono))
    gy = np.zeros(len(mono))
    for i in range(local.shape[1]):
        if grad[i, 0] >= 0:
            gx += ex[i] * local[:, i] * mono[:, grad[i, 0]]
        if grad[i, 1] >= 0:
            gy += ey[i] * local[:, i] * mono[:, grad[i, 1]]
    return pot, gx, gy


def complex_powers(z: np.ndarray, order: int) -> np.ndarray:
    """``(len(z), order + 1)`` powers ``z**k``, k = 0 ... order, by repeated products."""
    out = np.empty((len(z), order + 1), dtype=complex)
    out[:, 0] = 1.0
    for k in range(1, order + 1):
        out[:, k] = out[:, k - 1] * z
    return out


def term_matrix(rows: np.ndarray, n_out: int) -> np.ndarray:
    """``(len(rows), n_out)`` 0/1 matrix adding term ``t`` into output ``rows[t]``."""
    out = np.zeros((len(rows), n_out))
    out[np.arange(len(rows)), rows] = 1.0
    return out


def leaf_particles(node_count, node_start, node_end, node_child):
    """The leaves of a tree and their particles: ``(leaf, slot)`` for every particle slot."""
    leaves = np.flatnonzero(node_child[:node_count] < 0)
    owner, slot = expand(node_start[leaves], node_end[leaves])
    return leaves[owner], slot


def near_pairs(node_start, node_end, pairs, count) -> Iterator[tuple]:
    """Particle pairs of the leaf pairs ``pairs[:count]``, in the order of the Pyccel loops.

    A leaf paired with itself gives each of its pairs once (``ka < kb``). Yields chunks of
    ``(ka, kb)`` slot arrays.
    """
    a = np.asarray(pairs[:count, 0], dtype=np.intp)
    b = np.asarray(pairs[:count, 1], dtype=np.intp)
    size_a = node_end[a] - node_start[a]
    size_b = node_end[b] - node_start[b]
    for part in chunks(size_a * size_b):
        pa, pb = a[part], b[part]
        # every particle of a with every particle of b
        pair, ka = expand(node_start[pa], node_end[pa])
        rows = pb[pair]
        owner, kb = expand(node_start[rows], node_end[rows])
        ka = ka[owner]
        keep = (pa[pair][owner] != rows[owner]) | (ka < kb)
        yield ka[keep], kb[keep]


def target_pairs(t_start, t_end, t_child, t_count, row_start, partners, s_start, s_end):
    """Target slots and source slots of the near field of the target leaves.

    Target leaf ``n`` meets the particles of the source leaves
    ``partners[row_start[n]:row_start[n + 1]]``. Yields chunks of ``(k, j)`` arrays, the
    target slot and the source slot of every pair.
    """
    leaves = np.flatnonzero(t_child[:t_count] < 0)
    edge_leaf, edge = expand(row_start[leaves], row_start[leaves + 1])
    edge_leaf = leaves[edge_leaf]
    src = np.asarray(partners[edge], dtype=np.intp)
    work = (t_end[edge_leaf] - t_start[edge_leaf]) * (s_end[src] - s_start[src])
    for part in chunks(work):
        tl, sl = edge_leaf[part], src[part]
        owner, k = expand(t_start[tl], t_end[tl])
        rows = sl[owner]
        owner2, j = expand(s_start[rows], s_end[rows])
        yield k[owner2], j
