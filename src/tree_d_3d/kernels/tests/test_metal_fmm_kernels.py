"""The Metal versions of the FMM and tree-build kernels, against the CPU versions.

Each test calls a ``<name>_metal.py`` function and the Pyccel (or NumPy) version with
the same arrays. The GPU computes in float32; skipped without MLX and a Metal GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tree_d_3d.kernels import KERNELS
from tree_d_3d.kernels._metal import metal_available

pytestmark = pytest.mark.skipif(
    not metal_available(), reason="MLX with a Metal GPU is not available"
)

if metal_available():
    from tree_d_3d.flat_tree import FlatQuadtree, _binomial_table
    from tree_d_3d.kernels.fmm_downward.fmm_downward_metal import fmm_downward
    from tree_d_3d.kernels.fmm_evaluate.fmm_evaluate_metal import fmm_evaluate
    from tree_d_3d.kernels.fmm_m2l.fmm_m2l_metal import fmm_m2l
    from tree_d_3d.kernels.fmm_traverse.fmm_traverse_metal import fmm_traverse
    from tree_d_3d.kernels.log_quadtree_moments.log_quadtree_moments_metal import (
        log_quadtree_moments,
    )
    from tree_d_3d.kernels.quadtree_moments.quadtree_moments_metal import (
        quadtree_moments,
    )
    from tree_d_3d.kernels.quadtree_quadrants.quadtree_quadrants_metal import (
        quadtree_quadrants,
    )
    from tree_d_3d.kernels.quadtree_scatter.quadtree_scatter_metal import (
        quadtree_scatter,
    )
    from tree_d_3d.kernels.soften_pairs.soften_pairs_metal import soften_pairs
    from tree_d_3d.quad import Quad

ORDER = 8
SOFTENING = 0.01


def _close(gpu, cpu, atol=3e-5):
    scale = max(float(np.abs(cpu).max()), 1.0)
    np.testing.assert_allclose(gpu / scale, cpu / scale, rtol=0.0, atol=atol)


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(8)
    n = 2000
    # component-major: pos[0] is x of every particle
    pos = np.clip(
        np.concatenate(
            (rng.normal(0, 0.05, (2, n // 2)), rng.uniform(-1, 1, (2, n // 2))),
            axis=1,
        ),
        -0.99,
        0.99,
    )
    charge = rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)
    return pos, charge


@pytest.fixture(scope="module")
def tree(system):
    pos, charge = system
    return FlatQuadtree(
        pos, charge, Quad(0.0, 0.0, 1.0), kernel="log", order=ORDER, leaf_capacity=4
    )


def _traverse_lists(function, tree, theta):
    """All M2L and P2P pairs of the dual-tree traversal with ``function``."""
    cap = 40 * tree.node_count + 64
    m2l = np.empty((cap, 2), dtype=np.int64)
    p2p = np.empty((cap, 2), dtype=np.int64)
    frontier = np.zeros((1, 2), dtype=np.int64)
    counts = np.zeros(4, dtype=np.int64)
    while len(frontier):
        counts[2] = 0
        nxt = np.empty((cap, 2), dtype=np.int64)
        function(
            tree.node_center,
            tree.node_half,
            tree.node_start,
            tree.node_end,
            tree.node_child,
            theta,
            frontier,
            len(frontier),
            nxt,
            m2l,
            p2p,
            counts,
        )
        assert counts[3] == 0
        frontier = nxt[: counts[2]].copy()
    return m2l[: counts[0]], p2p[: counts[1]]


def test_fmm_traverse_lists_equal_the_cpu_lists(tree):
    cpu = _traverse_lists(KERNELS["fmm_traverse"], tree, 0.5)
    gpu = _traverse_lists(fmm_traverse, tree, 0.5)
    for a, b in zip(gpu, cpu):
        np.testing.assert_array_equal(a, b)  # same order: appended in pair order


def test_fmm_traverse_reports_a_small_buffer(tree):
    frontier = np.array([[0, 0]], dtype=np.int64)
    for function in (KERNELS["fmm_traverse"], fmm_traverse):
        nxt = np.empty((4, 2), dtype=np.int64)  # the root splits into 10 pairs
        m2l, p2p = np.empty((4, 2), dtype=np.int64), np.empty((4, 2), dtype=np.int64)
        counts = np.zeros(4, dtype=np.int64)
        function(
            tree.node_center,
            tree.node_half,
            tree.node_start,
            tree.node_end,
            tree.node_child,
            0.5,
            frontier,
            1,
            nxt,
            m2l,
            p2p,
            counts,
        )
        assert counts[3] == -1 and counts[2] == 10


def test_fmm_chain_m2l_downward_evaluate(system, tree):
    pos, _ = system
    m2l_pairs, p2p_pairs = _traverse_lists(KERNELS["fmm_traverse"], tree, 0.5)
    n_nodes = tree.node_count
    width = ORDER + 1
    binomial = _binomial_table(2 * ORDER + 1)
    local = {}
    for name, function in (("cpu", KERNELS["fmm_m2l"]), ("gpu", fmm_m2l)):
        re, im = np.zeros((n_nodes, width)), np.zeros((n_nodes, width))
        scratch = np.zeros(2 * ORDER + 2)
        function(
            m2l_pairs,
            len(m2l_pairs),
            tree.node_center,
            tree.moment_re,
            tree.moment_im,
            binomial,
            0,
            scratch,
            scratch.copy(),
            re,
            im,
        )
        local[name] = (re, im)
    _close(local["gpu"][0], local["cpu"][0], atol=1e-3)
    _close(local["gpu"][1], local["cpu"][1], atol=1e-3)

    # downward from identical (CPU) locals, level by level
    shifted = {}
    for name, function in (("cpu", KERNELS["fmm_downward"]), ("gpu", fmm_downward)):
        re, im = local["cpu"][0].copy(), local["cpu"][1].copy()
        scratch = np.zeros(width)
        for first, last in zip(tree.levels[:-2], tree.levels[1:-1]):
            function(
                int(first),
                int(last),
                tree.node_center,
                tree.node_child,
                binomial,
                scratch,
                scratch.copy(),
                re,
                im,
            )
        shifted[name] = (re, im)
    _close(shifted["gpu"][0], shifted["cpu"][0], atol=1e-3)
    _close(shifted["gpu"][1], shifted["cpu"][1], atol=1e-3)

    # evaluation from identical locals
    results = []
    for function in (KERNELS["fmm_evaluate"], fmm_evaluate):
        phi, field = np.zeros(pos.shape[1]), np.zeros((2, pos.shape[1]))
        status = np.zeros(1, dtype=np.int64)
        function(
            tree.pos,
            tree.charge,
            tree.perm,
            n_nodes,
            tree.node_center,
            tree.node_start,
            tree.node_end,
            tree.node_child,
            shifted["cpu"][0],
            shifted["cpu"][1],
            p2p_pairs,
            len(p2p_pairs),
            SOFTENING,
            1,
            phi,
            field,
            status,
        )
        assert status[0] == 0
        results.append((phi, field))
    _close(results[1][0], results[0][0], atol=1e-3)
    _close(results[1][1], results[0][1], atol=1e-3)


def test_soften_pairs(system):
    pos, charge = system
    n = pos.shape[1]
    rng = np.random.default_rng(1)
    pairs = np.unique(np.sort(rng.integers(0, n, (2000, 2)), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]].astype(np.int64)
    results = []
    for function in (KERNELS["soften_pairs"], soften_pairs):
        phi, field = np.zeros(n), np.zeros((2, n))
        status = np.zeros(1, dtype=np.int64)
        function(pos, charge, pairs, SOFTENING**2, 1, phi, field, status)
        assert status[0] == 0
        results.append((phi, field))
    _close(results[1][0], results[0][0], atol=1e-4)
    _close(results[1][1], results[0][1], atol=1e-4)


def test_soften_pairs_reports_coincident_charges():
    pos = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # (2, N): two coincide
    pairs = np.array([[0, 2], [0, 1]], dtype=np.int64)
    phi, field = np.zeros(3), np.zeros((2, 3))
    status = np.zeros(1, dtype=np.int64)
    soften_pairs(pos, np.ones(3), pairs, 1e-4, 1, phi, field, status)
    assert status[0] == 2


def test_tree_build_kernels_and_moments(system, tree):
    pos, _ = system
    n = pos.shape[1]
    # one level of the build from the root
    perm = np.arange(n, dtype=np.int64)
    slot_node = np.zeros(n, dtype=np.int64)
    quadrants = []
    for function in (KERNELS["quadtree_quadrants"], quadtree_quadrants):
        quadrant = np.zeros(n, dtype=np.int64)
        function(pos, perm, slot_node, tree.node_center, tree.node_child, quadrant)
        quadrants.append(quadrant)
    np.testing.assert_array_equal(quadrants[1], quadrants[0])

    quadrant = quadrants[0]
    below = np.cumsum(np.eye(4, dtype=np.int64)[quadrant], axis=0)
    node_start = tree.node_start.copy()
    counts = below[-1]
    node_start[tree.node_child[0] : tree.node_child[0] + 4] = np.concatenate(
        ([0], np.cumsum(counts)[:-1])
    )
    scattered = []
    for function in (KERNELS["quadtree_scatter"], quadtree_scatter):
        new_perm = np.zeros(n, dtype=np.int64)
        new_slot = np.zeros(n, dtype=np.int64)
        function(
            perm,
            slot_node,
            quadrant,
            below,
            node_start,
            tree.node_child,
            new_perm,
            new_slot,
        )
        scattered.append((new_perm, new_slot))
    for a, b in zip(*scattered):
        np.testing.assert_array_equal(a, b)

    # moments of every node, with the final tree
    count = tree.node_count
    cpu = [np.zeros(count), np.zeros(count), np.zeros((count, 2)), np.zeros((count, 3))]
    gpu = [np.zeros_like(a) for a in cpu]
    for out, function in ((cpu, KERNELS["quadtree_moments"]), (gpu, quadtree_moments)):
        function(
            tree.pos,
            tree.charge,
            tree.perm,
            count,
            tree.node_center,
            tree.node_start,
            tree.node_end,
            *out,
        )
    for a, b in zip(gpu, cpu):
        _close(a, b)

    cpu = [np.zeros(count), np.zeros((count, ORDER + 1)), np.zeros((count, ORDER + 1))]
    gpu = [np.zeros_like(a) for a in cpu]
    for out, function in (
        (cpu, KERNELS["log_quadtree_moments"]),
        (gpu, log_quadtree_moments),
    ):
        function(
            tree.pos,
            tree.charge,
            tree.perm,
            count,
            tree.node_center,
            tree.node_start,
            tree.node_end,
            *out,
        )
    for a, b in zip(gpu, cpu):
        _close(a, b)
