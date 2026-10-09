"""The vectorized NumPy versions of the Barnes-Hut kernels against their Python versions."""

from __future__ import annotations

import numpy as np
import pytest

from tree_d_3d.kernels import KERNELS


def _versions(name):
    implementations = KERNELS[name].host_kernel.kernel
    return implementations.get("numpy"), implementations.python


@pytest.fixture
def system():
    rng = np.random.default_rng(11)
    n = 300  # more than one block of the all-pairs arrays
    pos = rng.uniform(-1.0, 1.0, (2, n))  # component-major: pos[0] is x
    charge = rng.normal(size=n)
    charge[::13] = 0.0
    return pos, charge


@pytest.mark.parametrize("name", ["direct_sum", "direct_log_sum"])
def test_direct_sums(name, system):
    pos, charge = system
    n = pos.shape[1]
    exclude = np.arange(n, dtype=np.int64)
    exclude[::7] = -1
    results = []
    for kernel in _versions(name):
        phi, field, status = np.zeros(n), np.zeros((2, n)), np.zeros(1, dtype=np.int64)
        kernel(pos, exclude, pos, charge, 0.05, phi, field, status)
        results.append((phi, field, status))
    (phi, field, status), (phi_ref, field_ref, status_ref) = results
    assert status[0] == status_ref[0] == 0
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-11)
    np.testing.assert_allclose(field, field_ref, rtol=1e-11, atol=1e-12)


def test_pair_energy_and_coincidence(system):
    pos, charge = system
    energies = []
    for kernel in _versions("pair_energy"):
        status, energy = np.zeros(1, dtype=np.int64), np.zeros(1)
        kernel(pos, charge, 0.05, status, energy)
        energies.append(energy[0])
    assert energies[0] == pytest.approx(energies[1], rel=1e-12)
    pos = pos.copy()
    pos[:, 3] = pos[:, 40]
    status = np.zeros(1, dtype=np.int64)
    _versions("pair_energy")[0](pos, np.ones(pos.shape[1]), 0.0, status, np.zeros(1))
    assert status[0] == 1


def test_soften_pairs(system):
    pos, charge = system
    n = pos.shape[1]
    pairs = np.array([[0, 5], [5, 9], [2, 0], [7, 8]], dtype=np.int64)
    results = []
    for kernel in _versions("soften_pairs"):
        phi, field, status = np.zeros(n), np.zeros((2, n)), np.zeros(1, np.int64)
        kernel(pos, charge, pairs, 0.01, 1, phi, field, status)
        results.append((phi, field))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-12, atol=1e-15)


def test_moments(system):
    pos, charge = system
    n = pos.shape[1]
    perm = np.random.default_rng(1).permutation(n).astype(np.int64)
    node_start = np.array([0, 0, 100, 250], dtype=np.int64)
    node_end = np.array([n, 100, 250, n], dtype=np.int64)
    center = np.array([[0.0, 0.0], [0.3, -0.2], [-0.5, 0.5], [0.1, 0.9]])
    count = len(node_start)
    results = []
    for kernel in _versions("quadtree_moments"):
        out = [
            np.zeros(count),
            np.zeros(count),
            np.zeros((count, 2)),
            np.zeros((count, 3)),
        ]
        kernel(pos, charge, perm, count, center, node_start, node_end, *out)
        results.append(out)
    for a, b in zip(*results):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    results = []
    for kernel in _versions("log_quadtree_moments"):
        out = [np.zeros(count), np.zeros((count, 9)), np.zeros((count, 9))]
        kernel(pos, charge, perm, count, center, node_start, node_end, *out)
        results.append(out)
    for a, b in zip(*results):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)


def _flat(system, implementation, **options):
    """A log tree whose kernels are those of one host implementation."""
    import cunumpy as xp

    from tree_d_3d.flat_tree import FlatQuadtree
    from tree_d_3d.quad import Quad

    pos, charge = system
    with (
        xp.use_backend("numpy"),
        xp.kernels.use_host_kernel_implementation(implementation),
    ):
        tree = FlatQuadtree(pos, charge, Quad(0, 0, 1), kernel="log", **options)
        return tree, tree._traverse(0.5), tree.fmm(theta=0.5)


@pytest.mark.parametrize("leaf_capacity", [1, 5])
def test_build_traversal_and_downward_pass(system, leaf_capacity):
    """quadtree_quadrants/scatter, fmm_traverse and fmm_downward, through the FMM."""
    tree, lists, fields = _flat(system, "numpy", leaf_capacity=leaf_capacity, order=8)
    ref, ref_lists, ref_fields = _flat(
        system, "pyccel", leaf_capacity=leaf_capacity, order=8
    )
    for name in ("perm", "node_center", "node_half", "node_start", "node_end"):
        np.testing.assert_array_equal(getattr(tree, name), getattr(ref, name))
    np.testing.assert_array_equal(tree.node_child, ref.node_child)
    np.testing.assert_array_equal(tree.levels, ref.levels)
    (m2l, p2p, sizes), (m2l_ref, p2p_ref, sizes_ref) = lists, ref_lists
    np.testing.assert_array_equal(sizes, sizes_ref)
    np.testing.assert_array_equal(m2l[: sizes[0]], m2l_ref[: sizes[0]])
    np.testing.assert_array_equal(p2p[: sizes[1]], p2p_ref[: sizes[1]])
    for a, b in zip(fields, ref_fields):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_traversal_reports_overflow_and_needed_sizes(system):
    tree, _, _ = _flat(system, "pyccel")
    for implementation in _versions("fmm_traverse"):
        tiny = [np.zeros((3, 2), dtype=np.int64) for _ in range(4)]
        frontier, next_frontier, m2l, p2p = tiny
        counts = np.zeros(4, dtype=np.int64)
        # the root splits into 10 pairs, more than the next frontier holds
        implementation(
            tree.node_center,
            tree.node_half,
            tree.node_start,
            tree.node_end,
            tree.node_child,
            0.5,
            frontier,
            1,
            next_frontier,
            m2l,
            p2p,
            counts,
        )
        np.testing.assert_array_equal(counts, [0, 0, 10, -1])
