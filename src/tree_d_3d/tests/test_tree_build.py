"""The CPU tree build (quadtree_build) gives the same tree as the GPU's level kernels."""

import numpy as np
import pytest

from tree_d_3d import FlatQuadtree, Quad
from tree_d_3d.kernels.quadtree_build.quadtree_build_numpy import (
    quadtree_build as numpy_build,
)

ARRAYS = (
    "perm",
    "node_start",
    "node_end",
    "node_child",
    "node_depth",
    "node_center",
    "levels",
)


def _clouds():
    rng = np.random.default_rng(5)
    angle = rng.uniform(0, 2 * np.pi, 3000)
    return {
        "uniform": rng.uniform(-0.99, 0.99, (2, 4000)),
        "circle": 0.9 * np.stack((np.cos(angle), np.sin(angle))),
        "clustered": np.clip(rng.normal(0.3, 0.01, (2, 2000)), -1, 1),
        "lattice": np.stack(
            np.meshgrid(np.linspace(-0.5, 0.5, 33), np.linspace(-0.5, 0.5, 33))
        ).reshape(2, -1),
        "coincident": np.full((2, 40), 0.25),
    }


@pytest.mark.parametrize("name", list(_clouds()))
@pytest.mark.parametrize("compiled", [True, False])
def test_one_call_build_equals_the_level_kernels(name, compiled, monkeypatch):
    import tree_d_3d.flat_tree as flat_tree

    pos = _clouds()[name]
    q = np.ones(pos.shape[1])
    built = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=6, compiled=compiled)
    monkeypatch.setattr(flat_tree, "_HOST_BUILD", False)
    levelled = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=6)
    for attribute in ARRAYS:
        np.testing.assert_array_equal(
            getattr(built, attribute), getattr(levelled, attribute)
        )
    # every particle once, in tree order in the sorted positions
    assert np.array_equal(np.sort(built.perm), np.arange(pos.shape[1]))
    spos, _ = built._sorted_sources()
    np.testing.assert_array_equal(spos[:, : pos.shape[1]], pos[:, built.perm])


def test_a_too_small_node_buffer_restarts_the_build(monkeypatch):
    import tree_d_3d.flat_tree as flat_tree

    pos = _clouds()["clustered"]
    q = np.ones(pos.shape[1])
    expected = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=6)
    kernels = flat_tree._kernels(True)
    calls = []

    def counted(*args):
        calls.append(1)
        numpy_build(*args)

    original = FlatQuadtree._build_host
    monkeypatch.setattr(kernels, "quadtree_build", counted)
    monkeypatch.setattr(flat_tree, "_kernels", lambda compiled=True: kernels)
    monkeypatch.setattr(
        FlatQuadtree,
        "_build_host",
        lambda self, root, capacity: original(self, root, 8),
    )
    small = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=6)
    assert len(calls) > 1
    for attribute in ARRAYS:
        np.testing.assert_array_equal(
            getattr(small, attribute), getattr(expected, attribute)
        )


@pytest.mark.parametrize("kernel", ["log", "coulomb"])
def test_set_charges_equals_a_new_tree(kernel):
    rng = np.random.default_rng(8)
    pos = rng.uniform(-1, 1, (2, 3000))
    first, second = rng.normal(size=3000), rng.normal(size=3000)
    tree = FlatQuadtree(pos, first, Quad(0, 0, 1), kernel=kernel, leaf_capacity=16)
    if kernel == "log":
        tree.fmm()  # caches the interaction lists, which set_charges keeps
    tree.set_charges(second)
    fresh = FlatQuadtree(pos, second, Quad(0, 0, 1), kernel=kernel, leaf_capacity=16)
    if kernel == "log":
        a, b = tree.fmm()[1], fresh.fmm()[1]
    else:
        own = np.arange(3000)
        a, b = tree.evaluate(pos, exclude=own)[1], fresh.evaluate(pos, exclude=own)[1]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="one entry per particle"):
        tree.set_charges(second[:10])


def test_set_charges_on_an_empty_tree():
    tree = FlatQuadtree(np.zeros((2, 0)), np.zeros(0), Quad(0, 0, 1))
    tree.set_charges(np.zeros(0))
    assert tree.node_count == 1
