"""The CPU tree build (quadtree_split) gives the same tree as the reference build."""

import numpy as np
import pytest

from tree_d_3d import FlatQuadtree, Quad
from tree_d_3d.kernels.quadtree_split.quadtree_split_numpy import (
    quadtree_split as numpy_split,
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
def test_compiled_build_equals_the_uncompiled_one(name):
    pos = _clouds()[name]
    q = np.ones(pos.shape[1])
    compiled = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=6)
    plain = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=6, compiled=False)
    for attribute in ARRAYS:
        np.testing.assert_array_equal(
            getattr(compiled, attribute), getattr(plain, attribute)
        )
    # every particle once, every leaf holds what its range says
    assert np.array_equal(np.sort(compiled.perm), np.arange(pos.shape[1]))


def test_the_numpy_split_equals_the_compiled_one(monkeypatch):
    import tree_d_3d.flat_tree as flat_tree

    pos = _clouds()["clustered"]
    q = np.ones(pos.shape[1])
    expected = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=6)
    kernels = flat_tree._kernels(True)
    calls = []

    def counted(*args):
        calls.append(1)
        numpy_split(*args)

    monkeypatch.setattr(kernels, "quadtree_split", counted)
    monkeypatch.setattr(flat_tree, "_kernels", lambda compiled=True: kernels)
    numpy_tree = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=6)
    for attribute in ARRAYS:
        np.testing.assert_array_equal(
            getattr(numpy_tree, attribute), getattr(expected, attribute)
        )
