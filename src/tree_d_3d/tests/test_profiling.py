"""The opt-in scope-profiler regions: off by default, recorded once the tag is enabled."""

import numpy as np
import pytest
import scope_profiler as sp

from tree_d_3d import PROFILE_TAG
from tree_d_3d.flat_tree import FlatQuadtree
from tree_d_3d.quad import Quad

TREE_REGIONS = {
    "tree_d_3d.flat_tree.build",
    "tree_d_3d.flat_tree.moments",
    "tree_d_3d.flat_tree.evaluate",
}


@pytest.fixture
def profiler():
    """A clean profiler state before and after the test."""
    sp.reset()
    yield
    sp.reset()


def build_and_evaluate():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-0.9, 0.9, (2, 200))
    charge = rng.choice([-1.0, 1.0], 200)
    tree = FlatQuadtree(pos, charge, Quad(0, 0, 1))
    tree.evaluate(pos, exclude=np.arange(200), softening=0.01)


def recorded_regions(**options):
    sp.setup(output=None, **options)
    build_and_evaluate()
    return set(sp.finalize(verbose=False, return_results=True).region_names)


def test_regions_off_by_default(profiler):
    assert not {name for name in recorded_regions() if name.startswith("tree_d_3d.")}


@pytest.mark.parametrize("tag", [PROFILE_TAG, "*"])
def test_regions_recorded_when_enabled(profiler, tag):
    assert recorded_regions(enable_tags=[tag]) >= TREE_REGIONS
