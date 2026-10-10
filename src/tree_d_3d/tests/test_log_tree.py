"""The line-charge (log) kernel of the tree walk and the direct sum."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.kernels import kernels_compiled
from tree_d_3d.quad import Quad


def sources(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.2, (2, n)).clip(-0.99, 0.99), rng.normal(0.3, 1, n)


@pytest.mark.parametrize("order,tolerance", [(4, 1e-3), (8, 1e-5), (12, 2e-7)])
def test_log_tree_converges_with_expansion_order(order, tolerance):
    pos, q = sources()
    probes = np.random.default_rng(1).uniform(-1, 1, (2, 200))
    phi, field = FlatQuadtree(
        pos, q, Quad(0, 0, 1), kernel="log", order=order
    ).evaluate(probes, theta=0.6)
    ref_phi, ref_field = direct_field(probes, pos, q, kernel="log")
    assert np.max(np.abs(phi - ref_phi)) < tolerance * np.max(np.abs(ref_phi))
    assert np.max(np.abs(field - ref_field)) < tolerance * np.max(np.abs(ref_field))


def test_log_kernel_matches_a_line_charge_and_its_reference_radius():
    tree = FlatQuadtree([[0.0], [0.0]], [2.0], Quad(0, 0, 1), kernel="log")
    phi, field = tree.field_at([[3.0], [4.0]], k_e=1.0, reference_radius=2.0)
    assert phi[0] == pytest.approx(-2 * 2.0 * np.log(5.0 / 2.0))
    np.testing.assert_allclose(field[:, 0], 2 * 2.0 * np.array([3.0, 4.0]) / 25)
    with pytest.raises(ValueError, match="kernel"):
        FlatQuadtree([[0.0], [0.0]], [1.0], Quad(0, 0, 1), kernel="yukawa")


def test_compiled_log_kernels_match_python_source():
    if not kernels_compiled():
        pytest.skip("Pyccel kernels unavailable")
    pos, q = sources(500)
    probes = np.random.default_rng(2).uniform(-1, 1, (2, 50))
    compiled = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log")
    python = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", compiled=False)
    for a, b in zip(compiled.evaluate(probes), python.evaluate(probes)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    expected = direct_field(probes, pos, q, kernel="log")
    with xp.kernels.use_host_kernel_implementation("numpy"):
        vectorized = direct_field(probes, pos, q, kernel="log")
    for a, b in zip(expected, vectorized):
        np.testing.assert_allclose(a, b, rtol=1e-12)
