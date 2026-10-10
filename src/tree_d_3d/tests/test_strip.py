"""Line charges between grounded plates: conformal map plus the FMM."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from tree_d_3d.strip import GroundedStripField


def sheet_potential(sheets, sigma, points, length, epsilon):
    """Potential of charged sheets between grounded walls at 0 and ``length`` (1D)."""
    field_0 = -np.sum(sigma * (length - sheets)) / (epsilon * length)
    below = sheets[None, :] < points[:, None]
    return (
        -field_0 * points
        - np.sum(below * sigma * (points[:, None] - sheets), axis=1) / epsilon
    )


def cloud(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.stack((rng.uniform(0.05, 0.95, n), rng.uniform(-2, 2, n)), axis=0)
    return pos, rng.normal(0.2, 1.0, n)


def test_fmm_matches_direct_summation_in_the_mapped_plane():
    pos, q = cloud()
    phi_f, field_f = GroundedStripField(1.0, k_e=1.0).field_at_charges(pos, q)
    phi_d, field_d = GroundedStripField(1.0, k_e=1.0, method="direct").field_at_charges(
        pos, q
    )
    assert np.abs(phi_f - phi_d).max() < 1e-6 * np.abs(phi_d).max()
    assert np.abs(field_f - field_d).max() < 1e-5 * np.abs(field_d).max()


def test_plates_are_grounded_and_a_single_charge_matches_the_closed_form():
    strip = GroundedStripField(2.0, k_e=1.0, method="direct")
    source = np.array([[0.7], [0.3]])
    walls = strip.potential(source, [1.0], [[1e-12, 2.0 - 1e-12], [0.5, -1.0]])
    np.testing.assert_allclose(walls, 0.0, atol=1e-10)
    # Strip Green's function: -2 ln|sin((z - z0)/2a) / sin((z - conj z0)/2a)|
    # with a = L / pi, z = x + i y and conj z0 the mirror image in x = 0.
    probe = np.array([1.4, -0.6])
    z = complex(*probe)
    z0 = complex(0.7, 0.3)
    a = 2.0 / np.pi
    expected = -2 * np.log(
        abs(np.sin((z - z0) / (2 * a)) / np.sin((z + np.conj(z0)) / (2 * a)))
    )
    assert strip.potential(source, [1.0], probe[:, None])[0] == pytest.approx(expected)


def test_a_lone_charge_feels_only_its_image_force():
    strip = GroundedStripField(1.0, k_e=1.0, method="direct")
    _, middle = strip.field_at_charges([[0.5], [0.0]], [1.0])
    np.testing.assert_allclose(middle[:, 0], 0.0, atol=1e-10)
    _, near_left = strip.field_at_charges([[0.2], [0.0]], [1.0])
    assert near_left[0, 0] < 0 and abs(near_left[1, 0]) < 1e-10


def test_rows_of_line_charges_reproduce_sheets():
    rows = np.linspace(-20, 20, 800, endpoint=False)
    sheets, sigma = np.array([0.3, 0.6]), np.array([1.0, -0.4])
    lines = np.array([[x, y] for x in sheets for y in rows]).T
    per_line = np.repeat(sigma * 40 / len(rows), len(rows))
    xs = np.linspace(0.02, 0.98, 9)
    probes = np.stack((xs, np.zeros_like(xs)), axis=0)
    two_d = GroundedStripField(1.0, k_e=1.0).potential(lines, per_line, probes)
    one_d = sheet_potential(sheets, sigma, xs, 1.0, 1 / (4 * np.pi))
    assert np.abs(two_d - one_d).max() < 5e-3 * np.abs(one_d).max()


def test_validation():
    strip = GroundedStripField(1.0)
    with pytest.raises(ValueError, match="between the plates"):
        strip.field_at_charges([[1.0], [0.0]], [1.0])
    with pytest.raises(ValueError, match="method"):
        GroundedStripField(1.0, method="tree")
    phi, field = strip.field_at_charges(np.zeros((2, 0)), [])
    assert phi.shape == (0,) and field.shape == (2, 0)


@pytest.mark.parametrize("method", ["direct", "fmm"])
def test_plates_exert_no_net_force_along_y(method):
    # Translation symmetry in y: the total y-force on any configuration vanishes.
    pos, q = cloud(800, seed=4)
    _, field = GroundedStripField(1.0, k_e=1.0, method=method).field_at_charges(pos, q)
    force = q * field
    assert abs(force[1].sum()) < 1e-6 * np.abs(force[1]).sum()


def test_softening_is_a_local_correction():
    far = np.array([[0.3, 0.6], [0.0, 0.4]])
    q = np.array([1.0, -2.0])
    bare = GroundedStripField(1.0, k_e=1.0, method="direct")
    soft = GroundedStripField(1.0, k_e=1.0, method="direct", softening=0.01)
    # Beyond the cutoff (6 softening lengths) nothing changes.
    for a, b in zip(bare.field_at_charges(far, q), soft.field_at_charges(far, q)):
        np.testing.assert_array_equal(a, b)
    # A close pair: bare strip field with the free-space pair term softened.
    close = np.array([[0.3, 0.305], [0.0, 0.002]])
    _, e_bare = bare.field_at_charges(close, q)
    _, e_soft = soft.field_at_charges(close, q)
    d = close[:, 0] - close[:, 1]
    r2 = d @ d
    change = 2 * (1 / (r2 + 1e-4) - 1 / r2) * d
    np.testing.assert_allclose(e_soft[:, 0], e_bare[:, 0] + q[1] * change)
    np.testing.assert_allclose(e_soft[:, 1], e_bare[:, 1] - q[0] * change)
    # At half a softening length the pair force is already several times weaker.
    assert np.abs(q * e_soft).max() < 0.3 * np.abs(q * e_bare).max()


def test_softened_potential_and_no_net_y_force():
    pos, q = cloud(600, seed=5)
    soft = GroundedStripField(1.0, k_e=1.0, softening=0.02)
    _, field = soft.field_at_charges(pos, q)
    force = q * field
    assert abs(force[1].sum()) < 1e-6 * np.abs(force[1]).sum()
    probe = np.array([[0.5, 0.2], [0.0, 1.0]])
    far = GroundedStripField(1.0, k_e=1.0).potential(pos, q, probe)
    near = soft.potential(pos, q, probe)
    assert np.abs(near - far).max() < 0.05 * np.abs(far).max()
    with pytest.raises(ValueError, match="softening"):
        GroundedStripField(1.0, softening=-1)


def test_compiled_softening_and_force_only_mode_agree():
    pos, q = cloud(700, seed=6)
    strip = GroundedStripField(1.0, k_e=1.0, softening=0.03)
    phi, field = strip.field_at_charges(pos, q)
    _, field_only = strip.field_at_charges(pos, q, potential=False)
    np.testing.assert_allclose(field_only, field, rtol=1e-12, atol=1e-12)
    with xp.kernels.use_host_kernel_implementation("numpy"):
        phi_np, field_np = strip.field_at_charges(pos, q)
    np.testing.assert_allclose(phi_np, phi, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(field_np, field, rtol=1e-10, atol=1e-10)
