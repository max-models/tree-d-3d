"""The smooth-membership FMM prototype: exact gradients, exact antisymmetry, accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from tree_d_3d.quad import Quad
from tree_d_3d.smooth import (
    BoxTree,
    SmoothFMM,
    comoving_fields,
    direct_energy,
    direct_fields,
)


def cloud(n, seed=0, box=0.9):
    rng = np.random.default_rng(seed)
    return rng.uniform(-box, box, (2, n)), rng.choice([-1.0, 1.0], n)


def net_force(q, field):
    return np.abs((q * field).sum(axis=1)).max() / np.abs(q * field).sum()


def finite_difference_force(tree, pos, q, kw, indices, h=1e-6):
    out = {}
    for i in indices:
        for axis in range(2):
            pp, pm = pos.copy(), pos.copy()
            pp[axis, i] += h
            pm[axis, i] -= h
            up = SmoothFMM(tree, pp, q, **kw).energy
            um = SmoothFMM(tree, pm, q, **kw).energy
            out[i, axis] = -(up - um) / (2 * h)
    return out


def test_partition_of_unity_and_memberships():
    pos, q = cloud(400)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    for frac in (0.1, 0.5):
        f = SmoothFMM(tree, pos, q, band=frac * tree.min_leaf_half, order=4)
        total = np.bincount(f.part, f.w, minlength=400)
        np.testing.assert_allclose(total, 1.0, atol=1e-14)
        assert 1.0 < len(f.part) / 400 < 3.0
    with pytest.raises(ValueError, match="band"):
        SmoothFMM(tree, pos, q, band=1.01 * tree.min_leaf_half, order=4)


def test_sharp_membership_is_the_ordinary_tree():
    pos, q = cloud(600)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=16)
    f = SmoothFMM(tree, pos, q, order=12, theta=0.5, band=0.0)
    assert len(f.part) == 600 and np.all(f.w == 1.0)
    phi, field = f.fields()
    ref_phi, ref_field = direct_fields(pos, q)
    assert np.abs(field - ref_field).max() < 1e-6 * np.abs(ref_field).max()
    assert abs(f.energy - direct_energy(pos, q)) < 1e-7 * abs(direct_energy(pos, q))


@pytest.mark.parametrize("symmetric", [True, False])
def test_forces_are_the_exact_gradient_of_the_energy(symmetric):
    """With smooth membership, F = -dU/dx to finite-difference accuracy, for
    particles blended into several leaves as well as for the others."""
    pos, q = cloud(300, seed=2)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    kw = dict(
        order=8,
        theta=0.6,
        band=0.4 * tree.min_leaf_half,
        softening=0.02,
        symmetric=symmetric,
    )
    f = SmoothFMM(tree, pos, q, **kw)
    _, field = f.fields()
    force = q * field
    counts = np.bincount(f.part, minlength=300)
    rng = np.random.default_rng(5)
    blended = rng.choice(np.flatnonzero(counts > 1), 4, replace=False)
    single = rng.choice(np.flatnonzero(counts == 1), 2, replace=False)
    fd = finite_difference_force(tree, pos, q, kw, [*blended, *single])
    scale = np.abs(force).max()
    for (i, axis), value in fd.items():
        assert abs(value - force[axis, i]) < 1e-7 * scale


def test_symmetric_truncation_gives_exact_antisymmetry():
    pos, q = cloud(1500, seed=1)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=16)
    ref = direct_fields(pos, q)[1]
    for order in (8, 16):
        rect = SmoothFMM(tree, pos, q, order=order, theta=0.5, symmetric=False)
        tri = SmoothFMM(tree, pos, q, order=order, theta=0.5, symmetric=True)
        _, e_rect = rect.fields()
        _, e_tri = tri.fields()
        assert net_force(q, e_tri) < 1e-15
        # the rectangle's antisymmetry error is at the truncation order
        assert net_force(q, e_rect) > 1e-13
        # both converge; the triangle needs a higher order for the same error
        err_rect = np.abs(e_rect - ref).max() / np.abs(ref).max()
        err_tri = np.abs(e_tri - ref).max() / np.abs(ref).max()
        assert err_rect < err_tri < 100 * err_rect


def test_smooth_membership_keeps_the_accuracy():
    pos, q = cloud(800, seed=4)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    ref = direct_fields(pos, q)[1]
    errors = []
    for frac in (0.0, 0.25, 0.5):
        f = SmoothFMM(tree, pos, q, order=12, theta=0.5, band=frac * tree.min_leaf_half)
        _, e = f.fields()
        errors.append(np.abs(e - ref).max() / np.abs(ref).max())
    assert errors[0] < 1e-6
    # blended particles are evaluated slightly outside their leaves: a few times worse
    assert errors[2] < 50 * errors[0]


def _verlet(pos, vel, q, m, dt, steps, k_trap, fields):
    """Velocity Verlet in a harmonic trap; returns the relative energy error history
    and the largest relative net force seen."""
    x, v = pos.copy(), vel.copy()
    u, e = fields(x)
    e0 = u + 0.5 * k_trap * np.sum(x**2) + 0.5 * np.sum(m * np.sum(v**2, axis=0))
    acc = (q * e - k_trap * x) / m
    drift, net = [], 0.0
    for _ in range(steps):
        v += 0.5 * dt * acc
        x += dt * v
        u, e = fields(x)
        acc = (q * e - k_trap * x) / m
        v += 0.5 * dt * acc
        drift.append(
            (u + 0.5 * k_trap * np.sum(x**2) + 0.5 * np.sum(m * np.sum(v**2, axis=0)))
            / e0
            - 1.0
        )
        net = max(net, np.abs((q * e).sum(axis=1)).max() / np.abs(q * e).sum())
    return np.array(drift), net


def test_smooth_membership_in_comoving_boxes_conserves_energy_and_momentum():
    """Fixed boxes + smooth membership + co-moving frame: the energy error of the
    tree code equals the integrator's own (the direct sum's) and the net force is
    roundoff; hard membership on the same boxes drifts."""
    n, steps, dt, k_trap, eps = 150, 300, 0.002, 100.0, 0.1
    rng = np.random.default_rng(3)
    pos = rng.uniform(-0.8, 0.8, (2, n))
    q = rng.choice([-1.0, 1.0], n)
    vel = 0.2 * rng.normal(size=(2, n))
    m = np.ones(n)
    tree = BoxTree(pos, Quad(0, 0, 2.0), leaf_capacity=8)
    center = (m * pos).sum(axis=1) / m.sum()
    # a crude expansion, so that the membership effect dominates the integrator's error
    options = dict(order=2, theta=0.8, softening=eps, symmetric=True)

    def direct(x):
        return direct_energy(x, q, eps), direct_fields(x, q, eps)[1]

    def tree_fields(band_frac):
        def fields(x):
            u, _, e = comoving_fields(
                tree, x, q, m, center, band=band_frac * tree.min_leaf_half, **options
            )
            return u, e

        return fields

    reference, _ = _verlet(pos, vel, q, m, dt, steps, k_trap, direct)
    sharp, net_sharp = _verlet(pos, vel, q, m, dt, steps, k_trap, tree_fields(0.0))
    smooth, net_smooth = _verlet(pos, vel, q, m, dt, steps, k_trap, tree_fields(0.5))
    assert net_sharp < 1e-14 and net_smooth < 1e-14
    assert np.abs(smooth).max() < 1.5 * np.abs(reference).max()
    assert np.abs(sharp).max() > 5.0 * np.abs(smooth).max()
