"""Fast multipole method for line charges and the line-charge N-body mode."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erfi

from tree_d_3d.bodies import ChargedBody
from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.kernels import kernels_compiled
from tree_d_3d.quad import Quad
from tree_d_3d.simulation import ElectrostaticSimulation as Simulation


def cloud(n=3000, seed=0):
    """Positions ``(2, n)`` (component-major) and charges ``(n,)``."""
    rng = np.random.default_rng(seed)
    pos = np.concatenate(
        (
            rng.normal([[-0.3], [0.1]], 0.1, (2, n // 2)),
            rng.normal([[0.4], [-0.2]], 0.25, (2, n - n // 2)),
        ),
        axis=1,
    ).clip(-0.99, 0.99)
    return pos, rng.normal(0.3, 1, n)


def reference(pos, q):
    return direct_field(pos, pos, q, exclude=np.arange(pos.shape[1]), kernel="log")


def relative(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.mark.parametrize("order,tolerance", [(6, 1e-3), (10, 1e-5), (18, 1e-8)])
def test_fmm_converges_with_expansion_order(order, tolerance):
    pos, q = cloud()
    ref_phi, ref_field = reference(pos, q)
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=order)
    phi, field = tree.fmm(theta=0.5)
    assert relative(phi, ref_phi) < tolerance
    assert relative(field, ref_field) < 10 * tolerance
    m2l, p2p = tree.fmm_pairs
    assert m2l > 0 and p2p > 0


def test_fmm_agrees_for_any_leaf_size_and_with_python_source():
    pos, q = cloud(800, seed=1)
    _, ref_field = reference(pos, q)
    for leaf in (1, 8, 64):
        tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", leaf_capacity=leaf)
        _, field = tree.fmm()
        assert relative(field, ref_field) < 5e-7
    if kernels_compiled():
        compiled = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=8).fmm()
        python = FlatQuadtree(
            pos, q, Quad(0, 0, 1), kernel="log", order=8, compiled=False
        ).fmm()
        for a, b in zip(compiled, python):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)


def test_fmm_softening_neutral_markers_and_validation():
    pos, q = cloud(400, seed=2)
    q[::7] = 0.0
    pos[:, 2] = pos[:, 1]  # a coincident charged pair needs softening
    _, field = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log").fmm(softening=0.01)
    _, ref_field = direct_field(
        pos, pos, q, exclude=np.arange(pos.shape[1]), kernel="log", softening=0.01
    )
    # Softening acts on the exact near field only; far cells are unsoftened,
    # which is an O(eps**2 / d**2) approximation.
    assert relative(field, ref_field) < 1e-2
    with pytest.raises(ValueError, match="Coincident"):
        FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log").fmm()
    with pytest.raises(ValueError, match="Coincident"):
        FlatQuadtree(pos, q, Quad(0, 0, 1)).fmm()
    with pytest.raises(ValueError, match="theta"):
        FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log").fmm(theta=1.0)


def test_fmm_net_force_is_at_the_truncation_level():
    pos, q = cloud(6000, seed=3)
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=18)
    _, field = tree.fmm(theta=0.5)
    force = q * field  # (2, n)
    assert np.abs(force.sum(axis=1)).max() < 1e-11 * np.abs(force).sum()


def test_line_charge_pair_energy_and_reference_radius():
    bodies = [ChargedBody(0.0, 0.0, charge=2.0), ChargedBody(3.0, 4.0, charge=-1.0)]
    sim = Simulation(
        bodies, Quad(0, 0, 8), k_e=1.0, kernel="log", reference_radius=2.0, dt=0.1
    )
    expected = -2 * 2.0 * -1.0 * np.log(5.0 / 2.0)
    assert sim.get_potential_energy() == pytest.approx(expected)
    assert sim.get_potential_energy(exact=True) == pytest.approx(expected)
    sim.step()
    # Attraction: the line charges move toward each other.
    assert np.linalg.norm(bodies[1].pos - bodies[0].pos) < 5.0


def test_method_selection_and_validation():
    pos, q = cloud(1500, seed=4)
    bodies = [ChargedBody(*p, charge=c) for p, c in zip(pos.T, q)]
    sim = Simulation(bodies, Quad(0, 0, 1), k_e=1, kernel="log", softening=0.01)
    assert sim._resolve_method() == "fmm"
    with pytest.raises(ValueError, match="theta"):
        Simulation([], Quad(0, 0, 1), method="fmm", kernel="log", theta=1.2)


@pytest.mark.parametrize("method", ["tree", "fmm"])
def test_fast_methods_match_direct_trajectories(method):
    pos, q = cloud(600, seed=5)
    runs = {}
    for name in ("direct", method):
        bodies = [ChargedBody(*p, charge=c, mass=1.0) for p, c in zip(pos.T, q)]
        sim = Simulation(
            bodies,
            Quad(0, 0, 1),
            k_e=1e-3,
            dt=0.02,
            # Fast methods soften only the near field; keep that difference
            # far below the tolerance.
            softening=1e-4,
            kernel="log",
            method=name,
        )
        for _ in range(10):
            sim.step()
        runs[name] = sim.positions.copy()
    assert relative(runs[method], runs["direct"]) < 1e-6


def cylinder_scale(t, rate_constant):
    """s(t) for s'' = C / s, s(0) = 1, s'(0) = 0: t = sqrt(pi/2C) erfi(sqrt ln s)."""

    def time_of(s):
        return np.sqrt(np.pi / (2 * rate_constant)) * erfi(np.sqrt(np.log(s)))

    return brentq(lambda s: time_of(s) - t, 1.0, 1e6)


def test_uniform_cylinder_explodes_self_similarly():
    # A uniform disk of line charges has a linear field, E = 2 k Q r / R**2,
    # so it expands self-similarly with s'' = C / s, C = 2 k q Q / (m R**2).
    n, radius, total = 4000, 1.0, 1.0
    i = np.arange(n)
    r = radius * np.sqrt((i + 0.5) / n)
    angle = i * np.pi * (3 - np.sqrt(5))
    start = np.stack((r * np.cos(angle), r * np.sin(angle)))  # (2, n)
    bodies = [ChargedBody(*p, charge=total / n, mass=1.0 / n) for p in start.T]
    rate_constant = 2 * 1.0 * total / radius**2  # q/m = 1, k_e = 1
    tau = rate_constant**-0.5
    sim = Simulation(
        bodies,
        Quad(0, 0, radius),
        k_e=1.0,
        kernel="log",
        method="fmm",
        softening=0.25 * radius * np.sqrt(np.pi / n),
        dt=0.02 * tau,
    )
    energy = sim.get_total_energy(exact=True)
    for _ in range(150):
        sim.step()
    s = cylinder_scale(sim.current_time, rate_constant)
    rms = np.sqrt(np.mean(np.sum(sim.positions**2, axis=0)) / np.mean(r**2))
    assert rms / s - 1 == pytest.approx(0, abs=1e-3)
    assert abs(sim.get_total_energy(exact=True) / energy - 1) < 1e-4


def test_symmetric_truncation_conserves_momentum_to_roundoff():
    pos, q = cloud(6000, seed=3)
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=18, symmetric=True)
    _, field = tree.fmm(theta=0.5)
    force = q * field
    assert np.abs(force.sum(axis=1)).max() < 1e-15 * np.abs(force).sum()
    # same leading error order as the rectangle truncation, larger constant
    ref_phi, ref_field = reference(pos, q)
    rect = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=18).fmm(theta=0.5)[1]
    assert relative(rect, ref_field) < relative(field, ref_field) < 1e-6


@pytest.mark.parametrize("softening", [0.0, 0.02])
def test_coulomb_fmm_converges_with_order_and_conserves_momentum(softening):
    rng = np.random.default_rng(11)
    pos = np.concatenate(
        (rng.uniform(-0.9, 0.9, (2, 1500)), rng.normal(0.3, 0.05, (2, 1500))), axis=1
    ).clip(-0.99, 0.99)
    q = rng.normal(size=pos.shape[1])
    everyone = np.arange(pos.shape[1])
    phi_exact, exact = direct_field(pos, pos, q, exclude=everyone, softening=softening)
    errors = []
    for order in (2, 4, 6, 8):
        tree = FlatQuadtree(pos, q, Quad(0, 0, 1), order=order, leaf_capacity=16)
        phi, field = tree.fmm(theta=0.5, softening=softening)
        errors.append(float(np.max(np.abs(field - exact)) / np.max(np.abs(exact))))
        assert np.max(np.abs(phi - phi_exact)) < 0.05 * np.max(np.abs(phi_exact))
        # the total-degree truncation makes pair forces exactly antisymmetric
        net = np.abs((q * field).sum(axis=1)).max()
        assert net < 1e-12 * np.abs(q * field).sum()
    assert errors[-1] < 1e-2 * errors[0] and all(np.diff(errors) < 0)


def test_coulomb_fmm_without_pyccel_and_on_a_set_charges_tree():
    rng = np.random.default_rng(12)
    pos = rng.uniform(-0.9, 0.9, (2, 600))
    q, other = rng.normal(size=600), rng.normal(size=600)
    compiled = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=8)
    python = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=8, compiled=False)
    for a, b in zip(compiled.fmm(softening=0.01), python.fmm(softening=0.01)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    compiled.set_charges(other)
    fresh = FlatQuadtree(pos, other, Quad(0, 0, 1), leaf_capacity=8)
    np.testing.assert_allclose(
        compiled.fmm(softening=0.01)[1], fresh.fmm(softening=0.01)[1], atol=1e-12
    )


@pytest.mark.parametrize("kernel", ["coulomb", "log"])
def test_fmm_at_targets_matches_the_direct_sum(kernel):
    rng = np.random.default_rng(13)
    pos, q = cloud(4000, seed=13)
    targets = np.concatenate((rng.uniform(-0.9, 0.9, (2, 1500)), pos[:, :500]), axis=1)
    exclude = np.concatenate((np.full(1500, -1), np.arange(500)))
    order = 8 if kernel == "coulomb" else 16
    tree = FlatQuadtree(
        pos, q, Quad(0, 0, 1), kernel=kernel, order=order, leaf_capacity=16
    )
    # the log kernel's expansions are unsoftened, as in the walk: compare unsoftened
    eps = 0.01 if kernel == "coulomb" else 0.0
    phi, field = tree.fmm_at(targets, exclude=exclude, theta=0.5, softening=eps)
    phi_ref, ref = direct_field(
        targets, pos, q, exclude=exclude, softening=eps, kernel=kernel
    )
    tolerance = 2e-4
    assert np.max(np.abs(field - ref)) < tolerance * np.max(np.abs(ref))
    assert np.max(np.abs(phi - phi_ref)) < tolerance * np.max(np.abs(phi_ref))
    _, walk = tree.evaluate(targets, exclude=exclude, theta=0.5, softening=eps)
    assert np.max(np.abs(field - walk)) < 20 * tolerance * np.max(np.abs(ref))
    no_phi, same = tree.fmm_at(
        targets, exclude=exclude, theta=0.5, softening=eps, potential=False
    )
    assert not np.any(no_phi)
    np.testing.assert_allclose(same, field, rtol=1e-12, atol=1e-12)


def test_fmm_at_keeps_its_plan_across_new_charges_and_matches_python():
    rng = np.random.default_rng(14)
    pos = rng.uniform(-0.9, 0.9, (2, 800))
    targets = rng.uniform(-0.9, 0.9, (2, 300))
    q, other = rng.normal(size=800), rng.normal(size=800)
    other[:400] = 0.0  # cells without charge must not drop out of a kept plan
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", leaf_capacity=8)
    tree.fmm_at(targets)
    tree.set_charges(other)
    fresh = FlatQuadtree(pos, other, Quad(0, 0, 1), kernel="log", leaf_capacity=8)
    np.testing.assert_allclose(
        tree.fmm_at(targets)[1], fresh.fmm_at(targets)[1], rtol=1e-12, atol=1e-12
    )
    python = FlatQuadtree(
        pos, other, Quad(0, 0, 1), kernel="log", leaf_capacity=8, compiled=False
    )
    for a, b in zip(fresh.fmm_at(targets), python.fmm_at(targets)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_fmm_at_validation():
    pos = np.array([[0.1, 0.5], [0.2, -0.3]])
    tree = FlatQuadtree(pos, np.ones(2), Quad(0, 0, 1))
    with pytest.raises(ValueError, match="Coincident"):
        tree.fmm_at(pos[:, :1])
    phi, field = tree.fmm_at(pos[:, :1], exclude=[0])
    _, ref = direct_field(pos[:, :1], pos, np.ones(2), exclude=[0])
    np.testing.assert_allclose(field, ref, rtol=1e-12)
    with pytest.raises(ValueError, match="theta"):
        tree.fmm_at(pos, theta=1.0)
    assert tree.fmm_at(np.zeros((2, 0)))[1].shape == (2, 0)
