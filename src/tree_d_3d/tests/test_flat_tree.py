"""The array quadtree and compiled kernels against the reference object tree."""

import cunumpy as xp
import numpy as np
import pytest

import tree_d_3d.simulation as bh
from tree_d_3d.bodies import ChargedBody
from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.quad import Quad
from tree_d_3d.reference import ReferenceQuadtree
from tree_d_3d.simulation import ElectrostaticSimulation as Simulation


def cloud(n, seed=0, neutral_fraction=0.1):
    """Positions ``(2, n)`` (component-major) and charges ``(n,)``."""
    rng = np.random.default_rng(seed)
    # drawn as (n, 2) so the point set matches the tolerances tuned below
    pos = np.ascontiguousarray(rng.normal(0, 0.3, (n, 2)).T).clip(-0.99, 0.99)
    pos[:, : n // 10] = pos[:, :1]  # a coincident clump exercises max_depth buckets
    charge = np.where(rng.random(n) < 0.5, -1.0, 1.0) * rng.uniform(0.5, 2, n)
    charge[rng.random(n) < neutral_fraction] = 0.0
    return pos, charge


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.2])
@pytest.mark.parametrize("leaf_capacity,max_depth", [(1, 10), (8, 48)])
def test_flat_tree_matches_object_tree(theta, leaf_capacity, max_depth):
    pos, charge = cloud(300)
    bodies = [ChargedBody(*p, charge=q) for p, q in zip(pos.T, charge)]
    reference = ReferenceQuadtree(
        Quad(0, 0, 1), leaf_capacity=leaf_capacity, max_depth=max_depth
    )
    for body in bodies:
        reference.insert(body)
    tree = FlatQuadtree(
        pos, charge, Quad(0, 0, 1), leaf_capacity=leaf_capacity, max_depth=max_depth
    )
    assert tree.node_count == len(reference.collect_nodes()) + sum(
        child.count == 0
        for node in reference.collect_nodes()
        for child in node.children
        if child is not None
    )
    phi, field = tree.evaluate(
        pos, exclude=np.arange(pos.shape[1]), theta=theta, softening=0.01
    )
    assert field.shape == pos.shape
    for i, body in enumerate(bodies):
        ref_phi, ref_field = reference.field_at(
            body.pos, theta=theta, softening=0.01, k_e=1, exclude=body
        )
        scale = np.abs(charge).sum() / 0.01
        assert phi[i] == pytest.approx(ref_phi, abs=1e-12 * scale)
        np.testing.assert_allclose(field[:, i], ref_field, atol=1e-12 * scale / 0.01)


def test_compiled_kernels_match_python_source():
    if not bh.kernels_compiled():
        pytest.skip("Pyccel kernels unavailable")
    pos, charge = cloud(200, seed=1)
    exclude = np.arange(pos.shape[1])
    compiled = FlatQuadtree(pos, charge, Quad(0, 0, 1), leaf_capacity=2)
    python = FlatQuadtree(pos, charge, Quad(0, 0, 1), leaf_capacity=2, compiled=False)
    np.testing.assert_array_equal(compiled.perm, python.perm)
    np.testing.assert_array_equal(compiled.node_child, python.node_child)
    for a, b in zip(
        compiled.evaluate(pos, exclude=exclude, theta=0.7, softening=0.01),
        python.evaluate(pos, exclude=exclude, theta=0.7, softening=0.01),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-10)
    for a, b in zip(
        direct_field(pos, pos, charge, softening=0.01, exclude=exclude),
        direct_field(pos, pos, charge, softening=0.01, exclude=exclude, compiled=False),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-10)


def test_numpy_kernels_match_direct_kernel():
    pos, charge = cloud(150, seed=2)
    exclude = np.arange(pos.shape[1])
    expected = direct_field(pos, pos, charge, softening=0.02, exclude=exclude)
    with xp.kernels.use_host_kernel_implementation("numpy"):
        vectorized = direct_field(pos, pos, charge, softening=0.02, exclude=exclude)
        with pytest.raises(ValueError, match="Coincident"):
            direct_field(pos, pos, charge, softening=0.0, exclude=exclude)
    for a, b in zip(expected, vectorized):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)


def test_node_storage_grows_for_deep_coincident_chains():
    tree = FlatQuadtree(
        [[0.3, 0.3, -0.5], [0.3, 0.3, 0.2]],  # (2, N): two coincide
        [1, 1, -1],
        Quad(0, 0, 1),
        leaf_capacity=1,
        max_depth=48,
    )
    assert tree.node_count == 1 + 4 * 48
    assert tree.node_depth.max() == 48
    assert tree.root.count == 3


def test_positions_must_be_component_major():
    with pytest.raises(ValueError, match=r"\(2, N\)"):
        FlatQuadtree(np.zeros((3, 2)), np.zeros(3), Quad(0, 0, 1))
    tree = FlatQuadtree(np.zeros((2, 3)), np.zeros(3), Quad(0, 0, 1))
    with pytest.raises(ValueError, match=r"\(2, N\)"):
        tree.evaluate(np.zeros((5, 2)))
    with pytest.raises(ValueError, match=r"\(2, N\)"):
        direct_field(np.zeros((5, 2)), tree.pos, tree.charge)


def test_multi_probe_field_matches_direct_sum():
    pos, charge = cloud(100, seed=4)
    tree = FlatQuadtree(pos, charge, Quad(0, 0, 1))
    probes = np.array([[3.0, 0.0], [0.0, -4.0]])  # (2, probes): (3, 0) and (0, -4)
    phi, field = tree.field_at(probes, theta=0, k_e=2, softening=0.01)
    ref_phi, ref_field = direct_field(probes, pos, charge, softening=0.01)
    assert phi.shape == (2,) and field.shape == (2, 2)
    np.testing.assert_allclose(phi, 2 * ref_phi)
    np.testing.assert_allclose(field, 2 * ref_field)


def make_simulation(n=40, **kwargs):
    pos, charge = cloud(n, seed=5, neutral_fraction=0)
    pos[:, : n // 10] += np.linspace(0.01, 0.02, n // 10)
    bodies = [ChargedBody(*p, charge=q, mass=1.0) for p, q in zip(pos.T, charge)]
    kwargs = {"k_e": 1, "softening": 0.05, "dt": 1e-3, **kwargs}
    return Simulation(bodies, Quad(0, 0, 1), **kwargs)


@pytest.mark.parametrize("method", ["tree", "direct"])
def test_one_field_evaluation_per_step(monkeypatch, method):
    sim = make_simulation(method=method)
    calls = []
    evaluate, direct = FlatQuadtree.evaluate, bh.direct_field
    monkeypatch.setattr(
        FlatQuadtree,
        "evaluate",
        lambda *a, **k: calls.append("tree") or evaluate(*a, **k),
    )
    monkeypatch.setattr(
        bh, "direct_field", lambda *a, **k: calls.append("direct") or direct(*a, **k)
    )
    sim.step()
    assert calls == [method, method]
    for _ in range(3):
        sim.step()
    assert calls == [method] * 5
    sim.get_potential_energy()
    assert len(calls) == 5


def test_reused_forces_match_fresh_evaluation():
    cached = make_simulation(method="tree")
    for _ in range(5):
        cached.step()
    fresh = make_simulation(method="tree")
    for _ in range(5):
        fresh._field_state = None
        fresh.step()
    np.testing.assert_array_equal(cached.positions, fresh.positions)
    np.testing.assert_array_equal(cached.velocities, fresh.velocities)


def test_cache_invalidated_by_external_changes():
    sim = make_simulation()
    sim.step()
    before = sim._electric.copy()
    sim.bodies[0].pos = sim.bodies[0].pos + 0.1
    sim.compute_fields()
    assert not np.array_equal(before, sim._electric)
    before = sim._electric.copy()
    sim.bodies[1].charge = 5.0
    sim.compute_fields()
    assert not np.array_equal(before, sim._electric)
    before = sim._electric.copy()
    sim.softening = 0.1
    sim.compute_fields()
    assert not np.array_equal(before, sim._electric)


def test_bodies_are_views_of_simulation_arrays():
    sim = make_simulation(n=10)
    assert sim.positions.shape == sim.velocities.shape == sim.forces.shape == (2, 10)
    body = sim.bodies[3]
    body.vel[0] = 7.0
    assert sim.velocities[0, 3] == 7.0
    sim.positions[:, 3] = (0.5, -0.5)
    np.testing.assert_array_equal(body.pos, [0.5, -0.5])
    body.pos = (0.25, 0.75)
    np.testing.assert_array_equal(sim.positions[:, 3], [0.25, 0.75])
    body.fixed = True
    assert sim.fixed[3]


def test_bodies_added_after_construction_are_adopted():
    sim = make_simulation(n=10)
    extra = ChargedBody(0.9, 0.9, charge=-1)
    sim.bodies.append(extra)
    sim.step()
    assert sim.positions.shape == (2, 11)
    assert np.any(extra.force)


def test_auto_method_selection():
    assert make_simulation(n=20).step() is None
    # Uncompiled, the NumPy direct sum beats a Python tree walk at any size.
    tree_expected = FlatQuadtree if bh.kernels_compiled() else type(None)
    assert isinstance(make_simulation(n=20, direct_threshold=5).step(), tree_expected)
    assert make_simulation(n=20, direct_threshold=5, theta=0).step() is None
    with pytest.raises(ValueError, match="method"):
        make_simulation(method="fmm")
    with pytest.raises(ValueError, match="direct_threshold"):
        make_simulation(direct_threshold=-1)


def test_potential_energy_from_fields():
    exact = make_simulation(n=200, method="direct")
    tree = make_simulation(n=200, method="tree")
    reference = exact.get_potential_energy(exact=True)
    assert exact.get_potential_energy() == pytest.approx(reference, rel=1e-12)
    assert tree.get_potential_energy(exact=True) == pytest.approx(reference, rel=1e-12)
    assert tree.get_potential_energy() == pytest.approx(reference, rel=1e-3)
    assert tree.get_potential_energy() != reference


@pytest.mark.parametrize("method", ["tree", "direct"])
def test_fixed_particles_are_sources_but_not_targets_while_stepping(method):
    rng = np.random.default_rng(7)
    ions = [
        ChargedBody(*p, charge=1.0, fixed=True) for p in rng.uniform(-0.8, 0.8, (30, 2))
    ]
    electrons = [ChargedBody(*p, charge=-1.0) for p in rng.uniform(-0.8, 0.8, (30, 2))]
    kwargs = {"k_e": 1, "softening": 0.05, "dt": 1e-3, "method": method}
    sim = Simulation(ions + electrons, Quad(0, 0, 1), **kwargs)
    sim.step()
    assert np.all(sim.forces[:, :30] == 0) and np.any(sim.forces[:, 30:])
    # Asking for the energy fills in the fixed targets from the same fields.
    energy = sim.get_potential_energy()
    reference = Simulation(
        [
            ChargedBody(*b.pos, *b.vel, charge=b.charge, fixed=b.fixed)
            for b in sim.bodies
        ],
        Quad(0, 0, 1),
        **kwargs,
    )
    assert energy == pytest.approx(reference.get_potential_energy(), rel=1e-12)
    np.testing.assert_allclose(sim._electric, reference._electric, rtol=1e-12)
    # Electron forces match a full evaluation.
    reference.compute_fields()
    np.testing.assert_allclose(sim._electric[:, 30:], reference._electric[:, 30:])
