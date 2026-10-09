"""Analytic physics, direct-sum accuracy and pathological tree regressions."""

import numpy as np
import pytest

from tree_d_3d.bodies import ChargedBody
from tree_d_3d.quad import Quad
from tree_d_3d.reference import ReferenceQuadtree
from tree_d_3d.simulation import ElectrostaticSimulation as Simulation


def direct_field(bodies, position, softening, exclude=None):
    """Independent direct Coulomb reference."""
    phi = 0.0
    field = np.zeros(2)
    for body in bodies:
        if body is exclude:
            continue
        delta = position - body.pos
        radius = np.sqrt(np.dot(delta, delta) + softening**2)
        phi += body.charge / radius
        field += body.charge * delta / radius**3
    return phi, field


@pytest.mark.parametrize("charges", ["positive", "mixed", "neutral"])
def test_force_accuracy_and_opening_convergence(charges):
    rng = np.random.default_rng(2026)
    positions = rng.uniform(-1, 1, (128, 2))
    q = rng.uniform(0.5, 1.5, 128)
    if charges == "mixed":
        q[::2] *= -1
    elif charges == "neutral":
        q[:] = np.tile([1.0, -1.0], 64)
    bodies = [ChargedBody(*p, charge=c) for p, c in zip(positions, q)]
    tree = ReferenceQuadtree(Quad(0, 0, 1), leaf_capacity=2)
    for body in bodies:
        tree.insert(body)
    exact = np.array([direct_field(bodies, b.pos, 0.03, b)[1] for b in bodies])
    errors = []
    for theta in (0.8, 0.4, 0.0):
        fields = np.array(
            [
                tree.field_at(b.pos, theta=theta, softening=0.03, k_e=1, exclude=b)[1]
                for b in bodies
            ]
        )
        errors.append(np.linalg.norm(fields - exact) / np.linalg.norm(exact))
    assert errors[1] < errors[0]
    assert errors[1] < 0.003
    assert errors[2] < 1e-14


@pytest.mark.parametrize("imbalance", [0.0, 1e-10])
def test_neutral_and_nearly_neutral_dipoles(imbalance):
    bodies = [
        ChargedBody(-0.1, 0, charge=-1),
        ChargedBody(0.1, 0, charge=1 + imbalance),
    ]
    tree = ReferenceQuadtree(Quad(0, 0, 0.2), leaf_capacity=1)
    for body in bodies:
        tree.insert(body)
    point = np.array([10.0, 2.0])
    expected_phi, expected_e = direct_field(bodies, point, 0.02)
    phi, electric = tree.field_at(point, theta=0.5, softening=0.02, k_e=1)
    assert phi == pytest.approx(expected_phi, rel=3e-4)
    np.testing.assert_allclose(electric, expected_e, rtol=4e-4)


def test_neutral_quadrupole_and_potential_gradient():
    tree = ReferenceQuadtree(Quad(0, 0, 0.3), leaf_capacity=1)
    bodies = [
        ChargedBody(x, y, charge=q)
        for x, y, q in [(0.1, 0, 1), (-0.1, 0, 1), (0, 0.1, -1), (0, -0.1, -1)]
    ]
    for body in bodies:
        tree.insert(body)
    point = np.array([8.0, 3.0])
    kwargs = {"theta": 0.5, "softening": 0.7, "k_e": 1}
    phi, electric = tree.field_at(point, **kwargs)
    reference_phi, reference_e = direct_field(bodies, point, 0.7)
    assert phi == pytest.approx(reference_phi, rel=0.001)
    np.testing.assert_allclose(electric, reference_e, rtol=0.001)
    h = 1e-4
    gradient = np.array(
        [
            (
                tree.field_at(point + h * axis, **kwargs)[0]
                - tree.field_at(point - h * axis, **kwargs)[0]
            )
            / (2 * h)
            for axis in np.eye(2)
        ]
    )
    np.testing.assert_allclose(electric, -gradient, rtol=1e-8)


def test_self_force_and_softening_do_not_allow_containing_cell():
    bodies = [ChargedBody(-0.8, -0.8), ChargedBody(0.8, 0.8, charge=-2)]
    tree = ReferenceQuadtree(Quad(0, 0, 1), leaf_capacity=1)
    for body in bodies:
        tree.insert(body)
    for body in bodies:
        tree.compute_lorentz_force_on(body, theta=100, softening=100, k_e=1)
        expected = body.charge * direct_field(bodies, body.pos, 100, body)[1]
        np.testing.assert_allclose(body.force, expected, atol=1e-20)


def test_coincident_particles_and_neutral_markers():
    bodies = [ChargedBody(0, 0, charge=q) for q in (0, 1, -1, 1, -1) * 8]
    sim = Simulation(
        bodies,
        Quad(0, 0, 1),
        k_e=1,
        softening=0.1,
        leaf_capacity=2,
        max_depth=12,
        method="tree",
    )
    tree = sim.step()
    assert tree.root.count == 40
    assert len(tree.collect_nodes()) <= 13
    assert tree.root.total_abs_charge == 32
    assert all(np.array_equal(body.force, [0, 0]) for body in bodies)
    assert np.isfinite(sim.get_total_energy())


def test_unsoftened_collision_is_explicit():
    sim = Simulation(
        [ChargedBody(0, 0), ChargedBody(0, 0)], Quad(0, 0, 1), softening=0, k_e=1
    )
    with pytest.raises(ValueError, match="Coincident"):
        sim.step()
    with pytest.raises(ValueError, match="Coincident"):
        sim.get_potential_energy()


def test_neutral_particle_can_overlap_charged_source_without_softening():
    bodies = [ChargedBody(0, 0, charge=0), ChargedBody(0, 0)]
    sim = Simulation(bodies, Quad(0, 0, 1), k_e=1)
    sim.step()
    assert sim.softening == 0
    assert sim.get_total_energy() == 0
    assert all(np.array_equal(b.force, [0, 0]) for b in bodies)


def test_off_particle_probe_excludes_requested_source():
    bodies = [ChargedBody(-0.5, 0), ChargedBody(0.5, 0, charge=-1)]
    tree = ReferenceQuadtree(Quad(0, 0, 1), leaf_capacity=1)
    for body in bodies:
        tree.insert(body)
    point = np.array([20.0, 10.0])
    phi, field = tree.field_at(point, exclude=bodies[0], k_e=1)
    reference_phi, reference_field = direct_field(bodies, point, 0, bodies[0])
    assert phi == pytest.approx(reference_phi)
    np.testing.assert_allclose(field, reference_field)


def test_bounds_expand_and_retain_all_sources():
    bodies = [ChargedBody(-0.1, 0, vx=-5), ChargedBody(0.1, 0, vx=5)]
    sim = Simulation(
        bodies, Quad(0, 0, 0.2), k_e=1, dt=0.1, softening=0.2, method="tree"
    )
    tree = sim.step()
    assert tree.root.count == 2
    assert tree.root.total_charge == 2
    assert all(sim.bounds.contains(*b.pos) for b in bodies)
    assert bodies[0].force[0] < 0 < bodies[1].force[0]
    np.testing.assert_allclose(bodies[0].force, -bodies[1].force)


@pytest.mark.parametrize("charge,B_z,dt", [(1, 1, 0.02), (-1, 3, 0.7), (1, -2, 1.3)])
def test_exact_cyclotron_orbit_speed_and_force(charge, B_z, dt):
    body = ChargedBody(0, 0, vx=1, charge=charge)
    sim = Simulation([body], Quad(0, 0, 1), k_e=1, dt=dt, B_z=B_z)
    for _ in range(100):
        sim.step()
    omega = charge * B_z
    angle = omega * sim.current_time
    np.testing.assert_allclose(
        body.pos, [np.sin(angle) / omega, (np.cos(angle) - 1) / omega], atol=2e-11
    )
    np.testing.assert_allclose(body.vel, [np.cos(angle), -np.sin(angle)], atol=2e-11)
    assert sim.get_kinetic_energy() == pytest.approx(0.5, rel=2e-13)
    np.testing.assert_allclose(
        body.force, omega * np.array([body.vel[1], -body.vel[0]])
    )


def orbit(dt, B_z=0):
    softening = 0.1
    speed = (1 + softening**2) ** (-0.75)
    electron = ChargedBody(1, 0, vy=speed, charge=-1)
    ion = ChargedBody(0, 0, fixed=True)
    sim = Simulation(
        [electron, ion],
        Quad(0, 0, 2),
        k_e=1,
        dt=dt,
        theta=0,
        softening=softening,
        B_z=B_z,
    )
    initial_energy = sim.get_total_energy()
    for _ in range(round(2 / dt)):
        sim.step()
    assert np.array_equal(ion.pos, [0, 0])
    assert np.array_equal(ion.vel, [0, 0])
    return sim, electron, initial_energy, speed


def test_electric_orbit_second_order_and_energy():
    errors = []
    for dt in (0.04, 0.02):
        sim, electron, energy, speed = orbit(dt)
        exact = np.array([np.cos(2 * speed), np.sin(2 * speed)])
        errors.append(np.linalg.norm(electron.pos - exact))
        assert abs((sim.get_total_energy() - energy) / energy) < 2e-6
        assert sim.get_angular_momentum() == pytest.approx(speed, rel=1e-13)
    assert 3.8 < errors[0] / errors[1] < 4.2


def test_mixed_electric_magnetic_second_order():
    reference = orbit(0.00125, B_z=0.7)[1].pos
    coarse = np.linalg.norm(orbit(0.04, B_z=0.7)[1].pos - reference)
    fine = np.linalg.norm(orbit(0.02, B_z=0.7)[1].pos - reference)
    assert 3.8 < coarse / fine < 4.2


def test_pair_momentum_and_total_energy_conservation():
    bodies = [ChargedBody(-1, 0, vy=0.2, mass=2), ChargedBody(1, 0, vy=-0.4, mass=1)]
    sim = Simulation(bodies, Quad(0, 0, 2), k_e=1, dt=0.01, softening=0.05)
    energy = sim.get_total_energy()
    for _ in range(200):
        sim.step()
        np.testing.assert_allclose(sim.get_momentum(), [0, 0], atol=2e-14)
        assert abs((sim.get_total_energy() - energy) / energy) < 1e-5


def test_weighted_particles_preserve_charge_to_mass_and_scale_source():
    body = ChargedBody(1, 0, mass=2, charge=-3, weight=10)
    assert body.mass == 20
    assert body.charge == -30
    assert body.charge / body.mass == -1.5
    tree = ReferenceQuadtree(Quad(0, 0, 2))
    tree.insert(body)
    phi, field = tree.field_at([0, 0], k_e=1, softening=0)
    assert phi == -30
    np.testing.assert_allclose(field, [30, 0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": 0},
        {"mass": -1},
        {"mass": np.inf},
        {"charge": np.nan},
        {"weight": 0},
        {"weight": 1e-200, "mass": 1e-200},
        {"vx": np.inf},
        {"fixed": True, "vx": 1},
    ],
)
def test_invalid_particle_inputs(kwargs):
    with pytest.raises(ValueError):
        ChargedBody(0, 0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta": -1},
        {"softening": -1},
        {"dt": 0},
        {"dt": np.nan},
        {"k_e": -1},
        {"B_z": np.inf},
        {"max_depth": 0},
        {"leaf_capacity": 1.2},
    ],
)
def test_invalid_solver_inputs(kwargs):
    with pytest.raises(ValueError):
        Simulation([], Quad(0, 0, 1), **kwargs)


def test_tree_validation_and_empty_simulation():
    body = ChargedBody(0, 0)
    tree = ReferenceQuadtree(Quad(0, 0, 1))
    tree.insert(body)
    with pytest.raises(ValueError, match="twice"):
        tree.insert(body)
    with pytest.raises(ValueError, match="outside"):
        tree.insert(ChargedBody(2, 0))
    with pytest.raises(ValueError, match="distinct"):
        Simulation([body, body], Quad(0, 0, 1))
    sim = Simulation([], Quad(0, 0, 1), dt=0.1, method="tree")
    assert sim.step().root.count == 0
    assert sim.current_time == 0.1
    assert sim.get_total_energy() == 0
    np.testing.assert_array_equal(sim.get_center_of_mass(), [0, 0])
    with pytest.raises(ValueError, match="dt"):
        sim.step(dt=-1)
