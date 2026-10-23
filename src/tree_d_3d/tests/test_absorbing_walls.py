"""Absorbing walls: particles that cross a wall are removed and recorded."""

import numpy as np
import pytest

from tree_d_3d.bodies import ChargedBody
from tree_d_3d.quad import Quad
from tree_d_3d.simulation import ElectrostaticSimulation as Simulation

WALLS = {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0}


def test_free_particles_are_absorbed_at_the_wall_they_cross():
    bodies = [
        ChargedBody(0.5, 0.0, vx=1.0, charge=0),
        ChargedBody(-0.5, 0.0, vx=-1.0, charge=0),
        ChargedBody(0.0, 0.5, vy=1.0, charge=0),
        ChargedBody(0.0, -0.7, vy=-1.0, charge=0),
        ChargedBody(0.0, 0.0, charge=0),
    ]
    sim = Simulation(bodies, Quad(0, 0, 1), k_e=1, dt=0.1, absorbing_walls=WALLS)
    for _ in range(5):
        sim.step()
    # The y_min particle crosses on step 4; the others exactly reach x = 1 on
    # step 5 and stay, because a particle on the wall is still inside.
    assert sim.absorbed.counts() == {"y_min": 1}
    assert sim.absorbed.time[0] == pytest.approx(0.4)
    for _ in range(1):
        sim.step()
    assert sim.absorbed.counts() == {"x_max": 1, "x_min": 1, "y_max": 1, "y_min": 1}
    assert sim.bodies == [bodies[4]]
    assert sim.positions.shape == (2, 1)
    assert sim.absorbed.position.shape == sim.absorbed.velocity.shape == (2, 4)
    np.testing.assert_allclose(sim.absorbed.time, [0.4, 0.6, 0.6, 0.6])
    # Absorbed bodies are detached and keep their state at absorption.
    frozen = bodies[0].pos.copy()
    sim.step()
    np.testing.assert_array_equal(bodies[0].pos, frozen)
    assert bodies[0].pos[0] > 1.0
    assert sim.absorbed.kinetic_energy == pytest.approx(4 * 0.5)


def test_absorbed_particles_stop_being_sources():
    probe = ChargedBody(0.0, 0.0, charge=1, fixed=True)
    leaving = ChargedBody(0.9, 0.0, vx=10.0, charge=1)
    sim = Simulation(
        [probe, leaving], Quad(0, 0, 1), k_e=1, dt=0.05, absorbing_walls=WALLS
    )
    sim.step()
    assert len(sim.absorbed) == 1
    np.testing.assert_array_equal(probe.force, [0, 0])
    assert sim.get_potential_energy() == 0
    assert sim.get_total_charge() == 1
    assert sim.absorbed.total_charge == 1


@pytest.mark.parametrize("method", ["tree", "direct"])
def test_rebound_views_stay_consistent_after_removal(method):
    rng = np.random.default_rng(0)
    bodies = [
        ChargedBody(*rng.uniform(-0.9, 0.9, 2), *rng.normal(0, 3, 2), charge=q)
        for q in rng.choice([-1.0, 1.0], 60)
    ]
    sim = Simulation(
        bodies,
        Quad(0, 0, 1),
        k_e=1e-3,
        softening=0.05,
        dt=0.02,
        method=method,
        absorbing_walls=WALLS,
    )
    for _ in range(20):
        sim.step()
    assert 0 < len(sim.absorbed) < 60
    assert len(sim.bodies) + len(sim.absorbed) == 60
    for i, body in enumerate(sim.bodies):
        np.testing.assert_array_equal(body.pos, sim.positions[:, i])
        assert body.charge == sim.charges[i]
    assert np.all(np.abs(sim.positions) <= 1.0)
    # Remaining forces are those of the remaining sources only.
    fresh = Simulation(
        [ChargedBody(*b.pos, *b.vel, charge=b.charge) for b in sim.bodies],
        Quad(0, 0, 1),
        k_e=1e-3,
        softening=0.05,
        method=method,
    )
    fresh.compute_fields()
    np.testing.assert_allclose(sim._electric, fresh._electric, rtol=1e-12)


def test_open_axes_and_partial_walls():
    body = ChargedBody(0.0, 0.0, vx=-1.0, vy=5.0, charge=0)
    sim = Simulation([body], Quad(0, 0, 1), dt=0.5, absorbing_walls={"x_min": -1.0})
    sim.step()
    assert len(sim.absorbed) == 0 and body.pos[1] == 2.5
    sim.step()
    sim.step()
    assert sim.absorbed.counts() == {"x_min": 1}


def test_all_particles_absorbed_leaves_an_empty_simulation():
    sim = Simulation(
        [ChargedBody(0.9, 0, vx=1)], Quad(0, 0, 1), dt=0.5, absorbing_walls=WALLS
    )
    sim.step()
    assert sim.bodies == [] and len(sim.absorbed) == 1
    sim.step()
    assert sim.get_total_energy() == 0


@pytest.mark.parametrize(
    "walls,match",
    [
        ({"z_min": 0.0}, "Unknown"),
        ({"x_min": 1.0, "x_max": -1.0}, "below"),
        ({"x_max": np.nan}, "finite"),
        ({"x_max": 0.5}, "start inside"),
    ],
)
def test_invalid_walls(walls, match):
    with pytest.raises(ValueError, match=match):
        Simulation([ChargedBody(0.7, 0)], Quad(0, 0, 1), absorbing_walls=walls)


def test_particles_entering_an_absorbing_object_are_recorded_under_its_name():
    from simgeometry import Circle, Rectangle

    bodies = [
        ChargedBody(-0.45, 0.0, vx=1.0, charge=0),  # into the plate at x in [-0.2, 0.2]
        ChargedBody(0.0, 0.55, vy=1.0, charge=0),  # out of the round chamber
        ChargedBody(0.0, -0.5, charge=0),  # at rest, stays
    ]
    objects = {
        "plate": Rectangle(midpoint=(0.0, 0.0), side_lengths=(0.4, 0.4)),
        "chamber": Circle(midpoint=(0.0, 0.0), radius=0.8, inverted=True),
    }
    sim = Simulation(bodies, Quad(0, 0, 1), k_e=1, dt=0.1, absorbing_objects=objects)
    assert set(sim.absorbing_objects) == {"plate", "chamber"}
    for _ in range(4):
        sim.step()
    assert sim.absorbed.counts() == {"plate": 1, "chamber": 1}
    assert sim.bodies == [bodies[2]]
    np.testing.assert_allclose(sim.absorbed.time, [0.3, 0.3])
    assert list(sim.absorbed.wall) == ["plate", "chamber"]


def test_walls_take_precedence_over_objects():
    from simgeometry import Rectangle

    bodies = [ChargedBody(0.95, 0.0, vx=1.0, charge=0)]
    sim = Simulation(
        bodies,
        Quad(0, 0, 2),
        k_e=1,
        dt=0.1,
        absorbing_walls={"x_max": 1.0},
        absorbing_objects={
            "block": Rectangle(midpoint=(1.5, 0.0), side_lengths=(1, 1))
        },
    )
    sim.step()
    assert sim.absorbed.counts() == {"x_max": 1}


def test_absorbing_objects_are_validated():
    from simgeometry import Circle, Sphere

    inside = [ChargedBody(0.0, 0.0, charge=0)]
    kwargs = {"k_e": 1, "dt": 0.1}
    with pytest.raises(ValueError, match="outside the absorbing objects"):
        Simulation(
            inside,
            Quad(0, 0, 1),
            absorbing_objects={"disc": Circle(midpoint=(0, 0), radius=0.5)},
            **kwargs,
        )
    with pytest.raises(ValueError, match="2D simgeometry shape"):
        Simulation(
            inside,
            Quad(0, 0, 1),
            absorbing_objects={"ball": Sphere(midpoint=(5, 5, 5), radius=0.5)},
            **kwargs,
        )
    with pytest.raises(ValueError, match="wall name"):
        Simulation(
            inside,
            Quad(0, 0, 1),
            absorbing_objects={"x_max": Circle(midpoint=(5, 5), radius=0.5)},
            **kwargs,
        )


def _slit():
    from simgeometry import Aperture2D

    # plates 0.02 thick at x = 0, from |y| = 0.2 to 0.8
    return Aperture2D.slit(0.0, 0.2, 0.02, 0.6)


def test_steps_through_a_thin_plate_are_absorbed():
    bodies = [
        ChargedBody(-0.3, 0.0, vx=1.0, charge=0),  # through the aperture
        ChargedBody(-0.3, 0.5, vx=1.0, charge=0),  # at the upper plate
        ChargedBody(-0.3, -0.5, vx=1.0, charge=0),  # at the lower plate
        ChargedBody(-0.03, 0.15, vx=0.5, vy=1.0, charge=0),  # cuts a plate corner
    ]
    sim = Simulation(
        bodies, Quad(0, 0, 1), k_e=1, dt=0.2, absorbing_objects={"slit": _slit()}
    )
    for _ in range(3):
        sim.step()
    # each step (0.2) is ten times the plate thickness: the absorbed particles
    # stepped over the plate and ended beyond it
    assert sim.absorbed.counts() == {"slit": 3}
    assert sim.bodies == [bodies[0]] and bodies[0].pos[0] > 0.2
    assert np.all(sim.absorbed.position[0] > 0.01)
    np.testing.assert_allclose(sim.absorbed.time, [0.2, 0.4, 0.4])


def test_the_object_crossed_first_takes_the_particle():
    from simgeometry import Circle, Polygon

    square = Polygon([(0.3, -0.1), (0.3, 0.1), (0.4, 0.1), (0.4, -0.1)])
    disc = Circle(midpoint=(0.6, 0.0), radius=0.05)
    bodies = [
        ChargedBody(0.0, 0.0, vx=1.0, charge=0),  # square, then disc
        ChargedBody(0.9, 0.0, vx=-1.0, charge=0),  # disc, then square
        ChargedBody(0.0, 0.5, vx=1.0, charge=0),  # misses both
    ]
    for objects in ({"square": square, "disc": disc}, {"disc": disc, "square": square}):
        sim = Simulation(
            [ChargedBody(*b.pos, *b.vel, charge=0) for b in bodies],
            Quad(0, 0, 1),
            k_e=1,
            dt=1.0,
            absorbing_objects=objects,
        )
        sim.step()
        assert list(sim.absorbed.wall) == ["square", "disc"]
        assert sim.num_particles == 1


def test_absorbed_and_remaining_charge_add_up():
    from simgeometry import Circle

    rng = np.random.default_rng(3)
    n = 200
    positions = np.vstack([rng.uniform(-0.6, -0.05, n), rng.uniform(-0.6, 0.6, n)])
    velocities = np.vstack([rng.uniform(0.5, 3.0, n), rng.normal(0, 0.5, n)])
    charges = rng.choice([-1.0, 2.0], n)
    bodies = [
        ChargedBody(*positions[:, i], *velocities[:, i], charge=charges[i])
        for i in range(n)
    ]
    objects = {
        "slit": _slit(),
        "chamber": Circle(midpoint=(0.0, 0.0), radius=0.95, inverted=True),
    }
    sim = Simulation(
        bodies,
        Quad(0, 0, 1),
        k_e=1e-4,
        softening=0.05,
        dt=0.05,
        method="direct",
        absorbing_objects=objects,
    )
    for _ in range(20):
        sim.step()
    assert set(sim.absorbed.counts()) == {"slit", "chamber"}
    assert len(sim.bodies) + len(sim.absorbed) == n
    per_object = sim.absorbed.charges()
    assert sum(per_object.values()) == pytest.approx(sim.absorbed.total_charge)
    assert sim.get_total_charge() + sim.absorbed.total_charge == pytest.approx(
        charges.sum()
    )
    # no remaining particle sits in a plate or outside the chamber
    assert not np.any(objects["slit"].contains_points(sim.positions))
    assert np.all(np.hypot(*sim.positions) <= 0.95)
