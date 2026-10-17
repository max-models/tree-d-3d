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
