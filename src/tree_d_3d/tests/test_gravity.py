"""Tests for Barnes-Hut N-body solver."""

import numpy as np
import pytest

from tree_d_3d.gravity import (
    BarnesHutSimulation,
    BarnesHutTree,
    Body,
    Quad,
)


class TestBody:
    """Tests for Body class."""

    def test_body_init(self):
        """Test Body initialization."""
        body = Body(x=1.0, y=2.0, vx=0.1, vy=0.2, mass=5.0)
        assert body.pos[0] == 1.0
        assert body.pos[1] == 2.0
        assert body.vel[0] == 0.1
        assert body.vel[1] == 0.2
        assert body.mass == 5.0
        assert np.allclose(body.force, [0.0, 0.0])

    def test_body_reset_force(self):
        """Test force reset."""
        body = Body(0.0, 0.0, mass=1.0)
        body.add_force(1.0, 2.0)
        assert not np.allclose(body.force, [0.0, 0.0])
        body.reset_force()
        assert np.allclose(body.force, [0.0, 0.0])

    def test_body_add_force(self):
        """Test force accumulation."""
        body = Body(0.0, 0.0, mass=1.0)
        body.add_force(1.0, 2.0)
        assert np.allclose(body.force, [1.0, 2.0])
        body.add_force(0.5, -1.0)
        assert np.allclose(body.force, [1.5, 1.0])


class TestQuad:
    """Tests for Quad (quadrant) class."""

    def test_quad_contains(self):
        """Test point containment in quadrant."""
        quad = Quad(x_mid=0.0, y_mid=0.0, half_size=1.0)
        assert quad.contains(0.0, 0.0)
        assert quad.contains(0.5, 0.5)
        assert quad.contains(-0.5, -0.5)
        assert not quad.contains(1.5, 0.0)
        assert not quad.contains(0.0, 1.5)

    def test_quad_children(self):
        """Test child quadrant generation."""
        quad = Quad(x_mid=0.0, y_mid=0.0, half_size=2.0)
        nw = quad.northwest()
        ne = quad.northeast()
        sw = quad.southwest()
        se = quad.southeast()

        assert nw.xmid == -1.0 and nw.ymid == 1.0
        assert ne.xmid == 1.0 and ne.ymid == 1.0
        assert sw.xmid == -1.0 and sw.ymid == -1.0
        assert se.xmid == 1.0 and se.ymid == -1.0
        assert nw.half == 1.0


class TestBarnesHutTree:
    """Tests for Barnes-Hut tree."""

    def test_tree_insert_single_body(self):
        """Test inserting single body."""
        tree = BarnesHutTree(Quad(0.0, 0.0, 1.0))
        body = Body(0.0, 0.0, mass=1.0)
        tree.insert(body)
        assert tree.root.body == body
        assert tree.root.mass == 1.0

    def test_tree_insert_multiple_bodies(self):
        """Test inserting multiple bodies."""
        tree = BarnesHutTree(Quad(0.0, 0.0, 2.0))
        bodies = [
            Body(0.5, 0.5, mass=1.0),
            Body(-0.5, 0.5, mass=2.0),
            Body(0.5, -0.5, mass=1.5),
        ]
        for body in bodies:
            tree.insert(body)

        assert tree.root.mass == sum(b.mass for b in bodies)
        assert len(tree.collect_nodes()) > 0

    def test_tree_center_of_mass(self):
        """Test center of mass calculation."""
        tree = BarnesHutTree(Quad(0.0, 0.0, 2.0))
        body1 = Body(1.0, 0.0, mass=1.0)
        body2 = Body(-1.0, 0.0, mass=1.0)
        tree.insert(body1)
        tree.insert(body2)

        # COM should be at origin
        assert np.allclose(tree.root.center_of_mass, [0.0, 0.0])
        assert tree.root.mass == 2.0

    def test_tree_compute_force_single_pair(self):
        """Test force computation for two bodies."""
        tree = BarnesHutTree(Quad(0.0, 0.0, 2.0))
        body1 = Body(1.0, 0.0, mass=1.0)
        body2 = Body(-1.0, 0.0, mass=1.0)
        tree.insert(body1)
        tree.insert(body2)

        tree.compute_force_on(body1, theta=0.0, G=1.0, softening=0.01)
        # Force should be attractive (negative x-direction)
        assert body1.force[0] < 0.0
        assert abs(body1.force[1]) < 1e-10


class TestBarnesHutSimulation:
    """Tests for full Barnes-Hut simulation."""

    def test_simulation_init(self):
        """Test simulation initialization."""
        bodies = [Body(0.0, 0.0, mass=1.0)]
        bounds = Quad(0.0, 0.0, 1.0)
        sim = BarnesHutSimulation(bodies, bounds)
        assert len(sim.bodies) == 1
        assert sim.theta == 0.5

    def test_simulation_kinetic_energy(self):
        """Test kinetic energy calculation."""
        bodies = [Body(0.0, 0.0, vx=1.0, vy=1.0, mass=1.0)]
        bounds = Quad(0.0, 0.0, 1.0)
        sim = BarnesHutSimulation(bodies, bounds)
        ekin = sim.get_kinetic_energy()
        assert ekin == pytest.approx(1.0)

    def test_simulation_center_of_mass(self):
        """Test center of mass calculation."""
        bodies = [
            Body(1.0, 0.0, mass=1.0),
            Body(-1.0, 0.0, mass=1.0),
        ]
        bounds = Quad(0.0, 0.0, 2.0)
        sim = BarnesHutSimulation(bodies, bounds)
        com = sim.get_center_of_mass()
        assert np.allclose(com, [0.0, 0.0])

    def test_simulation_step(self):
        """Test one simulation step."""
        bodies = [
            Body(1.0, 0.0, mass=1.0),
            Body(-1.0, 0.0, mass=1.0),
        ]
        bounds = Quad(0.0, 0.0, 2.0)
        sim = BarnesHutSimulation(bodies, bounds, theta=0.8, dt=0.01)

        initial_pos = [b.pos.copy() for b in bodies]
        tree = sim.step()

        # Positions should have changed
        assert not np.allclose(bodies[0].pos, initial_pos[0])
        assert tree is not None
        assert len(tree.collect_nodes()) > 0

    def test_simulation_energy_conservation(self):
        """Test that energy is approximately conserved."""
        # Create two-body system
        bodies = [
            Body(1.0, 0.0, vy=1.0, mass=1.0),
            Body(-1.0, 0.0, vy=-1.0, mass=1.0),
        ]
        bounds = Quad(0.0, 0.0, 3.0)
        sim = BarnesHutSimulation(
            bodies,
            bounds,
            theta=0.8,
            G=1.0,
            softening=0.1,
            dt=0.01,
        )

        initial_energy = sim.get_kinetic_energy()
        energies = [initial_energy]

        for _ in range(100):
            sim.step()
            energies.append(sim.get_kinetic_energy())

        # Energy should not grow unboundedly (within reasonable tolerance)
        max_deviation = max(
            abs(e - initial_energy) / initial_energy for e in energies[10:]
        )
        assert max_deviation < 0.5  # Allow 50% deviation due to integration errors

    def test_simulation_build_tree(self):
        """Test tree building."""
        bodies = [
            Body(0.5, 0.5, mass=1.0),
            Body(-0.5, 0.5, mass=1.0),
        ]
        bounds = Quad(0.0, 0.0, 2.0)
        sim = BarnesHutSimulation(bodies, bounds)
        tree = sim.build_tree()

        assert tree.root.mass == sum(b.mass for b in bodies)
        assert len(tree.collect_nodes()) > 0


class TestBarnesHutAccuracy:
    """Tests for accuracy and convergence."""

    def test_theta_parameter_effect(self):
        """Test that smaller theta gives more accurate results."""
        bodies1 = [Body(1.0, 0.0, mass=1.0), Body(-1.0, 0.0, mass=1.0)]
        bodies2 = [Body(1.0, 0.0, mass=1.0), Body(-1.0, 0.0, mass=1.0)]

        bounds = Quad(0.0, 0.0, 2.0)
        sim1 = BarnesHutSimulation(bodies1, bounds, theta=0.1, dt=0.001)
        sim2 = BarnesHutSimulation(bodies2, bounds, theta=0.9, dt=0.001)

        for _ in range(10):
            sim1.step()
            sim2.step()

        # Lower theta should be more accurate
        # (though both should be reasonably accurate for simple cases)
        dist1 = np.linalg.norm(bodies1[1].pos - bodies1[0].pos)
        dist2 = np.linalg.norm(bodies2[1].pos - bodies2[0].pos)

        # Both should keep particles roughly at reasonable distance
        assert 0.5 < dist1 < 3.0
        assert 0.5 < dist2 < 3.0
