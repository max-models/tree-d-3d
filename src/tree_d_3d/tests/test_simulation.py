"""Tests for the Barnes-Hut Vlasov-Poisson (Lorentz-force) solver."""

from __future__ import annotations

import numpy as np
import pytest

from tree_d_3d.bodies import ChargedBody
from tree_d_3d.flat_tree import FlatQuadtree
from tree_d_3d.quad import Quad
from tree_d_3d.reference import ReferenceQuadtree
from tree_d_3d.simulation import ElectrostaticSimulation


class TestChargedBody:
    """Tests for the ChargedBody class."""

    def test_init_defaults(self):
        body = ChargedBody(x=1.0, y=2.0)
        assert body.pos[0] == pytest.approx(1.0)
        assert body.pos[1] == pytest.approx(2.0)
        assert body.vel[0] == pytest.approx(0.0)
        assert body.vel[1] == pytest.approx(0.0)
        assert body.mass == pytest.approx(1.0)
        assert body.charge == pytest.approx(1.0)
        assert np.allclose(body.force, [0.0, 0.0])

    def test_init_explicit(self):
        body = ChargedBody(x=3.0, y=-1.0, vx=0.5, vy=-0.5, mass=2.0, charge=-1.6e-19)
        assert body.vel[0] == pytest.approx(0.5)
        assert body.charge == pytest.approx(-1.6e-19)
        assert body.mass == pytest.approx(2.0)

    def test_reset_force(self):
        body = ChargedBody(0.0, 0.0)
        body.add_force(3.0, -4.0)
        assert not np.allclose(body.force, [0.0, 0.0])
        body.reset_force()
        assert np.allclose(body.force, [0.0, 0.0])

    def test_add_force_accumulates(self):
        body = ChargedBody(0.0, 0.0)
        body.add_force(1.0, 2.0)
        body.add_force(-0.5, 0.5)
        assert body.force[0] == pytest.approx(0.5)
        assert body.force[1] == pytest.approx(2.5)


class TestReferenceQuadtree:
    """Tests for the ReferenceQuadtree."""

    def test_insert_single_body(self):
        tree = ReferenceQuadtree(Quad(0.0, 0.0, 1.0))
        body = ChargedBody(0.0, 0.0, charge=2.0)
        tree.insert(body)
        assert tree.root.body is body
        assert tree.root.total_charge == pytest.approx(2.0)

    def test_insert_two_same_sign_charges(self):
        """Two positive charges: total charge = sum, COM between them."""
        tree = ReferenceQuadtree(Quad(0.0, 0.0, 2.0))
        b1 = ChargedBody(1.0, 0.0, charge=1.0)
        b2 = ChargedBody(-1.0, 0.0, charge=1.0)
        tree.insert(b1)
        tree.insert(b2)
        assert tree.root.total_charge == pytest.approx(2.0)
        assert np.allclose(tree.root.center_of_charge, [0.0, 0.0])

    def test_insert_opposite_charges(self):
        """Opposite charges: total charge → 0, centre is geometric midpoint."""
        tree = ReferenceQuadtree(Quad(0.0, 0.0, 2.0))
        b1 = ChargedBody(1.0, 0.0, charge=1.0)
        b2 = ChargedBody(-1.0, 0.0, charge=-1.0)
        tree.insert(b1)
        tree.insert(b2)
        assert tree.root.total_charge == pytest.approx(0.0)

    def test_repulsion_same_sign(self):
        """Like charges should push each other apart."""
        tree = ReferenceQuadtree(Quad(0.0, 0.0, 2.0))
        b1 = ChargedBody(0.5, 0.0, charge=1.0)
        b2 = ChargedBody(-0.5, 0.0, charge=1.0)
        tree.insert(b1)
        tree.insert(b2)

        tree.compute_lorentz_force_on(b1, theta=0.0, k_e=1.0, softening=0.01)
        # b1 is to the right of b2: repulsion pushes b1 further right (+x)
        assert b1.force[0] > 0.0
        assert abs(b1.force[1]) < 1e-12

    def test_attraction_opposite_sign(self):
        """Opposite charges should attract each other."""
        tree = ReferenceQuadtree(Quad(0.0, 0.0, 2.0))
        b1 = ChargedBody(0.5, 0.0, charge=1.0)
        b2 = ChargedBody(-0.5, 0.0, charge=-1.0)
        tree.insert(b1)
        tree.insert(b2)

        tree.compute_lorentz_force_on(b1, theta=0.0, k_e=1.0, softening=0.01)
        # b1 is to the right of b2: attraction pulls b1 left (−x)
        assert b1.force[0] < 0.0
        assert abs(b1.force[1]) < 1e-12

    def test_force_magnitude_coulomb_law(self):
        """Force magnitude should follow Coulomb's law F = k_e q1 q2 / r^2."""
        tree = ReferenceQuadtree(Quad(0.0, 0.0, 5.0))
        separation = 2.0
        b1 = ChargedBody(separation / 2, 0.0, charge=1.0)
        b2 = ChargedBody(-separation / 2, 0.0, charge=1.0)
        tree.insert(b1)
        tree.insert(b2)

        k_e = 1.0
        tree.compute_lorentz_force_on(b1, theta=0.0, k_e=k_e, softening=0.0)
        expected = k_e * 1.0 * 1.0 / (separation**2)
        assert b1.force[0] == pytest.approx(expected, rel=1e-6)

    def test_magnetic_force_perpendicular(self):
        """Magnetic force on a particle moving in x under B_z should be in y."""
        tree = ReferenceQuadtree(Quad(0.0, 0.0, 2.0))
        # Isolated body with no other charges: only magnetic force
        body = ChargedBody(0.0, 0.0, vx=1.0, vy=0.0, charge=1.0)
        tree.compute_lorentz_force_on(body, theta=0.5, k_e=1.0, softening=0.1, B_z=1.0)
        # F_mag = q * v × B = (0, -q*vx*Bz) = (0, -1)
        assert body.force[0] == pytest.approx(0.0, abs=1e-12)
        assert body.force[1] == pytest.approx(-1.0)

    def test_collect_nodes_nonempty(self):
        tree = ReferenceQuadtree(Quad(0.0, 0.0, 2.0))
        for x in [0.5, -0.5, 0.0]:
            tree.insert(ChargedBody(x, 0.0, charge=1.0))
        assert len(tree.collect_nodes()) > 0


class TestElectrostaticSimulation:
    """Tests for the full Vlasov-Poisson Barnes-Hut simulation."""

    def _make_sim(self, **kwargs) -> ElectrostaticSimulation:
        bodies = [
            ChargedBody(1.0, 0.0, charge=1.0, mass=1.0),
            ChargedBody(-1.0, 0.0, charge=1.0, mass=1.0),
        ]
        bounds = Quad(0.0, 0.0, 3.0)
        return ElectrostaticSimulation(
            bodies=bodies,
            bounds=bounds,
            k_e=1.0,
            softening=0.1,
            dt=0.01,
            **kwargs,
        )

    def test_init(self):
        sim = self._make_sim()
        assert len(sim.bodies) == 2
        assert sim.theta == pytest.approx(0.5)
        assert sim.B_z == pytest.approx(0.0)

    def test_total_charge(self):
        sim = self._make_sim()
        assert sim.get_total_charge() == pytest.approx(2.0)

    def test_kinetic_energy_zero_initial(self):
        sim = self._make_sim()
        assert sim.get_kinetic_energy() == pytest.approx(0.0)

    def test_kinetic_energy_nonzero(self):
        bodies = [ChargedBody(0.0, 0.0, vx=1.0, vy=1.0, mass=2.0, charge=1.0)]
        sim = ElectrostaticSimulation(
            bodies=bodies,
            bounds=Quad(0.0, 0.0, 2.0),
            k_e=1.0,
        )
        assert sim.get_kinetic_energy() == pytest.approx(2.0)

    def test_center_of_mass(self):
        sim = self._make_sim()
        com = sim.get_center_of_mass()
        assert np.allclose(com, [0.0, 0.0])

    def test_momentum_zero(self):
        sim = self._make_sim()
        p = sim.get_momentum()
        assert np.allclose(p, [0.0, 0.0])

    def test_step_moves_particles(self):
        """A step with like charges should push particles apart."""
        sim = self._make_sim()
        initial_pos = [b.pos.copy() for b in sim.bodies]
        sim.step()
        # Particles must have moved
        for b, p0 in zip(sim.bodies, initial_pos):
            assert not np.allclose(b.pos, p0)

    def test_like_charges_repel(self):
        """Like-charge particles should increase their separation over time."""
        sim = self._make_sim()
        initial_sep = np.linalg.norm(sim.bodies[1].pos - sim.bodies[0].pos)
        for _ in range(50):
            sim.step()
        final_sep = np.linalg.norm(sim.bodies[1].pos - sim.bodies[0].pos)
        assert final_sep > initial_sep

    def test_opposite_charges_attract(self):
        """Opposite-charge particles should decrease their separation."""
        bodies = [
            ChargedBody(1.0, 0.0, charge=1.0, mass=1.0),
            ChargedBody(-1.0, 0.0, charge=-1.0, mass=1.0),
        ]
        sim = ElectrostaticSimulation(
            bodies=bodies,
            bounds=Quad(0.0, 0.0, 5.0),
            k_e=1.0,
            softening=0.1,
            dt=0.005,
        )
        initial_sep = np.linalg.norm(bodies[1].pos - bodies[0].pos)
        for _ in range(30):
            sim.step()
        final_sep = np.linalg.norm(bodies[1].pos - bodies[0].pos)
        assert final_sep < initial_sep

    def test_magnetic_circular_motion(self):
        """A charged particle in a uniform B field should orbit (constant speed)."""
        body = ChargedBody(1.0, 0.0, vx=0.0, vy=1.0, mass=1.0, charge=1.0)
        sim = ElectrostaticSimulation(
            bodies=[body],
            bounds=Quad(0.0, 0.0, 5.0),
            k_e=1.0,
            softening=0.1,
            dt=0.005,
            B_z=1.0,
        )
        initial_speed = np.linalg.norm(body.vel)
        for _ in range(200):
            sim.step()
        final_speed = np.linalg.norm(body.vel)
        # Speed should be conserved (B does no work)
        assert final_speed == pytest.approx(initial_speed, rel=0.05)

    def test_build_tree_returns_tree(self):
        sim = self._make_sim()
        tree = sim.build_tree()
        assert isinstance(tree, FlatQuadtree)
        assert tree.root.total_charge == pytest.approx(2.0)

    def test_step_returns_tree(self):
        sim = self._make_sim(method="tree")
        tree = sim.step()
        assert isinstance(tree, FlatQuadtree)

    def test_small_systems_sum_directly(self):
        sim = self._make_sim()
        assert sim.step() is None

    def test_softened_source_small_amplitude_oscillation(self):
        """The core of a softened source is harmonic, not a bulk plasma mode."""
        softening = 0.2
        amplitude = 1e-4
        omega = np.sqrt(1 / softening**3)
        period = 2 * np.pi / omega
        electron = ChargedBody(amplitude, 0.0, charge=-1.0, mass=1.0)
        ion = ChargedBody(0.0, 0.0, charge=1.0, fixed=True)
        sim = ElectrostaticSimulation(
            bodies=[electron, ion],
            bounds=Quad(0.0, 0.0, 2.0),
            k_e=1.0,
            softening=softening,
            dt=period / 400,
        )
        for _ in range(400):
            sim.step()
            expected = amplitude * np.cos(omega * sim.current_time)
            assert abs(electron.pos[0] - expected) < 1e-4 * amplitude
        assert np.array_equal(ion.pos, [0, 0])
