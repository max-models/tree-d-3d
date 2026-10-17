"""Tests for :mod:`tree_d_3d.electrodes`."""

from __future__ import annotations

import numpy as np
import pytest
from simgeometry import Aperture2D, Circle, Polygon

from tree_d_3d.electrodes import (
    BoundaryIntegralElectrodes,
    Conductor,
    conductors_from_shape,
)


def ring(center, radius: float, n: int, potential: float) -> Conductor:
    """A circular conductor of ``n`` panels, from a simgeometry ``Circle``."""
    circle = Circle(midpoint=center, radius=radius)
    (conductor,) = conductors_from_shape(circle, potential, 2 * np.pi * radius / n)
    return conductor


def _three_image_force(a: float, d: float, q0: float, k_e: float) -> float:
    """Force on a line charge ``q0`` at distance ``d`` from a grounded circle of radius ``a``."""

    def field_from(q, z, at):
        dx = at - z
        return 2 * k_e * q * dx / dx**2

    z_im, q_im = a * a / d, -q0
    q_c = -q0 * np.log(d / a) / np.log(a)
    return q0 * (field_from(q_im, z_im, d) + field_from(q_c, 0.0, d))


def test_grounded_circle_force_matches_three_image_closed_form():
    a, d, q0, k_e = 1.5, 5.0, 1.0, 1.0
    elec = BoundaryIntegralElectrodes([ring((0, 0), a, 400, 0.0)], k_e=k_e)
    elec.solve(np.array([[d], [0.0]]), [q0])
    _, field = elec.force_on_sources()
    f_bem = q0 * field[0, 0]
    f_exact = _three_image_force(a, d, q0, k_e)
    assert f_bem == pytest.approx(f_exact, rel=1e-3)


def test_grounded_circle_boundary_potential_is_small():
    a, d, q0 = 1.5, 5.0, 1.0
    elec = BoundaryIntegralElectrodes([ring((0, 0), a, 200, 0.0)], k_e=1.0)
    elec.solve(np.array([[d], [0.0]]), [q0])
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    boundary = np.stack((a * np.cos(theta), a * np.sin(theta)))
    phi = elec.potential(boundary)
    # not at the collocation points, so not machine zero, but small vs. the source scale
    assert np.abs(phi).max() < 0.2


def test_cylindrical_capacitor_matches_exact_profile_and_charge():
    a, b, v1, v2 = 1.0, 3.0, 1.0, 0.0
    cap = BoundaryIntegralElectrodes(
        [
            ring((0, 0), a, 200, v1),
            ring((0, 0), b, 200, v2),
        ]
    )
    cap.solve()
    r = np.linspace(a + 0.1, b - 0.1, 20)
    phi = cap.potential(np.stack((r, np.zeros_like(r))))
    exact = v1 + (v2 - v1) * np.log(r / a) / np.log(b / a)
    assert phi == pytest.approx(exact, abs=5e-3)

    q_exact = -(v2 - v1) / (2 * cap.k_e * np.log(b / a))
    assert cap.total_charge(0) == pytest.approx(q_exact, rel=5e-3)
    assert cap.total_charge() == pytest.approx(
        [cap.total_charge(0), cap.total_charge(1)]
    )


def test_two_conductors_can_float_independently():
    # a dipole-like pair: equal and opposite potentials, no external charge
    left = ring((-2, 0), 0.5, 80, 1.0)
    right = ring((2, 0), 0.5, 80, -1.0)
    elec = BoundaryIntegralElectrodes([left, right])
    elec.solve()
    # by symmetry the induced charges are equal and opposite
    charges = elec.total_charge()
    assert charges[0] == pytest.approx(-charges[1], rel=1e-6)


def test_conductor_needs_at_least_two_vertices():
    with pytest.raises(ValueError, match="2"):
        BoundaryIntegralElectrodes([Conductor(np.zeros((2, 1)), 0.0)])


def test_open_conductor_has_one_fewer_panel_than_vertices():
    (v,) = Circle(midpoint=(0, 0), radius=1.0).outlines(2 * np.pi / 10)
    closed = BoundaryIntegralElectrodes([Conductor(v, 0.0, closed=True)])
    open_ = BoundaryIntegralElectrodes([Conductor(v, 0.0, closed=False)])
    assert closed.num_panels == 10
    assert open_.num_panels == 9


def test_conductors_from_shape_has_one_closed_conductor_per_loop():
    aperture = Aperture2D(
        axis=0,
        position=0.0,
        vertices=((-0.1, 0.5), (0.1, 0.5), (0.1, 3.0), (-0.1, 3.0)),
    )
    lower, upper = conductors_from_shape(aperture, 2.0, 0.1)
    assert lower.closed and upper.closed
    assert lower.potential == upper.potential == 2.0
    for conductor in (lower, upper):
        loop = np.concatenate((conductor.vertices, conductor.vertices[:, :1]), axis=1)
        assert np.max(np.hypot(*np.diff(loop, axis=1))) <= 0.1 + 1e-12
    assert np.all(upper.vertices[1] >= 0.5) and np.all(lower.vertices[1] <= -0.5)
    assert ring((0, 0), 1.0, 37, 0.0).vertices.shape == (2, 37)


def test_graded_panels_are_short_near_the_aperture_tip():
    aperture = Aperture2D(
        axis=0,
        position=0.0,
        vertices=((-0.1, 0.5), (0.1, 0.5), (0.1, 3.0), (-0.1, 3.0)),
    )
    _, upper = conductors_from_shape(
        aperture, 0.0, lambda p: 0.02 + 0.1 * (np.abs(p[1]) - 0.5)
    )
    electrodes = BoundaryIntegralElectrodes([upper])
    tip = electrodes.midpoints[1] < 0.55
    far = electrodes.midpoints[1] > 2.5
    assert electrodes.lengths[tip].max() < electrodes.lengths[far].min()


def test_polygon_conductor_interior_converges_to_equipotential_and_field_free():
    """An L-shaped conductor from a simgeometry Polygon screens its interior.

    The flat-panel midpoint rule converges at first order in the panel length.
    """
    l_shape = Polygon(
        vertices=np.array([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], dtype=float)
    )
    inside = np.array([[0.5, 1.5, 0.5, 0.4], [0.5, 0.5, 1.5, 0.4]])
    assert l_shape.contains_points(inside).all()
    errors = []
    for panel in (0.04, 0.02):
        electrodes = BoundaryIntegralElectrodes(
            conductors_from_shape(l_shape, 1.0, panel), k_e=1.0
        )
        electrodes.solve()
        phi, field = electrodes.probe(inside)
        errors.append(max(np.abs(phi - 1.0).max(), np.abs(field).max()))
    assert errors[1] < 5e-3
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.1)
