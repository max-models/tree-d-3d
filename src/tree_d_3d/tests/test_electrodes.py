"""Tests for :mod:`tree_d_3d.electrodes`."""

from __future__ import annotations

import numpy as np
import pytest
from simgeometry import Aperture2D, Circle, Polygon

from tree_d_3d.electrodes import (
    BoundaryIntegralElectrodes,
    Conductor,
    conductors_from_shape,
    segment_field,
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
    # (3.2): the near panels enter exactly, so it is far below the plain midpoint rule's
    assert np.abs(phi).max() < 5e-3
    midpoint_rule = BoundaryIntegralElectrodes(
        [ring((0, 0), a, 200, 0.0)], k_e=1.0, near_field=0, far_points=1
    )
    midpoint_rule.solve(np.array([[d], [0.0]]), [q0])
    assert np.abs(midpoint_rule.potential(boundary)).max() > 10 * np.abs(phi).max()


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

    With exact near panels the error falls faster than second order in the panel
    length (the plain midpoint rule converges at first order).
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
    assert errors[1] < 5e-5
    assert errors[0] / errors[1] > 4.0


def test_an_open_polyline_becomes_an_open_conductor():
    from simgeometry import Polyline

    plate = Polyline([(0.0, 0.0), (1.0, 0.0)])
    (conductor,) = conductors_from_shape(plate, 1.0, 0.1)
    assert conductor.closed is False
    electrodes = BoundaryIntegralElectrodes([conductor], k_e=1.0)
    assert electrodes.num_panels == conductor.vertices.shape[1] - 1 == 10


def _segment_by_quadrature(point, start, end, pieces=200, nodes=40):
    """Composite Gauss--Legendre over ``pieces`` equal sub-segments (unit total charge)."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    along = ((np.arange(pieces)[:, None] + 0.5 * (x + 1.0)) / pieces).ravel()
    weight = np.tile(w, pieces) / (2.0 * pieces)
    source = start[:, None] + along * (end - start)[:, None]
    d = point[:, None] - source
    r2 = np.sum(d * d, axis=0)
    return np.sum(weight * -np.log(r2)), np.sum(weight * 2 * d / r2, axis=1)


@pytest.mark.parametrize(
    "point",
    [
        (0.7, 2.0),  # far
        (0.31, -0.19),  # next to the start
        (1.12, 0.41),  # next to the end
        (1.5, 0.7),  # on the line, beyond the end
        (0.75, 0.15),  # just off the middle
        (-3.0, 1.0),
    ],
)
def test_segment_field_matches_quadrature(point):
    start, end = np.array([0.3, -0.2]), np.array([1.1, 0.4])
    phi, field = segment_field(np.array(point)[:, None], start[:, None], end[:, None])
    phi_q, field_q = _segment_by_quadrature(np.array(point), start, end)
    assert phi[0] == pytest.approx(phi_q, abs=1e-10)
    np.testing.assert_allclose(field[:, 0], field_q, atol=1e-8)


def test_segment_field_on_the_segment_and_at_its_ends():
    start, end = np.array([[0.0], [0.0]]), np.array([[2.0], [0.0]])
    # at the midpoint: the analytic self term, no normal field
    phi, field = segment_field(np.array([[1.0], [0.0]]), start, end)
    assert phi[0] == pytest.approx(2.0 * (1.0 - np.log(1.0)))
    np.testing.assert_allclose(field[:, 0], 0.0, atol=1e-14)
    # at an end: finite potential  -(2/L) (L ln L - L)
    phi, _ = segment_field(start, start, end)
    assert phi[0] == pytest.approx(-2.0 * (np.log(2.0) - 1.0))


def test_image_force_and_capacitor_charge_converge_at_high_order():
    a, d, q0 = 1.5, 5.0, 1.0
    force = _three_image_force(a, d, q0, 1.0)
    q_exact = 1.0 / (2.0 * np.log(3.0))
    force_errors, charge_errors = [], []
    for n in (50, 100, 200):
        elec = BoundaryIntegralElectrodes([ring((0, 0), a, n, 0.0)], k_e=1.0)
        elec.solve(np.array([[d], [0.0]]), [q0])
        force_errors.append(abs(q0 * elec.force_on_sources()[1][0, 0] / force - 1))
        cap = BoundaryIntegralElectrodes(
            [ring((0, 0), 1.0, n, 1.0), ring((0, 0), 3.0, n, 0.0)], k_e=1.0
        )
        cap.solve()
        charge_errors.append(abs(cap.total_charge(0) / q_exact - 1))
    assert force_errors[-1] < 5e-4 and charge_errors[-1] < 1e-5
    # second order for the force, third for the induced charge
    assert all(e0 / e1 > 3.5 for e0, e1 in zip(force_errors, force_errors[1:]))
    assert all(e0 / e1 > 6.0 for e0, e1 in zip(charge_errors, charge_errors[1:]))


def test_probes_on_the_boundary_are_finite_and_close_to_the_potential():
    elec = BoundaryIntegralElectrodes([ring((0, 0), 0.5, 100, 1.0)], k_e=1.0)
    elec.solve()
    np.testing.assert_allclose(elec.potential(elec.midpoints), 1.0, atol=1e-12)
    np.testing.assert_allclose(elec.potential(elec.starts), 1.0, atol=5e-3)
    for midpoint_rule in ({"near_field": 0}, {"far_points": 1, "near_field": 0}):
        rough = BoundaryIntegralElectrodes(
            [ring((0, 0), 0.5, 100, 1.0)], k_e=1.0, **midpoint_rule
        )
        rough.solve()
        assert np.isfinite(rough.potential(rough.midpoints)).all()


def test_tabulated_field_interpolates_the_probe():
    elec = BoundaryIntegralElectrodes(
        [ring((0, 0), 0.5, 100, 1.0), ring((0, 0), 2.0, 100, 0.0)], k_e=1.0
    )
    elec.solve()
    table = elec.tabulate((-1.2, -1.2), (1.2, 1.2), (241, 241))
    assert table.potential.shape == (241, 241) and table.field.shape == (2, 241, 241)
    rng = np.random.default_rng(1)
    r = rng.uniform(0.7, 1.15, 200)
    angle = rng.uniform(0, 2 * np.pi, 200)
    points = np.stack((r * np.cos(angle), r * np.sin(angle)))
    phi, field = table(points)
    phi_p, field_p = elec.probe(points)
    assert table.contains(points).all()
    np.testing.assert_allclose(phi, phi_p, atol=1e-4)
    np.testing.assert_allclose(field, field_p, atol=1e-3)
    # outside the table: clamped to the edge, and reported by contains()
    outside = np.array([[5.0], [0.0]])
    assert not table.contains(outside)[0]
    edge = table(np.array([[1.2], [0.0]]))
    np.testing.assert_allclose(table(outside)[1], edge[1])
    with pytest.raises(ValueError, match="2 grid lines"):
        elec.tabulate((0, 0), (1, 1), (1, 5))


def test_invalid_near_field_settings_are_rejected():
    conductor = ring((0, 0), 0.5, 20, 1.0)
    with pytest.raises(ValueError, match="near_field"):
        BoundaryIntegralElectrodes([conductor], near_field=-1)
    with pytest.raises(ValueError, match="far_points"):
        BoundaryIntegralElectrodes([conductor], far_points=0)
