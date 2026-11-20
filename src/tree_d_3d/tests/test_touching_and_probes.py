"""Touching conductors, probes inside conductors and the surface field."""

import numpy as np
import pytest
from simgeometry import Circle, Polygon

from tree_d_3d import BoundaryIntegralElectrodes, Conductor, Face, conductors_from_shape


def _square(x0, y0, x1, y1):
    return Polygon(vertices=((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


def test_two_touching_squares_are_their_union():
    halves = conductors_from_shape(_square(0, 0, 1, 1), 1.0, 0.125)
    halves += conductors_from_shape(_square(1, 0, 2, 1), 1.0, 0.125)
    joined = BoundaryIntegralElectrodes(halves, k_e=1.0)
    union = BoundaryIntegralElectrodes(
        conductors_from_shape(_square(0, 0, 2, 1), 1.0, 0.125), k_e=1.0
    )
    assert joined.num_shared_panels == 16
    assert joined.num_panels == union.num_panels == 48
    joined.solve()
    union.solve()
    assert joined.total_charge().sum() == pytest.approx(
        union.total_charge(0), rel=1e-10
    )
    points = np.array([[-0.5, 1.0, 3.0], [0.5, 1.7, -1.0]])
    for a, b in zip(joined.probe(points), union.probe(points)):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12)


def test_a_plate_split_by_the_period_seam_is_one_plate():
    period = 1.0
    # one plate from y = 0.8 to 1.2, drawn as two halves that meet on the seams y = 0, 1
    lower = conductors_from_shape(_square(0.4, 0.0, 0.6, 0.2), 1.0, 0.05)
    upper = conductors_from_shape(_square(0.4, 0.8, 0.6, 1.0), 1.0, 0.05)
    split = BoundaryIntegralElectrodes(lower + upper, period=period, k_e=1.0)
    whole = BoundaryIntegralElectrodes(
        conductors_from_shape(_square(0.4, 0.8, 0.6, 1.2), 1.0, 0.05),
        period=period,
        k_e=1.0,
    )
    assert split.num_shared_panels == 8
    split.solve()
    whole.solve()
    assert split.total_charge().sum() == pytest.approx(whole.total_charge(0), rel=1e-7)
    points = np.array([[0.0, 1.0, 0.5], [0.3, 0.5, 0.5]])
    for a, b in zip(split.probe(points), whole.probe(points)):
        assert np.allclose(a, b, rtol=1e-6, atol=1e-7)


def test_touching_conductors_at_different_potentials_or_panels_are_rejected():
    left = conductors_from_shape(_square(0, 0, 1, 1), 1.0, 0.125)
    with pytest.raises(ValueError, match="different potentials"):
        BoundaryIntegralElectrodes(
            left + conductors_from_shape(_square(1, 0, 2, 1), 0.0, 0.125)
        )
    with pytest.raises(ValueError, match="panelled differently"):
        BoundaryIntegralElectrodes(
            left + conductors_from_shape(_square(1, 0, 2, 1), 1.0, 0.1)
        )
    # a single conductor whose only panels pair up
    there = np.array([[0.0, 1.0], [0.0, 0.0]])
    back = there[:, ::-1]
    with pytest.raises(ValueError, match="shared edge"):
        BoundaryIntegralElectrodes(
            [Conductor(there, 1.0, closed=False), Conductor(back, 1.0, closed=False)]
        )


def test_probes_inside_solid_conductors_get_the_exact_values():
    box = conductors_from_shape(_square(0, 0, 1, 1), 2.0, 0.1)
    electrodes = BoundaryIntegralElectrodes(
        box, period=3.0, face=Face("x", -1.0), k_e=1.0
    )
    electrodes.solve(np.array([[2.0], [0.5]]), np.array([1.0]))
    points = np.array([[0.5, 0.5, 0.2, 1.5], [0.5, 3.5, -2.8, 0.5]])
    assert list(electrodes.inside(points)) == [0, 0, 0, -1]
    phi, field = electrodes.probe(points)
    assert np.all(phi[:3] == 2.0) and np.all(field[:, :3] == 0.0)
    # outside, the panel sums
    assert np.allclose(electrodes.panel_field(points[:, 3:])[1] != 0.0, True)
    # a free charge inside the loop makes it a shell: its inside is not metal
    electrodes.solve(np.array([[0.5], [0.5]]), np.array([1.0]))
    assert np.all(electrodes.inside(points) == -1)


def test_shells_around_other_conductors_are_not_metal():
    inner = conductors_from_shape(Circle(midpoint=(0, 0), radius=0.5), 1.0, 0.05)
    outer = conductors_from_shape(Circle(midpoint=(0, 0), radius=2.0), 0.0, 0.1)
    electrodes = BoundaryIntegralElectrodes(inner + outer, k_e=1.0)
    electrodes.solve()
    points = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert list(electrodes.inside(points)) == [0, -1]
    phi, field = electrodes.probe(points)
    assert phi[0] == 1.0 and np.all(field[:, 0] == 0.0)
    # the coaxial profile between the two
    expected = np.log(2.0 / 1.0) / np.log(2.0 / 0.5)
    assert phi[1] == pytest.approx(expected, abs=1e-3)


def test_surface_field_is_gauss_on_a_charged_circle():
    radius, potential = 0.7, 3.0
    electrodes = BoundaryIntegralElectrodes(
        conductors_from_shape(Circle(midpoint=(0, 0), radius=radius), potential, 0.01),
        k_e=1.0,
    )
    electrodes.solve()
    surface = electrodes.surface_field()
    total = electrodes.total_charge(0)
    # E = 2 k_e Q / r at the surface of a line charge Q
    assert np.allclose(surface, 2.0 * total / radius, rtol=1e-4)
    # and the probe's field just outside approaches it
    outside = np.array([[radius * 1.02], [0.0]])
    assert electrodes.field(outside)[0, 0] == pytest.approx(
        2.0 * total / (radius * 1.02), rel=1e-3
    )
    with pytest.raises(RuntimeError):
        BoundaryIntegralElectrodes(
            conductors_from_shape(Circle(midpoint=(0, 0), radius=radius), 1.0, 0.1)
        ).surface_field()
