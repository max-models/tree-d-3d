"""Panels with a period along y and with a Neumann or Dirichlet face."""

import numpy as np
import pytest
from simgeometry import Circle

from tree_d_3d import (
    BoundaryIntegralElectrodes,
    Conductor,
    Face,
    conductors_from_shape,
    periodic_log_kernel,
)


def test_periodic_kernel_is_the_row_sum():
    period = 1.3
    d = np.array([[0.3, -0.7, 2.0, 5.0], [0.2, 0.4, -0.1, 0.6]])
    m = np.arange(-4000, 4001)
    rel = d[:, :, None] - np.array([0.0, 1.0])[:, None, None] * m * period
    r2 = np.sum(rel**2, axis=0)
    # the truncated row's field: the far rows act as two half sheets, error ~ 1 / m
    field = np.sum(2.0 * rel / r2, axis=-1)
    phi, kernel_field = periodic_log_kernel(d, period)
    assert np.allclose(kernel_field, field, atol=3e-3)
    # the potential up to the renormalised constant
    differences = np.sum(-np.log(r2), axis=-1) - phi
    assert np.ptp(differences) < 2e-2


def test_periodic_kernel_next_to_a_charge_and_far_from_the_row():
    period = 2.0
    phi, field = periodic_log_kernel(np.array([[1e-7, 0.0], [1e-7, 1e-6]]), period)
    assert np.allclose(phi, -np.log([2e-14, 1e-12]), rtol=1e-12)
    assert np.allclose(field[:, 1], [0.0, 2e6], rtol=1e-8)
    # a charged sheet: 2 pi / P per unit charge, either side, also beyond the asymptote
    far = np.array([[-80.0, -8.0, 8.0, 80.0], [0.3, 0.3, 0.3, 0.3]])
    phi, field = periodic_log_kernel(far, period)
    assert np.allclose(field[0], np.pi * np.array([-1, -1, 1, 1]), rtol=1e-10)
    assert np.allclose(field[1], 0.0, atol=1e-10)
    assert np.allclose(phi, -np.pi * np.abs(far[0]) + 2 * np.log(np.pi), rtol=1e-10)


def _slab_reference(points, period, width, source, voltage):
    """A periodic row between a Dirichlet face at x = 0 (``voltage``) and a grounded x = width.

    The image series with period ``2 width`` converges only up to a linear term, fixed
    by the potential on the plate.
    """

    def series(points):
        phi = np.zeros(points.shape[1])
        field = np.zeros_like(points)
        for k in range(-30, 31):
            for x, sign in (
                (2 * k * width + source[0], 1.0),
                (2 * k * width - source[0], -1.0),
            ):
                p, e = periodic_log_kernel(points - [[x], [source[1]]], period)
                phi += sign * p
                field += sign * e
        return phi, field

    phi, field = series(points)
    slope = -series(np.array([[width], [0.0]]))[0][0] / width - voltage / width
    return phi + slope * points[0] + voltage, field - [[slope], [0.0]]


def test_slab_between_a_dirichlet_face_and_a_periodic_plate():
    period, width, voltage = 1.0, 1.0, 2.0
    source = np.array([0.3, 0.4])
    points = np.array([[0.1, 0.5, 0.25, 0.9], [0.0, 0.9, 0.35, 0.1]])
    exact_phi, exact_field = _slab_reference(points, period, width, source, voltage)
    errors = []
    for n in (25, 100):
        y = np.linspace(0.0, period, n + 1)
        plate = Conductor(np.stack((np.full_like(y, width), y)), 0.0, closed=False)
        electrodes = BoundaryIntegralElectrodes(
            [plate],
            period=period,
            face=Face("x", 0.0, "dirichlet", voltage),
            k_e=1.0,
        )
        electrodes.solve(source[:, None], np.array([1.0]))
        phi, field = electrodes.probe(points)
        errors.append(np.abs(field - exact_field).max())
        assert np.abs(phi - exact_phi).max() < 1e-5
        # Gauss: the source's flux toward the plate and the face's uniform field
        assert electrodes.total_charge(0) == pytest.approx(
            -source[0] / width - voltage * period / (4 * np.pi * width), abs=1e-6
        )
    assert errors[1] < errors[0] / 10 and errors[1] < 1e-6


def test_neumann_face_equals_the_explicit_mirror_image():
    (circle,) = conductors_from_shape(
        Circle(midpoint=(1.0, 0.3), radius=0.4), 1.0, 0.05
    )
    mirror = Conductor(circle.vertices[:, ::-1] * [[-1.0], [1.0]], 1.0)
    sources = np.array([[1.6, 0.9], [0.0, 0.4]])
    charges = np.array([0.7, -0.2])
    faced = BoundaryIntegralElectrodes([circle], face=Face("x", 0.0), k_e=1.0)
    faced.solve(sources, charges)
    both = BoundaryIntegralElectrodes([circle, mirror], k_e=1.0)
    both.solve(
        np.concatenate((sources, sources * [[-1.0], [1.0]]), axis=1),
        np.concatenate((charges, charges)),
    )
    points = np.array([[0.5, 2.0, 0.01], [1.0, -1.0, 0.2]])
    for a, b in zip(faced.probe(points), both.probe(points)):
        assert np.allclose(a, b, rtol=1e-11, atol=1e-11)
    assert faced.total_charge(0) == pytest.approx(both.total_charge(0), rel=1e-12)
    # the force on the sources includes their own images
    assert np.allclose(
        faced.force_on_sources()[1], both.force_on_sources()[1][:, :2], rtol=1e-11
    )
    # no normal field on the face
    on_face = np.array([[0.0, 0.0], [-0.5, 1.5]])
    assert np.allclose(faced.field(on_face)[0], 0.0, atol=1e-12)
    # panels alone: everything but the sources themselves, so a tree code can add them
    panels_phi, panels_field = faced.panel_field(points)
    alone_phi, alone_field = faced._sources_alone(points)
    total_phi, total_field = faced.probe(points)
    assert np.allclose(panels_phi + alone_phi, total_phi)
    assert np.allclose(panels_field + alone_field, total_field)


@pytest.mark.parametrize("face_potential", [0.0, 2.0])
def test_cylinder_over_a_dirichlet_face(face_potential):
    radius, height = 0.3, 1.0
    (circle,) = conductors_from_shape(
        Circle(midpoint=(0.0, height), radius=radius), 1.0, 2 * np.pi * radius / 400
    )
    electrodes = BoundaryIntegralElectrodes(
        [circle], face=Face("y", 0.0, "dirichlet", face_potential), k_e=1.0
    )
    electrodes.solve()
    # charge per length 2 pi eps_0 (V - V_face) / acosh(h / a), with k_e = 1 / (4 pi eps_0)
    exact = (1.0 - face_potential) / (2.0 * np.arccosh(height / radius))
    assert electrodes.total_charge(0) == pytest.approx(exact, rel=2e-5)
    face = np.array([[-0.5, 0.0, 3.0], [0.0, 0.0, 0.0]])
    assert np.allclose(electrodes.potential(face), face_potential, atol=1e-12)
    surface = np.array([[0.0], [height + radius]])
    assert electrodes.potential(surface)[0] == pytest.approx(1.0, abs=1e-4)


def test_periodic_pair_is_the_limit_of_explicit_stacks():
    period = 2.0
    plus = conductors_from_shape(Circle(midpoint=(0.5, 0.0), radius=0.25), 1.0, 0.03)
    minus = conductors_from_shape(Circle(midpoint=(-0.5, 0.3), radius=0.25), -1.0, 0.03)
    periodic = BoundaryIntegralElectrodes(plus + minus, period=period, k_e=1.0)
    periodic.solve()
    points = np.array([[0.0, 1.0, -1.2], [0.1, 0.7, -0.4]])
    phi, field = periodic.probe(points)
    errors = []
    for copies in (3, 12):
        stack = [
            Conductor(c.vertices + [[0.0], [k * period]], c.potential)
            for k in range(-copies, copies + 1)
            for c in plus + minus
        ]
        explicit = BoundaryIntegralElectrodes(stack, k_e=1.0)
        explicit.solve()
        errors.append(np.abs(explicit.field(points) - field).max())
    # a stack of neutral cells converges like 1 / copies
    assert errors[1] < errors[0] / 3
    # the periodic cell is neutral by symmetry
    assert periodic.total_charge(0) == pytest.approx(-periodic.total_charge(1))
    # copies one period apart see the same field
    shifted_phi, shifted_field = periodic.probe(points + [[0.0], [period]])
    assert np.allclose(shifted_phi, phi, atol=1e-10)
    assert np.allclose(shifted_field, field, atol=1e-10)


def test_periodic_sources_and_their_images():
    period = 1.0
    (circle,) = conductors_from_shape(
        Circle(midpoint=(1.0, 0.5), radius=0.2), 0.0, 0.02
    )
    electrodes = BoundaryIntegralElectrodes(
        [circle], period=period, face=Face("x", 0.0), k_e=1.0
    )
    sources = np.array([[0.5, 2.0], [0.2, 0.9]])
    charges = np.array([1.0, -0.5])
    electrodes.solve(sources, charges)
    phi, field = electrodes.force_on_sources()
    # moving a source by a period changes nothing
    electrodes.solve(sources + [[0.0], [period]], charges)
    shifted_phi, shifted_field = electrodes.force_on_sources()
    assert np.allclose(shifted_phi, phi, atol=1e-10)
    assert np.allclose(shifted_field, field, atol=1e-10)
    # the conductor stays grounded
    on_surface = np.array([[1.2], [0.5]])
    assert electrodes.potential(on_surface)[0] == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"period": 0.0}, "period"),
        ({"period": np.inf}, "period"),
        ({"period": 1.0, "face": Face("y", 0.0)}, "periodic y axis"),
        ({"face": Face("x", 1.0)}, "one side"),
    ],
)
def test_invalid_images(kwargs, match):
    (circle,) = conductors_from_shape(Circle(midpoint=(1.0, 0.0), radius=0.4), 1.0, 0.1)
    with pytest.raises(ValueError, match=match):
        BoundaryIntegralElectrodes([circle], **kwargs)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("z", 0.0), "axis"),
        (("x", 0.0, "robin"), "neumann"),
        (("x", 0.0, "neumann", 1.0), "no potential"),
    ],
)
def test_invalid_faces(args, match):
    with pytest.raises(ValueError, match=match):
        Face(*args)
