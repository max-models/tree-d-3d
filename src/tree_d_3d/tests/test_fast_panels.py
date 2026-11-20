"""The iterative panel solve: GMRES with fast multipole products."""

import numpy as np
import pytest
from simgeometry import Circle, Polygon

import tree_d_3d.electrodes as electrodes_module
from tree_d_3d import BoundaryIntegralElectrodes, Face, conductors_from_shape


def _geometry(panel):
    conductors = conductors_from_shape(Circle(midpoint=(0, 0), radius=1.0), 1.0, panel)
    box = Polygon(vertices=((1.5, -1), (3, -1), (3, 1), (1.5, 1)))
    return conductors + conductors_from_shape(box, -1.0, panel)


@pytest.mark.parametrize("far_points", [1, 2, 3])
@pytest.mark.parametrize("face", [None, Face("y", -2.0, "dirichlet", 0.5)])
def test_gmres_matches_the_direct_solve(far_points, face):
    conductors = _geometry(0.03)
    sources = np.array([[0.0, 1.2], [1.5, 0.0]])
    charges = np.array([0.3, -0.1])
    solved = {}
    for solver in ("direct", "gmres"):
        electrodes = BoundaryIntegralElectrodes(
            conductors, solver=solver, far_points=far_points, face=face, k_e=1.0
        )
        electrodes.solve(sources, charges)
        solved[solver] = electrodes
    direct, iterative = solved["direct"], solved["gmres"]
    assert direct.iterations == 0 and 10 < iterative.iterations < 200
    assert np.allclose(iterative.charge, direct.charge, rtol=0, atol=1e-7)
    points = np.array([[0.0, 4.0, 1.2], [-1.5, 0.0, 1.6]])
    for a, b in zip(iterative.probe(points), direct.probe(points)):
        assert np.allclose(a, b, rtol=1e-7, atol=1e-7)


def test_the_products_are_the_dense_matrix():
    conductors = _geometry(0.03)
    direct = BoundaryIntegralElectrodes(conductors, solver="direct", k_e=1.0)
    iterative = BoundaryIntegralElectrodes(conductors, solver="gmres", k_e=1.0)
    charge = np.random.default_rng(0).normal(size=direct.num_panels)
    product = direct._matrix @ charge
    assert (
        np.abs(iterative._apply(charge) - product).max() < 1e-10 * np.abs(product).max()
    )


def test_auto_iterates_above_the_limit(monkeypatch):
    conductors = _geometry(0.05)
    assert BoundaryIntegralElectrodes(conductors).solver == "direct"
    monkeypatch.setattr(electrodes_module, "DIRECT_LIMIT", 100)
    assert BoundaryIntegralElectrodes(conductors).solver == "gmres"
    # never with a period
    assert BoundaryIntegralElectrodes(conductors, period=10.0).solver == "direct"


def test_unconverged_solves_and_invalid_options_raise():
    conductors = _geometry(0.05)
    electrodes = BoundaryIntegralElectrodes(
        conductors, solver="gmres", tolerance=1e-14, max_iterations=3
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        electrodes.solve()
    with pytest.raises(ValueError, match="solver"):
        BoundaryIntegralElectrodes(conductors, solver="fmm")
    with pytest.raises(ValueError, match="period"):
        BoundaryIntegralElectrodes(conductors, solver="gmres", period=10.0)
    with pytest.raises(ValueError, match="tolerance"):
        BoundaryIntegralElectrodes(conductors, tolerance=0.0)


def test_tree_probes_match_the_direct_sum(monkeypatch):
    conductors = _geometry(0.03)
    electrodes = BoundaryIntegralElectrodes(conductors, k_e=1.0)
    rng = np.random.default_rng(2)
    sources = rng.uniform(-0.5, 0.5, (2, 300)) + [[4.0], [0.0]]
    charges = rng.normal(size=300)
    electrodes.solve(sources, charges)
    points = rng.uniform(-3.0, 4.0, (2, 2000))
    direct = electrodes.probe(points)
    monkeypatch.setattr(electrodes_module, "TREE_PAIRS", 0)
    tree = electrodes.probe(points)
    assert len(electrodes._trees) == 1  # the panels' tree, kept for the next probe
    for a, b in zip(tree, direct):
        assert np.abs(a - b).max() < 1e-11 * np.abs(b).max()
    # a second solve recharges the kept tree
    electrodes.solve(sources, -charges)
    again = electrodes.probe(points)
    monkeypatch.setattr(electrodes_module, "TREE_PAIRS", 10**18)
    reference = electrodes.probe(points)
    for a, b in zip(again, reference):
        assert np.abs(a - b).max() < 1e-11 * np.abs(b).max()
