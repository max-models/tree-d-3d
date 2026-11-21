"""The Coulomb explosion and diocotron examples run, write their figures and match theory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_coulomb_explosion_example_writes_its_figures(tmp_path):
    example = _load("coulomb_explosion")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "gaussian_explosion",
            "uniform_expansion",
            "phase_space",
            "density_profile",
        )
    }


def test_uniform_disk_follows_the_self_similar_solution():
    example = _load("coulomb_explosion")
    run = example.explode(600, "uniform", 1.0, dt=0.01)
    exact = run["rms"][0] * example.stretch(run["t"], 2.0)
    assert np.max(np.abs(run["rms"] / exact - 1.0)) < 2e-3
    assert 0.9 < example.breaking_time("gaussian") < 1.2


def test_diocotron_example_writes_its_figures(tmp_path):
    example = _load("diocotron_instability")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in ("roll_up", "mode_growth", "growth_rates", "invariants")
    }


def test_diocotron_growth_rate_theory():
    example = _load("diocotron_instability")
    assert example.growth_rate(2) == 0.0
    assert example.growth_rate(8) == 0.0
    assert example.growth_rate(5) == pytest.approx(0.312, abs=1e-3)
    times = np.linspace(0.0, 20.0, 201)
    assert example.fit_rate(times, 1e-3 * np.exp(0.3 * times)) == pytest.approx(0.3)
    assert example.fit_rate(times, np.full(times.size, 1e-3)) == 0.0


def test_galaxy_collision_example_writes_its_figures(tmp_path):
    example = _load("galaxy_collision")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in ("collision", "core_orbit", "survival", "energy")
    }


def test_galaxy_cores_start_on_the_parabola():
    example = _load("galaxy_collision")
    state = example.initial_state(100)
    separation = np.hypot(*(state["pos"][:, 1] - state["pos"][:, 0]))
    assert separation == pytest.approx(example.START)
    speed = np.hypot(*(state["vel"][:, 1] - state["vel"][:, 0]))
    assert speed == pytest.approx(np.sqrt(2 * 2 * example.CORE / example.START))
    assert example.kepler_separation(
        np.array([state["t_peri"]]), state["t_peri"]
    ) == pytest.approx(example.PERICENTRE, rel=1e-3)


def test_cold_collapse_example_writes_its_figures(tmp_path):
    example = _load("cold_collapse")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in ("collapse", "fragmentation", "homologous_fall", "virial_ratio")
    }


def test_maclaurin_disk_pulls_linearly():
    from tree_d_3d import direct_field

    example = _load("cold_collapse")
    position, _, mass = example.maclaurin_disk(4000, 0.0, quiet=True)
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    for r in (0.3, 0.6):
        targets = r * np.stack([np.cos(angles), np.sin(angles)])
        _, field = direct_field(targets, position, mass, softening=0.02)
        pull = -np.mean(np.sum(field * targets, axis=0)) / r**2
        assert pull == pytest.approx(-(example.OMEGA0**2), rel=0.1)


def test_two_stream_example_writes_its_figures(tmp_path):
    example = _load("two_stream_instability")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in ("phase_space", "mapped_tree", "mode_growth", "growth_rates")
    }


def test_two_stream_periodic_field_matches_the_row_kernel():
    from tree_d_3d import periodic_log_kernel

    example = _load("two_stream_instability")
    rng = np.random.default_rng(0)
    ions = np.stack([rng.uniform(0, example.L, 60), rng.uniform(-2, 2, 60)])
    electrons = np.stack([rng.uniform(0, example.L, 120), rng.uniform(-2, 2, 120)])
    charge, ion_charges = -0.01, np.full(60, 0.02)
    field, _ = example.periodic_field(electrons, charge, ions, ion_charges, 0.0)
    sources = np.concatenate([electrons, ions], axis=1)
    charges = np.concatenate([np.full(120, charge), ion_charges])
    for i in range(0, 120, 17):
        others = np.arange(sources.shape[1]) != i
        d = electrons[:, i : i + 1] - sources[:, others]
        _, row = periodic_log_kernel(d[::-1], example.L)  # the period along x
        expected = np.sum(charges[others] * row[::-1], axis=1)
        np.testing.assert_allclose(field[:, i], expected, rtol=1e-5, atol=1e-8)
    assert example.growth_rate(np.sqrt(3 / 8)) == pytest.approx(1 / np.sqrt(8))
    assert example.growth_rate(1.2) == 0.0
