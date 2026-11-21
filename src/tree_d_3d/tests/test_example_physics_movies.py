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
