"""The example scripts run and write their figures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib not installed"
)
def test_conserving_tree_code_example_writes_its_figures(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    example = _load("conserving_tree_code")
    output = example.main(["--quick", "--output", str(tmp_path)])
    assert {p.name for p in output.glob("*.gif")} == {
        "cloud.gif",
        "orbits.gif",
        "membership.gif",
    }
    names = {p.name for p in output.glob("*.png")}
    assert names == {
        "truncation_accuracy.png",
        "truncation_momentum.png",
        "partition_of_unity.png",
        "gradient_check.png",
        "energy_drift.png",
        "momentum_drift.png",
        "trajectories.png",
        "trajectories_gentle.png",
    }


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib not installed"
)
def test_studies_example_writes_its_figures(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    example = _load("conserving_tree_code_studies")
    output = example.main(["--quick", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.png")}
    assert names == {
        "kernels.png",
        "weights.png",
        "band_study.png",
        "cost_matched.png",
        "long_run.png",
        "error_map.png",
        "correction_term.png",
    }
