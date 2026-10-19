"""The charge-exchange grid-erosion example runs and writes its figures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


def test_cex_grid_erosion_example_writes_its_figures(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("plasmacoll")
    spec = importlib.util.spec_from_file_location(
        "cex_grid_erosion", EXAMPLES / "cex_grid_erosion.py"
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    sys.modules["cex_grid_erosion"] = example
    spec.loader.exec_module(example)
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "geometry",
            "trajectories",
            "impact_map",
            "impact_distributions",
            "erosion",
            "pressure_scan",
        )
    }
