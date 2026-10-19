"""The multipactor example runs and writes its figures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "multipactor.py"


def _load():
    spec = importlib.util.spec_from_file_location("multipactor", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["multipactor"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_multipactor_example_writes_its_figures(tmp_path):
    example = _load()
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "geometry",
            "resonant_trajectory",
            "secondary_yield",
            "electron_growth",
            "susceptibility_chart",
        )
    }


def test_vaughan_yield_has_its_peak_and_threshold():
    pytest.importorskip("plotly")
    example = _load()
    assert example.vaughan_yield(example.E_MAX, 0.0) == pytest.approx(example.DELTA_MAX)
    assert example.vaughan_yield(example.E_THRESHOLD - 1.0, 0.0) == 0.0
    assert example.E_THRESHOLD < example.first_crossover() < example.E_MAX
