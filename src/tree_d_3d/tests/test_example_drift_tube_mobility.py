"""The drift_tube_mobility example runs and writes its figures."""

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


def test_drift_tube_mobility_example_writes_its_figures(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("plasmacoll")
    example = _load("drift_tube_mobility")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "geometry",
            "arrival",
            "mobility",
            "diffusion",
            "resolving_power",
        )
    }
