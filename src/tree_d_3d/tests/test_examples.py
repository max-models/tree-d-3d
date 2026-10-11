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
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_conserving_tree_code_example_writes_its_figures(tmp_path):
    example = _load("conserving_tree_code")
    output = example.main(["--quick", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "truncation_accuracy",
            "truncation_momentum",
            "partition_of_unity",
            "gradient_check",
            "energy_drift",
            "momentum_drift",
            "trajectories",
            "trajectories_gentle",
            "cloud",
            "orbits",
            "membership",
            "tree",
        )
    }


@pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_studies_example_writes_its_figures(tmp_path):
    example = _load("conserving_tree_code_studies")
    output = example.main(["--quick", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "kernels",
            "weights",
            "band_study",
            "cost_matched",
            "long_run",
            "error_map",
            "correction_term",
        )
    }


@pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_barnes_hut_tree_example_writes_its_figures(tmp_path):
    example = _load("barnes_hut_tree")
    output = example.main(["--quick", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "tree_boxes",
            "opening_criterion",
            "theta_tradeoff",
            "scaling",
            "cluster_merge",
        )
    }


@pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_fmm_scaling_example_writes_its_figures(tmp_path):
    example = _load("fmm_scaling")
    output = example.main(["--quick", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "order_accuracy",
            "theta_accuracy",
            "interaction_counts",
            "scaling",
            "interaction_list",
        )
    }


@pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_rational_far_field_example_writes_its_figures(tmp_path):
    example = _load("rational_far_field")
    output = example.main(["--quick", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "cell_accuracy",
            "cluster_size",
            "fit_validity",
            "hierarchical",
            "pair_counts",
        )
    }


@pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_implicit_push_example_writes_its_figures(tmp_path):
    example = _load("implicit_push")
    output = example.main(["--quick", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json" for stem in ("energy", "cost", "kinetic", "convergence")
    }


def test_boltzmann_tree_example_writes_its_figures(tmp_path):
    example = _load("boltzmann_tree")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in ("profiles", "iterations", "convergence", "leaves")
    }


@pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_plasma_sheath_example_writes_its_figures(tmp_path):
    example = _load("plasma_sheath")
    output = example.main(["--quick", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "setup",
            "density_profile",
            "potential_profile",
            "field_profile",
            "particle_loss",
            "sheath_formation",
        )
    }


def test_child_langmuir_diode_example_writes_its_figures(tmp_path):
    example = _load("child_langmuir_diode")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "current",
            "potential",
            "density",
            "phase_space",
            "stability",
            "emission",
        )
    }


def test_segment_beam_example_writes_its_figures(tmp_path):
    example = _load("segment_beam")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "segment_kernel",
            "trajectories",
            "convergence",
            "envelope_error",
            "ribbon",
        )
    }


def test_boundary_integral_electrodes_example_writes_its_figures(tmp_path):
    example = _load("boundary_integral_electrodes")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "grounded_circle",
            "image_force",
            "capacitor_profile",
            "capacitor_charge",
            "corner_potential",
        )
    }


def test_einzel_lens_example_writes_its_figures(tmp_path):
    example = _load("einzel_lens")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "geometry",
            "focusing_trajectories",
            "polarity_asymmetry",
            "energy_conservation",
            "focal_length",
        )
    }


def test_ion_extraction_example_writes_its_figures(tmp_path):
    example = _load("ion_extraction")
    output = example.main(["--smoke", "--output", str(tmp_path)])
    names = {p.name for p in output.glob("*.plotly.json")}
    assert names == {
        f"{stem}.plotly.json"
        for stem in (
            "geometry",
            "beam_snapshot",
            "envelope",
            "transmission_scaling",
            "beam_formation",
        )
    }
