"""The CuPy code path of the Barnes-Hut/FMM solver, without a GPU.

``fake_cupy_check.py`` runs the solver with cunumpy's fake CuPy and
emulated CUDA kernels; it needs a fresh process because the fake CuPy must be
installed before cunumpy is imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from cunumpy.kernel_testing import emulation_compiler

SCRIPT = Path(__file__).with_name("fake_cupy_check.py")


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
def test_cupy_backend_matches_numpy_with_emulated_cuda_kernels():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "evaluate_quadtree" in result.stdout and "fmm_m2l" in result.stdout
    assert "absorb_polygons_2d" in result.stdout
