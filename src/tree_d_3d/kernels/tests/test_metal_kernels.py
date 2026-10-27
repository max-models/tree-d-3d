"""The Metal versions of the kernels on the GPU of an Apple-silicon Mac, against the CPU.

Each test calls a ``<name>_metal.py`` function and the Pyccel (or NumPy) version
with the same arrays. The GPU computes in float32, so the results agree to about
1e-5 of the largest value. Skipped without MLX and a Metal GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from cunumpy.kernels import metal_available

from tree_d_3d.kernels import KERNELS

pytestmark = pytest.mark.skipif(
    not metal_available(), reason="MLX with a Metal GPU is not available"
)

from tree_d_3d.flat_tree import FlatQuadtree
from tree_d_3d.kernels.direct_log_sum.direct_log_sum_metal import (
    direct_log_sum,
)
from tree_d_3d.kernels.direct_sum.direct_sum_metal import direct_sum
from tree_d_3d.kernels.evaluate_log_quadtree.evaluate_log_quadtree_metal import (
    evaluate_log_quadtree,
)
from tree_d_3d.kernels.evaluate_quadtree.evaluate_quadtree_metal import (
    evaluate_quadtree,
)
from tree_d_3d.kernels.pair_energy.pair_energy_metal import pair_energy
from tree_d_3d.quad import Quad

TOL = {"rtol": 0.0, "atol": 3e-5}
SOFTENING = 0.01


def _close(gpu, cpu):
    scale = max(float(np.abs(cpu).max()), 1.0)
    np.testing.assert_allclose(gpu / scale, cpu / scale, **TOL)


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(8)
    n = 3000
    # component-major (2, N)
    pos = np.clip(
        np.concatenate(
            (rng.normal(0, 0.05, (2, n // 2)), rng.uniform(-1, 1, (2, n // 2))),
            axis=1,
        ),
        -0.99,
        0.99,
    )
    charge = rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)
    return pos, charge


def _outputs(n):
    return np.zeros(n), np.zeros((2, n)), np.zeros(1, dtype=np.int64)


@pytest.mark.parametrize(
    ("name", "metal"), [("direct_sum", direct_sum), ("direct_log_sum", direct_log_sum)]
)
def test_direct_sums(system, name, metal):
    pos, charge = system
    exclude = np.arange(pos.shape[1], dtype=np.int64)
    results = []
    for function in (KERNELS[name], metal):
        phi, field, status = _outputs(pos.shape[1])
        function(pos, exclude, pos, charge, SOFTENING, phi, field, status)
        assert status[0] == 0
        results.append((phi, field))
    _close(results[1][0], results[0][0])
    _close(results[1][1], results[0][1])


@pytest.mark.parametrize(
    ("name", "metal"), [("direct_sum", direct_sum), ("direct_log_sum", direct_log_sum)]
)
def test_direct_sums_report_coincident_charges(name, metal):
    pos = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]).T  # (2, N)
    charge = np.ones(3)
    exclude = np.full(3, -1, dtype=np.int64)
    phi, field, status = _outputs(3)
    metal(pos, exclude, pos, charge, 0.0, phi, field, status)
    assert status[0] == 1  # the first failing target, as on the CPU


def test_pair_energy(system):
    pos, charge = system
    energies = []
    for function in (KERNELS["pair_energy"], pair_energy):
        status, energy = np.zeros(1, dtype=np.int64), np.zeros(1)
        function(pos, charge, SOFTENING, status, energy)
        assert status[0] == 0
        energies.append(energy[0])
    np.testing.assert_allclose(energies[1], energies[0], rtol=1e-4)


@pytest.mark.parametrize("kernel", ["coulomb", "log"])
def test_tree_evaluation(system, kernel):
    pos, charge = system
    tree = FlatQuadtree(pos, charge, Quad(0.0, 0.0, 1.0), kernel=kernel, order=10)
    exclude = np.arange(pos.shape[1], dtype=np.int64)
    common = (pos, exclude, tree.pos, tree.charge, tree.perm)
    nodes = (tree.node_center, tree.node_half, tree.node_start, tree.node_end)
    if kernel == "log":
        name, metal = "evaluate_log_quadtree", evaluate_log_quadtree
        moments = (tree.node_child, tree.node_qabs, tree.moment_re, tree.moment_im)
    else:
        name, metal = "evaluate_quadtree", evaluate_quadtree
        moments = (
            tree.node_child,
            tree.node_q,
            tree.node_qabs,
            tree.node_p,
            tree.node_s,
        )
    stack = np.empty(512, dtype=np.int64)
    results = []
    for function in (KERNELS[name], metal):
        phi, field, status = _outputs(pos.shape[1])
        function(*common, *nodes, *moments, 0.6, SOFTENING, stack, phi, field, status)
        assert status[0] == 0
        results.append((phi, field))
    _close(results[1][0], results[0][0])
    _close(results[1][1], results[0][1])
