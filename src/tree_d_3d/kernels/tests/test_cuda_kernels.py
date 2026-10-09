"""The CUDA versions of the Barnes-Hut and FMM kernels on a GPU, against the CPU.

Each test runs :class:`FlatQuadtree`, :func:`direct_field` or a kernel on CuPy
arrays (the CUDA versions) and on NumPy arrays (the Pyccel versions). Without CuPy
they are skipped; ``test_barnes_hut_kernels_emulated`` runs the same CUDA sources
on the CPU, and ``tree_d_3d/tests/test_cupy_path.py`` the solver's
CuPy code path with a fake CuPy.
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.kernels import KERNEL_NAMES, KERNELS
from tree_d_3d.quad import Quad

pytestmark = pytest.mark.skipif(
    not xp.cupy_available(), reason="CuPy not installed or not functional"
)

ROOT = Quad(0.0, 0.0, 1.0)
TOL = {"rtol": 1e-10, "atol": 1e-10}


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(8)
    n = 5000
    # component-major: pos[0] is x of every particle
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


def _on(backend, function):
    with xp.use_backend(backend):
        result = function()
        return tuple(np.asarray(xp.to_numpy(a)) for a in result)


def _close(gpu, cpu):
    scale = max(np.abs(cpu).max(), 1.0)
    np.testing.assert_allclose(gpu / scale, cpu / scale, **TOL)


def test_every_kernel_has_a_cuda_version():
    for name in KERNEL_NAMES:
        assert KERNELS[name].cuda_kernel is not None, name


@pytest.mark.parametrize("leaf_capacity", [1, 8])
def test_tree_build_is_identical(system, leaf_capacity):
    pos, charge = system

    def build():
        tree = FlatQuadtree(pos, charge, ROOT, leaf_capacity=leaf_capacity)
        return (
            tree.perm,
            tree.node_center,
            tree.node_start,
            tree.node_child,
            tree.node_q,
        )

    gpu, cpu = _on("cupy", build), _on("numpy", build)
    for a, b in zip(gpu[:4], cpu[:4]):
        np.testing.assert_array_equal(a, b)
    _close(gpu[4], cpu[4])  # moments: sums in another order


@pytest.mark.parametrize("kernel", ["coulomb", "log"])
def test_tree_and_direct_evaluation(system, kernel):
    pos, charge = system
    exclude = np.arange(pos.shape[1])

    def evaluate():
        tree = FlatQuadtree(pos, charge, ROOT, kernel=kernel, order=10)
        return (
            *tree.evaluate(pos, exclude=exclude, theta=0.6, softening=0.01),
            *direct_field(
                pos, pos, charge, softening=0.01, exclude=exclude, kernel=kernel
            ),
        )

    for a, b in zip(_on("cupy", evaluate), _on("numpy", evaluate)):
        _close(a, b)


@pytest.mark.parametrize("potential", [True, False])
def test_fmm(system, potential):
    pos, charge = system

    def fmm():
        tree = FlatQuadtree(pos, charge, ROOT, kernel="log", order=12, leaf_capacity=4)
        phi, field = tree.fmm(theta=0.5, softening=0.005, potential=potential)
        return phi, field, np.asarray(tree.fmm_pairs)

    gpu, cpu = _on("cupy", fmm), _on("numpy", fmm)
    np.testing.assert_array_equal(gpu[2], cpu[2])  # the same pairs, in another order
    _close(gpu[0], cpu[0])
    _close(gpu[1], cpu[1])


def test_pair_energy_and_soften_pairs(system):
    pos, charge = system
    n = pos.shape[1]
    rng = np.random.default_rng(1)
    pairs = np.unique(np.sort(rng.integers(0, n, (2000, 2)), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]].astype(np.int64)

    def run():
        p, q = xp.asarray(pos), xp.asarray(charge)
        status, energy = xp.zeros(1, dtype=np.int64), xp.zeros(1)
        KERNELS["pair_energy"](p, q, 0.01, status, energy)
        phi, field = xp.zeros(n), xp.zeros((2, n))
        soft_status = xp.zeros(1, dtype=np.int64)
        KERNELS["soften_pairs"](
            p,
            q,
            xp.asarray(pairs),
            0.01**2,
            1,
            phi,
            field,
            soft_status,
            n_threads=len(pairs),
        )
        return status, energy, phi, field, soft_status

    gpu, cpu = _on("cupy", run), _on("numpy", run)
    assert gpu[0][0] == cpu[0][0] == 0 and gpu[4][0] == cpu[4][0] == 0
    for a, b in zip(gpu[1:4], cpu[1:4]):
        _close(a, b)


def test_coincident_charges_are_reported_on_the_gpu():
    pos = np.array([[0.1, 0.1, 0.5], [0.1, 0.1, -0.2]])  # (2, N): two coincide
    with xp.use_backend("cupy"), pytest.raises(ValueError, match="Coincident"):
        direct_field(pos, pos, np.ones(3), exclude=np.arange(3))
