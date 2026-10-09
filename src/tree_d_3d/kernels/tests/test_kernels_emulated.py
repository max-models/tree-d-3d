"""The CUDA versions of the Barnes-Hut and FMM kernels against their Python versions.

Each CUDA kernel is compiled as C++ with the emulation stubs of
``test_cuda_kernels_emulated`` and run thread by thread, so the kernels are checked on
machines without a GPU (the threads run one after another: atomic adds are exact sums,
in another order than on the CPU).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from cunumpy.kernel_testing import emulate_cuda_kernel

import tree_d_3d.simulation as bh
from tree_d_3d.flat_tree import FlatQuadtree, _binomial_table
from tree_d_3d.kernels import KERNELS
from tree_d_3d.kernels.tests._emulation import COMPILER, _run
from tree_d_3d.quad import Quad

pytestmark = pytest.mark.skipif(COMPILER is None, reason="no C++ compiler")

ORDER = 6
THETA = 0.6
SOFTENING = 0.01


def _python(name):
    return KERNELS[name].host_kernel.kernel.python


def _views(arrays):
    return {name: array.ndim for name, array in arrays.items()}


def _cuda(tmp_path, name, arrays, scalars, n_threads, outputs):
    """Run the CUDA kernel ``name``: ``arrays`` and ``scalars`` in signature order."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    parameters = [p.name for p in KERNELS[name].cuda_kernel.signature]
    args = [f"{p}_view" if p in arrays else repr(scalars[p]) for p in parameters]
    call = (n_threads, f"{name}({', '.join(args)})")
    return _run(
        tmp_path, KERNELS[name].cuda_kernel, arrays, _views(arrays), call, outputs
    )


TREE = {"leaf_capacity": 4, "max_depth": 20}
ROOT = Quad(0.0, 0.0, 1.0 + 1e-9)


def _emulated(name):
    """The CUDA version of kernel ``name``, run by cunumpy's emulator, as a kernel."""
    kernel = KERNELS[name].cuda_kernel

    def run(*args, n_threads=None):
        # one thread per entry of the last axis of the first array (the marker count
        # of a component-major (2, N) array), as tree_d_3d's launch default
        emulate_cuda_kernel(kernel, *args, n_threads=n_threads or args[0].shape[-1])

    return run


def _with_emulated(kernels, *names):
    return SimpleNamespace(**{**vars(kernels), **{n: _emulated(n) for n in names}})


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(3)
    n = 80
    pos = rng.uniform(-1.0, 1.0, (2, n))  # component-major: pos[0] is x
    charge = rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)
    charge[::17] = 0.0  # some neutral markers
    flat = FlatQuadtree(pos, charge, ROOT, compiled=False, **TREE)
    tree = {
        name: getattr(flat, name)
        for name in ("perm", "node_center", "node_half", "node_start", "node_end")
    }
    tree["node_child"] = flat.node_child
    return {
        "pos": pos,
        "charge": charge,
        "tree": tree,
        "levels": flat.levels,
        "count": flat.node_count,
    }


def test_quadtree_build(monkeypatch, system):
    """The level-by-level build with the CUDA quadrant and scatter kernels."""
    python = bh._kernels(compiled=False)
    monkeypatch.setattr(
        bh,
        "_kernels",
        lambda compiled=True: _with_emulated(
            python, "quadtree_quadrants", "quadtree_scatter"
        ),
    )
    flat = FlatQuadtree(system["pos"], system["charge"], ROOT, compiled=False, **TREE)
    assert flat.node_count == system["count"]
    np.testing.assert_array_equal(flat.levels, system["levels"])
    assert len(flat.levels) > 4  # several levels were split
    for name, expected in system["tree"].items():
        np.testing.assert_array_equal(getattr(flat, name), expected, err_msg=name)


def _node_arrays(system):
    count, tree = system["count"], system["tree"]
    return {
        k: tree[k][:count].copy()
        for k in ("node_center", "node_half", "node_start", "node_end", "node_child")
    }


def test_quadtree_moments_and_evaluation(tmp_path, system):
    count = system["count"]
    nodes = _node_arrays(system)
    common = {
        "pos": system["pos"],
        "charge": system["charge"],
        "perm": system["tree"]["perm"],
    }
    moments = {
        "node_q": np.zeros(count),
        "node_qabs": np.zeros(count),
        "node_p": np.zeros((count, 2)),
        "node_s": np.zeros((count, 3)),
    }
    expected = {k: v.copy() for k, v in moments.items()}
    _python("quadtree_moments")(
        common["pos"],
        common["charge"],
        common["perm"],
        count,
        nodes["node_center"],
        nodes["node_start"],
        nodes["node_end"],
        *expected.values(),
    )
    arrays = {
        **common,
        **{k: nodes[k] for k in ("node_center", "node_start", "node_end")},
        **moments,
    }
    results = _cuda(
        tmp_path / "m",
        "quadtree_moments",
        arrays,
        {"node_count": count},
        count,
        list(moments),
    )
    for name, result in zip(moments, results):
        np.testing.assert_allclose(
            result, expected[name], rtol=1e-13, atol=1e-13, err_msg=name
        )

    n = system["pos"].shape[1]
    targets = {"targets": system["pos"], "exclude": np.arange(n, dtype=np.int64)}
    out = {
        "stack": np.zeros(256, dtype=np.int64),
        "phi": np.zeros(n),
        "field": np.zeros((2, n)),
        "status": np.zeros(1, dtype=np.int64),
    }
    reference = {k: v.copy() for k, v in out.items()}
    _python("evaluate_quadtree")(
        targets["targets"],
        targets["exclude"],
        *common.values(),
        *nodes.values(),
        *expected.values(),
        THETA,
        SOFTENING,
        *reference.values(),
    )
    arrays = {**targets, **common, **nodes, **expected, **out}
    phi, field, status = _cuda(
        tmp_path / "e",
        "evaluate_quadtree",
        arrays,
        {"theta": THETA, "softening": SOFTENING},
        n,
        ["phi", "field", "status"],
    )
    assert status[0] == 0
    np.testing.assert_allclose(phi, reference["phi"], rtol=1e-12)
    np.testing.assert_allclose(field, reference["field"], rtol=1e-12)


@pytest.mark.parametrize("name", ["direct_sum", "direct_log_sum"])
def test_direct_sums(tmp_path, system, name):
    n = system["pos"].shape[1]
    arrays = {
        "targets": system["pos"],
        "exclude": np.arange(n, dtype=np.int64),
        "pos": system["pos"],
        "charge": system["charge"],
        "phi": np.zeros(n),
        "field": np.zeros((2, n)),
        "status": np.zeros(1, dtype=np.int64),
    }
    reference = {k: v.copy() for k, v in arrays.items()}
    _python(name)(
        *(reference[k] for k in ("targets", "exclude", "pos", "charge")),
        SOFTENING,
        reference["phi"],
        reference["field"],
        reference["status"],
    )
    phi, field, status = _cuda(
        tmp_path, name, arrays, {"softening": SOFTENING}, n, ["phi", "field", "status"]
    )
    assert status[0] == 0
    np.testing.assert_allclose(phi, reference["phi"], rtol=1e-12)
    np.testing.assert_allclose(field, reference["field"], rtol=1e-12)


def test_coincident_charges_are_reported(tmp_path, system):
    pos = system["pos"].copy()
    pos[:, 5] = pos[:, 9]
    n = pos.shape[1]
    charge = np.ones(n)
    arrays = {
        "targets": pos,
        "exclude": np.full(n, -1, dtype=np.int64),
        "pos": pos,
        "charge": charge,
        "phi": np.zeros(n),
        "field": np.zeros((2, n)),
        "status": np.zeros(1, dtype=np.int64),
    }
    (status,) = _cuda(tmp_path, "direct_sum", arrays, {"softening": 0.0}, n, ["status"])
    assert status[0] > 0


def test_pair_energy(tmp_path, system):
    arrays = {
        "pos": system["pos"],
        "charge": system["charge"],
        "status": np.zeros(1, dtype=np.int64),
        "energy": np.zeros(1),
    }
    reference = {k: v.copy() for k, v in arrays.items()}
    _python("pair_energy")(
        reference["pos"],
        reference["charge"],
        SOFTENING,
        reference["status"],
        reference["energy"],
    )
    status, energy = _cuda(
        tmp_path,
        "pair_energy",
        arrays,
        {"softening": SOFTENING},
        system["pos"].shape[1],
        ["status", "energy"],
    )
    assert status[0] == 0
    np.testing.assert_allclose(energy, reference["energy"], rtol=1e-12)


@pytest.fixture(scope="module")
def log_tree(system):
    count = system["count"]
    nodes = _node_arrays(system)
    moments = {
        "node_qabs": np.zeros(count),
        "moment_re": np.zeros((count, ORDER + 1)),
        "moment_im": np.zeros((count, ORDER + 1)),
    }
    _python("log_quadtree_moments")(
        system["pos"],
        system["charge"],
        system["tree"]["perm"],
        count,
        nodes["node_center"],
        nodes["node_start"],
        nodes["node_end"],
        *moments.values(),
    )
    return nodes, moments


def test_log_moments_and_evaluation(tmp_path, system, log_tree):
    count = system["count"]
    nodes, expected = log_tree
    common = {
        "pos": system["pos"],
        "charge": system["charge"],
        "perm": system["tree"]["perm"],
    }
    moments = {k: np.zeros_like(v) for k, v in expected.items()}
    arrays = {
        **common,
        **{k: nodes[k] for k in ("node_center", "node_start", "node_end")},
        **moments,
    }
    results = _cuda(
        tmp_path / "m",
        "log_quadtree_moments",
        arrays,
        {"node_count": count},
        count,
        list(moments),
    )
    for name, result in zip(moments, results):
        np.testing.assert_allclose(
            result, expected[name], rtol=1e-12, atol=1e-13, err_msg=name
        )

    n = system["pos"].shape[1]
    targets = {"targets": system["pos"], "exclude": np.arange(n, dtype=np.int64)}
    out = {
        "stack": np.zeros(256, dtype=np.int64),
        "phi": np.zeros(n),
        "field": np.zeros((2, n)),
        "status": np.zeros(1, dtype=np.int64),
    }
    reference = {k: v.copy() for k, v in out.items()}
    _python("evaluate_log_quadtree")(
        targets["targets"],
        targets["exclude"],
        *common.values(),
        *nodes.values(),
        *expected.values(),
        THETA,
        SOFTENING,
        *reference.values(),
    )
    arrays = {**targets, **common, **nodes, **expected, **out}
    phi, field, status = _cuda(
        tmp_path / "e",
        "evaluate_log_quadtree",
        arrays,
        {"theta": THETA, "softening": SOFTENING},
        n,
        ["phi", "field", "status"],
    )
    assert status[0] == 0
    np.testing.assert_allclose(phi, reference["phi"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(field, reference["field"], rtol=1e-11, atol=1e-11)


def test_fmm_pipeline(tmp_path, system, log_tree):
    """Traversal, M2L, downward pass and evaluation, each against its Python version."""
    count = system["count"]
    nodes, moments = log_tree
    flat = FlatQuadtree(
        system["pos"],
        system["charge"],
        ROOT,
        compiled=False,
        kernel="log",
        order=ORDER,
        **TREE,
    )
    m2l_ref, p2p_ref, sizes_ref = flat._traverse(THETA)
    python = flat._kernels
    flat._kernels = _with_emulated(python, "fmm_traverse")
    m2l, p2p, sizes = flat._traverse(THETA)
    np.testing.assert_array_equal(sizes, sizes_ref)
    n_m2l, n_p2p = (int(c) for c in sizes_ref[:2])
    assert n_m2l and n_p2p
    np.testing.assert_array_equal(m2l[:n_m2l], m2l_ref[:n_m2l])
    np.testing.assert_array_equal(p2p[:n_p2p], p2p_ref[:n_p2p])
    reference = {"m2l": m2l_ref, "p2p": p2p_ref}

    binomial = _binomial_table(2 * ORDER + 1)
    local = {
        "local_re": np.zeros((count, ORDER + 1)),
        "local_im": np.zeros((count, ORDER + 1)),
    }
    scratch = {"inv_re": np.zeros(2 * ORDER + 2), "inv_im": np.zeros(2 * ORDER + 2)}
    expected = {k: v.copy() for k, v in local.items()}
    _python("fmm_m2l")(
        reference["m2l"],
        n_m2l,
        nodes["node_center"],
        moments["moment_re"],
        moments["moment_im"],
        binomial,
        *scratch.values(),
        *expected.values(),
    )
    arrays = {
        "pairs": reference["m2l"][: max(n_m2l, 1)].copy(),
        "node_center": nodes["node_center"],
        "moment_re": moments["moment_re"],
        "moment_im": moments["moment_im"],
        "binomial": binomial,
        **scratch,
        **local,
    }
    results = _cuda(
        tmp_path / "m", "fmm_m2l", arrays, {"count": n_m2l}, n_m2l, list(local)
    )
    for name, result in zip(local, results):
        np.testing.assert_allclose(
            result, expected[name], rtol=1e-11, atol=1e-11, err_msg=name
        )

    shifted = {k: v.copy() for k, v in expected.items()}
    power = {"power_re": np.zeros(ORDER + 1), "power_im": np.zeros(ORDER + 1)}
    emulated = {k: v.copy() for k, v in expected.items()}
    levels = system["levels"]
    for lo, hi in zip(levels[:-2], levels[1:-1]):
        common = (int(lo), int(hi), nodes["node_center"], nodes["node_child"], binomial)
        _python("fmm_downward")(*common, *power.values(), *shifted.values())
        _emulated("fmm_downward")(
            *common, *power.values(), *emulated.values(), n_threads=int(hi - lo)
        )
    for name in local:
        np.testing.assert_allclose(
            emulated[name], shifted[name], rtol=1e-11, atol=1e-11, err_msg=name
        )

    n = system["pos"].shape[1]
    out = {
        "phi": np.zeros(n),
        "field": np.zeros((2, n)),
        "status": np.zeros(1, dtype=np.int64),
    }
    ref_out = {k: v.copy() for k, v in out.items()}
    common = {
        "pos": system["pos"],
        "charge": system["charge"],
        "perm": system["tree"]["perm"],
    }
    _python("fmm_evaluate")(
        *common.values(),
        count,
        nodes["node_center"],
        nodes["node_start"],
        nodes["node_end"],
        nodes["node_child"],
        *shifted.values(),
        reference["p2p"],
        n_p2p,
        SOFTENING,
        1,
        *ref_out.values(),
    )
    arrays = {
        **common,
        **{
            k: nodes[k] for k in ("node_center", "node_start", "node_end", "node_child")
        },
        **shifted,
        "pairs": reference["p2p"][: max(n_p2p, 1)].copy(),
        **out,
    }
    phi, field, status = _cuda(
        tmp_path / "e",
        "fmm_evaluate",
        arrays,
        {
            "node_count": count,
            "count": n_p2p,
            "softening": SOFTENING,
            "with_potential": 1,
        },
        max(count, n_p2p),
        ["phi", "field", "status"],
    )
    assert status[0] == 0
    np.testing.assert_allclose(phi, ref_out["phi"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(field, ref_out["field"], rtol=1e-11, atol=1e-11)


def test_soften_pairs(tmp_path, system):
    n = system["pos"].shape[1]
    rng = np.random.default_rng(5)
    pairs = np.unique(np.sort(rng.integers(0, n, (40, 2)), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]].astype(np.int64)
    arrays = {
        "pos": system["pos"],
        "charge": system["charge"],
        "pairs": pairs,
        "phi": np.zeros(n),
        "field": np.zeros((2, n)),
        "status": np.zeros(1, dtype=np.int64),
    }
    reference = {k: v.copy() for k, v in arrays.items()}
    _python("soften_pairs")(
        reference["pos"],
        reference["charge"],
        reference["pairs"],
        0.01,
        1,
        reference["phi"],
        reference["field"],
        reference["status"],
    )
    phi, field, status = _cuda(
        tmp_path,
        "soften_pairs",
        arrays,
        {"eps2": 0.01, "with_potential": 1},
        len(pairs),
        ["phi", "field", "status"],
    )
    assert status[0] == 0
    np.testing.assert_allclose(phi, reference["phi"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(field, reference["field"], rtol=1e-12, atol=1e-14)
