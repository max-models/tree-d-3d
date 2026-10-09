"""Run the Barnes-Hut/FMM solver on the CuPy backend without a GPU.

CuPy is replaced by cunumpy's fake CuPy and each CUDA kernel launch is run on the
CPU by ``cunumpy.kernel_testing.emulate_cuda_kernel``, so the whole CuPy code path
of ``tree_d_3d.flat_tree`` (array allocation, kernel
dispatch, host conversion) is checked against the NumPy backend.

Usage: python fake_cupy_check.py   (prints OK, exits 1 on a mismatch)
"""

import os
import sys

os.environ["CUNUMPY_FAKE_CUPY"] = "1"

import cunumpy as xp
import numpy as np
from cunumpy import _cuda_kernel
from cunumpy.kernel_testing import emulate_cuda_kernel

LAUNCHES = []


def _numpy(arg):
    return arg._a if type(arg).__name__ == "ndarray" and hasattr(arg, "_a") else arg


def _emulated_call(self, *args, n_threads=None, grid=None, block=None, **_):
    LAUNCHES.append(self.name)
    emulate_cuda_kernel(
        self, *map(_numpy, args), n_threads=n_threads, grid=grid, block=block
    )


_cuda_kernel.CudaKernel.__call__ = _emulated_call

from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.quad import Quad


def run(n=300):
    rng = np.random.default_rng(4)
    pos = rng.uniform(-0.9, 0.9, (2, n))  # component-major: pos[0] is x
    q = rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)
    exclude = np.arange(n)
    out = {}
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=4)
    out["bh"] = tree.evaluate(pos, exclude=exclude, theta=0.6, softening=0.01)
    out["direct"] = direct_field(pos, pos, q, softening=0.01, exclude=exclude)
    log = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=10)
    out["log_bh"] = log.evaluate(pos, exclude=exclude, theta=0.5)
    out["fmm"] = log.fmm()
    return {k: tuple(np.asarray(xp.to_numpy(a)) for a in v) for k, v in out.items()}


with xp.use_backend("numpy"):
    expected = run()
with xp.use_backend("cupy"):
    assert xp.get_backend() == "cupy"
    actual = run()

failed = False
for key in expected:
    for a, b in zip(expected[key], actual[key]):
        scale = np.abs(a).max()
        err = np.abs(a - b).max() / scale
        ok = err < 1e-10
        failed |= not ok
        print(f"{key:7s} rel. err {err:.2e} {'ok' if ok else 'MISMATCH'}")
print("CUDA launches:", sorted(set(LAUNCHES)))
print("FAILED" if failed else "OK")
sys.exit(1 if failed else 0)
