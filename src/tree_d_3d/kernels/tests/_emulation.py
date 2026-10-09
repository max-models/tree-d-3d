"""Run a CUDA kernel body on the CPU: compiled as C++ with the CUDA builtins stubbed.

:func:`_run` calls the kernel once per thread index, serially (atomics are then plain
additions). This checks the indexing and arithmetic, not the concurrency.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cunumpy as xp
import numpy as np

import tree_d_3d

COMPILER = shutil.which("c++")
# the directory containing the tree_d_3d package, for #include "tree_d_3d/..."
SOURCE_ROOT = Path(tree_d_3d.__file__).resolve().parents[1]

STUBS = r"""
#include <cmath>
#include <cstdio>
using std::floor;
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
struct dim3_ { unsigned x, y, z; };
static dim3_ threadIdx, blockDim, blockIdx, gridDim;
#define __shared__
inline void __syncthreads() {} // every emulated block has one thread
double block_field[6144];
inline double atomicAdd(double* p, double v) { double o = *p; *p += v; return o; }
inline float atomicAdd(float* p, float v) { float o = *p; *p += v; return o; }
inline int atomicAdd(int* p, int v) { int o = *p; *p += v; return o; }
inline unsigned atomicAdd(unsigned* p, unsigned v) { unsigned o = *p; *p += v; return o; }
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { unsigned long long o = *p; *p += v; return o; }
inline double __longlong_as_double(long long) { return 0; }
inline long long __double_as_longlong(double) { return 0; }
inline unsigned long long atomicCAS(unsigned long long* p, unsigned long long compare, unsigned long long v) { unsigned long long o = *p; if (o == compare) *p = v; return o; }
"""


#: The C type of each array dtype: masks are ``char`` arrays on the GPU.
C_TYPES = {"f": "double", "b": "char", "i": "long long"}


def _c_array(name, values):
    values = np.asarray(values)
    if values.dtype.kind == "f":
        flat = ", ".join(
            "NAN" if np.isnan(v) else repr(float(v)) for v in np.ravel(values)
        )
    else:
        flat = ", ".join(str(int(v)) for v in np.ravel(values))
    return f"static {C_TYPES[values.dtype.kind]} {name}[] = {{{flat}}};\n"


def _view(name, array, ndim):
    shape = ", ".join(str(n) for n in array.shape)
    strides = ", ".join(str(s // array.itemsize) for s in array.strides)
    c_type = C_TYPES[array.dtype.kind]
    return (
        f"Array{ndim}D<{c_type}> {name}_view{{{name}, {{{shape}}}, {{{strides}}}}};\n"
    )


def _run(tmp_path, kernel, arrays, views, call, outputs):
    """Compile `kernel` with a main that calls it for each thread and prints `outputs`."""
    n_threads, call = call
    body = "".join(_c_array(name, a) for name, a in arrays.items())
    body += f"int main() {{ blockDim.x = 1; gridDim.x = {n_threads};\n"
    body += "".join(_view(name, arrays[name], ndim) for name, ndim in views.items())
    body += (
        f"for (long long t = 0; t < {n_threads}; ++t) "
        f"{{ blockIdx.x = (unsigned)t; {call}; }}\n"
    )
    body += "".join(
        f"for (long long i = 0; i < {arrays[name].size}; ++i) "
        f'printf("%.17g\\n", (double){name}[i]);\n'
        for name in outputs
    )
    body += "return 0; }\n"
    source = tmp_path / "kernel.cpp"
    exe = tmp_path / "kernel"
    source.write_text(STUBS + kernel.source.replace('extern "C"', "") + body)
    subprocess.run(
        [
            COMPILER,
            "-std=c++17",
            "-O1",
            f"-I{xp.cuda.cuda_include_dir()}",
            f"-I{SOURCE_ROOT}",  # tree_d_3d/kernels/barnes_hut.cuh
            str(source),
            "-o",
            str(exe),
        ],
        check=True,
    )
    printed = subprocess.run(
        [str(exe)], capture_output=True, text=True, check=True
    ).stdout
    values = np.array([float(v) for v in printed.split()])
    results, start = [], 0
    for name in outputs:
        size = arrays[name].size
        results.append(values[start : start + size].reshape(arrays[name].shape))
        start += size
    return results
