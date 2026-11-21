---
title: Installation
description: Install tree-d-3d and its optional GPU backends.
---

tree-d-3d needs Python 3.10 or newer, [cunumpy](https://github.com/max-models/cunumpy),
[simgeometry](https://github.com/max-models/simgeometry) for electrode and wall shapes, and
[Pyccel](https://github.com/pyccel/pyccel) (with a C compiler) for the compiled CPU kernels.
Without a working Pyccel the NumPy versions of the kernels run instead, which is much
slower for the tree kernels.

```bash
pip install "tree-d-3d @ git+https://github.com/max-models/tree-d-3d"
```

The kernels are compiled on first use and cached in `~/.cache/tree-d-3d/kernels`
(`TREE_D_3D_KERNEL_CACHE` moves it). They are built with OpenMP; `TREE_D_3D_OPENMP=0` builds them
serial, `OMP_NUM_THREADS` limits the threads.

`TREE_D_3D_FLAGS` adds C compiler flags to the builds (with cunumpy's devel branch, kept apart in
the cache): `TREE_D_3D_FLAGS="-march=native -ffast-math"` made the tree walk of point charges
1.45x and a GMRES panel solve 1.2x faster on one Xeon core, other parts 1.0-1.2x. Such builds
may not run on another CPU, and `-ffast-math` changes the rounding, so none are set by default.

## Optional extras

| Extra   | Installs                                                  |
| ------- | --------------------------------------------------------- |
| `gpu`   | CuPy, for the CUDA kernels                                |
| `metal` | MLX, for the Metal kernels on Apple-silicon Macs          |
| `test`  | `pytest` and `coverage`                                   |
| `docs`  | the notebook runner and griffe for the API reference      |
| `dev`   | formatters and linters, plus the `test` and `docs` extras |

For development, clone the repository and run `make install` (uv), or

```bash
pip install -e ".[dev]"
```
