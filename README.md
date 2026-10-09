# tree-d-3d


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/tree-d-3d/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/tree-d-3d/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/tree-d-3d/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/tree-d-3d/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/tree-d-3d/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/tree-d-3d/)
[![codecov](https://codecov.io/gh/max-models/tree-d-3d/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/tree-d-3d)
[![PyPI](https://img.shields.io/pypi/v/tree-d-3d.png)](https://pypi.org/project/tree-d-3d/)
[![Python](https://img.shields.io/pypi/pyversions/tree-d-3d.png)](https://pypi.org/project/tree-d-3d/)

Barnes-Hut trees and fast multipole methods for charged particles in a
plane, on NumPy, CuPy or the Apple GPU. Extracted from
[PICNIC](https://github.com/max-models/picnic).

- `FlatQuadtree`: a flat array quadtree with multipole moments. Fields
  of point charges (the 3D Coulomb kernel, `kernel="coulomb"`) or of
  line charges (the 2D Poisson kernel, `kernel="log"`) at any targets,
  by Barnes-Hut tree walk or, for line charges, the O(N) fast multipole
  method (`FlatQuadtree.fmm`).
- `direct_field`: the exact all-pairs sum, for small systems and
  reference results.
- `ElectrostaticSimulation`: time integration of `ChargedBody`
  particles, second order, with a uniform magnetic field `B_z`,
  absorbing walls and cached fields (one field evaluation per step).
- `ReferenceQuadtree`: a readable object tree computing the same as the
  flat tree.
- `tree_d_3d.gravity`: the classic gravitational Barnes-Hut N-body code.

Every kernel has a Pyccel version (compiled on first use, cached in
`~/.cache/tree-d-3d/kernels`), a NumPy version (used if Pyccel is
unavailable), a CUDA version for CuPy arrays and a Metal version
(float32, via MLX). The array backend comes from
[cunumpy](https://github.com/max-models/cunumpy).

Documentation: <https://max-models.github.io/tree-d-3d/>

## Install

``` bash
pip install "tree-d-3d @ git+https://github.com/max-models/tree-d-3d"
pip install "tree-d-3d[gpu] @ git+https://github.com/max-models/tree-d-3d"    # CuPy for CUDA
pip install "tree-d-3d[metal] @ git+https://github.com/max-models/tree-d-3d"  # MLX for Apple GPUs
```

or, for development, with [uv](https://docs.astral.sh/uv/):

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
```

## Usage

Arrays of particles are component-major: positions have shape `(2, N)`.

``` python
import numpy as np

from tree_d_3d import (
    ChargedBody,
    ElectrostaticSimulation,
    FlatQuadtree,
    Quad,
    direct_field,
)

rng = np.random.default_rng(0)
pos = rng.uniform(-1.0, 1.0, (2, 5000))  # pos[0] is x
charge = rng.choice([-1.0, 1.0], 5000)
others = np.arange(5000)  # exclude each particle's own charge

# Barnes-Hut tree walk against the direct sum (fields without k_e)
tree = FlatQuadtree(pos, charge, Quad(0.0, 0.0, 1.0))
phi, field = tree.evaluate(pos, exclude=others, theta=0.5, softening=1e-3)
phi_ref, field_ref = direct_field(pos, pos, charge, softening=1e-3, exclude=others)

# line charges with the fast multipole method
phi, field = FlatQuadtree(
    pos, charge, Quad(0.0, 0.0, 1.0), kernel="log", order=12
).fmm()

# a cloud of protons in normalized units
bodies = [ChargedBody(x, y) for x, y in rng.normal(0.0, 0.1, (200, 2))]
sim = ElectrostaticSimulation(
    bodies, Quad(0.0, 0.0, 1.0), k_e=1.0, softening=1e-3, dt=1e-4
)
for _ in range(100):
    sim.step()
print(sim.get_total_energy())
```

On the CuPy backend (`cunumpy.use_backend("cupy")`) the trees and direct
sums take and return CuPy arrays and run the CUDA kernels.

## Development

``` bash
make lint        # ruff check, ruff format --check, pyright
make test        # pytest with coverage
```

The CUDA tests need a GPU. Without one, the CUDA kernels are checked on
the CPU by compiling them as C++ (`test_kernels_emulated.py`), and the
CuPy code path of the solver runs on cunumpy’s fake CuPy with those
emulated kernels (`test_cupy_path.py`); both need a C++ compiler. The
Metal tests run on Apple-silicon Macs with `tree-d-3d[metal]`.

## Build docs

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-dev         # live preview at http://localhost:4321/tree-d-3d/
make docs-build       # the static site in docs/dist
```

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/) (`make readme`).
