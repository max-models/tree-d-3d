---
title: Quickstart
description: Evaluate fields with a tree, the FMM or a direct sum, and run a simulation.
---

First, ensure that `tree-d-3d` is [installed](/tree-d-3d/getting-started/installation/).

Arrays of particles are component-major: positions have shape `(2, N)`.

```python
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

On the CuPy backend (`cunumpy.use_backend("cupy")`) the trees and direct sums take and
return CuPy arrays and run the CUDA kernels.


Fields are returned without the Coulomb constant: multiply by `tree_d_3d.K_E` for SI units.
The simulation includes it (`k_e=tree_d_3d.K_E` by default).

See the [API reference](/tree-d-3d/api/tree_d_3d/) for every module and function.
