---
title: Parallel runs
description: Share a simulation or a panel solve over MPI ranks, and how far it scales.
---

`ElectrostaticSimulation`, `BoundaryIntegralElectrodes` and `FlatQuadtree` take a
`comm=` argument, an MPI communicator. With it the ranks share the work and every rank
gets the same answer as a serial run (to round-off). Without it, or under one process,
nothing changes.

Install mpi4py next to an MPI library (Open MPI or MPICH). tree-d-3d imports it through
[maybempi](https://github.com/max-models/maybempi), which falls back to a serial stand-in
when the script is not started by `mpirun`, so the same script runs either way.

## A simulation

Each rank passes its own particles. Diagnostics such as `get_total_energy` are sums over
every rank, so every rank must call them (they are collective):

```python
import numpy as np
from maybempi import MPI, collectives

from tree_d_3d import ChargedBody, ElectrostaticSimulation, Quad

comm = MPI.COMM_WORLD
rng = np.random.default_rng(1)
positions = rng.uniform(-0.9, 0.9, (2, 400_000))
charges = rng.choice([-1.0, 1.0], 400_000)

mine = collectives.block(400_000, comm)  # this rank's share
bodies = [
    ChargedBody(*positions[:, i], charge=charges[i])
    for i in range(mine.start, mine.stop)
]
sim = ElectrostaticSimulation(
    bodies, Quad(0, 0, 1), k_e=1e-4, dt=1e-3, softening=1e-3, kernel="log", comm=comm
)
for _ in range(10):
    sim.step()
energy = sim.get_total_energy()  # on every rank
if comm.Get_rank() == 0:
    print(energy)
```

```sh
mpirun -n 4 python run.py
```

Particles stay on the rank that created them; absorbed ones leave it. Give the ranks
similar numbers of particles for an even load.

## A panel solve

Here every rank passes the same conductors, free charges and probe points; the ranks
share the probes, the right-hand side, the preconditioner blocks and the GMRES products
(or the rows of the dense matrix), and gather the results:

```python
electrodes = BoundaryIntegralElectrodes(conductors, k_e=1.0, comm=comm)
electrodes.solve(sources, charges)
phi, field = electrodes.probe(points)  # the whole result on every rank
```

## How it works and how far it scales

The sources are replicated and the targets distributed. Every rank gathers all the
particles (`MPI_Allgatherv`) and builds the same tree. It then evaluates the field only at
its own particles, from the whole tree. The cells' multipole moments, the FMM's M2L and
near-field pairs and the panel solver's sums are split evenly over the ranks, and are
completed by an allreduce or a gather.

This suits a few to tens of ranks on one node or a few nodes. Each rank still holds every
particle and builds the whole tree, so memory does not shrink with the ranks and the
build and the gather cap the speed-up (Amdahl). A distributed tree, with particles
migrating between subdomains as in a PIC code, would be the next step beyond that.

Strong scaling, one thread per rank, on one 4-core machine (mean of 2 runs; the ranks
share its memory bandwidth):

| Workload | 1 rank | 2 ranks | 4 ranks | Speed-up at 4 |
|---|---|---|---|---|
| 3 steps, 400k line charges, FMM | 7.63 s | 4.75 s | 3.03 s | 2.5x |
| 3 steps, 400k point charges, tree | 14.6 s | 7.79 s | 4.14 s | 3.5x |
| 3 steps, 400k line charges, tree | 44.1 s | 24.6 s | 13.2 s | 3.3x |
| 100k probes, 2.7k panels (LU) | 0.31 s | 0.17 s | 0.092 s | 3.4x |
| 100k probes, 21k panels | 0.99 s | 0.65 s | 0.34 s | 2.9x |
| GMRES set-up, 21k panels | 0.59 s | 0.37 s | 0.27 s | 2.2x |
| GMRES solve, 21k panels | 8.12 s | 5.37 s | 3.68 s | 2.2x |
| LU set-up, 2.7k panels | 0.66 s | 0.62 s | 0.51 s | 1.3x |

The LU factorisation stays on every rank, so only the matrix assembly is shared.
