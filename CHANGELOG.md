# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- tree-d-3d runs in the browser with Pyodide, and a live demo page in the docs builds a
  quadtree there: sliders for the particles, leaf capacity, opening angle and a target redraw
  the cells, the target's Barnes-Hut walk and the field errors against the direct sum. It uses
  starlight-theme's `PyodideDemo`; the docs workflow builds the wheels it installs
  (`pyodide: true`), and a Pyodide check in CI (`.github/workflows/pyodide.yml`) installs the
  wheel, imports every module and runs the demo. Pyccel is not installed there
  (`sys_platform != 'emscripten'`).
- Examples `diocotron_instability` and `coulomb_explosion`: physics processes with the
  evolving quadtree animated by `plasma_plots.animate_quadtree`. The diocotron example drifts
  line-charge guiding centres by the FMM and matches Davidson's growth rates of a thin
  annulus to within 3%; the Coulomb explosion follows the exact self-similar expansion of a
  uniform disk to 3e-4 and the wave breaking of a Gaussian cloud.
- Examples `galaxy_collision` (Toomre-style tidal tails of a prograde disk, a retrograde one
  intact, the cores on the two-body parabola, energy error against the opening angle),
  `cold_collapse` (a Maclaurin disk falling in, violent relaxation to virial equilibrium,
  and a rotating one fragmenting) and `two_stream_instability` (electron holes in phase
  space; a periodic slab summed by the FMM after a conformal map onto an annulus; growth
  rates against the cold-beam dispersion relation).
- MPI: `ElectrostaticSimulation`, `BoundaryIntegralElectrodes` and `FlatQuadtree` take
  `comm=`, an MPI communicator (via maybempi, a new dependency), and agree with serial runs
  to round-off. Sources are replicated and targets distributed: every rank builds the whole
  tree from the gathered particles and evaluates its own. The tree moments, the FMM's pair
  lists, probes, panel sums, the preconditioner blocks and the dense-matrix rows are split
  over the ranks. At 4 ranks on 4 cores, a 400k-particle step runs 2.5x (FMM) to 3.5x (tree)
  faster, 100k probes 2.9-3.4x and a 21k-panel GMRES solve 2.2x faster
  ("Parallel runs" in the docs). The gathers and sums come from `maybempi.collectives`; the tests run the parallel paths on
  thread ranks (`maybempi.testing`).

### Changed

- Without Pyccel (in Pyodide, or where it fails to compile) the CPU tree walk runs a
  vectorized NumPy walk, all targets descending the tree together, instead of the Pyccel
  loops in the interpreter, and the tree is built with the level-by-level NumPy kernels: in
  Pyodide, 5000 particles take 0.33 instead of 7.9 s to walk and 0.04 instead of 0.2 s to
  build.
- GMRES products walk the panels' tree at the midpoints instead of running the FMM over
  every Gauss node: 0.11 instead of 0.17 s per product at 21k panels on one core.
- Faster panel solves and sums (measured on a 4-core CPU, 3 runs each):
  - GMRES keeps one FMM tree per geometry and only recharges it (`FlatQuadtree.set_charges`,
    interaction lists cached per opening angle), and is preconditioned by the exact panel
    matrix of each run of 256 consecutive panels (block Jacobi): 150 to 51 iterations at
    21k panels, the solve 27.9 to 7.6 s.
  - Probes and sums over free charges above `TREE_PAIRS` pairs walk a multipole tree, kept
    for the panels: 100k probes 9.2 to 0.75 s at 21k panels, 1.27 to 0.18 s at 2.7k.
    `inside` uses simgeometry's compiled containment test.
  - The panels' multipole sums run at theta 0.7 with order 32 and leaves of 16, the measured
    optimum at their accuracy (about 1e-12).
  - Periodic sums and GMRES products run in the plane mapped by `exp(2 pi z / period)`, where
    a row of copies is one charge: 20k periodic probes 57 s to 0.13 s, a solve with 20k free
    charges 2.6 to 0.02 s; `solver="gmres"` now works with a period (within `MAP_REACH`
    periods along x).
  - The CPU tree build splits a level in one `quadtree_split` kernel call instead of
    `(N, 4)` prefix sums: builds 1.1-1.6x faster, the trees unchanged.
- Faster single-core CPU paths (200k particles, one thread, against the previous commit):
  tree walk 1.9 to 0.85 s (Coulomb) and 4.8 to 2.4 s (log), FMM 1.55 to 0.72 s (leaves of
  8, order 12) and 1.24 to 0.62 s (32, 18), a simulation step 2x, 50k panel probes 0.43 to
  0.09 s. Results agree with the previous ones to round-off. The GPU kernels are unchanged.
  - The walk (`FlatQuadtree.evaluate`) takes nearby targets in groups of 32 (tree or
    Morton order), the group walking the tree once with a mask of the targets still
    walking, so each target gets the same cells as alone; sources are read in tree order.
  - FMM: M2L by destination cell (`fmm_m2l_host`, rows from `pair_csr`), as a real
    matrix on scaled moments, without atomics; the near field on the particles in tree
    order (`fmm_evaluate_host`).
  - Moments in one upward pass (leaves from their particles, then M2M), instead of every
    cell summing its particles; the CPU build is one `quadtree_build` call, which replaces
    `quadtree_split`.
  - `FlatQuadtree.evaluate(potential=False)` skips the log kernel's logarithms; steps of
    `ElectrostaticSimulation` with the log kernel's tree or FMM skip the potential, which
    `get_potential_energy` computes when asked (`compute_fields(potential=...)`).
  - Panel probes find exact hits on panel nodes without a full nearest-neighbour search.

## [0.2.0] - 2026-11-20

### Changed

- The kernels are built with cunumpy's `KernelPackage` (`tree_d_3d.kernels.KERNEL_PACKAGE`):
  `kernels/_build.py`, `kernels/compile.py` and `kernels/_metal_atomic.py` are gone, and
  `kernels/_metal.py` keeps only the tree helpers (`BH_HEADER`, `merge_status`). The build
  cache is still `TREE_D_3D_KERNEL_CACHE`; `metal_available` is cunumpy's.

### Added

- A Numerics section in the docs (header and sidebar): an overview of the model's equations
  (`docs/src/content/docs/numerics/index.mdx`, with the main cards) and one page per module,
  from the trees to the solvers built on them (starlight-theme's `methodcard: { pages: true }`).
- Method cards (methodcard) on the equations of the package: the direct sums of both kernels,
  the tree and its moments, the Barnes-Hut walk with the Coulomb and log expansions, the FMM
  (traversal, M2L, L2L, L2P, P2P), the reference radius, macroparticles, the pusher with the
  exact gyration and the pair energies, the grounded strip with its self term, image force and
  softening correction, the uniformly charged segment, the boundary-integral electrodes and
  their tabulated field, the smooth-membership FMM and the gravitational Barnes-Hut code, with
  their assumptions and references. `test_method_cards.py` checks them: SymPy rules for the
  closed forms (the kernels, the moments and expansions, the strip's Green's function and image
  force, the segment integral, the compact softening, the gyration drift, the pair energies),
  the observed order of the tree walk and the FMM against the size of the cells' charges, of
  the electrodes against the panel length, of the tabulated field and of the pusher, and the
  cards of the docs against those of the code. Adds the `methodcard` dependency.

- Opt-in scope-profiler regions (`tree_d_3d.<module>.<operation>`) around the tree build,
  moments, tree walk, FMM, direct sum, simulation step and field evaluation, electrode
  factorisation, solve and panel field, and the grounded-strip field. They record nothing
  unless the application enables the `tree_d_3d` tag (`scope_profiler.setup(enable_tags=
  ["tree_d_3d"])`, `SCOPE_PROFILER_ENABLE_TAGS` or `scope-profiler run --enable-tag`).
  Needs `scope-profiler>=0.7.5`, which uv takes from scope-profiler's `devel` branch
  (`[tool.uv.sources]`) until that release is on PyPI.

- `BoundaryIntegralElectrodes` treats panels within `near_field` (4) of their own lengths of a
  target with the exact potential and field of a uniformly charged segment
  (`segment_field`), and sums farther panels fast as `far_points` (2) Gauss-Legendre point
  charges each. The boundary potential between collocation points drops from about a tenth to
  a few thousandths of the source scale, and the method converges at second to third order
  instead of first; `near_field=0, far_points=1` is the old midpoint rule.
- `BoundaryIntegralElectrodes.tabulate()` and `TabulatedField`: the solved field on a grid,
  interpolated bilinearly; `panel_field()`: the panels' field without the free charges. The
  multipactor, reflectron and Pierce-gun examples use them.

- Examples beyond the tree code's home ground, each checked against a closed form or an analytic
  rate: charge-exchange ions and grid erosion in an ion-thruster grid set
  (`cex_grid_erosion.py`), a multipactor susceptibility chart (`multipactor.py`), a reflectron
  time-of-flight mass spectrometer (`reflectron_tof.py`), a planar Pierce gun (`pierce_gun.py`)
  and an ion-mobility drift tube (`drift_tube_mobility.py`). plasmacoll joins the `examples`
  extra for their collisions.

- simgeometry is a dependency. `conductors_from_shape` turns any 2D simgeometry shape into
  `BoundaryIntegralElectrodes` conductors, with constant or graded panel lengths.
- `ElectrostaticSimulation(absorbing_objects={name: shape})` removes particles that end a
  drift inside a simgeometry shape, recorded under its name in `absorbed`.
- The einzel-lens, ion-extraction and boundary-integral examples build their electrodes from
  simgeometry shapes (`Aperture2D`, `Circle`, `Polygon`).
- Absorbing objects test the whole step, not just its end: a particle whose step crosses an
  object's boundary is absorbed by the object it meets first, so none jumps over a plate
  thinner than `|v| dt`. `Polygon` and `Aperture2D` objects go through simgeometry's
  compiled swept polygon kernel (`simgeometry.polygon_objects`, shared with picnic), other
  shapes through their `crossings`. `AbsorbedParticles.charges()` gives the charge collected
  by each wall and object. The ion-extraction example uses the swept test for its plate hits
  instead of its own mid-plane check.
- `BoundaryIntegralElectrodes(period=...)`: geometries periodic along y, through the closed-form
  row of line charges `periodic_log_kernel` (with a method card); the panel copies near a target
  still enter exactly. `Face(axis, position, kind, potential)` and `face=`: a Neumann or
  Dirichlet domain face by mirror images, combinable with the period; free charges get their
  images too, and `force_on_sources` includes them.
- Touching conductors at one potential act as their union: panel pairs on a shared edge, also
  across the period seam, are dropped (`num_shared_panels`) instead of making the system singular.
- `BoundaryIntegralElectrodes.inside` (the solid conductor holding a point; loops around other
  conductors or free charges are shells), exact potential and zero field from `probe` inside a
  solid conductor, and `surface_field()` (the normal field `4 pi k_e sigma` just outside each
  panel).
- `solver="gmres"` (and `"auto"` above `DIRECT_LIMIT` panels): GMRES with fast multipole products
  and a self-term preconditioner, so tens of thousands of panels solve without a dense matrix.
- Notebooks "Periodic electrodes", "Fast panel solves", "Absorbing objects" and "Kernels".
- The example figures are web figures on the theme's `PlotlyFigure` (site colours, light and dark
  posters, the legend below the plot on a phone), and every example's figures are laid out for a
  phone: stacked panels, short titles, LaTeX labels, decade ticks, animations that start on their
  first frame. The examples extra installs `plasma-plots[plotly,web]>=0.3`.

### Changed

- simgeometry comes from PyPI (`simgeometry>=0.2.1`, for the swept polygon kernel) instead of
  git; cunumpy at least 0.7.2.

- `BoundaryIntegralElectrodes` LU-factorises its panel matrix once; each `solve()` reuses it.
- The test extra installs the examples extra, so CI runs every example test.

### Removed

- `circle_vertices`: use `conductors_from_shape(Circle(...), ...)`.

### Fixed

- The math in the docstrings renders on the API pages: `$$` blocks instead of indented text,
  and code spans instead of `:func:` roles. The docstrings of `flat_tree` (it has the log
  kernel too), `electrodes` (far panels are two Gauss-Legendre point charges), `gravity` (no
  octree), `FlatQuadtree.fmm` and `fmm_traverse` (P2P pairs need not touch),
  `ElectrostaticSimulation` (`"auto"` sums directly without the compiled kernels) and
  `ReferenceQuadtree.field_at` (an `exclude` away from the probe makes the sum exact) say what
  the code does.

- For an odd `far_points` the middle Gauss node is exactly the panel midpoint.
- The drift-tube example passes the gas temperature to plasmacoll as kT in eV (it passed 300 K
  as 300 eV); mobility, diffusion and resolving power now match their theory within a few percent.
- `conductors_from_shape` keeps an open curve (a simgeometry `Polyline`) an open conductor
  instead of closing it.
- The einzel-lens example no longer claims round einzel lenses focus equally for both
  polarities.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
