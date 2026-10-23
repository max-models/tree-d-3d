# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Changed

- simgeometry comes from PyPI (`simgeometry>=0.1.0`) instead of git.

- `BoundaryIntegralElectrodes` LU-factorises its panel matrix once; each `solve()` reuses it.
- The test extra installs the examples extra, so CI runs every example test.

### Removed

- `circle_vertices`: use `conductors_from_shape(Circle(...), ...)`.

### Fixed

- `conductors_from_shape` keeps an open curve (a simgeometry `Polyline`) an open conductor
  instead of closing it.
- The einzel-lens example no longer claims round einzel lenses focus equally for both
  polarities.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
