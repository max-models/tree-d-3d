# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Changed

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
