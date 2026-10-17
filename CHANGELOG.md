# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

- The einzel-lens example no longer claims round einzel lenses focus equally for both
  polarities.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
