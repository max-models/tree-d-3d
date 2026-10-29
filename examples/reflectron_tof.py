"""Reflectron mass spectrometer.

A gridless reflectron time-of-flight mass spectrometer, and its energy focusing.

A time-of-flight mass spectrometer accelerates ions to (nearly) the same energy ``U`` and
times their flight: at fixed charge the time scales as ``sqrt(m)``, so the arrival time
gives the mass. The resolving power is set by the energy spread: in a linear
instrument ``t = L sqrt(m / 2U)`` and an energy spread ``dU / U`` smears the arrival by
``dt / t = dU / 2U``. A reflectron (Mamyrin, 1973) sends the ions into a retarding field
and back; faster ions penetrate deeper and spend longer turning, which compensates their
shorter drift. For a single uniform field ``E`` after a field-free drift of total length
``L`` (in and out), the flight time is ``t(U) = L sqrt(m / 2U) + 2 sqrt(2 m U) / (q E)``,
and ``dt / dU = 0`` at ``U0`` when ``q E = 4 U0 / L``: the ions turn at a depth of a
quarter of the drift length. This is the closed-form first-order focusing condition the
example checks.

The reflectron here is gridless, as in many modern instruments: a stack of slit
electrodes (simgeometry ``Aperture2D.slit`` plates) at linearly increasing potentials,
closed by a back plate, after a grounded flight tube (two thin ``Polyline`` walls).
:class:`tree_d_3d.BoundaryIntegralElectrodes` solves the electrode charges, the static
field is tabulated once on a grid around the ion paths, and ions are traced through it by
velocity Verlet. Arrival times come from simgeometry's ``crossings()`` of each step with
the detector line, interpolated within the step. Code units ``q = 1``, ``U0 = 1`` and
reference mass ``m = 1`` (read as 100 u); other masses follow exactly from
``t ~ sqrt(m)``, since the field is static and there is no space charge.

1. ``geometry``: the flight tube, the ring stack and the back plate, the potential
   around them, and a few ion paths from the source to the detector.
2. ``axial_potential``: the potential along the ion path inside the reflectron against
   the ideal linear ramp of a uniform field: the gridless entrance softens the field over
   about one aperture width.
3. ``energy_focusing``: ``dt / dU`` at ``U0`` against the reflectron field, through zero
   near the closed-form ``q E = 4 U0 / L``.
4. ``time_of_flight``: flight time against energy at the focusing field, flat to first
   order around ``U0``, against a linear instrument of the same flight time.
5. ``mass_spectrum``: arrival-time spectra of three neighbouring masses with a 1 %
   energy spread: resolved by the reflectron, merged in the linear instrument, with the
   resolving powers ``m / dm = t / (2 dt)``.

Usage::

    python examples/reflectron_tof.py [--output examples/output] [--quick]

``--quick`` uses fewer field scans and ions (well under a minute); ``--smoke`` is the tiny
configuration the test suite runs. Needs Plotly
(``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from simgeometry import Aperture2D, Polyline

from tree_d_3d import BoundaryIntegralElectrodes, Conductor, conductors_from_shape

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}
K_E = 1.0  # the field is set by the electrode potentials; k_e only scales their charges

U0 = 1.0  # ion energy (code units, q = 1)
REFLECTRON_X = 12.0  # entrance ring; source and detector at x = 0
TUBE_HALF_HEIGHT = 2.0
RING_HALF_WIDTH = 1.5
RING_PITCH = 0.4
RINGS = 21  # depth (RINGS - 1) * RING_PITCH = 8
DEPTH = (RINGS - 1) * RING_PITCH
FIELD_FREE_LENGTH = 2 * REFLECTRON_X  # in and out
E_FOCUS = 4 * U0 / FIELD_FREE_LENGTH  # q E L = 4 U0
SOURCE_Y = -0.2
SLOPE = 0.015  # the ions return 2 SLOPE x (path length) above the source
DETECTOR = Polyline([(0.0, 0.05), (0.0, 1.0)])


def build(field: float, spacing: float) -> tuple[BoundaryIntegralElectrodes, list]:
    """Electrodes for a reflectron field ``field`` (V per unit length); return them and the shapes."""
    top = field * DEPTH
    shapes = []
    parts: list[Conductor] = []
    for sign in (-1, 1):
        wall = Polyline(
            [
                (-1.0, sign * TUBE_HALF_HEIGHT),
                (REFLECTRON_X - 0.3, sign * TUBE_HALF_HEIGHT),
            ]
        )
        shapes.append((wall, 0.0))
        parts += conductors_from_shape(wall, 0.0, 4 * spacing)
    for k in range(RINGS):
        ring = Aperture2D.slit(
            REFLECTRON_X + k * RING_PITCH,
            half_width=RING_HALF_WIDTH,
            thickness=0.1,
            length=TUBE_HALF_HEIGHT - RING_HALF_WIDTH + 0.5,
        )
        potential = top * k / (RINGS - 1)
        shapes.append((ring, potential))
        parts += conductors_from_shape(ring, potential, spacing)
    back_x = REFLECTRON_X + DEPTH + RING_PITCH
    back = Polyline(
        [(back_x, -TUBE_HALF_HEIGHT - 0.5), (back_x, TUBE_HALF_HEIGHT + 0.5)]
    )
    back_potential = top * RINGS / (RINGS - 1)
    shapes.append((back, back_potential))
    parts += conductors_from_shape(back, back_potential, spacing)
    electrodes = BoundaryIntegralElectrodes(parts, k_e=K_E)
    electrodes.solve()
    return electrodes, shapes


class FieldMap:
    """The electrodes' field tabulated on a grid around the ion paths (bilinear)."""

    def __init__(self, electrodes: BoundaryIntegralElectrodes, h: float) -> None:
        x = np.arange(-0.5, REFLECTRON_X + DEPTH + 0.2, h)
        y = np.arange(-1.2, 1.2 + h / 2, h)
        self.table = electrodes.tabulate((x[0], y[0]), (x[-1], y[-1]), (x.size, y.size))
        self.x, self.y = self.table.x, self.table.y
        self.phi = self.table.potential

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        return self.table(pos)[1]


def flight_times(
    fields: FieldMap, energies: np.ndarray, dt: float, slope: float = SLOPE
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Arrival times at the detector of ions (q = m = 1) of the given energies.

    Each starts at ``(0, SOURCE_Y)`` heading into the reflectron at a small angle,
    ``slope``, so that it returns onto the detector. Returns the times (NaN for ions
    that miss) and the paths.
    """
    n = len(energies)
    speed = np.sqrt(2 * energies)
    direction = np.array([1.0, slope]) / np.hypot(1.0, slope)
    pos = np.stack((np.zeros(n), np.full(n, SOURCE_Y)))
    vel = direction[:, None] * speed
    acc = fields(pos)
    t = np.zeros(n)
    arrival = np.full(n, np.nan)
    flying = np.ones(n, dtype=bool)
    paths = [pos.copy()]
    max_steps = int(6 * (REFLECTRON_X + DEPTH) / (speed.min() * dt))
    for _ in range(max_steps):
        if not flying.any():
            break
        old = pos.copy()
        vel = vel + 0.5 * dt * acc
        pos = pos + dt * vel
        acc = fields(pos)
        vel = vel + 0.5 * dt * acc
        t = t + dt
        returning = flying & (vel[0] < 0)
        if returning.any():
            hits = DETECTOR.crossings(old[:, returning], pos[:, returning])
            hit = np.asarray(hits.hit, dtype=bool)
            index = np.flatnonzero(returning)[hit]
            arrival[index] = t[index] - dt * (1 - np.asarray(hits.fraction)[hit])
            flying[index] = False
            flying &= ~(returning & (pos[0] < -0.5))
        flying &= np.abs(pos[1]) < fields.y[-1]  # left the tabulated region: lost
        paths.append(pos.copy())
    return arrival, paths


def main(argv: list[str] | None = None) -> Path:
    """Run the reflectron checks and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        spacing, h, dt, scan, n_spec = 0.2, 0.1, 0.05, 3, 200
    elif args.quick:
        spacing, h, dt, scan, n_spec = 0.08, 0.04, 0.02, 7, 3000
    else:
        spacing, h, dt, scan, n_spec = 0.05, 0.025, 0.01, 11, 10000
    probe = U0 * (1 + np.array([-0.03, -0.015, 0.0, 0.015, 0.03]))

    # 3: dt/dU against the reflectron field
    field_scan = E_FOCUS * np.linspace(0.85, 1.4, scan)  # deeper turns leave the stack
    slopes, maps = [], {}
    for field in field_scan:
        electrodes, shapes = build(field, spacing)
        fields = FieldMap(electrodes, h)
        times, _ = flight_times(fields, probe, dt)
        ok = np.isfinite(times)
        slopes.append(
            np.polyfit(probe[ok] - U0, times[ok], 1)[0] if ok.sum() > 2 else np.nan
        )
        maps[field] = (electrodes, shapes, fields)
    slopes = np.array(slopes)
    good = np.isfinite(slopes)
    field_scan, slopes = field_scan[good], slopes[good]
    sign = np.flatnonzero(np.diff(np.sign(slopes)))
    if len(sign):
        i = sign[0]
        e_best = field_scan[i] - slopes[i] * (field_scan[i + 1] - field_scan[i]) / (
            slopes[i + 1] - slopes[i]
        )
    else:
        e_best = field_scan[np.argmin(np.abs(slopes))]
    print(
        f"focusing field: simulated {e_best:.4f}, closed form 4 U0 / L = {E_FOCUS:.4f} "
        f"(ratio {e_best / E_FOCUS:.3f})"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=field_scan / E_FOCUS,
            y=slopes,
            mode="lines+markers",
            line={"color": COLORS["C0"]},
            name="dt/dU at U0 (traced)",
        )
    )
    fig.add_hline(y=0.0, line={"color": "gray", "dash": "dot"})
    fig.add_vline(
        x=1.0,
        line={"color": "black", "dash": "dash"},
        annotation_text="q E L = 4 U0",
    )
    fig.update_xaxes(title_text="reflectron field / (4 U0 / q L)")
    fig.update_yaxes(title_text="dt / dU at U0")
    fig.update_layout(title="First-order energy focusing of the reflectron")
    save_figure(fig, output / "energy_focusing")

    # the reflectron at the focusing field
    electrodes, shapes = build(e_best, spacing)
    fields = FieldMap(electrodes, h)

    # 1: geometry
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=fields.x,
            y=fields.y,
            z=fields.phi.T,
            colorscale="Viridis",
            contours={"coloring": "heatmap"},
            colorbar={"title": "phi"},
            opacity=0.6,
        )
    )
    for shape, _ in shapes:
        for loop in shape.outlines():
            closed = getattr(shape, "closed", True)
            xs = np.append(loop[0], loop[0, 0]) if closed else loop[0]
            ys = np.append(loop[1], loop[1, 0]) if closed else loop[1]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    fill="toself" if closed else None,
                    line={"color": "black", "width": 1},
                    fillcolor="gray",
                    showlegend=False,
                )
            )
    _, paths = flight_times(fields, probe[[0, 2, 4]], dt)
    path = np.array(paths)
    for j, color in zip(range(3), ("C0", "C1", "C3"), strict=True):
        fig.add_trace(
            go.Scatter(
                x=path[:, 0, j],
                y=path[:, 1, j],
                mode="lines",
                line={"color": COLORS[color]},
                name=f"U = {probe[[0, 2, 4]][j]:.3f}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=DETECTOR.vertices[:, 0],
            y=DETECTOR.vertices[:, 1],
            mode="lines",
            line={"color": COLORS["C2"], "width": 5},
            name="detector",
        )
    )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(
        title_text="y", range=[-TUBE_HALF_HEIGHT - 0.6, TUBE_HALF_HEIGHT + 0.6]
    )
    fig.update_layout(
        title="Gridless reflectron: flight tube, ring stack and ion paths",
        legend={"orientation": "h", "y": -0.15},
    )
    save_figure(fig, output / "geometry")

    # 2: axial potential against the ideal ramp
    on_path = np.argmin(np.abs(fields.y - 0.0))
    x = fields.x
    ideal = np.clip(e_best * (x - REFLECTRON_X), 0.0, None)
    inside = (x > REFLECTRON_X - 3) & (x < REFLECTRON_X + DEPTH)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x[inside],
            y=fields.phi[inside, on_path],
            mode="lines",
            line={"color": COLORS["C0"]},
            name="ring stack (panel solution)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[inside],
            y=ideal[inside],
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="uniform field from the entrance ring",
        )
    )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="potential on the axis")
    fig.update_layout(title="Potential inside the gridless reflectron")
    save_figure(fig, output / "axial_potential")

    # 4: time of flight against energy
    energies = U0 * np.linspace(0.94, 1.06, 13 if not args.smoke else 5)
    times, _ = flight_times(fields, energies, dt)
    t0 = np.interp(U0, energies, times)
    linear = t0 * np.sqrt(U0 / energies)  # same flight time at U0, no reflectron
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=energies,
            y=times / t0 - 1,
            mode="lines+markers",
            line={"color": COLORS["C0"]},
            name="reflectron (traced)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=energies,
            y=linear / t0 - 1,
            mode="lines",
            line={"color": COLORS["C3"], "dash": "dash"},
            name="linear instrument, t ~ U^-1/2",
        )
    )
    fig.update_xaxes(title_text="ion energy / U0")
    fig.update_yaxes(title_text="relative flight-time change")
    fig.update_layout(title="Flight time against energy at the focusing field")
    save_figure(fig, output / "time_of_flight")

    # 5: mass spectrum, 1 % energy spread; t ~ sqrt(m) at fixed charge
    rng = np.random.default_rng(0)
    spread = U0 * (1 + 0.01 * rng.standard_normal(n_spec))
    spread = np.clip(spread, 0.95 * U0, 1.05 * U0)
    grid = U0 * np.linspace(0.95, 1.05, 21 if not args.smoke else 7)
    grid_times, _ = flight_times(fields, grid, dt)
    t_ref = np.interp(spread, grid, grid_times)  # mass 100 u
    t_lin = t0 * np.sqrt(U0 / spread)
    masses = (100.0, 101.0, 102.0)
    fig = go.Figure()
    print("mass resolving power m / dm = t / (2 FWHM):")
    for instrument, base, dash in (
        ("reflectron", t_ref, "solid"),
        ("linear", t_lin, "dash"),
    ):
        all_times = np.concatenate([base * np.sqrt(m / 100.0) for m in masses])
        fwhm = 2.3548 * np.std(base)
        resolving = np.mean(base) / (2 * fwhm)
        print(f"  {instrument}: {resolving:.0f}")
        counts, edges = np.histogram(all_times, bins=300)
        fig.add_trace(
            go.Scatter(
                x=0.5 * (edges[1:] + edges[:-1]),
                y=counts,
                mode="lines",
                line={"dash": dash},
                name=f"{instrument}, m/dm = {resolving:.0f}",
            )
        )
    fig.update_xaxes(title_text="arrival time (code units, m = 100 u at t ~ sqrt(m))")
    fig.update_yaxes(title_text="ions per bin")
    fig.update_layout(title="Masses 100, 101 and 102 u with a 1 % energy spread")
    save_figure(fig, output / "mass_spectrum")

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
