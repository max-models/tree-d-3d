"""Two-stream instability.

Two cold electron beams passing through each other bunch up and trap themselves in phase space.

Two equal electron beams stream through each other at ``+v0`` and ``-v0`` across a fixed ion
background that neutralises them. A small ripple of density along the beams is amplified: a
beam slows where the potential is high and bunches there, which raises the potential further.
For cold beams of plasma frequency ``omega_b = omega_p / sqrt(2)`` each, a mode ``k`` along the
beams grows at ``gamma^2 = (sqrt(1 + 8 a^2) - 1 - 2 a^2) / 2`` (in ``omega_p``, with
``a = k v0 / omega_p``) when ``a < 1``, fastest, at ``gamma = omega_p / sqrt(8)``, for
``a^2 = 3/8``. In a slab of plasma this holds for every bulk mode, whatever its shape across
the slab. The growth stops when the beams' electrons are trapped in the potential wells: the
phase space ``(x, v_x)`` rolls up into vortices (electron holes), the textbook picture of a
kinetic instability.

The plasma is a slab ``|y| < H/2``, periodic along the beams (``x``, period ``L``), with
electrons and ions as line charges. A periodic sum of line charges is a free-space sum after
the conformal map ``zeta = exp(2 pi i (x + i y) / L)``, which wraps the slab around an
annulus: the field comes from the FMM of ``FlatQuadtree`` over the mapped charges (the
system is neutral, so the map adds no field of its own), times ``d zeta / dz``. The electrons
are pushed with the leapfrog; the ions do not move. Units ``k_e = e = m_e = 1``,
``omega_p = v0 = 1``, ``L = 20`` (modes ``k = 0.314 m``), ``H = 5``; the electrons start on
the ion sites (a quiet start), displaced by a small ripple.

1. ``phase_space``: the electrons in ``(x, v_x)``, colored by their beam, as the most unstable
   mode (``m = 2``) grows, saturates and rolls the two beams up into two vortices (electron
   holes) of trapped electrons, which then slowly mix.
2. ``mapped_tree``: the same run in the plane of the map, with the quadtree the FMM builds
   every step: the slab is the annulus, the beams' bunches are the spokes in which the cells
   crowd.
3. ``mode_growth``: the density modulation of the seeded mode against time, one run per mode
   ``m = 1 ... 4``, with the slope of the cold-beam growth rate (dashed); ``m = 4``
   (``a = 1.26``) is stable. Each run is seeded with the lowest bulk mode of the slab, the
   displacement ``grad(cos(k x) cos(pi y / H))``, whose potential vanishes at the edges.
4. ``growth_rates``: the growth rates fitted to the runs against the dispersion relation.
   ``m = 1``, ``2`` and ``4`` agree with it; ``m = 3``, near the cutoff, grows faster, at
   about the rate of ``m = 2``: a mode of the edges of the slab, which the bulk theory leaves
   out. It weakens as the slab gets thicker (fitted 0.37, 0.29 and 0.26 at ``H = 5``, 10 and
   15, against 0.19 for the bulk).
Usage::

    python examples/two_stream_instability.py [--output examples/output] [--quick]

``--quick`` uses fewer electrons (about a minute); ``--smoke`` is the tiny configuration the
test suite runs. Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plasma_plots import animate_quadtree, save_figure

from tree_d_3d import FlatQuadtree, Quad

#: Period along the beams, slab thickness, beam speed; the plasma frequency is 1.
L, H, V0 = 20.0, 5.0, 1.0
K0 = 2.0 * np.pi / L
#: The charge density of the ions: omega_p^2 = 4 pi k_e n q^2 / m = 1.
DENSITY = 1.0 / (4.0 * np.pi)
MODES = (1, 2, 3, 4)
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def growth_rate(k: float | np.ndarray) -> np.ndarray:
    """The cold symmetric two-stream growth rate at wavenumber ``k``, in ``omega_p``."""
    a2 = (np.asarray(k, dtype=float) * V0) ** 2
    return np.sqrt(np.maximum(0.5 * (np.sqrt(1.0 + 8.0 * a2) - 1.0 - 2.0 * a2), 0.0))


# -- the plasma ----------------------------------------------------------------------------------


def slab(spacing: float, ripples: dict[int, float]) -> dict:
    """Ions on a lattice filling the slab, and two beams of electrons on the same sites.

    ``ripples`` maps a mode number to the amplitude of a displacement
    ``sum(eps sin(k_m x))`` of the electrons along x.
    """
    nx, ny = int(round(L / spacing)), int(round(H / spacing))
    x = (np.arange(nx) + 0.5) * L / nx
    y = (np.arange(ny) + 0.5) * H / ny - 0.5 * H
    gx, gy = (a.ravel() for a in np.meshgrid(x, y))
    ions = np.stack([gx, gy])
    charge = DENSITY * L * H / gx.size  # per ion; each electron carries half of it
    # the displacement of the lowest bulk mode, the gradient of cos(k x) cos(pi y / H): its
    # potential vanishes at the edges of the slab, so it excites no surface mode
    shift = np.zeros((2, gx.size))
    for m, eps in ripples.items():
        k = m * K0
        shift[0] += eps * np.sin(k * gx) * np.cos(np.pi * gy / H)
        shift[1] += eps * (np.pi / (k * H)) * np.cos(k * gx) * np.sin(np.pi * gy / H)
    electrons = np.concatenate([ions + shift, ions + shift], axis=1)
    electrons[0] %= L
    velocity = np.zeros_like(electrons)
    velocity[0] = np.concatenate([np.full(gx.size, V0), np.full(gx.size, -V0)])
    return {
        "ions": ions,
        "ion_charge": np.full(gx.size, charge),
        "pos": electrons,
        "vel": velocity,
        "charge": -0.5
        * charge,  # per electron; q / m = -1, so its mass is 0.5 * charge
        "beam": np.concatenate([np.ones(gx.size), -np.ones(gx.size)]),
        "spacing": spacing,
    }


def mapped(positions: np.ndarray) -> np.ndarray:
    """``zeta = exp(i K0 (x + i y))`` as ``(2, N)`` points of the plane of the map."""
    zeta = np.exp(-K0 * positions[1]) * np.exp(1j * K0 * positions[0])
    return np.stack([zeta.real, zeta.imag])


def periodic_field(electrons, charge, ions, ion_charges, softening, keep_tree=False):
    """The field at the electrons of everything, periodic along x, by the FMM in the map.

    Returns the field ``(2, N)`` and, with ``keep_tree``, the tree of the mapped charges.
    """
    zeta_e, zeta_i = mapped(electrons), mapped(ions)
    sources = np.concatenate([zeta_e, zeta_i], axis=1)
    charges = np.concatenate([np.full(electrons.shape[1], charge), ion_charges])
    tree = FlatQuadtree(
        sources,
        charges,
        Quad(0.0, 0.0, 1.05 * float(np.abs(sources).max())),
        kernel="log",
        leaf_capacity=16,
    )
    _, field = tree.fmm(theta=0.5, softening=softening, potential=False)
    n = electrons.shape[1]
    # E_x - i E_y = (F_x - i F_y) d zeta / dz, with d zeta / dz = i K0 zeta
    complex_field = (
        (field[0, :n] - 1j * field[1, :n]) * 1j * K0 * (zeta_e[0] + 1j * zeta_e[1])
    )
    # the map counts a charge's own row as a sheet of charge -q: put it back
    e = np.stack([complex_field.real, -complex_field.imag - K0 * charge])
    return (e, tree) if keep_tree else (e, None)


class Beams:
    """Leapfrog of the electrons in the periodic field; the ions stay fixed."""

    def _acceleration(self) -> np.ndarray:
        field, self.tree = periodic_field(
            self.pos,
            self.p["charge"],
            self.p["ions"],
            self.p["ion_charge"],
            self.softening,
            keep_tree=True,
        )
        return -field  # q / m = -1

    def __init__(self, plasma: dict):
        self.p = plasma
        self.pos = plasma["pos"].copy()
        self.vel = plasma["vel"].copy()
        # a softening of half the lattice spacing, measured where |zeta| = 1 (y = 0)
        self.softening = 0.5 * K0 * plasma["spacing"]
        self.tree = None
        self.acc = self._acceleration()

    def step(self, dt: float) -> None:
        """One kick-drift-kick step."""
        self.vel += 0.5 * dt * self.acc
        self.pos += dt * self.vel
        self.pos[0] %= L
        self.acc = self._acceleration()
        self.vel += 0.5 * dt * self.acc

    def amplitude(self, mode: int) -> float:
        """The density modulation of the mode ``m``: ``|<exp(i k_m x)>|`` over the electrons."""
        return float(np.abs(np.mean(np.exp(1j * mode * K0 * self.pos[0]))))


# -- runs -------------------------------------------------------------------------------------------


def roll_up(spacing: float, t_end: float, dt: float, frames: int) -> dict:
    """The m = 2 mode seeded, with a little m = 1: snapshots in (x, v_x) and in the map."""
    beams = Beams(slab(spacing, {2: 1e-3, 1: 1e-4}))
    steps = int(round(t_end / dt))
    keep = set(np.linspace(0, steps, frames).round().astype(int))
    times, x, vx, zeta, trees = [], [], [], [], []
    for step in range(steps + 1):
        if step in keep:
            times.append(step * dt)
            x.append(beams.pos[0].copy())
            vx.append(beams.vel[0].copy())
            zeta.append(mapped(beams.pos))
            trees.append(beams.tree)
        if step < steps:
            beams.step(dt)
    markers = xr.Dataset(
        {
            "x": (("t", "marker"), np.array(x), {"long_name": "$x$"}),
            "v_x": (("t", "marker"), np.array(vx), {"long_name": "$v_x$"}),
        },
        coords={"t": ("t", np.array(times), {"long_name": "$t\\,\\omega_p$"})},
        attrs={"label": "Two-stream instability"},
    )
    return {"markers": markers, "zeta": zeta, "trees": trees, "beam": beams.p["beam"]}


def seeded_growth(mode: int, spacing: float, t_end: float, dt: float):
    """The amplitude of the mode ``m`` of a slab rippled in that mode only."""
    beams = Beams(slab(spacing, {mode: 1e-3}))
    steps = int(round(t_end / dt))
    times, amplitude = [], []
    for step in range(steps + 1):
        times.append(step * dt)
        amplitude.append(beams.amplitude(mode))
        if step < steps:
            beams.step(dt)
    return np.array(times), np.array(amplitude)


#: A mode that has not tripled by this time counts as stable.
T_LINEAR = 15.0


def fit_rate(times: np.ndarray, amplitude: np.ndarray) -> float:
    """The exponential growth rate from three times the seed to 0.05 (0 for a stable mode)."""
    tripled = np.flatnonzero(amplitude > 3.0 * amplitude[0])
    if not tripled.size or times[tripled[0]] > T_LINEAR:
        return 0.0
    saturated = np.flatnonzero(amplitude > 0.05)
    end = saturated[0] if saturated.size else amplitude.size
    if end - tripled[0] < 5:
        return 0.0
    window = slice(tripled[0], end)
    return float(np.polyfit(times[window], np.log(amplitude[window]), 1)[0])


# -- figures -----------------------------------------------------------------------------------------


def plot_phase_space(run: dict, output: Path) -> None:
    """Figure 1: the electrons in (x, v_x), by the markers accessor of plasma-plots."""
    markers = run["markers"].assign(
        beam=(("t", "marker"), np.broadcast_to(run["beam"], run["markers"]["x"].shape))
    )
    result = markers.plasma.plot.animation(
        x="x",
        y="v_x",
        color="beam",
        color_at=0,
        s=2,
        cmap="coolwarm",
        aspect="auto",
        interval=100,
        backend="plotly",
    )
    # the poster shows the vortices at their clearest
    poster = int(np.argmin(np.abs(markers["t"].values - 30.0)))
    save_figure(result, output / "phase_space", frame=poster, width=900, height=620)


def plot_mapped_tree(run: dict, output: Path) -> None:
    """Figure 2: the run in the plane of the map, with the FMM's tree, by ``animate_quadtree``."""
    reach = 1.05 * np.exp(0.5 * K0 * H)
    result = animate_quadtree(
        run["trees"],
        points=run["zeta"],
        point_color=run["beam"],
        times=run["markers"]["t"].values,
        extent=(-reach, reach, -reach, reach),
        leaves_only=True,
        s=2,
        interval=100,
        title="t ω_p = {time}: {leaves} leaf cells in the plane of the map",
        backend="plotly",
    )
    poster = int(np.argmin(np.abs(run["markers"]["t"].values - 30.0)))
    save_figure(result, output / "mapped_tree", frame=poster, width=720, height=760)


def plot_mode_growth(curves: dict, output: Path) -> None:
    """Figure 3: the amplitude of each seeded mode, with the cold-beam slopes."""
    fig = go.Figure()
    for color, (mode, (times, amplitude)) in zip(PALETTE, curves.items()):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=amplitude,
                mode="lines",
                name=f"m = {mode}, k = {mode * K0:.2f}",
                line={"color": color},
            )
        )
        gamma = float(growth_rate(mode * K0))
        if gamma > 0:
            anchor = int(np.argmin(np.abs(np.log(amplitude / 1e-2))))
            line = amplitude[anchor] * np.exp(gamma * (times - times[anchor]))
            shown = (line > amplitude[0]) & (line < 0.3)
            fig.add_trace(
                go.Scatter(
                    x=times[shown],
                    y=line[shown],
                    mode="lines",
                    showlegend=False,
                    line={"color": color, "dash": "dash", "width": 1},
                )
            )
    fig.update_xaxes(title_text="$t\\,\\omega_p$")
    fig.update_yaxes(
        type="log",
        exponentformat="power",
        title_text="density modulation |⟨exp(i k x)⟩|",
    )
    fig.update_layout(title="Seeded modes against the cold-beam growth (dashed)")
    save_figure(fig, output / "mode_growth")


def plot_growth_rates(curves: dict, output: Path) -> None:
    """Figure 4: the fitted growth rates against the dispersion relation."""
    k = np.linspace(0.0, 1.3, 400)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k,
            y=growth_rate(k),
            mode="lines",
            name="cold beams (theory)",
            line={"color": "black"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[m * K0 for m in curves],
            y=[fit_rate(*c) for c in curves.values()],
            mode="markers",
            name="tree code (FMM)",
            marker={"size": 10, "color": PALETTE[1]},
        )
    )
    fig.update_xaxes(title_text="wavenumber $k v_0 / \\omega_p$")
    fig.update_yaxes(title_text="growth rate $\\gamma / \\omega_p$", rangemode="tozero")
    fig.update_layout(title="The two-stream dispersion relation")
    save_figure(fig, output / "growth_rates")


# -- main ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run the seeded modes and the roll-up, write the figures, return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        spacing, t_growth, t_roll, frames, dt = 1.0, 2.0, 2.0, 5, 0.1
    elif args.quick:
        spacing, t_growth, t_roll, frames, dt = 0.25, 30.0, 45.0, 60, 0.05
    else:
        spacing, t_growth, t_roll, frames, dt = 0.125, 30.0, 45.0, 70, 0.025

    print("1. and 2. roll-up")
    run = roll_up(spacing, t_roll, dt, frames)
    print(f"   {run['beam'].size} electrons")
    plot_phase_space(run, output)
    plot_mapped_tree(run, output)

    print("3. seeded modes")
    curves = {}
    for mode in MODES:
        curves[mode] = seeded_growth(mode, spacing, t_growth, dt)
        print(
            f"   m = {mode}: growth rate {fit_rate(*curves[mode]):.3f}, "
            f"theory {float(growth_rate(mode * K0)):.3f}"
        )
    plot_mode_growth(curves, output)

    print("4. growth rates")
    plot_growth_rates(curves, output)
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
