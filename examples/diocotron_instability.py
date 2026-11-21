"""Diocotron instability.

A hollow electron column in a strong magnetic field rolls up into vortices; the tree follows them.

In a strong axial field ``B`` the electrons of a nonneutral plasma move with the E x B drift
``v = E x B / B^2`` of their guiding centres: a 2D incompressible flow whose vorticity is the
charge density, the same dynamics as an ideal fluid's (2D Euler). A thin annulus of charge
between the radii ``a-`` and ``a+`` is unstable, like a shear layer (Kelvin-Helmholtz): its
edges ripple, the ripples grow and the ring breaks up into vortices that orbit and merge.
For a uniform annulus with no wall, the azimuthal mode ``l`` grows at
``gamma / omega_D = sqrt(c_l - b_l^2 / 4)``, with ``b_l = l (1 - x^2)``,
``c_l = l (1 - x^2) - (1 - x^(2 l))``, ``x = a- / a+`` and ``omega_D`` the E x B rotation
frequency of a uniform column of the annulus density (Davidson, *Physics of Nonneutral
Plasmas*, ch. 6). The guiding centres are line charges (``kernel="log"``); their field comes
from the FMM of ``FlatQuadtree.fmm`` and they are pushed with the classic fourth-order
Runge-Kutta method. Units ``k_e = B = omega_D = 1``, ``a+ = 1``, ``a- = 0.8``, particles
started on rings (a quiet start, so that only the seeded mode grows).

1. ``roll_up``: the annulus seeded with a small random ripple, rolling up into vortices, with
   the cells of the quadtree the FMM builds every step; the particles are colored by their
   starting radius (blue at the inner edge, red at the outer), so the vortices show their
   layers wound up into spirals. The tree refines into the vortices and coarsens in the empty
   space between them.
2. ``mode_growth``: the amplitude of the seeded mode against time, one run per mode
   ``l = 2 ... 6`` and ``l = 8``, measured by the multipole moment ``|<(z / a+)^l>|`` of the charge, with
   the slope of the theory's growth rate (dashed). ``l = 2`` and ``l = 8`` are stable (``l = 7`` is only
   marginally so, and left out): they
   only oscillate, until the unstable modes seeded by the discreteness of the particles grow
   large enough to drive them nonlinearly.
3. ``growth_rates``: the growth rates fitted to the runs (from three times the seed to 0.1;
   a mode that has not tripled by ``t omega_D = 15`` counts as stable) against the theory's.
4. ``invariants``: the electrostatic energy and the angular momentum (the sum of ``q r^2``),
   which the exact guiding-centre dynamics conserves, during the roll-up: their drift
   measures the FMM's and the time step's errors.

Usage::

    python examples/diocotron_instability.py [--output examples/output] [--quick]

``--quick`` uses fewer particles and a shorter roll-up (about a minute); ``--smoke`` is the
tiny configuration the test suite runs. Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import animate_quadtree, save_figure

from tree_d_3d import FlatQuadtree, Quad

#: The radii of the annulus.
A_IN, A_OUT = 0.8, 1.0
#: The charge density: omega_D = 2 pi k_e rho / B = 1.
RHO = 1.0 / (2.0 * np.pi)
MODES = (2, 3, 4, 5, 6, 8)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def growth_rate(mode: int, x: float = A_IN / A_OUT) -> float:
    """The theory's growth rate of the mode ``l`` of a uniform annulus, in ``omega_D``."""
    b = mode * (1.0 - x**2)
    c = mode * (1.0 - x**2) - (1.0 - x ** (2 * mode))
    return float(np.sqrt(max(c - 0.25 * b * b, 0.0)))


def annulus(n: int, mode: int = 0, ripple: float = 0.0, seed: int | None = None):
    """About ``n`` guiding centres on rings filling the annulus, the edges rippled.

    With ``mode`` every radius is stretched by ``1 + ripple cos(mode theta)``; with ``seed``
    instead by a random combination of the modes 1 to 12 of size ``ripple``. Returns the
    positions, the charges, the starting radii and the ring spacing.
    """
    area = np.pi * (A_OUT**2 - A_IN**2)
    spacing = float(np.sqrt(area / n))
    rings = max(int(round((A_OUT - A_IN) / spacing)), 2)
    radii = A_IN + (np.arange(rings) + 0.5) * (A_OUT - A_IN) / rings
    if seed is None:

        def ripples(theta):
            return ripple * np.cos(mode * theta)
    else:
        rng = np.random.default_rng(seed)
        amplitudes = rng.normal(size=12) / np.sqrt(12)
        phases = rng.uniform(0, 2 * np.pi, 12)

        def ripples(theta):
            return ripple * sum(
                a * np.cos((m + 1) * theta + p)
                for m, (a, p) in enumerate(zip(amplitudes, phases))
            )

    columns, starts = [], []
    for radius in radii:
        count = int(round(2.0 * np.pi * radius / spacing))
        theta = 2.0 * np.pi * (np.arange(count) + 0.5) / count
        r = radius * (1.0 + ripples(theta))
        columns.append(np.stack([r * np.cos(theta), r * np.sin(theta)]))
        starts.append(np.full(count, radius))
    positions = np.concatenate(columns, axis=1)
    charges = np.full(positions.shape[1], RHO * area / positions.shape[1])
    return positions, charges, np.concatenate(starts), spacing


class GuidingCentres:
    """E x B drift of line charges in a uniform ``B_z = 1``, with the field from the FMM."""

    def __init__(self, positions, charges, softening: float, leaf_capacity: int = 16):
        self.positions = positions.copy()
        self.charges = charges
        self.softening = softening
        self.leaf_capacity = leaf_capacity
        self.tree: FlatQuadtree | None = None

    def _drift(self, positions: np.ndarray, keep: bool = False):
        half = 1.05 * float(np.abs(positions).max())
        tree = FlatQuadtree(
            positions,
            self.charges,
            Quad(0.0, 0.0, half),
            kernel="log",
            leaf_capacity=self.leaf_capacity,
        )
        if keep:
            self.tree = tree
        _, field = tree.fmm(theta=0.5, softening=self.softening, potential=False)
        return np.stack([field[1], -field[0]])  # E x z: v = (E_y, -E_x) / B

    def step(self, dt: float) -> None:
        """One fourth-order Runge-Kutta step; keeps the tree of its first stage."""
        x = self.positions
        k1 = self._drift(x, keep=True)
        k2 = self._drift(x + 0.5 * dt * k1)
        k3 = self._drift(x + 0.5 * dt * k2)
        k4 = self._drift(x + dt * k3)
        self.positions = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def amplitude(self, mode: int) -> float:
        """The size of the mode ``l`` of the charge, its multipole moment ``|<z^l>|``.

        ``z = (x + i y) / a+``; a ripple of the edges in that mode makes it nonzero from the
        start, unlike the density moment ``<exp(i l theta)>``.
        """
        z = (self.positions[0] + 1j * self.positions[1]) / A_OUT
        return float(np.abs(np.mean(z**mode)))

    def invariants(self) -> tuple[float, float]:
        """The electrostatic energy and the angular momentum ``sum(q r^2)``."""
        tree = FlatQuadtree(
            self.positions,
            self.charges,
            Quad(0.0, 0.0, 1.05 * float(np.abs(self.positions).max())),
            kernel="log",
        )
        phi, _ = tree.fmm(theta=0.5, softening=self.softening)
        return (
            0.5 * float(np.sum(self.charges * phi)),
            float(np.sum(self.charges * np.sum(self.positions**2, axis=0))),
        )


# -- 1 and 4: the roll-up ------------------------------------------------------------------


def roll_up(n: int, t_end: float, dt: float, frames: int) -> dict:
    """The annulus with a random ripple, run to ``t_end``; snapshots and invariants."""
    positions, charges, starts, spacing = annulus(n, ripple=2e-3, seed=7)
    system = GuidingCentres(positions, charges, softening=0.5 * spacing)
    steps = int(round(t_end / dt))
    keep = set(np.linspace(0, steps, frames).round().astype(int))
    snapshots, times, energy, momentum = [], [], [], []
    every = max(steps // 100, 1)
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            e, p = system.invariants()
            times.append(step * dt)
            energy.append(e)
            momentum.append(p)
        if step in keep:
            system._drift(system.positions, keep=True)  # the tree of these positions
            snapshots.append(
                {"t": step * dt, "pos": system.positions.copy(), "tree": system.tree}
            )
        if step < steps:
            system.step(dt)
    return {
        "snapshots": snapshots,
        "starts": starts,
        "n": positions.shape[1],
        "t": np.array(times),
        "energy": np.array(energy),
        "momentum": np.array(momentum),
    }


def plot_roll_up(run: dict, output: Path) -> None:
    """Figure 1: the roll-up with the tree of every frame, by ``animate_quadtree``."""
    snaps = run["snapshots"]
    middle = 0.5 * (A_IN + A_OUT)
    result = animate_quadtree(
        [s["tree"] for s in snaps],
        points=[s["pos"] for s in snaps],
        point_color=run["starts"] - middle,
        times=[s["t"] for s in snaps],
        extent=(-1.35, 1.35, -1.35, 1.35),
        leaves_only=True,
        s=3,
        interval=120,
        title="t ω_D = {time}: {leaves} leaf cells, depth {depth}",
        backend="plotly",
    )
    save_figure(result, output / "roll_up", frame=-1, width=720, height=760)


def plot_invariants(run: dict, output: Path) -> None:
    """Figure 4: the relative drift of the energy and the angular momentum."""
    fig = go.Figure()
    for values, name, color in (
        (run["energy"], "electrostatic energy", PALETTE[0]),
        (run["momentum"], "angular momentum Σ q r²", PALETTE[3]),
    ):
        drift = np.abs(values / values[0] - 1.0)
        fig.add_trace(
            go.Scatter(
                x=run["t"][1:],
                y=drift[1:],
                mode="lines",
                name=name,
                line={"color": color},
            )
        )
    fig.update_xaxes(title_text="$t\\,\\omega_D$")
    fig.update_yaxes(type="log", exponentformat="power", title_text="relative drift")
    fig.update_layout(title=f"Conserved quantities during the roll-up (N = {run['n']})")
    save_figure(fig, output / "invariants")


# -- 2 and 3: linear growth -----------------------------------------------------------------


def seeded_growth(
    mode: int, n: int, t_end: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """The amplitude of the mode ``l`` of an annulus rippled in that mode only."""
    positions, charges, _, spacing = annulus(n, mode=mode, ripple=1e-3)
    system = GuidingCentres(
        positions, charges, softening=0.5 * spacing, leaf_capacity=32
    )
    steps = int(round(t_end / dt))
    times, amplitude = [], []
    for step in range(steps + 1):
        times.append(step * dt)
        amplitude.append(system.amplitude(mode))
        if step < steps:
            system.step(dt)
    return np.array(times), np.array(amplitude)


#: A mode that has not tripled by this time counts as stable: later growth comes from the
#: unstable modes the discreteness seeds, coupled into it nonlinearly.
T_LINEAR = 15.0


def fit_rate(times: np.ndarray, amplitude: np.ndarray) -> float:
    """The exponential growth rate from three times the seed to 0.1 (0 for a stable mode)."""
    tripled = np.flatnonzero(amplitude > 3.0 * amplitude[0])
    if not tripled.size or times[tripled[0]] > T_LINEAR:
        return 0.0
    saturated = np.flatnonzero(amplitude > 0.1)
    end = saturated[0] if saturated.size else amplitude.size
    window = slice(tripled[0], end)
    if end - tripled[0] < 5:
        return 0.0
    return float(np.polyfit(times[window], np.log(amplitude[window]), 1)[0])


def plot_mode_growth(curves: dict, output: Path) -> None:
    """Figure 2: the amplitude of each seeded mode, with the theory's slopes."""
    fig = go.Figure()
    for color, (mode, (times, amplitude)) in zip(PALETTE, curves.items()):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=amplitude,
                mode="lines",
                name=f"l = {mode}",
                line={"color": color},
            )
        )
        gamma = growth_rate(mode)
        if gamma > 0:
            # the theory's slope, through the run at amplitude 0.01
            anchor = int(np.argmin(np.abs(np.log(amplitude / 1e-2))))
            line = amplitude[anchor] * np.exp(gamma * (times - times[anchor]))
            shown = (line > 2e-4) & (line < 0.3)
            fig.add_trace(
                go.Scatter(
                    x=times[shown],
                    y=line[shown],
                    mode="lines",
                    showlegend=False,
                    line={"color": color, "dash": "dash", "width": 1},
                )
            )
    fig.update_xaxes(title_text="$t\\,\\omega_D$")
    fig.update_yaxes(
        type="log", exponentformat="power", title_text="mode amplitude |⟨(z / a₊)ˡ⟩|"
    )
    fig.update_layout(title="Seeded modes against the theory's growth (dashed)")
    save_figure(fig, output / "mode_growth")


def plot_growth_rates(curves: dict, output: Path) -> None:
    """Figure 3: the fitted growth rates against the theory's."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(curves),
            y=[growth_rate(m) for m in curves],
            mode="markers",
            name="theory (Davidson)",
            marker={"symbol": "circle-open", "size": 14, "color": PALETTE[3]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(curves),
            y=[fit_rate(*curve) for curve in curves.values()],
            mode="markers",
            name="tree code (FMM)",
            marker={"size": 8, "color": PALETTE[0]},
        )
    )
    fig.update_xaxes(title_text="azimuthal mode number $l$", dtick=1)
    fig.update_yaxes(title_text="growth rate $\\gamma / \\omega_D$", rangemode="tozero")
    fig.update_layout(title=f"Growth rates of a thin annulus, a-/a+ = {A_IN / A_OUT:g}")
    save_figure(fig, output / "growth_rates")


# -- main -----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run the growth study and the roll-up, write the figures, return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        n_growth, t_growth, n_roll, t_roll, frames, dt = 400, 2.0, 400, 2.0, 5, 0.1
    elif args.quick:
        n_growth, t_growth, n_roll, t_roll, frames, dt = (
            2500,
            36.0,
            3000,
            60.0,
            60,
            0.05,
        )
    else:
        n_growth, t_growth, n_roll, t_roll, frames, dt = (
            6000,
            36.0,
            6000,
            80.0,
            70,
            0.05,
        )

    print("1. roll-up")
    run = roll_up(n_roll, t_roll, dt, frames)
    plot_roll_up(run, output)

    print("2. seeded modes")
    curves = {}
    for mode in MODES:
        curves[mode] = seeded_growth(mode, n_growth, t_growth, dt)
        print(
            f"   l = {mode}: growth rate {fit_rate(*curves[mode]):.3f}, "
            f"theory {growth_rate(mode):.3f}"
        )
    plot_mode_growth(curves, output)

    print("3. growth rates")
    plot_growth_rates(curves, output)

    print("4. invariants")
    plot_invariants(run, output)
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
