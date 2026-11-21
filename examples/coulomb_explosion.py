"""Coulomb explosion.

A cold cloud of charge of one sign blows itself apart, and the tree follows it outward.

A cylinder of ions with no electrons to hold it together accelerates outward under its own
space charge. With line charges (``kernel="log"``, the 2D field ``2 k_e Q(r) / r`` of the
charge ``Q(r)`` inside a radius) a cold shell starting at ``r0`` keeps its enclosed charge
until another shell overtakes it, and then moves exactly: ``r = r0 f(t)`` with
``f'' = kappa / f``, ``kappa = 2 k_e (q/m) Q(r0) / r0^2``, so that
``t = sqrt(2 / kappa) (sqrt(pi) / 2) erfi(sqrt(ln f))``. A uniform disk has the same
``kappa`` for every shell: it expands self-similarly and stays uniform. A Gaussian cloud has a
larger ``kappa`` inside than outside, so the inner shells catch up with the outer ones: the
flow folds over in phase space (wave breaking) and a dense shell, a collisionless shock,
runs out at the edge of the cloud. Units ``k_e = q/m = 1``, total charge 1, particles started
on rings of equal charge (a quiet start); the fields come from the FMM of
``ElectrostaticSimulation(kernel="log", method="fmm")``.

1. ``gaussian_explosion``: the Gaussian cloud exploding, with the cells of the quadtree the
   FMM builds every step; the particles are colored by their starting radius (blue inside,
   red outside, by the charge inside their starting
   radius), so the inner shells can be seen overtaking the outer ones. The tree is
   deepest where the charge piles up, in the shock.
2. ``uniform_expansion``: the rms radius of the uniform disk against the exact self-similar
   solution, and their relative difference (the FMM's truncation and the softening).
3. ``phase_space``: radius against radial velocity of the Gaussian cloud at four times,
   against the cold single-stream solution. Before the breaking time the particles lie on
   the curve; after it the exact curve folds back on itself (three streams at one radius),
   and the particles that have crossed feel a changed enclosed charge, so they leave it.
4. ``density_profile``: the charge density of the Gaussian cloud against radius, at the
   start and at the end: the peak at the edge is the shock.

Usage::

    python examples/coulomb_explosion.py [--output examples/output] [--quick]

``--quick`` uses fewer particles (well under a minute); ``--smoke`` is the tiny configuration
the test suite runs. Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import animate_quadtree, save_figure
from plotly.subplots import make_subplots
from scipy.special import erfi

from tree_d_3d import ChargedBody, ElectrostaticSimulation, Quad

#: The width of the Gaussian cloud, and the radius it is cut at (in widths).
SIGMA, CUT = 0.25, 4.0
#: The radius of the uniform disk.
RADIUS = 1.0
COLORS = {"sim": "#1f77b4", "theory": "#d62728", "gray": "rgb(150,150,150)"}


# -- the exact cold-shell solution -----------------------------------------------------------


def enclosed(r0: np.ndarray, profile: str) -> np.ndarray:
    """The fraction of the charge inside the radius ``r0`` at the start."""
    r0 = np.asarray(r0, dtype=float)
    if profile == "uniform":
        return np.clip(r0 / RADIUS, 0.0, 1.0) ** 2
    cut = 1.0 - np.exp(-0.5 * CUT**2)
    return np.clip((1.0 - np.exp(-0.5 * (r0 / SIGMA) ** 2)) / cut, 0.0, 1.0)


def kappa(r0: np.ndarray, profile: str) -> np.ndarray:
    """The acceleration constant ``2 Q(r0) / r0^2`` of the shell starting at ``r0``."""
    r0 = np.asarray(r0, dtype=float)
    return 2.0 * enclosed(r0, profile) / r0**2


_F = np.geomspace(1.0, 1e4, 4000)
_TAU = 0.5 * np.sqrt(np.pi) * erfi(np.sqrt(np.log(_F)))


def stretch(time: float | np.ndarray, k: np.ndarray) -> np.ndarray:
    """The expansion factor ``f`` of shells of constant ``kappa = k`` at ``time``."""
    return np.interp(np.asarray(time) * np.sqrt(0.5 * np.asarray(k)), _TAU, _F)


def shell_state(r0: np.ndarray, time: float, profile: str):
    """Radius and radial velocity of cold shells starting at ``r0``, before they cross."""
    k = kappa(r0, profile)
    f = stretch(time, k)
    return r0 * f, r0 * np.sqrt(2.0 * k * np.log(f))


def breaking_time(profile: str = "gaussian") -> float:
    """The first time two shells of the exact solution meet (``inf`` if never)."""
    r0 = np.linspace(1e-3, CUT * SIGMA, 2000)
    for time in np.linspace(0.0, 5.0, 2001)[1:]:
        if np.any(np.diff(shell_state(r0, time, profile)[0]) < 0):
            return float(time)
    return float("inf")


# -- the simulation -----------------------------------------------------------------------------


def quiet_cloud(n: int, profile: str) -> tuple[np.ndarray, np.ndarray]:
    """About ``n`` positions on rings of equal charge, and each particle's starting radius.

    Every ring holds the same number of particles at an angle offset by half a spacing from
    the ring inside, so there is no random noise to seed shell crossing early.
    """
    rings = max(int(np.sqrt(n / np.pi)), 2)
    per_ring = max(n // rings, 3)
    share = (np.arange(rings) + 0.5) / rings  # the enclosed charge at each ring
    if profile == "uniform":
        radii = RADIUS * np.sqrt(share)
    else:
        cut = 1.0 - np.exp(-0.5 * CUT**2)
        radii = SIGMA * np.sqrt(-2.0 * np.log(1.0 - share * cut))
    angles = 2.0 * np.pi * np.arange(per_ring) / per_ring
    columns = [
        r
        * np.stack(
            [
                np.cos(angles + np.pi * j / per_ring),
                np.sin(angles + np.pi * j / per_ring),
            ]
        )
        for j, r in enumerate(radii)
    ]
    positions = np.concatenate(columns, axis=1)
    return positions, np.repeat(radii, per_ring)


def explode(
    n: int,
    profile: str,
    t_end: float,
    dt: float,
    frames: int = 0,
    leaf_capacity: int = 16,
) -> dict:
    """Run a cloud of ``n`` particles to ``t_end``; keep the rms radius and some snapshots."""
    positions, r0 = quiet_cloud(n, profile)
    count = positions.shape[1]
    bodies = [
        ChargedBody(x, y, mass=1.0 / count, charge=1.0 / count) for x, y in positions.T
    ]
    spacing = float(np.sqrt(np.pi) * r0.max() / np.sqrt(count))
    sim = ElectrostaticSimulation(
        bodies,
        Quad(0.0, 0.0, 1.05 * float(r0.max())),
        theta=0.5,
        k_e=1.0,
        softening=0.2 * spacing,
        dt=dt,
        kernel="log",
        method="fmm",
        leaf_capacity=leaf_capacity,
    )
    steps = int(round(t_end / dt))
    keep = set(np.linspace(0, steps, frames).round().astype(int)) if frames else set()
    times, rms, snapshots = [], [], []
    tree = sim.build_tree()
    for step in range(steps + 1):
        times.append(sim.current_time)
        rms.append(float(np.sqrt(np.mean(np.sum(sim.positions**2, axis=0)))))
        if step in keep:
            snapshots.append(
                {
                    "t": sim.current_time,
                    "pos": sim.positions.copy(),
                    "vel": sim.velocities.copy(),
                    "tree": tree,
                }
            )
        if step < steps:
            tree = sim.step()
    return {
        "t": np.array(times),
        "rms": np.array(rms),
        "r0": r0,
        "snapshots": snapshots,
        "n": count,
    }


# -- figures ------------------------------------------------------------------------------------


def plot_gaussian_explosion(run: dict, output: Path, extent: float) -> None:
    """Figure 1: the explosion with the tree of every frame, by ``animate_quadtree``."""
    snaps = run["snapshots"]
    result = animate_quadtree(
        [s["tree"] for s in snaps],
        points=[s["pos"] for s in snaps],
        point_color=enclosed(run["r0"], "gaussian") - 0.5,
        times=[s["t"] for s in snaps],
        extent=(-extent, extent, -extent, extent),
        leaves_only=True,
        s=3,
        interval=120,
        title="t = {time}: {leaves} leaf cells, depth {depth}",
        backend="plotly",
    )
    save_figure(result, output / "gaussian_explosion", frame=-1, width=720, height=760)


def plot_uniform_expansion(run: dict, output: Path) -> None:
    """Figure 2: the rms radius of the uniform disk against the exact solution."""
    t, rms = run["t"], run["rms"]
    exact = rms[0] * stretch(t, kappa(RADIUS, "uniform"))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(
        go.Scatter(
            x=t[::5],
            y=rms[::5],
            mode="markers",
            name="tree code (FMM)",
            marker={"color": COLORS["sim"], "size": 5},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=exact, mode="lines", name="exact", line={"color": COLORS["theory"]}
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t[1:],
            y=np.abs(rms[1:] / exact[1:] - 1.0),
            mode="lines",
            showlegend=False,
            line={"color": COLORS["sim"]},
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="rms radius", row=1, col=1)
    fig.update_yaxes(
        type="log",
        exponentformat="power",
        title_text="relative difference",
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="$t$", row=2, col=1)
    fig.update_layout(
        title=f"A uniform disk expands self-similarly (N = {run['n']})", height=640
    )
    save_figure(fig, output / "uniform_expansion", height=640)


def plot_phase_space(run: dict, output: Path, t_break: float) -> None:
    """Figure 3: radius against radial velocity, simulation and the cold-shell solution."""
    snaps = run["snapshots"]
    picks = [
        snaps[i] for i in np.linspace(0, len(snaps) - 1, 5).round().astype(int)[1:]
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"t = {s['t']:.2f}" for s in picks],
        horizontal_spacing=0.1,
        vertical_spacing=0.14,
    )
    r0_line = np.linspace(1e-3, CUT * SIGMA, 600)
    rng = np.random.default_rng(0)
    for i, snap in enumerate(picks):
        row, col = i // 2 + 1, i % 2 + 1
        pos, vel = snap["pos"], snap["vel"]
        r = np.hypot(pos[0], pos[1])
        v_r = np.sum(pos * vel, axis=0) / r
        shown = rng.choice(r.size, min(r.size, 1500), replace=False)
        fig.add_trace(
            go.Scattergl(
                x=r[shown],
                y=v_r[shown],
                mode="markers",
                name="tree code",
                marker={"size": 3, "color": COLORS["sim"]},
                showlegend=i == 0,
            ),
            row=row,
            col=col,
        )
        r_exact, v_exact = shell_state(r0_line, snap["t"], "gaussian")
        fig.add_trace(
            go.Scatter(
                x=r_exact,
                y=v_exact,
                mode="lines",
                name="cold shells, no crossing",
                line={"color": COLORS["theory"], "width": 1.5},
                showlegend=i == 0,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="$r$" if row == 2 else None, row=row, col=col)
        fig.update_yaxes(title_text="$v_r$" if col == 1 else None, row=row, col=col)
    fig.update_layout(
        title=f"Wave breaking at t = {t_break:.2f}: the flow folds over", height=720
    )
    save_figure(fig, output / "phase_space", height=720)


def plot_density_profile(run: dict, output: Path) -> None:
    """Figure 4: the charge density against radius at the start and at the end."""
    first, last = run["snapshots"][0], run["snapshots"][-1]
    fig = go.Figure()
    for snap, color, label in (
        (first, COLORS["gray"], f"t = {first['t']:.2f}"),
        (last, COLORS["sim"], f"t = {last['t']:.2f}"),
    ):
        r = np.hypot(*snap["pos"])
        edges = np.linspace(0.0, r.max() * 1.02, 60)
        counts, _ = np.histogram(r, edges)
        density = counts / run["n"] / (np.pi * np.diff(edges**2))
        centres = 0.5 * (edges[1:] + edges[:-1])
        fig.add_trace(
            go.Scatter(
                x=centres,
                y=density / density[: len(density) // 4].mean(),
                mode="lines",
                line={"color": color, "shape": "hvh"},
                name=label,
            )
        )
    fig.update_xaxes(title_text="$r$")
    fig.update_yaxes(title_text="charge density / central density")
    fig.update_layout(title="The shock: charge piles up at the edge")
    save_figure(fig, output / "density_profile")


# -- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run both explosions, write the figures and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    n, frames = (300, 6) if args.smoke else (2500, 50) if args.quick else (6000, 60)
    t_end = 0.6 if args.smoke else 2.0

    t_break = breaking_time("gaussian")
    print(f"cold-shell breaking time of the Gaussian cloud: t = {t_break:.3f}")

    print("1. Gaussian cloud")
    gaussian = explode(n, "gaussian", t_end, dt=0.004, frames=frames)
    extent = 1.05 * float(np.abs(gaussian["snapshots"][-1]["pos"]).max())
    plot_gaussian_explosion(gaussian, output, extent)

    print("2. uniform disk")
    uniform = explode(n, "uniform", 3.0 if not args.smoke else 0.5, dt=0.01)
    exact = (
        uniform["rms"][-1] / (uniform["rms"][0] * stretch(uniform["t"][-1], 2.0)) - 1
    )
    print(
        f"   rms radius at t = {uniform['t'][-1]:.1f}: relative difference {exact:.1e}"
    )
    plot_uniform_expansion(uniform, output)

    print("3. phase space")
    plot_phase_space(gaussian, output, t_break)

    print("4. density profile")
    plot_density_profile(gaussian, output)
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
