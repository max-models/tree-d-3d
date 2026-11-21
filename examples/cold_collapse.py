"""Cold collapse.

A cold disk of stars falls in on itself, bounces and relaxes; spun up, it breaks into clumps.

The Maclaurin (Kalnajs) disk, of surface density ``Sigma0 sqrt(1 - r^2 / R^2)``, is the one
flat mass distribution whose gravity is exactly linear inside it: the pull towards the centre
is ``Omega0^2 r`` with ``Omega0^2 = pi^2 G Sigma0 / (2 R)``. Started at rest, every star
falls as ``r0 cos(Omega0 t)`` and the whole disk reaches the centre at once, at
``t = pi / (2 Omega0)``. But a cold disk is gravitationally unstable on every scale, so the
graininess of ``N`` stars grows during the fall (the stars start on rings of equal mass to keep
it small) and breaks the homology before the end: the disk ends in a violent bounce, throws
out about a third of its mass, and the bound rest trades energy in the fluctuating potential
until it is in virial equilibrium, ``2 K = -sum(m r . a)``, with a dense core and a halo
(violent relaxation, Lynden-Bell 1967). Spun at
``Omega0`` the same disk is in exact rotational equilibrium, but it is cold (Toomre's
``Q = 0``), so every wavelength is gravitationally unstable: it breaks up into clumps that
spiral together. The bodies are point masses in a plane with the ``1/r^2`` force, by the
Barnes-Hut walk of ``FlatQuadtree`` (``theta = 0.5``, Plummer softening 0.03) and the
kick-drift-kick leapfrog.
Units ``G = M = R = 1``, so ``Omega0 = sqrt(3 pi / 4)`` and the collapse time is 1.023.

1. ``collapse``: the cold disk falling in, bouncing and relaxing, with the quadtree of every
   frame: the cells shrink with the disk, reach their deepest at the bounce and spread out
   into the halo.
2. ``fragmentation``: the rotating disk breaking into clumps, with its quadtree: each clump
   becomes a knot of deep cells.
3. ``homologous_fall``: the median radius of the collapsing disk against the exact
   ``cos(Omega0 t)``: it follows it closely for the first half of the fall, then the growing
   clumps pull the stars in faster; after the bounce the core stays a tenth of the disk.
4. ``virial_ratio``: the virial ratio ``2 K / |sum(m r . a)|`` of the bound stars of both
   disks (exact for the softened forces too, unlike ``2 K / |W|``), and the bound fraction of
   the mass: the collapsing disk overshoots and settles to 1 having lost a third of its mass;
   the rotating disk starts at 1 (rotation supports it) and stays near it while its ordered
   rotation turns into random motion in the clumps.

Usage::

    python examples/cold_collapse.py [--output examples/output] [--quick]

``--quick`` uses fewer stars (well under a minute); ``--smoke`` is the tiny configuration the
test suite runs. Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import animate_quadtree, save_figure
from plotly.subplots import make_subplots

from tree_d_3d import FlatQuadtree, Quad

#: The angular frequency of the Maclaurin disk of unit mass and radius, and its collapse time.
OMEGA0 = float(np.sqrt(3.0 * np.pi / 4.0))
T_COLLAPSE = 0.5 * np.pi / OMEGA0
THETA = 0.5
COLORS = {"collapse": "#1f77b4", "rotating": "#d62728", "theory": "black"}


def maclaurin_disk(n: int, spin: float, quiet: bool = False, seed: int = 0):
    """About ``n`` stars of a unit Maclaurin disk, turning at ``spin * Omega0`` (0: at rest).

    ``quiet`` puts them on rings of equal mass, evenly spaced along each ring, instead of at
    random, to seed the disk's instability as little as possible.
    """
    if quiet:
        rings = max(int(np.sqrt(n)), 2)
        per_ring = max(n // rings, 3)
        share = (np.arange(rings) + 0.5) / rings
        r = np.repeat(np.sqrt(1.0 - (1.0 - share) ** (2.0 / 3.0)), per_ring)
        steps = np.tile(np.arange(per_ring), rings) + 0.5 * (
            np.arange(r.size) // per_ring % 2
        )
        phi = 2.0 * np.pi * steps / per_ring
    else:
        rng = np.random.default_rng(seed)
        # the mass inside r is 1 - (1 - r^2)^(3/2): invert it
        r = np.sqrt(1.0 - (1.0 - rng.uniform(0.0, 1.0, n)) ** (2.0 / 3.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
    position = r * np.stack([np.cos(phi), np.sin(phi)])
    velocity = spin * OMEGA0 * np.stack([-position[1], position[0]])
    return position, velocity, np.full(r.size, 1.0 / r.size)


class Gravity:
    """Leapfrog for self-gravitating point masses with the Barnes-Hut tree of ``FlatQuadtree``."""

    def _forces(self):
        half = 1.05 * float(np.abs(self.pos).max())
        tree = FlatQuadtree(self.pos, self.mass, Quad(0.0, 0.0, half), leaf_capacity=8)
        phi, field = tree.evaluate(
            self.pos, exclude=self.others, theta=THETA, softening=self.softening
        )
        return -np.asarray(field), np.asarray(phi), tree  # gravity attracts

    def __init__(self, positions, velocities, masses, softening: float):
        self.pos = positions.copy()
        self.vel = velocities.copy()
        self.mass = masses
        self.softening = softening
        self.others = np.arange(masses.size)
        self.acc, self.phi, self.tree = self._forces()

    def step(self, dt: float) -> None:
        """One kick-drift-kick step."""
        self.vel += 0.5 * dt * self.acc
        self.pos += dt * self.vel
        self.acc, self.phi, self.tree = self._forces()
        self.vel += 0.5 * dt * self.acc

    def virial(self) -> tuple[float, float]:
        """The virial ratio ``2 K / |sum(m r . a)|`` of the bound stars, and their mass fraction.

        Bound: negative energy in the potential of all the others. Positions and velocities
        are taken from the bound stars' centre of mass.
        """
        bound = 0.5 * np.sum(self.vel**2, axis=0) - self.phi < 0
        mass = self.mass[bound]
        weights = mass / mass.sum()
        pos = self.pos[:, bound] - (self.pos[:, bound] @ weights)[:, None]
        vel = self.vel[:, bound] - (self.vel[:, bound] @ weights)[:, None]
        kinetic = 0.5 * float(np.sum(mass * np.sum(vel**2, axis=0)))
        virial = float(np.sum(mass * np.sum(pos * self.acc[:, bound], axis=0)))
        return 2.0 * kinetic / abs(virial), float(mass.sum() / self.mass.sum())

    def energy(self) -> float:
        """The total energy."""
        kinetic = 0.5 * float(np.sum(self.mass * np.sum(self.vel**2, axis=0)))
        return kinetic - 0.5 * float(np.sum(self.mass * self.phi))


def evolve(n: int, spin: float, t_end: float, dt: float, frames: int) -> dict:
    """Run a disk to ``t_end``: snapshots, median radius, virial ratio and energy."""
    position, velocity, mass = maclaurin_disk(n, spin, quiet=spin == 0)
    system = Gravity(position, velocity, mass, softening=0.03)
    steps = int(round(t_end / dt))
    keep = set(np.linspace(0, steps, frames).round().astype(int))
    every = max(steps // 400, 1)
    times, median, ratio, bound, energy, snapshots = [], [], [], [], [], []
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            times.append(step * dt)
            median.append(float(np.median(np.hypot(*system.pos))))
            q, f = system.virial()
            ratio.append(q)
            bound.append(f)
            energy.append(system.energy())
        if step in keep:
            snapshots.append(
                {"t": step * dt, "pos": system.pos.copy(), "tree": system.tree}
            )
        if step < steps:
            system.step(dt)
    return {
        "t": np.array(times),
        "median": np.array(median),
        "virial": np.array(ratio),
        "bound": np.array(bound),
        "energy": np.array(energy),
        "snapshots": snapshots,
        "n": mass.size,
        "r0": np.hypot(*position),
    }


# -- figures --------------------------------------------------------------------------------


def animate(run: dict, output: Path, name: str, extent: float, title: str) -> None:
    """Figures 1 and 2: a disk with the quadtree of every frame, by ``animate_quadtree``."""
    snaps = run["snapshots"]
    result = animate_quadtree(
        [s["tree"] for s in snaps],
        points=[s["pos"] for s in snaps],
        point_color=run["r0"] - 0.5,  # by starting radius: blue inside, red outside
        times=[s["t"] for s in snaps],
        extent=(-extent, extent, -extent, extent),
        leaves_only=True,
        s=3,
        interval=100,
        title=title,
        backend="plotly",
    )
    save_figure(result, output / name, frame=-1, width=720, height=760)


def plot_homologous_fall(run: dict, output: Path) -> None:
    """Figure 3: the median radius against ``median(0) cos(Omega0 t)``."""
    t = run["t"]
    fall = t <= T_COLLAPSE
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t[fall] / T_COLLAPSE,
            y=run["median"][0] * np.cos(OMEGA0 * t[fall]),
            mode="lines",
            name="exact: cos(Ω₀ t)",
            line={"color": COLORS["theory"], "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t / T_COLLAPSE,
            y=run["median"],
            mode="lines",
            name=f"tree code, N = {run['n']}",
            line={"color": COLORS["collapse"], "dash": "dash"},
        )
    )
    fig.update_xaxes(title_text="time / collapse time")
    fig.update_yaxes(title_text="median radius", rangemode="tozero")
    fig.update_layout(title="The Maclaurin disk falls in, bounces and settles")
    save_figure(fig, output / "homologous_fall")


def plot_virial_ratio(runs: dict, output: Path) -> None:
    """Figure 4: the virial ratio of the bound stars, and their mass fraction."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for name, run in runs.items():
        color = COLORS[name.split()[0]]
        fig.add_trace(
            go.Scatter(
                x=run["t"] / T_COLLAPSE,
                y=run["virial"],
                mode="lines",
                name=name,
                line={"color": color},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=run["t"] / T_COLLAPSE,
                y=run["bound"],
                mode="lines",
                showlegend=False,
                line={"color": color},
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=1.0, line={"color": "gray", "dash": "dot"}, row=1, col=1)
    fig.update_yaxes(title_text="virial ratio", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="bound fraction", range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="time / collapse time", row=2, col=1)
    fig.update_layout(title="Violent relaxation to virial equilibrium", height=700)
    save_figure(fig, output / "virial_ratio", height=700)


# -- main -----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run both disks, write the figures and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        n, collapses, dt, frames = 200, 1.5, 0.004, 5
    elif args.quick:
        n, collapses, dt, frames = 3000, 6.0, 0.002, 60
    else:
        n, collapses, dt, frames = 10000, 8.0, 0.001, 80
    t_end = collapses * T_COLLAPSE
    print(f"collapse time {T_COLLAPSE:.3f}")

    print("1. collapse")
    collapse = evolve(n, 0.0, t_end, dt, frames)
    animate(
        collapse,
        output,
        "collapse",
        1.5,
        "t = {time}: {leaves} leaf cells, depth {depth}",
    )

    print("2. fragmentation")
    rotating = evolve(n, 1.0, t_end, dt, frames)
    animate(
        rotating,
        output,
        "fragmentation",
        1.5,
        "t = {time}: {leaves} leaf cells, depth {depth}",
    )

    print("3. homologous fall")
    plot_homologous_fall(collapse, output)
    early = collapse["t"] <= 0.4 * T_COLLAPSE
    exact = collapse["median"][0] * np.cos(OMEGA0 * collapse["t"][early])
    error = np.max(np.abs(collapse["median"][early] / exact - 1.0))
    print(
        f"   median radius up to 0.4 collapse times: within {error:.1e} of cos(Omega0 t)"
    )

    print("4. virial ratio")
    plot_virial_ratio(
        {"collapse (at rest)": collapse, "rotating disk": rotating}, output
    )
    for name, run in (("collapse", collapse), ("rotating", rotating)):
        drift = abs(run["energy"][-1] / run["energy"][0] - 1.0)
        print(
            f"   {name}: virial ratio {run['virial'][-1]:.2f}, bound fraction "
            f"{run['bound'][-1]:.2f}, energy error {drift:.1e}"
        )
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
