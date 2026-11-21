"""Galaxy collision.

Two disk galaxies fly past each other and pull out tidal tails, the classic Barnes-Hut problem.

Each galaxy is a heavy core (its bulge and halo, one softened point mass) with a light disk of
stars on circular orbits, all in one plane. The cores fall towards each other on a parabolic
orbit, and every body feels every other through the Barnes-Hut walk of ``FlatQuadtree``
(``kernel="coulomb"``, the ``1/r^2`` force of point masses, attractive here), integrated with
the kick-drift-kick leapfrog. Toomre and Toomre (1972) showed with this setup that the long
tails and bridges of interacting galaxies are tides: a disk turning the same way as the orbit
(prograde) stays in resonance with the passing companion and is torn into a bridge and a tail,
while a disk turning the other way (retrograde) is barely disturbed. Here galaxy A (red)
turns prograde and galaxy B (blue) retrograde. Units ``G = 1``, core masses 1, disk radius 1,
disk mass 2% of the core, pericentre 1.5.

1. ``collision``: the encounter, with the quadtree of every frame: the cells crowd into the
   cores and stretch out along the tails as the prograde disk is torn apart.
2. ``core_orbit``: the separation of the cores against the exact two-body parabola (Barker's
   equation). The disks' light mass and the stars they lose bend the orbit only slightly.
3. ``survival``: the fraction of each disk's stars still within one disk radius of their own
   core six time units after pericentre, against their starting radius: the prograde disk loses its outer half,
   the retrograde disk keeps almost all of it.
4. ``energy``: the energy error (relative to the kinetic energy at the start) of shorter runs
   with half the stars at several opening angles, and their cost per step. The total energy
   of a parabolic encounter is nearly zero, so the error is measured against the kinetic
   energy. The leapfrog's own error is small: what drifts is the tree's force error. The
   heavy cores make large ``theta`` costly: a cell holding a core is taken as one multipole
   while the core sits far from the cell's centre, so ``theta = 0.6`` loses a few percent and
   ``theta = 0.3`` (the main run) under one percent, while the direct sum is limited by the
   leapfrog alone.

Usage::

    python examples/galaxy_collision.py [--output examples/output] [--quick]

``--quick`` uses fewer stars (well under a minute); ``--smoke`` is the tiny configuration the
test suite runs. Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import animate_quadtree, save_figure

from tree_d_3d import FlatQuadtree, Quad

#: Core mass, disk mass, disk radii, pericentre and starting separation.
CORE, DISK, R_IN, R_OUT = 1.0, 0.02, 0.2, 1.0
PERICENTRE, START = 1.5, 8.0
SOFTENING, THETA = 0.05, 0.3
COLORS = {"A": "#d62728", "B": "#1f77b4", "theory": "#2ca02c"}


# -- the two-body orbit --------------------------------------------------------------------


def parabola(anomaly: np.ndarray, mu: float = 2.0 * CORE) -> tuple:
    """Separation, relative velocity and time from pericentre at the true anomaly ``f``."""
    p = 2.0 * PERICENTRE
    r = p / (1.0 + np.cos(anomaly))
    position = r * np.stack([np.cos(anomaly), np.sin(anomaly)])
    scale = np.sqrt(mu / p)
    velocity = scale * np.stack([-np.sin(anomaly), 1.0 + np.cos(anomaly)])
    d = np.tan(0.5 * anomaly)
    time = 0.5 * np.sqrt(p**3 / mu) * (d + d**3 / 3.0)  # Barker's equation
    return r, position, velocity, time


def start_anomaly() -> float:
    """The true anomaly (negative: before pericentre) at the starting separation."""
    return -float(np.arccos(2.0 * PERICENTRE / START - 1.0))


# -- the galaxies ----------------------------------------------------------------------------


def disk(n: int, sense: float, rng: np.random.Generator):
    """``n`` stars of a disk around a core at the origin, on circular orbits.

    Uniform surface density between ``R_IN`` and ``R_OUT``; the circular speed uses the core
    and the disk mass inside each radius; ``sense`` is +1 counter-clockwise, -1 clockwise.
    """
    r = np.sqrt(rng.uniform(R_IN**2, R_OUT**2, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    inside = CORE + DISK * (r**2 - R_IN**2) / (R_OUT**2 - R_IN**2)
    speed = np.sqrt(inside / r)
    position = r * np.stack([np.cos(phi), np.sin(phi)])
    velocity = sense * speed * np.stack([-np.sin(phi), np.cos(phi)])
    return position, velocity, r


def initial_state(n_stars: int, seed: int = 1) -> dict:
    """Both galaxies: the cores first (indices 0 and 1), then the stars of A, then of B."""
    rng = np.random.default_rng(seed)
    _, rel_pos, rel_vel, time = parabola(np.array(start_anomaly()))
    half = n_stars // 2
    pos_a, vel_a, r_a = disk(half, +1.0, rng)  # the orbit turns counter-clockwise
    pos_b, vel_b, r_b = disk(half, -1.0, rng)
    # the centre of mass at rest: each galaxy carries half the relative motion
    centre_a, centre_b = -0.5 * rel_pos, 0.5 * rel_pos
    drift_a, drift_b = -0.5 * rel_vel, 0.5 * rel_vel
    positions = np.concatenate(
        [
            np.stack([centre_a, centre_b], axis=1),
            pos_a + centre_a[:, None],
            pos_b + centre_b[:, None],
        ],
        axis=1,
    )
    velocities = np.concatenate(
        [
            np.stack([drift_a, drift_b], axis=1),
            vel_a + drift_a[:, None],
            vel_b + drift_b[:, None],
        ],
        axis=1,
    )
    masses = np.concatenate([[CORE, CORE], np.full(2 * half, DISK / half)])
    galaxy = np.concatenate([[0, 1], np.zeros(half, int), np.ones(half, int)])
    return {
        "pos": positions,
        "vel": velocities,
        "mass": masses,
        "galaxy": galaxy,
        "r0": np.concatenate([[0.0, 0.0], r_a, r_b]),
        "t_peri": float(-time),
    }


class Gravity:
    """Leapfrog for self-gravitating point masses with the Barnes-Hut tree of ``FlatQuadtree``."""

    def _forces(self):
        half = 1.05 * float(np.abs(self.pos).max())
        tree = FlatQuadtree(self.pos, self.mass, Quad(0.0, 0.0, half), leaf_capacity=8)
        phi, field = tree.evaluate(
            self.pos, exclude=self.others, theta=self.theta, softening=self.softening
        )
        # the field of positive "charges" points away from them: gravity pulls the other way
        return -np.asarray(field), np.asarray(phi), tree

    def __init__(self, positions, velocities, masses, theta=THETA, softening=SOFTENING):
        self.pos = positions.copy()
        self.vel = velocities.copy()
        self.mass = masses
        self.theta = theta
        self.softening = softening
        self.others = np.arange(masses.size)
        self.acc, self.phi, self.tree = self._forces()

    def step(self, dt: float) -> None:
        """One kick-drift-kick step."""
        self.vel += 0.5 * dt * self.acc
        self.pos += dt * self.vel
        self.acc, self.phi, self.tree = self._forces()
        self.vel += 0.5 * dt * self.acc

    def energy(self) -> float:
        kinetic = 0.5 * float(np.sum(self.mass * np.sum(self.vel**2, axis=0)))
        return kinetic - 0.5 * float(np.sum(self.mass * self.phi))


def collide(n_stars: int, t_after: float, dt: float, frames: int) -> dict:
    """Run from the start to ``t_after`` past pericentre; keep frames and diagnostics."""
    state = initial_state(n_stars)
    system = Gravity(state["pos"], state["vel"], state["mass"])
    t_end = state["t_peri"] + t_after
    steps = int(round(t_end / dt))
    keep = set(np.linspace(0, steps, frames).round().astype(int))
    times, separation, snapshots = [], [], []
    for step in range(steps + 1):
        times.append(step * dt)
        separation.append(float(np.hypot(*(system.pos[:, 1] - system.pos[:, 0]))))
        if step in keep:
            snapshots.append(
                {"t": step * dt, "pos": system.pos.copy(), "tree": system.tree}
            )
        if step < steps:
            system.step(dt)
    return {
        **state,
        "t": np.array(times),
        "separation": np.array(separation),
        "snapshots": snapshots,
        "final": system.pos.copy(),
    }


# -- figures ----------------------------------------------------------------------------------


def plot_collision(run: dict, output: Path) -> None:
    """Figure 1: the encounter with its quadtree, by ``animate_quadtree``."""
    snaps = run["snapshots"]
    side = np.where(run["galaxy"] == 0, 1.0, -1.0)  # A red, B blue
    result = animate_quadtree(
        [s["tree"] for s in snaps],
        points=[s["pos"] for s in snaps],
        point_color=side,
        times=[s["t"] - run["t_peri"] for s in snaps],
        sweep="t - t_peri",
        extent=(-7.0, 7.0, -7.0, 7.0),
        leaves_only=True,
        s=3,
        interval=100,
        title="t − t_peri = {time}: {leaves} leaf cells, depth {depth}",
        backend="plotly",
    )
    # the poster shows the tail at its longest inside the view, three time units after pericentre
    poster = int(np.argmin([abs(s["t"] - run["t_peri"] - 3.0) for s in snaps]))
    save_figure(result, output / "collision", frame=poster, width=720, height=760)


def kepler_separation(times: np.ndarray, t_peri: float) -> np.ndarray:
    """The two-body separation at ``times`` (from the start), by inverting Barker's equation."""
    anomaly = np.linspace(start_anomaly(), -start_anomaly() * 1.15, 20000)
    r, _, _, time = parabola(anomaly)
    return np.interp(times - t_peri, time, r)


def plot_core_orbit(run: dict, output: Path) -> None:
    """Figure 2: core separation against the two-body parabola."""
    t = run["t"] - run["t_peri"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=kepler_separation(run["t"], run["t_peri"]),
            mode="lines",
            name="two-body parabola",
            line={"color": COLORS["theory"], "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=run["separation"],
            mode="lines",
            name="tree code",
            line={"color": "black", "dash": "dash"},
        )
    )
    fig.update_xaxes(title_text="time from pericentre")
    fig.update_yaxes(title_text="core separation", rangemode="tozero")
    fig.update_layout(title="The cores follow a parabola")
    save_figure(fig, output / "core_orbit")


def plot_survival(run: dict, output: Path) -> None:
    """Figure 3: the fraction of stars still bound to their own core, by starting radius."""
    final, galaxy, r0 = run["final"], run["galaxy"], run["r0"]
    edges = np.linspace(R_IN, R_OUT, 9)
    centres = 0.5 * (edges[1:] + edges[:-1])
    fig = go.Figure()
    for index, name, label in ((0, "A", "A, prograde"), (1, "B", "B, retrograde")):
        stars = np.flatnonzero(galaxy == index)
        stars = stars[stars > 1]  # the stars, without the cores
        distance = np.hypot(*(final[:, stars] - final[:, [index]]))
        kept = distance < R_OUT
        bins = np.digitize(r0[stars], edges) - 1
        fraction = [
            kept[bins == b].mean() if np.any(bins == b) else np.nan for b in range(8)
        ]
        fig.add_trace(
            go.Scatter(
                x=centres,
                y=fraction,
                mode="lines+markers",
                name=label,
                line={"color": COLORS[name]},
            )
        )
    fig.update_xaxes(title_text="starting radius")
    fig.update_yaxes(title_text="fraction still within the disk", range=[0, 1.05])
    fig.update_layout(title="Tides strip the prograde disk, not the retrograde one")
    save_figure(fig, output / "survival")


def energy_study(n_stars: int, t_end: float, dt: float, thetas) -> dict:
    """The energy error over ``t_end`` and the seconds per step at each opening angle."""
    state = initial_state(n_stars, seed=2)
    studies = {}
    for theta in thetas:
        system = Gravity(state["pos"], state["vel"], state["mass"], theta=theta)
        kinetic = 0.5 * float(np.sum(system.mass * np.sum(system.vel**2, axis=0)))
        start = system.energy()
        times, error = [0.0], [0.0]
        steps = int(round(t_end / dt))
        clock = time.perf_counter()
        for step in range(1, steps + 1):
            system.step(dt)
            if step % 10 == 0 or step == steps:
                times.append(step * dt)
                error.append((system.energy() - start) / kinetic)
        cost = (time.perf_counter() - clock) / steps
        studies[theta] = (np.array(times), np.array(error), cost)
        print(
            f"   theta = {theta}: energy error {error[-1]:+.1e}, {1e3 * cost:.1f} ms/step"
        )
    return studies


def plot_energy(studies: dict, n: int, output: Path) -> None:
    """Figure 4: energy error at several opening angles, with the cost per step."""
    fig = go.Figure()
    shades = ["#9467bd", "#ff7f0e", "#1f77b4", "black"]
    for shade, (theta, (times, error, cost)) in zip(shades, studies.items()):
        name = "direct sum" if theta == 0 else f"θ = {theta}"
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.abs(error),
                mode="lines",
                name=f"{name}: {1e3 * cost:.0f} ms/step",
                line={"color": shade},
            )
        )
    fig.update_xaxes(title_text="time from the start")
    fig.update_yaxes(
        type="log",
        exponentformat="power",
        title_text="|energy error| / kinetic energy",
    )
    fig.update_layout(title=f"Accuracy against cost of the tree walk (N = {n})")
    save_figure(fig, output / "energy")


# -- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run the encounter, write the figures and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        n, t_after, dt, frames, t_study = 200, -8.0, 0.02, 5, 0.2
    elif args.quick:
        n, t_after, dt, frames, t_study = 3000, 6.0, 0.01, 60, 12.0
    else:
        n, t_after, dt, frames, t_study = 8000, 6.0, 0.005, 70, 16.0

    print("1. collision")
    run = collide(n, t_after, dt, frames)
    plot_collision(run, output)
    print(f"   pericentre at t = {run['t_peri']:.2f}, {run['mass'].size} bodies")

    print("2. core orbit")
    plot_core_orbit(run, output)

    print("3. survival")
    plot_survival(run, output)

    print("4. energy")
    studies = energy_study(n // 2, t_study, 0.01, (0.8, 0.6, 0.3, 0.0))
    plot_energy(studies, n // 2 + 2, output)
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
