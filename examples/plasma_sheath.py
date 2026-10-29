"""Plasma sheath.

A plasma sheath between two grounded (Dirichlet) walls.

Runs a two-species (electron/ion) particle simulation confined in the strip
``0 < x < L`` between two grounded conducting plates, using ``GroundedStripField``:
its image-charge Green's function enforces ``phi = 0`` exactly at ``x = 0`` and
``x = L`` (a Dirichlet boundary condition), not approximately. A quasi-neutral
plasma starts uniform and Maxwellian; electrons, being lighter and faster, are
lost to and repelled from the grounded walls more than the slower ions, leaving a
thin layer of excess positive space charge against each wall -- the classic plasma
sheath -- while the bulk stays quasi-neutral. Saves one figure per claim:

1. ``setup``: the initial electron and ion positions, uniform and quasi-neutral,
   with the two grounded walls marked.
2. ``density_profile``: electron and ion density against x, time-averaged over a
   late window once the sheath has formed; quasi-neutral in the bulk, with an
   excess of ions over electrons in thin layers against both walls.
3. ``potential_profile``: the electrostatic potential against x, same time
   average; it returns to (near) zero at both grounded plates -- the Dirichlet
   condition the image charges enforce exactly -- while the bulk floats higher,
   with most of the drop concentrated in the sheath layers.
4. ``field_profile``: the field E_x against x, same time average; close to zero
   in the quasi-neutral bulk and pointing from the bulk toward each wall inside
   the sheath layers.
5. ``particle_loss``: electrons and ions remaining against time; the walls are a
   real particle sink, with the bulk only partly depleted over the plotted window.
6. ``sheath_formation``: an (x, v_x) phase-space animation of both species, from
   the initial uniform state to the developed sheath.

Usage::

    python examples/plasma_sheath.py [--output examples/output] [--quick]

``--quick`` uses fewer particles and a shorter run (well under a minute instead of
roughly ten). Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure

from tree_d_3d.strip import GroundedStripField

MPL_COLORS = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
}
ELECTRON, ION = 0, 1


def _clustered_edges(low: float, high: float, n: int) -> np.ndarray:
    """``n + 1`` edges over ``[low, high]``, dense at both ends, sparse in the middle.

    The sheath is only a few Debye lengths thick -- a small fraction of the domain --
    so a uniform grid puts it inside a single bin. This Chebyshev-like spacing
    (``cos`` of an evenly spaced angle) concentrates resolution at the two walls,
    where it matters, without needing more samples in the already-featureless bulk.
    """
    theta = np.linspace(0.0, np.pi, n + 1)
    return 0.5 * (low + high) - 0.5 * (high - low) * np.cos(theta)


def _add_player(
    fig: go.Figure, frames: list[go.Frame], *, duration: int = 40, prefix: str = "t = "
) -> None:
    """A Play button and a frame slider, as ``plasma_plots.heatmap_movie`` does for heatmaps."""
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.18,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": duration, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                ],
            },
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in frames
                ],
                "x": 0.12,
                "len": 0.88,
                "y": -0.1,
                "currentvalue": {"prefix": prefix},
            },
        ],
    )


def _wall_shapes(L: float, y0: float, y1: float) -> list[dict]:
    """Two vertical lines marking the grounded plates at x = 0 and x = L."""
    return [
        {
            "type": "line",
            "x0": x,
            "x1": x,
            "y0": y0,
            "y1": y1,
            "line": {"color": "black", "width": 2},
        }
        for x in (0.0, L)
    ]


class Sheath:
    """Leapfrog state of the two-species plasma between two grounded plates.

    Units: lengths in the electron Debye length lambda_D, time in 1/omega_pe,
    velocities in the electron thermal speed v_te, with T_e = m_e = 1, electron
    charge -1 and ion charge +1 (div E = 4 pi rho, so k_e = 1, the convention this
    repo's other plasma examples use). ``Ly`` is kept comparable to ``L`` (not
    stretched out for better statistics): ``GroundedStripField``'s conformal map
    compresses or expands distances by exp(pi y / L) across the strip, so a tall,
    thin sampling box raises charges to huge multipole powers and loses all
    numerical precision; a tall box is the wrong way to add more particles here.
    """

    def _accel(self) -> np.ndarray:
        _, field = self.field.field_at_charges(self.pos, self.charge, potential=False)
        return (self.charge * field) / self.mass[None, :]

    def __init__(
        self,
        L: float,
        Ly: float,
        n_half: int,
        *,
        mass_ratio: float = 100.0,
        v_te: float = 1.0,
        softening: float = 0.15,
        dt: float = 0.05,
        order: int = 8,
        theta: float = 0.6,
        leaf_capacity: int = 32,
        seed: int = 0,
    ) -> None:
        self.L, self.Ly, self.dt = L, Ly, dt
        self.n_half = n_half
        self.field = GroundedStripField(
            L,
            order=order,
            theta=theta,
            leaf_capacity=leaf_capacity,
            k_e=1.0,
            softening=softening,
        )
        rng = np.random.default_rng(seed)
        v_ti = v_te / np.sqrt(mass_ratio)
        x = rng.uniform(0.02 * L, 0.98 * L, 2 * n_half)
        y = rng.uniform(-Ly / 2, Ly / 2, 2 * n_half)
        vx = np.concatenate([rng.normal(0, v_te, n_half), rng.normal(0, v_ti, n_half)])
        vy = np.concatenate([rng.normal(0, v_te, n_half), rng.normal(0, v_ti, n_half)])
        self.pos = np.vstack([x, y])
        self.vel = np.vstack([vx, vy])
        self.charge = np.concatenate([-np.ones(n_half), np.ones(n_half)])
        self.mass = np.concatenate([np.ones(n_half), np.full(n_half, mass_ratio)])
        self.species = np.concatenate([np.full(n_half, ELECTRON), np.full(n_half, ION)])
        self.time = 0.0
        self.history_t = [0.0]
        self.history_n = [(n_half, n_half)]
        self._acc = self._accel()

    def counts(self) -> tuple[int, int]:
        return (
            int(np.sum(self.species == ELECTRON)),
            int(np.sum(self.species == ION)),
        )

    def step(self) -> None:
        """Leapfrog: half-kick, drift (with a periodic y wrap), absorb, half-kick."""
        self.vel += 0.5 * self.dt * self._acc
        self.pos += self.dt * self.vel
        self.pos[1] = (self.pos[1] + self.Ly / 2) % self.Ly - self.Ly / 2
        hit = (self.pos[0] <= 0) | (self.pos[0] >= self.L)
        if np.any(hit):
            keep = ~hit
            self.pos = np.ascontiguousarray(self.pos[:, keep])
            self.vel = np.ascontiguousarray(self.vel[:, keep])
            self.charge, self.mass, self.species = (
                self.charge[keep],
                self.mass[keep],
                self.species[keep],
            )
        self._acc = self._accel()
        self.vel += 0.5 * self.dt * self._acc
        self.time += self.dt
        self.history_t.append(self.time)
        self.history_n.append(self.counts())

    def density_profile(self, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Electron and ion linear density (particles / length) over [0, L]; centers, n_e, n_i.

        Bins are :func:`_clustered_edges`, so each count is divided by its own width:
        a raw count would read as a higher "density" in the wide bulk bins for no
        reason other than their width.
        """
        edges = _clustered_edges(0.0, self.L, bins)
        width = np.diff(edges)
        n_e, _ = np.histogram(self.pos[0, self.species == ELECTRON], bins=edges)
        n_i, _ = np.histogram(self.pos[0, self.species == ION], bins=edges)
        return 0.5 * (edges[:-1] + edges[1:]), n_e / width, n_i / width

    def probe_profile(self, probe_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Potential and E_x at y = 0, probed at ``probe_x`` (strictly inside (0, L))."""
        probe = np.vstack([probe_x, np.zeros_like(probe_x)])
        phi, field = self.field.probe(self.pos, self.charge, probe)
        return phi, field[0]


def run_and_profile(
    L: float,
    Ly: float,
    n_half: int,
    *,
    burst_steps: int,
    average_steps: int,
    sample_every: int,
    bins: int,
    quick: bool,
) -> dict:
    """Run to a late, quasi-steady window and average the profiles over it."""
    sheath = Sheath(L, Ly, n_half, seed=1)
    lam_D = np.sqrt(1.0 / (4 * np.pi * (2 * n_half / (L * Ly))))
    print(
        f"  L = {L:.0f} = {L / lam_D:.0f} lambda_D, N = {2 * n_half}, "
        f"dt * omega_pe = {sheath.dt:.3f}"
    )
    for _ in range(burst_steps):
        sheath.step()

    centers, n_e_sum, n_i_sum = sheath.density_profile(bins)
    n_e_sum, n_i_sum = n_e_sum.astype(float), n_i_sum.astype(float)
    probe_x = _clustered_edges(0.01 * L, 0.99 * L, bins)
    phi_sum = np.zeros_like(probe_x)
    ex_sum = np.zeros_like(probe_x)
    samples = 0
    for i in range(average_steps):
        sheath.step()
        if (i + 1) % sample_every == 0:
            _, n_e, n_i = sheath.density_profile(bins)
            n_e_sum += n_e
            n_i_sum += n_i
            phi, ex = sheath.probe_profile(probe_x)
            phi_sum += phi
            ex_sum += ex
            samples += 1
    samples = max(samples, 1)
    n_e_end, n_i_end = sheath.counts()
    print(
        f"  t = {sheath.time:.1f}: N_e = {n_e_end}, N_i = {n_i_end}, "
        f"averaged over {samples} late snapshots"
    )
    return {
        "lam_D": lam_D,
        "x": centers,
        "n_e": n_e_sum / (1 + samples),
        "n_i": n_i_sum / (1 + samples),
        "probe_x": probe_x,
        "phi": phi_sum / samples,
        "field": ex_sum / samples,
        "history_t": np.array(sheath.history_t),
        "history_n": np.array(sheath.history_n),
    }


# -- 1: setup ------------------------------------------------------------------------------


def plot_setup(output: Path, L: float, Ly: float, n_half: int) -> None:
    """Figure 1."""
    sheath = Sheath(L, Ly, n_half, seed=2)
    fig = go.Figure()
    for species, name, color in (
        (ELECTRON, "electrons", MPL_COLORS["C0"]),
        (ION, "ions", MPL_COLORS["C3"]),
    ):
        sel = sheath.species == species
        fig.add_trace(
            go.Scatter(
                x=sheath.pos[0, sel],
                y=sheath.pos[1, sel],
                mode="markers",
                marker={"size": 3, "color": color, "opacity": 0.6},
                name=name,
            )
        )
    fig.update_layout(shapes=_wall_shapes(L, -Ly / 2, Ly / 2))
    fig.update_xaxes(title_text="x", range=[-0.02 * L, 1.02 * L], showgrid=False)
    fig.update_yaxes(title_text="y", scaleanchor="x", showgrid=False)
    fig.update_layout(
        title="A quasi-neutral plasma starts uniform between two grounded walls (black)",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "setup")


# -- 2-4: profiles ---------------------------------------------------------------------------


def plot_density_profile(output: Path, profile: dict) -> None:
    """Figure 2."""
    fig = go.Figure()
    for key, name, color in (
        ("n_e", "electrons", MPL_COLORS["C0"]),
        ("n_i", "ions", MPL_COLORS["C3"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=profile["x"],
                y=profile[key],
                mode="lines",
                line={"color": color},
                name=name,
            )
        )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="linear density, particles / length (time-averaged)")
    fig.update_layout(
        title=(
            f"Quasi-neutral bulk, excess ions against both walls (domain ≈ "
            f"{profile['x'][-1] / profile['lam_D']:.0f} λ_D wide)"
        ),
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "density_profile")


def plot_potential_profile(output: Path, profile: dict) -> None:
    """Figure 3."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=profile["probe_x"],
            y=profile["phi"],
            mode="lines",
            line={"color": MPL_COLORS["C2"]},
            showlegend=False,
        )
    )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="potential φ (time-averaged)")
    fig.update_layout(
        title="φ = 0 at both grounded plates (Dirichlet, enforced by the image charges), bulk floats higher"
    )
    save_figure(fig, output / "potential_profile")


def plot_field_profile(output: Path, profile: dict) -> None:
    """Figure 4."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=profile["probe_x"],
            y=profile["field"],
            mode="lines",
            line={"color": MPL_COLORS["C1"]},
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line={"color": "black", "width": 0.5})
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="E_x (time-averaged)")
    fig.update_layout(
        title="Field near zero in the quasi-neutral bulk, pointing toward each wall in its sheath"
    )
    save_figure(fig, output / "field_profile")


# -- 5: particle loss -------------------------------------------------------------------------


def plot_particle_loss(output: Path, profile: dict) -> None:
    """Figure 5."""
    t = profile["history_t"]
    n = profile["history_n"]
    fig = go.Figure()
    for col, name, color in (
        (0, "electrons", MPL_COLORS["C0"]),
        (1, "ions", MPL_COLORS["C3"]),
    ):
        fig.add_trace(
            go.Scatter(x=t, y=n[:, col], mode="lines", line={"color": color}, name=name)
        )
    fig.update_xaxes(title_text="time")
    fig.update_yaxes(title_text="particles remaining")
    fig.update_layout(
        title="The grounded walls are a real particle sink; the bulk is only partly depleted here",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "particle_loss")


# -- 6: sheath formation (animation) -----------------------------------------------------------


def animate_sheath_formation(
    output: Path, L: float, Ly: float, n_half: int, *, frames: int, steps_per_frame: int
) -> None:
    """Figure 6."""
    sheath = Sheath(L, Ly, n_half, seed=3)
    v_scale = 3.0

    fig = go.Figure()
    for _species, name, color in (
        (ELECTRON, "electrons", MPL_COLORS["C0"]),
        (ION, "ions", MPL_COLORS["C3"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker={"size": 3, "color": color, "opacity": 0.6},
                name=name,
            )
        )
    fig.update_layout(shapes=_wall_shapes(L, -v_scale, v_scale))
    fig.update_xaxes(title_text="x", range=[-0.02 * L, 1.02 * L])
    fig.update_yaxes(title_text="v_x", range=[-v_scale, v_scale])
    fig.update_layout(legend={"font": {"size": 10}})

    frame_list = []
    for frame_i in range(frames):
        for _ in range(steps_per_frame):
            sheath.step()
        n_e, n_i = sheath.counts()
        title = f"t = {sheath.time:.2f}: N_e = {n_e}, N_i = {n_i}"
        data = []
        for species in (ELECTRON, ION):
            sel = sheath.species == species
            data.append(go.Scatter(x=sheath.pos[0, sel], y=sheath.vel[0, sel]))
        frame_list.append(
            go.Frame(
                name=f"{frame_i}",
                data=data,
                traces=[0, 1],
                layout=go.Layout(title=title),
            )
        )
    fig.frames = frame_list
    if frame_list:
        fig.update_layout(title=frame_list[0].layout.title.text)
    _add_player(fig, frame_list, duration=60, prefix="frame ")
    save_figure(fig, output / "sheath_formation", frame=-1)


# -- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run everything and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    quick = args.quick
    L = 30.0

    print("1. setup")
    plot_setup(output, L, L, n_half=1000 if quick else 6000)

    print("2-5. density, potential, field profiles and particle loss")
    profile = run_and_profile(
        L,
        L,
        n_half=1000 if quick else 6000,
        burst_steps=120 if quick else 350,
        average_steps=80 if quick else 200,
        sample_every=10,
        bins=24 if quick else 80,
        quick=quick,
    )
    plot_density_profile(output, profile)
    plot_potential_profile(output, profile)
    plot_field_profile(output, profile)
    plot_particle_loss(output, profile)

    print("6. sheath formation")
    animate_sheath_formation(
        output,
        L,
        L,
        n_half=80 if quick else 300,
        frames=25 if quick else 60,
        steps_per_frame=3,
    )
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
