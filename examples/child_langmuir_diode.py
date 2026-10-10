"""Space-charge-limited emission in a gridless code: the planar Child-Langmuir diode.

A beam source emits at the space-charge limit when the normal field at the emitter is
zero. In a mesh code that condition is imposed through the cell next to the emitter; in
a gridless code the field at the surface is available directly, from the same summation
that pushes the particles. Here the emitter is the cathode plate of a planar diode and
the summation is :class:`tree_d_3d.GroundedStripField`, whose image charges make both
plates exact equipotentials; the applied voltage is the harmonic function ``V x / d``
added on top. Each step the field at the cathode is probed per emitter segment and the
sheet charge ``sigma = eps0 E_x(0)`` that cancels it is injected at rest (Gauss's law
emission, the rule of PIC emission models). The run starts from vacuum, so it also shows
the overdriven transient: a virtual cathode that oscillates before the steady
Child-Langmuir flow sets in. Code units ``eps0 = e = m = d = V = 1``, so the
Child-Langmuir current density is ``J_CL = (4/9) sqrt(2) = 0.6285`` and the transit time
``3 d / v(d) = 2.12``.

1. ``current``: emitted current density, the current density collected at the anode and
   the one returned to the cathode against time, over ``J_CL``: the virtual-cathode
   transient and the steady state. The collected current is ``J_CL``; the return current
   is the artefact of injecting cold point charges into a layer the previous step's
   particles have not yet left, and shrinks with the time step.
2. ``potential``: the time-averaged potential and field along the gap at the emitter's
   centre against the Child-Langmuir profile ``phi = V (x/d)^(4/3)``; the field vanishes
   at the cathode and reaches ``4/3 V/d`` at the anode.
3. ``density``: the time-averaged electron density along the gap against the
   Child-Langmuir ``n = J / v = (J / sqrt(2 V)) (x/d)^(-2/3)``, which the point charges
   follow up to the layer where they are injected.
4. ``phase_space``: the electrons in the ``(x, v_x)`` plane at the end of the run, on the
   Child-Langmuir curve ``v = sqrt(2 phi)``; the width of the band is the noise of the
   emission rule and of the point charges' mutual repulsion.
5. ``stability``: the collected current against time for several time steps, at the
   same macroparticle charge; the emission rule feeds the field back one step late, and
   the transient grows noisier with the step, but the steady state is reached for every
   step tested.
6. ``emission``: an ``(x, y)`` animation of the electrons from vacuum through the
   virtual-cathode burst to the steady flow, with the emitting segments marked.

Usage::

    python examples/child_langmuir_diode.py [--output examples/output] [--quick]

``--quick`` uses a narrower emitter and a shorter run (a few minutes instead of a
quarter of an hour); ``--smoke`` is the tiny configuration the test suite runs. Needs
Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from plotly.subplots import make_subplots

from tree_d_3d import GroundedStripField

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}

# code units: eps0 = 1 (so k_e = 1 / 4 pi), electron charge -1, mass 1, gap d = 1, V = 1
K_E = 1.0 / (4.0 * np.pi)
GAP = 1.0
VOLTAGE = 1.0
J_CL = 4.0 / 9.0 * np.sqrt(2.0) * VOLTAGE**1.5 / GAP**2
TRANSIT = 3.0 * GAP / np.sqrt(2.0 * VOLTAGE)


@dataclass
class Diode:
    """A planar diode with Gauss's-law emission from the cathode plate at ``x = 0``.

    The emitter spans ``|y| < width / 2`` in ``segments`` pieces. The injection layer is
    ``0 < x < layer``; the field is probed just above it at ``x = 2 * layer`` and the
    layer's own charge is added back by Gauss's law to get the field at the surface.
    Diagnostics use the central ``|y| < core / 2`` part of the emitter, away from the
    edges.
    """

    width: float
    segments: int
    dt: float
    particle_charge: float  # the magnitude of one macroparticle's line charge
    softening: float = 0.01
    layer: float = 0.01
    core: float = 2.0
    seed: int = 0
    time: float = 0.0
    x: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))
    v: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))
    emitted: list[float] = field(default_factory=list)
    collected: list[float] = field(default_factory=list)
    returned: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.strip = GroundedStripField(
            GAP, k_e=K_E, softening=self.softening, order=12
        )
        self.edges = np.linspace(-self.width / 2, self.width / 2, self.segments + 1)
        self.centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.dy = self.edges[1] - self.edges[0]
        self.deficit = np.zeros(self.segments)  # charge owed per segment, carried over
        self.q = np.zeros(0)
        self._core = np.abs(self.centres) < self.core / 2

    # --- fields -----------------------------------------------------------------

    def fields(self, positions):
        """Total potential and field at ``positions`` ``(2, M)``: image charges + applied."""
        if self.q.size:
            phi, e = self.strip.probe(self.x, self.q, positions)
        else:
            phi, e = np.zeros(positions.shape[1]), np.zeros_like(positions)
        phi = phi + VOLTAGE * positions[0] / GAP
        e = e.copy()
        e[0] -= VOLTAGE / GAP
        return phi, e

    def field_at_particles(self):
        if not self.q.size:
            return np.zeros_like(self.x)
        _, e = self.strip.field_at_charges(self.x, self.q, potential=False)
        e[0] -= VOLTAGE / GAP
        return e

    def surface_field(self) -> np.ndarray:
        """``E_x`` at the cathode surface per emitter segment, by Gauss's law."""
        probes = np.stack((np.full(self.segments, 2 * self.layer), self.centres))
        _, e = self.fields(probes)
        # charge of the injection layer below the probes, per unit emitter length
        if self.q.size:
            below = self.x[0] < 2 * self.layer
            sigma = np.histogram(self.x[1, below], self.edges, weights=self.q[below])[0]
            sigma /= self.dy
        else:
            sigma = np.zeros(self.segments)
        return e[0] - sigma  # eps0 = 1

    # --- the step ---------------------------------------------------------------

    def emit(self) -> float:
        """Inject the sheet charge that cancels the surface field; return the current."""
        e0 = self.surface_field()
        wanted = np.where(e0 < 0, -e0, 0.0) * self.dy  # magnitude of charge to inject
        total = wanted + self.deficit
        count = np.floor(total / self.particle_charge).astype(int)
        self.deficit = total - count * self.particle_charge
        n = int(count.sum())
        if n:
            seg = np.repeat(np.arange(self.segments), count)
            y = self.centres[seg] + self.dy * (self.rng.random(n) - 0.5)
            x = self.layer * self.rng.random(n)
            self.x = np.concatenate((self.x, np.stack((x, y))), axis=1)
            self.v = np.concatenate((self.v, np.zeros((2, n))), axis=1)
            self.q = np.concatenate((self.q, np.full(n, -self.particle_charge)))
        core_charge = (count * self.particle_charge)[self._core].sum()
        return core_charge / (self.core * self.dt)

    def absorb(self) -> tuple[float, float]:
        """Remove electrons past the anode or back on the cathode; return both currents."""
        core = np.abs(self.x[1]) < self.core / 2
        gone_anode = self.x[0] >= GAP
        gone_cathode = self.x[0] <= 0
        collected = -self.q[gone_anode & core].sum() / (self.core * self.dt)
        returned = -self.q[gone_cathode & core].sum() / (self.core * self.dt)
        keep = ~(gone_anode | gone_cathode)
        self.x, self.v, self.q = self.x[:, keep], self.v[:, keep], self.q[keep]
        return collected, returned

    def step(self) -> None:
        """Leapfrog (kick-drift-kick) with emission before the push and absorption after."""
        emitted = self.emit()
        a = -self.field_at_particles()  # q = -1, m = 1
        self.v += 0.5 * self.dt * a
        self.x += self.dt * self.v
        collected, returned = self.absorb()
        if self.q.size:
            self.v += 0.5 * self.dt * (-self.field_at_particles())
        self.time += self.dt
        self.times.append(self.time)
        self.emitted.append(emitted)
        self.collected.append(collected)
        self.returned.append(returned)

    # --- diagnostics ------------------------------------------------------------

    def core_density(self, bins: np.ndarray) -> np.ndarray:
        core = np.abs(self.x[1]) < self.core / 2
        counts = np.histogram(self.x[0, core], bins, weights=-self.q[core])[0]
        return counts / (np.diff(bins) * self.core)


def run(
    diode: Diode,
    end: float,
    *,
    average_from: float | None = None,
    profile_x: np.ndarray | None = None,
    frames: int = 0,
):
    """Advance to ``end``; average profiles after ``average_from``; keep animation frames."""
    averages = {"phi": 0.0, "e": 0.0, "n": 0.0, "count": 0}
    bins = np.linspace(0, GAP, 41)
    snapshots = []
    steps = int(round(end / diode.dt))
    stride = max(1, steps // frames) if frames else 0
    for i in range(steps):
        diode.step()
        if (
            average_from is not None
            and diode.time >= average_from
            and profile_x is not None
        ):
            probes = np.stack((profile_x, np.zeros_like(profile_x)))
            phi, e = diode.fields(probes)
            averages["phi"] = averages["phi"] + phi
            averages["e"] = averages["e"] + e[0]
            averages["n"] = averages["n"] + diode.core_density(bins)
            averages["count"] += 1
        if frames and i % stride == 0:
            snapshots.append((diode.time, diode.x.copy()))
    if averages["count"]:
        for key in ("phi", "e", "n"):
            averages[key] = averages[key] / averages["count"]
    averages["bins"] = bins
    return averages, snapshots


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Running mean over ``window`` samples (same length, edges shortened)."""
    if window <= 1:
        return np.asarray(y)
    c = np.cumsum(np.insert(np.asarray(y, dtype=float), 0, 0.0))
    out = np.empty(len(y))
    for i in range(len(y)):
        lo, hi = max(0, i - window // 2), min(len(y), i + window // 2 + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def main(argv: list[str] | None = None) -> Path:
    """Run the diode and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        width, segments, dt, end, per_step, scan = 3.0, 24, 0.05, 6.0, 1.0, (0.05, 0.1)
    elif args.quick:
        width, segments, dt, end, per_step, scan = (
            4.0,
            32,
            0.02,
            20.0,
            1.0,
            (0.02, 0.05, 0.1, 0.2),
        )
    else:
        width, segments, dt, end, per_step, scan = (
            6.0,
            64,
            0.01,
            30.0,
            2.0,
            (0.01, 0.02, 0.05, 0.1, 0.2),
        )
    core = min(2.0, width - 2.0)
    dy = width / segments
    # one macroparticle carries 1 / per_step of a segment's steady-state charge per step
    particle_charge = J_CL * dy * dt / per_step
    print(
        f"emitter width {width}, {segments} segments, dt = {dt}, macroparticle charge "
        f"{particle_charge:.2e}, J_CL = {J_CL:.4f}, transit time {TRANSIT:.2f}"
    )

    # 1. the base run
    diode = Diode(width, segments, dt, particle_charge, core=core)
    profile_x = np.linspace(0.02, 0.98, 49)
    averages, snapshots = run(
        diode, end, average_from=0.6 * end, profile_x=profile_x, frames=60
    )
    t = np.asarray(diode.times)
    window = max(1, int(0.2 * TRANSIT / dt))
    late = t >= 0.6 * end
    j_collected = float(np.mean(np.asarray(diode.collected)[late]))
    j_emitted = float(np.mean(np.asarray(diode.emitted)[late]))
    print(
        f"steady state: emitted {j_emitted / J_CL:.4f} J_CL, collected "
        f"{j_collected / J_CL:.4f} J_CL, {diode.q.size} electrons in the gap"
    )

    fig = go.Figure()
    for name, data, color in (
        ("emitted", diode.emitted, COLORS["C1"]),
        ("collected at the anode", diode.collected, COLORS["C0"]),
        ("returned to the cathode", diode.returned, COLORS["C3"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=t / TRANSIT,
                y=smooth(np.asarray(data) / J_CL, window),
                mode="lines",
                name=f"{name} (running mean over 0.2 transit times)",
                line={"color": color},
            )
        )
    fig.add_hline(
        y=1.0, line={"dash": "dash", "color": "black"}, annotation_text="J_CL"
    )
    fig.update_xaxes(title_text="time / transit time")
    fig.update_yaxes(title_text="current density / J_CL", range=[-0.2, 3])
    fig.update_layout(title="Emitted and collected current density of the planar diode")
    save_figure(fig, output / "current")

    # 2. potential and field profiles
    fig = make_subplots(rows=1, cols=2, subplot_titles=("potential", "field E_x"))
    x_cl = np.linspace(0, GAP, 200)
    fig.add_trace(
        go.Scatter(
            x=profile_x,
            y=averages["phi"],
            mode="markers",
            name="tree, time average",
            marker={"color": COLORS["C0"]},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_cl,
            y=VOLTAGE * (x_cl / GAP) ** (4 / 3),
            mode="lines",
            name="Child-Langmuir V (x/d)^(4/3)",
            line={"color": "black", "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_cl,
            y=VOLTAGE * x_cl / GAP,
            mode="lines",
            name="vacuum V x/d",
            line={"color": "gray", "dash": "dot"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=profile_x,
            y=averages["e"],
            mode="markers",
            name="tree, time average",
            marker={"color": COLORS["C0"]},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x_cl,
            y=-4 / 3 * VOLTAGE / GAP * (x_cl / GAP) ** (1 / 3),
            mode="lines",
            name="Child-Langmuir -4/3 V/d (x/d)^(1/3)",
            line={"color": "black", "dash": "dash"},
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="x / d")
    fig.update_yaxes(title_text="phi / V", row=1, col=1)
    fig.update_yaxes(title_text="E_x d / V", row=1, col=2)
    fig.update_layout(title="Potential and field along the gap at the emitter's centre")
    save_figure(fig, output / "potential")

    # 3. density profile
    bins = averages["bins"]
    mid = 0.5 * (bins[1:] + bins[:-1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mid,
            y=averages["n"],
            mode="markers",
            name="tree, time average",
            marker={"color": COLORS["C0"]},
        )
    )
    x_n = np.linspace(0.01, GAP, 300)
    fig.add_trace(
        go.Scatter(
            x=x_n,
            y=J_CL / np.sqrt(2 * VOLTAGE) * (x_n / GAP) ** (-2 / 3),
            mode="lines",
            name="Child-Langmuir J_CL / v(x)",
            line={"color": "black", "dash": "dash"},
        )
    )
    fig.update_xaxes(title_text="x / d")
    fig.update_yaxes(
        title_text="electron density", range=[0, 4 * J_CL / np.sqrt(2 * VOLTAGE)]
    )
    fig.update_layout(title="Electron density along the gap")
    save_figure(fig, output / "density")

    # 4. phase space
    in_core = np.abs(diode.x[1]) < diode.core / 2
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=diode.x[0, in_core],
            y=diode.v[0, in_core],
            mode="markers",
            marker={"size": 3, "color": COLORS["C0"], "opacity": 0.5},
            name="electrons at the end of the run",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_cl,
            y=np.sqrt(2 * VOLTAGE * (x_cl / GAP) ** (4 / 3)),
            mode="lines",
            name="Child-Langmuir sqrt(2 phi)",
            line={"color": "black", "dash": "dash"},
        )
    )
    fig.update_xaxes(title_text="x / d")
    fig.update_yaxes(title_text="v_x")
    fig.update_layout(title="Electron phase space (x, v_x) in the steady state")
    save_figure(fig, output / "phase_space")

    # 5. the emission loop against the time step
    fig = go.Figure()
    for i, step in enumerate(scan):
        if step == dt:
            times, current = t, diode.collected
        else:
            other = Diode(width, segments, step, particle_charge, core=core)
            run(other, end)
            times, current = np.asarray(other.times), other.collected
            late = times >= 0.6 * end
            print(
                f"  dt = {step}: collected {np.mean(np.asarray(current)[late]) / J_CL:.4f} J_CL"
            )
        fig.add_trace(
            go.Scatter(
                x=times / TRANSIT,
                y=smooth(np.asarray(current) / J_CL, max(1, int(0.2 * TRANSIT / step))),
                mode="lines",
                name=f"dt = {step}",
                line={"color": COLORS[f"C{i % 4}"]},
            )
        )
    fig.add_hline(y=1.0, line={"dash": "dash", "color": "black"})
    fig.update_xaxes(title_text="time / transit time")
    fig.update_yaxes(title_text="collected current density / J_CL", range=[-0.2, 3])
    fig.update_layout(
        title="Collected current for several time steps (same macroparticle charge)"
    )
    save_figure(fig, output / "stability")

    # 6. animation
    frames = []
    for time, pos in snapshots:
        frames.append(
            go.Frame(
                data=[
                    go.Scattergl(
                        x=pos[0],
                        y=pos[1],
                        mode="markers",
                        marker={"size": 2, "color": COLORS["C0"]},
                    )
                ],
                name=f"{time:.2f}",
                layout={"title": f"t = {time / TRANSIT:.2f} transit times"},
            )
        )
    fig = go.Figure(data=frames[0].data, frames=frames)
    for yc in diode.edges:
        fig.add_shape(
            type="line", x0=-0.02, x1=0.0, y0=yc, y1=yc, line={"color": "black"}
        )
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=-width / 2,
        y1=width / 2,
        line={"color": "black", "width": 3},
    )
    fig.add_shape(
        type="line",
        x0=GAP,
        x1=GAP,
        y0=-width / 2,
        y1=width / 2,
        line={"color": COLORS["C3"], "width": 3},
    )
    fig.update_xaxes(title_text="x / d", range=[-0.05, GAP + 0.05])
    fig.update_yaxes(title_text="y / d", range=[-width / 2 - 0.1, width / 2 + 0.1])
    fig.update_layout(
        title="Electrons from vacuum to the Child-Langmuir flow",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    }
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": f.name,
                        "method": "animate",
                    }
                    for f in frames
                ]
            }
        ],
    )
    save_figure(fig, output / "emission", frame=len(frames) - 1)

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
