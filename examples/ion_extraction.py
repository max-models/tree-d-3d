"""Ion extraction through an aperture, by the tree code plus boundary-integral electrodes.

A two-slit extraction system, the 2D analogue of the plasma/extractor electrode pair used
to pull an ion beam out of a plasma source (as in ``picnic``'s electrode setups): a grounded
"plasma electrode" with a narrow aperture, a gap, then an "extractor electrode" at a large
negative bias with a wider aperture. Each electrode is a simgeometry ``Aperture2D`` (two
plates of finite thickness), panelled by :func:`tree_d_3d.conductors_from_shape` with panels
graded toward the aperture tips, and its ``contains_points`` tells which ions hit a plate.
Ions are continuously supplied from a small reservoir pressed against the plasma-electrode
aperture (representing the plasma meniscus) and are pushed by the combined field of the two
electrodes (:class:`tree_d_3d.BoundaryIntegralElectrodes`, re-solved every step for the
electrodes' response to the current ion cloud) and of the other ions
(:class:`tree_d_3d.FlatQuadtree`'s Barnes-Hut tree walk, log kernel). An ion that is
collected past the extractor or strikes a plate is immediately re-injected at the reservoir,
so the simulation runs at a fixed particle count and reaches a steady extracted beam rather
than draining a finite source.

The beam's own space charge matters here just as it does in a real source: too little and
the beam threads both apertures almost losslessly; enough of it and the self-repulsion
flares the beam out faster than the extractor aperture can accept, intercepting a growing
fraction of it on the extractor plate -- the extraction analogue of a space-charge-limited
diode, without needing a 1D planar approximation.

1. ``geometry``: the two slit electrodes, their potentials, and the two apertures.
2. ``beam_snapshot``: a steady-state snapshot of ion positions with the electrodes drawn in,
   showing the beam narrow through the plasma-electrode aperture and spread again toward the
   extractor.
3. ``envelope``: the beam's rms half-width against x across the whole device, time-averaged
   over a late steady window.
4. ``transmission_scaling``: the fraction of ions that make it past the extractor against
   the macroparticle charge (the simulation's space-charge "knob") -- transmission falls as
   the beam's own space charge grows, the extraction analogue of a space-charge-limited diode.
5. ``beam_formation``: an animation of the real-space (x, y) ion positions, from the initial
   fill to the steady extracted beam.

Usage::

    python examples/ion_extraction.py [--output examples/output] [--quick]

``--quick`` uses fewer particles, panels and steps (well under a minute instead of a few
minutes); ``--smoke`` is the tiny configuration the test suite runs. Needs Plotly
(``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from simgeometry import Aperture2D

from tree_d_3d import (
    BoundaryIntegralElectrodes,
    FlatQuadtree,
    Quad,
    conductors_from_shape,
    direct_field,
)

MPL_COLORS = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
}
K_E = 1.0  # code units


def _add_player(
    fig: go.Figure, frames: list[go.Frame], *, duration: int = 60, prefix: str = "t = "
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


def slit_electrode(x: float, half_aperture: float) -> Aperture2D:
    """The two plates (above and below the aperture) of a slit electrode centred at ``x``.

    A simgeometry ``Aperture2D``: each plate is ``THICKNESS`` thick and ``PLATE_LENGTH``
    long, its tip ``half_aperture`` off the axis.
    """
    x0, x1 = x - 0.5 * THICKNESS, x + 0.5 * THICKNESS
    a, b = half_aperture, half_aperture + PLATE_LENGTH
    return Aperture2D(
        axis=0, position=0.0, vertices=((x0, a), (x1, a), (x1, b), (x0, b))
    )


def panel_length(tip_panel: float, half_aperture: float):
    """Panel lengths graded from ``tip_panel`` at the aperture tip to ``THICKNESS``.

    Short panels where the field varies fastest, but not much shorter than the ions'
    softening length there: ions pass close to the tip, and the panel method's point-charge
    approximation (see :mod:`tree_d_3d.electrodes`) needs each panel's own length to stay
    comparable to it. The cap keeps panels shorter than the gap across the plate.
    """

    def length(points: np.ndarray) -> np.ndarray:
        distance = np.maximum(np.abs(points[1]) - half_aperture, 0.0)
        return np.minimum(tip_panel + 0.1 * distance, THICKNESS)

    return length


A0, A1 = 0.6, 1.0  # plasma-electrode, extractor-electrode aperture half-widths
PLATE_LENGTH = 10.0
THICKNESS = 0.4
GAP = 4.0  # plasma electrode to extractor
X0, X1 = 0.0, GAP
RESERVOIR_X = (-1.5, -0.3)  # ion source, pressed against the plasma-electrode aperture
V_DRIFT, V_THERMAL = 4.0, 0.12
SOFTENING = 0.5
DT = 0.03


class ExtractionSource:
    """Leapfrog state of ions pushed by the two electrodes and by each other.

    An ion absorbed by a plate, collected past ``x_final``, or flung far enough off-axis to
    have left the device entirely is immediately re-injected at the reservoir, so ``n`` (and
    the macroparticle charge ``q``, the simulation's space-charge "knob") stay fixed and the
    system settles into a steady extracted beam instead of draining its own source.
    """

    def __init__(
        self,
        n: int,
        q: float,
        *,
        v_ext: float,
        x_final: float,
        tip_panel: float,
        leaf_capacity: int = 16,
        theta: float = 0.6,
        seed: int = 0,
    ) -> None:
        self.n, self.q, self.x_final = n, q, x_final
        self.leaf_capacity, self.theta = leaf_capacity, theta
        #: the plasma electrode and the extractor, as ``(x, half-aperture, shape)``
        self.plates = [
            (X0, A0, slit_electrode(X0, A0)),
            (X1, A1, slit_electrode(X1, A1)),
        ]
        conductors = [
            conductor
            for (_, half_ap, shape), potential in zip(self.plates, (0.0, -v_ext))
            for conductor in conductors_from_shape(
                shape, potential, panel_length(tip_panel, half_ap)
            )
        ]
        self.electrodes = BoundaryIntegralElectrodes(conductors, k_e=K_E)
        self.rng = np.random.default_rng(seed)
        self.charge = np.full(n, q)
        half = 0.5 * (x_final - RESERVOIR_X[0]) + 2.0
        self.root = Quad(0.5 * (RESERVOIR_X[0] + x_final), 0.0, half)
        self.pos, self.vel = self._fresh(n)
        self.time = 0.0
        self.n_collected = 0
        self.n_lost = 0
        self.history_t = [0.0]
        self.history_collected = [0]
        self.history_lost = [0]
        self._acc = self._accel(self.pos)

    def _fresh(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """``k`` freshly injected ions at the reservoir, with thermal spread and drift."""
        x = self.rng.uniform(*RESERVOIR_X, k)
        y = self.rng.uniform(-0.5 * A0, 0.5 * A0, k)
        vx = self.rng.normal(V_DRIFT, V_THERMAL, k)
        vy = self.rng.normal(0.0, V_THERMAL, k)
        return np.vstack([x, y]), np.vstack([vx, vy])

    def _accel(self, pos: np.ndarray) -> np.ndarray:
        """Field from the electrodes (panel-only) plus from the other ions (tree walk)."""
        self.electrodes.solve(pos, self.charge)
        e_panel = (
            direct_field(
                pos,
                self.electrodes.midpoints,
                self.electrodes.charge,
                kernel="log",
                softening=SOFTENING,
            )[1]
            * self.electrodes.k_e
        )
        tree = FlatQuadtree(
            pos, self.charge, self.root, kernel="log", leaf_capacity=self.leaf_capacity
        )
        e_tree = tree.evaluate(
            pos, exclude=np.arange(pos.shape[1]), theta=self.theta, softening=SOFTENING
        )[1]
        return K_E * (e_panel + e_tree)

    def step(self) -> None:
        """Leapfrog: half-kick, drift, absorb/collect/recycle, half-kick."""
        self.vel = self.vel + 0.5 * DT * self._acc
        x_prev = self.pos[0].copy()
        self.pos = self.pos + DT * self.vel

        # inside a plate at the end of the step, or through its mid-plane outside the
        # aperture during it (so a fast ion cannot step over a plate)
        hit_plate = np.zeros(self.pos.shape[1], dtype=bool)
        for x_e, half_ap, shape in self.plates:
            crossed = (x_prev - x_e) * (self.pos[0] - x_e) <= 0
            hit_plate |= crossed & (np.abs(self.pos[1]) > half_ap)
            hit_plate |= np.asarray(shape.contains_points(self.pos), dtype=bool)
        collected = self.pos[0] > self.x_final
        off_axis = np.abs(self.pos[1]) > PLATE_LENGTH + max(A0, A1) + 2
        back = self.pos[0] < RESERVOIR_X[0] - 1.0
        recycle = hit_plate | collected | off_axis | back
        n_recycle = int(np.sum(recycle))
        self.n_collected += int(np.sum(collected))
        self.n_lost += int(np.sum(hit_plate | off_axis | back))
        if n_recycle:
            new_pos, new_vel = self._fresh(n_recycle)
            self.pos[:, recycle] = new_pos
            self.vel[:, recycle] = new_vel

        self._acc = self._accel(self.pos)
        self.vel = self.vel + 0.5 * DT * self._acc
        self.time += DT
        self.history_t.append(self.time)
        self.history_collected.append(self.n_collected)
        self.history_lost.append(self.n_lost)

    def transmission(self) -> float:
        """Fraction of ions leaving the reservoir that are collected rather than lost."""
        total = self.n_collected + self.n_lost
        return self.n_collected / total if total else float("nan")


def add_plates(fig: go.Figure, source: ExtractionSource, color: str = "black") -> None:
    """Draw the electrode plates as filled polygons from their simgeometry outlines."""
    for _, _, shape in source.plates:
        for loop in shape.outlines():
            fig.add_trace(
                go.Scatter(
                    x=np.append(loop[0], loop[0, 0]),
                    y=np.append(loop[1], loop[1, 0]),
                    mode="lines",
                    fill="toself",
                    fillcolor=color,
                    line={"color": color, "width": 1},
                    showlegend=False,
                )
            )


# -- 1: geometry -----------------------------------------------------------------------------


def plot_geometry(output: Path, source: ExtractionSource, v_ext: float) -> None:
    """Figure 1."""
    fig = go.Figure()
    add_plates(fig, source, MPL_COLORS["C0"])
    midpoints = source.electrodes.midpoints
    fig.add_trace(
        go.Scatter(
            x=midpoints[0],
            y=midpoints[1],
            mode="markers",
            marker={"size": 3, "color": "black"},
            name=f"panel midpoints ({source.electrodes.num_panels})",
        )
    )
    for x, v, label in ((X0, 0.0, "plasma electrode"), (X1, -v_ext, "extractor")):
        fig.add_annotation(
            x=x, y=PLATE_LENGTH + 2.0, text=f"{label}<br>V = {v:g}", showarrow=False
        )
    fig.update_xaxes(title_text="x", range=[X0 - 2, X1 + 2], showgrid=False)
    fig.update_yaxes(title_text="y", scaleanchor="x", showgrid=False)
    fig.update_layout(
        title="Plasma electrode and extractor, each a grounded or biased slit"
    )
    save_figure(fig, output / "geometry")


# -- 2: beam snapshot --------------------------------------------------------------------------


def plot_beam_snapshot(
    output: Path, source: ExtractionSource, burst_steps: int
) -> None:
    """Figure 2."""
    for _ in range(burst_steps):
        source.step()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=source.pos[0],
            y=source.pos[1],
            mode="markers",
            marker={"size": 3, "color": MPL_COLORS["C0"], "opacity": 0.6},
            showlegend=False,
        )
    )
    add_plates(fig, source)
    fig.update_xaxes(title_text="x", showgrid=False)
    fig.update_yaxes(title_text="y", scaleanchor="x", showgrid=False)
    fig.update_layout(
        title=f"Steady extracted beam (t = {source.time:.1f}, "
        f"transmission = {100 * source.transmission():.0f}%)"
    )
    save_figure(fig, output / "beam_snapshot")


# -- 3: envelope -------------------------------------------------------------------------------


def plot_envelope(
    output: Path,
    source: ExtractionSource,
    *,
    average_steps: int,
    sample_every: int,
    bins: int,
) -> None:
    """Figure 3."""
    x_min, x_max = RESERVOIR_X[0], source.x_final
    edges = np.linspace(x_min, x_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sq_sum = np.zeros(bins)
    count_sum = np.zeros(bins)
    for i in range(average_steps):
        source.step()
        if (i + 1) % sample_every == 0:
            idx = np.digitize(source.pos[0], edges) - 1
            valid = (idx >= 0) & (idx < bins)
            np.add.at(sq_sum, idx[valid], source.pos[1, valid] ** 2)
            np.add.at(count_sum, idx[valid], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(sq_sum / count_sum)
    for x_probe in (X0, 0.5 * (X0 + X1), X1, source.x_final - 1.0):
        i = min(np.searchsorted(edges, x_probe) - 1, bins - 1)
        print(f"  rms half-width at x = {centers[i]:.1f}: {rms[i]:.3f}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=rms,
            mode="lines",
            line={"color": MPL_COLORS["C0"]},
            showlegend=False,
        )
    )
    for x_e, half_ap in ((X0, A0), (X1, A1)):
        fig.add_hline(y=half_ap, line={"color": "gray", "dash": "dot"})
        fig.add_vline(x=x_e, line={"color": "gray", "dash": "dot"})
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="beam rms half-width (time-averaged)")
    fig.update_layout(
        title="Beam envelope: narrow at the plasma electrode, spreading beyond it"
    )
    save_figure(fig, output / "envelope")


# -- 4: transmission scaling -------------------------------------------------------------------


def plot_transmission_scaling(
    output: Path,
    charges: np.ndarray,
    *,
    n: int,
    v_ext: float,
    tip_panel: float,
    burst_steps: int,
) -> None:
    """Figure 4."""
    transmission = []
    for q in charges:
        source = ExtractionSource(
            n, float(q), v_ext=v_ext, x_final=X1 + 12.0, tip_panel=tip_panel, seed=1
        )
        for _ in range(burst_steps):
            source.step()
        transmission.append(source.transmission())
        print(f"  q = {q:.4f}: transmission = {100 * transmission[-1]:.1f}%")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=charges,
            y=100 * np.array(transmission),
            mode="lines+markers",
            line={"color": MPL_COLORS["C0"]},
            showlegend=False,
        )
    )
    fig.update_xaxes(title_text="macroparticle charge (space-charge strength)")
    fig.update_yaxes(title_text="transmission past the extractor (%)")
    fig.update_layout(
        title="Space-charge-limited extraction: the extractor intercepts more as the beam's own repulsion grows"
    )
    save_figure(fig, output / "transmission_scaling")


# -- 5: beam formation (animation) --------------------------------------------------------------


def animate_beam_formation(
    output: Path, source: ExtractionSource, *, frames: int, steps_per_frame: int
) -> None:
    """Figure 5."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 3, "color": MPL_COLORS["C0"], "opacity": 0.6},
            showlegend=False,
        )
    )
    add_plates(fig, source)
    fig.update_xaxes(
        title_text="x", range=[RESERVOIR_X[0] - 1, source.x_final + 1], showgrid=False
    )
    fig.update_yaxes(
        title_text="y",
        range=[-PLATE_LENGTH - 2, PLATE_LENGTH + 2],
        scaleanchor="x",
        showgrid=False,
    )

    frame_list = []
    for frame_i in range(frames):
        for _ in range(steps_per_frame):
            source.step()
        title = (
            f"t = {source.time:.2f}: transmission = {100 * source.transmission():.0f}%"
        )
        frame_list.append(
            go.Frame(
                name=f"{frame_i}",
                data=[go.Scatter(x=source.pos[0], y=source.pos[1])],
                traces=[0],
                layout=go.Layout(title=title),
            )
        )
    fig.frames = frame_list
    if frame_list:
        fig.update_layout(title=frame_list[0].layout.title.text)
    _add_player(fig, frame_list, prefix="frame ")
    save_figure(fig, output / "beam_formation", frame=-1)


# -- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run everything and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    v_ext = 8.0
    x_final = X1 + 12.0
    if args.smoke:
        n, tip_panel, burst, avg, bins = 60, 0.4, 20, 10, 6
        charges = np.array([1 / 400, 1 / 100])
        frames, steps_per_frame = 3, 3
    elif args.quick:
        n, tip_panel, burst, avg, bins = 400, 0.25, 250, 100, 20
        charges = np.array([1 / 2000, 1 / 1000, 1 / 600, 1 / 400, 1 / 250])
        frames, steps_per_frame = 15, 6
    else:
        n, tip_panel, burst, avg, bins = 1500, 0.15, 600, 300, 40
        charges = np.array([1 / 2000, 1 / 1200, 1 / 800, 1 / 500, 1 / 350, 1 / 250])
        frames, steps_per_frame = 40, 8

    q = 1 / 500
    source = ExtractionSource(
        n, q, v_ext=v_ext, x_final=x_final, tip_panel=tip_panel, seed=2
    )
    print(f"1. geometry ({source.electrodes.num_panels} panels)")
    plot_geometry(output, source, v_ext)

    print("2. beam snapshot")
    plot_beam_snapshot(output, source, burst)

    print("3. envelope")
    plot_envelope(
        output, source, average_steps=avg, sample_every=max(1, avg // 20), bins=bins
    )

    print("4. transmission scaling")
    plot_transmission_scaling(
        output, charges, n=n, v_ext=v_ext, tip_panel=tip_panel, burst_steps=burst
    )

    print("5. beam formation")
    anim_source = ExtractionSource(
        n // 4 if n >= 60 else n,
        q,
        v_ext=v_ext,
        x_final=x_final,
        tip_panel=tip_panel,
        seed=3,
    )
    animate_beam_formation(
        output, anim_source, frames=frames, steps_per_frame=steps_per_frame
    )

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
