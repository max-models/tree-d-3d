"""A planar Pierce electron gun: a space-charge-limited sheet beam kept parallel.

In a planar diode the space-charge-limited current density is the Child-Langmuir
``J_CL = (4 / 9) eps0 sqrt(2 q / m) V^(3/2) / d^2``, with the potential
``V (x / d)^(4/3)`` across the gap. A beam of finite width breaks that solution at its
edge: outside the beam the potential would be linear, the mismatch pushes transverse
fields into the beam and its edge spreads. Pierce (1940) found the electrodes that make
the vacuum outside the beam continue the Child-Langmuir potential with no transverse
field at the edge: for a sheet beam, the harmonic continuation
``Re[(x + i (y - w))^(4/3)]``, whose zero equipotential leaves the beam edge at 67.5
degrees from the beam and whose ``V`` equipotential is the curve
``r cos(4 theta / 3)^(3/4) = d``. Those are the cathode's focus electrodes and the anode.

The gun here builds both as thin simgeometry ``Polyline`` electrodes (the cathode strip
with its 67.5-degree wings, and the anode: the two curves joined by a grid across the
beam, which the electrons pass through, as in Pierce's planar solution; an open anode
slit would lower the limiting current), solves them with
:class:`tree_d_3d.BoundaryIntegralElectrodes`, and finds the steady beam by the gun-code
fixed point: trace the electron rays, turn them into line charges (each segment of a ray
of current ``dI`` holds ``dI dt``), whose field the tree code evaluates and to which the
electrodes respond, and trace again. Rays start a small distance ``x0`` from the cathode
at the local potential; the thin layer they skip is added as the Child-Langmuir charge
``3 x0 dI / v0``, and Gauss's law across it gives the cathode field
``E(0) = E(x0) - sigma_layer`` (in units ``eps0 = 1``): the current is space-charge
limited when it vanishes. Code units ``eps0 = m = |q| = d = V = 1``, so ``J_CL = 0.6285``.

1. ``geometry``: the Pierce electrodes, the potential and the converged rays at
   ``J_CL``.
2. ``axial_potential``: the potential along the beam axis against the Child-Langmuir
   ``V (x / d)^(4/3)``.
3. ``cathode_field``: the cathode field at the beam centre against the current density;
   it vanishes near ``J_CL``, the space-charge limit: at about ``1.07 J_CL`` with the
   ``--quick`` rays, a value that stays put when the panels are refined fourfold, so the
   excess is not an electrode error.
4. ``beam_edge``: the outermost ray with the Pierce electrodes and in a flat diode of the
   same gap (wide cathode and gridded anode planes, emitting only over the beam width):
   with them the edge stays within a few percent of parallel (drawn slightly in by the
   discrete rays and the truncated focus electrodes), without them it spreads by about a
   fifth across the gap.
5. ``convergence``: the change of the rays per fixed-point iteration, with
   under-relaxation 0.5, down to the noise of the discrete rays.

Usage::

    python examples/pierce_gun.py [--output examples/output] [--quick]

``--quick`` uses fewer rays and current values (under a minute); ``--smoke`` is the tiny
configuration the test suite runs. Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from simgeometry import Polyline

from tree_d_3d import (
    BoundaryIntegralElectrodes,
    FlatQuadtree,
    Quad,
    conductors_from_shape,
)

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}
K_E = 1.0 / (4.0 * np.pi)  # eps0 = 1
GAP = 1.0
VOLTAGE = 1.0
J_CL = 4.0 / 9.0 * np.sqrt(2.0) * VOLTAGE**1.5 / GAP**2
HALF_WIDTH = 0.5  # beam half-width w
START = 0.02  # x0: rays start here
PIERCE_ANGLE = np.radians(67.5)
END = 1.3 * GAP  # rays are traced to here, past the anode


def pierce_electrodes() -> list[tuple[Polyline, float]]:
    """The cathode with its 67.5-degree focus wings, and the two anode curves."""
    wing = 1.2 * GAP
    dx, dy = wing * np.cos(PIERCE_ANGLE), wing * np.sin(PIERCE_ANGLE)
    w = HALF_WIDTH
    cathode = Polyline([(dx, -w - dy), (0.0, -w), (0.0, w), (dx, w + dy)])
    theta = np.linspace(0.0, np.radians(55.0), 40)
    r = GAP / np.cos(4 * theta / 3) ** 0.75
    upper = np.stack((r * np.cos(theta), w + r * np.sin(theta)), axis=1)
    lower = np.stack((r * np.cos(theta), -w - r * np.sin(theta)), axis=1)
    # one anode: the lower curve, the grid across the beam, the upper curve
    anode = Polyline(np.concatenate((lower[::-1], upper)))
    return [(cathode, 0.0), (anode, VOLTAGE)]


def flat_electrodes() -> list[tuple[Polyline, float]]:
    """A flat diode of the same gap: wide cathode and anode planes, no focus electrodes."""
    outer = HALF_WIDTH + 1.2 * GAP
    cathode = Polyline([(0.0, -outer), (0.0, outer)])
    anode = Polyline([(GAP, -outer), (GAP, outer)])
    return [(cathode, 0.0), (anode, VOLTAGE)]


@dataclass
class Gun:
    """The rays of a sheet beam of current density ``j`` between ``electrodes``."""

    electrodes: list[tuple[Polyline, float]]
    j: float
    rays: int
    dt: float
    spacing: float
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        parts = [
            c
            for s, v in self.electrodes
            for c in conductors_from_shape(s, v, self.spacing)
        ]
        self.bem = BoundaryIntegralElectrodes(parts, k_e=K_E)
        self.y0 = HALF_WIDTH * (2 * (np.arange(self.rays) + 0.5) / self.rays - 1)
        self.d_i = self.j * 2 * HALF_WIDTH / self.rays  # current of each ray
        self.softening = 0.5 * 2 * HALF_WIDTH / self.rays
        self.root = Quad(0.6 * GAP, 0.0, 2.0 * GAP)
        self.steps = int(np.ceil(4 * END / (np.sqrt(2 * VOLTAGE) * self.dt)))
        self.x = None  # rays (steps + 1, 2, rays), NaN after a ray ends
        self.v0 = None

    def sources(self) -> tuple[np.ndarray, np.ndarray]:
        """Electron line charges: ray segments (``-dI dt``) and the skipped layer."""
        if self.x is None:
            return np.zeros((2, 0)), np.zeros(0)
        a, b = self.x[:-1], self.x[1:]
        ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
        mid = 0.5 * (a + b)
        xy = np.stack((mid[:, 0][ok], mid[:, 1][ok]))
        q = np.full(xy.shape[1], -self.d_i * self.dt)
        layer = np.stack((np.full(self.rays, START / 4), self.y0))
        q_layer = -self.d_i * 3 * START / self.v0
        return np.concatenate((xy, layer), axis=1), np.concatenate((q, q_layer))

    def probe(self, points: np.ndarray, tree) -> tuple[np.ndarray, np.ndarray]:
        """Potential and field: electrodes (panels) plus the beam (tree code)."""
        phi, e = self.bem.panel_field(points)
        if tree is not None:
            phi_b, e_b = tree.evaluate(points, theta=0.5, softening=self.softening)
            phi, e = phi + K_E * np.asarray(phi_b), e + K_E * np.asarray(e_b)
        return phi, e

    def iterate(self, iterations: int, omega: float = 0.5) -> None:
        """Fixed-point iterations: solve the electrodes and the beam, retrace the rays."""
        for _ in range(iterations):
            xy, q = self.sources()
            self.bem.solve(xy, q)
            tree = (
                FlatQuadtree(xy, q, self.root, kernel="log", order=12, leaf_capacity=32)
                if q.size
                else None
            )
            start = np.stack((np.full(self.rays, START), self.y0))
            phi0, _ = self.probe(start, tree)
            v0 = np.sqrt(2 * np.maximum(phi0, 1e-4 * VOLTAGE))
            x = np.full((self.steps + 1, 2, self.rays), np.nan)
            pos, vel = start.copy(), np.stack((v0, np.zeros(self.rays)))
            x[0] = pos
            alive = np.ones(self.rays, dtype=bool)
            _, e = self.probe(pos, tree)
            for i in range(self.steps):
                vel = vel - 0.5 * self.dt * e  # q / m = -1
                pos = pos + self.dt * vel
                _, e = self.probe(pos, tree)
                vel = vel - 0.5 * self.dt * e
                alive &= (pos[0] >= 0.0) & (pos[0] <= END)
                x[i + 1][:, alive] = pos[:, alive]
                if not alive.any():
                    break
            if self.x is not None:
                change = float(np.nanmax(np.abs(x[:, 1] - self.x[:, 1])))
                self.history.append(change)
                x = np.where(
                    np.isfinite(self.x) & np.isfinite(x),
                    (1 - omega) * self.x + omega * x,
                    x,
                )
            self.x, self.v0 = x, v0
        xy, q = self.sources()
        self.bem.solve(xy, q)
        self.tree = FlatQuadtree(
            xy, q, self.root, kernel="log", order=12, leaf_capacity=32
        )

    def cathode_slope(self) -> float:
        """``dphi/dx`` at the cathode at the beam centre, by Gauss's law across the layer."""
        centre = np.argmin(np.abs(self.y0))
        point = np.array([[START], [self.y0[centre]]])
        _, e = self.probe(point, self.tree)
        sigma = self.d_i / (2 * HALF_WIDTH / self.rays) * 3 * START / self.v0[centre]
        return float(-e[0, 0] - sigma)

    def edge(self) -> tuple[np.ndarray, np.ndarray]:
        """The outermost ray's ``x`` and ``y``."""
        ok = np.isfinite(self.x[:, 0, -1])
        return self.x[ok, 0, -1], self.x[ok, 1, -1]


def main(argv: list[str] | None = None) -> Path:
    """Run the gun and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        rays, dt, spacing, iterations, scan = 8, 0.05, 0.1, 3, (0.8, 1.2)
    elif args.quick:
        rays, dt, spacing, iterations, scan = (
            20,
            0.02,
            0.04,
            10,
            (0.7, 0.85, 1.0, 1.15, 1.3),
        )
    else:
        rays, dt, spacing, iterations, scan = (
            40,
            0.01,
            0.02,
            15,
            tuple(np.linspace(0.6, 1.4, 9)),
        )

    # the gun at J_CL
    gun = Gun(pierce_electrodes(), J_CL, rays, dt, spacing)
    gun.iterate(iterations)
    print(f"Pierce gun at J_CL: cathode dphi/dx = {gun.cathode_slope():+.4f} (V/d = 1)")

    # 1: geometry
    xs = np.linspace(-0.1, 1.4, 120)
    ys = np.linspace(-1.6, 1.6, 120)
    xx, yy = np.meshgrid(xs, ys)
    phi, _ = gun.probe(np.stack((xx.ravel(), yy.ravel())), gun.tree)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=np.clip(phi.reshape(xx.shape), -0.2, 1.2),
            colorscale="Viridis",
            contours={"start": 0.0, "end": 1.0, "size": 0.1},
            colorbar={"title": "phi", "x": 1.02},
            opacity=0.7,
        )
    )
    for shape, potential in gun.electrodes:
        (loop,) = shape.outlines()
        fig.add_trace(
            go.Scatter(
                x=loop[0],
                y=loop[1],
                mode="lines",
                line={"color": "black", "width": 4},
                showlegend=False,
                hovertext=f"V = {potential:g}",
            )
        )
    for k in range(rays):
        ok = np.isfinite(gun.x[:, 0, k])
        fig.add_trace(
            go.Scatter(
                x=gun.x[ok, 0, k],
                y=gun.x[ok, 1, k],
                mode="lines",
                line={"color": COLORS["C3"], "width": 1},
                showlegend=False,
            )
        )
    fig.update_xaxes(title_text="x / d", range=[-0.1, 1.4], constrain="domain")
    fig.update_yaxes(title_text="y / d", range=[-1.6, 1.6], scaleanchor="x")
    fig.update_layout(title="Pierce sheet-beam gun at the Child-Langmuir current")
    save_figure(fig, output / "geometry")

    # 2: axial potential
    xa = np.linspace(START, 0.98 * GAP, 60)  # off the cathode's and the grid's panels
    phi_a, _ = gun.probe(np.stack((xa, np.zeros_like(xa))), gun.tree)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xa,
            y=phi_a,
            mode="markers",
            marker={"color": COLORS["C0"]},
            name="self-consistent beam",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xa,
            y=VOLTAGE * (xa / GAP) ** (4 / 3),
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="Child-Langmuir V (x/d)^(4/3)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xa,
            y=VOLTAGE * xa / GAP,
            mode="lines",
            line={"color": "gray", "dash": "dot"},
            name="vacuum V x/d",
        )
    )
    fig.update_xaxes(title_text="x / d")
    fig.update_yaxes(title_text="potential on the axis")
    fig.update_layout(title="Potential along the beam axis")
    save_figure(fig, output / "axial_potential")
    inner = (xa > 0.1) & (xa < 0.9)
    deviation = np.max(np.abs(phi_a[inner] - (xa[inner] / GAP) ** (4 / 3)))
    print(
        f"axial potential: max deviation from (x/d)^(4/3) on 0.1 < x < 0.9: {deviation:.4f}"
    )

    # 3: cathode field against the current density
    slopes = []
    for ratio in scan:
        g = Gun(pierce_electrodes(), ratio * J_CL, rays, dt, spacing)
        g.iterate(iterations)
        slopes.append(g.cathode_slope())
    slopes = np.array(slopes)
    ratios = np.array(scan)
    crossing = np.flatnonzero(np.diff(np.sign(slopes)))
    if len(crossing):
        i = crossing[0]
        j_scl = ratios[i] - slopes[i] * (ratios[i + 1] - ratios[i]) / (
            slopes[i + 1] - slopes[i]
        )
        print(f"space-charge limit: J = {j_scl:.3f} J_CL (cathode field zero)")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ratios,
            y=slopes,
            mode="lines+markers",
            line={"color": COLORS["C0"]},
            name="dphi/dx at the cathode (Gauss's law)",
        )
    )
    fig.add_hline(y=0.0, line={"color": "gray", "dash": "dot"})
    fig.add_vline(
        x=1.0, line={"color": "black", "dash": "dash"}, annotation_text="J_CL"
    )
    fig.update_xaxes(title_text="J / J_CL")
    fig.update_yaxes(title_text="cathode field dphi/dx (V/d)")
    fig.update_layout(title="The space-charge limit: the cathode field vanishes")
    save_figure(fig, output / "cathode_field")

    # 4: beam edge, Pierce against flat
    flat = Gun(flat_electrodes(), J_CL, rays, dt, spacing)
    flat.iterate(iterations)
    fig = go.Figure()
    for g, name, color in (
        (gun, "Pierce electrodes", "C0"),
        (flat, "flat diode", "C3"),
    ):
        ex, ey = g.edge()
        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey / HALF_WIDTH,
                mode="lines",
                line={"color": COLORS[color]},
                name=name,
            )
        )
        at_anode = np.interp(GAP, ex, ey) / g.y0[-1]
        print(f"{name}: outermost ray at the anode / at the cathode = {at_anode:.4f}")
    fig.add_vline(x=GAP, line={"color": "gray", "dash": "dot"}, annotation_text="anode")
    fig.update_xaxes(title_text="x / d")
    fig.update_yaxes(title_text="outermost ray y / w")
    fig.update_layout(title="Beam edge with and without the Pierce electrodes at J_CL")
    save_figure(fig, output / "beam_edge")

    # 5: convergence
    fig = go.Figure()
    for g, name, color in ((gun, "Pierce, J_CL", "C0"), (flat, "flat, J_CL", "C3")):
        fig.add_trace(
            go.Scatter(
                x=np.arange(2, len(g.history) + 2),
                y=g.history,
                mode="lines+markers",
                line={"color": COLORS[color]},
                name=name,
            )
        )
    fig.update_xaxes(title_text="iteration")
    fig.update_yaxes(type="log", title_text="max change of the rays (d)")
    fig.update_layout(title="Fixed-point convergence of the gun iteration")
    save_figure(fig, output / "convergence")

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
