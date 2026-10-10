"""Steady-state space-charge optics without a mesh: trajectories as line charges.

A steady beam is a bundle of trajectories; the trajectory carrying the current ``dI``
(per unit depth, in 2D) at speed ``v`` is a curve of line charge ``dI / v`` per unit
length. Its space-charge field is then a sum over the charged segments of all
trajectories, which the tree code evaluates in O(N) once the segments are turned into
sources, and the self-consistent beam is the fixed point "trace in the field of the
segments, rebuild the segments". Mesh codes (IBSimu, picnic's ``BeamOptics``) deposit
the trajectories on a grid and solve Poisson's equation instead. This example runs the
gridless loop on the one 2D case with an exact answer, a uniform sheet beam expanding
under its own space charge: inside a uniform slab the field is linear in ``y``, so the
beam stays uniform and every trajectory follows ``y(x) = y0 (1 + kappa x^2)`` with
``kappa = q I / (4 eps0 m v0^3)``, the planar analogue of the envelope equation. Units
``q = m = eps0 = v0 = 1``, half-width 1, ``I = 0.01`` (``kappa = 1/400``, so the beam
doubles in width over 20 units).

Each segment is represented by Gauss-Legendre points carrying its charge
``dI * dt``, so no new kernel is needed; the segments of a trajectory are smoothed over a
fraction of the trajectory spacing (the "ribbon", a Plummer softening), which is what
regularises the field a trajectory sees from itself and from its neighbours. A beam that starts and stops has
fringe fields at its ends that the infinite slab does not (about ``a / (pi s)`` of the
slab field at distance ``s`` from an end); mesh codes remove them with Neumann planes at
the domain ends, and the gridless code does the same by mirroring the segments at
``x = 0`` and ``x = L`` (``images`` reflections on each side), which is exact for the
self-similar beam at the entrance, where its mirror image is the converging beam.

1. ``segment_kernel``: field error of 1, 2 and 3 Gauss-Legendre points per segment
   against the closed-form field of a uniform segment, against the distance over the
   segment length: the point representation is exponentially accurate beyond a segment
   length and only the near field needs the exact segment.
2. ``trajectories``: the converged trajectories, the envelope against the exact
   parabola, and the field-free straight rays the iteration starts from.
3. ``convergence``: the change of the trajectories per iteration for several
   under-relaxation factors; for this beam plain substitution (``omega = 1``) converges
   fastest, and under-relaxation only slows it down.
4. ``envelope_error``: the envelope's deviation from the parabola along the beam for
   several trajectory counts (ribbon width 0.3 spacings) and for several numbers of
   mirror images: the trajectory count hardly matters, the images do, because a
   mirrored beam is still finite and the missing far slabs at distance ``s`` cost about
   ``a / (pi s)`` of the field; the deviation falls as the images extend the beam.
5. ``ribbon``: the same deviation against the ribbon width in trajectory spacings: a
   Plummer ribbon as wide as the spacing weakens the neighbours' field (the softened
   field at one spacing is half the bare one, a deficit of about one trajectory's worth
   in the whole slab), a narrow one leaves the discrete trajectories' noise; the width
   is a numerical parameter of the method, as the cell size is for a mesh.

Usage::

    python examples/segment_beam.py [--output examples/output] [--quick]

``--quick`` uses fewer trajectories (a few minutes instead of ten); ``--smoke`` is the
tiny configuration the test suite runs. Needs Plotly
(``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure

from tree_d_3d import FlatQuadtree, Quad

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}

K_E = 1.0 / (4.0 * np.pi)  # eps0 = 1
CURRENT = 0.01  # per unit depth; kappa = I / 4
HALF_WIDTH = 1.0
LENGTH = 30.0
KAPPA = CURRENT / 4.0

GAUSS = {
    1: (np.array([0.5]), np.array([1.0])),
    2: (0.5 + np.array([-1, 1]) / (2 * np.sqrt(3)), np.array([0.5, 0.5])),
    3: (0.5 + np.array([-1, 0, 1]) * np.sqrt(0.6) / 2, np.array([5, 8, 5]) / 18),
}


def segment_field_exact(
    z: np.ndarray, w1: complex, w2: complex, lam: float
) -> np.ndarray:
    """``E_x - i E_y`` of a uniform segment ``w1 -> w2`` of line-charge density ``lam``."""
    length = abs(w2 - w1)
    return 2 * K_E * lam * length / (w2 - w1) * np.log((z - w1) / (z - w2))


def segment_field_points(z: np.ndarray, w1: complex, w2: complex, lam: float, n: int):
    """The same field from ``n`` Gauss-Legendre point charges on the segment."""
    nodes, weights = GAUSS[n]
    out = np.zeros_like(z)
    for t, w in zip(nodes, weights, strict=True):
        out += 2 * K_E * lam * abs(w2 - w1) * w / (z - (w1 + t * (w2 - w1)))
    return out


@dataclass
class Beam:
    """The sheet beam's trajectories and the fixed-point iteration."""

    rays: int
    dx: float = 0.2
    softening_spacings: float = 0.3
    points_per_segment: int = 2
    images: int = 6

    def __post_init__(self) -> None:
        self.y0 = HALF_WIDTH * (2 * (np.arange(self.rays) + 0.5) / self.rays - 1)
        self.d_i = CURRENT / self.rays
        self.steps = int(round(LENGTH / self.dx))
        self.dt = self.dx  # v0 = 1
        self.softening = self.softening_spacings * 2 * HALF_WIDTH / self.rays
        self.root = Quad(LENGTH / 2, 0.0, (2 * self.images + 2) * LENGTH)
        # trajectories: positions (steps + 1, 2, rays) and velocities, start field-free
        self.x = np.zeros((self.steps + 1, 2, self.rays))
        self.v = np.zeros((self.steps + 1, 2, self.rays))
        self.x[:, 0] = np.arange(self.steps + 1)[:, None] * self.dx
        self.x[:, 1] = self.y0[None, :]
        self.v[:, 0] = 1.0

    def sources(self) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Legendre point charges of every segment of the current trajectories."""
        nodes, weights = GAUSS[self.points_per_segment]
        a, b = self.x[:-1], self.x[1:]
        pts = [a + t * (b - a) for t in nodes]  # each (steps, 2, rays)
        xy = np.concatenate([p.transpose(1, 0, 2).reshape(2, -1) for p in pts], axis=1)
        q = np.concatenate(
            [np.full(self.steps * self.rays, w * self.d_i * self.dt) for w in weights]
        )
        # mirror images at x = 0 and x = L: x -> 2 k L +- x
        copies, charges = [], []
        for k in range(-self.images, self.images + 1):
            for sign in (1, -1):
                if k == 0 and sign == 1:
                    continue
                copies.append(np.stack((2 * k * LENGTH + sign * xy[0], xy[1])))
                charges.append(q)
        if copies:
            xy = np.concatenate([xy, *copies], axis=1)
            q = np.concatenate([q, *charges])
        return xy, q

    def trace(self, tree: FlatQuadtree):
        """Integrate all rays through the field of ``tree`` (midpoint rule)."""
        x = np.zeros_like(self.x)
        v = np.zeros_like(self.v)
        x[0] = self.x[0]
        v[0] = self.v[0]

        def accel(pos):
            _, e = tree.evaluate(pos, theta=0.5, softening=self.softening)
            return K_E * np.asarray(e)

        for i in range(self.steps):
            a1 = accel(x[i])
            xm = x[i] + 0.5 * self.dt * v[i]
            vm = v[i] + 0.5 * self.dt * a1
            am = accel(xm)
            x[i + 1] = x[i] + self.dt * vm
            v[i + 1] = v[i] + self.dt * am
        return x, v

    def iterate(self, omega: float, max_iter: int, tol: float = 1e-6) -> list[float]:
        """Fixed-point iteration with under-relaxation ``omega``; return the residuals."""
        history = []
        for _ in range(max_iter):
            xy, q = self.sources()
            tree = FlatQuadtree(
                xy, q, self.root, kernel="log", order=12, leaf_capacity=32
            )
            x_new, v_new = self.trace(tree)
            residual = float(np.abs(x_new[:, 1] - self.x[:, 1]).max())
            history.append(residual)
            self.x = (1 - omega) * self.x + omega * x_new
            self.v = (1 - omega) * self.v + omega * v_new
            if residual < tol:
                break
        return history

    def envelope(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Axial positions, outermost-ray height, and the exact self-similar height."""
        xs = self.x[:, 0, -1]
        return xs, self.x[:, 1, -1], self.y0[-1] * (1 + KAPPA * xs**2)


def main(argv: list[str] | None = None) -> Path:
    """Run the beam and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        rays, scan, omegas, max_iter = 16, (8, 16), (0.5, 1.0), 6
    elif args.quick:
        rays, scan, omegas, max_iter = 64, (16, 32, 64), (0.3, 0.6, 1.0), 25
    else:
        rays, scan, omegas, max_iter = 128, (32, 64, 128, 256), (0.3, 0.6, 1.0), 40

    # 1. point representation of a segment
    w1, w2 = -0.5 + 0j, 0.5 + 0j
    angles = np.linspace(0, np.pi, 37)
    distances = np.geomspace(0.3, 20, 60)
    fig = go.Figure()
    for n, color in ((1, COLORS["C0"]), (2, COLORS["C1"]), (3, COLORS["C2"])):
        errors = []
        for d in distances:
            z = d * np.exp(1j * angles)
            exact = segment_field_exact(z, w1, w2, 1.0)
            approx = segment_field_points(z, w1, w2, 1.0, n)
            errors.append(np.max(np.abs(approx - exact) / np.abs(exact)))
        fig.add_trace(
            go.Scatter(
                x=distances,
                y=errors,
                mode="lines",
                name=f"{n} point(s) per segment",
                line={"color": color},
            )
        )
    fig.update_xaxes(
        type="log", title_text="distance from the segment centre / segment length"
    )
    fig.update_yaxes(type="log", title_text="max relative field error over directions")
    fig.update_layout(title="Gauss-Legendre points against the exact uniform segment")
    save_figure(fig, output / "segment_kernel")

    # 2. the base beam
    beam = Beam(rays)
    straight = beam.x.copy()
    history = beam.iterate(1.0, max_iter)
    xs, env, exact = beam.envelope()
    inner = (xs > 10) & (xs < 20)
    print(f"{rays} rays: {len(history)} iterations, last residual {history[-1]:.2e}")
    print(f"  envelope error over 10 < x < 20: {np.abs(env - exact)[inner].max():.3e}")

    fig = go.Figure()
    every = max(1, rays // 16)
    for k in range(0, rays, every):
        fig.add_trace(
            go.Scatter(
                x=straight[:, 0, k],
                y=straight[:, 1, k],
                mode="lines",
                line={"color": "lightgray", "width": 1},
                showlegend=k == 0,
                name="field-free start",
            )
        )
    for k in range(0, rays, every):
        fig.add_trace(
            go.Scatter(
                x=beam.x[:, 0, k],
                y=beam.x[:, 1, k],
                mode="lines",
                line={"color": COLORS["C0"], "width": 1},
                showlegend=k == 0,
                name="converged trajectories",
            )
        )
    for sign in (1, -1):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=sign * exact,
                mode="lines",
                line={"color": "black", "dash": "dash"},
                name="exact envelope y0 (1 + kappa x^2)",
                showlegend=sign == 1,
            )
        )
    fig.update_xaxes(title_text="x", showgrid=False)
    fig.update_yaxes(title_text="y", scaleanchor="x", showgrid=False)
    fig.update_layout(title="Sheet beam expanding under its own space charge")
    save_figure(fig, output / "trajectories")

    # 3. convergence against the relaxation factor
    fig = go.Figure()
    for i, omega in enumerate(omegas):
        if omega == 1.0:
            h = history
        else:
            h = Beam(rays).iterate(omega, max_iter)
        print(
            f"  omega = {omega}: {len(h)} iterations, residuals {h[0]:.2e} -> {h[-1]:.2e}"
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(h) + 1),
                y=h,
                mode="lines+markers",
                name=f"omega = {omega}",
                line={"color": COLORS[f"C{i}"]},
            )
        )
    fig.update_xaxes(title_text="iteration")
    fig.update_yaxes(
        type="log", title_text="max change of a trajectory's y per iteration"
    )
    fig.update_layout(title="Convergence of the trajectory / space-charge fixed point")
    save_figure(fig, output / "convergence")

    # 4. envelope error against the number of rays
    fig = go.Figure()
    for i, k in enumerate(scan):
        b = beam if k == rays else Beam(k)
        if k != rays:
            b.iterate(1.0, max_iter)
        xs, env, exact = b.envelope()
        err = (env - exact) / (exact - b.y0[-1] + 1e-300)
        inner = (xs > 10) & (xs < 20)
        print(
            f"  {k} rays: relative envelope error over 10 < x < 20: {np.abs(err[inner]).max():.3e}"
        )
        fig.add_trace(
            go.Scatter(
                x=xs[1:],
                y=err[1:],
                mode="lines",
                name=f"{k} rays, 6 mirror images per side",
                line={"color": COLORS[f"C{i % 4}"]},
            )
        )
    for images, dash in ((0, "dot"), (2, "dash")):
        b = Beam(rays, images=images)
        b.iterate(1.0, max_iter)
        xs, env, exact = b.envelope()
        err = (env - exact) / (exact - b.y0[-1] + 1e-300)
        inner = (xs > 10) & (xs < 20)
        print(
            f"  {rays} rays, {images} images per side: relative envelope error "
            f"{np.abs(err[inner]).max():.3e}"
        )
        fig.add_trace(
            go.Scatter(
                x=xs[1:],
                y=err[1:],
                mode="lines",
                name=f"{rays} rays, {images} mirror images per side",
                line={"color": "black", "dash": dash},
            )
        )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(
        title_text="(envelope - parabola) / (parabola - y0)", range=[-0.3, 0.1]
    )
    fig.update_layout(title="Envelope error against the trajectory count")
    save_figure(fig, output / "envelope_error")

    # 5. ribbon width
    fig = go.Figure()
    k = scan[len(scan) // 2]
    for i, width in enumerate((0.1, 0.3, 1.0)):
        b = Beam(k, softening_spacings=width)
        b.iterate(1.0, max_iter)
        xs, env, exact = b.envelope()
        err = (env - exact) / (exact - b.y0[-1] + 1e-300)
        inner = (xs > 10) & (xs < 20)
        print(
            f"  {k} rays, ribbon {width} spacings: envelope error {np.abs(err[inner]).max():.3e}"
        )
        fig.add_trace(
            go.Scatter(
                x=xs[1:],
                y=err[1:],
                mode="lines",
                name=f"ribbon width {width} spacings",
                line={"color": COLORS[f"C{i}"]},
            )
        )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(
        title_text="(envelope - parabola) / (parabola - y0)", range=[-0.2, 0.05]
    )
    fig.update_layout(title=f"Envelope error against the ribbon width ({k} rays)")
    save_figure(fig, output / "ribbon")

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
