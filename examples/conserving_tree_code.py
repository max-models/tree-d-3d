"""A tree code that conserves both momentum and energy: the evidence, as figures.

Runs the experiments behind ``tree_d_3d.smooth`` and saves one figure per claim:

1. ``truncation_accuracy``: field error against the expansion order for the
   rectangle (standard) and triangle (symmetric) M2L truncations.
2. ``truncation_momentum``: the net force of the same runs; the triangle gives
   roundoff at every order, the rectangle the truncation error.
3. ``partition_of_unity``: the leaf boxes of an adaptive tree and the weight of one
   leaf with its smooth band, plus the sum of all weights (one everywhere).
4. ``gradient_check``: analytic force against finite differences of the energy, for
   particles blended into several leaves.
5. ``energy_drift``: relative energy error over a Verlet run for the direct sum,
   hard membership (fixed boxes and rebuilt every step), smooth membership, and
   smooth membership with co-moving boxes.
6. ``momentum_drift``: the net force over the same runs.
7. ``trajectories``: the particle trajectories of the conserving run over the fixed
   leaf boxes, and the distance of each variant's trajectories from the direct sum's.
8. ``trajectories_gentle``: a small ring of like charges orbiting in the trap,
   where the orbits are recognisable: the conserving tree code's trajectories over the
   direct sum's, and the energy error of both.
9. ``cloud``: the trapped cloud under the conserving scheme, animated, with its
   energy error.
10. ``orbits``: the ring of like charges, tree code over direct sum, animated.
11. ``membership``: a tracer crossing leaf edges, animated, with its leaf weights.
12. ``tree``: the quadtree rebuilt every frame while the particles move, animated:
    leaves that split and merge.

Usage::

    python examples/conserving_tree_code.py [--output examples/output] [--quick]

``--quick`` uses fewer particles and steps (about 20 s instead of a few minutes).
Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from plotly.subplots import make_subplots

from tree_d_3d.quad import Quad
from tree_d_3d.smooth import (
    BoxTree,
    SmoothFMM,
    comoving_fields,
    direct_energy,
    direct_fields,
)

# The Matplotlib "Cn" cycle, kept so the colours stay the ones each figure was designed around.
MPL_CYCLE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
MPL_COLORS = {f"C{i}": color for i, color in enumerate(MPL_CYCLE)}
MPL_COLORS["k"] = "black"


def _c(code: str) -> str:
    """A Matplotlib color code ("C0", "k", ...) as a Plotly color, or ``code`` unchanged."""
    return MPL_COLORS.get(code, code)


def _rect(
    cx: float,
    cy: float,
    h: float,
    *,
    color: str = "rgb(153,153,153)",
    width: float = 0.4,
    dash: str | None = None,
    fillcolor: str | None = None,
    opacity: float | None = None,
    xref: str = "x",
    yref: str = "y",
) -> dict:
    """A Plotly rect shape for a square box of center ``(cx, cy)`` and half-width ``h``."""
    line: dict = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    shape = {
        "type": "rect",
        "x0": cx - h,
        "y0": cy - h,
        "x1": cx + h,
        "y1": cy + h,
        "xref": xref,
        "yref": yref,
        "line": line,
    }
    if fillcolor is not None:
        shape["fillcolor"] = fillcolor
        shape["opacity"] = 1.0 if opacity is None else opacity
    return shape


def _box_shapes(tree, **kw) -> list[dict]:
    """A rect shape per leaf of ``tree``."""
    return [
        _rect(tree.center[L].real, tree.center[L].imag, tree.half[L], **kw)
        for L in tree.leaves
    ]


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


def cloud(n: int, seed: int, box: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Random positions in a square and random unit charges of both signs."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-box, box, (2, n)), rng.choice([-1.0, 1.0], n)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Max-norm error of ``a`` relative to ``b``."""
    return float(np.abs(a - b).max() / np.abs(b).max())


def net_force(q: np.ndarray, field: np.ndarray) -> float:
    """Net force relative to the sum of the force magnitudes."""
    return float(np.abs((q * field).sum(axis=1)).max() / np.abs(q * field).sum())


# -- 1, 2: truncation ---------------------------------------------------------


def truncation_study(n: int, orders: list[int], thetas: tuple[float, ...]):
    """Field error and net force for both truncations, every order and theta."""
    pos, q = cloud(n, seed=1)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=16)
    _, ref = direct_fields(pos, q)
    rows = []
    for theta in thetas:
        for p in orders:
            for symmetric in (False, True):
                f = SmoothFMM(tree, pos, q, order=p, theta=theta, symmetric=symmetric)
                _, e = f.fields()
                rows.append((theta, p, symmetric, relative(e, ref), net_force(q, e)))
                print(
                    f"  theta={theta} p={p:2d} {'triangle ' if symmetric else 'rectangle'}"
                    f" error {rows[-1][3]:.2e} net force {rows[-1][4]:.1e}"
                )
    return rows


def plot_truncation(rows, output: Path) -> None:
    """Figures 1 and 2."""
    thetas = sorted({r[0] for r in rows})
    theta_colors = dict(zip(thetas, (MPL_COLORS["C0"], MPL_COLORS["C3"])))
    for key, name, ylabel, title in (
        (
            3,
            "truncation_accuracy",
            "max field error relative to the direct sum",
            "Accuracy: the triangle needs ~4 more orders",
        ),
        (
            4,
            "truncation_momentum",
            "net force / sum of |forces|",
            "Momentum: pair antisymmetry of the M2L truncation",
        ),
    ):
        fig = go.Figure()
        for theta in thetas:
            color = theta_colors[theta]
            for symmetric, dash, symbol, label in (
                (False, "solid", "circle", "rectangle (standard)"),
                (True, "dash", "square", "triangle (symmetric)"),
            ):
                sel = [r for r in rows if r[0] == theta and r[2] == symmetric]
                fig.add_trace(
                    go.Scatter(
                        x=[r[1] for r in sel],
                        y=[max(r[key], 1e-18) for r in sel],
                        mode="lines+markers",
                        line={"color": color, "dash": dash},
                        marker={"symbol": symbol, "color": color},
                        name=f"{label}, θ = {theta}",
                    )
                )
        fig.update_layout(
            title=title,
            xaxis_title="expansion order p",
            yaxis_title=ylabel,
            yaxis_type="log",
            legend={"font": {"size": 10}},
        )
        save_figure(fig, output / name)


# -- 3: partition of unity -------------------------------------------------------


def plot_partition_of_unity(output: Path) -> None:
    """Figure 3."""
    pos, q = cloud(300, seed=7)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    band = 0.5 * tree.min_leaf_half
    # the weight of one leaf and the sum of all weights on a grid of probe points
    grid = np.linspace(-1, 1, 321)
    gx, gy = np.meshgrid(grid, grid)
    probes = np.stack((gx.ravel(), gy.ravel()))
    zp = probes[0] + 1j * probes[1]
    part, leaf, w, _, _ = tree.memberships(zp, band)
    total = np.bincount(part, w, minlength=len(zp)).reshape(gx.shape)
    # pick a leaf near the middle with a neighbour of a different size if possible
    leaves = tree.leaves
    middle = leaves[np.argmin(np.abs(tree.center[leaves] - (0.1 + 0.1j)))]
    w_leaf = np.zeros(len(zp))
    sel = leaf == middle
    w_leaf[part[sel]] = w[sel]
    w_leaf = w_leaf.reshape(gx.shape)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "weight w_L of one leaf (band = half its width / 2)",
            "sum of all leaf weights",
        ),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Heatmap(
            z=w_leaf,
            x=grid,
            y=grid,
            zmin=0,
            zmax=1.0,
            colorscale="Viridis",
            colorbar={"x": 0.43, "len": 0.9},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=total,
            x=grid,
            y=grid,
            zmin=0,
            zmax=1.001,
            colorscale="Viridis",
            colorbar={"x": 1.0, "len": 0.9},
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        shapes=_box_shapes(tree, color="white", width=0.6, xref="x", yref="y")
        + _box_shapes(tree, color="white", width=0.6, xref="x2", yref="y2")
    )
    for col in (1, 2):
        fig.add_trace(
            go.Scatter(
                x=pos[0],
                y=pos[1],
                mode="markers",
                marker={"size": 3, "color": "black", "opacity": 0.6},
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    c, h = tree.center[middle], tree.half[middle]
    fig.update_xaxes(
        range=[c.real - 4 * h, c.real + 4 * h],
        scaleanchor="y",
        showgrid=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[c.imag - 4 * h, c.imag + 4 * h], showgrid=False, row=1, col=1
    )
    fig.update_xaxes(scaleanchor="y2", showgrid=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=2)
    fig.add_annotation(
        text=f"max |Σw − 1| = {np.abs(total - 1).max():.1e}",
        x=-0.95,
        y=-0.95,
        xref="x2",
        yref="y2",
        showarrow=False,
        font={"color": "white"},
    )
    fig.update_layout(
        title="Smooth membership is a partition of unity on the adaptive leaves"
    )
    save_figure(fig, output / "partition_of_unity")


# -- 4: gradient check ------------------------------------------------------------


def plot_gradient_check(output: Path, n: int = 300) -> None:
    """Figure 4."""
    pos, q = cloud(n, seed=2)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    kw = {
        "order": 8,
        "theta": 0.6,
        "band": 0.4 * tree.min_leaf_half,
        "softening": 0.02,
        "symmetric": True,
    }
    f = SmoothFMM(tree, pos, q, **kw)
    _, e = f.fields()
    force = q * e
    counts = np.bincount(f.part, minlength=n)
    rng = np.random.default_rng(5)
    picks = rng.choice(
        np.flatnonzero(counts > 1), min(25, int((counts > 1).sum())), replace=False
    )
    analytic, numeric, members = [], [], []
    h = 1e-6
    for i in picks:
        for axis in range(2):
            pp, pm = pos.copy(), pos.copy()
            pp[axis, i] += h
            pm[axis, i] -= h
            fd = -(
                SmoothFMM(tree, pp, q, **kw).energy
                - SmoothFMM(tree, pm, q, **kw).energy
            ) / (2 * h)
            analytic.append(force[axis, i])
            numeric.append(fd)
            members.append(counts[i])
    analytic, numeric, members = map(np.array, (analytic, numeric, members))
    scale = np.abs(force).max()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Forces are the exact gradient of the energy",
            f"max mismatch {np.abs(numeric - analytic).max() / scale:.1e}",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[-scale, scale],
            y=[-scale, scale],
            mode="lines",
            line={"color": "black", "width": 0.8},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=analytic,
            y=numeric,
            mode="markers",
            marker={
                "size": 7,
                "color": members,
                "cmin": 1,
                "cmax": 4,
                "colorscale": "Viridis",
                "colorbar": {"title": "leaves the particle belongs to", "x": 0.43},
            },
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="analytic force component  2q Σ [w G + Φ ∇w]", row=1, col=1
    )
    fig.update_yaxes(title_text="−dU/dx by central differences", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            y=np.abs(numeric - analytic) / scale,
            mode="markers",
            marker={"size": 5},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=1e-8,
        line={"color": "black", "dash": "dash", "width": 0.8},
        annotation_text="finite-difference noise level",
        annotation_position="top left",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        type="log", range=[-13, -5], title_text="|F_fd − F| / max|F|", row=1, col=2
    )
    fig.update_xaxes(title_text="sample", row=1, col=2)
    save_figure(fig, output / "gradient_check")


# -- 5, 6: conservation over time -------------------------------------------------


def verlet(pos, vel, q, m, dt, steps, k_trap, fields: Callable, every: int = 5):
    """Velocity Verlet in a harmonic trap; energy error and net force histories."""
    x, v = pos.copy(), vel.copy()
    u, e = fields(x)
    e0 = u + 0.5 * k_trap * np.sum(x**2) + 0.5 * np.sum(m * np.sum(v**2, axis=0))
    acc = (q * e - k_trap * x) / m
    times, drift, net, history = [], [], [], []
    for s in range(steps):
        v += 0.5 * dt * acc
        x += dt * v
        u, e = fields(x)
        acc = (q * e - k_trap * x) / m
        v += 0.5 * dt * acc
        if s % every == 0:
            times.append((s + 1) * dt)
            total = (
                u + 0.5 * k_trap * np.sum(x**2) + 0.5 * np.sum(m * np.sum(v**2, axis=0))
            )
            drift.append(total / e0 - 1.0)
            net.append(net_force(q, e))
            history.append(x.copy())
    return np.array(times), np.array(drift), np.array(net), np.array(history)


def conservation_runs(n: int, steps: int, verbose: bool = True):
    """The five Verlet runs of figures 5 and 6."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(-0.8, 0.8, (2, n))
    q = rng.choice([-1.0, 1.0], n)
    vel = 0.2 * rng.normal(size=(2, n))
    m = np.ones(n)
    dt, k_trap, eps = 0.002, 100.0, 0.1
    order, theta, leaf_capacity = 4, 0.7, 8
    root = Quad(0, 0, 2.0)
    tree0 = BoxTree(pos, root, leaf_capacity=leaf_capacity)
    center = (m * pos).sum(axis=1) / m.sum()
    options = dict(order=order, theta=theta, softening=eps, symmetric=True)

    def direct(x):
        return direct_energy(x, q, eps), direct_fields(x, q, eps)[1]

    def fixed(band_frac):
        def fields(x):
            f = SmoothFMM(tree0, x, q, band=band_frac * tree0.min_leaf_half, **options)
            return f.energy, f.fields()[1]

        return fields

    def rebuilt(x):
        t = BoxTree(x, root, leaf_capacity=leaf_capacity)
        f = SmoothFMM(t, x, q, **options)
        return f.energy, f.fields()[1]

    def comoving(x):
        u, _, e = comoving_fields(
            tree0, x, q, m, center, band=0.5 * tree0.min_leaf_half, **options
        )
        return u, e

    cases = [
        ("direct sum (integrator alone)", direct, "k"),
        ("hard membership, boxes rebuilt every step", rebuilt, "C3"),
        ("hard membership, fixed boxes", fixed(0.0), "C1"),
        ("smooth membership, fixed boxes", fixed(0.5), "C0"),
        ("smooth membership, fixed co-moving boxes", comoving, "C2"),
    ]
    results = []
    for label, fields, color in cases:
        t0 = time.time()
        times, drift, net, history = verlet(pos, vel, q, m, dt, steps, k_trap, fields)
        if verbose:
            print(
                f"  {label:44s} max|dE/E| {np.abs(drift).max():.2e}  "
                f"final {drift[-1]:+.2e}  net force {net.max():.1e}  ({time.time() - t0:.0f}s)"
            )
        results.append((label, color, times, drift, net, history))
    return results, dict(n=n, dt=dt, order=order, theta=theta, eps=eps, tree=tree0, q=q)


def plot_conservation(results, info, output: Path) -> None:
    """Figures 5 and 6."""
    fig = go.Figure()
    for label, color, times, drift, _, _ in results:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=drift,
                mode="lines",
                line={"color": _c(color), "width": 1.2},
                name=label,
            )
        )
    fig.add_hline(y=0, line={"color": "black", "width": 0.5})
    fig.update_layout(
        title=(
            f"Energy: N = {info['n']}, p = {info['order']}, θ = {info['theta']}, "
            f"Δt = {info['dt']} (Verlet)"
        ),
        xaxis_title="time",
        yaxis_title="relative energy error  (E − E₀) / |E₀|",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "energy_drift")

    fig = go.Figure()
    for label, color, times, _, net, _ in results:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.maximum(net, 1e-18),
                mode="lines",
                line={"color": _c(color), "width": 1.2},
                name=label,
            )
        )
    fig.update_layout(
        title="Momentum: symmetric truncation, and co-moving boxes for the smooth terms",
        xaxis_title="time",
        yaxis_title="net force / sum of |forces|",
        yaxis_type="log",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "momentum_drift")


def plot_trajectories(results, info, output: Path) -> None:
    """Figure 7."""
    tree, q = info["tree"], info["q"]
    reference = results[0][5]
    conserving = results[-1]
    history = conserving[5]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{conserving[0]}<br><span style='font-size:11px'>"
            "red: +, blue: −, dots: start; grey: the fixed leaf boxes</span>",
            "Distance from the direct-sum trajectories<br>"
            "<span style='font-size:11px'>(grows exponentially for every variant: N-body chaos)</span>",
        ),
    )
    fig.update_layout(shapes=_box_shapes(tree, color="rgb(153,153,153)", width=0.3))
    for i in range(history.shape[2]):
        color = MPL_COLORS["C3"] if q[i] > 0 else MPL_COLORS["C0"]
        fig.add_trace(
            go.Scatter(
                x=history[:, 0, i],
                y=history[:, 1, i],
                mode="lines",
                line={"color": color, "width": 0.5},
                opacity=0.7,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=history[0, 0],
            y=history[0, 1],
            mode="markers",
            marker={"size": 3, "color": "black"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(range=[-1.3, 1.3], scaleanchor="y", showgrid=False, row=1, col=1)
    fig.update_yaxes(range=[-1.3, 1.3], showgrid=False, row=1, col=1)

    for label, color, times, _, _, hist in results[1:]:
        distance = np.sqrt(((hist - reference) ** 2).sum(axis=1)).max(axis=1)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.maximum(distance, 1e-17),
                mode="lines",
                line={"color": _c(color), "width": 1.2},
                name=label,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="time", row=1, col=2)
    fig.update_yaxes(
        title_text="max over particles of |x − x_direct|", type="log", row=1, col=2
    )
    fig.update_layout(legend={"font": {"size": 10}})
    save_figure(fig, output / "trajectories")


def plot_gentle_trajectories(output: Path, steps: int = 2000):
    """Figure 8."""
    n = 8
    rng = np.random.default_rng(11)
    angle = 2 * np.pi * np.arange(n) / n + rng.uniform(-0.15, 0.15, n)
    radius = 0.5 + rng.uniform(-0.08, 0.08, n)
    pos = radius * np.stack((np.cos(angle), np.sin(angle)))
    q = np.ones(n)  # like charges: no collapse, ordered orbits
    vel = 1.0 * np.stack((-np.sin(angle), np.cos(angle)))
    m = np.ones(n)
    dt, k_trap, eps = 0.002, 20.0, 0.1
    root = Quad(0, 0, 1.5)
    tree = BoxTree(pos, root, leaf_capacity=1)
    center = (m * pos).sum(axis=1) / m.sum()
    # a high order: even eight like charges are chaotic, and the truncation error of a
    # low order would be amplified into visibly different orbits within a few periods
    options = dict(order=16, theta=0.5, softening=eps, symmetric=True)

    def direct(x):
        return direct_energy(x, q, eps), direct_fields(x, q, eps)[1]

    def conserving(x):
        u, _, e = comoving_fields(
            tree, x, q, m, center, band=0.5 * tree.min_leaf_half, **options
        )
        return u, e

    times, drift_d, _, hist_d = verlet(
        pos, vel, q, m, dt, steps, k_trap, direct, every=2
    )
    _, drift_t, net_t, hist_t = verlet(
        pos, vel, q, m, dt, steps, k_trap, conserving, every=2
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "8 like charges in a trap: tree code (colour) over direct sum (dashed)<br>"
            "<span style='font-size:11px'>max trajectory difference "
            f"{np.sqrt(((hist_t - hist_d) ** 2).sum(axis=1)).max():.1e}</span>",
            f"energy; net force of the tree code ≤ {net_t.max():.0e}",
        ),
    )
    fig.update_layout(shapes=_box_shapes(tree, color="rgb(153,153,153)", width=0.3))
    for i in range(n):
        fig.add_trace(
            go.Scatter(
                x=hist_d[:, 0, i],
                y=hist_d[:, 1, i],
                mode="lines",
                line={"color": "black", "width": 1.6, "dash": "dash"},
                opacity=0.5,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist_t[:, 0, i],
                y=hist_t[:, 1, i],
                mode="lines",
                line={"color": MPL_COLORS[f"C{i % 10}"], "width": 0.8},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=pos[0],
            y=pos[1],
            mode="markers",
            marker={"size": 6, "color": "black"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(range=[-1.5, 1.5], scaleanchor="y", showgrid=False, row=1, col=1)
    fig.update_yaxes(range=[-1.5, 1.5], showgrid=False, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=times,
            y=drift_d,
            mode="lines",
            line={"color": "black", "width": 1.2, "dash": "dash"},
            name="direct sum",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=drift_t,
            mode="lines",
            line={"color": MPL_COLORS["C2"], "width": 1.0},
            name="smooth membership, fixed co-moving boxes",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line={"color": "black", "width": 0.5}, row=1, col=2)
    fig.update_xaxes(title_text="time", row=1, col=2)
    fig.update_yaxes(title_text="relative energy error", row=1, col=2)
    fig.update_layout(legend={"font": {"size": 10}})
    save_figure(fig, output / "trajectories_gentle")
    return times, hist_d, hist_t, tree, q


# -- 9: animations -------------------------------------------------------------------


def animate_cloud(results, info, output: Path, frames: int = 160) -> None:
    """``cloud``: the conserving run, particles over the fixed boxes, energy error."""
    label, color, times, drift, _, history = results[-1]
    step = max(1, len(times) // frames)
    idx = np.arange(0, len(times), step)
    q, tree = info["q"], info["tree"]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.47, 0.53],
        subplot_titles=(label, "energy error (faint: the other variants)"),
    )
    fig.update_layout(shapes=_box_shapes(tree, color="rgb(153,153,153)", width=0.3))
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 6, "color": MPL_COLORS["C3"]},
            name="+",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 6, "color": MPL_COLORS["C0"]},
            name="−",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(range=[-1.3, 1.3], scaleanchor="y", showgrid=False, row=1, col=1)
    fig.update_yaxes(range=[-1.3, 1.3], showgrid=False, row=1, col=1)

    for _, col_, t_, d_, _, _ in results:
        fig.add_trace(
            go.Scatter(
                x=t_,
                y=d_,
                mode="lines",
                line={"color": _c(col_), "width": 0.8},
                opacity=0.35,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line={"color": _c(color), "width": 1.6},
            name=label,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line={"color": "black", "width": 0.5}, row=1, col=2)
    lim = 1.2 * max(np.abs(r[3]).max() for r in results)
    fig.update_xaxes(title_text="time", range=[0, times[-1]], row=1, col=2)
    fig.update_yaxes(
        title_text="relative energy error", range=[-lim, lim], row=1, col=2
    )
    fig.update_layout(legend={"font": {"size": 10}})

    trace_pos_plus, trace_pos_minus = 0, 1
    trace_energy = len(results) + 2
    frame_list = []
    for k in idx:
        x = history[k]
        frame_list.append(
            go.Frame(
                name=f"{times[k]:.3f}",
                data=[
                    go.Scatter(x=x[0, q > 0], y=x[1, q > 0]),
                    go.Scatter(x=x[0, q < 0], y=x[1, q < 0]),
                    go.Scatter(x=times[: k + 1], y=drift[: k + 1]),
                ],
                traces=[trace_pos_plus, trace_pos_minus, trace_energy],
            )
        )
    fig.frames = frame_list
    _add_player(fig, frame_list, duration=40)
    save_figure(fig, output / "cloud", frame=-1)


def animate_orbits(
    times, hist_d, hist_t, tree, q, output: Path, frames: int = 200, trail: int = 60
) -> None:
    """``orbits``: the ring of like charges, tree code (colour) over the direct sum (black)."""
    n = hist_t.shape[2]
    step = max(1, len(times) // frames)
    idx = np.arange(0, len(times), step)

    fig = go.Figure()
    fig.update_layout(shapes=_box_shapes(tree, color="rgb(153,153,153)", width=0.3))
    for _ in range(n):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line={"color": "black", "width": 2.2},
                opacity=0.35,
                showlegend=False,
            )
        )
    for i in range(n):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line={"color": MPL_COLORS[f"C{i % 10}"], "width": 1.0},
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 7, "color": "black"},
            showlegend=False,
        )
    )
    fig.update_xaxes(range=[-1.2, 1.2], scaleanchor="y", showgrid=False)
    fig.update_yaxes(range=[-1.2, 1.2], showgrid=False)

    frame_list = []
    for k in idx:
        lo = max(0, k - trail)
        data = [
            go.Scatter(x=hist_d[lo : k + 1, 0, i], y=hist_d[lo : k + 1, 1, i])
            for i in range(n)
        ]
        data += [
            go.Scatter(x=hist_t[lo : k + 1, 0, i], y=hist_t[lo : k + 1, 1, i])
            for i in range(n)
        ]
        data.append(go.Scatter(x=hist_t[k, 0], y=hist_t[k, 1]))
        dist = np.sqrt(((hist_t[k] - hist_d[k]) ** 2).sum(axis=0)).max()
        frame_list.append(
            go.Frame(
                name=f"{times[k]:.2f}",
                data=data,
                traces=list(range(2 * n + 1)),
                layout=go.Layout(
                    title=(
                        "8 like charges: tree code (colour) over direct sum (black)<br>"
                        f"t = {times[k]:.2f}, max |x − x_direct| = {dist:.1e}"
                    )
                ),
            )
        )
    fig.frames = frame_list
    if frame_list:
        fig.update_layout(title=frame_list[0].layout.title.text)
    _add_player(fig, frame_list, duration=40)
    save_figure(fig, output / "orbits", frame=-1)


def animate_membership(output: Path, frames: int = 150) -> None:
    """``membership``: a tracer crossing leaf edges; its leaf weights sum to one."""
    pos, _ = cloud(300, seed=7)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    band = 0.5 * tree.min_leaf_half
    # a straight path through the middle of the tree that crosses several edges and a corner
    path = np.linspace(-0.55, 0.55, frames)
    track = path + 1j * (0.35 * path + 0.08)

    static_shapes = _box_shapes(tree, color="rgb(128,128,128)", width=0.3) + [
        _rect(
            tree.center[L].real,
            tree.center[L].imag,
            tree.half[L] + band,
            color=MPL_COLORS["C0"],
            width=0.3,
            dash="dot",
        )
        for L in tree.leaves
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            "tracer (red); leaves (grey), leaves grown by the band (dotted)",
            "membership weights (they sum to 1)",
        ),
    )
    fig.update_layout(shapes=static_shapes)
    fig.add_trace(
        go.Scatter(
            x=pos[0],
            y=pos[1],
            mode="markers",
            marker={"size": 3, "color": "rgb(179,179,179)"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=track.real,
            y=track.imag,
            mode="lines",
            line={"color": "rgb(204,204,204)", "width": 0.8},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 10, "color": MPL_COLORS["C3"]},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=[], y=[], marker_color=MPL_COLORS["C1"], showlegend=False),
        row=1,
        col=2,
    )
    fig.update_xaxes(range=[-0.75, 0.75], scaleanchor="y", showgrid=False, row=1, col=1)
    fig.update_yaxes(range=[-0.45, 0.6], showgrid=False, row=1, col=1)
    fig.update_yaxes(
        range=[0, 1.05], title_text="weight w_L of the tracer in leaf L", row=1, col=2
    )

    tracer_trace, bar_trace = 2, 3
    frame_list = []
    for frame_i, z in enumerate(track):
        part, leaf, w, _, _ = tree.memberships(np.array([z]), band)
        order = np.argsort(-w)
        leaf_sorted, w_sorted = leaf[order], w[order]
        shaded = [
            _rect(
                tree.center[L].real,
                tree.center[L].imag,
                tree.half[L],
                color="rgba(0,0,0,0)",
                width=0,
                fillcolor=MPL_COLORS["C1"],
                opacity=float(0.6 * weight),
            )
            for L, weight in zip(leaf_sorted, w_sorted)
        ]
        labels = [f"leaf {int(L)}" for L in leaf_sorted] + ["sum"]
        values = [*w_sorted.tolist(), float(w_sorted.sum())]
        colors = [MPL_COLORS["C1"]] * len(leaf_sorted) + ["rgb(102,102,102)"]
        frame_list.append(
            go.Frame(
                name=f"{frame_i}",
                data=[
                    go.Scatter(x=[z.real], y=[z.imag]),
                    go.Bar(x=labels, y=values, marker_color=colors),
                ],
                traces=[tracer_trace, bar_trace],
                layout=go.Layout(shapes=static_shapes + shaded),
            )
        )
    fig.frames = frame_list
    _add_player(fig, frame_list, duration=50, prefix="frame ")
    save_figure(fig, output / "membership", frame=-1)


def animate_tree(
    output: Path, n: int = 300, frames: int = 200, steps_per_frame: int = 4
) -> None:
    """``tree``: the quadtree rebuilt every frame while the particles move.

    The dynamics is the direct sum (the point is the tree, not the force
    approximation). Leaves that did not exist in the previous frame are drawn in
    green, leaves that vanished in red, so splits and merges are visible.
    """
    rng = np.random.default_rng(3)
    pos = rng.uniform(-0.8, 0.8, (2, n))
    q = rng.choice([-1.0, 1.0], n)
    vel = 0.4 * rng.normal(size=(2, n))
    m = np.ones(n)
    dt, k_trap, eps, capacity = 0.002, 60.0, 0.1, 8
    root = Quad(0, 0, 2.0)
    x, v = pos.copy(), vel.copy()
    _, e = direct_fields(x, q, eps)
    acc = (q * e - k_trap * x) / m

    def key(tree, L):
        return (
            round(tree.center[L].real, 9),
            round(tree.center[L].imag, 9),
            round(tree.half[L], 9),
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.53, 0.47],
        subplot_titles=(
            "",
            f"leaf capacity {capacity}: the box set changes constantly",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 5, "color": MPL_COLORS["C3"]},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 5, "color": MPL_COLORS["C0"]},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(range=[-1.6, 1.6], scaleanchor="y", showgrid=False, row=1, col=1)
    fig.update_yaxes(range=[-1.6, 1.6], showgrid=False, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line={"color": "black", "width": 1.2},
            name="leaves",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line={"color": MPL_COLORS["C2"], "width": 1.2},
            name="leaves created this frame",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line={"color": MPL_COLORS["C3"], "width": 1.2},
            name="leaves removed this frame",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text="time", range=[0, frames * steps_per_frame * dt], row=1, col=2
    )
    fig.update_yaxes(title_text="count", row=1, col=2)
    fig.update_layout(legend={"font": {"size": 9}})

    dots_plus, dots_minus, line_leaves, line_new, line_gone = 0, 1, 2, 3, 4
    previous: set = set()
    times, counts, created, removed = [], [], [], []
    frame_list = []
    for frame_i in range(frames):
        for _ in range(steps_per_frame):
            v += 0.5 * dt * acc
            x += dt * v
            _, e = direct_fields(x, q, eps)
            acc = (q * e - k_trap * x) / m
            v += 0.5 * dt * acc
        tree = BoxTree(x, root, leaf_capacity=capacity)
        current = {key(tree, L): L for L in tree.leaves}
        new, gone = set(current) - previous, previous - set(current)
        shapes = []
        for k_, L in current.items():
            color, width = (
                (MPL_COLORS["C2"], 2.0) if k_ in new else ("rgb(140,140,140)", 0.6)
            )
            shapes.append(
                _rect(
                    tree.center[L].real,
                    tree.center[L].imag,
                    tree.half[L],
                    color=color,
                    width=width,
                )
            )
        for cx, cy, h in gone:
            shapes.append(
                _rect(cx, cy, h, color=MPL_COLORS["C3"], width=1.4, dash="dash")
            )
        previous = set(current)
        t = (frame_i + 1) * steps_per_frame * dt
        times.append(t)
        counts.append(len(current))
        created.append(len(new) if frame_i else 0)
        removed.append(len(gone))
        title = (
            f"t = {t:.2f}: {len(current)} leaves, depth {int(tree.depth.max())}; "
            "green: new this frame, red dashed: just removed"
        )
        frame_list.append(
            go.Frame(
                name=f"{t:.2f}",
                data=[
                    go.Scatter(x=x[0, q > 0], y=x[1, q > 0]),
                    go.Scatter(x=x[0, q < 0], y=x[1, q < 0]),
                    go.Scatter(x=times.copy(), y=counts.copy()),
                    go.Scatter(x=times.copy(), y=created.copy()),
                    go.Scatter(x=times.copy(), y=removed.copy()),
                ],
                traces=[dots_plus, dots_minus, line_leaves, line_new, line_gone],
                layout=go.Layout(shapes=shapes, title=title),
            )
        )
    fig.frames = frame_list
    if frame_list:
        fig.update_layout(
            shapes=frame_list[0].layout.shapes, title=frame_list[0].layout.title.text
        )
    fig.update_yaxes(range=[0, max(counts) * 1.1], row=1, col=2)
    _add_player(fig, frame_list, duration=50, prefix="frame ")
    save_figure(fig, output / "tree", frame=-1)


# -- main -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run everything and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    quick = args.quick

    print("1-2. truncation study")
    rows = truncation_study(
        n=600 if quick else 1500,
        orders=[4, 8, 12, 16] if quick else [4, 8, 12, 16, 20, 24],
        thetas=(0.5, 0.7),
    )
    plot_truncation(rows, output)

    print("3. partition of unity")
    plot_partition_of_unity(output)

    print("4. gradient check")
    plot_gradient_check(output, n=150 if quick else 300)

    print("5-6. conservation over time")
    results, info = conservation_runs(
        n=120 if quick else 200, steps=200 if quick else 800
    )
    plot_conservation(results, info, output)
    plot_trajectories(results, info, output)

    print("8. gentle trajectories")
    gentle = plot_gentle_trajectories(output, steps=600 if quick else 2000)

    print("9. animations")
    animate_cloud(results, info, output, frames=60 if quick else 160)
    animate_orbits(*gentle, output, frames=80 if quick else 200)
    animate_membership(output, frames=60 if quick else 150)
    animate_tree(output, n=120 if quick else 300, frames=60 if quick else 200)
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
