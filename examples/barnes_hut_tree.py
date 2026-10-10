"""The Barnes-Hut tree: how a quadtree approximates distant charges.

Runs the experiments behind ``FlatQuadtree.evaluate``, the library's Barnes-Hut tree
walk for point and line charges, and saves one figure per claim:

1. ``tree_boxes``: a cloud of random charges and the leaf boxes of the quadtree built
   on them; boxes shrink where particles are dense and stay large where they are sparse.
2. ``opening_criterion``: for one target particle, which boxes the tree accepts as a
   single multipole (far enough), which it visits leaf by leaf (too close), and which
   it merely opens on the way down.
3. ``theta_tradeoff``: field error against the direct sum, and wall-clock time, both
   against the opening angle theta: smaller theta is more accurate and slower.
4. ``scaling``: wall-clock time against N for the direct sum and the tree walk, on a
   log-log plot, next to reference slopes for their N^2 and N log N complexities.
5. ``cluster_merge``: two colliding star clusters evolved with the pure-Python
   Barnes-Hut gravity solver, animated, with the quadtree redrawn every frame.

Usage::

    python examples/barnes_hut_tree.py [--output examples/output] [--quick]

``--quick`` uses fewer particles and a narrower N range (well under a minute instead
of a few minutes). Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from plotly.subplots import make_subplots

from tree_d_3d import FlatQuadtree, Quad, direct_field
from tree_d_3d.gravity import BarnesHutSimulation, Body

MPL_COLORS = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
}


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
        "line": line,
    }
    if fillcolor is not None:
        shape["fillcolor"] = fillcolor
        shape["opacity"] = 1.0 if opacity is None else opacity
    return shape


def _leaf_shapes(nodes, **kw) -> list[dict]:
    """A rect shape per leaf of a list of ``FlatNode`` or ``gravity.Node``."""
    return [
        _rect(node.quad.xmid, node.quad.ymid, node.quad.half, **kw)
        for node in nodes
        if node.is_external()
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


# -- 1: tree boxes ---------------------------------------------------------------------


def plot_tree_boxes(output: Path, n: int = 3000) -> None:
    """Figure 1."""
    pos, q = cloud(n, seed=0)
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=8)
    fig = go.Figure()
    fig.update_layout(
        shapes=_leaf_shapes(tree.collect_nodes(), color="rgb(160,160,160)", width=0.4)
    )
    fig.add_trace(
        go.Scatter(
            x=pos[0, q > 0],
            y=pos[1, q > 0],
            mode="markers",
            marker={"size": 3, "color": MPL_COLORS["C3"]},
            name="+",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pos[0, q < 0],
            y=pos[1, q < 0],
            mode="markers",
            marker={"size": 3, "color": MPL_COLORS["C0"]},
            name="−",
        )
    )
    fig.update_xaxes(range=[-1.05, 1.05], scaleanchor="y", showgrid=False)
    fig.update_yaxes(range=[-1.05, 1.05], showgrid=False)
    fig.update_layout(
        title=f"The quadtree on N = {n} random charges: leaf capacity 8",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "tree_boxes")


# -- 2: opening criterion ---------------------------------------------------------------


def _classify(tree: FlatQuadtree, target: np.ndarray, theta: float):
    """Walk the tree as ``evaluate`` would for one target: accepted / leaf / internal nodes."""
    accepted, leaves, internal = [], [], []

    def walk(node) -> None:
        quad = node.quad
        if node.is_external():
            leaves.append(node)
            return
        distance = float(np.hypot(target[0] - quad.xmid, target[1] - quad.ymid))
        if not quad.contains(*target) and 2 * quad.half < theta * distance:
            accepted.append(node)
            return
        internal.append(node)
        for child in node.children:
            if child is not None and child.count > 0:
                walk(child)

    walk(tree.root)
    return accepted, leaves, internal


def plot_opening_criterion(output: Path, n: int = 800, theta: float = 0.5) -> None:
    """Figure 2."""
    pos, q = cloud(n, seed=2)
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1), leaf_capacity=8)
    target = pos[:, 0]
    accepted, leaves, internal = _classify(tree, target, theta)
    shapes = (
        [
            _rect(node.quad.xmid, node.quad.ymid, node.quad.half, width=0.3)
            for node in internal
        ]
        + [
            _rect(
                node.quad.xmid,
                node.quad.ymid,
                node.quad.half,
                color=MPL_COLORS["C1"],
                width=0.9,
            )
            for node in leaves
        ]
        + [
            _rect(
                node.quad.xmid,
                node.quad.ymid,
                node.quad.half,
                color=MPL_COLORS["C2"],
                width=0,
                fillcolor=MPL_COLORS["C2"],
                opacity=0.35,
            )
            for node in accepted
        ]
    )
    fig = go.Figure()
    fig.update_layout(shapes=shapes)
    fig.add_trace(
        go.Scatter(
            x=pos[0],
            y=pos[1],
            mode="markers",
            marker={"size": 3, "color": "rgb(150,150,150)"},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[target[0]],
            y=[target[1]],
            mode="markers",
            marker={"size": 14, "symbol": "star", "color": "black"},
            name="target",
        )
    )
    fig.update_xaxes(range=[-1.05, 1.05], scaleanchor="y", showgrid=False)
    fig.update_yaxes(range=[-1.05, 1.05], showgrid=False)
    fig.update_layout(
        title=(
            f"Opening test for one target, θ = {theta}: {len(accepted)} boxes accepted as "
            f"one multipole (green), {len(leaves)} leaves visited particle by particle "
            f"(orange); plain boxes were opened on the way down"
        ),
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "opening_criterion")


# -- 3: theta tradeoff -------------------------------------------------------------------


def plot_theta_tradeoff(output: Path, n: int = 6000) -> None:
    """Figure 3."""
    pos, q = cloud(n, seed=3)
    others = np.arange(n)
    reference = direct_field(pos, pos, q, softening=1e-3, exclude=others)[1]
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1))
    thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    errors, times_ms = [], []
    for theta in thetas:
        t0 = time.time()
        field = tree.evaluate(pos, exclude=others, theta=theta, softening=1e-3)[1]
        times_ms.append((time.time() - t0) * 1e3)
        errors.append(float(np.abs(field - reference).max() / np.abs(reference).max()))
        print(f"  theta={theta:.1f} error {errors[-1]:.2e} time {times_ms[-1]:.1f} ms")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "accuracy: smaller θ opens more boxes",
            "cost: smaller θ visits more node pairs",
        ),
    )
    fig.add_trace(
        go.Scatter(x=thetas, y=errors, mode="lines+markers", showlegend=False),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        type="log",
        title_text="max field error relative to the direct sum",
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=thetas, y=times_ms, mode="lines+markers", showlegend=False),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="wall-clock time [ms]", row=1, col=2)
    for col in (1, 2):
        fig.update_xaxes(title_text="opening angle θ", row=1, col=col)
    fig.update_layout(title=f"Accuracy/speed tradeoff of the opening angle (N = {n})")
    save_figure(fig, output / "theta_tradeoff")


# -- 4: scaling ---------------------------------------------------------------------------


def plot_scaling(output: Path, quick: bool) -> None:
    """Figure 4."""
    direct_ns = [300, 1000, 3000, 8000] if quick else [500, 2000, 8000, 20000, 50000]
    tree_ns = direct_ns + ([20000, 50000] if quick else [150000, 400000])
    reps = 2
    rng = np.random.default_rng(4)

    def make(n: int):
        pos = rng.uniform(-1, 1, (2, n))
        q = rng.choice([-1.0, 1.0], n)
        return pos, q, np.arange(n)

    # warm up the compiled kernels once, outside the timed loop
    warm_pos, warm_q, warm_others = make(64)
    direct_field(warm_pos, warm_pos, warm_q, softening=1e-3, exclude=warm_others)
    FlatQuadtree(warm_pos, warm_q, Quad(0, 0, 1)).evaluate(
        warm_pos, exclude=warm_others, theta=0.5, softening=1e-3
    )

    direct_times = []
    for n in direct_ns:
        pos, q, others = make(n)
        t0 = time.time()
        for _ in range(reps):
            direct_field(pos, pos, q, softening=1e-3, exclude=others)
        direct_times.append((time.time() - t0) / reps)
        print(f"  N={n:6d} direct sum {direct_times[-1] * 1e3:9.1f} ms")

    tree_times = []
    for n in tree_ns:
        pos, q, others = make(n)
        tree = FlatQuadtree(pos, q, Quad(0, 0, 1))
        t0 = time.time()
        for _ in range(reps):
            tree.evaluate(pos, exclude=others, theta=0.5, softening=1e-3)
        tree_times.append((time.time() - t0) / reps)
        print(f"  N={n:6d} tree walk   {tree_times[-1] * 1e3:9.1f} ms")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=direct_ns,
            y=direct_times,
            mode="lines+markers",
            line={"color": "black"},
            name="direct sum",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tree_ns,
            y=tree_times,
            mode="lines+markers",
            line={"color": MPL_COLORS["C0"]},
            name="Barnes-Hut tree walk",
        )
    )
    n_ref = np.array([float(direct_ns[0]), float(tree_ns[-1])])
    anchor_n, anchor_t = direct_ns[-1], direct_times[-1]
    fig.add_trace(
        go.Scatter(
            x=n_ref,
            y=anchor_t * (n_ref / anchor_n) ** 2,
            mode="lines",
            line={"color": "rgb(153,153,153)", "dash": "dash"},
            name="∝ N²",
        )
    )
    anchor_n2, anchor_t2 = tree_ns[0], tree_times[0]
    fig.add_trace(
        go.Scatter(
            x=n_ref,
            y=anchor_t2 * (n_ref * np.log(n_ref)) / (anchor_n2 * np.log(anchor_n2)),
            mode="lines",
            line={"color": "rgb(153,153,153)", "dash": "dot"},
            name="∝ N log N",
        )
    )
    fig.update_xaxes(type="log", title_text="N")
    fig.update_yaxes(type="log", title_text="wall-clock time [s]")
    fig.update_layout(
        title="The tree walk scales as N log N, the direct sum as N²",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "scaling")


# -- 5: cluster merge animation -----------------------------------------------------------


def _two_clusters(n_each: int, seed: int):
    """Two Gaussian blobs of bodies, offset and moving toward each other."""
    rng = np.random.default_rng(seed)
    offset, speed, spread, vspread = 0.6, 0.6, 0.12, 0.03
    bodies: list[Body] = []
    labels = []
    for sign in (-1.0, 1.0):
        pos = rng.normal(0.0, spread, (n_each, 2)) + np.array([sign * offset, 0.0])
        vel = rng.normal(0.0, vspread, (n_each, 2)) + np.array([-sign * speed, 0.0])
        for (x, y), (vx, vy) in zip(pos, vel):
            bodies.append(Body(x, y, vx, vy, mass=1.0))
            labels.append(sign)
    return bodies, np.array(labels)


def animate_cluster_merge(
    output: Path, n_each: int = 75, frames: int = 50, steps_per_frame: int = 2
) -> None:
    """``cluster_merge``: two star clusters colliding, animated with the current tree."""
    bodies, labels = _two_clusters(n_each, seed=6)
    sim = BarnesHutSimulation(
        bodies, Quad(0, 0, 2.0), theta=0.6, G=2e-3, softening=0.05, dt=0.03
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 5, "color": MPL_COLORS["C3"]},
            name="cluster A",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 5, "color": MPL_COLORS["C0"]},
            name="cluster B",
        )
    )
    fig.update_xaxes(range=[-1.4, 1.4], scaleanchor="y", showgrid=False)
    fig.update_yaxes(range=[-1.4, 1.4], showgrid=False)
    fig.update_layout(legend={"font": {"size": 10}})

    frame_list = []
    for frame_i in range(frames):
        for _ in range(steps_per_frame):
            tree = sim.step()
        pos = np.array([body.pos for body in bodies])
        n_leaves = sum(1 for node in tree.collect_nodes() if node.is_external())
        t = (frame_i + 1) * steps_per_frame * sim.dt
        title = f"t = {t:.2f}: {n_leaves} leaf boxes"
        frame_list.append(
            go.Frame(
                name=f"{frame_i}",
                data=[
                    go.Scatter(x=pos[labels < 0, 0], y=pos[labels < 0, 1]),
                    go.Scatter(x=pos[labels > 0, 0], y=pos[labels > 0, 1]),
                ],
                traces=[0, 1],
                layout=go.Layout(
                    shapes=_leaf_shapes(
                        tree.collect_nodes(), color="rgb(160,160,160)", width=0.4
                    ),
                    title=title,
                ),
            )
        )
    fig.frames = frame_list
    if frame_list:
        fig.update_layout(
            shapes=frame_list[0].layout.shapes, title=frame_list[0].layout.title.text
        )
    _add_player(fig, frame_list, duration=60, prefix="frame ")
    save_figure(fig, output / "cluster_merge", frame=-1)


# -- main -----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run everything and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    quick = args.quick

    print("1. tree boxes")
    plot_tree_boxes(output, n=1200 if quick else 3000)

    print("2. opening criterion")
    plot_opening_criterion(output, n=400 if quick else 800)

    print("3. theta tradeoff")
    plot_theta_tradeoff(output, n=2000 if quick else 6000)

    print("4. scaling")
    plot_scaling(output, quick)

    print("5. cluster merge animation")
    animate_cluster_merge(
        output,
        n_each=75 if quick else 150,
        frames=50 if quick else 90,
        steps_per_frame=2 if quick else 3,
    )
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
