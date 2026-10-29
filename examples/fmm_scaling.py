"""Fast multipole method.

The fast multipole method computes the fields of line charges in O(N).

Runs the experiments behind ``FlatQuadtree.fmm``, the library's O(N) fast multipole
method for ``kernel="log"``, and saves one figure per claim:

1. ``order_accuracy``: FMM field error against the expansion order, for a few theta;
   multipole expansions converge almost exponentially in the order.
2. ``theta_accuracy``: FMM field error against theta, for a few fixed orders.
3. ``interaction_counts``: the number of M2L and P2P node pairs the dual-tree
   traversal produces, against N; the interaction list grows linearly, not
   quadratically.
4. ``scaling``: wall-clock time against N for the direct sum, the Barnes-Hut tree
   walk and the FMM, on a log-log plot; the FMM overtakes the tree walk at large N.
5. ``interaction_list``: the quadtree with one leaf box highlighted, its P2P
   (near-field, exact) partner leaves in one color and its M2L (far-field,
   multipole-translated) partner boxes in another.

Usage::

    python examples/fmm_scaling.py [--output examples/output] [--quick]

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

from tree_d_3d import FlatQuadtree, Quad, direct_field

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
    fillcolor: str | None = None,
    opacity: float | None = None,
) -> dict:
    """A Plotly rect shape for a square box of center ``(cx, cy)`` and half-width ``h``."""
    shape: dict = {
        "type": "rect",
        "x0": cx - h,
        "y0": cy - h,
        "x1": cx + h,
        "y1": cy + h,
        "line": {"color": color, "width": width},
    }
    if fillcolor is not None:
        shape["fillcolor"] = fillcolor
        shape["opacity"] = 1.0 if opacity is None else opacity
    return shape


def cloud(n: int, seed: int, box: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Random positions in a square and random unit charges of both signs."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-box, box, (2, n)), rng.choice([-1.0, 1.0], n)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Max-norm error of ``a`` relative to ``b``."""
    return float(np.abs(a - b).max() / np.abs(b).max())


# -- 1: order accuracy -------------------------------------------------------------------


def plot_order_accuracy(output: Path, n: int = 2000) -> None:
    """Figure 1."""
    pos, q = cloud(n, seed=1)
    others = np.arange(n)
    reference = direct_field(pos, pos, q, exclude=others, kernel="log")[1]
    orders = [2, 4, 6, 8, 10, 12, 16, 20]
    thetas = (0.3, 0.5, 0.7)

    fig = go.Figure()
    for theta, color in zip(
        thetas, (MPL_COLORS["C0"], MPL_COLORS["C1"], MPL_COLORS["C3"])
    ):
        errors = []
        for order in orders:
            tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=order)
            field = tree.fmm(theta=theta)[1]
            errors.append(max(relative(field, reference), 1e-18))
            print(f"  theta={theta} order={order:2d} error {errors[-1]:.2e}")
        fig.add_trace(
            go.Scatter(
                x=orders,
                y=errors,
                mode="lines+markers",
                line={"color": color},
                name=f"θ = {theta}",
            )
        )
    fig.update_xaxes(title_text="expansion order")
    fig.update_yaxes(
        type="log", title_text="max field error relative to the direct sum"
    )
    fig.update_layout(
        title="Multipole expansions converge almost exponentially in the order",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "order_accuracy")


# -- 2: theta accuracy --------------------------------------------------------------------


def plot_theta_accuracy(output: Path, n: int = 2000) -> None:
    """Figure 2."""
    pos, q = cloud(n, seed=1)
    others = np.arange(n)
    reference = direct_field(pos, pos, q, exclude=others, kernel="log")[1]
    thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    orders = (6, 10, 16)

    fig = go.Figure()
    for order, color in zip(
        orders, (MPL_COLORS["C0"], MPL_COLORS["C1"], MPL_COLORS["C3"])
    ):
        tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=order)
        errors = []
        for theta in thetas:
            field = tree.fmm(theta=theta)[1]
            errors.append(max(relative(field, reference), 1e-18))
        print(f"  order={order}: " + ", ".join(f"{e:.1e}" for e in errors))
        fig.add_trace(
            go.Scatter(
                x=thetas,
                y=errors,
                mode="lines+markers",
                line={"color": color},
                name=f"order = {order}",
            )
        )
    fig.update_xaxes(title_text="opening angle θ")
    fig.update_yaxes(
        type="log", title_text="max field error relative to the direct sum"
    )
    fig.update_layout(
        title="A larger θ accepts more distant pairs as well separated, at some cost in accuracy",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "theta_accuracy")


# -- 3: interaction counts ----------------------------------------------------------------


def plot_interaction_counts(output: Path, quick: bool) -> None:
    """Figure 3."""
    ns = [300, 1000, 3000, 8000] if quick else [300, 1000, 3000, 8000, 20000, 50000]
    rng = np.random.default_rng(2)
    m2l_counts, p2p_counts = [], []
    for n in ns:
        pos = rng.uniform(-1, 1, (2, n))
        q = rng.choice([-1.0, 1.0], n)
        tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=8)
        tree.fmm(theta=0.5)
        n_m2l, n_p2p = tree.fmm_pairs
        m2l_counts.append(n_m2l)
        p2p_counts.append(n_p2p)
        print(f"  N={n:6d} M2L pairs {n_m2l:7d}  P2P pairs {n_p2p:7d}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=m2l_counts,
            mode="lines+markers",
            line={"color": MPL_COLORS["C0"]},
            name="M2L pairs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=p2p_counts,
            mode="lines+markers",
            line={"color": MPL_COLORS["C1"]},
            name="P2P pairs",
        )
    )
    anchor_n, anchor_count = ns[0], p2p_counts[0]
    n_ref = np.array([float(ns[0]), float(ns[-1])])
    fig.add_trace(
        go.Scatter(
            x=n_ref,
            y=anchor_count * (n_ref / anchor_n),
            mode="lines",
            line={"color": "rgb(153,153,153)", "dash": "dash"},
            name="∝ N",
        )
    )
    fig.update_xaxes(type="log", title_text="N")
    fig.update_yaxes(type="log", title_text="number of node pairs")
    fig.update_layout(
        title="The dual-tree traversal's interaction list grows linearly with N",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "interaction_counts")


# -- 4: scaling ---------------------------------------------------------------------------


def plot_scaling(output: Path, quick: bool) -> None:
    """Figure 4."""
    direct_ns = [300, 1000, 3000, 8000] if quick else [500, 2000, 8000, 20000, 50000]
    tree_ns = direct_ns + ([20000, 50000] if quick else [150000, 400000])
    reps = 2
    rng = np.random.default_rng(5)

    def make(n: int):
        pos = rng.uniform(-1, 1, (2, n))
        q = rng.choice([-1.0, 1.0], n)
        return pos, q, np.arange(n)

    # warm up the compiled kernels once, outside the timed loop
    warm_pos, warm_q, warm_others = make(64)
    direct_field(warm_pos, warm_pos, warm_q, exclude=warm_others, kernel="log")
    warm_tree = FlatQuadtree(warm_pos, warm_q, Quad(0, 0, 1), kernel="log", order=10)
    warm_tree.evaluate(warm_pos, exclude=warm_others, theta=0.5)
    warm_tree.fmm(theta=0.5)

    direct_times = []
    for n in direct_ns:
        pos, q, others = make(n)
        t0 = time.time()
        for _ in range(reps):
            direct_field(pos, pos, q, exclude=others, kernel="log")
        direct_times.append((time.time() - t0) / reps)
        print(f"  N={n:6d} direct sum {direct_times[-1] * 1e3:9.1f} ms")

    tree_times, fmm_times = [], []
    for n in tree_ns:
        pos, q, others = make(n)
        tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=10)
        t0 = time.time()
        for _ in range(reps):
            tree.evaluate(pos, exclude=others, theta=0.5)
        tree_times.append((time.time() - t0) / reps)
        t0 = time.time()
        for _ in range(reps):
            tree.fmm(theta=0.5)
        fmm_times.append((time.time() - t0) / reps)
        print(
            f"  N={n:6d} tree walk   {tree_times[-1] * 1e3:9.1f} ms   "
            f"fmm {fmm_times[-1] * 1e3:9.1f} ms"
        )

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
    fig.add_trace(
        go.Scatter(
            x=tree_ns,
            y=fmm_times,
            mode="lines+markers",
            line={"color": MPL_COLORS["C3"]},
            name="fast multipole method",
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
    anchor_n3, anchor_t3 = tree_ns[0], fmm_times[0]
    fig.add_trace(
        go.Scatter(
            x=n_ref,
            y=anchor_t3 * (n_ref / anchor_n3),
            mode="lines",
            line={"color": "rgb(153,153,153)", "dash": "longdash"},
            name="∝ N",
        )
    )
    fig.update_xaxes(type="log", title_text="N")
    fig.update_yaxes(type="log", title_text="wall-clock time [s]")
    fig.update_layout(
        title="At large N the O(N) fast multipole method overtakes the O(N log N) tree walk",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "scaling")


# -- 5: interaction list ------------------------------------------------------------------


def plot_interaction_list(output: Path, n: int = 600) -> None:
    """Figure 5."""
    pos, q = cloud(n, seed=7)
    theta = 0.5
    tree = FlatQuadtree(pos, q, Quad(0, 0, 1), kernel="log", order=8, leaf_capacity=12)
    m2l, p2p, sizes = tree._traverse(theta)
    m2l, p2p = m2l[: sizes[0]], p2p[: sizes[1]]
    leaves = [node for node in tree.collect_nodes() if node.is_external()]
    target = min(leaves, key=lambda node: abs(node.quad.xmid) + abs(node.quad.ymid))

    p2p_partners: set[int] = set()
    for a, b in p2p:
        a, b = int(a), int(b)
        if a == target.index and b != target.index:
            p2p_partners.add(b)
        elif b == target.index and a != target.index:
            p2p_partners.add(a)
    m2l_partners: set[int] = set()
    for a, b in m2l:
        a, b = int(a), int(b)
        if a == target.index:
            m2l_partners.add(b)
        elif b == target.index:
            m2l_partners.add(a)
    print(
        f"  target leaf {target.index}: {len(p2p_partners)} P2P partners, "
        f"{len(m2l_partners)} M2L partners, {len(leaves)} leaves total"
    )

    shapes = [
        _rect(node.quad.xmid, node.quad.ymid, node.quad.half, width=0.3)
        for node in leaves
        if node.index not in m2l_partners and node.index not in p2p_partners
    ]
    shapes += [
        _rect(
            tree.node_center[i, 0],
            tree.node_center[i, 1],
            tree.node_half[i],
            color=MPL_COLORS["C0"],
            width=0,
            fillcolor=MPL_COLORS["C0"],
            opacity=0.4,
        )
        for i in m2l_partners
    ]
    shapes += [
        _rect(
            tree.node_center[i, 0],
            tree.node_center[i, 1],
            tree.node_half[i],
            color=MPL_COLORS["C1"],
            width=0,
            fillcolor=MPL_COLORS["C1"],
            opacity=0.55,
        )
        for i in p2p_partners
    ]
    shapes.append(
        _rect(
            target.quad.xmid,
            target.quad.ymid,
            target.quad.half,
            color="black",
            width=1.8,
            fillcolor="black",
            opacity=0.8,
        )
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
    # legend-only traces, the shapes carry the actual colors
    for color, name in (
        ("black", "target leaf"),
        (MPL_COLORS["C1"], "P2P: exact near field"),
        (MPL_COLORS["C0"], "M2L: one multipole-to-local translation"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": "square"},
                name=name,
            )
        )
    fig.update_xaxes(range=[-1.05, 1.05], scaleanchor="y", showgrid=False)
    fig.update_yaxes(range=[-1.05, 1.05], showgrid=False)
    fig.update_layout(
        title=(
            "Who the target leaf (black) talks to: near leaves exactly (orange), farther "
            "well-separated boxes with one translation each (blue), regardless of their size"
        ),
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "interaction_list")


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

    print("1. order accuracy")
    plot_order_accuracy(output, n=1000 if quick else 2000)

    print("2. theta accuracy")
    plot_theta_accuracy(output, n=1000 if quick else 2000)

    print("3. interaction counts")
    plot_interaction_counts(output, quick)

    print("4. scaling")
    plot_scaling(output, quick)

    print("5. interaction list")
    plot_interaction_list(output, n=400 if quick else 600)
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
