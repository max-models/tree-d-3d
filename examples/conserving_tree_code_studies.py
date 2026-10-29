"""Conserving tree studies.

The parameter studies behind the conserving tree code, as figures.

Companion of ``conserving_tree_code.py``; each figure answers one question:

1. ``kernels``: the pair potential and field of the bare, Plummer and compact
   (smooth cloud) softening; why the far field of the compact one is exact.
2. ``weights``: the smoothstep weights of two neighbouring leaves across their
   edge and the derivative that enters the force.
3. ``band_study``: field error, memberships per particle and energy error against
   the band width.
4. ``cost_matched``: the cheapest (order, theta) reaching a given field error for
   the rectangle and the triangle truncation, with the M2L and near-field work counted.
5. ``long_run``: energy error over a long Verlet run: no secular drift with smooth
   membership on fixed co-moving boxes.
6. ``error_map``: where the field error sits, for hard and smooth membership (it is
   not concentrated at leaf edges, and blending does not change its distribution).
7. ``correction_term``: the size of the ``Phi grad w`` force term relative to the
   ordinary ``w G`` term, which is what restores the exact gradient.

Usage::

    python examples/conserving_tree_code_studies.py [--output examples/output] [--quick]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from plotly.subplots import make_subplots

from tree_d_3d.quad import Quad
from tree_d_3d.smooth import (
    BoxTree,
    SmoothFMM,
    _disk_kernel,
    _smoothstep,
    comoving_fields,
    direct_energy,
    direct_fields,
)

# The Matplotlib "Cn" cycle, kept so the colours stay the ones each figure was designed around.
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
    xref: str = "x",
    yref: str = "y",
) -> dict:
    """A Plotly rect shape for a square box of center ``(cx, cy)`` and half-width ``h``."""
    line: dict = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    return {
        "type": "rect",
        "x0": cx - h,
        "y0": cy - h,
        "x1": cx + h,
        "y1": cy + h,
        "xref": xref,
        "yref": yref,
        "line": line,
    }


def _box_shapes(tree, **kw) -> list[dict]:
    """A rect shape per leaf of ``tree``."""
    return [
        _rect(tree.center[L].real, tree.center[L].imag, tree.half[L], **kw)
        for L in tree.leaves
    ]


def cloud(n: int, seed: int, box: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Random positions in a square and random unit charges of both signs."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-box, box, (2, n)), rng.choice([-1.0, 1.0], n)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Max-norm error of ``a`` relative to ``b``."""
    return float(np.abs(a - b).max() / np.abs(b).max())


# -- 1: kernels ----------------------------------------------------------------------


def plot_kernels(output: Path) -> None:
    """Figure 1."""
    eps = 1.0
    r = np.linspace(0.02, 4.0, 400)
    dz = r + 0.0j
    bare_phi, bare_g = _disk_kernel(dz, 0.0)
    compact_phi, compact_g = _disk_kernel(dz, eps**2)
    plummer_phi = 0.5 * np.log(r**2 + eps**2)
    plummer_g = r / (r**2 + eps**2)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "potential",
            "field: the compact cloud equals the bare kernel beyond ε",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=bare_phi,
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="bare: ln r",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=plummer_phi,
            mode="lines",
            line={"color": MPL_COLORS["C1"]},
            name="Plummer: ½ ln(r² + ε²)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=compact_phi,
            mode="lines",
            line={"color": MPL_COLORS["C2"]},
            name="compact cloud (density ∝ (1 − r²/ε²)² inside ε)",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=eps, line={"color": "rgb(153,153,153)", "width": 0.8}, row=1, col=1)
    fig.update_xaxes(title_text="r / ε", row=1, col=1)
    fig.update_yaxes(title_text="pair potential Φ", range=[-3, 1.6], row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=r,
            y=bare_g.real,
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="bare: 1/r",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=plummer_g,
            mode="lines",
            line={"color": MPL_COLORS["C1"]},
            name="Plummer",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=compact_g.real,
            mode="lines",
            line={"color": MPL_COLORS["C2"]},
            name="compact cloud",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=np.abs(plummer_g - bare_g.real),
            mode="lines",
            line={"color": MPL_COLORS["C1"], "dash": "dot"},
            name="|Plummer − bare| (never zero)",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=eps, line={"color": "rgb(153,153,153)", "width": 0.8}, row=1, col=2)
    fig.update_xaxes(title_text="r / ε", row=1, col=2)
    fig.update_yaxes(title_text="|∇Φ|", range=[0, 2.2], row=1, col=2)
    fig.update_layout(
        title="Softening kernels: only a compact kernel keeps the multipole far field exact",
        legend={"font": {"size": 9}},
    )
    save_figure(fig, output / "kernels")


# -- 2: weights ----------------------------------------------------------------------


def plot_weights(output: Path) -> None:
    """Figure 2."""
    band = 0.25
    x = np.linspace(-1, 1, 801)
    # two leaves sharing the edge x = 0, each of half-width 1
    s_right, ds_right = _smoothstep(x / band)  # weight of the leaf on the right
    s_left, ds_left = _smoothstep(-x / band)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "quintic smoothstep: C², s(t) + s(−t) = 1",
            "the derivatives are opposite: the Φ∇w forces see Φ_L − Φ_L′",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=s_left,
            mode="lines",
            line={"color": MPL_COLORS["C0"]},
            name="w_L (left leaf)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=s_right,
            mode="lines",
            line={"color": MPL_COLORS["C1"]},
            name="w_L' (right leaf)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=s_left + s_right,
            mode="lines",
            line={"color": "black", "dash": "dash", "width": 0.8},
            name="sum",
        ),
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=-band,
        x1=band,
        fillcolor="rgb(230,230,230)",
        opacity=0.6,
        line_width=0,
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="x (edge at 0)", row=1, col=1)
    fig.update_yaxes(title_text="weight", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=-ds_left / band,
            mode="lines",
            line={"color": MPL_COLORS["C0"]},
            name="dw_L/dx",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ds_right / band,
            mode="lines",
            line={"color": MPL_COLORS["C1"]},
            name="dw_L'/dx",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line={"color": "black", "width": 0.5}, row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="dw/dx", row=1, col=2)
    fig.update_layout(legend={"font": {"size": 9}})
    save_figure(fig, output / "weights")


# -- 3: band study -------------------------------------------------------------------


def verlet_energy(pos, vel, q, m, dt, steps, k_trap, fields) -> np.ndarray:
    """Relative energy error history of velocity Verlet in a harmonic trap."""
    x, v = pos.copy(), vel.copy()
    u, e = fields(x)
    e0 = u + 0.5 * k_trap * np.sum(x**2) + 0.5 * np.sum(m * np.sum(v**2, axis=0))
    acc = (q * e - k_trap * x) / m
    out = []
    for _ in range(steps):
        v += 0.5 * dt * acc
        x += dt * v
        u, e = fields(x)
        acc = (q * e - k_trap * x) / m
        v += 0.5 * dt * acc
        out.append(
            (u + 0.5 * k_trap * np.sum(x**2) + 0.5 * np.sum(m * np.sum(v**2, axis=0)))
            / e0
            - 1
        )
    return np.array(out)


def plot_band_study(output: Path, quick: bool) -> None:
    """Figure 3."""
    fractions = [0.0, 0.05, 0.1, 0.25, 0.5]
    # accuracy and memberships on a static cloud
    pos, q = cloud(600 if quick else 1200, seed=4)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    _, ref = direct_fields(pos, q)
    errors, members = [], []
    for frac in fractions:
        f = SmoothFMM(tree, pos, q, order=12, theta=0.5, band=frac * tree.min_leaf_half)
        _, e = f.fields()
        errors.append(relative(e, ref))
        members.append(len(f.part) / len(q))
    # energy error on a Verlet run
    n, steps = (100, 150) if quick else (150, 300)
    rng = np.random.default_rng(3)
    xpos = rng.uniform(-0.8, 0.8, (2, n))
    xq = rng.choice([-1.0, 1.0], n)
    vel = 0.2 * rng.normal(size=(2, n))
    m = np.ones(n)
    xtree = BoxTree(xpos, Quad(0, 0, 2.0), leaf_capacity=8)
    center = (m * xpos).sum(axis=1) / m.sum()
    options = dict(order=2, theta=0.8, softening=0.1, symmetric=True)
    drifts = []
    for frac in fractions:

        def fields(x, frac=frac):
            u, _, e = comoving_fields(
                xtree, x, xq, m, center, band=frac * xtree.min_leaf_half, **options
            )
            return u, e

        drifts.append(
            np.abs(verlet_energy(xpos, vel, xq, m, 0.002, steps, 100.0, fields)).max()
        )
        print(
            f"  band {frac:.2f}: error {errors[len(drifts) - 1]:.2e}, "
            f"memberships {members[len(drifts) - 1]:.2f}, max|dE/E| {drifts[-1]:.2e}"
        )
    reference = np.abs(
        verlet_energy(
            xpos,
            vel,
            xq,
            m,
            0.002,
            steps,
            100.0,
            lambda x: (direct_energy(x, xq, 0.1), direct_fields(x, xq, 0.1)[1]),
        )
    ).max()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "accuracy: non-monotonic; a wide band averages\ntwo leaves' expansions and can even help",
            "cost of P2M, L2P and the near field",
            "energy: any band > 0 removes the drift",
        ),
    )
    fig.add_trace(
        go.Scatter(x=fractions, y=errors, mode="lines+markers", showlegend=False),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        type="log", title_text="max field error (p = 12, θ = 0.5)", row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=fractions, y=members, mode="lines+markers", showlegend=False),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="memberships per particle", row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=fractions,
            y=drifts,
            mode="lines+markers",
            name="tree code (p = 2, θ = 0.8, co-moving)",
        ),
        row=1,
        col=3,
    )
    fig.add_hline(
        y=reference,
        line={"color": "black", "dash": "dash", "width": 0.8},
        annotation_text="direct sum (integrator alone)",
        row=1,
        col=3,
    )
    fig.update_yaxes(type="log", title_text="max |ΔE/E| over the run", row=1, col=3)
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="band δ / smallest leaf half-width", row=1, col=col)
    fig.update_layout(legend={"font": {"size": 9}})
    save_figure(fig, output / "band_study")


# -- 4: cost at matched accuracy -----------------------------------------------------


def plot_cost_matched(output: Path, quick: bool) -> None:
    """Figure 4."""
    pos, q = cloud(800 if quick else 1500, seed=1)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=16)
    _, ref = direct_fields(pos, q)
    thetas = [0.4, 0.5, 0.6, 0.7, 0.8]
    orders = [4, 8, 12, 16, 20, 24] if quick else [4, 8, 12, 16, 20, 24, 28]
    rows = []
    for theta in thetas:
        for p in orders:
            for symmetric in (False, True):
                t0 = time.time()
                f = SmoothFMM(tree, pos, q, order=p, theta=theta, symmetric=symmetric)
                _, e = f.fields()
                n_m2l, n_p2p = len(f.m2l_pairs), len(f.p2p_pairs)
                # work model: M2L is p^2 (rectangle) or p^2/2 (triangle) per pair and
                # direction; the near field is capacity^2 per leaf pair
                m2l_work = 2 * n_m2l * (p * p / 2 if symmetric else p * p)
                p2p_work = n_p2p * 16 * 16
                rows.append(
                    (
                        theta,
                        p,
                        symmetric,
                        relative(e, ref),
                        m2l_work + p2p_work,
                        time.time() - t0,
                    )
                )
    tolerances = np.logspace(-3, -11, 17)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "work at matched accuracy, best over θ ∈ [0.4, 0.8] and p ≤ 28",
            "every (θ, p) combination",
        ),
    )
    for symmetric, color, label in (
        (False, MPL_COLORS["C0"], "rectangle (standard)"),
        (True, MPL_COLORS["C3"], "triangle (symmetric)"),
    ):
        best = []
        for tol in tolerances:
            ok = [r for r in rows if r[2] == symmetric and r[3] <= tol]
            best.append(min(r[4] for r in ok) if ok else np.nan)
        fig.add_trace(
            go.Scatter(
                x=tolerances,
                y=best,
                mode="lines+markers",
                line={"color": color},
                name=label,
            ),
            row=1,
            col=1,
        )
        sel = [r for r in rows if r[2] == symmetric]
        fig.add_trace(
            go.Scatter(
                x=[r[3] for r in sel],
                y=[r[4] for r in sel],
                mode="markers",
                marker={"size": 7, "color": color, "opacity": 0.6},
                name=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(
        type="log", autorange="reversed", title_text="target field error", row=1, col=1
    )
    fig.update_yaxes(
        type="log",
        title_text="cheapest work (M2L + near field, model units)",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        type="log", autorange="reversed", title_text="field error", row=1, col=2
    )
    fig.update_yaxes(type="log", title_text="work", row=1, col=2)
    fig.update_layout(
        title="Exact momentum (triangle) costs about the same as the standard truncation at equal accuracy",
        legend={"font": {"size": 9}},
    )
    save_figure(fig, output / "cost_matched")


# -- 5: long run ---------------------------------------------------------------------


def plot_long_run(output: Path, quick: bool) -> None:
    """Figure 5."""
    n, steps = (100, 800) if quick else (150, 4000)
    rng = np.random.default_rng(3)
    pos = rng.uniform(-0.8, 0.8, (2, n))
    q = rng.choice([-1.0, 1.0], n)
    vel = 0.2 * rng.normal(size=(2, n))
    m = np.ones(n)
    dt, k_trap, eps = 0.002, 100.0, 0.1
    tree = BoxTree(pos, Quad(0, 0, 2.0), leaf_capacity=8)
    center = (m * pos).sum(axis=1) / m.sum()
    options = dict(order=3, theta=0.8, softening=eps, symmetric=True)

    def direct(x):
        return direct_energy(x, q, eps), direct_fields(x, q, eps)[1]

    def tree_fields(frac):
        def fields(x):
            u, _, e = comoving_fields(
                tree, x, q, m, center, band=frac * tree.min_leaf_half, **options
            )
            return u, e

        return fields

    times = dt * np.arange(1, steps + 1)
    fig = go.Figure()
    for label, fields, color in (
        ("direct sum (integrator alone)", direct, "black"),
        ("hard membership, fixed boxes", tree_fields(0.0), MPL_COLORS["C1"]),
        (
            "smooth membership, fixed co-moving boxes",
            tree_fields(0.5),
            MPL_COLORS["C2"],
        ),
    ):
        t0 = time.time()
        drift = verlet_energy(pos, vel, q, m, dt, steps, k_trap, fields)
        print(
            f"  {label:42s} max|dE/E| {np.abs(drift).max():.2e} final {drift[-1]:+.2e} ({time.time() - t0:.0f}s)"
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=drift,
                mode="lines",
                line={"color": color, "width": 0.9},
                name=label,
            )
        )
    fig.add_hline(y=0, line={"color": "black", "width": 0.5})
    fig.update_layout(
        title=(
            f"{steps} steps, N = {n}, p = {options['order']}, θ = {options['theta']}: "
            "no secular drift with smooth membership"
        ),
        xaxis_title="time",
        yaxis_title="relative energy error",
        legend={"font": {"size": 9}},
    )
    save_figure(fig, output / "long_run")


# -- 6: error map --------------------------------------------------------------------


def plot_error_map(output: Path, quick: bool) -> None:
    """Figure 6."""
    pos, q = cloud(600 if quick else 1000, seed=8)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    _, ref = direct_fields(pos, q)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(" ", " "))
    vmin, vmax = 1e-9, 1e-4
    ticks = np.linspace(np.log10(vmin), np.log10(vmax), 6)
    shapes: list[dict] = []
    for col, frac, title in (
        (1, 0.0, "hard membership"),
        (2, 0.5, "smooth membership, band = 0.5 h_min"),
    ):
        f = SmoothFMM(tree, pos, q, order=8, theta=0.6, band=frac * tree.min_leaf_half)
        _, e = f.fields()
        err = np.sqrt(((e - ref) ** 2).sum(axis=0)) / np.abs(ref).max()
        xref, yref = (f"x{col}", f"y{col}") if col > 1 else ("x", "y")
        shapes += _box_shapes(
            tree, color="rgb(153,153,153)", width=0.3, xref=xref, yref=yref
        )
        log_err = np.log10(np.clip(err, vmin, vmax))
        fig.add_trace(
            go.Scatter(
                x=pos[0],
                y=pos[1],
                mode="markers",
                marker={
                    "size": 6,
                    "color": log_err,
                    "colorscale": "Magma",
                    "cmin": ticks[0],
                    "cmax": ticks[-1],
                    "showscale": col == 2,
                    "colorbar": (
                        {
                            "title": "field error / max |E| (p = 8, θ = 0.6)",
                            "tickvals": ticks,
                            "ticktext": [f"{10**t:.0e}" for t in ticks],
                        }
                        if col == 2
                        else None
                    ),
                },
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.layout.annotations[col - 1].update(
            text=f"{title}: max error {err.max():.1e}"
        )
        fig.update_xaxes(scaleanchor=yref, showgrid=False, row=1, col=col)
        fig.update_yaxes(showgrid=False, row=1, col=col)
    fig.update_layout(
        shapes=shapes,
        title=(
            "Where the error is: smooth membership leaves the error distribution unchanged "
            "(no concentration at leaf edges)"
        ),
    )
    save_figure(fig, output / "error_map")


# -- 7: correction term --------------------------------------------------------------


def plot_correction_term(output: Path, quick: bool) -> None:
    """Figure 7."""
    pos, q = cloud(600 if quick else 1200, seed=5)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    fig = go.Figure()
    bins = np.logspace(-8, 0, 40)
    for frac, color in (
        (0.1, MPL_COLORS["C0"]),
        (0.25, MPL_COLORS["C1"]),
        (0.5, MPL_COLORS["C2"]),
    ):
        f = SmoothFMM(
            tree,
            pos,
            q,
            order=8,
            theta=0.6,
            band=frac * tree.min_leaf_half,
            softening=0.02,
        )
        n = len(q)
        wg = np.zeros(n, dtype=complex)
        pdw = np.zeros(n, dtype=complex)
        np.add.at(wg, f.part, f.w * f.g_m)
        np.add.at(pdw, f.part, f.phi_m * f.dw)
        counts = np.bincount(f.part, minlength=n)
        blended = counts > 1
        ratio = np.abs(pdw[blended]) / np.abs(wg).max()
        counts_hist, edges = np.histogram(np.clip(ratio, bins[0], bins[-1]), bins=bins)
        fig.add_trace(
            go.Scatter(
                x=edges[:-1],
                y=counts_hist,
                mode="lines",
                line={"color": color, "width": 1.5, "shape": "hv"},
                name=f"band = {frac} h_min  ({blended.sum()} blended particles)",
            )
        )
    fig.update_xaxes(
        type="log", title_text="|Φ ∇w term| / max |w G term|   (per blended particle)"
    )
    fig.update_yaxes(title_text="particles")
    fig.update_layout(
        title="The exact-gradient correction is small (truncation error / band width) but not negligible",
        legend={"font": {"size": 9}},
    )
    save_figure(fig, output / "correction_term")


# -- main ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run every study and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    print("1. kernels")
    plot_kernels(output)
    print("2. weights")
    plot_weights(output)
    print("3. band study")
    plot_band_study(output, args.quick)
    print("4. cost at matched accuracy")
    plot_cost_matched(output, args.quick)
    print("5. long run")
    plot_long_run(output, args.quick)
    print("6. error map")
    plot_error_map(output, args.quick)
    print("7. correction term")
    plot_correction_term(output, args.quick)
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
