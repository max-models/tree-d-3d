"""Rational far fields.

Rational far fields compare equivalent charges with free poles against the multipole series.

The multipole (Taylor) far field of a cell converges like ``rho**p`` with ``rho`` the
cell's half-diagonal over the target distance, which is why a fast multipole method
needs ``rho`` well below one and many accepted cell pairs. A rational approximant
(AAA, Nakatsukasa-Sete-Trefethen) of the same far field, read off in partial fractions,
is a set of ``m`` *equivalent complex charges at optimised positions*; its convergence
rate is set by the distance of the poles rather than by the convergence radius of a
series, so it can be accurate where the series is not. This example measures what that
buys, at equal cost per target (``p`` series terms against ``m = p/2`` charges, two
parameters each).

1. ``cell_accuracy``: field error of a 32-charge cell against the number of
   parameters, for the Taylor series and the AAA equivalent charges, at separation
   ratios 0.5, 0.7 and 0.9; the rational far field wins by orders of magnitude where
   the series barely converges.
2. ``cluster_size``: the gain (Taylor error over rational error) against the number of
   sources in the cell; it decays, because the far field of ``n`` charges has ``n``
   branch points and ``m`` free poles cannot resolve them all.
3. ``fit_validity``: error against the evaluation radius for a rational fit made on one
   circle; the series keeps improving with distance, the rational fit plateaus at its
   fit accuracy, so a fit must be made at the closest accepted separation.
4. ``hierarchical``: error of a cell's representation against its level in a quadtree,
   for the Taylor series (exact upward translation), a flat AAA fit from all the cell's
   particles, and a hierarchical AAA fit that compresses the children's equivalent
   charges; the hierarchy is what keeps the small-cluster gains at the top of the tree.
5. ``pair_counts``: the number of accepted (far-field) and near-field cell pairs of the
   dual-tree traversal against theta, for a fixed cloud; what a larger theta would buy.

Usage::

    python examples/rational_far_field.py [--output examples/output] [--quick]

``--quick`` uses smaller clouds and fewer orders (about 20 s instead of a few minutes).
Needs Plotly (``pip install 'tree-d-3d[examples]'``). The scripts behind the tables in the
research notes are ``benchmarks/rational_far_field*.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.linalg as sla
from plasma_plots import save_figure

from tree_d_3d.quad import Quad
from tree_d_3d.smooth import BoxTree, SmoothFMM

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}
R_UNIT = np.sqrt(2.0)  # half-diagonal of the unit square [-1, 1]^2


# -- the rational tools --------------------------------------------------------------------


def aaa(F: np.ndarray, Z: np.ndarray, m: int, tol: float = 1e-13):
    """AAA rational approximation of the samples ``F`` at ``Z`` with up to ``m`` poles.

    Returns the support points, their values and the barycentric weights.
    """
    M = len(Z)
    J = np.arange(M)
    z: list[complex] = []
    f: list[complex] = []
    R = np.full(M, F.mean())
    w = np.ones(1)
    for _ in range(m + 1):
        j = int(np.argmax(np.abs(F - R)))
        z.append(Z[j])
        f.append(F[j])
        J = J[J != j]
        zz, ff = np.array(z), np.array(f)
        C = 1.0 / (Z[J][:, None] - zz[None, :])
        A = (F[J][:, None] - ff[None, :]) * C
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        w = Vh[-1].conj()
        R = F.copy()
        R[J] = (C @ (w * ff)) / (C @ w)
        if np.abs(F - R).max() <= tol * np.abs(F).max():
            break
    return np.array(z), np.array(f), w


def aaa_poles_residues(zz, ff, w):
    """Poles and residues of the barycentric rational function."""
    m = len(zz)
    E = np.zeros((m + 1, m + 1), complex)
    E[0, 1:] = w
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(zz)
    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0
    ev = sla.eigvals(E, B)
    poles = ev[np.isfinite(ev) & (np.abs(ev) < 1e12)]
    N = ((w * ff)[None, :] / (poles[:, None] - zz[None, :])).sum(axis=1)
    Dp = -(w[None, :] / (poles[:, None] - zz[None, :]) ** 2).sum(axis=1)
    return poles, N / Dp


def equivalent_charges(zs, q, centre, d, m, samples=512):
    """``m`` equivalent complex charges of the sources ``(zs, q)``.

    Their field matches the sources' on the circle of radius ``d`` about ``centre``:
    they are the partial fractions of the AAA fit of ``h(zeta) = sum q/(1 - z zeta)``.
    """
    phi = np.linspace(0, 2 * np.pi, samples, endpoint=False) + np.pi / samples
    zeta = np.exp(-1j * phi) / d  # 1 / (d e^{i phi})
    rel = zs - centre
    h = (q[None, :] / (1 - rel[None, :] * zeta[:, None])).sum(axis=1)
    zz, ff, w = aaa(h, zeta, m)
    poles, res = aaa_poles_residues(zz, ff, w)
    rho = centre + 1 / poles
    c = -res / poles
    # the monopole is fixed exactly by a correction charge at the centre
    return np.r_[rho, centre], np.r_[c, q.sum() - c.sum()]


def field_of_charges(rho, c, Z):
    """``W'(Z) = sum c_k / (Z - rho_k)``."""
    return (c[None, :] / (Z[:, None] - rho[None, :])).sum(axis=1)


def taylor_field(zs, q, centre, p, Z):
    """``W'`` from the ``p``-term multipole series about ``centre``."""
    rel = zs - centre
    a = np.array([np.sum(q * rel**k) for k in range(p + 1)])
    u = Z - centre
    return a[0] / u + sum(a[k] / u ** (k + 1) for k in range(1, p + 1))


def exact_field(zs, q, Z):
    """``W'`` of the point charges, exactly."""
    return (q[None, :] / (Z[:, None] - zs[None, :])).sum(axis=1)


def circle(centre, d, n=1024, offset=0.0):
    """``n`` points on the circle of radius ``d`` about ``centre``."""
    return centre + d * np.exp(
        1j * (np.linspace(0, 2 * np.pi, n, endpoint=False) + offset)
    )


def relative(a, b):
    """Max-norm error of ``a`` relative to ``b``."""
    return float(np.abs(a - b).max() / np.abs(b).max())


def cloud(n, seed, box=1.0):
    """Random positions in a square and random unit charges of both signs."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-box, box, n) + 1j * rng.uniform(-box, box, n), rng.choice(
        [-1.0, 1.0], n
    )


# -- 1: accuracy per parameter -------------------------------------------------------------


def plot_cell_accuracy(output: Path, quick: bool) -> None:
    """Figure 1."""
    orders = [4, 8, 12, 16, 24] if quick else [4, 8, 12, 16, 20, 24, 32]
    seeds = [1, 2] if quick else [1, 2, 3]
    fig = go.Figure()
    for rho, color in zip((0.5, 0.7, 0.9), (COLORS["C0"], COLORS["C1"], COLORS["C3"])):
        d = R_UNIT / rho
        err_t, err_r = [], []
        for p in orders:
            m = p // 2
            et, er = [], []
            for seed in seeds:
                zs, q = cloud(32, seed)
                Z = circle(0.0, d)
                ref = exact_field(zs, q, Z)
                et.append(relative(taylor_field(zs, q, 0.0, p, Z), ref))
                rho_k, c_k = equivalent_charges(zs, q, 0.0, d, m)
                er.append(relative(field_of_charges(rho_k, c_k, Z), ref))
            err_t.append(np.mean(et))
            err_r.append(np.mean(er))
            print(
                f"  rho={rho} p={p:2d} m={m:2d}: Taylor {err_t[-1]:.1e}  rational {err_r[-1]:.1e}"
            )
        fig.add_trace(
            go.Scatter(
                x=orders,
                y=err_t,
                mode="lines+markers",
                line={"color": color},
                name=f"Taylor series, ρ = {rho}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=orders,
                y=err_r,
                mode="lines+markers",
                line={"color": color, "dash": "dash"},
                marker={"symbol": "square"},
                name=f"m = p/2 equivalent charges, ρ = {rho}",
            )
        )
    fig.update_xaxes(title_text="complex parameters (p series terms, or 2 per charge)")
    fig.update_yaxes(type="log", title_text="max field error on the circle, relative")
    fig.update_layout(
        title="A 32-charge cell: rational far field against the multipole series at equal cost",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "cell_accuracy")


# -- 2: cluster size -----------------------------------------------------------------------


def plot_cluster_size(output: Path, quick: bool) -> None:
    """Figure 2."""
    sizes = [8, 32, 128, 512] if quick else [8, 32, 128, 512, 2048]
    p, m = 24, 12
    fig = go.Figure()
    for rho, color in zip((0.5, 0.7, 0.9), (COLORS["C0"], COLORS["C1"], COLORS["C3"])):
        d = R_UNIT / rho
        gains = []
        for n in sizes:
            zs, q = cloud(n, 11)
            Z = circle(0.0, d)
            ref = exact_field(zs, q, Z)
            et = relative(taylor_field(zs, q, 0.0, p, Z), ref)
            rho_k, c_k = equivalent_charges(zs, q, 0.0, d, m)
            er = relative(field_of_charges(rho_k, c_k, Z), ref)
            gains.append(max(et / max(er, 1e-17), 1e-3))
            print(
                f"  rho={rho} n={n:5d}: Taylor {et:.1e} rational {er:.1e} gain {gains[-1]:.1f}"
            )
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=gains,
                mode="lines+markers",
                line={"color": color},
                name=f"ρ = {rho}",
            )
        )
    fig.add_hline(y=1.0, line={"color": "black", "width": 0.8, "dash": "dot"})
    fig.update_xaxes(type="log", title_text="sources in the cell")
    fig.update_yaxes(
        type="log", title_text="gain: Taylor error / rational error (p = 24, m = 12)"
    )
    fig.update_layout(
        title="The rational gain decays with the cell population (m free poles, n branch points)"
    )
    save_figure(fig, output / "cluster_size")


# -- 3: validity away from the fitting circle ---------------------------------------------


def plot_fit_validity(output: Path) -> None:
    """Figure 3."""
    factors = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    p, m = 24, 12
    zs, q = cloud(32, 3)
    fig = go.Figure()
    for rho, color in zip((0.5, 0.9), (COLORS["C0"], COLORS["C3"])):
        d = R_UNIT / rho
        rho_k, c_k = equivalent_charges(zs, q, 0.0, d, m)
        et, er = [], []
        for f in factors:
            Z = circle(0.0, f * d)
            ref = exact_field(zs, q, Z)
            et.append(max(relative(taylor_field(zs, q, 0.0, p, Z), ref), 1e-17))
            er.append(max(relative(field_of_charges(rho_k, c_k, Z), ref), 1e-17))
        fig.add_trace(
            go.Scatter(
                x=factors,
                y=et,
                mode="lines+markers",
                line={"color": color},
                name=f"Taylor, fit ρ = {rho}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=factors,
                y=er,
                mode="lines+markers",
                line={"color": color, "dash": "dash"},
                marker={"symbol": "square"},
                name=f"rational, fit ρ = {rho}",
            )
        )
    fig.update_xaxes(title_text="evaluation radius / fitting radius")
    fig.update_yaxes(type="log", title_text="max field error, relative")
    fig.update_layout(
        title="A rational fit holds its accuracy outward but does not improve; the series does",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "fit_validity")


# -- 4: hierarchical compression --------------------------------------------------------------


def plot_hierarchical(output: Path, quick: bool) -> None:
    """Figure 4.

    A uniform quadtree of ``levels`` levels over a random cloud with about ``per_leaf``
    particles per leaf. For every cell at every level, three representations of its far
    field are compared on the circle of radius ``d = R_cell (2/theta - 1)`` about the cell
    centre (the closest a particle of an accepted partner cell can be under the mutual
    MAC): the Taylor series of order ``p`` (whose moments are summed exactly over all
    particles), a flat AAA fit of ``m`` charges from all the particles, and the
    hierarchical AAA fit of ``m`` charges from the four children's charges (leaves use
    their particles when they hold at most ``m``). The figure shows the max error over
    the cells of a level.
    """
    levels = 4 if quick else 5
    per_leaf = 8
    m = 12
    p = 2 * m
    n = per_leaf * 4**levels
    zs, q = cloud(n, 5)
    thetas = (0.5, 0.9)
    fig = go.Figure()
    for theta, color in zip(thetas, (COLORS["C0"], COLORS["C3"])):
        # cells per level: level L has 4^L cells of half-width 1 / 2^L
        reps: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        err_t = np.zeros(levels + 1)
        err_flat = np.zeros(levels + 1)
        err_h = np.zeros(levels + 1)
        for level in range(levels, -1, -1):
            h = 1.0 / 2**level
            r_cell = np.sqrt(2.0) * h
            d = r_cell * (2.0 / theta - 1.0)
            for ix in range(2**level):
                for iy in range(2**level):
                    centre = complex(-1 + (2 * ix + 1) * h, -1 + (2 * iy + 1) * h)
                    inside = (np.abs(zs.real - centre.real) <= h) & (
                        np.abs(zs.imag - centre.imag) <= h
                    )
                    if not inside.any():
                        reps[level, ix, iy] = (np.zeros(0, complex), np.zeros(0))
                        continue
                    zc, qc = zs[inside], q[inside]
                    Z = circle(centre, d, n=256)
                    ref = exact_field(zc, qc, Z)
                    err_t[level] = max(
                        err_t[level], relative(taylor_field(zc, qc, centre, p, Z), ref)
                    )
                    # flat
                    if len(zc) <= m:
                        rho_flat, c_flat = zc, qc
                    else:
                        rho_flat, c_flat = equivalent_charges(
                            zc, qc, centre, d, m, samples=256
                        )
                    err_flat[level] = max(
                        err_flat[level],
                        relative(field_of_charges(rho_flat, c_flat, Z), ref),
                    )
                    # hierarchical
                    if level == levels:
                        rho_h, c_h = rho_flat, c_flat
                    else:
                        kids = [
                            reps[level + 1, 2 * ix + a, 2 * iy + b]
                            for a in range(2)
                            for b in range(2)
                        ]
                        rho_c = np.concatenate([k[0] for k in kids])
                        c_c = np.concatenate([k[1] for k in kids])
                        if len(rho_c) <= m:
                            rho_h, c_h = rho_c, c_c
                        else:
                            rho_h, c_h = equivalent_charges(
                                rho_c, c_c, centre, d, m, samples=256
                            )
                    reps[level, ix, iy] = (rho_h, c_h)
                    err_h[level] = max(
                        err_h[level], relative(field_of_charges(rho_h, c_h, Z), ref)
                    )
            print(
                f"  theta={theta} level {level} ({4**level} cells, ~{n // 4**level} particles each): Taylor {err_t[level]:.1e}  flat {err_flat[level]:.1e}  hierarchical {err_h[level]:.1e}"
            )
        x = list(range(levels, -1, -1))
        pop = [n // 4**lv for lv in x]
        labels = [f"{lv} ({pp} per cell)" for lv, pp in zip(x, pop)]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[max(err_t[lv], 1e-17) for lv in x],
                mode="lines+markers",
                line={"color": color},
                name=f"Taylor p = {p}, θ = {theta}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[max(err_flat[lv], 1e-17) for lv in x],
                mode="lines+markers",
                line={"color": color, "dash": "dot"},
                marker={"symbol": "diamond"},
                name=f"flat AAA m = {m}, θ = {theta}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[max(err_h[lv], 1e-17) for lv in x],
                mode="lines+markers",
                line={"color": color, "dash": "dash"},
                marker={"symbol": "square"},
                name=f"hierarchical AAA m = {m}, θ = {theta}",
            )
        )
    fig.update_xaxes(title_text="tree level (leaves first; particles per cell)")
    fig.update_yaxes(
        type="log",
        title_text="max over cells of the far-field error at the closest accepted separation",
    )
    fig.update_layout(
        title=f"Up the tree: Taylor (exact M2M) vs flat vs hierarchical rational fits, N = {n}",
        legend={"font": {"size": 9}},
    )
    save_figure(fig, output / "hierarchical")


# -- 5: pair counts against theta ------------------------------------------------------------


def plot_pair_counts(output: Path, quick: bool) -> None:
    """Figure 5."""
    n = 2000 if quick else 6000
    rng = np.random.default_rng(2)
    pos = rng.uniform(-0.9, 0.9, (2, n))
    q = rng.choice([-1.0, 1.0], n)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=16)
    thetas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    m2l, p2p = [], []
    for theta in thetas:
        f = SmoothFMM(tree, pos, q, order=2, theta=theta)
        m2l.append(len(f.m2l_pairs))
        p2p.append(len(f.p2p_pairs))
        print(f"  theta={theta}: {m2l[-1]} far-field pairs, {p2p[-1]} near-field pairs")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thetas,
            y=m2l,
            mode="lines+markers",
            line={"color": COLORS["C0"]},
            name="far-field (M2L) cell pairs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thetas,
            y=p2p,
            mode="lines+markers",
            line={"color": COLORS["C1"]},
            name="near-field (P2P) leaf pairs",
        )
    )
    fig.update_xaxes(title_text="θ")
    fig.update_yaxes(type="log", title_text="pairs")
    fig.update_layout(
        title=f"What a larger θ buys: cell pairs of the dual-tree traversal, N = {n}, leaf capacity 16"
    )
    save_figure(fig, output / "pair_counts")


# -- main -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run every study and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    print("1. cell accuracy")
    plot_cell_accuracy(output, args.quick)
    print("2. cluster size")
    plot_cluster_size(output, args.quick)
    print("3. fit validity")
    plot_fit_validity(output)
    print("4. hierarchical")
    plot_hierarchical(output, args.quick)
    print("5. pair counts")
    plot_pair_counts(output, args.quick)
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
