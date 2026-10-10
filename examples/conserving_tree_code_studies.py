"""Parameter studies behind the conserving tree code, as figures.

Companion of ``conserving_tree_code.py``; each figure answers one question:

1. ``kernels.png``: the pair potential and field of the bare, Plummer and compact
   (smooth cloud) softening; why the far field of the compact one is exact.
2. ``weights.png``: the smoothstep weights of two neighbouring leaves across their
   edge and the derivative that enters the force.
3. ``band_study.png``: field error, memberships per particle and energy error against
   the band width.
4. ``cost_matched.png``: the cheapest (order, theta) reaching a given field error for
   the rectangle and the triangle truncation, with the M2L and near-field work counted.
5. ``long_run.png``: energy error over a long Verlet run: no secular drift with smooth
   membership on fixed co-moving boxes.
6. ``error_map.png``: where the field error sits, for hard and smooth membership (it is
   not concentrated at leaf edges, and blending does not change its distribution).
7. ``correction_term.png``: the size of the ``Phi grad w`` force term relative to the
   ordinary ``w G`` term, which is what restores the exact gradient.

Usage::

    python examples/conserving_tree_code_studies.py [--output examples/output] [--quick]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

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
    import matplotlib.pyplot as plt

    eps = 1.0
    r = np.linspace(0.02, 4.0, 400)
    dz = r + 0.0j
    bare_phi, bare_g = _disk_kernel(dz, 0.0)
    compact_phi, compact_g = _disk_kernel(dz, eps**2)
    plummer_phi = 0.5 * np.log(r**2 + eps**2)
    plummer_g = r / (r**2 + eps**2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(r, bare_phi, "k--", label="bare: ln r")
    ax.plot(r, plummer_phi, "C1", label="Plummer: ½ ln(r² + ε²)")
    ax.plot(
        r, compact_phi, "C2", label="compact cloud (density ∝ (1 − r²/ε²)² inside ε)"
    )
    ax.axvline(eps, color="0.6", lw=0.8)
    ax.set_xlabel("r / ε")
    ax.set_ylabel("pair potential Φ")
    ax.set_ylim(-3, 1.6)
    ax.legend(fontsize=8)
    ax.set_title("potential", fontsize=10)
    ax = axes[1]
    ax.plot(r, bare_g.real, "k--", label="bare: 1/r")
    ax.plot(r, plummer_g, "C1", label="Plummer")
    ax.plot(r, compact_g.real, "C2", label="compact cloud")
    ax.plot(
        r, np.abs(plummer_g - bare_g.real), "C1:", label="|Plummer − bare| (never zero)"
    )
    ax.axvline(eps, color="0.6", lw=0.8)
    ax.set_xlabel("r / ε")
    ax.set_ylabel("|∇Φ|")
    ax.set_ylim(0, 2.2)
    ax.legend(fontsize=8)
    ax.set_title(
        "field: the compact cloud equals the bare kernel beyond ε", fontsize=10
    )
    fig.suptitle(
        "Softening kernels: only a compact kernel keeps the multipole far field exact"
    )
    fig.tight_layout()
    fig.savefig(output / "kernels.png", dpi=150)
    plt.close(fig)


# -- 2: weights ----------------------------------------------------------------------


def plot_weights(output: Path) -> None:
    """Figure 2."""
    import matplotlib.pyplot as plt

    band = 0.25
    x = np.linspace(-1, 1, 801)
    # two leaves sharing the edge x = 0, each of half-width 1
    s_right, ds_right = _smoothstep(x / band)  # weight of the leaf on the right
    s_left, ds_left = _smoothstep(-x / band)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(x, s_left, "C0", label="$w_{L}$ (left leaf)")
    ax.plot(x, s_right, "C1", label="$w_{L'}$ (right leaf)")
    ax.plot(x, s_left + s_right, "k--", lw=0.8, label="sum")
    ax.axvspan(-band, band, color="0.9", label="band ±δ")
    ax.set_xlabel("x (edge at 0)")
    ax.set_ylabel("weight")
    ax.legend(fontsize=8)
    ax.set_title("quintic smoothstep: C², s(t) + s(−t) = 1", fontsize=10)
    ax = axes[1]
    ax.plot(x, -ds_left / band, "C0", label="$dw_L/dx$")
    ax.plot(x, ds_right / band, "C1", label="$dw_{L'}/dx$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("dw/dx")
    ax.legend(fontsize=8)
    ax.set_title(
        "the derivatives are opposite: the Φ∇w forces see Φ_L − Φ_L′", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(output / "weights.png", dpi=150)
    plt.close(fig)


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
    import matplotlib.pyplot as plt

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
            f"  band {frac:.2f}: error {errors[len(drifts) - 1]:.2e}, memberships {members[len(drifts) - 1]:.2f}, max|dE/E| {drifts[-1]:.2e}"
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
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].semilogy(fractions, errors, "o-")
    axes[0].set_ylabel("max field error (p = 12, θ = 0.5)")
    axes[0].set_title(
        "accuracy: non-monotonic; a wide band averages\ntwo leaves' expansions and can even help",
        fontsize=9,
    )
    axes[1].plot(fractions, members, "o-")
    axes[1].set_ylabel("memberships per particle")
    axes[1].set_title("cost of P2M, L2P and the near field", fontsize=9)
    axes[2].semilogy(
        fractions, drifts, "o-", label="tree code (p = 2, θ = 0.8, co-moving)"
    )
    axes[2].axhline(
        reference, color="k", ls="--", lw=0.8, label="direct sum (integrator alone)"
    )
    axes[2].set_ylabel("max |ΔE/E| over the run")
    axes[2].legend(fontsize=8)
    axes[2].set_title("energy: any band > 0 removes the drift", fontsize=9)
    for ax in axes:
        ax.set_xlabel("band δ / smallest leaf half-width")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output / "band_study.png", dpi=150)
    plt.close(fig)


# -- 4: cost at matched accuracy -----------------------------------------------------


def plot_cost_matched(output: Path, quick: bool) -> None:
    """Figure 4."""
    import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for symmetric, color, label in (
        (False, "C0", "rectangle (standard)"),
        (True, "C3", "triangle (symmetric)"),
    ):
        best = []
        for tol in tolerances:
            ok = [r for r in rows if r[2] == symmetric and r[3] <= tol]
            best.append(min(r[4] for r in ok) if ok else np.nan)
        axes[0].loglog(tolerances, best, "o-", color=color, label=label)
        sel = [r for r in rows if r[2] == symmetric]
        axes[1].scatter(
            [r[3] for r in sel],
            [r[4] for r in sel],
            s=14,
            color=color,
            alpha=0.6,
            label=label,
        )
    axes[0].invert_xaxis()
    axes[0].set_xlabel("target field error")
    axes[0].set_ylabel("cheapest work (M2L + near field, model units)")
    axes[0].set_title(
        "work at matched accuracy, best over θ ∈ [0.4, 0.8] and p ≤ 28", fontsize=9
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("field error")
    axes[1].set_ylabel("work")
    axes[1].set_title("every (θ, p) combination", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.suptitle(
        "Exact momentum (triangle) costs about the same as the standard truncation at equal accuracy"
    )
    fig.tight_layout()
    fig.savefig(output / "cost_matched.png", dpi=150)
    plt.close(fig)


# -- 5: long run ---------------------------------------------------------------------


def plot_long_run(output: Path, quick: bool) -> None:
    """Figure 5."""
    import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, fields, color in (
        ("direct sum (integrator alone)", direct, "k"),
        ("hard membership, fixed boxes", tree_fields(0.0), "C1"),
        ("smooth membership, fixed co-moving boxes", tree_fields(0.5), "C2"),
    ):
        t0 = time.time()
        drift = verlet_energy(pos, vel, q, m, dt, steps, k_trap, fields)
        print(
            f"  {label:42s} max|dE/E| {np.abs(drift).max():.2e} final {drift[-1]:+.2e} ({time.time() - t0:.0f}s)"
        )
        ax.plot(times, drift, color=color, lw=0.9, label=label)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("relative energy error")
    ax.set_title(
        f"{steps} steps, N = {n}, p = {options['order']}, θ = {options['theta']}: no secular drift with smooth membership",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "long_run.png", dpi=150)
    plt.close(fig)


# -- 6: error map --------------------------------------------------------------------


def plot_error_map(output: Path, quick: bool) -> None:
    """Figure 6."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    pos, q = cloud(600 if quick else 1000, seed=8)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    _, ref = direct_fields(pos, q)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    vmin, vmax = 1e-9, 1e-4
    for ax, frac, title in (
        (axes[0], 0.0, "hard membership"),
        (axes[1], 0.5, "smooth membership, band = 0.5 h_min"),
    ):
        f = SmoothFMM(tree, pos, q, order=8, theta=0.6, band=frac * tree.min_leaf_half)
        _, e = f.fields()
        err = np.sqrt(((e - ref) ** 2).sum(axis=0)) / np.abs(ref).max()
        for L in tree.leaves:
            c, h = tree.center[L], tree.half[L]
            ax.add_patch(
                Rectangle(
                    (c.real - h, c.imag - h), 2 * h, 2 * h, fill=False, lw=0.3, ec="0.6"
                )
            )
        sc = ax.scatter(
            pos[0],
            pos[1],
            c=np.clip(err, vmin, vmax),
            s=9,
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_aspect("equal")
        ax.set_title(f"{title}: max error {err.max():.1e}", fontsize=9)
    fig.colorbar(
        sc, ax=axes, shrink=0.85, label="field error / max |E| (p = 8, θ = 0.6)"
    )
    fig.suptitle(
        "Where the error is: smooth membership leaves the error distribution unchanged "
        "(no concentration at leaf edges)",
        fontsize=10,
    )
    fig.savefig(output / "error_map.png", dpi=150)
    plt.close(fig)


# -- 7: correction term --------------------------------------------------------------


def plot_correction_term(output: Path, quick: bool) -> None:
    """Figure 7."""
    import matplotlib.pyplot as plt

    pos, q = cloud(600 if quick else 1200, seed=5)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-8, 0, 40)
    for frac, color in ((0.1, "C0"), (0.25, "C1"), (0.5, "C2")):
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
        ax.hist(
            np.clip(ratio, bins[0], bins[-1]),
            bins=bins,
            histtype="step",
            lw=1.5,
            color=color,
            label=f"band = {frac} h_min  ({blended.sum()} blended particles)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("|Φ ∇w term| / max |w G term|   (per blended particle)")
    ax.set_ylabel("particles")
    ax.set_title(
        "The exact-gradient correction is small (truncation error / band width) but not negligible",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output / "correction_term.png", dpi=150)
    plt.close(fig)


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
