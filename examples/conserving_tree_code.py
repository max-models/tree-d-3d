"""A tree code that conserves both momentum and energy: the evidence, as figures.

Runs the experiments behind ``tree_d_3d.smooth`` and saves one figure per claim:

1. ``truncation_accuracy.png``: field error against the expansion order for the
   rectangle (standard) and triangle (symmetric) M2L truncations.
2. ``truncation_momentum.png``: the net force of the same runs; the triangle gives
   roundoff at every order, the rectangle the truncation error.
3. ``partition_of_unity.png``: the leaf boxes of an adaptive tree and the weight of one
   leaf with its smooth band, plus the sum of all weights (one everywhere).
4. ``gradient_check.png``: analytic force against finite differences of the energy, for
   particles blended into several leaves.
5. ``energy_drift.png``: relative energy error over a Verlet run for the direct sum,
   hard membership (fixed boxes and rebuilt every step), smooth membership, and
   smooth membership with co-moving boxes.
6. ``momentum_drift.png``: the net force over the same runs.
7. ``trajectories.png``: the particle trajectories of the conserving run over the fixed
   leaf boxes, and the distance of each variant's trajectories from the direct sum's.
8. ``trajectories_gentle.png``: a small ring of like charges orbiting in the trap,
   where the orbits are recognisable: the conserving tree code's trajectories over the
   direct sum's, and the energy error of both.

And three animations (GIF): ``cloud.gif`` (the trapped cloud under the conserving scheme
with its energy error), ``orbits.gif`` (the ring of like charges, tree code over direct
sum) and ``membership.gif`` (a tracer crossing leaf edges, with its leaf weights).

Usage::

    python examples/conserving_tree_code.py [--output examples/output] [--quick]

``--quick`` uses fewer particles and steps (about 20 s instead of a few minutes).
Needs matplotlib (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from tree_d_3d.quad import Quad
from tree_d_3d.smooth import (
    BoxTree,
    SmoothFMM,
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
    import matplotlib.pyplot as plt

    thetas = sorted({r[0] for r in rows})
    for key, name, ylabel in (
        (3, "truncation_accuracy.png", "max field error relative to the direct sum"),
        (4, "truncation_momentum.png", "net force / sum of |forces|"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for theta, color in zip(thetas, ("C0", "C3")):
            for symmetric, style, label in (
                (False, "o-", "rectangle (standard)"),
                (True, "s--", "triangle (symmetric)"),
            ):
                sel = [r for r in rows if r[0] == theta and r[2] == symmetric]
                ax.semilogy(
                    [r[1] for r in sel],
                    [max(r[key], 1e-18) for r in sel],
                    style,
                    color=color,
                    label=f"{label}, θ = {theta}",
                )
        ax.set_xlabel("expansion order p")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        if key == 4:
            ax.set_title("Momentum: pair antisymmetry of the M2L truncation")
        else:
            ax.set_title("Accuracy: the triangle needs ~4 more orders")
        fig.tight_layout()
        fig.savefig(output / name, dpi=150)
        plt.close(fig)


# -- 3: partition of unity -------------------------------------------------------


def plot_partition_of_unity(output: Path) -> None:
    """Figure 3."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, field, title in (
        (axes[0], w_leaf, "weight $w_L$ of one leaf (band = half its width / 2)"),
        (axes[1], total, "sum of all leaf weights"),
    ):
        im = ax.imshow(
            field,
            extent=(-1, 1, -1, 1),
            origin="lower",
            vmin=0,
            vmax=1.0 if ax is axes[0] else 1.001,
            cmap="viridis",
        )
        for L in leaves:
            c, h = tree.center[L], tree.half[L]
            ax.add_patch(
                Rectangle(
                    (c.real - h, c.imag - h),
                    2 * h,
                    2 * h,
                    fill=False,
                    lw=0.4,
                    ec="white",
                )
            )
        ax.plot(pos[0], pos[1], ".", color="k", ms=2, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
    c, h = tree.center[middle], tree.half[middle]
    axes[0].set_xlim(c.real - 4 * h, c.real + 4 * h)
    axes[0].set_ylim(c.imag - 4 * h, c.imag + 4 * h)
    axes[1].text(
        -0.95,
        -0.95,
        f"max |Σw − 1| = {np.abs(total - 1).max():.1e}",
        color="white",
        fontsize=9,
    )
    fig.suptitle("Smooth membership is a partition of unity on the adaptive leaves")
    fig.tight_layout()
    fig.savefig(output / "partition_of_unity.png", dpi=150)
    plt.close(fig)


# -- 4: gradient check ------------------------------------------------------------


def plot_gradient_check(output: Path, n: int = 300) -> None:
    """Figure 4."""
    import matplotlib.pyplot as plt

    pos, q = cloud(n, seed=2)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    kw = dict(
        order=8,
        theta=0.6,
        band=0.4 * tree.min_leaf_half,
        softening=0.02,
        symmetric=True,
    )
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot([-scale, scale], [-scale, scale], "k-", lw=0.8)
    sc = axes[0].scatter(
        analytic, numeric, c=members, cmap="viridis", s=22, vmin=1, vmax=4
    )
    fig.colorbar(sc, ax=axes[0], label="leaves the particle belongs to")
    axes[0].set_xlabel("analytic force component  2q Σ [w G + Φ ∇w]")
    axes[0].set_ylabel("−dU/dx by central differences")
    axes[0].set_title("Forces are the exact gradient of the energy")
    axes[1].semilogy(np.abs(numeric - analytic) / scale, "o", ms=4)
    axes[1].axhline(
        1e-8, color="k", ls="--", lw=0.8, label="finite-difference noise level"
    )
    axes[1].set_xlabel("sample")
    axes[1].set_ylabel("|F_fd − F| / max|F|")
    axes[1].set_ylim(1e-13, 1e-5)
    axes[1].legend()
    axes[1].set_title(f"max mismatch {np.abs(numeric - analytic).max() / scale:.1e}")
    fig.tight_layout()
    fig.savefig(output / "gradient_check.png", dpi=150)
    plt.close(fig)


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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, color, times, drift, _, _ in results:
        ax.plot(times, drift, color=color, lw=1.2, label=label)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("relative energy error  (E − E₀) / |E₀|")
    ax.set_title(
        f"Energy: N = {info['n']}, p = {info['order']}, θ = {info['theta']}, "
        f"Δt = {info['dt']} (Verlet)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "energy_drift.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, color, times, _, net, _ in results:
        ax.semilogy(times, np.maximum(net, 1e-18), color=color, lw=1.2, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("net force / sum of |forces|")
    ax.set_title(
        "Momentum: symmetric truncation, and co-moving boxes for the smooth terms"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output / "momentum_drift.png", dpi=150)
    plt.close(fig)


def plot_trajectories(results, info, output: Path) -> None:
    """Figure 7."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tree, q = info["tree"], info["q"]
    reference = results[0][5]
    conserving = results[-1]
    history = conserving[5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    for L in tree.leaves:
        c, h = tree.center[L], tree.half[L]
        ax.add_patch(
            Rectangle(
                (c.real - h, c.imag - h), 2 * h, 2 * h, fill=False, lw=0.3, ec="0.6"
            )
        )
    for i in range(history.shape[2]):
        ax.plot(
            history[:, 0, i],
            history[:, 1, i],
            "-",
            lw=0.5,
            color="C3" if q[i] > 0 else "C0",
            alpha=0.7,
        )
    ax.plot(history[0, 0], history[0, 1], "k.", ms=2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title(
        f"{conserving[0]}\nred: +, blue: −, dots: start; grey: the fixed leaf boxes",
        fontsize=9,
    )
    ax = axes[1]
    for label, color, times, _, _, hist in results[1:]:
        distance = np.sqrt(((hist - reference) ** 2).sum(axis=1)).max(axis=1)
        ax.semilogy(
            times, np.maximum(distance, 1e-17), color=color, lw=1.2, label=label
        )
    ax.set_xlabel("time")
    ax.set_ylabel("max over particles of |x − x_direct|")
    ax.set_title(
        "Distance from the direct-sum trajectories\n(grows exponentially for every variant: N-body chaos)",
        fontsize=9,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=150)
    plt.close(fig)


def plot_gentle_trajectories(output: Path, steps: int = 2000) -> None:
    """Figure 8."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

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

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    for L in tree.leaves:
        c, h = tree.center[L], tree.half[L]
        ax.add_patch(
            Rectangle(
                (c.real - h, c.imag - h), 2 * h, 2 * h, fill=False, lw=0.3, ec="0.6"
            )
        )
    for i in range(n):
        ax.plot(hist_d[:, 0, i], hist_d[:, 1, i], "k--", lw=1.6, alpha=0.5)
        ax.plot(
            hist_t[:, 0, i],
            hist_t[:, 1, i],
            "-",
            lw=0.8,
            color=f"C{i % 10}",
        )
    ax.plot(pos[0], pos[1], "k.", ms=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(
        "8 like charges in a trap: tree code (colour) over direct sum (dashed)\n"
        f"max trajectory difference {np.sqrt(((hist_t - hist_d) ** 2).sum(axis=1)).max():.1e}",
        fontsize=9,
    )
    ax = axes[1]
    ax.plot(times, drift_d, "k--", lw=1.2, label="direct sum")
    ax.plot(
        times, drift_t, "C2-", lw=1.0, label="smooth membership, fixed co-moving boxes"
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("relative energy error")
    ax.set_title(f"energy; net force of the tree code ≤ {net_t.max():.0e}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "trajectories_gentle.png", dpi=150)
    plt.close(fig)
    return times, hist_d, hist_t, tree, q


def _writer(fps: int):
    from matplotlib.animation import PillowWriter

    return PillowWriter(fps=fps)


def _draw_boxes(ax, tree, ec="0.6") -> None:
    from matplotlib.patches import Rectangle

    for L in tree.leaves:
        c, h = tree.center[L], tree.half[L]
        ax.add_patch(
            Rectangle((c.real - h, c.imag - h), 2 * h, 2 * h, fill=False, lw=0.3, ec=ec)
        )


def animate_cloud(results, info, output: Path, frames: int = 160) -> None:
    """``cloud.gif``: the conserving run, particles over the fixed boxes, energy error."""
    import matplotlib.pyplot as plt

    label, color, times, drift, _, history = results[-1]
    step = max(1, len(times) // frames)
    idx = np.arange(0, len(times), step)
    q, tree = info["q"], info["tree"]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.1]}
    )
    _draw_boxes(ax, tree)
    pos_plus = ax.plot([], [], "o", color="C3", ms=3, label="+")[0]
    pos_minus = ax.plot([], [], "o", color="C0", ms=3, label="−")[0]
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    for _, col, t_, d_, _, _ in results:
        ax2.plot(t_, d_, color=col, lw=0.8, alpha=0.35)
    trace = ax2.plot([], [], color=color, lw=1.6, label=label)[0]
    ax2.set_xlim(0, times[-1])
    lim = 1.2 * max(np.abs(r[3]).max() for r in results)
    ax2.set_ylim(-lim, lim)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("time")
    ax2.set_ylabel("relative energy error")
    ax2.set_title("energy error (faint: the other variants)", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    writer = _writer(fps=20)
    with writer.saving(fig, output / "cloud.gif", dpi=90):
        for k in idx:
            x = history[k]
            pos_plus.set_data(x[0, q > 0], x[1, q > 0])
            pos_minus.set_data(x[0, q < 0], x[1, q < 0])
            trace.set_data(times[: k + 1], drift[: k + 1])
            writer.grab_frame()
    plt.close(fig)


def animate_orbits(
    times, hist_d, hist_t, tree, q, output: Path, frames: int = 200, trail: int = 60
) -> None:
    """``orbits.gif``: the ring of like charges, tree code (colour) over the direct sum (black)."""
    import matplotlib.pyplot as plt

    n = hist_t.shape[2]
    step = max(1, len(times) // frames)
    idx = np.arange(0, len(times), step)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    _draw_boxes(ax, tree)
    trails_d = [
        ax.plot([], [], "-", color="k", lw=2.2, alpha=0.35)[0] for _ in range(n)
    ]
    trails_t = [ax.plot([], [], "-", color=f"C{i % 10}", lw=1.0)[0] for i in range(n)]
    dots = ax.plot([], [], "o", color="k", ms=4)[0]
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    title = ax.set_title("", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    writer = _writer(fps=25)
    with writer.saving(fig, output / "orbits.gif", dpi=90):
        for k in idx:
            lo = max(0, k - trail)
            for i in range(n):
                trails_d[i].set_data(hist_d[lo : k + 1, 0, i], hist_d[lo : k + 1, 1, i])
                trails_t[i].set_data(hist_t[lo : k + 1, 0, i], hist_t[lo : k + 1, 1, i])
            dots.set_data(hist_t[k, 0], hist_t[k, 1])
            dist = np.sqrt(((hist_t[k] - hist_d[k]) ** 2).sum(axis=0)).max()
            title.set_text(
                f"8 like charges: tree code (colour) over direct sum (black)\n"
                f"t = {times[k]:.2f}, max |x − x_direct| = {dist:.1e}"
            )
            writer.grab_frame()
    plt.close(fig)


def animate_membership(output: Path, frames: int = 150) -> None:
    """``membership.gif``: a tracer crossing leaf edges; its leaf weights sum to one."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    pos, _ = cloud(300, seed=7)
    tree = BoxTree(pos, Quad(0, 0, 1), leaf_capacity=8)
    band = 0.5 * tree.min_leaf_half
    # a straight path through the middle of the tree that crosses several edges and a corner
    path = np.linspace(-0.55, 0.55, frames)
    track = path + 1j * (0.35 * path + 0.08)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    _draw_boxes(ax, tree, ec="0.5")
    for L in tree.leaves:
        c, h = tree.center[L], tree.half[L]
        ax.add_patch(
            Rectangle(
                (c.real - h - band, c.imag - h - band),
                2 * (h + band),
                2 * (h + band),
                fill=False,
                lw=0.3,
                ls=":",
                ec="C0",
            )
        )
    ax.plot(pos[0], pos[1], ".", color="0.7", ms=2)
    ax.plot(track.real, track.imag, "-", color="0.8", lw=0.8)
    tracer = ax.plot([], [], "o", color="C3", ms=7)[0]
    shaded = []
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.45, 0.6)
    ax.set_aspect("equal")
    ax.set_title(
        "tracer (red); leaves (grey), leaves grown by the band (dotted)", fontsize=9
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("weight $w_L$ of the tracer in leaf L")
    ax2.set_title("membership weights (they sum to 1)", fontsize=9)
    writer = _writer(fps=20)
    with writer.saving(fig, output / "membership.gif", dpi=90):
        for z in track:
            part, leaf, w, _, _ = tree.memberships(np.array([z]), band)
            order = np.argsort(-w)
            leaf, w = leaf[order], w[order]
            tracer.set_data([z.real], [z.imag])
            for patch in shaded:
                patch.remove()
            shaded = []
            for L, weight in zip(leaf, w):
                c, h = tree.center[L], tree.half[L]
                shaded.append(
                    ax.add_patch(
                        Rectangle(
                            (c.real - h, c.imag - h),
                            2 * h,
                            2 * h,
                            color="C1",
                            alpha=0.6 * weight,
                        )
                    )
                )
            ax2.cla()
            ax2.bar([f"leaf {int(L)}" for L in leaf], w, color="C1")
            ax2.bar(["sum"], [w.sum()], color="0.4")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("weight $w_L$ of the tracer in leaf L")
            ax2.set_title("membership weights (they sum to 1)", fontsize=9)
            writer.grab_frame()
    plt.close(fig)


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
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
