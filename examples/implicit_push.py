"""Implicit push.

An implicit, exactly energy-conserving push for N-body plasma, and what it costs.

Explicit integrators must resolve the plasma oscillation (``dt * omega_pe`` below about
one). An implicit midpoint step can take longer steps; with Gonzalez's discrete gradient
in place of the midpoint force it conserves the energy exactly for *any* potential, and
with a Jacobian-free Newton-Krylov solver each nonlinear iteration costs one force
evaluation, i.e. one fast-summation pass in a tree code. This example measures, with
direct sums on a small trapped cloud of line charges, how many force evaluations such a
step needs as the step grows, and whether the solver converges at all, which is the
whole question of whether the scheme can beat explicit subcycling once the force comes
from an FMM.

1. ``energy``: relative energy error over the run for explicit Verlet at a small step
   and for the implicit midpoint and discrete-gradient steps at ``dt * omega_pe = 0.5``;
   the discrete gradient holds the energy to the solver tolerance.
2. ``cost``: force (residual) evaluations per step of the two implicit schemes against
   ``dt * omega_pe``, with the number of explicit steps the same ``dt`` covers; the
   scheme pays only where the evaluations per step fall below that line. Steps on which
   the unpreconditioned Newton-Krylov solver did not converge are marked.
3. ``kinetic``: kinetic energy against time for the explicit reference and the
   discrete-gradient scheme at the three steps; what a step that does not resolve the
   plasma oscillation does to the dynamics.
4. ``convergence``: the residual history of one Newton-Krylov solve per step size; why
   the large steps fail without a preconditioner.

Usage::

    python examples/implicit_push.py [--output examples/output] [--quick]

``--quick`` uses a smaller cloud and shorter runs (about a minute instead of several).
Needs Plotly (``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from scipy.optimize import NoConvergence, newton_krylov

from tree_d_3d.smooth import direct_energy, direct_fields

COLORS = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
    "C4": "#9467bd",
}


class System:
    """A neutral cloud of line charges in a harmonic trap; counts force evaluations."""

    def __init__(self, n: int, seed: int = 3, k_trap: float = 100.0, eps: float = 0.1):
        rng = np.random.default_rng(seed)
        self.pos = rng.uniform(-0.8, 0.8, (2, n))
        self.q = rng.choice([-1.0, 1.0], n)
        self.vel = 0.2 * rng.normal(size=(2, n))
        self.m = np.ones(n)
        self.k = k_trap
        self.eps = eps
        self.evaluations = 0
        # 2D line charges: div E = 4 pi rho in these units, omega_pe^2 = 4 pi n q^2 / m
        density = n / 1.6**2
        self.omega_pe = float(np.sqrt(4 * np.pi * density))

    def energy(self, x: np.ndarray) -> float:
        """Potential energy, Coulomb plus trap."""
        return direct_energy(x, self.q, self.eps) + 0.5 * self.k * np.sum(x**2)

    def force(self, x: np.ndarray) -> np.ndarray:
        """Force on every particle (counted)."""
        self.evaluations += 1
        return self.q * direct_fields(x, self.q, self.eps)[1] - self.k * x

    def total(self, x: np.ndarray, v: np.ndarray) -> float:
        """Total energy."""
        return self.energy(x) + 0.5 * np.sum(self.m * np.sum(v**2, axis=0))


def verlet(sys_: System, dt: float, steps: int, every: int = 1):
    """Explicit velocity Verlet; returns (time, total energy, kinetic energy) samples."""
    x, v = sys_.pos.copy(), sys_.vel.copy()
    acc = sys_.force(x) / sys_.m
    out = []
    for s in range(steps):
        v += 0.5 * dt * acc
        x += dt * v
        acc = sys_.force(x) / sys_.m
        v += 0.5 * dt * acc
        if s % every == 0:
            out.append(((s + 1) * dt, sys_.total(x, v), 0.5 * np.sum(v**2)))
    return out


class ImplicitStep:
    """The nonlinear system of one implicit midpoint / discrete-gradient step.

    Unknown ``x1``; ``v1 = v0 + dt F / m`` with ``F`` the midpoint force or Gonzalez's
    discrete-gradient force, and ``x1 = x0 + dt (v0 + v1) / 2``.
    """

    def __init__(self, sys_: System, x0, v0, dt: float, discrete_gradient: bool):
        self.sys = sys_
        self.x0, self.v0, self.dt = x0, v0, dt
        self.discrete_gradient = discrete_gradient
        self.u0 = sys_.energy(x0)
        self.history: list[float] = []

    def mean_force(self, x1: np.ndarray) -> np.ndarray:
        f = self.sys.force(0.5 * (self.x0 + x1))
        if self.discrete_gradient:
            dx = x1 - self.x0
            norm2 = np.sum(dx * dx)
            if norm2 > 0:
                # Gonzalez: F_dg . dx = -(U(x1) - U(x0)) exactly
                f = f - (self.sys.energy(x1) - self.u0 + np.sum(f * dx)) / norm2 * dx
        return f

    def residual(self, x1_flat: np.ndarray) -> np.ndarray:
        x1 = x1_flat.reshape(2, -1)
        v1 = self.v0 + self.dt * self.mean_force(x1) / self.sys.m
        r = (x1 - self.x0 - 0.5 * self.dt * (self.v0 + v1)).ravel()
        self.history.append(float(np.abs(r).max()))
        return r


def implicit(
    sys_: System,
    dt: float,
    steps: int,
    discrete_gradient: bool,
    f_tol: float = 1e-11,
    maxiter: int = 100,
):
    """Run ``steps`` implicit steps with Jacobian-free Newton-Krylov (unpreconditioned GMRES).

    Returns the samples, the force evaluations per step, the number of failed steps
    (no convergence within ``maxiter`` Newton iterations: the last iterate is used) and
    the residual history of the first step.
    """
    x, v = sys_.pos.copy(), sys_.vel.copy()
    out, counts, failed, first_history = [], [], 0, None
    for s in range(steps):
        step = ImplicitStep(sys_, x.copy(), v.copy(), dt, discrete_gradient)
        before = sys_.evaluations
        guess = (step.x0 + dt * step.v0).ravel()
        try:
            x1 = newton_krylov(
                step.residual,
                guess,
                method="lgmres",
                f_tol=f_tol * dt,
                maxiter=maxiter,
                inner_maxiter=50,
            )
        except NoConvergence as exc:
            failed += 1
            x1 = np.asarray(exc.args[0])
        x1 = x1.reshape(2, -1)
        v1 = step.v0 + dt * step.mean_force(x1) / sys_.m
        counts.append(sys_.evaluations - before)
        if first_history is None:
            first_history = step.history
        x, v = x1, v1
        out.append(((s + 1) * dt, sys_.total(x, v), 0.5 * np.sum(v**2)))
    return out, counts, failed, first_history


def main(argv: list[str] | None = None) -> Path:
    """Run the comparison and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    n = 100 if args.quick else 200
    horizon = 1.0 if args.quick else 2.0
    sys_ = System(n)
    wpe = sys_.omega_pe
    print(f"N = {n}, omega_pe = {wpe:.1f}, horizon {horizon}")

    dt_ref = 0.1 / wpe
    steps_ref = int(round(horizon / dt_ref))
    sys_.evaluations = 0
    ref = verlet(sys_, dt_ref, steps_ref, every=max(1, steps_ref // 400))
    e0 = ref[0][1]
    print(
        f"  Verlet dt*wpe=0.1: {steps_ref} steps, 1 evaluation/step, "
        f"max|dE/E| {max(abs(r[1] / e0 - 1) for r in ref):.1e}"
    )

    factors = [0.5, 2.0, 10.0]
    results = {}
    for factor in factors:
        dt = factor / wpe
        steps = max(int(round(horizon / dt)), 3)
        for dg in (False, True):
            label = "discrete gradient" if dg else "midpoint"
            sys_.evaluations = 0
            hist, counts, failed, history = implicit(sys_, dt, steps, dg)
            err = max(abs(h[1] / e0 - 1) for h in hist)
            print(
                f"  {label:17s} dt*wpe={factor:4.1f}: {steps:4d} steps, "
                f"{np.mean(counts):6.1f} evaluations/step "
                f"(explicit steps covered: {factor / 0.1:.0f}), "
                f"{failed} failed, max|dE/E| {err:.1e}"
            )
            results[factor, dg] = (hist, counts, failed, history)

    # 1. energy
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r[0] for r in ref],
            y=[r[1] / e0 - 1 for r in ref],
            mode="lines",
            line={"color": "black"},
            name="explicit Verlet, dt·ω_pe = 0.1",
        )
    )
    for dg, color in ((False, COLORS["C1"]), (True, COLORS["C2"])):
        hist = results[0.5, dg][0]
        fig.add_trace(
            go.Scatter(
                x=[h[0] for h in hist],
                y=[h[1] / e0 - 1 for h in hist],
                mode="lines+markers",
                line={"color": color},
                name=("discrete gradient" if dg else "implicit midpoint")
                + ", dt·ω_pe = 0.5",
            )
        )
    fig.update_xaxes(title_text="time")
    fig.update_yaxes(title_text="relative energy error", exponentformat="e")
    fig.update_layout(
        title=f"Energy: the discrete gradient conserves it to the solver tolerance (N = {n}, ω_pe = {wpe:.0f})",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "energy")

    # 2. cost
    fig = go.Figure()
    for dg, color in ((False, COLORS["C1"]), (True, COLORS["C2"])):
        means = [np.mean(results[f, dg][1]) for f in factors]
        fails = [results[f, dg][2] for f in factors]
        fig.add_trace(
            go.Scatter(
                x=factors,
                y=means,
                mode="lines+markers+text",
                text=[f"{k} failed" if k else "" for k in fails],
                textposition="top center",
                line={"color": color},
                name="discrete gradient" if dg else "implicit midpoint",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=factors,
            y=[f / 0.1 for f in factors],
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="explicit steps covered (dt·ω_pe = 0.1 each)",
        )
    )
    fig.update_xaxes(type="log", title_text="dt · ω_pe")
    fig.update_yaxes(type="log", title_text="force evaluations per step")
    fig.update_layout(
        title="Cost: Jacobian-free Newton-Krylov without preconditioning",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "cost")

    # 3. kinetic energy
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r[0] for r in ref],
            y=[r[2] for r in ref],
            mode="lines",
            line={"color": "black"},
            name="explicit reference",
        )
    )
    for factor, color in zip(factors, (COLORS["C0"], COLORS["C2"], COLORS["C3"])):
        hist = results[factor, True][0]
        fig.add_trace(
            go.Scatter(
                x=[h[0] for h in hist],
                y=[h[2] for h in hist],
                mode="lines+markers",
                line={"color": color},
                name=f"discrete gradient, dt·ω_pe = {factor}",
            )
        )
    fig.update_xaxes(title_text="time")
    fig.update_yaxes(title_text="kinetic energy")
    fig.update_layout(
        title="Dynamics: a step that does not resolve ω_pe misses the oscillation",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "kinetic")

    # 4. convergence
    fig = go.Figure()
    for factor, color in zip(factors, (COLORS["C0"], COLORS["C2"], COLORS["C3"])):
        history = results[factor, True][3] or []
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history) + 1)),
                y=[max(h, 1e-18) for h in history],
                mode="lines",
                line={"color": color},
                name=f"dt·ω_pe = {factor}",
            )
        )
    fig.update_xaxes(
        title_text="residual evaluation (Newton iterations and GMRES matvecs)"
    )
    fig.update_yaxes(type="log", title_text="max |residual| of the first step")
    fig.update_layout(
        title="Convergence of one unpreconditioned Newton-Krylov solve (discrete gradient)",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "convergence")
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
