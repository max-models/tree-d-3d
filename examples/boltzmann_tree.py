"""Boltzmann electrons.

Boltzmann electrons on the quadtree, with the tree leaves as the quadrature of a nonlinear source.

A gridless ion code has no mesh to put a fluid electron model on. The quadtree leaves are
an adaptive partition of the plane, so put the Boltzmann electron charge of each leaf,
``-n0 A_L exp(phi_L / T)``, at quadrature points inside it and solve the nonlinear
problem ``phi = G(q_ions + q_e(phi))`` with the same summation the ions use. This example
does that on the textbook case with an exact answer, Debye shielding of a test line
charge in a neutral background (linear regime: ``phi = 2 Q K0(r / lambda_D)``; nonlinear
regime: the radial Poisson-Boltzmann ODE), with direct sums over the leaf charges, and
measures what the solve costs.

1. ``profiles``: the potential against radius from the tree solve, the linearised
   analytic solution ``2 Q K0(r / lambda_D)`` and the radial nonlinear ODE, for a weak and
   a strong test charge.
2. ``iterations``: potential evaluations to converge against the leaf size over the Debye
   length at a strong test charge (``Q = 1.5``, ``phi / T`` up to about 5), for damped
   Picard (fixed point) and Jacobian-free Newton-Krylov, both with continuation in the test
   charge; the damped Picard cost is independent of the resolution, the unpreconditioned
   Newton-Krylov cost grows with it.
3. ``convergence``: error of the tree solution against the radial ODE against the leaf
   size, with one and four quadrature points per leaf: roughly first order, with the
   leaf-charge representation (point charges plus a mean self-potential) as the limit.
4. ``leaves``: the adaptive leaves coloured by their electron charge for the strong test
   charge; refinement follows the potential gradient, not the ions.

Usage::

    python examples/boltzmann_tree.py [--output examples/output] [--quick]

``--quick`` uses coarser trees (about ten minutes instead of hours); ``--smoke`` is the
tiny configuration the test suite runs (a minute). Needs Plotly
(``pip install 'tree-d-3d[examples]'``) and SciPy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from scipy.integrate import solve_bvp
from scipy.optimize import NoConvergence, newton_krylov
from scipy.special import k0

COLORS = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
}

# units: line charges with phi = -2 q ln r, so div E = 4 pi rho and
# lambda_D^2 = T / (4 pi n0) for electrons of unit charge; e = 1, T in units of e phi.


def mean_self_log(samples: int = 400) -> float:
    """Mean of ln|r - r'| over a unit square against itself (for a cell's self-potential)."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, (samples * samples, 2))
    b = rng.uniform(0, 1, (samples * samples, 2))
    return float(np.mean(np.log(np.linalg.norm(a - b, axis=1))))


SELF_LOG = mean_self_log()


class LeafTree:
    """Square leaves over the disk of radius ``radius``, refined toward the origin.

    ``level`` is the uniform refinement level; leaves within ``refine_radius`` of the
    origin are split ``extra`` more times. Each leaf carries ``points`` quadrature points
    (1: its centre; 4: a 2x2 Gauss rule).
    """

    def __init__(
        self, radius: float, level: int, extra: int, refine_radius: float, points: int
    ):
        self.radius = radius
        h = radius / 2**level
        cells: list[tuple[complex, float]] = []
        for ix in range(-(2**level), 2**level):
            for iy in range(-(2**level), 2**level):
                c = complex((ix + 0.5) * h, (iy + 0.5) * h)
                if abs(c) > radius:
                    continue
                if abs(c) < refine_radius and extra:
                    sub = 2**extra
                    hh = h / sub
                    for jx in range(sub):
                        for jy in range(sub):
                            cells.append(
                                (
                                    c
                                    + complex(
                                        (jx + 0.5) * hh - h / 2, (jy + 0.5) * hh - h / 2
                                    ),
                                    hh,
                                )
                            )
                else:
                    cells.append((c, h))
        self.centre = np.array([c for c, _ in cells])
        self.size = np.array([s for _, s in cells])
        if points == 1:
            g = np.array([0.0])
            w = np.array([1.0])
        else:
            g = np.array([-1, 1]) / (2 * np.sqrt(3))
            w = np.array([0.5, 0.5])
        gx, gy = np.meshgrid(g, g)
        wx, wy = np.meshgrid(w, w)
        self.points = (
            self.centre[:, None]
            + self.size[:, None] * (gx.ravel() + 1j * gy.ravel())[None, :]
        ).ravel()
        self.weights = (
            self.size[:, None] ** 2 * (wx * wy).ravel()[None, :]
        ).ravel()  # areas
        self.leaf_of_point = np.repeat(np.arange(len(cells)), len(g) ** 2)
        self.n_points = len(g) ** 2
        # self term of a quadrature point: mean potential of its sub-cell over itself
        own = self.size[self.leaf_of_point] / np.sqrt(self.n_points)
        self.self_potential = -2.0 * (np.log(own) + SELF_LOG)

    def potential(self, charges: np.ndarray, test_charge: float) -> np.ndarray:
        """Potential at the quadrature points from the point charges and the test charge."""
        out = -2.0 * test_charge * np.log(np.abs(self.points))
        out += self.self_potential * charges
        chunk = 2048
        for lo in range(0, len(self.points), chunk):
            d = self.points[lo : lo + chunk, None] - self.points[None, :]
            with np.errstate(divide="ignore"):
                g = -2.0 * np.log(np.abs(d))
            idx = np.arange(lo, min(lo + chunk, len(self.points)))
            g[idx - lo, idx] = 0.0
            out[lo : lo + chunk] += g @ charges
        return out


def solve(
    tree: LeafTree,
    n0: float,
    T: float,
    test_charge: float,
    method: str,
    max_iter: int = 400,
    damping: float = 0.5,
):
    """Solve ``phi = G(q_background + q_e(phi)) + phi_test`` by continuation in the test charge.

    The charge is raised in steps (0.05, 0.2, 0.5, 1, 1.5, ... up to ``test_charge``), each
    solve starting from the previous solution, with damped Picard iteration or
    Jacobian-free Newton-Krylov. Returns phi, the total number of residual (potential)
    evaluations, and whether every stage converged.
    """
    q_bg = n0 * tree.weights  # neutralising background, fixed

    def electrons(phi):
        return -n0 * tree.weights * np.exp(np.clip(phi / T, -50, 50))

    stages = [
        q for q in (0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0) if q < test_charge
    ] + [test_charge]
    phi = np.zeros(len(tree.points))
    total, ok = 0, True
    for stage in stages:
        if method == "picard":
            converged = False
            for _ in range(max_iter):
                new = tree.potential(q_bg + electrons(phi), stage)
                total += 1
                step = np.abs(new - phi).max()
                phi = (1 - damping) * phi + damping * new
                if not np.isfinite(step) or step > 1e8:
                    break
                if step < 1e-9 * max(np.abs(phi).max(), 1e-30):
                    converged = True
                    break
            ok = ok and converged
        else:
            count = [0]

            def residual(p, stage=stage, count=count):
                count[0] += 1
                return p - tree.potential(q_bg + electrons(p), stage)

            try:
                phi = newton_krylov(
                    residual, phi, method="lgmres", f_tol=1e-10, maxiter=max_iter
                )
            except NoConvergence as exc:
                phi = np.asarray(exc.args[0])
                ok = False
            total += count[0]
    return phi, total, ok


def radial_ode(n0: float, T: float, test_charge: float, radius: float):
    """The nonlinear radial Poisson-Boltzmann problem, solved as a boundary value problem.

    ``phi'' + phi'/r = 4 pi n0 (exp(phi/T) - 1)``, ``r phi' -> -2 Q`` at the test charge,
    ``phi(radius) = 0`` (the tree's neutral far field).
    """
    r0 = 1e-4 * radius

    def rhs(r, y):
        return np.vstack(
            (y[1], 4 * np.pi * n0 * (np.exp(np.clip(y[0] / T, -50, 50)) - 1) - y[1] / r)
        )

    def bc(ya, yb):
        return np.array([r0 * ya[1] + 2 * test_charge, yb[0]])

    r = np.geomspace(r0, radius, 400)
    lam = np.sqrt(T / (4 * np.pi * n0))
    guess = np.vstack(
        (2 * test_charge * k0(np.maximum(r / lam, 1e-6)), -2 * test_charge / r)
    )
    sol = solve_bvp(rhs, bc, r, guess, tol=1e-8, max_nodes=20000)
    return sol


def main(argv: list[str] | None = None) -> Path:
    """Run the studies and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    n0, T = 1.0 / (4 * np.pi), 1.0  # lambda_D = 1
    lam = np.sqrt(T / (4 * np.pi * n0))
    radius = 8.0 * lam
    print(f"lambda_D = {lam:.3f}, domain radius {radius / lam:.0f} lambda_D")

    # 1. profiles
    level = 3 if args.smoke else 4 if args.quick else 5
    tree = LeafTree(radius, level, extra=2, refine_radius=1.5 * lam, points=4)
    print(
        f"  profile tree: {len(tree.centre)} leaves, {len(tree.points)} points, smallest leaf {tree.size.min() / lam:.3f} lambda_D"
    )
    fig = go.Figure()
    r_plot = np.geomspace(0.05 * lam, radius, 300)
    for test_charge, color in ((0.05, COLORS["C0"]), (2.0, COLORS["C3"])):
        phi, its, ok = solve(tree, n0, T, test_charge, "newton")
        print(
            f"  Q = {test_charge}: Newton-Krylov {its} residuals, converged {ok}, max phi/T {phi.max() / T:.2f}"
        )
        rr = np.abs(tree.points)
        order = np.argsort(rr)
        fig.add_trace(
            go.Scatter(
                x=rr[order] / lam,
                y=phi[order],
                mode="markers",
                marker={"color": color, "size": 3},
                name=f"tree leaves, Q = {test_charge}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=r_plot / lam,
                y=2 * test_charge * k0(r_plot / lam),
                mode="lines",
                line={"color": color, "dash": "dot"},
                name=f"linear: 2 Q K0(r/λ_D), Q = {test_charge}",
            )
        )
        sol = radial_ode(n0, T, test_charge, radius)
        fig.add_trace(
            go.Scatter(
                x=r_plot / lam,
                y=sol.sol(r_plot)[0],
                mode="lines",
                line={"color": color},
                name=f"radial nonlinear ODE, Q = {test_charge}",
            )
        )
    fig.update_xaxes(type="log", title_text="r / λ_D")
    fig.update_yaxes(title_text="φ / T")
    fig.update_layout(
        title="Debye shielding of a test line charge: Boltzmann electrons on the tree leaves",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "profiles")

    # 2. iterations vs leaf size
    levels = [1, 2] if args.smoke else [1, 2, 3, 4] if args.quick else [1, 2, 3, 4, 5]
    strong = 1.5
    fig = go.Figure()
    for method, color in (("picard", COLORS["C1"]), ("newton", COLORS["C2"])):
        xs, ys, texts = [], [], []
        for level in levels:
            t = LeafTree(radius, level, extra=0, refine_radius=0.0, points=1)
            _, its, ok = solve(t, n0, T, strong, method, max_iter=300)
            xs.append(t.size.min() / lam)
            ys.append(its)
            texts.append("" if ok else "diverged")
            print(
                f"  {method:7s} h/lambda_D = {xs[-1]:.2f}: {its} potential evaluations, converged {ok}"
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers+text",
                text=texts,
                textposition="top center",
                line={"color": color},
                name="damped Picard (0.5), with continuation in Q"
                if method == "picard"
                else "Newton-Krylov (residual evaluations), with continuation in Q",
            )
        )
    fig.update_xaxes(type="log", title_text="leaf size / λ_D")
    fig.update_yaxes(
        type="log", title_text="potential evaluations to converge (Q = 1.5)"
    )
    fig.update_layout(
        title="Cost of the nonlinear solve against the leaf size",
        legend={"font": {"size": 10}},
    )
    save_figure(fig, output / "iterations")

    # 3. convergence vs leaf size
    fig = go.Figure()
    test_charge = 1.0
    sol = radial_ode(n0, T, test_charge, radius)
    sizes = [2, 3] if args.smoke else [2, 3, 4] if args.quick else [2, 3, 4, 5]
    for points, color in ((1, COLORS["C0"]), (4, COLORS["C2"])):
        xs, ys = [], []
        for level in sizes:
            t = LeafTree(radius, level, extra=1, refine_radius=1.5 * lam, points=points)
            phi, _, ok = solve(t, n0, T, test_charge, "newton")
            rr = np.abs(t.points)
            mask = (rr > 0.5 * lam) & (rr < 5 * lam)
            err = (
                np.abs(phi[mask] - sol.sol(rr[mask])[0]).max()
                / np.abs(sol.sol(rr[mask])[0]).max()
            )
            xs.append(t.size.min() / lam)
            ys.append(err)
            print(f"  {points} point(s): h/lambda_D = {xs[-1]:.3f}, error {err:.2e}")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line={"color": color},
                name=f"{points} quadrature point(s) per leaf",
            )
        )
    fig.update_xaxes(type="log", title_text="smallest leaf size / λ_D")
    fig.update_yaxes(
        type="log", title_text="max relative error of φ for 0.5 < r/λ_D < 5"
    )
    fig.update_layout(title="Convergence against the radial nonlinear ODE (Q = 1)")
    save_figure(fig, output / "convergence")

    # 4. leaves
    phi, _, _ = solve(tree, n0, T, 2.0, "newton")
    q_e = -n0 * tree.weights * np.exp(np.clip(phi / T, -50, 50))
    leaf_q = np.bincount(tree.leaf_of_point, q_e, minlength=len(tree.centre))
    leaf_area = tree.size**2
    fig = go.Figure()
    shapes = []
    density = leaf_q / leaf_area / (-n0)
    for c, s, dens in zip(tree.centre, tree.size, density):
        shade = min(1.0, max(0.0, (dens - 1.0) / 6.0))
        shapes.append(
            {
                "type": "rect",
                "x0": c.real - s / 2,
                "y0": c.imag - s / 2,
                "x1": c.real + s / 2,
                "y1": c.imag + s / 2,
                "line": {"color": "rgb(120,120,120)", "width": 0.4},
                "fillcolor": f"rgba(214,39,40,{shade:.2f})",
            }
        )
    fig.update_layout(
        shapes=shapes,
        title="Adaptive leaves and their electron density n_e / n0 - 1 (red) around the strong test charge",
        xaxis={"range": [-3 * lam, 3 * lam], "scaleanchor": "y", "showgrid": False},
        yaxis={"range": [-3 * lam, 3 * lam], "showgrid": False},
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker={"color": "black", "size": 8},
            name="test charge",
        )
    )
    save_figure(fig, output / "leaves")
    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
