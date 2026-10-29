"""Boundary-integral electrodes.

Conductors at prescribed potentials, solved by the boundary-integral (panel) method.

A particle code with electrodes needs conductors held at prescribed potentials, not just
grounded absorbing walls or the two infinite grounded plates of
:class:`tree_d_3d.GroundedStripField`. :class:`tree_d_3d.BoundaryIntegralElectrodes`
discretises each conductor's boundary into straight panels (here from
:mod:`simgeometry` shapes, by :func:`tree_d_3d.conductors_from_shape`), collocates the Dirichlet
condition at their midpoints and solves the resulting dense linear system for the
induced line-charge density -- the standard first-kind single-layer boundary-element
method, with a closed-form self-term for each panel and the exact field of a uniformly
charged segment for the panels near a target (farther ones are summed fast as Gauss
points). This example checks it against two
cases with an exact answer in the code's own units, both line-charge electrostatics with
the log kernel ``phi = -2 k_e q ln r``:

A grounded circular conductor with one external line charge has a closed-form solution
with *three* real images: the usual inverse-point image that cancels the source on the
circle, plus a charge at the circle's centre that removes the constant potential offset
the first image leaves behind (a detail of the log kernel's reference point, only
visible written out -- and the reason the single-conductor problem degenerates when the
circle's radius equals the kernel's reference length). Two concentric circular
conductors at different potentials is the classic cylindrical capacitor, with an exact
logarithmic potential profile between them and a known induced charge from Gauss's law.

1. ``grounded_circle``: the boundary potential around a grounded circle with an external
   charge, sampled between the panels (not at the collocation points), against the
   target of zero.
2. ``image_force``: the force the circle exerts back on the external charge against the
   three-image closed form, over the panel count -- the convergence of the method, at
   second order in the panel length, since panels near a target enter with their exact
   field (the plain midpoint rule converges at first order).
3. ``capacitor_profile``: the potential across the gap of a two-conductor cylindrical
   capacitor against the exact logarithmic profile.
4. ``capacitor_charge``: the induced charge on the inner conductor against the exact
   Gauss's-law value, over the panel count: third order.
5. ``corner_potential``: a filled contour of the potential around a grounded, non-convex
   (L-shaped) conductor with a nearby charge -- panels need no special treatment at a
   corner, unlike methods that assume a smooth boundary; the boundary potential error is
   printed, there being no closed form for this shape.

Usage::

    python examples/boundary_integral_electrodes.py [--output examples/output] [--quick]

``--quick`` uses fewer panels in the convergence studies (under a minute instead of a
few); ``--smoke`` is the tiny configuration the test suite runs. Needs Plotly
(``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from plotly.subplots import make_subplots
from simgeometry import Circle, Polygon

from tree_d_3d import BoundaryIntegralElectrodes, Conductor, conductors_from_shape

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}
K_E = 1.0  # code units: the log kernel's own reference length is the unit of distance


def three_image_force(a: float, d: float, q0: float) -> float:
    """Force on a line charge ``q0`` at distance ``d`` from a grounded circle of radius ``a``.

    Exact for the code's log kernel: the usual inverse-point image ``-q0`` at ``a**2/d``
    plus a charge at the centre that cancels the constant offset the first image leaves
    on the circle (see the module docstring).
    """

    def field_from(q, z, at):
        dx = at - z
        return 2 * K_E * q * dx / dx**2

    z_im, q_im = a * a / d, -q0
    q_c = -q0 * np.log(d / a) / np.log(a)
    return q0 * (field_from(q_im, z_im, d) + field_from(q_c, 0.0, d))


def capacitor_charge(a: float, b: float, v1: float, v2: float) -> float:
    """Induced charge on the inner conductor of a cylindrical capacitor (Gauss's law)."""
    return -(v2 - v1) / (2 * K_E * np.log(b / a))


def circle(radius: float, n: int, potential: float) -> Conductor:
    """A circular conductor of ``n`` panels centred on the origin, from a simgeometry ``Circle``."""
    (conductor,) = conductors_from_shape(
        Circle(midpoint=(0.0, 0.0), radius=radius), potential, 2 * np.pi * radius / n
    )
    return conductor


def main(argv: list[str] | None = None) -> Path:
    """Run the electrode checks and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        panels_scan = (25, 50)
        grid = 30
    elif args.quick:
        panels_scan = (25, 50, 100, 200, 400)
        grid = 80
    else:
        panels_scan = (25, 50, 100, 200, 400, 800, 1600)
        grid = 200

    # 1 & 2: grounded circle with an external charge, against the three-image solution
    a, d, q0 = 1.5, 5.0, 1.0
    f_exact = three_image_force(a, d, q0)
    n_detail = panels_scan[len(panels_scan) // 2]
    elec = BoundaryIntegralElectrodes([circle(a, n_detail, 0.0)], k_e=K_E)
    elec.solve(np.array([[d], [0.0]]), [q0])
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    boundary = np.stack((a * np.cos(theta), a * np.sin(theta)))
    phi_boundary = elec.potential(boundary)
    print(
        f"grounded circle, {n_detail} panels: max boundary |phi| = "
        f"{np.abs(phi_boundary).max():.3e} (source scale "
        f"{abs(2 * K_E * q0 * np.log(d)):.3f})"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theta,
            y=phi_boundary,
            mode="lines",
            line={"color": COLORS["C0"]},
            name=f"BEM, {n_detail} panels",
        )
    )
    fig.update_xaxes(title_text="angle around the circle")
    fig.update_yaxes(title_text="phi (target: 0)")
    fig.update_layout(
        title="Boundary potential of a grounded circle with an external charge"
    )
    save_figure(fig, output / "grounded_circle")

    fig = go.Figure()
    errors = []
    for n in panels_scan:
        e = BoundaryIntegralElectrodes([circle(a, n, 0.0)], k_e=K_E)
        e.solve(np.array([[d], [0.0]]), [q0])
        _, field = e.force_on_sources()
        f_bem = q0 * field[0, 0]
        err = abs(f_bem - f_exact) / abs(f_exact)
        errors.append(err)
        print(f"  n={n:5d}: force relative error {err:.3e}")
    fig.add_trace(
        go.Scatter(
            x=panels_scan,
            y=errors,
            mode="lines+markers",
            line={"color": COLORS["C0"]},
            name="BEM force vs the three-image closed form",
        )
    )
    ref = errors[0] * (np.array(panels_scan) / panels_scan[0]) ** -2.0
    fig.add_trace(
        go.Scatter(
            x=panels_scan,
            y=ref,
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="1 / n^2",
        )
    )
    fig.update_xaxes(type="log", title_text="panels")
    fig.update_yaxes(type="log", title_text="relative error of the force on the charge")
    fig.update_layout(title="Image force on the external charge: convergence")
    save_figure(fig, output / "image_force")

    # 3 & 4: cylindrical capacitor
    a2, b2, v1, v2 = 1.0, 3.0, 1.0, 0.0
    q_exact = capacitor_charge(a2, b2, v1, v2)
    n_cap = panels_scan[len(panels_scan) // 2]
    inner = circle(a2, n_cap, v1)
    outer = circle(b2, n_cap, v2)
    cap = BoundaryIntegralElectrodes([inner, outer], k_e=K_E)
    cap.solve()
    r = np.linspace(a2 + 0.05, b2 - 0.05, 60)
    profile = cap.potential(np.stack((r, np.zeros_like(r))))
    exact_profile = v1 + (v2 - v1) * np.log(r / a2) / np.log(b2 / a2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=r,
            y=profile,
            mode="markers",
            marker={"color": COLORS["C0"]},
            name=f"BEM, {n_cap} panels per conductor",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=exact_profile,
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="exact V1 + (V2-V1) ln(r/a) / ln(b/a)",
        )
    )
    fig.update_xaxes(title_text="r")
    fig.update_yaxes(title_text="phi")
    fig.update_layout(title="Potential across the gap of a cylindrical capacitor")
    save_figure(fig, output / "capacitor_profile")

    fig = go.Figure()
    charge_errors = []
    for n in panels_scan:
        c = BoundaryIntegralElectrodes(
            [circle(a2, n, v1), circle(b2, n, v2)],
            k_e=K_E,
        )
        c.solve()
        q_bem = c.total_charge(0)
        err = abs(q_bem - q_exact) / abs(q_exact)
        charge_errors.append(err)
        print(f"  n={n:5d}: induced-charge relative error {err:.3e}")
    fig.add_trace(
        go.Scatter(
            x=panels_scan,
            y=charge_errors,
            mode="lines+markers",
            line={"color": COLORS["C0"]},
            name="inner conductor's induced charge",
        )
    )
    ref = charge_errors[0] * (np.array(panels_scan) / panels_scan[0]) ** -3.0
    fig.add_trace(
        go.Scatter(
            x=panels_scan,
            y=ref,
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="1 / n^3",
        )
    )
    fig.update_xaxes(type="log", title_text="panels per conductor")
    fig.update_yaxes(type="log", title_text="relative error of the induced charge")
    fig.update_layout(
        title="Induced charge on the capacitor's inner conductor: convergence"
    )
    save_figure(fig, output / "capacitor_charge")

    # 5: a non-convex conductor, visual + boundary-residual capstone
    corners = np.array([[0, 3, 3, 1, 1, 0], [0, 0, 1, 1, 3, 3]], dtype=float)
    corners -= corners.mean(axis=1, keepdims=True)
    l_shape = Polygon(vertices=corners.T)
    n_l = max(panels_scan[len(panels_scan) // 2], 60)
    perimeter = 12.0
    (l_conductor,) = conductors_from_shape(l_shape, 0.0, perimeter / n_l)
    vertices = l_conductor.vertices
    l_elec = BoundaryIntegralElectrodes([l_conductor], k_e=K_E)
    q_pos = np.array([[3.5], [2.0]])
    l_elec.solve(q_pos, [1.0])
    theta_l = np.linspace(0, 1, 4001, endpoint=False)
    edge_idx = (theta_l * vertices.shape[1]).astype(int) % vertices.shape[1]
    # offset away from 0 and 0.5 (panel endpoints and midpoints) to avoid exact
    # coincidence with a panel's own collocation point
    frac = (theta_l * vertices.shape[1] + 0.2137) % 1.0
    next_idx = (edge_idx + 1) % vertices.shape[1]
    boundary_l = vertices[:, edge_idx] + frac * (
        vertices[:, next_idx] - vertices[:, edge_idx]
    )
    residual = l_elec.potential(boundary_l)
    print(
        f"L-shaped conductor, {vertices.shape[1]} panels: max boundary |phi| = "
        f"{np.abs(residual).max():.3e}"
    )

    lo, hi = vertices.min(axis=1) - 1.5, vertices.max(axis=1) + 1.5
    xs = np.linspace(lo[0], hi[0], grid)
    ys = np.linspace(lo[1], hi[1], grid)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.stack((xx.ravel(), yy.ravel()))
    phi_grid = l_elec.potential(grid_pts).reshape(grid, grid)
    inside = np.asarray(l_shape.contains_points(grid_pts)).reshape(grid, grid)
    phi_grid = np.where(inside, np.nan, phi_grid)

    fig = make_subplots()
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=phi_grid,
            colorscale="RdBu",
            reversescale=True,
            contours={"coloring": "heatmap"},
            colorbar={"title": "phi"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.append(vertices[0], vertices[0, 0]),
            y=np.append(vertices[1], vertices[1, 0]),
            mode="lines",
            line={"color": "black", "width": 2},
            name="conductor",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q_pos[0],
            y=q_pos[1],
            mode="markers",
            marker={"color": "black", "size": 10, "symbol": "x"},
            name="source charge",
        )
    )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y", scaleanchor="x")
    fig.update_layout(title="Potential around a grounded L-shaped conductor")
    save_figure(fig, output / "corner_potential")

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
