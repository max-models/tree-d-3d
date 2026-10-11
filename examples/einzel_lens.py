"""A slit (2D) einzel lens: why the round-aperture "focuses either way" rule weakens here.

A practical use of :class:`tree_d_3d.BoundaryIntegralElectrodes`, as opposed to
``examples/boundary_integral_electrodes.py``'s closed-form validation cases: three slit
electrodes on the beam axis, the outer two grounded and the centre one held at a lens
voltage, act on a beam of test charges -- the two-dimensional (slit, not round-aperture)
analogue of the einzel lenses used in real ion-source LEBTs. Rays are traced through the
static field by a plain RK4 push (ballistic: no self space charge, one lens voltage held
fixed), the same approximation IBSimu and `BeamOptics` make for linear-optics design.

The textbook einzel (unipotential) lens result is that a round-aperture lens focuses a
beam *regardless of the sign* of the centre electrode's voltage, with equal strength for
either sign, because the paraxial ray equation reduces to a Sturm-Liouville form with an
unconditionally positive restoring term ``(V'/V)^2`` that is also exactly even in the
voltage -- a term that comes specifically from the ``1/r`` curvature of the cylindrically
symmetric Laplacian. A 2D slit lens has no such term (a Cartesian Laplacian has no
curvature in the transverse direction), so there is no guarantee of either property. This
example checks that directly: both polarities do still focus the beam at the voltages and
beam energy tried here, but at very different distances -- decelerating the beam (for a
positive test charge, a positive centre voltage: naively repulsive) gives a *much*
stronger lens than accelerating it by the same voltage, because a decelerated particle
spends longer near the lens, where the transverse field acts on it, than an accelerated
one does. The round-aperture lens's clean sign-symmetry does not transfer to the slit.

1. ``geometry``: the three slit electrodes and their potentials.
2. ``focusing_trajectories``: rays converging to a focus for the decelerating polarity.
3. ``polarity_asymmetry``: the same rays for both polarities of equal magnitude -- both
   converge, but the decelerating one focuses several times closer than the accelerating
   one, unlike the round-aperture lens's sign-symmetric strength.
4. ``energy_conservation``: the kinetic energy of the decelerating-case ray along its
   path, from well upstream of the lens to a symmetric point downstream (the two
   electrodes are not individually neutral, so the solved field has a net monopole and
   decays only as ``1/r``, the usual 2D electrostatics catch, so "upstream" and
   "downstream" need to be equidistant from the lens, not merely "before" and "after" it,
   for the potential -- and so the kinetic energy -- to return to its starting value).
5. ``focal_length``: the focus distance against the centre voltage, for the decelerating
   polarity only (the accelerating one needs several times the voltage-scan range's beam
   energy before it is this strong, which starts to overlap the beam's own KE): shorter
   (stronger focusing) at larger voltage, up to where the beam no longer has enough energy
   to cross the centre electrode's potential barrier at all.

Usage::

    python examples/einzel_lens.py [--output examples/output] [--quick]

``--quick`` uses fewer panels and voltage-scan points (under a minute instead of a few);
``--smoke`` is the tiny configuration the test suite runs. Needs Plotly
(``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure

from tree_d_3d import BoundaryIntegralElectrodes, Conductor

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}
K_E = 1.0  # code units

HALF_APERTURE = 0.5
GAP = 3.0
ELECTRODE_X = (0.0, GAP, 2 * GAP)
PLATE_LENGTH = 15.0
VX0 = 5.0  # beam speed (q = m = 1, so KE0 = 12.5; stay well below that for V_lens)
START_X = -10.0  # upstream of the first electrode, where the field is small


def slit_electrode(
    x: float, potential: float, n: int, grade: float = 2.0
) -> list[Conductor]:
    """Two plate conductors (above and below the aperture) forming a slit at ``x``.

    Panels are graded (denser near the aperture tip, where the field varies fastest) by
    ``t**grade`` spacing along each plate.
    """
    t = np.linspace(0, 1, n + 1) ** grade
    y = HALF_APERTURE + PLATE_LENGTH * t
    upper = np.stack((np.full(n + 1, x), y))
    lower = np.stack((np.full(n + 1, x), -y))
    return [
        Conductor(upper, potential, closed=False),
        Conductor(lower, potential, closed=False),
    ]


def build_lens(v_lens: float, n: int) -> BoundaryIntegralElectrodes:
    """The three-slit lens: outer electrodes grounded, centre one at ``v_lens``."""
    conductors = (
        slit_electrode(ELECTRODE_X[0], 0.0, n)
        + slit_electrode(ELECTRODE_X[1], v_lens, n)
        + slit_electrode(ELECTRODE_X[2], 0.0, n)
    )
    lens = BoundaryIntegralElectrodes(conductors, k_e=K_E)
    lens.solve()  # no free charges: the test rays are ballistic probes, not sources
    return lens


def trace(
    lens: BoundaryIntegralElectrodes, y0: float, vx0: float, x_final: float, dt: float
):
    """RK4 push of a unit test charge (q = m = 1) from ``(0, y0)`` with velocity ``(vx0, 0)``.

    Starts at ``x = START_X``, well upstream of the first electrode (not at ``x = 0``,
    which sits inside the aperture where the potential is not the asymptotic upstream
    value, so energy conservation against the entry KE would otherwise not hold). Stops
    early if ``x`` reaches ``x_final``. Returns the position and velocity history, each
    ``(2, steps)``.
    """
    max_steps = int(np.ceil(2 * (x_final - START_X) / (vx0 * dt)))
    pos = np.array([START_X, y0])
    vel = np.array([vx0, 0.0])
    xs, vs = [pos.copy()], [vel.copy()]

    def accel(p):
        _, field = lens.probe(p.reshape(2, 1))
        return field[:, 0]  # q = m = 1

    for _ in range(max_steps):
        k1v = accel(pos)
        k1x = vel
        k2v = accel(pos + 0.5 * dt * k1x)
        k2x = vel + 0.5 * dt * k1v
        k3v = accel(pos + 0.5 * dt * k2x)
        k3x = vel + 0.5 * dt * k2v
        k4v = accel(pos + dt * k3x)
        k4x = vel + dt * k3v
        pos = pos + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        vel = vel + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        xs.append(pos.copy())
        vs.append(vel.copy())
        if pos[0] > x_final:
            break
    return np.array(xs).T, np.array(vs).T


def focus_crossing(xs: np.ndarray) -> float:
    """The x where the trajectory's y crosses zero after the lens (linear interpolation)."""
    x, y = xs
    past_lens = x > ELECTRODE_X[-1]
    sign_change = np.where(past_lens[:-1] & (np.sign(y[:-1]) != np.sign(y[1:])))[0]
    if not len(sign_change):
        return np.nan
    i = sign_change[0]
    frac = y[i] / (y[i] - y[i + 1])
    return x[i] + frac * (x[i + 1] - x[i])


def main(argv: list[str] | None = None) -> Path:
    """Run the einzel-lens checks and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        n_panels, scan_voltages, dt = 10, (3.0, 6.0, 9.0), 0.02
    elif args.quick:
        n_panels, scan_voltages, dt = 25, np.linspace(3, 9, 7), 0.005
    else:
        n_panels, scan_voltages, dt = 50, np.linspace(3, 9, 13), 0.002
    v_decel, v_accel = 6.0, -6.0
    x_final = 45.0
    x_final_energy = (
        2 * ELECTRODE_X[1] - START_X
    )  # symmetric about the centre electrode
    y0s = (0.05, 0.1, 0.15, 0.2, 0.25)

    # 1. geometry
    lens = build_lens(v_decel, n_panels)
    fig = go.Figure()
    for conductor in lens.conductors:
        v = conductor.vertices
        fig.add_trace(
            go.Scatter(
                x=v[0],
                y=v[1],
                mode="lines",
                line={"color": COLORS["C0"], "width": 3},
                showlegend=False,
            )
        )
    for x, v in zip(ELECTRODE_X, (0.0, v_decel, 0.0), strict=True):
        fig.add_annotation(
            x=x,
            y=3.5,
            text=f"V = {v:g}",
            showarrow=False,
        )
    fig.update_xaxes(title_text="x", range=[ELECTRODE_X[0] - 3, ELECTRODE_X[-1] + 3])
    fig.update_yaxes(title_text="y", range=[-4, 4])
    fig.update_layout(title=f"The three-slit lens (centre at V = {v_decel:g} shown)")
    save_figure(fig, output / "geometry")

    # 2. focusing trajectories, the decelerating (stronger) polarity
    fig = go.Figure()
    crossings = []
    for y0 in y0s:
        xs, _ = trace(lens, y0, VX0, x_final, dt)
        crossings.append(focus_crossing(xs))
        fig.add_trace(
            go.Scatter(
                x=xs[0],
                y=xs[1],
                mode="lines",
                line={"color": COLORS["C0"]},
                name=f"y0 = {y0}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs[0],
                y=-xs[1],
                mode="lines",
                line={"color": COLORS["C0"]},
                showlegend=False,
            )
        )
    print(f"V_lens = {v_decel:g}: focus crossings at x = {np.round(crossings, 3)}")
    for x in ELECTRODE_X:
        fig.add_vline(x=x, line={"color": "gray", "dash": "dot"})
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    fig.update_layout(
        title=f"Rays focused by the lens (centre electrode at V = {v_decel:g})"
    )
    save_figure(fig, output / "focusing_trajectories")

    # 3. polarity asymmetry: both focus, but at very different distances
    fig = go.Figure()
    for v_lens, color, dash, label in (
        (v_decel, COLORS["C0"], "solid", "decelerating, strong"),
        (v_accel, COLORS["C3"], "dash", "accelerating, weak"),
    ):
        lens_v = build_lens(v_lens, n_panels)
        crossings_v = []
        for i, y0 in enumerate(y0s):
            xs, _ = trace(lens_v, y0, VX0, x_final, dt)
            crossings_v.append(focus_crossing(xs))
            fig.add_trace(
                go.Scatter(
                    x=xs[0],
                    y=xs[1],
                    mode="lines",
                    line={"color": color, "dash": dash},
                    name=f"V = {v_lens:g} ({label})" if i == 0 else None,
                    showlegend=i == 0,
                )
            )
        print(f"V_lens = {v_lens:g}: focus crossings at x = {np.round(crossings_v, 3)}")
    for x in ELECTRODE_X:
        fig.add_vline(x=x, line={"color": "gray", "dash": "dot"})
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    fig.update_layout(
        title="Equal-magnitude voltages focus at very different distances"
    )
    save_figure(fig, output / "polarity_asymmetry")

    # 4. energy conservation: symmetric upstream/downstream points (see the module
    # docstring for why "before" and "after" the lens is not quite enough in 2D)
    xs, vs = trace(lens, y0s[2], VX0, x_final_energy, dt)
    ke = 0.5 * np.sum(vs**2, axis=0)
    ke0 = 0.5 * VX0**2
    rel_drift = abs(ke[-1] - ke0) / ke0
    print(f"energy conservation: final/initial KE - 1 = {rel_drift:.3e}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs[0],
            y=ke,
            mode="lines",
            line={"color": COLORS["C0"]},
            name="kinetic energy",
        )
    )
    fig.add_hline(
        y=ke0, line={"color": "black", "dash": "dash"}, annotation_text="initial KE"
    )
    for x in ELECTRODE_X:
        fig.add_vline(x=x, line={"color": "gray", "dash": "dot"})
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="kinetic energy")
    fig.update_layout(
        title="Kinetic energy along a ray: equal at equal distances from the lens"
    )
    save_figure(fig, output / "energy_conservation")

    # 5. focal length vs centre voltage, converging polarity only
    fig = go.Figure()
    focal = []
    for v in scan_voltages:
        lens_v = build_lens(float(v), n_panels)
        xs, _ = trace(lens_v, y0s[1], VX0, x_final, dt)
        crossing = focus_crossing(xs)
        focal.append(crossing - ELECTRODE_X[1] if np.isfinite(crossing) else np.nan)
        print(
            f"  V_lens = {v:6.2f}: focus at "
            f"{crossing - ELECTRODE_X[1] if np.isfinite(crossing) else float('nan'):.3f} "
            "past the centre"
        )
    fig.add_trace(
        go.Scatter(
            x=scan_voltages,
            y=focal,
            mode="lines+markers",
            line={"color": COLORS["C0"]},
            name="focus distance past the centre electrode",
        )
    )
    fig.update_xaxes(title_text="centre electrode voltage (converging polarity)")
    fig.update_yaxes(title_text="focal distance")
    fig.update_layout(title="Focal distance against the lens voltage")
    save_figure(fig, output / "focal_length")

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
