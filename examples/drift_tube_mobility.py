"""An ion-mobility drift tube: drift velocity, diffusion and resolving power.

Ion-mobility spectrometry separates ions by how fast they drift through a buffer gas in
a weak field: in the low-field limit the drift velocity is ``v_d = K E`` with a mobility
``K`` set by the ion-neutral collisions, and the ion packet spreads by diffusion with
``D = K k T / q`` (Einstein's relation). Diffusion alone then limits the resolving power
``t_d / dt_FWHM`` to ``sqrt(q V / (16 ln 2 k T))``, which grows only as the square root of
the drift voltage ``V``.

The tube here is a stack of guard rings (simgeometry ``Aperture2D.slit`` plates) at
linearly falling potentials, closed by a gate plate and a grounded collector plate (thin
``Polyline`` electrodes); the panel solution of
:class:`tree_d_3d.BoundaryIntegralElectrodes` gives a nearly uniform field inside, which
is tabulated once on a grid. Ions of 100 u drift through nitrogen at 300 K and about
1 mbar, with plasmacoll's Monte Carlo collisions. For an ion polarising a neutral, the
interaction is ``-alpha q^2 / (2 r^4)``, whose momentum-transfer rate is independent of
speed: ``k_m = 1.105 k_L``, with the Langevin capture rate
``k_L = 2 pi q sqrt(alpha' / (4 pi eps0 mu))`` (McDaniel and Mason). plasmacoll's Maxwell
molecule cross section with isotropic scattering at that rate reproduces it, and then
the mobility is exactly ``K = q / (mu N k_m)``, the polarization limit (at standard
density, ``13.9 / sqrt(alpha mu)`` cm^2 / V s with ``alpha`` in A^3 and ``mu`` in u). The
field is set so the drift is a fifth of the gas thermal speed, low enough for the
low-field relations. Arrival times are recorded where each ion first crosses several
planes along the tube (simgeometry ``crossings()`` of ``Polyline`` planes), so one run
gives the packet at several drift lengths, i.e. drift voltages.

1. ``geometry``: the gate, the guard rings, the collector and the potential in the tube, with the
   recording planes.
2. ``arrival``: arrival-time distributions at the recording planes, against Gaussians of
   the predicted mean and width (``v_d = K E``, ``D = K k T / q``).
3. ``mobility``: mean arrival time against the drift length; its slope gives the drift
   velocity, compared with ``K E`` of the polarization limit. The Monte Carlo step allows
   one collision per step, which lowers the collision rate by
   ``(1 - exp(-nu dt)) / (nu dt)`` and raises the mobility by about 2 % at the
   ``nu dt = 0.04`` used here.
4. ``diffusion``: the variance of the arrival time against drift length; its slope gives
   the longitudinal diffusion coefficient, compared with Einstein's ``K k T / q``.
5. ``resolving_power``: the resolving power against the drift voltage, against the
   diffusion limit ``sqrt(q V / (16 ln 2 k T))``.

Usage::

    python examples/drift_tube_mobility.py [--output examples/output] [--quick]

``--quick`` uses fewer ions and drift voltages up to ``60 k T / q`` (about a minute); ``--smoke`` is the
tiny configuration the test suite runs. Needs Plotly and plasmacoll
(``pip install 'tree-d-3d[examples]' plasmacoll``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plasmacoll
import plotly.graph_objects as go
from plasma_plots import save_figure
from plasmacoll.constants import ATOMIC_MASS, BOLTZMANN, ELEMENTARY_CHARGE
from scipy.interpolate import RegularGridInterpolator
from simgeometry import Aperture2D, Polyline

from tree_d_3d import BoundaryIntegralElectrodes, conductors_from_shape

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}
K_E = 8.9875517923e9  # SI
EPS0 = 1.0 / (4 * np.pi * K_E)

# the ion, the gas and the collision model (SI)
ION_MASS = 100.0 * ATOMIC_MASS
GAS_MASS = 28.0 * ATOMIC_MASS  # N2
POLARIZABILITY = 1.74e-30  # alpha' of N2, m^3
TEMPERATURE = 300.0
MU = ION_MASS * GAS_MASS / (ION_MASS + GAS_MASS)
K_LANGEVIN = (
    2 * np.pi * ELEMENTARY_CHARGE * np.sqrt(POLARIZABILITY / (4 * np.pi * EPS0 * MU))
)
K_MOMENTUM = 1.105 * K_LANGEVIN
KT = BOLTZMANN * TEMPERATURE
DRIFT_OVER_THERMAL = 0.2  # v_d / sqrt(k T / M): the low-field regime

# the tube (SI): rings every PITCH from the gate at x = 0, collector at LENGTH
LENGTH = 1.0e-2
PITCH = 5.0e-4
HALF_WIDTH = 3.0e-3
START = 1.0e-3  # the ions start here, in the uniform field
PLANES_KT = (20.0, 40.0, 60.0, 80.0, 100.0)  # q V / k T from START to each plane


def build(voltage: float, spacing: float):
    """The guard rings from ``voltage`` at the gate down to the grounded collector."""
    rings = int(round(LENGTH / PITCH))
    shapes, parts = [], []
    for k in range(rings):
        x = k * PITCH
        ring = Aperture2D.slit(
            x, half_width=HALF_WIDTH, thickness=1.0e-4, length=2.0e-3
        )
        potential = voltage * (1 - x / LENGTH)
        shapes.append((ring, potential))
        parts += conductors_from_shape(ring, potential, spacing)
    collector = Polyline([(LENGTH, -HALF_WIDTH - 2e-3), (LENGTH, HALF_WIDTH + 2e-3)])
    gate = Polyline([(-PITCH, -HALF_WIDTH - 2e-3), (-PITCH, HALF_WIDTH + 2e-3)])
    for plate, potential in ((collector, 0.0), (gate, voltage * (1 + PITCH / LENGTH))):
        shapes.append((plate, potential))
        parts += conductors_from_shape(plate, potential, spacing)
    electrodes = BoundaryIntegralElectrodes(parts, k_e=K_E)
    electrodes.solve()
    return electrodes, shapes


class FieldMap:
    """The electrodes' field tabulated on a grid inside the tube (bilinear)."""

    def __init__(self, electrodes: BoundaryIntegralElectrodes, h: float) -> None:
        self.x = np.arange(0.0, LENGTH + h / 2, h)
        self.y = np.arange(-0.8 * HALF_WIDTH, 0.8 * HALF_WIDTH + h / 2, h)
        xx, yy = np.meshgrid(self.x, self.y, indexing="ij")
        phi, e = electrodes.probe(np.stack((xx.ravel(), yy.ravel())))
        self.phi = phi.reshape(xx.shape)
        kw = {"bounds_error": False, "fill_value": None}
        self._ex = RegularGridInterpolator(
            (self.x, self.y), e[0].reshape(xx.shape), **kw
        )
        self._ey = RegularGridInterpolator(
            (self.x, self.y), e[1].reshape(xx.shape), **kw
        )

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        return np.stack((self._ex(pos.T), self._ey(pos.T)))


def drift(
    fields: FieldMap,
    planes: list[Polyline],
    n: int,
    density: float,
    dt: float,
    t_max: float,
    seed: int,
):
    """Ions released at ``START`` at rest on average; first-crossing times at each plane.

    Stops when every ion has crossed the last plane, or at ``t_max``.
    """
    mcc = plasmacoll.MonteCarloCollisions(
        {"ion": ION_MASS},
        [plasmacoll.NeutralBackground("N2", density, TEMPERATURE, GAS_MASS)],
        {
            "ion": [
                plasmacoll.CollisionProcess(
                    "elastic",
                    "N2",
                    plasmacoll.CrossSection.maxwell_molecule(K_MOMENTUM, MU),
                )
            ]
        },
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    pos = np.stack((np.full(n, START), 1e-4 * rng.standard_normal(n)))
    vel = np.sqrt(KT / ION_MASS) * rng.standard_normal((3, n))
    weights = np.ones(n)
    arrival = np.full((len(planes), n), np.nan)
    t = 0.0
    accel = ELEMENTARY_CHARGE / ION_MASS
    while np.isnan(arrival[-1]).any() and t < t_max:
        old = pos.copy()
        vel[:2] += dt * accel * fields(pos)
        pos = pos + dt * vel[:2]
        result = mcc.collide("ion", pos.T, vel.T, weights, dt)
        vel = np.asarray(result.velocities).T.copy()
        t += dt
        for k, plane in enumerate(planes):
            # only steps that span the plane's x can cross it
            x_plane = plane.vertices[0, 0]
            near = (np.minimum(old[0], pos[0]) <= x_plane) & (
                np.maximum(old[0], pos[0]) >= x_plane
            )
            waiting = np.isnan(arrival[k]) & near
            if not waiting.any():
                continue
            hits = plane.crossings(old[:, waiting], pos[:, waiting])
            hit = np.asarray(hits.hit, dtype=bool)
            index = np.flatnonzero(waiting)[hit]
            arrival[k, index] = t - dt * (1 - np.asarray(hits.fraction)[hit])
    return arrival


def main(argv: list[str] | None = None) -> Path:
    """Run the drift tube and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        n, planes_kt, spacing, h = 40, (5.0, 10.0), 2e-4, 2e-4
    elif args.quick:
        n, planes_kt, spacing, h = 400, (15.0, 30.0, 45.0, 60.0), 1e-4, 1e-4
    else:
        n, planes_kt, spacing, h = 2000, PLANES_KT, 5e-5, 5e-5

    # the field: v_d a fifth of the gas thermal speed; the density sets the scale
    v_drift = DRIFT_OVER_THERMAL * np.sqrt(KT / GAS_MASS)
    v_last = (
        planes_kt[-1] * KT / ELEMENTARY_CHARGE
    )  # volts from START to the last plane
    field = PLANES_KT[-1] * KT / ELEMENTARY_CHARGE / (0.8 * LENGTH)
    density = ELEMENTARY_CHARGE * field / (MU * K_MOMENTUM * v_drift)
    mobility = ELEMENTARY_CHARGE / (MU * density * K_MOMENTUM)
    diffusion = mobility * KT / ELEMENTARY_CHARGE
    n0 = 2.6868e25  # Loschmidt
    print(
        f"N2 at {density * KT:.0f} Pa, E = {field:.0f} V/m, E/N = "
        f"{field / density * 1e21:.2f} Td; polarization-limit K0 = "
        f"{mobility * density / n0 * 1e4:.2f} cm^2/V s "
        f"(13.876 / sqrt(alpha mu) = "
        f"{13.876 / np.sqrt(1.74 * MU / ATOMIC_MASS):.2f})"
    )

    electrodes, shapes = build(field * LENGTH, spacing)
    fields = FieldMap(electrodes, h)
    axis = np.argmin(np.abs(fields.y))
    field_measured = field
    for _ in range(2):  # the field over the span the ions cross, and the planes in it
        distances = np.array(planes_kt) * KT / ELEMENTARY_CHARGE / field_measured
        span = (fields.x >= START) & (fields.x <= START + distances[-1])
        field_measured = -np.polyfit(fields.x[span], fields.phi[span, axis], 1)[0]
    distances = np.array(planes_kt) * KT / ELEMENTARY_CHARGE / field_measured
    planes = [
        Polyline([(START + d, -HALF_WIDTH), (START + d, HALF_WIDTH)]) for d in distances
    ]
    nu = density * K_MOMENTUM
    t_max = 3 * distances[-1] / v_drift
    nu_dt = 0.04
    arrival = drift(fields, planes, n, density, nu_dt / nu, t_max, seed=3)
    print(
        f"tube field {field_measured:.1f} V/m (ideal {field:.1f}), drop to {v_last:.2f} V"
    )

    mean_t = np.nanmean(arrival, axis=1)
    var_t = np.nanvar(arrival, axis=1)
    v_sim = 1.0 / np.polyfit(distances, mean_t, 1)[0]
    k_sim = v_sim / field_measured
    # sigma_x^2 = 2 D t, sigma_t = sigma_x / v_d, so d(sigma_t^2)/dx = 2 D / v_d^3
    d_sim = 0.5 * np.polyfit(distances, var_t, 1)[0] * v_sim**3
    # the null-collision step allows one collision per step: rate (1 - e^-x) / dt
    step_bias = nu_dt / (1 - np.exp(-nu_dt))
    print(
        f"mobility: simulated {k_sim * 1e4 * density / n0:.3f}, polarization limit "
        f"{mobility * 1e4 * density / n0:.3f} cm^2/V s (K0, ratio {k_sim / mobility:.3f}; "
        f"the one-collision-per-step bias alone is {step_bias:.3f})"
    )
    print(
        f"diffusion: simulated {d_sim:.4g}, Einstein K k T / q = {diffusion:.4g} m^2/s "
        f"(ratio {d_sim / diffusion:.3f})"
    )

    # 1: geometry
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=fields.x * 1e3,
            y=fields.y * 1e3,
            z=fields.phi.T,
            colorscale="Viridis",
            contours={"coloring": "heatmap"},
            colorbar={"title": "phi (V)", "x": 1.02},
            opacity=0.7,
        )
    )
    for shape, _ in shapes:
        for loop in shape.outlines():
            closed = getattr(shape, "closed", True)
            xs = np.append(loop[0], loop[0, 0]) if closed else loop[0]
            ys = np.append(loop[1], loop[1, 0]) if closed else loop[1]
            fig.add_trace(
                go.Scatter(
                    x=xs * 1e3,
                    y=ys * 1e3,
                    mode="lines",
                    fill="toself" if closed else None,
                    fillcolor="gray",
                    line={"color": "black", "width": 1},
                    showlegend=False,
                )
            )
    for plane, kt in zip(planes, planes_kt, strict=True):
        fig.add_trace(
            go.Scatter(
                x=plane.vertices[:, 0] * 1e3,
                y=plane.vertices[:, 1] * 1e3,
                mode="lines",
                line={"color": COLORS["C3"], "dash": "dot"},
                showlegend=False,
                hovertext=f"qV/kT = {kt:g}",
            )
        )
    fig.update_xaxes(title_text="x (mm)")
    fig.update_yaxes(title_text="y (mm)")
    fig.update_layout(title="Guard-ring drift tube and the recording planes (dotted)")
    save_figure(fig, output / "geometry")

    # 2: arrival-time distributions against the predicted Gaussians
    fig = go.Figure()
    for k, (d, kt) in enumerate(zip(distances, planes_kt, strict=True)):
        times = arrival[k][np.isfinite(arrival[k])] * 1e6
        counts, edges = np.histogram(times, bins=40, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        fig.add_trace(go.Bar(x=centres, y=counts, opacity=0.5, name=f"qV/kT = {kt:g}"))
        t_pred = d / (mobility * field_measured)
        s_pred = np.sqrt(2 * diffusion * t_pred) / (mobility * field_measured)
        tt = np.linspace(edges[0], edges[-1], 200)
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=np.exp(-0.5 * ((tt - t_pred * 1e6) / (s_pred * 1e6)) ** 2)
                / (np.sqrt(2 * np.pi) * s_pred * 1e6),
                mode="lines",
                line={"color": "black"},
                showlegend=k == 0,
                name="K E drift, Einstein diffusion",
            )
        )
    fig.update_layout(barmode="overlay", title="Arrival-time distributions")
    fig.update_xaxes(title_text="arrival time (us)")
    fig.update_yaxes(title_text="probability density (1/us)")
    save_figure(fig, output / "arrival")

    # 3: mobility
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distances * 1e3,
            y=mean_t * 1e6,
            mode="markers",
            marker={"color": COLORS["C0"], "size": 9},
            name=f"simulated, K0 = {k_sim * 1e4 * density / n0:.3f} cm^2/V s",
        )
    )
    dd = np.linspace(0, distances[-1], 50)
    fig.add_trace(
        go.Scatter(
            x=dd * 1e3,
            y=dd / (mobility * field_measured) * 1e6,
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name=f"polarization limit, K0 = {mobility * 1e4 * density / n0:.3f} cm^2/V s",
        )
    )
    fig.update_xaxes(title_text="drift length (mm)")
    fig.update_yaxes(title_text="mean arrival time (us)")
    fig.update_layout(title="Drift velocity: mean arrival time against drift length")
    save_figure(fig, output / "mobility")

    # 4: diffusion
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distances * 1e3,
            y=var_t * 1e12,
            mode="markers",
            marker={"color": COLORS["C0"], "size": 9},
            name=f"simulated, D = {d_sim:.3g} m^2/s",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dd * 1e3,
            y=2 * diffusion * dd / (mobility * field_measured) ** 3 * 1e12,
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name=f"Einstein, D = K k T / q = {diffusion:.3g} m^2/s",
        )
    )
    fig.update_xaxes(title_text="drift length (mm)")
    fig.update_yaxes(title_text="arrival-time variance (us^2)")
    fig.update_layout(title="Longitudinal diffusion of the ion packet")
    save_figure(fig, output / "diffusion")

    # 5: resolving power
    resolving = mean_t / (2.3548 * np.sqrt(var_t))
    kts = np.array(planes_kt)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kts * KT / ELEMENTARY_CHARGE,
            y=resolving,
            mode="markers",
            marker={"color": COLORS["C0"], "size": 9},
            name="simulated t / FWHM",
        )
    )
    vv = np.linspace(0.5 * kts[0], kts[-1], 50) * KT / ELEMENTARY_CHARGE
    fig.add_trace(
        go.Scatter(
            x=vv,
            y=np.sqrt(ELEMENTARY_CHARGE * vv / (16 * np.log(2) * KT)),
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="diffusion limit sqrt(qV / 16 ln2 kT)",
        )
    )
    fig.update_xaxes(title_text="drift voltage (V)")
    fig.update_yaxes(title_text="resolving power")
    fig.update_layout(title="Diffusion-limited resolving power")
    save_figure(fig, output / "resolving_power")
    for kt, r in zip(planes_kt, resolving, strict=True):
        limit = np.sqrt(kt / (16 * np.log(2)))
        print(
            f"  qV/kT = {kt:5.0f}: resolving power {r:.2f}, diffusion limit {limit:.2f}"
        )

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
