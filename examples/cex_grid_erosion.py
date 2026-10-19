"""Charge-exchange ions in an ion-thruster grid set: where they land and what they erode.

A gridded ion thruster accelerates xenon ions through a stack of slit electrodes, and the
grids wear out from the inside: unionised propellant leaks through the apertures, a fast
beam ion meets a slow xenon atom, and resonant charge exchange (``Xe+ + Xe -> Xe + Xe+``)
leaves a fast neutral and a slow ion. The slow ion is born at the local potential and falls
into the most negative electrode nearby, the accelerator grid, which it sputters. This is
the classic lifetime limit of gridded thrusters (the "pits and grooves" worn into the accel
grid's downstream face, and the barrel erosion that widens its apertures).

The grid set is a three-grid slit geometry with the thicknesses and gaps of a 30 cm
NSTAR-class thruster: a screen grid at +1100 V, an accel grid at -250 V and a grounded decel
grid, each 0.38 mm thick, with 0.66 / 1.0 mm gaps and half-apertures of 0.95, 0.45 and
0.9 mm. The accel slit is narrower than NSTAR's round accel holes (0.57 mm radius) because a
slit lets the screen potential through much more than a round hole: at 0.57 mm and -180 V
the potential on the axis of the accel aperture would be positive (about +16 V), and the
grid would no longer stop beam-plasma electrons from streaming back. Each grid
is a simgeometry ``Aperture2D.slit``, panelled by :func:`tree_d_3d.conductors_from_shape`
with panels graded toward the aperture tips. Two wide plates close the problem: one upstream
at the discharge-plasma potential (1125 V) and a grounded one far downstream, standing in
for the beam plasma; without them the net charge of the isolated grids would set the far
field. :class:`tree_d_3d.BoundaryIntegralElectrodes` solves for the vacuum field once.

Model and assumptions, all in a 2D slice (per unit length of slit), lengths in mm, energies
in eV and code units with ``q = m = 1`` (so a kinetic energy is ``v^2 / 2``):

* Primary ions start 1 mm upstream of the screen grid with the energy they gain from the
  plasma potential and are pushed ballistically (no beam space charge: the potential
  saddle and the trajectories are the vacuum ones; a high-perveance beamlet would need the
  tree-code space charge of ``examples/ion_extraction.py``). Each primary carries
  ``I' / e / N`` ions per second per metre of slit, with ``I' = 0.13 A/m``, the beamlet
  current density of an NSTAR-class grid (about 70 A/m^2) times the screen aperture.
* The neutral xenon density ``n = p / (k_B T)`` (T = 300 K) is uniform. Real neutral
  efflux peaks at the grids and thins out downstream, so the absolute rates here are
  indicative; the scaling and the impact pattern are the point.
* Resonant charge exchange uses the fit of Miller et al., J. Appl. Phys. 91, 984 (2002),
  ``sigma = (87.3 - 13.6 log10(E / eV)) * 1e-20 m^2`` at the primary's lab energy ``E``,
  tabulated as a plasmacoll ``CrossSection`` (used here from 1 eV to 1.3 keV). Along each
  primary step of length ``ds`` a charge exchange happens with probability
  ``n sigma(E) ds``; the primary current is attenuated by the same optical depth (the fast
  neutral leaves). Charge-exchange events are rare (``n sigma L`` is about 1e-3 at the
  reference pressure), so the sampling boosts the probability by a constant factor and
  gives each slow ion the inverse as its weight.
* Each slow ion is born with the thermal velocity of the gas, pushed ballistically in the
  electrode field (velocity Verlet with a step limited to 0.02 mm), and absorbed at the
  first boundary crossing of a step: ``shape.crossings(start, end)`` tests the whole step,
  so no ion jumps through a plate, and gives the impact point and the surface normal, from
  which the impact energy and angle of incidence follow. Slow ions do not collide again.
* Sputtering uses the Bohdansky formula, Nucl. Instrum. Methods B 2, 587 (1984),
  ``Y(E) = Q s_n(eps) (1 - (E_th / E)^(2/3)) (1 - E_th / E)^2``, with the Kr-C reduced
  nuclear stopping ``s_n`` (Eckstein) and the Lindhard screening length, for xenon on
  molybdenum at normal incidence. The threshold ``E_th = 50 eV`` and ``Q = 4`` are
  illustrative values that put the yield near one atom per ion at 1 keV and the threshold
  near those of published Xe -> Mo fits; replace them with a vetted fit before using the
  erosion numbers quantitatively. Angular effects are left out (the angles are reported).

The production of slow ions is checked against the expected number, the sum over primary
steps of the boosted probability ``B n sigma(E) ds exp(-tau)``, within Poisson error.

1. ``geometry``: the potential of the three grids and the boundary plates, with the
   negative saddle in the accel aperture (about -100 V on the axis) that keeps beam-plasma
   electrons out.
2. ``trajectories``: the primary beamlet and a sample of charge-exchange ions, coloured by
   where they end: most fall back onto the accel grid, from the gap and from downstream.
3. ``impact_map``: the flux of charge-exchange ions onto each face of the accel grid
   (upstream face, aperture barrel, downstream face) against position along the face.
4. ``impact_distributions``: impact energy and angle of incidence of the ions that hit the
   accel grid, weighted by flux.
5. ``erosion``: the estimated erosion rate of each face of the accel grid, in micrometres
   per thousand hours of operation (an estimate: see the sputter-yield caveat above). The
   barrel wears fastest, at its upstream edge, which widens the aperture.
6. ``pressure_scan``: charge-exchange current to the accel grid and the peak erosion rate
   against neutral pressure: linear at low pressure, falling below linear only once the
   beam's optical depth ``n sigma L`` approaches one (by then slow ions would collide again
   too, which the model leaves out).

Usage::

    python examples/cex_grid_erosion.py [--output examples/output] [--quick]

``--quick`` uses fewer primaries, slow ions and scan points (well under a minute);
``--smoke`` is the tiny configuration the test suite runs. Needs Plotly and plasmacoll
(``pip install 'tree-d-3d[examples]' plasmacoll``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from plasmacoll import CrossSection
from plotly.subplots import make_subplots
from simgeometry import Aperture2D, Rectangle

from tree_d_3d import BoundaryIntegralElectrodes, conductors_from_shape

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}
K_E = 1.0  # code units: potentials in V, lengths in mm, fields in V/mm

# -- the grid set (mm, V) ---------------------------------------------------------------------
THICKNESS = 0.38
PLATE_LENGTH = 6.0
PLASMA_POTENTIAL = 1125.0
#: name -> (axial centre, half-aperture, potential)
GRIDS = {
    "screen": (0.0, 0.95, 1100.0),
    "accel": (1.04, 0.45, -250.0),
    "decel": (2.42, 0.90, 0.0),
}
#: the boundary plates: name -> (axial centre, potential)
WALLS = {"upstream plasma": (-1.5, PLASMA_POTENTIAL), "downstream ground": (9.0, 0.0)}
WALL_HALF_HEIGHT = 9.0
X_START = -1.0  # primaries start here, 1 mm upstream of the screen grid
X_END = 8.0  # and are followed to here

# -- physics ----------------------------------------------------------------------------------
E_CHARGE = 1.602176634e-19  # C
K_B = 1.380649e-23  # J/K
GAS_TEMPERATURE = 300.0  # K
THERMAL_EV = K_B * GAS_TEMPERATURE / E_CHARGE
BEAMLET_CURRENT = 0.13  # A per metre of slit
REFERENCE_PRESSURE = 4e-3  # Pa, about 1e18 m^-3 of neutral xenon
MO_DENSITY = 6.42e28  # molybdenum atoms per m^3
HOURS = 3600.0
STEP_LENGTH = 0.02  # mm, the largest distance a particle moves in one step


def miller_cross_section(energy: np.ndarray) -> np.ndarray:
    """Xe+ + Xe resonant charge exchange in m^2 (Miller et al. 2002), ``energy`` in eV."""
    return (87.3 - 13.6 * np.log10(energy)) * 1e-20


CX = CrossSection.from_function(
    miller_cross_section,
    np.geomspace(1.0, 1300.0, 200),
    label="Xe+ + Xe resonant charge exchange (Miller et al. 2002)",
)


def sigma_cx(energy: np.ndarray) -> np.ndarray:
    """The tabulated cross section (m^2) at ``energy`` (eV), as a NumPy array."""
    return np.asarray(CX(np.clip(energy, 1.0, None)), dtype=float)


def sputter_yield(energy: np.ndarray, e_th: float = 50.0, q: float = 4.0) -> np.ndarray:
    """Bohdansky yield of molybdenum under xenon at normal incidence (atoms per ion).

    ``e_th`` and ``q`` are illustrative (see the module docstring). Reduced energy with the
    Lindhard screening length and the Kr-C nuclear stopping, as in Eckstein's fits.
    """
    z1, m1, z2, m2 = 54.0, 131.29, 42.0, 95.95
    a_l = 0.04685 / np.sqrt(z1 ** (2 / 3) + z2 ** (2 / 3))  # nm
    eps = energy * m2 / (m1 + m2) * a_l / (z1 * z2 * 1.44)
    eps = np.maximum(eps, 1e-12)
    s_n = (
        0.5
        * np.log1p(1.2288 * eps)
        / (eps + 0.1728 * np.sqrt(eps) + 0.008 * eps**0.1504)
    )
    ratio = np.clip(e_th / np.maximum(energy, 1e-12), 0.0, 1.0)
    return q * s_n * (1 - ratio ** (2 / 3)) * (1 - ratio) ** 2


def panel_length(half_aperture: float):
    """Panels 0.03 mm long at an aperture tip, growing away from it, at most 0.3 mm."""

    def length(points: np.ndarray) -> np.ndarray:
        distance = np.maximum(np.abs(points[1]) - half_aperture, 0.0)
        return np.minimum(0.03 + 0.12 * distance, 0.3)

    return length


def wall_panel_length(points: np.ndarray) -> np.ndarray:
    """Panels on the boundary plates: short near the axis, at most 0.25 mm."""
    return np.minimum(0.1 + 0.1 * np.abs(points[1]), 0.25)


def build_geometry():
    """The grid and wall shapes, by name, and the solved electrodes."""
    shapes = {}
    conductors = []
    for name, (x, half_aperture, potential) in GRIDS.items():
        shapes[name] = Aperture2D.slit(x, half_aperture, THICKNESS, PLATE_LENGTH)
        conductors += conductors_from_shape(
            shapes[name], potential, panel_length(half_aperture)
        )
    for name, (x, potential) in WALLS.items():
        shapes[name] = Rectangle((x, 0.0), (0.3, 2 * WALL_HALF_HEIGHT))
        conductors += conductors_from_shape(shapes[name], potential, wall_panel_length)
    electrodes = BoundaryIntegralElectrodes(conductors, k_e=K_E)
    electrodes.solve()
    return shapes, electrodes


@dataclass
class Fate:
    """Where each pushed particle ended."""

    where: np.ndarray  # name of the shape hit, "escaped" or "in flight"
    point: np.ndarray  # (2, N) end or impact point
    velocity: np.ndarray  # (2, N) velocity at the end or impact
    normal: np.ndarray  # (2, N) outward surface normal at an impact (NaN otherwise)


def push(electrodes, shapes, pos, vel, *, max_steps, record=None, on_step=None):
    """Velocity-Verlet push of unit charges (q = m = 1) until each hits a shape or escapes.

    Each particle has its own step, limited so that it moves at most ``STEP_LENGTH``; every
    step is tested against every shape with ``crossings``. ``record`` (indices) keeps the
    paths of those particles; ``on_step(start, end, alive)`` sees each step.
    """
    n = pos.shape[1]
    pos, vel = pos.copy(), vel.copy()
    where = np.full(n, "in flight", dtype=object)
    point = np.full((2, n), np.nan)
    v_end = np.full((2, n), np.nan)
    normal = np.full((2, n), np.nan)
    alive = np.ones(n, dtype=bool)
    paths = {int(i): [pos[:, i].copy()] for i in (record if record is not None else [])}
    acc = electrodes.field(pos)
    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        p0, v0, a0 = pos[:, idx], vel[:, idx], acc[:, idx]
        speed = np.hypot(*v0)
        accel = np.hypot(*a0)
        dt = np.minimum(
            STEP_LENGTH / np.maximum(speed, 1e-12),
            np.sqrt(2 * STEP_LENGTH / np.maximum(accel, 1e-12)),
        )
        p1 = p0 + v0 * dt + 0.5 * a0 * dt**2
        a1 = electrodes.field(p1)
        v1 = v0 + 0.5 * (a0 + a1) * dt
        first = np.full(idx.size, np.inf)
        hit_name = np.full(idx.size, "", dtype=object)
        hit_point = np.full((2, idx.size), np.nan)
        hit_normal = np.full((2, idx.size), np.nan)
        for name, shape in shapes.items():
            c = shape.crossings(p0, p1)
            frac = np.where(np.asarray(c.hit), np.asarray(c.fraction), np.inf)
            closer = frac < first
            first[closer] = frac[closer]
            hit_name[closer] = name
            hit_point[:, closer] = np.asarray(c.point)[:, closer]
            hit_normal[:, closer] = np.asarray(c.normal)[:, closer]
        hit = np.isfinite(first)
        if on_step is not None:
            on_step(idx, p0, np.where(hit, hit_point, p1), v0, v1)
        out = ~hit & ((p1[0] > X_END) | (p1[0] < X_START - 0.4))
        out |= ~hit & (np.abs(p1[1]) > WALL_HALF_HEIGHT)
        f = np.where(hit, first, 1.0)
        v_hit = v0 + f * (v1 - v0)
        done = idx[hit | out]
        where[idx[hit]] = hit_name[hit]
        where[idx[out]] = "escaped"
        point[:, done] = np.where(hit, hit_point, p1)[:, hit | out]
        v_end[:, done] = v_hit[:, hit | out]
        normal[:, idx[hit]] = hit_normal[:, hit]
        alive[done] = False
        pos[:, idx], vel[:, idx], acc[:, idx] = p1, v1, a1
        for i, j in zip(idx, range(idx.size)):
            if int(i) in paths:
                paths[int(i)].append(
                    hit_point[:, j]
                    if hit[j]
                    else (p1[:, j] if not out[j] else p1[:, j])
                )
    point[:, alive] = pos[:, alive]
    v_end[:, alive] = vel[:, alive]
    return Fate(where, point, v_end, normal), {
        i: np.array(p).T for i, p in paths.items()
    }


@dataclass
class ChargeExchange:
    """Slow ions born along the primary beam, with what the pressure scan needs."""

    birth: np.ndarray  # (2, M) birth points
    tau_per_density: np.ndarray  # (M,) primary optical depth per unit density at birth
    boost: float  # sampling boost: each slow ion stands for 1 / boost events
    expected: float  # expected number of sampled events (validation)
    n_primary: int


def sample_charge_exchange(electrodes, shapes, n_primary, target, rng, density):
    """Push the primaries and draw slow-ion births along them at ``density`` (m^-3)."""
    y0 = np.linspace(-0.8, 0.8, n_primary) * GRIDS["screen"][1]
    pos = np.stack((np.full(n_primary, X_START), y0))
    phi, _ = electrodes.probe(pos)
    speed = np.sqrt(2 * (PLASMA_POTENTIAL - phi))
    angle = rng.normal(0.0, np.radians(2.0), n_primary)
    vel = speed * np.stack((np.cos(angle), np.sin(angle)))

    segments = []  # (start, end, energy, length) per primary step

    def collect(idx, p0, p1, v0, v1):
        energy = 0.25 * (np.sum(v0**2, axis=0) + np.sum(v1**2, axis=0))
        segments.append((idx, p0, p1, energy, np.hypot(*(p1 - p0))))

    fate, paths = push(
        electrodes,
        shapes,
        pos,
        vel,
        max_steps=4000,
        record=range(0, n_primary, max(1, n_primary // 12)),
        on_step=collect,
    )
    # the optical depth per unit density accumulates along each primary
    idx = np.concatenate([s[0] for s in segments])
    p0 = np.concatenate([s[1] for s in segments], axis=1)
    p1 = np.concatenate([s[2] for s in segments], axis=1)
    energy = np.concatenate([s[3] for s in segments])
    length_m = np.concatenate([s[4] for s in segments]) * 1e-3
    depth = sigma_cx(energy) * length_m  # per unit density, per step
    order = np.argsort(idx, kind="stable")
    cumulative = np.zeros_like(depth)
    for i in np.unique(idx):
        own = order[idx[order] == i]
        cumulative[own] = np.cumsum(depth[own]) - depth[own]
    weight = density * depth * np.exp(-density * cumulative)
    boost = target / max(weight.sum(), 1e-300)
    count = rng.poisson(boost * weight)
    step = np.repeat(np.arange(count.size), count)
    t = rng.uniform(0.0, 1.0, step.size)
    birth = p0[:, step] + t * (p1 - p0)[:, step]
    cx = ChargeExchange(
        birth=birth,
        tau_per_density=cumulative[step] + t * depth[step],
        boost=boost,
        expected=float(boost * weight.sum()),
        n_primary=n_primary,
    )
    per_primary = weight.sum() / n_primary
    return fate, paths, cx, per_primary


def slow_ion_weights(cx: ChargeExchange, density: float, reference: float):
    """Slow ions per second per metre of slit that each sampled ion stands for at ``density``.

    The births were drawn at ``reference``; another density rescales the probability
    linearly and the beam attenuation exponentially.
    """
    primaries_per_second = BEAMLET_CURRENT / E_CHARGE  # per metre of slit
    base = primaries_per_second / cx.n_primary / cx.boost
    reweight = (density / reference) * np.exp(
        -(density - reference) * cx.tau_per_density
    )
    return base * reweight


def classify_faces(point, normal):
    """The face of a slit plate each impact is on: upstream, downstream, barrel or end."""
    face = np.full(point.shape[1], "end", dtype=object)
    face[normal[0] < -0.5] = "upstream face"
    face[normal[0] > 0.5] = "downstream face"
    barrel = (np.abs(normal[1]) > 0.5) & (normal[1] * point[1] < 0)
    face[barrel] = "barrel"
    return face


def main(argv: list[str] | None = None) -> Path:
    """Run the charge-exchange study and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        n_primary, target, pressures, grid = 12, 150, np.geomspace(1e-3, 1.0, 4), 50
    elif args.quick:
        n_primary, target, pressures, grid = 40, 1500, np.geomspace(1e-4, 3.0, 9), 120
    else:
        n_primary, target, pressures, grid = 120, 8000, np.geomspace(1e-4, 3.0, 13), 220
    rng = np.random.default_rng(7)
    density = REFERENCE_PRESSURE / (K_B * GAS_TEMPERATURE)

    shapes, electrodes = build_geometry()
    print(f"{electrodes.num_panels} panels; neutral density {density:.2e} m^-3")

    # 1: geometry and potential
    xs = np.linspace(-1.3, 8.5, grid)
    ys = np.linspace(-4.0, 4.0, int(grid * 0.8))
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack((xx.ravel(), yy.ravel()))
    phi = electrodes.potential(pts).reshape(xx.shape)
    solid = np.zeros(pts.shape[1], dtype=bool)
    for shape in shapes.values():
        solid |= np.asarray(shape.contains_points(pts), dtype=bool)
    phi = np.where(solid.reshape(xx.shape), np.nan, phi)
    fig = go.Figure(
        go.Contour(
            x=xs,
            y=ys,
            z=phi,
            colorscale="RdBu",
            reversescale=True,
            contours={"start": -250, "end": 1100, "size": 50},
            colorbar={"title": "phi (V)"},
        )
    )
    add_plates(fig, shapes)
    fig.update_xaxes(title_text="x (mm)", range=[-1.3, 8.5])
    fig.update_yaxes(title_text="y (mm)", range=[-4.0, 4.0], scaleanchor="x")
    fig.update_layout(title="Screen (+1100 V), accel (-250 V) and decel (0 V) grids")
    save_figure(fig, output / "geometry")
    saddle_phi = float(electrodes.potential(np.array([[GRIDS["accel"][0]], [0.0]]))[0])
    print(f"potential on axis in the accel aperture: {saddle_phi:.1f} V")

    # 2: primaries and slow ions
    fate_p, primary_paths, cx, per_primary = sample_charge_exchange(
        electrodes, shapes, n_primary, target, rng, density
    )
    n_cx = cx.birth.shape[1]
    sigma_full = float(sigma_cx(np.array([PLASMA_POTENTIAL]))[0])
    print(
        f"primaries: {np.sum(fate_p.where == 'escaped')} of {n_primary} leave the grids "
        f"({', '.join(f'{k}: {np.sum(fate_p.where == k)}' for k in GRIDS)} hit a grid)"
    )
    print(
        f"charge exchange per primary: {per_primary:.3e} (rough n sigma(E_beam) L: "
        f"{density * sigma_full * (X_END - X_START) * 1e-3:.3e})"
    )
    sd = np.sqrt(cx.expected)
    print(
        f"validation: {n_cx} slow ions sampled, {cx.expected:.1f} +- {sd:.1f} expected "
        f"({(n_cx - cx.expected) / sd:+.2f} sigma)"
    )

    energy_birth = THERMAL_EV
    theta = rng.uniform(0, 2 * np.pi, n_cx)
    speed = np.sqrt(2 * rng.exponential(energy_birth, n_cx))
    vel = speed * np.stack((np.cos(theta), np.sin(theta)))
    shown = rng.choice(n_cx, size=min(n_cx, 80), replace=False)
    fate, cx_paths = push(
        electrodes, shapes, cx.birth, vel, max_steps=6000, record=shown
    )
    weights = slow_ion_weights(cx, density, density)
    total = weights.sum()
    summary = {
        name: weights[fate.where == name].sum() / total
        for name in [*GRIDS, *WALLS, "escaped", "in flight"]
    }
    print(
        "slow ions end on: "
        + ", ".join(f"{k} {100 * v:.1f}%" for k, v in summary.items() if v > 0)
    )

    fig = go.Figure()
    add_plates(fig, shapes)
    for i, path in primary_paths.items():
        fig.add_trace(
            go.Scatter(
                x=path[0],
                y=path[1],
                mode="lines",
                line={"color": "gray", "width": 1},
                name="primary ions",
                showlegend=i == min(primary_paths),
            )
        )
    end_colors = {"accel": COLORS["C3"], "decel": COLORS["C2"], "screen": COLORS["C1"]}
    legend_done = set()
    for i, path in cx_paths.items():
        end = str(fate.where[i])
        color = end_colors.get(end, COLORS["C0"])
        label = f"slow ion -> {end}"
        fig.add_trace(
            go.Scatter(
                x=path[0],
                y=path[1],
                mode="lines",
                line={"color": color, "width": 1},
                name=label,
                showlegend=label not in legend_done,
            )
        )
        legend_done.add(label)
    fig.update_xaxes(title_text="x (mm)", range=[-1.3, 8.5])
    fig.update_yaxes(title_text="y (mm)", range=[-4.0, 4.0], scaleanchor="x")
    fig.update_layout(
        title="Primary beamlet and charge-exchange ions, by where they end"
    )
    save_figure(fig, output / "trajectories")

    # 3-5: impacts on the accel grid
    on_accel = fate.where == "accel"
    face = classify_faces(fate.point[:, on_accel], fate.normal[:, on_accel])
    w_accel = weights[on_accel]
    v_imp = fate.velocity[:, on_accel]
    energy = 0.5 * np.sum(v_imp**2, axis=0)
    n_out = fate.normal[:, on_accel]
    cos_angle = np.clip(-np.sum(v_imp * n_out, axis=0) / np.hypot(*v_imp), -1, 1)
    angle = np.degrees(np.arccos(cos_angle))
    p = fate.point[:, on_accel]
    x_a, r_a, _ = GRIDS["accel"]
    faces = {
        # face -> (coordinate along it, bins in mm)
        "upstream face": (np.abs(p[1]), np.linspace(r_a, r_a + 3.0, 31)),
        "barrel": (p[0], np.linspace(x_a - THICKNESS / 2, x_a + THICKNESS / 2, 11)),
        "downstream face": (np.abs(p[1]), np.linspace(r_a, r_a + 3.0, 31)),
    }
    current_density = {}
    erosion = {}
    yield_ = sputter_yield(energy)
    for name, (coord, bins) in faces.items():
        own = face == name
        # both plates (y > 0 and y < 0) share a coordinate: halve to get one plate's flux
        flux, _ = np.histogram(coord[own], bins, weights=w_accel[own] / 2)
        eroded, _ = np.histogram(coord[own], bins, weights=(w_accel * yield_)[own] / 2)
        width_m = np.diff(bins) * 1e-3
        current_density[name] = (bins, flux / width_m * E_CHARGE)  # A/m^2
        erosion[name] = (bins, eroded / width_m / MO_DENSITY * 1e6 * 1000 * HOURS)
    for name in faces:
        share = w_accel[face == name].sum() / max(total, 1e-300)
        print(f"  accel {name}: {100 * share:.1f}% of the slow ions")
    cx_current = w_accel.sum() * E_CHARGE / BEAMLET_CURRENT
    print(
        f"charge-exchange current to the accel grid: {100 * cx_current:.3f}% of the beam"
    )

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(faces))
    for col, (name, (bins, j)) in enumerate(current_density.items(), start=1):
        centres = 0.5 * (bins[1:] + bins[:-1])
        fig.add_trace(
            go.Bar(x=centres, y=j, marker_color=COLORS["C3"], showlegend=False),
            row=1,
            col=col,
        )
        fig.update_xaxes(
            title_text="x (mm)" if name == "barrel" else "distance from axis (mm)",
            row=1,
            col=col,
        )
    fig.update_yaxes(title_text="charge-exchange current density (A/m^2)", row=1, col=1)
    fig.update_layout(title="Where the slow ions hit the accel grid")
    save_figure(fig, output / "impact_map")

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("impact energy", "angle of incidence")
    )
    for col, (values, bins, label) in enumerate(
        (
            (energy, np.linspace(0, 400, 41), "energy (eV)"),
            (angle, np.linspace(0, 90, 31), "angle from the normal (deg)"),
        ),
        start=1,
    ):
        hist, _ = np.histogram(values, bins, weights=w_accel)
        centres = 0.5 * (bins[1:] + bins[:-1])
        fig.add_trace(
            go.Bar(
                x=centres,
                y=hist / max(hist.sum(), 1e-300),
                marker_color=COLORS["C0"],
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text=label, row=1, col=col)
    fig.update_yaxes(title_text="fraction of the flux", row=1, col=1)
    fig.update_layout(title="Impacts of charge-exchange ions on the accel grid")
    save_figure(fig, output / "impact_distributions")
    mean_energy = np.sum(w_accel * energy) / max(w_accel.sum(), 1e-300)
    print(f"mean impact energy on the accel grid: {mean_energy:.0f} eV")

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(erosion))
    for col, (name, (bins, rate)) in enumerate(erosion.items(), start=1):
        centres = 0.5 * (bins[1:] + bins[:-1])
        fig.add_trace(
            go.Scatter(
                x=centres,
                y=rate,
                mode="lines+markers",
                line={"color": COLORS["C3"]},
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(
            title_text="x (mm)" if name == "barrel" else "distance from axis (mm)",
            row=1,
            col=col,
        )
    fig.update_yaxes(title_text="erosion rate (um per 1000 h, estimate)", row=1, col=1)
    fig.update_layout(title="Estimated sputter erosion of the accel grid's faces")
    save_figure(fig, output / "erosion")
    peaks = {
        name: (0.5 * (b[1:] + b[:-1]))[np.argmax(r)] if r.max() > 0 else np.nan
        for name, (b, r) in erosion.items()
    }
    peak_face = max(erosion, key=lambda k: erosion[k][1].max())
    where = "x" if peak_face == "barrel" else "distance from the axis"
    print(
        f"peak erosion: {erosion[peak_face][1].max():.2f} um per 1000 h on the accel "
        f"{peak_face}, at {where} = {peaks[peak_face]:.2f} mm"
    )

    # 6: pressure scan, by reweighting the same slow ions
    scan_current, scan_erosion = [], []
    for p_scan in pressures:
        n_scan = p_scan / (K_B * GAS_TEMPERATURE)
        w = slow_ion_weights(cx, n_scan, density)[on_accel]
        scan_current.append(w.sum() * E_CHARGE / BEAMLET_CURRENT)
        peak = 0.0
        for name, (coord, bins) in faces.items():
            own = face == name
            eroded, _ = np.histogram(coord[own], bins, weights=(w * yield_)[own] / 2)
            rate = eroded / (np.diff(bins) * 1e-3) / MO_DENSITY * 1e6 * 1000 * HOURS
            peak = max(peak, rate.max())
        scan_erosion.append(peak)
    scan_current = np.array(scan_current)
    scan_erosion = np.array(scan_erosion)
    linear = scan_current[0] * pressures / pressures[0]
    deviation = scan_current / linear
    print(
        "pressure scan: CEX current / linear extrapolation = "
        + ", ".join(f"{p:.0e} Pa: {d:.3f}" for p, d in zip(pressures, deviation))
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=pressures,
            y=scan_current,
            mode="lines+markers",
            name="CEX current to accel / beam current",
            line={"color": COLORS["C3"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pressures,
            y=linear,
            mode="lines",
            name="linear in pressure",
            line={"color": "black", "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pressures,
            y=scan_erosion,
            mode="lines+markers",
            name="peak erosion (um / 1000 h)",
            line={"color": COLORS["C0"]},
        ),
        secondary_y=True,
    )
    fig.update_xaxes(type="log", title_text="neutral xenon pressure (Pa)")
    fig.update_yaxes(type="log", title_text="current fraction", secondary_y=False)
    fig.update_yaxes(type="log", title_text="um per 1000 h", secondary_y=True)
    fig.update_layout(title="Charge-exchange current and erosion against pressure")
    save_figure(fig, output / "pressure_scan")

    print(f"figures in {output}")
    return output


def add_plates(fig: go.Figure, shapes) -> None:
    """Draw the grids and the boundary plates as filled outlines."""
    for name, shape in shapes.items():
        for loop in shape.outlines():
            fig.add_trace(
                go.Scatter(
                    x=np.append(loop[0], loop[0, 0]),
                    y=np.append(loop[1], loop[1, 0]),
                    mode="lines",
                    fill="toself",
                    fillcolor="black" if name in GRIDS else "lightgray",
                    line={"color": "black", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )


if __name__ == "__main__":
    main()
