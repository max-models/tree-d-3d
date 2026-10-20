"""Two-surface multipactor between parallel plates, with surface impacts from simgeometry.

Multipactor is a resonant electron avalanche in vacuum RF devices (cavities, couplers,
waveguide windows, gaps between accelerator grids): an electron emitted from one wall is
accelerated across the gap by the RF field, strikes the opposite wall just as the field
reverses, and knocks out more than one secondary electron, which repeat the trip in the
next half period. The classic setting is two parallel plates a distance ``d`` apart with
a voltage ``V0 sin(omega t)`` between them, and the classic result is the *susceptibility
chart*: the zones in the ``(f d, V0)`` plane where the electron number grows.

This example builds that chart from the max-models building blocks rather than from the
analytic theory, and then compares it with the theory:

* the two plates are simgeometry ``Rectangle`` electrodes, panelled by
  :func:`tree_d_3d.conductors_from_shape` and solved once at unit voltage by
  :class:`tree_d_3d.BoundaryIntegralElectrodes`; the RF field is that static solution
  times ``V0 sin(omega t)``, the quasi-static approximation (the gap, at most a few
  millimetres, is far smaller than the RF wavelength of tens of centimetres). It is
  tabulated in the gap by :meth:`~tree_d_3d.BoundaryIntegralElectrodes.tabulate`, accurate
  up to the plate surfaces since panels near a point enter with their exact field;
* electrons are pushed nonrelativistically (impact energies stay below a few keV, well
  under the 511 keV rest energy) by a velocity-Verlet step in the RF phase, and wall
  impacts come from simgeometry's ``shape.crossings(start, end)``, which tests the whole
  step, so no electron steps through a plate, and returns the impact point and the
  surface normal, hence the impact energy and angle;
* each impact emits one secondary with the yield of Vaughan's formula (J. R. M. Vaughan,
  "A new formula for secondary emission yield", IEEE Trans. Electron Devices 36, 1963
  (1989)), carried as a statistical weight: the weight is multiplied by the yield
  ``delta(E, theta)``, so the population grows where ``delta > 1`` at the resonant impact
  energy. The secondary leaves along a cosine (Lambert) distribution about the normal,
  with energies from the flux of a 1 eV Maxwellian (mean 2 eV). The yield parameters,
  ``delta_max = 2.0`` at ``E_max = 300 eV`` with threshold ``E_0 = 12.5 eV``, are
  illustrative values of the order quoted for air-exposed copper, not a measured set;
  electrons arriving below ``E_0`` are absorbed (backscattered primaries are left out).

The theory overlaid on the chart is the parallel-plate return map with a constant
emission velocity (the setting of Hatch and Williams, Phys. Rev. 112, 681 (1958)): an
electron leaves a plate at a fixed speed and phase, crosses a uniform field, and its
impact phase becomes the next emission phase, a map iterated to its fixed point. Its
closed-form resonances (odd orders ``N = 1, 3, 5``: transit in ``N`` half periods) are
drawn as lines; the zone where the map's fixed point has ``delta > 1`` is drawn as a
contour. The simulation has none of the map's idealisations (real 2D field from the
panel solve, spread of emission energies and angles, oblique impacts), so where the two
agree the agreement is a check of the whole chain, and where they differ the difference
is physics the theory leaves out.

1. ``geometry``: the plates and the field magnitude in the gap, from the panel solve at
   unit voltage, against the uniform ``1/d`` of ideal plates.
2. ``resonant_trajectory``: one electron at a first-order resonance crossing the gap in
   each half period, against the RF voltage.
3. ``secondary_yield``: Vaughan's yield against impact energy at normal and oblique
   incidence, with the first crossover ``delta = 1`` that sets the lower voltage edge of
   the zones.
4. ``electron_growth``: the weighted electron number against RF cycles for a voltage
   inside the first-order zone and one below it.
5. ``susceptibility_chart``: the growth rate per RF cycle over ``(f d, V0)``, with the
   theory's resonance lines and multipactor zone overlaid; the printed overlap counts
   how many simulated growth points the theory's zone contains and vice versa.

Usage::

    python examples/multipactor.py [--output examples/output] [--quick]

``--quick`` uses a coarser chart and fewer electrons (under a minute instead of about
five);
``--smoke`` is the tiny configuration the test suite runs. Needs Plotly
(``pip install 'tree-d-3d[examples]'``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plasma_plots import save_figure
from simgeometry import Rectangle

from tree_d_3d import (
    BoundaryIntegralElectrodes,
    TabulatedField,
    conductors_from_shape,
)

COLORS = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "C3": "#d62728"}

E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837015e-31  # kg

GAP = 1.0e-3  # m: the plate separation d
PLATE_THICKNESS = 0.5 * GAP
PLATE_HALF_LENGTH = 40 * GAP
FIELD_HALF_WIDTH = 20 * GAP  # electrons drifting further sideways are counted as lost

# Vaughan (1989), illustrative values for air-exposed copper (see the module docstring)
DELTA_MAX = 2.0
E_MAX = 300.0  # eV
E_THRESHOLD = 12.5  # eV, no secondaries below
SMOOTHNESS = 1.0
EMISSION_TEMPERATURE = 1.0  # eV: Maxwellian flux, mean emission energy 2 eV

STEPS_PER_CYCLE = 100


def vaughan_yield(energy, theta):
    """Vaughan's secondary emission yield at impact ``energy`` (eV) and angle ``theta``.

    ``theta`` is measured from the surface normal. The peak rises and moves to higher
    energy at oblique incidence (``smoothness`` 1, a typical technical surface).
    """
    energy = np.asarray(energy, dtype=float)
    theta = np.asarray(theta, dtype=float)
    delta_max = DELTA_MAX * (1 + SMOOTHNESS * theta**2 / (2 * np.pi))
    e_max = E_MAX * (1 + SMOOTHNESS * theta**2 / np.pi)
    w = (energy - E_THRESHOLD) / (e_max - E_THRESHOLD)
    k = np.where(w < 1, 0.62, 0.25)
    w = np.clip(w, 1e-300, None)
    delta = delta_max * (w * np.exp(1 - w)) ** k
    return np.where(energy > E_THRESHOLD, delta, 0.0)


def first_crossover() -> float:
    """The impact energy (eV) above which the normal-incidence yield first exceeds 1."""
    energies = np.linspace(E_THRESHOLD, E_MAX, 20001)
    return float(energies[np.argmax(vaughan_yield(energies, 0.0) > 1.0)])


# -- field ------------------------------------------------------------------------------


@dataclass
class GapField:
    """The panel-solved field of the gap at unit voltage, tabulated for interpolation.

    The lower plate is at 0 V and the upper at 1 V, so the field points from the upper
    plate to the lower one; ``x`` runs along the plates, ``y`` across the gap.
    """

    table: TabulatedField
    plates: tuple[Rectangle, Rectangle]
    num_panels: int

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Bilinear field ``(2, N)`` at ``(2, N)`` points (clamped to the table)."""
        return self.table(points)[1]

    @property
    def x(self) -> np.ndarray:
        return self.table.x

    @property
    def y(self) -> np.ndarray:
        return self.table.y

    @property
    def ex(self) -> np.ndarray:
        """``(len(y), len(x))``, V/m per volt."""
        return self.table.field[0].T

    @property
    def ey(self) -> np.ndarray:
        return self.table.field[1].T


def solve_gap_field(panel: float, nx: int, ny: int) -> GapField:
    """Panel-solve the two plates at 0 V and 1 V and tabulate the field in the gap.

    Panels are ``panel * GAP`` long on the faces near the middle of the gap and grow
    towards the plate ends, where the field matters little.
    """
    lower = Rectangle(
        midpoint=(0.0, -PLATE_THICKNESS / 2),
        side_lengths=(2 * PLATE_HALF_LENGTH, PLATE_THICKNESS),
    )
    upper = Rectangle(
        midpoint=(0.0, GAP + PLATE_THICKNESS / 2),
        side_lengths=(2 * PLATE_HALF_LENGTH, PLATE_THICKNESS),
    )

    def length(points: np.ndarray) -> np.ndarray:
        grow = np.maximum(np.abs(points[0]) - FIELD_HALF_WIDTH, 0.0)
        return panel * GAP + 0.2 * grow

    conductors = conductors_from_shape(lower, 0.0, length) + conductors_from_shape(
        upper, 1.0, length
    )
    electrodes = BoundaryIntegralElectrodes(conductors, k_e=1.0)
    electrodes.solve()
    # stop a hair short of the plates: on a panel its own normal field is the average
    # of the two sides of its surface charge; the table clamps beyond its edges
    inset = 1e-4 * GAP
    table = electrodes.tabulate(
        (-FIELD_HALF_WIDTH, inset), (FIELD_HALF_WIDTH, GAP - inset), (nx, ny)
    )
    return GapField(
        table=table,
        plates=(lower, upper),
        num_panels=electrodes.num_panels,
    )


# -- electrons --------------------------------------------------------------------------


def emit(normal: np.ndarray, omega: np.ndarray, rng) -> np.ndarray:
    """Secondary velocities ``(2, N)`` in m per radian of RF phase.

    Energies from the flux of a Maxwellian at ``EMISSION_TEMPERATURE`` (a Gamma(2, T)
    distribution), directions from a cosine law about the unit ``normal``.
    """
    n = normal.shape[1]
    energy = rng.gamma(2.0, EMISSION_TEMPERATURE, n)
    speed = np.sqrt(2 * energy * E_CHARGE / M_E)
    sin_theta = np.sqrt(rng.uniform(0, 1, n)) * rng.choice((-1.0, 1.0), n)
    cos_theta = np.sqrt(1 - sin_theta**2)
    tangent = np.stack((-normal[1], normal[0]))
    direction = cos_theta * normal + sin_theta * tangent
    return speed * direction / omega


@dataclass
class Population:
    """Weighted electrons of many RF cases advanced together in the RF phase ``tau``.

    Each electron belongs to the case of its column: ``omega`` (rad/s) and ``v0`` (V)
    per electron. Positions are in m and velocities in m per radian of phase, so one
    phase step serves every frequency. ``log_weight`` is ``-inf`` for absorbed or lost
    electrons.
    """

    pos: np.ndarray
    vel: np.ndarray
    omega: np.ndarray
    v0: np.ndarray
    log_weight: np.ndarray
    impacts: list


def seed(case_omega, case_v0, per_case: int, rng) -> Population:
    """``per_case`` electrons per case, emitted from random points of both plates."""
    omega = np.repeat(case_omega, per_case)
    v0 = np.repeat(case_v0, per_case)
    n = omega.size
    on_upper = rng.uniform(size=n) < 0.5
    x = rng.uniform(-0.2 * GAP, 0.2 * GAP, n)
    y = np.where(on_upper, GAP, 0.0)
    normal = np.stack((np.zeros(n), np.where(on_upper, -1.0, 1.0)))
    return Population(
        pos=np.stack((x, y)) + 1e-9 * normal,
        vel=emit(normal, omega, rng),
        omega=omega,
        v0=v0,
        log_weight=np.zeros(n),
        impacts=[],
    )


def advance(pop: Population, field: GapField, tau: float, dtau: float, rng) -> None:
    """One velocity-Verlet step in phase, with wall impacts and secondary emission."""
    alive = np.isfinite(pop.log_weight)
    # acceleration in m / rad^2: -(e / m) V0 sin(tau) E_unit / omega^2
    coefficient = -E_CHARGE / M_E * pop.v0 / pop.omega**2

    def accel(points, phase):
        return coefficient * np.sin(phase) * field(points)

    half = pop.vel + 0.5 * dtau * accel(pop.pos, tau)
    new = pop.pos + dtau * half
    first = np.full(pop.pos.shape[1], np.inf)
    point = np.full(pop.pos.shape, np.nan)
    normal = np.full(pop.pos.shape, np.nan)
    for plate in field.plates:
        hits = plate.crossings(pop.pos[:, alive], new[:, alive])
        fraction = np.where(np.asarray(hits.hit), np.asarray(hits.fraction), np.inf)
        idx = np.flatnonzero(alive)
        closer = fraction < first[idx]
        first[idx[closer]] = fraction[closer]
        point[:, idx[closer]] = np.asarray(hits.point)[:, closer]
        normal[:, idx[closer]] = np.asarray(hits.normal)[:, closer]
    pop.vel = half + 0.5 * dtau * accel(new, tau + dtau)
    pop.pos = new
    hit = np.isfinite(first)
    if np.any(hit):
        # impact velocity: interpolate within the step (to first order)
        v_impact = pop.omega[hit] * half[:, hit]
        speed = np.hypot(*v_impact)
        energy = 0.5 * M_E * speed**2 / E_CHARGE
        cos_theta = np.clip(-np.sum(v_impact * normal[:, hit], axis=0) / speed, 0, 1)
        theta = np.arccos(cos_theta)
        delta = vaughan_yield(energy, theta)
        with np.errstate(divide="ignore"):
            pop.log_weight[hit] += np.log(delta)
        pop.impacts.append((tau + dtau, energy))
        pop.pos[:, hit] = point[:, hit] + 1e-9 * normal[:, hit]
        pop.vel[:, hit] = emit(normal[:, hit], pop.omega[hit], rng)
    lost = np.abs(pop.pos[0]) > FIELD_HALF_WIDTH
    pop.log_weight[lost] = -np.inf


def run_cases(case_omega, case_v0, field, *, per_case, cycles, seed_value=0):
    """Advance every case for ``cycles`` RF cycles; return ``(cycles + 1, cases)`` of ln N.

    ``ln N`` is the log of the total weight of each case, relative to its seed number.
    """
    rng = np.random.default_rng(seed_value)
    pop = seed(np.asarray(case_omega), np.asarray(case_v0), per_case, rng)
    num_cases = np.size(case_omega)
    dtau = 2 * np.pi / STEPS_PER_CYCLE
    history = [np.zeros(num_cases)]
    tau = 0.0
    for _ in range(cycles):
        for _ in range(STEPS_PER_CYCLE):
            advance(pop, field, tau, dtau, rng)
            tau += dtau
        weights = np.exp(pop.log_weight).reshape(num_cases, per_case)
        with np.errstate(divide="ignore"):
            history.append(np.log(weights.sum(axis=1) / per_case))
    return np.array(history)


def growth_rate(history: np.ndarray) -> np.ndarray:
    """Growth of ln N per RF cycle, a least-squares slope over the second half.

    Cases that died out (every weight zero) get ``-inf``.
    """
    cycles = history.shape[0] - 1
    late = slice(cycles // 2, cycles + 1)
    t = np.arange(cycles + 1)[late]
    values = history[late]
    dead = ~np.all(np.isfinite(values), axis=0)
    values = np.where(np.isfinite(values), values, 0.0)
    slope = np.polyfit(t, values, 1)[0]
    return np.where(dead, -np.inf, slope)


# -- theory -----------------------------------------------------------------------------


def resonance_voltage(fd, order: int, phase: float = 0.0, emission_ev: float = 2.0):
    """Closed-form resonance voltage (V) for transit in ``order`` half periods.

    Uniform field, emission at ``emission_ev`` along the normal at RF phase ``phase``:
    ``V0 = m omega d (omega d - N pi v_e) / (e (N pi cos(phase) + 2 sin(phase)))``.
    ``fd`` is in GHz mm.
    """
    omega_d = 2 * np.pi * np.asarray(fd) * 1e6  # (rad/s) m
    v_e = np.sqrt(2 * emission_ev * E_CHARGE / M_E)
    n_pi = order * np.pi
    return (
        M_E
        * omega_d
        * (omega_d - n_pi * v_e)
        / (E_CHARGE * (n_pi * np.cos(phase) + 2 * np.sin(phase)))
    )


def theory_zone(fd, v0, *, emission_ev=2.0, bounces=40, samples=600):
    """Multipactor in the constant-emission-velocity return map, for each ``(fd, v0)``.

    An electron leaves a plate at ``emission_ev`` along the normal in a uniform field;
    its impact phase (shifted by pi when it reaches the other plate) is the next
    emission phase. From four starting phases the map is iterated ``bounces`` times
    (transits up to three RF cycles, so orders up to ``N = 5``; the crossing time is
    interpolated between ``samples`` phase samples). Returns the larger, over the
    starts, of the mean ``ln delta`` per impact over the last half: positive where the
    theory multipacts.
    """
    fd = np.asarray(fd, dtype=float).ravel()
    v0 = np.asarray(v0, dtype=float).ravel()
    omega = 2 * np.pi * fd * 1e6 / GAP  # fd in GHz mm with d = GAP
    c = E_CHARGE * v0 / (M_E * GAP * omega**2) / GAP  # in units of d
    u = np.sqrt(2 * emission_ev * E_CHARGE / M_E) / (omega * GAP)
    tau = np.linspace(0, 6 * np.pi, samples)[1:]
    rows = np.arange(fd.size)

    def height(t, p):
        return u[:, None] * t + c[:, None] * (t * np.cos(p) - np.sin(t + p) + np.sin(p))

    best = np.full(fd.size, -np.inf)
    for start in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        phase = np.full(fd.size, start)
        log_delta = []
        for _ in range(bounces):
            p = phase[:, None]
            y = height(tau, p)
            out = (y >= 1.0) | (y < 0.0)
            ok = out.any(axis=1)
            index = np.where(ok, out.argmax(axis=1), tau.size - 1)
            other_plate = y[rows, index] >= 1.0
            # interpolate the crossing of y = 1 (or y = 0) inside the last interval
            before = np.maximum(index - 1, 0)
            y0 = np.where(index > 0, y[rows, before], 0.0)
            t0 = np.where(index > 0, tau[before], 0.0)
            level = np.where(other_plate, 1.0, 0.0)
            y1 = y[rows, index]
            weight = np.clip((level - y0) / np.where(y1 != y0, y1 - y0, 1.0), 0, 1)
            t = t0 + weight * (tau[index] - t0)
            speed = u + c * (np.cos(phase) - np.cos(t + phase))
            energy = 0.5 * M_E * (speed * omega * GAP) ** 2 / E_CHARGE
            delta = np.where(ok, vaughan_yield(energy, 0.0), 0.0)
            with np.errstate(divide="ignore"):
                log_delta.append(np.log(delta))
            phase = np.mod(phase + t + np.where(other_plate, np.pi, 0.0), 2 * np.pi)
        late = np.array(log_delta[bounces // 2 :])
        mean = np.where(np.all(np.isfinite(late), axis=0), late.mean(axis=0), -np.inf)
        best = np.maximum(best, mean)
    return best


# -- figures ----------------------------------------------------------------------------


def plot_geometry(field: GapField, output: Path) -> None:
    """Figure 1: plates and the tabulated field magnitude against the ideal ``1/d``."""
    magnitude = np.hypot(field.ex, field.ey) * GAP
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=field.x * 1e3,
            y=field.y * 1e3,
            z=magnitude,
            colorscale="Viridis",
            colorbar={"title": "|E| d / V"},
        )
    )
    for plate in field.plates:
        (loop,) = plate.outlines()
        closed = np.concatenate((loop, loop[:, :1]), axis=1) * 1e3
        fig.add_trace(
            go.Scatter(
                x=closed[0],
                y=closed[1],
                mode="lines",
                fill="toself",
                fillcolor="gray",
                line={"color": "black"},
                showlegend=False,
            )
        )
    fig.update_xaxes(title_text="x (mm)", range=[-3, 3])
    fig.update_yaxes(title_text="y (mm)", range=[-0.6, 1.6])
    fig.update_layout(
        title=f"Plates and the field at unit voltage ({field.num_panels} panels)"
    )
    save_figure(fig, output / "geometry")


def plot_resonant_trajectory(
    field: GapField, fd: float, v0: float, output: Path
) -> None:
    """Figure 2: one electron at ``v0`` inside the first-order zone, locking to it."""
    phase = 0.3
    omega = 2 * np.pi * fd * 1e6 / GAP
    rng = np.random.default_rng(1)
    # emitted from the lower plate along the normal at 2 eV, phase chosen so that the
    # field first pushes it into the gap
    pop = Population(
        pos=np.array([[0.0], [1e-9]]),
        vel=np.array([[0.0], [np.sqrt(2 * 2.0 * E_CHARGE / M_E) / omega]]),
        omega=np.array([omega]),
        v0=np.array([v0]),
        log_weight=np.zeros(1),
        impacts=[],
    )
    # the upper plate is at +V0 sin(tau): the field pushes electrons up when sin > 0
    tau = phase
    dtau = 2 * np.pi / STEPS_PER_CYCLE
    taus, ys = [tau], [pop.pos[1, 0]]
    for _ in range(4 * STEPS_PER_CYCLE):
        advance(pop, field, tau, dtau, rng)
        # keep the trajectory deterministic: re-emit along the normal at 2 eV
        if pop.impacts and pop.impacts[-1][0] == tau + dtau:
            normal = 1.0 if pop.pos[1, 0] < 0.5 * GAP else -1.0
            pop.vel[:, 0] = (0.0, normal * np.sqrt(2 * 2.0 * E_CHARGE / M_E) / omega)
        tau += dtau
        taus.append(tau)
        ys.append(pop.pos[1, 0])
    cycles = (np.array(taus) - taus[0]) / (2 * np.pi)
    impacts = [(t - taus[0]) / (2 * np.pi) for t, _ in pop.impacts]
    energies = [float(e[0]) for _, e in pop.impacts]
    print(
        f"resonant trajectory at f d = {fd} GHz mm, V0 = {v0:.1f} V: impacts after "
        f"{np.round(np.diff([0.0, *impacts]), 3)} cycles at {np.round(energies, 1)} eV"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=np.array(ys) / GAP,
            mode="lines",
            line={"color": COLORS["C0"]},
            name="electron, y / d",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=0.5 + 0.5 * np.sin(np.array(taus)),
            mode="lines",
            line={"color": COLORS["C1"], "dash": "dash"},
            name="RF voltage (scaled)",
        )
    )
    fig.update_xaxes(title_text="time (RF cycles)")
    fig.update_yaxes(title_text="y / d")
    fig.update_layout(
        title=f"First-order resonance: one gap crossing per half period (V0 = {v0:.0f} V)"
    )
    save_figure(fig, output / "resonant_trajectory")


def plot_secondary_yield(output: Path) -> float:
    """Figure 3: Vaughan's yield; returns the first crossover energy (eV)."""
    energy = np.linspace(0, 1500, 600)
    e1 = first_crossover()
    fig = go.Figure()
    for theta, color, name in (
        (0.0, COLORS["C0"], "normal incidence"),
        (np.pi / 3, COLORS["C2"], "60 degrees from the normal"),
    ):
        fig.add_trace(
            go.Scatter(
                x=energy,
                y=vaughan_yield(energy, theta),
                mode="lines",
                line={"color": color},
                name=name,
            )
        )
    fig.add_hline(y=1.0, line={"color": "gray", "dash": "dot"})
    fig.add_vline(
        x=e1,
        line={"color": "gray", "dash": "dash"},
        annotation_text=f"E1 = {e1:.0f} eV",
    )
    fig.update_xaxes(title_text="impact energy (eV)")
    fig.update_yaxes(title_text="secondary yield delta")
    fig.update_layout(title="Vaughan's secondary emission yield (illustrative copper)")
    save_figure(fig, output / "secondary_yield")
    return e1


def main(argv: list[str] | None = None) -> Path:
    """Run the multipactor study and write the figures; return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="smaller and faster")
    parser.add_argument("--smoke", action="store_true", help="tiny, for the tests")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        panel, nx, ny = 0.2, 41, 11
        n_fd, n_v0, per_case, cycles = 4, 4, 4, 4
    elif args.quick:
        panel, nx, ny = 0.1, 161, 21
        n_fd, n_v0, per_case, cycles = 14, 14, 12, 16
    else:
        panel, nx, ny = 0.05, 321, 41
        n_fd, n_v0, per_case, cycles = 24, 24, 24, 24

    # 1. geometry and field
    field = solve_gap_field(panel, nx, ny)
    middle = np.hypot(field.ex, field.ey)[:, np.abs(field.x) < 5 * GAP] * GAP
    print(
        f"{field.num_panels} panels; |E| d / V over |x| < 5 d: "
        f"{middle.min():.4f} to {middle.max():.4f} (ideal plates: 1)"
    )
    plot_geometry(field, output)

    # 2. a resonant trajectory, in the middle of the theory's first-order zone at
    # f d = 2 GHz mm; 3. the yield
    fd_example = 2.0
    v_scan = np.geomspace(50.0, 1000.0, 120)
    zone = v_scan[theory_zone(np.full(v_scan.size, fd_example), v_scan) > 0]
    v_resonant = float(np.sqrt(zone.min() * zone.max()))
    print(
        f"theory's first-order zone at f d = {fd_example} GHz mm: "
        f"{zone.min():.0f} to {zone.max():.0f} V"
    )
    plot_resonant_trajectory(field, fd_example, v_resonant, output)
    e1 = plot_secondary_yield(output)
    print(f"first crossover E1 = {e1:.1f} eV")

    # 4. growth in and below the first-order zone
    v_below = 0.3 * v_resonant
    omega = 2 * np.pi * fd_example * 1e6 / GAP
    history = run_cases(
        [omega, omega],
        [v_resonant, v_below],
        field,
        per_case=4 * per_case,
        cycles=cycles,
        seed_value=2,
    )
    rates = growth_rate(history)
    print(
        f"growth per cycle at f d = {fd_example}: {rates[0]:.3f} at V0 = "
        f"{v_resonant:.0f} V, {rates[1]:.3f} at V0 = {v_below:.0f} V"
    )
    fig = go.Figure()
    for k, (v, color) in enumerate(
        ((v_resonant, COLORS["C3"]), (v_below, COLORS["C0"]))
    ):
        fig.add_trace(
            go.Scatter(
                x=np.arange(history.shape[0]),
                y=history[:, k],
                mode="lines+markers",
                line={"color": color},
                name=f"V0 = {v:.0f} V",
            )
        )
    fig.update_xaxes(title_text="RF cycles")
    fig.update_yaxes(title_text="ln(N / N0)")
    fig.update_layout(
        title=f"Electron number at f d = {fd_example} GHz mm: avalanche or decay"
    )
    save_figure(fig, output / "electron_growth")

    # 5. susceptibility chart
    fd_axis = np.geomspace(0.3, 3.0, n_fd)
    v0_axis = np.geomspace(10.0, 3000.0, n_v0)
    fd_grid, v0_grid = np.meshgrid(fd_axis, v0_axis)
    omega_grid = 2 * np.pi * fd_grid * 1e6 / GAP
    history = run_cases(
        omega_grid.ravel(),
        v0_grid.ravel(),
        field,
        per_case=per_case,
        cycles=cycles,
        seed_value=3,
    )
    rate = growth_rate(history).reshape(v0_grid.shape)
    theory = theory_zone(fd_grid, v0_grid).reshape(v0_grid.shape)
    simulated = rate > 0
    predicted = theory > 0
    both = np.sum(simulated & predicted)
    print(
        f"chart: {simulated.sum()} simulated growth points, {predicted.sum()} in the "
        f"theory's zone, {both} in both ({both / max(simulated.sum(), 1):.0%} of the "
        f"simulated, {both / max(predicted.sum(), 1):.0%} of the theory's)"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=fd_axis,
            y=v0_axis,
            z=np.where(np.isfinite(rate), np.clip(rate, -1.0, 1.0), -1.0),
            zmid=0.0,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "ln N per cycle (clipped)"},
        )
    )
    fig.add_trace(
        go.Contour(
            x=fd_axis,
            y=v0_axis,
            z=predicted.astype(float),
            contours={"start": 0.5, "end": 0.5, "coloring": "none"},
            line={"color": "black", "width": 2},
            showscale=False,
            name="theory: multipactor zone",
            showlegend=True,
        )
    )
    fd_line = np.geomspace(fd_axis[0], fd_axis[-1], 200)
    for order, dash in ((1, "solid"), (3, "dash"), (5, "dot")):
        v_line = resonance_voltage(fd_line, order, 0.0)
        keep = (v_line > v0_axis[0]) & (v_line < v0_axis[-1])
        fig.add_trace(
            go.Scatter(
                x=fd_line[keep],
                y=v_line[keep],
                mode="lines",
                line={"color": COLORS["C2"], "dash": dash},
                name=f"resonance N = {order} (phase 0)",
            )
        )
    fig.update_xaxes(type="log", title_text="f d (GHz mm)")
    fig.update_yaxes(type="log", title_text="V0 (V)")
    fig.update_layout(
        title="Multipactor susceptibility: simulation against theory",
        legend={"orientation": "h", "y": -0.18},
    )
    save_figure(fig, output / "susceptibility_chart")

    print(f"figures in {output}")
    return output


if __name__ == "__main__":
    main()
