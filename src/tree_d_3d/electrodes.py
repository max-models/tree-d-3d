"""Conductors at prescribed potentials, by the boundary-integral (panel) method.

A conductor's surface carries an induced line-charge density whose potential, added to
the free charges' potential, makes the conductor an equipotential: the single-layer
formulation of :mod:`tree_d_3d`'s research note "10 Boundary-integral electrodes"
(``obsidian-notes/notes/tree-plasma-ideas``). Discretising a conductor into straight
panels and collocating at panel midpoints turns this into a small dense linear system
``S Q = g``: ``S`` couples panel charges to panel potentials, a panel's own column uses
the closed-form potential of a uniformly charged straight segment at its own midpoint
(finite, unlike a point charge's self-potential). A panel close to its target -- within
``near_field`` of its own lengths, another panel's midpoint or a probe point -- also
enters with that exact segment potential and field (:func:`segment_field`); farther
panels are point line charges at their midpoints, the usual flat-panel approximation,
whose error falls as ``(L / r)**2`` and is summed fast by :func:`tree_d_3d.direct_field`.
Without the near-field correction the potential between collocation points is off by
about a tenth of the source scale next to a panel and the method converges at first
order; with it the boundary is an equipotential to within the piecewise-constant charge
density's own error.

Panel counts here are the hundreds to low thousands the validation examples need, so the
system is solved directly, by an LU factorisation done once per geometry and reused by
every :meth:`BoundaryIntegralElectrodes.solve`; the note's plan of an FMM-accelerated
GMRES solve, needed once panel and particle counts both grow large, is not implemented.

The log kernel is the one :class:`tree_d_3d.FlatQuadtree` and :func:`tree_d_3d.direct_field`
use with ``kernel="log"``: potential ``-2 q ln r`` and field ``2 q d / r**2`` per unit
charge, without ``k_e`` (applied here, as in :class:`tree_d_3d.GroundedStripField`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scope_profiler as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import KDTree

from tree_d_3d._profiling import _PROFILE_TAGS
from tree_d_3d.constants import K_E
from tree_d_3d.flat_tree import direct_field

#: Panels closer to a target than this many of their own lengths enter exactly.
NEAR_FIELD: float = 4.0
#: Gauss--Legendre points per panel in the fast far-field sum.
FAR_POINTS: int = 2


def segment_field(points, start, end) -> tuple[np.ndarray, np.ndarray]:
    """Potential ``(K,)`` and field ``(2, K)`` of uniformly charged straight segments.

    Pairs column by column: ``points``, ``start`` and ``end`` are ``(2, K)``, and segment
    ``k`` from ``start[:, k]`` to ``end[:, k]`` carries unit total charge spread evenly
    along it, in the log kernel of :func:`tree_d_3d.direct_field` (a point charge ``q``
    has potential ``-2 q ln r``), without ``k_e``. With ``u`` and ``v`` the target's
    coordinates along and across the segment of length ``L`` from its start,

    ``phi = -(2 / L) [u ln r_a - (u - L) ln r_b - L + |v| theta]``,
    ``E = (2 / L) [ln(r_a / r_b) t + sign(v) theta n]``,

    where ``r_a`` and ``r_b`` are the distances to the two ends, ``theta`` the angle the
    segment subtends at the target, and ``t`` and ``n`` the unit tangent and normal. On
    the segment itself the normal field (a jump of the surface charge) is taken as zero;
    at an end point the potential is finite but the field diverges logarithmically.
    """
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    d = end - start
    length = np.hypot(d[0], d[1])
    t = d / length
    n = np.stack((-t[1], t[0]))
    rel = points - start
    u = np.sum(rel * t, axis=0)
    v = np.sum(rel * n, axis=0)
    av = np.abs(v)
    ra2 = u * u + v * v
    rb2 = (u - length) ** 2 + v * v
    theta = np.arctan2(u, av) - np.arctan2(u - length, av)
    # at an end point the potential stays finite and the field diverges (log)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_a = np.log(ra2)
        log_b = np.log(rb2)
        ends = np.where(ra2 > 0, u * log_a, 0.0) - np.where(
            rb2 > 0, (u - length) * log_b, 0.0
        )
        along = 0.5 * (log_a - log_b)
        field = (2.0 / length) * (along * t + np.sign(v) * theta * n)
    integral = 0.5 * ends - length + av * theta
    phi = -2.0 * integral / length
    return phi, field


def _point_field(points, sources) -> tuple[np.ndarray, np.ndarray]:
    """The point line-charge kernel, pairwise: ``-2 ln r`` and ``2 d / r**2``."""
    d = points - sources
    r2 = np.sum(d * d, axis=0)
    return -np.log(r2), 2.0 * d / r2


@dataclass(frozen=True)
class TabulatedField:
    """Potential and field sampled on a regular grid, interpolated bilinearly.

    ``x`` ``(nx,)`` and ``y`` ``(ny,)`` are the grid lines, ``potential`` ``(nx, ny)``
    and ``field`` ``(2, nx, ny)`` the samples, as :meth:`BoundaryIntegralElectrodes.tabulate`
    makes them. Calling it with ``(2, M)`` points returns the interpolated potential
    ``(M,)`` and field ``(2, M)``; points outside the grid take the value at the
    nearest edge (see :meth:`contains`).
    """

    x: np.ndarray
    y: np.ndarray
    potential: np.ndarray
    field: np.ndarray

    def contains(self, points) -> np.ndarray:
        """Mask of the ``(2, M)`` points inside the tabulated rectangle."""
        points = np.asarray(points, dtype=float).reshape(2, -1)
        return (
            (points[0] >= self.x[0])
            & (points[0] <= self.x[-1])
            & (points[1] >= self.y[0])
            & (points[1] <= self.y[-1])
        )

    def __call__(self, points) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float).reshape(2, -1)
        i, fx = self._cell(self.x, points[0])
        j, fy = self._cell(self.y, points[1])
        weights = (
            ((1 - fx) * (1 - fy), i, j),
            (fx * (1 - fy), i + 1, j),
            ((1 - fx) * fy, i, j + 1),
            (fx * fy, i + 1, j + 1),
        )
        phi = np.zeros(points.shape[1])
        field = np.zeros((2, points.shape[1]))
        for w, a, b in weights:
            phi += w * self.potential[a, b]
            field += w * self.field[:, a, b]
        return phi, field

    @staticmethod
    def _cell(grid: np.ndarray, coordinate: np.ndarray):
        """Index of the cell holding each coordinate (clamped) and the fraction across it."""
        f = np.clip((coordinate - grid[0]) / (grid[1] - grid[0]), 0.0, grid.size - 1.0)
        index = np.asarray(np.minimum(f.astype(np.int64), grid.size - 2))
        return index, np.asarray(f - index)


class OutlinedShape(Protocol):
    """A 2D shape with boundary loops, such as any 2D :mod:`simgeometry` shape."""

    def outlines(self, max_spacing=None) -> list[np.ndarray]: ...


@dataclass
class Conductor:
    """A conductor's boundary: a polyline of vertices ``(2, n)`` held at one potential.

    Consecutive vertices are panel endpoints; ``closed`` connects the last vertex back
    to the first (a loop), giving ``n`` panels instead of ``n - 1``.
    """

    vertices: np.ndarray
    potential: float
    closed: bool = True


def conductors_from_shape(
    shape: OutlinedShape,
    potential: float,
    max_panel_length: float | Callable[[np.ndarray], np.ndarray],
) -> list[Conductor]:
    """Closed conductors on the boundary of a 2D :mod:`simgeometry` shape.

    One :class:`Conductor` per boundary loop of ``shape.outlines(max_panel_length)``
    (an ``Aperture2D`` gives two, one per plate), all held at ``potential``; an open
    curve such as a ``Polyline`` (a plate of zero thickness) gives an open conductor. Panels are
    at most ``max_panel_length`` long; it can also be a function of ``(2, m)`` points
    returning ``(m,)`` lengths, to grade the panels (short where the field varies
    fastest, such as an aperture tip). Keep panels shorter than the gaps between
    surfaces, a plate's thickness included, so that the charge on one face does not
    have to resolve the field of the facing one.
    """
    closed = bool(getattr(shape, "closed", True))
    return [
        Conductor(loop, potential, closed=closed)
        for loop in shape.outlines(max_panel_length)
    ]


class BoundaryIntegralElectrodes:
    """Conductors held at prescribed potentials, by the single-layer boundary integral.

    ``conductors`` is a sequence of :class:`Conductor`. Call :meth:`solve` once per
    configuration of free charges (or with none, for a pure capacitance problem); then
    :meth:`probe` (and :meth:`potential`, :meth:`field`) give the total field --
    electrodes plus free charges -- at arbitrary points, and :meth:`force_on_sources`
    the force the electrodes and the other free charges exert back on the free charges.

    Panels within ``near_field`` of their own lengths of a target use the exact field of
    a uniformly charged segment (:func:`segment_field`); farther ones are summed fast as
    ``far_points`` Gauss--Legendre point charges each, whose error falls as
    ``(L / r) ** (2 * far_points)``. ``near_field=0, far_points=1`` gives the plain
    midpoint rule.
    """

    def __init__(
        self,
        conductors: Sequence[Conductor],
        *,
        k_e: float = K_E,
        near_field: float = NEAR_FIELD,
        far_points: int = FAR_POINTS,
    ) -> None:
        if not conductors:
            raise ValueError("need at least one conductor")
        if not near_field >= 0:
            raise ValueError("near_field must be >= 0")
        if int(far_points) != far_points or far_points < 1:
            raise ValueError("far_points must be a positive integer")
        self.k_e = float(k_e)
        self.near_field = float(near_field)
        self.far_points = int(far_points)
        self.conductors = list(conductors)
        midpoints, lengths, group, target, starts, ends = [], [], [], [], [], []
        for i, c in enumerate(self.conductors):
            v = np.asarray(c.vertices, dtype=float)
            if v.ndim != 2 or v.shape[0] != 2 or v.shape[1] < 2:
                raise ValueError("a conductor needs vertices of shape (2, n >= 2)")
            a = v if c.closed else v[:, :-1]
            b = np.roll(v, -1, axis=1) if c.closed else v[:, 1:]
            seg_length = np.linalg.norm(b - a, axis=0)
            if np.any(seg_length <= 0):
                raise ValueError("a panel has zero length")
            midpoints.append(0.5 * (a + b))
            starts.append(a)
            ends.append(b)
            lengths.append(seg_length)
            group.append(np.full(a.shape[1], i))
            target.append(np.full(a.shape[1], float(c.potential)))
        #: panel midpoints, ``(2, num_panels)``
        self.midpoints = np.concatenate(midpoints, axis=1)
        #: panel lengths, ``(num_panels,)``
        self.lengths = np.concatenate(lengths)
        #: panel end points, ``(2, num_panels)`` each
        self.starts = np.concatenate(starts, axis=1)
        self.ends = np.concatenate(ends, axis=1)
        #: index of the conductor each panel belongs to, ``(num_panels,)``
        self.group = np.concatenate(group)
        #: the panel's conductor's potential, ``(num_panels,)``
        self.target_potential = np.concatenate(target)
        self.num_panels = self.midpoints.shape[1]
        self._panel_tree = KDTree(self.midpoints.T)
        # far panels as far_points Gauss--Legendre point charges each, panel-major
        x, w = np.polynomial.legendre.leggauss(self.far_points)
        self._weights = 0.5 * w
        along = 0.5 * (x + 1.0)
        self._nodes = (
            self.starts[:, :, None]
            + (self.ends - self.starts)[:, :, None] * along[None, None, :]
        ).reshape(2, -1)
        self._node_weights = np.tile(self._weights, self.num_panels)
        self._node_tree = KDTree(self._nodes.T)
        with sp.region("tree_d_3d.electrodes.factorize", tags=_PROFILE_TAGS):
            self._matrix = self._build_matrix()
            self._lu = lu_factor(self._matrix)
        #: solved panel charges, set by :meth:`solve`
        self.charge: np.ndarray | None = None
        self._source_positions = np.zeros((2, 0))
        self._source_charges = np.zeros(0)

    def _near_pairs(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Target and panel indices of the pairs closer than ``near_field`` panel lengths."""
        empty = np.zeros(0, dtype=np.int64)
        if not self.near_field or not points.shape[1]:
            return empty, empty
        radius = self.near_field * float(self.lengths.max())
        pairs = KDTree(points.T).sparse_distance_matrix(
            self._panel_tree, radius, output_type="ndarray"
        )
        if not pairs.size:
            return empty, empty
        target = pairs["i"].astype(np.int64)
        panel = pairs["j"].astype(np.int64)
        distance = pairs["v"]
        keep = distance < self.near_field * self.lengths[panel]
        return target[keep], panel[keep]

    @sp.profile("tree_d_3d.electrodes.panel_field", tags=_PROFILE_TAGS)
    def _panel_field(self, points: np.ndarray, charge: np.ndarray):
        """Raw potential and field at ``points`` from panels carrying ``charge``.

        Far panels are summed fast as ``far_points`` Gauss--Legendre point charges each;
        for the near pairs those point terms are taken back out (or excluded, for a
        target exactly on a node, where they diverge) and the exact segment terms added.
        """
        k = self.far_points
        node_charge = np.repeat(charge, k) * self._node_weights
        # a target exactly on a node: leave that node out of the point sum
        distance, nearest = self._node_tree.query(points.T)
        exclude = np.where(distance == 0.0, nearest, -1).astype(np.int64)
        target, panel = self._near_pairs(points)
        # the near pairs' node terms, one row per pair and node
        node = (panel[:, None] * k + np.arange(k)).ravel()
        node_target = np.repeat(target, k)
        on_node = node == exclude[node_target]
        phi, e = direct_field(
            points, self._nodes, node_charge, kernel="log", exclude=exclude
        )
        phi, e = np.array(phi), np.array(e)
        if target.size:
            seg_phi, seg_e = segment_field(
                points[:, target], self.starts[:, panel], self.ends[:, panel]
            )
            off = ~on_node
            pt_phi, pt_e = _point_field(
                points[:, node_target[off]], self._nodes[:, node[off]]
            )
            m = points.shape[1]
            q = charge[panel]
            qn = node_charge[node[off]]
            phi += np.bincount(target, q * seg_phi, m)
            phi -= np.bincount(node_target[off], qn * pt_phi, m)
            for axis in (0, 1):
                e[axis] += np.bincount(target, q * seg_e[axis], m)
                e[axis] -= np.bincount(node_target[off], qn * pt_e[axis], m)
        return phi, e

    def _build_matrix(self) -> np.ndarray:
        """The dense ``S``: Gauss nodes for far pairs, exact segments for near ones."""
        k = self.far_points
        matrix = np.zeros((self.num_panels, self.num_panels))
        with np.errstate(divide="ignore"):
            for g in range(k):
                nodes = self._nodes[:, g::k]
                diff = self.midpoints[:, :, None] - nodes[:, None, :]
                # -2 ln r = -ln r^2, per unit panel charge
                matrix -= self._weights[g] * np.log(np.sum(diff**2, axis=0))
        target, panel = self._near_pairs(self.midpoints)
        off = target != panel
        target, panel = target[off], panel[off]
        if target.size:
            matrix[target, panel] = segment_field(
                self.midpoints[:, target], self.starts[:, panel], self.ends[:, panel]
            )[0]
        # self term: the raw potential at a panel's own midpoint from its charge spread
        # uniformly over its length L, integral_{-L/2}^{L/2} -2 ln|t| dt / L * L = ...
        self_term = 2.0 * (1.0 - np.log(self.lengths / 2.0))
        np.fill_diagonal(matrix, self_term)
        return matrix

    @sp.profile("tree_d_3d.electrodes.solve", tags=_PROFILE_TAGS)
    def solve(self, source_positions=None, source_charges=None) -> np.ndarray:
        """Solve for the panel charges that make every conductor an equipotential.

        ``source_positions`` ``(2, N)`` and ``source_charges`` ``(N,)`` are free line
        charges (omit both for a pure capacitance problem). Returns and caches the
        panel charges, used by :meth:`probe` and :meth:`force_on_sources`.
        """
        if source_positions is None:
            source_positions = np.zeros((2, 0))
            source_charges = np.zeros(0)
        self._source_positions = np.asarray(source_positions, dtype=float).reshape(
            2, -1
        )
        self._source_charges = np.asarray(source_charges, dtype=float).ravel()
        if self._source_charges.size:
            free_raw, _ = direct_field(
                self.midpoints,
                self._source_positions,
                self._source_charges,
                kernel="log",
            )
        else:
            free_raw = np.zeros(self.num_panels)
        rhs = self.target_potential / self.k_e - free_raw
        charge = lu_solve(self._lu, rhs)
        self.charge = charge
        return charge

    def probe(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Potential (V) and field (V/m, shape ``(2, M)``) at ``points`` ``(2, M)``.

        Includes the solved panel charges and the free charges passed to :meth:`solve`;
        ``points`` are probes, not sources, so no self term applies. Call :meth:`solve`
        first.
        """
        if self.charge is None:
            raise RuntimeError("call solve() before probe()")
        points = np.asarray(points, dtype=float).reshape(2, -1)
        phi, e = self._panel_field(points, self.charge)
        if self._source_charges.size:
            phi_s, e_s = direct_field(
                points, self._source_positions, self._source_charges, kernel="log"
            )
            phi, e = phi + phi_s, e + e_s
        return self.k_e * phi, self.k_e * e

    def panel_field(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Potential (V) and field (V/m) at ``points`` ``(2, M)`` of the panels alone.

        The induced charge from the last :meth:`solve`, without the free charges passed
        to it: for codes that add the free charges' field themselves (by a tree code,
        say). Near panels enter exactly, as in :meth:`probe`.
        """
        if self.charge is None:
            raise RuntimeError("call solve() before panel_field()")
        points = np.asarray(points, dtype=float).reshape(2, -1)
        phi, e = self._panel_field(points, self.charge)
        return self.k_e * phi, self.k_e * e

    def tabulate(self, lower, upper, shape) -> TabulatedField:
        """:meth:`probe` on a regular grid, for fast bilinear interpolation afterwards.

        ``lower`` and ``upper`` are the corners ``(x, y)`` of the rectangle and
        ``shape`` ``(nx, ny)`` the number of grid lines along each axis (at least 2).
        Like :meth:`probe` it includes the free charges passed to :meth:`solve`, so
        tabulate a static configuration and scale it, for example by an RF voltage.
        """
        nx, ny = (int(n) for n in shape)
        if nx < 2 or ny < 2:
            raise ValueError("a table needs at least 2 grid lines along each axis")
        x = np.linspace(float(lower[0]), float(upper[0]), nx)
        y = np.linspace(float(lower[1]), float(upper[1]), ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        phi, field = self.probe(np.stack((xx.ravel(), yy.ravel())))
        return TabulatedField(
            x=x,
            y=y,
            potential=np.asarray(phi).reshape(nx, ny),
            field=np.asarray(field).reshape(2, nx, ny),
        )

    def potential(self, points) -> np.ndarray:
        """Potential (V) at ``points`` ``(2, M)``; see :meth:`probe`."""
        return self.probe(points)[0]

    def field(self, points) -> np.ndarray:
        """Field (V/m, shape ``(2, M)``) at ``points`` ``(2, M)``; see :meth:`probe`."""
        return self.probe(points)[1]

    def force_on_sources(self) -> tuple[np.ndarray, np.ndarray]:
        """Potential and field (V/m, shape ``(2, N)``) at the free charges from everything else.

        Excludes each source's own singular self-term, so multiplying the field by a
        source's charge gives the physical force on it from the panels and the other
        sources. Call :meth:`solve` first.
        """
        if self.charge is None:
            raise RuntimeError("call solve() before force_on_sources()")
        n = self._source_charges.shape[0]
        phi_p, e_p = self._panel_field(self._source_positions, self.charge)
        if n > 1:
            phi_s, e_s = direct_field(
                self._source_positions,
                self._source_positions,
                self._source_charges,
                exclude=np.arange(n),
                kernel="log",
            )
            phi_p, e_p = phi_p + phi_s, e_p + e_s
        return self.k_e * phi_p, self.k_e * e_p

    def total_charge(self, conductor_index: int | None = None):
        """Induced charge of one conductor, or of every conductor if ``index`` is ``None``."""
        if self.charge is None:
            raise RuntimeError("call solve() before total_charge()")
        if conductor_index is None:
            return np.bincount(
                self.group, weights=self.charge, minlength=len(self.conductors)
            )
        return float(self.charge[self.group == conductor_index].sum())
