"""Conductors at prescribed potentials, by the boundary-integral (panel) method.

A conductor's surface carries an induced line-charge density whose potential, added to
the free charges' potential, makes the conductor an equipotential: the single-layer
formulation of :mod:`tree_d_3d`'s research note "10 Boundary-integral electrodes"
(``obsidian-notes/notes/tree-plasma-ideas``). Discretising a conductor into straight
panels and collocating at panel midpoints turns this into a small dense linear system
``S Q = g``: ``S`` couples panel charges to panel potentials, a panel's own column uses
the closed-form potential of a uniformly charged straight segment averaged over itself
(finite, unlike a point charge's self-potential), and every other entry -- panel to
panel, panel to free charge -- is the ordinary point-charge log kernel at the panel's
midpoint, which is the standard flat-panel BEM approximation (exact for panels well
separated from the target; see the note's "Cost" section for the near-panel correction
this skips).

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
from scipy.linalg import lu_factor, lu_solve

from tree_d_3d.constants import K_E
from tree_d_3d.flat_tree import direct_field


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
    (an ``Aperture2D`` gives two, one per plate), all held at ``potential``. Panels are
    at most ``max_panel_length`` long; it can also be a function of ``(2, m)`` points
    returning ``(m,)`` lengths, to grade the panels (short where the field varies
    fastest, such as an aperture tip). Keep panels shorter than the gaps between
    surfaces, a plate's thickness included: the off-diagonal entries treat each panel
    as a point charge at its midpoint.
    """
    return [Conductor(loop, potential) for loop in shape.outlines(max_panel_length)]


class BoundaryIntegralElectrodes:
    """Conductors held at prescribed potentials, by the single-layer boundary integral.

    ``conductors`` is a sequence of :class:`Conductor`. Call :meth:`solve` once per
    configuration of free charges (or with none, for a pure capacitance problem); then
    :meth:`probe` (and :meth:`potential`, :meth:`field`) give the total field --
    electrodes plus free charges -- at arbitrary points, and :meth:`force_on_sources`
    the force the electrodes and the other free charges exert back on the free charges.
    """

    def __init__(self, conductors: Sequence[Conductor], *, k_e: float = K_E) -> None:
        if not conductors:
            raise ValueError("need at least one conductor")
        self.k_e = float(k_e)
        self.conductors = list(conductors)
        midpoints, lengths, group, target = [], [], [], []
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
            lengths.append(seg_length)
            group.append(np.full(a.shape[1], i))
            target.append(np.full(a.shape[1], float(c.potential)))
        #: panel midpoints, ``(2, num_panels)``
        self.midpoints = np.concatenate(midpoints, axis=1)
        #: panel lengths, ``(num_panels,)``
        self.lengths = np.concatenate(lengths)
        #: index of the conductor each panel belongs to, ``(num_panels,)``
        self.group = np.concatenate(group)
        #: the panel's conductor's potential, ``(num_panels,)``
        self.target_potential = np.concatenate(target)
        self.num_panels = self.midpoints.shape[1]
        self._matrix = self._build_matrix()
        self._lu = lu_factor(self._matrix)
        #: solved panel charges, set by :meth:`solve`
        self.charge: np.ndarray | None = None
        self._source_positions = np.zeros((2, 0))
        self._source_charges = np.zeros(0)

    def _build_matrix(self) -> np.ndarray:
        """The dense ``S`` with the analytic self-term on its diagonal."""
        diff = self.midpoints[:, :, None] - self.midpoints[:, None, :]
        r2 = np.sum(diff**2, axis=0)
        with np.errstate(divide="ignore"):
            matrix = -np.log(r2)  # -2 ln r = -ln r^2, per unit panel charge
        # self term: the raw potential at a panel's own midpoint from its charge spread
        # uniformly over its length L, integral_{-L/2}^{L/2} -2 ln|t| dt / L * L = ...
        self_term = 2.0 * (1.0 - np.log(self.lengths / 2.0))
        np.fill_diagonal(matrix, self_term)
        return matrix

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
        phi, e = direct_field(points, self.midpoints, self.charge, kernel="log")
        if self._source_charges.size:
            phi_s, e_s = direct_field(
                points, self._source_positions, self._source_charges, kernel="log"
            )
            phi, e = phi + phi_s, e + e_s
        return self.k_e * phi, self.k_e * e

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
        phi_p, e_p = direct_field(
            self._source_positions, self.midpoints, self.charge, kernel="log"
        )
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
