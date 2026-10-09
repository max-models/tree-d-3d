"""Charged particles, alone or adopted by a simulation."""

from __future__ import annotations

import numpy as np

from tree_d_3d._utils import _finite


class ChargedBody:
    """A physical particle or weighted macroparticle in consistent units.

    Input mass and charge are per physical particle. Weight multiplies both,
    preserving q/m; stored mass and charge are the totals. A fixed source
    stays immobile. SI inputs give SI outputs.

    State lives in arrays. A simulation adopts its bodies, after which
    ``pos``, ``vel`` and ``force`` are views of columns of the simulation's
    ``(2, N)`` arrays, and reading or assigning any attribute reads or writes
    them.
    """

    pos = property(
        lambda self: self._pos[:, self._index],
        lambda self, value: self._pos.__setitem__((slice(None), self._index), value),
    )
    vel = property(
        lambda self: self._vel[:, self._index],
        lambda self, value: self._vel.__setitem__((slice(None), self._index), value),
    )
    force = property(
        lambda self: self._force[:, self._index],
        lambda self, value: self._force.__setitem__((slice(None), self._index), value),
    )
    mass = property(
        lambda self: float(self._mass[self._index]),
        lambda self, value: self._mass.__setitem__(self._index, value),
    )
    charge = property(
        lambda self: float(self._charge[self._index]),
        lambda self, value: self._charge.__setitem__(self._index, value),
    )
    fixed = property(
        lambda self: bool(self._fixed[self._index]),
        lambda self, value: self._fixed.__setitem__(self._index, value),
    )

    def _bind(self, pos, vel, force, mass, charge, fixed, index) -> None:
        self._pos, self._vel, self._force = pos, vel, force
        self._mass, self._charge, self._fixed = mass, charge, fixed
        self._index = index

    def __init__(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        mass: float = 1.0,
        charge: float = 1.0,
        *,
        weight: float = 1.0,
        fixed: bool = False,
    ) -> None:
        pos = np.array([[x], [y]], dtype=float)
        vel = np.array([[vx], [vy]], dtype=float)
        if not np.all(np.isfinite([pos, vel])):
            raise ValueError("Particle position and velocity must be finite")
        self.weight = _finite("weight", weight, positive=True)
        total_mass = _finite("mass", mass, positive=True) * self.weight
        total_charge = _finite("charge", charge) * self.weight
        _finite("weighted mass", total_mass, positive=True)
        _finite("weighted charge", total_charge)
        if fixed and np.any(vel):
            raise ValueError("A fixed source must have zero velocity")
        self._bind(
            pos,
            vel,
            np.zeros((2, 1)),
            np.array([total_mass]),
            np.array([total_charge]),
            np.array([bool(fixed)]),
            0,
        )

    def reset_force(self) -> None:
        self.force.fill(0.0)

    def add_force(self, fx: float, fy: float) -> None:
        self.force += (fx, fy)
