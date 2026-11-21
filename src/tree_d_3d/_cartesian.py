"""Index tables of the Cartesian expansions of the Coulomb FMM.

The softened Coulomb kernel of charges in the plane, ``f(x, y) = (x^2 + y^2 + eps^2)^(-1/2)``,
is expanded in Taylor series in the two plane coordinates. A coefficient carries a
multi-index ``(a, b)``, the powers of ``x`` and ``y``; the coefficients of total degree
``n = a + b <= p`` are stored in one array, degree by degree, at
``n (n + 1) / 2 + b``. The kernels read the tables built here (one set per order,
cached), so their loops are flat lists of terms.

* multipole moments of a cell about its centre ``c``: ``m_a = sum_j q_j (r_j - c)^a``;
* local expansion about a cell centre: ``phi(c + u) = sum_b T_b u^b``;
* Taylor coefficients ``A_g(t) = D^g f(t) / g!``, by the recurrence (for ``n = |g|``)
  ``n R^2 A_g = -(2n - 1)(t_x A_{g-e_x} + t_y A_{g-e_y}) - (n - 1)(A_{g-2e_x} + A_{g-2e_y})``,
  ``R^2 = |t|^2 + eps^2``, ``A_0 = 1 / R``;
* M2L: ``T_b += sum_a (-1)^|a| C(a + b, a) m_a A_{a+b}(c_dst - c_src)`` over
  ``|a| + |b| <= p``;
* M2M and L2L: shifts by ``s`` with the binomial ``C(g, b) s^(g - b)``.
"""

from __future__ import annotations

from math import comb

import numpy as np

_TABLES: dict[int, dict[str, np.ndarray]] = {}


def size(order: int) -> int:
    """The number of coefficients of total degree at most ``order``."""
    return (order + 1) * (order + 2) // 2


def index(a: int, b: int) -> int:
    """The position of coefficient ``(a, b)``."""
    n = a + b
    return n * (n + 1) // 2 + b


def tables(order: int) -> dict[str, np.ndarray]:
    """The index tables of order ``order`` (see the module docstring), cached.

    ``ex``, ``ey``: the powers of each coefficient. ``mono_parent``, ``mono_dir``: a
    monomial ``u^g`` is ``u^parent`` times ``u_x`` (dir 0) or ``u_y`` (dir 1).
    ``rec``: per coefficient the positions of ``g - e_x``, ``g - e_y``, ``g - 2e_x``,
    ``g - 2e_y`` (-1: none). ``m2l``: rows ``(b, a, a + b)`` with coefficient
    ``m2l_coef``, sorted by ``b``. ``shift``: rows ``(g, b, g - b)`` for ``b <= g``
    with coefficient ``shift_coef`` = ``C(g, b)``, sorted by ``g`` (M2M sums over
    ``b`` into ``g``; L2L sums over ``g`` into ``b``). ``grad``: per coefficient the
    positions of ``g - e_x`` and ``g - e_y`` (-1: none), for the field of a local
    expansion.
    """
    if order in _TABLES:
        return _TABLES[order]
    n_coef = size(order)
    ex = np.empty(n_coef, dtype=np.int64)
    ey = np.empty(n_coef, dtype=np.int64)
    for n in range(order + 1):
        for b in range(n + 1):
            ex[index(n - b, b)] = n - b
            ey[index(n - b, b)] = b

    def at(a: int, b: int) -> int:
        return index(a, b) if a >= 0 and b >= 0 else -1

    mono_parent = np.zeros(n_coef, dtype=np.int64)
    mono_dir = np.zeros(n_coef, dtype=np.int64)
    rec = np.full((n_coef, 4), -1, dtype=np.int64)
    grad = np.full((n_coef, 2), -1, dtype=np.int64)
    for i in range(n_coef):
        a, b = int(ex[i]), int(ey[i])
        if a > 0:
            mono_parent[i], mono_dir[i] = at(a - 1, b), 0
        elif b > 0:
            mono_parent[i], mono_dir[i] = at(a, b - 1), 1
        rec[i] = (at(a - 1, b), at(a, b - 1), at(a - 2, b), at(a, b - 2))
        grad[i] = (at(a - 1, b), at(a, b - 1))
    m2l = []
    for ib in range(n_coef):
        bx, by = int(ex[ib]), int(ey[ib])
        for ia in range(size(order - bx - by)):
            ax, ay = int(ex[ia]), int(ey[ia])
            coef = (-1) ** (ax + ay) * comb(ax + bx, ax) * comb(ay + by, ay)
            m2l.append((ib, ia, index(ax + bx, ay + by), coef))
    shift = []
    for ig in range(n_coef):
        gx, gy = int(ex[ig]), int(ey[ig])
        for bx in range(gx + 1):
            for by in range(gy + 1):
                coef = comb(gx, bx) * comb(gy, by)
                shift.append((ig, index(bx, by), index(gx - bx, gy - by), coef))
    m2l_rows = np.array([row[:3] for row in m2l], dtype=np.int64)
    shift_rows = np.array([row[:3] for row in shift], dtype=np.int64)
    result = {
        "ex": ex,
        "ey": ey,
        "mono_parent": mono_parent,
        "mono_dir": mono_dir,
        "rec": rec,
        "grad": grad,
        "m2l": m2l_rows,
        "m2l_coef": np.array([row[3] for row in m2l], dtype=float),
        "shift": shift_rows,
        "shift_coef": np.array([row[3] for row in shift], dtype=float),
    }
    _TABLES[order] = result
    return result
