"""The version of the ``pair_csr`` kernel for machines without Pyccel.

``pair_csr_pyccel.py`` as a stable sort of the pair ends by node: the same rows, each
in pair order.
"""

from __future__ import annotations

import numpy as np


def pair_csr(pairs, count, mutual, row_start, partner, cursor):
    """Rows by node of ``pairs[:count]``: see ``pair_csr_pyccel`` (``cursor`` is unused)."""
    a, b = pairs[:count, 0], pairs[:count, 1]
    if mutual != 0:
        # (a, b) then (b, a) for every pair, the second dropped for a == b
        rows = np.column_stack((a, b)).reshape(-1)
        values = np.column_stack((b, a)).reshape(-1)
        keep = np.column_stack((np.ones(count, dtype=bool), a != b)).reshape(-1)
        rows, values = rows[keep], values[keep]
    else:
        rows, values = a, b
    rows = rows.astype(np.intp)  # np.bincount takes the platform's index type
    order = np.argsort(rows, kind="stable")
    partner[: len(order)] = values[order]
    row_start[0] = 0
    row_start[1:] = np.cumsum(np.bincount(rows, minlength=len(row_start) - 1))
