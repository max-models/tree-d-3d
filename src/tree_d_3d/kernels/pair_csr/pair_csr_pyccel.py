"""The Pyccel version (CPU) of the ``pair_csr`` kernel of the FMM.

``pair_csr_numpy.py`` next to this file is the version for machines without Pyccel,
with the same arguments; the folder's ``__init__.py`` declares the kernel (see
`tree_d_3d.kernels`).
"""

# ruff: noqa: F821 -- "int64[:]" annotations are Pyccel types, not Python names.
# mypy: ignore-errors


def pair_csr(
    pairs: "int64[:,:]",
    count: int,
    mutual: int,
    row_start: "int64[:]",
    partner: "int64[:]",
    cursor: "int64[:]",
) -> None:
    """Rows by node of ``pairs[:count]``.

    ``row_start`` (``node_count + 1``, zero on entry) becomes the offsets and
    ``partner[row_start[n]:row_start[n + 1]]`` the partners of node ``n``, in pair
    order. With ``mutual`` nonzero, a pair ``(a, b)`` lists ``b`` under ``a`` and ``a``
    under ``b`` (once if ``a == b``); otherwise ``b`` under ``a`` only. ``cursor``
    (``node_count``) is scratch.
    """
    n_nodes = row_start.shape[0] - 1
    for p in range(count):
        a = pairs[p, 0]
        b = pairs[p, 1]
        row_start[a + 1] += 1
        if mutual != 0 and a != b:
            row_start[b + 1] += 1
    for n in range(n_nodes):
        row_start[n + 1] += row_start[n]
        cursor[n] = row_start[n]
    for p in range(count):
        a = pairs[p, 0]
        b = pairs[p, 1]
        partner[cursor[a]] = b
        cursor[a] += 1
        if mutual != 0 and a != b:
            partner[cursor[b]] = a
            cursor[b] += 1
