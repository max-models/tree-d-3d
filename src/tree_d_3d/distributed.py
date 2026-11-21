"""Running tree-d-3d on several MPI ranks: sources replicated, targets distributed.

Each rank owns a share of the targets (its particles, its probe points, its share of the
panels) and evaluates fields there only. The sources are gathered to every rank, which builds
the whole tree itself: a build is a small part of an evaluation, so the work scales with the
ranks until the replicated build and the gather dominate.

``comm`` is an mpi4py communicator, or `maybempi`'s serial stand-in; ``None`` means one
process without MPI. `maybempi` imports mpi4py only when the process was launched under MPI,
so the serial path costs nothing::

    from maybempi import MPI

    sim = ElectrostaticSimulation(my_bodies, bounds, comm=MPI.COMM_WORLD)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def size(comm: Any) -> int:
    """The number of ranks of ``comm``, 1 for ``None``."""
    return 1 if comm is None else int(comm.Get_size())


def rank(comm: Any) -> int:
    """This process's rank in ``comm``, 0 for ``None``."""
    return 0 if comm is None else int(comm.Get_rank())


def is_parallel(comm: Any) -> bool:
    """Whether ``comm`` has more than one rank."""
    return size(comm) > 1


def gather_columns(comm: Any, local: np.ndarray) -> tuple[np.ndarray, int]:
    """Every rank's columns, in rank order, and the offset of this rank's first.

    ``local`` is ``(k, n_rank)`` or ``(n_rank,)``; the result is ``(k, N)`` or ``(N,)``
    with ``N`` the sum of the ranks' ``n_rank``, the same on every rank.
    """
    local = np.asarray(local, dtype=float)
    if not is_parallel(comm):
        return local.copy(), 0
    flat = local.ndim == 1
    rows = local.reshape(1, -1) if flat else local
    k, n = rows.shape
    counts = np.asarray(comm.allgather(n), dtype=np.int64)
    offset = int(counts[: rank(comm)].sum())
    send = np.ascontiguousarray(rows.T)  # (n, k): one rank's block is contiguous
    received = np.empty((int(counts.sum()), k))
    comm.Allgatherv(send, [received, counts * k])
    gathered = np.ascontiguousarray(received.T)
    return (gathered[0] if flat else gathered), offset


def total(comm: Any, value):
    """The sum of ``value`` (a number or an array) over the ranks."""
    if not is_parallel(comm):
        return value
    if np.ndim(value):
        array = np.ascontiguousarray(value, dtype=float)
        out = np.empty_like(array)
        from maybempi import MPI

        comm.Allreduce(array, out, op=MPI.SUM)
        return out
    return comm.allreduce(value)


def block(n: int, comm: Any) -> slice:
    """This rank's contiguous share of ``range(n)``, the ranks' shares differing by one at most."""
    ranks, me = size(comm), rank(comm)
    base, extra = divmod(n, ranks)
    start = me * base + min(me, extra)
    return slice(start, start + base + (me < extra))


def gather_blocks(comm: Any, local: np.ndarray, n: int) -> np.ndarray:
    """The ``(..., n)`` whole of per-rank ``block(n, comm)`` pieces ``(..., n_rank)``."""
    if not is_parallel(comm):
        return np.asarray(local)
    gathered, _ = gather_columns(comm, np.asarray(local, dtype=float))
    return gathered
