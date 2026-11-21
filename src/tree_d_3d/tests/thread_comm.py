"""An in-process stand-in for an MPI communicator: one thread per rank.

It implements the collectives tree-d-3d uses (``allgather``, ``allreduce``,
``Allgatherv``, ``Allreduce``, ``Barrier``), so the parallel code paths run under
plain pytest, without mpirun.
"""

import threading

import numpy as np
from maybempi import MPI


class _World:
    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=120)
        self.slots: list = [None] * size


class ThreadComm:
    """Rank ``rank`` of a `_World` of threads."""

    def __init__(self, world: _World, rank: int):
        self._world = world
        self._rank = rank

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._world.size

    def _exchange(self, value) -> list:
        """Every rank's ``value``, in rank order."""
        world = self._world
        world.slots[self._rank] = value
        world.barrier.wait()
        values = list(world.slots)
        world.barrier.wait()
        return values

    def Barrier(self) -> None:
        self._world.barrier.wait()

    def allgather(self, value) -> list:
        return self._exchange(value)

    def allreduce(self, value, op=None):
        values = self._exchange(value)
        return max(values) if op is MPI.MAX else sum(values)

    def Allgatherv(self, send, receive) -> None:
        buffer, _ = receive
        parts = self._exchange(np.array(send, copy=True).ravel())
        buffer.reshape(-1)[:] = np.concatenate(parts)

    def Allreduce(self, send, out, op=None) -> None:
        parts = self._exchange(np.array(send, copy=True))
        out[...] = np.max(parts, axis=0) if op is MPI.MAX else np.sum(parts, axis=0)


def run_ranks(size: int, function) -> list:
    """``function(comm)`` on ``size`` thread ranks; their results in rank order."""
    world = _World(size)
    results: list = [None] * size
    errors: list = []

    def target(rank: int) -> None:
        try:
            results[rank] = function(ThreadComm(world, rank))
        except BaseException as error:  # noqa: BLE001 -- re-raised below
            errors.append(error)
            world.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results
