"""Run independent jobs on several CPU cores.

Route attempts and the sim runs that pick marks don't depend on each other,
so :class:`Workers` hands them to a pool of processes (Python threads would
take turns on one core). Each worker gets a copy of the shared state once, when
it starts, not with every job: for a trail that's the part of the map around
Start, not the whole city.

With one worker (or one job) everything runs here, in this process, on the
caller's own state. Jobs must not depend on which process runs them, so a run
gives the same trail whatever the number of workers.

Processes are started the way Windows does it ("spawn") on every platform,
so what's tested on Linux is what runs on Brian's PC.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

_STATE: dict | None = None  # this worker's copy of the shared state


def auto_workers() -> int:
    """One worker per logical CPU (a Ryzen 5 3600 has 12)."""
    return max(1, os.cpu_count() or 1)


def _init(payload: bytes, setup: Callable[[dict], None] | None) -> None:
    global _STATE
    _STATE = pickle.loads(payload)
    if setup is not None:
        setup(_STATE)


def _call(item):
    fn, job = item
    return fn(_STATE, job)


class Workers:
    """``map(fn, jobs)`` = ``[fn(state, job) for job in jobs]``, spread over ``n`` processes.

    ``local`` is the state for jobs run in this process. ``remote()`` builds the
    (usually smaller) state to send to workers; it's only called if a pool is
    needed. ``setup(state)`` runs in each worker after the state arrives (to
    rebuild caches that aren't worth sending).
    """

    def __init__(self, n: int, local: dict, remote: Callable[[], dict] | None = None,
                 setup: Callable[[dict], None] | None = None):
        self.n = max(1, int(n))
        self.local = local
        self._remote = remote or (lambda: local)
        self._setup = setup
        self._pool: ProcessPoolExecutor | None = None

    def map(self, fn: Callable, jobs) -> list:
        jobs = list(jobs)
        if self.n <= 1 or len(jobs) <= 1:
            return [fn(self.local, j) for j in jobs]
        if self._pool is None:
            payload = pickle.dumps(self._remote(), protocol=pickle.HIGHEST_PROTOCOL)
            self._pool = ProcessPoolExecutor(max_workers=self.n, mp_context=mp.get_context("spawn"),
                                             initializer=_init, initargs=(payload, self._setup))
        return list(self._pool.map(_call, [(fn, j) for j in jobs]))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def __enter__(self) -> "Workers":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
