# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Progress/timing helpers for long, GIL-releasing Rust passes.

Small, dependency-light utilities shared by the pipeline drivers (e.g.
:mod:`sfmtool._embed_patches`, :mod:`sfmtool._commands.cluster_patches`): a
timed-step context manager for start/elapsed logging, and a background poller
that echoes a :class:`ProgressCounter`'s ``value``/``total`` while a counted
pass runs.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any

from sfmtool._sfmtool import ProgressCounter


@contextmanager
def _timed_step(log: Any, label: str):
    """Log ``label`` at the start of a step and its elapsed wall time on
    completion, so each blocking Rust pass shows a start line (and a "done"
    line proving it advanced) rather than one silent block. A no-op when
    ``log`` is ``None``."""
    if log is None:
        yield
        return
    log(label)
    t0 = time.perf_counter()
    yield
    log(f"    ...done ({time.perf_counter() - t0:.1f}s)")


def _progress_poll_loop(
    log: Any,
    read: Any,
    total: int,
    stop: threading.Event,
    interval: float,
) -> None:
    """Until ``stop`` is set, every ``interval`` seconds read the current work
    count via ``read()`` and echo a ``done/total`` line through ``log``. Split
    out from :func:`_poll_progress` so the reporting logic is unit-testable with
    an injected ``read`` (the live path reads a :class:`ProgressCounter`)."""
    while not stop.wait(interval):
        done = read()
        # `done` can momentarily read 0 before the first patches finish, and
        # equal `total` just as the pass ends; only report genuine mid-pass
        # progress so we never print a redundant 0%/100% around the done line.
        if 0 < done < total:
            log(f"    {done}/{total} patches ({100 * done // total}%)")


@contextmanager
def _poll_progress(log: Any, total: int, *, interval: float = 5.0):
    """Yield a :class:`ProgressCounter` and, while the body runs, echo its
    ``value``/``total`` from a background thread every ``interval`` seconds.

    The counted Rust pass runs with the GIL released (``py.detach``), so the
    poller thread can read the shared counter and report intra-pass progress
    that would otherwise be one opaque blocking step. Yields ``None`` (no
    counter) when ``log`` is ``None`` or ``total`` is trivial, so the caller
    passes ``progress=None`` and the pass runs uninstrumented."""
    if log is None or total <= 0:
        yield None
        return
    counter = ProgressCounter()
    stop = threading.Event()
    poller = threading.Thread(
        target=_progress_poll_loop,
        args=(log, lambda: counter.value, total, stop, interval),
        daemon=True,
    )
    poller.start()
    try:
        yield counter
    finally:
        stop.set()
        poller.join(timeout=interval)
