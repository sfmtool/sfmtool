# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the progress-polling instrumentation (``sfmtool._progress``)."""

from __future__ import annotations

import threading

from sfmtool._progress import _poll_progress, _progress_poll_loop
from sfmtool._sfmtool import ProgressCounter


def _run_poll_once(value: int, total: int) -> list[str]:
    """Drive ``_progress_poll_loop`` through exactly one iteration reading
    ``value``, and return the lines it logged."""
    lines: list[str] = []
    stop = threading.Event()

    def read() -> int:
        stop.set()  # end the loop after this single iteration
        return value

    _progress_poll_loop(lines.append, read, total, stop, 0.0)
    return lines


def test_progress_poll_loop_reports_midpass_value():
    """A count strictly between 0 and total is reported as a done/total line."""
    assert _run_poll_once(500, 1000) == ["    500/1000 patches (50%)"]


def test_progress_poll_loop_skips_zero_and_complete():
    """A 0 count (nothing done yet) and a full count (pass ending) are both
    suppressed, so the poller never prints a redundant 0%/100% line."""
    assert _run_poll_once(0, 1000) == []
    assert _run_poll_once(1000, 1000) == []


def test_poll_progress_uninstrumented_when_no_log_or_trivial_total():
    """``_poll_progress`` yields ``None`` (so the caller passes
    ``progress=None``) when there is no log sink or no work to report."""
    with _poll_progress(None, 100) as counter:
        assert counter is None
    with _poll_progress(lambda _s: None, 0) as counter:
        assert counter is None


def test_poll_progress_yields_counter_and_emits_nothing_for_fast_body():
    """With a log sink and real work, ``_poll_progress`` yields a live
    ``ProgressCounter``; a body that finishes before the poll interval logs
    nothing (fast passes stay quiet)."""
    lines: list[str] = []
    with _poll_progress(lines.append, 100, interval=3600) as counter:
        assert isinstance(counter, ProgressCounter)
        assert counter.value == 0
    assert lines == []


def test_progress_counter_starts_at_zero_and_resets():
    counter = ProgressCounter()
    assert counter.value == 0
    counter.reset()
    assert counter.value == 0
