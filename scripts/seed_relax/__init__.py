# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The seed relaxation: a rotation-only member's finite sibling.

A rotation-only member claims bearing without range.  The observations its
model refused are the near points, and over a frame pair they carry the
pair's baseline, so the member already holds the evidence for where its
cameras were and how far its points are.  This package turns that evidence
into a finite member beside the original: baselines from the refused rows,
camera centres by translation averaging, structure filled in from the source
clusters the admission never held, and a lens read on the result.

Module map (each module carries its own provenance comment):

* :mod:`quat` -- quaternion and rotation-matrix conversions, numpy only.
* :mod:`graph` -- the admission covisibility graph and the per-edge baseline
  direction read off the refused rows.
* :mod:`averaging` -- camera centres from pairwise directions.
* :mod:`orientation` -- the global orientation bit by parallax-weighted
  cheirality.
* :mod:`structure` -- point estimation, the adjustment's arrays and its
  settled schedule.
* :mod:`relaxation` -- graph to oriented, adjusted state.
* :mod:`lens` -- the lens released on bearings and on the relaxed state.
* :mod:`rings` -- the radius bands the fill-in admits clusters in.
* :mod:`fill` -- the source clusters the admission never held, by radius.
* :mod:`report` -- the runaway-frame report.
* :mod:`pipeline` -- the six stages on one member.
* :mod:`release` -- the arrays and manifest blocks the writer needs.

Nothing here samples, and nothing reads a clock into a record: same inputs,
same output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The late release's knot count on the fisheye chart.  A wide field carries
#: shape a two-knot profile cannot express, and the wider profile pulls the
#: frames the relaxation placed worst toward their neighbours; the pinhole
#: chart's count stays the seed's own, because a narrow field's admission
#: cannot tell a radial profile from a focal error.
DEFAULT_KNOTS_FISHEYE = 4
#: The seed's own knot count (`exp_fast_seed.BSPLINE_KNOTS`).
DEFAULT_KNOTS_PINHOLE = 2


def relax_on():
    """Whether the relaxation rung runs this session (`SFMTOOL_RELAX`).

    Read from the environment rather than imported so a caller can ask the
    question before anything in this package has been imported."""
    return (os.environ.get("SFMTOOL_RELAX", "1") or "1").strip() != "0"


def trace_on():
    """Whether the per-stage census is echoed to stdout."""
    return (os.environ.get("SFMTOOL_RELAX_TRACE", "0") or "0").strip() != "0"


@dataclass(frozen=True)
class Options:
    """Everything the caller can move, in one value.

    ``ring_cap`` is an absolute count of clusters admitted per ring, with
    ``0`` meaning no count at all: a ring then admits its whole band, which is
    the population the band already states.  ``knots_fisheye`` and
    ``knots_pinhole`` are the late release's knot counts, one per radial
    chart."""

    ring_cap: int = 0
    knots_fisheye: int = DEFAULT_KNOTS_FISHEYE
    knots_pinhole: int = DEFAULT_KNOTS_PINHOLE
    trace: bool = False


def _int_env(name, default):
    raw = (os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def options():
    """The run's options, from the environment.

    ``SFMTOOL_RELAX_RING_CAP`` (0, the default, admits a ring's whole band; a
    positive integer is an absolute count per ring),
    ``SFMTOOL_RELAX_KNOTS`` (the fisheye chart's late-release knot count) and
    ``SFMTOOL_RELAX_TRACE``.  The pinhole chart's knot count is not tunable:
    it is the seed's own."""
    return Options(
        ring_cap=max(0, _int_env("SFMTOOL_RELAX_RING_CAP", 0)),
        knots_fisheye=max(1, _int_env("SFMTOOL_RELAX_KNOTS", DEFAULT_KNOTS_FISHEYE)),
        knots_pinhole=DEFAULT_KNOTS_PINHOLE,
        trace=trace_on(),
    )


def run(member, source, opts=None):
    """Relax one rotation-only member.  See :func:`pipeline.run_member`."""
    from .pipeline import run_member

    return run_member(member, source, opts)
