# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The contract every track-at-pixel candidate implements.

A candidate is a module under ``candidates/`` exposing

    build_track(ctx: HoldoutContext, image: int, pixel: tuple[float, float],
                options: dict) -> TrackAtPixelResult

that either returns a track-stage :class:`EditableTrack` centred at (or very
near) ``pixel`` in ``image``, or raises :class:`TrackAtPixelError` saying which
stage refused and why. ``ctx`` is everything the function may read: the
reconstruction with the track under test removed, the photographs, the
descriptor index, the cluster-patches ``.matches`` file and the neighbourhood
queries (see ``context.py``). It must not read the held-out point any other
way; the harness is only honest while that holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TrackAtPixelError(Exception):
    """No high-quality track could be built; ``stage`` names the step that said so.

    ``reason`` is the sentence a person would be shown. ``diagnostics`` carries
    whatever the candidate measured on the way, so a failure can be read the
    same way a success is.
    """

    def __init__(self, stage: str, reason: str, diagnostics: dict | None = None):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.diagnostics = diagnostics or {}


@dataclass
class TrackAtPixelResult:
    """A built track, plus the trail of how it was built.

    ``track`` is a track-stage ``sfmtool._sfmtool.bench.EditableTrack`` that has
    been evaluated, so every ``in`` observation carries its leave-one-out
    ``zncc`` and friends. ``query_observation`` is the index of the observation
    that sits in the queried image (``None`` if the candidate moved off it).
    """

    track: Any
    query_observation: int | None
    diagnostics: dict = field(default_factory=dict)
