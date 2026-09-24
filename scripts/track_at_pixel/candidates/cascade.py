# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Cascade: ask each way of finding the sightings in turn, most precise first.

The other candidates find the pixel's sightings from different evidence, and
they fail on different pixels: the cluster file only reaches pixels near a
cluster the refinement kept, the neighbour transfer needs enough reconstructed
points of one surface around the pixel, the sweep needs any at all, and the
descriptor search needs a constellation that matches. Each one's gates are its
own judgement of whether it succeeded, so the first to return a track is taken
and the rest are not run.

The order is by how rarely each is wrong when it does return a track:

1. ``clusters`` -- correspondences the cluster refinement already verified.
2. ``transfer`` -- the neighbours' own matched keypoints.
3. ``sweep`` -- a surface guess the photographs have to confirm.
4. ``baseline`` with the shared finish -- the descriptor constellation.

A refusal from every member is reported with each member's stage and reason.
"""

from __future__ import annotations

import importlib

from api import TrackAtPixelError

DEFAULTS = {
    # (candidate module, options passed to it)
    "members": [
        ["clusters", {}],
        ["transfer", {}],
        ["sweep", {}],
        ["baseline", {"finish": "common"}],
    ],
}


def build_track(ctx, image: int, pixel, options: dict | None = None):
    opts = {**DEFAULTS, **(options or {})}
    refusals = []
    for name, member_opts in opts["members"]:
        module = importlib.import_module(f"candidates.{name}")
        try:
            result = module.build_track(ctx, image, pixel, member_opts)
        except TrackAtPixelError as e:
            refusals.append({"member": name, "stage": e.stage, "reason": e.reason})
            continue
        result.diagnostics = {
            "member": name,
            "refusals": refusals,
            **result.diagnostics,
        }
        return result
    last = refusals[-1] if refusals else {"stage": "cascade", "reason": "no members"}
    raise TrackAtPixelError(
        "cascade",
        f"every member refused; the last, {last.get('member')}, at "
        f"{last['stage']}: {last['reason']}",
        {"refusals": refusals},
    )
