# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Core vote: every Rust cascade member runs on its own, and their tracks vote on the depth.

The cascade returns the first member's track that passes that member's own
gates, so a self-consistent wrong track from an early member (most often
``clusters``) is returned although a later member would have found the right
one. The ensemble (``candidates/ensemble.py``) showed that letting the members
vote on the depth fixes many of those, but it runs every Python member at three
sizes and is twenty times slower. This is the cheap part of it: each member of
the Rust operation runs once, alone, through ``bench.build_track_at_pixel``
with ``members=[member]``; the tracks that pass are grouped by
:func:`candidates.ensemble.agree` (points within the larger half-extent, or
bearings within ``bearing_tolerance_deg``); the group supported by the most
members wins, then the highest summed ``score_track``; and within it the
member earliest in the cascade's order is returned, preferring one whose
queried view peaks within ``prefer_centred_px`` of the pixel when that is set.

``extra`` adds Python candidates to the pool as further voters.
"""

from __future__ import annotations

import importlib

from api import TrackAtPixelError
from candidates import core_cascade
from candidates.common import score_track
from candidates.ensemble import agree, query_shift

DEFAULTS = {
    "members": ["clusters", "transfer", "sweep", "constellation"],
    "core_options": {"finish.min_in_views": 2, "finish.min_zncc_median": 0.7},
    # (candidate module, options) added to the pool.
    "extra": [],
    "bearing_tolerance_deg": 0.2,
    "prefer_centred_px": None,
    # When set, the members after the first that passes run only if its track
    # has fewer than this many `in` views (a long track is rarely wrong).
    "confirm_below_views": None,
}


def build_track(ctx, image: int, pixel, options: dict | None = None):
    opts = {**DEFAULTS, **(options or {})}
    pool, refusals = [], []
    voters = [("core", m) for m in opts["members"]] + [
        ("py", (name, o)) for name, o in opts["extra"]
    ]
    for kind, spec in voters:
        if (
            opts["confirm_below_views"] is not None
            and pool
            and pool[0]["result"].track.verdict_counts[0] >= opts["confirm_below_views"]
        ):
            break
        try:
            if kind == "core":
                name = spec
                result = core_cascade.build_track(
                    ctx,
                    image,
                    pixel,
                    {"members": [spec], "core_options": opts["core_options"]},
                )
            else:
                name, member_opts = spec
                module = importlib.import_module(f"candidates.{name}")
                result = module.build_track(ctx, image, pixel, member_opts)
        except TrackAtPixelError as e:
            refusals.append({"member": name, "stage": e.stage, "reason": e.reason})
            continue
        pool.append(
            {"member": name, "result": result, "score": score_track(result.track)}
        )
    if not pool:
        last = refusals[-1] if refusals else None
        reason = (
            f"every member refused; the last, {last['member']}, at "
            f"{last['stage']}: {last['reason']}"
            if last
            else "no members are configured"
        )
        raise TrackAtPixelError("vote", reason, {"refusals": refusals})
    best, best_key = None, None
    for entry in pool:
        group = [
            o
            for o in pool
            if o is entry or agree(entry["result"].track, o["result"].track, opts)
        ]
        key = (len({o["member"] for o in group}), sum(o["score"] for o in group))
        if best_key is None or key > best_key:
            best, best_key = group, key
    order = [o["member"] for o in pool]
    centred = opts["prefer_centred_px"]

    def rank(o):
        off = centred is not None and query_shift(o["result"]) > centred
        return (off, order.index(o["member"]))

    chosen = min(best, key=rank)
    result = chosen["result"]
    result.diagnostics = {
        "member": chosen["member"],
        "pool": [(o["member"], o["score"]) for o in pool],
        "support": [o["member"] for o in best],
        "refusals": refusals,
        **{k: v for k, v in result.diagnostics.items() if k != "member"},
    }
    return result
