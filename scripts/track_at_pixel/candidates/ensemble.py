# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Ensemble: run every way of finding the sightings, and let them vote on the depth.

The cascade takes the first member whose own gates pass. A member's gates only
say that its track agrees with itself, so a wrong track that is self-consistent
is returned whenever it comes first, and a pixel no member reaches at its own
patch size is refused although a larger patch would have held.

Every track a member returns is anchored on the queried pixel, so it lies on
that pixel's ray: tracks can only differ in how far along it they put the
point. Two members that found their sightings from different evidence and
still agree on the depth are unlikely to be wrong in the same way. So:

1. **Pool.** Each member runs at each patch size in ``sizes`` (a multiplier on
   the size the member picks for itself), with the ray consensus in the finish
   (:func:`candidates.common.ray_consensus`). Every track that passes the
   member's own gates joins the pool.
2. **Groups.** Two tracks agree when their points are within the larger of the
   two patches' half-extents of each other (the good-track bar's own unit), or
   when both are bearings within ``bearing_tolerance_deg``. Each track's group
   is the tracks that agree with it.
3. **Vote.** The group supported by the most distinct members wins, then the
   group with the highest summed score (kept views times median ZNCC). The
   vote decides the depth; which of the group's tracks is returned is decided
   by the cascade's order, smallest patch first. The highest-scoring track is
   not taken: the score counts kept views, and the largest patches keep the
   most views, wrong ones included.
"""

from __future__ import annotations

import importlib

import numpy as np

from api import TrackAtPixelError
from candidates.common import score_track

DEFAULTS = {
    # (candidate module, options passed to it), in the cascade's order.
    "members": [
        ["clusters", {}],
        ["transfer", {}],
        ["sweep", {}],
        ["planesweep", {}],
        ["baseline", {"finish": "common"}],
    ],
    # Patch-size multipliers each member is run at. The baseline sizes its
    # patch from the descriptor, so it runs once.
    "sizes": [1.0, 1.5, 2.0],
    "sized": ["clusters", "transfer", "sweep", "planesweep"],
    # Options given to every member, under each member's own. The gates are
    # the good-track bar's own ZNCC and no view count past the pixel and one
    # other: the vote, not each member, is what refuses a wrong track.
    "shared": {"ray_consensus": True, "min_zncc_median": 0.7, "min_in_views": 2},
    "bearing_tolerance_deg": 0.2,
    # When set, the winning group's tracks whose queried view peaks within this
    # many pixels of the pixel are preferred over the rest (see query_shift).
    "prefer_centred_px": None,
}


def _half(track) -> float:
    p = track.placement
    return float(np.linalg.norm(p["u_halfvec"])) if p is not None else 0.0


def query_shift(result) -> float:
    """How far the queried view's correlation peak sits from its keypoint."""
    o = result.track.observations[result.query_observation]
    v = o.get("track", {}).get("seed_shift_px")
    return float(v) if v is not None else float("inf")


def agree(a, b, opts) -> bool:
    if a.at_infinity != b.at_infinity:
        return False
    if a.at_infinity:
        u, v = np.asarray(a.direction, float), np.asarray(b.direction, float)
        c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        return np.degrees(np.arccos(np.clip(c, -1, 1))) <= opts["bearing_tolerance_deg"]
    d = np.linalg.norm(np.asarray(a.position) - np.asarray(b.position))
    return d <= max(_half(a), _half(b))


def build_track(ctx, image: int, pixel, options: dict | None = None):
    opts = {**DEFAULTS, **(options or {})}
    pool, refusals = [], []
    for name, member_opts in opts["members"]:
        module = importlib.import_module(f"candidates.{name}")
        sizes = opts["sizes"] if name in opts["sized"] else [1.0]
        for size in sizes:
            run_opts = {**opts["shared"], **member_opts, "size_scale": size}
            try:
                result = module.build_track(ctx, image, pixel, run_opts)
            except TrackAtPixelError as e:
                refusals.append(
                    {"member": name, "size": size, "stage": e.stage, "reason": e.reason}
                )
                continue
            pool.append(
                {
                    "member": name,
                    "size": size,
                    "result": result,
                    "score": score_track(result.track),
                }
            )
    if not pool:
        last = refusals[-1] if refusals else None
        reason = (
            f"every member refused at every size; the last, {last['member']} at "
            f"{last['size']}x, at {last['stage']}: {last['reason']}"
            if last
            else "no members are configured"
        )
        raise TrackAtPixelError("ensemble", reason, {"refusals": refusals})
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
    names = [m for m, _ in opts["members"]]
    centred = opts["prefer_centred_px"]

    def rank(o):
        off = centred is not None and query_shift(o["result"]) > centred
        return (off, o["size"], names.index(o["member"]))

    chosen = min(best, key=rank)
    result = chosen["result"]
    result.diagnostics = {
        "member": chosen["member"],
        "size": chosen["size"],
        "pool": [(o["member"], o["size"], o["score"]) for o in pool],
        "support": [(o["member"], o["size"]) for o in best],
        "refusals": refusals,
        **result.diagnostics,
    }
    return result
