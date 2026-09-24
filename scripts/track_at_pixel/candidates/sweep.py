# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep: place the patch on the neighbours' surface and let the photographs correct it.

No descriptor is read. The reconstruction's own points around the pixel say
what surface the pixel is most likely on, so the candidate guesses the patch
from them and projects it:

1. **Surface hypotheses.** The observations near the pixel in the queried image
   are grouped into depth modes (one surface, or the two sides of an edge). For
   each mode, a plane through its points' centroid with their weighted mean
   normal is met by the pixel's ray; that is one hypothesis of where the patch
   is, with the mode's normal and apparent size.
2. **Projection.** Each hypothesis is projected into every photograph it
   plausibly shows in (inside the frame, facing the camera, not behind the
   reconstruction's own surface there), and the views that see it most nearly
   face-on become its first sightings.
3. **Fit.** The sightings are upgraded to a track, the patch is tilted and sized
   to the hypothesis, and the fit localizes every view against the others,
   anchored on the pixel. The photographs correct the depth: a guess a few
   pixels off in another view is within the localizer's reach.
4. The hypothesis whose track reads best (kept views times median ZNCC) is
   finished by :func:`candidates.common.finish`: grown by geometry search and
   gated.
"""

from __future__ import annotations

import numpy as np

from api import TrackAtPixelError
from candidates.common import (
    FINISH_DEFAULTS,
    anchored_fit,
    finish,
    score_track,
    shape_track,
    track_from_sightings,
    visible_views,
)

DEFAULTS = {
    **FINISH_DEFAULTS,
    "prior_radius_px": 40.0,
    "prior_k": 10,
    "default_radius_px": 8.0,
    "min_radius_px": 4.0,
    "max_radius_px": 40.0,
    "depth_mode_gap": 1.15,
    "min_mode_size": 2,
    "seed_views": 5,
    "max_view_angle_deg": 70.0,
}


def surface_hypotheses(ctx, image: int, pixel, opts: dict) -> list[dict]:
    """One hypothesis per depth mode of the observations near the pixel."""
    cam = ctx.camera(image)
    ray = cam.ray(pixel)
    center = cam.center
    near = [
        o
        for o in ctx.observations_near(image, pixel, opts["prior_radius_px"])
        if not o["at_infinity"] and o["depth"] and o["depth"] > 0
    ]
    if not near:
        return []
    by_depth = sorted(near, key=lambda o: o["depth"])
    modes, cur = [], [by_depth[0]]
    for o in by_depth[1:]:
        if o["depth"] / cur[-1]["depth"] > opts["depth_mode_gap"]:
            modes.append(cur)
            cur = []
        cur.append(o)
    modes.append(cur)
    ds = ctx.dataset
    out = []
    for mode in modes:
        if len(mode) < opts["min_mode_size"] and len(modes) > 1:
            continue
        same = sorted(mode, key=lambda o: o["distance_px"])[: opts["prior_k"]]
        weights = np.asarray([1.0 / (1.0 + o["distance_px"]) for o in same])
        positions = np.asarray([ds.point_xyz[o["point"]] for o in same])
        normals = np.asarray([o["normal"] for o in same])
        normal = (weights[:, None] * normals).sum(0)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        if normal @ (center - positions.mean(0)) < 0:
            normal = -normal
        centroid = (weights[:, None] * positions).sum(0) / weights.sum()
        denom = float(normal @ ray)
        if abs(denom) > 0.15:
            t = float(normal @ (centroid - center)) / denom
        else:
            t = float(np.linalg.norm(centroid - center))
        if t <= 0:
            continue
        xyz = center + t * ray
        depth = cam.depth(xyz)
        half_px = np.median(
            [o["half_px"] for o in same if np.isfinite(o["half_px"])]
            or [opts["default_radius_px"]]
        )
        half_px = float(np.clip(half_px, opts["min_radius_px"], opts["max_radius_px"]))
        half_px *= opts["size_scale"]
        out.append(
            {
                "xyz": xyz,
                "normal": normal,
                "half_px": half_px,
                "half": half_px * depth / cam.focal,
                "support": len(mode),
                "nearest_px": float(same[0]["distance_px"]),
            }
        )
    out.sort(key=lambda h: h["nearest_px"])
    return out


def build_track(ctx, image: int, pixel, options: dict | None = None):
    from sfmtool._sfmtool import bench as B

    opts = {**DEFAULTS, **(options or {})}
    pixel = (float(pixel[0]), float(pixel[1]))
    diag: dict = {}
    hyps = surface_hypotheses(ctx, image, pixel, opts)
    diag["hypotheses"] = [
        {
            "support": h["support"],
            "half_px": h["half_px"],
            "nearest_px": h["nearest_px"],
        }
        for h in hyps
    ]
    if not hyps:
        raise TrackAtPixelError(
            "prior",
            "no reconstructed point near the pixel says what surface it is on",
            diag,
        )
    best, best_score, tried = None, -1.0, []
    for h in hyps:
        views = visible_views(
            ctx,
            h["xyz"],
            h["normal"],
            exclude={image},
            max_view_angle_deg=opts["max_view_angle_deg"],
        )[: opts["seed_views"]]
        record = {"views": [v[0] for v in views]}
        tried.append(record)
        if not views:
            record["error"] = "no photograph sees the hypothesis"
            continue
        try:
            track = track_from_sightings(
                ctx, image, pixel, h["half_px"], [(v[0], v[1]) for v in views]
            )
            track = shape_track(ctx, track, 0, pixel, h["normal"], h["half"])
            track = anchored_fit(ctx, track, 0, pixel, opts["anchor_refits"])
            track, _ = B.apply_thresholds(track)
        except ValueError as e:
            record["error"] = str(e)
            continue
        s = score_track(track)
        record["score"] = s
        if s > best_score:
            best, best_score = track, s
    diag["tried"] = tried
    if best is None:
        raise TrackAtPixelError(
            "hypothesis",
            "no surface hypothesis could be fitted in any photograph",
            diag,
        )
    return finish(ctx, best, 0, pixel, opts, diag)
