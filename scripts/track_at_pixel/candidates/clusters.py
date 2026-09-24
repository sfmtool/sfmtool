# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Clusters: read the correspondences the cluster-patches ``.matches`` file already holds.

The file groups SIFT detections across the photographs into clusters and
refines each against a reference patch, so a cluster with a member near the
pixel is a ready-made set of sightings of nearly the same piece of surface. No
descriptor search and no geometry runs to find them.

1. **Nearby clusters.** The clusters with a member in the queried image within
   ``search_radius_px`` of the pixel, nearest first.
2. **Transfer.** The pixel is not at the member, so the offset from the member
   to the pixel is carried into each other member's image through the two
   members' affine shapes: ``shape_other @ inv(shape_query)`` maps a pixel step
   in the queried image to the matching step in the other. Only the members the
   refinement kept (and the reference) are used, one per image.
3. **Fit.** Those sightings are upgraded to a track and fitted, anchored on the
   pixel. The best-reading cluster is finished by
   :func:`candidates.common.finish`.
"""

from __future__ import annotations

import numpy as np

from api import TrackAtPixelError
from candidates.common import (
    FINISH_DEFAULTS,
    anchored_fit,
    finish,
    in_frame,
    score_track,
    track_from_sightings,
)

DEFAULTS = {
    **FINISH_DEFAULTS,
    "search_radius_px": 16.0,
    # How far from the member the pixel may be, in units of the member's own
    # patch scale (the square root of its shape's determinant).
    "max_offset_in_scales": 15.0,
    "max_clusters": 3,
    "use_statuses": ["reference", "kept"],
    "radius_in_scales": 2.0,  # patch half-width in member scales
    "min_radius_px": 4.0,
    "max_radius_px": 40.0,
}


def cluster_sightings(ctx, image: int, pixel, cluster: dict, opts: dict):
    """The pixel carried into every other image the cluster reaches."""
    m = cluster["member"]
    shape_q = np.asarray(m["shape"], float)
    det = abs(np.linalg.det(shape_q))
    if det <= 1e-12:
        return None, None
    scale = float(np.sqrt(det))
    offset = np.asarray(pixel, float) - np.asarray(m["position"], float)
    if np.linalg.norm(offset) > opts["max_offset_in_scales"] * scale:
        return None, None
    inv_q = np.linalg.inv(shape_q)
    by_image: dict[int, tuple[float, np.ndarray]] = {}
    for o in cluster["members"]:
        if o["image"] < 0 or o["image"] == image:
            continue
        if o["status"] not in opts["use_statuses"]:
            continue
        pred = np.asarray(o["position"], float) + np.asarray(o["shape"], float) @ (
            inv_q @ offset
        )
        if not in_frame(ctx, o["image"], pred):
            continue
        z = o["zncc"] if np.isfinite(o["zncc"]) else 1.0
        if o["image"] not in by_image or z > by_image[o["image"]][0]:
            by_image[o["image"]] = (z, pred)
    sightings = [(i, p) for i, (_, p) in by_image.items()]
    return sightings, scale


def build_track(ctx, image: int, pixel, options: dict | None = None):
    from sfmtool._sfmtool import bench as B

    opts = {**DEFAULTS, **(options or {})}
    pixel = (float(pixel[0]), float(pixel[1]))
    diag: dict = {}
    near = ctx.clusters_near(image, pixel, opts["search_radius_px"])
    diag["clusters_near"] = len(near)
    if not near:
        raise TrackAtPixelError(
            "clusters",
            f"no cluster has a member within {opts['search_radius_px']:.0f} px of the pixel",
            diag,
        )
    best, best_score, tried = None, -1.0, []
    for cluster in near[: opts["max_clusters"]]:
        record = {"cluster": cluster["cluster"], "distance_px": cluster["distance_px"]}
        tried.append(record)
        sightings, scale = cluster_sightings(ctx, image, pixel, cluster, opts)
        if not sightings:
            record["error"] = "no kept member in another image to carry the pixel to"
            continue
        radius_px = float(
            np.clip(
                opts["radius_in_scales"] * scale,
                opts["min_radius_px"],
                opts["max_radius_px"],
            )
        )
        record["sightings"] = len(sightings)
        try:
            track = track_from_sightings(ctx, image, pixel, radius_px, sightings)
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
            "clusters",
            "no nearby cluster carried the pixel into another photograph",
            diag,
        )
    return finish(ctx, best, 0, pixel, opts, diag)
