# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline: the bench steps in the order a person would press them.

1. **Local prior** -- the reconstruction's observations near the pixel in the
   queried image give a patch size (their apparent half-width), a normal and
   the depth structure (one surface, or an edge between two).
2. **Cluster** at the pixel, sized from the prior.
3. **Constellation search** from the pixel, then a **lateral** search from the
   best-supported images it found, so a pixel with a thin neighbourhood of
   keypoints in its own image can borrow a richer one elsewhere.
4. **Cluster evaluation** and the thresholds' verdicts, with the queried
   sighting pinned ``in``.
5. **Upgrade** to the track stage (triangulate, frame, fit).
6. **Normal prior**: tilt toward the neighbours' normal and refit; keep
   whichever of the two reads better.
7. **Sampling ratio** (``size_policy``): optionally resize the patch so one
   chosen view's bitmap samples the image at ``texel_scale_target`` pixels per
   texel -- the view the patch appears largest in, the median view, or the
   smallest -- and refit. ``"prior"`` keeps the size the local prior gave.
8. **Geometry search** from the queried sighting, refit, thresholds, refit.
9. **Gates**: enough ``in`` views, a good median ZNCC, and the queried sighting
   still ``in`` and still on the pixel.
"""

from __future__ import annotations

import numpy as np

from api import TrackAtPixelError, TrackAtPixelResult

DEFAULTS = {
    "prior_radius_px": 40.0,  # neighbourhood read for the local prior
    "prior_k": 8,  # nearest observations the size/normal prior uses
    "default_radius_px": 8.0,  # patch half-width when no neighbour says
    "min_radius_px": 4.0,
    "max_radius_px": 40.0,
    "depth_mode_gap": 1.15,  # depth ratio that separates two surfaces
    "constellation_target": 50,  # keypoints the constellation should hold
    "min_inliers": 6,
    "lateral_searches": 2,
    "normal_prior": True,
    "geometry_search": True,
    # Which view's sampling ratio the patch is sized by: "prior" (no resize),
    # "largest_view", "median_view" or "smallest_view".
    "size_policy": "prior",
    "texel_scale_target": 1.0,  # image pixels per bitmap texel in that view
    "min_in_views": 3,
    "min_zncc_median": 0.8,
    "max_query_offset_px": 2.0,
}


def local_prior(ctx, image: int, pixel, opts: dict) -> dict:
    """Size, normal and depth structure the neighbours suggest."""
    near = ctx.observations_near(image, pixel, opts["prior_radius_px"])
    prior: dict = {"neighbours": len(near)}
    finite = [o for o in near if not o["at_infinity"] and o["depth"] and o["depth"] > 0]
    if not finite:
        return prior

    # Depth modes: sort, split where consecutive depths jump by the gap ratio.
    depths = np.sort([o["depth"] for o in finite])
    modes, start = [], 0
    for i in range(1, len(depths)):
        if depths[i] / depths[i - 1] > opts["depth_mode_gap"]:
            modes.append(depths[start:i])
            start = i
    modes.append(depths[start:])
    prior["depth_modes"] = [
        {"median": float(np.median(m)), "count": int(len(m))} for m in modes
    ]
    prior["edge"] = sum(len(m) >= 2 for m in modes) >= 2

    # The surface the pixel is most likely on is the nearest neighbour's; the
    # size and normal come from that neighbour's depth mode only.
    nearest_depth = finite[0]["depth"]
    mode = next(m for m in modes if m[0] <= nearest_depth <= m[-1])
    same = [o for o in finite if mode[0] <= o["depth"] <= mode[-1]][: opts["prior_k"]]
    prior["depth"] = float(np.median([o["depth"] for o in same]))
    half_px = [o["half_px"] for o in same if np.isfinite(o["half_px"])]
    if half_px:
        prior["half_px"] = float(np.median(half_px))
    weights = np.asarray([1.0 / (1.0 + o["distance_px"]) for o in same])
    normals = np.asarray([o["normal"] for o in same])
    mean = (weights[:, None] * normals).sum(0)
    if np.linalg.norm(mean) > 1e-9:
        mean /= np.linalg.norm(mean)
        prior["normal"] = mean
        prior["normal_spread_deg"] = float(
            np.degrees(np.arccos(np.clip(normals @ mean, -1, 1))).mean()
        )
    return prior


def _median_zncc(track) -> float:
    z = [
        o["track"]["zncc"]
        for o in track.observations
        if o["verdict"] == "in" and o.get("track", {}).get("zncc") is not None
    ]
    return float(np.median(z)) if z else float("-inf")


def _size_by_sampling_ratio(ctx, track, opts: dict, diag: dict):
    """Resize (about the centre) so the policy's view samples at the target.

    The sampling ratio is proportional to the half-extent, so one resize by
    ``target / ratio`` lands the chosen view on the target; the refit after it
    re-reads every view at the new size, and the change is kept only if the
    refit succeeds.
    """
    from sfmtool._sfmtool import bench as B

    scales = ctx.texel_scales(track)
    if not scales:
        diag["size_policy"] = {"error": "no in view projects the patch"}
        return track
    values = np.asarray([s["scale"] for s in scales.values()])
    pick = {
        "largest_view": values.max(),
        "median_view": float(np.median(values)),
        "smallest_view": values.min(),
    }.get(opts["size_policy"])
    if pick is None:
        raise ValueError(f"unknown size_policy {opts['size_policy']!r}")
    half = float(np.linalg.norm(track.placement["u_halfvec"]))
    new_half = half * opts["texel_scale_target"] / float(pick)
    record = {
        "policy": opts["size_policy"],
        "scales_before": values.tolist(),
        "half_before": half,
        "half_after": new_half,
        "zncc_before": _median_zncc(track),
    }
    try:
        sized, _ = B.resize_patch(track, ctx.edited, new_half)
        sized, _ = B.fit(sized, ctx.edited, ctx.pyramids)
    except ValueError as e:
        record["error"] = str(e)
        diag["size_policy"] = record
        return track
    record["zncc_after"] = _median_zncc(sized)
    diag["size_policy"] = record
    return sized


def build_track(
    ctx, image: int, pixel, options: dict | None = None
) -> TrackAtPixelResult:
    from sfmtool._sfmtool import bench as B
    from sfmtool._sfmtool.spatial import radius_for_feature_count

    opts = {**DEFAULTS, **(options or {})}
    pixel = (float(pixel[0]), float(pixel[1]))
    diag: dict = {}

    def fail(stage: str, reason: str):
        raise TrackAtPixelError(stage, reason, diag)

    # 1. Local prior.
    prior = local_prior(ctx, image, pixel, opts)
    diag["prior"] = {k: v for k, v in prior.items() if k != "normal"}
    radius_px = float(
        np.clip(
            prior.get("half_px", opts["default_radius_px"]),
            opts["min_radius_px"],
            opts["max_radius_px"],
        )
    )
    diag["radius_px"] = radius_px

    # 2. Cluster at the pixel; the queried sighting is pinned in.
    _, track = B.create_cluster(
        B.Bench(), image, ctx.image_stem(image), pixel, radius_px=radius_px
    )
    track, _ = B.set_verdict(track, 0, "in")

    # 3. Constellation search, then lateral searches from the best finds.
    xy, affine = ctx.keypoints(image)
    w, h = ctx.image_size(image)
    search_radius = float(
        radius_for_feature_count(w, h, len(xy), opts["constellation_target"])
    )
    diag["search_radius_px"] = search_radius
    track, found = B.search_descriptors(
        track,
        0,
        xy,
        affine,
        ctx.forest,
        radius_px=search_radius,
        min_inliers=opts["min_inliers"],
    )
    diag["constellation"] = {
        "keypoints": found["constellation"],
        "images": [(m["image"], m["inliers"]) for m in found["matches"]],
    }
    added = sorted(
        (m for m in found["matches"] if m["found"] == "added"),
        key=lambda m: -m["inliers"],
    )
    lateral = []
    for m in added[: opts["lateral_searches"]]:
        other = m["image"]
        oxy, oaffine = ctx.keypoints(other)
        ow, oh = ctx.image_size(other)
        try:
            track, more = B.search_descriptors(
                track,
                m["observation"],
                oxy,
                oaffine,
                ctx.forest,
                radius_px=float(
                    radius_for_feature_count(
                        ow, oh, len(oxy), opts["constellation_target"]
                    )
                ),
                min_inliers=opts["min_inliers"],
            )
            lateral.append({"from": other, "added": more["added"]})
        except ValueError as e:
            lateral.append({"from": other, "error": str(e)})
    diag["lateral"] = lateral
    if track.observation_count < 2:
        fail(
            "constellation",
            f"the constellation of {found['constellation']} keypoints within "
            f"{search_radius:.0f} px matched no other image with "
            f"{opts['min_inliers']}+ inliers",
        )

    # 4. Cluster evaluation and the thresholds' verdicts.
    try:
        track, _ = B.evaluate(track, ctx.edited, ctx.pyramids)
    except ValueError as e:
        fail("cluster evaluate", str(e))
    track, _ = B.apply_thresholds(track)
    diag["cluster"] = {
        "observations": track.observation_count,
        "in": track.verdict_counts[0],
        "statuses": [o.get("cluster", {}).get("status") for o in track.observations],
    }
    if track.verdict_counts[0] < 2:
        fail(
            "cluster evaluate",
            f"{track.observation_count - 1} candidate sighting(s) found, none "
            "registered against the queried patch well enough to keep",
        )

    # 5. Upgrade to the track stage.
    try:
        track, staged = B.set_stage(track, ctx.edited, ctx.pyramids, "track")
    except ValueError as e:
        fail("upgrade", str(e))
    classification = (staged.get("fit") or {}).get("classification") or {}
    diag["upgrade"] = {
        "at_infinity": bool(track.at_infinity),
        "reason": classification.get("reason"),
        "zncc_median": _median_zncc(track),
    }

    # 6. Normal prior: tilt, refit, keep the better reading.
    if opts["normal_prior"] and "normal" in prior and not track.at_infinity:
        normal = np.asarray(prior["normal"], float)
        to_cam = ctx.camera(image).center - np.asarray(track.position)
        if normal @ to_cam < 0:
            normal = -normal
        try:
            tilted, tilt = B.tilt_patch(track, ctx.edited, tuple(normal))
            tilted, _ = B.fit(tilted, ctx.edited, ctx.pyramids)
            before, after = _median_zncc(track), _median_zncc(tilted)
            diag["normal_prior"] = {
                "degrees": tilt["degrees"],
                "stopped": tilt["stopped"] is not None,
                "zncc_before": before,
                "zncc_after": after,
                "kept": after >= before,
            }
            if after >= before:
                track = tilted
        except ValueError as e:
            diag["normal_prior"] = {"error": str(e)}

    # 7. Sampling ratio: resize so the chosen view samples at the target.
    if opts["size_policy"] != "prior":
        track = _size_by_sampling_ratio(ctx, track, opts, diag)

    # 8. Geometry search from the queried sighting, then refit.
    if opts["geometry_search"]:
        try:
            grown, geo = B.search_geometry(track, 0, ctx.edited, ctx.pyramids)
            diag["geometry_search"] = {
                "added": geo["added"],
                "self_agreement": geo["self_agreement"],
            }
            if geo["added"]:
                grown, _ = B.fit(grown, ctx.edited, ctx.pyramids)
                grown, painted = B.apply_thresholds(grown)
                grown, _ = B.fit(grown, ctx.edited, ctx.pyramids)
                track = grown
        except ValueError as e:
            diag["geometry_search"] = {"error": str(e)}

    # 9. Gates.
    track, _ = B.apply_thresholds(track)
    q = track.observations[0]
    kp = q.get("track", {}).get("keypoint")
    offset = float(np.linalg.norm(np.asarray(kp) - pixel)) if kp is not None else None
    n_in, zncc = track.verdict_counts[0], _median_zncc(track)
    diag["final"] = {"in": n_in, "zncc_median": zncc, "query_offset_px": offset}
    if q["verdict"] != "in":
        fail("gate", "the queried sighting did not survive the final thresholds")
    if offset is None or offset > opts["max_query_offset_px"]:
        fail(
            "gate",
            f"the fitted sighting sits {offset if offset is not None else float('nan'):.1f} px "
            f"from the pixel asked about (bar {opts['max_query_offset_px']} px)",
        )
    if n_in < opts["min_in_views"]:
        fail("gate", f"only {n_in} views kept (bar {opts['min_in_views']})")
    if zncc < opts["min_zncc_median"]:
        fail("gate", f"median ZNCC {zncc:.3f} is under {opts['min_zncc_median']}")
    return TrackAtPixelResult(track=track, query_observation=0, diagnostics=diag)
