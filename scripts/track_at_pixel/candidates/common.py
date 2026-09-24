# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Steps several candidates share once they hold a track-stage track.

The candidates differ in how they find the other photographs' sightings of the
pixel. Once a track stands, what is left is the same for all of them: keep it
on the pixel asked about, grow it into the views it was not found in, and
decide whether it is good enough to return. That is :func:`finish`.

**Anchoring.** A track-stage fit localizes every sighting, the queried one
included, against the consensus of the others. On a patch with a stronger
feature beside the pixel the whole patch slides a few pixels toward it, in
every photograph at once: the sightings stay mutually consistent (they are the
same piece of surface, one patch-width over), but the track is no longer at the
pixel. :func:`anchor` slides the patch back across its own plane until its
centre in the queried photograph is the pixel again. Every sighting moves by
the same in-plane displacement, so a slide that was coherent is undone in all
views, and the reading after it scores the sightings where they now sit.

:func:`finish` runs, in order: an anchored fit; a tilt toward the neighbours'
normal, kept unless the reading drops; the geometry search from the queried
sighting and an anchored refit; the cleaning, which turns out the views that
disagree with the track's own geometry and refits; and the gates. A view the
geometry search added has no keypoint until a fit places it, which is why an
anchored fit always takes a fit that places one.
"""

from __future__ import annotations

import numpy as np

from api import TrackAtPixelError, TrackAtPixelResult

FINISH_DEFAULTS = {
    "anchor": True,
    # Refits after anchoring: each is followed by another anchor, so the fit
    # may refine the other sightings and the depth but not move the pixel.
    "anchor_refits": 1,
    # Photometric search over the normal after anchoring (see search_normal).
    "normal_search": False,
    # Tilt toward the neighbours' normal (see neighbour_normal) and keep the
    # tilt unless the reading falls by more than normal_prior_tolerance.
    "normal_prior": True,
    "normal_prior_radius_px": 40.0,
    "normal_prior_k": 10,
    "normal_prior_tolerance": 0.01,
    # Cleaning: an `in` view whose own peak sits further than clean_max_shift_px
    # from its keypoint, or whose keypoint sits further than
    # clean_max_projection_px from the point's projection, is turned out and
    # the track refit, up to clean_rounds times.
    "clean": True,
    "clean_max_shift_px": 1.5,
    "clean_max_projection_px": 1.5,
    "clean_rounds": 2,
    # Gate: the largest projection offset over the kept views.
    "max_projection_offset_px": 1.5,
    "geometry_search": True,
    "min_in_views": 3,
    "min_zncc_median": 0.8,
    "max_query_offset_px": 2.0,
}


def median_zncc(track) -> float:
    z = [
        o["track"]["zncc"]
        for o in track.observations
        if o["verdict"] == "in" and o.get("track", {}).get("zncc") is not None
    ]
    return float(np.median(z)) if z else float("-inf")


def query_offset(track, q: int, pixel) -> float | None:
    kp = track.observations[q].get("track", {}).get("keypoint")
    if kp is None:
        return None
    return float(np.linalg.norm(np.asarray(kp) - np.asarray(pixel, float)))


def anchor(ctx, track, q: int, pixel):
    """Slide the patch so its centre in observation ``q`` is ``pixel``, then read it."""
    from sfmtool._sfmtool import bench as B

    moved, _ = B.translate_patch_to_pixel(track, ctx.edited, q, list(pixel))
    moved, _ = B.evaluate(moved, ctx.edited, ctx.pyramids)
    return moved


def unplaced(track) -> bool:
    """Whether an ``in`` observation has no keypoint yet (a fit has not placed it)."""
    return any(
        o["verdict"] == "in" and o.get("track", {}).get("keypoint") is None
        for o in track.observations
    )


def anchored_fit(ctx, track, q: int, pixel, refits: int):
    """``refits`` rounds of fit-then-anchor, then an anchor; keeps the best reading."""
    from sfmtool._sfmtool import bench as B

    best = anchor(ctx, track, q, pixel)
    for _ in range(refits):
        try:
            fitted, _ = B.fit(best, ctx.edited, ctx.pyramids)
            fitted = anchor(ctx, fitted, q, pixel)
        except ValueError:
            break
        # A view the fit placed for the first time has no keypoint before it,
        # so the fit is taken whatever the reading says; otherwise only an
        # improvement is.
        if not unplaced(best) and median_zncc(fitted) < median_zncc(best):
            break
        best = fitted
    return best


def in_frame(ctx, image: int, px, margin: float = 2.0) -> bool:
    w, h = ctx.image_size(image)
    return margin <= px[0] < w - margin and margin <= px[1] < h - margin


def visible_views(
    ctx,
    xyz,
    normal=None,
    *,
    exclude=(),
    max_view_angle_deg: float = 75.0,
    occlusion_radius_px: float = 12.0,
    occlusion_ratio: float = 0.93,
) -> list[tuple[int, np.ndarray, float]]:
    """Images a surface point at ``xyz`` is plausibly seen in, with its pixel.

    A view is kept when the point projects inside the photograph, faces the
    camera within ``max_view_angle_deg`` (when a ``normal`` is given), and is
    not behind the reconstruction's own surface there: when the observations
    within ``occlusion_radius_px`` of the projection sit, at their median,
    nearer than ``occlusion_ratio`` of the point's depth, something else is in
    front of it. Returns ``(image, pixel, view_angle_deg)`` sorted by angle.
    """
    xyz = np.asarray(xyz, float)
    out = []
    cos_max = np.cos(np.radians(max_view_angle_deg))
    for image in range(len(ctx.dataset.cameras)):
        if image in exclude:
            continue
        cam = ctx.camera(image)
        px = cam.project(xyz)
        if px is None or not in_frame(ctx, image, px):
            continue
        to_cam = cam.center - xyz
        to_cam /= np.linalg.norm(to_cam)
        angle = 0.0
        if normal is not None:
            c = float(np.asarray(normal) @ to_cam)
            if c < cos_max:
                continue
            angle = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        depth = cam.depth(xyz)
        near = [
            o["depth"]
            for o in ctx.observations_near(image, px, occlusion_radius_px)
            if o["depth"]
        ]
        if near and np.median(near) < occlusion_ratio * depth:
            continue
        out.append((image, px, angle))
    out.sort(key=lambda v: v[2])
    return out


def track_from_sightings(ctx, image: int, pixel, radius_px: float, sightings):
    """A track-stage track from the queried pixel plus ``(image, pixel)`` sightings.

    Every sighting is set ``in`` and the cluster is upgraded without a cluster
    reading: the upgrade triangulates the sightings as given and runs the
    track-stage fit over the frame it builds. Observation 0 is the query.
    """
    from sfmtool._sfmtool import bench as B

    _, track = B.create_cluster(
        B.Bench(), image, ctx.image_stem(image), tuple(pixel), radius_px=radius_px
    )
    track, _ = B.set_verdict(track, 0, "in")
    for other, px in sightings:
        track, _ = B.add_observation(
            track, int(other), tuple(map(float, px)), provenance="sweep"
        )
        track, _ = B.set_verdict(track, track.observation_count - 1, "in")
    track, _ = B.set_stage(track, ctx.edited, ctx.pyramids, "track")
    return track


def shape_track(ctx, track, q: int, pixel, normal=None, half=None):
    """Tilt toward ``normal`` and resize to ``half``, keeping the centre on ``pixel``."""
    from sfmtool._sfmtool import bench as B

    if track.at_infinity:
        return track
    if normal is not None:
        normal = np.asarray(normal, float)
        to_cam = ctx.camera(int(track.observations[q]["image"])).center - np.asarray(
            track.position
        )
        if normal @ to_cam < 0:
            normal = -normal
        try:
            track, _ = B.tilt_patch(track, ctx.edited, tuple(normal))
        except ValueError:
            pass
    if half is not None and half > 0:
        try:
            track, _ = B.resize_patch(track, ctx.edited, float(half))
        except ValueError:
            pass
    return track


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def patch_normal(track) -> np.ndarray | None:
    placement = track.placement
    if placement is None:
        return None
    n = np.cross(placement["u_halfvec"], placement["v_halfvec"])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else None


def search_normal(ctx, track, q: int, pixel, steps_deg=(20.0, 10.0, 5.0)):
    """Coordinate search over the patch normal by photometric agreement.

    From the current normal, tilts of ``step`` degrees toward each of four
    tangent directions are tried; a tilt is kept when the anchored reading's
    median ZNCC improves, and the step halves when none does. The sightings are
    read, not refit, at each trial, so the search costs one reading per trial.
    """
    from sfmtool._sfmtool import bench as B

    if track.at_infinity:
        return track
    best, best_z = track, median_zncc(track)
    for step in steps_deg:
        improved = True
        while improved:
            improved = False
            n = patch_normal(best)
            if n is None:
                return best
            e1, e2 = _tangent_basis(n)
            t = np.tan(np.radians(step))
            for d in (e1, -e1, e2, -e2):
                try:
                    tilted, _ = B.tilt_patch(best, ctx.edited, tuple(n + t * d))
                    tilted = anchor(ctx, tilted, q, pixel)
                except ValueError:
                    continue
                z = median_zncc(tilted)
                if z > best_z + 1e-4:
                    best, best_z, improved = tilted, z, True
                    break
    return best


def score_track(track) -> float:
    """A candidate's own reading of how well a track stands: kept views, weighted by agreement."""
    z = median_zncc(track)
    if not np.isfinite(z):
        return 0.0
    return track.verdict_counts[0] * max(z, 0.0)


def _reading(o: dict, key: str) -> float:
    v = o.get("track", {}).get(key)
    return float(v) if v is not None else float("nan")


def max_projection_offset(track) -> float:
    vals = [
        _reading(o, "projection_offset_px")
        for o in track.observations
        if o["verdict"] == "in"
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else float("inf")


def clean(ctx, track, q: int, pixel, opts: dict, diag: dict):
    """Turn out the views that disagree with the track's own geometry, and refit."""
    from sfmtool._sfmtool import bench as B

    removed = []
    for _ in range(opts["clean_rounds"]):
        worst = []
        for i, o in enumerate(track.observations):
            if i == q or o["verdict"] != "in":
                continue
            shift = _reading(o, "seed_shift_px")
            off = _reading(o, "projection_offset_px")
            if (
                shift > opts["clean_max_shift_px"]
                or off > opts["clean_max_projection_px"]
            ):
                worst.append(i)
        if not worst:
            break
        for i in worst:
            track, _ = B.set_verdict(track, i, "out")
        removed.extend(int(track.observations[i]["image"]) for i in worst)
        if track.verdict_counts[0] < 2:
            break
        try:
            track = anchored_fit(ctx, track, q, pixel, opts["anchor_refits"])
        except ValueError:
            break
    diag["clean"] = {"removed_images": removed}
    return track


def neighbour_normal(ctx, image: int, pixel, depth: float, opts: dict):
    """Distance-weighted mean normal of the nearby points on the track's own surface.

    Only neighbours within 15% of ``depth`` (along the queried camera's axis)
    count, so across an edge the other side's normal is not averaged in.
    """
    near = [
        o
        for o in ctx.observations_near(image, pixel, opts["normal_prior_radius_px"])
        if o["depth"] and abs(o["depth"] / depth - 1.0) < 0.15
    ][: opts["normal_prior_k"]]
    if not near:
        return None
    w = np.asarray([1.0 / (1.0 + o["distance_px"]) for o in near])
    n = (w[:, None] * np.asarray([o["normal"] for o in near])).sum(0)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else None


def finish(ctx, track, q: int, pixel, opts: dict, diag: dict) -> TrackAtPixelResult:
    """Anchor, grow by geometry search, threshold, gate. ``track`` is track-stage."""
    from sfmtool._sfmtool import bench as B

    def fail(stage: str, reason: str):
        raise TrackAtPixelError(stage, reason, diag)

    if opts["anchor"]:
        try:
            track = anchored_fit(ctx, track, q, pixel, opts["anchor_refits"])
        except ValueError as e:
            fail("anchor", str(e))
        track, _ = B.apply_thresholds(track)
        diag["anchor"] = {"in": track.verdict_counts[0], "zncc": median_zncc(track)}

    if opts["normal_prior"] and not track.at_infinity:
        image = int(track.observations[q]["image"])
        depth = ctx.camera(image).depth(np.asarray(track.position))
        normal = neighbour_normal(ctx, image, pixel, depth, opts) if depth > 0 else None
        if normal is not None:
            before = median_zncc(track)
            try:
                tilted = shape_track(ctx, track, q, pixel, normal)
                tilted = anchored_fit(ctx, tilted, q, pixel, opts["anchor_refits"])
                tilted, _ = B.apply_thresholds(tilted)
                after = median_zncc(tilted)
                kept = after >= before - opts["normal_prior_tolerance"]
                if kept:
                    track = tilted
                diag["normal_prior"] = {"before": before, "after": after, "kept": kept}
            except ValueError as e:
                diag["normal_prior"] = {"error": str(e)}

    if opts["normal_search"] and track.verdict_counts[0] >= 3:
        before = median_zncc(track)
        track = search_normal(ctx, track, q, pixel)
        track, _ = B.apply_thresholds(track)
        diag["normal_search"] = {
            "zncc_before": before,
            "zncc_after": median_zncc(track),
        }

    if opts["geometry_search"] and track.verdict_counts[0] >= 2:
        try:
            grown, geo = B.search_geometry(track, q, ctx.edited, ctx.pyramids)
            diag["geometry_search"] = {
                "added": geo["added"],
                "self_agreement": geo["self_agreement"],
            }
            if geo["added"]:
                grown, _ = B.evaluate(grown, ctx.edited, ctx.pyramids)
                grown, _ = B.apply_thresholds(grown)
                if opts["anchor"]:
                    grown = anchored_fit(ctx, grown, q, pixel, opts["anchor_refits"])
                else:
                    grown, _ = B.fit(grown, ctx.edited, ctx.pyramids)
                track = grown
        except ValueError as e:
            diag["geometry_search"] = {"error": str(e)}

    track, _ = B.apply_thresholds(track)
    if opts["clean"]:
        track = clean(ctx, track, q, pixel, opts, diag)
    row = track.observations[q]
    offset = query_offset(track, q, pixel)
    n_in, zncc = track.verdict_counts[0], median_zncc(track)
    diag["final"] = {"in": n_in, "zncc_median": zncc, "query_offset_px": offset}
    if row["verdict"] != "in":
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
    worst = max_projection_offset(track)
    diag["final"]["max_projection_offset_px"] = worst
    if worst > opts["max_projection_offset_px"]:
        fail(
            "gate",
            f"a kept view sits {worst:.1f} px from the point's projection "
            f"(bar {opts['max_projection_offset_px']} px)",
        )
    return TrackAtPixelResult(track=track, query_observation=q, diagnostics=diag)
