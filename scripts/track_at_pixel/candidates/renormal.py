# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Renormal: the core cascade's track, with better fallbacks and a better normal.

The core cascade (``candidates/core_cascade.py``, the Rust operation) finds the
sightings. This candidate changes three things around it, all measured in the
harness on seoul_bull and the Kerry Park candidate ground truth:

1. **Gates.** Two views and a median ZNCC of 0.7 are enough to return a track
   (``core_options``, and ``min_in_views`` / ``min_zncc_median`` here), where
   the finish asks for three and 0.8. Most of the cascade's refusals were
   "only one view kept past the query" or "only two", and many of those
   tracks meet the good-track bar.
2. **Fallbacks.** When the cascade refuses, ``fallback`` tries, in order, the
   Python ``clusters`` member at 1.5 and 2 times its patch size and then
   ``planesweep`` at 1 and 1.5 times, each with the ray consensus. They only
   run on refused queries, so they cost little at the median; ``planesweep``
   needs only the poses and the photographs, so it is what rescues the empty
   pass.
3. **The normal.** The returned track is re-oriented by the first ``source``
   in a chain that gives a normal, tilted, given an anchored fit, thresholded,
   cleaned and gated with the finish's gates; it replaces the cascade's track
   when it passes them (``keep``):

   - ``nbchain``: the reconstruction's observations near the pixel on the
     track's own surface, from a tight neighbourhood to a wide one
     (``nb_chain``): 10 px within 5% of the track's depth, then 20 px within
     10%, 30 px within 15%, 40 px within 30%. The first neighbourhood holding a
     neighbour gives the principal axis of its normals, weighted
     ``1/(1+d)^2``. Tight first, because a close neighbour on the same surface
     is a much better guide than the average of a wide disc.
   - ``nb3d``: the mean normal of the ``nb3d_k`` points nearest the track in
     3D, for a pixel with no image-space neighbour at its depth.
   - ``photo``: ``PatchCloud.refine_normals`` over the track's ``in`` views,
     started from the mean direction to their cameras, over a 45 degree range
     with a small fronto-parallel prior, on a patch ``photo_scale`` (2) times
     the track's half-size. This is the only source in the empty pass.

   The other sources (``nb2d``, ``nb2d_principal``, ``photo_from_nb``) and
   options (``challenger``, ``regrow``, ``photo_scales``, ``photo_views``)
   are the variants the harness measured and did not keep.
"""

from __future__ import annotations

import importlib

import numpy as np

from api import TrackAtPixelError, TrackAtPixelResult
from candidates import core_cascade
from candidates.common import (
    FINISH_DEFAULTS,
    anchored_fit,
    clean,
    max_projection_offset,
    median_zncc,
    patch_normal,
    query_offset,
    score_track,
    shape_track,
    visible_views,
)

_FALLBACK_GATES = {"min_in_views": 2, "min_zncc_median": 0.7, "ray_consensus": True}

DEFAULTS = {
    **FINISH_DEFAULTS,
    # The gates: two views and the good-track bar's ZNCC (the finish's own
    # defaults are three and 0.8), here and in the core cascade.
    "min_in_views": 2,
    "min_zncc_median": 0.7,
    "source": "nbchain+nb3d+photo",
    "nb_radius_px": 20.0,
    "nb_k": 5,
    "nb_depth_tol": 0.15,
    "nb_fallback_radius_px": 40.0,
    "nb3d_k": 6,
    "nb_chain": [
        [10.0, 5, 0.05, "inv2", "principal"],
        [20.0, 5, 0.10, "inv2", "principal"],
        [30.0, 8, 0.15, "inv2", "principal"],
        [40.0, 12, 0.30, "inv2", "principal"],
    ],
    "photo_scale": 2.0,
    # When set, the photometric normal is estimated at each of these scales and
    # the results averaged.
    "photo_scales": None,
    "photo_range_deg": 45.0,
    "photo_views": "in",
    "photo_iters": 1,
    # Where the photometric search starts when there is no neighbours' normal:
    # the track's own normal, or the mean direction to its `in` cameras.
    "photo_init": "mean_view",
    "photo_max_view_angle_deg": 75.0,
    "photo_kwargs": {"fronto_prior_weight": 0.05},
    "keep": "force",
    "tolerance": 0.01,
    "clean_after": True,
    # Passed to the core cascade (see candidates/core_cascade.py).
    "core_options": {"finish.min_in_views": 2, "finish.min_zncc_median": 0.7},
    # (candidate module, options) tried in turn when the core cascade refuses.
    "fallback": [
        ["clusters", {**_FALLBACK_GATES, "size_scale": 1.5}],
        ["clusters", {**_FALLBACK_GATES, "size_scale": 2.0}],
        ["planesweep", _FALLBACK_GATES],
        ["planesweep", {**_FALLBACK_GATES, "size_scale": 1.5}],
    ],
    # The candidate whose track is re-oriented (core_cascade takes core_options).
    "base": "core_cascade",
    "base_options": {},
    # After the tilt, run the geometry search again and refit.
    "regrow": False,
    # Depth probe (see depth_probe): the factors along the ray tried when the
    # track has fewer than probe_below_views `in` views; a probe replaces the
    # track when its score is over probe_margin times the track's.
    "probe_scales": [],
    "probe_below_views": 6,
    "probe_margin": 1.0,
    # (candidate module, options) also run when the core cascade's track has
    # fewer than challenge_below_views `in` views; the higher score_track wins.
    "challenger": None,
    "challenge_below_views": 5,
}


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def _face(n, to_cam):
    if n is None:
        return None
    return n if n @ to_cam >= 0 else -n


def _tangent(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(n, a))
    return e1, np.cross(n, e1)


def neighbours_2d(ctx, image, pixel, depth, opts):
    for radius in (opts["nb_radius_px"], opts["nb_fallback_radius_px"]):
        near = [
            o
            for o in ctx.observations_near(image, pixel, radius)
            if o["depth"] and abs(o["depth"] / depth - 1.0) < opts["nb_depth_tol"]
        ][: opts["nb_k"]]
        if near:
            return near
    return []


def nb2d_normal(ctx, image, pixel, depth, to_cam, opts, principal=False):
    near = neighbours_2d(ctx, image, pixel, depth, opts)
    if not near:
        return None
    w = np.asarray([1.0 / (1.0 + o["distance_px"]) for o in near])
    ns = [_face(o["normal"], to_cam) for o in near]
    if principal:
        M = sum(wi * np.outer(n, n) for n, wi in zip(ns, w))
        return _face(np.linalg.eigh(M)[1][:, -1], to_cam)
    return _face(_unit((w[:, None] * ns).sum(0)), to_cam)


def nb_chain_normal(ctx, image, pixel, depth, to_cam, opts):
    """The first neighbourhood of ``nb_chain`` that holds a neighbour gives the normal.

    Each link is ``[radius_px, k, depth_tol, weighting, mode]``: the ``k``
    observations nearest the pixel within ``radius_px`` whose depth is within
    ``depth_tol`` of the track's, weighted ``1/(1+d)`` (``inv``) or
    ``1/(1+d)^2`` (``inv2``), combined by weighted ``mean`` or as the
    ``principal`` axis of the weighted normal outer products. The chain runs
    from tight to wide, so a pixel with close neighbours on its own surface is
    not averaged with farther ones.
    """
    widest = max(link[0] for link in opts["nb_chain"])
    allnear = [o for o in ctx.observations_near(image, pixel, widest) if o["depth"]]
    for radius, k, tol, weighting, mode in opts["nb_chain"]:
        near = [
            o
            for o in allnear
            if o["distance_px"] <= radius and abs(o["depth"] / depth - 1.0) < tol
        ][:k]
        if not near:
            continue
        d = np.asarray([o["distance_px"] for o in near])
        w = 1.0 / (1.0 + d) if weighting == "inv" else 1.0 / (1.0 + d) ** 2
        ns = np.asarray([_face(o["normal"], to_cam) for o in near])
        if mode == "principal":
            M = (w[:, None, None] * ns[:, :, None] * ns[:, None, :]).sum(0)
            return _face(np.linalg.eigh(M)[1][:, -1], to_cam)
        return _face(_unit((w[:, None] * ns).sum(0)), to_cam)
    return None


def nb3d_normal(ctx, xyz, half, to_cam, opts):
    near = ctx.points_near(xyz, k=opts["nb3d_k"])
    if not near:
        return None
    w = np.asarray([1.0 / (1.0 + o["distance"] / max(half, 1e-9)) for o in near])
    ns = [_face(o["normal"], to_cam) for o in near]
    return _face(_unit((w[:, None] * ns).sum(0)), to_cam)


def camera_views(ctx):
    from sfmtool import patches

    ds = ctx.dataset
    views = getattr(ds, "_camera_views", None)
    if views is None:
        rec = ds.recon
        views = patches.CameraViews(
            list(rec.cameras),
            np.asarray(rec.quaternions_wxyz, float),
            np.asarray(rec.translations, float),
            np.asarray(rec.camera_indexes, np.uint32),
        )
        ds._camera_views = views
    return views


def photo_normal(ctx, track, init, to_cam, opts, **kw):
    """The photometric normal; with several ``photo_scales``, their weighted mean."""
    scales = opts["photo_scales"] or [opts["photo_scale"]]
    found = []
    for scale in scales:
        n = _photo_normal_at(
            ctx, track, init, to_cam, {**opts, "photo_scale": scale}, **kw
        )
        if n is not None:
            found.append(n)
    if not found:
        return None
    return _face(_unit(sum(found)), to_cam)


def _photo_normal_at(ctx, track, init, to_cam, opts, **kw):
    from sfmtool import patches

    placement = track.placement
    half = float(np.linalg.norm(placement["u_halfvec"])) * opts["photo_scale"]
    images = [
        int(o["image"])
        for o in track.observations
        if opts["photo_views"] == "all" or o["verdict"] == "in"
    ]
    if opts["photo_views"] == "visible":
        # Every photograph the patch projects into and faces, not only the
        # track's sightings: more views constrain the normal better.
        seen = visible_views(
            ctx,
            placement["center"],
            None,
            max_view_angle_deg=opts["photo_max_view_angle_deg"],
        )
        images = sorted(set(images) | {int(i) for i, _, _ in seen})
    if len(images) < 2:
        return None
    n = init
    for it in range(opts["photo_iters"]):
        e1, e2 = _tangent(n)
        cloud = patches.PatchCloud.from_halfvec_arrays(
            np.asarray([e1 * half], np.float32),
            np.asarray([e2 * half], np.float32),
            np.asarray([placement["center"]], np.float64),
        )
        try:
            res = cloud.refine_normals(
                camera_views(ctx),
                ctx.pyramids,
                view_indices=[images],
                use_stored_keypoints=False,
                min_views=2,
                angular_range_deg=opts["photo_range_deg"] if it == 0 else 20.0,
                **{**opts["photo_kwargs"], **kw},
            )
        except (ValueError, RuntimeError):
            return None
        found = _unit(res["normal"][0])
        if found is None or not np.all(np.isfinite(found)):
            return None
        n = _face(found, to_cam)
    return n


def estimate(ctx, track, image, pixel, opts):
    xyz = np.asarray(track.position, float)
    cam = ctx.camera(image)
    to_cam = _unit(cam.center - xyz)
    depth = cam.depth(xyz)
    current = _face(patch_normal(track), to_cam)
    if opts["photo_init"] == "mean_view":
        dirs = [
            _unit(ctx.camera(int(o["image"])).center - xyz)
            for o in track.observations
            if o["verdict"] == "in"
        ]
        current = _face(_unit(sum(dirs)), to_cam)
    half = float(np.linalg.norm(track.placement["u_halfvec"]))
    # A chain "a+b+c": the first source that gives a normal wins, except that
    # "photo_from_nb" starts the photometric search from the neighbours' normal.
    for src in opts["source"].split("+"):
        if src == "none":
            return None
        if src == "nb2d":
            n = nb2d_normal(ctx, image, pixel, depth, to_cam, opts)
        elif src == "nb2d_principal":
            n = nb2d_normal(ctx, image, pixel, depth, to_cam, opts, principal=True)
        elif src == "nbchain":
            n = nb_chain_normal(ctx, image, pixel, depth, to_cam, opts)
        elif src == "nb3d":
            n = nb3d_normal(ctx, xyz, half, to_cam, opts)
        elif src == "photo":
            n = photo_normal(ctx, track, current, to_cam, opts)
        elif src == "photo_from_nb":
            nb = nb2d_normal(ctx, image, pixel, depth, to_cam, opts)
            n = photo_normal(
                ctx, track, nb if nb is not None else current, to_cam, opts
            )
        else:
            raise ValueError(f"unknown source {src!r}")
        if n is not None:
            return n
    return None


def depth_probe(ctx, track, q, pixel, opts, info):
    """Try the track at other depths along the queried pixel's ray, and grow it there.

    A track found in a few photographs close together fixes its depth badly,
    and the geometry search from a wrong depth projects the patch beside the
    surface in the photographs far away, so the track never reaches the views
    that would fix it. For each factor in ``probe_scales`` the patch is moved
    along the ray to that multiple of its distance from the queried camera,
    anchored on the pixel, grown by the geometry search, refit and cleaned.
    The one that passes the gates with the highest ``score_track`` wins,
    the track as it stood included.
    """
    from sfmtool import bench as B

    cam = ctx.camera(int(track.observations[q]["image"]))
    best, best_score = track, score_track(track)
    tried = []
    for factor in opts["probe_scales"]:
        try:
            placement = track.placement
            u = _unit(np.asarray(placement["u_halfvec"], float))
            v = _unit(np.asarray(placement["v_halfvec"], float))
            n = _unit(np.cross(u, v))
            d = (factor - 1.0) * (np.asarray(track.position, float) - cam.center)
            moved, _ = B.translate_patch(
                track, ctx.edited, (float(d @ u), float(d @ v), float(d @ n))
            )
            moved, _ = B.translate_patch_to_pixel(moved, ctx.edited, q, list(pixel))
            moved, _ = B.evaluate(moved, ctx.edited, ctx.pyramids)
            grown, geo = B.search_geometry(moved, q, ctx.edited, ctx.pyramids)
            grown, _ = B.evaluate(grown, ctx.edited, ctx.pyramids)
            grown, _ = B.apply_thresholds(grown)
            if grown.verdict_counts[0] < 2:
                tried.append({"factor": factor, "in": grown.verdict_counts[0]})
                continue
            grown = anchored_fit(ctx, grown, q, pixel, opts["anchor_refits"])
            grown, _ = B.apply_thresholds(grown)
            grown = clean(ctx, grown, q, pixel, opts, {})
        except ValueError as e:
            tried.append({"factor": factor, "error": str(e)})
            continue
        sc = score_track(grown)
        ok = passes_gates(grown, q, pixel, opts)
        tried.append(
            {"factor": factor, "in": grown.verdict_counts[0], "score": sc, "gates": ok}
        )
        if ok and sc > best_score * opts["probe_margin"]:
            best, best_score = grown, sc
    info["depth_probe"] = {"tried": tried, "moved": best is not track}
    return best


def passes_gates(track, q, pixel, opts) -> bool:
    if track.observations[q]["verdict"] != "in":
        return False
    off = query_offset(track, q, pixel)
    if off is None or off > opts["max_query_offset_px"]:
        return False
    if track.verdict_counts[0] < opts["min_in_views"]:
        return False
    if median_zncc(track) < opts["min_zncc_median"]:
        return False
    return max_projection_offset(track) <= opts["max_projection_offset_px"]


def build_track(ctx, image: int, pixel, options: dict | None = None):
    from sfmtool import bench as B

    opts = {**DEFAULTS, **(options or {})}
    try:
        if opts["base"] == "core_cascade":
            result = core_cascade.build_track(
                ctx, image, pixel, {"core_options": opts["core_options"]}
            )
        else:
            base = importlib.import_module(f"candidates.{opts['base']}")
            result = base.build_track(ctx, image, pixel, opts["base_options"])
    except TrackAtPixelError as refusal:
        result = None
        for name, member_opts in opts["fallback"]:
            module = importlib.import_module(f"candidates.{name}")
            try:
                result = module.build_track(ctx, image, pixel, member_opts)
            except TrackAtPixelError:
                continue
            result.diagnostics = {"fallback": name, **result.diagnostics}
            break
        if result is None:
            raise refusal from None
    if (
        opts["challenger"]
        and "fallback" not in result.diagnostics
        and result.track.verdict_counts[0] < opts["challenge_below_views"]
    ):
        name, member_opts = opts["challenger"]
        module = importlib.import_module(f"candidates.{name}")
        try:
            other = module.build_track(ctx, image, pixel, member_opts)
        except TrackAtPixelError:
            other = None
        info = {"ran": name, "own": score_track(result.track)}
        if other is not None:
            info["challenger"] = score_track(other.track)
            if info["challenger"] > info["own"]:
                other.diagnostics = {"challenger_won": name, **other.diagnostics}
                result = other
        result.diagnostics["challenge"] = info
    track, q, diag = result.track, result.query_observation, result.diagnostics
    if track.at_infinity or opts["source"] == "none":
        return result
    normal = estimate(ctx, track, image, pixel, opts)
    info = {"estimated": normal is not None}
    diag["renormal"] = info
    if normal is None:
        return result
    before = patch_normal(track)
    info["tilt_deg"] = float(
        np.degrees(np.arccos(np.clip(abs(float(normal @ before)), -1, 1)))
    )
    try:
        tilted = shape_track(ctx, track, q, pixel, normal)
        tilted = anchored_fit(ctx, tilted, q, pixel, opts["anchor_refits"])
        tilted, _ = B.apply_thresholds(tilted)
        if opts["regrow"] and tilted.verdict_counts[0] >= 2:
            grown, geo = B.search_geometry(tilted, q, ctx.edited, ctx.pyramids)
            info["regrow_added"] = geo["added"]
            if geo["added"]:
                grown, _ = B.evaluate(grown, ctx.edited, ctx.pyramids)
                grown, _ = B.apply_thresholds(grown)
                grown = anchored_fit(ctx, grown, q, pixel, opts["anchor_refits"])
                tilted, _ = B.apply_thresholds(grown)
        if opts["clean_after"]:
            tilted = clean(ctx, tilted, q, pixel, opts, info)
    except ValueError as e:
        info["error"] = str(e)
        return result
    final = patch_normal(tilted)
    if final is not None:
        info["final_vs_estimate_deg"] = float(
            np.degrees(np.arccos(np.clip(abs(float(final @ normal)), -1, 1)))
        )
    ok = passes_gates(tilted, q, pixel, opts)
    z0, z1 = median_zncc(track), median_zncc(tilted)
    info.update(zncc_before=z0, zncc_after=z1, gates=ok)
    if ok and (opts["keep"] == "force" or z1 >= z0 - opts["tolerance"]):
        info["kept"] = True
        out = tilted
    else:
        info["kept"] = False
        out = track
    if (
        opts["probe_scales"]
        and not out.at_infinity
        and out.verdict_counts[0] < opts["probe_below_views"]
    ):
        out = depth_probe(ctx, out, q, pixel, opts, info)
    info["query_shift_px"] = out.observations[q].get("track", {}).get("seed_shift_px")
    return TrackAtPixelResult(track=out, query_observation=q, diagnostics=diag)


__all__ = ["build_track", "DEFAULTS", "TrackAtPixelError"]
