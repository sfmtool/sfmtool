# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Anchors: reinforced depth readings near a pixel, the first step of building a track.

Before a track is built at a pixel, the depth there is unknown. An **anchor**
is a 3D point near the pixel that several photographs agree on, with the
pixel it sits at in the queried image: a place to start from and walk toward
the pixel. :func:`find_anchors` looks for them in its sources, strongest
first, and stops once it has enough close to the pixel (``stop="enough"``) or
runs them all (``stop="never"``, what the harness's ``anchors`` mode uses so
each source is measured):

1. **Tracks** (``tracks``): the reconstruction's own points observed within
   ``track_radius_px`` of the pixel, finite, seen in ``track_min_views`` or
   more images, every observation within ``max_reproj_px`` of the point's
   projection. These are the strongest readings: a solver already agreed on
   them.
2. **Clusters** (``clusters``): the cluster-patches clusters with a member in
   the queried image within ``cluster_radius_px``, vetted by triangulation
   (:func:`vet_cluster`).
3. **Guided matching** (``guided``): the keypoints near the pixel matched by
   descriptor among the keypoints whose rays pass close to theirs
   (:func:`from_guided`).
4. **Constellation** (``constellation``): the SIFT index's constellation query
   from the pixel itself. Each image it matches carries the pixel into its
   own frame by the constellation's affine warp, and those sightings are
   triangulated, dropping the worst while three or more remain, within
   ``constellation_max_reproj_px``. Its anchor sits at the pixel.
5. **Infinity** (``infinity``): whether the pixel is further than the
   photographs can tell from infinity (:func:`from_infinity`), run after the
   others when they leave the pixel's depth open.

Each anchor carries the **range** of distances along its pixel's ray that its
views allow. Photographs a few metres apart looking at a point a hundred
metres away fix its distance only loosely, and two views from adjacent frames
may not bound it at all. An anchor is usable when its range is bounded, or has
no far end and starts well beyond the cameras' spread (a far reading). An
anchor is **supported** by each other usable anchor whose range overlaps its
own and whose photographs are not a subset of its own (or it of theirs): two
independent readings of one structure.

The scene near a pixel can hold surfaces at very different depths: a railing in
front, the pixel's surface, a skyline behind. An anchor near the pixel may be
real geometry of a different surface, so the anchors are hypotheses, grouped
into **layers** of overlapping ranges and ranked by what supports each: the
pixel's own patch read at the layer's distances, whole and in the middle
(:func:`read_patch`), with the anchors' own features beside it.

:func:`score_anchors` measures anchors against the ground truth, which
:func:`find_anchors` never sees; :func:`summarize` reports a pass.
"""

from __future__ import annotations

import time

import numpy as np

DEFAULTS = {
    "sources": "tracks+clusters+guided+constellation",
    # "enough": stop after the first source that leaves `min_anchors` anchors
    # within `enough_px` of the pixel; "never": run every source.
    "stop": "enough",
    "min_anchors": 2,
    "enough_px": 20.0,
    "track_radius_px": 40.0,
    "track_max": 8,
    "track_min_views": 2,
    "cluster_radius_px": 48.0,
    "cluster_max": 16,
    "constellation_target": 50,
    "constellation_min_inliers": 6,
    "constellation_radius_px": 6.0,
    "max_reproj_px": 2.0,
    # The constellation's sightings are affine predictions, not refined.
    "constellation_max_reproj_px": 3.0,
    # An anchor's range is where every view stays within `range_px` (or half a
    # pixel more than its own error); it is bounded when finite at both ends and
    # no wider than `max_span`, far over near.
    "range_px": 1.0,
    "max_span": 3.0,
    # A range with no far end is a far reading when its near end is at least
    # `far_spread` times the largest distance between two cameras.
    "far_spread": 5.0,
    # Infinity: after the other sources, when they gave no usable anchor, more
    # than one layer, or none at the pixel ("needed"), or always, or never. The pixel's patch is
    # compared with every image at its position at infinity; it is at infinity
    # when `inf_min_views` or more, and `inf_min_share` of the images it lands
    # in, read `inf_min_zncc` or better.
    "infinity": "needed",
    "inf_radius_px": 8.0,
    "inf_min_zncc": 0.8,
    "inf_min_views": 1,
    "inf_min_share": 0.5,
    # The agreeing images are then read at the distances that move the pixel
    # 1, 2, 4, ... up to `inf_max_shift_px` pixels from its position at
    # infinity in the image that moves it most. A finite distance that reads
    # `inf_peak_margin` better than infinity rejects the reading; the range's
    # near end is where the reading falls `inf_drop` below infinity's.
    "inf_max_shift_px": 128.0,
    # When set, the test must also pass with a patch of this radius, whose
    # reading is otherwise unused: a small patch can repeat along the epipolar
    # line where a larger one does not.
    "inf_confirm_radius_px": 0.0,
    # The patch of `inf_radius_px` can be mostly background when the pixel is
    # on a small near object. Its middle (the central samples, about half its
    # width, read from the same samples) must then also read
    # `inf_centre_min_zncc` on average at infinity in the agreeing images, when
    # it has texture (grey standard deviation of `inf_centre_min_std` or more);
    # a flat middle has no say.
    "inf_centre": True,
    "inf_centre_min_std": 8.0,
    "inf_centre_min_zncc": 0.8,
    # Evidence per layer (`layer_evidence`): the pixel's own patch of
    # `layer_radius_px` read at `layer_samples` distances across the layer's
    # range, in every image it lands in.
    "layer_evidence": True,
    "layer_radius_px": 8.0,
    "layer_samples": 5,
    "inf_peak_margin": 0.02,
    "inf_drop": 0.05,
    # Clusters: "kept" uses the reference and kept members, and the queried
    # image's member must be one of them; "any" uses every member in a
    # reconstruction image and lets the triangulation drop the bad ones.
    "cluster_members": "any",
    # Constellation: "pixel" queries from the pixel; "keypoints" also from up to
    # `lateral_max` keypoints within `lateral_radius_px` of it.
    "constellation_at": "pixel",
    "lateral_max": 4,
    "lateral_radius_px": 24.0,
    # Guided: the keypoints within `guided_radius_px` of the pixel (up to
    # `guided_max`, skipping those within `guided_skip_px`) are matched along
    # their epipolar curves in every other image.
    "guided_radius_px": 24.0,
    "guided_max": 8,
    "guided_skip_px": -1.0,
    "guided_epipolar_px": 2.0,
    "guided_ratio": 0.8,
    "guided_max_dist": 250.0,
    # A match past `guided_max_dist` but within this, that passes the ratio
    # test, is added after the first triangulation if it agrees with it.
    "guided_loose_dist": 400.0,
    "guided_min_views": 2,
    # Sweep: the plane sweep at the pixel (candidates/planesweep.py).
    "sweep_radius_px": 12.0,
    "sweep_min_zncc": 0.8,
    "sweep_min_views": 2,
    "sweep_samples": 96,
}


def triangulate(ctx, sightings):
    """The least-squares meeting point of ``(image, pixel)`` rays, and each reprojection error."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for image, px in sightings:
        cam = ctx.camera(image)
        d = cam.ray(px)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ cam.center
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    errs = []
    for image, px in sightings:
        cam = ctx.camera(image)
        p = cam.project(X)
        if p is None or cam.depth(X) <= 0:
            return None, None
        errs.append(float(np.linalg.norm(p - np.asarray(px, float))))
    return X, errs


def vet_cluster(ctx, image, cluster, max_reproj_px: float, policy: str = "kept"):
    """The cluster's usable members when they triangulate cleanly, else ``None``.

    The queried image's member must be the reference or kept. The reference
    and kept members (the best-reading one per image) must meet in front of
    every camera with every reprojection error within ``max_reproj_px``; the
    worst member is dropped and the rest tried again while three or more
    remain, but never the queried image's.
    """
    usable = ("reference", "kept")
    if policy == "any":
        usable = (*usable, "rejected_low_zncc", "rejected_shift", "not_evaluated")
    if cluster["member"]["status"] not in usable:
        return None
    best: dict[int, dict] = {}
    for m in cluster["members"]:
        if m["image"] < 0 or m["status"] not in usable:
            continue
        if m["image"] == image and m["index"] != cluster["member"]["index"]:
            continue
        # Prefer the reference and kept members, then the best-reading one.
        z = m["zncc"] if np.isfinite(m["zncc"]) else 0.0
        if m["status"] in ("reference", "kept"):
            z += 2.0
        if m["image"] not in best or z > best[m["image"]]["_z"]:
            best[m["image"]] = {**m, "_z": z}
    members = list(best.values())
    while len(members) >= 2:
        X, errs = triangulate(ctx, [(m["image"], m["position"]) for m in members])
        if X is not None and max(errs) <= max_reproj_px:
            return members
        if X is None or len(members) < 3:
            return None
        worst = int(np.argmax(errs))
        if members[worst]["image"] == image:
            return None
        members.pop(worst)
    return None


def _robust(ctx, sightings, keep_first: bool, tol: float):
    """Triangulate, dropping the worst sighting while three or more remain."""
    s = list(sightings)
    while len(s) >= 2:
        X, errs = triangulate(ctx, s)
        if X is not None and max(errs) <= tol:
            return X, s, errs
        if X is None or len(s) < 3:
            return None, None, None
        worst = int(np.argmax(errs))
        if keep_first and worst == 0:
            return None, None, None
        s.pop(worst)
    return None, None, None


def _ray_angle(ctx, X, images) -> float:
    dirs = [ctx.camera(i).center - X for i in images]
    dirs = [d / np.linalg.norm(d) for d in dirs]
    best = 0.0
    for a in range(len(dirs)):
        for b in range(a + 1, len(dirs)):
            best = max(
                best, float(np.degrees(np.arccos(np.clip(dirs[a] @ dirs[b], -1, 1))))
            )
    return best


def _anchor(ctx, source, ident, X, sightings, errs, image, qpix, pixel):
    return {
        "source": source,
        "id": ident,
        "position": [float(x) for x in X],
        "views": [[int(i), float(p[0]), float(p[1])] for i, p in sightings],
        "query_pixel": [float(qpix[0]), float(qpix[1])],
        "distance_px": float(
            np.linalg.norm(np.asarray(qpix, float) - np.asarray(pixel, float))
        ),
        "n_views": len(sightings),
        "max_reproj_px": float(max(errs)),
        "max_ray_angle_deg": _ray_angle(ctx, X, [i for i, _ in sightings]),
        "depth": float(ctx.camera(image).depth(X)),
    }


def from_tracks(ctx, image, pixel, opts):
    out = []
    for o in ctx.observations_near(image, pixel, opts["track_radius_px"]):
        if len(out) >= opts["track_max"]:
            break
        if o["position"] is None or o["track_length"] < opts["track_min_views"]:
            continue
        rec = ctx.edited.point(int(o["point"]))
        sightings = [
            (int(i), np.asarray(p, float))
            for i, p in zip(rec["image_indexes"], rec["keypoints_xy"])
        ]
        X = np.asarray(o["position"], float)
        errs = []
        for i, p in sightings:
            q = ctx.camera(i).project(X)
            errs.append(float("inf") if q is None else float(np.linalg.norm(q - p)))
        if max(errs) > opts["max_reproj_px"]:
            continue
        out.append(
            _anchor(
                ctx,
                "tracks",
                int(o["point"]),
                X,
                sightings,
                errs,
                image,
                o["keypoint"],
                pixel,
            )
        )
    return out


def from_clusters(ctx, image, pixel, opts):
    out = []
    for c in ctx.clusters_near(image, pixel, opts["cluster_radius_px"])[
        : opts["cluster_max"]
    ]:
        members = vet_cluster(
            ctx, image, c, opts["max_reproj_px"], opts["cluster_members"]
        )
        if members is None:
            continue
        sightings = [
            (int(m["image"]), np.asarray(m["position"], float)) for m in members
        ]
        X, errs = triangulate(ctx, sightings)
        if X is None:
            continue
        q = next(p for i, p in sightings if i == image)
        out.append(
            _anchor(
                ctx, "clusters", int(c["cluster"]), X, sightings, errs, image, q, pixel
            )
        )
    return out


def _constellation_at(ctx, image, at, pixel, opts):
    from sfmtool._sfmtool import bench as B
    from sfmtool._sfmtool.spatial import radius_for_feature_count

    _, track = B.create_cluster(
        B.Bench(),
        image,
        ctx.image_stem(image),
        tuple(map(float, at)),
        radius_px=opts["constellation_radius_px"],
    )
    track, _ = B.set_verdict(track, 0, "in")
    xy, affine = ctx.keypoints(image)
    w, h = ctx.image_size(image)
    radius = float(
        radius_for_feature_count(w, h, len(xy), opts["constellation_target"])
    )
    try:
        track, found = B.search_descriptors(
            track,
            0,
            xy,
            affine,
            ctx.forest,
            radius_px=radius,
            min_inliers=opts["constellation_min_inliers"],
        )
    except ValueError:
        return None
    if track.observation_count < 2:
        return None
    # Each matched image's sighting is where the constellation's affine warp
    # carries the pixel. The cluster refinement is not used to read them: on
    # a grazing surface it rejects the true matches (point 309 of Kerry Park).
    sightings = [(image, np.asarray(at, float))]
    for o in track.observations[1:]:
        p = (o.get("cluster") or {}).get("seed_position")
        if p is not None:
            sightings.append((int(o["image"]), np.asarray(p, float)))
    X, kept, errs = _robust(ctx, sightings, True, opts["constellation_max_reproj_px"])
    if X is None:
        return None
    return _anchor(ctx, "constellation", None, X, kept, errs, image, at, pixel)


def _keypoints_near(ctx, image, pixel, radius_px, skip_px, limit):
    """Rows of ``image``'s keypoints within ``radius_px`` of the pixel and
    further than ``skip_px``, nearest first."""
    xy, _ = ctx.keypoints(image)
    d = np.linalg.norm(np.asarray(xy, float) - np.asarray(pixel, float), axis=1)
    rows = np.flatnonzero((d <= radius_px) & (d > skip_px))
    return rows[np.argsort(d[rows])][:limit]


def from_constellation(ctx, image, pixel, opts):
    out = []
    a = _constellation_at(ctx, image, pixel, pixel, opts)
    if a is not None:
        out.append(a)
    if opts["constellation_at"] == "keypoints":
        xy, _ = ctx.keypoints(image)
        for k in _keypoints_near(
            ctx, image, pixel, opts["lateral_radius_px"], 1.0, opts["lateral_max"]
        ):
            a = _constellation_at(ctx, image, xy[k], pixel, opts)
            if a is not None:
                a["id"] = int(k)
                out.append(a)
    return out


def _cache(ctx) -> dict:
    ds = ctx.dataset
    if getattr(ds, "_anchors_cache", None) is None:
        ds._anchors_cache = {"desc": {}, "rays": {}}
    return ds._anchors_cache


def _descriptors(ctx, image) -> np.ndarray:
    c = _cache(ctx)["desc"]
    if image not in c:
        from sfmtool.sift.file import SiftReader, get_sift_path_for_image

        ds = ctx.dataset
        reader = SiftReader(
            get_sift_path_for_image(ds.prepared.workspace / ds.image_names[image])
        )
        c[image] = np.asarray(reader.read_descriptors(), np.float32)
        reader.close()
    return c[image]


def _rays(ctx, image) -> np.ndarray:
    """World-frame unit rays through every keypoint of ``image``."""
    c = _cache(ctx)["rays"]
    if image not in c:
        xy, _ = ctx.keypoints(image)
        cam = ctx.camera(image)
        r = np.asarray(
            cam.intrinsics.pixel_to_ray_batch(np.asarray(xy, np.float64)), float
        )
        r = r @ cam.R
        c[image] = r / np.linalg.norm(r, axis=1, keepdims=True)
    return c[image]


def from_guided(ctx, image, pixel, opts):
    """Keypoints near the pixel, matched by descriptor along their epipolar curves.

    With the cameras posed, a keypoint's match in another image lies where that
    image's keypoint ray passes within ``guided_epipolar_px`` of the keypoint's
    own ray. Among those, the nearest descriptor is the match when it is within
    ``guided_max_dist`` and, if there is a second, ``guided_ratio`` of it. Each
    keypoint's matches are triangulated with it, dropping the worst while three
    or more remain, and kept with ``guided_min_views`` or more views. A match
    that passes the ratio test but not ``guided_max_dist``, within
    ``guided_loose_dist``, is then added if the triangulation with it still
    meets every view within ``max_reproj_px``, most distinct first.
    """
    rows = _keypoints_near(
        ctx,
        image,
        pixel,
        opts["guided_radius_px"],
        opts["guided_skip_px"],
        opts["guided_max"],
    )
    if not len(rows):
        return []
    xy_q, _ = ctx.keypoints(image)
    cq = ctx.camera(image).center
    fq = ctx.camera(image).focal
    rq = _rays(ctx, image)[rows]
    dq = _descriptors(ctx, image)[rows]
    sightings = [[(image, np.asarray(xy_q[k], float))] for k in rows]
    loose: list[list] = [[] for _ in rows]
    for j in range(len(ctx.dataset.cameras)):
        if j == image:
            continue
        rj = _rays(ctx, j)
        if not len(rj):
            continue
        cj = ctx.camera(j).center
        fj = ctx.camera(j).focal
        # The closest approach of each query ray and each keypoint ray.
        w0 = cq - cj
        b = rq @ rj.T
        d = rq @ w0
        e = rj @ w0
        den = 1.0 - b * b
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (b * e[None, :] - d[:, None]) / den
            t = (e[None, :] - b * d[:, None]) / den
            gap = np.linalg.norm(
                (cq + s[..., None] * rq[:, None, :])
                - (cj + t[..., None] * rj[None, :, :]),
                axis=2,
            )
            err = np.maximum(fq * gap / np.abs(s), fj * gap / np.abs(t))
        ok = (den > 1e-8) & (s > 0) & (t > 0) & (err <= opts["guided_epipolar_px"])
        if not ok.any():
            continue
        dj = _descriptors(ctx, j)
        xy_j, _ = ctx.keypoints(j)
        for a in np.flatnonzero(ok.any(axis=1)):
            cand = np.flatnonzero(ok[a])
            dist = np.linalg.norm(dj[cand] - dq[a], axis=1)
            order = np.argsort(dist)
            best = dist[order[0]]
            ratio = best / dist[order[1]] if len(order) > 1 else 0.0
            if ratio > opts["guided_ratio"]:
                continue
            match = (j, np.asarray(xy_j[cand[order[0]]], float))
            if best <= opts["guided_max_dist"]:
                sightings[a].append(match)
            elif best <= opts["guided_loose_dist"]:
                loose[a].append((ratio, match))
    out = []
    for k, sg, extra in zip(rows, sightings, loose):
        if len(sg) < 2:
            continue
        X, kept, errs = _robust(ctx, sg, True, opts["max_reproj_px"])
        if X is None:
            continue
        for _, match in sorted(extra, key=lambda e: e[0]):
            X2, errs2 = triangulate(ctx, [*kept, match])
            if X2 is not None and max(errs2) <= opts["max_reproj_px"]:
                X, kept, errs = X2, [*kept, match], errs2
        if len(kept) < opts["guided_min_views"]:
            continue
        out.append(
            _anchor(ctx, "guided", int(k), X, kept, errs, image, sg[0][1], pixel)
        )
    return out


def from_sweep(ctx, image, pixel, opts):
    """The plane sweep at the pixel: the depth its photographs agree on best.

    The sweep of ``candidates/planesweep.py`` with a patch of
    ``sweep_radius_px``. The depth with the highest score is kept when
    ``sweep_min_views`` or more other photographs read ``sweep_min_zncc`` or
    better there.
    """
    from candidates import planesweep

    sopts = {**planesweep.DEFAULTS, "depth_samples": opts["sweep_samples"]}
    swept = planesweep.sweep(ctx, image, pixel, opts["sweep_radius_px"], sopts)
    if swept is None:
        return []
    depths, others, zncc, centre = swept
    top = np.sort(zncc, axis=1)[:, ::-1][:, : sopts["top_k"]]
    score = np.where(top > 0, top, 0).sum(axis=1)
    i = int(np.argmax(score))
    if not np.isfinite(depths[i]):
        return []
    views = [v for v in range(len(others)) if zncc[i, v] >= opts["sweep_min_zncc"]]
    if len(views) < opts["sweep_min_views"]:
        return []
    cam = ctx.camera(image)
    ray = cam.ray(pixel)
    ray = ray / np.linalg.norm(ray)
    # The sweep's planes face the pixel's ray, and its depths are along it.
    X = cam.center + ray * float(depths[i])
    sightings = [(image, np.asarray(pixel, float))] + [
        (others[v], centre[i, v]) for v in views
    ]
    _, errs = triangulate(ctx, sightings)
    a = _anchor(ctx, "sweep", None, X, sightings, errs or [0.0], image, pixel, pixel)
    a["sweep_score"] = float(score[i])
    return [a]


SOURCES = {
    "tracks": from_tracks,
    "clusters": from_clusters,
    "constellation": from_constellation,
    "guided": from_guided,
    "sweep": from_sweep,
}


def distance_range(camera, image, qpix, sightings, t0, tol_px):
    """The distances along ``qpix``'s ray in ``image`` at which every other
    sighting stays within its tolerance: ``(near, far)``, ``far`` may be ``inf``.

    ``camera`` maps an image to its :class:`context.Camera`; ``t0`` is the
    distance the sightings were triangulated at (``inf`` for a point at
    infinity). The tolerance is ``tol_px``, or half a pixel more than the
    largest error at ``t0`` when that is larger, so a reading that only meets
    within 2 px still has a range. The range is found by doubling away from
    ``t0`` until a view leaves its tolerance, then bisecting in log distance.
    """
    cq = camera(image)
    ray = cq.ray(qpix)
    ray = ray / np.linalg.norm(ray)
    others = [(i, np.asarray(p, float)) for i, p in sightings if i != image]

    def err(t):
        if not others:
            return 0.0
        worst = 0.0
        for i, p in others:
            cam = camera(i)
            q = (
                cam.project(ray, 0.0)
                if np.isinf(t)
                else cam.project(cq.center + t * ray)
            )
            if q is None:
                return float("inf")
            worst = max(worst, float(np.linalg.norm(q - p)))
        return worst

    tol = max(tol_px, err(t0) + 0.5)

    def bisect(inside, outside):
        for _ in range(10):
            mid = float(np.sqrt(inside * outside))
            if err(mid) <= tol:
                inside = mid
            else:
                outside = mid
        return inside

    start = t0 if np.isfinite(t0) else 1e7
    near, t = 0.0, start
    for _ in range(40):
        if err(t / 2) > tol:
            near = bisect(t, t / 2)
            break
        t /= 2
    if np.isinf(t0) or err(np.inf) <= tol:
        far = float("inf")
    else:
        far, t = float("inf"), t0
        for _ in range(40):
            if err(t * 2) > tol:
                far = bisect(t, t * 2)
                break
            t *= 2
    return near, far


def _overlap(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _bounded(a, max_span: float) -> bool:
    near, far = a["range"]
    return near > 0 and np.isfinite(far) and far / near <= max_span


def _camera_spread(ctx) -> float:
    c = _cache(ctx)
    if "spread" not in c:
        C = np.asarray([cam.center for cam in ctx.dataset.cameras])
        c["spread"] = float(np.linalg.norm(C[:, None] - C[None], axis=2).max())
    return c["spread"]


def _usable(a) -> bool:
    return bool(a["bounded"] or a["far"])


def read_patch(ctx, image, pixel, radius, depths, views=None):
    """The pixel's patch read along its ray, whole and in the middle.

    The sampling of :func:`candidates.planesweep.sweep`: a square grid of
    ``radius`` around the pixel, on a plane facing the queried camera at each
    of ``depths``, sampled in each other image. Each read gives two ZNCCs from
    the same samples: of the whole grid, and of its middle (the central
    samples, about half the grid's width). A match of the whole patch that the
    middle does not share is carried by the parts away from the pixel.

    Returns ``(others, whole, middle, centres, middle_std)``: the images read,
    ``(depths, images)`` arrays of ZNCC (``-1`` where the patch cannot be read),
    each read's centre pixel, and the grey standard deviation of the query's
    middle; or ``None`` when the query's patch is flat or off the image.
    """
    from candidates import planesweep
    from candidates.common import in_frame

    ds = ctx.dataset
    grey = planesweep.grey_images(ds, planesweep.DEFAULTS["blur_sigma"])
    cam = ctx.camera(image)
    n = planesweep.DEFAULTS["grid"]
    s = np.linspace(-radius, radius, n)
    gx, gy = np.meshgrid(s, s)
    grid = np.asarray(pixel, float) + np.column_stack([gx.ravel(), gy.ravel()])
    template = planesweep.sample(grey[image], grid)
    if not np.all(np.isfinite(template)) or template.std() < 1e-3:
        return None
    c, h = n // 2, n // 4
    ii, jj = np.meshgrid(np.arange(n), np.arange(n))
    middle = ((abs(ii - c) <= h) & (abs(jj - c) <= h)).ravel()
    rays = np.asarray(cam.intrinsics.pixel_to_ray_batch(grid), float) @ cam.R
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    along = rays @ rays[(n * n) // 2]
    if views is None:
        views = [i for i in range(len(ds.cameras)) if i != image]
    whole = np.full((len(depths), len(views)), -1.0)
    mid = np.full((len(depths), len(views)), -1.0)
    centres = np.full((len(depths), len(views), 2), np.nan)
    for di, t in enumerate(depths):
        for vi, other in enumerate(views):
            oc = ctx.camera(other)
            if np.isfinite(t):
                pc = (cam.center + rays * (t / along)[:, None]) @ oc.R.T + oc.t
            else:
                pc = rays @ oc.R.T
            if np.any(-pc[:, 2] <= 1e-9):
                continue
            px = np.asarray(
                oc.intrinsics.ray_to_pixel_batch(
                    pc / np.linalg.norm(pc, axis=1, keepdims=True)
                ),
                float,
            )
            centre = px[(n * n) // 2]
            if not in_frame(ctx, other, centre, margin=radius * 0.5):
                continue
            vals = planesweep.sample(grey[other], px)
            if not np.all(np.isfinite(vals)):
                continue
            whole[di, vi] = planesweep.zncc_rows(template, vals[None, :])[0]
            mid[di, vi] = planesweep.zncc_rows(template[middle], vals[None, middle])[0]
            centres[di, vi] = centre
    return views, whole, mid, centres, float(template[middle].std())


def from_infinity(ctx, image, pixel, opts):
    """The pixel at infinity: one slice of the plane sweep.

    At infinity the pixel lands at one position in every other image, fixed by
    the cameras' rotations alone, so there is nothing to search. The patch of
    ``inf_radius_px`` around the pixel is compared by ZNCC with each image it
    lands in. A point at infinity reads well in nearly all of them; a finite
    point only in the images whose baseline is too short to show its parallax.

    Reading well at infinity is not enough where the texture repeats or barely
    changes along the pixel's epipolar curve: on seoul_bull, a point 11 m away
    reads 0.9 at infinity and 0.99 at its true distance in the one image that
    sees it. So the agreeing images are also read along the ray, in from
    infinity (:func:`_zncc_along`). If some finite distance reads better than
    infinity, there is no reading; otherwise the range's near end is no further
    than where the reading starts to fall (``near_limit``).
    """
    read = read_patch(ctx, image, pixel, opts["inf_radius_px"], [np.inf])
    if read is None:
        return []
    others, zncc, mid, centre, mid_std = read
    z = zncc[0]
    seen = int((z > -1).sum())
    agree = [v for v in np.argsort(-z) if z[v] >= opts["inf_min_zncc"]]
    if len(agree) < opts["inf_min_views"] or len(agree) < opts["inf_min_share"] * seen:
        return []
    if (
        opts["inf_centre"]
        and mid_std >= opts["inf_centre_min_std"]
        and float(np.mean(mid[0, agree])) < opts["inf_centre_min_zncc"]
    ):
        return []
    # Pixels of shift per unit of inverse distance, in the image that moves
    # most: parallax is linear in inverse distance.
    cam = ctx.camera(image)
    ray = cam.ray(pixel)
    ray = ray / np.linalg.norm(ray)
    probe = 1e3 * _camera_spread(ctx)
    rate = 0.0
    for v in agree:
        oc = ctx.camera(int(others[v]))
        a, b = oc.project(ray, 0.0), oc.project(cam.center + probe * ray)
        if a is not None and b is not None:
            rate = max(rate, float(np.linalg.norm(b - a)) * probe)
    if rate <= 0:
        return []
    shifts = 2.0 ** np.arange(0, np.log2(opts["inf_max_shift_px"]) + 1)
    inv = shifts / rate
    along = _zncc_along(
        ctx, image, pixel, [int(others[v]) for v in agree], 1.0 / inv, opts
    )
    at_inf = float(np.mean(z[agree]))
    if along is None or np.nanmax(along) > at_inf + opts["inf_peak_margin"]:
        return []
    near_limit = float("inf")
    for d, score in zip(1.0 / inv, along):
        if not score >= at_inf - opts["inf_drop"]:
            break
        near_limit = float(d)
    views = [[int(image), float(pixel[0]), float(pixel[1])]]
    views += [
        [int(others[v]), float(centre[0, v, 0]), float(centre[0, v, 1])] for v in agree
    ]
    return [
        {
            "source": "infinity",
            "id": None,
            "position": [float(x) for x in ray],
            "w": 0.0,
            "views": views,
            "query_pixel": [float(pixel[0]), float(pixel[1])],
            "distance_px": 0.0,
            "n_views": len(views),
            "max_reproj_px": 0.0,
            "max_ray_angle_deg": 0.0,
            "depth": float("inf"),
            "agree_share": len(agree) / max(seen, 1),
            "near_limit": near_limit,
        }
    ]


def _zncc_along(ctx, image, pixel, views, depths, opts):
    """Mean ZNCC of the pixel's patch over ``views`` at each of ``depths`` on its ray.

    The sampling of :func:`candidates.planesweep.sweep`, for a few images only.
    A depth at which no image can be read gives ``nan``.
    """
    from candidates import planesweep

    ds = ctx.dataset
    grey = planesweep.grey_images(ds, planesweep.DEFAULTS["blur_sigma"])
    cam = ctx.camera(image)
    r = opts["inf_radius_px"]
    n = planesweep.DEFAULTS["grid"]
    s = np.linspace(-r, r, n)
    gx, gy = np.meshgrid(s, s)
    grid = np.asarray(pixel, float) + np.column_stack([gx.ravel(), gy.ravel()])
    template = planesweep.sample(grey[image], grid)
    if not np.all(np.isfinite(template)) or template.std() < 1e-3:
        return None
    rays = np.asarray(cam.intrinsics.pixel_to_ray_batch(grid), float) @ cam.R
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    along = rays @ rays[(n * n) // 2]
    out = []
    for t in depths:
        pts = cam.center + rays * (t / along)[:, None]
        scores = []
        for other in views:
            oc = ctx.camera(other)
            pc = pts @ oc.R.T + oc.t
            if np.any(-pc[:, 2] <= 1e-9):
                continue
            px = np.asarray(
                oc.intrinsics.ray_to_pixel_batch(
                    pc / np.linalg.norm(pc, axis=1, keepdims=True)
                ),
                float,
            )
            vals = planesweep.sample(grey[other], px)
            if np.all(np.isfinite(vals)):
                scores.append(planesweep.zncc_rows(template, vals[None, :])[0])
        out.append(float(np.mean(scores)) if scores else float("nan"))
    return np.asarray(out)


def _from_infinity_confirmed(ctx, image, pixel, opts):
    found = _from_infinity_once(ctx, image, pixel, opts)
    if found and opts["inf_confirm_radius_px"] > 0:
        wide = {**opts, "inf_radius_px": opts["inf_confirm_radius_px"]}
        if not _from_infinity_once(ctx, image, pixel, wide):
            return []
    return found


_from_infinity_once = from_infinity
SOURCES["infinity"] = _from_infinity_confirmed


def find_anchors(ctx, image: int, pixel, options: dict | None = None) -> dict:
    """The anchors near ``pixel`` in ``image``, grouped into depth layers, and
    what each source did.

    Each anchor carries ``range``, the distances along its pixel's ray that its
    views allow (:func:`distance_range`, within ``range_px``); ``bounded``,
    whether that range is finite at both ends and no wider than ``max_span``
    (far over near); and ``far``, whether it has no far end and its near end is
    at least ``far_spread`` times the cameras' spread, a reading that the point
    is further than the photographs can tell apart from infinity. Bounded and
    far anchors are usable. Two anchors support each other when both are
    usable, their ranges overlap, and neither's images are all among the
    other's, so the two readings do not rest on the same photographs. Near the
    pixel the scene can hold surfaces at very different depths, so the anchors
    are hypotheses, not one estimate: ``layers`` groups the usable anchors
    whose ranges overlap, nearest first, each with its range and the anchors in
    it. With ``layer_evidence``, each layer also carries what supports it
    (:func:`layer_evidence`), a ``score`` from the pixel's own patch read at the
    layer's distances, whole and in the middle, and its ``rank`` by that score:
    the best-supported depth is rank 1.

    The infinity test (:func:`from_infinity`) runs after the other sources, when
    they gave no usable anchor, more than one layer, or no usable anchor
    within a pixel of the pixel (``infinity="needed"``). A pixel at infinity
    often has one layer of nearer anchors beside it and none at it.
    """
    opts = {**DEFAULTS, **(options or {})}
    anchors, stages = [], []
    far_near = opts["far_spread"] * _camera_spread(ctx)
    cq = ctx.camera(image)

    def run(name):
        t0 = time.perf_counter()
        found = SOURCES[name](ctx, image, pixel, opts)
        for a in found:
            if a.get("w", 1.0) == 0:
                t = float("inf")
            else:
                X = np.asarray(a["position"], float)
                ray = cq.ray(a["query_pixel"])
                t = float((X - cq.center) @ (ray / np.linalg.norm(ray)))
            views = [(int(v[0]), v[1:]) for v in a["views"]]
            a["distance"] = t
            a["range"] = list(
                distance_range(
                    ctx.camera, image, a["query_pixel"], views, t, opts["range_px"]
                )
            )
            if "near_limit" in a:
                a["range"][0] = min(a["range"][0], a["near_limit"])
            a["bounded"] = _bounded(a, opts["max_span"])
            a["far"] = bool(np.isinf(a["range"][1]) and a["range"][0] >= far_near)
        stages.append(
            {"source": name, "found": len(found), "seconds": time.perf_counter() - t0}
        )
        anchors.extend(found)

    for name in opts["sources"].split("+"):
        run(name)
        close = [
            a for a in anchors if _usable(a) and a["distance_px"] <= opts["enough_px"]
        ]
        if opts["stop"] == "enough" and len(close) >= opts["min_anchors"]:
            break
    if opts["infinity"] == "always" or (
        opts["infinity"] == "needed"
        and (
            len(_layers(anchors)) != 1
            or not any(_usable(a) and a["distance_px"] <= 1.0 for a in anchors)
        )
    ):
        run("infinity")
    for a in anchors:
        mine = {int(v[0]) for v in a["views"]}
        a["support"] = sum(
            1
            for b in anchors
            if b is not a
            and _usable(a)
            and _usable(b)
            and _overlap(a["range"], b["range"])
            and not (mine <= {int(v[0]) for v in b["views"]})
            and not ({int(v[0]) for v in b["views"]} <= mine)
        )
    layers = _layers(anchors)
    if opts["layer_evidence"] and layers:
        t0 = time.perf_counter()
        reads, centre = _layer_reads(ctx, image, pixel, layers, opts)
        for n, L in enumerate(layers):
            L["evidence"] = layer_evidence(anchors, L, reads, n, centre)
            e = L["evidence"]
            # The pixel's own patch there, whole and in the middle: the mean
            # of the whole patch's reading and the lesser of the two.
            L["score"] = 0.5 * (e["photo"] + e["photo_both"])
        order = sorted(range(len(layers)), key=lambda n: -layers[n]["score"])
        for rank, n in enumerate(order, 1):
            layers[n]["rank"] = rank
        stages.append(
            {"source": "evidence", "found": 0, "seconds": time.perf_counter() - t0}
        )
    return {"anchors": anchors, "layers": layers, "stages": stages}


def _layers(anchors) -> list[dict]:
    """The usable anchors grouped by overlapping ranges, nearest first."""
    layers = []
    for k in sorted(
        (k for k, a in enumerate(anchors) if _usable(a)),
        key=lambda k: anchors[k]["range"][0],
    ):
        a = anchors[k]
        if layers and a["range"][0] <= layers[-1]["range"][1]:
            L = layers[-1]
            L["range"][1] = max(L["range"][1], a["range"][1])
            L["anchors"].append(k)
        else:
            layers.append({"range": list(a["range"]), "anchors": [k]})
    for L in layers:
        L["nearest_px"] = min(anchors[k]["distance_px"] for k in L["anchors"])
        L["views"] = max(anchors[k]["n_views"] for k in L["anchors"])
    return layers


def _layer_reads(ctx, image, pixel, layers, opts):
    """The pixel's patch read at each layer: ``(layers, images)``, the best ZNCC
    over ``layer_samples`` distances across the layer's range (even in inverse
    distance; from infinity in for a far layer), ``-1`` where it cannot be read;
    for the whole patch and for its middle (:func:`read_patch`)."""

    depths, owner = [], []
    for n, L in enumerate(layers):
        near, far = L["range"]
        lo = 0.0 if not np.isfinite(far) else 1.0 / far
        hi = 1.0 / near if near > 0 else lo
        for v in np.linspace(lo, hi, opts["layer_samples"]):
            depths.append(np.inf if v == 0 else 1.0 / v)
            owner.append(n)
    read = read_patch(ctx, image, pixel, opts["layer_radius_px"], depths)
    reads = np.full((len(layers), len(ctx.dataset.cameras) - 1), -1.0)
    mids = np.full_like(reads, -1.0)
    if read is None:
        return reads, mids
    _, whole, mid, _, _ = read
    for k, n in enumerate(owner):
        better = whole[k] > reads[n]
        reads[n] = np.where(better, whole[k], reads[n])
        mids[n] = np.where(better, mid[k], mids[n])
    return reads, mids


def layer_evidence(anchors, layer, reads, n, centre=None) -> dict:
    """What supports one layer: its anchors, and the photographs' votes.

    The anchors give how many readings there are, how many rest on different
    photographs (none of whose images are all among another's), their most
    views and widest ray angle, and how near the pixel the nearest sits. The
    pixel's own patch is read at every layer in every image (``reads``); an
    image votes for the layer it reads best, when that reading is 0.7 or better
    and 0.05 better than at any other layer. Short-baseline images read all
    layers alike and do not vote. With ``centre``, the same reads for a smaller
    patch, a vote also needs the middle of the patch to read 0.7 or better at
    that layer: when it does not, the parts of the patch away from the pixel
    are what match. ``votes_all`` counts the votes without that condition.
    """
    members = [anchors[k] for k in layer["anchors"]]
    sets = {frozenset(int(v[0]) for v in a["views"]) for a in members}
    independent = [a for a in sets if not any(a < b for b in sets)]
    images = set().union(*sets) if sets else set()
    mine = reads[n]
    if len(reads) > 1:
        others = np.max(np.delete(reads, n, axis=0), axis=0)
    else:
        others = np.full_like(mine, -1.0)
    readable = mine > -1
    voting = readable & (mine >= 0.7) & (mine >= others + 0.05)
    votes_all = int(voting.sum())
    if centre is not None:
        voting &= centre[n] >= 0.7
    votes = int(voting.sum())

    def best3(x):
        x = np.sort(x[readable])[::-1][:3]
        return float(np.mean(x)) if len(x) else -1.0

    return {
        "n_anchors": len(members),
        "n_independent": len(independent),
        "n_images": len(images),
        "max_views": max((a["n_views"] for a in members), default=0),
        "max_ray_angle": max((a["max_ray_angle_deg"] for a in members), default=0.0),
        "nearest_px": min((a["distance_px"] for a in members), default=None),
        "at_pixel": any(a["distance_px"] <= 1.0 for a in members),
        "sources": sorted({a["source"] for a in members}),
        "support": float(
            sum(
                np.log2(1 + a["n_views"]) * np.exp(-a["distance_px"] / 20.0)
                for a in members
            )
        ),
        "photo": float(np.mean(np.sort(mine[readable])[::-1][:3]))
        if readable.any()
        else -1.0,
        "votes": votes,
        "votes_all": votes_all,
        "photo_mid": best3(centre[n]) if centre is not None else None,
        "photo_both": best3(np.minimum(mine, centre[n]))
        if centre is not None
        else None,
    }


# --- scoring against the ground truth (the harness's side) ------------------

# An anchor within AT_PIXEL_PX of the pixel is a reading of the pixel's own
# distance. An anchor is checked against the ground truth when a true point is
# observed within CHECK_PX of its pixel in the queried image: it is right when
# the two ranges overlap. The ground truth does not hold every surface, so an
# anchor with no true point at its pixel is unchecked, not wrong. An anchor is on
# the pixel's layer when its range overlaps the true point's surface along its
# own ray (`_layer_range`).
AT_PIXEL_PX = 1.0
CHECK_PX = 1.5
RANGE_PX = 1.0


def _truth_range(ds, point: int, image: int, pixel) -> tuple[float, float]:
    """The true point's own range along ``pixel``'s ray, from its track's views."""
    cache = ds.__dict__.setdefault("_truth_ranges", {})
    key = (point, image, round(float(pixel[0]), 2), round(float(pixel[1]), 2))
    if key not in cache:
        cq = ds.cameras[image]
        ray = cq.ray(pixel)
        ray = ray / np.linalg.norm(ray)
        if ds.point_w[point] != 0:
            t0 = float((ds.point_xyz[point] - cq.center) @ ray)
        else:
            t0 = float("inf")
        views = list(zip(ds.point_images[point].tolist(), ds.point_keypoints[point]))
        cache[key] = distance_range(
            lambda i: ds.cameras[i], image, pixel, views, t0, RANGE_PX
        )
    return cache[key]


def _layer_range(ds, point, image, pixel, truth, qpix):
    """The range on the true point's surface along ``qpix``'s ray.

    Where the ray meets the true point's plane, scaled by the true point's own
    range at the pixel over its distance, so a neighbour on the same sloping
    surface is compared at its own distance and with the ground truth's own
    uncertainty. A point at infinity, or a ray that meets the plane behind the
    camera or nearly along it, falls back to the true point's range itself.
    """
    if ds.point_w[point] == 0:
        return truth
    cq = ds.cameras[image]
    G, N = ds.point_xyz[point], ds.point_normal[point]
    ray0 = cq.ray(pixel)
    t_gt = float((G - cq.center) @ (ray0 / np.linalg.norm(ray0)))
    ray = cq.ray(qpix)
    ray = ray / np.linalg.norm(ray)
    den = float(ray @ N)
    if abs(den) < 0.05:
        return truth
    t = float((G - cq.center) @ N) / den
    if t <= 0:
        return truth
    return (t * truth[0] / t_gt, t * truth[1] / t_gt)


def score_anchors(ds, point: int, image: int, pixel, found: dict) -> dict:
    truth = _truth_range(ds, point, image, pixel)
    tree = ds._obs_tree[image]
    rows = []
    for a in found["anchors"]:
        r = {
            k: a[k]
            for k in (
                "source",
                "distance_px",
                "n_views",
                "max_ray_angle_deg",
                "support",
                "bounded",
                "far",
            )
        }
        r["range"] = a["range"]
        r["on_layer"] = bool(
            _usable(a)
            and _overlap(
                a["range"],
                _layer_range(ds, point, image, pixel, truth, a["query_pixel"]),
            )
        )
        # The ground truth at the anchor's own pixel, if it has a point there.
        q = np.asarray([a["query_pixel"]], float)
        offsets, idx = tree.within_radius(q, CHECK_PX)
        hits = [int(ds._obs_point[image][i]) for i in idx[offsets[0] : offsets[1]]]
        r["checked"] = bool(hits)
        if hits:
            r["agrees"] = any(
                _overlap(a["range"], _truth_range(ds, p, image, a["query_pixel"]))
                for p in hits
            )
        rows.append(r)
    out = {
        "anchors_scored": rows,
        "anchors": found["anchors"],
        "layers": found["layers"],
        "stages": found["stages"],
        "truth_range": list(truth),
        "finite_gt": bool(ds.point_w[point] != 0),
        "n_layers": len(found["layers"]),
        "layers_right": [bool(_overlap(L["range"], truth)) for L in found["layers"]],
    }
    for src in (*SOURCES, "all"):
        sub = [r for r in rows if src == "all" or r["source"] == src]
        at = [r for r in sub if _usable(r) and r["distance_px"] <= AT_PIXEL_PX]
        layer = [r for r in sub if r["on_layer"]]
        out[f"{src}_n"] = len(sub)
        out[f"{src}_bounded"] = sum(_usable(r) for r in sub)
        out[f"{src}_at_pixel"] = bool(at)
        out[f"{src}_at_pixel_right"] = any(r["on_layer"] for r in at)
        out[f"{src}_on_layer"] = bool(layer)
        out[f"{src}_on_layer_px"] = min((r["distance_px"] for r in layer), default=None)
        out[f"{src}_supported_layer"] = any(r["support"] > 0 for r in layer)
        out[f"{src}_checked"] = sum(r["checked"] for r in sub)
        out[f"{src}_agrees"] = sum(bool(r.get("agrees")) for r in sub)
    return out


def summarize(rows: list[dict]) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return "no rows"
    lines = [
        f"{len(ok)} queries, {np.median([r['seconds'] for r in ok]):.3f} s median, "
        f"{np.percentile([r['seconds'] for r in ok], 90):.3f} s p90, "
        f"{np.mean([r['n_layers'] for r in ok]):.2f} layers a query",
        f"{'source':14s} {'has':>6s} {'mean n':>7s} {'bnd':>6s} {'at px':>6s} "
        f"{'right':>6s} {'wrong':>6s} {'layer':>6s} {'lyr px':>7s} {'l+sup':>6s} "
        f"{'chk':>6s} {'agree':>6s} {'s':>7s}",
    ]
    n = len(ok)
    for src in (*SOURCES, "all"):
        if f"{src}_n" not in ok[0]:
            continue
        has = [r for r in ok if r[f"{src}_n"] > 0]
        if not has and src != "all":
            continue
        total = sum(r[f"{src}_n"] for r in ok)
        checked = sum(r[f"{src}_checked"] for r in ok)
        at = sum(r[f"{src}_at_pixel"] for r in ok)
        right = sum(r[f"{src}_at_pixel_right"] for r in ok)
        lpx = [r[f"{src}_on_layer_px"] for r in ok if r[f"{src}_on_layer"]]
        secs = [
            sum(st["seconds"] for st in r["stages"] if src in ("all", st["source"]))
            for r in ok
        ]
        lines.append(
            f"{src:14s} {len(has) / n:6.3f} "
            f"{total / n:7.2f} "
            f"{sum(r[f'{src}_bounded'] for r in ok) / max(total, 1):6.3f} "
            f"{at / n:6.3f} {right / n:6.3f} {(at - right) / n:6.3f} "
            f"{len(lpx) / n:6.3f} {np.median(lpx) if lpx else float('nan'):7.1f} "
            f"{sum(r[f'{src}_supported_layer'] for r in ok) / n:6.3f} "
            f"{checked / max(total, 1):6.3f} "
            f"{sum(r[f'{src}_agrees'] for r in ok) / max(checked, 1):6.3f} "
            f"{np.mean(secs):7.3f}"
        )
    lines.append(
        "has: share of queries with an anchor; mean n: anchors a query; bnd: share "
        "of anchors that are usable (bounded or far); at px, right, wrong: share of queries with "
        "a usable anchor at the pixel, and with one whose range does or does not "
        "overlap the true point's; layer: share with an anchor on the pixel's layer "
        "(its range overlaps the true point's), lyr px: median pixels to the nearest "
        "one, l+sup: share with one another anchor supports; chk: share of anchors "
        "with a true point at their own pixel, agree: share of those whose ranges "
        "overlap; s: mean seconds a query"
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULTS",
    "distance_range",
    "find_anchors",
    "score_anchors",
    "summarize",
]
