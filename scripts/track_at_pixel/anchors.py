# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Anchors: reinforced depth readings near a pixel, the first step of building a track.

Before a track is built at a pixel, the depth there is unknown. An **anchor**
is a 3D point near the pixel that several photographs agree on, with the
pixel it sits at in the queried image: a place to start from and walk toward
the pixel. :func:`find_anchors` looks for them in three sources, strongest
first, and stops once it has enough close to the pixel (``stop="enough"``) or
runs all three (``stop="never"``, what the harness's ``anchors`` mode uses so
each source is measured):

1. **Tracks** (``tracks``): the reconstruction's own points observed within
   ``track_radius_px`` of the pixel, finite, seen in ``track_min_views`` or
   more images, every observation within ``max_reproj_px`` of the point's
   projection. These are the strongest readings: a solver already agreed on
   them.
2. **Clusters** (``clusters``): the cluster-patches clusters with a member in
   the queried image within ``cluster_radius_px``, vetted by triangulation
   (:func:`vet_cluster`): the queried image's member must be the
   reference or kept, and the reference and kept members must meet in front of
   every camera within ``max_reproj_px``.
3. **Constellation** (``constellation``): the SIFT index's constellation query
   from the pixel itself. Each image it matches carries the pixel into its
   own frame by the constellation's affine warp, and those sightings are
   triangulated, dropping the worst while three or more remain, within
   ``constellation_max_reproj_px``. Its anchor sits at the pixel.

An anchor is **supported** by each anchor of another source (or another
cluster) that lies within ``agree_fraction`` of its depth of it in 3D: two
independent readings of one structure.

:func:`score_anchors` measures anchors against the ground truth, which
:func:`find_anchors` never sees; :func:`summarize` reports a pass.
"""

from __future__ import annotations

import time

import numpy as np

DEFAULTS = {
    "sources": "tracks+clusters+constellation",
    # "enough": stop after the first source that leaves `min_anchors` anchors
    # within `enough_px` of the pixel; "never": run every source.
    "stop": "enough",
    "min_anchors": 2,
    "enough_px": 20.0,
    "track_radius_px": 40.0,
    "track_max": 8,
    "track_min_views": 2,
    "cluster_radius_px": 24.0,
    "cluster_max": 8,
    "constellation_target": 50,
    "constellation_min_inliers": 6,
    "constellation_radius_px": 6.0,
    "max_reproj_px": 2.0,
    # The constellation's sightings are affine predictions, not refined.
    "constellation_max_reproj_px": 3.0,
    "agree_fraction": 0.1,
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


def vet_cluster(ctx, image, cluster, max_reproj_px: float):
    """The cluster's usable members when they triangulate cleanly, else ``None``.

    The queried image's member must be the reference or kept. The reference
    and kept members (the best-reading one per image) must meet in front of
    every camera with every reprojection error within ``max_reproj_px``; the
    worst member is dropped and the rest tried again while three or more
    remain, but never the queried image's.
    """
    if cluster["member"]["status"] not in ("reference", "kept"):
        return None
    best: dict[int, dict] = {}
    for m in cluster["members"]:
        if m["image"] < 0 or m["status"] not in ("reference", "kept"):
            continue
        z = m["zncc"] if np.isfinite(m["zncc"]) else 1.0
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
        members = vet_cluster(ctx, image, c, opts["max_reproj_px"])
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


def from_constellation(ctx, image, pixel, opts):
    from sfmtool._sfmtool import bench as B
    from sfmtool._sfmtool.spatial import radius_for_feature_count

    _, track = B.create_cluster(
        B.Bench(),
        image,
        ctx.image_stem(image),
        tuple(map(float, pixel)),
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
        return []
    if track.observation_count < 2:
        return []
    # Each matched image's sighting is where the constellation's affine warp
    # carries the pixel. The cluster refinement is not used to read them: on
    # a grazing surface it rejects the true matches (point 309 of Kerry Park).
    sightings = [(image, np.asarray(pixel, float))]
    for o in track.observations[1:]:
        p = (o.get("cluster") or {}).get("seed_position")
        if p is not None:
            sightings.append((int(o["image"]), np.asarray(p, float)))
    X, kept, errs = _robust(ctx, sightings, True, opts["constellation_max_reproj_px"])
    if X is None:
        return []
    return [_anchor(ctx, "constellation", None, X, kept, errs, image, pixel, pixel)]


SOURCES = {
    "tracks": from_tracks,
    "clusters": from_clusters,
    "constellation": from_constellation,
}


def find_anchors(ctx, image: int, pixel, options: dict | None = None) -> dict:
    """The anchors near ``pixel`` in ``image``, and what each source did."""
    opts = {**DEFAULTS, **(options or {})}
    anchors, stages = [], []
    for name in opts["sources"].split("+"):
        t0 = time.perf_counter()
        found = SOURCES[name](ctx, image, pixel, opts)
        stages.append(
            {"source": name, "found": len(found), "seconds": time.perf_counter() - t0}
        )
        anchors.extend(found)
        close = [a for a in anchors if a["distance_px"] <= opts["enough_px"]]
        if opts["stop"] == "enough" and len(close) >= opts["min_anchors"]:
            break
    for a in anchors:
        a["support"] = sum(
            1
            for b in anchors
            if b is not a
            and (b["source"], b["id"]) != (a["source"], a["id"])
            and np.linalg.norm(np.subtract(b["position"], a["position"]))
            <= opts["agree_fraction"] * a["depth"]
        )
    return {"anchors": anchors, "stages": stages}


# --- scoring against the ground truth (the harness's side) ------------------

# An anchor is near the true point within NEAR_HALVES of its half-sizes in 3D,
# and on its surface when also within PLANE_HALVES of its plane and
# SURFACE_HALVES of it. An anchor within AT_PIXEL_PX of the pixel is a reading
# of the pixel's own depth.
NEAR_HALVES = 2.0
PLANE_HALVES = 0.25
SURFACE_HALVES = 8.0
AT_PIXEL_PX = 1.0


def score_anchors(ds, point: int, image: int, pixel, found: dict) -> dict:
    cam = ds.cameras[image]
    G, N, H = ds.point_xyz[point], ds.point_normal[point], float(ds.point_half[point])
    finite = ds.point_w[point] != 0
    rows = []
    for a in found["anchors"]:
        X = np.asarray(a["position"])
        r = {
            k: a[k]
            for k in (
                "source",
                "distance_px",
                "n_views",
                "max_ray_angle_deg",
                "support",
            )
        }
        if finite:
            r["dist_halves"] = float(np.linalg.norm(X - G)) / H
            r["plane_halves"] = abs(float((X - G) @ N)) / H
            r["on_surface"] = (
                r["plane_halves"] <= PLANE_HALVES and r["dist_halves"] <= SURFACE_HALVES
            )
            if a["distance_px"] <= AT_PIXEL_PX:
                r["depth_ratio"] = a["depth"] / cam.depth(G)
        rows.append(r)
    out = {
        "anchors_scored": rows,
        "anchors": found["anchors"],
        "stages": found["stages"],
        "finite_gt": bool(finite),
        "settled_on": found["stages"][-1]["source"] if found["stages"] else None,
    }
    for src in ("tracks", "clusters", "constellation", "all"):
        sub = [r for r in rows if src == "all" or r["source"] == src]
        out[f"{src}_n"] = len(sub)
        out[f"{src}_nearest_px"] = min((r["distance_px"] for r in sub), default=None)
        if finite and sub:
            out[f"{src}_nearest_halves"] = min(r["dist_halves"] for r in sub)
            out[f"{src}_on_surface"] = any(r["on_surface"] for r in sub)
            out[f"{src}_supported"] = any(r["support"] > 0 for r in sub)
            at = [abs(r["depth_ratio"] - 1) for r in sub if "depth_ratio" in r]
            out[f"{src}_at_pixel_depth_err"] = min(at) if at else None
    return out


def summarize(rows: list[dict]) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return "no rows"
    fin = [r for r in ok if r.get("finite_gt")]
    lines = [
        f"{len(ok)} queries ({len(fin)} with a finite ground truth), "
        f"{np.median([r['seconds'] for r in ok]):.3f} s median, "
        f"{np.percentile([r['seconds'] for r in ok], 90):.3f} s p90",
        f"{'source':14s} {'has':>6s} {'mean n':>7s} {'near px':>8s} {'3D h':>6s} "
        f"{'<=2h':>6s} {'surface':>8s} {'at px':>6s} {'<5%':>6s} {'support':>8s}",
    ]
    for src in ("tracks", "clusters", "constellation", "all"):
        has = [r for r in ok if r[f"{src}_n"] > 0]
        fh = [r for r in fin if r[f"{src}_n"] > 0]
        near = [r[f"{src}_nearest_px"] for r in has]
        h3 = [r[f"{src}_nearest_halves"] for r in fh]
        at = [r.get(f"{src}_at_pixel_depth_err") for r in fh]
        at = [v for v in at if v is not None]
        n = max(len(fin), 1)
        lines.append(
            f"{src:14s} {len(has) / len(ok):6.3f} "
            f"{np.mean([r[f'{src}_n'] for r in ok]):7.2f} "
            f"{np.median(near) if near else float('nan'):8.1f} "
            f"{np.median(h3) if h3 else float('nan'):6.2f} "
            f"{sum(v <= NEAR_HALVES for v in h3) / n:6.3f} "
            f"{sum(bool(r[f'{src}_on_surface']) for r in fh) / n:8.3f} "
            f"{len(at) / n:6.3f} {sum(v <= 0.05 for v in at) / n:6.3f} "
            f"{sum(bool(r[f'{src}_supported']) for r in fh) / n:8.3f}"
        )
    lines.append(
        "has: share of queries with an anchor; near px, 3D h: median nearest, in "
        "pixels and in true half-sizes; <=2h, surface, at px, <5%, support: shares "
        "of the finite queries with an anchor within 2 half-sizes, on the true "
        "surface, at the pixel, at the pixel within 5% of the true depth, and "
        "supported by another source"
    )
    return "\n".join(lines)


__all__ = ["DEFAULTS", "find_anchors", "score_anchors", "summarize"]
