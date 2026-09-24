# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a built track against the ground-truth track it was built in place of.

Every number is computed from the two tracks and the cameras alone, so a
candidate cannot influence how it is scored. Photometric numbers (the
leave-one-out ZNCC and friends) are read off the tracks' own evaluations, and
the ground truth's are from the same ``evaluate`` run on the ground-truth point,
so the two are one measurement taken of two tracks.
"""

from __future__ import annotations

import numpy as np

from context import texel_scale


def _texel_scales(dataset, track) -> dict[int, dict]:
    placement = track.placement
    if placement is None:
        return {}
    out = {}
    for i, o in enumerate(track.observations):
        s = texel_scale(
            dataset.cameras[int(o["image"])],
            placement["center"],
            placement["u_halfvec"],
            placement["v_halfvec"],
            float(placement["w"]),
        )
        if s is not None:
            out[i] = s
    return out


def _scale_stats(prefix: str, scales: list[dict]) -> dict:
    if not scales:
        return {f"{prefix}_{k}": None for k in ("min", "median", "max", "aniso_max")}
    values = np.asarray([s["scale"] for s in scales])
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_aniso_max": float(max(s["anisotropy"] for s in scales)),
    }


def _in_rows(track) -> list[dict]:
    return [o for o in track.observations if o["verdict"] == "in"]


def _track_numbers(rows: list[dict], key: str) -> np.ndarray:
    vals = [o.get("track", {}).get(key) for o in rows]
    return np.asarray([v for v in vals if v is not None], dtype=float)


def _stats(prefix: str, values: np.ndarray) -> dict:
    if values.size == 0:
        return {f"{prefix}_median": None, f"{prefix}_min": None}
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_min": float(np.min(values)),
    }


def _angle_deg(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def ground_truth_reading(dataset, point: int) -> dict:
    """The ground-truth track, put on the bench and evaluated (moves nothing)."""
    from sfmtool._sfmtool import bench as B
    from sfmtool._sfmtool.reconstruction import EditedReconstruction

    edited = EditedReconstruction(dataset.recon)
    _, track = B.create_track(B.Bench(), edited, int(point))
    track, _ = B.evaluate(track, edited, dataset.pyramids)
    rows = _in_rows(track)
    return {
        "track": track,
        **_stats("gt_zncc", _track_numbers(rows, "zncc")),
        **_stats("gt_localizability", _track_numbers(rows, "localizability")),
        "gt_reproj_median": _stats("x", _track_numbers(rows, "reprojection_error"))[
            "x_median"
        ],
    }


def score(
    dataset, point: int, query_image: int, query_pixel, result, gt_read: dict
) -> dict:
    """The metric row for one query. ``result`` is a ``TrackAtPixelResult``."""
    track = result.track
    cam = dataset.cameras[query_image]
    q = np.asarray(query_pixel, float)

    gt_xyz = dataset.point_xyz[point]
    gt_w = float(dataset.point_w[point])
    gt_images = [int(i) for i in dataset.point_images[point]]
    gt_kp = {int(i): dataset.point_keypoints[point][j] for j, i in enumerate(gt_images)}
    gt_normal = dataset.point_normal[point]
    gt_half = float(dataset.point_half[point])

    rows = _in_rows(track)
    images = [int(o["image"]) for o in rows]
    shared = sorted(set(images) & set(gt_images))
    m: dict = {
        "n_in": len(rows),
        "n_gt": len(gt_images),
        "n_shared": len(shared),
        "image_precision": len(shared) / len(images) if images else None,
        "image_recall": len(shared) / len(gt_images),
    }

    kp_err = []
    for o in rows:
        kp = o.get("track", {}).get("keypoint")
        if kp is not None and int(o["image"]) in gt_kp:
            kp_err.append(
                float(np.linalg.norm(np.asarray(kp) - gt_kp[int(o["image"])]))
            )
    m["kp_err_median_px"] = float(np.median(kp_err)) if kp_err else None
    m["kp_err_max_px"] = float(np.max(kp_err)) if kp_err else None

    # Geometric precision: an `in` view is right when its keypoint lies within
    # VIEW_TOLERANCE_PX of the ground-truth point's projection in that image.
    # Unlike image precision this credits a correct sighting in a photograph
    # the ground-truth track happens not to list (ground-truth tracks are not
    # complete), and still charges one on another piece of surface.
    right = 0
    for o in rows:
        kp = o.get("track", {}).get("keypoint")
        proj = dataset.cameras[int(o["image"])].project(gt_xyz, 1.0 if gt_w else 0.0)
        if kp is not None and proj is not None:
            right += float(np.linalg.norm(np.asarray(kp) - proj)) <= VIEW_TOLERANCE_PX
    m["view_precision"] = right / len(rows) if rows else None

    # Where the track sits in the queried photograph, which is the contract:
    # centred at, or very near, the pixel asked about.
    qo = result.query_observation
    q_row = track.observations[qo] if qo is not None else None
    m["query_image_in"] = bool(q_row is not None and q_row["verdict"] == "in")
    q_kp = q_row.get("track", {}).get("keypoint") if q_row is not None else None
    m["query_keypoint_offset_px"] = (
        float(np.linalg.norm(np.asarray(q_kp) - q)) if q_kp is not None else None
    )

    at_inf = bool(track.at_infinity)
    coord = np.asarray(track.direction if at_inf else track.position, float)
    proj = cam.project(coord, 0.0 if at_inf else 1.0)
    m["query_projection_offset_px"] = (
        float(np.linalg.norm(proj - q)) if proj is not None else None
    )

    m["at_infinity"] = at_inf
    m["gt_at_infinity"] = gt_w == 0
    if not at_inf and gt_w != 0:
        err = coord - gt_xyz
        gt_depth = cam.depth(gt_xyz)
        ray = cam.ray(q)
        along = float(err @ ray)
        m["position_err"] = float(np.linalg.norm(err))
        m["position_err_along_ray"] = along
        m["position_err_lateral"] = float(np.linalg.norm(err - along * ray))
        m["position_err_rel_depth"] = float(np.linalg.norm(err)) / gt_depth
        m["position_err_in_gt_halves"] = float(np.linalg.norm(err)) / gt_half
        # Angle the two points subtend from the query camera: a pixel-free
        # "same line of sight" number that ignores depth error.
        c = cam.center
        m["position_err_angle_deg"] = _angle_deg(coord - c, gt_xyz - c)
    else:
        gt_dir = gt_xyz if gt_w == 0 else gt_xyz - cam.center
        m["position_err_angle_deg"] = _angle_deg(
            coord if at_inf else coord - cam.center, gt_dir
        )

    placement = track.placement
    if placement is not None:
        u = np.asarray(placement["u_halfvec"], float)
        v = np.asarray(placement["v_halfvec"], float)
        normal = np.cross(u, v)
        m["normal_err_deg"] = _angle_deg(normal, gt_normal) if gt_w != 0 else None
        half = float(np.linalg.norm(u))
        m["half_extent"] = half
        m["gt_half_extent"] = gt_half
        m["half_extent_ratio"] = half / gt_half if gt_half > 0 else None
        if not at_inf:
            depth = cam.depth(coord)
            m["half_px_in_query"] = half * cam.focal / depth if depth > 0 else None

    # Sampling ratio: image pixels per bitmap texel, per `in` view, for the
    # built track and for the ground truth's own frame over its own images.
    built_scales = [
        s
        for i, s in _texel_scales(dataset, track).items()
        if track.observations[i]["verdict"] == "in"
    ]
    m.update(_scale_stats("texel_scale", built_scales))
    gt_scales = []
    for image in gt_images:
        s = texel_scale(
            dataset.cameras[image],
            gt_xyz,
            dataset.point_u[point],
            dataset.point_v[point],
            gt_w,
        )
        if s is not None:
            gt_scales.append(s)
    m.update(_scale_stats("gt_texel_scale", gt_scales))

    zncc = _track_numbers(rows, "zncc")
    m.update(_stats("zncc", zncc))
    m.update(_stats("localizability", _track_numbers(rows, "localizability")))
    reproj = _track_numbers(rows, "reprojection_error")
    m["reproj_median"] = float(np.median(reproj)) if reproj.size else None
    for key in (
        "gt_zncc_median",
        "gt_zncc_min",
        "gt_localizability_median",
        "gt_reproj_median",
    ):
        m[key] = gt_read.get(key)
    if m["zncc_median"] is not None and gt_read.get("gt_zncc_median") is not None:
        m["zncc_median_delta"] = m["zncc_median"] - gt_read["gt_zncc_median"]
    m["good_failures"] = good_failures(m)
    m["good"] = not m["good_failures"]
    return m


# How far an `in` keypoint may sit from the ground-truth point's projection in
# its image and still count as a sighting of that point.
VIEW_TOLERANCE_PX = 3.0

# The harness's own bar for a good track, the same for every candidate, so
# candidates with different gates are compared on one scale. A candidate's gates
# decide what it returns; this decides whether what it returned was right.
GOOD_BAR = {
    "max_err_in_gt_halves": 1.0,  # the point lies within the true patch
    "max_bearing_err_deg": 0.5,  # the same, for a bearing
    "min_view_precision": 0.75,  # its views mostly see the true point
    "min_zncc_median": 0.7,  # and they agree photometrically
}


def good_failures(m: dict) -> list[str]:
    """Which parts of :data:`GOOD_BAR` a scored row misses (empty = good)."""
    bar = GOOD_BAR
    out = []
    # The queried sighting has to stay in the track, but its keypoint may settle
    # away from the pixel: the position bar is what keeps the track on the spot.
    if not m.get("query_image_in"):
        out.append("query out")
    halves = m.get("position_err_in_gt_halves")
    if halves is not None:
        if halves > bar["max_err_in_gt_halves"]:
            out.append("position")
    else:
        angle = m.get("position_err_angle_deg")
        if angle is None or angle > bar["max_bearing_err_deg"]:
            out.append("position")
    precision = m.get("view_precision")
    if precision is None or precision < bar["min_view_precision"]:
        out.append("precision")
    zncc = m.get("zncc_median")
    if zncc is None or zncc < bar["min_zncc_median"]:
        out.append("zncc")
    return out
