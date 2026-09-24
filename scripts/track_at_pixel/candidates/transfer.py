# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Transfer: carry the pixel into other photographs by the neighbours' own sightings.

The reconstructed points around the pixel are already matched across the
photographs. Over a small enough neighbourhood of one surface, the map from the
queried image to another is close to affine, so the neighbours' pairs of
keypoints fix that map and it carries the pixel over. No descriptor, no pose
and no depth guess is used to find the sightings; the poses enter only at the
fit.

1. **Neighbourhood.** The observations near the pixel in the queried image,
   split into depth modes; each mode with enough points is a hypothesis (one
   surface, or either side of an edge).
2. **Local affine per image.** For every other photograph, the mode's points
   that are seen there give pairs ``(keypoint here, keypoint there)``. A
   weighted least-squares affine (nearer neighbours weigh more) is fitted,
   the pairs it misses by more than ``max_residual_px`` are dropped and it is
   fitted again. The pixel's image under it is a sighting.
3. **Fit.** The sightings are upgraded to a track and fitted, anchored on the
   pixel. The best-reading hypothesis is finished by
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
    "neighbour_radius_px": 60.0,
    "max_neighbours": 16,
    "depth_mode_gap": 1.15,
    "min_pairs": 4,
    "max_residual_px": 3.0,
    "default_radius_px": 8.0,
    "min_radius_px": 4.0,
    "max_radius_px": 40.0,
}


def fit_affine(src: np.ndarray, dst: np.ndarray, w: np.ndarray):
    """Weighted least-squares ``dst ~ A @ src + b``; returns (A, b) or None."""
    X = np.column_stack([src, np.ones(len(src))])
    sw = np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(X * sw, dst * sw, rcond=None)
    if not np.all(np.isfinite(sol)):
        return None
    return sol[:2].T, sol[2]


def depth_modes(near: list[dict], gap: float) -> list[list[dict]]:
    finite = sorted(
        (o for o in near if not o["at_infinity"] and o["depth"] and o["depth"] > 0),
        key=lambda o: o["depth"],
    )
    if not finite:
        return []
    modes, cur = [], [finite[0]]
    for o in finite[1:]:
        if o["depth"] / cur[-1]["depth"] > gap:
            modes.append(cur)
            cur = []
        cur.append(o)
    modes.append(cur)
    return modes


def transfer_sightings(ctx, image: int, pixel, mode: list[dict], opts: dict):
    ds = ctx.dataset
    pixel = np.asarray(pixel, float)
    mode = sorted(mode, key=lambda o: o["distance_px"])[: opts["max_neighbours"]]
    pairs: dict[int, list] = {}
    for o in mode:
        p = o["point"]
        for img, kp in zip(ds.point_images[p], ds.point_keypoints[p]):
            if int(img) == image:
                continue
            pairs.setdefault(int(img), []).append(
                (o["keypoint"], np.asarray(kp, float), 1.0 / (1.0 + o["distance_px"]))
            )
    sightings, fits = [], {}
    for img, rows in pairs.items():
        if len(rows) < opts["min_pairs"]:
            continue
        src = np.asarray([r[0] for r in rows]) - pixel
        dst = np.asarray([r[1] for r in rows])
        w = np.asarray([r[2] for r in rows])
        keep = np.ones(len(rows), bool)
        model = None
        for _ in range(3):
            if keep.sum() < opts["min_pairs"]:
                model = None
                break
            model = fit_affine(src[keep], dst[keep], w[keep])
            if model is None:
                break
            A, b = model
            resid = np.linalg.norm(src @ A.T + b - dst, axis=1)
            new_keep = resid <= opts["max_residual_px"]
            if (new_keep == keep).all():
                break
            keep = new_keep
        if model is None or keep.sum() < opts["min_pairs"]:
            continue
        A, b = model
        if np.linalg.det(A) <= 0:
            continue
        pred = b  # the pixel is the origin of src
        if not in_frame(ctx, img, pred):
            continue
        sightings.append((img, pred))
        fits[img] = {
            "pairs": int(keep.sum()),
            "scale": float(np.sqrt(np.linalg.det(A))),
        }
    return sightings, fits


def build_track(ctx, image: int, pixel, options: dict | None = None):
    from sfmtool._sfmtool import bench as B

    opts = {**DEFAULTS, **(options or {})}
    pixel = (float(pixel[0]), float(pixel[1]))
    diag: dict = {}
    near = ctx.observations_near(image, pixel, opts["neighbour_radius_px"])
    modes = [
        m
        for m in depth_modes(near, opts["depth_mode_gap"])
        if len(m) >= opts["min_pairs"]
    ]
    modes.sort(key=lambda m: min(o["distance_px"] for o in m))
    diag["modes"] = [len(m) for m in modes]
    if not modes:
        raise TrackAtPixelError(
            "neighbourhood",
            f"fewer than {opts['min_pairs']} reconstructed points of one surface lie "
            f"within {opts['neighbour_radius_px']:.0f} px of the pixel",
            diag,
        )
    best, best_score, tried = None, -1.0, []
    for mode in modes:
        sightings, fits = transfer_sightings(ctx, image, pixel, mode, opts)
        record = {"support": len(mode), "images": sorted(fits)}
        tried.append(record)
        if not sightings:
            record["error"] = "no other photograph shares enough of these neighbours"
            continue
        half_px = [o["half_px"] for o in mode[:8] if np.isfinite(o["half_px"])]
        radius_px = float(
            np.clip(
                np.median(half_px) if half_px else opts["default_radius_px"],
                opts["min_radius_px"],
                opts["max_radius_px"],
            )
        )
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
            "transfer",
            "no neighbourhood carried the pixel into another photograph",
            diag,
        )
    return finish(ctx, best, 0, pixel, opts, diag)
