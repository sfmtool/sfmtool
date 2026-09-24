# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Plane sweep: search the queried pixel's ray for the depth the photographs agree on.

The other candidates start from something already reconstructed near the pixel:
a cluster, the neighbours' matched keypoints, or the neighbours' surface. Where
there is none, or where it describes a different surface, they have nothing to
start from. This candidate needs only the poses and the photographs.

The point lies on the pixel's ray; only the distance along it is unknown. For
each distance on a grid (uniform in inverse depth, down to the point at
infinity), a small square patch facing the queried camera is placed there,
its grid of sample points is projected into every other photograph, and each
view's samples are compared with the queried photograph's by ZNCC. At the
right depth the views that see the point agree with the query; at a wrong
depth they sample some other piece of the photograph.

1. **Sweep.** ``depth_samples`` inverse depths between the nearest and
   furthest reconstructed points the queried camera sees (widened by
   ``depth_margin``), plus infinity. Each depth's score is the sum of its
   ``top_k`` best view ZNCCs.
2. **Hypotheses.** The ``max_hypotheses`` best local maxima of that score. A
   hypothesis's sightings are the views whose ZNCC there is at least
   ``min_view_zncc``, at the pixel where the patch centre projects.
3. **Fit.** Each hypothesis is upgraded to a track and fitted, anchored on the
   pixel; the best-reading one is finished by :func:`candidates.common.finish`.

The patch size is the median apparent size of the reconstructed points within
``size_radius_px`` of the pixel, or ``default_radius_px`` when there are none.
The images are compared in grey, blurred by ``blur_sigma`` pixels so that a
view seeing the patch smaller than the query does not alias.
"""

from __future__ import annotations

import cv2
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
    "depth_samples": 96,
    "depth_margin": 2.0,
    "grid": 11,
    "size_radius_px": 40.0,
    "default_radius_px": 12.0,
    "min_radius_px": 5.0,
    "max_radius_px": 40.0,
    "blur_sigma": 1.0,
    "top_k": 3,
    "max_hypotheses": 3,
    "min_view_zncc": 0.7,
    "min_views": 2,
}


def grey_images(ds, sigma: float) -> list[np.ndarray]:
    """The photographs in grey as float32, blurred, cached on the dataset."""
    key = ("_planesweep_grey", sigma)
    cache = getattr(ds, "_planesweep_cache", None)
    if cache is None:
        cache = {}
        ds._planesweep_cache = cache
    if key not in cache:
        out = []
        for im in ds.images:
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if sigma > 0:
                g = cv2.GaussianBlur(g, (0, 0), sigma)
            out.append(g)
        cache[key] = out
    return cache[key]


def zncc_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ZNCC of one template ``a`` (n,) with each row of ``b`` (m, n)."""
    a = a - a.mean()
    b = b - b.mean(axis=1, keepdims=True)
    den = np.linalg.norm(a) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (b @ a) / den
    return np.where(den > 1e-6, z, -1.0)


def sample(img: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``img`` at pixel centres ``px`` (n, 2)."""
    m = px.astype(np.float32).reshape(1, -1, 2)
    # Pixel (x, y) is the centre of texel (x - 0.5, y - 0.5) in array terms.
    out = cv2.remap(
        img,
        m[..., 0] - 0.5,
        m[..., 1] - 0.5,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    return out.reshape(-1)


def depth_range(ctx, image: int, margin: float) -> tuple[float, float]:
    ds = ctx.dataset
    cam = ctx.camera(image)
    finite = ds.point_w != 0
    d = np.asarray(
        [
            cam.depth(x)
            for x, f in zip(ds.point_xyz, finite)
            if f and cam.project(x) is not None
        ]
    )
    d = d[d > 0]
    if d.size == 0:
        return 0.1, 100.0
    return float(np.percentile(d, 1)) / margin, float(np.percentile(d, 99)) * margin


def sweep(ctx, image: int, pixel, radius_px: float, opts: dict, depths=None):
    """Per depth: per-view ZNCC and centre pixels.

    ``depths`` defaults to the inverse-depth grid over the scene plus infinity.
    """
    ds = ctx.dataset
    grey = grey_images(ds, opts["blur_sigma"])
    cam = ctx.camera(image)
    n = opts["grid"]
    s = np.linspace(-radius_px, radius_px, n)
    gx, gy = np.meshgrid(s, s)
    grid_px = np.asarray(pixel, float) + np.column_stack([gx.ravel(), gy.ravel()])
    template = sample(grey[image], grid_px)
    if not np.all(np.isfinite(template)) or template.std() < 1e-3:
        return None
    rays = np.asarray(cam.intrinsics.pixel_to_ray_batch(grid_px), float) @ cam.R
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    axis = rays[(n * n) // 2]
    if depths is None:
        near, far = depth_range(ctx, image, opts["depth_margin"])
        inv = np.linspace(1.0 / near, 1.0 / far, opts["depth_samples"])
        depths = list(1.0 / inv) + [np.inf]
    others = [i for i in range(len(ds.cameras)) if i != image]
    zncc = np.full((len(depths), len(others)), -1.0)
    centre = np.full((len(depths), len(others), 2), np.nan)
    # The plane faces the query camera: every ray meets it at distance t
    # along the axis, so ray k is scaled by t / (ray_k . axis).
    along = rays @ axis
    for di, t in enumerate(depths):
        for vi, other in enumerate(others):
            oc = ctx.camera(other)
            if np.isfinite(t):
                pts = cam.center + rays * (t / along)[:, None]
                pc = pts @ oc.R.T + oc.t
            else:
                pc = rays @ oc.R.T
            front = -pc[:, 2]
            if np.any(front <= 1e-9):
                continue
            px = np.asarray(
                oc.intrinsics.ray_to_pixel_batch(
                    pc / np.linalg.norm(pc, axis=1, keepdims=True)
                ),
                float,
            )
            c = px[(n * n) // 2]
            if not in_frame(ctx, other, c, margin=radius_px * 0.5):
                continue
            vals = sample(grey[other], px)
            if not np.all(np.isfinite(vals)):
                continue
            zncc[di, vi] = zncc_rows(template, vals[None, :])[0]
            centre[di, vi] = c
    return depths, others, zncc, centre


def build_track(ctx, image: int, pixel, options: dict | None = None):
    from sfmtool._sfmtool import bench as B

    opts = {**DEFAULTS, **(options or {})}
    pixel = (float(pixel[0]), float(pixel[1]))
    diag: dict = {}
    near = ctx.observations_near(image, pixel, opts["size_radius_px"])
    half_px = [o["half_px"] for o in near[:10] if np.isfinite(o["half_px"])]
    radius_px = float(
        np.clip(
            np.median(half_px) if half_px else opts["default_radius_px"],
            opts["min_radius_px"],
            opts["max_radius_px"],
        )
    )
    radius_px *= opts["size_scale"]
    diag["radius_px"] = radius_px
    swept = sweep(ctx, image, pixel, radius_px, opts)
    if swept is None:
        raise TrackAtPixelError(
            "sweep", "the patch around the pixel has no texture to compare", diag
        )
    depths, others, zncc, centre = swept
    top = np.sort(zncc, axis=1)[:, ::-1][:, : opts["top_k"]]
    score = np.where(top > 0, top, 0).sum(axis=1)
    peaks = [
        i
        for i in range(len(depths))
        if score[i] > 0
        and (i == 0 or score[i] >= score[i - 1])
        and (i == len(depths) - 1 or score[i] >= score[i + 1])
    ]
    peaks.sort(key=lambda i: -score[i])
    diag["peaks"] = [
        {"depth": float(depths[i]), "score": float(score[i])} for i in peaks[:5]
    ]
    best, best_score, tried = None, -1.0, []
    for i in peaks[: opts["max_hypotheses"]]:
        views = [
            (others[v], centre[i, v])
            for v in np.argsort(-zncc[i])
            if zncc[i, v] >= opts["min_view_zncc"]
        ]
        record = {"depth": float(depths[i]), "views": [v for v, _ in views]}
        tried.append(record)
        if len(views) < opts["min_views"] - 1:
            record["error"] = "too few views agree"
            continue
        try:
            track = track_from_sightings(ctx, image, pixel, radius_px, views)
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
            "sweep",
            "no depth along the pixel's ray is agreed on by enough photographs",
            diag,
        )
    return finish(ctx, best, 0, pixel, opts, diag)
