# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment: the equidistant-fisheye columns of the structure-free focal vote.

Prototype gate for `specs/core/geometry/focal-vote.md` § "Camera-Model Columns
(Equidistant Fisheye)".  Nothing here touches `crates/` or `src/sfmtool/`;
it is a numpy/scipy stand-in for the kernel the spec section proposes, run
on real captures so the section's two search criteria can be checked before
any Rust lands.

What it builds
--------------
A **column** is a pixel->unit-ray map parameterized by a focal:

  * ``pinhole``     ray ~ ((x-cx)/f, (y-cy)/f, 1)
  * ``equidistant`` theta = r/f, ray = (sin th * dx/r, sin th * dy/r, cos th)

and a **cell** is a per-pair search criterion evaluated over a log grid of
candidate focals:

  * ``epipolar`` — robustly fit the ray-space epipolar matrix, cost =
    essentialness residual (s1 - s2) / (s1 + s2) of the consensus refit.
  * ``rotation`` — robustly fit a rotation of rays (Kabsch + trimming),
    cost = trimmed RMS angular residual over a frozen support set.

Estimation and residuals live on the ray SPHERE throughout — no ray is ever
projected to a normalized plane, because a >180 deg fisheye really does
produce ``theta >= 90 deg`` correspondences and those are exactly the
model-informative ones.  Epipolar residual = ``asin(|x2^T E x1|)`` for unit
rays; rotation residual = the angle between the fitted and the measured ray.

Both maps shrink every ray angle like ``1/f``, so a raw angular residual has
a slowly decaying tail across the focal grid.  The rotation cell survives it
only because its support set is FROZEN (found once by RANSAC over the grid,
then reused at every candidate): with a frozen support the angular minimum
is interior and lands on the same focal as the scale-normalized one, while a
per-focal inlier set lets a bad focal buy a low cost by keeping fewer points
and pins the minimum at the top of the grid.  Both costs are recorded —
``cost_rad`` (angular, what the gates use) and ``cost`` (the same residual
through the map's local pixels-per-radian ``dr/dtheta``: ``f`` for
equidistant, ``f (1 + (r/f)^2)`` for pinhole) — because the angular one
transfers across capture resolutions and the scaled one does not.

``--gate`` selects the RANSAC consensus bound: a fixed angle, or the same
angle derived per candidate focal from a pixel localization tolerance.  The
bound is an angle either way; only its numeric value differs.

The epipolar cell additionally runs a PLANE-SUBSET comparison arm: the
correspondences projectable at every candidate focal (a frozen population,
so the curve is not a population artifact) go through the native
``estimate_fundamental``, after which the consensus is re-selected
angularly on the sphere and refit.  That arm measures what reusing the
shipped RANSAC would cost, and what the beyond-hemisphere periphery is
worth.

Both cells are run under BOTH columns, with the same cost function, the same
grid, the same RANSAC sample sets and the same gates, so the four cells are
directly comparable and the spec's "model verdict = column with the greater
certified mass" can be measured.  (The shipped pinhole kernel uses closed
forms — Bougnoux, ``K^-1 H K`` orthogonality — rather than a scan; the
native ``focal_vote`` binding is also run per dataset and reported as the
reference pinhole verdict.)

Run::

    pixi run -e dev python scripts/exp_fisheye_vote.py [--out DIR] [--quick]

Writes ``results.json`` (per-pair rows, full cost curves, per-dataset
summaries) plus ``pairs.tsv`` / ``summary.tsv`` under the output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from sfmtool._sfmtool.geometry import (
    estimate_fundamental,
    focal_vote as native_focal_vote,
)
from sfmtool._sfmtool.io import MatchesFile, read_sift_partial

# ── Datasets ─────────────────────────────────────────────────────────────────
#
# ``gt_focal_px`` is the calibrated focal for fisheye captures (the rig
# config's OPENCV_FISHEYE fx/fy mean) and the accepted pinhole focal for the
# controls.  ``gt_model`` is the truth label the model verdict is scored
# against.  ``gt_distortion`` holds the OPENCV_FISHEYE k1..k4 so the
# equidistant column's own ground truth (the best-fit equidistant focal over
# the observed radius distribution) can be derived — the calibrated model is
# NOT equidistant.

PROTO = Path("C:/DataSets/workspace-prep/fisheye-vote-proto")

DATASETS = [
    {
        "name": "kerry_park",
        "ws": PROTO / "kerry_ws",
        "matches": PROTO / "kerry_ws/tvg-matches/kerry_park-clusters.matches",
        "gt_model": "equidistant",
        "gt_focal_px": 0.5 * (129.1499937015594 + 129.2573627423474),
        "gt_distortion": [
            0.038113353966529886,
            -0.00800851799065643,
            0.008329720504707577,
            -0.0026901578801066814,
        ],
        "note": "checked-in test-data/images/kerry_park, 24 frames x 2 fisheyes @480",
    },
    {
        "name": "kp360_sub",
        "ws": PROTO / "kp360_ws",
        "matches": PROTO / "kp360_ws/tvg-matches/kp360-clusters.matches",
        "gt_model": "equidistant",
        "gt_focal_px": 0.5 * (257.9354095 + 257.43220425),
        "gt_distortion": [0.042219, -0.011493, 0.010094, -0.003034],
        "note": "KerryPark360 frames 300..620 step 8, native 960x960",
    },
    {
        "name": "c2f_seoul_b100",
        "ws": Path("C:/DataSets/c2f_seoul/b100"),
        "matches": Path(
            "C:/DataSets/c2f_seoul/b100/matches/lvl-clusters-patches.matches"
        ),
        "gt_model": "pinhole",
        "gt_focal_px": 320.3154226420564,
        "gt_distortion": None,
        "note": "pinhole control (phone video, 270x480)",
    },
    {
        "name": "dino_dog_toy",
        "ws": Path("C:/DataSets/DinoDogToyWS"),
        "matches": Path(
            "C:/DataSets/DinoDogToyWS/matches/dino_dog_toy-clusters-patches.matches"
        ),
        "gt_model": "pinhole",
        "gt_focal_px": None,  # filled from the native pinhole vote
        "gt_distortion": None,
        "note": "pinhole control (DSLR stills, 2040x1536)",
    },
]

# ── Search configuration ─────────────────────────────────────────────────────

BAND_LO, BAND_HI = 0.075, 3.0  # x max(width, height); the wide scan band
N_GRID = 64  # log-spaced candidate focals (coarse pass uses N_GRID // 3)
N_SAMPLES = 128  # RANSAC minimal samples, drawn once per pair and reused
MAX_CORR = 600  # correspondences per pair fed to the estimators
MAX_PAIRS = 24  # candidate pairs per dataset
MAX_CLUSTER_SIZE = 10  # skip mega-clusters when expanding correspondences
TOL_PX = 3.0  # scale-derived consensus bound, pixels (--gate pixel)
GATE_RAD = 0.02  # fixed angular consensus bound, radians (--gate angular)
TRIM_Q = 0.90  # rotation-cell trimmed-residual quantile inside the support
MIN_ROT_SUPPORT = 20  # far-field correspondences a rotation vote needs

# FOV-derived window for the equidistant column: under theta = r / f the
# focal and the field of view are the same statement, f = r_edge / th_edge,
# so credible focals are a credible half-FOV range evaluated at the image's
# own edge radius.  The scan still runs on the wide band (comparability and
# honest shape metrics); this window is what the vote is restricted to.
FOV_HALF_LO_DEG, FOV_HALF_HI_DEG = 50.0, 110.0
GATE_MODE = "pixel"  # set from --gate


# ── Ray maps ─────────────────────────────────────────────────────────────────


def rays_pinhole(uv: np.ndarray, f: float) -> np.ndarray:
    """Unit rays for principal-point-centred pixels under the pinhole map."""
    v = np.empty((len(uv), 3))
    v[:, :2] = uv / f
    v[:, 2] = 1.0
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rays_equidistant(uv: np.ndarray, f: float) -> np.ndarray:
    """Unit rays for principal-point-centred pixels under theta = r / f."""
    r = np.linalg.norm(uv, axis=1)
    th = r / f
    s = np.divide(np.sin(th), r, out=np.zeros_like(r), where=r > 1e-12)
    v = np.empty((len(uv), 3))
    v[:, 0] = uv[:, 0] * s
    v[:, 1] = uv[:, 1] * s
    v[:, 2] = np.cos(th)
    return v


def pixels_pinhole(v: np.ndarray, f: float) -> np.ndarray:
    """Rays back to centred pixels; rays at or behind the plane give NaN."""
    z = np.where(v[..., 2] > 1e-9, v[..., 2], np.nan)
    return f * v[..., :2] / z[..., None]


def pixels_equidistant(v: np.ndarray, f: float) -> np.ndarray:
    th = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
    rho = np.linalg.norm(v[..., :2], axis=-1)
    s = np.divide(f * th, rho, out=np.zeros_like(rho), where=rho > 1e-12)
    return v[..., :2] * s[..., None]


def scale_pinhole(uv: np.ndarray, f: float) -> np.ndarray:
    """Local dr/dtheta of the pinhole map at each pixel (pixels per radian)."""
    r = np.linalg.norm(uv, axis=-1)
    return f * (1.0 + (r / f) ** 2)


def scale_equidistant(uv: np.ndarray, f: float) -> np.ndarray:
    return np.full(uv.shape[:-1], f)


COLUMNS = {
    "pinhole": (rays_pinhole, pixels_pinhole, scale_pinhole),
    "equidistant": (rays_equidistant, pixels_equidistant, scale_equidistant),
}


def column_valid(column: str, f: float, r_hi: float) -> bool:
    """Whether a candidate focal keeps the column's map injective.

    The equidistant map folds once ``theta = r / f`` passes pi; the pinhole
    map is injective for every positive focal.
    """
    return column != "equidistant" or (r_hi / f) < math.pi


# ── Robust ray-space estimators ──────────────────────────────────────────────


def _epipolar_rows(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """(N, 9) rows of the linear constraint r2^T E r1 = 0."""
    return (r2[:, :, None] * r1[:, None, :]).reshape(len(r1), 9)


def _null_vectors(a: np.ndarray) -> np.ndarray:
    """Right singular vector of the smallest singular value, batched or not."""
    _, _, vt = np.linalg.svd(a)
    return vt[..., -1, :]


def _epi_residual(
    e: np.ndarray, r1: np.ndarray, r2: np.ndarray, side: int
) -> np.ndarray:
    """Angular epipolar residual, batched over leading axes of ``e``.

    ``side=2`` measures the angle between each image-2 ray and the epipolar
    plane ``E r1`` (a residual "in image 2"); ``side=1`` measures the image-1
    ray against ``E^T r2``.  Both are true angles, so they are comparable
    across candidate focals; the two sides are what the direction-agreement
    analog is built from.
    """
    if side == 2:
        n = r1 @ np.swapaxes(e, -1, -2)  # (..., N, 3) = (E r1)^T
        other = r2
    else:
        n = r2 @ e  # (..., N, 3) = (E^T r2)^T
        other = r1
    nn = np.linalg.norm(n, axis=-1)
    dot = np.abs(np.einsum("...ij,ij->...i", n, other))
    return np.arcsin(np.clip(dot / np.maximum(nn, 1e-15), 0.0, 1.0))


def fit_epipolar(
    r1: np.ndarray,
    r2: np.ndarray,
    tol_rad: np.ndarray,
    samples: np.ndarray,
    side: int,
) -> dict | None:
    """LO-RANSAC ray-space epipolar matrix with a fixed minimal-sample set.

    Everything is on the unit sphere: the consensus bound is the angular
    ray-to-epipolar-plane residual, so rays beyond the hemisphere take part
    like any others.  The sample index sets are drawn once per pair and
    reused for every candidate focal, so the cost curve carries no RANSAC
    jitter.
    """
    a = _epipolar_rows(r1, r2)
    cand = _null_vectors(a[samples]).reshape(-1, 3, 3)  # (S, 3, 3)
    res = _epi_residual(cand, r1, r2, side)  # (S, N), radians
    best = int(np.argmax((res < tol_rad).sum(axis=1)))
    inl = res[best] < tol_rad
    if inl.sum() < 8:
        return None
    e = None
    for _ in range(3):  # local optimization on the consensus set
        e = _null_vectors(a[inl]).reshape(3, 3)
        new = _epi_residual(e, r1, r2, side) < tol_rad
        if new.sum() < 8:
            break
        done = np.array_equal(new, inl)
        inl = new
        if done:
            break
    if e is None or inl.sum() < 8:
        return None
    sv = np.linalg.svd(e, compute_uv=False)
    ess = float((sv[0] - sv[1]) / (sv[0] + sv[1]))
    return {
        "cost": ess,
        "inliers": inl,
        "n_inliers": int(inl.sum()),
        "resid_rad": _epi_residual(e, r1, r2, side)[inl],
    }


def plane_population(column: str, uv1: np.ndarray, uv2: np.ndarray, f_min: float):
    """Correspondences projectable to the z=1 plane at EVERY candidate focal.

    The population is frozen at the smallest focal of the scan, so the
    plane-subset arm's cost curve is not a population artifact.  Under the
    equidistant map ``r < f pi/2`` is exactly ``theta < 90 deg``; the pinhole
    map images every pixel, so nothing is dropped there.
    """
    if column != "equidistant":
        return np.ones(len(uv1), bool)
    r_cap = 0.999 * f_min * math.pi / 2.0
    return (np.linalg.norm(uv1, axis=1) < r_cap) & (np.linalg.norm(uv2, axis=1) < r_cap)


def fit_epipolar_plane(
    r1: np.ndarray, r2: np.ndarray, tol_rad: np.ndarray, side: int, seed: int
) -> dict | None:
    """Epipolar matrix from the NATIVE estimator on normalized-plane points.

    The hypothesis comes from ``estimate_fundamental`` run on the z=1
    projections (this is the arm that would let the kernel reuse the shipped
    RANSAC instead of a hand-rolled sphere sampler).  Its planar inlier
    bound overweights the periphery as ``tan theta`` grows, so the consensus
    is re-selected ANGULARLY on the unit rays and refit there, exactly as in
    the sphere arm; a plane-homogeneous F is already the ray-space epipolar
    matrix, so no conversion is needed.
    """
    p1 = np.ascontiguousarray(r1[:, :2] / r1[:, 2:3])
    p2 = np.ascontiguousarray(r2[:, :2] / r2[:, 2:3])
    # Angular tolerance -> plane units: d(tan th)/d th = 1 / cos^2 th, taken
    # at the population's median obliquity.
    stretch = float(np.median(1.0 / np.maximum(r2[:, 2] ** 2, 1e-6)))
    est = estimate_fundamental(
        p1,
        p2,
        max_error_px=float(np.median(tol_rad) * stretch),
        min_inliers=8,
        seed=seed,
    )
    if est is None:
        return None
    a = _epipolar_rows(r1, r2)
    e = np.asarray(est["f_matrix"])
    inl = None
    for _ in range(3):
        new = _epi_residual(e, r1, r2, side) < tol_rad
        if new.sum() < 8:
            return None
        done = inl is not None and np.array_equal(new, inl)
        inl = new
        if done:
            break
        e = _null_vectors(a[inl]).reshape(3, 3)
    sv = np.linalg.svd(e, compute_uv=False)
    return {
        "cost": float((sv[0] - sv[1]) / (sv[0] + sv[1])),
        "inliers": inl,
        "n_inliers": int(inl.sum()),
        "resid_rad": _epi_residual(e, r1, r2, side)[inl],
    }


def _kabsch(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Rotation taking ``r1`` onto ``r2`` in the least-squares sense."""
    u, _, vt = np.linalg.svd(r1.T @ r2)
    d = np.sign(np.linalg.det(u @ vt))
    return (u * np.array([1.0, 1.0, d])) @ vt


def _kabsch_batch(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Batched Kabsch over (S, k, 3) minimal samples -> (S, 3, 3)."""
    u, _, vt = np.linalg.svd(np.swapaxes(r1, -1, -2) @ r2)
    d = np.sign(np.linalg.det(u @ vt))
    u = u * np.stack([np.ones_like(d), np.ones_like(d), d], axis=-1)[:, None, :]
    return u @ vt


def _rot_residual(r1: np.ndarray, r2: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Angle between the rotated and the measured ray, batched over ``rot``."""
    pred = r1 @ np.swapaxes(rot, -1, -2)
    return np.arccos(np.clip(np.einsum("...ij,ij->...i", pred, r2), -1.0, 1.0))


def rotation_support(
    r1: np.ndarray, r2: np.ndarray, tol_rad: np.ndarray, samples: np.ndarray
) -> np.ndarray | None:
    """LO-RANSAC far-field support set of a pair at one candidate focal."""
    cand = _kabsch_batch(r1[samples], r2[samples])
    res = _rot_residual(r1, r2, cand)
    best = int(np.argmax((res < tol_rad).sum(axis=1)))
    inl = res[best] < tol_rad
    if inl.sum() < MIN_ROT_SUPPORT:
        return None
    for _ in range(3):
        new = _rot_residual(r1, r2, _kabsch(r1[inl], r2[inl])) < tol_rad
        if new.sum() < MIN_ROT_SUPPORT:
            return None
        done = np.array_equal(new, inl)
        inl = new
        if done:
            break
    return inl


def fit_rotation(
    r1: np.ndarray, r2: np.ndarray, px_scale: np.ndarray, support: np.ndarray
) -> dict:
    """Rotation fitted on a FIXED support set; angular and scaled costs.

    Freezing the support (the pair's far-field population, found once by
    RANSAC over the grid) is what makes the cost comparable across candidate
    focals: a per-focal inlier set would let a bad focal buy a low cost by
    keeping fewer points.  ``cost_rad`` is the trimmed RMS angular residual
    the spec names; ``cost`` is the same residual carried through the map's
    local pixels-per-radian, which is what removes the ``1/f`` drift.
    """
    idx = np.flatnonzero(support)
    keep = idx
    rot = _kabsch(r1[keep], r2[keep])
    k = max(MIN_ROT_SUPPORT, int(round(TRIM_Q * len(idx))))
    for _ in range(2):  # light trimming inside the frozen support
        res = _rot_residual(r1[idx], r2[idx], rot)
        keep = idx[np.argsort(res)[:k]]
        rot = _kabsch(r1[keep], r2[keep])
    res = _rot_residual(r1[idx], r2[idx], rot)
    order = np.argsort(res)[:k]
    return {
        "cost": float(np.sqrt(np.mean((res[order] * px_scale[idx][order]) ** 2))),
        "cost_rad": float(np.sqrt(np.mean(res[order] ** 2))),
        "inliers": support,
        "n_inliers": int(support.sum()),
        "resid_rad": res[order],
    }


# ── Scan over candidate focals ───────────────────────────────────────────────


def scan(
    cell: str,
    column: str,
    uv1: np.ndarray,
    uv2: np.ndarray,
    grid: np.ndarray,
    samples: np.ndarray,
    r_hi: float,
    side: int = 2,
    fov_window: tuple | None = None,
) -> dict:
    """Cost curve of one cell x column over ``grid``, plus its shape metrics."""
    ray, _to_px, scale = COLUMNS[column]
    valid = [k for k, f in enumerate(grid) if column_valid(column, float(f), r_hi)]
    uv_side = uv1 if side == 1 else uv2

    def tol(f: float) -> np.ndarray:
        """Per-point angular consensus bound at this candidate focal."""
        if GATE_MODE == "angular":
            return np.full(len(uv_side), GATE_RAD)
        return TOL_PX / scale(uv_side, f)

    empty = (np.full(len(grid), np.nan), np.zeros(len(grid), int), grid, {})
    support = None
    if cell == "rotation":
        # One pass over a coarse sub-grid fixes the pair's far-field support:
        # the largest rotation consensus any candidate focal can muster.
        best_n = 0
        for k in valid[::4]:
            f = float(grid[k])
            s = rotation_support(ray(uv1, f), ray(uv2, f), tol(f), samples)
            if s is not None and int(s.sum()) > best_n:
                best_n, support = int(s.sum()), s
        if support is None:
            return _curve_metrics(*empty, fov_window=fov_window)
    costs = np.full(len(grid), np.nan)
    costs_rad = np.full(len(grid), np.nan)
    ninl = np.zeros(len(grid), int)
    inliers_at, resid_at = {}, {}
    t0 = time.perf_counter()
    for k in valid:
        f = float(grid[k])
        r1, r2 = ray(uv1, f), ray(uv2, f)
        if cell == "epipolar":
            out = fit_epipolar(r1, r2, tol(f), samples, side)
        else:
            out = fit_rotation(r1, r2, scale(uv_side, f), support)
        if out is None:
            continue
        costs[k] = out["cost"]
        costs_rad[k] = out.get("cost_rad", out["cost"])
        ninl[k] = out["n_inliers"]
        inliers_at[k] = out["inliers"]
        resid_at[k] = out["resid_rad"]
    seconds = time.perf_counter() - t0
    m = _curve_metrics(costs, ninl, grid, inliers_at, fov_window)
    m["seconds"] = seconds
    m["seconds_per_point"] = seconds / max(len(valid), 1)
    m["costs_rad"] = costs_rad.tolist()
    kmin = m.pop("_kmin")
    r = resid_at.get(kmin)
    m["resid_rad_p50"] = float(np.median(r)) if r is not None and len(r) else None
    m["resid_rad_p90"] = (
        float(np.quantile(r, 0.9)) if r is not None and len(r) else None
    )
    # Raw-angular counterpart of the same scan: where the minimum sits when
    # the cost is NOT carried through the map scale (the spec's literal
    # wording for the rotation cell).
    ok = np.isfinite(costs_rad)
    if ok.sum() >= 5:
        ridx = np.flatnonzero(ok)
        rm = _minimum(costs_rad, grid, ridx)
        m["rad_focal_px"] = rm["focal_px"]
        m["rad_sharpness"] = rm["sharpness"]
        m["rad_edge"] = rm["edge"]
        m["cost_min_rad"] = rm["cost_min"]
    else:
        m["rad_focal_px"] = m["rad_sharpness"] = m["rad_edge"] = None
        m["cost_min_rad"] = None
    return m


def scan_plane(
    column: str,
    uv1: np.ndarray,
    uv2: np.ndarray,
    grid: np.ndarray,
    fov_window: tuple,
    side: int = 2,
) -> dict:
    """Plane-subset comparison arm of the epipolar cell, on the FOV window.

    Runs on the same grid POINTS as the sphere arm's FOV-window restriction,
    so voted focal, minimum sharpness and certified counts compare directly;
    the only differences are the frozen projectable population and the
    native estimator supplying the hypothesis.
    """
    ray, _to_px, scale = COLUMNS[column]
    lo, hi = fov_window
    widx = np.flatnonzero((grid >= lo) & (grid <= hi))
    out = {
        "focal_px": None,
        "cost_min": None,
        "cost_median": None,
        "sharpness": None,
        "rel_width": None,
        "edge": None,
        "n_inliers_at_min": 0,
        "n_population": 0,
        "frac_population": 0.0,
        "coverage_p90": None,
        "seconds": 0.0,
        "seconds_per_point": 0.0,
    }
    if len(widx) < 5:
        return out
    sub = plane_population(column, uv1, uv2, float(grid[widx[0]]))
    out["n_population"] = int(sub.sum())
    out["frac_population"] = float(sub.mean())
    if sub.sum() < 40:
        return out
    a1, a2 = uv1[sub], uv2[sub]
    uv_side = a1 if side == 1 else a2
    t0 = time.perf_counter()
    costs = np.full(len(grid), np.nan)
    ninl = np.zeros(len(grid), int)
    inl_at = {}
    for k in widx:
        f = float(grid[k])
        tol = (
            np.full(len(uv_side), GATE_RAD)
            if GATE_MODE == "angular"
            else TOL_PX / scale(uv_side, f)
        )
        res = fit_epipolar_plane(ray(a1, f), ray(a2, f), tol, side, seed=k)
        if res is None:
            continue
        costs[k] = res["cost"]
        ninl[k] = res["n_inliers"]
        inl_at[k] = res["inliers"]
    out["seconds"] = time.perf_counter() - t0
    out["seconds_per_point"] = out["seconds"] / max(len(widx), 1)
    ok = np.flatnonzero(np.isfinite(costs))
    if len(ok) < 5:
        return out
    m = _minimum(costs, grid, ok)
    out.update(
        {
            s: m[s]
            for s in (
                "focal_px",
                "cost_min",
                "cost_median",
                "sharpness",
                "rel_width",
                "edge",
            )
        }
    )
    out["n_inliers_at_min"] = int(ninl[m["k"]])
    inl = inl_at.get(m["k"])
    if inl is not None:
        r_in = np.r_[np.linalg.norm(a1[inl], axis=1), np.linalg.norm(a2[inl], axis=1)]
        out["_r_in_p90"] = float(np.quantile(r_in, 0.90))
    out["costs"] = costs.tolist()
    return out


def _minimum(costs: np.ndarray, grid: np.ndarray, idx: np.ndarray) -> dict:
    """Location and shape of the cost minimum over the grid points ``idx``."""
    lg = np.log(grid)
    k = int(idx[np.argmin(costs[idx])])
    c_min, c_med = float(costs[k]), float(np.median(costs[idx]))
    near = idx[costs[idx] < 2.0 * max(c_min, 1e-12)]
    span = lg[idx[-1]] - lg[idx[0]]
    # Parabolic refinement of the winning bracket in log f.
    f_star = float(grid[k])
    if idx[0] < k < idx[-1] and np.isfinite(costs[k - 1]) and np.isfinite(costs[k + 1]):
        y0, y1, y2 = costs[k - 1], costs[k], costs[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            step = 0.5 * (y0 - y2) / den
            if abs(step) <= 1.0:
                f_star = float(np.exp(lg[k] + step * (lg[k + 1] - lg[k])))
    return {
        "k": k,
        "focal_px": f_star,
        "cost_min": c_min,
        "cost_median": c_med,
        "sharpness": c_min / c_med if c_med > 0 else 1.0,
        "rel_width": float((lg[near[-1]] - lg[near[0]]) / span) if span > 0 else 1.0,
        "edge": bool(k == idx[0] or k == idx[-1]),
    }


def _curve_metrics(
    costs: np.ndarray,
    ninl: np.ndarray,
    grid: np.ndarray,
    inliers_at: dict,
    fov_window: tuple | None = None,
) -> dict:
    ok = np.isfinite(costs)
    out = {
        "costs": costs.tolist(),
        "n_inliers": ninl.tolist(),
        "n_valid": int(ok.sum()),
        "focal_px": None,
        "cost_min": None,
        "cost_median": None,
        "sharpness": None,
        "rel_width": None,
        "edge": None,
        "n_inliers_at_min": 0,
        "inliers_at_min": None,
        "fov_focal_px": None,
        "fov_cost_min": None,
        "fov_sharpness": None,
        "fov_edge": None,
        "fov_contains_global_min": None,
        "_kmin": -1,
    }
    if ok.sum() < 5:
        return out
    idx = np.flatnonzero(ok)
    g = _minimum(costs, grid, idx)
    k = g["k"]
    out.update(
        {
            s: g[s]
            for s in (
                "focal_px",
                "cost_min",
                "cost_median",
                "sharpness",
                "rel_width",
                "edge",
            )
        }
    )
    out["_kmin"] = k
    out["n_inliers_at_min"] = int(ninl[k])
    out["inliers_at_min"] = inliers_at.get(k)
    if fov_window is not None:
        lo, hi = fov_window
        widx = idx[(grid[idx] >= lo) & (grid[idx] <= hi)]
        if len(widx) >= 3:
            wm = _minimum(costs, grid, widx)
            out["fov_focal_px"] = wm["focal_px"]
            out["fov_cost_min"] = wm["cost_min"]
            out["fov_sharpness"] = wm["sharpness"]
            out["fov_edge"] = wm["edge"]
            out["fov_contains_global_min"] = bool(lo <= grid[k] <= hi)
    return out


# ── Data loading ─────────────────────────────────────────────────────────────


def load_dataset(spec: dict) -> dict:
    """Cluster observations of one capture as flat arrays."""
    mf = MatchesFile(str(spec["matches"]))
    names = list(mf.image_names)
    dims = np.asarray(mf.image_dims).astype(np.int64)
    starts = np.asarray(mf.cluster_starts).astype(np.int64)
    mimg = np.asarray(mf.member_images).astype(np.int64)
    if mf.has_cluster_patches:
        pos = np.asarray(mf.member_positions())
    else:
        mfeat = np.asarray(mf.member_features).astype(np.int64)
        pos = np.zeros((len(mfeat), 2))
        from sfmtool.sift.file import get_sift_path_for_image

        for j in np.unique(mimg):
            m = mimg == j
            p = read_sift_partial(
                str(get_sift_path_for_image(spec["ws"] / names[j])),
                int(mfeat[m].max()) + 1,
            )["positions_xy"]
            pos[m] = p[mfeat[m]].astype(np.float64)
    if len(np.unique(dims, axis=0)) != 1:
        raise SystemExit(f"{spec['name']}: mixed image dimensions, unsupported")
    w, h = int(dims[0, 0]), int(dims[0, 1])
    return {
        "names": names,
        "width": w,
        "height": h,
        "starts": starts,
        "member_images": mimg,
        "positions": pos,
    }


def pair_tables(data: dict) -> dict:
    """Shared-cluster count and mean displacement per covisible image pair."""
    starts, mimg, pos = data["starts"], data["member_images"], data["positions"]
    acc: dict[tuple[int, int], list] = {}
    for c in range(len(starts) - 1):
        lo, hi = int(starts[c]), int(starts[c + 1])
        if hi - lo < 2 or hi - lo > MAX_CLUSTER_SIZE:
            continue
        for a in range(lo, hi):
            for b in range(a + 1, hi):
                i, j = int(mimg[a]), int(mimg[b])
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                ka, kb = (a, b) if i < j else (b, a)
                d = float(np.hypot(*(pos[ka] - pos[kb])))
                e = acc.get(key)
                if e is None:
                    acc[key] = [1, d, [(ka, kb)]]
                else:
                    e[0] += 1
                    e[1] += d
                    e[2].append((ka, kb))
    return acc


def select_epipolar_pairs(acc: dict, width: int, height: int) -> list:
    """Spec's epipolar candidates: most-covisible pairs with some parallax."""
    diag = math.hypot(width, height)
    rows = [
        (k, v[0], v[1] / v[0], v[2])
        for k, v in acc.items()
        if v[0] >= 30 and (v[1] / v[0]) >= 0.02 * diag
    ]
    rows.sort(key=lambda r: -r[1])
    used: dict[int, int] = {}
    out = []
    for key, n, mdisp, links in rows:
        i, j = key
        if used.get(i, 0) >= 2 or used.get(j, 0) >= 2:
            continue
        used[i] = used.get(i, 0) + 1
        used[j] = used.get(j, 0) + 1
        out.append({"images": key, "n_shared": n, "mean_disp": mdisp, "links": links})
        if len(out) >= MAX_PAIRS:
            break
    return out


def select_rotation_pairs(acc: dict, width: int, height: int, n_images: int) -> list:
    """Spec's rotation candidates: each image's widest well-covisible partner.

    Small-displacement homographies are near identity and observe no focal,
    so the partner must also be far enough away; each unordered pair is
    taken once.
    """
    diag = math.hypot(width, height)
    widest: dict[int, tuple] = {}
    for (i, j), v in acc.items():
        if v[0] < 25:
            continue
        d = v[1] / v[0]
        if d < 0.08 * diag:
            continue
        for a, b in ((i, j), (j, i)):
            if a not in widest or d > widest[a][0]:
                widest[a] = (d, b, (i, j), v[0], v[2])
    seen, out = set(), []
    for _, (d, _b, key, n, links) in sorted(widest.items()):
        if key in seen:
            continue
        seen.add(key)
        out.append({"images": key, "n_shared": n, "mean_disp": d, "links": links})
    out.sort(key=lambda r: -r["n_shared"])
    return out[:MAX_PAIRS]


# ── Ground truth for the equidistant column ──────────────────────────────────


def opencv_fisheye_forward(th: np.ndarray, f: float, k: list) -> np.ndarray:
    """r = f * (th + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9)."""
    t2 = th * th
    return f * th * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def calibrated_theta_lut(spec: dict) -> tuple | None:
    """(r, theta) lookup for the capture's calibrated OPENCV_FISHEYE model.

    Truncated at the theta where ``dr/dtheta`` first turns over: past that
    radius the calibration is not invertible, which is also the practical
    edge of the imaged circle.
    """
    if spec["gt_distortion"] is None or spec["gt_focal_px"] is None:
        return None
    th = np.linspace(1e-6, math.pi, 20001)
    r = opencv_fisheye_forward(th, spec["gt_focal_px"], spec["gt_distortion"])
    turn = np.flatnonzero(np.diff(r) <= 0)
    stop = int(turn[0]) if len(turn) else len(th) - 1
    return r[: stop + 1], th[: stop + 1]


def equidistant_gt(spec: dict, radii: np.ndarray) -> dict | None:
    """Best-fit equidistant focal for the capture's calibrated fisheye model.

    The equidistant column cannot recover the calibrated focal when the true
    model is OPENCV_FISHEYE; what it *can* recover is the ``f`` best matching
    ``r = f theta`` over the radii the correspondences actually occupy.  The
    calibrated polynomial is inverted by monotone interpolation, truncated at
    the theta where ``dr/dtheta`` first turns over (beyond that radius the
    calibration itself is not invertible, i.e. outside the imaged circle).
    """
    lut = calibrated_theta_lut(spec)
    if lut is None:
        return None
    f_cal = spec["gt_focal_px"]
    r_grid, th_grid = lut
    used = radii[radii <= r_grid[-1]]
    if len(used) < 100:
        return None
    th = np.interp(used, r_grid, th_grid)
    f_px = float(np.sum(used * th) / np.sum(th**2))  # pixel-domain LS
    f_ang = float(np.sum(used**2) / np.sum(used * th))  # angle-domain LS
    return {
        "f_equidistant_px": f_px,
        "f_equidistant_angle_px": f_ang,
        "bias_vs_calibrated": float(math.log(f_px / f_cal)),
        "r_max_valid": float(r_grid[-1]),
        "frac_radii_used": float(len(used) / len(radii)),
        "theta_p90_deg": float(np.degrees(np.quantile(th, 0.90))),
        "theta_max_deg": float(np.degrees(th.max())),
        "r_p90": float(np.quantile(radii, 0.90)),
    }


# ── Per-dataset run ──────────────────────────────────────────────────────────


def run_dataset(spec: dict, quick: bool) -> dict:
    t0 = time.perf_counter()
    data = load_dataset(spec)
    w, h = data["width"], data["height"]
    cx, cy = 0.5 * w, 0.5 * h
    maxdim, halfdiag = float(max(w, h)), 0.5 * math.hypot(w, h)
    theta_lut = calibrated_theta_lut(spec)
    acc = pair_tables(data)
    groups = {
        "epipolar": select_epipolar_pairs(acc, w, h),
        "rotation": select_rotation_pairs(acc, w, h, len(data["names"])),
    }
    pos = data["positions"]
    print(
        f"[{spec['name']}] {len(data['names'])} images {w}x{h}, "
        f"{len(data['starts']) - 1} clusters, {len(acc)} covisible pairs, "
        f"epi-cands={len(groups['epipolar'])} rot-cands={len(groups['rotation'])}"
        f"  [{time.perf_counter() - t0:.1f}s]"
    )

    # Reference pinhole verdict from the shipped kernel.
    starts, mimg = data["starts"], data["member_images"]
    sizes = np.diff(starts)
    cl_of = np.repeat(np.arange(len(sizes)), sizes)
    native = native_focal_vote(
        cl_of.astype(np.uint32), mimg.astype(np.uint32), pos, w, h, seed=0
    )

    n_grid = N_GRID // 3 if quick else N_GRID
    rng = np.random.default_rng(12345)
    rows = []
    all_radii = []
    for cell, pairs in groups.items():
        for pi, pair in enumerate(pairs):
            links = pair["links"]
            if len(links) > MAX_CORR:
                sel = rng.choice(len(links), MAX_CORR, replace=False)
                links = [links[int(s)] for s in sel]
            ka = np.array([a for a, _ in links])
            kb = np.array([b for _, b in links])
            uv1 = pos[ka] - np.array([cx, cy])
            uv2 = pos[kb] - np.array([cx, cy])
            n = len(uv1)
            if n < 40:
                continue
            rad = np.r_[np.linalg.norm(uv1, axis=1), np.linalg.norm(uv2, axis=1)]
            r_hi = float(np.quantile(rad, 0.99))
            all_radii.append(rad)
            grid = np.exp(
                np.linspace(
                    math.log(BAND_LO * maxdim), math.log(BAND_HI * maxdim), n_grid
                )
            )
            # FOV window: f = r_edge / theta_edge at this pair's own edge
            # radius (the p99 of its correspondences).
            fov_window = (
                r_hi / math.radians(FOV_HALF_HI_DEG),
                r_hi / math.radians(FOV_HALF_LO_DEG),
            )
            s_epi = rng.integers(0, n, size=(N_SAMPLES, 8))
            s_rot = rng.integers(0, n, size=(N_SAMPLES, 3))

            row = {
                "dataset": spec["name"],
                "cell": cell,
                "pair": [int(pair["images"][0]), int(pair["images"][1])],
                "n_shared": int(pair["n_shared"]),
                "mean_disp_frac": pair["mean_disp"] / math.hypot(w, h),
                "n_corr": n,
                "cells": {},
            }
            for column in COLUMNS:
                samples = s_epi if cell == "epipolar" else s_rot
                res = scan(
                    cell, column, uv1, uv2, grid, samples, r_hi, fov_window=fov_window
                )
                inl = res.pop("inliers_at_min")
                if inl is not None:
                    r_in = np.r_[
                        np.linalg.norm(uv1[inl], axis=1),
                        np.linalg.norm(uv2[inl], axis=1),
                    ]
                    res["coverage_p90"] = float(np.quantile(r_in, 0.90) / halfdiag)
                    res["coverage_p50"] = float(np.quantile(r_in, 0.50) / halfdiag)
                    # Beyond-hemisphere participation: theta of the vote's own
                    # inliers under the calibrated model where one exists,
                    # else under the column's winning equidistant focal.
                    th_in = (
                        np.interp(r_in, *theta_lut)
                        if theta_lut is not None
                        else r_in / max(res["focal_px"] or 1.0, 1e-6)
                    )
                    res["theta_p90_deg"] = float(np.degrees(np.quantile(th_in, 0.9)))
                    res["frac_theta_ge_90"] = float(np.mean(th_in >= 0.5 * math.pi))
                else:
                    res["coverage_p90"] = res["coverage_p50"] = None
                    res["theta_p90_deg"] = res["frac_theta_ge_90"] = None
                if cell == "epipolar":
                    # Direction-agreement analog: the same scan with the
                    # residual measured on the other side.
                    other = scan(
                        cell,
                        column,
                        uv1,
                        uv2,
                        grid,
                        s_epi,
                        r_hi,
                        side=1,
                        fov_window=fov_window,
                    )
                    other.pop("inliers_at_min", None)
                    res["focal_px_side1"] = other["focal_px"]
                    res["cost_min_side1"] = other["cost_min"]
                    if res["focal_px"] and other["focal_px"]:
                        res["dir_disagreement"] = abs(
                            math.log(res["focal_px"] / other["focal_px"])
                        )
                    else:
                        res["dir_disagreement"] = None
                    # Plane-subset comparison arm (native estimate_fundamental
                    # on the projectable population).
                    pl = scan_plane(column, uv1, uv2, grid, fov_window)
                    if pl.get("_r_in_p90") is not None:
                        pl["coverage_p90"] = pl.pop("_r_in_p90") / halfdiag
                    pl.pop("_r_in_p90", None)
                    pl1 = scan_plane(column, uv1, uv2, grid, fov_window, side=1)
                    pl["dir_disagreement"] = (
                        abs(math.log(pl["focal_px"] / pl1["focal_px"]))
                        if pl["focal_px"] and pl1["focal_px"]
                        else None
                    )
                    pl["seconds"] += pl1["seconds"]
                    res["plane"] = pl
                row["cells"][column] = res
            rows.append(row)
            print(
                f"  {cell} {pi + 1}/{len(pairs)} {row['pair']} n={n} "
                + " ".join(
                    f"{k}={(v['focal_px'] or float('nan')):.0f}"
                    f"/{(v['cost_min'] or float('nan')):.3f}"
                    f"/{(v['sharpness'] or float('nan')):.2f}"
                    for k, v in row["cells"].items()
                )
            )

    radii = np.concatenate(all_radii) if all_radii else np.zeros(1)
    theta_stats = None
    if theta_lut is not None:
        r_grid, th_grid = theta_lut
        th = np.interp(radii, r_grid, th_grid)
        theta_stats = {
            "r_max_invertible": float(r_grid[-1]),
            "r_at_theta_90": float(np.interp(0.5 * math.pi, th_grid, r_grid)),
            "half_fov_at_image_edge_deg": float(
                np.degrees(np.interp(0.5 * max(w, h), r_grid, th_grid))
            ),
            "half_fov_at_half_diagonal_deg": float(
                np.degrees(np.interp(halfdiag, r_grid, th_grid))
            ),
            "theta_p50_deg": float(np.degrees(np.median(th))),
            "theta_p90_deg": float(np.degrees(np.quantile(th, 0.90))),
            "theta_p99_deg": float(np.degrees(np.quantile(th, 0.99))),
            "theta_max_deg": float(np.degrees(th.max())),
            "frac_theta_ge_90": float(np.mean(th >= 0.5 * math.pi)),
            "frac_radii_beyond_invertible": float(np.mean(radii > r_grid[-1])),
        }
    return {
        "spec": {k: (str(v) if isinstance(v, Path) else v) for k, v in spec.items()},
        "width": w,
        "height": h,
        "n_images": len(data["names"]),
        "native_pinhole_vote": {
            k: (v if not isinstance(v, list) else len(v)) for k, v in native.items()
        },
        "equidistant_gt": equidistant_gt(spec, radii),
        "theta_stats": theta_stats,
        "rows": rows,
        "seconds": time.perf_counter() - t0,
    }


# ── Reporting ────────────────────────────────────────────────────────────────


def log_median(xs: list) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return float(math.exp(np.median(np.log(xs)))) if xs else None


def certify(c: dict, cell: str, gates: dict) -> tuple[bool, bool]:
    """(certified, model-informative) verdict of one scan under ``gates``.

    The epipolar floor is on the dimensionless essentialness residual; the
    rotation floor is on the ANGULAR cost (radians), which transfers across
    capture resolutions where the pixel cost does not.  The plane-subset arm
    has no angular cost of its own, so it falls back to its scaled cost.
    """
    if c["focal_px"] is None or c["edge"]:
        return False, False
    if cell == "rotation":
        cost = c.get("cost_min_rad")
        floor = gates["rotation_floor_rad"]
    else:
        cost, floor = c["cost_min"], gates["epipolar_floor"]
    if cost is None or cost > floor:
        return False, False
    if (c["sharpness"] if c["sharpness"] is not None else 1.0) > gates["shape"]:
        return False, False
    if cell == "epipolar":
        d = c.get("dir_disagreement")
        if d is None or d > gates["dir_band"]:
            return False, False
    return True, (c["coverage_p90"] or 0.0) >= gates["coverage"]


def _pool(scans: list, cell: str, gates: dict, key: str = "focal_px") -> dict:
    cert, informative = [], []
    for c in scans:
        ok, info = certify(c, cell, gates)
        if not ok:
            continue
        cert.append(c[key])
        if info:
            informative.append(c[key])
    return {
        "n_scanned": len(scans),
        "n_certified": len(cert),
        "n_informative": len(informative),
        "focal_certified": log_median(cert),
        "focal_informative": log_median(informative),
    }


def summarize(res: dict, gates: dict) -> dict:
    """Per-cell pooled focal and certified counts under a gate setting."""
    out = {}
    for cell in ("epipolar", "rotation"):
        for column in COLUMNS:
            scans = [r["cells"][column] for r in res["rows"] if r["cell"] == cell]
            out[f"{column}:{cell}"] = _pool(scans, cell, gates)
            if cell == "epipolar":
                # Sphere arm restricted to the FOV window, and the
                # plane-subset arm on the same window: the three-way
                # comparison the column decision rests on.
                # The FOV window's own edge rule applies to BOTH arms, else
                # the narrow window penalizes only the plane arm.
                fov = [
                    {
                        **c,
                        "focal_px": c["fov_focal_px"],
                        "cost_min": c["fov_cost_min"],
                        "sharpness": c["fov_sharpness"],
                        "edge": c["fov_edge"],
                    }
                    for c in scans
                ]
                out[f"{column}:epipolar/sphere-fov"] = _pool(fov, cell, gates)
                pl = [c["plane"] for c in scans]
                out[f"{column}:epipolar/plane-fov"] = _pool(pl, cell, gates)
                out[f"{column}:epipolar/plane-fov"]["seconds"] = sum(
                    c["plane"].get("seconds", 0.0) for c in scans
                )
                out[f"{column}:epipolar/plane-fov"]["frac_population"] = (
                    float(np.mean([c["plane"]["frac_population"] for c in scans]))
                    if scans
                    else 0.0
                )
    # Column-level roll-up: the pooled median over both cells' certified
    # votes, and the model-informative mass the arbitration reads.
    for column in COLUMNS:
        cert, info = [], []
        for r in res["rows"]:
            c = r["cells"][column]
            ok, inf = certify(c, r["cell"], gates)
            if ok:
                cert.append(c["focal_px"])
                if inf:
                    info.append(c["focal_px"])
        out[f"{column}:POOL"] = {
            "n_scanned": len(res["rows"]),
            "n_certified": len(cert),
            "n_informative": len(info),
            "focal_certified": log_median(cert),
            "focal_informative": log_median(info),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROTO / "out"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated dataset names")
    ap.add_argument(
        "--gate",
        choices=("pixel", "angular"),
        default="pixel",
        help="RANSAC consensus bound: a fixed angle (GATE_RAD) or the same "
        "angle derived per focal from TOL_PX through the map scale",
    )
    args = ap.parse_args()
    global GATE_MODE
    GATE_MODE = args.gate
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    want = set(args.only.split(",")) if args.only else None
    results = {}
    for spec in DATASETS:
        if want and spec["name"] not in want:
            continue
        if not Path(spec["matches"]).exists():
            print(f"[{spec['name']}] SKIP: {spec['matches']} missing")
            continue
        res = run_dataset(spec, args.quick)
        if spec["gt_focal_px"] is None:
            res["spec"]["gt_focal_px"] = res["native_pinhole_vote"].get("focal_px")
        results[spec["name"]] = res

    # Gates derived from the measured distributions (results.json carries the
    # curves each value came from):
    #   epipolar_floor 0.03 — essentialness at the minimum, correct column,
    #       p90 = 0.023 / 0.023 / 0.022 / 0.008 across the four captures.
    #   rotation_floor_rad 0.02 (1.15 deg) — correct-column angular cost p90
    #       = 0.63 / 0.59 deg, wrong column ~34 deg; the pixel equivalent
    #       doubles with resolution (1.5 -> 2.8 px) while the angle does not.
    #   shape 0.5 — the pinhole rotation cell's existing 2*cost < median rule.
    #   dir_band 0.05 — the epipolar family's existing agreement band.
    #   coverage 0.50 — inert on three captures; on the narrow-FOV pinhole
    #       control it drops the equidistant column's centre-hugging votes
    #       and widens the model margin 1.33x -> 2.29x.
    gates = {
        "epipolar_floor": 0.03,
        "rotation_floor_rad": 0.02,
        "shape": 0.5,
        "dir_band": 0.05,
        "coverage": 0.50,
    }
    for name, res in results.items():
        res["summary"] = summarize(res, gates)

    (outdir / "results.json").write_text(
        json.dumps(
            {"gates": gates, "gate_mode": GATE_MODE, "datasets": results},
            indent=1,
            default=str,
        )
    )

    lines = ["dataset\tcell\tn_scan\tn_cert\tn_info\tf_cert\tf_info\tgt\tgt_equi"]
    for name, res in results.items():
        gt = res["spec"]["gt_focal_px"]
        gte = (res["equidistant_gt"] or {}).get("f_equidistant_px")
        for key, s in res["summary"].items():
            lines.append(
                f"{name}\t{key}\t{s['n_scanned']}\t{s['n_certified']}\t"
                f"{s['n_informative']}\t{_f(s['focal_certified'])}\t"
                f"{_f(s['focal_informative'])}\t{_f(gt)}\t{_f(gte)}"
            )
    (outdir / "summary.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    prows = [
        "dataset\tpair\tn_corr\tdisp\tcell\tcolumn\tfocal\tcost_min\tcost_med\t"
        "sharp\trel_width\tcov_p90\tdir_dis\tn_inl\tedge\tcert\tinfo\t"
        "theta_p90\tfrac_th90\tf_fov\tf_plane\tplane_cost\tplane_sharp\tf_rad"
    ]
    for name, res in results.items():
        for r in res["rows"]:
            for column, c in r["cells"].items():
                ok, info = certify(c, r["cell"], gates)
                prows.append(
                    f"{name}\t{r['pair'][0]}-{r['pair'][1]}\t{r['n_corr']}\t"
                    f"{r['mean_disp_frac']:.4f}\t{r['cell']}\t{column}\t"
                    f"{_f(c['focal_px'])}\t{_f(c['cost_min'], 5)}\t"
                    f"{_f(c['cost_median'], 5)}\t{_f(c['sharpness'], 4)}\t"
                    f"{_f(c['rel_width'], 4)}\t{_f(c['coverage_p90'], 4)}\t"
                    f"{_f(c.get('dir_disagreement'), 4)}\t{c['n_inliers_at_min']}\t"
                    f"{int(bool(c['edge']))}\t{int(ok)}\t{int(info)}\t"
                    f"{_f(c.get('theta_p90_deg'), 1)}\t"
                    f"{_f(c.get('frac_theta_ge_90'), 3)}\t"
                    f"{_f(c.get('fov_focal_px'))}\t"
                    f"{_f((c.get('plane') or {}).get('focal_px'))}\t"
                    f"{_f((c.get('plane') or {}).get('cost_min'), 5)}\t"
                    f"{_f((c.get('plane') or {}).get('sharpness'), 4)}\t"
                    f"{_f(c.get('rad_focal_px'))}"
                )
    (outdir / "pairs.tsv").write_text("\n".join(prows) + "\n")
    print(f"\nwrote {outdir}/results.json, summary.tsv, pairs.tsv")


def _f(x, nd: int = 2) -> str:
    return "" if x is None else f"{x:.{nd}f}"


if __name__ == "__main__":
    main()
