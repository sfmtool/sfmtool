# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The seed's candidate-evaluation battery (specs/core/geometry/seed-candidate-evaluation.md).

When rung 1's hypothesis loop has committed its candidate set, every member is
measured HERE, once, before the product is written.  The channels attach to the
member's own record in the release manifest, so a later selection pass ranks,
refuses and trims on stored evidence rather than re-deriving it from artifacts.

The battery is:

* **Fit** -- inlier fractions at fixed pixel radii, median reprojection, and
  per-frame observation counts, over the member's own released geometry.
* **Hold-out self-resection** -- each frame is re-resected against structure
  re-triangulated from the member's OTHER frames only, with the depth agreement
  and the support conditioning that say whether the reading means anything.
* **Non-member resection** -- the capture's best-connected held-out images are
  resected against the member's stored structure, and the result is checked
  against a two-view essential estimate that never touched that structure.
* **Settling probe** -- a short staged bundle adjustment with the intrinsics
  frozen, read gauge-free: how far the member moves when it is allowed to
  settle, and whether its residual grows.
* **Focal-vote consistency** -- the released equivalent focal against the
  capture's structure-free vote, as a signed fraction.
* **Per-frame support and coherence**, and **peer corroboration** -- the
  channels the run already had, gathered into the same record.

THE BATTERY NEVER ALTERS THE RUN.  It reads the member's finished arrays,
writes nothing but manifest and evolution fields, and draws every random sample
from its own fixed seeds, so a released `.sfmr` is byte-identical whether the
battery ran or not.  `SFMTOOL_SEED_EVAL=0` turns the whole thing off.

Every channel reports whether it could be measured.  An unmeasurable reading
carries its reason and is never silently dropped, and a reading taken below a
conditioning floor is marked as a non-measurement rather than as a
disagreement.  Frames and images are named by their full relative path:
basenames collide across rig sensor directories.
"""

import itertools
import math
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from sfmtool._sfmtool.analysis import triangulate_batch
from sfmtool._sfmtool.geometry import (
    bundle_adjust,
    estimate_absolute_pose,
    estimate_essential_rays,
    refine_absolute_pose,
    reprojection_residuals,
)

# ── Constants ───────────────────────────────────────────────────────────────
#
# The resection settings are the bootstrap's own (a hold-out resection has to
# be the same measurement the pipeline would take, or it measures the settings
# instead of the member).  The floors are arithmetic ones: below them the
# statistic is not a weaker reading, it is a different quantity.

#: RANSAC P3P pixel bound, converted to an angular bound by the camera itself.
RANSAC_PX = 4.0
#: Trimmed pose refinement: rounds, retained fraction, final inlier bound.
TRIM_ROUNDS, KEEP_FRACTION, FINAL_INLIER_PX = 5, 0.6, 3.0
#: Every sampler in this module. Same inputs + same seed => same output.
SEED = 0

#: Correspondences below which a resection interpolates rather than fits.
MIN_RESECT_POINTS = 10
#: Below five paired depths a rank correlation is an accident, not a statistic.
MIN_STRUCT_PTS = 5
#: Supporting observations a point needs on OTHER frames to be re-triangulated.
MIN_SUPPORT_OBS = 2

#: Held-out images resected per member, and the correspondence floor to try one.
K_HELD_OUT, MIN_CORR = 5, 12
#: The two-view witness: pixel tolerance (carried to radians through the
#: member's own equivalent focal), match floor, subsample cap, minimal samples.
E_TOL_PX, E_MIN_CORR, E_MAX_CORR, E_SAMPLES = 3.0, 20, 3000, 1024
#: Correspondences from one member image before its inlier rate is reported.
MIN_MEM_CORR = 8
#: A pair subtending fewer than this many of the epipolar estimator's own
#: consensus bounds is CONDITIONING-LIMITED: its direction delta is reported
#: and is not read as a disagreement.  The unit is derived per camera from the
#: member's own map; only the multiplier is a choice, and it ships in the block.
E_PARALLAX_BAR = 2.0

#: A frame whose re-triangulated support subtends less than this median
#: angle is CONDITIONING-LIMITED: its pose deltas are reported and are not
#: gate-eligible, because a pose fit to depths that came out of a fraction of
#: a degree of baseline disagrees with nothing.  Set where human-ruled files
#: leave room -- the order of magnitude below it is arithmetic on noise -- and
#: shipped in the block so a selection pass can re-derive it on a wider
#: population than one capture offers.
TRI_ANGLE_BAR_DEG = 1.0
#: Support centres sampled per point for the triangulation-angle conditioning.
#: A wide track's extremes carry the answer, so the sample is even, not a head.
TRI_ANGLE_MAX_CENTRES = 16

#: The settling probe: the kernel's own staged schedule, opening permissive.
#: A single late-stage round would trim the re-triangulated points before the
#: solve ever reached them, which measures the trim and not the settling.
SETTLE_SCHEDULE = [(50.0, 5.0), (12.0, 2.0), (4.0, 1.0)]
#: Short: a settling probe asks where the member goes, not where it ends.
SETTLE_MAX_ITERS = 20
#: A refit whose median residual grows past this multiple has diverged.
DIVERGE_RATIO = 1.5
#: Baselines below this fraction of the member's scale carry no direction.
MIN_BASELINE_FRAC = 1e-4
#: Frame pairs the gauge-free pose comparison reads (evenly sampled above it).
MAX_POSE_PAIRS = 2000

#: Per-frame support quantities need this many finite points to mean anything.
MIN_SUPPORT_POINTS = 5

#: The warp channel's pixel->ray Jacobian step, in pixels.  A central
#: difference; the residual is flat across half and double this, so it is an
#: arithmetic choice rather than a knob.
WARP_JAC_STEP_PX = 0.25
#: A pair whose rays subtend less than this at the point is
#: CONDITIONING-LIMITED: with the rays that close to parallel the whole warp is
#: nearly pose-determined and its normal-free content constrains almost
#: nothing.  Reported, not gate-eligible, and the pair vergence distribution
#: ships beside it so a fleet pass can re-derive the floor.
WARP_VERGENCE_FLOOR_DEG = 1.0
#: A track seen in more views than this is thinned to this many partners per
#: view.  A compute cap on the pair count, never a threshold on the data.
WARP_MAX_VIEWS_ALL_PAIRS, WARP_PARTNERS_PER_VIEW = 16, 8
#: Below this many pairs the member's warp aggregates are an accident.
WARP_MIN_PAIRS = 30

#: Neighbours in the 3D k-NN neighbourhood the surface-variation channel reads.
SURF_K = 12
#: The stranger neighbourhood: candidates scanned for covisibility
#: disjointness, strangers kept, and the support floor below which the point
#: has no stranger surface to be measured against.
SURF_C_CAND, SURF_C_USE, SURF_C_MIN = 64, 12, 6
#: The range-vetted image-space neighbourhood: the same three numbers.
SURF_BV_CAND, SURF_BV_USE, SURF_BV_MIN = 32, 12, 6
#: The robust plane: redescending IRLS passes and the Tukey constant, the same
#: shape as the adjacency-surfel fit (specs/core/analysis/adjacency-surfel-normals.md).
SURF_IRLS_ITERS, SURF_TUKEY_C = 3, 4.685
#: Rows per batch in the neighbourhood distance products.
SURF_CHUNK = 20000

#: The rotation-only battery.  A far-field member has one rotation per frame,
#: every camera centre at the same point, and one DIRECTION per explained
#: cluster; every channel below is the rotation-only form of a finite one.
#:
#: Trimmed rotation fit: rounds and retained fraction, the shape the pipeline's
#: own pose refinement uses.
ROT_TRIM_ROUNDS, ROT_KEEP_FRACTION = 5, 0.6
#: Directions a hold-out rotation resection needs before it is a fit.
ROT_MIN_SUPPORT = 10
#: Other-frame observations a direction needs to be re-derived without the
#: frame being held out.
ROT_MIN_SUPPORT_OBS = 2
#: Frames the hold-out channel resects, evenly sampled past it.  A capture-wide
#: far layer poses hundreds of frames; this is a compute cap, not a threshold.
ROT_MAX_HOLDOUT_FRAMES = 64
#: Alternating rounds of the rotation-only settling probe (rotations from the
#: directions, then directions from the rotations).
ROT_SETTLE_ROUNDS = 5
#: Residual magnitude in pixels below which a residual VECTOR carries no
#: direction, as a quantile of the frame's own residuals.
ROT_RESIDUE_QUANTILE = 50
#: Observations a frame needs before its parallax residue is a reading.
ROT_MIN_RESIDUE_OBS = 12
#: The mean |cos| between a residual vector and a radial direction under
#: isotropic noise: the parallax-residue channel's own null level.
ROT_RADIAL_NULL = 2.0 / math.pi
#: The cycle channel: points two frames must share before their relative
#: rotation is a measurement, and the largest fundamental cycle basis read.
#: Both are compute caps on a graph that can hold tens of thousands of cycles,
#: never thresholds on the readings.
ROT_CYCLE_MIN_SHARED, ROT_CYCLE_MAX_CYCLES = 10, 512

#: The exact photometric witness.  Pairs sampled per member, points per
#: pair, and the sampling grid across the patch: compute caps
#: on an I/O-bound reading, not thresholds.  The window's EXTENT comes from
#: each observation's own affine shape, floored and capped only where a shape
#: degenerates to a sub-pixel or a whole-image window.
ROT_PHOTO_MAX_PAIRS, ROT_PHOTO_MAX_POINTS = 20, 48
ROT_PHOTO_MIN_SHARED, ROT_PHOTO_GRID = 10, 9
ROT_PHOTO_MIN_RADIUS_PX, ROT_PHOTO_MAX_RADIUS_PX = 2.0, 64.0
#: Samples a window needs before its correlation is a correlation, and how
#: many images are held decoded at once.
ROT_PHOTO_MIN_SAMPLES, ROT_PHOTO_CACHE = 25, 8

#: Parallax-bearing points listed by name in the record, loudest first.  The
#: census counts them all; this caps what the record carries.
ROT_PARALLAX_MAX_LISTED = 512
#: The share of a member's points that must be parallax-bearing before the
#: member's whole field is called that rather than part of it.
ROT_PARALLAX_MAJORITY = 0.5
#: Pairs the warp channels read before the set is evenly thinned.  A far layer
#: over hundreds of frames offers millions; the residual distribution does not
#: need them.
WARP_MAX_PAIRS = 100000


def eval_on():
    """Whether the battery runs (``SFMTOOL_SEED_EVAL=0`` turns it off)."""
    return (os.environ.get("SFMTOOL_SEED_EVAL", "1") or "1").strip() != "0"


# ── Small shared numerics ───────────────────────────────────────────────────


def _finite(vals):
    """The finite floats of ``vals``, dropping ``None`` and non-finite."""
    out = []
    for v in vals:
        if v is None:
            continue
        f = float(v)
        if math.isfinite(f):
            out.append(f)
    return out


def _q(vals, p):
    """``percentile(vals, p)`` over the finite readings, or None when empty."""
    v = _finite(vals)
    return float(np.percentile(v, p)) if v else None


def _extreme(rows, key, worst_is_max=True):
    """``(value, name)`` of the row with the extreme ``key``, or ``(None, None)``.

    Ties go to the first row, which is capture order."""
    have = [r for r in rows if r.get(key) is not None and math.isfinite(r[key])]
    if not have:
        return None, None
    pick = (
        max(have, key=lambda r: r[key])
        if worst_is_max
        else min(have, key=lambda r: r[key])
    )
    return float(pick[key]), pick.get("name")


def _rank(a):
    """Ordinal ranks of ``a`` (stable, so equal values keep capture order).

    Depths are continuous, so this is the tie-averaged rank wherever the
    distinction exists."""
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
    return r


def _spearman(x, y):
    """Spearman rank correlation, or None when either side is degenerate."""
    if len(x) < 3 or len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        return None
    rx, ry = _rank(np.asarray(x)), _rank(np.asarray(y))
    sx, sy = rx.std(), ry.std()
    if not (sx > 0 and sy > 0):
        return None
    rho = float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))
    return rho if math.isfinite(rho) else None


def _quat(rvec):
    """Rotation vectors as WXYZ quaternions, the kernels' pose convention."""
    return np.ascontiguousarray(
        Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64)).as_quat()[
            :, [3, 0, 1, 2]
        ]
    )


def _centres(rot, tvec):
    """Camera centres ``-Rᵀ·t`` for world-to-camera ``(R, t)``."""
    return -np.einsum("nji,nj->ni", rot, np.asarray(tvec, dtype=np.float64))


def _residual_norms(cam, rvec, tvec, pts, uv, obs_i, obs_c):
    """Per-observation reprojection norms in pixels (``inf`` where invalid)."""
    res = np.asarray(
        reprojection_residuals(
            cam,
            _quat(rvec),
            np.ascontiguousarray(tvec, dtype=np.float64),
            np.ascontiguousarray(pts, dtype=np.float64),
            np.ascontiguousarray(uv, dtype=np.float64),
            np.ascontiguousarray(obs_i, dtype=np.uint32),
            np.ascontiguousarray(obs_c, dtype=np.uint32),
            invalid_residual=float("inf"),
        )
    )
    return np.linalg.norm(res, axis=1)


def _world_rays(cam, rot, obs_i, uv):
    """Unit WORLD rays of ``uv`` seen by images ``obs_i`` at poses ``rot``.

    The pixels unproject through the member's own camera model, so a fisheye
    member is read on its own map rather than on a pinhole stand-in."""
    loc = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(uv, np.float64)))
    n = np.linalg.norm(loc, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        loc = loc / np.where(n > 0, n, 1.0)
    return np.einsum("nji,nj->ni", rot[obs_i], loc)


def _csr(keys, n):
    """``(order, bounds)`` grouping ``keys`` (values in ``[0, n)``) CSR-style."""
    order = np.argsort(keys, kind="stable")
    bounds = np.searchsorted(keys[order], np.arange(n + 1))
    return order, bounds


def _pose_pairs(n):
    """Index pairs over ``n`` items, evenly thinned past ``MAX_POSE_PAIRS``.

    Thinned evenly rather than truncated: the pair that moved most has to be
    free to land anywhere in the member."""
    pairs = np.array(list(itertools.combinations(range(n), 2)), dtype=np.int64)
    if len(pairs) > MAX_POSE_PAIRS:
        take = np.unique(
            np.linspace(0, len(pairs) - 1, MAX_POSE_PAIRS).round().astype(np.int64)
        )
        pairs = pairs[take]
    return pairs


def _relative_pose_deltas(rot_a, cen_a, rot_b, cen_b, pairs, floor=0.0):
    """Gauge-free per-pair ``(rotation deg, translation-direction deg)``.

    Both readings are between-frame quantities, so a global rotation, a global
    translation or a global scale applied to either side cancels: nothing is
    aligned and no absolute pose is ever compared.  A pair whose baseline is
    shorter than ``floor`` on either side has no direction to compare and reads
    NaN rather than an angle between two rounding errors."""
    i, j = pairs[:, 0], pairs[:, 1]
    rel_a = np.einsum("nij,nkj->nik", rot_a[j], rot_a[i])
    rel_b = np.einsum("nij,nkj->nik", rot_b[j], rot_b[i])
    rot_deg = np.degrees(
        Rotation.from_matrix(np.einsum("nji,njk->nik", rel_a, rel_b)).magnitude()
    )
    # The pair's baseline stated in frame i's own camera frame, so the
    # comparison is free of the world frame, then read as a DIRECTION (the two
    # sides may differ by a global scale).
    d_a = np.einsum("nij,nj->ni", rot_a[i], cen_a[j] - cen_a[i])
    d_b = np.einsum("nij,nj->ni", rot_b[i], cen_b[j] - cen_b[i])
    na, nb = np.linalg.norm(d_a, axis=1), np.linalg.norm(d_b, axis=1)
    live = (na > floor) & (nb > floor)
    tdir = np.full(len(pairs), np.nan)
    if live.any():
        cos = np.sum(d_a[live] * d_b[live], axis=1) / (na[live] * nb[live])
        tdir[live] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return rot_deg, tdir


def _agg(vals, prefix, out, worst=True):
    """``med``/``p90``/``worst`` of ``vals`` under ``prefix`` into ``out``."""
    v = _finite(vals)
    out[f"{prefix}_med"] = _q(v, 50)
    out[f"{prefix}_p90"] = _q(v, 90)
    out[f"{prefix}_worst"] = (max(v) if worst else min(v)) if v else None
    return out


# ── The member view ─────────────────────────────────────────────────────────


class Member:
    """One committed candidate's finished arrays, in one image and cluster space.

    ``obs_*`` are the member's own working set (the admission its solve ran on),
    ``pts`` its released structure by that set's cluster id, and the poses are
    lifted into the loader's image frame, which every member of the capture
    shares.  Nothing here is copied out of an artifact: the battery reads what
    the run already holds."""

    def __init__(
        self,
        idx,
        model,
        names,
        camera,
        f_eq,
        rvec,
        tvec,
        posed,
        pts,
        obs,
        shapes=None,
        keep=None,
        dropped=None,
    ):
        self.idx = int(idx)
        self.model = model
        self.names = [str(n).replace("\\", "/") for n in names]
        self.camera = camera
        self.f_eq = None if f_eq is None else float(f_eq)
        self.rvec = np.ascontiguousarray(rvec, dtype=np.float64)
        self.tvec = np.ascontiguousarray(tvec, dtype=np.float64)
        self.rot = Rotation.from_rotvec(self.rvec).as_matrix()
        self.centre = _centres(self.rot, self.tvec)
        self.posed = np.asarray(posed, dtype=bool)
        # FRAMES THIS MEMBER ITSELF REJECTED.  A member restricted to a core
        # is not posed on them and they are not outsiders either: an image the
        # member cut is a rejected part of the member, so a held-out channel
        # never asks the core to explain it.  Empty for a member as committed.
        self.dropped = (
            np.zeros(len(self.posed), bool)
            if dropped is None
            else np.asarray(dropped, dtype=bool)
        )
        self.pts = None if pts is None else np.asarray(pts, dtype=np.float64)
        self.n_img = len(self.names)
        obs_c, obs_i, obs_uv, obs_f = obs
        self.obs_c = np.asarray(obs_c, dtype=np.int64)
        self.obs_i = np.asarray(obs_i, dtype=np.int64)
        self.obs_uv = np.asarray(obs_uv, dtype=np.float64)
        self.obs_f = None if obs_f is None else np.asarray(obs_f, dtype=np.int64)
        # The member's stored affine SHAPES, one 2x2 per observation row: the
        # map from the detector's canonical unit frame onto that image's
        # pixels.  Supplied by the loader alongside the pixels, so the shape
        # and the keypoint a residual is anchored at come from one row.
        self.obs_shape = (
            None if shapes is None else np.asarray(shapes, dtype=np.float64)
        )
        self.n_cl = 0 if self.pts is None else len(self.pts)
        # THE MEMBER'S OWN OBSERVATIONS: the rows on its posed frames whose
        # cluster it actually placed.  Everything the battery measures is a
        # statement about these rows.
        self.finite = (
            np.zeros(0, bool) if self.pts is None else np.isfinite(self.pts).all(axis=1)
        )
        live = self.posed[self.obs_i]
        if self.n_cl:
            live &= self.finite[self.obs_c]
        # THE FULL ADMISSION on the member's posed frames, kept beside the
        # member's own rows.  A rotation-only member's membership is the model's
        # own inlier set, and the observations it did NOT explain are the
        # evidence the parallax-residue channel reads.
        self.rows_all = np.nonzero(live)[0]
        if keep is not None:
            live &= np.asarray(keep, dtype=bool)
        self.rows = np.nonzero(live)[0]
        self.frames = np.nonzero(self.posed)[0]
        # Per-frame slices of the member rows, in image order.
        ordr, bounds = _csr(self.obs_i[self.rows], self.n_img)
        self._rows_by_img = self.rows[ordr]
        self._img_bounds = bounds

    def frame_rows(self, j):
        """The member rows of image ``j``."""
        return self._rows_by_img[self._img_bounds[j] : self._img_bounds[j + 1]]

    def scene_scale(self):
        """The member's median camera-to-structure distance.

        The denominator of every translation fraction: per posed frame the
        median RANGE to its own finite support, then the median of those over
        frames.  Range, never a coordinate axis, so the quantity means the same
        thing under a fisheye member as under a pinhole one."""
        meds = []
        for j in self.frames:
            rows = self.frame_rows(j)
            if not len(rows):
                continue
            cl = np.unique(self.obs_c[rows])
            if len(cl) < MIN_SUPPORT_POINTS:
                continue
            d = np.linalg.norm(self.pts[cl] - self.centre[j], axis=1)
            d = d[np.isfinite(d) & (d > 0)]
            if len(d) >= MIN_SUPPORT_POINTS:
                meds.append(float(np.median(d)))
        return float(np.median(meds)) if meds else None

    def keep_mask(self):
        """The membership mask over observation rows, as it was handed in.

        A rotation-only member's membership is its model's own inlier set, so
        it is not recoverable from the poses alone; it is recovered here from
        the rows the member kept, which is the same set."""
        mask = np.zeros(len(self.obs_c), bool)
        mask[self.rows] = True
        return mask

    def restricted(self, keep_names, min_obs=2):
        """The member over ``keep_names`` alone, as a member in its own right.

        The frames outside the set stop being posed, every point left with
        fewer than ``min_obs`` observations on the frames that remain stops
        being finite, and a frame the point cull emptied stops being posed
        too: the same three steps, in the same order, that produce the core a
        trim writes.  Nothing else moves -- the poses, the camera, the
        keypoints and the shapes are the member's own -- so what comes back is
        the surviving geometry itself and not an aggregate of readings taken
        before the cut."""
        want = {str(n).replace("\\", "/") for n in keep_names}
        posed = self.posed & np.asarray([n in want for n in self.names], bool)
        pts = None if self.pts is None else self.pts.copy()
        live = posed[self.obs_i]
        if pts is not None:
            counts = np.bincount(self.obs_c[live], minlength=self.n_cl)
            pts[counts < int(min_obs)] = np.nan
            live &= np.isfinite(pts[self.obs_c]).all(axis=1)
        seen = np.zeros(self.n_img, bool)
        if live.any():
            seen[np.unique(self.obs_i[live])] = True
        posed &= seen
        return Member(
            self.idx,
            self.model,
            self.names,
            self.camera,
            self.f_eq,
            self.rvec,
            self.tvec,
            posed,
            pts,
            (self.obs_c, self.obs_i, self.obs_uv, self.obs_f),
            shapes=self.obs_shape,
            keep=self.keep_mask(),
            dropped=self.dropped | (self.posed & ~posed),
        )


def member_arrays(m):
    """Everything a :class:`Member` was built from, as plain arrays.

    The member's own finished arrays, nothing copied out of an artifact and
    nothing derived: enough to state the member again, and so enough to state
    any subset of it.  ``keep`` is the membership mask, which a rotation-only
    member needs and a finite one is unchanged by."""
    return {
        "idx": int(m.idx),
        "model": m.model,
        "names": list(m.names),
        "camera": m.camera.to_dict(),
        "f_eq": m.f_eq,
        "rvec": m.rvec,
        "tvec": m.tvec,
        "posed": m.posed,
        "pts": m.pts,
        "obs_c": m.obs_c,
        "obs_i": m.obs_i,
        "obs_uv": m.obs_uv,
        "obs_f": m.obs_f,
        "obs_shape": m.obs_shape,
        "keep": m.keep_mask(),
    }


def member_from_arrays(d):
    """A :class:`Member` from what :func:`member_arrays` holds."""
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    return Member(
        d["idx"],
        d["model"],
        d["names"],
        CameraIntrinsics.from_dict(d["camera"]),
        d["f_eq"],
        d["rvec"],
        d["tvec"],
        d["posed"],
        d["pts"],
        (d["obs_c"], d["obs_i"], d["obs_uv"], d.get("obs_f")),
        shapes=d.get("obs_shape"),
        keep=d.get("keep"),
    )


# ── Channel: fit ────────────────────────────────────────────────────────────


def fit_channels(m):
    """Inlier fractions at the two fixed radii, residual quantiles, and the
    census of what the member actually holds."""
    # THE MEMBER'S WHOLE ADMISSION on its posed frames, which is the same set
    # as its own rows for a finite member and a wider one for a rotation-only
    # member, whose membership is its model's inlier set.  Measuring the fit
    # over the membership alone would report the trim bar, not the fit.
    rows = m.rows_all
    out = {
        "n_posed": int(m.posed.sum()),
        "n_points": 0 if m.pts is None else int(m.finite.sum()),
        "n_obs": int(len(rows)),
        "median_px": None,
        "p90_px": None,
        "inlier_2px": None,
        "inlier_4px": None,
        "measurable": False,
    }
    if m.pts is None or not len(rows):
        out["unmeasurable_reason"] = "no_structure" if m.pts is None else "no_obs"
        return out
    r = _residual_norms(
        m.camera,
        m.rvec,
        m.tvec,
        m.pts,
        m.obs_uv[rows],
        m.obs_i[rows],
        m.obs_c[rows],
    )
    fin = r[np.isfinite(r)]
    if not len(fin):
        out["unmeasurable_reason"] = "no_finite_residual"
        return out
    out.update(
        median_px=float(np.median(fin)),
        p90_px=float(np.percentile(fin, 90)),
        inlier_2px=float((fin < 2.0).mean()),
        inlier_4px=float((fin < 4.0).mean()),
        measurable=True,
    )
    return out


# ── Channel: per-frame support and coherence ────────────────────────────────


def support_channels(m):
    """Per-frame support, depth-scale coherence and observation floors.

    A frame whose whole support sits far away has an unobservable translation:
    its orientation can be right while its position is anything, and no
    member-level residual sees that.  The near-support ratio is that reading,
    made dimensionless by the member's own scale so it compares across a fleet
    whose captures differ in scale by orders of magnitude."""
    scale = m.scene_scale()
    frames = []
    out = {"scene_scale": scale, "frames": frames}
    if m.pts is None:
        out["unmeasurable_reason"] = "no_structure"
        return out
    for j in m.frames:
        rows = m.frame_rows(j)
        rec = {"name": m.names[j], "n_obs": int(len(rows))}
        all_rows = np.nonzero(m.obs_i == j)[0]
        rec["n_obs_admitted"] = int(len(all_rows))
        rec["finite_frac"] = float(len(rows) / len(all_rows)) if len(all_rows) else None
        cl = np.unique(m.obs_c[rows]) if len(rows) else np.zeros(0, np.int64)
        rec["n_points"] = int(len(cl))
        if len(cl) >= MIN_SUPPORT_POINTS:
            d = np.linalg.norm(m.pts[cl] - m.centre[j], axis=1)
            d = d[np.isfinite(d) & (d > 0)]
            if len(d) >= MIN_SUPPORT_POINTS:
                rec["near_p10"] = float(np.percentile(d, 10))
                rec["dist_med"] = float(np.median(d))
                if scale and scale > 0:
                    rec["near_ratio"] = rec["near_p10"] / scale
                    rec["depth_ratio"] = rec["dist_med"] / scale
                    rec["depth_log_dev"] = abs(math.log10(rec["dist_med"] / scale))
        if len(rows):
            r = _residual_norms(
                m.camera,
                m.rvec,
                m.tvec,
                m.pts,
                m.obs_uv[rows],
                m.obs_i[rows],
                m.obs_c[rows],
            )
            fin = r[np.isfinite(r)]
            if len(fin):
                rec["reproj_med"] = float(np.median(fin))
        frames.append(rec)
    if not frames:
        out["unmeasurable_reason"] = "no_posed_frames"
        return out
    obs = [f["n_obs"] for f in frames]
    out["n_frames"] = len(frames)
    out["obs_min"] = int(min(obs))
    out["obs_p10"] = _q(obs, 10)
    out["obs_med"] = _q(obs, 50)
    out["finite_frac_min"] = _q([f.get("finite_frac") for f in frames], 0)
    ratios = [f.get("near_ratio") for f in frames]
    out["near_ratio_med"] = _q(ratios, 50)
    out["near_ratio_p90"] = _q(ratios, 90)
    out["near_ratio_worst"], out["near_ratio_worst_frame"] = _extreme(
        frames, "near_ratio"
    )
    dev = [f.get("depth_log_dev") for f in frames]
    out["depth_log_dev_med"] = _q(dev, 50)
    out["depth_log_dev_worst"], out["depth_log_dev_worst_frame"] = _extreme(
        frames, "depth_log_dev"
    )
    out["reproj_med"] = _q([f.get("reproj_med") for f in frames], 50)
    out["reproj_worst"], out["reproj_worst_frame"] = _extreme(frames, "reproj_med")
    return out


# ── Channel: hold-out self-resection ────────────────────────────────────────


def _tri_angle_max(centres, offsets, points):
    """Each point's MAXIMUM pairwise angle subtended at it by its supporting
    camera centres, in degrees.

    The widest baseline the point actually had.  Measured at the
    RE-TRIANGULATED position, because that is the depth the resection was
    handed; a point whose whole support subtends a fraction of a degree carries
    a depth that is arithmetic on noise, and a pose fit to it is not a
    disagreement with anything."""
    out = np.full(len(points), np.nan)
    for k in range(len(points)):
        cc = centres[offsets[k] : offsets[k + 1]]
        if len(cc) > TRI_ANGLE_MAX_CENTRES:
            take = np.unique(
                np.linspace(0, len(cc) - 1, TRI_ANGLE_MAX_CENTRES)
                .round()
                .astype(np.int64)
            )
            cc = cc[take]
        d = cc - points[k]
        n = np.linalg.norm(d, axis=1)
        keep = np.isfinite(n) & (n > 0)
        if int(keep.sum()) < 2:
            continue
        u = d[keep] / n[keep, None]
        out[k] = np.degrees(np.arccos(np.clip(np.min(u @ u.T), -1.0, 1.0)))
    return out


def _loo_frame(m, j, scale):
    """One frame's hold-out self-resection reading."""
    rec = {"name": m.names[j], "status": "ok"}
    rows_j = m.frame_rows(j)
    rec["n_obs"] = int(len(rows_j))
    if not len(rows_j):
        rec["status"] = "no_observations"
        return rec
    # ONE OBSERVATION PER POINT, the frame's first, which is the pixel the
    # resection is later handed.
    uniq, first = np.unique(m.obs_c[rows_j], return_index=True)
    rec["n_finite_pts"] = int(len(uniq))
    # The support: every OTHER posed frame's observation of those points.  The
    # stored positions are contaminated for this purpose -- `j` helped
    # triangulate them -- so nothing about them is reused below.
    sel = m.rows[(m.obs_i[m.rows] != j) & np.isin(m.obs_c[m.rows], uniq)]
    if not len(sel):
        rec["status"] = "no_other_support"
        return rec
    op, oi, ou = m.obs_c[sel], m.obs_i[sel], m.obs_uv[sel]
    order = np.argsort(op, kind="stable")
    op, oi, ou = op[order], oi[order], ou[order]
    upts, counts = np.unique(op, return_counts=True)
    good = counts >= MIN_SUPPORT_OBS
    rec["n_candidate_points"] = int(good.sum())
    rec["cross_frac"] = float(good.sum() / max(len(uniq), 1))
    if int(good.sum()) < MIN_RESECT_POINTS:
        rec["status"] = "few_support_points"
        return rec
    keep = np.repeat(good, counts)
    op, oi, ou = op[keep], oi[keep], ou[keep]
    upts, counts = upts[good], counts[good]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    dirs = _world_rays(m.camera, m.rot, oi, ou)
    if not np.isfinite(dirs).all():
        rec["status"] = "bad_rays"
        return rec
    cen = m.centre[oi]
    tri = triangulate_batch(
        np.ascontiguousarray(dirs), np.ascontiguousarray(cen), offsets
    )
    xr = np.asarray(tri["points"])
    ok = np.isfinite(xr).all(axis=1) & np.asarray(tri["in_front_of_all_cameras"])
    rec["n_retri"] = int(ok.sum())
    if int(ok.sum()) < MIN_STRUCT_PTS:
        rec["status"] = "few_retriangulated"
        return rec
    xr_ok, pts_ok = np.ascontiguousarray(xr[ok]), upts[ok]
    # SUPPORT CONDITIONING first, and the depth agreement with it: both survive
    # a resection that never converges, and both are what says whether a pose
    # delta from this frame is evidence.
    off_ok = np.concatenate([[0], np.cumsum(counts[ok])]).astype(np.int64)
    cen_ok = cen[np.repeat(ok, counts)]
    ang = _tri_angle_max(cen_ok, off_ok, xr_ok)
    fin_ang = ang[np.isfinite(ang)]
    if len(fin_ang):
        rec["tri_angle_med"] = float(np.median(fin_ang))
        rec["tri_angle_p90"] = float(np.percentile(fin_ang, 90))
        rec["tri_angle_max"] = float(np.max(fin_ang))
    _loo_structure(m, j, rec, pts_ok, xr_ok, scale)
    if int(ok.sum()) < MIN_RESECT_POINTS:
        rec["status"] = "few_retriangulated"
        return rec
    pos = {int(p): k for k, p in enumerate(uniq)}
    sel_first = first[[pos[int(p)] for p in pts_ok]]
    uv = np.ascontiguousarray(m.obs_uv[rows_j[sel_first]])
    rec["n_points"] = int(len(uv))
    ans = estimate_absolute_pose(
        uv, xr_ok, camera=m.camera, max_error_px=RANSAC_PX, seed=SEED
    )
    if ans is None:
        rec["status"] = "no_consensus"
        return rec
    rec["ransac_inlier_frac"] = float(np.asarray(ans["inliers"]).mean())
    ref = refine_absolute_pose(
        m.camera,
        uv,
        xr_ok,
        np.asarray(ans["quaternion_wxyz"], dtype=np.float64),
        np.asarray(ans["translation"], dtype=np.float64),
        TRIM_ROUNDS,
        KEEP_FRACTION,
        FINAL_INLIER_PX,
    )
    rec["inlier_frac"] = float(ref["inlier_fraction"])
    qn = np.asarray(ref["quaternion_wxyz"], dtype=np.float64)
    rn = Rotation.from_quat(qn[[1, 2, 3, 0]]).as_matrix()
    cn = -rn.T @ np.asarray(ref["translation"], dtype=np.float64)
    rec["rot_delta_deg"] = float(
        np.degrees(Rotation.from_matrix(rn @ m.rot[j].T).magnitude())
    )
    d = float(np.linalg.norm(cn - m.centre[j]))
    rec["trans_delta"] = d
    rec["trans_delta_frac"] = (d / scale) if (scale and scale > 0) else None
    return rec


def _loo_structure(m, j, rec, pts_ok, xr_ok, scale):
    """The re-triangulation depth agreement at frame ``j``.

    Depth is the RAY RANGE from the frame's own centre under every camera
    model, so the correlation compares the same quantity a fisheye member and a
    pinhole member both have."""
    if len(pts_ok) < MIN_STRUCT_PTS:
        return
    xc = m.pts[pts_ok]
    dc = np.linalg.norm(xc - m.centre[j], axis=1)
    dr = np.linalg.norm(xr_ok - m.centre[j], axis=1)
    ok = np.isfinite(dc) & np.isfinite(dr) & (dc > 0) & (dr > 0)
    if int(ok.sum()) < MIN_STRUCT_PTS:
        return
    dc, dr = dc[ok], dr[ok]
    rec["retri_n"] = int(ok.sum())
    # Out of the member's own field, reported and never filtered: the model's
    # own map decides, by refusing to project.
    cam_pts = np.einsum("ij,nj->ni", m.rot[j], xr_ok[ok]) + m.tvec[j]
    proj = np.asarray(m.camera.ray_to_pixel_batch(np.ascontiguousarray(cam_pts)))
    rec["retri_n_outfield"] = int((~np.isfinite(proj).all(axis=1)).sum())
    rho = _spearman(dc, dr)
    if rho is not None:
        rec["retri_rho"] = rho
    ratio = dr / dc
    ld = np.abs(np.log(ratio))
    rec["retri_ratio_med"] = float(np.median(ratio))
    rec["retri_logdev_med"] = float(np.median(ld))
    rec["retri_logdev_worst"] = float(np.max(ld))
    if scale and scale > 0:
        rec["retri_disp_med"] = float(
            np.median(np.linalg.norm(xr_ok[ok] - xc[ok], axis=1)) / scale
        )


def self_resection_channels(m, scale):
    """The member's hold-out self-resection: per frame, then the aggregates.

    The pose aggregates are gated by an inlier-fraction FLOOR taken as a
    quantile of the capture's own readings: a resection that fits nothing
    returns a pose that is noise, and reading that noise as a pose
    disagreement is what makes a sound member look welded.  A frame below the
    floor, or below the conditioning bar, is a non-measurement -- its deltas
    are reported, and they are not gate-eligible."""
    frames = []
    out = {"frames": frames}
    if m.pts is None:
        out["unmeasurable_reason"] = "no_structure"
        out["n_measured"] = 0
        return out
    for j in m.frames:
        frames.append(_loo_frame(m, j, scale))
    ok = [f for f in frames if f["status"] == "ok"]
    sf = [f for f in frames if f.get("retri_n")]
    out["n_frames_tried"] = len(frames)
    out["n_measured"] = len(ok)
    out["unmeasurable_frac"] = 1.0 - len(ok) / len(frames) if frames else None
    out["struct_n"] = len(sf)
    out["struct_frac"] = (len(sf) / len(frames)) if frames else None
    out["cross_frac_min"] = _q([f.get("cross_frac") for f in frames], 0)
    out["cross_frac_med"] = _q([f.get("cross_frac") for f in frames], 50)
    if ok:
        rots = [f.get("rot_delta_deg") for f in ok]
        trs = [f.get("trans_delta_frac") for f in ok]
        _agg(rots, "rot", out)
        _agg(trs, "trans", out)
        out["rot_worst"], out["rot_worst_frame"] = _extreme(ok, "rot_delta_deg")
        out["trans_worst"], out["trans_worst_frame"] = _extreme(ok, "trans_delta_frac")
        out["inlier_min"] = _q([f.get("inlier_frac") for f in ok], 0)
        out["inlier_med"] = _q([f.get("inlier_frac") for f in ok], 50)
        out["points_med"] = _q([f.get("n_points") for f in ok], 50)
    if sf:
        rho = [f.get("retri_rho") for f in sf]
        out["retri_rho_med"] = _q(rho, 50)
        out["retri_rho_p10"] = _q(rho, 10)
        out["retri_rho_min"], out["retri_rho_min_frame"] = _extreme(
            sf, "retri_rho", worst_is_max=False
        )
        ld = [f.get("retri_logdev_med") for f in sf]
        out["retri_logdev_med"] = _q(ld, 50)
        out["retri_logdev_p90"] = _q(ld, 90)
        out["retri_logdev_worst"], out["retri_logdev_worst_frame"] = _extreme(
            sf, "retri_logdev_med"
        )
        out["retri_logdev_tail_med"] = _q([f.get("retri_logdev_worst") for f in sf], 50)
        out["retri_logdev_tail_worst"] = _q(
            [f.get("retri_logdev_worst") for f in sf], 100
        )
        out["retri_disp_med"] = _q([f.get("retri_disp_med") for f in sf], 50)
        out["retri_disp_worst"] = _q([f.get("retri_disp_med") for f in sf], 100)
        out["tri_angle_med"] = _q([f.get("tri_angle_med") for f in sf], 50)
        out["tri_angle_min"] = _q([f.get("tri_angle_med") for f in sf], 0)
    return out


def apply_inlier_floor(blocks, floor=None):
    """Gate the self-resection pose aggregates on the CAPTURE's own floor.

    The floor is the 10th percentile of every per-frame resection inlier
    fraction this capture produced, pooled over all its members -- a quantile
    of the relevant population, not a constant.  The per-frame readings ship
    beside it, so a fleet pass can re-derive the floor on a wider population
    without re-running anything.

    A caller measuring a SUBSET of one member passes the capture's own floor
    in: conditioning is respected, never re-derived, and one member's frames
    are not the population the floor is a quantile of."""
    pool = [
        f["inlier_frac"]
        for b in blocks
        for f in b.get("frames", [])
        if f.get("inlier_frac") is not None
    ]
    if floor is None:
        floor = float(np.percentile(pool, 10)) if pool else None
    for b in blocks:
        b["inlier_floor"] = floor
        b["inlier_floor_source"] = "capture p10 of per-frame resection inlier"
        b["tri_angle_bar_deg"] = TRI_ANGLE_BAR_DEG
        frames = b.get("frames", [])
        n = len(frames)
        gated = []
        for f in frames:
            elig = (
                f["status"] == "ok"
                and f.get("inlier_frac") is not None
                and (floor is None or f["inlier_frac"] >= floor)
            )
            f["gate_eligible"] = bool(elig)
            if not elig:
                f["gate_blocked_by"] = (
                    "status" if f["status"] != "ok" else "inlier_floor"
                )
            elif (
                f.get("tri_angle_med") is not None
                and f["tri_angle_med"] < TRI_ANGLE_BAR_DEG
            ):
                # Conditioning-limited: reported, not gate-eligible.
                f["gate_eligible"] = False
                f["gate_blocked_by"] = "conditioning"
            if f["gate_eligible"]:
                gated.append(f)
        b["gated_n"] = len(gated)
        b["gated_lost_frac"] = (1.0 - len(gated) / n) if n else None
        b["rot_worst_gated"], b["rot_worst_gated_frame"] = _extreme(
            gated, "rot_delta_deg"
        )
        b["trans_worst_gated"], b["trans_worst_gated_frame"] = _extreme(
            gated, "trans_delta_frac"
        )
    return floor


# ── Channel: non-member resection ───────────────────────────────────────────


class PairGraph:
    """The capture's match graph, indexed by image, for the two-view witness.

    A cluster IS the match-transitive group, so two images' shared clusters
    are their raw 2D-2D matches -- evidence no member's structure ever
    touched.  Built once per capture and grouped CSR-style, because the
    witness asks for a handful of images out of a table with millions of rows.
    """

    def __init__(self, obs_c, obs_i, obs_uv, n_img):
        obs_i = np.asarray(obs_i, dtype=np.int64)
        order, self._bounds = _csr(obs_i, n_img)
        self._cl = np.asarray(obs_c, dtype=np.int64)[order]
        self._uv = np.asarray(obs_uv, dtype=np.float64)[order]

    def rows_of(self, j):
        """``(cluster ids, pixels)`` of image ``j``, cluster-sorted."""
        sl = slice(self._bounds[j], self._bounds[j + 1])
        cl = self._cl[sl]
        o = np.argsort(cl, kind="stable")
        return cl[o], self._uv[sl][o]


def _decompose_essential(e, x1, x2):
    """``(R, t, front_fraction)`` for ``x2 ~ R·x1 + t`` from a ray-space E.

    Cheirality is a ray-space midpoint solve, not a ``z > 0`` test, so a
    fisheye ray behind the pinhole half-space participates like any other."""
    u, _, vt = np.linalg.svd(np.asarray(e, dtype=np.float64))
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vt) < 0:
        vt[-1, :] *= -1
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    best = (None, None, -1)
    for rot in (u @ w @ vt, u @ w.T @ vt):
        r1 = x1 @ rot.T
        for s in (1.0, -1.0):
            t = s * u[:, 2]
            a11 = np.einsum("ij,ij->i", r1, r1)
            a12 = -np.einsum("ij,ij->i", r1, x2)
            a22 = np.einsum("ij,ij->i", x2, x2)
            b1, b2 = -(r1 @ t), x2 @ t
            det = a11 * a22 - a12 * a12
            ok = np.abs(det) > 1e-12
            l1 = np.zeros(len(x1))
            l2 = np.zeros(len(x1))
            with np.errstate(invalid="ignore", divide="ignore"):
                l1[ok] = (a22[ok] * b1[ok] - a12[ok] * b2[ok]) / det[ok]
                l2[ok] = (a11[ok] * b2[ok] - a12[ok] * b1[ok]) / det[ok]
            n = int((ok & (l1 > 0) & (l2 > 0)).sum())
            if n > best[2]:
                best = (rot, t, n)
    rot, t, n = best
    return rot, t, float(n) / max(len(x1), 1)


def _pair_parallax_deg(rot, x1, x2):
    """Median triangulation angle of the pair's inlier rays, in degrees."""
    r1 = x1 @ rot.T
    cos = np.clip(np.einsum("ij,ij->i", r1, x2), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(cos))))


def _two_view_witness(m, rec, gh, jm, pair_obs):
    """The pair's own relative pose, from the raw matches alone.

    Structure cannot fake agreement between a resection against it and an
    epipolar estimate that never saw it, which is the whole point of the
    witness.  The pair's matches are the clusters both images carry in the
    capture's admission -- 2D-2D evidence, no point ever consulted."""
    if pair_obs is None:
        rec["e_status"] = "no_pair_source"
        return
    cl_a, uv_a = pair_obs.rows_of(gh)
    cl_b, uv_b = pair_obs.rows_of(jm)
    if not len(cl_a) or not len(cl_b):
        rec["e_status"] = "few_pair_matches"
        return
    _, ia, ib = np.intersect1d(cl_a, cl_b, return_indices=True)
    rec["n_pair_matches"] = int(len(ia))
    if len(ia) < E_MIN_CORR:
        rec["e_status"] = "few_pair_matches"
        return
    ph, pm = uv_a[ia], uv_b[ib]
    if len(ph) > E_MAX_CORR:
        take = np.sort(
            np.random.default_rng(SEED).choice(len(ph), E_MAX_CORR, replace=False)
        )
        ph, pm = ph[take], pm[take]
    x1 = np.asarray(m.camera.pixel_to_ray_batch(np.ascontiguousarray(ph)))
    x2 = np.asarray(m.camera.pixel_to_ray_batch(np.ascontiguousarray(pm)))
    live = np.isfinite(x1).all(axis=1) & np.isfinite(x2).all(axis=1)
    x1, x2 = x1[live], x2[live]
    if len(x1) < E_MIN_CORR:
        rec["e_status"] = "few_pair_matches"
        return
    x1 = np.ascontiguousarray(x1 / np.linalg.norm(x1, axis=1, keepdims=True))
    x2 = np.ascontiguousarray(x2 / np.linalg.norm(x2, axis=1, keepdims=True))
    tol = E_TOL_PX / m.f_eq
    er = estimate_essential_rays(
        x1,
        x2,
        max_angle_rad=tol,
        min_inliers=E_MIN_CORR,
        samples=E_SAMPLES,
        seed=SEED,
    )
    if er is None:
        rec["e_status"] = "no_essential"
        return
    ein = np.asarray(er["inliers"], dtype=bool)
    rec["e_inlier_frac"] = float(ein.mean())
    rec["e_essentialness"] = float(er["essentialness"])
    rec["e_rms_px"] = float(er["rms_rad"]) * m.f_eq
    re_, te, front = _decompose_essential(np.asarray(er["e_matrix"]), x1[ein], x2[ein])
    rec["e_cheirality_frac"] = front
    par = _pair_parallax_deg(re_, x1[ein], x2[ein])
    rec["e_parallax_deg"] = par
    # PARALLAX IN THE ESTIMATOR'S OWN UNIT: how many of its consensus bounds
    # the pair actually subtends.  Derived from the member's camera, never a
    # fleet constant, and a pair below the bar is conditioning-limited.
    rec["e_parallax_bounds"] = math.radians(par) / tol
    rec["e_conditioned"] = bool(rec["e_parallax_bounds"] >= E_PARALLAX_BAR)
    # The PnP-implied relative pose of the same pair, and the two deltas.
    rot_mh = m.rot[jm] @ rec["_rot_h"].T
    t_mh = m.tvec[jm] - rot_mh @ rec["_t_h"]
    rec["rot_delta_deg"] = float(
        np.degrees(Rotation.from_matrix(re_ @ rot_mh.T).magnitude())
    )
    n = float(np.linalg.norm(t_mh))
    if n > 0:
        cos = float(np.clip(np.dot(te, t_mh / n), -1.0, 1.0))
        rec["tdir_delta_deg"] = float(np.degrees(np.arccos(abs(cos))))
        rec["tdir_signed_deg"] = float(np.degrees(np.arccos(cos)))
    rec["e_status"] = "ok"


def nonmember_channels(m, scale, pair_obs):
    """Resect the capture's best-connected HELD-OUT images against the member.

    The held-out image never touched the fit, so the member's stored points are
    legitimate evidence here -- which is exactly what makes this the
    complement of the hold-out self-resection.  Ranking is two-dimensional:
    how many of the member's images an outsider links to leads, its raw
    correspondence count breaks the tie."""
    images = []
    out = {
        "images": images,
        "k_requested": K_HELD_OUT,
        "parallax_bar_bounds": E_PARALLAX_BAR,
    }
    if m.pts is None or not len(m.rows):
        out["unmeasurable_reason"] = "no_structure"
        out["n_measured"] = 0
        return out
    anchored = np.nonzero(m.finite)[0]
    out["n_anchor_clusters"] = int(len(anchored))
    if not len(anchored):
        out["unmeasurable_reason"] = "no_anchor_clusters"
        out["n_measured"] = 0
        return out
    # Rows of the member's own working set on anchored clusters, held by images
    # the member never posed.
    on_anchor = m.finite[m.obs_c]
    held = on_anchor & ~m.posed[m.obs_i] & ~m.dropped[m.obs_i]
    rows_h = np.nonzero(held)[0]
    if not len(rows_h):
        out["unmeasurable_reason"] = "no_heldout_observations"
        out["n_measured"] = 0
        return out
    n_corr = np.bincount(m.obs_i[rows_h], minlength=m.n_img)
    # Connectivity dimension one: distinct member images sharing a cluster.
    seen = np.zeros((m.n_img, m.n_cl), dtype=bool)
    seen[m.obs_i[m.rows], m.obs_c[m.rows]] = True
    mem_seen = seen[m.frames]
    n_link = np.zeros(m.n_img, dtype=np.int64)
    ordr, bounds = _csr(m.obs_i[rows_h], m.n_img)
    rows_h = rows_h[ordr]
    avail = np.nonzero(n_corr >= MIN_CORR)[0]
    for g in avail:
        cl = m.obs_c[rows_h[bounds[g] : bounds[g + 1]]]
        n_link[g] = int(mem_seen[:, np.unique(cl)].any(axis=1).sum())
    out["n_heldout_available"] = int(len(avail))
    order = sorted(avail.tolist(), key=lambda g: (-int(n_link[g]), -int(n_corr[g]), g))[
        :K_HELD_OUT
    ]
    out["n_heldout_tried"] = len(order)
    for rank, g in enumerate(order):
        rec = _resect_one(m, g, rows_h[bounds[g] : bounds[g + 1]], mem_seen, scale)
        rec["rank"] = rank
        rec["n_link"] = int(n_link[g])
        if rec["status"] == "ok":
            _two_view_witness(m, rec, g, rec["_best_member"], pair_obs)
        for k in [k for k in rec if k.startswith("_")]:
            del rec[k]
        images.append(rec)
    ok = [r for r in images if r["status"] == "ok"]
    e = [r for r in ok if r.get("e_status") == "ok"]
    out["n_measured"] = len(ok)
    out["n_unmeasurable"] = K_HELD_OUT - len(ok)
    out["coverage"] = len(ok) / K_HELD_OUT
    out["n_e_measured"] = len(e)
    if not ok:
        out.setdefault("unmeasurable_reason", "no_resectable_heldout_image")
        return out
    out["inlier_med"] = _q([r.get("inlier_frac") for r in ok], 50)
    out["inlier_worst"] = _q([r.get("inlier_frac") for r in ok], 0)
    out["ncorr_med"] = _q([r.get("n_corr") for r in ok], 50)
    out["ncorr_min"] = _q([r.get("n_corr") for r in ok], 0)
    out["link_med"] = _q([r.get("n_link") for r in ok], 50)
    out["mem_spread_med"] = _q([r.get("mem_inlier_spread") for r in ok], 50)
    out["mem_spread_worst"] = _q([r.get("mem_inlier_spread") for r in ok], 100)
    out["mem_inlier_min"] = _q([r.get("mem_inlier_min") for r in ok], 0)
    out["reproj_med"] = _q([r.get("reproj_med_px") for r in ok], 50)
    if e:
        out["rot_med"] = _q([r.get("rot_delta_deg") for r in e], 50)
        out["rot_worst"], out["rot_worst_image"] = _extreme(e, "rot_delta_deg")
        out["tdir_med"] = _q([r.get("tdir_delta_deg") for r in e], 50)
        out["tdir_worst"], out["tdir_worst_image"] = _extreme(e, "tdir_delta_deg")
        out["e_inlier_med"] = _q([r.get("e_inlier_frac") for r in e], 50)
        out["e_parallax_med"] = _q([r.get("e_parallax_deg") for r in e], 50)
        out["e_parallax_min"] = _q([r.get("e_parallax_deg") for r in e], 0)
        out["e_parallax_bounds_med"] = _q([r.get("e_parallax_bounds") for r in e], 50)
        out["e_parallax_bounds_min"] = _q([r.get("e_parallax_bounds") for r in e], 0)
        out["e_cheirality_med"] = _q([r.get("e_cheirality_frac") for r in e], 50)
        cond = [r for r in e if r.get("e_conditioned")]
        out["n_conditioned"] = len(cond)
        out["rot_worst_conditioned"], _ = _extreme(cond, "rot_delta_deg")
        out["tdir_worst_conditioned"], _ = _extreme(cond, "tdir_delta_deg")
        out["tdir_med_conditioned"] = _q([r.get("tdir_delta_deg") for r in cond], 50)
    return out


def _resect_one(m, g, rows, mem_seen, scale):
    """One held-out image resected against the member's stored structure."""
    rec = {"name": m.names[g], "status": "ok", "n_corr_raw": int(len(rows))}
    cl = m.obs_c[rows]
    # A feature claimed by two DIFFERENT clusters is kept: that conflict is the
    # measurement.  An exact duplicate claim is collapsed.
    if m.obs_f is not None:
        key = m.obs_f[rows].astype(np.int64) * (np.int64(1) << np.int64(32)) + cl
        _, uniq = np.unique(key, return_index=True)
        rows, cl = rows[np.sort(uniq)], cl[np.sort(uniq)]
    rec["n_corr"] = int(len(rows))
    if len(rows) < MIN_CORR:
        rec["status"] = "few_correspondences"
        return rec
    uv = np.ascontiguousarray(m.obs_uv[rows])
    x = np.ascontiguousarray(m.pts[cl])
    ans = estimate_absolute_pose(
        uv, x, camera=m.camera, max_error_px=RANSAC_PX, seed=SEED
    )
    if ans is None:
        rec["status"] = "no_consensus"
        return rec
    rec["ransac_inlier_frac"] = float(np.asarray(ans["inliers"]).mean())
    ref = refine_absolute_pose(
        m.camera,
        uv,
        x,
        np.asarray(ans["quaternion_wxyz"], dtype=np.float64),
        np.asarray(ans["translation"], dtype=np.float64),
        TRIM_ROUNDS,
        KEEP_FRACTION,
        FINAL_INLIER_PX,
    )
    rec["inlier_frac"] = float(ref["inlier_fraction"])
    qh = np.asarray(ref["quaternion_wxyz"], dtype=np.float64)
    rot_h = Rotation.from_quat(qh[[1, 2, 3, 0]]).as_matrix()
    t_h = np.asarray(ref["translation"], dtype=np.float64)
    rec["_rot_h"], rec["_t_h"] = rot_h, t_h
    ch = -rot_h.T @ t_h
    # Support recomputed on the member's own map: the model refuses to project
    # what it cannot see, which is the field test.
    xc = x @ rot_h.T + t_h
    proj = np.asarray(m.camera.ray_to_pixel_batch(np.ascontiguousarray(xc)))
    res = np.linalg.norm(proj - uv, axis=1)
    inl = np.isfinite(res) & (res <= FINAL_INLIER_PX)
    rec["n_inliers"] = int(inl.sum())
    rec["reproj_med_px"] = float(np.median(res[inl])) if inl.any() else None
    # PER MEMBER IMAGE: an inconsistent member hands its outsiders conflicting
    # correspondences, and the spread over the images that supplied them is
    # where that shows.
    rates = []
    for b, j in enumerate(m.frames):
        sup = mem_seen[b][cl]
        if int(sup.sum()) >= MIN_MEM_CORR:
            rates.append((float(inl[sup].mean()), int(sup.sum()), m.names[j]))
    rec["n_mem_rated"] = len(rates)
    if len(rates) >= 2:
        rr = [r[0] for r in rates]
        rec["mem_inlier_spread"] = max(rr) - min(rr)
        rec["mem_inlier_min"] = min(rr)
        rec["mem_inlier_worst_image"] = min(rates)[2]
    # The best-connected member image is the pair the two-view witness reads.
    counts = [int(mem_seen[b][cl].sum()) for b in range(len(m.frames))]
    best = int(np.argmax(counts))
    rec["_best_member"] = int(m.frames[best])
    rec["best_member"] = m.names[m.frames[best]]
    rec["best_member_corr"] = counts[best]
    if scale and scale > 0:
        d = np.linalg.norm(m.pts[cl[inl]] - ch, axis=1) if inl.any() else np.zeros(0)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d):
            rec["depth_med_frac"] = float(np.median(d)) / scale
    return rec


# ── Channel: settling probe ─────────────────────────────────────────────────


def settling_channels(m):
    """A SHORT staged bundle adjustment of the whole member, intrinsics frozen.

    The question is where the member goes when it is allowed to settle, read
    gauge-free as the between-frame pose change, plus whether its residual
    grows.  The schedule opens permissive on purpose: a single late-stage round
    would trim the freshly re-triangulated points before the solve reached
    them, and that measures the trim gate, not the member.

    A diverging refit is read from the WORST aggregates -- a member whose
    residual grew has no median worth quoting."""
    out = {"measurable": False}
    if m.pts is None or not len(m.rows):
        out["unmeasurable_reason"] = "no_structure" if m.pts is None else "no_obs"
        return out
    rows = m.rows
    if len(rows) < 12:
        out["unmeasurable_reason"] = "too_few_observations"
        return out
    q0 = _quat(m.rvec)
    t0 = np.ascontiguousarray(m.tvec, dtype=np.float64)
    pts0 = np.ascontiguousarray(m.pts, dtype=np.float64)
    before = _residual_norms(
        m.camera, m.rvec, m.tvec, m.pts, m.obs_uv[rows], m.obs_i[rows], m.obs_c[rows]
    )
    fin_b = before[np.isfinite(before)]
    out["residual_med_before"] = float(np.median(fin_b)) if len(fin_b) else None
    try:
        ba = bundle_adjust(
            m.camera,
            q0,
            t0,
            pts0,
            np.ascontiguousarray(m.obs_uv[rows], dtype=np.float64),
            np.ascontiguousarray(m.obs_i[rows], dtype=np.uint32),
            np.ascontiguousarray(m.obs_c[rows], dtype=np.uint32),
            opt_f=False,
            schedule=SETTLE_SCHEDULE,
            max_iters=SETTLE_MAX_ITERS,
            min_track=2,
            min_obs=12,
        )
    except Exception as exc:  # noqa: BLE001 — evaluation never kills the run
        out["unmeasurable_reason"] = f"{type(exc).__name__}: {exc}"
        return out
    q1 = np.asarray(ba["quaternions_wxyz"], dtype=np.float64)
    t1 = np.asarray(ba["translations"], dtype=np.float64)
    pts1 = np.asarray(ba["points"], dtype=np.float64)
    res1 = np.asarray(ba["residual_norms"], dtype=np.float64)
    if not np.isfinite(res1).any():
        out["unmeasurable_reason"] = "ba_degenerate"
        return out
    fin_a = res1[np.isfinite(res1)]
    out["residual_med_after"] = float(np.median(fin_a))
    out["n_obs"] = int(len(rows))
    out["n_points_before"] = int(m.finite.sum())
    out["n_points_after"] = int(np.isfinite(pts1).all(axis=1).sum())
    out["schedule"] = [list(s) for s in SETTLE_SCHEDULE]
    if out["residual_med_before"] and out["residual_med_after"]:
        out["residual_ratio"] = out["residual_med_after"] / out["residual_med_before"]
        # A DIVERGING REFIT.  Read from the worst aggregates, never the median:
        # a solve that starves most of its frames leaves them exactly where
        # they were, and the median of an untouched population is zero.
        out["diverged"] = bool(out["residual_ratio"] > DIVERGE_RATIO)
    frames = m.frames
    if len(frames) < 2:
        out["unmeasurable_reason"] = "too_few_posed_frames"
        return out
    rot_b = Rotation.from_quat(q1[frames][:, [1, 2, 3, 0]]).as_matrix()
    cen_b = _centres(rot_b, t1[frames])
    pairs = _pose_pairs(len(frames))
    scale = m.scene_scale()
    rot_deg, tdir = _relative_pose_deltas(
        m.rot[frames],
        m.centre[frames],
        rot_b,
        cen_b,
        pairs,
        floor=MIN_BASELINE_FRAC * (scale if scale and scale > 0 else 1.0),
    )
    out["n_pairs"] = int(len(pairs))
    _agg(rot_deg.tolist(), "rot", out)
    _agg(tdir.tolist(), "tdir", out)
    # Where the settling LOCALIZES: each frame's median over the pairs it is in.
    per_frame = []
    for k, j in enumerate(frames):
        touch = (pairs[:, 0] == k) | (pairs[:, 1] == k)
        rec = {"name": m.names[j]}
        for tag, v in (("rot", rot_deg), ("tdir", tdir)):
            vals = v[touch]
            vals = vals[np.isfinite(vals)]
            rec[f"{tag}_med"] = float(np.median(vals)) if len(vals) else None
        per_frame.append(rec)
    out["frames"] = per_frame
    out["rot_worst_frame"] = _extreme(per_frame, "rot_med")[1]
    out["tdir_worst_frame"] = _extreme(per_frame, "tdir_med")[1]
    out["measurable"] = True
    return out


# ── Channel: warp epipolar consistency ──────────────────────────────────────


def _inv2(mat):
    """``(inverse, determinant)`` of a batch of 2x2 matrices."""
    det = mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 0, 1] * mat[:, 1, 0]
    out = np.empty_like(mat)
    out[:, 0, 0] = mat[:, 1, 1]
    out[:, 1, 1] = mat[:, 0, 0]
    out[:, 0, 1] = -mat[:, 0, 1]
    out[:, 1, 0] = -mat[:, 1, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / det[:, None, None]
    return out, det


def _tangent_basis(u):
    """``(n, 3, 2)`` orthonormal tangent basis of the sphere at unit rays ``u``.

    Deterministic: the seed axis is the canonical axis the ray is least aligned
    with, so the basis is a function of the ray alone."""
    n = len(u)
    e = np.zeros((n, 3))
    e[np.arange(n), np.argmin(np.abs(u), axis=1)] = 1.0
    b1 = e - u * np.sum(u * e, axis=1, keepdims=True)
    nb = np.linalg.norm(b1, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b1 = b1 / np.where(nb > 0, nb, 1.0)
    return np.stack([b1, np.cross(u, b1)], axis=2)


def _angular_chart(cam, xy, h=WARP_JAC_STEP_PX):
    """``(u, B, P, r, J)`` at pixels ``xy``: unit ray, tangent basis, the
    pixel-to-angular map ``P = Bᵀ J / |r|``, and the raw ray and Jacobian.

    The chart is ANGULAR, not normalized-pinhole, which is what makes the warp
    residuals camera-model-generic: a fisheye member past 90 degrees off axis
    has no perspective chart, and always has a tangent plane."""
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    r = np.asarray(cam.pixel_to_ray_batch(xy), dtype=np.float64)
    rho = np.linalg.norm(r, axis=1)
    safe = np.where(rho > 0, rho, 1.0)
    u = r / safe[:, None]
    basis = _tangent_basis(u)
    cols = []
    for k in (0, 1):
        d = np.zeros_like(xy)
        d[:, k] = h
        rp = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(xy + d)))
        rm = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(xy - d)))
        cols.append((rp - rm) / (2.0 * h))
    jac = np.stack(cols, axis=2)
    return u, basis, np.einsum("nab,nac->nbc", basis, jac) / safe[:, None, None], r, jac


def _track_pairs(point_index, obs_index):
    """``(i, j, point)`` over UNORDERED observation pairs sharing a point.

    Past ``WARP_MAX_VIEWS_ALL_PAIRS`` views a track is thinned to evenly spaced
    partners per view, which keeps every view represented while bounding the
    cost.  Deterministic: the offsets are a linspace and the survivors are
    taken in sorted key order."""
    order = np.argsort(point_index, kind="stable")
    pts, obs = point_index[order], obs_index[order]
    bounds = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1], True])
    ii, jj, pt = [], [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        k = int(e - s)
        if k < 2:
            continue
        grp = obs[s:e]
        if k <= WARP_MAX_VIEWS_ALL_PAIRS:
            a, b = np.triu_indices(k, 1)
        else:
            offs = np.unique(
                np.linspace(1, k - 1, WARP_PARTNERS_PER_VIEW).round().astype(np.int64)
            )
            a0 = np.repeat(np.arange(k), len(offs))
            b0 = (np.tile(offs, k) + a0) % k
            lo, hi = np.minimum(a0, b0), np.maximum(a0, b0)
            _, uniq = np.unique(lo * k + hi, return_index=True)
            a, b = lo[np.sort(uniq)], hi[np.sort(uniq)]
        ii.append(grp[a])
        jj.append(grp[b])
        pt.append(np.full(len(a), pts[s]))
    if not ii:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    ii, jj, pt = np.concatenate(ii), np.concatenate(jj), np.concatenate(pt)
    if len(ii) > WARP_MAX_PAIRS:
        take = np.unique(
            np.linspace(0, len(ii) - 1, WARP_MAX_PAIRS).round().astype(np.int64)
        )
        ii, jj, pt = ii[take], jj[take], pt[take]
    return ii, jj, pt


def _warp_residuals(warp, chart, ii, jj, img_i, img_j, rot, tvec, d_i, d_j):
    """``(nf_res, epi_res, vergence_deg)`` for a batch of observation pairs.

    Two readings of the SAME two normal-free numbers.  ``nf_res`` is the
    tangent-chart form: the component of the measured warp's departure from the
    pose-only warp that is perpendicular to the epipolar direction, over the
    pose-only warp's own magnitude.  ``epi_res`` is the classical form in
    pixels, from differentiating the epipolar constraint along the
    correspondence.  The remaining two numbers of the warp are the tangent
    plane's, and neither reading touches them."""
    u, basis, chart_p, ray, jac = chart
    u_i = u[ii]
    b_i, b_j = basis[ii], basis[jj]
    n = len(ii)
    rel = np.einsum("nab,ncb->nac", rot[img_j], rot[img_i])
    t_rel = tvec[img_j] - np.einsum("nab,nb->na", rel, tvec[img_i])
    w0 = np.einsum("nab,nac->nbc", b_j, np.einsum("nab,nbc->nac", rel, b_i))
    a = np.einsum("nab,na->nb", b_j, np.einsum("nab,nb->na", rel, u_i))
    pi_inv, _ = _inv2(chart_p[ii])
    atil = np.einsum(
        "nab,nbc->nac", chart_p[jj], np.einsum("nab,nbc->nac", warp, pi_inv)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        c = d_i / d_j
        diff = atil / c[:, None, None] - w0
    na = np.linalg.norm(a, axis=1)
    ahat = a / np.where(na > 0, na, 1.0)[:, None]
    along = np.einsum("na,nab->nb", ahat, diff)
    perp = diff - ahat[:, :, None] * along[:, None, :]
    w0n = np.linalg.norm(w0.reshape(n, 4), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nf = np.linalg.norm(perp.reshape(n, 4), axis=1) / np.where(w0n > 0, w0n, np.nan)
    verg = np.degrees(np.arcsin(np.clip(na, 0.0, 1.0)))
    # The classical epipolar residual, in each image's own pixels.  `l` is the
    # local epipolar-line normal there, so no perspective chart is needed.
    skew = np.zeros((n, 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -t_rel[:, 2], t_rel[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = t_rel[:, 2], -t_rel[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -t_rel[:, 1], t_rel[:, 0]
    ess = np.einsum("nab,nbc->nac", skew, rel)
    l_i = np.einsum("nab,na->nb", jac[ii], np.einsum("nba,nb->na", ess, ray[jj]))
    l_j = np.einsum("nab,na->nb", jac[jj], np.einsum("nab,nb->na", ess, ray[ii]))
    at_lj = np.einsum("nba,nb->na", warp, l_j)
    den = np.linalg.norm(l_i, axis=1) + np.linalg.norm(at_lj, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        epi = np.linalg.norm(at_lj + l_i, axis=1) / np.where(den > 0, den, np.nan)
    return nf, epi, verg


def warp_channels(m):
    """The member's stored affine shapes against its own relative poses.

    Every other geometry channel re-derives something the member already
    produced.  This one reads a measurement the member never used: the patch
    refinement fitted each cluster member's shape photometrically, and two
    members of one cluster therefore carry a measured pixel-to-pixel warp that
    the member's pose and point predict up to the tangent-plane normal."""
    out = {
        "measurable": False,
        "vergence_floor_deg": WARP_VERGENCE_FLOOR_DEG,
        "jacobian_step_px": WARP_JAC_STEP_PX,
        "frames": [],
    }
    if m.pts is None or not len(m.rows):
        out["unmeasurable_reason"] = "no_structure" if m.pts is None else "no_obs"
        return out
    if m.obs_shape is None:
        out["unmeasurable_reason"] = "no_member_shapes"
        return out
    rows = m.rows
    shp = m.obs_shape[rows]
    det = shp[:, 0, 0] * shp[:, 1, 1] - shp[:, 0, 1] * shp[:, 1, 0]
    usable = np.isfinite(shp).all(axis=(1, 2)) & (np.abs(det) > 0)
    out["n_obs"] = int(len(rows))
    out["shape_frac"] = float(usable.mean()) if len(rows) else None
    idx = rows[usable]
    if len(idx) < 2 * WARP_MIN_PAIRS:
        out["unmeasurable_reason"] = "few_shaped_observations"
        return out
    ii, jj, pt = _track_pairs(m.obs_c[idx], idx)
    out["n_pairs"] = int(len(ii))
    out["n_points"] = int(len(np.unique(pt))) if len(pt) else 0
    if len(ii) < WARP_MIN_PAIRS:
        out["unmeasurable_reason"] = "few_pairs"
        return out
    # One chart batch per image: five map evaluations an image, not five a row.
    n_rows = len(m.obs_uv)
    u = np.full((n_rows, 3), np.nan)
    basis = np.full((n_rows, 3, 2), np.nan)
    chart_p = np.full((n_rows, 2, 2), np.nan)
    ray = np.full((n_rows, 3), np.nan)
    jac = np.full((n_rows, 3, 2), np.nan)
    for j in m.frames:
        sel = idx[m.obs_i[idx] == j]
        if not len(sel):
            continue
        u[sel], basis[sel], chart_p[sel], ray[sel], jac[sel] = _angular_chart(
            m.camera, m.obs_uv[sel]
        )
    warp = np.einsum("nab,nbc->nac", m.obs_shape[jj], _inv2(m.obs_shape[ii])[0])
    x = m.pts[pt]
    img_i, img_j = m.obs_i[ii], m.obs_i[jj]
    d_i = np.linalg.norm(x - m.centre[img_i], axis=1)
    d_j = np.linalg.norm(x - m.centre[img_j], axis=1)
    nf, epi, verg = _warp_residuals(
        warp,
        (u, basis, chart_p, ray, jac),
        ii,
        jj,
        img_i,
        img_j,
        m.rot,
        m.tvec,
        d_i,
        d_j,
    )
    live = np.isfinite(nf) & np.isfinite(verg)
    out["n_pairs_finite"] = int(live.sum())
    if int(live.sum()) < WARP_MIN_PAIRS:
        out["unmeasurable_reason"] = "few_finite_pairs"
        return out
    nf, epi, verg = nf[live], epi[live], verg[live]
    img_i, img_j = img_i[live], img_j[live]
    cond = verg >= WARP_VERGENCE_FLOOR_DEG
    out["n_conditioned"] = int(cond.sum())
    out["conditioned_frac"] = float(cond.mean())
    out["verg_med"] = _q(verg, 50)
    out["verg_p10"] = _q(verg, 10)
    out["verg_p90"] = _q(verg, 90)
    for tag, v in (("nf", nf), ("epi", epi)):
        out[f"{tag}_med"] = _q(v, 50)
        out[f"{tag}_p90"] = _q(v, 90)
        out[f"{tag}_p99"] = _q(v, 99)
        if int(cond.sum()) >= WARP_MIN_PAIRS:
            out[f"{tag}_cond_med"] = _q(v[cond], 50)
            out[f"{tag}_cond_p90"] = _q(v[cond], 90)
    # Per frame, over every pair the frame takes part in on either side.
    side_img = np.concatenate([img_i, img_j])
    side_nf = np.concatenate([nf, nf])
    side_epi = np.concatenate([epi, epi])
    side_verg = np.concatenate([verg, verg])
    order, bnd = _csr(side_img, m.n_img)
    for j in m.frames:
        take = order[bnd[j] : bnd[j + 1]]
        rec = {"name": m.names[j], "n_pairs": int(len(take))}
        if len(take):
            v = side_verg[take]
            c = v >= WARP_VERGENCE_FLOOR_DEG
            rec["verg_med"] = _q(v, 50)
            rec["n_pairs_cond"] = int(c.sum())
            for tag, arr in (("nf", side_nf[take]), ("epi", side_epi[take])):
                rec[f"{tag}_med"] = _q(arr, 50)
                rec[f"{tag}_p90"] = _q(arr, 90)
                if int(c.sum()):
                    rec[f"{tag}_cond_med"] = _q(arr[c], 50)
        out["frames"].append(rec)
    out["nf_frame_worst"], out["nf_worst_frame"] = _extreme(out["frames"], "nf_med")
    out["epi_frame_worst"], out["epi_worst_frame"] = _extreme(out["frames"], "epi_med")
    out["nf_frame_med"] = _q([f.get("nf_med") for f in out["frames"]], 50)
    out["epi_frame_med"] = _q([f.get("epi_med") for f in out["frames"]], 50)
    out["measurable"] = True
    return out


# ── Channels: local surface shape ───────────────────────────────────────────


def _pair_spacing(nbrs):
    """Each neighbourhood's median nearest-neighbour distance INSIDE itself.

    The local length every residual below is divided by.  It scales exactly
    like the residual under a similarity transform, which is what makes the
    ratio gauge-free."""
    n, k, _ = nbrs.shape
    out = np.empty(n, dtype=np.float64)
    diag = np.arange(k)
    for lo in range(0, n, SURF_CHUNK):
        hi = min(lo + SURF_CHUNK, n)
        d = np.linalg.norm(nbrs[lo:hi, :, None, :] - nbrs[lo:hi, None, :, :], axis=-1)
        d[:, diag, diag] = np.inf
        out[lo:hi] = np.median(d.min(axis=2), axis=1)
    return out


def _robust_plane(nbrs, spacing):
    """Robust total-least-squares plane through each neighbourhood.

    Tukey-redescending IRLS on the orthogonal residual, in the same shape as
    the repo's adjacency-surfel fit: a fixed pass count, no randomness, and a
    per-point stall exit so an all-zero scatter never yields an arbitrary axis.
    The robust scale is floored at a fraction of the neighbourhood's OWN
    spacing, so the floor is a length the data supplied."""
    n, k, _ = nbrs.shape
    w = np.ones((n, k), dtype=np.float64)
    floor = np.where(spacing > 0, 1e-3 * spacing, 1.0)
    cen = np.zeros((n, 3))
    nrm = np.zeros((n, 3))
    stalled = np.zeros(n, dtype=bool)
    for it in range(SURF_IRLS_ITERS + 1):
        wsum = w.sum(axis=1)
        wsum = np.where(wsum > 0, wsum, 1.0)
        cen = np.einsum("nk,nkj->nj", w, nbrs) / wsum[:, None]
        dev = nbrs - cen[:, None, :]
        mom = np.einsum("nk,nki,nkj->nij", w, dev, dev) / wsum[:, None, None]
        nrm = np.linalg.eigh(mom)[1][:, :, 0]
        r = np.abs(np.einsum("nkj,nj->nk", dev, nrm))
        sig = np.maximum(1.4826 * np.median(r, axis=1), floor)
        if it == SURF_IRLS_ITERS:
            break
        t = r / (SURF_TUKEY_C * sig[:, None])
        w_new = np.where(t < 1.0, (1.0 - t * t) ** 2, 0.0)
        dead = w_new.sum(axis=1) <= 1e-12
        stalled |= dead
        w = np.where(dead[:, None] | stalled[:, None], w, w_new)
    return cen, nrm


def _plane_stats(p, nbrs):
    """``(residual, surface variation)`` for query points and neighbourhoods.

    The residual is LEAVE-ONE-OUT: the plane never sees ``p``, so a point
    pushed off its own surface is measured against the surface rather than
    against a plane it helped define.  The surface variation DOES include
    ``p``, because it describes the neighbourhood's shape."""
    n = p.shape[0]
    if not n:
        return np.zeros(0), np.zeros(0)
    spacing = _pair_spacing(nbrs)
    cen, nrm = _robust_plane(nbrs, spacing)
    d = np.abs(np.einsum("nj,nj->n", p - cen, nrm))
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.where(spacing > 0, d / spacing, np.nan)
    full = np.concatenate([p[:, None, :], nbrs], axis=1)
    dev = full - full.mean(axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", dev, dev) / full.shape[1])
    tot = ev.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sv = np.where(tot > 0, ev[:, 0] / tot, np.nan)
    return res, sv


def _grouped_plane(p, groups, counts):
    """``(residual, surface variation)`` for variable-size neighbourhoods."""
    n = p.shape[0]
    res, sv = np.full(n, np.nan), np.full(n, np.nan)
    for size in np.unique(counts):
        if size < 3:
            continue
        sel = np.nonzero(counts == size)[0]
        nbrs = np.stack([groups[i][:size] for i in sel], axis=0)
        res[sel], sv[sel] = _plane_stats(p[sel], nbrs)
    return res, sv


def _observer_bitsets(m, slot, n_pts):
    """``(n_points, words)`` packed sets of the images observing each point."""
    words = (m.n_img + 63) // 64
    bits = np.zeros((n_pts, words), dtype=np.uint64)
    rows = m.rows
    row_slot = slot[m.obs_c[rows]]
    live = row_slot >= 0
    img = m.obs_i[rows][live]
    np.bitwise_or.at(
        bits,
        (row_slot[live], (img >> 6).astype(np.int64)),
        (np.uint64(1) << (img & 63).astype(np.uint64)),
    )
    return bits


def _shares_any(bits, ii, jj):
    """Whether rows ``ii`` and ``jj`` of the bitset table share a set bit."""
    out = np.zeros(len(ii), dtype=bool)
    for lo in range(0, len(ii), SURF_CHUNK):
        hi = min(lo + SURF_CHUNK, len(ii))
        out[lo:hi] = (np.bitwise_and(bits[ii[lo:hi]], bits[jj[lo:hi]]) != 0).any(axis=1)
    return out


def _pick_neighbours(mask, cand, positions, keep, floor):
    """``(neighbourhoods, counts, chosen)`` from a per-row candidate mask."""
    counts = np.minimum(mask.sum(axis=1), keep)
    groups, ok = [], np.zeros(len(mask), dtype=bool)
    for i in range(len(mask)):
        if counts[i] < floor:
            groups.append(np.zeros((0, 3)))
            continue
        sel = np.nonzero(mask[i])[0][:keep]
        groups.append(positions[cand[i, sel]])
        ok[i] = True
    return groups, np.where(ok, counts, 0), ok


def _frame_point_slots(m, slot):
    """``{image: slots}`` of the finite points each posed frame observes."""
    out = {}
    for j in m.frames:
        rows = m.frame_rows(j)
        s = slot[m.obs_c[rows]] if len(rows) else np.zeros(0, np.int64)
        out[int(j)] = s[s >= 0] if len(s) else np.zeros(0, np.int64)
    return out


def _frame_rollup(per_frame, key, out, prefix):
    """Median and worst of a per-frame reading, with the worst frame named."""
    out[f"{prefix}_frame_med"] = _q([f.get(key) for f in per_frame], 50)
    out[f"{prefix}_frame_worst"], out[f"{prefix}_worst_frame"] = _extreme(
        per_frame, key
    )


def surface_channels(m):
    """The three surface readings of the member's own point cloud.

    All three share one neighbour search and one plane primitive, and none of
    them consults a residual, an inlier count or a gauge: they ask only whether
    the cloud the member released looks like the surfaces it was taken of."""
    from scipy.spatial import cKDTree

    stranger = {"measurable": False, "frames": []}
    variation = {"measurable": False, "k": SURF_K, "frames": []}
    vetted = {"measurable": False, "frames": []}
    blocks = {
        "stranger_surface": stranger,
        "surface_variation": variation,
        "range_vetted_surface": vetted,
    }
    stranger.update(
        k_candidates=SURF_C_CAND, k_used=SURF_C_USE, support_floor=SURF_C_MIN
    )
    vetted.update(
        k_candidates=SURF_BV_CAND, k_used=SURF_BV_USE, support_floor=SURF_BV_MIN
    )
    if m.pts is None or not len(m.rows):
        why = "no_structure" if m.pts is None else "no_obs"
        for b in blocks.values():
            b["unmeasurable_reason"] = why
        return blocks
    fin = np.nonzero(m.finite)[0]
    fin = fin[np.isfinite(m.pts[fin]).all(axis=1)]
    n_pts = len(fin)
    for b in blocks.values():
        b["n_points"] = int(n_pts)
        b["n_points_infinite"] = int((~m.finite).sum())
    if n_pts < SURF_K + 2:
        for b in blocks.values():
            b["unmeasurable_reason"] = "few_finite_points"
        return blocks
    x = m.pts[fin]
    slot = np.full(m.n_cl, -1, dtype=np.int64)
    slot[fin] = np.arange(n_pts)
    tree = cKDTree(x)
    kq = min(max(SURF_K, SURF_C_CAND) + 1, n_pts)
    dist, idx = tree.query(x, k=kq, workers=-1)

    # -- local surface variation, and the plain k-NN residual beside it -----
    a_res = a_sv = None
    if SURF_K + 1 <= kq:
        a_res, a_sv = _plane_stats(x, x[idx[:, 1 : SURF_K + 1]])
        variation["point_sv_med"] = _q(a_sv, 50)
        variation["point_sv_p90"] = _q(a_sv, 90)
        variation["point_res_med"] = _q(a_res, 50)
        variation["point_res_p90"] = _q(a_res, 90)
        variation["measurable"] = True
    else:
        variation["unmeasurable_reason"] = "few_finite_points"

    # -- stranger-surface membership ---------------------------------------
    bits = _observer_bitsets(m, slot, n_pts)
    ncand = min(SURF_C_CAND, kq - 1)
    cand = idx[:, 1 : ncand + 1]
    cdist = dist[:, 1 : ncand + 1]
    ii = np.repeat(np.arange(n_pts), ncand)
    disj = ~_shares_any(bits, ii, cand.reshape(-1)).reshape(n_pts, ncand)
    groups, counts, ok = _pick_neighbours(disj, cand, x, SURF_C_USE, SURF_C_MIN)
    own = np.median(cdist[:, : min(SURF_K, ncand)], axis=1)
    loc = np.full(n_pts, np.nan)
    for i in np.nonzero(ok)[0]:
        sel = np.nonzero(disj[i])[0][:SURF_C_USE]
        if own[i] > 0:
            loc[i] = float(np.median(cdist[i, sel]) / own[i])
    stranger["n_measurable"] = int(ok.sum())
    stranger["frac_measurable"] = float(ok.mean())
    stranger["disjoint_avail_med"] = float(np.median(disj.sum(axis=1)))
    c_res = np.full(n_pts, np.nan)
    if int(ok.sum()):
        c_res, _ = _grouped_plane(x, groups, counts)
        stranger["res_med"] = _q(c_res, 50)
        stranger["res_p90"] = _q(c_res, 90)
        stranger["locality_med"] = _q(loc, 50)
        # The strangers whose surface is no further away than the point's own
        # neighbourhood -- a within-member quantile, not a length.
        bar = stranger["locality_med"]
        if bar is not None:
            tight = np.isfinite(loc) & (loc <= bar)
            if int(tight.sum()) >= MIN_SUPPORT_POINTS:
                stranger["res_local_med"] = _q(c_res[tight], 50)
                stranger["n_local"] = int(tight.sum())
        if a_res is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = c_res / np.where(a_res > 0, a_res, np.nan)
            stranger["over_local_med"] = _q(ratio, 50)
        stranger["measurable"] = True
    else:
        stranger["unmeasurable_reason"] = "no_covisibility_disjoint_neighbour"

    # -- per frame: the two point readings, and the range-vetted one --------
    by_frame = _frame_point_slots(m, slot)
    bv_sum = np.zeros(n_pts)
    bv_cnt = np.zeros(n_pts)
    bars = []
    for j in m.frames:
        name = m.names[j]
        sp = by_frame[int(j)]
        rec = {"name": name, "n_points": int(len(sp))}
        srec = {"name": name, "n_points": int(len(sp))}
        if len(sp):
            if a_sv is not None:
                rec["sv_med"] = _q(a_sv[sp], 50)
                rec["res_med"] = _q(a_res[sp], 50)
            srec["res_med"] = _q(c_res[sp], 50)
            srec["res_p90"] = _q(c_res[sp], 90)
            srec["n_measurable"] = int(np.isfinite(c_res[sp]).sum())
        variation["frames"].append(rec)
        stranger["frames"].append(srec)
        rows = m.frame_rows(j)
        keep_row = slot[m.obs_c[rows]] >= 0
        rows = rows[keep_row]
        vrec = {"name": name, "n_obs": int(len(rows))}
        if len(rows) < SURF_BV_MIN + 2:
            vrec["status"] = "few_observations"
            vetted["frames"].append(vrec)
            continue
        vrec["status"] = "ok"
        sp_row = slot[m.obs_c[rows]]
        uv = m.obs_uv[rows]
        t2 = cKDTree(uv)
        kq2 = min(SURF_BV_CAND + 1, len(uv))
        _, i2 = t2.query(uv, k=kq2, workers=-1)
        pos = x[sp_row]
        rng = np.linalg.norm(pos - m.centre[j], axis=1)
        cv = i2[:, 1:kq2]
        with np.errstate(invalid="ignore", divide="ignore"):
            dr = np.abs(rng[:, None] - rng[cv]) / np.maximum(
                0.5 * (rng[:, None] + rng[cv]), 1e-12
            )
        # THE VET BAR IS THIS FRAME'S OWN MEDIAN relative range difference: a
        # within-frame quantile, so a near scene and a far one are vetted at
        # the same strictness in their own terms.
        bar = float(np.median(dr[np.isfinite(dr)])) if np.isfinite(dr).any() else None
        if bar is None:
            vrec["status"] = "no_range_bar"
            vetted["frames"].append(vrec)
            continue
        vrec["range_bar"] = bar
        bars.append(bar)
        groups_v, counts_v, ok_v = _pick_neighbours(
            dr <= bar, cv, x[sp_row], SURF_BV_USE, SURF_BV_MIN
        )
        vrec["frac_measurable"] = float(ok_v.mean())
        if int(ok_v.sum()):
            res_v, sv_v = _grouped_plane(pos, groups_v, counts_v)
            vrec["res_med"] = _q(res_v, 50)
            vrec["res_p90"] = _q(res_v, 90)
            vrec["sv_med"] = _q(sv_v, 50)
            good = np.isfinite(res_v)
            np.add.at(bv_sum, sp_row[good], res_v[good])
            np.add.at(bv_cnt, sp_row[good], 1.0)
        vetted["frames"].append(vrec)

    if variation["frames"]:
        _frame_rollup(variation["frames"], "sv_med", variation, "sv")
        _frame_rollup(variation["frames"], "res_med", variation, "res")
    if stranger["frames"]:
        _frame_rollup(stranger["frames"], "res_med", stranger, "res")
    if bars:
        vetted["range_bar_med"] = float(np.median(bars))
        vetted["frac_measurable_med"] = _q(
            [f.get("frac_measurable") for f in vetted["frames"]], 50
        )
        _frame_rollup(vetted["frames"], "res_med", vetted, "res")
    live = bv_cnt > 0
    if int(live.sum()) >= MIN_SUPPORT_POINTS:
        per_point = bv_sum[live] / bv_cnt[live]
        vetted["res_med"] = _q(per_point, 50)
        vetted["res_p90"] = _q(per_point, 90)
        vetted["n_points_measurable"] = int(live.sum())
        vetted["measurable"] = True
    else:
        vetted.setdefault("unmeasurable_reason", "no_range_vetted_neighbourhood")
    return blocks


# ── Channels: the rotation-only member ──────────────────────────────────────
#
# A far-field member claims BEARING WITHOUT RANGE: one rotation per frame, one
# direction per explained cluster, no baseline anywhere.  Every finite channel
# has a rotation-only form, and the forms are not weaker readings of the same
# thing -- a hold-out resection of a rotation is a different measurement from a
# hold-out resection of a pose, and the warp channel is STRONGER here, because
# a pure rotation predicts all four numbers of a cluster's warp instead of two.
# The surface channels have no rotation-only form at all: there is no cloud.


def _kabsch(a, b):
    """The rotation ``R`` minimizing the angle between ``b`` and ``R a``."""
    u, _, vt = np.linalg.svd(np.asarray(a).T @ np.asarray(b))
    d = float(np.sign(np.linalg.det(vt.T @ u.T)))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _rot_residuals(rot, a, b):
    """Angles between ``b`` and ``R a``, in radians."""
    return np.arccos(np.clip(np.einsum("ij,ij->i", b, a @ rot.T), -1.0, 1.0))


def _trimmed_rotation(a, b, tol_rad):
    """``(R, inlier_fraction, n_inliers, residuals)`` by TRIMMED Kabsch.

    Closed-form and global at every round, so no sampling and no seed: the fit
    drops its own worst readings and refits, which is the shape the pipeline's
    trimmed pose refinement already uses."""
    n = len(a)
    if n < 3:
        return None, None, 0, None
    keep = np.ones(n, dtype=bool)
    rot = None
    for _round in range(ROT_TRIM_ROUNDS):
        rot = _kabsch(a[keep], b[keep])
        ang = _rot_residuals(rot, a, b)
        take = max(3, int(round(ROT_KEEP_FRACTION * n)))
        keep = np.zeros(n, dtype=bool)
        keep[np.argsort(ang, kind="stable")[:take]] = True
    ang = _rot_residuals(rot, a, b)
    inl = ang <= tol_rad
    return rot, float(inl.mean()), int(inl.sum()), ang


def _local_rays(cam, uv):
    """Unit camera-frame rays of ``uv``, or NaN where the model refuses."""
    r = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(uv, np.float64)))
    n = np.linalg.norm(r, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return r / np.where(n > 0, n, np.nan)


def _world_dirs_from(m, rows):
    """Each cluster's world direction as the mean of its rotated rays.

    Returns ``(cluster ids, directions, counts)`` over the clusters ``rows``
    covers.  The mean of unit rays, renormalized: with the rays this tight it
    is the same answer as a spherical mean and costs one pass."""
    cl = m.obs_c[rows]
    order = np.argsort(cl, kind="stable")
    cl, rows = cl[order], rows[order]
    uniq, starts, counts = np.unique(cl, return_index=True, return_counts=True)
    loc = _local_rays(m.camera, m.obs_uv[rows])
    world = np.einsum("nji,nj->ni", m.rot[m.obs_i[rows]], loc)
    acc = np.zeros((len(uniq), 3))
    slot = np.repeat(np.arange(len(uniq)), counts)
    good = np.isfinite(world).all(axis=1)
    np.add.at(acc, slot[good], world[good])
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = acc / np.where(n > 0, n, np.nan)
    del starts
    return uniq, acc, counts


def _rot_tol(m):
    """The angular consensus bound, from the member's own equivalent focal."""
    return E_TOL_PX / m.f_eq if m.f_eq else math.radians(0.5)


def rot_self_resection_channels(m):
    """Hold out a frame and re-estimate its ROTATION from the other frames.

    The frame's directions are re-derived from its OTHER observing frames
    alone (the stored ones are contaminated: this frame helped fix them), and
    the frame's own rotation is then refitted against them.  The comparison is
    a rotation delta in a world frame the other frames define, so nothing here
    compares an absolute pose across independently fit geometry.

    Conditioning is the support's own angular SPREAD: a frame whose support
    directions lie in a narrow cone leaves the rotation about that cone's axis
    undetermined, and a delta read there is arithmetic on noise."""
    frames = []
    out = {"measurable": False, "frames": []}
    if m.pts is None or not len(m.rows):
        out["unmeasurable_reason"] = "no_directions" if m.pts is None else "no_obs"
        return out
    tol = _rot_tol(m)
    out["tol_rad"] = tol
    picks = m.frames
    if len(picks) > ROT_MAX_HOLDOUT_FRAMES:
        take = np.unique(
            np.linspace(0, len(picks) - 1, ROT_MAX_HOLDOUT_FRAMES)
            .round()
            .astype(np.int64)
        )
        picks = picks[take]
    out["n_frames_posed"] = int(len(m.frames))
    out["n_frames_tried"] = int(len(picks))
    for j in picks:
        rec = {"name": m.names[j], "status": "ok"}
        rows_j = m.frame_rows(j)
        rec["n_obs"] = int(len(rows_j))
        if not len(rows_j):
            rec["status"] = "no_observations"
            frames.append(rec)
            continue
        uniq, first = np.unique(m.obs_c[rows_j], return_index=True)
        sel = m.rows[(m.obs_i[m.rows] != j) & np.isin(m.obs_c[m.rows], uniq)]
        if not len(sel):
            rec["status"] = "no_other_support"
            frames.append(rec)
            continue
        pts_o, dirs_o, counts = _world_dirs_from(m, sel)
        good = (counts >= ROT_MIN_SUPPORT_OBS) & np.isfinite(dirs_o).all(axis=1)
        rec["n_candidate_points"] = int(good.sum())
        rec["cross_frac"] = float(good.sum() / max(len(uniq), 1))
        if int(good.sum()) < ROT_MIN_SUPPORT:
            rec["status"] = "few_support_directions"
            frames.append(rec)
            continue
        pts_o, dirs_o = pts_o[good], dirs_o[good]
        pos = {int(p): k for k, p in enumerate(uniq)}
        rays = _local_rays(
            m.camera, m.obs_uv[rows_j[first[[pos[int(p)] for p in pts_o]]]]
        )
        live = np.isfinite(rays).all(axis=1)
        pts_o, dirs_o, rays = pts_o[live], dirs_o[live], rays[live]
        rec["n_points"] = int(len(rays))
        if len(rays) < ROT_MIN_SUPPORT:
            rec["status"] = "few_support_directions"
            frames.append(rec)
            continue
        # CONDITIONING: how wide a cone the support subtends, in the frame's
        # own view.  Two nearly-parallel directions fix two of the rotation's
        # three degrees of freedom and leave the third to noise.
        rec["support_spread_deg"] = float(
            np.degrees(
                np.arccos(np.clip(np.min(rays @ rays.T), -1.0, 1.0))
                if len(rays) > 1
                else 0.0
            )
        )
        # STRUCTURE: how far the re-derived directions sit from the stored
        # ones.  The rotation-only twin of the re-triangulation depth
        # agreement, and it survives a rotation fit that never converges.
        dev = np.degrees(
            np.arccos(np.clip(np.einsum("ij,ij->i", dirs_o, m.pts[pts_o]), -1.0, 1.0))
        )
        dev = dev[np.isfinite(dev)]
        if len(dev):
            rec["dir_dev_med_deg"] = float(np.median(dev))
            rec["dir_dev_p90_deg"] = float(np.percentile(dev, 90))
            rec["dir_dev_worst_deg"] = float(np.max(dev))
        rot, frac, n_inl, _ang = _trimmed_rotation(dirs_o, rays, tol)
        if rot is None:
            rec["status"] = "no_rotation_fit"
            frames.append(rec)
            continue
        rec["inlier_frac"] = frac
        rec["n_inliers"] = n_inl
        rec["rot_delta_deg"] = float(
            np.degrees(Rotation.from_matrix(rot @ m.rot[j].T).magnitude())
        )
        frames.append(rec)
    out["frames"] = frames
    ok = [f for f in frames if f["status"] == "ok"]
    out["n_measured"] = len(ok)
    out["unmeasurable_frac"] = 1.0 - len(ok) / len(frames) if frames else None
    out["cross_frac_med"] = _q([f.get("cross_frac") for f in frames], 50)
    if not ok:
        out["unmeasurable_reason"] = "no_frame_resected"
        return out
    _agg([f.get("rot_delta_deg") for f in ok], "rot", out)
    out["rot_worst"], out["rot_worst_frame"] = _extreme(ok, "rot_delta_deg")
    out["inlier_med"] = _q([f.get("inlier_frac") for f in ok], 50)
    out["inlier_min"] = _q([f.get("inlier_frac") for f in ok], 0)
    out["dir_dev_med_deg"] = _q([f.get("dir_dev_med_deg") for f in ok], 50)
    out["dir_dev_p90_deg"] = _q([f.get("dir_dev_med_deg") for f in ok], 90)
    out["dir_dev_worst_deg"], out["dir_dev_worst_frame"] = _extreme(
        ok, "dir_dev_med_deg"
    )
    out["support_spread_med_deg"] = _q([f.get("support_spread_deg") for f in ok], 50)
    out["support_spread_min_deg"] = _q([f.get("support_spread_deg") for f in ok], 0)
    out["measurable"] = True
    return out


def apply_rot_floors(blocks, floor=None, spread_bar=None):
    """Gate the rotation hold-out on the CAPTURE's own floors.

    Two quantiles of this capture's own per-frame readings, pooled over its
    rotation-only members: the p10 of the resection inlier fractions, and the
    p10 of the support spreads.  A frame under either is a non-measurement --
    its delta is reported, and it is not gate-eligible.  A caller measuring a
    subset of one member passes the capture's own two floors in, for the same
    reason :func:`apply_inlier_floor` takes one."""
    pool_inl = [
        f["inlier_frac"]
        for b in blocks
        for f in b.get("frames", [])
        if f.get("inlier_frac") is not None
    ]
    pool_spread = [
        f["support_spread_deg"]
        for b in blocks
        for f in b.get("frames", [])
        if f.get("support_spread_deg") is not None
    ]
    if floor is None:
        floor = float(np.percentile(pool_inl, 10)) if pool_inl else None
    if spread_bar is None:
        spread_bar = float(np.percentile(pool_spread, 10)) if pool_spread else None
    for b in blocks:
        b["inlier_floor"] = floor
        b["inlier_floor_source"] = "capture p10 of per-frame rotation-fit inlier"
        b["support_spread_bar_deg"] = spread_bar
        b["support_spread_bar_source"] = "capture p10 of per-frame support spread"
        gated = []
        frames = b.get("frames", [])
        for f in frames:
            elig = (
                f.get("status") == "ok"
                and f.get("inlier_frac") is not None
                and (floor is None or f["inlier_frac"] >= floor)
            )
            f["gate_eligible"] = bool(elig)
            if not elig:
                f["gate_blocked_by"] = (
                    "status" if f.get("status") != "ok" else "inlier_floor"
                )
            elif spread_bar is not None and (
                f.get("support_spread_deg") is None
                or f["support_spread_deg"] < spread_bar
            ):
                f["gate_eligible"] = False
                f["gate_blocked_by"] = "conditioning"
            if f["gate_eligible"]:
                gated.append(f)
        b["gated_n"] = len(gated)
        b["gated_lost_frac"] = (1.0 - len(gated) / len(frames)) if frames else None
        b["rot_worst_gated"], b["rot_worst_gated_frame"] = _extreme(
            gated, "rot_delta_deg"
        )
        b["dir_dev_worst_gated"], b["dir_dev_worst_gated_frame"] = _extreme(
            gated, "dir_dev_med_deg"
        )
    return floor


def rot_nonmember_channels(m, pair_obs):
    """Resect the capture's best-connected HELD-OUT images as ROTATIONS.

    The held-out image's rotation is fitted against the member's stored
    directions, and the result is checked against a two-view witness estimated
    from the pair's raw matches alone.  The witness is the ROTATION-ONLY
    homography -- a parallax-free pair simply is a rotation of unit rays -- and
    not an essential matrix: a model with no baseline has no epipolar geometry
    to estimate, and asking for one would fit noise."""
    from sfmtool._sfmtool.geometry import fit_ray_rotation

    images = []
    out = {"measurable": False, "images": images, "k_requested": K_HELD_OUT}
    if m.pts is None or not len(m.rows):
        out["unmeasurable_reason"] = "no_directions" if m.pts is None else "no_obs"
        out["n_measured"] = 0
        return out
    tol = _rot_tol(m)
    out["tol_rad"] = tol
    anchored = np.nonzero(m.finite)[0]
    out["n_anchor_clusters"] = int(len(anchored))
    on_anchor = m.finite[m.obs_c]
    rows_h = np.nonzero(on_anchor & ~m.posed[m.obs_i] & ~m.dropped[m.obs_i])[0]
    if not len(rows_h):
        out["unmeasurable_reason"] = "no_heldout_observations"
        out["n_measured"] = 0
        return out
    n_corr = np.bincount(m.obs_i[rows_h], minlength=m.n_img)
    seen = np.zeros((m.n_img, m.n_cl), dtype=bool)
    seen[m.obs_i[m.rows], m.obs_c[m.rows]] = True
    mem_seen = seen[m.frames]
    n_link = np.zeros(m.n_img, dtype=np.int64)
    ordr, bounds = _csr(m.obs_i[rows_h], m.n_img)
    rows_h = rows_h[ordr]
    avail = np.nonzero(n_corr >= MIN_CORR)[0]
    for g in avail:
        cl = m.obs_c[rows_h[bounds[g] : bounds[g + 1]]]
        n_link[g] = int(mem_seen[:, np.unique(cl)].any(axis=1).sum())
    out["n_heldout_available"] = int(len(avail))
    order = sorted(avail.tolist(), key=lambda g: (-int(n_link[g]), -int(n_corr[g]), g))[
        :K_HELD_OUT
    ]
    out["n_heldout_tried"] = len(order)
    for rank, g in enumerate(order):
        rows = rows_h[bounds[g] : bounds[g + 1]]
        rec = {
            "name": m.names[g],
            "rank": rank,
            "status": "ok",
            "n_link": int(n_link[g]),
            "n_corr": int(len(rows)),
        }
        cl = m.obs_c[rows]
        rays = _local_rays(m.camera, m.obs_uv[rows])
        dirs = m.pts[cl]
        live = np.isfinite(rays).all(axis=1) & np.isfinite(dirs).all(axis=1)
        if int(live.sum()) < MIN_CORR:
            rec["status"] = "few_correspondences"
            images.append(rec)
            continue
        rot_g, frac, n_inl, _ = _trimmed_rotation(dirs[live], rays[live], tol)
        rec["inlier_frac"], rec["n_inliers"] = frac, n_inl
        if rot_g is None:
            rec["status"] = "no_rotation_fit"
            images.append(rec)
            continue
        if n_inl < MIN_CORR:
            # A fit nothing consented to is not a reading of the member: the
            # held-out image simply cannot be explained as a turn of this
            # member's directions.  Reported, not read as a disagreement.
            rec["status"] = "no_rotation_consensus"
            images.append(rec)
            continue
        counts = [int(mem_seen[b][cl].sum()) for b in range(len(m.frames))]
        best = int(np.argmax(counts))
        jm = int(m.frames[best])
        rec["best_member"], rec["best_member_corr"] = m.names[jm], counts[best]
        # THE WITNESS: the same pair's relative rotation from raw matches, and
        # never from any direction the member placed.
        rec["e_status"] = "no_pair_source"
        if pair_obs is not None:
            cl_a, uv_a = pair_obs.rows_of(g)
            cl_b, uv_b = pair_obs.rows_of(jm)
            _, ia, ib = np.intersect1d(cl_a, cl_b, return_indices=True)
            rec["n_pair_matches"] = int(len(ia))
            if len(ia) < E_MIN_CORR:
                rec["e_status"] = "few_pair_matches"
            else:
                ph, pm = uv_a[ia], uv_b[ib]
                if len(ph) > E_MAX_CORR:
                    take = np.sort(
                        np.random.default_rng(SEED).choice(
                            len(ph), E_MAX_CORR, replace=False
                        )
                    )
                    ph, pm = ph[take], pm[take]
                x1, x2 = _local_rays(m.camera, ph), _local_rays(m.camera, pm)
                good = np.isfinite(x1).all(axis=1) & np.isfinite(x2).all(axis=1)
                x1, x2 = x1[good], x2[good]
                if len(x1) < E_MIN_CORR:
                    rec["e_status"] = "few_pair_matches"
                else:
                    fit = fit_ray_rotation(
                        np.ascontiguousarray(x1),
                        np.ascontiguousarray(x2),
                        max_angle_rad=tol,
                        min_inliers=E_MIN_CORR,
                        samples=E_SAMPLES,
                        seed=SEED,
                    )
                    if fit is None:
                        rec["e_status"] = "no_rotation_consensus"
                    else:
                        inl = np.asarray(fit["inliers"], dtype=bool)
                        rot_2v = np.asarray(fit["rotation"], dtype=np.float64)
                        rec["e_inlier_frac"] = float(inl.mean())
                        rec["e_n_inliers"] = int(inl.sum())
                        rec["e_rms_px"] = float(fit["rms_rad"]) * (m.f_eq or 1.0)
                        rel = m.rot[jm] @ rot_g.T
                        rec["rot_delta_deg"] = float(
                            np.degrees(Rotation.from_matrix(rot_2v @ rel.T).magnitude())
                        )
                        rec["e_status"] = "ok"
        images.append(rec)
    ok = [r for r in images if r["status"] == "ok"]
    wit = [r for r in ok if r.get("e_status") == "ok"]
    out["n_measured"] = len(ok)
    out["n_unmeasurable"] = K_HELD_OUT - len(ok)
    out["coverage"] = len(ok) / K_HELD_OUT
    out["n_witnessed"] = len(wit)
    if not ok:
        out["unmeasurable_reason"] = "no_resectable_heldout_image"
        return out
    out["inlier_med"] = _q([r.get("inlier_frac") for r in ok], 50)
    out["inlier_worst"] = _q([r.get("inlier_frac") for r in ok], 0)
    out["ncorr_med"] = _q([r.get("n_corr") for r in ok], 50)
    if wit:
        out["rot_med"] = _q([r.get("rot_delta_deg") for r in wit], 50)
        out["rot_worst"], out["rot_worst_image"] = _extreme(wit, "rot_delta_deg")
        out["e_inlier_med"] = _q([r.get("e_inlier_frac") for r in wit], 50)
    out["measurable"] = True
    return out


def rot_settling_channels(m):
    """Let the rotation-only member settle: rotations free, directions free.

    One alternating pass: each direction is the mean of its own rotated rays,
    each rotation is refitted against the directions.  There is no scale and no
    baseline to trade, so the probe is closed-form at every step and needs no
    solver.  Read gauge-free, as the between-frame relative rotation change."""
    out = {"measurable": False, "rounds": ROT_SETTLE_ROUNDS}
    if m.pts is None or len(m.frames) < 2 or not len(m.rows):
        out["unmeasurable_reason"] = "no_directions" if m.pts is None else "too_small"
        return out
    rows = m.rows
    tol = _rot_tol(m)
    loc = _local_rays(m.camera, m.obs_uv[rows])
    live = np.isfinite(loc).all(axis=1)
    rows, loc = rows[live], loc[live]
    if len(rows) < ROT_MIN_SUPPORT:
        out["unmeasurable_reason"] = "too_few_observations"
        return out
    img, cl = m.obs_i[rows], m.obs_c[rows]
    rot = m.rot.copy()
    dirs = np.array(m.pts, dtype=np.float64, copy=True)

    # THE EARNED ROWS.  Each cluster's direction is the ray of its first
    # observation, so that row's residual is zero by construction; a median
    # over a set half made of them measures the construction, not the fit.
    earned = np.ones(len(rows), dtype=bool)
    earned[np.unique(cl, return_index=True)[1]] = False
    out["n_obs"] = int(len(rows))
    out["n_obs_earned"] = int(earned.sum())

    def _residual_deg(rot_now, dirs_now):
        """Angles between each observation's ray and the direction the model
        transports into that frame, over the rows the model earned."""
        ang = _rot_residuals(
            np.eye(3), np.einsum("nij,nj->ni", rot_now[img], dirs_now[cl]), loc
        )
        ang = ang[earned & np.isfinite(ang)]
        return float(np.degrees(np.median(ang))) if len(ang) else None

    out["residual_med_before_deg"] = _residual_deg(rot, dirs)
    for _round in range(ROT_SETTLE_ROUNDS):
        world = np.einsum("nji,nj->ni", rot[img], loc)
        acc = np.zeros_like(dirs)
        np.add.at(acc, cl, world)
        n = np.linalg.norm(acc, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            upd = acc / np.where(n > 0, n, np.nan)
        dirs = np.where(np.isfinite(upd), upd, dirs)
        for j in m.frames:
            sel = img == j
            if int(sel.sum()) < 3:
                continue
            d = dirs[cl[sel]]
            good = np.isfinite(d).all(axis=1)
            if int(good.sum()) < 3:
                continue
            fit, _f, _n, _a = _trimmed_rotation(d[good], loc[sel][good], tol)
            if fit is not None:
                rot[j] = fit
    out["residual_med_after_deg"] = _residual_deg(rot, dirs)
    tol_deg = math.degrees(tol)
    out["tol_deg"] = tol_deg
    if out["residual_med_before_deg"] and out["residual_med_after_deg"]:
        out["residual_ratio"] = (
            out["residual_med_after_deg"] / out["residual_med_before_deg"]
        )
        # A refit that grew a residual still far inside the member's own
        # consensus bound has not diverged; it has moved within the noise.
        out["diverged"] = bool(
            out["residual_ratio"] > DIVERGE_RATIO
            and out["residual_med_after_deg"] > tol_deg
        )
    frames = m.frames
    pairs = _pose_pairs(len(frames))
    rel_a = np.einsum(
        "nij,nkj->nik", m.rot[frames][pairs[:, 1]], m.rot[frames][pairs[:, 0]]
    )
    rel_b = np.einsum(
        "nij,nkj->nik", rot[frames][pairs[:, 1]], rot[frames][pairs[:, 0]]
    )
    deg = np.degrees(
        Rotation.from_matrix(np.einsum("nji,njk->nik", rel_a, rel_b)).magnitude()
    )
    out["n_pairs"] = int(len(pairs))
    _agg(deg.tolist(), "rot", out)
    per_frame = []
    for k, j in enumerate(frames):
        touch = (pairs[:, 0] == k) | (pairs[:, 1] == k)
        vals = deg[touch]
        vals = vals[np.isfinite(vals)]
        per_frame.append(
            {
                "name": m.names[j],
                "rot_med": float(np.median(vals)) if len(vals) else None,
            }
        )
    out["frames"] = per_frame
    out["rot_worst_frame"] = _extreme(per_frame, "rot_med")[1]
    out["measurable"] = True
    return out


def rot_warp_channels(m):
    """The stored affine shapes against a pure rotation, in FULL form.

    Under a rotation-only model the pairwise image map of a track carries no
    surface term: the two views' depth ratio is one and there is no
    tangent-plane normal to absorb anything, so the relative rotation and the
    camera model predict ALL FOUR numbers of the measured warp.  The residual
    is the whole departure from that prediction, not the two-number
    projection a finite member is limited to."""
    out = {"measurable": False, "frames": []}
    if m.pts is None or not len(m.rows):
        out["unmeasurable_reason"] = "no_directions" if m.pts is None else "no_obs"
        return out
    if m.obs_shape is None:
        out["unmeasurable_reason"] = "no_member_shapes"
        return out
    rows = m.rows
    shp = m.obs_shape[rows]
    det = shp[:, 0, 0] * shp[:, 1, 1] - shp[:, 0, 1] * shp[:, 1, 0]
    usable = np.isfinite(shp).all(axis=(1, 2)) & (np.abs(det) > 0)
    out["n_obs"] = int(len(rows))
    out["shape_frac"] = float(usable.mean()) if len(rows) else None
    idx = rows[usable]
    if len(idx) < 2 * WARP_MIN_PAIRS:
        out["unmeasurable_reason"] = "few_shaped_observations"
        return out
    ii, jj, pt = _track_pairs(m.obs_c[idx], idx)
    out["n_pairs"] = int(len(ii))
    out["n_points"] = int(len(np.unique(pt))) if len(pt) else 0
    if len(ii) < WARP_MIN_PAIRS:
        out["unmeasurable_reason"] = "few_pairs"
        return out
    n_rows = len(m.obs_uv)
    basis = np.full((n_rows, 3, 2), np.nan)
    chart_p = np.full((n_rows, 2, 2), np.nan)
    for j in m.frames:
        sel = idx[m.obs_i[idx] == j]
        if not len(sel):
            continue
        _u, basis[sel], chart_p[sel], _r, _j = _angular_chart(m.camera, m.obs_uv[sel])
    warp = np.einsum("nab,nbc->nac", m.obs_shape[jj], _inv2(m.obs_shape[ii])[0])
    img_i, img_j = m.obs_i[ii], m.obs_i[jj]
    rel = np.einsum("nab,ncb->nac", m.rot[img_j], m.rot[img_i])
    w0 = np.einsum("nab,nac->nbc", basis[jj], np.einsum("nab,nbc->nac", rel, basis[ii]))
    pi_inv, _ = _inv2(chart_p[ii])
    atil = np.einsum(
        "nab,nbc->nac", chart_p[jj], np.einsum("nab,nbc->nac", warp, pi_inv)
    )
    n = len(ii)
    w0n = np.linalg.norm(w0.reshape(n, 4), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        full = np.linalg.norm((atil - w0).reshape(n, 4), axis=1) / np.where(
            w0n > 0, w0n, np.nan
        )
    good = np.isfinite(full)
    out["n_pairs_finite"] = int(good.sum())
    if int(good.sum()) < WARP_MIN_PAIRS:
        out["unmeasurable_reason"] = "few_finite_pairs"
        return out
    full, img_i, img_j = full[good], img_i[good], img_j[good]
    out["full_med"] = _q(full, 50)
    out["full_p90"] = _q(full, 90)
    out["full_p99"] = _q(full, 99)
    side_img = np.concatenate([img_i, img_j])
    side_val = np.concatenate([full, full])
    order, bnd = _csr(side_img, m.n_img)
    for j in m.frames:
        take = order[bnd[j] : bnd[j + 1]]
        rec = {"name": m.names[j], "n_pairs": int(len(take))}
        if len(take):
            rec["full_med"] = _q(side_val[take], 50)
            rec["full_p90"] = _q(side_val[take], 90)
        out["frames"].append(rec)
    out["full_frame_med"] = _q([f.get("full_med") for f in out["frames"]], 50)
    out["full_frame_worst"], out["full_worst_frame"] = _extreme(
        out["frames"], "full_med"
    )
    out["measurable"] = True
    return out


def _parallax_points(m, rows, mag, out):
    """Per-point parallax residue, in the member's own admission bounds.

    Writes the census into ``out`` and returns ``(measured, bearing)``, two
    masks over the member's cluster ids."""
    measured = np.zeros(m.n_cl, bool)
    bearing = np.zeros(m.n_cl, bool)
    keep = np.zeros(len(m.obs_c), bool)
    keep[m.rows] = True
    inside = mag[keep[rows] & np.isfinite(mag)]
    # THE MEMBER'S OWN BOUND: the largest residual its model kept.  Taken from
    # the member rather than assumed, so the reading is in the units the
    # member itself admitted at.
    bound = float(inside.max()) if inside.size else None
    out["bound_px"] = bound
    out["bound_source"] = "the largest residual the member's own model kept"
    if bound is None or not bound > 0:
        out["unmeasurable_point_reason"] = "no_admission_bound"
        return measured, bearing
    cl = m.obs_c[rows]
    order = np.argsort(cl, kind="stable")
    cls, mgs = cl[order], mag[order]
    cuts = np.flatnonzero(np.diff(cls)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(cls)]))
    ids, res, nobs = [], [], []
    for lo, hi in zip(starts, ends):
        v = mgs[lo:hi]
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        ids.append(int(cls[lo]))
        res.append(float(np.median(v)) / bound)
        nobs.append(int(len(v)))
    if not ids:
        out["unmeasurable_point_reason"] = "no_finite_residual"
        return measured, bearing
    ids_a = np.asarray(ids, np.int64)
    res_a = np.asarray(res, float)
    measured[ids_a] = True
    bearing[ids_a] = res_a > 1.0
    n_bear = int((res_a > 1.0).sum())
    frac = n_bear / len(ids_a)
    out["n_points_measured"] = int(len(ids_a))
    out["n_points_parallax"] = n_bear
    out["frac_points_parallax"] = float(frac)
    out["point_residue_med"] = _q(res_a, 50)
    out["point_residue_p90"] = _q(res_a, 90)
    out["point_residue_p99"] = _q(res_a, 99)
    out["point_residue_max"] = float(res_a.max())
    out["parallax_class"] = (
        "none"
        if n_bear == 0
        else ("majority" if frac >= ROT_PARALLAX_MAJORITY else "some")
    )
    take = np.argsort(-res_a, kind="stable")[:ROT_PARALLAX_MAX_LISTED]
    take = take[res_a[take] > 1.0]
    out["points_source"] = "the member's own cluster ids"
    out["points_listed"] = int(len(take))
    out["points"] = [
        {
            "point": int(ids_a[i]),
            "residue_bounds": float(res_a[i]),
            "n_obs": int(nobs[i]),
        }
        for i in take
    ]
    return measured, bearing


def parallax_residue_channels(m):
    """Which of a rotation-only member's points carry parallax, and how much.

    A point at infinity is an approximation, not a claim: it holds over a
    narrow-parallax subset and stops holding over a wider one, and a point
    that stops holding is a point to be GRADUATED to a finite depth, not a
    defect of the member.  A rotation-only member is valued for its
    ORIENTATIONS, so this channel measures the departure and never accuses:
    the readings here are graduation evidence.

    Per point, over the frames that observe it in the member's whole
    admission: the median residual against the rotation-only prediction, in
    units of the member's own admission bound -- the largest residual its
    model kept.  A point past one bound is parallax-bearing.  The census (how
    many, what share, which ones) ships with the member.

    The residual FIELD's shape is read beside it, per frame: a camera that
    travelled leaves residual vectors pointing along the line from the frame's
    epipole through the observation, so fitting that epipole and reading how
    radial the field is around it says whether the departure is baseline or
    noise.  Isotropic noise reads at the null level.

    The frame's whole admission is read, not the member's own inlier set: the
    observations the rotation model REFUSED are the near objects, and they are
    exactly what this channel is looking for."""
    out = {"measurable": False, "frames": [], "radial_null": ROT_RADIAL_NULL}
    if m.pts is None or not len(m.rows_all):
        out["unmeasurable_reason"] = "no_directions" if m.pts is None else "no_obs"
        return out
    rows = m.rows_all
    img = m.obs_i[rows]
    x_cam = np.einsum("nij,nj->ni", m.rot[img], m.pts[m.obs_c[rows]])
    proj = np.asarray(m.camera.ray_to_pixel_batch(np.ascontiguousarray(x_cam)))
    resid = proj - m.obs_uv[rows]
    mag = np.linalg.norm(resid, axis=1)
    pt_measured, pt_bearing = _parallax_points(m, rows, mag, out)
    order, bnd = _csr(img, m.n_img)
    for j in m.frames:
        take = order[bnd[j] : bnd[j + 1]]
        rec = {"name": m.names[j], "n_obs": int(len(take))}
        seen = np.unique(m.obs_c[rows[take]])
        seen = seen[pt_measured[seen]]
        rec["n_points_measured"] = int(len(seen))
        if len(seen):
            rec["n_points_parallax"] = int(pt_bearing[seen].sum())
            rec["frac_points_parallax"] = float(pt_bearing[seen].mean())
        r, x, mg = resid[take], m.obs_uv[rows[take]], mag[take]
        fin = np.isfinite(mg)
        rec["n_finite"] = int(fin.sum())
        if int(fin.sum()) >= 1:
            rec["res_med_px"] = float(np.median(mg[fin]))
            rec["res_p90_px"] = float(np.percentile(mg[fin], 90))
            # The share of the frame's admission the rotation model refused:
            # by construction those are the observations with parallax.
            rec["rejected_frac"] = float((mg[fin] >= 2.0).mean())
        if int(fin.sum()) < ROT_MIN_RESIDUE_OBS:
            rec["status"] = "few_observations"
            out["frames"].append(rec)
            continue
        r, x, mg = r[fin], x[fin], mg[fin]
        bar = float(np.percentile(mg, ROT_RESIDUE_QUANTILE))
        sel = mg > bar
        if int(sel.sum()) < ROT_MIN_RESIDUE_OBS:
            rec["status"] = "no_directed_residual"
            out["frames"].append(rec)
            continue
        rs, xs = r[sel], x[sel]
        # The epipole: the point every residual line passes through, in the
        # least-squares sense.  `cross(r, x - e) = 0` is linear in `e`.
        a = np.stack([rs[:, 1], -rs[:, 0]], axis=1)
        b = rs[:, 1] * xs[:, 0] - rs[:, 0] * xs[:, 1]
        try:
            epi, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            rec["status"] = "no_epipole"
            out["frames"].append(rec)
            continue
        radial = xs - epi
        nr = np.linalg.norm(radial, axis=1)
        nrs = np.linalg.norm(rs, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.abs(np.einsum("ij,ij->i", radial, rs)) / (nr * nrs)
        cos = cos[np.isfinite(cos)]
        if not len(cos):
            rec["status"] = "no_epipole"
            out["frames"].append(rec)
            continue
        rec["status"] = "ok"
        rec["epipole"] = [float(epi[0]), float(epi[1])]
        rec["radial_align"] = float(np.median(np.clip(cos, 0.0, 1.0)))
        rec["n_directed"] = int(sel.sum())
        cov = np.cov(rs.T)
        ev = np.linalg.eigvalsh(cov)
        rec["residual_aniso"] = float(math.sqrt(ev[1] / ev[0])) if ev[0] > 0 else None
        out["frames"].append(rec)
    ok = [f for f in out["frames"] if f.get("status") == "ok"]
    out["n_frames"] = len(out["frames"])
    out["n_measured"] = len(ok)
    out["rejected_frac_med"] = _q([f.get("rejected_frac") for f in out["frames"]], 50)
    out["res_med_px"] = _q([f.get("res_med_px") for f in out["frames"]], 50)
    if not ok:
        out["unmeasurable_reason"] = "no_frame_with_directed_residual"
        return out
    out["radial_align_med"] = _q([f.get("radial_align") for f in ok], 50)
    out["radial_align_p90"] = _q([f.get("radial_align") for f in ok], 90)
    out["radial_align_worst"], out["radial_align_worst_frame"] = _extreme(
        ok, "radial_align"
    )
    out["aniso_med"] = _q([f.get("residual_aniso") for f in ok], 50)
    out["measurable"] = True
    return out


def _frame_rays(m):
    """Per posed frame, `(cluster ids, unit camera rays, rows)` of its own rows.

    One row per (frame, cluster): a feature claimed twice in one frame would
    make the pair's correspondence ambiguous, so the first claim stands.
    Memoized on the member: two channels ask the same question."""
    held = getattr(m, "_frame_rays_memo", None)
    if held is not None:
        return held
    out = {}
    for j in m.frames:
        rows = m.frame_rows(j)
        if not len(rows):
            continue
        cl = m.obs_c[rows]
        order = np.argsort(cl, kind="stable")
        cl, rows = cl[order], rows[order]
        uniq, first = np.unique(cl, return_index=True)
        rays = _local_rays(m.camera, m.obs_uv[rows[first]])
        out[int(j)] = (uniq, rays, rows[first])
    m._frame_rays_memo = out
    return out


def _covisibility(m, per_frame, floor):
    """`{(i, j): n_shared}` over posed frame pairs above a shared-point floor.

    Memoized on the member per floor: two channels ask the same question."""
    memo = getattr(m, "_covis_memo", None)
    if memo is None:
        memo = m._covis_memo = {}
    if floor in memo:
        return memo[floor]
    ids = sorted(per_frame)
    if len(ids) < 3:
        return memo.setdefault(floor, {})
    table = np.zeros((len(ids), m.n_cl), bool)
    for k, j in enumerate(ids):
        table[k, per_frame[j][0]] = True
    counts = table.astype(np.int32) @ table.astype(np.int32).T
    out = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            n = int(counts[a, b])
            if n >= floor:
                out[(ids[a], ids[b])] = n
    memo[floor] = out
    return out


def _pair_rotation(m, per_frame, i, j, tol):
    """The relative rotation `R_ij` MEASURED from the pair's own shared rays.

    Not the member's `R_j R_i^T`: a composition of those is the identity by
    construction and says nothing.  This is a trimmed Kabsch fit of the two
    frames' unit rays over the points they share, so a cycle of them closes
    only if the pairwise measurements agree with each other."""
    ci, ri, _ = per_frame[i]
    cj, rj, _ = per_frame[j]
    shared, ii, jj = np.intersect1d(ci, cj, assume_unique=True, return_indices=True)
    if len(shared) < ROT_CYCLE_MIN_SHARED:
        return None, 0, None
    a, b = ri[ii], rj[jj]
    good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[good], b[good]
    if len(a) < ROT_CYCLE_MIN_SHARED:
        return None, 0, None
    rot, inl, _n, _ang = _trimmed_rotation(a, b, tol)
    return rot, len(a), inl


def rot_cycle_channels(m):
    """Do the member's PAIRWISE relative rotations close around a cycle?

    Every edge of the member's covisibility graph carries a relative rotation
    measured from that pair's shared rays alone.  Under a pure rotation those
    measurements are one consistent field, so composing them around any closed
    walk returns the identity; parallax, a mis-associated track or a frame
    fitted to the wrong points break the closure.  The reading is the residual
    angle of the composition, and it is gauge-free and internal: no referee,
    no absolute pose, and nothing outside the member.

    The cycle set is a spanning tree's fundamental basis -- every non-tree edge
    plus the tree path that closes it -- which spans the graph's whole cycle
    space.  The longest such cycle is always read, because a long walk
    accumulates what a triangle can hide."""
    out = {"measurable": False, "frames": []}
    if m.pts is None or not len(m.rows):
        out["unmeasurable_reason"] = "no_directions" if m.pts is None else "no_obs"
        return out
    per_frame = _frame_rays(m)
    out["min_shared"] = ROT_CYCLE_MIN_SHARED
    edges = _covisibility(m, per_frame, ROT_CYCLE_MIN_SHARED)
    out["n_frames"] = len(per_frame)
    out["n_edges"] = len(edges)
    if len(edges) < 3:
        out["unmeasurable_reason"] = "covisibility graph carries no cycle"
        return out
    # The spanning tree, grown from the best-connected frame over the
    # best-shared edges first, so the tree paths are the trustworthy ones.
    adj = {}
    for (a, b), n in edges.items():
        adj.setdefault(a, []).append((n, b))
        adj.setdefault(b, []).append((n, a))
    for v in adj.values():
        v.sort(key=lambda kv: -kv[0])
    root = max(adj, key=lambda k: len(adj[k]))
    parent, depth, seen = {root: None}, {root: 0}, {root}
    stack = [root]
    while stack:
        u = stack.pop(0)
        for _n, v in adj[u]:
            if v not in seen:
                seen.add(v)
                parent[v], depth[v] = u, depth[u] + 1
                stack.append(v)
    tree = {tuple(sorted((v, p))) for v, p in parent.items() if p is not None}
    extra = [(n, e) for e, n in edges.items() if e not in tree and e[0] in seen]

    def path_len(a, b):
        pa, pb = a, b
        da, db = depth.get(a, 0), depth.get(b, 0)
        steps = 0
        while da > db:
            pa, da, steps = parent[pa], da - 1, steps + 1
        while db > da:
            pb, db, steps = parent[pb], db - 1, steps + 1
        while pa != pb:
            pa, pb, steps = parent[pa], parent[pb], steps + 2
        return steps

    extra.sort(key=lambda kv: -kv[0])
    chosen = extra[:ROT_CYCLE_MAX_CYCLES]
    if extra:
        longest = max(extra, key=lambda kv: path_len(*kv[1]))
        if longest not in chosen:
            chosen.append(longest)
    tol = _rot_tol(m)
    cache = {}

    def rot_of(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            cache[key] = _pair_rotation(m, per_frame, key[0], key[1], tol)
        rot, n, inl = cache[key]
        if rot is None:
            return None
        return rot if (a, b) == key else rot.T

    def walk(a, b):
        """The frames of the tree path from `a` up to `b`'s side, in order."""
        pa, pb, up_a, up_b = a, b, [], []
        while depth[pa] > depth[pb]:
            up_a.append(pa)
            pa = parent[pa]
        while depth[pb] > depth[pa]:
            up_b.append(pb)
            pb = parent[pb]
        while pa != pb:
            up_a.append(pa)
            up_b.append(pb)
            pa, pb = parent[pa], parent[pb]
        return up_a + [pa] + up_b[::-1]

    cycles, per_frame_worst = [], {}
    for _n, (a, b) in chosen:
        nodes = walk(a, b)
        if len(nodes) < 3:
            continue
        loop = nodes + [nodes[0]]
        prod = np.eye(3)
        ok = True
        for u, v in zip(loop[:-1], loop[1:]):
            r = rot_of(u, v)
            if r is None:
                ok = False
                break
            prod = r @ prod
        if not ok:
            continue
        cos = (np.trace(prod) - 1.0) / 2.0
        res = float(np.degrees(math.acos(max(-1.0, min(1.0, cos)))))
        cycles.append({"length": len(nodes), "residual_deg": res})
        for f in nodes:
            per_frame_worst[f] = max(per_frame_worst.get(f, 0.0), res)
    if not cycles:
        out["unmeasurable_reason"] = "no cycle carried a measurable edge"
        return out
    res = [c["residual_deg"] for c in cycles]
    out["n_cycles"] = len(cycles)
    out["n_edges_fitted"] = sum(1 for v in cache.values() if v[0] is not None)
    out["res_med"] = _q(res, 50)
    out["res_p90"] = _q(res, 90)
    out["res_worst"] = float(max(res))
    out["cycle_len_med"] = _q([c["length"] for c in cycles], 50)
    big = max(cycles, key=lambda c: c["length"])
    out["largest_cycle_len"] = big["length"]
    out["largest_cycle_res_deg"] = big["residual_deg"]
    for j in m.frames:
        rec = {"name": m.names[j]}
        if int(j) in per_frame_worst:
            rec["cycle_worst_deg"] = float(per_frame_worst[int(j)])
        out["frames"].append(rec)
    out["res_frame_worst"], out["res_worst_frame"] = _extreme(
        out["frames"], "cycle_worst_deg"
    )
    out["measurable"] = True
    return out


def _zncc(a, b):
    """Zero-mean normalized cross-correlation of two sample sets, or NaN."""
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < ROT_PHOTO_MIN_SAMPLES:
        return float("nan")
    x, y = a[good], b[good]
    x = x - x.mean()
    y = y - y.mean()
    nx, ny = math.sqrt(float(x @ x)), math.sqrt(float(y @ y))
    if nx <= 0 or ny <= 0:
        return float("nan")
    return float((x @ y) / (nx * ny))


class ImageCache:
    """A few of the capture's images as grayscale float, by relative path."""

    def __init__(self, loader, size=None):
        self.loader = loader
        self.size = int(size or ROT_PHOTO_CACHE)
        self._held = {}
        self._order = []

    def get(self, name):
        if name in self._held:
            return self._held[name]
        img = None
        try:
            img = self.loader(name)
        except Exception:  # noqa: BLE001 — a missing image is a non-measurement
            img = None
        while len(self._order) >= self.size:
            self._held.pop(self._order.pop(0), None)
        self._held[name] = img
        self._order.append(name)
        return img


def _patch_radius(shape):
    """The stored affine shape's own extent in pixels.

    `S` maps the detector's canonical unit frame onto this image's pixels, so
    the patch's half-width is that map's largest singular value.  The window
    is the feature's own, not a constant."""
    if shape is None or not np.isfinite(shape).all():
        return None
    sv = np.linalg.svd(np.asarray(shape, float).reshape(2, 2), compute_uv=False)
    r = float(sv[0])
    if not math.isfinite(r) or r <= 0:
        return None
    return float(min(max(r, ROT_PHOTO_MIN_RADIUS_PX), ROT_PHOTO_MAX_RADIUS_PX))


def rot_warp_photometric_channels(m, images):
    """The EXACT photometric witness of a pure rotation.

    Under a rotation the pairwise image map carries no surface term at all:
    a pixel in `i` unprojects to a ray, the relative rotation turns it, and the
    camera projects it into `j`.  So the map is fully predicted, and it can be
    checked against the images themselves rather than against a fitted warp.

    At each stored keypoint of a shared point, a window whose extent is that
    observation's own affine shape is laid over image `i`, carried into image
    `j` through the member's camera model and its relative rotation, and the
    two samplings are compared by ZNCC.  The window is anchored at the STORED
    keypoint, never at a reprojection: the question is whether the model
    explains the content the member actually matched.

    A sample that leaves either image, a mapped ray that leaves the camera's
    field, and a window with no contrast in either view are non-measurements,
    counted with their reason.  The reading is the photometric DISAGREEMENT,
    `1 - ZNCC`, so it grows with the defect like every other residual here."""
    out = {"measurable": False, "frames": [], "grid": ROT_PHOTO_GRID}
    if images is None:
        out["unmeasurable_reason"] = "no_image_source"
        return out
    if m.pts is None or not len(m.rows) or m.obs_shape is None:
        out["unmeasurable_reason"] = (
            "no_directions"
            if m.pts is None
            else ("no_obs" if not len(m.rows) else "no_affine_shapes")
        )
        return out
    from scipy.ndimage import map_coordinates

    per_frame = _frame_rays(m)
    ids = sorted(per_frame)
    if len(ids) < 2:
        out["unmeasurable_reason"] = "one_posed_frame"
        return out
    edges = _covisibility(m, per_frame, ROT_PHOTO_MIN_SHARED)
    if not edges:
        out["unmeasurable_reason"] = "no_pair_shares_enough_points"
        return out
    # A BOUNDED, EVENLY SPREAD set of the pairs that actually share points:
    # the reading is a distribution over the member, not a census of it, and
    # every pair costs two image decodes.  Spread over the member's own frame
    # order, then grouped by first frame so a decoded image is reused.
    keys = sorted(edges)
    pick = np.unique(
        np.round(
            np.linspace(0, len(keys) - 1, min(len(keys), ROT_PHOTO_MAX_PAIRS))
        ).astype(int)
    )
    order = sorted((keys[k] for k in pick), key=lambda e: (e[0], e[1]))
    sub = per_frame
    step = np.linspace(-1.0, 1.0, ROT_PHOTO_GRID)
    gx, gy = np.meshgrid(step, step)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    census = {"pairs": 0, "points": 0, "off_image": 0, "off_field": 0, "flat": 0}
    per_pair, by_frame = [], {}
    for i, j in order:
        img_i, img_j = images.get(m.names[i]), images.get(m.names[j])
        if img_i is None or img_j is None:
            continue
        ci, _ri, rows_i = sub[i]
        cj, _rj, rows_j = sub[j]
        shared, ii, jj = np.intersect1d(ci, cj, assume_unique=True, return_indices=True)
        if not len(shared):
            continue
        take = np.linspace(0, len(shared) - 1, min(len(shared), ROT_PHOTO_MAX_POINTS))
        take = np.unique(np.round(take).astype(int))
        rel = m.rot[j] @ m.rot[i].T
        vals = []
        for k in take:
            row_i = rows_i[ii[k]]
            rad = _patch_radius(m.obs_shape[row_i])
            if rad is None:
                census["flat"] += 1
                continue
            uv_i = m.obs_uv[row_i] + grid * rad
            ray = np.asarray(m.camera.pixel_to_ray_batch(np.ascontiguousarray(uv_i)))
            turned = ray @ rel.T
            uv_j = np.asarray(m.camera.ray_to_pixel_batch(np.ascontiguousarray(turned)))
            if not np.isfinite(uv_j).all():
                census["off_field"] += 1
                continue
            pi = map_coordinates(
                img_i,
                [uv_i[:, 1], uv_i[:, 0]],
                order=1,
                mode="constant",
                cval=np.nan,
            )
            pj = map_coordinates(
                img_j,
                [uv_j[:, 1], uv_j[:, 0]],
                order=1,
                mode="constant",
                cval=np.nan,
            )
            if not np.isfinite(pi).all() or not np.isfinite(pj).all():
                census["off_image"] += 1
                continue
            z = _zncc(pi, pj)
            if not math.isfinite(z):
                census["flat"] += 1
                continue
            vals.append(1.0 - z)
            census["points"] += 1
        if not vals:
            continue
        census["pairs"] += 1
        rec = {
            "image_a": m.names[i],
            "image_b": m.names[j],
            "n_points": len(vals),
            "res_med": _q(vals, 50),
            "res_p90": _q(vals, 90),
            "zncc_med": 1.0 - _q(vals, 50),
            "zncc_p10": 1.0 - _q(vals, 90),
        }
        per_pair.append(rec)
        for f in (i, j):
            by_frame.setdefault(f, []).extend(vals)
    out["census"] = census
    if not per_pair:
        out["unmeasurable_reason"] = "no_pair_produced_a_measurable_window"
        return out
    allv = [v for vals in by_frame.values() for v in vals]
    out["n_pairs"] = len(per_pair)
    out["n_points"] = census["points"]
    out["res_med"] = _q(allv, 50)
    out["res_p90"] = _q(allv, 90)
    out["zncc_med"] = 1.0 - _q(allv, 50)
    out["zncc_p10"] = 1.0 - _q(allv, 90)
    out["pairs"] = per_pair
    for j in m.frames:
        rec = {"name": m.names[j]}
        vals = by_frame.get(int(j))
        if vals:
            rec["n_points"] = len(vals)
            rec["res_med"] = _q(vals, 50)
            rec["zncc_med"] = 1.0 - _q(vals, 50)
        out["frames"].append(rec)
    out["res_frame_med"] = _q([f.get("res_med") for f in out["frames"]], 50)
    out["res_frame_worst"], out["res_worst_frame"] = _extreme(out["frames"], "res_med")
    out["measurable"] = True
    return out


def rot_support_channels(m):
    """Per-frame observation counts, for a member with no range.

    A frame the member posed on almost nothing is not a frame the member
    measured: whatever its rotation reads, it reads on too little to be a
    reading.  So the counts are aggregated into a member channel of their
    own -- the most starved frame, and its STARVATION, that frame's count
    against the member's own median frame, which is a ratio inside one member
    and so carries no capture scale.

    The depth-scale coherence and near-support readings have no rotation-only
    form: every direction sits at the same unit range by construction, so
    there is nothing for them to compare."""
    out = {
        "measurable": False,
        "unmeasurable_reason": "no_finite_structure",
        "frames": [],
    }
    if m.pts is None:
        return out
    for j in m.frames:
        rows = m.frame_rows(j)
        rec = {"name": m.names[j], "n_obs": int(len(rows))}
        rec["n_obs_admitted"] = int((m.obs_i[m.rows_all] == j).sum())
        if rec["n_obs_admitted"]:
            rec["explained_frac"] = rec["n_obs"] / rec["n_obs_admitted"]
        out["frames"].append(rec)
    counts = [f["n_obs"] for f in out["frames"]]
    if not counts:
        out["unmeasurable_reason"] = "no_posed_frame"
        return out
    med = float(np.median(counts))
    out.pop("unmeasurable_reason", None)
    out["measurable"] = True
    out["n_frames"] = len(counts)
    out["obs_med"] = med
    out["obs_min"], out["obs_min_frame"] = _extreme(
        out["frames"], "n_obs", worst_is_max=False
    )
    # A frame holding NOTHING is at least as starved as one holding a single
    # observation, so its deficit is read at one rather than left unmeasured:
    # the most starved frame of all must not be the one the ratio cannot see.
    for f in out["frames"]:
        f["obs_deficit"] = med / max(f["n_obs"], 1)
    out["obs_deficit_worst"], out["obs_deficit_worst_frame"] = _extreme(
        out["frames"], "obs_deficit"
    )
    return out


# ── Channel: focal-vote consistency ─────────────────────────────────────────


def focal_vote_channel(f_eq, f_vote):
    """The member's released equivalent focal against the capture's vote.

    Reported as a signed fraction and never gated here: the vote is a
    structure-free reading and the release is a structural one, and which of
    them is wrong is a selection-pass question."""
    out = {"f_released_eq": None if f_eq is None else float(f_eq), "f_vote": None}
    if f_vote is None or not math.isfinite(float(f_vote)) or float(f_vote) <= 0:
        out["measurable"] = False
        out["unmeasurable_reason"] = "no_capture_vote"
        return out
    out["f_vote"] = float(f_vote)
    if f_eq is None or not math.isfinite(float(f_eq)):
        out["measurable"] = False
        out["unmeasurable_reason"] = "no_released_focal"
        return out
    out["signed_fraction"] = (float(f_eq) - float(f_vote)) / float(f_vote)
    out["abs_fraction"] = abs(out["signed_fraction"])
    out["measurable"] = True
    return out


# ── The battery ─────────────────────────────────────────────────────────────


def _guard_none(fn, *args):
    """Build a helper, or None when building it fails."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 — instrumentation never kills the run
        print(f"  [candidate evaluation: {fn.__name__} failed: {exc}]")
        return None


def _guard(fn, *args):
    """Run one channel, turning a failure into a recorded non-measurement.

    Evaluation is instrumentation: it may not kill the run it instruments, and
    a channel that raised is an unmeasurable reading with a reason, which is
    exactly what the battery is required to report."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 — see above
        return {
            "measurable": False,
            "unmeasurable_reason": f"{type(exc).__name__}: {exc}",
        }


#: Every channel that asks a question only a member with depth can answer.  A
#: rotation-only layer records all of them as unmeasurable, with the reason.
GEOMETRY_CHANNELS = (
    "fit",
    "support",
    "self_resection",
    "nonmember_resection",
    "settling",
    "warp_epipolar",
    "stranger_surface",
    "surface_variation",
    "range_vetted_surface",
)

#: The channels only a member with DEPTH can answer.  A rotation-only member
#: records these unmeasurable and answers the rotation-only family instead.
FINITE_ONLY_CHANNELS = (
    "self_resection",
    "nonmember_resection",
    "settling",
    "warp_epipolar",
    "stranger_surface",
    "surface_variation",
    "range_vetted_surface",
)

#: The rotation-only family: the same questions, asked of bearing without
#: range.  Every one of them ships on a rotation-only member's record, and
#: every one is recorded unmeasurable on a finite member's.
ROTATION_CHANNELS = (
    "rot_self_resection",
    "rot_nonmember_resection",
    "rot_settling",
    "rot_warp",
    "rot_cycles",
    "rot_photometric",
    "parallax_residue",
)

#: The blocks `surface_channels` produces in one pass over the cloud.
SURFACE_CHANNELS = ("stranger_surface", "surface_variation", "range_vetted_surface")


def evaluate(members, f_vote, pair_obs=None, floors=None, images=None):
    """Measure every committed member and return one channel block per member.

    ``members`` are :class:`Member` views in one image frame; ``pair_obs`` is
    the capture-level ``(obs_c, obs_i, obs_uv)`` the two-view witness draws its
    raw matches from.  ``floors`` supplies the capture's own conditioning
    floors (``inlier_floor``, ``rot_inlier_floor``, ``rot_support_spread_bar``)
    when the caller is measuring part of a capture rather than all of it, so
    they are respected rather than re-derived on too small a population.
    ``images`` reads one of the capture's images as a grayscale array by its
    relative path; without it the photometric witness is a non-measurement.
    Returns ``{member idx: channels}``."""
    floors = floors or {}
    cache = ImageCache(images) if images is not None else None
    graph = None
    if pair_obs is not None and members:
        graph = _guard_none(
            PairGraph, pair_obs[0], pair_obs[1], pair_obs[2], members[0].n_img
        )
    blocks, loo_blocks, rot_blocks = {}, [], []
    for m in members:
        t0 = time.perf_counter()
        block = {"enabled": True, "model": m.model}
        block["focal_vote"] = _guard(focal_vote_channel, m.f_eq, f_vote)
        if m.pts is None:
            for name in GEOMETRY_CHANNELS + ROTATION_CHANNELS:
                block[name] = {
                    "measurable": False,
                    "unmeasurable_reason": "no_structure",
                }
            block["elapsed_s"] = round(time.perf_counter() - t0, 4)
            blocks[m.idx] = block
            continue
        if m.model == "rotation_only":
            # THE SAME QUESTIONS, ASKED OF BEARING WITHOUT RANGE.  A far-field
            # member is not an unmeasurable finite one: it is a different model
            # with its own hold-out, its own witness, its own settling and a
            # STRONGER warp reading, plus the one channel only it has -- what a
            # rotation left behind on frames that travelled.
            for name in FINITE_ONLY_CHANNELS:
                block[name] = {
                    "measurable": False,
                    "unmeasurable_reason": "no_finite_structure",
                }
            timings = {}
            for name, fn, args in (
                ("fit", fit_channels, (m,)),
                ("support", rot_support_channels, (m,)),
                ("rot_self_resection", rot_self_resection_channels, (m,)),
                ("rot_nonmember_resection", rot_nonmember_channels, (m, graph)),
                ("rot_settling", rot_settling_channels, (m,)),
                ("rot_warp", rot_warp_channels, (m,)),
                ("rot_cycles", rot_cycle_channels, (m,)),
                ("rot_photometric", rot_warp_photometric_channels, (m, cache)),
                ("parallax_residue", parallax_residue_channels, (m,)),
            ):
                t1 = time.perf_counter()
                block[name] = _guard(fn, *args)
                timings[name] = round(time.perf_counter() - t1, 4)
            rot_blocks.append(block["rot_self_resection"])
            block["timings_s"] = timings
            block["elapsed_s"] = round(time.perf_counter() - t0, 4)
            blocks[m.idx] = block
            continue
        for name in ROTATION_CHANNELS:
            block[name] = {
                "measurable": False,
                "unmeasurable_reason": "finite_structure",
            }
        scale = m.scene_scale()
        block["scene_scale"] = scale
        # ONE CHANNEL'S FAILURE COSTS THAT CHANNEL, not the battery: a member
        # whose settling probe blew up still ships its resection evidence.
        timings = {}
        for name, fn, args in (
            ("fit", fit_channels, (m,)),
            ("support", support_channels, (m,)),
            ("self_resection", self_resection_channels, (m, scale)),
            ("nonmember_resection", nonmember_channels, (m, scale, graph)),
            ("settling", settling_channels, (m,)),
            ("warp_epipolar", warp_channels, (m,)),
        ):
            t1 = time.perf_counter()
            block[name] = _guard(fn, *args)
            timings[name] = round(time.perf_counter() - t1, 4)
        # The three surface readings share one neighbour search, so they are
        # measured together and split into their own blocks afterwards.
        t1 = time.perf_counter()
        surf = _guard(surface_channels, m)
        timings["surface"] = round(time.perf_counter() - t1, 4)
        for name in SURFACE_CHANNELS:
            block[name] = surf.get(name, dict(surf))
        loo_blocks.append(block["self_resection"])
        block["timings_s"] = timings
        block["elapsed_s"] = round(time.perf_counter() - t0, 4)
        blocks[m.idx] = block
    # The floors are quantiles of the CAPTURE's own readings, so they can only
    # be taken once every member of each family has reported.
    apply_inlier_floor(loo_blocks, floors.get("inlier_floor"))
    apply_rot_floors(
        rot_blocks,
        floors.get("rot_inlier_floor"),
        floors.get("rot_support_spread_bar"),
    )
    return blocks


def disabled_block():
    """The block a member carries when the battery is switched off."""
    return {"enabled": False, "reason": "SFMTOOL_SEED_EVAL=0"}
