// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Staged bundle adjustment for images sharing one camera model.
//!
//! Jointly refines world-to-camera poses, world points, and optionally the
//! shared focal length and radial coefficient by minimizing soft-L1 pixel
//! reprojection error over a trim schedule with inter-round retriangulation —
//! the multi-view generalization of [`crate::geometry::pose_refine`], and the
//! native replacement for the cluster-bootstrap experiments' scipy BA
//! (`specs/core/bundle-adjustment.md`).
//!
//! Canonical camera frame throughout (the camera looks along `−Z`; a point in
//! front has `z < 0`). Each Levenberg–Marquardt step is taken over a local
//! `SO(3) × ℝ³` perturbation per image, `ℝ³` per point, and the optional
//! shared camera parameters (the two scalars, or the radial spline
//! coefficients), with analytic Jacobians; points are eliminated by a Schur
//! complement and the dense reduced camera system is solved by LU.

use nalgebra::{DMatrix, DVector, Matrix3, Point3, SMatrix, UnitQuaternion, Vector2, Vector3};

use crate::camera::distortion::bspline::{
    basis_at, bspline_is_monotone, BSPLINE_SUPPORT, MIN_BSPLINE_COEFFS,
};
use crate::camera::{CameraModel, PixelJacobian};
use crate::geometry::numeric::{cam_with, cam_with_bspline};
use crate::reconstruction::triangulation::triangulate_batch;
use crate::CameraIntrinsics;

/// A point behind the camera / outside the model domain contributes this pixel
/// residual per component — large enough to be trimmed, finite so the robust
/// cost stays well-posed (matches `reprojection_residuals` / `pose_refine`).
const INVALID_RESIDUAL: f64 = 1e6;

/// One round of the trim schedule.
#[derive(Clone, Copy, Debug)]
pub struct BaSchedule {
    /// Pre-round trim threshold on the reprojection residual norm, px.
    pub trim_px: f64,
    /// Soft-L1 scale for the round's solve, px.
    pub loss_scale: f64,
}

/// The default staged schedule (gross-outlier trim → tighten → final).
pub const DEFAULT_SCHEDULE: [BaSchedule; 3] = [
    BaSchedule {
        trim_px: 50.0,
        loss_scale: 5.0,
    },
    BaSchedule {
        trim_px: 12.0,
        loss_scale: 2.0,
    },
    BaSchedule {
        trim_px: 4.0,
        loss_scale: 1.0,
    },
];

/// Default widening multiplier on a stage's `loss_scale` for protected
/// observations (see [`bundle_adjust`]'s `protected`).
pub const DEFAULT_PROTECTED_LOSS_SCALE: f64 = 3.0;

/// Result of [`bundle_adjust`]. Poses and points are refined in place; this
/// carries what has no in-place home.
#[derive(Clone, Debug)]
pub struct BundleAdjustment {
    /// The shared focal length after the solve (the input focal unless
    /// `opt_f`).
    pub focal: f64,
    /// The shared radial coefficient after the solve — the input `k1` unless
    /// `opt_k1`, and `0.0` for models that have no such parameter.
    pub k1: f64,
    /// The shared radial spline coefficients after the solve — the input
    /// ones unless `opt_bspline`, and empty for models that carry no spline.
    pub bspline: Vec<f64>,
    /// Unweighted reprojection residual norm of every supplied observation at
    /// the final state; `+∞` where the point is non-finite, behind the
    /// camera, or outside the model domain. All-`∞` signals the degenerate
    /// exit (fewer than `min_obs` observations survived a trim).
    pub residual_norms: Vec<f64>,
}

/// Soft-L1 robust cost of a squared-residual-over-scale² argument:
/// `ρ(z) = 2·(√(1 + z) − 1)`, applied per residual COMPONENT (matching
/// scipy's element-wise `loss="soft_l1"` that this kernel replaces).
#[inline]
fn rho(z: f64) -> f64 {
    2.0 * ((1.0 + z).sqrt() - 1.0)
}

/// Second-order (Triggs-style) robust scaling of one residual component,
/// exactly scipy's `scale_for_robust_loss_function`: with `z = (r/s)²`,
/// scale the Jacobian row by `√(ρ' + 2·ρ''·z)` and the residual by
/// `ρ'/√(ρ' + 2·ρ''·z)`. For soft-L1 the curvature term collapses to
/// `ρ' + 2ρ''z = (1 + z)^(−3/2)`, so the row scale is `(1 + z)^(−¾)` and the
/// residual scale `(1 + z)^(+¼)`; the resulting `Jᵀr` equals the true robust
/// gradient `ρ'·Jᵀr` while `JᵀJ` carries the corrected curvature.
#[inline]
fn robust_scales(z: f64) -> (f64, f64) {
    let js = (1.0 + z).powf(-0.75);
    let rs = (1.0 + z).powf(0.25);
    (js, rs)
}

/// Projected pixel and the 2×3 projection Jacobian `∂(u, v)/∂p_cam` at a
/// camera-frame point. Analytic for the perspective family; a central
/// difference of `ray_to_pixel` for fisheye / equirectangular models, which
/// have no analytic Jacobian yet (same fallback as `pose_refine`). `None`
/// when the point is outside the model domain.
fn project_with_jac(
    cam: &CameraIntrinsics,
    p_cam: Vector3<f64>,
    analytic: bool,
) -> Option<PixelJacobian> {
    if analytic {
        return cam.ray_to_pixel_with_jacobian([p_cam.x, p_cam.y, p_cam.z]);
    }
    let uv = cam.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z])?;
    let h = 1e-6;
    let mut j = [[0.0f64; 3]; 2];
    for c in 0..3 {
        let mut pp = p_cam;
        let mut pm = p_cam;
        pp[c] += h;
        pm[c] -= h;
        let (up, vp) = cam.ray_to_pixel([pp.x, pp.y, pp.z])?;
        let (um, vm) = cam.ray_to_pixel([pm.x, pm.y, pm.z])?;
        j[0][c] = (up - um) / (2.0 * h);
        j[1][c] = (vp - vm) / (2.0 * h);
    }
    Some((uv, j))
}

/// Linearization of one observation: weighted residual, the weighted
/// camera-side (2×`CAM_COLS`) and point-side (2×3) Jacobian blocks, and the
/// reduced-camera-system column index of each camera-block column.
struct ObsBlocks<const CAM_COLS: usize> {
    /// Compact point index.
    cp: usize,
    res: Vector2<f64>,
    cam_j: SMatrix<f64, 2, CAM_COLS>,
    pt_j: SMatrix<f64, 2, 3>,
    /// Reduced-system column of each `cam_j` column, per observation:
    /// `[f, k1, (active spline coefficients,) δθ×3, δt×3]`. A spline slot
    /// whose active basis function is one of the gauge-anchored pair (full
    /// index < 2 — no coefficient, no column) points at [`K1_SLOT`], which is
    /// always pinned under the spline release, so its exactly-zero column
    /// accumulates exact zeros there.
    idx: [usize; CAM_COLS],
}

/// Width of one observation's camera-side Jacobian block in the base
/// instantiation: the two shared camera scalars (`f`, `k1`) plus the image's
/// six pose DOFs. Both scalar slots are always present — pinned in the
/// reduced system when unreleased — so the indexing is uniform.
const BASE_CAM_COLS: usize = 2 + 6;

/// Width under the spline release: the two scalar slots, the
/// [`BSPLINE_SUPPORT`] basis functions active at the observation's incidence
/// angle (cubic local support — the only nonzero columns of `∂(u, v)/∂c` for
/// that observation), and the six pose DOFs.
const BSPLINE_CAM_COLS: usize = 2 + BSPLINE_SUPPORT + 6;

/// The reduced camera system's slot of the shared focal.
const F_SLOT: usize = 0;
/// The reduced camera system's slot of the shared radial coefficient.
const K1_SLOT: usize = 1;
/// The reduced camera system's slot of the first shared spline coefficient
/// (coefficient `i` lives at `BSPLINE_SLOT0 + i`; the pose blocks follow the
/// whole coefficient vector).
const BSPLINE_SLOT0: usize = 2;

/// `∂(u, v)/∂k1` at a camera-frame point, for the one model `opt_k1` admits.
///
/// `SIMPLE_RADIAL_FISHEYE` projects a ray through `x_d = θ_d·ûx` with
/// `θ_d = θ·(1 + k1·θ²)`, so `∂x_d/∂k1 = θ³·ûx` and the pixel column is
/// `f·θ³·(ûx, ûy)` — exact at every incidence angle, the periphery past 90°
/// included, since `θ` comes from the ray rather than from a pixel radius.
/// `(ûx, ûy)` is the unit image direction in the OPTICAL frame
/// (`S = diag(1, −1, −1)` off canonical), which is where the `v` axis picks
/// up its sign.
///
/// On the optical axis the column is exactly zero: `θ³·û → 0` as `θ → 0`
/// whatever the (undefined) direction is.
///
/// A direction (point at infinity) takes this column unchanged — it projects
/// through the very same map, at `R·d` instead of `R·X + t`.
#[inline]
fn k1_column(f: f64, p_cam: Vector3<f64>) -> (f64, f64) {
    // Canonical → optical frame: (rx, ry, rz) = S·p_cam.
    let (rx, ry, rz) = (p_cam.x, -p_cam.y, -p_cam.z);
    let rho = rx.hypot(ry);
    if rho == 0.0 {
        return (0.0, 0.0);
    }
    let theta = rho.atan2(rz);
    // f·θ³·û with û = (rx, ry)/ρ.
    let s = f * theta * theta * theta / rho;
    (s * rx, s * ry)
}

/// Whether `θ_d = θ·(1 + k1·θ²)` is strictly increasing over the field the
/// solve actually images — the plausibility guard on a `k1` step, the
/// counterpart of the focal's `f > 0`.
///
/// `dθ_d/dθ = 1 + 3·k1·θ²` is positive everywhere for `k1 ≥ 0`. For `k1 < 0`
/// it vanishes at `θ_fold = 1/√(−3·k1)`, past which the map folds back: two
/// incidence angles share a pixel radius, `pixel_to_ray` picks the wrong
/// branch, and the projection stops being invertible. Since
/// `k1·θ_fold² = −1/3`, the outermost pixel radius still on the rising branch
/// is `f·θ_d(θ_fold) = (2/3)·f·θ_fold`, so the step is admissible exactly
/// when the field's outer edge sits inside that — or when the fold is past
/// `θ = π` and therefore past every physical ray.
///
/// `field_r` is the largest observed pixel radius from the principal point,
/// measured over the kept observations: the model's imaged field as the data
/// reports it, not a fixed constant.
fn k1_step_admissible(f: f64, k1: f64, field_r: f64) -> bool {
    if !k1.is_finite() {
        return false;
    }
    if k1 >= 0.0 {
        return true;
    }
    let theta_fold = 1.0 / (-3.0 * k1).sqrt();
    if theta_fold >= std::f64::consts::PI {
        return true;
    }
    (2.0 / 3.0) * f * theta_fold > field_r
}

/// The active `∂(u, v)/∂cᵢ` columns at a camera-frame point, for the one
/// model `opt_bspline` admits.
///
/// `SFMTOOL_FISHEYE` projects a ray through `x_d = θ_d·ûx` with
/// `θ_d = θ + Σ cᵢ·Bᵢ(θ)`, so `∂x_d/∂cᵢ = Bᵢ(θ)·ûx` and the pixel column is
/// `f·Bᵢ(θ)·(ûx, ûy)` — exact at every incidence angle, the periphery past
/// 90° included, since `θ` comes from the ray rather than from a pixel
/// radius (the same property as [`k1_column`], with `Bᵢ(θ)` in place of
/// `θ³`). Past `θ_max` the correction is held constant at `δ(θ_max)`, whose
/// coefficient derivative is `Bᵢ(θ_max)` — exactly what the clamp inside
/// `basis_at` evaluates, so the column is exact on the linear tail too.
/// `(ûx, ûy)` is the unit image direction in the OPTICAL frame
/// (`S = diag(1, −1, −1)` off canonical), like the `k1` column.
///
/// Returns the full-basis index of the first active function and its
/// [`BSPLINE_SUPPORT`] columns in basis order; entries whose full index is
/// below 2 are the gauge-anchored pair — no coefficient, and the caller must
/// not scatter them. On the optical axis every column is exactly zero (the
/// coefficient-bearing basis functions all vanish at `θ = 0` by the
/// center-anchored gauge, whatever the undefined `û` is).
///
/// A direction (point at infinity) takes these columns unchanged — it
/// projects through the very same map, at `R·d` instead of `R·X + t`.
fn bspline_columns(
    f: f64,
    n_coeffs: usize,
    theta_max: f64,
    p_cam: Vector3<f64>,
) -> (usize, [[f64; 2]; BSPLINE_SUPPORT]) {
    // Canonical → optical frame: (rx, ry, rz) = S·p_cam.
    let (rx, ry, rz) = (p_cam.x, -p_cam.y, -p_cam.z);
    let rho = rx.hypot(ry);
    if rho == 0.0 {
        return (0, [[0.0; 2]; BSPLINE_SUPPORT]);
    }
    let theta = rho.atan2(rz);
    let (first, values, _derivs) = basis_at(n_coeffs, theta_max, theta);
    let (ux, uy) = (rx / rho, ry / rho);
    let mut cols = [[0.0; 2]; BSPLINE_SUPPORT];
    for (col, &b) in cols.iter_mut().zip(&values) {
        // f·Bᵢ(θ)·û.
        let s = f * b;
        *col = [s * ux, s * uy];
    }
    (first, cols)
}

/// Whether candidate coefficients keep `θ_d = θ + δ(θ)` strictly increasing
/// over the spline's whole domain `[0, θ_max]` — the plausibility guard on a
/// spline step, the counterpart of [`k1_step_admissible`].
///
/// The whole domain rather than just the imaged field, because monotonicity
/// is the model's construction invariant: it is what gives the Newton solve
/// behind `pixel_to_ray` a guaranteed bracket, and the accepted spline is
/// persisted into a camera whose inverse must stay well-defined everywhere.
/// Beyond `θ_max` the slope is exactly `1`, so `[0, θ_max]` is the entire
/// risk region — and coefficient slots with no observation support are
/// pinned at their input values, so past the data the spline never moves
/// and the wider check costs no legitimate steps.
fn bspline_step_admissible(bspline: &[f64], theta_max: f64) -> bool {
    bspline.iter().all(|c| c.is_finite()) && bspline_is_monotone(bspline, theta_max, theta_max)
}

/// Staged bundle adjustment over images sharing one camera model.
///
/// Per schedule round: retriangulate every point from all supplied
/// observations at the current poses (rounds after the first), trim to
/// observations under `trim_px` with in-front depth and a finite point whose
/// track keeps at least `min_track` survivors, then run one robust sparse LM
/// solve at the round's `loss_scale`. Poses and points are refined in place;
/// the returned [`BundleAdjustment`] carries the focal and the per-observation
/// residual norms at the final state (`+∞` where invalid — and everywhere,
/// with the state passed through, when fewer than `min_obs` observations
/// survive a trim).
///
/// `point_at_infinity` optionally marks per-point directions: a marked row of
/// `points` is a world-frame direction (normalized on input and output) whose
/// observations depend on rotation and camera model only — see "Points at
/// infinity" in `specs/core/bundle-adjustment.md`. An absent mask is an
/// all-`false` mask, which reduces the solve to the finite-only one.
///
/// `protected` optionally marks per-observation protection (parallel to the
/// observation arrays): a protected observation is never removed by the
/// inter-round trim gates — it stays in the solve set every round regardless
/// of its residual and always counts toward `min_track` survival — and passes
/// through the robust loss at the wider scale
/// `protected_loss_scale · loss_scale` (bounded pull, never trimmed nor
/// dominating). See "Protected observations" in
/// `specs/core/bundle-adjustment.md`. An absent or all-`false` mask
/// reproduces the unprotected behavior bit for bit.
///
/// `opt_f` releases the shared focal (SIMPLE_PINHOLE, EQUIDISTANT_FISHEYE,
/// SIMPLE_RADIAL_FISHEYE and SFMTOOL_FISHEYE — the models this kernel's
/// analytic focal column `(u − cx)/f` is exact for), `opt_k1` the shared
/// radial coefficient (SIMPLE_RADIAL_FISHEYE only, the one model carrying
/// it), and `opt_bspline` the shared radial spline coefficients
/// (SFMTOOL_FISHEYE only, likewise). The binding rejects other models
/// loudly; the core silently degrades them to a fixed-parameter solve, never
/// a half-modeled DOF. `opt_k1` and `opt_bspline` are naturally exclusive
/// (no model carries both parameters). Callers stage the releases — fixed →
/// `opt_f` → `opt_f` plus the model's distortion release — so the distortion
/// rung opens on a focal that has already settled.
#[allow(clippy::too_many_arguments)]
pub fn bundle_adjust(
    cam: &CameraIntrinsics,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    point_at_infinity: Option<&[bool]>,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
) -> BundleAdjustment {
    if let Some(mask) = protected {
        assert_eq!(
            mask.len(),
            obs_img.len(),
            "protected and observation length mismatch"
        );
    }
    // An all-`false` protection mask is exactly no mask.
    let protected = protected.filter(|m| m.iter().any(|&b| b));
    // No mask is an all-`false` mask. The staged loop reduces exactly to a
    // finite-only solve when nothing is marked: every direction-specific
    // branch in it is guarded by the per-point flag.
    let n_pt = points.len();
    let no_directions: Vec<bool>;
    let is_dir: &[bool] = match point_at_infinity {
        Some(mask) => {
            assert_eq!(
                mask.len(),
                n_pt,
                "point_at_infinity and points length mismatch"
            );
            mask
        }
        None => {
            no_directions = vec![false; n_pt];
            &no_directions
        }
    };
    bundle_adjust_staged(
        cam,
        quats,
        trans,
        points,
        uv,
        obs_img,
        obs_pt,
        is_dir,
        protected,
        protected_loss_scale,
        opt_f,
        opt_k1,
        opt_bspline,
        schedule,
        max_iters,
        min_track,
        min_obs,
    )
}

// ── Points at infinity ──────────────────────────────────────────────────────
//
// The staged loop below handles per-point direction masks
// (`specs/core/bundle-adjustment.md`, "Points at infinity"). A marked row of
// `points` is a world-frame direction `d` projecting as
// `uv_pred = ray_to_pixel(R·d)` — no translation dependence — parameterized
// by a 2-DOF tangent-plane perturbation `d ← normalize(d + B(d)·δ)`. With no
// row marked, every direction branch is skipped and what remains is the
// finite-only solve.

/// Normalize a direction row. Zero-norm and non-finite rows come back `NaN`
/// (a `NaN` direction behaves like a `NaN` finite point: invalid until
/// re-estimated).
fn normalized_dir(p: [f64; 3]) -> [f64; 3] {
    let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        [p[0] / n, p[1] / n, p[2] / n]
    } else {
        [f64::NAN; 3]
    }
}

/// An orthonormal basis `B(d) = [b1 | b2]` of the tangent plane `d⊥` of a
/// unit direction (rebuilt at each linearization).
fn tangent_basis(d: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let anchor = if d.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let b1 = d.cross(&anchor).normalize();
    let b2 = d.cross(&b1);
    (b1, b2)
}

/// Mixed-path residual norms and in-front measures. Finite observations
/// report the canonical depth `−z_cam` (checked against the `1e-3·f` floor by
/// the caller); direction observations report `−(R·d)_z` (cheirality: any
/// positive value is in front). Invalid observations report
/// `INVALID_RESIDUAL` and a non-positive in-front measure.
///
/// **Model-aware in-front measure.** `−z_cam` is the perspective family's
/// notion of "in front": its projection is only defined for `z_cam < 0`, so
/// the sign of `−z` *is* the domain test. A ray-path model (fisheye /
/// equirectangular) images directions all the way out to `θ = π`, where
/// `−z_cam ≤ 0` for every observation past 90° off-axis — a real, in-domain
/// observation of a >180° capture. For those models the in-front measure is
/// the range `‖p_cam‖` instead, which keeps the floor doing the only job it
/// can still do there (reject a point sitting on the camera centre, where the
/// direction is undefined) and leaves the domain test to `ray_to_pixel`.
#[allow(clippy::too_many_arguments)]
fn residual_norms_depths(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &[[f64; 3]],
    is_dir: &[bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
) -> (Vec<f64>, Vec<f64>) {
    let n_obs = obs_img.len();
    let mut norms = vec![INVALID_RESIDUAL; n_obs];
    let mut depths = vec![f64::NEG_INFINITY; n_obs];
    let ray_path = cam.model.needs_ray_path();
    for k in 0..n_obs {
        let pi = obs_pt[k] as usize;
        let p = points[pi];
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            continue;
        }
        let i = obs_img[k] as usize;
        let rot = quats[i] * Vector3::new(p[0], p[1], p[2]);
        let c = if is_dir[pi] { rot } else { rot + trans[i] };
        depths[k] = if ray_path { c.norm() } else { -c.z };
        if let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) {
            norms[k] = (u - uv[k][0]).hypot(v - uv[k][1]);
        }
    }
    (norms, depths)
}

/// Re-estimation (rounds after the first): finite points rebuild from all
/// supplied observations at the current poses by ray-midpoint batch
/// triangulation; direction points re-estimate in closed form as the
/// normalized mean of their observations' back-rotated rays
/// `R_iᵀ · pixel_to_ray(uv)`. Tracks with fewer than two observations — and
/// points with none — become `NaN`; callers refill from their full
/// observation set (the bootstrap's post-BA refill rule).
#[allow(clippy::too_many_arguments)]
fn reestimate_points(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &mut [[f64; 3]],
    is_dir: &[bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
) {
    let n_obs = obs_img.len();
    let mut order: Vec<u32> = (0..n_obs as u32).collect();
    order.sort_unstable_by_key(|&k| obs_pt[k as usize]);

    // Direction means accumulate directly; finite tracks feed the batch
    // triangulation.
    let mut dir_sum: Vec<(usize, Vector3<f64>, usize)> = Vec::new();
    let mut dirs = Vec::new();
    let mut centers = Vec::new();
    let mut offsets = Vec::new();
    let mut track_pt = Vec::new();
    let mut prev: Option<u32> = None;
    for &k in &order {
        let k = k as usize;
        let p = obs_pt[k];
        let pu = p as usize;
        if prev != Some(p) {
            if is_dir[pu] {
                dir_sum.push((pu, Vector3::zeros(), 0));
            } else {
                offsets.push(dirs.len());
                track_pt.push(pu);
            }
            prev = Some(p);
        }
        let i = obs_img[k] as usize;
        let r_inv = quats[i].inverse();
        let d = cam.pixel_to_ray(uv[k][0], uv[k][1]);
        let world_ray = r_inv * Vector3::new(d[0], d[1], d[2]);
        if is_dir[pu] {
            let last = dir_sum.last_mut().unwrap();
            last.1 += world_ray;
            last.2 += 1;
        } else {
            dirs.push(world_ray);
            centers.push(Point3::from(-(r_inv * trans[i])));
        }
    }
    offsets.push(dirs.len());

    for p in points.iter_mut() {
        *p = [f64::NAN; 3];
    }
    let tris = triangulate_batch(&dirs, &centers, &offsets);
    for (t, tri) in tris.iter().enumerate() {
        if offsets[t + 1] - offsets[t] >= 2 {
            points[track_pt[t]] = [tri.point.x, tri.point.y, tri.point.z];
        }
    }
    for &(p, sum, count) in &dir_sum {
        if count >= 2 {
            let mean = sum / count as f64;
            points[p] = normalized_dir([mean.x, mean.y, mean.z]);
        }
    }
}

/// Robust cost over the kept observations at a candidate state. `cp_dir`
/// flags direction points by compact index; `s2s` is the per-kept-observation
/// squared loss scale (uniform except where a protected observation widens
/// it).
#[allow(clippy::too_many_arguments)]
fn robust_cost(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &[Vector3<f64>],
    cp_dir: &[bool],
    uv: &[[f64; 2]],
    kept: &[usize],
    obs_ci: &[usize],
    obs_cp: &[usize],
    s2s: &[f64],
) -> f64 {
    kept.iter()
        .enumerate()
        .map(|(kk, &k)| {
            let s2 = s2s[kk];
            let p = points[obs_cp[kk]];
            // A non-finite point (possible only for protected observations,
            // which the trim never excludes) is penalized like an
            // out-of-domain projection.
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                return s2 * rho(INVALID_RESIDUAL * INVALID_RESIDUAL / s2);
            }
            let rot = quats[obs_ci[kk]] * p;
            let c = if cp_dir[obs_cp[kk]] {
                rot
            } else {
                rot + trans[obs_ci[kk]]
            };
            match cam.ray_to_pixel([c.x, c.y, c.z]) {
                Some((u, v)) => {
                    let dx = u - uv[k][0];
                    let dy = v - uv[k][1];
                    s2 * (rho(dx * dx / s2) + rho(dy * dy / s2))
                }
                None => s2 * rho(INVALID_RESIDUAL * INVALID_RESIDUAL / s2),
            }
        })
        .sum()
}

/// One robust sparse LM solve over the kept observations with mixed finite
/// and direction points. Direction points use 2-DOF tangent-plane parameters
/// stored in the first two slots of the uniform 3-wide point block (the third
/// slot carries exact zeros and is pinned at the Schur inversion); images
/// whose kept observations are all directions have their translation slots
/// pinned in the reduced system (frozen for the round).
///
/// `CAM_COLS` selects the per-observation camera-block width:
/// [`BASE_CAM_COLS`] for every solve without the spline release (the
/// original layout, byte for byte), [`BSPLINE_CAM_COLS`] when the staged
/// loop releases an `SFMTOOL_FISHEYE` spline — the reduced system then
/// carries one shared slot per coefficient (`n_shared = 2 + n_coeffs`, still
/// dynamic) while each observation's block stays compile-time sized at the
/// spline's local support. `bspline0` is the current coefficient vector
/// (read-only outside the spline instantiation, where the camera's own
/// fixed spline rides along inside `cam0`).
#[allow(clippy::too_many_arguments)]
fn solve_lm<const CAM_COLS: usize>(
    cam0: &CameraIntrinsics,
    f0: f64,
    k1_0: f64,
    bspline0: &[f64],
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    is_dir: &[bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    kept: &[usize],
    opt_f: bool,
    opt_k1: bool,
    loss_scale: f64,
    max_iters: usize,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
) -> (f64, f64, Vec<f64>) {
    // The spline instantiation is selected by width; the staged loop only
    // requests it for a released, well-formed SFMTOOL_FISHEYE spline.
    let opt_bspline = CAM_COLS == BSPLINE_CAM_COLS;
    debug_assert!(
        !(opt_bspline && opt_k1),
        "opt_k1 and opt_bspline live on different models"
    );
    let (n_coeffs, theta_max) = match cam0.model {
        CameraModel::SfmtoolFisheye {
            bspline_theta_max, ..
        } if opt_bspline => (bspline0.len(), bspline_theta_max),
        _ => (0, 0.0),
    };
    // Compact the images and points the kept observations touch.
    let mut img_ids: Vec<usize> = kept.iter().map(|&k| obs_img[k] as usize).collect();
    img_ids.sort_unstable();
    img_ids.dedup();
    let mut pt_ids: Vec<usize> = kept.iter().map(|&k| obs_pt[k] as usize).collect();
    pt_ids.sort_unstable();
    pt_ids.dedup();
    let n_im = img_ids.len();
    let n_pt = pt_ids.len();
    let ci_of: std::collections::HashMap<usize, usize> =
        img_ids.iter().enumerate().map(|(c, &i)| (i, c)).collect();
    let cp_of: std::collections::HashMap<usize, usize> =
        pt_ids.iter().enumerate().map(|(c, &p)| (p, c)).collect();
    let obs_ci: Vec<usize> = kept
        .iter()
        .map(|&k| ci_of[&(obs_img[k] as usize)])
        .collect();
    let obs_cp: Vec<usize> = kept.iter().map(|&k| cp_of[&(obs_pt[k] as usize)]).collect();
    let cp_dir: Vec<bool> = pt_ids.iter().map(|&p| is_dir[p]).collect();

    // Translation observability: an image whose kept observations are all
    // directions gets its translation pinned for this round (a direction
    // observation's translation Jacobian is identically zero, so the block
    // would otherwise be pure zero curvature in the reduced system).
    let mut img_has_finite = vec![false; n_im];
    for (kk, &ci) in obs_ci.iter().enumerate() {
        if !cp_dir[obs_cp[kk]] {
            img_has_finite[ci] = true;
        }
    }
    let any_frozen = img_has_finite.iter().any(|&h| !h);

    // Per-point observation lists (compact indices into `kept`).
    let mut pt_obs: Vec<Vec<usize>> = vec![Vec::new(); n_pt];
    for (kk, &cp) in obs_cp.iter().enumerate() {
        pt_obs[cp].push(kk);
    }

    // Working state (compact copies). Direction rows arrive unit-normalized
    // (input normalization / re-estimation) and every accepted step
    // re-normalizes them.
    let mut f = f0;
    let mut k1 = k1_0;
    let mut bspline: Vec<f64> = bspline0.to_vec();
    let mut q: Vec<UnitQuaternion<f64>> = img_ids.iter().map(|&i| quats[i]).collect();
    let mut t: Vec<Vector3<f64>> = img_ids.iter().map(|&i| trans[i]).collect();
    let mut x: Vec<Vector3<f64>> = pt_ids
        .iter()
        .map(|&p| Vector3::new(points[p][0], points[p][1], points[p][2]))
        .collect();

    // Reduced camera system: [f | k1 | (spline coefficients) | 6 per image];
    // the scalar shared slots are always present (pinned when unreleased) to
    // keep the indexing uniform, the coefficient slots only under the
    // spline release.
    let n_shared = 2 + n_coeffs;
    let d = n_shared + 6 * n_im;
    // First pose slot of compact image `ci` in the reduced camera system.
    let img_slot = |ci: usize| n_shared + 6 * ci;
    // First pose column within an observation's camera block.
    let pose_c = CAM_COLS - 6;
    // The camera at a candidate shared state. Off the spline instantiation
    // this is exactly the scalar builder (a fixed SFMTOOL_FISHEYE spline
    // rides along inside `cam0` untouched).
    let build_cam = |fv: f64, k1v: f64, bsv: &[f64]| {
        if opt_bspline {
            cam_with_bspline(cam0, fv, bsv)
        } else {
            cam_with(cam0, fv, k1v)
        }
    };
    // The imaged field, as the kept observations report it: the largest pixel
    // radius from the principal point. The `k1` step guard asks whether the
    // distorted map stays monotone out to here.
    let field_r = {
        let (cx, cy) = cam0.principal_point();
        kept.iter()
            .map(|&k| (uv[k][0] - cx).hypot(uv[k][1] - cy))
            .fold(0.0f64, f64::max)
    };
    // Per-kept-observation squared loss scale: the round's scale everywhere,
    // widened by `protected_loss_scale` for protected observations.
    let s2 = loss_scale * loss_scale;
    let s2s: Vec<f64> = kept
        .iter()
        .map(|&k| match protected {
            Some(m) if m[k] => {
                let s = loss_scale * protected_loss_scale;
                s * s
            }
            _ => s2,
        })
        .collect();
    let mut lambda = 1e-3;
    let mut tiny_steps = 0usize;
    let mut cam = build_cam(f, k1, &bspline);
    let mut prev_cost = robust_cost(&cam, &q, &t, &x, &cp_dir, uv, kept, &obs_ci, &obs_cp, &s2s);

    let analytic = cam.model.supports_pixel_jacobian();
    for _ in 0..max_iters {
        // ── Linearize at the current state ───────────────────────────────
        // Tangent bases B(d) = [b1 | b2] for the direction points, rebuilt at
        // each linearization.
        let bases: Vec<(Vector3<f64>, Vector3<f64>)> = x
            .iter()
            .zip(&cp_dir)
            .map(|(xd, &dir)| {
                if dir {
                    tangent_basis(xd)
                } else {
                    (Vector3::zeros(), Vector3::zeros())
                }
            })
            .collect();
        let (cx, cy) = cam.principal_point();
        let blocks: Vec<ObsBlocks<CAM_COLS>> = kept
            .iter()
            .enumerate()
            .map(|(kk, &k)| {
                let ci = obs_ci[kk];
                let cp = obs_cp[kk];
                let s2 = s2s[kk];
                let dir = cp_dir[cp];
                let rot_pt = q[ci] * x[cp];
                let p_cam = if dir { rot_pt } else { rot_pt + t[ci] };
                let mut res = Vector2::new(INVALID_RESIDUAL, 0.0);
                let mut cam_j = SMatrix::<f64, 2, CAM_COLS>::zeros();
                let mut pt_j = SMatrix::<f64, 2, 3>::zeros();
                // Column indices: `[f, k1, (spline), δθ×3, δt×3]`. Spline
                // slots start at the pinned K1_SLOT dummy and are pointed at
                // their coefficient's shared slot below, where the
                // observation actually carries one.
                let mut idx = [K1_SLOT; CAM_COLS];
                idx[F_SLOT] = F_SLOT;
                let o = img_slot(ci);
                for (j, slot) in idx[pose_c..].iter_mut().enumerate() {
                    *slot = o + j;
                }
                // A non-finite point (protected observations only — the trim
                // never excludes them) keeps the penalized residual and zero
                // Jacobian rows: penalized, never steering.
                let proj = if x[cp].x.is_finite() && x[cp].y.is_finite() && x[cp].z.is_finite() {
                    project_with_jac(&cam, p_cam, analytic)
                } else {
                    None
                };
                if let Some(((u, v), jp)) = proj {
                    res = Vector2::new(u - uv[k][0], v - uv[k][1]);
                    let jp = SMatrix::<f64, 2, 3>::from_rows(&[
                        SMatrix::<f64, 1, 3>::from_row_slice(&jp[0]),
                        SMatrix::<f64, 1, 3>::from_row_slice(&jp[1]),
                    ]);
                    if opt_f {
                        // ∂(u, v)/∂f. Exact for every model the `opt_f` gate
                        // admits: the focal is a pure multiplier of an
                        // `f`-independent distorted coordinate, so the
                        // derivative is that coordinate.
                        cam_j[(0, F_SLOT)] = (u - cx) / f;
                        cam_j[(1, F_SLOT)] = (v - cy) / f;
                    }
                    if opt_k1 {
                        // ∂(u, v)/∂k1 = f·θ³·û — direction rows included,
                        // they project through the same map.
                        let (du, dv) = k1_column(f, p_cam);
                        cam_j[(0, K1_SLOT)] = du;
                        cam_j[(1, K1_SLOT)] = dv;
                    }
                    if opt_bspline {
                        // ∂(u, v)/∂cᵢ = f·Bᵢ(θ)·û for the ≤ 4 active basis
                        // functions — direction rows included, they project
                        // through the same map. Gauge-anchored functions
                        // (full index < 2) carry no coefficient: their slot
                        // keeps the pinned K1_SLOT dummy and their column
                        // stays exactly zero.
                        let (first, cols) = bspline_columns(f, n_coeffs, theta_max, p_cam);
                        for (j, col) in cols.iter().enumerate() {
                            let full = first + j;
                            if full < 2 {
                                continue;
                            }
                            idx[2 + j] = BSPLINE_SLOT0 + (full - 2);
                            cam_j[(0, 2 + j)] = col[0];
                            cam_j[(1, 2 + j)] = col[1];
                        }
                    }
                    // Rotation block: ∂p_cam/∂δθ = −[R·X]ₓ (finite) or
                    // −[R·d]ₓ (direction) — same composition either way.
                    let nskew = Matrix3::new(
                        0.0, rot_pt.z, -rot_pt.y, //
                        -rot_pt.z, 0.0, rot_pt.x, //
                        rot_pt.y, -rot_pt.x, 0.0,
                    );
                    cam_j
                        .fixed_view_mut::<2, 3>(0, pose_c)
                        .copy_from(&(jp * nskew));
                    let r_mat: Matrix3<f64> = q[ci].to_rotation_matrix().into_inner();
                    if dir {
                        // Translation block: zero (a direction observes no
                        // translation). Point block: 2-DOF tangent-plane
                        // parameters, ∂p_cam/∂δ = R·B(d) (columns b1, b2;
                        // the third slot stays exactly zero).
                        let (b1, b2) = bases[cp];
                        let col0 = jp * (r_mat * b1);
                        let col1 = jp * (r_mat * b2);
                        pt_j.set_column(0, &col0);
                        pt_j.set_column(1, &col1);
                    } else {
                        // Translation block: identity.
                        cam_j.fixed_view_mut::<2, 3>(0, pose_c + 3).copy_from(&jp);
                        // Point block: ∂p_cam/∂X = R.
                        pt_j.copy_from(&(jp * r_mat));
                    }
                }
                for row in 0..2 {
                    let z = res[row] * res[row] / s2;
                    let (js, rs) = robust_scales(z);
                    res[row] *= rs;
                    for col in 0..CAM_COLS {
                        cam_j[(row, col)] *= js;
                    }
                    for col in 0..3 {
                        pt_j[(row, col)] *= js;
                    }
                }
                ObsBlocks {
                    cp,
                    res,
                    cam_j,
                    pt_j,
                    idx,
                }
            })
            .collect();

        // ── Accumulate the normal-equation blocks ────────────────────────
        let mut h_cc = DMatrix::<f64>::zeros(d, d);
        let mut g_c = DVector::<f64>::zeros(d);
        let mut v_pp: Vec<Matrix3<f64>> = vec![Matrix3::zeros(); n_pt];
        let mut g_p: Vec<Vector3<f64>> = vec![Vector3::zeros(); n_pt];
        let mut w_cp: Vec<SMatrix<f64, CAM_COLS, 3>> = Vec::with_capacity(blocks.len());
        for b in &blocks {
            let idx = &b.idx;
            let h_local = b.cam_j.transpose() * b.cam_j;
            let g_local = b.cam_j.transpose() * b.res;
            for (a, &ia) in idx.iter().enumerate() {
                g_c[ia] += g_local[a];
                for (c, &ic) in idx.iter().enumerate() {
                    h_cc[(ia, ic)] += h_local[(a, c)];
                }
            }
            v_pp[b.cp] += b.pt_j.transpose() * b.pt_j;
            g_p[b.cp] += b.pt_j.transpose() * b.res;
            w_cp.push(b.cam_j.transpose() * b.pt_j);
        }

        // ── Damping ladder: re-damp and re-solve from this linearization ──
        let mut improved = false;
        for _ in 0..12 {
            let mut s = h_cc.clone();
            for dd in 0..d {
                s[(dd, dd)] += lambda * h_cc[(dd, dd)].max(1e-12);
            }
            let mut g_red = g_c.clone();
            // Schur-eliminate the points. A direction's block is 2×2 in the
            // first two slots (its third row/column carry exact zeros); the
            // third diagonal is pinned to 1 so the uniform 3×3 inversion
            // stays regular while contributing an exactly-zero update.
            let mut v_inv: Vec<Matrix3<f64>> = Vec::with_capacity(n_pt);
            let mut singular = false;
            for (p, v) in v_pp.iter().enumerate() {
                let mut vd = *v;
                for dd in 0..3 {
                    vd[(dd, dd)] += lambda * v[(dd, dd)].max(1e-12);
                }
                if cp_dir[p] {
                    vd[(2, 2)] = 1.0;
                }
                match vd.try_inverse() {
                    Some(inv) => v_inv.push(inv),
                    None => {
                        singular = true;
                        break;
                    }
                }
            }
            if singular {
                lambda *= 4.0;
                continue;
            }
            for (p, obs) in pt_obs.iter().enumerate() {
                let y = v_inv[p] * g_p[p];
                for &a in obs {
                    let wa = &w_cp[a];
                    let ia = &blocks[a].idx;
                    let contrib = wa * y;
                    for (r, &ir) in ia.iter().enumerate() {
                        g_red[ir] -= contrib[r];
                    }
                    for &b in obs {
                        let m = wa * v_inv[p] * w_cp[b].transpose();
                        let ib = &blocks[b].idx;
                        for (r, &ir) in ia.iter().enumerate() {
                            for (c, &ic) in ib.iter().enumerate() {
                                s[(ir, ic)] -= m[(r, c)];
                            }
                        }
                    }
                }
            }
            // Pin the unreleased shared-camera slots (their columns are
            // already exactly zero; this keeps the reduced system regular).
            // Under the spline release the same treatment covers each
            // coefficient slot with no observation support in this
            // linearization: no kept observation touches its basis span, so
            // its column carries zero curvature (`h_cc` diagonal exactly
            // zero — every contribution is a square) and the LU would be
            // singular. A pinned coefficient holds its value exactly, like a
            // frozen translation.
            for slot in 0..n_shared {
                let released = match slot {
                    F_SLOT => opt_f,
                    K1_SLOT => opt_k1,
                    _ => h_cc[(slot, slot)] > 0.0,
                };
                if released {
                    continue;
                }
                for dd in 0..d {
                    s[(slot, dd)] = 0.0;
                    s[(dd, slot)] = 0.0;
                }
                s[(slot, slot)] = 1.0;
                g_red[slot] = 0.0;
            }
            if any_frozen {
                // Pin the translation slots of all-direction images (frozen
                // for the round; their rotations still update).
                for (c, &has_finite) in img_has_finite.iter().enumerate() {
                    if has_finite {
                        continue;
                    }
                    for r in 0..3 {
                        let slot = img_slot(c) + 3 + r;
                        for dd in 0..d {
                            s[(slot, dd)] = 0.0;
                            s[(dd, slot)] = 0.0;
                        }
                        s[(slot, slot)] = 1.0;
                        g_red[slot] = 0.0;
                    }
                }
            }

            let Some(delta) = s.lu().solve(&(-g_red)) else {
                lambda *= 4.0;
                continue;
            };

            // Candidate state.
            let f_cand = if opt_f { f + delta[F_SLOT] } else { f };
            if opt_f && !(f_cand.is_finite() && f_cand > 1e-6) {
                lambda *= 4.0;
                continue;
            }
            let k1_cand = if opt_k1 { k1 + delta[K1_SLOT] } else { k1 };
            // The curvature rung's plausibility guard: a step that folds the
            // distorted map inside the imaged field is rejected the way a
            // non-positive focal is.
            if opt_k1 && !k1_step_admissible(f_cand, k1_cand, field_r) {
                lambda *= 4.0;
                continue;
            }
            // The spline rung's plausibility guard: a step that folds the
            // spline map anywhere on its domain is rejected the same way.
            let bspline_cand: Option<Vec<f64>> = if opt_bspline {
                let mut pc = bspline.clone();
                for (i, c) in pc.iter_mut().enumerate() {
                    let dv = delta[BSPLINE_SLOT0 + i];
                    // Pinned (unsupported) slots solve to exactly zero; skip
                    // the add so a `−0.0` coefficient keeps its sign (the
                    // frozen-translation precedent).
                    if dv != 0.0 {
                        *c += dv;
                    }
                }
                if !bspline_step_admissible(&pc, theta_max) {
                    lambda *= 4.0;
                    continue;
                }
                Some(pc)
            } else {
                None
            };
            let mut q_cand = q.clone();
            let mut t_cand = t.clone();
            for c in 0..n_im {
                let o = img_slot(c);
                let dtheta = Vector3::new(delta[o], delta[o + 1], delta[o + 2]);
                q_cand[c] = UnitQuaternion::from_scaled_axis(dtheta) * q[c];
                if img_has_finite[c] {
                    t_cand[c] = t[c] + Vector3::new(delta[o + 3], delta[o + 4], delta[o + 5]);
                }
                // Frozen images keep their translation untouched (a `+ 0.0`
                // would still flip the sign of a `−0.0` component).
            }
            let mut x_cand = x.clone();
            for (p, obs) in pt_obs.iter().enumerate() {
                // δp = −V⁻¹(g_p + Wᵀ δc), the Wᵀδc gathered over the point's
                // observations' camera blocks.
                let mut wt_dc = Vector3::zeros();
                for &a in obs {
                    let ia = &blocks[a].idx;
                    let mut dc = SMatrix::<f64, CAM_COLS, 1>::zeros();
                    for (r, &ir) in ia.iter().enumerate() {
                        dc[r] = delta[ir];
                    }
                    wt_dc += w_cp[a].transpose() * dc;
                }
                let dxp = v_inv[p] * (g_p[p] + wt_dc);
                if cp_dir[p] {
                    // d ← normalize(d + B(d)·δ), the 2-DOF tangent update.
                    let (b1, b2) = bases[p];
                    x_cand[p] = (x[p] - (b1 * dxp[0] + b2 * dxp[1])).normalize();
                } else {
                    x_cand[p] = x[p] - dxp;
                }
            }

            let cam_cand = match &bspline_cand {
                Some(pc) => build_cam(f_cand, k1_cand, pc),
                None => build_cam(f_cand, k1_cand, &bspline),
            };
            let new_cost = robust_cost(
                &cam_cand, &q_cand, &t_cand, &x_cand, &cp_dir, uv, kept, &obs_ci, &obs_cp, &s2s,
            );
            if new_cost < prev_cost {
                let rel = (prev_cost - new_cost) / prev_cost.max(1e-300);
                f = f_cand;
                k1 = k1_cand;
                if let Some(pc) = bspline_cand {
                    bspline = pc;
                }
                q = q_cand;
                t = t_cand;
                x = x_cand;
                cam = cam_cand;
                prev_cost = new_cost;
                lambda = (lambda * 0.5).max(1e-12);
                improved = true;
                // Converged only after tiny improvements twice in a row: a
                // single small step is how a traverse of a nearly-flat
                // valley STARTS (the focal release walks −20% through one),
                // so one is not proof of convergence.
                if rel < 1e-8 {
                    tiny_steps += 1;
                    if tiny_steps >= 2 {
                        lambda = f64::INFINITY;
                    }
                } else {
                    tiny_steps = 0;
                }
                break;
            }
            lambda *= 4.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved || lambda.is_infinite() {
            break;
        }
    }

    // Scatter the compact state back.
    for (c, &i) in img_ids.iter().enumerate() {
        quats[i] = q[c];
        trans[i] = t[c];
    }
    for (c, &p) in pt_ids.iter().enumerate() {
        points[p] = [x[c].x, x[c].y, x[c].z];
    }
    (f, k1, bspline)
}

/// The staged loop: direction-aware residuals, trims, re-estimation, and the
/// finite-survivors-only `min_obs` floor. With an all-`false` `is_dir` every
/// direction branch is skipped and this is the finite-only solve.
#[allow(clippy::too_many_arguments)]
fn bundle_adjust_staged(
    cam: &CameraIntrinsics,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    is_dir: &[bool],
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
) -> BundleAdjustment {
    let n_obs = obs_img.len();
    assert_eq!(obs_pt.len(), n_obs, "obs_img and obs_pt length mismatch");
    assert_eq!(uv.len(), n_obs, "uv and obs_img length mismatch");
    let is_prot = |k: usize| protected.is_some_and(|m| m[k]);

    // Direction rows are world-frame directions: normalized on input (and
    // kept normalized throughout, so they return normalized too).
    for (p, row) in points.iter_mut().enumerate() {
        if is_dir[p] {
            *row = normalized_dir(*row);
        }
    }

    // Which models release the focal is a property of this implementation's
    // focal column, not of the camera: the analytic `∂(u, v)/∂f = (u − cx)/f`
    // is exact exactly when the focal is a pure multiplier of a distorted
    // coordinate that does not itself read `f` — `u = f·x_d + cx` with
    // `x_d = rx/(−rz)` (SIMPLE_PINHOLE), `x_d = θ·ûx`, `θ = atan2(ρ, rz)`
    // (EQUIDISTANT_FISHEYE), or `x_d = θ·(1 + k1·θ²)·ûx` with the same
    // ray-derived `θ` (SIMPLE_RADIAL_FISHEYE — the distortion rides on `θ`,
    // not on `r/f`). Every other model fails that test, via a second focal
    // `fy` this kernel has no slot for or via coefficients applied to a
    // normalized coordinate whose relation to the pixel is `f`-dependent
    // (the multi-coefficient fisheye family: `x_d = θ·g(θ²)·û` with `θ`
    // recovered from `r/f`), and degrades to a fixed-focal solve (the binding
    // rejects it loudly first). SFMTOOL_FISHEYE passes the test the same way
    // SIMPLE_RADIAL_FISHEYE does: its radial spline is dimensionless and
    // rides on the ray-derived `θ` (`x_d = (θ + δ(θ))·ûx`), so `f` never
    // appears inside the distorted coordinate and `(u − cx)/f` stays exact.
    // `numeric::cam_at` mirrors this gate.
    let opt_f = opt_f
        && matches!(
            cam.model,
            CameraModel::SimplePinhole { .. }
                | CameraModel::EquidistantFisheye { .. }
                | CameraModel::SimpleRadialFisheye { .. }
                | CameraModel::SfmtoolFisheye { .. }
        );
    // The curvature rung exists on exactly one model: `SIMPLE_RADIAL_FISHEYE`
    // is the only one whose single radial coefficient acts on the ray's own
    // `θ`, which is what makes `∂(u, v)/∂k1 = f·θ³·û` exact. Same degrade.
    let opt_k1 = opt_k1 && matches!(cam.model, CameraModel::SimpleRadialFisheye { .. });
    // The spline rung likewise exists on exactly one model:
    // `SFMTOOL_FISHEYE`, whose dimensionless spline coefficients act on the
    // ray's own `θ`, making `∂(u, v)/∂cᵢ = f·Bᵢ(θ)·û` exact — and only when
    // its spline is actually defined (at least `MIN_BSPLINE_COEFFS`
    // coefficients on a positive finite `θ_max`; anything shorter evaluates
    // as the identity and carries nothing to release). Same degrade; `opt_k1`
    // and `opt_bspline` are therefore naturally exclusive, which the spline
    // instantiation's pinned-K1 dummy slot relies on.
    let opt_bspline = opt_bspline
        && match cam.model {
            CameraModel::SfmtoolFisheye {
                bspline_theta_max,
                ref bspline,
                ..
            } => {
                bspline.len() >= MIN_BSPLINE_COEFFS
                    && bspline_theta_max.is_finite()
                    && bspline_theta_max > 0.0
            }
            _ => false,
        };

    let mut f = cam.focal_lengths().0;
    let mut k1 = match cam.model {
        CameraModel::SimpleRadialFisheye {
            radial_distortion_k1,
            ..
        } => radial_distortion_k1,
        _ => 0.0,
    };
    let mut bspline: Vec<f64> = match &cam.model {
        CameraModel::SfmtoolFisheye { bspline, .. } => bspline.clone(),
        _ => Vec::new(),
    };

    for (rnd, stage) in schedule.iter().enumerate() {
        let cam_now = if opt_bspline {
            cam_with_bspline(cam, f, &bspline)
        } else {
            cam_with(cam, f, k1)
        };
        if rnd > 0 {
            reestimate_points(&cam_now, quats, trans, points, is_dir, uv, obs_img, obs_pt);
        }
        let (norms, depths) =
            residual_norms_depths(&cam_now, quats, trans, points, is_dir, uv, obs_img, obs_pt);
        // In-front: the model-aware measure from `residual_norms_depths`
        // (canonical depth for the perspective family, range for a ray-path
        // model) over the 1e-3·f floor for finite observations; cheirality
        // (R·d)_z < 0 for directions. Protected observations bypass the trim
        // gates entirely.
        let mut keep: Vec<bool> = (0..n_obs)
            .map(|k| {
                let floor = if is_dir[obs_pt[k] as usize] {
                    0.0
                } else {
                    1e-3 * f
                };
                is_prot(k) || (norms[k] < stage.trim_px && depths[k] > floor)
            })
            .collect();
        // Track survival: drop observations of points with < min_track kept
        // (protected observations count as survivors and are never dropped).
        let mut surv = vec![0usize; points.len()];
        for k in 0..n_obs {
            if keep[k] {
                surv[obs_pt[k] as usize] += 1;
            }
        }
        for k in 0..n_obs {
            keep[k] = keep[k] && (is_prot(k) || surv[obs_pt[k] as usize] >= min_track);
        }
        let kept: Vec<usize> = (0..n_obs).filter(|&k| keep[k]).collect();
        // The degenerate floor counts finite-point survivors only: direction
        // observations constrain no structure depth, so they cannot vouch
        // for a usable finite solve.
        let kept_finite = kept
            .iter()
            .filter(|&&k| !is_dir[obs_pt[k] as usize])
            .count();
        if kept_finite < min_obs {
            // Degenerate (e.g. a wildly wrong focal): state passes through.
            return BundleAdjustment {
                focal: f,
                k1,
                bspline,
                residual_norms: vec![f64::INFINITY; n_obs],
            };
        }
        (f, k1, bspline) = if opt_bspline {
            solve_lm::<BSPLINE_CAM_COLS>(
                cam,
                f,
                k1,
                &bspline,
                quats,
                trans,
                points,
                is_dir,
                uv,
                obs_img,
                obs_pt,
                &kept,
                opt_f,
                opt_k1,
                stage.loss_scale,
                max_iters,
                protected,
                protected_loss_scale,
            )
        } else {
            solve_lm::<BASE_CAM_COLS>(
                cam,
                f,
                k1,
                &bspline,
                quats,
                trans,
                points,
                is_dir,
                uv,
                obs_img,
                obs_pt,
                &kept,
                opt_f,
                opt_k1,
                stage.loss_scale,
                max_iters,
                protected,
                protected_loss_scale,
            )
        };
    }

    let cam_final = if opt_bspline {
        cam_with_bspline(cam, f, &bspline)
    } else {
        cam_with(cam, f, k1)
    };
    let (norms, _depths) = residual_norms_depths(
        &cam_final, quats, trans, points, is_dir, uv, obs_img, obs_pt,
    );
    let residual_norms = norms
        .iter()
        .map(|&r| {
            if r >= INVALID_RESIDUAL {
                f64::INFINITY
            } else {
                r
            }
        })
        .collect();
    BundleAdjustment {
        focal: f,
        k1,
        bspline,
        residual_norms,
    }
}

#[cfg(test)]
mod tests;
