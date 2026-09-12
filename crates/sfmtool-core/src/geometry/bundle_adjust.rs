// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Staged bundle adjustment for images sharing one camera model.
//!
//! Jointly refines world-to-camera poses, world points, and optionally the
//! shared focal length and radial coefficient by minimizing soft-L1 pixel
//! reprojection error over a trim schedule with inter-round retriangulation —
//! the multi-view generalization of [`crate::geometry::pose_refine`], and the
//! native replacement for the cluster-bootstrap experiments' scipy BA
//! (`specs/core/geometry/bundle-adjustment.md`).
//!
//! Canonical camera frame throughout (the camera looks along `−Z`; a point in
//! front has `z < 0`). Each Levenberg–Marquardt step is taken over a local
//! `SO(3) × ℝ³` perturbation per image, `ℝ³` per point, and the optional
//! shared camera parameters (the two scalars, or the radial spline
//! coefficients), with analytic Jacobians; points are eliminated by a Schur
//! complement and the dense reduced camera system is solved by LU.

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, UnitQuaternion, Vector2, Vector3};

use crate::camera::distortion::bspline::{
    basis_at, bspline_is_monotone, BSPLINE_SUPPORT, MIN_BSPLINE_COEFFS,
};
use crate::camera::intrinsics::SplineRadial;
use crate::camera::{CameraModel, PixelJacobian};
use crate::progress::Progress;
use crate::progress_info;
use crate::reconstruction::point_estimation::{
    estimate_points_from_observations, tangent_basis, FewObservations, ObservationSet,
    PointDistance, PointRules,
};
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

/// Default constant `c` in the noise-floor angle `θ_floor = c·s/f` a crossing
/// free point is classified by (see [`FreePointPolicy`]).
pub const DEFAULT_NOISE_FLOOR_SCALE: f64 = 2.0;

/// What the solve owns of a point, orthogonal to the representation the point
/// currently carries.
///
/// See "Point constraints" in `specs/core/geometry/bundle-adjustment.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PointConstraint {
    /// The solve owns the point: its position moves, and under
    /// [`FreePointPolicy::cross`] its representation is re-decided from its
    /// rays between rounds.
    #[default]
    Free,
    /// The caller owns the point's distance from a reference and the solve owns
    /// its direction: the point is `X = O + r·d` with `d` its only parameter.
    Ranged,
    /// The caller owns the point: its coordinate is fixed for the whole solve,
    /// its observations still form residuals and camera blocks, and it has no
    /// parameters.
    Held,
}

/// Where a ranged point's distance is measured from.
///
/// A reference is a function of the camera poses, never a fixed world point: a
/// distance measured from a coordinate the cameras are free to move away from
/// constrains nothing about the cameras, because the adjustment's gauge is
/// free. Both forms resolve to a world position through
/// `C_k = −R_kᵀ · t_k` at whatever poses the round holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DistanceReference {
    /// One image's camera centre: "the landmark is `r` from where this
    /// photograph was taken".
    Image(u32),
    /// The mean of several images' camera centres, `O = (1/|K|)·Σ C_k`: a
    /// capture station whose frames sit close together and none of which is the
    /// survey point on its own.
    ImageMean(Vec<u32>),
}

impl DistanceReference {
    /// The image indices whose camera centres are averaged, in the caller's
    /// order.
    pub fn images(&self) -> &[u32] {
        match self {
            Self::Image(k) => std::slice::from_ref(k),
            Self::ImageMean(ks) => ks,
        }
    }
}

/// The constraint on every point, and what the ranged ones are held at.
///
/// The three arrays are parallel to the adjustment's `points`. `distance` and
/// `reference` are read only where `constraint` is [`PointConstraint::Ranged`]:
/// `distance` carries the distance (`+∞` for a direction, which needs no
/// reference) and `reference` the pose-relative origin it is measured from.
///
/// [`PointConstraints::all_free`] plus the two setters is the intended way to
/// build one, so a caller states only the points it owns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointConstraints {
    /// One constraint per point.
    pub constraint: Vec<PointConstraint>,
    /// One distance per point; read where the constraint is ranged, `NaN`
    /// elsewhere.
    pub distance: Vec<f64>,
    /// One reference per point; read where the constraint is ranged at a finite
    /// distance, `None` elsewhere.
    pub reference: Vec<Option<DistanceReference>>,
}

impl PointConstraints {
    /// `n_pt` free points: the state a caller starts from and edits.
    pub fn all_free(n_pt: usize) -> Self {
        Self {
            constraint: vec![PointConstraint::Free; n_pt],
            distance: vec![f64::NAN; n_pt],
            reference: vec![None; n_pt],
        }
    }

    /// Hold point `p` at whatever coordinate the caller hands the adjustment.
    pub fn hold(&mut self, p: usize) {
        self.constraint[p] = PointConstraint::Held;
        self.distance[p] = f64::NAN;
        self.reference[p] = None;
    }

    /// Range point `p` at `distance` from `reference`, the solve keeping only
    /// its direction. `f64::INFINITY` is a direction and takes no reference.
    pub fn constrain_distance(
        &mut self,
        p: usize,
        distance: f64,
        reference: Option<DistanceReference>,
    ) {
        self.constraint[p] = PointConstraint::Ranged;
        self.distance[p] = distance;
        self.reference[p] = reference;
    }

    /// The constraint set three parallel per-point arrays state, or `None` when
    /// they state nothing beyond "every point free" -- which is the off position
    /// the kernel's parity requirement is stated against, and is why this is not
    /// simply always a set.
    ///
    /// `held` marks the points the caller owns outright, `distance` carries a
    /// ranged point's distance (`NaN` where the point is not ranged), and
    /// `reference` where a finite distance is measured from. The arrays are the
    /// shape a caller reading columns off a file or arrays out of a numpy world
    /// already has, and the checks below are the ones neither of them should be
    /// writing twice: the two constraints are exclusive on one point, a distance
    /// is strictly positive or `+inf`, a finite one names a reference, and a
    /// reference names an image that exists.
    ///
    /// The messages name the arrays -- `distance[p]`, `distance_from[p]` -- so a
    /// caller that took them from a keyword argument can hand the sentence
    /// straight to its user.
    pub fn from_arrays(
        held: Option<&[bool]>,
        distance: Option<&[f64]>,
        reference: &[Option<DistanceReference>],
        n_pt: usize,
        n_img: usize,
    ) -> Result<Option<Self>, PointConstraintsError> {
        let mut constraints = PointConstraints::all_free(n_pt);
        let mut any = false;
        for p in 0..n_pt {
            let is_held = held.is_some_and(|h| h[p]);
            let r = distance.map_or(f64::NAN, |d| d[p]);
            let is_ranged = !r.is_nan();
            if is_held && is_ranged {
                return Err(PointConstraintsError::HeldAndRanged(p));
            }
            if is_held {
                constraints.hold(p);
                any = true;
                continue;
            }
            if !is_ranged {
                continue;
            }
            if r <= 0.0 {
                return Err(PointConstraintsError::NotADistance {
                    point: p,
                    distance: r,
                });
            }
            let origin = if r.is_finite() {
                let origin = reference.get(p).cloned().flatten().ok_or(
                    PointConstraintsError::NoReference {
                        point: p,
                        distance: r,
                    },
                )?;
                for &k in origin.images() {
                    if k as usize >= n_img {
                        return Err(PointConstraintsError::ReferenceOutOfRange {
                            point: p,
                            image: k,
                            image_count: n_img,
                        });
                    }
                }
                Some(origin)
            } else {
                // A direction is measured from nothing, so an origin given here
                // is read as the caller stating a reference the geometry never
                // needs.
                None
            };
            constraints.constrain_distance(p, r, origin);
            any = true;
        }
        Ok(any.then_some(constraints))
    }
}

/// Why per-point arrays state no constraint set.
///
/// Every variant is a statement the arrays make that the adjustment cannot
/// honour, rather than a failure of the solve: the caller is building the
/// constraints, and what it hears back is which point it got wrong.
#[derive(Debug, Clone, PartialEq)]
pub enum PointConstraintsError {
    /// One point is both held and ranged.
    HeldAndRanged(usize),
    /// A ranged point's distance is not one: zero, negative, or `-inf`.
    NotADistance {
        /// The point.
        point: usize,
        /// The distance stated for it.
        distance: f64,
    },
    /// A ranged point's finite distance names no reference to measure from.
    NoReference {
        /// The point.
        point: usize,
        /// The distance stated for it.
        distance: f64,
    },
    /// A reference names an image the table does not hold.
    ReferenceOutOfRange {
        /// The point.
        point: usize,
        /// The image named.
        image: u32,
        /// How many images there are.
        image_count: usize,
    },
}

impl std::fmt::Display for PointConstraintsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PointConstraintsError::HeldAndRanged(p) => write!(
                f,
                "point {p} is both held and ranged; a held point owns its whole \
                 coordinate, so there is no direction left for a distance to constrain"
            ),
            PointConstraintsError::NotADistance { point, distance } => write!(
                f,
                "distance[{point}] = {distance} is not a distance; a ranged point needs a \
                 strictly positive one, +inf for a direction, or NaN to be left free"
            ),
            PointConstraintsError::NoReference { point, distance } => write!(
                f,
                "distance[{point}] = {distance} is finite and needs a distance_from: a \
                 distance is measured from a camera centre, not from the world frame"
            ),
            PointConstraintsError::ReferenceOutOfRange {
                point,
                image,
                image_count,
            } => write!(
                f,
                "distance_from[{point}] names image {image}, past the {image_count} images"
            ),
        }
    }
}

impl std::error::Error for PointConstraintsError {}

/// How a free point's representation is decided.
///
/// [`FreePointPolicy::default`] is the off position: the caller's
/// `point_at_infinity` mask is honoured for the whole solve, which is the
/// kernel the parity requirement is stated against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreePointPolicy {
    /// Re-decide every free point's representation at each inter-round
    /// re-estimation, from its own rays at the current geometry: a track whose
    /// widest ray pair opens past the noise floor is finite, one that closes
    /// below it is a direction, and so is one that solves behind a camera that
    /// observes it.
    pub cross: bool,
    /// The constant `c` in the noise-floor angle `θ_floor = c·s/f`, with `s`
    /// the round's loss scale in pixels and `f` the camera's current focal.
    /// Read only when [`Self::cross`] is set.
    pub noise_floor_scale: f64,
}

impl Default for FreePointPolicy {
    fn default() -> Self {
        Self {
            cross: false,
            noise_floor_scale: DEFAULT_NOISE_FLOOR_SCALE,
        }
    }
}

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
    /// exit (fewer than `min_obs` observations, finite and direction alike,
    /// survived a trim).
    pub residual_norms: Vec<f64>,
    /// The representation each point ended with, one entry per point: `true`
    /// where the returned row is a world-frame direction and `false` where it
    /// is a position. A free point's entry is the caller's input mask unless
    /// [`FreePointPolicy::cross`] let the re-estimation re-decide it, a held
    /// point's is its input value, and a ranged point's is whether its distance
    /// is infinite.
    pub point_at_infinity: Vec<bool>,
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
    /// The reference-camera columns of a ranged point's observation: the
    /// reduced-system column and its `∂(u, v)/∂parameter`, one entry per DOF of
    /// every reference camera the round is solving. Empty for every observation
    /// of a free or held point, and for a ranged point whose reference cameras
    /// no kept observation touches. A column here may share its reduced-system
    /// slot with one of [`Self::idx`] -- the observing camera can be its own
    /// reference -- and the two then simply add.
    ref_cols: Vec<(usize, Vector2<f64>)>,
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

/// The active `∂(u, v)/∂cᵢ` columns at a camera-frame point, for the two
/// models `opt_bspline` admits.
///
/// Both project a ray through `x_d = d_d·ûx` with `d_d = d + Σ cᵢ·Bᵢ(d)` over
/// the model's radial coordinate `d` (`radial`): the incidence angle `θ` for
/// `SFMTOOL_FISHEYE`, the normalized image-plane radius `ρ = ρ_xy/rz` for
/// `SFMTOOL_PINHOLE`. So `∂x_d/∂cᵢ = Bᵢ(d)·ûx` and the pixel column is
/// `f·Bᵢ(d)·(ûx, ûy)` — exact everywhere in the field, since `d` comes from
/// the ray rather than from a pixel radius (the same property as
/// [`k1_column`], with `Bᵢ(d)` in place of `θ³`), the fisheye's periphery past
/// 90° included. Past `d_max` the correction continues along its end tangent,
/// `δ(d) = δ(d_max) + δ'(d_max)·(d − d_max)`, so the exact coefficient
/// derivative there is `Bᵢ(d_max) + B'ᵢ(d_max)·(d − d_max)`; the clamped
/// `basis_at` returns both terms, and the column carries that tail exactly
/// rather than approximating it by the endpoint basis alone.
/// `(ûx, ûy)` is the unit image direction in the OPTICAL frame
/// (`S = diag(1, −1, −1)` off canonical), like the `k1` column.
///
/// Returns the full-basis index of the first active function and its
/// [`BSPLINE_SUPPORT`] columns in basis order; entries whose full index is
/// below 2 are the gauge-anchored pair — no coefficient, and the caller must
/// not scatter them. On the optical axis every column is exactly zero (the
/// coefficient-bearing basis functions all vanish at `d = 0` by the
/// center-anchored gauge, whatever the undefined `û` is).
///
/// A direction (point at infinity) takes these columns unchanged — it
/// projects through the very same map, at `R·d` instead of `R·X + t`.
fn bspline_columns(
    f: f64,
    n_coeffs: usize,
    d_max: f64,
    radial: SplineRadial,
    p_cam: Vector3<f64>,
) -> (usize, [[f64; 2]; BSPLINE_SUPPORT]) {
    // Canonical → optical frame: (rx, ry, rz) = S·p_cam.
    let (rx, ry, rz) = (p_cam.x, -p_cam.y, -p_cam.z);
    let rho = rx.hypot(ry);
    if rho == 0.0 {
        return (0, [[0.0; 2]; BSPLINE_SUPPORT]);
    }
    // The caller only reaches this for an observation the model projected, so
    // the pinhole quotient is taken on a strictly positive `rz`.
    let d = match radial {
        SplineRadial::IncidenceAngle => rho.atan2(rz),
        SplineRadial::ImagePlaneRadius => rho / rz,
    };
    let (first, values, derivs) = basis_at(n_coeffs, d_max, d);
    // Zero inside the domain; the tail's lever arm past it. `basis_at` clamped,
    // so `values`/`derivs` are the endpoint basis and its slope there.
    let over = (d - d_max).max(0.0);
    let (ux, uy) = (rx / rho, ry / rho);
    let mut cols = [[0.0; 2]; BSPLINE_SUPPORT];
    for ((col, &b), &bp) in cols.iter_mut().zip(&values).zip(&derivs) {
        // f·(Bᵢ(d) + B'ᵢ(d)·over)·û.
        let s = f * (b + bp * over);
        *col = [s * ux, s * uy];
    }
    (first, cols)
}

/// Whether candidate coefficients keep `d_d = d + δ(d)` strictly increasing
/// over the spline's whole domain `[0, d_max]` and its linear continuation past
/// it — the plausibility guard on a spline step, the counterpart of
/// [`k1_step_admissible`]. The check is
/// arithmetic on the coefficients and the domain end, so it reads the same for
/// either radial coordinate.
///
/// The whole domain rather than just the imaged field, because monotonicity
/// is the model's construction invariant: it is what gives the Newton solve
/// behind `pixel_to_ray` a guaranteed bracket, and the accepted spline is
/// persisted into a camera whose inverse must stay well-defined everywhere.
/// Beyond `d_max` the map continues with the constant slope `1 + δ'(d_max)`,
/// which `bspline_is_monotone` decides along with the domain — and coefficient
/// slots with no observation support are pinned at their input values, so past
/// the data the spline never moves and the wider check costs no legitimate
/// steps.
fn bspline_step_admissible(bspline: &[f64], d_max: f64) -> bool {
    bspline.iter().all(|c| c.is_finite()) && bspline_is_monotone(bspline, d_max, d_max)
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
/// with the state passed through, when fewer than `min_obs` observations —
/// finite and direction alike — survive a trim).
///
/// `point_at_infinity` optionally marks per-point directions: a marked row of
/// `points` is a world-frame direction (normalized on input and output) whose
/// observations depend on rotation and camera model only — see "Points at
/// infinity" in `specs/core/geometry/bundle-adjustment.md`. An absent mask is an
/// all-`false` mask, which reduces the solve to the finite-only one.
///
/// `constraints` optionally states a constraint per point (free, ranged or
/// held) and what the ranged ones are held at; an absent one is every point
/// free. `free_points` says whether a free point's representation is re-decided
/// from its rays between rounds and at what noise floor. See "Point
/// constraints" in
/// `specs/core/geometry/bundle-adjustment.md`. Absent constraints and a default
/// [`FreePointPolicy`] reproduce the kernel without them bit for bit.
///
/// `protected` optionally marks per-observation protection (parallel to the
/// observation arrays): a protected observation is never removed by the
/// inter-round trim gates — it stays in the solve set every round regardless
/// of its residual and always counts toward `min_track` survival — and passes
/// through the robust loss at the wider scale
/// `protected_loss_scale · loss_scale` (bounded pull, never trimmed nor
/// dominating). See "Protected observations" in
/// `specs/core/geometry/bundle-adjustment.md`. An absent or all-`false` mask
/// reproduces the unprotected behavior bit for bit.
///
/// `opt_f` releases the shared focal (SIMPLE_PINHOLE, EQUIDISTANT_FISHEYE,
/// SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE and SFMTOOL_PINHOLE — the models
/// this kernel's analytic focal column `(u − cx)/f` is exact for), `opt_k1`
/// the shared radial coefficient (SIMPLE_RADIAL_FISHEYE only, the one model
/// carrying it), and `opt_bspline` the shared radial spline coefficients
/// (SFMTOOL_FISHEYE and SFMTOOL_PINHOLE, the two carrying a spline). The
/// binding rejects other models loudly; the core silently degrades them to a
/// fixed-parameter solve, never a half-modeled DOF. `opt_k1` and `opt_bspline` are naturally exclusive
/// (no model carries both parameters). Callers stage the releases — fixed →
/// `opt_f` → `opt_f` plus the model's distortion release — so the distortion
/// rung opens on a focal that has already settled.
///
/// `progress` is where the rounds and the LM iterations inside them are
/// reported: one phase per schedule round, carrying that round's trim and loss
/// scale, and an iteration count under it, so a caller knows which round a long
/// solve is in. It is also how this call is asked to stop, which it is between
/// rounds and between iterations; a stopped solve returns the state it had
/// reached rather than an error, since it has no `Result` to put one in, and
/// the caller asks `Progress::is_cancelled` again to find out. Pass
/// `&Progress::none()` to report nothing and never stop: every method on it is
/// a branch on a null sink, so the solve runs exactly as it did before this
/// parameter existed.
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
    constraints: Option<&PointConstraints>,
    free_points: FreePointPolicy,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
    progress: &Progress<'_>,
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
    let cons = Constraints::resolve(constraints, n_pt, quats.len());
    // A ranged point's representation is its distance's: finite where the
    // distance is, a direction at `r = ∞`, whatever the caller's mask says.
    let mut is_dir_state: Vec<bool> = is_dir.to_vec();
    for (p, dir) in is_dir_state.iter_mut().enumerate() {
        if cons.constraint[p] == PointConstraint::Ranged {
            *dir = !cons.distance[p].is_finite();
        }
    }
    bundle_adjust_staged(
        cam,
        quats,
        trans,
        points,
        uv,
        obs_img,
        obs_pt,
        &mut is_dir_state,
        &cons,
        free_points,
        protected,
        protected_loss_scale,
        opt_f,
        opt_k1,
        opt_bspline,
        schedule,
        max_iters,
        min_track,
        min_obs,
        progress,
    )
}

// ── Points at infinity ──────────────────────────────────────────────────────
//
// The staged loop below handles per-point direction masks
// (`specs/core/geometry/bundle-adjustment.md`, "Points at infinity"). A marked row of
// `points` is a world-frame direction `d` projecting as
// `uv_pred = ray_to_pixel(R·d)` — no translation dependence — parameterized
// by a 2-DOF tangent-plane perturbation `d ← normalize(d + B(d)·δ)`. With no
// row marked, every direction branch is skipped and what remains is the
// finite-only solve.

/// The caller's [`PointConstraints`] as the staged loop reads them: one entry
/// per point, validated once at the entry point, with a ranged point's
/// reference flattened to the image indices whose centres are averaged.
struct Constraints {
    constraint: Vec<PointConstraint>,
    /// A ranged point's distance from its reference; `NaN` for every other
    /// point.
    distance: Vec<f64>,
    /// A ranged point's reference images, deduplicated in the caller's order.
    /// Empty for every other point and at `r = ∞`, which needs no reference.
    refs: Vec<Vec<usize>>,
    any_ranged: bool,
    any_held: bool,
}

impl Constraints {
    /// Every point free: what an absent [`PointConstraints`] means, and the
    /// state every branch below is inert under.
    fn all_free(n_pt: usize) -> Self {
        Self {
            constraint: vec![PointConstraint::Free; n_pt],
            distance: vec![f64::NAN; n_pt],
            refs: vec![Vec::new(); n_pt],
            any_ranged: false,
            any_held: false,
        }
    }

    /// Validate the caller's constraints against the state arrays and flatten
    /// them. A ranged point needs a strictly positive distance, and a finite
    /// one needs a reference naming images the adjustment holds.
    fn resolve(constraints: Option<&PointConstraints>, n_pt: usize, n_img: usize) -> Self {
        let Some(c) = constraints else {
            return Self::all_free(n_pt);
        };
        assert_eq!(
            c.constraint.len(),
            n_pt,
            "point constraints and points mismatch"
        );
        let mut out = Self::all_free(n_pt);
        for p in 0..n_pt {
            out.constraint[p] = c.constraint[p];
            match c.constraint[p] {
                PointConstraint::Free => {}
                PointConstraint::Held => out.any_held = true,
                PointConstraint::Ranged => {
                    out.any_ranged = true;
                    assert_eq!(
                        c.distance.len(),
                        n_pt,
                        "constraint distances and points mismatch"
                    );
                    let r = c.distance[p];
                    assert!(r > 0.0, "point {p} is ranged at a non-positive distance");
                    out.distance[p] = r;
                    if !r.is_finite() {
                        continue;
                    }
                    assert_eq!(
                        c.reference.len(),
                        n_pt,
                        "constraint references and points mismatch"
                    );
                    let reference = c.reference[p]
                        .as_ref()
                        .unwrap_or_else(|| panic!("point {p} is ranged at {r} with no reference"));
                    let mut images: Vec<usize> = Vec::new();
                    for &k in reference.images() {
                        let k = k as usize;
                        assert!(k < n_img, "point {p} references image {k} of {n_img}");
                        if !images.contains(&k) {
                            images.push(k);
                        }
                    }
                    assert!(!images.is_empty(), "point {p} references no image");
                    out.refs[p] = images;
                }
            }
        }
        out
    }

    /// Whether point `p` is held, so that it owns no parameters.
    #[inline]
    fn held(&self, p: usize) -> bool {
        self.any_held && self.constraint[p] == PointConstraint::Held
    }

    /// A ranged point's distance where it is finite: the scale on its tangent
    /// Jacobian block, and the radius its position is rebuilt at.
    #[inline]
    fn finite_distance(&self, p: usize) -> Option<f64> {
        (self.any_ranged
            && self.constraint[p] == PointConstraint::Ranged
            && self.distance[p].is_finite())
        .then(|| self.distance[p])
    }
}

/// A ranged point's reference as one round's solve reads it: the reference
/// cameras the round can move, the constant part of the sum from the ones it
/// cannot, and `1/|K|`.
struct DistanceOrigin {
    /// Compact image indices of the reference cameras the round is solving.
    live: Vec<usize>,
    /// `Σ C_k` over reference cameras no kept observation touches, which the
    /// round holds fixed.
    fixed_sum: Vector3<f64>,
    /// `1/|K|` over the whole reference set, live and fixed alike.
    inv_k: f64,
    /// The distance the direction is carried at.
    distance: f64,
}

/// One image's centre `C = −Rᵀ·t`.
#[inline]
fn camera_centre(q: &UnitQuaternion<f64>, t: &Vector3<f64>) -> Vector3<f64> {
    -(q.inverse() * t)
}

/// `[v]ₓ`, the cross-product matrix.
#[inline]
fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

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

/// Re-estimation (rounds after the first): the shared point-estimation
/// operation ([`crate::reconstruction::point_estimation`]) at the round's
/// geometry with the adjustment's settings. `marks` is on with the round's
/// direction mask, `few` is `absent`, and the floor, cheirality and bar rules
/// are off. A finite track rebuilds from all supplied observations by
/// ray-midpoint batch triangulation; a direction track re-estimates in closed
/// form as the normalized mean of its observations' back-rotated rays
/// `R_iᵀ · pixel_to_ray(uv)`. Tracks with fewer than two usable observations,
/// and points with none, become `NaN`; callers refill from their full
/// observation set (the bootstrap's post-BA refill rule). The adjustment's
/// trim, not the operation, decides what a point behind a camera means, which
/// is why cheirality stays off.
///
/// The operation builds its CSR grouping with a **stable** sort, so a track's
/// observations accumulate in the order the caller listed them and the result
/// is defined by the input order rather than by a sort's tie-breaking.
///
/// `cross_floor` is the crossing policy for free points: `None` keeps every
/// free point's representation as it came in (the mask is honoured for the
/// whole solve), and `Some(θ_floor)` re-decides it from the rays at this
/// geometry -- marks off, the floor at `θ_floor`, cheirality on -- and writes
/// the verdict back into `is_dir` for the next linearization. A ranged point is
/// carried by the distance rule at whatever origin its reference resolves to
/// now, and a held point is not re-estimated at all.
#[allow(clippy::too_many_arguments)]
fn reestimate_points(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &mut [[f64; 3]],
    is_dir: &mut [bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    cons: &Constraints,
    cross_floor: Option<f64>,
) {
    let mut quats_wxyz = Vec::with_capacity(quats.len() * 4);
    for q in quats {
        quats_wxyz.extend_from_slice(&[q.w, q.i, q.j, q.k]);
    }
    let mut translations = Vec::with_capacity(trans.len() * 3);
    for t in trans {
        translations.extend_from_slice(&[t.x, t.y, t.z]);
    }
    // A crossing free point is solved from its rays, so it carries no mark; the
    // marks of the points the crossing does not touch are what they were.
    let marks: Vec<bool> = (0..points.len())
        .map(|p| {
            if cross_floor.is_some() && cons.constraint[p] == PointConstraint::Free {
                false
            } else {
                is_dir[p]
            }
        })
        .collect();
    // A ranged point's origin is a function of the poses, read here at the
    // round's own geometry.
    let distances: Option<Vec<PointDistance>> = cons.any_ranged.then(|| {
        (0..points.len())
            .map(|p| {
                if cons.constraint[p] != PointConstraint::Ranged {
                    return PointDistance::NONE;
                }
                let mut o = Vector3::zeros();
                for &k in &cons.refs[p] {
                    o += camera_centre(&quats[k], &trans[k]);
                }
                let n = cons.refs[p].len().max(1) as f64;
                PointDistance {
                    distance: cons.distance[p],
                    origin: [o.x / n, o.y / n, o.z / n],
                }
            })
            .collect()
    });
    let est = estimate_points_from_observations(
        cam,
        ObservationSet {
            uv: uv.as_flattened(),
            obs_image: obs_img,
            obs_point: obs_pt,
            quats_wxyz: &quats_wxyz,
            translations: &translations,
            n_tracks: points.len(),
        },
        Some(&marks),
        PointRules {
            distance: distances.as_deref(),
            floor_rad: cross_floor,
            cheirality: cross_floor.is_some(),
            few: FewObservations::Absent,
            ..Default::default()
        },
    );
    for (p, (row, e)) in points.iter_mut().zip(&est.xyzw).enumerate() {
        // A held point owns its coordinate; the estimate for it is discarded.
        if cons.held(p) {
            continue;
        }
        *row = [e[0], e[1], e[2]];
        // The crossing verdict, where the estimate says anything: an absent
        // track (`NaN`) leaves the representation it had, so a track that
        // momentarily loses its observations does not also change representation.
        if cross_floor.is_some() && cons.constraint[p] == PointConstraint::Free && e[3].is_finite()
        {
            is_dir[p] = e[3] == 0.0;
        }
    }
}

/// Robust cost over the kept observations at a candidate state. `points` are
/// world positions by compact index -- a ranged point's already resolved
/// through its reference and distance; `cp_dir` flags direction points; `s2s`
/// is the per-kept-observation squared loss scale (uniform except where a
/// protected observation widens it).
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

/// Everything one linearization of the solve reads that does not vary from
/// observation to observation: the camera at the current shared state, the
/// current poses and points, and the per-point flags that say what each point's
/// block looks like.
///
/// It exists so the per-observation blocks are built by a function a test can
/// call, which is how the reference-camera columns of a ranged point are
/// checked against a difference of the residual they claim to differentiate.
struct LinState<'a> {
    cam: &'a CameraIntrinsics,
    /// Whether the model carries an analytic pixel Jacobian.
    analytic: bool,
    cx: f64,
    cy: f64,
    f: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    n_coeffs: usize,
    d_max: f64,
    radial: SplineRadial,
    /// Compact poses.
    q: &'a [UnitQuaternion<f64>],
    t: &'a [Vector3<f64>],
    /// Compact world positions -- a ranged point's already resolved through its
    /// reference and distance.
    xp: &'a [Vector3<f64>],
    /// Tangent bases of the points parameterized on the sphere.
    bases: &'a [(Vector3<f64>, Vector3<f64>)],
    cp_dir: &'a [bool],
    cp_held: &'a [bool],
    cp_origin: &'a [Option<DistanceOrigin>],
    /// Every supplied observation's pixel, indexed by the caller's own index.
    uv: &'a [[f64; 2]],
    /// Width of the reduced system's shared-parameter head.
    n_shared: usize,
}

impl LinState<'_> {
    /// First pose slot of compact image `ci` in the reduced camera system.
    #[inline]
    fn img_slot(&self, ci: usize) -> usize {
        self.n_shared + 6 * ci
    }
}

/// Linearize one observation: its weighted residual and the camera-side,
/// point-side and reference-camera blocks that differentiate it.
///
/// `k` is the observation's index in the caller's arrays, `ci` and `cp` its
/// compact image and point, and `s2` the squared loss scale it is weighted at.
fn observation_blocks<const CAM_COLS: usize>(
    st: &LinState<'_>,
    k: usize,
    ci: usize,
    cp: usize,
    s2: f64,
) -> ObsBlocks<CAM_COLS> {
    // First pose column within an observation's camera block.
    let pose_c = CAM_COLS - 6;
    let (q, t, xp, uv, cam) = (st.q, st.t, st.xp, st.uv, st.cam);
    let f = st.f;
    let dir = st.cp_dir[cp];
    let rot_pt = q[ci] * xp[cp];
    let p_cam = if dir { rot_pt } else { rot_pt + t[ci] };
    let mut res = Vector2::new(INVALID_RESIDUAL, 0.0);
    let mut cam_j = SMatrix::<f64, 2, CAM_COLS>::zeros();
    let mut pt_j = SMatrix::<f64, 2, 3>::zeros();
    let mut ref_cols: Vec<(usize, Vector2<f64>)> = Vec::new();
    // Column indices: `[f, k1, (spline), δθ×3, δt×3]`. Spline
    // slots start at the pinned K1_SLOT dummy and are pointed at
    // their coefficient's shared slot below, where the
    // observation actually carries one.
    let mut idx = [K1_SLOT; CAM_COLS];
    idx[F_SLOT] = F_SLOT;
    let o = st.img_slot(ci);
    for (j, slot) in idx[pose_c..].iter_mut().enumerate() {
        *slot = o + j;
    }
    // A non-finite point (protected observations only — the trim
    // never excludes them) keeps the penalized residual and zero
    // Jacobian rows: penalized, never steering.
    let proj = if xp[cp].x.is_finite() && xp[cp].y.is_finite() && xp[cp].z.is_finite() {
        project_with_jac(cam, p_cam, st.analytic)
    } else {
        None
    };
    if let Some(((u, v), jp)) = proj {
        res = Vector2::new(u - uv[k][0], v - uv[k][1]);
        let jp = SMatrix::<f64, 2, 3>::from_rows(&[
            SMatrix::<f64, 1, 3>::from_row_slice(&jp[0]),
            SMatrix::<f64, 1, 3>::from_row_slice(&jp[1]),
        ]);
        if st.opt_f {
            // ∂(u, v)/∂f. Exact for every model the `opt_f` gate
            // admits: the focal is a pure multiplier of an
            // `f`-independent distorted coordinate, so the
            // derivative is that coordinate.
            cam_j[(0, F_SLOT)] = (u - st.cx) / f;
            cam_j[(1, F_SLOT)] = (v - st.cy) / f;
        }
        if st.opt_k1 {
            // ∂(u, v)/∂k1 = f·θ³·û — direction rows included,
            // they project through the same map.
            let (du, dv) = k1_column(f, p_cam);
            cam_j[(0, K1_SLOT)] = du;
            cam_j[(1, K1_SLOT)] = dv;
        }
        if st.opt_bspline {
            // ∂(u, v)/∂cᵢ = f·Bᵢ(θ)·û for the ≤ 4 active basis
            // functions — direction rows included, they project
            // through the same map. Gauge-anchored functions
            // (full index < 2) carry no coefficient: their slot
            // keeps the pinned K1_SLOT dummy and their column
            // stays exactly zero.
            let (first, cols) = bspline_columns(f, st.n_coeffs, st.d_max, st.radial, p_cam);
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
            // the third slot stays exactly zero). A held direction
            // owns no parameters, so it takes no point block.
            if !st.cp_held[cp] {
                let (b1, b2) = st.bases[cp];
                let col0 = jp * (r_mat * b1);
                let col1 = jp * (r_mat * b2);
                pt_j.set_column(0, &col0);
                pt_j.set_column(1, &col1);
            }
        } else if let Some(origin) = &st.cp_origin[cp] {
            // A ranged point at a finite distance. Translation
            // block: identity, as for any finite point.
            cam_j.fixed_view_mut::<2, 3>(0, pose_c + 3).copy_from(&jp);
            // Point block: r·J_X·B(d), the direction's tangent
            // columns scaled by the distance the caller holds.
            let (b1, b2) = st.bases[cp];
            let col0 = jp * (r_mat * b1) * origin.distance;
            let col1 = jp * (r_mat * b2) * origin.distance;
            pt_j.set_column(0, &col0);
            pt_j.set_column(1, &col1);
            // Reference cameras: X moves with O, so every
            // observation of the point also lands
            // J_X·∂O/∂pose_k / |K| in each reference camera's own
            // block, with ∂C/∂t = −Rᵀ and ∂C/∂ω = −Rᵀ·[t]ₓ under
            // this kernel's R ← exp(ω)·R update.
            let jx = jp * r_mat;
            for &c in &origin.live {
                let rct = q[c].to_rotation_matrix().into_inner().transpose();
                let d_omega = -(rct * skew(t[c])) * origin.inv_k;
                let d_trans = -rct * origin.inv_k;
                let b_omega = jx * d_omega;
                let b_trans = jx * d_trans;
                let o = st.img_slot(c);
                for j in 0..3 {
                    ref_cols.push((o + j, Vector2::new(b_omega[(0, j)], b_omega[(1, j)])));
                }
                for j in 0..3 {
                    ref_cols.push((o + 3 + j, Vector2::new(b_trans[(0, j)], b_trans[(1, j)])));
                }
            }
        } else {
            // Translation block: identity.
            cam_j.fixed_view_mut::<2, 3>(0, pose_c + 3).copy_from(&jp);
            // Point block: ∂p_cam/∂X = R. A held point owns no
            // parameters and takes none.
            if !st.cp_held[cp] {
                pt_j.copy_from(&(jp * r_mat));
            }
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
        for (_, col) in ref_cols.iter_mut() {
            col[row] *= js;
        }
    }
    ObsBlocks {
        cp,
        res,
        cam_j,
        pt_j,
        idx,
        ref_cols,
    }
}

/// One robust sparse LM solve over the kept observations with mixed finite
/// and direction points. Direction points use 2-DOF tangent-plane parameters
/// stored in the first two slots of the uniform 3-wide point block (the third
/// slot carries exact zeros and is pinned at the Schur inversion); images
/// whose kept observations carry no translation Jacobian have their
/// translation slots pinned in the reduced system (frozen for the round).
///
/// `cons` carries the point constraints. A ranged point at a finite distance takes
/// the same 2-DOF tangent parameters as a direction -- its state here IS that
/// direction, its position `O(pose) + r·d` resolved wherever a position is
/// wanted -- with its Jacobian block scaled by the distance and its reference
/// cameras' own blocks accumulated alongside the observing camera's. A held
/// point takes no point block, no Schur block and no update, so it comes back
/// exactly as it went in.
///
/// `CAM_COLS` selects the per-observation camera-block width:
/// [`BASE_CAM_COLS`] for every solve without the spline release (the
/// original layout, byte for byte), [`BSPLINE_CAM_COLS`] when the staged
/// loop releases a radial spline — the reduced system then carries one
/// shared slot per coefficient (`n_shared = 2 + n_coeffs`, still
/// dynamic) while each observation's block stays compile-time sized at the
/// spline's local support. `bspline0` is the current coefficient vector
/// (read-only outside the spline instantiation, where the camera's own
/// fixed spline rides along inside `cam0`).
///
/// `progress` counts the iterations, names the stages inside one of them at the
/// detail level, and is polled at the top of each: an iteration boundary is
/// where this solve holds a consistent state, since a candidate step is only
/// ever written once it has improved the cost. A stopped solve scatters back
/// what the last accepted step left, which is what it would have returned had
/// the budget run out there.
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
    cons: &Constraints,
    opt_f: bool,
    opt_k1: bool,
    loss_scale: f64,
    max_iters: usize,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    progress: &Progress<'_>,
) -> (f64, f64, Vec<f64>) {
    // The spline instantiation is selected by width; the staged loop only
    // requests it for a released, well-formed spline.
    let opt_bspline = CAM_COLS == BSPLINE_CAM_COLS;
    debug_assert!(
        !(opt_bspline && opt_k1),
        "opt_k1 and opt_bspline live on different models"
    );
    let (n_coeffs, d_max, radial) = match cam0.model.radial_spline() {
        Some((_, d_max, radial)) if opt_bspline => (bspline0.len(), d_max, radial),
        _ => (0, 0.0, SplineRadial::IncidenceAngle),
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
    // A held point owns no parameters: no point block, no Schur block, no
    // update.
    let cp_held: Vec<bool> = pt_ids.iter().map(|&p| cons.held(p)).collect();
    // A ranged point at a finite distance is parameterized like a direction --
    // two tangent DOFs -- with its Jacobian block scaled by that distance, so
    // the two families share the tangent basis, the 2×2 Schur block and the
    // normalizing update.
    let cp_distance: Vec<Option<f64>> = pt_ids.iter().map(|&p| cons.finite_distance(p)).collect();
    let cp_tangent: Vec<bool> = (0..n_pt)
        .map(|c| cp_dir[c] || cp_distance[c].is_some())
        .collect();

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

    // Ranged points: their reference set, split into the images this round
    // solves and the ones it does not (a reference no kept observation touches
    // cannot move, so its centre folds into a constant). Their state is the
    // direction `d`, so the incoming position is read as one -- a caller whose
    // position does not sit at exactly `r` from the reference is snapped onto
    // the sphere the distance names.
    let cp_origin: Vec<Option<DistanceOrigin>> = pt_ids
        .iter()
        .enumerate()
        .map(|(c, &p)| {
            let distance = cp_distance[c]?;
            let mut live = Vec::new();
            let mut fixed_sum = Vector3::zeros();
            for &k in &cons.refs[p] {
                match ci_of.get(&k) {
                    Some(&ci) => live.push(ci),
                    None => fixed_sum += camera_centre(&quats[k], &trans[k]),
                }
            }
            Some(DistanceOrigin {
                live,
                fixed_sum,
                inv_k: 1.0 / cons.refs[p].len() as f64,
                distance,
            })
        })
        .collect();
    let has_ranged = cp_origin.iter().any(Option::is_some);
    // The reference origin at a candidate state.
    let origin_of = |o: &DistanceOrigin, q: &[UnitQuaternion<f64>], t: &[Vector3<f64>]| {
        let mut s = o.fixed_sum;
        for &c in &o.live {
            s += camera_centre(&q[c], &t[c]);
        }
        s * o.inv_k
    };
    // The world positions a state stands for: the state itself, except that a
    // ranged point holds its direction and sits at `O(pose) + r·d`.
    let positions = |x: &[Vector3<f64>], q: &[UnitQuaternion<f64>], t: &[Vector3<f64>]| {
        let mut out = x.to_vec();
        for (c, o) in cp_origin.iter().enumerate() {
            if let Some(o) = o {
                out[c] = origin_of(o, q, t) + o.distance * x[c];
            }
        }
        out
    };
    for (c, o) in cp_origin.iter().enumerate() {
        if let Some(o) = o {
            x[c] = (x[c] - origin_of(o, &q, &t)).normalize();
        }
    }
    // Reduced camera system: [f | k1 | (spline coefficients) | 6 per image];
    // the scalar shared slots are always present (pinned when unreleased) to
    // keep the indexing uniform, the coefficient slots only under the
    // spline release.
    let n_shared = 2 + n_coeffs;
    let d = n_shared + 6 * n_im;
    // First pose slot of compact image `ci` in the reduced camera system.
    let img_slot = |ci: usize| n_shared + 6 * ci;
    // The camera at a candidate shared state. Off the spline instantiation
    // this is exactly the scalar builder (a fixed spline
    // rides along inside `cam0` untouched).
    let build_cam = |fv: f64, k1v: f64, bsv: &[f64]| {
        if opt_bspline {
            cam0.with_focal_bspline(fv, bsv)
        } else {
            cam0.with_focal_k1(fv, k1v)
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
    // The robust cost at a candidate state, read through the positions above.
    let cost_at = |cam: &CameraIntrinsics,
                   q: &[UnitQuaternion<f64>],
                   t: &[Vector3<f64>],
                   x: &[Vector3<f64>]| {
        let owned;
        let xp: &[Vector3<f64>] = if has_ranged {
            owned = positions(x, q, t);
            &owned
        } else {
            x
        };
        robust_cost(cam, q, t, xp, &cp_dir, uv, kept, &obs_ci, &obs_cp, &s2s)
    };
    let mut lambda = 1e-3;
    let mut tiny_steps = 0usize;
    let mut cam = build_cam(f, k1, &bspline);
    let mut prev_cost = cost_at(&cam, &q, &t, &x);

    let analytic = cam.model.supports_pixel_jacobian();
    for iter in 0..max_iters {
        // An iteration boundary is this solve's stopping point: every state
        // below is either the last accepted step's or a candidate nothing has
        // been written from.
        if progress.is_cancelled() {
            break;
        }
        progress.count(iter as u64, Some(max_iters as u64), "iteration");
        let linearise = progress.detail_phase("linearise");
        // ── Linearize at the current state ───────────────────────────────
        // Tangent bases B(d) = [b1 | b2] for the direction points, rebuilt at
        // each linearization.
        let bases: Vec<(Vector3<f64>, Vector3<f64>)> = x
            .iter()
            .zip(&cp_tangent)
            .map(|(xd, &dir)| {
                if dir {
                    tangent_basis(xd)
                } else {
                    (Vector3::zeros(), Vector3::zeros())
                }
            })
            .collect();
        // A ranged point's state is a direction; its position is where the
        // reference and the distance put it at this linearization.
        let owned_positions;
        let xp: &[Vector3<f64>] = if has_ranged {
            owned_positions = positions(&x, &q, &t);
            &owned_positions
        } else {
            &x
        };
        let (cx, cy) = cam.principal_point();
        let blocks: Vec<ObsBlocks<CAM_COLS>> = {
            let st = LinState {
                cam: &cam,
                analytic,
                cx,
                cy,
                f,
                opt_f,
                opt_k1,
                opt_bspline,
                n_coeffs,
                d_max,
                radial,
                q: &q,
                t: &t,
                xp,
                bases: &bases,
                cp_dir: &cp_dir,
                cp_held: &cp_held,
                cp_origin: &cp_origin,
                uv,
                n_shared,
            };
            kept.iter()
                .enumerate()
                .map(|(kk, &k)| observation_blocks(&st, k, obs_ci[kk], obs_cp[kk], s2s[kk]))
                .collect()
        };
        drop(linearise);

        // ── Accumulate the normal-equation blocks ────────────────────────
        let equations = progress.detail_phase("normal equations");
        let mut h_cc = DMatrix::<f64>::zeros(d, d);
        let mut g_c = DVector::<f64>::zeros(d);
        let mut v_pp: Vec<Matrix3<f64>> = vec![Matrix3::zeros(); n_pt];
        let mut g_p: Vec<Vector3<f64>> = vec![Vector3::zeros(); n_pt];
        let mut w_cp: Vec<SMatrix<f64, CAM_COLS, 3>> = Vec::with_capacity(blocks.len());
        // The reference columns' own rows of `W`, parallel to `w_cp`; empty
        // unless a ranged point is in the solve.
        let mut w_ref: Vec<Vec<(usize, Vector3<f64>)>> = if has_ranged {
            Vec::with_capacity(blocks.len())
        } else {
            Vec::new()
        };
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
            // The reference columns are extra columns of the same camera-side
            // Jacobian, so they carry the same three products: their own square,
            // their cross terms with the observing camera's block (both
            // triangles, which is what makes an observing camera that is its own
            // reference come out right), and their gradient entry.
            for &(ij, cj) in &b.ref_cols {
                g_c[ij] += cj.dot(&b.res);
                for (a, &ia) in idx.iter().enumerate() {
                    let v = cj[0] * b.cam_j[(0, a)] + cj[1] * b.cam_j[(1, a)];
                    h_cc[(ia, ij)] += v;
                    h_cc[(ij, ia)] += v;
                }
                for &(il, cl) in &b.ref_cols {
                    h_cc[(ij, il)] += cj.dot(&cl);
                }
            }
            v_pp[b.cp] += b.pt_j.transpose() * b.pt_j;
            g_p[b.cp] += b.pt_j.transpose() * b.res;
            w_cp.push(b.cam_j.transpose() * b.pt_j);
            if has_ranged {
                w_ref.push(
                    b.ref_cols
                        .iter()
                        .map(|&(ij, cj)| (ij, b.pt_j.transpose() * cj))
                        .collect(),
                );
            }
        }

        drop(equations);

        // ── Damping ladder: re-damp and re-solve from this linearization ──
        let ladder = progress.detail_phase("damping ladder");
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
                // A held point has no point block at all; the identity keeps
                // the uniform indexing while its zero gradient and zero `W`
                // make every contribution exactly zero.
                if cp_held[p] {
                    v_inv.push(Matrix3::identity());
                    continue;
                }
                let mut vd = *v;
                for dd in 0..3 {
                    vd[(dd, dd)] += lambda * v[(dd, dd)].max(1e-12);
                }
                if cp_tangent[p] {
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
                // A held point is eliminated by having no block to eliminate.
                if cp_held[p] {
                    continue;
                }
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
                // The reference columns take the same elimination, over the
                // three blocks the fixed loop above does not reach: reference
                // against camera, camera against reference, and reference
                // against reference.
                if has_ranged {
                    for &a in obs {
                        for &(ir, ra) in &w_ref[a] {
                            let rv: Vector3<f64> = v_inv[p].transpose() * ra;
                            g_red[ir] -= ra.dot(&y);
                            for &b in obs {
                                let ib = &blocks[b].idx;
                                let m = w_cp[b] * rv;
                                for (c, &ic) in ib.iter().enumerate() {
                                    s[(ir, ic)] -= m[c];
                                }
                                for &(jc, rb) in &w_ref[b] {
                                    s[(ir, jc)] -= rv.dot(&rb);
                                }
                            }
                        }
                        let wa = &w_cp[a];
                        let ia = &blocks[a].idx;
                        for &b in obs {
                            for &(jc, rb) in &w_ref[b] {
                                let m = wa * (v_inv[p] * rb);
                                for (r, &ir) in ia.iter().enumerate() {
                                    s[(ir, jc)] -= m[r];
                                }
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
                if !bspline_step_admissible(&pc, d_max) {
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
                // A held point does not move.
                if cp_held[p] {
                    continue;
                }
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
                    if has_ranged {
                        for &(ir, ra) in &w_ref[a] {
                            wt_dc += ra * delta[ir];
                        }
                    }
                }
                let dxp = v_inv[p] * (g_p[p] + wt_dc);
                if cp_tangent[p] {
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
            let new_cost = cost_at(&cam_cand, &q_cand, &t_cand, &x_cand);
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
        drop(ladder);
        if !improved || lambda.is_infinite() {
            break;
        }
    }

    // Scatter the compact state back. A ranged point comes back as the
    // position its distance and its reference put it at, re-read at the poses
    // the round settled on; a held point is not written at all.
    let owned_final;
    let xf: &[Vector3<f64>] = if has_ranged {
        owned_final = positions(&x, &q, &t);
        &owned_final
    } else {
        &x
    };
    for (c, &i) in img_ids.iter().enumerate() {
        quats[i] = q[c];
        trans[i] = t[c];
    }
    for (c, &p) in pt_ids.iter().enumerate() {
        if cp_held[c] {
            continue;
        }
        points[p] = [xf[c].x, xf[c].y, xf[c].z];
    }
    (f, k1, bspline)
}

/// The staged loop: direction-aware residuals, trims, re-estimation, and the
/// `min_obs` floor over every trim survivor. With an all-`false` `is_dir`
/// every direction branch is skipped and this is the finite-only solve.
#[allow(clippy::too_many_arguments)]
fn bundle_adjust_staged(
    cam: &CameraIntrinsics,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    is_dir: &mut [bool],
    cons: &Constraints,
    free_points: FreePointPolicy,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
    progress: &Progress<'_>,
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
    // rejects it loudly first). The two sfmtool spline models pass the test
    // the same way SIMPLE_RADIAL_FISHEYE does: their radial spline is
    // dimensionless and rides on the ray-derived radial coordinate
    // (`x_d = (θ + δ(θ))·ûx` for SFMTOOL_FISHEYE, `x_d = (ρ + δ(ρ))·ûx` with
    // `ρ = ρ_xy/rz` for SFMTOOL_PINHOLE), so `f` never appears inside the
    // distorted coordinate and `(u − cx)/f` stays exact. `CameraIntrinsics::with_focal`
    // mirrors this gate.
    let opt_f = opt_f
        && matches!(
            cam.model,
            CameraModel::SimplePinhole { .. }
                | CameraModel::EquidistantFisheye { .. }
                | CameraModel::SimpleRadialFisheye { .. }
                | CameraModel::SfmtoolFisheye { .. }
                | CameraModel::SfmtoolPinhole { .. }
        );
    // The curvature rung exists on exactly one model: `SIMPLE_RADIAL_FISHEYE`
    // is the only one whose single radial coefficient acts on the ray's own
    // `θ`, which is what makes `∂(u, v)/∂k1 = f·θ³·û` exact. Same degrade.
    let opt_k1 = opt_k1 && matches!(cam.model, CameraModel::SimpleRadialFisheye { .. });
    // The spline rung exists on the two models that carry a spline,
    // `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`, whose dimensionless
    // coefficients act on the ray's own radial coordinate `d`, making
    // `∂(u, v)/∂cᵢ = f·Bᵢ(d)·û` exact — and only when the spline is actually
    // defined (at least `MIN_BSPLINE_COEFFS` coefficients on a positive finite
    // domain end; anything shorter evaluates as the identity and carries
    // nothing to release). Same degrade; `opt_k1` and `opt_bspline` are
    // therefore naturally exclusive, which the spline instantiation's
    // pinned-K1 dummy slot relies on.
    let opt_bspline = opt_bspline
        && cam
            .model
            .radial_spline()
            .is_some_and(|(bspline, d_max, _)| {
                bspline.len() >= MIN_BSPLINE_COEFFS && d_max.is_finite() && d_max > 0.0
            });

    let mut f = cam.focal_lengths().0;
    let mut k1 = match cam.model {
        CameraModel::SimpleRadialFisheye {
            radial_distortion_k1,
            ..
        } => radial_distortion_k1,
        _ => 0.0,
    };
    let mut bspline: Vec<f64> = cam
        .model
        .radial_spline()
        .map(|(bspline, _, _)| bspline.to_vec())
        .unwrap_or_default();

    // A share of the range per round. Even shares, because the rounds run the
    // same solve over a tightening trim and none of them is predictably the
    // expensive one; an estimate, like every other set of weights.
    // The schedule belongs to the solve rather than to any one round: the
    // rounds fold into a single row for a reader, and a trim that is true of
    // only the round that happened to run last is worse than no trim at all.
    //
    // Not said at all for an empty schedule, which is how a caller asks for the
    // residuals at the state it handed over: no round runs, so there is no
    // schedule to report and "0 rounds" is noise in front of the answer.
    if !schedule.is_empty() {
        progress_info!(
            progress,
            "{} rounds, trim {} px",
            schedule.len(),
            schedule
                .iter()
                .map(|stage| format!("{}", stage.trim_px))
                .collect::<Vec<_>>()
                .join("/")
        );
    }
    let rounds = progress.split_evenly(schedule.len());
    for ((rnd, stage), p_round) in schedule.iter().enumerate().zip(rounds) {
        // Between rounds the poses and the points hold what the last round
        // settled on, which is a whole answer to hand back.
        if progress.is_cancelled() {
            break;
        }
        let round = p_round.phase("round");
        let cam_now = if opt_bspline {
            cam.with_focal_bspline(f, &bspline)
        } else {
            cam.with_focal_k1(f, k1)
        };
        // The noise floor a crossing free point is classified at: the parallax
        // this stage's own residual scale cannot tell from noise, `c·s/f` in
        // radians, so a wide-baseline stage keeps more tracks finite than a
        // tight one and the boundary moves with the focal as the release walks
        // it.
        let cross_floor = free_points.cross.then(|| {
            let (fx, fy) = cam_now.focal_lengths();
            free_points.noise_floor_scale * stage.loss_scale / (0.5 * (fx + fy))
        });
        if rnd > 0 {
            reestimate_points(
                &cam_now,
                quats,
                trans,
                points,
                is_dir,
                uv,
                obs_img,
                obs_pt,
                cons,
                cross_floor,
            );
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
        // The degenerate floor counts every trim survivor, finite and
        // direction alike: the floor asks whether the round retained enough
        // evidence to solve on, and a direction vouches for the rotations and
        // the camera model exactly as a finite observation does. Translation
        // observability is handled separately — an image whose survivors are
        // all directions has its translation frozen for the round.
        if kept.len() < min_obs {
            // Degenerate (e.g. a wildly wrong focal): state passes through.
            return BundleAdjustment {
                focal: f,
                k1,
                bspline,
                residual_norms: vec![f64::INFINITY; n_obs],
                point_at_infinity: is_dir.to_vec(),
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
                cons,
                opt_f,
                opt_k1,
                stage.loss_scale,
                max_iters,
                protected,
                protected_loss_scale,
                &round,
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
                cons,
                opt_f,
                opt_k1,
                stage.loss_scale,
                max_iters,
                protected,
                protected_loss_scale,
                &round,
            )
        };
        drop(round);
        progress.count(rnd as u64 + 1, Some(schedule.len() as u64), "round");
    }

    let cam_final = if opt_bspline {
        cam.with_focal_bspline(f, &bspline)
    } else {
        cam.with_focal_k1(f, k1)
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
        point_at_infinity: is_dir.to_vec(),
    }
}

#[cfg(test)]
mod tests;
