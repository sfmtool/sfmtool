// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Staged bundle adjustment for images taken through one or more cameras.
//!
//! Jointly refines world-to-camera poses, world points, and optionally each
//! camera's focal length, radial coefficient or spline, and principal point by
//! minimizing soft-L1 pixel reprojection error over a trim schedule with
//! inter-round retriangulation. It is the multi-view generalization of
//! [`crate::geometry::pose_refine`]. The design is in
//! `specs/core/geometry/bundle-adjustment.md`.
//!
//! Canonical camera frame throughout (the camera looks along `−Z`; a point in
//! front has `z < 0`). Each Levenberg–Marquardt step is taken over a local
//! `SO(3) × ℝ³` perturbation per image, `ℝ³` per point, and the optional
//! per-camera lens parameters (the two scalars or the radial spline
//! coefficients, and the principal point), with analytic Jacobians; points are eliminated by a Schur
//! complement and the dense reduced camera system is solved by LU.

use std::borrow::Cow;

use nalgebra::{
    DMatrix, DVector, Matrix2, Matrix3, Point3, SMatrix, UnitQuaternion, Vector2, Vector3,
};
use rayon::prelude::*;

use crate::analysis::reprojection_noise::{
    camera_keypoint_resolution_px, gated_reprojection_noise, ObservationResidual,
    ReprojectionNoise, OUTLIER_GATE,
};
use crate::camera::distortion::bspline::{
    basis_at, bspline_is_monotone, delta_and_deriv as bspline_delta_and_deriv, smoothing_weight,
    BSPLINE_SUPPORT, MIN_BSPLINE_COEFFS,
};
use crate::camera::intrinsics::{ImageIntrinsicsList, IntrinsicAdjustment, SplineRadial};
use crate::camera::CameraModel;
use crate::geometry::pose_refine::project_with_jac;
use crate::progress::Progress;
use crate::progress_info;
use crate::reconstruction::triangulation::points::{
    tangent_basis, triangulate_points_through_cameras, FewObservations, ObservationSet,
    PointDistance, PointRules,
};
use crate::reconstruction::triangulation::{
    bearing_score, fit_point_and_bearing, is_finite, observed_ray, PointBearingFitOptions,
    DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
};
use crate::CameraIntrinsics;

/// A point behind the camera / outside the model domain contributes this pixel
/// residual per component — large enough to be trimmed, finite so the robust
/// cost stays well-posed (matches `reprojection_residuals` / `pose_refine`).
const INVALID_RESIDUAL: f64 = 1e6;

/// The trim's in-front floor for a finite observation, as a fraction of the
/// round's scene scale ([`in_front_floor`]). A point whose in-front measure is
/// at or under it sits on its camera's centre, where the projection is
/// singular; nothing a camera can image in focus is a millionth of the scene's
/// median depth away from it.
const IN_FRONT_FLOOR_FRACTION: f64 = 1e-6;

/// One round of the trim schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaSchedule {
    /// Pre-round trim threshold on the reprojection residual norm, px.
    /// `f64::INFINITY` trims nothing on the residual: every observation with a
    /// valid projection is kept. An observation whose projection fails is
    /// trimmed whatever the threshold.
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

/// What the solve owns of a point, orthogonal to the representation the point
/// currently carries.
///
/// See "Point constraints" in `specs/core/geometry/bundle-adjustment.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PointConstraint {
    /// The solve owns the point: its position moves, and under
    /// [`FreePointPolicy::cross`] it is solved in inverse depth and its
    /// representation is decided from its rays at the end of the solve.
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
    /// A reference names an image past the solve's `n_img` images.
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
/// [`FreePointPolicy::default`] crosses: every caller in the crate passes it,
/// so free points are solved in inverse depth and decided at the end of each
/// solve unless a caller opts out. [`FreePointPolicy::NO_CROSS`] is the opt-out,
/// the crossing off: the caller's `point_at_infinity` mask is honoured for the
/// whole solve, so every free point keeps the representation it is handed in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreePointPolicy {
    /// Solve every free point in inverse depth about a fixed anchor, so that it
    /// can move between near and infinity within a round, and decide at the end
    /// of the solve whether it is stored as a position or as a direction, on the
    /// point-or-bearing test at the noise level the final round's residuals
    /// measure. See "Free points: inverse depth and the storage decision" in
    /// `specs/core/geometry/bundle-adjustment.md`.
    pub cross: bool,
}

impl FreePointPolicy {
    /// Free points are solved in inverse depth and stored as the storage
    /// decision says: the default.
    pub const CROSS: FreePointPolicy = FreePointPolicy { cross: true };
    /// Free points keep the representation the caller handed in, for the
    /// whole solve: the opt-out, `cross: false`.
    pub const NO_CROSS: FreePointPolicy = FreePointPolicy { cross: false };
}

impl Default for FreePointPolicy {
    fn default() -> Self {
        Self::CROSS
    }
}

/// The end-of-solve storage decision under [`FreePointPolicy::cross`]: the
/// noise level the free points were decided at and what the decision changed
/// against the caller's input representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreePointDecision {
    /// The noise level the free points are decided at: the RMS per-axis
    /// residual, in pixels, over the final round's kept observations of finite
    /// points at the state the solve ended at, outliers gated out as
    /// [`crate::analysis::reprojection_noise::OUTLIER_GATE`] does, and never
    /// under the cameras' keypoint resolution. `None` where the final round
    /// kept no observation of a finite point.
    pub sigma_px: Option<f64>,
    /// How many observations `sigma_px` is measured over.
    pub observation_count: usize,
    /// How many were left out as outliers.
    pub outlier_count: usize,
    /// Whether the test was read. It is unless there is no level to read it at
    /// or the solve was cancelled, and each free point is then stored as the
    /// solve left it: a position where its inverse depth is positive and a
    /// direction where it is zero.
    pub decided: bool,
    /// Whether the final round met its convergence test, rather than stopping
    /// on its iteration budget or settling with a camera's distortion held at
    /// its step guard (a `k1` at the fold of the imaged field, a spline at its
    /// monotonicity limit) while the step that would lower the cost further was
    /// one the guard forbids. The decision is read either way; a round that
    /// stopped short measures a level that still carries pose error, which errs
    /// toward a direction, and this says so.
    pub converged: bool,
    /// Free points the caller handed in as directions that are stored finite.
    pub to_finite: usize,
    /// Free points the caller handed in as positions that are stored as
    /// directions.
    pub to_direction: usize,
    /// Free points with an estimate that the test could not score, because
    /// the final round kept fewer than two usable observations of them (the
    /// trim or `min_track` dropped the rest). Each is stored in the
    /// representation the caller handed it in -- as the solve left it where
    /// that is the same, and as handed in otherwise; one handed in with no
    /// estimate keeps the representation the re-estimation gave it -- and is
    /// counted in neither `to_finite` nor `to_direction`. Zero where the test
    /// was not read.
    pub unscored: usize,
}

/// The cameras of a solve, and which of them took each image.
///
/// The camera list and the image-to-camera column only mean something
/// together, so they travel as one argument and are checked in one place: an
/// empty camera list, an `image_camera` whose length is not the image count,
/// or an index past the list is a caller error, and [`bundle_adjust`] panics
/// on it like on its other shape checks.
///
/// `image_camera` is a [`Cow`] so that [`BaCameras::shared`] can own the
/// all-zero column it builds, while a caller with a column of its own lends it
/// (`image_camera: column.as_slice().into()`).
#[derive(Clone, Debug)]
pub struct BaCameras<'a> {
    /// One entry per camera; each carries its own model, parameters and
    /// initial focal.
    pub cameras: &'a [CameraIntrinsics],
    /// One entry per image: the index into `cameras` of the camera that took
    /// it.
    pub image_camera: Cow<'a, [u32]>,
    /// What each camera may release, one entry per camera, or `None` for every
    /// camera taking the solve's release flags as they are.
    ///
    /// An entry narrows the flags for its camera: the focal is released under
    /// `opt_f && focal`, `k1` under `opt_k1 && distortion`, the spline under
    /// `opt_bspline && distortion` and the principal point under
    /// `opt_pp && principal_point`, each still only where the camera's model
    /// admits it. So a caller that decides the releases camera by camera passes
    /// every flag on and states the choice here. A length other than the camera
    /// count is a caller error, checked with the others.
    pub releases: Option<&'a [CameraRelease]>,
    /// Each image's intrinsic adjustment, one per image, or `None` for every
    /// image read through its camera exactly.
    ///
    /// Every observation is projected, differentiated, trimmed and re-estimated
    /// through its camera with its image's adjustment applied, while a released
    /// focal, distortion or principal point is still the camera's shared one,
    /// the adjustment applied on top of it. The solve holds the adjustments
    /// unless the image's camera releases them per image (`opt_image_f`,
    /// `opt_image_pp`), and returns them in
    /// [`BundleAdjustment::image_adjustments`]. A length other than the image
    /// count is a caller error, checked with the others. All-zero adjustments
    /// with no per-image release are the same solve as `None`, to the bit.
    pub image_adjustments: Option<Cow<'a, [IntrinsicAdjustment]>>,
}

/// What the solve may move of one camera's lens, besides the poses and points
/// every solve moves.
///
/// The per-camera half of a release decision: [`BaCameras::releases`] holds one
/// per camera, and the reconstruction-level adjustment takes its list in this
/// type too. A camera with none of the five is held, which is also `Default`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CameraRelease {
    /// Release the focal length.
    pub focal: bool,
    /// Release the lens distortion: `k1` on `SIMPLE_RADIAL_FISHEYE`, the radial
    /// spline on `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`.
    pub distortion: bool,
    /// Release the principal point, `(cx, cy)`, on any model.
    pub principal_point: bool,
    /// Release each image's focal scale, the `focal` of its
    /// [`IntrinsicAdjustment`], on the models the focal itself is released on.
    /// The camera's shared focal is held during the solve and takes the images'
    /// mean after every round, so this moves the shared focal as well, and with
    /// [`Self::focal`] it is the same release.
    pub image_focal: bool,
    /// Release each image's principal-point offset, on any model. The camera's
    /// shared principal point is held during the solve and takes the images'
    /// mean after every round, so this moves the shared principal point as
    /// well, and with [`Self::principal_point`] it is the same release.
    pub image_principal_point: bool,
}

impl CameraRelease {
    /// Nothing released: the camera is held.
    pub const HELD: CameraRelease = CameraRelease {
        focal: false,
        distortion: false,
        principal_point: false,
        image_focal: false,
        image_principal_point: false,
    };
    /// The focal alone.
    pub const FOCAL: CameraRelease = CameraRelease {
        focal: true,
        ..Self::HELD
    };
    /// The focal and the lens distortion.
    pub const FOCAL_AND_DISTORTION: CameraRelease = CameraRelease {
        focal: true,
        distortion: true,
        ..Self::HELD
    };
    /// Everything a camera can release: what a camera takes when
    /// [`BaCameras::releases`] is `None`, so that the solve's flags apply to it
    /// as they are.
    pub const ALL: CameraRelease = CameraRelease {
        focal: true,
        distortion: true,
        principal_point: true,
        image_focal: true,
        image_principal_point: true,
    };
}

/// The widths of one camera's prior on its images' intrinsic adjustments: how
/// far, as a Gaussian's standard deviation, each image's adjustment is expected
/// to lie from the mean of the camera's images.
///
/// `focal` is the width of the log focal scale `t = ln(1 + s)`, which reads as
/// a fraction of the focal for small values, as the scale `s` does, since
/// `t ≈ s`; `principal_point` is the principal-point offset's, in the
/// short-side unit `min(width, height)` and the same on both axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePriorWidths {
    /// The width of the log focal scale `ln(1 + s)`, read as a fraction.
    pub focal: f64,
    /// The principal-point offset's width, in the short-side unit.
    pub principal_point: f64,
}

/// How the widths of the prior on a per-image release are chosen.
///
/// By default each camera's widths are derived from the solve itself: the first
/// round runs under `loose`, and before every later round each width becomes
/// the robust spread of the camera's images' deviations from their mean, with
/// the prior's pull undone and the measurement noise subtracted, clamped to
/// `[floor, loose]`. `fixed` sets every camera's widths for every round and
/// derives nothing. See "The per-image release" in
/// `specs/core/geometry/bundle-adjustment.md`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePriorOptions {
    /// The widths of the first round, and the most a derived width may be.
    pub loose: ImagePriorWidths,
    /// The least a derived width may be.
    pub floor: ImagePriorWidths,
    /// When set, every camera's prior has these widths in every round, used as
    /// they are, and nothing is derived.
    pub fixed: Option<ImagePriorWidths>,
}

impl Default for ImagePriorOptions {
    fn default() -> Self {
        Self {
            loose: ImagePriorWidths {
                focal: 0.1,
                principal_point: 0.1,
            },
            floor: ImagePriorWidths {
                focal: 0.001,
                principal_point: 0.001,
            },
            fixed: None,
        }
    }
}

/// The prior one camera's per-image release ran under, as the solve reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePrior {
    /// The widths the final round ran under. `NaN` for a parameter the camera
    /// does not release per image.
    pub widths: ImagePriorWidths,
    /// Whether the widths were derived from the solve, rather than set by
    /// [`ImagePriorOptions::fixed`].
    pub derived: bool,
    /// The last derivation's spread before it was clamped to the floor and the
    /// loose width, so a reader sees which decided the width. `NaN` for a
    /// parameter not released per image, for one whose derivation had fewer
    /// than two images with a positive weight, and for both when nothing was
    /// derived.
    pub spread: ImagePriorWidths,
}

impl<'a> BaCameras<'a> {
    /// Every one of `n_img` images taken by one camera: the single-camera
    /// case.
    pub fn shared(camera: &'a CameraIntrinsics, n_img: usize) -> BaCameras<'a> {
        BaCameras {
            cameras: std::slice::from_ref(camera),
            image_camera: Cow::Owned(vec![0; n_img]),
            releases: None,
            image_adjustments: None,
        }
    }

    /// Panic on a camera list or column that does not describe `n_img` images.
    fn check(&self, n_img: usize) {
        assert!(!self.cameras.is_empty(), "BaCameras holds no camera");
        assert_eq!(
            self.image_camera.len(),
            n_img,
            "image_camera and quats length mismatch"
        );
        if let Some(&bad) = self
            .image_camera
            .iter()
            .find(|&&j| j as usize >= self.cameras.len())
        {
            panic!(
                "image_camera names camera {bad}, past the {} cameras",
                self.cameras.len()
            );
        }
        if let Some(releases) = self.releases {
            assert_eq!(
                releases.len(),
                self.cameras.len(),
                "releases and cameras length mismatch"
            );
        }
        if let Some(adjustments) = &self.image_adjustments {
            assert_eq!(
                adjustments.len(),
                n_img,
                "image_adjustments and quats length mismatch"
            );
        }
    }
}

/// Result of [`bundle_adjust`]. Poses and points are refined in place; this
/// carries what has no in-place home.
#[derive(Clone, Debug)]
pub struct BundleAdjustment {
    /// The cameras after the solve, one per input camera and in its order: each
    /// the input camera with its released parameters replaced by the solved
    /// ones (the focal under `opt_f`, `k1` under `opt_k1`, the spline under
    /// `opt_bspline`, where its model admits the release, and the principal
    /// point under `opt_pp`), and equal to the input camera otherwise.
    pub cameras: Vec<CameraIntrinsics>,
    /// Unweighted reprojection residual norm of every supplied observation at
    /// the final state; `+∞` where the point is non-finite, behind the
    /// camera, or outside the model domain. All-`∞` signals the degenerate
    /// exit (fewer than `min_obs` observations, finite and direction alike,
    /// survived a trim).
    pub residual_norms: Vec<f64>,
    /// The representation each point ended with, one entry per point: `true`
    /// where the returned row is a world-frame direction and `false` where it
    /// is a position. A free point's entry is the caller's input mask unless
    /// [`FreePointPolicy::cross`] solved it in inverse depth, a held point's is
    /// its input value, and a ranged point's is whether its distance is
    /// infinite.
    pub point_at_infinity: Vec<bool>,
    /// The end-of-solve storage decision under [`FreePointPolicy::cross`];
    /// `None` with the crossing off, and when the solve exited degenerate.
    pub free_point_decision: Option<FreePointDecision>,
    /// Every image's intrinsic adjustment after the solve, one per image: the
    /// solved adjustment, re-centred on its camera's mean, for an image of a
    /// camera with a per-image release, and the input's otherwise (zero where
    /// [`BaCameras::image_adjustments`] is `None`).
    pub image_adjustments: Vec<IntrinsicAdjustment>,
    /// One entry per camera: the prior its per-image release ran under, or
    /// `None` for a camera with no per-image release.
    pub image_priors: Vec<Option<ImagePrior>>,
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
    /// `[f, k1, (active spline coefficients,) cx, cy, δθ×3, δt×3]`, the lens
    /// columns in the block of the camera that took the observation's image. A
    /// spline
    /// slot that carries no coefficient (an active basis function of the
    /// gauge-anchored pair, full index < 2, or any spline slot of a camera
    /// whose spline is not released) points at that camera's [`K1_SLOT`]. Its
    /// column is exactly zero, so it adds exact zeros there whether or not that
    /// slot is released.
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
/// instantiation: its camera's two lens scalars (`f`, `k1`) plus the image's
/// six pose DOFs. Both scalar slots are always present — pinned in the
/// reduced system when unreleased — so the indexing is uniform.
const BASE_CAM_COLS: usize = 2 + 6;

/// Width when any camera's spline is released: the two scalar slots, the
/// [`BSPLINE_SUPPORT`] basis functions active at the observation's incidence
/// angle (cubic local support — the only nonzero columns of `∂(u, v)/∂c` for
/// that observation), and the six pose DOFs.
const BSPLINE_CAM_COLS: usize = 2 + BSPLINE_SUPPORT + 6;

/// [`BASE_CAM_COLS`] with the principal point's two columns (`cx`, `cy`)
/// before the pose DOFs: the width when any camera releases its principal
/// point and none its spline.
const BASE_PP_CAM_COLS: usize = BASE_CAM_COLS + 2;

/// [`BSPLINE_CAM_COLS`] with the principal point's two columns before the pose
/// DOFs: the width when some camera releases its spline and some its principal
/// point.
const BSPLINE_PP_CAM_COLS: usize = BSPLINE_CAM_COLS + 2;

/// An image's pose columns: `δθ×3`, `δt×3`.
const POSE_COLS: usize = 6;

/// An image's intrinsic-adjustment columns, after its pose: the log focal scale
/// `t = ln(1 + s)`, then the principal-point offset `a_x`, `a_y`, each in the
/// image's own block of the reduced system.
const IMG_ADJ_COLS: usize = 3;

/// The log focal scale's slot, relative to the start of an image's block in the
/// reduced system; `a_x` and `a_y` follow it.
const IMG_S_SLOT: usize = POSE_COLS;

/// [`BASE_PP_CAM_COLS`] with the image's three adjustment columns after the
/// pose: the width when some camera releases a per-image parameter and none its
/// spline. A per-image release always carries the shared principal point's
/// columns as well, so that the per-image widths are two instantiations rather
/// than four.
const BASE_IMG_CAM_COLS: usize = BASE_PP_CAM_COLS + IMG_ADJ_COLS;

/// [`BSPLINE_PP_CAM_COLS`] with the image's three adjustment columns after the
/// pose: the width when some camera releases a per-image parameter and some its
/// spline.
const BSPLINE_IMG_CAM_COLS: usize = BSPLINE_PP_CAM_COLS + IMG_ADJ_COLS;

// The six widths name six instantiations, and `carries_pp_cols` and
// `carries_image_cols` tell them apart by width alone.
const _: () = assert!(
    BASE_CAM_COLS != BASE_PP_CAM_COLS
        && BASE_CAM_COLS != BSPLINE_CAM_COLS
        && BASE_PP_CAM_COLS != BSPLINE_CAM_COLS
        && BSPLINE_CAM_COLS != BSPLINE_PP_CAM_COLS
        && BASE_IMG_CAM_COLS != BASE_CAM_COLS
        && BASE_IMG_CAM_COLS != BSPLINE_CAM_COLS
        && BASE_IMG_CAM_COLS != BASE_PP_CAM_COLS
        && BASE_IMG_CAM_COLS != BSPLINE_PP_CAM_COLS
        && BSPLINE_IMG_CAM_COLS != BASE_CAM_COLS
        && BSPLINE_IMG_CAM_COLS != BSPLINE_CAM_COLS
        && BSPLINE_IMG_CAM_COLS != BASE_PP_CAM_COLS
        && BSPLINE_IMG_CAM_COLS != BSPLINE_PP_CAM_COLS
        && BSPLINE_IMG_CAM_COLS != BASE_IMG_CAM_COLS
);

/// Whether an observation block `cam_cols` wide carries each image's three
/// adjustment columns. Only the solves that release a per-image parameter pay
/// for them, as only those that release a principal point pay for its two.
#[inline]
const fn carries_image_cols(cam_cols: usize) -> bool {
    cam_cols == BASE_IMG_CAM_COLS || cam_cols == BSPLINE_IMG_CAM_COLS
}

/// Whether an observation block `cam_cols` wide carries the principal point's
/// two columns. Only the solves that release a principal point, shared or per
/// image, pay for them: a wider block costs every observation pair of every
/// point in the Schur elimination, which grows with the square of the width.
#[inline]
const fn carries_pp_cols(cam_cols: usize) -> bool {
    cam_cols == BASE_PP_CAM_COLS || cam_cols == BSPLINE_PP_CAM_COLS || carries_image_cols(cam_cols)
}

/// The width of an image's block in the reduced system for observation blocks
/// `cam_cols` wide: its six pose slots, and its three adjustment slots in the
/// instantiations that carry them.
#[inline]
const fn image_block_cols(cam_cols: usize) -> usize {
    if carries_image_cols(cam_cols) {
        POSE_COLS + IMG_ADJ_COLS
    } else {
        POSE_COLS
    }
}

/// A camera's focal slot, relative to the start of its lens block.
const F_SLOT: usize = 0;
/// A camera's radial-coefficient slot, relative to the start of its lens block.
const K1_SLOT: usize = 1;
/// A camera's first spline-coefficient slot, relative to the start of its lens
/// block (coefficient `i` lives at `BSPLINE_SLOT0 + i`; the principal point's
/// two slots follow the whole coefficient vector, at [`pp_slot`]).
const BSPLINE_SLOT0: usize = 2;

/// A camera's `cx` slot, relative to the start of its lens block, for a block
/// carrying `n_coeffs` spline coefficient slots (zero where its spline is not
/// released); `cy` is the slot after it, and the next camera's block, or the
/// pose blocks after the last camera, follow `cy`. After the spline rather
/// than before it, so that [`F_SLOT`], [`K1_SLOT`] and [`BSPLINE_SLOT0`] keep
/// the values they had before the principal point had slots.
#[inline]
const fn pp_slot(n_coeffs: usize) -> usize {
    BSPLINE_SLOT0 + n_coeffs
}

/// Width of a camera's lens block in the reduced system: `f`, `k1`, its
/// `n_coeffs` spline coefficient slots, `cx` and `cy`.
#[inline]
const fn lens_block_width(n_coeffs: usize) -> usize {
    pp_slot(n_coeffs) + 2
}

/// Whether a candidate log focal scale `t = ln(1 + s)` is one a lens can have:
/// finite, with the image's focal `f·(1 + s)` finite and above `1e-6`, the
/// counterpart of the shared focal's guard. The focal is read through the
/// stored scale `s = exp(t) − 1`, as the projection reads it, so `1 + s > 0`
/// follows.
fn image_focal_step_admissible(f: f64, t: f64) -> bool {
    let f_image = f * (1.0 + t.exp_m1());
    t.is_finite() && f_image.is_finite() && f_image > 1e-6
}

/// The log focal scale `t = ln(1 + s)` of a stored focal scale `s`, the
/// parameter the per-image focal release solves, holds and re-centres.
#[inline]
fn log_focal_scale(s: f64) -> f64 {
    s.ln_1p()
}

/// Whether a candidate principal point is one a lens can have: finite, and on
/// the sensor (`cx` in `[0, width]`, `cy` in `[0, height]`). The guard on a
/// principal-point step, the counterpart of the focal's `f > 0`: no physical
/// lens has its optical axis off the sensor, and a solve that tries to put it
/// there is following a degenerate direction rather than the data.
fn pp_step_admissible(cx: f64, cy: f64, width: u32, height: u32) -> bool {
    cx.is_finite()
        && cy.is_finite()
        && (0.0..=f64::from(width)).contains(&cx)
        && (0.0..=f64::from(height)).contains(&cy)
}

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
/// which `bspline_is_monotone` decides along with the domain. Past the data the
/// coefficients move as the spline penalty continues them, and a continuation
/// that would fold there is a step this guard holds.
fn bspline_step_admissible(bspline: &[f64], d_max: f64) -> bool {
    bspline.iter().all(|c| c.is_finite()) && bspline_is_monotone(bspline, d_max, d_max)
}

/// Staged bundle adjustment over images taken through one or more cameras.
///
/// `cameras` holds the camera list and, per image, the index of the camera
/// that took it; [`BaCameras::shared`] is the single-camera case. Every
/// observation is projected, differentiated, trimmed and re-estimated through
/// the camera of its own image, and each camera keeps its own lens block in the
/// reduced system, so two cameras' lens parameters couple only through the
/// poses and points they both observe. Where [`BaCameras::image_adjustments`]
/// gives an image an intrinsic adjustment, its observations are read through
/// its camera with that adjustment applied, and the adjustment is held: the
/// released lens parameters are the camera's shared ones.
///
/// Per schedule round: retriangulate every point from all supplied
/// observations at the current poses (rounds after the first), trim to
/// observations under `trim_px` with in-front depth and a finite point whose
/// track keeps at least `min_track` survivors, then run one robust sparse LM
/// solve at the round's `loss_scale`. Poses and points are refined in place;
/// the returned [`BundleAdjustment`] carries the cameras and the
/// per-observation residual norms at the final state (`+∞` where invalid — and
/// everywhere, with the state passed through, when fewer than `min_obs`
/// observations — finite and direction alike — survive a trim).
///
/// `point_at_infinity` optionally marks per-point directions: a marked row of
/// `points` is a world-frame direction (normalized on input and output) whose
/// observations depend on rotation and camera model only — see "Points at
/// infinity" in `specs/core/geometry/bundle-adjustment.md`. An absent mask is an
/// all-`false` mask, which reduces the solve to the finite-only one.
///
/// `constraints` optionally states a constraint per point (free, ranged or
/// held) and what the ranged ones are held at; an absent one is every point
/// free. `free_points` says whether a free point is solved in inverse depth and
/// its representation decided at the end of the solve, on the point-or-bearing
/// test at the noise level the final round's residuals measure. See "Point
/// constraints" in
/// `specs/core/geometry/bundle-adjustment.md`. The default policy crosses;
/// under [`FreePointPolicy::NO_CROSS`] every free point keeps the representation
/// it is handed in. Absent constraints and all-free ones are the same solve to
/// the bit.
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
/// `opt_f` releases each camera's focal (SIMPLE_PINHOLE, EQUIDISTANT_FISHEYE,
/// SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE and SFMTOOL_PINHOLE — the models
/// this kernel's analytic focal column `(u − cx)/f` is exact for), `opt_k1`
/// each camera's radial coefficient (SIMPLE_RADIAL_FISHEYE only, the one model
/// carrying it), and `opt_bspline` each camera's radial spline coefficients
/// (SFMTOOL_FISHEYE and SFMTOOL_PINHOLE, the two carrying a spline). Each is a
/// request to every camera, decided per camera: a camera whose model the
/// release is not exact for keeps that parameter fixed, never a half-modeled
/// DOF, while the others release theirs. The binding rejects such a model
/// loudly instead. `opt_pp` releases each camera's principal point, on every
/// model: every model projects as `u = fx·x_d + cx`, `v = fy·y_d + cy` with a
/// distorted coordinate that does not read the principal point, so its columns
/// are the identity, exact everywhere, and it needs no focal release beside it.
/// A candidate step whose principal point is not finite or leaves the image
/// (`cx` outside `[0, width]`, `cy` outside `[0, height]`) is rejected whole
/// and re-damped. [`BaCameras::releases`] narrows the four flags camera by
/// camera, so a solve can release one camera's lens and hold another's.
/// `opt_k1` and `opt_bspline` are exclusive on any one camera
/// (no model carries both parameters). Callers stage the releases — fixed →
/// `opt_f` → `opt_f` plus the model's distortion release — so the distortion
/// rung opens on a focal that has already settled.
///
/// `opt_image_f` and `opt_image_pp` release each image's intrinsic adjustment:
/// its focal scale (on the models `opt_f` admits) and its principal-point
/// offset (on every model), narrowed per camera by
/// [`CameraRelease::image_focal`] and [`CameraRelease::image_principal_point`].
/// A camera that releases a per-image parameter holds its shared value of the
/// same kind during the solve, so asking for both is the per-image release
/// alone. Each image is held toward the mean adjustment of its camera's images
/// in the round by a Gaussian prior whose widths `image_prior` chooses (`None`
/// for [`ImagePriorOptions::default`], which derives them from the solve), and
/// after every round the camera takes that mean while each image keeps its
/// difference from it, which leaves every image's intrinsics as they were. The
/// focal scale `s` is solved, held and re-centred as its logarithm
/// `t = ln(1 + s)`, since a focal scale is multiplicative: the mean it is held
/// toward and re-centred by is the mean of `t`, the geometric mean of `1 + s`.
/// It is stored and returned as `s`.
/// The solved adjustments come back in [`BundleAdjustment::image_adjustments`]
/// and each camera's widths in [`BundleAdjustment::image_priors`]. With
/// neither flag on, or no camera releasing either, the solve is the one before
/// these flags existed, bit for bit. See "The per-image release" in
/// `specs/core/geometry/bundle-adjustment.md`.
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
    cameras: &BaCameras<'_>,
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
    opt_pp: bool,
    opt_image_f: bool,
    opt_image_pp: bool,
    image_prior: Option<&ImagePriorOptions>,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
    progress: &Progress<'_>,
) -> BundleAdjustment {
    cameras.check(quats.len());
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
        cameras.cameras,
        &cameras.image_camera,
        cameras.image_adjustments.as_deref(),
        cameras.releases,
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
        opt_pp,
        opt_image_f,
        opt_image_pp,
        &image_prior.copied().unwrap_or_default(),
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

/// Mixed-path residual norms and in-front measures, each observation through
/// the camera of its own image (`cams[image_camera[i]]`). Invalid observations
/// report `INVALID_RESIDUAL` and a non-positive in-front measure.
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
///
/// Finite observations are checked against the scene-scale floor of
/// [`in_front_floor`] by the caller.
/// A direction observation reports the same measure at `R·d` and is checked
/// against zero: for the perspective family that is `(R·d)_z < 0`, and for a
/// ray-path model, whose range of a unit direction is one, it passes and the
/// direction is in front exactly when `ray_to_pixel(R·d)` is defined, which the
/// residual already reports.
#[allow(clippy::too_many_arguments)]
fn residual_norms_depths(
    cams: &[CameraIntrinsics],
    image_camera: &[u32],
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
    let ray_path: Vec<bool> = cams.iter().map(|c| c.model.needs_ray_path()).collect();
    for k in 0..n_obs {
        let pi = obs_pt[k] as usize;
        let p = points[pi];
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            continue;
        }
        let i = obs_img[k] as usize;
        let j = image_camera[i] as usize;
        let rot = quats[i] * Vector3::new(p[0], p[1], p[2]);
        let c = if is_dir[pi] { rot } else { rot + trans[i] };
        depths[k] = if ray_path[j] { c.norm() } else { -c.z };
        if let Some((u, v)) = cams[j].ray_to_pixel([c.x, c.y, c.z]) {
            norms[k] = (u - uv[k][0]).hypot(v - uv[k][1]);
        }
    }
    (norms, depths)
}

/// The trim's in-front floor for finite observations at one round's state:
/// [`IN_FRONT_FLOOR_FRACTION`] of the scene scale, the median in-front measure
/// over the observations of finite points that are in front at all. Zero when
/// no such observation exists, which leaves the sign test.
///
/// The floor is a length, as the measure is, and it is a fraction of a length
/// the scene states, so scaling the world scales the floor with it and the
/// trim keeps exactly the observations it kept before. The median is a
/// selection, not a sum, so it is the same value whatever order the
/// observations come in.
fn in_front_floor(depths: &[f64], is_dir: &[bool], obs_pt: &[u32]) -> f64 {
    let mut front: Vec<f64> = depths
        .iter()
        .zip(obs_pt)
        .filter(|&(&d, &p)| !is_dir[p as usize] && d > 0.0 && d.is_finite())
        .map(|(&d, _)| d)
        .collect();
    if front.is_empty() {
        return 0.0;
    }
    let mid = front.len() / 2;
    let (_, median, _) = front.select_nth_unstable_by(mid, f64::total_cmp);
    IN_FRONT_FLOOR_FRACTION * *median
}

/// Re-estimation (rounds after the first): the shared retriangulation
/// operation ([`crate::reconstruction::triangulation`]) at the round's
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
/// Each observation's ray is `pixel_to_ray` through the camera of its own
/// image, so a track seen by two cameras back-projects each observation
/// through its own lens.
///
/// `cross_trim` is the inverse-depth policy for free points: `None` with the
/// crossing off, and with it on, the trim threshold of the round about to
/// start. Off, every free point keeps its representation as it came in (the
/// mask is honoured for the whole solve). On, each free point starts the next
/// round from whichever of three states fits its whole track best, read as the
/// sum over its observations of the squared residual capped at `cross_trim`:
/// the state the last round left it in, its midpoint, and its mean ray at
/// `ρ = 0`, preferred in that order on a tie. The midpoint is read with
/// cheirality on, so one behind a camera that observes the point is the mean
/// ray. A state in the other representation than the last round left is taken
/// only when at least `min_track` of the track's observations reproject under
/// `cross_trim` from it, the number the next round's trim needs to keep the
/// track in the solve; otherwise the last state stands, since a change of
/// representation the next round cannot solve on would come back unscored and
/// be reset to the row it was handed in with. A point with no estimate takes
/// its midpoint, or its mean ray where that costs less, with no guard. The
/// choice is not a decision about how the point is stored: the next
/// round solves it in inverse depth from there, where it can move to or away
/// from infinity, and the storage decision is taken at the end of the solve.
/// It is a choice among starting values because neither closed form is right
/// for every track: the midpoint of nearly parallel rays can land near the
/// cameras, where its residuals put the whole track past the trim, and the mean
/// ray of a near point's rays is off by its parallax. A free point with fewer
/// than two usable observations becomes `NaN`, as any track does. A ranged
/// point is carried by the distance rule at whatever origin its reference
/// resolves to now, and a held point is not re-estimated at all.
#[allow(clippy::too_many_arguments)]
fn retriangulate_round(
    cams: &[CameraIntrinsics],
    image_camera: &[u32],
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &mut [[f64; 3]],
    is_dir: &mut [bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    cons: &Constraints,
    cross_trim: Option<f64>,
    min_track: usize,
) {
    let free_from_rays = cross_trim.is_some();
    let mut quats_wxyz = Vec::with_capacity(quats.len() * 4);
    for q in quats {
        quats_wxyz.extend_from_slice(&[q.w, q.i, q.j, q.k]);
    }
    let mut translations = Vec::with_capacity(trans.len() * 3);
    for t in trans {
        translations.extend_from_slice(&[t.x, t.y, t.z]);
    }
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
    // The operation over every track, the free points marked as `free` says
    // where it is given, and every point otherwise as the round's mask has it.
    let estimate = |is_dir: &[bool], free: Option<bool>| {
        let marks: Vec<bool> = (0..points.len())
            .map(|p| match free {
                Some(m) if cons.constraint[p] == PointConstraint::Free => m,
                _ => is_dir[p],
            })
            .collect();
        triangulate_points_through_cameras(
            cams,
            image_camera,
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
                cheirality: free_from_rays,
                few: FewObservations::Absent,
                ..Default::default()
            },
        )
        .xyzw
    };
    let Some(trim_px) = cross_trim else {
        let est = estimate(is_dir, None);
        for (p, (row, e)) in points.iter_mut().zip(&est).enumerate() {
            // A held point owns its coordinate; the estimate for it is discarded.
            if cons.held(p) {
                continue;
            }
            *row = [e[0], e[1], e[2]];
        }
        return;
    };
    // Under the crossing: every free point's midpoint (its mean ray where the
    // midpoint lies behind an observing camera) and its mean ray.
    let mid = estimate(is_dir, Some(false));
    let ray = estimate(is_dir, Some(true));
    let mid_rows: Vec<[f64; 3]> = mid.iter().map(|e| [e[0], e[1], e[2]]).collect();
    let mid_dirs: Vec<bool> = mid.iter().map(|e| e[3] == 0.0).collect();
    let ray_rows: Vec<[f64; 3]> = ray.iter().map(|e| [e[0], e[1], e[2]]).collect();
    let ray_dirs = vec![true; points.len()];
    // A candidate state's cost per point, the squared residual of each of its
    // observations capped at the next round's trim, and how many of them
    // reproject under that trim.
    let capped_cost = |rows: &[[f64; 3]], dirs: &[bool]| {
        let (norms, _) = residual_norms_depths(
            cams,
            image_camera,
            quats,
            trans,
            rows,
            dirs,
            uv,
            obs_img,
            obs_pt,
        );
        let mut cost = vec![0.0; rows.len()];
        let mut under = vec![0usize; rows.len()];
        for (k, &r) in norms.iter().enumerate() {
            let p = obs_pt[k] as usize;
            cost[p] += r.min(trim_px).powi(2);
            under[p] += usize::from(r < trim_px);
        }
        (cost, under)
    };
    let (cost_now, _) = capped_cost(points, is_dir);
    let (cost_mid, under_mid) = capped_cost(&mid_rows, &mid_dirs);
    let (cost_ray, under_ray) = capped_cost(&ray_rows, &ray_dirs);
    for p in 0..points.len() {
        // A held point owns its coordinate; the estimate for it is discarded.
        if cons.held(p) {
            continue;
        }
        // A ranged point is carried by the distance rule, and a free point
        // with fewer than two usable observations has no estimate.
        if cons.constraint[p] != PointConstraint::Free || !mid[p][3].is_finite() {
            points[p] = mid_rows[p];
            continue;
        }
        // A state with no estimate (`NaN`) is replaced by the midpoint, as any
        // track's is; otherwise a state in the other representation must keep
        // enough of the track under the trim for the next round to solve it.
        let now_finite = points[p].iter().all(|v| v.is_finite());
        let admissible =
            |dir: bool, under: usize| !now_finite || dir == is_dir[p] || under >= min_track;
        let was_dir = is_dir[p];
        let mut best = cost_now[p];
        if !now_finite || (cost_mid[p] < best && admissible(mid_dirs[p], under_mid[p])) {
            best = cost_mid[p];
            points[p] = mid_rows[p];
            is_dir[p] = mid_dirs[p];
        }
        if ray[p][3].is_finite()
            && cost_ray[p] < best
            && (!now_finite || was_dir || under_ray[p] >= min_track)
        {
            points[p] = ray_rows[p];
            is_dir[p] = true;
        }
    }
}

/// The noise level the storage decision reads: the gated RMS per-axis residual
/// over `kept` observations of finite points at the state the arrays hold, the
/// estimator of [`crate::analysis::reprojection_noise`] (each camera's
/// residuals gated at [`OUTLIER_GATE`] robust spreads, no degrees-of-freedom
/// correction), never under the cameras' keypoint resolution.
///
/// `kept` is the final round's solve set and the state the one that solve
/// settled on, so the level measures the residuals the adjustment has just
/// minimised rather than its schedule's loss scale. A direction's residuals
/// are left out, as the stored measure leaves out points at infinity: they are
/// the bearing's, which is the model under question. An observation the camera
/// model cannot image at that state carries no residual and is left out too.
///
/// Each residual is read through its image's intrinsics, `intrinsics`, and gated
/// with the other residuals of its image's camera, `cams[image_camera[i]]`,
/// whose keypoint resolutions are the floor.
#[allow(clippy::too_many_arguments)]
fn round_noise(
    cams: &[CameraIntrinsics],
    image_camera: &[u32],
    intrinsics: &ImageIntrinsicsList<'_>,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &[[f64; 3]],
    is_dir: &[bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    kept: &[usize],
) -> ReprojectionNoise {
    let residuals: Vec<ObservationResidual> = kept
        .par_iter()
        .filter_map(|&k| {
            let p = obs_pt[k] as usize;
            if is_dir[p] {
                return None;
            }
            let x = points[p];
            if !(x[0].is_finite() && x[1].is_finite() && x[2].is_finite()) {
                return None;
            }
            let i = obs_img[k] as usize;
            let camera = image_camera[i] as usize;
            let c = quats[i] * Vector3::new(x[0], x[1], x[2]) + trans[i];
            let (u, v) = intrinsics.of(i).ray_to_pixel([c.x, c.y, c.z])?;
            let residual = [u - uv[k][0], v - uv[k][1]];
            (residual[0].is_finite() && residual[1].is_finite())
                .then_some(ObservationResidual { camera, residual })
        })
        .collect();
    let mut noise = gated_reprojection_noise(&residuals, cams.len(), OUTLIER_GATE);
    let resolution = cams
        .iter()
        .map(camera_keypoint_resolution_px)
        .fold(f64::from(f32::EPSILON), f64::max);
    noise.sigma_px = noise.sigma_px.map(|s| s.max(resolution));
    noise
}

/// Robust cost over the kept observations at a candidate state. `points` are
/// world positions by compact index -- a ranged point's already resolved
/// through its reference and distance; `cp_dir` flags direction points; `s2s`
/// is the per-kept-observation squared loss scale (uniform except where a
/// protected observation widens it). `cams` are the cameras at the candidate
/// state, an adjusted image's own entry among them, and `obs_cam` the entry
/// each kept observation is read through. A point with an
/// anchor in `inverse` is the direction `points[c]` and the inverse depth
/// `inv_depth[c]` about that anchor (see [`inverse_depth_ray`]).
#[allow(clippy::too_many_arguments)]
fn robust_cost(
    cams: &[CameraIntrinsics],
    obs_cam: &[usize],
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &[Vector3<f64>],
    cp_dir: &[bool],
    inverse: &[Option<Vector3<f64>>],
    inv_depth: &[f64],
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
            let cp = obs_cp[kk];
            let p = points[cp];
            // A non-finite point (possible only for protected observations,
            // which the trim never excludes) is penalized like an
            // out-of-domain projection.
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                return s2 * rho(INVALID_RESIDUAL * INVALID_RESIDUAL / s2);
            }
            let ci = obs_ci[kk];
            let c = if let Some(a) = inverse.get(cp).copied().flatten() {
                inverse_depth_ray(&quats[ci], &trans[ci], &p, inv_depth[cp], &a).1
            } else {
                let rot = quats[ci] * p;
                if cp_dir[cp] {
                    rot
                } else {
                    rot + trans[ci]
                }
            };
            match cams[obs_cam[kk]].ray_to_pixel([c.x, c.y, c.z]) {
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

/// The camera-frame ray of a point held in inverse depth `ρ` about the anchor
/// `a` with unit direction `u`, seen from the image at `(q, t)`: returned as
/// `R·(u + ρ·a)`, the part the rotation block differentiates, and
/// `p̃ = R·(u + ρ·a) + ρ·t`.
///
/// `p̃` is `ρ·(R·X + t)` for the point `X = a + u/ρ`, a positive multiple of the
/// camera-frame point, so it projects to the same pixel and sits on the same
/// side of the camera; at `ρ = 0` it is `R·u`, the direction's own ray. Every
/// projection the solve makes is homogeneous of degree zero in the ray, so it
/// reads `p̃` directly and never forms the point, which is what keeps a far
/// point well conditioned.
#[inline]
fn inverse_depth_ray(
    q: &UnitQuaternion<f64>,
    t: &Vector3<f64>,
    u: &Vector3<f64>,
    inv_depth: f64,
    anchor: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let rot = q * (u + anchor * inv_depth);
    (rot, rot + t * inv_depth)
}

/// Everything one linearization of the solve reads that does not vary from
/// observation to observation: every camera at the current lens state, the
/// current poses and points, and the per-point flags that say what each point's
/// block looks like.
///
/// It exists so the per-observation blocks are built by a function a test can
/// call, which is how the reference-camera columns of a ranged point are
/// checked against a difference of the residual they claim to differentiate.
struct LinState<'a> {
    /// One entry per camera, that camera at the current state and its lens
    /// block, then one per compact image with an intrinsic adjustment.
    lenses: &'a [LinLens<'a>],
    /// The entry of `lenses` each compact image is read through: its
    /// camera's, or its own where it has an adjustment.
    ci_cam: &'a [usize],
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
    /// The anchor of each point solved in inverse depth, whose `xp` entry is
    /// then its unit direction from the anchor; empty, or `None` per point,
    /// where no point is.
    cp_inverse: &'a [Option<Vector3<f64>>],
    /// The inverse depth of each point solved in inverse depth, parallel to
    /// `cp_inverse`.
    inv_depth: &'a [f64],
    /// Every supplied observation's pixel, indexed by the caller's own index.
    uv: &'a [[f64; 2]],
    /// Width of the reduced system's lens head: every camera's block.
    n_shared: usize,
    /// Width of each image's block: its six pose slots, and its three
    /// adjustment slots in a solve that releases a per-image parameter.
    img_cols: usize,
}

/// One camera as a linearization reads it: the camera at the current state,
/// which of its parameters are released, and where its lens block starts in
/// the reduced system.
///
/// An image with an intrinsic adjustment is read through an entry of its own:
/// `cam`, `cx`, `cy` and `f_image` are the image's, with the adjustment
/// applied, while `f`, the releases and `slot0` are its camera's, so its lens
/// columns land in the shared camera's block.
struct LinLens<'a> {
    cam: &'a CameraIntrinsics,
    /// Whether the model carries an analytic pixel Jacobian.
    analytic: bool,
    /// The principal point the observation is projected about: the camera's,
    /// moved by the image's adjustment.
    cx: f64,
    cy: f64,
    /// The camera's shared focal, the parameter the focal slot holds.
    f: f64,
    /// The focal the observation is projected with, `f·(1 + s)` under the
    /// image's focal scale `s`, and `f` itself without one.
    f_image: f64,
    /// The camera's short side `min(width, height)`, the unit of the
    /// principal-point offset.
    m: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    opt_pp: bool,
    /// Whether the image's focal scale is released, and its principal-point
    /// offset. Read only in the instantiations that carry their columns.
    opt_image_f: bool,
    opt_image_pp: bool,
    n_coeffs: usize,
    d_max: f64,
    radial: SplineRadial,
    /// The reduced-system slot of this camera's focal; its `k1`, spline and
    /// principal-point slots follow.
    slot0: usize,
}

impl LinState<'_> {
    /// First pose slot of compact image `ci` in the reduced camera system.
    #[inline]
    fn img_slot(&self, ci: usize) -> usize {
        self.n_shared + self.img_cols * ci
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
    // First pose column within an observation's camera block, and, in the
    // instantiations that carry them, the principal point's two columns just
    // before it and the image's three adjustment columns just after the pose.
    let pose_c = CAM_COLS - image_block_cols(CAM_COLS);
    let pp_cols = carries_pp_cols(CAM_COLS);
    let pp_c = pose_c - 2;
    let adj_c = pose_c + IMG_S_SLOT;
    let lens = &st.lenses[st.ci_cam[ci]];
    let (q, t, xp, uv, cam) = (st.q, st.t, st.xp, st.uv, lens.cam);
    let (f, f_image) = (lens.f, lens.f_image);
    let dir = st.cp_dir[cp];
    let inverse = st.cp_inverse.get(cp).copied().flatten();
    let (rot_pt, p_cam) = match inverse {
        Some(a) => inverse_depth_ray(&q[ci], &t[ci], &xp[cp], st.inv_depth[cp], &a),
        None => {
            let rot_pt = q[ci] * xp[cp];
            (rot_pt, if dir { rot_pt } else { rot_pt + t[ci] })
        }
    };
    let mut res = Vector2::new(INVALID_RESIDUAL, 0.0);
    let mut cam_j = SMatrix::<f64, 2, CAM_COLS>::zeros();
    let mut pt_j = SMatrix::<f64, 2, 3>::zeros();
    let mut ref_cols: Vec<(usize, Vector2<f64>)> = Vec::new();
    // Column indices: `[f, k1, (spline), cx, cy, δθ×3, δt×3, (s, a_x, a_y)]`, the lens
    // columns in the block of this observation's camera. Spline slots start at
    // that camera's K1_SLOT dummy and are pointed at their coefficient's slot
    // below, where the observation actually carries one.
    let mut idx = [lens.slot0 + K1_SLOT; CAM_COLS];
    idx[F_SLOT] = lens.slot0 + F_SLOT;
    if pp_cols {
        let pp0 = lens.slot0 + pp_slot(lens.n_coeffs);
        idx[pp_c] = pp0;
        idx[pp_c + 1] = pp0 + 1;
    }
    let o = st.img_slot(ci);
    for (j, slot) in idx[pose_c..].iter_mut().enumerate() {
        *slot = o + j;
    }
    // A non-finite point (protected observations only — the trim
    // never excludes them) keeps the penalized residual and zero
    // Jacobian rows: penalized, never steering.
    let proj = if xp[cp].x.is_finite() && xp[cp].y.is_finite() && xp[cp].z.is_finite() {
        project_with_jac(cam, p_cam, lens.analytic)
    } else {
        None
    };
    if let Some(((u, v), jp)) = proj {
        res = Vector2::new(u - uv[k][0], v - uv[k][1]);
        let jp = SMatrix::<f64, 2, 3>::from_rows(&[
            SMatrix::<f64, 1, 3>::from_row_slice(&jp[0]),
            SMatrix::<f64, 1, 3>::from_row_slice(&jp[1]),
        ]);
        if lens.opt_f {
            // ∂(u, v)/∂f. Exact for every model the `opt_f` gate
            // admits: the focal is a pure multiplier of an
            // `f`-independent distorted coordinate, so the
            // derivative is that coordinate. Under an image's focal
            // scale the image's focal is `f·(1 + s)`, so the column of
            // the shared `f` is `(1 + s)·x_d = (u − cx_i)/f`, read about
            // the image's own principal point.
            cam_j[(0, F_SLOT)] = (u - lens.cx) / f;
            cam_j[(1, F_SLOT)] = (v - lens.cy) / f;
        }
        if lens.opt_k1 {
            // ∂(u, v)/∂k1 = f_i·θ³·û at the image's focal — direction
            // rows included, they project through the same map.
            let (du, dv) = k1_column(f_image, p_cam);
            cam_j[(0, K1_SLOT)] = du;
            cam_j[(1, K1_SLOT)] = dv;
        }
        if lens.opt_bspline {
            // ∂(u, v)/∂cᵢ = f_i·Bᵢ(θ)·û at the image's focal, for the
            // ≤ 4 active basis functions — direction rows included, they
            // project through the same map. Gauge-anchored functions
            // (full index < 2) carry no coefficient: their slot
            // keeps the pinned K1_SLOT dummy and their column
            // stays exactly zero.
            let (first, cols) =
                bspline_columns(f_image, lens.n_coeffs, lens.d_max, lens.radial, p_cam);
            for (j, col) in cols.iter().enumerate() {
                let full = first + j;
                if full < 2 {
                    continue;
                }
                idx[2 + j] = lens.slot0 + BSPLINE_SLOT0 + (full - 2);
                cam_j[(0, 2 + j)] = col[0];
                cam_j[(1, 2 + j)] = col[1];
            }
        }
        if pp_cols && lens.opt_pp {
            // ∂(u, v)/∂(cx, cy) = I on every model: each projects as
            // `u = fx·x_d + cx`, `v = fy·y_d + cy` with `(x_d, y_d)` read off
            // the ray alone -- direction rows included. An image's offset
            // `a·m` is added to the shared `(cx, cy)`, so its column is the
            // same identity.
            cam_j[(0, pp_c)] = 1.0;
            cam_j[(1, pp_c + 1)] = 1.0;
        }
        if carries_image_cols(CAM_COLS) {
            if lens.opt_image_f {
                // ∂(u, v)/∂t for the image's log focal scale
                // `t = ln(1 + s)`: the image's focal is `f·exp(t)` (both
                // focals on a two-focal model), a pure multiplier of the
                // distorted coordinate, so the column is
                // `f·exp(t)·x_d = u − cx_i`, read about the image's own
                // principal point.
                cam_j[(0, adj_c)] = u - lens.cx;
                cam_j[(1, adj_c)] = v - lens.cy;
            }
            if lens.opt_image_pp {
                // ∂(u, v)/∂(a_x, a_y) = m·I: the offset is in the short-side
                // unit, added to the principal point as `a·m`.
                cam_j[(0, adj_c + 1)] = lens.m;
                cam_j[(1, adj_c + 2)] = lens.m;
            }
        }
        // Rotation block: ∂p_cam/∂δθ = −[R·X]ₓ (finite), −[R·d]ₓ
        // (direction) or −[R·(u + ρ·a)]ₓ (inverse depth) — same composition
        // every way.
        let nskew = Matrix3::new(
            0.0, rot_pt.z, -rot_pt.y, //
            -rot_pt.z, 0.0, rot_pt.x, //
            rot_pt.y, -rot_pt.x, 0.0,
        );
        cam_j
            .fixed_view_mut::<2, 3>(0, pose_c)
            .copy_from(&(jp * nskew));
        let r_mat: Matrix3<f64> = q[ci].to_rotation_matrix().into_inner();
        if let Some(a) = inverse {
            // A free point in inverse depth, p̃ = R·(u + ρ·a) + ρ·t.
            // Translation block: ρ·J, which vanishes at ρ = 0 as a
            // direction's does. Point block: the two tangent columns of `u`,
            // J·R·B(u), and the inverse depth's, J·(R·a + t), the anchor seen
            // from this camera.
            let rho = st.inv_depth[cp];
            cam_j
                .fixed_view_mut::<2, 3>(0, pose_c + 3)
                .copy_from(&(jp * rho));
            let (b1, b2) = st.bases[cp];
            pt_j.set_column(0, &(jp * (r_mat * b1)));
            pt_j.set_column(1, &(jp * (r_mat * b2)));
            pt_j.set_column(2, &(jp * (r_mat * a + t[ci])));
        } else if dir {
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

/// Whether this kernel can release the focal of a camera of this model: true
/// for `SIMPLE_PINHOLE`, `EQUIDISTANT_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`,
/// `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`, false for every other model.
///
/// This is a property of this implementation's focal column, not of the
/// camera. The analytic `∂(u, v)/∂f = (u − cx)/f` is exact only when the
/// projection is `u = f·x_d + cx` with one focal and a distorted coordinate
/// `x_d` that does not itself read `f`. The five models have that form:
/// `x_d = rx/(−rz)` for `SIMPLE_PINHOLE`, `x_d = θ·ûx` with
/// `θ = atan2(ρ, rz)` for `EQUIDISTANT_FISHEYE`, `x_d = θ·(1 + k1·θ²)·ûx`
/// with the same ray-derived `θ` for `SIMPLE_RADIAL_FISHEYE`, and, for the two
/// spline models, a dimensionless radial spline on the ray's own radial
/// coordinate (`x_d = (θ + δ(θ))·ûx` for `SFMTOOL_FISHEYE`,
/// `x_d = (ρ + δ(ρ))·ûx` with `ρ = ρ_xy/rz` for `SFMTOOL_PINHOLE`). Every
/// other model either has a second focal `fy`, which this kernel has no slot
/// for, or applies its coefficients to a normalized coordinate that depends on
/// `f` (the multi-coefficient fisheye family recovers `θ` from `r/f`).
///
/// This is the one list of those models. The focal gate in [`Lens::new`]
/// reads it, and so does the public per-camera
/// `reconstruction::bundle_adjust::focal_is_releasable`, which the viewer and
/// the Python binding ask.
pub(crate) fn focal_is_releasable(model: &CameraModel) -> bool {
    // Exhaustive on purpose, with no `_` arm: a new model has to be
    // classified here before it will build.
    match model {
        CameraModel::SimplePinhole { .. }
        | CameraModel::EquidistantFisheye { .. }
        | CameraModel::SimpleRadialFisheye { .. }
        | CameraModel::SfmtoolFisheye { .. }
        | CameraModel::SfmtoolPinhole { .. } => true,
        CameraModel::Pinhole { .. }
        | CameraModel::SimpleRadial { .. }
        | CameraModel::Radial { .. }
        | CameraModel::OpenCV { .. }
        | CameraModel::OpenCVFisheye { .. }
        | CameraModel::RadialFisheye { .. }
        | CameraModel::ThinPrismFisheye { .. }
        | CameraModel::RadTanThinPrismFisheye { .. }
        | CameraModel::FullOpenCV { .. }
        | CameraModel::Equirectangular { .. } => false,
    }
}

/// One camera's state across the staged loop: the camera the caller handed in,
/// the releases its model admits, and the current values of the parameters
/// those releases move.
#[derive(Clone, Debug)]
struct Lens<'a> {
    /// The input camera. A fixed spline, and every parameter no release
    /// touches, ride along inside it.
    base: &'a CameraIntrinsics,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
    opt_pp: bool,
    /// Whether each image's focal scale, and its principal-point offset, is
    /// released. Each holds the shared value of its kind in the solve: a
    /// camera with `opt_image_f` has `opt_f` off, and one with `opt_image_pp`
    /// has `opt_pp` off, the shared value moving only when the images'
    /// adjustments are re-centred after a round.
    opt_image_f: bool,
    opt_image_pp: bool,
    /// The focal, the model's first where it carries two.
    f: f64,
    /// The radial coefficient; `0.0` for a model without one.
    k1: f64,
    /// The spline coefficients; empty for a model without a spline.
    bspline: Vec<f64>,
    /// The principal point, `(cx, cy)`.
    pp: [f64; 2],
}

impl<'a> Lens<'a> {
    /// `base` with the releases its own model admits. Each request is decided
    /// on this camera alone, and a model a release is not exact for keeps that
    /// parameter fixed, whatever the other cameras of the solve do. The
    /// principal point is released on every model, shared or per image. A
    /// per-image release holds the shared value of its kind.
    fn new(
        base: &'a CameraIntrinsics,
        opt_f: bool,
        opt_k1: bool,
        opt_bspline: bool,
        opt_pp: bool,
        opt_image_f: bool,
        opt_image_pp: bool,
    ) -> Self {
        // The analytic focal column `∂(u, v)/∂f = (u − cx)/f` is exact only
        // on the models `focal_is_releasable` admits (its doc says why), and
        // the image's log-focal-scale column `u − cx_i` on the same ones. Every other model degrades to a fixed focal here; the binding
        // rejects it loudly first. A per-image release holds the shared value
        // of its kind for the solve.
        let opt_image_f = opt_image_f && focal_is_releasable(&base.model);
        let opt_f = opt_f && focal_is_releasable(&base.model) && !opt_image_f;
        let opt_pp = opt_pp && !opt_image_pp;
        // The curvature rung exists on exactly one model:
        // `SIMPLE_RADIAL_FISHEYE` is the only one whose single radial
        // coefficient acts on the ray's own `θ`, which is what makes
        // `∂(u, v)/∂k1 = f·θ³·û` exact. Same degrade.
        let opt_k1 = opt_k1 && matches!(base.model, CameraModel::SimpleRadialFisheye { .. });
        // The spline rung exists on the two models that carry a spline,
        // `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`, whose dimensionless
        // coefficients act on the ray's own radial coordinate `d`, making
        // `∂(u, v)/∂cᵢ = f·Bᵢ(d)·û` exact — and only when the spline is
        // actually defined (at least `MIN_BSPLINE_COEFFS` coefficients on a
        // positive finite domain end; anything shorter evaluates as the
        // identity and carries nothing to release). Same degrade; `opt_k1` and
        // `opt_bspline` are therefore exclusive on any one camera, which the
        // spline instantiation's pinned-K1 dummy slot relies on.
        let opt_bspline = opt_bspline
            && base
                .model
                .radial_spline()
                .is_some_and(|(bspline, d_max, _)| {
                    bspline.len() >= MIN_BSPLINE_COEFFS && d_max.is_finite() && d_max > 0.0
                });
        let k1 = match base.model {
            CameraModel::SimpleRadialFisheye {
                radial_distortion_k1,
                ..
            } => radial_distortion_k1,
            _ => 0.0,
        };
        let bspline = base
            .model
            .radial_spline()
            .map(|(bspline, _, _)| bspline.to_vec())
            .unwrap_or_default();
        let (cx, cy) = base.principal_point();
        Self {
            base,
            opt_f,
            opt_k1,
            opt_bspline,
            opt_pp,
            opt_image_f,
            opt_image_pp,
            f: base.focal_lengths().0,
            k1,
            bspline,
            pp: [cx, cy],
        }
    }

    /// Whether any per-image parameter of this camera is released.
    fn releases_image(&self) -> bool {
        self.opt_image_f || self.opt_image_pp
    }

    /// The camera at the current state.
    fn camera(&self) -> CameraIntrinsics {
        self.at(self.f, self.k1, &self.bspline, self.pp)
    }

    /// The camera at a candidate state. Off the spline release this is
    /// exactly the scalar builder, and a fixed spline rides along inside
    /// `base` untouched; off the principal-point releases, shared and per
    /// image, the principal point rides along inside `base` the same way. A
    /// per-image principal-point release moves the shared one when the images'
    /// offsets are re-centred, so it is written here.
    fn at(&self, f: f64, k1: f64, bspline: &[f64], pp: [f64; 2]) -> CameraIntrinsics {
        let camera = if self.opt_bspline {
            self.base.with_focal_bspline(f, bspline)
        } else {
            self.base.with_focal_k1(f, k1)
        };
        if self.opt_pp || self.opt_image_pp {
            camera.with_principal_point(pp[0], pp[1])
        } else {
            camera
        }
    }
}

/// The second-difference penalty on one camera's released spline for one
/// round: `E = w · f_ref² · Σᵢ (c_{i−1} − 2cᵢ + c_{i+1})²` over every interior
/// coefficient `i` (`1 ≤ i ≤ N − 2`, so no row reaches the two gauge-anchored
/// coefficients), with `w = SMOOTHING · rows_data / N` read through
/// [`smoothing_weight`], the camera refit's weight. Each row is a residual
/// `√w · f_ref · (c_{i−1} − 2cᵢ + c_{i+1})` in pixels, so the penalty enters the
/// normal equations as `PᵀP` and `PᵀP·c`, at the same scale as the data rows'
/// `JᵀJ` and `Jᵀr` under a cost of `Σ r²`.
///
/// Every interior row enters whether or not data reach its coefficients: a
/// coefficient the solve moves without data support is placed by its rows
/// alone, as the smoothest continuation of its supported neighbours.
///
/// `f_ref` is the camera's focal at the start of the round, so the penalty is
/// exactly quadratic in the coefficients and has no focal column, and the cost
/// the damping ladder compares is one function for the whole round.
struct SplinePenalty {
    /// The camera whose spline this is.
    cam: usize,
    /// The slot of the spline's first coefficient in the reduced system.
    slot: usize,
    /// The spline's coefficient count `N`.
    n_coeffs: usize,
    /// `w · f_ref²`, the squared weight of each row.
    scale: f64,
}

impl SplinePenalty {
    /// The penalty of camera `cam`'s released spline of `n_coeffs`
    /// coefficients starting at reduced-system slot `slot`, for a round with
    /// `n_obs` kept observations of that camera and focal `f_ref` at its start.
    /// `None` when the spline has no interior coefficient (`n_coeffs < 3`).
    fn for_round(
        cam: usize,
        slot: usize,
        n_coeffs: usize,
        f_ref: f64,
        n_obs: usize,
    ) -> Option<Self> {
        if n_coeffs < 3 {
            return None;
        }
        Some(Self {
            cam,
            slot,
            n_coeffs,
            scale: smoothing_weight(2 * n_obs, n_coeffs) * f_ref * f_ref,
        })
    }

    /// The interior coefficients `i`, each with its row `(i − 1, i, i + 1)`.
    fn rows(&self) -> std::ops::Range<usize> {
        1..self.n_coeffs - 1
    }

    /// The penalty at coefficients `c`.
    fn cost(&self, c: &[f64]) -> f64 {
        self.scale
            * self
                .rows()
                .map(|i| (c[i - 1] - 2.0 * c[i] + c[i + 1]).powi(2))
                .sum::<f64>()
    }

    /// Adds the penalty's `PᵀP` to `h` and `PᵀP·c` to `g`, at coefficients `c`.
    fn accumulate(&self, c: &[f64], h: &mut DMatrix<f64>, g: &mut DVector<f64>) {
        const STENCIL: [f64; 3] = [1.0, -2.0, 1.0];
        for i in self.rows() {
            let r = c[i - 1] - 2.0 * c[i] + c[i + 1];
            for (a, sa) in STENCIL.iter().enumerate() {
                let ia = self.slot + i - 1 + a;
                g[ia] += self.scale * sa * r;
                for (b, sb) in STENCIL.iter().enumerate() {
                    h[(ia, self.slot + i - 1 + b)] += self.scale * sa * sb;
                }
            }
        }
    }
}

/// Whether a released spline's coefficients move in a linearization whose
/// data diagonal over them is `data_diag`: when at least two are supported (a
/// positive data diagonal, from the data alone). Two supported coefficients
/// are the fewest that anchor the linear continuation the penalty extends past
/// them, so the penalty then places every unsupported coefficient. With fewer,
/// it would place them relative to at most one value the data hold, so the
/// spline is held for that linearization.
fn spline_anchored(data_diag: &[f64]) -> bool {
    data_diag.iter().filter(|&&d| d > 0.0).count() >= 2
}

/// How one round's Levenberg–Marquardt solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoundEnd {
    /// Two accepted full steps in a row each improved the cost by less than
    /// `1e-8` relative, or no rung of a damping ladder found a downhill step
    /// and none of them had to hold a distortion.
    Converged,
    /// The poses and points settled with a camera's distortion held at its
    /// step guard: two accepted held steps in a row each improved the cost by
    /// less than `1e-8` relative, or no rung of a ladder in which some rung
    /// held a distortion found a downhill step. Not converged: the step that
    /// would lower the cost further is one the guard forbids.
    SettledAtGuard,
    /// The iteration budget ran out, or the solve was cancelled.
    Stopped,
}

/// A candidate step's lens state, as the step guards read it.
enum LensStep {
    /// Every camera's candidate passes its guards: the candidate focals, `k1`s,
    /// splines (`None` for a camera whose spline does not move) and principal
    /// points.
    Admissible(Vec<f64>, Vec<f64>, Vec<Option<Vec<f64>>>, Vec<[f64; 2]>),
    /// The cameras, not yet held, whose finite distortion step fails its
    /// guard. Every focal and principal point passed.
    Inadmissible(Vec<usize>),
    /// A focal that is not finite and above `1e-6`, a principal point that is
    /// not finite or lies off the image, a distortion step that is not finite,
    /// or a held `k1` that the candidate focal folds: the whole step is
    /// rejected.
    Rejected,
}

/// What one round's solve reads for the per-image release.
struct ImageRound<'a> {
    /// Each camera's prior widths for the round.
    widths: &'a [ImagePriorWidths],
    /// The noise level `σ_n` the prior is weighed at, px: the prior's weight on
    /// a deviation `d` of width `σ` is `σ_n²·d²/σ²`, in the units of the data's
    /// `Σ r²`.
    sigma_n: f64,
}

/// What one round's solve hands back for the per-image release.
#[derive(Debug, Default)]
struct ImageRoundResult {
    /// The images the round solved, by image index: the posed images with a
    /// kept observation, in index order.
    images: Vec<usize>,
    /// Per image of `images`, the inverse of its data-only block of the
    /// reduced system at the round's last linearisation (the points eliminated,
    /// no prior, no damping), over the adjustment slots its camera releases, in
    /// the order log focal scale, `a_x`, `a_y`, and zero in a slot its camera does
    /// not release. Multiplied by `σ_n²` it is the adjustments' covariance with
    /// every other image and the shared camera held. `None` where the block is
    /// not positive definite, or the camera releases nothing per image.
    inv_info: Vec<Option<Matrix3<f64>>>,
}

/// Component `k` of an adjustment in the order of an image's adjustment slots:
/// the log focal scale `ln(1 + s)`, `a_x`, `a_y`.
#[inline]
fn adj_component(a: &IntrinsicAdjustment, k: usize) -> f64 {
    match k {
        0 => log_focal_scale(a.focal),
        1 => a.principal_point[0],
        _ => a.principal_point[1],
    }
}

/// The prior's cost on the compact adjustments `adj`: per camera with a
/// per-image release, `w·Σ (x_i − x̄)²` over its images `cam_imgs[j]` for each
/// released parameter (the focal as its log scale, [`adj_component`]), with
/// `w = σ_n²/σ²`.
fn image_prior_cost(
    lenses: &[Lens<'_>],
    cam_imgs: &[Vec<usize>],
    weights: &[[f64; 2]],
    adj: &[IntrinsicAdjustment],
) -> f64 {
    let mut cost = 0.0;
    for (j, imgs) in cam_imgs.iter().enumerate() {
        if imgs.len() < 2 {
            continue;
        }
        let n = imgs.len() as f64;
        let [w_f, w_pp] = weights[j];
        if lenses[j].opt_image_f {
            let mean = imgs.iter().map(|&c| adj_component(&adj[c], 0)).sum::<f64>() / n;
            cost += w_f
                * imgs
                    .iter()
                    .map(|&c| (adj_component(&adj[c], 0) - mean).powi(2))
                    .sum::<f64>();
        }
        if lenses[j].opt_image_pp {
            for axis in 0..2 {
                let mean = imgs
                    .iter()
                    .map(|&c| adj[c].principal_point[axis])
                    .sum::<f64>()
                    / n;
                cost += w_pp
                    * imgs
                        .iter()
                        .map(|&c| (adj[c].principal_point[axis] - mean).powi(2))
                        .sum::<f64>();
            }
        }
    }
    cost
}

/// The cameras a round's observations are read through at the shared cameras
/// `cams`: `cams` themselves, then entry `cams.len() + r` as camera
/// `view_camera[cams.len() + r]` with `view_adjustment[r]` applied. Borrowed
/// when no entry is adjusted, so a solve with no adjustment reads the cameras
/// it always did.
fn entries_at<'c>(
    cams: &'c [CameraIntrinsics],
    view_camera: &[usize],
    view_adjustment: &[IntrinsicAdjustment],
) -> Cow<'c, [CameraIntrinsics]> {
    if view_adjustment.is_empty() {
        return Cow::Borrowed(cams);
    }
    let mut out = cams.to_vec();
    out.extend(
        view_adjustment
            .iter()
            .zip(&view_camera[cams.len()..])
            .map(|(a, &j)| cams[j].with_adjustment(a)),
    );
    Cow::Owned(out)
}

/// Pins `slot` of a reduced system: an identity row and column and a zero
/// gradient entry, so the slot solves to zero and holds its value.
fn pin_slot(s: &mut DMatrix<f64>, g: &mut DVector<f64>, slot: usize) {
    s.row_mut(slot).fill(0.0);
    s.column_mut(slot).fill(0.0);
    s[(slot, slot)] = 1.0;
    g[slot] = 0.0;
}

/// The summed penalty of `penalties` with each camera's coefficients read
/// through `spline_of`.
fn penalty_cost<'a>(penalties: &[SplinePenalty], spline_of: impl Fn(usize) -> &'a [f64]) -> f64 {
    penalties.iter().map(|p| p.cost(spline_of(p.cam))).sum()
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
/// `anchors`, under [`FreePointPolicy::cross`], holds one anchor per point,
/// and every free point in the solve with a finite anchor is solved in inverse
/// depth about it: a unit direction `u` in its tangent plane and `ρ ≥ 0`, three
/// slots like a finite point's, with the point at `a + u/ρ`. A step that would
/// take `ρ` below zero stops it at zero, and a point already at zero whose
/// gradient asks for a negative `ρ` has its `ρ` slot pinned for that
/// iteration, so it steps in `u` alone. The point is handed in and back in the
/// arrays' own representation: a position where `ρ > 0` and the direction `u`
/// where `ρ = 0`, which is what `is_dir` is updated to say. An image's
/// translation is pinned for an iteration in which none of its kept
/// observations carries a translation column, which an inverse-depth point at
/// `ρ = 0` does not.
///
/// A released spline is regularized by a [`SplinePenalty`] over all its
/// interior coefficients, fixed for the round. Its rows are added to the camera
/// block after the data and before the damping, and its value to the cost the
/// damping ladder compares. A coefficient no kept observation supports, here
/// or at the round's first linearization, is placed by those rows; one that
/// loses its support within the round keeps its value until the next. The
/// support test reads the data alone, and a spline with fewer than two
/// supported coefficients in a linearization is held there
/// ([`spline_anchored`]).
///
/// A candidate step whose only fault is a camera's distortion (a `k1` that
/// folds the imaged field, or a spline that breaks monotonicity) is solved
/// again at the same damping with that camera's distortion slots pinned, so
/// the focal, the principal point, the poses and the points still move. Only a
/// step that fails with the distortion held, whose focal or distortion is not
/// finite, or whose principal point is not finite or lies off the image, is
/// rejected whole and re-damped.
///
/// Returns how the round ended: [`RoundEnd::Converged`],
/// [`RoundEnd::SettledAtGuard`] (said on `progress`, naming the held
/// cameras), or [`RoundEnd::Stopped`] on `max_iters` or a cancellation.
///
/// The reduced system opens with one lens block per camera,
/// `[f_j, k1_j, (c_j,0..c_j,N_j−1), cx_j, cy_j | …]`, then the poses. `lenses`
/// carries each camera's state in and its solved state out. A camera none of
/// whose images has a kept observation in this round has its whole block pinned
/// and comes back exactly as it went in.
///
/// `CAM_COLS` selects the per-observation camera-block width:
/// [`BASE_CAM_COLS`] for every solve in which no camera releases a spline or a
/// principal point (the original layout, byte for byte), [`BSPLINE_CAM_COLS`]
/// when a spline is released — a camera whose spline is released then carries
/// one slot per coefficient in its block (`4 + N_j`, still dynamic) while each
/// observation's block stays compile-time sized at the spline's local support —
/// and [`BASE_PP_CAM_COLS`] or [`BSPLINE_PP_CAM_COLS`] when, besides, a
/// principal point is. The lens block holds its `cx` and `cy` slots whatever
/// the width; an instantiation without their columns leaves them pinned.
///
/// `progress` counts the iterations, names the stages inside one of them at the
/// detail level, and is polled at the top of each: an iteration boundary is
/// where this solve holds a consistent state, since a candidate step is only
/// ever written once it has improved the cost. A stopped solve scatters back
/// what the last accepted step left, which is what it would have returned had
/// the budget run out there.
#[allow(clippy::too_many_arguments)]
fn solve_lm<const CAM_COLS: usize>(
    lenses: &mut [Lens<'_>],
    image_camera: &[u32],
    adjustments: &mut [IntrinsicAdjustment],
    image_round: Option<&ImageRound<'_>>,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    is_dir: &mut [bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    kept: &[usize],
    cons: &Constraints,
    anchors: Option<&[Vector3<f64>]>,
    loss_scale: f64,
    max_iters: usize,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    progress: &Progress<'_>,
) -> (RoundEnd, ImageRoundResult) {
    // The spline instantiation is selected by width; the staged loop requests
    // it exactly when some camera releases a well-formed spline.
    debug_assert!(
        CAM_COLS == BSPLINE_CAM_COLS
            || CAM_COLS == BSPLINE_PP_CAM_COLS
            || CAM_COLS == BSPLINE_IMG_CAM_COLS
            || lenses.iter().all(|l| !l.opt_bspline),
        "a released spline needs the spline instantiation"
    );
    debug_assert!(
        carries_pp_cols(CAM_COLS) || lenses.iter().all(|l| !l.opt_pp),
        "a released principal point needs an instantiation with its columns"
    );
    debug_assert!(
        lenses.iter().all(|l| !(l.opt_bspline && l.opt_k1)),
        "opt_k1 and opt_bspline live on different models"
    );
    // The per-image columns, and the slots each image's block carries for
    // them, only in the instantiations a per-image release selects.
    let image_cols = carries_image_cols(CAM_COLS);
    let img_cols = image_block_cols(CAM_COLS);
    let pose_c = CAM_COLS - img_cols;
    debug_assert!(
        image_cols || lenses.iter().all(|l| !l.releases_image()),
        "a per-image release needs an instantiation with its columns"
    );
    debug_assert!(
        !image_cols || (image_round.is_some() && !adjustments.is_empty()),
        "a per-image release needs its prior and every image's adjustment"
    );
    let n_cam = lenses.len();
    // Each camera's spline shape where its spline is released, and where its
    // lens block starts in the reduced system.
    let shapes: Vec<(usize, f64, SplineRadial)> = lenses
        .iter()
        .map(|l| match l.base.model.radial_spline() {
            Some((_, d_max, radial)) if l.opt_bspline => (l.bspline.len(), d_max, radial),
            _ => (0, 0.0, SplineRadial::IncidenceAngle),
        })
        .collect();
    let mut slot0 = Vec::with_capacity(n_cam);
    let mut n_shared = 0usize;
    for &(n_coeffs, _, _) in &shapes {
        slot0.push(n_shared);
        n_shared += lens_block_width(n_coeffs);
    }
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
    // The camera of each compact image and of each kept observation, and which
    // cameras this round's observations reach at all.
    let ci_cam: Vec<usize> = img_ids.iter().map(|&i| image_camera[i] as usize).collect();
    let obs_cam: Vec<usize> = obs_ci.iter().map(|&ci| ci_cam[ci]).collect();
    let mut live = vec![false; n_cam];
    for &j in &obs_cam {
        live[j] = true;
    }
    // The entry each compact image is read through: its camera's, or, for an
    // image with an intrinsic adjustment, an entry of its own after the
    // cameras'. `view_camera` is the camera each entry belongs to, and
    // `view_adjustment` the adjustment of each entry past the cameras. With no
    // adjustment every image reads its camera's entry, as before adjustments
    // existed. An image of a camera that releases a per-image parameter has an
    // entry of its own whatever its adjustment, since the round moves it.
    let image_released: Vec<bool> = ci_cam.iter().map(|&j| lenses[j].releases_image()).collect();
    // The compact images' adjustments, which a per-image release moves; zero
    // where the caller handed none.
    let mut adj: Vec<IntrinsicAdjustment> = if adjustments.is_empty() {
        vec![IntrinsicAdjustment::default(); img_ids.len()]
    } else {
        img_ids.iter().map(|&i| adjustments[i]).collect()
    };
    let mut ci_view: Vec<usize> = ci_cam.clone();
    let mut view_camera: Vec<usize> = (0..n_cam).collect();
    // The compact image of each entry past the cameras.
    let mut view_image: Vec<usize> = Vec::new();
    if !adjustments.is_empty() {
        for (ci, a) in adj.iter().enumerate() {
            if a.is_zero() && !image_released[ci] {
                continue;
            }
            ci_view[ci] = view_camera.len();
            view_camera.push(ci_cam[ci]);
            view_image.push(ci);
        }
    }
    let views_of = |adj: &[IntrinsicAdjustment]| -> Vec<IntrinsicAdjustment> {
        view_image.iter().map(|&ci| adj[ci]).collect()
    };
    let mut view_adjustment: Vec<IntrinsicAdjustment> = views_of(&adj);
    let obs_view: Vec<usize> = obs_ci.iter().map(|&ci| ci_view[ci]).collect();
    // Each camera's images in the round, the set its prior's mean is over.
    let mut cam_imgs: Vec<Vec<usize>> = vec![Vec::new(); n_cam];
    if image_cols {
        for (ci, &j) in ci_cam.iter().enumerate() {
            if lenses[j].releases_image() {
                cam_imgs[j].push(ci);
            }
        }
    }
    // The prior's weights `σ_n²/σ²` per camera, `[focal, principal point]`.
    let prior_weights: Vec<[f64; 2]> = match image_round {
        Some(r) if image_cols => r
            .widths
            .iter()
            .map(|w| {
                let s2 = r.sigma_n * r.sigma_n;
                [
                    s2 / (w.focal * w.focal),
                    s2 / (w.principal_point * w.principal_point),
                ]
            })
            .collect(),
        _ => vec![[0.0; 2]; n_cam],
    };
    let prior_cost = |adj: &[IntrinsicAdjustment]| -> f64 {
        if image_cols {
            image_prior_cost(lenses, &cam_imgs, &prior_weights, adj)
        } else {
            0.0
        }
    };
    // Free points solved in inverse depth: the anchor of each, `None` for every
    // other point. A free point whose position sits exactly on its anchor has
    // no direction from it, and stays a position for the round.
    let cp_inverse: Vec<Option<Vector3<f64>>> = match anchors {
        None => Vec::new(),
        Some(anchors) => pt_ids
            .iter()
            .map(|&p| {
                if cons.constraint[p] != PointConstraint::Free {
                    return None;
                }
                let a = anchors[p];
                if !(a.x.is_finite() && a.y.is_finite() && a.z.is_finite()) {
                    return None;
                }
                let x = Vector3::new(points[p][0], points[p][1], points[p][2]);
                let usable = if is_dir[p] {
                    x.norm() > 0.0
                } else {
                    let n = (x - a).norm();
                    n > 0.0 && n.is_finite()
                };
                usable.then_some(a)
            })
            .collect(),
    };
    let has_inverse = cp_inverse.iter().any(Option::is_some);
    let is_inverse = |c: usize| has_inverse && cp_inverse[c].is_some();
    // A point in inverse depth is never a direction here: its `ρ` says whether
    // it is at infinity.
    let cp_dir: Vec<bool> = pt_ids
        .iter()
        .enumerate()
        .map(|(c, &p)| is_dir[p] && !is_inverse(c))
        .collect();
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

    // Translation observability: an image none of whose kept observations
    // carries a translation column gets its translation pinned (a direction's
    // translation Jacobian is identically zero, and so is that of a point in
    // inverse depth at `ρ = 0`, so the block would otherwise be pure zero
    // curvature in the reduced system). Without points in inverse depth this
    // is the same every iteration, and pinned for the whole round.
    let translation_support = |inv_depth: &[f64]| {
        let mut support = vec![false; n_im];
        for (kk, &ci) in obs_ci.iter().enumerate() {
            let cp = obs_cp[kk];
            let carries = if is_inverse(cp) {
                inv_depth[cp] > 0.0
            } else {
                !cp_dir[cp]
            };
            if carries {
                support[ci] = true;
            }
        }
        support
    };

    // Per-point observation lists (compact indices into `kept`).
    let mut pt_obs: Vec<Vec<usize>> = vec![Vec::new(); n_pt];
    for (kk, &cp) in obs_cp.iter().enumerate() {
        pt_obs[cp].push(kk);
    }

    // Working state (compact copies). Direction rows arrive unit-normalized
    // (input normalization / re-estimation) and every accepted step
    // re-normalizes them.
    let mut fs: Vec<f64> = lenses.iter().map(|l| l.f).collect();
    let mut k1s: Vec<f64> = lenses.iter().map(|l| l.k1).collect();
    let mut bsplines: Vec<Vec<f64>> = lenses.iter().map(|l| l.bspline.clone()).collect();
    let mut pps: Vec<[f64; 2]> = lenses.iter().map(|l| l.pp).collect();
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
    // A point in inverse depth holds its unit direction from the anchor in `x`
    // and its inverse depth here: a position `X` is `u = (X − a)/‖X − a‖`,
    // `ρ = 1/‖X − a‖`, and a direction `d` is `u = d`, `ρ = 0`.
    let mut inv_depth: Vec<f64> = if has_inverse {
        vec![0.0; n_pt]
    } else {
        Vec::new()
    };
    if has_inverse {
        for (c, a) in cp_inverse.iter().enumerate() {
            let Some(a) = a else { continue };
            if is_dir[pt_ids[c]] {
                x[c] = x[c].normalize();
            } else {
                let w = x[c] - a;
                let n = w.norm();
                x[c] = w / n;
                inv_depth[c] = 1.0 / n;
            }
        }
    }
    // Reduced camera system: every camera's lens block, then 6 per image, or 9
    // in a solve that releases a per-image parameter. A block's two scalar
    // slots and two principal-point slots are always present (pinned when
    // unreleased) to keep the indexing uniform, its coefficient slots only
    // under its own spline release, and an image's three adjustment slots
    // likewise whenever the instantiation carries them.
    let d = n_shared + img_cols * n_im;
    // First pose slot of compact image `ci` in the reduced camera system.
    let img_slot = |ci: usize| n_shared + img_cols * ci;
    // Each camera's imaged field, as its kept observations report it: the
    // largest pixel radius from its principal point at the start of the round.
    // The `k1` step guard asks whether that camera's distorted map stays
    // monotone out to here. An adjusted image's radius is measured from its own
    // principal point and divided by its focal scale `1 + s`, so that it is
    // compared with the fold radius of the shared focal: the image folds at
    // `(1 + s)` times the shared radius.
    let mut field_r = vec![0.0f64; n_cam];
    for (kk, &k) in kept.iter().enumerate() {
        let j = obs_cam[kk];
        let [cx, cy] = lenses[j].pp;
        let v = obs_view[kk];
        let r = if v < n_cam {
            (uv[k][0] - cx).hypot(uv[k][1] - cy)
        } else {
            let a = view_adjustment[v - n_cam];
            let m = f64::from(lenses[j].base.width.min(lenses[j].base.height));
            let (cx, cy) = (cx + a.principal_point[0] * m, cy + a.principal_point[1] * m);
            (uv[k][0] - cx).hypot(uv[k][1] - cy) / (1.0 + a.focal)
        };
        field_r[j] = f64::max(field_r[j], r);
    }
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
    let cost_at = |cams: &[CameraIntrinsics],
                   view_adjustment: &[IntrinsicAdjustment],
                   q: &[UnitQuaternion<f64>],
                   t: &[Vector3<f64>],
                   x: &[Vector3<f64>],
                   inv_depth: &[f64]| {
        let owned;
        let xp: &[Vector3<f64>] = if has_ranged {
            owned = positions(x, q, t);
            &owned
        } else {
            x
        };
        robust_cost(
            &entries_at(cams, &view_camera, view_adjustment),
            &obs_view,
            q,
            t,
            xp,
            &cp_dir,
            &cp_inverse,
            inv_depth,
            uv,
            kept,
            &obs_ci,
            &obs_cp,
            &s2s,
        )
    };
    let mut lambda = 1e-3;
    // Runs of accepted steps that each improved the cost by less than `1e-8`
    // relative: full steps, and steps retried with a distortion held.
    let mut tiny_full = 0usize;
    let mut tiny_held = 0usize;
    let mut ending: Option<RoundEnd> = None;
    // The cameras whose distortion the latest retry held.
    let mut guard_cams: Vec<usize> = Vec::new();
    let releases_distortion =
        (0..n_cam).any(|j| live[j] && (lenses[j].opt_k1 || lenses[j].opt_bspline));
    let mut cams: Vec<CameraIntrinsics> = lenses.iter().map(Lens::camera).collect();
    // The spline penalties, fixed for the round, and the cost the ladder
    // compares: the data's plus theirs.
    let penalties: Vec<SplinePenalty> = (0..n_cam)
        .filter(|&j| live[j] && lenses[j].opt_bspline)
        .filter_map(|j| {
            let n_obs = obs_cam.iter().filter(|&&c| c == j).count();
            SplinePenalty::for_round(j, slot0[j] + BSPLINE_SLOT0, shapes[j].0, fs[j], n_obs)
        })
        .collect();
    let mut prev_cost = cost_at(&cams, &view_adjustment, &q, &t, &x, &inv_depth)
        + penalty_cost(&penalties, |j| &bsplines[j])
        + prior_cost(&adj);
    // The latest linearisation's data-only block of each image's slots, the
    // points eliminated, for the derivation of the prior's widths, and which
    // images carried a translation column in it.
    let mut info: Vec<SMatrix<f64, 9, 9>> = Vec::new();
    let mut info_translation: Vec<bool> = Vec::new();
    let mut round_support: Vec<bool> = Vec::new();

    let analytic: Vec<bool> = cams
        .iter()
        .map(|c| c.model.supports_pixel_jacobian())
        .collect();
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
            .enumerate()
            .map(|(c, (xd, &dir))| {
                if dir || is_inverse(c) {
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
        let blocks: Vec<ObsBlocks<CAM_COLS>> = {
            let entries = entries_at(&cams, &view_camera, &view_adjustment);
            let lin: Vec<LinLens<'_>> = (0..entries.len())
                .map(|v| {
                    let j = view_camera[v];
                    let (cx, cy) = entries[v].principal_point();
                    let s = if v < n_cam {
                        0.0
                    } else {
                        view_adjustment[v - n_cam].focal
                    };
                    let f_image = if v < n_cam { fs[j] } else { fs[j] * (1.0 + s) };
                    LinLens {
                        cam: &entries[v],
                        analytic: analytic[j],
                        cx,
                        cy,
                        f: fs[j],
                        f_image,
                        m: f64::from(lenses[j].base.width.min(lenses[j].base.height)),
                        opt_f: lenses[j].opt_f,
                        opt_k1: lenses[j].opt_k1,
                        opt_bspline: lenses[j].opt_bspline,
                        opt_pp: lenses[j].opt_pp,
                        opt_image_f: lenses[j].opt_image_f,
                        opt_image_pp: lenses[j].opt_image_pp,
                        n_coeffs: shapes[j].0,
                        d_max: shapes[j].1,
                        radial: shapes[j].2,
                        slot0: slot0[j],
                    }
                })
                .collect();
            let st = LinState {
                lenses: &lin,
                ci_cam: &ci_view,
                q: &q,
                t: &t,
                xp,
                bases: &bases,
                cp_dir: &cp_dir,
                cp_held: &cp_held,
                cp_origin: &cp_origin,
                cp_inverse: &cp_inverse,
                inv_depth: &inv_depth,
                uv,
                n_shared,
                img_cols,
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
        // The lens slots' curvature from the data alone: a spline coefficient
        // is supported in this linearization when it is positive, whatever the
        // penalty adds below.
        let data_diag: Vec<f64> = (0..n_shared).map(|slot| h_cc[(slot, slot)]).collect();
        // Which coefficients the round started with support for. One that
        // loses its support within the round keeps the value the round's data
        // gave it, rather than being placed by the penalty alone in the middle
        // of a step that moved its observations out of its span.
        if iter == 0 {
            round_support = data_diag.iter().map(|&d| d > 0.0).collect();
        }
        for p in &penalties {
            p.accumulate(&bsplines[p.cam], &mut h_cc, &mut g_c);
        }
        // A point in inverse depth at the bound `ρ = 0` whose gradient asks for
        // a negative `ρ` keeps it at zero this iteration: its `ρ` column is
        // dropped, so it is eliminated as a direction is, over its two tangent
        // slots, and the camera step is not solved as though it could move
        // through infinity.
        let mut cp_bound = vec![false; if has_inverse { n_pt } else { 0 }];
        if has_inverse {
            for (c, bound) in cp_bound.iter_mut().enumerate() {
                if !(is_inverse(c) && inv_depth[c] == 0.0 && g_p[c][2] >= 0.0) {
                    continue;
                }
                *bound = true;
                for r in 0..3 {
                    v_pp[c][(r, 2)] = 0.0;
                    v_pp[c][(2, r)] = 0.0;
                }
                g_p[c][2] = 0.0;
                for &a in &pt_obs[c] {
                    w_cp[a].column_mut(2).fill(0.0);
                }
            }
        }
        // Which images carry a translation column this iteration.
        let img_has_finite = translation_support(&inv_depth);
        let any_frozen = img_has_finite.iter().any(|&h| !h);

        if image_cols {
            // Each released image's own block of the reduced system from the
            // data alone, the points eliminated undamped: what the derivation
            // of the prior's widths reads its variances from. The coupling a
            // ranged point's reference columns add to an image's pose is left
            // out, being a share of another image's observation.
            info = vec![SMatrix::<f64, 9, 9>::zeros(); n_im];
            for (ci, block) in info.iter_mut().enumerate() {
                if image_released[ci] {
                    let o = img_slot(ci);
                    *block = h_cc.fixed_view::<9, 9>(o, o).into_owned();
                }
            }
            for (p, obs) in pt_obs.iter().enumerate() {
                if cp_held[p] {
                    continue;
                }
                let mut vd = v_pp[p];
                if cp_tangent[p] || (has_inverse && cp_bound[p]) {
                    vd[(2, 2)] = 1.0;
                }
                let Some(v_inv) = vd.try_inverse() else {
                    continue;
                };
                for &a in obs {
                    let ci = obs_ci[a];
                    if !image_released[ci] {
                        continue;
                    }
                    let wa = w_cp[a].fixed_rows::<9>(pose_c).into_owned();
                    for &b in obs.iter().filter(|&&b| obs_ci[b] == ci) {
                        let wb = w_cp[b].fixed_rows::<9>(pose_c).into_owned();
                        info[ci] -= wa * v_inv * wb.transpose();
                    }
                }
            }
            info_translation = img_has_finite.clone();
            // The prior on each camera's images: `w·(I − 𝟏𝟏ᵀ/n)` over the
            // camera's images in the round, per released parameter, and
            // `w·(x_i − x̄)` in each image's gradient entry. Added exactly: the
            // reduced system is dense, so the coupling between a camera's
            // images costs nothing the factorization does not already pay.
            for (j, imgs) in cam_imgs.iter().enumerate() {
                if imgs.len() < 2 {
                    continue;
                }
                let n = imgs.len() as f64;
                let [w_f, w_pp] = prior_weights[j];
                let mut kinds: Vec<(usize, f64)> = Vec::new();
                if lenses[j].opt_image_f {
                    kinds.push((0, w_f));
                }
                if lenses[j].opt_image_pp {
                    kinds.extend([(1, w_pp), (2, w_pp)]);
                }
                for &(k, w) in &kinds {
                    let mean = imgs.iter().map(|&c| adj_component(&adj[c], k)).sum::<f64>() / n;
                    for &ca in imgs {
                        let ia = img_slot(ca) + IMG_S_SLOT + k;
                        g_c[ia] += w * (adj_component(&adj[ca], k) - mean);
                        for &cb in imgs {
                            let ib = img_slot(cb) + IMG_S_SLOT + k;
                            let delta = if ca == cb { 1.0 } else { 0.0 };
                            h_cc[(ia, ib)] += w * (delta - 1.0 / n);
                        }
                    }
                }
            }
        }

        drop(equations);

        // ── Damping ladder: re-damp and re-solve from this linearization ──
        let ladder = progress.detail_phase("damping ladder");
        let mut improved = false;
        // Whether a rung of this ladder retried its step with a distortion held.
        let mut ladder_held = false;
        for _ in 0..12 {
            // Every rejection raises `λ` fourfold and moves to the next rung,
            // so the cap read here ends the ladder after any of them.
            if lambda > 1e12 {
                break;
            }
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
            // The damped tangent blocks of the points in inverse depth, for the
            // step in `u` alone that replaces one taking `ρ` through zero.
            let mut v_tangent: Vec<Matrix2<f64>> = if has_inverse {
                Vec::with_capacity(n_pt)
            } else {
                Vec::new()
            };
            let mut singular = false;
            for (p, v) in v_pp.iter().enumerate() {
                // A held point has no point block at all; the identity keeps
                // the uniform indexing while its zero gradient and zero `W`
                // make every contribution exactly zero.
                if cp_held[p] {
                    v_inv.push(Matrix3::identity());
                    if has_inverse {
                        v_tangent.push(Matrix2::identity());
                    }
                    continue;
                }
                let mut vd = *v;
                for dd in 0..3 {
                    vd[(dd, dd)] += lambda * v[(dd, dd)].max(1e-12);
                }
                if cp_tangent[p] || (has_inverse && cp_bound[p]) {
                    vd[(2, 2)] = 1.0;
                }
                if has_inverse {
                    v_tangent.push(vd.fixed_view::<2, 2>(0, 0).into_owned());
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
            // Pin every lens slot that is not released (its column is already
            // exactly zero; this keeps the reduced system regular), and every
            // slot of a camera no kept observation reaches, whose columns are
            // zero whatever it releases. A coefficient of a released spline
            // that no kept observation supports (its data diagonal exactly
            // zero, every contribution being a square) is placed by its
            // penalty rows, which tie it to its supported neighbours, when it
            // had no support at the round's first linearization either; one
            // that has lost its support within the round is pinned at its
            // value until the next round. A spline with fewer than two
            // supported coefficients, which the rows cannot anchor, has every
            // coefficient slot pinned. A pinned slot holds its value exactly,
            // like a frozen translation.
            for (j, lens) in lenses.iter().enumerate() {
                let n_coeffs = shapes[j].0;
                let c0 = slot0[j] + BSPLINE_SLOT0;
                let spline_free = n_coeffs > 0 && spline_anchored(&data_diag[c0..c0 + n_coeffs]);
                for local in 0..lens_block_width(n_coeffs) {
                    let slot = slot0[j] + local;
                    let released = live[j]
                        && match local {
                            F_SLOT => lens.opt_f,
                            K1_SLOT => lens.opt_k1,
                            l if l >= pp_slot(n_coeffs) => lens.opt_pp,
                            _ => spline_free && (data_diag[slot] > 0.0 || !round_support[slot]),
                        };
                    if !released {
                        pin_slot(&mut s, &mut g_red, slot);
                    }
                }
            }
            if any_frozen {
                // Pin the translation slots of all-direction images (frozen
                // for the round; their rotations still update).
                for (c, &has_finite) in img_has_finite.iter().enumerate() {
                    if has_finite {
                        continue;
                    }
                    for r in 0..3 {
                        pin_slot(&mut s, &mut g_red, img_slot(c) + 3 + r);
                    }
                }
            }
            if image_cols {
                // Pin each image's adjustment slots its camera does not
                // release.
                for (c, &j) in ci_cam.iter().enumerate() {
                    let o = img_slot(c) + IMG_S_SLOT;
                    if !lenses[j].opt_image_f {
                        pin_slot(&mut s, &mut g_red, o);
                    }
                    if !lenses[j].opt_image_pp {
                        pin_slot(&mut s, &mut g_red, o + 1);
                        pin_slot(&mut s, &mut g_red, o + 2);
                    }
                }
            }

            // The reduced system before its factorization, kept for a retry
            // with a camera's distortion held; only a solve that releases a
            // distortion can need one.
            let reduced = releases_distortion.then(|| (s.clone(), g_red.clone()));
            let Some(mut delta) = s.lu().solve(&(-g_red)) else {
                lambda *= 4.0;
                continue;
            };

            // Candidate lens state, camera by camera, each step checked against
            // that camera's own model and field. A camera no kept observation
            // reaches is not stepped at all, and a camera in `held` keeps its
            // distortion.
            let lens_step = |delta: &DVector<f64>, held: &[bool]| {
                let mut f_cand = fs.clone();
                let mut k1_cand = k1s.clone();
                let mut bspline_cand: Vec<Option<Vec<f64>>> = vec![None; n_cam];
                let mut pp_cand = pps.clone();
                let mut inadmissible = Vec::new();
                for (j, lens) in lenses.iter().enumerate() {
                    if !live[j] {
                        continue;
                    }
                    let o = slot0[j];
                    if lens.opt_f {
                        f_cand[j] = fs[j] + delta[o + F_SLOT];
                        if !(f_cand[j].is_finite() && f_cand[j] > 1e-6) {
                            return LensStep::Rejected;
                        }
                    }
                    // The principal point's guard: finite and on the sensor.
                    if lens.opt_pp {
                        let p = o + pp_slot(shapes[j].0);
                        pp_cand[j] = [pps[j][0] + delta[p], pps[j][1] + delta[p + 1]];
                        let [cx, cy] = pp_cand[j];
                        if !pp_step_admissible(cx, cy, lens.base.width, lens.base.height) {
                            return LensStep::Rejected;
                        }
                    }
                    // The curvature rung's plausibility guard: a step that
                    // folds the distorted map inside the imaged field marks
                    // the camera for a retry with its `k1` held. A held `k1` is
                    // checked again, since the guard reads the candidate focal.
                    if lens.opt_k1 {
                        if held[j] {
                            if !k1_step_admissible(f_cand[j], k1s[j], field_r[j]) {
                                return LensStep::Rejected;
                            }
                        } else {
                            k1_cand[j] = k1s[j] + delta[o + K1_SLOT];
                            if !k1_cand[j].is_finite() {
                                return LensStep::Rejected;
                            }
                            if !k1_step_admissible(f_cand[j], k1_cand[j], field_r[j]) {
                                inadmissible.push(j);
                            }
                        }
                    }
                    // The spline rung's plausibility guard: a step that folds
                    // the spline map anywhere on its domain marks the camera the
                    // same way. A held spline is the current, admissible one.
                    if lens.opt_bspline && !held[j] {
                        let mut pc = bsplines[j].clone();
                        for (i, c) in pc.iter_mut().enumerate() {
                            let dv = delta[o + BSPLINE_SLOT0 + i];
                            // Pinned slots (an unanchored spline, or a
                            // coefficient that lost its support within the
                            // round) solve to exactly zero;
                            // skip the add so a `−0.0` coefficient keeps its
                            // sign (the frozen-translation precedent).
                            if dv != 0.0 {
                                *c += dv;
                            }
                        }
                        if !pc.iter().all(|c| c.is_finite()) {
                            return LensStep::Rejected;
                        }
                        if bspline_step_admissible(&pc, shapes[j].1) {
                            bspline_cand[j] = Some(pc);
                        } else {
                            inadmissible.push(j);
                        }
                    }
                }
                if inadmissible.is_empty() {
                    LensStep::Admissible(f_cand, k1_cand, bspline_cand, pp_cand)
                } else {
                    LensStep::Inadmissible(inadmissible)
                }
            };
            // A step whose only fault is a camera's distortion is solved again
            // at the same damping with that distortion held, so the focal, the
            // poses and the points still move. Pinning a slot after the Schur
            // elimination removes that variable from the damped system, so the
            // back-substitution below is the exact solution with it held. The
            // held set only grows, so this ends within `n_cam` solves.
            let mut held = vec![false; n_cam];
            let step = loop {
                match lens_step(&delta, &held) {
                    LensStep::Inadmissible(marked) => {
                        let Some((s0, g0)) = &reduced else {
                            break LensStep::Rejected;
                        };
                        ladder_held = true;
                        for j in marked {
                            held[j] = true;
                        }
                        guard_cams = (0..n_cam).filter(|&j| held[j]).collect();
                        let (mut s_held, mut g_held) = (s0.clone(), g0.clone());
                        for j in (0..n_cam).filter(|&j| held[j]) {
                            if lenses[j].opt_k1 {
                                pin_slot(&mut s_held, &mut g_held, slot0[j] + K1_SLOT);
                            }
                            for i in 0..shapes[j].0 {
                                pin_slot(&mut s_held, &mut g_held, slot0[j] + BSPLINE_SLOT0 + i);
                            }
                        }
                        match s_held.lu().solve(&(-g_held)) {
                            Some(dh) => delta = dh,
                            None => break LensStep::Rejected,
                        }
                    }
                    other => break other,
                }
            };
            let LensStep::Admissible(f_cand, k1_cand, bspline_cand, pp_cand) = step else {
                lambda *= 4.0;
                continue;
            };
            // Each image's candidate adjustment, and its guards: an image's
            // log focal scale stays finite and keeps its focal finite and
            // positive, and an image's principal point, where its camera
            // releases one and the image's is not the shared one, stays on the
            // image. A failure rejects the whole step. The step is in the log
            // focal scale `t`, and the stored scale is `s = exp(t) − 1`.
            let mut adj_cand = adj.clone();
            let mut image_ok = true;
            for (c, &j) in ci_cam.iter().enumerate() {
                let lens = &lenses[j];
                if image_cols {
                    let o = img_slot(c) + IMG_S_SLOT;
                    if lens.opt_image_f {
                        let t = log_focal_scale(adj_cand[c].focal) + delta[o];
                        if !image_focal_step_admissible(f_cand[j], t) {
                            image_ok = false;
                            break;
                        }
                        adj_cand[c].focal = t.exp_m1();
                    }
                    if lens.opt_image_pp {
                        adj_cand[c].principal_point[0] += delta[o + 1];
                        adj_cand[c].principal_point[1] += delta[o + 2];
                    }
                }
                if (lens.opt_pp || lens.opt_image_pp)
                    && (lens.opt_image_pp || !adj_cand[c].is_zero())
                {
                    let m = f64::from(lens.base.width.min(lens.base.height));
                    let [ax, ay] = adj_cand[c].principal_point;
                    let [cx, cy] = pp_cand[j];
                    if !pp_step_admissible(
                        cx + ax * m,
                        cy + ay * m,
                        lens.base.width,
                        lens.base.height,
                    ) {
                        image_ok = false;
                        break;
                    }
                }
            }
            if !image_ok {
                lambda *= 4.0;
                continue;
            }
            let view_adjustment_cand = if image_cols {
                views_of(&adj_cand)
            } else {
                view_adjustment.clone()
            };
            let held_step = held.contains(&true);
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
            let mut inv_depth_cand = inv_depth.clone();
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
                let rhs = g_p[p] + wt_dc;
                let mut dxp = v_inv[p] * rhs;
                if is_inverse(p) {
                    // u ← normalize(u + B(u)·δ), ρ ← ρ + δρ, with ρ held at
                    // its bound: a step past zero from above stops at zero, and
                    // one past zero from zero itself is replaced by the step in
                    // `u` alone, as the point fit of the test does.
                    let mut rho = inv_depth[p] - dxp[2];
                    if rho < 0.0 {
                        if inv_depth[p] == 0.0 {
                            if let Some(inv) = v_tangent[p].try_inverse() {
                                let du = inv * Vector2::new(rhs[0], rhs[1]);
                                dxp = Vector3::new(du[0], du[1], 0.0);
                            }
                        }
                        rho = 0.0;
                    }
                    let (b1, b2) = bases[p];
                    x_cand[p] = (x[p] - (b1 * dxp[0] + b2 * dxp[1])).normalize();
                    inv_depth_cand[p] = rho;
                } else if cp_tangent[p] {
                    // d ← normalize(d + B(d)·δ), the 2-DOF tangent update.
                    let (b1, b2) = bases[p];
                    x_cand[p] = (x[p] - (b1 * dxp[0] + b2 * dxp[1])).normalize();
                } else {
                    x_cand[p] = x[p] - dxp;
                }
            }

            let cams_cand: Vec<CameraIntrinsics> = (0..n_cam)
                .map(|j| {
                    if !live[j] {
                        return cams[j].clone();
                    }
                    let bs = bspline_cand[j].as_deref().unwrap_or(&bsplines[j]);
                    lenses[j].at(f_cand[j], k1_cand[j], bs, pp_cand[j])
                })
                .collect();
            let new_cost = cost_at(
                &cams_cand,
                &view_adjustment_cand,
                &q_cand,
                &t_cand,
                &x_cand,
                &inv_depth_cand,
            ) + penalty_cost(&penalties, |j| {
                bspline_cand[j].as_deref().unwrap_or(&bsplines[j])
            }) + prior_cost(&adj_cand);
            if new_cost < prev_cost {
                let rel = (prev_cost - new_cost) / prev_cost.max(1e-300);
                fs = f_cand;
                k1s = k1_cand;
                pps = pp_cand;
                for (j, pc) in bspline_cand.into_iter().enumerate() {
                    if let Some(pc) = pc {
                        bsplines[j] = pc;
                    }
                }
                q = q_cand;
                t = t_cand;
                x = x_cand;
                inv_depth = inv_depth_cand;
                cams = cams_cand;
                adj = adj_cand;
                view_adjustment = view_adjustment_cand;
                prev_cost = new_cost;
                lambda = (lambda * 0.5).max(1e-12);
                improved = true;
                // Converged only after tiny improvements twice in a row: a
                // single small step is how a traverse of a nearly-flat
                // valley STARTS (the focal release walks −20% through one),
                // so one is not proof of convergence. Full and held steps
                // count separately, and a step of one kind ends the other's
                // run: two tiny full steps converge, two tiny held steps
                // settle against the guard.
                let (run, other) = if held_step {
                    (&mut tiny_held, &mut tiny_full)
                } else {
                    (&mut tiny_full, &mut tiny_held)
                };
                *other = 0;
                *run = if rel < 1e-8 { *run + 1 } else { 0 };
                if tiny_full >= 2 {
                    ending = Some(RoundEnd::Converged);
                } else if tiny_held >= 2 {
                    ending = Some(RoundEnd::SettledAtGuard);
                }
                break;
            }
            lambda *= 4.0;
        }
        drop(ladder);
        // No rung found a downhill step. Where a rung had to hold a distortion,
        // the step that would lower the cost is the one the guard forbids.
        if !improved {
            ending = Some(if ladder_held {
                RoundEnd::SettledAtGuard
            } else {
                RoundEnd::Converged
            });
        }
        if ending.is_some() {
            break;
        }
    }
    let ending = ending.unwrap_or(RoundEnd::Stopped);
    if ending == RoundEnd::SettledAtGuard {
        let held: Vec<String> = guard_cams
            .iter()
            .map(|&j| {
                if lenses[j].opt_bspline {
                    let (_, slope) =
                        bspline_delta_and_deriv(&bsplines[j], shapes[j].1, shapes[j].1);
                    format!("camera {j} (spline end slope {:.3})", 1.0 + slope)
                } else {
                    format!("camera {j} (k1 {:.4})", k1s[j])
                }
            })
            .collect();
        progress_info!(
            progress,
            "the round settled with a distortion held at its step guard: {}",
            held.join(", ")
        );
    }

    // Scatter the compact state back. A ranged point comes back as the
    // position its distance and its reference put it at, re-read at the poses
    // the round settled on; a held point is not written at all. A point in
    // inverse depth comes back as a position where `ρ > 0` and as its direction
    // where `ρ = 0` (or where `a + u/ρ` overflows).
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
        if is_inverse(c) {
            let a = cp_inverse[c].expect("an anchor");
            let rho = inv_depth[c];
            debug_assert!(rho >= 0.0, "an inverse depth below its bound: {rho}");
            let x = a + xf[c] / rho;
            if rho > 0.0 && x.iter().all(|v| v.is_finite()) {
                points[p] = [x.x, x.y, x.z];
                is_dir[p] = false;
            } else {
                points[p] = [xf[c].x, xf[c].y, xf[c].z];
                is_dir[p] = true;
            }
            continue;
        }
        points[p] = [xf[c].x, xf[c].y, xf[c].z];
    }
    for (j, lens) in lenses.iter_mut().enumerate() {
        lens.f = fs[j];
        lens.k1 = k1s[j];
        lens.bspline = std::mem::take(&mut bsplines[j]);
        lens.pp = pps[j];
    }
    let mut image_result = ImageRoundResult::default();
    if image_cols {
        for (ci, &i) in img_ids.iter().enumerate() {
            adjustments[i] = adj[ci];
        }
        image_result.images = img_ids.clone();
        image_result.inv_info = (0..n_im)
            .map(|ci| {
                let lens = &lenses[ci_cam[ci]];
                if info.is_empty() || !lens.releases_image() {
                    return None;
                }
                image_adjustment_inv_info(
                    &info[ci],
                    info_translation[ci],
                    lens.opt_image_f,
                    lens.opt_image_pp,
                )
            })
            .collect();
    }
    (ending, image_result)
}

/// The inverse of one image's data-only block `h` (its pose and adjustment
/// slots) over the slots that move, read back on the adjustment slots: the
/// three rotation slots, the translation where `translation` says the image
/// carries one, and the released adjustment slots. Inverting the whole block
/// counts each adjustment's coupling with the same image's pose against it.
/// The result is in the order log focal scale, `a_x`, `a_y`, zero in a slot not
/// released. `None` where that block is not positive definite.
fn image_adjustment_inv_info(
    h: &SMatrix<f64, 9, 9>,
    translation: bool,
    focal: bool,
    principal_point: bool,
) -> Option<Matrix3<f64>> {
    let mut slots: Vec<usize> = (0..3).collect();
    if translation {
        slots.extend(3..6);
    }
    let first_adj = slots.len();
    if focal {
        slots.push(IMG_S_SLOT);
    }
    if principal_point {
        slots.extend([IMG_S_SLOT + 1, IMG_S_SLOT + 2]);
    }
    let n = slots.len();
    let sub = DMatrix::from_fn(n, n, |r, c| h[(slots[r], slots[c])]);
    let inv = sub.cholesky()?.inverse();
    let mut out = Matrix3::zeros();
    for (r, &sr) in slots.iter().enumerate().skip(first_adj) {
        for (c, &sc) in slots.iter().enumerate().skip(first_adj) {
            out[(sr - IMG_S_SLOT, sc - IMG_S_SLOT)] = inv[(r, c)];
        }
    }
    Some(out)
}

/// The anchor of every free point solved in inverse depth: the centroid of the
/// camera centres of the images observing it, one term per observation, at the
/// poses given. `NaN` for a point that is not free or has no observation.
///
/// It is the default anchor of the test's own point fit
/// ([`fit_point_and_bearing`]), so the solve and the fit read `ρ` about the same
/// place.
fn free_point_anchors(
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    obs_img: &[u32],
    obs_pt: &[u32],
    cons: &Constraints,
    n_pt: usize,
) -> Vec<Vector3<f64>> {
    let centres: Vec<Vector3<f64>> = quats
        .iter()
        .zip(trans)
        .map(|(q, t)| camera_centre(q, t))
        .collect();
    let mut sum = vec![Vector3::zeros(); n_pt];
    let mut count = vec![0usize; n_pt];
    for (&i, &p) in obs_img.iter().zip(obs_pt) {
        let p = p as usize;
        sum[p] += centres[i as usize];
        count[p] += 1;
    }
    (0..n_pt)
        .map(|p| {
            if cons.constraint[p] == PointConstraint::Free && count[p] > 0 {
                sum[p] / count[p] as f64
            } else {
                Vector3::repeat(f64::NAN)
            }
        })
        .collect()
}

/// The storage decision: each free point with an estimate, scored at the poses
/// and cameras the solve ended at and at the noise level `sigma_px`, over the
/// observations of its track in `kept` (the final round's solve set), is stored
/// as a position where [`is_finite`] says its rays ask for a depth and as a
/// direction where they do not. Returns, per point, whether it has an estimate
/// but fewer than two usable kept observations and so is not scored. Such a
/// point is stored as the solve left it where that is in the representation
/// the caller handed in (`input_points`, `input_is_dir`), and otherwise as the
/// caller handed it in, so that a point the test never read is never stored in
/// a representation it did not choose. A point handed in with no estimate has
/// no row to go back to and keeps the representation the re-estimation gave it.
///
/// - Only kept observations are read, so an observation the trim, the in-front
///   floor or `min_track` rejected changes neither the verdict nor the stored
///   bearing, and a track the trim starved is not stored from rays the solve
///   did not fit.
/// - A finite verdict on a point the solve left at a position keeps that
///   position, and on one it left at `ρ = 0` places it at the test's own point
///   fit ([`fit_point_and_bearing`]), where that point lies in front of every
///   observing camera; otherwise the direction stands.
/// - A bearing verdict stores the bearing the test fits, where that bearing
///   lies in front of every observing camera, whether the solve left the point
///   at a position or at `ρ = 0`, so the stored bearing does not depend on
///   which side of the bound the solve's own fit happened to end. A bearing
///   behind a camera describes no sighting there, and what the solve left
///   stands.
#[allow(clippy::too_many_arguments)]
fn decide_free_points(
    cams: &[CameraIntrinsics],
    image_camera: &[u32],
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &mut [[f64; 3]],
    is_dir: &mut [bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    kept: &[usize],
    cons: &Constraints,
    sigma_px: f64,
    input_points: &[[f64; 3]],
    input_is_dir: &[bool],
) -> Vec<bool> {
    let n_pt = points.len();
    let mut track: Vec<Vec<usize>> = vec![Vec::new(); n_pt];
    for &k in kept {
        track[obs_pt[k] as usize].push(k);
    }
    let centres: Vec<Point3<f64>> = quats
        .iter()
        .zip(trans)
        .map(|(q, t)| Point3::from(camera_centre(q, t)))
        .collect();
    let fit_options = PointBearingFitOptions::default();
    // Per point: the representation to store (the row and whether it is a
    // direction), where it changes, and whether the point has an estimate but
    // too few kept rays to score.
    type Stored = Option<([f64; 3], bool)>;
    let decided: Vec<(Stored, bool)> = (0..n_pt)
        .into_par_iter()
        .map(|p| {
            if cons.constraint[p] != PointConstraint::Free {
                return (None, false);
            }
            let row = points[p];
            if !row.iter().all(|v| v.is_finite()) {
                return (None, false);
            }
            let (mut dirs, mut cs, mut weights) = (Vec::new(), Vec::new(), Vec::new());
            for &k in &track[p] {
                let i = obs_img[k] as usize;
                let cam = &cams[image_camera[i] as usize];
                if let Some(r) = observed_ray(cam, &quats[i], uv[k], sigma_px) {
                    dirs.push(r.dir);
                    cs.push(centres[i]);
                    weights.push(r.weight);
                }
            }
            let Some(score) = bearing_score(&dirs, &cs, &weights) else {
                return (None, true);
            };
            let finite = is_finite(&score, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD);
            let stored = match (finite, is_dir[p]) {
                (true, true) => {
                    fit_point_and_bearing(&dirs, &cs, &weights, None, None, &fit_options)
                        .and_then(|fit| fit.point.filter(|_| fit.in_front_of_all_cameras))
                        .map(|x| ([x.x, x.y, x.z], false))
                }
                (false, _) if score.bearing_in_front_of_all_cameras => {
                    let b = score.bearing;
                    Some(([b.x, b.y, b.z], true))
                }
                _ => None,
            };
            (stored, false)
        })
        .collect();
    let mut unscored = vec![false; n_pt];
    for (p, (d, not_scored)) in decided.into_iter().enumerate() {
        if let Some((row, dir)) = d {
            points[p] = row;
            is_dir[p] = dir;
        }
        let handed_in = input_points[p];
        if not_scored && is_dir[p] != input_is_dir[p] && handed_in.iter().all(|v| v.is_finite()) {
            points[p] = handed_in;
            is_dir[p] = input_is_dir[p];
        }
        unscored[p] = not_scored;
    }
    unscored
}

/// The staged loop: direction-aware residuals, trims, re-estimation, and the
/// `min_obs` floor over every trim survivor. With an all-`false` `is_dir`
/// every direction branch is skipped and this is the finite-only solve.
#[allow(clippy::too_many_arguments)]
fn bundle_adjust_staged(
    cameras: &[CameraIntrinsics],
    image_camera: &[u32],
    adjustments: Option<&[IntrinsicAdjustment]>,
    releases: Option<&[CameraRelease]>,
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
    opt_pp: bool,
    opt_image_f: bool,
    opt_image_pp: bool,
    image_prior: &ImagePriorOptions,
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

    // Each camera's releases: the flags, narrowed by the camera's own entry
    // where the caller gave one, then decided on its own model.
    let mut lenses: Vec<Lens<'_>> = cameras
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let r = releases.map_or(CameraRelease::ALL, |r| r[j]);
            Lens::new(
                c,
                opt_f && r.focal,
                opt_k1 && r.distortion,
                opt_bspline && r.distortion,
                opt_pp && r.principal_point,
                opt_image_f && r.image_focal,
                opt_image_pp && r.image_principal_point,
            )
        })
        .collect();
    let spline_cols = lenses.iter().any(|l| l.opt_bspline);
    let pp_cols = lenses.iter().any(|l| l.opt_pp);
    let image_cols = lenses.iter().any(Lens::releases_image);
    let n_img = quats.len();
    // The adjustments the solve reads, and under a per-image release moves:
    // every image's, zero where the caller gave none. Without a per-image
    // release, all-zero adjustments are none at all, so the solve reads every
    // camera as it is.
    let mut adj_state: Vec<IntrinsicAdjustment> = match adjustments {
        _ if image_cols => adjustments.map_or_else(
            || vec![IntrinsicAdjustment::default(); n_img],
            <[_]>::to_vec,
        ),
        Some(a) if a.iter().any(|a| !a.is_zero()) => a.to_vec(),
        _ => Vec::new(),
    };
    // The per-image prior's widths, per camera: the loose ones for the first
    // round and derived before each later one, or the fixed ones throughout.
    let derive = image_cols && image_prior.fixed.is_none();
    let mut widths: Vec<ImagePriorWidths> =
        vec![image_prior.fixed.unwrap_or(image_prior.loose); cameras.len()];
    let no_widths = ImagePriorWidths {
        focal: f64::NAN,
        principal_point: f64::NAN,
    };
    let mut spreads: Vec<ImagePriorWidths> = vec![no_widths; cameras.len()];
    let mut round_widths: Vec<ImagePriorWidths> = widths.clone();
    // A one-round schedule runs its round a second time under widths derived
    // from the first, so that every derived prior is derived at least once.
    let repeated: [BaSchedule; 2];
    let schedule: &[BaSchedule] = if derive && schedule.len() == 1 {
        repeated = [schedule[0], schedule[0]];
        &repeated
    } else {
        schedule
    };

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
    // The caller's representation, which the storage decision reports its
    // changes against and restores to a point it cannot score.
    let (input_points, input_is_dir): (Vec<[f64; 3]>, Vec<bool>) = if free_points.cross {
        (points.to_vec(), is_dir.to_vec())
    } else {
        (Vec::new(), Vec::new())
    };
    // The solve set of the last round, whose residuals the storage decision
    // measures, and how that round ended.
    let mut prev_kept: Vec<usize> = Vec::new();
    let mut last_end = RoundEnd::Stopped;
    let rounds = progress.split_evenly(schedule.len());
    for ((rnd, stage), p_round) in schedule.iter().enumerate().zip(rounds) {
        // Between rounds the poses and the points hold what the last round
        // settled on, which is a whole answer to hand back.
        if progress.is_cancelled() {
            break;
        }
        let round = p_round.phase("round");
        // The round's trim, never above the invalid residual, so an
        // observation whose projection fails is trimmed even under an
        // unbounded trim.
        let trim_px = stage.trim_px.min(INVALID_RESIDUAL);
        let cams_now: Vec<CameraIntrinsics> = lenses.iter().map(Lens::camera).collect();
        // Every image's intrinsics at the round's cameras: what the
        // re-estimation casts rays through and the trim measures with.
        let intrinsics_now = ImageIntrinsicsList::new(&cams_now, image_camera, &adj_state);
        if rnd > 0 {
            retriangulate_round(
                &intrinsics_now.intrinsics,
                &intrinsics_now.image_entry,
                quats,
                trans,
                points,
                is_dir,
                uv,
                obs_img,
                obs_pt,
                cons,
                free_points.cross.then_some(trim_px),
                min_track,
            );
        }
        // Under the crossing every free point is solved in inverse depth about
        // the centroid of its observing cameras at the poses the round starts
        // from, fixed for the round.
        let anchors: Option<Vec<Vector3<f64>>> = free_points
            .cross
            .then(|| free_point_anchors(quats, trans, obs_img, obs_pt, cons, points.len()));
        let (norms, depths) = residual_norms_depths(
            &intrinsics_now.intrinsics,
            &intrinsics_now.image_entry,
            quats,
            trans,
            points,
            is_dir,
            uv,
            obs_img,
            obs_pt,
        );
        // In-front: the model-aware measure from `residual_norms_depths`
        // (canonical depth for the perspective family, range for a ray-path
        // model) over the scene-scale floor for finite observations, and over
        // zero for directions, the floor's limit at zero inverse depth.
        // Protected observations bypass the trim gates entirely.
        let finite_floor = in_front_floor(&depths, is_dir, obs_pt);
        let mut keep: Vec<bool> = (0..n_obs)
            .map(|k| {
                let floor = if is_dir[obs_pt[k] as usize] {
                    0.0
                } else {
                    finite_floor
                };
                is_prot(k) || (norms[k] < trim_px && depths[k] > floor)
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
                cameras: lenses.iter().map(Lens::camera).collect(),
                residual_norms: vec![f64::INFINITY; n_obs],
                point_at_infinity: is_dir.to_vec(),
                free_point_decision: None,
                image_adjustments: returned_adjustments(&adj_state, adjustments, n_img),
                image_priors: image_priors_of(&lenses, &round_widths, derive, &spreads),
            };
        }
        let anchors_now = anchors.as_deref();
        // The per-image prior's noise level: the round's reprojection noise at
        // the state it starts from, over its kept observations, or the round's
        // loss scale where that measures nothing.
        let sigma_n = if image_cols {
            round_noise(
                &cams_now,
                image_camera,
                &intrinsics_now,
                quats,
                trans,
                points,
                is_dir,
                uv,
                obs_img,
                obs_pt,
                &kept,
            )
            .sigma_px
            .unwrap_or(stage.loss_scale)
        } else {
            0.0
        };
        round_widths.clone_from(&widths);
        let image_round = ImageRound {
            widths: &round_widths,
            sigma_n,
        };
        // One round at the observation-block width the releases need.
        macro_rules! round_at {
            ($cols:expr) => {
                solve_lm::<{ $cols }>(
                    &mut lenses,
                    image_camera,
                    &mut adj_state,
                    image_cols.then_some(&image_round),
                    quats,
                    trans,
                    points,
                    is_dir,
                    uv,
                    obs_img,
                    obs_pt,
                    &kept,
                    cons,
                    anchors_now,
                    stage.loss_scale,
                    max_iters,
                    protected,
                    protected_loss_scale,
                    &round,
                )
            };
        }
        let (end, image_result) = match (spline_cols, pp_cols, image_cols) {
            (false, false, false) => round_at!(BASE_CAM_COLS),
            (true, false, false) => round_at!(BSPLINE_CAM_COLS),
            (false, true, false) => round_at!(BASE_PP_CAM_COLS),
            (true, true, false) => round_at!(BSPLINE_PP_CAM_COLS),
            (false, _, true) => round_at!(BASE_IMG_CAM_COLS),
            (true, _, true) => round_at!(BSPLINE_IMG_CAM_COLS),
        };
        last_end = end;
        if image_cols {
            // Before every round after the first, each camera's widths are
            // derived from the round just finished; then the camera takes its
            // images' mean adjustment, and each image keeps its difference.
            if derive && rnd + 1 < schedule.len() {
                // The noise the solved adjustments carry: the level of the
                // round's kept residuals at the state it settled on, rather
                // than the level it started from, which the prior's weight
                // was stated at and which still holds the start's error.
                let cams_end: Vec<CameraIntrinsics> = lenses.iter().map(Lens::camera).collect();
                let intrinsics_end = ImageIntrinsicsList::new(&cams_end, image_camera, &adj_state);
                let sigma_end = round_noise(
                    &cams_end,
                    image_camera,
                    &intrinsics_end,
                    quats,
                    trans,
                    points,
                    is_dir,
                    uv,
                    obs_img,
                    obs_pt,
                    &kept,
                )
                .sigma_px
                .unwrap_or(sigma_n);
                for (j, lens) in lenses.iter().enumerate() {
                    if !lens.releases_image() {
                        continue;
                    }
                    let samples: Vec<(IntrinsicAdjustment, Option<Matrix3<f64>>)> = image_result
                        .images
                        .iter()
                        .zip(&image_result.inv_info)
                        .filter(|(&i, _)| image_camera[i] as usize == j)
                        .map(|(&i, &inv)| (adj_state[i], inv))
                        .collect();
                    let (next, spread) = derive_image_prior_widths(
                        &samples,
                        lens.opt_image_f,
                        lens.opt_image_pp,
                        round_widths[j],
                        sigma_n,
                        sigma_end,
                        image_prior,
                    );
                    widths[j] = next;
                    spreads[j] = spread;
                }
            }
            recentre_adjustments(
                &mut lenses,
                image_camera,
                &mut adj_state,
                &image_result.images,
            );
        }
        prev_kept = kept;
        drop(round);
        progress.count(rnd as u64 + 1, Some(schedule.len() as u64), "round");
    }

    let cams_final: Vec<CameraIntrinsics> = lenses.iter().map(Lens::camera).collect();
    let intrinsics_final = ImageIntrinsicsList::new(&cams_final, image_camera, &adj_state);
    // The storage decision: every free point is scored at the state the solve
    // ended at and the noise level its final round measures, over the
    // observations that round kept, and stored as the test says. A final round
    // that stopped on its iteration budget or settled with a distortion held
    // at its step guard is decided too, at the level it measures, and the
    // decision says it did not converge.
    let free_point_decision = free_points.cross.then(|| {
        let noise = round_noise(
            &cams_final,
            image_camera,
            &intrinsics_final,
            quats,
            trans,
            points,
            is_dir,
            uv,
            obs_img,
            obs_pt,
            &prev_kept,
        );
        // A cancelled solve hands back the state it reached, and is not
        // decided on.
        let sigma = noise.sigma_px.filter(|_| !progress.is_cancelled());
        let unscored_points = sigma.map_or_else(Vec::new, |s| {
            decide_free_points(
                &intrinsics_final.intrinsics,
                &intrinsics_final.image_entry,
                quats,
                trans,
                points,
                is_dir,
                uv,
                obs_img,
                obs_pt,
                &prev_kept,
                cons,
                s,
                &input_points,
                &input_is_dir,
            )
        });
        let unscored = unscored_points.iter().filter(|&&u| u).count();
        // A point not scored is not counted as changed, whatever row it keeps.
        let (mut to_finite, mut to_direction) = (0, 0);
        for p in 0..points.len() {
            if cons.constraint[p] != PointConstraint::Free
                || unscored_points.get(p).copied().unwrap_or(false)
            {
                continue;
            }
            match (input_is_dir[p], is_dir[p]) {
                (true, false) => to_finite += 1,
                (false, true) => to_direction += 1,
                _ => {}
            }
        }
        // The decision in one line, said whenever there was a round to decide
        // after: an empty schedule only measures residuals.
        if !schedule.is_empty() {
            let unconverged = match last_end {
                RoundEnd::Converged => "",
                RoundEnd::SettledAtGuard => {
                    "; the final round settled with a distortion held at its step guard"
                }
                RoundEnd::Stopped => "; the final round stopped on its iteration budget",
            };
            let unscored_clause = if unscored > 0 {
                format!(", {unscored} not scored: too few kept observations")
            } else {
                String::new()
            };
            match sigma {
                Some(s) => progress_info!(
                    progress,
                    "free points decided at noise {s:.3} px: {to_finite} to finite, \
                     {to_direction} to directions{unscored_clause}{unconverged}"
                ),
                None if !progress.is_cancelled() => progress_info!(
                    progress,
                    "free points not decided: the final round kept no observation of a \
                     finite point"
                ),
                None => {}
            }
        }
        FreePointDecision {
            sigma_px: noise.sigma_px,
            decided: sigma.is_some(),
            observation_count: noise.observation_count,
            outlier_count: noise.outlier_count,
            converged: last_end == RoundEnd::Converged,
            to_finite,
            to_direction,
            unscored,
        }
    });
    let (norms, _depths) = residual_norms_depths(
        &intrinsics_final.intrinsics,
        &intrinsics_final.image_entry,
        quats,
        trans,
        points,
        is_dir,
        uv,
        obs_img,
        obs_pt,
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
        cameras: cams_final,
        residual_norms,
        point_at_infinity: is_dir.to_vec(),
        free_point_decision,
        image_adjustments: returned_adjustments(&adj_state, adjustments, n_img),
        image_priors: image_priors_of(&lenses, &round_widths, derive, &spreads),
    }
}

/// The adjustments a solve returns: the state it moved under a per-image
/// release, and the caller's otherwise, zero where the caller gave none.
fn returned_adjustments(
    state: &[IntrinsicAdjustment],
    input: Option<&[IntrinsicAdjustment]>,
    n_img: usize,
) -> Vec<IntrinsicAdjustment> {
    if state.len() == n_img {
        return state.to_vec();
    }
    input.map_or_else(
        || vec![IntrinsicAdjustment::default(); n_img],
        <[_]>::to_vec,
    )
}

/// Each camera's [`ImagePrior`] for the result: the widths its last round ran
/// under and the last derivation's spread, `NaN` for a parameter it does not
/// release per image, and `None` for a camera with no per-image release.
fn image_priors_of(
    lenses: &[Lens<'_>],
    widths: &[ImagePriorWidths],
    derived: bool,
    spreads: &[ImagePriorWidths],
) -> Vec<Option<ImagePrior>> {
    lenses
        .iter()
        .enumerate()
        .map(|(j, lens)| {
            lens.releases_image().then(|| {
                let keep = |w: ImagePriorWidths| ImagePriorWidths {
                    focal: if lens.opt_image_f { w.focal } else { f64::NAN },
                    principal_point: if lens.opt_image_pp {
                        w.principal_point
                    } else {
                        f64::NAN
                    },
                };
                ImagePrior {
                    widths: keep(widths[j]),
                    derived,
                    spread: keep(spreads[j]),
                }
            })
        })
        .collect()
}

/// After a round: each camera with a per-image release takes the mean of its
/// released adjustments over its images the round solved, `solved`, and every
/// image of the camera keeps its difference from that mean, so that every
/// image's intrinsics are unchanged and the mean adjustment over `solved` is
/// zero. The principal point moves by the mean offset, `cx += mean(a_x)·m`;
/// the focal takes the mean `t̄` of the log focal scales `t = ln(1 + s)`,
/// `f ← f·exp(t̄)` and `t −= t̄`, which multiplies the focal by the geometric
/// mean of `1 + s` and divides each `1 + s` by it. Applied to every image of
/// the camera, the ones the round did not solve included.
fn recentre_adjustments(
    lenses: &mut [Lens<'_>],
    image_camera: &[u32],
    adj: &mut [IntrinsicAdjustment],
    solved: &[usize],
) {
    for (j, lens) in lenses.iter_mut().enumerate() {
        if !lens.releases_image() {
            continue;
        }
        let of_camera: Vec<usize> = solved
            .iter()
            .copied()
            .filter(|&i| image_camera[i] as usize == j)
            .collect();
        if of_camera.is_empty() {
            continue;
        }
        let n = of_camera.len() as f64;
        let images: Vec<usize> = (0..adj.len())
            .filter(|&i| image_camera[i] as usize == j)
            .collect();
        if lens.opt_image_pp {
            let m = f64::from(lens.base.width.min(lens.base.height));
            let mut mean = [0.0; 2];
            for &i in &of_camera {
                mean[0] += adj[i].principal_point[0];
                mean[1] += adj[i].principal_point[1];
            }
            mean = [mean[0] / n, mean[1] / n];
            lens.pp = [lens.pp[0] + mean[0] * m, lens.pp[1] + mean[1] * m];
            for &i in &images {
                adj[i].principal_point[0] -= mean[0];
                adj[i].principal_point[1] -= mean[1];
            }
        }
        if lens.opt_image_f {
            let mean = of_camera
                .iter()
                .map(|&i| log_focal_scale(adj[i].focal))
                .sum::<f64>()
                / n;
            lens.f *= mean.exp();
            for &i in &images {
                adj[i].focal = (log_focal_scale(adj[i].focal) - mean).exp_m1();
            }
        }
    }
}

/// The weighted median of `(value, weight)` samples: the smallest value at
/// which the cumulative weight, in increasing order of value, reaches half the
/// total. `NaN` for no sample of positive weight.
fn weighted_median(samples: &mut [(f64, f64)]) -> f64 {
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = samples.iter().map(|s| s.1).sum();
    if total.is_nan() || total <= 0.0 {
        return f64::NAN;
    }
    let mut cumulative = 0.0;
    for &(value, weight) in samples.iter() {
        cumulative += weight;
        if cumulative >= 0.5 * total {
            return value;
        }
    }
    samples.last().map_or(f64::NAN, |s| s.0)
}

/// The spread `s` of de-shrunk deviations `(d̂, v)` over `slots` of `dev`,
/// each sample of variance `s² + v`: `s² = max(0, m² − median_w v)` with
/// `m = 1.4826·median_w |d̂ − c|`, `c` each slot's weighted median of `d̂`,
/// and every median weighted by `1/(s² + v)`.
///
/// The weight is each sample's precision as a sample of the spread, so an
/// image whose own noise swamps the spread counts little and images whose
/// noise is well under it count alike. Since it reads `s`, it is iterated from
/// the round's width, the current estimate, with `s` never under `floor` in
/// the weight. The centre is each slot's weighted median rather than the mean
/// the deviations are taken from, so an image that moves the mean by its share,
/// which moves every deviation alike, does not widen the spread.
fn robust_spread(dev: &[Vec<(f64, f64)>; 3], slots: &[usize], start: f64, floor: f64) -> f64 {
    let mut s = start;
    for _ in 0..8 {
        let weight_of = |v: f64| {
            let w = s.max(floor);
            1.0 / (w * w + v)
        };
        let mut distances: Vec<(f64, f64)> = Vec::new();
        let mut variances: Vec<(f64, f64)> = Vec::new();
        for &k in slots {
            let mut signed: Vec<(f64, f64)> =
                dev[k].iter().map(|&(d, v)| (d, weight_of(v))).collect();
            let centre = weighted_median(&mut signed);
            for &(d, v) in &dev[k] {
                distances.push(((d - centre).abs(), weight_of(v)));
                variances.push((v, weight_of(v)));
            }
        }
        let m = 1.4826 * weighted_median(&mut distances);
        let next = (m * m - weighted_median(&mut variances)).max(0.0).sqrt();
        let settled = (next - s).abs() <= 1e-6 * s.max(floor);
        s = next;
        if settled {
            break;
        }
    }
    s
}

/// Derive one camera's prior widths for the next round from the round just
/// finished: per released parameter, the robust spread of the images'
/// deviations from their mean, with the round's prior pull undone and the
/// measurement noise subtracted, clamped to `[floor, loose]`. The focal's
/// parameter is the log focal scale `ln(1 + s)` ([`adj_component`]), the one
/// the round solved and its prior held.
///
/// `samples` holds each image of the camera the round solved: its adjustment,
/// and the inverse `H_i⁻¹` of its data-only block over the adjustment slots
/// ([`ImageRoundResult::inv_info`]). The deviation `d_i` from the mean is
/// de-shrunk as `d̂_i = (I + H_i⁻¹·W)·d_i`, with `W = σ_prior²·P⁻¹` the
/// round's prior curvature (`P` its prior variances, `σ_prior` the level its
/// weight was stated at), every released parameter together: exactly the
/// pull the round's prior had on the image's own block. The adjustment's
/// variance, its covariance `V_i = σ_noise²·H_i⁻¹` with every other image held,
/// is read at `σ_noise`, the level of the round's solved residuals. The spread is
/// [`robust_spread`]'s; the principal point pools its two axes as two samples
/// per image. A parameter with fewer than two images of usable variance keeps
/// the loose width and reports a `NaN` spread. Returns the next widths and the
/// spreads.
fn derive_image_prior_widths(
    samples: &[(IntrinsicAdjustment, Option<Matrix3<f64>>)],
    focal: bool,
    principal_point: bool,
    round: ImagePriorWidths,
    sigma_prior: f64,
    sigma_noise: f64,
    options: &ImagePriorOptions,
) -> (ImagePriorWidths, ImagePriorWidths) {
    let mut next = options.loose;
    let mut spread = ImagePriorWidths {
        focal: f64::NAN,
        principal_point: f64::NAN,
    };
    if samples.is_empty() {
        return (next, spread);
    }
    let n = samples.len() as f64;
    let released = [focal, principal_point, principal_point];
    let mut mean = [0.0; 3];
    for (a, _) in samples {
        for (k, m) in mean.iter_mut().enumerate() {
            *m += adj_component(a, k);
        }
    }
    for m in mean.iter_mut() {
        *m /= n;
    }
    let inv_var = |w: f64, on: bool| {
        if on {
            sigma_prior * sigma_prior / (w * w)
        } else {
            0.0
        }
    };
    let prior_curvature = Vector3::new(
        inv_var(round.focal, focal),
        inv_var(round.principal_point, principal_point),
        inv_var(round.principal_point, principal_point),
    );
    // Per slot (log focal scale, `a_x`, `a_y`): the de-shrunk deviation `d̂` of
    // each image with a usable variance, and that variance `v`; and per
    // parameter kind how many images have one.
    let mut dev: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut weighted_images = [0usize; 2];
    for (a, inv) in samples {
        let Some(inv) = inv else { continue };
        let v_mat = inv * (sigma_noise * sigma_noise);
        let d = Vector3::from_fn(|k, _| {
            if released[k] {
                adj_component(a, k) - mean[k]
            } else {
                0.0
            }
        });
        let d_hat = d + inv * prior_curvature.component_mul(&d);
        let mut counted = [false; 2];
        for k in 0..3 {
            if !released[k] {
                continue;
            }
            let v = v_mat[(k, k)];
            if !(v.is_finite() && v > 0.0) {
                continue;
            }
            dev[k].push((d_hat[k], v));
            counted[usize::from(k > 0)] = true;
        }
        for kind in 0..2 {
            weighted_images[kind] += usize::from(counted[kind]);
        }
    }
    for (kind, on) in [focal, principal_point].into_iter().enumerate() {
        if !on || weighted_images[kind] < 2 {
            continue;
        }
        let slots: &[usize] = if kind == 0 { &[0] } else { &[1, 2] };
        let (round_width, floor) = if kind == 0 {
            (round.focal, options.floor.focal)
        } else {
            (round.principal_point, options.floor.principal_point)
        };
        let s = robust_spread(&dev, slots, round_width, floor);
        if kind == 0 {
            spread.focal = s;
            next.focal = s.max(options.floor.focal).min(options.loose.focal);
        } else {
            spread.principal_point = s;
            next.principal_point = s
                .max(options.floor.principal_point)
                .min(options.loose.principal_point);
        }
    }
    (next, spread)
}

#[cfg(test)]
mod tests;
