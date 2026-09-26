// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bundle-adjust a whole reconstruction.
//!
//! `specs/core/reconstruction/bundle-adjust.md` is the design. The function
//! here is pure: an [`SfmrReconstruction`] goes in, a new one and a report come
//! out, and the input is left exactly as it was. It is the reconstruction-level
//! caller of the array kernel in
//! [`crate::geometry::bundle_adjust()`]: it gathers the poses, the points and the
//! observation pixels the kernel takes, runs it once, and writes the answer back
//! into a value of the same shape.

use nalgebra::{Point3, UnitQuaternion, Vector3};

use super::data::SfmrReconstruction;
use std::ops::RangeInclusive;

use super::outermost_keypoint::{outermost_keypoints, KeypointReach};
use crate::camera::distortion::bspline::MIN_BSPLINE_COEFFS;
use crate::camera::intrinsics::SplineRadial;
use crate::camera::refit_intrinsics::{
    refit_spline, MonotoneConstraint, RefitError, MAX_COEFF_COUNT,
};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::bundle_adjust::{
    BaCameras, BaSchedule, DistanceReference, FreePointPolicy, PointConstraints,
    PointConstraintsError, DEFAULT_PROTECTED_LOSS_SCALE, DEFAULT_SCHEDULE,
};
use crate::numeric::median_in_place;
use crate::progress::Progress;
use crate::progress_info;
use sfmtool_sfmr_format::{NO_REFERENCE_IMAGE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED};

/// LM iteration budget per round, the kernel's own default.
const DEFAULT_MAX_ITERS: usize = 60;
/// Trim survivors a point needs to stay in the solve, the kernel's own default.
const DEFAULT_MIN_TRACK: usize = 2;
/// Trim survivors below which a round exits degenerate, the kernel's own
/// default.
const DEFAULT_MIN_OBS: usize = 12;

/// The spline coefficient counts [`BundleAdjustOptions::spline_coeff_count`]
/// accepts. The floor is the fewest coefficients a spline is defined with
/// (fewer evaluate as the identity, which has nothing to release); the ceiling
/// is the refit's own, [`MAX_COEFF_COUNT`], past which the knot spans are
/// narrower than a lens calibration can support.
pub const SPLINE_COEFF_COUNT_RANGE: RangeInclusive<usize> = MIN_BSPLINE_COEFFS..=MAX_COEFF_COUNT;

/// What the adjustment is allowed to move, and how hard it tries.
///
/// The defaults are the kernel's own, so an adjustment asked for with nothing
/// stated is the one every other caller in this crate runs.
#[derive(Debug, Clone)]
pub struct BundleAdjustOptions {
    /// Release the focal length of every camera the posed images use, each
    /// camera its own. Off by default, because a focal that moves is a
    /// different claim about the capture than a pose that does, and the caller
    /// should be the one making it.
    ///
    /// Only the models the kernel's analytic focal column is exact for accept
    /// it -- `SIMPLE_PINHOLE`, `EQUIDISTANT_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`,
    /// `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`. When any camera in the solve has
    /// another model this is refused rather than silently ignored: a caller that
    /// asked for a focal solve and got a fixed-focal one back would have no way
    /// to tell.
    pub opt_f: bool,
    /// Release each camera's lens distortion where its model admits one: `k1`
    /// on `SIMPLE_RADIAL_FISHEYE`, the radial spline on `SFMTOOL_FISHEYE` and
    /// `SFMTOOL_PINHOLE`, each camera its own. Off by default.
    ///
    /// The distortion is released only together with the focal: neither `k1`
    /// nor the spline can change the scale at the centre of the image (both
    /// leave the slope of the radial map on the axis at one), and a distortion
    /// released against a held focal could only bend the periphery around a
    /// scale it cannot fix. So this is refused without [`Self::opt_f`], and
    /// refused when no camera in the solve has distortion to release. A camera
    /// of any other model keeps its distortion where it is, and the report says
    /// which cameras released theirs.
    pub opt_distortion: bool,
    /// Change the coefficient count of every spline camera in the solve to
    /// this, before the solve starts. `None`, the default, keeps each count.
    ///
    /// Each spline camera whose count differs is refitted by
    /// [`refit_spline`] as the same spline model with this many coefficients,
    /// its domain end held, fitted over the whole domain: the best
    /// least-squares description of the old spline's curve on the new
    /// coefficient scheme. The solve then starts from the refitted cameras.
    /// Refused unless [`Self::opt_distortion`] is on, because a new
    /// coefficient scheme can only approximate the old curve and it is the
    /// solve that brings it back to the observations; refused when no camera
    /// in the solve is a spline model; refused outside
    /// [`SPLINE_COEFF_COUNT_RANGE`]; and refused, naming the camera, when a
    /// refit is.
    pub spline_coeff_count: Option<usize>,
    /// Move the domain end of every spline camera in the solve to this
    /// incidence angle, in degrees, before the solve starts. `None`, the
    /// default, keeps each domain.
    ///
    /// Each spline camera whose domain end differs is refitted by
    /// [`refit_spline`] on the new domain, over the whole of it, together with
    /// any new [`Self::spline_coeff_count`] in the same refit. The old model is
    /// defined everywhere, on its domain and along its linear tail past it, so
    /// the new domain may be shorter or longer than the old one. The refusals
    /// are the coefficient count's: it needs [`Self::opt_distortion`] and a
    /// spline camera in the solve, and a refit refused, here for a domain end
    /// the model cannot have, refuses the adjustment naming the camera.
    pub spline_domain_deg: Option<f64>,
    /// The staged trim schedule, `(trim_px, loss_scale)` per round.
    pub schedule: Vec<BaSchedule>,
    /// LM iteration budget per round.
    pub max_iters: usize,
    /// Trim survivors a point needs to stay in a round's solve. A point that
    /// falls below it is dropped from the result.
    pub min_track: usize,
    /// Trim survivors below which the round exits degenerate, which this call
    /// refuses.
    pub min_obs: usize,
}

impl Default for BundleAdjustOptions {
    fn default() -> Self {
        Self {
            opt_f: false,
            opt_distortion: false,
            spline_coeff_count: None,
            spline_domain_deg: None,
            schedule: DEFAULT_SCHEDULE.to_vec(),
            max_iters: DEFAULT_MAX_ITERS,
            min_track: DEFAULT_MIN_TRACK,
            min_obs: DEFAULT_MIN_OBS,
        }
    }
}

/// Why an adjustment produced nothing. Every variant names what did not hold,
/// because the caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum BundleAdjustError {
    /// The observations carry no pixel: a `sift_files` value without the
    /// format's optional inline keypoint column.
    NoKeypoints,
    /// The focal release was asked for, and a camera in the solve has a model
    /// the kernel's focal column is not exact for.
    FocalNotReleasable {
        /// The camera's index in the reconstruction's camera table.
        camera: usize,
        /// Its model's name.
        model: &'static str,
    },
    /// The distortion release was asked for without the focal release.
    DistortionWithoutFocal,
    /// The distortion release was asked for, and no camera in the solve has a
    /// model whose distortion the adjustment can release.
    DistortionNotReleasable,
    /// A spline coefficient count or domain was asked for without the
    /// distortion release.
    SplineRefitWithoutDistortion,
    /// A spline coefficient count or domain was asked for, and no camera in
    /// the solve is a spline model.
    SplineRefitWithoutSpline,
    /// A spline coefficient count outside [`SPLINE_COEFF_COUNT_RANGE`].
    SplineCoeffCount {
        /// The count asked for.
        count: usize,
    },
    /// The refit of one spline camera to the new count or domain was
    /// refused.
    SplineRefit {
        /// The camera's index in the reconstruction's camera table.
        camera: usize,
        /// The refit's refusal.
        error: RefitError,
    },
    /// No image of the reconstruction carries a usable pose.
    NoPosedImages,
    /// No observation of a live point falls in a posed image.
    NoObservations,
    /// The schedule has no rounds, so there is nothing to run.
    EmptySchedule,
    /// The value's constraint columns state something the adjustment cannot
    /// honour.
    Constraints(PointConstraintsError),
    /// The caller asked the adjustment to stop, and it did, so there is no
    /// solved state to write back.
    Cancelled,
    /// A round exited degenerate: fewer than `min_obs` observations survived a
    /// trim, and the state passed through untouched.
    Degenerate {
        /// How many observations went in.
        observations: usize,
        /// The floor they had to keep.
        min_obs: usize,
    },
}

impl std::fmt::Display for BundleAdjustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BundleAdjustError::NoKeypoints => write!(
                f,
                "the adjustment needs a pixel per observation, and this reconstruction's \
                 observations are .sift feature indexes with no inline keypoints"
            ),
            BundleAdjustError::FocalNotReleasable { camera, model } => write!(
                f,
                "the focal cannot be released on camera {camera}, a {model}; the \
                 adjustment's focal column is exact for SIMPLE_PINHOLE, \
                 EQUIDISTANT_FISHEYE, SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE and \
                 SFMTOOL_PINHOLE"
            ),
            BundleAdjustError::DistortionWithoutFocal => write!(
                f,
                "the lens distortion is released only together with the focal length, \
                 because neither k1 nor the spline can change the scale at the centre of \
                 the image"
            ),
            BundleAdjustError::DistortionNotReleasable => write!(
                f,
                "no camera of this reconstruction has lens distortion the adjustment can \
                 release; it releases k1 on SIMPLE_RADIAL_FISHEYE and the spline on \
                 SFMTOOL_FISHEYE and SFMTOOL_PINHOLE"
            ),
            BundleAdjustError::SplineRefitWithoutDistortion => write!(
                f,
                "a spline's coefficient count or domain is changed only while the lens \
                 distortion is released, because the refitted spline can only approximate \
                 the old curve until the solve fits it to the observations"
            ),
            BundleAdjustError::SplineRefitWithoutSpline => write!(
                f,
                "no camera of this reconstruction has a spline whose coefficient count or \
                 domain could change; only SFMTOOL_FISHEYE and SFMTOOL_PINHOLE carry one"
            ),
            BundleAdjustError::SplineCoeffCount { count } => write!(
                f,
                "a spline takes {} to {} coefficients, not {count}",
                SPLINE_COEFF_COUNT_RANGE.start(),
                SPLINE_COEFF_COUNT_RANGE.end()
            ),
            BundleAdjustError::SplineRefit { camera, error } => write!(
                f,
                "the spline of camera {camera} could not be refitted: {error}"
            ),
            BundleAdjustError::NoPosedImages => {
                write!(f, "no image of this reconstruction carries a pose")
            }
            BundleAdjustError::NoObservations => write!(
                f,
                "no observation of this reconstruction falls in a posed image"
            ),
            BundleAdjustError::EmptySchedule => {
                write!(f, "the schedule has no rounds to run")
            }
            BundleAdjustError::Cancelled => write!(
                f,
                "the adjustment was asked to stop before it had an answer, so nothing \
                 was written back"
            ),
            BundleAdjustError::Constraints(e) => write!(f, "{e}"),
            BundleAdjustError::Degenerate {
                observations,
                min_obs,
            } => write!(
                f,
                "the adjustment exited degenerate: of {observations} observations fewer \
                 than {min_obs} survived a trim, so nothing was solved"
            ),
        }
    }
}

impl std::error::Error for BundleAdjustError {}

impl From<PointConstraintsError> for BundleAdjustError {
    fn from(e: PointConstraintsError) -> Self {
        BundleAdjustError::Constraints(e)
    }
}

/// What one adjustment did.
#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustReport {
    /// Posed images the solve moved.
    pub images: usize,
    /// Points that went into the solve.
    pub points: usize,
    /// Observations that went into the solve.
    pub observations: usize,
    /// Points the solve left unsupported, which the result does not hold.
    pub points_deleted: usize,
    /// Median unweighted reprojection residual before the solve, in pixels,
    /// over the observations that went into it.
    pub median_residual_before: f64,
    /// The same median after it. Taken over the observations of the points that
    /// survived, which is the population the value that comes back describes.
    pub median_residual_after: f64,
    /// One entry per camera in the solve, in camera-table order.
    pub cameras: Vec<CameraAdjustment>,
}

/// What one adjustment did to one camera.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraAdjustment {
    /// The camera's index in the reconstruction's camera table.
    pub camera: usize,
    /// Posed images taken through it that were in the solve.
    pub images: usize,
    /// Its focal before the solve, the model's first where it carries two.
    pub focal_before: f64,
    /// Its focal after the solve, which is the one before it unless the focal
    /// was released.
    pub focal_after: f64,
    /// Whether its focal was released.
    pub focal_released: bool,
    /// Whether its lens distortion was released: `k1` on a
    /// `SIMPLE_RADIAL_FISHEYE`, the spline on a spline model.
    pub distortion_released: bool,
    /// The refit that gave its spline a new coefficient count or domain before
    /// the solve, or `None` where both were kept.
    pub spline_refit: Option<SplineRefit>,
    /// The outermost of the observations of its images, measured under the
    /// camera the solve returned. Only the observations: the adjustment reads
    /// nothing off disk, so the features detected in the images' `.sift` files
    /// are [`outermost_keypoints`]'s to report.
    pub outermost_observed: Option<KeypointReach>,
}

/// How one camera's spline was refitted to a new coefficient count or domain
/// before the solve.
#[derive(Debug, Clone, PartialEq)]
pub struct SplineRefit {
    /// The coefficient count it had.
    pub coeffs_before: usize,
    /// The coefficient count it was refitted to.
    pub coeffs_after: usize,
    /// Where its domain ended, as an incidence angle in degrees.
    pub domain_before_deg: f64,
    /// Where the refitted domain ends, in degrees.
    pub domain_after_deg: f64,
    /// RMS pixel distance between the old and the refitted camera over the
    /// refit's samples, which cover the whole spline domain.
    pub rms_px: f64,
    /// The largest such distance.
    pub max_px: f64,
    /// What the refit's monotonicity constraint did: where it held the new
    /// spline's slope at the floor, and so departed from the old curve.
    pub monotone_constraint: MonotoneConstraint,
}

/// Bundle-adjust `recon`, returning the adjusted value and a report.
///
/// Every posed image's pose, every live point's position and, under
/// [`BundleAdjustOptions::opt_f`], each camera's focal (and under
/// [`BundleAdjustOptions::opt_distortion`] each camera's `k1` or radial spline)
/// are refined together
/// against every observation that carries a pixel, by the staged robust solve in
/// [`crate::geometry::bundle_adjust()`]. The posed images may be taken through
/// any number of the table's cameras; each keeps its own lens in the solve, and a
/// camera no posed image uses is not in it and comes back as it went in. A point
/// at infinity goes in as the
/// direction it is and comes back as one: the caller's representation is honoured
/// for the whole solve, so no point crosses between a bearing and a position
/// here. A point's constraint -- free, ranged or held -- is the one its
/// [`PointConstraintColumns`](super::data::PointConstraintColumns) state, so a
/// held point comes back exactly as it went in.
///
/// A point the solve leaves unsupported -- its position non-finite, or every one
/// of its observations invalid at the final state -- is **deleted** from the
/// result, because a point no observation can still be reconciled with is a
/// claim about the scene with nothing behind it. That is also what becomes of a
/// track the trim wears below [`BundleAdjustOptions::min_track`].
///
/// The patch frame of a point that moved is **rescaled** by the ratio of its
/// placement distances, so the patch keeps the angular size it had; the bitmap,
/// the normal, the colour and the constraint are the point's own and are kept.
/// The stored per-point error becomes the RMS of that point's own residuals at
/// the state that came back, because the column would otherwise describe a
/// geometry the value no longer holds.
///
/// `progress` is where this call names its four stages (gathering the kernel's
/// arrays, the residuals it starts from, the solve, writing the answer back),
/// says how big the problem is, and passes on the rounds and iterations the
/// kernel reports underneath. It is also how the call is asked to stop: a
/// cancelled adjustment returns [`BundleAdjustError::Cancelled`] and writes
/// nothing, because a half-converged state is not an answer anybody asked for.
/// Pass `&Progress::none()` to report nothing and never stop, which is one
/// branch per report and no behaviour change at all.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::progress::Progress;
/// use sfmtool_core::reconstruction::bundle_adjust::{bundle_adjust, BundleAdjustOptions};
/// # fn run(recon: &sfmtool_core::SfmrReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let (next, report) =
///     bundle_adjust(recon, &BundleAdjustOptions::default(), &Progress::none())?;
/// println!(
///     "{} px -> {} px over {} observations",
///     report.median_residual_before, report.median_residual_after, report.observations
/// );
/// # Ok(())
/// # }
/// ```
pub fn bundle_adjust(
    recon: &SfmrReconstruction,
    options: &BundleAdjustOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, BundleAdjustReport), BundleAdjustError> {
    if options.schedule.is_empty() {
        return Err(BundleAdjustError::EmptySchedule);
    }
    // Every observation states a pixel, or there is nothing to reproject
    // against. The optional inline column answers for `sift_files` too.
    let keypoints = recon
        .point_set
        .keypoints_xy()
        .ok_or(BundleAdjustError::NoKeypoints)?;

    let table = &recon.image_table;
    // The solve's image list, and the map from the table's indexes onto it. An
    // unposed image is not a camera the residuals can be written against, so it
    // sits the solve out and its observations do not enter.
    let mut old_to_new: Vec<Option<u32>> = vec![None; table.images.len()];
    let mut posed: Vec<usize> = Vec::with_capacity(table.images.len());
    for (i, image) in table.images.iter().enumerate() {
        if is_posed(&image.quaternion_wxyz, &image.translation_xyz) {
            old_to_new[i] = Some(posed.len() as u32);
            posed.push(i);
        }
    }
    if posed.is_empty() {
        return Err(BundleAdjustError::NoPosedImages);
    }

    // The cameras of the solve: the table's entries the posed images use, in
    // table order, and each posed image's index into that list. A camera only
    // unposed images use sits the solve out with them.
    let mut used: Vec<u32> = posed
        .iter()
        .map(|&i| table.images[i].camera_index)
        .collect();
    used.sort_unstable();
    used.dedup();
    let source_cameras: Vec<CameraIntrinsics> = used
        .iter()
        .map(|&c| table.cameras[c as usize].clone())
        .collect();
    let image_camera: Vec<u32> = posed
        .iter()
        .map(|&i| {
            used.binary_search(&table.images[i].camera_index)
                .expect("every posed image's camera is in the list") as u32
        })
        .collect();
    // The release refuses rather than degrades, over every camera in the
    // solve: the kernel would hold a camera it cannot release and move the rest,
    // and a report saying the focal was released would then be false of it.
    if options.opt_f {
        if let Some((&c, camera)) = used
            .iter()
            .zip(&source_cameras)
            .find(|(_, camera)| !focal_is_releasable(camera))
        {
            return Err(BundleAdjustError::FocalNotReleasable {
                camera: c as usize,
                model: camera.model_name(),
            });
        }
    }
    if options.opt_distortion && !options.opt_f {
        return Err(BundleAdjustError::DistortionWithoutFocal);
    }
    // A new coefficient count or domain is a refit of each spline camera whose
    // count or domain differs, before the solve, and the solve starts from the
    // refitted cameras. The "before" residuals are still measured through the cameras
    // the value holds, so the report's two medians describe the input and the
    // output.
    let mut spline_refits: Vec<Option<SplineRefit>> = vec![None; used.len()];
    let mut cameras = source_cameras.clone();
    if options.spline_coeff_count.is_some() || options.spline_domain_deg.is_some() {
        if !options.opt_distortion {
            return Err(BundleAdjustError::SplineRefitWithoutDistortion);
        }
        if let Some(count) = options.spline_coeff_count {
            if !SPLINE_COEFF_COUNT_RANGE.contains(&count) {
                return Err(BundleAdjustError::SplineCoeffCount { count });
            }
        }
        if !cameras.iter().any(|c| c.model.radial_spline().is_some()) {
            return Err(BundleAdjustError::SplineRefitWithoutSpline);
        }
        for (j, &c) in used.iter().enumerate() {
            let Some((bspline, _, _)) = source_cameras[j].model.radial_spline() else {
                continue;
            };
            let coeffs_before = bspline.len();
            let domain_before_deg = spline_domain_deg(&source_cameras[j]).unwrap_or(f64::NAN);
            let count = options.spline_coeff_count.unwrap_or(coeffs_before);
            // A domain within a nanodegree of the one the camera has is the
            // same domain, which is then copied exactly rather than taken
            // through degrees and back. A value that is not a number is passed
            // on, for the refit to refuse.
            let same = |deg: f64| (deg - domain_before_deg).abs() <= 1e-9;
            let domain = options.spline_domain_deg.filter(|&deg| !same(deg));
            if count == coeffs_before && domain.is_none() {
                continue;
            }
            let refit = refit_spline(&source_cameras[j], count, domain).map_err(|error| {
                BundleAdjustError::SplineRefit {
                    camera: c as usize,
                    error,
                }
            })?;
            spline_refits[j] = Some(SplineRefit {
                coeffs_before,
                coeffs_after: count,
                domain_before_deg,
                domain_after_deg: refit.spline_domain_deg.unwrap_or(f64::NAN),
                rms_px: refit.rms_px,
                max_px: refit.max_px,
                monotone_constraint: refit.monotone_constraint,
            });
            cameras[j] = refit.camera;
        }
    }
    if options.opt_distortion && !cameras.iter().any(distortion_is_releasable) {
        return Err(BundleAdjustError::DistortionNotReleasable);
    }
    let source_ba_cameras = BaCameras {
        cameras: &source_cameras,
        image_camera: image_camera.as_slice().into(),
    };
    let ba_cameras = BaCameras {
        cameras: &cameras,
        image_camera: image_camera.as_slice().into(),
    };

    // The solve is all of the time; the other three walk arrays the size of the
    // reconstruction once. An estimate, as every set of weights is.
    let [p_gather, p_before, p_solve, p_write] = progress.split([0.04, 0.04, 0.90, 0.02]);

    let gather = p_gather.phase("gather arrays");
    // ── The arrays the kernel takes ────────────────────────────────────────
    let mut quats: Vec<UnitQuaternion<f64>> = posed
        .iter()
        .map(|&i| table.images[i].quaternion_wxyz)
        .collect();
    let mut trans: Vec<Vector3<f64>> = posed
        .iter()
        .map(|&i| table.images[i].translation_xyz)
        .collect();
    let n_pt = recon.point_set.points.len();
    let mut points: Vec<[f64; 3]> = recon
        .point_set
        .points
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let is_dir: Vec<bool> = recon
        .point_set
        .points
        .iter()
        .map(|p| p.is_at_infinity())
        .collect();

    let mut uv: Vec<[f64; 2]> = Vec::with_capacity(recon.point_set.tracks.len());
    let mut obs_img: Vec<u32> = Vec::with_capacity(recon.point_set.tracks.len());
    let mut obs_pt: Vec<u32> = Vec::with_capacity(recon.point_set.tracks.len());
    let mut in_solve = vec![false; n_pt];
    for (row, observation) in recon.point_set.tracks.iter().enumerate() {
        let Some(image) = old_to_new[observation.image_index as usize] else {
            continue;
        };
        uv.push([
            f64::from(keypoints[[row, 0]]),
            f64::from(keypoints[[row, 1]]),
        ]);
        obs_img.push(image);
        obs_pt.push(observation.point_index);
        in_solve[observation.point_index as usize] = true;
    }
    if uv.is_empty() {
        return Err(BundleAdjustError::NoObservations);
    }
    let points_in_solve = in_solve.iter().filter(|&&s| s).count();

    // A constraint describes its own point, and the reference it measures from
    // is an image: the same re-indexing the observations went through applies,
    // and a point whose reference is not in the solve comes back free rather
    // than carrying a distance from a camera the solve does not hold.
    let constraints = match &recon.point_set.point_constraints {
        Some(columns) => {
            let mut columns = columns.clone();
            columns.remap_images(&old_to_new);
            let held: Vec<bool> = columns
                .point_constraints
                .iter()
                .map(|&k| k == POINT_CONSTRAINT_HELD)
                .collect();
            let distance: Vec<f64> = columns
                .point_constraints
                .iter()
                .zip(&columns.constraint_distances)
                .map(|(&k, &r)| {
                    if k == POINT_CONSTRAINT_RANGED {
                        r
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            let reference: Vec<Option<DistanceReference>> = columns
                .constraint_reference_images
                .iter()
                .map(|&k| (k != NO_REFERENCE_IMAGE).then_some(DistanceReference::Image(k)))
                .collect();
            PointConstraints::from_arrays(
                Some(&held),
                Some(&distance),
                &reference,
                n_pt,
                posed.len(),
            )?
        }
        None => None,
    };

    drop(gather);
    progress_info!(
        progress,
        "{} images, {} points, {} observations",
        posed.len(),
        points_in_solve,
        uv.len()
    );

    // ── The two solves ────────────────────────────────────────────────────
    //
    // The first is the kernel over an **empty** schedule, which runs no round
    // and reports the residuals at the state it was handed: the "before" median
    // is then measured by the same projection the "after" one is, rather than by
    // a second spelling of it here.
    let residuals = p_before.phase("residuals before");
    let before = crate::geometry::bundle_adjust::bundle_adjust(
        &source_ba_cameras,
        &mut quats.clone(),
        &mut trans.clone(),
        &mut points.clone(),
        &uv,
        &obs_img,
        &obs_pt,
        Some(&is_dir),
        constraints.as_ref(),
        FreePointPolicy::default(),
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        false,
        false,
        false,
        &[],
        options.max_iters,
        options.min_track,
        options.min_obs,
        &residuals,
    );
    let (median_before, measured) = median_finite(&before.residual_norms);
    progress_info!(
        residuals,
        "median {median_before:.3} px over {measured} observations"
    );
    drop(residuals);

    let solve = p_solve.phase("solve");
    let solved = crate::geometry::bundle_adjust::bundle_adjust(
        &ba_cameras,
        &mut quats,
        &mut trans,
        &mut points,
        &uv,
        &obs_img,
        &obs_pt,
        Some(&is_dir),
        constraints.as_ref(),
        FreePointPolicy::default(),
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        options.opt_f,
        // One request, both kernel rungs: `k1` and the spline live on different
        // models, and the kernel decides each per camera.
        options.opt_distortion,
        options.opt_distortion,
        &options.schedule,
        options.max_iters,
        options.min_track,
        options.min_obs,
        &solve,
    );
    let (median_after, measured) = median_finite(&solved.residual_norms);
    progress_info!(
        solve,
        "median {median_after:.3} px over {measured} observations"
    );
    drop(solve);
    // A stopped solve broke off between rounds, so what it holds is the state
    // of whichever round last finished: a refusal rather than an answer, and
    // nothing is written back, because the caller would otherwise keep a value
    // built from a solve that never converged.
    if progress.is_cancelled() {
        return Err(BundleAdjustError::Cancelled);
    }
    // Every residual infinite is the kernel's degenerate exit, which passes the
    // state through: read as a per-point verdict it would delete the whole
    // point set, so it is the call's refusal instead.
    if solved.residual_norms.iter().all(|r| !r.is_finite()) {
        return Err(BundleAdjustError::Degenerate {
            observations: uv.len(),
            min_obs: options.min_obs,
        });
    }

    let _write_back = p_write.phase("write back");
    // ── Writing the answer back ───────────────────────────────────────────
    let mut out = recon.clone_for_edit();
    for (slot, &i) in posed.iter().enumerate() {
        out.image_table.images[i].quaternion_wxyz = quats[slot];
        out.image_table.images[i].translation_xyz = trans[slot];
    }
    // Each camera in the solve takes the one the kernel returned, which is
    // itself at its solved focal under `opt_f`, with its solved `k1` or spline
    // under `opt_distortion` where its model has one, and itself unchanged
    // otherwise:
    // nothing else about any lens moves.
    for (&c, solved_camera) in used.iter().zip(&solved.cameras) {
        out.image_table.cameras[c as usize] = solved_camera.clone();
    }
    // The outermost observation of each camera's images, under the camera the
    // solve returned; the value carries inline keypoints, so nothing is read.
    let used_indexes: Vec<usize> = used.iter().map(|&c| c as usize).collect();
    let outermost = outermost_keypoints(&out, &used_indexes, false);
    let camera_reports: Vec<CameraAdjustment> = used
        .iter()
        .enumerate()
        .map(|(j, &c)| CameraAdjustment {
            camera: c as usize,
            images: image_camera.iter().filter(|&&k| k as usize == j).count(),
            focal_before: source_cameras[j].focal_lengths().0,
            focal_after: solved.cameras[j].focal_lengths().0,
            focal_released: options.opt_f,
            distortion_released: options.opt_distortion && distortion_is_releasable(&cameras[j]),
            spline_refit: spline_refits[j].clone(),
            outermost_observed: outermost[j].observed,
        })
        .collect();

    // Each point's own residuals, for the stored error column and for the
    // verdict on whether anything still sees it.
    let mut squares = vec![0.0f64; n_pt];
    let mut counts = vec![0usize; n_pt];
    for (k, &r) in solved.residual_norms.iter().enumerate() {
        if r.is_finite() {
            let p = obs_pt[k] as usize;
            squares[p] += r * r;
            counts[p] += 1;
        }
    }

    let mut keep = vec![true; n_pt];
    let mut points_deleted = 0usize;
    for p in 0..n_pt {
        if !in_solve[p] {
            continue;
        }
        let row = points[p];
        if !row.iter().all(|c| c.is_finite()) || counts[p] == 0 {
            keep[p] = false;
            points_deleted += 1;
            continue;
        }
        let was_at_infinity = recon.point_set.points[p].is_at_infinity();
        let now_at_infinity = solved.point_at_infinity[p];
        let before_position = recon.point_set.points[p].position;
        let after_position = Point3::new(row[0], row[1], row[2]);
        rescale_patch_frame(
            &mut out,
            p,
            (!was_at_infinity).then(|| recon.image_table.placement_scale(&before_position)),
            (!now_at_infinity).then_some(&after_position),
        );
        let point = &mut out.point_set.points[p];
        point.position = after_position;
        point.w = if now_at_infinity { 0.0 } else { 1.0 };
        point.error = (squares[p] / counts[p] as f64).sqrt() as f32;
    }

    let mut out = if points_deleted > 0 {
        out.filter_points_by_mask(&keep)
    } else {
        out
    };
    out.rebuild_derived_fields();

    let report = BundleAdjustReport {
        images: posed.len(),
        points: points_in_solve,
        observations: uv.len(),
        points_deleted,
        median_residual_before: median_residual(&before.residual_norms, &obs_pt, &keep),
        median_residual_after: median_residual(&solved.residual_norms, &obs_pt, &keep),
        cameras: camera_reports,
    };
    Ok((out, report))
}

/// Whether a stored pose is one at all. Every `.sfmr` image row has the fields;
/// a non-finite one is a placeholder rather than a registration.
///
/// Crate-visible because the same question decides which images enter this
/// solve and whether
/// [`move_camera`](super::move_camera::move_camera) has a pose to replace.
pub(crate) fn is_posed(rotation: &UnitQuaternion<f64>, translation: &Vector3<f64>) -> bool {
    rotation.coords.iter().all(|c| c.is_finite()) && translation.iter().all(|c| c.is_finite())
}

/// Whether the kernel's analytic focal column is exact for this camera, which is
/// what decides whether the focal can be released at all. The kernel degrades a
/// model it is not exact for to a fixed-focal solve;
/// [`bundle_adjust`] refuses instead, and a caller offering the release as a
/// choice asks this first so it can grey the choice rather than take it and
/// refuse.
pub fn focal_is_releasable(camera: &CameraIntrinsics) -> bool {
    matches!(
        camera.model,
        CameraModel::SimplePinhole { .. }
            | CameraModel::EquidistantFisheye { .. }
            | CameraModel::SimpleRadialFisheye { .. }
            | CameraModel::SfmtoolFisheye { .. }
            | CameraModel::SfmtoolPinhole { .. }
    )
}

/// Whether this camera has lens distortion the adjustment can release: a
/// `SIMPLE_RADIAL_FISHEYE`, whose `k1` the kernel frees, or an
/// `SFMTOOL_FISHEYE` or `SFMTOOL_PINHOLE` whose spline is defined, at least two
/// coefficients on a positive finite domain end. A shorter spline evaluates as
/// the identity and has nothing to release. The kernel's releases are exact on
/// these models alone, so every other model's distortion stays where it is. A
/// caller offering the release as a choice asks this first, as it asks
/// [`focal_is_releasable`].
pub fn distortion_is_releasable(camera: &CameraIntrinsics) -> bool {
    if matches!(camera.model, CameraModel::SimpleRadialFisheye { .. }) {
        return true;
    }
    camera
        .model
        .radial_spline()
        .is_some_and(|(bspline, d_max, _)| {
            bspline.len() >= MIN_BSPLINE_COEFFS && d_max.is_finite() && d_max > 0.0
        })
}

/// Where a spline camera's domain ends, as an incidence angle in degrees:
/// `bspline_theta_max` itself for `SFMTOOL_FISHEYE`, and the angle whose
/// tangent is `bspline_rho_max` for `SFMTOOL_PINHOLE`. `None` for a camera with
/// no spline. A caller showing the domain as an editable value reads it here,
/// in the unit [`BundleAdjustOptions::spline_domain_deg`] takes.
pub fn spline_domain_deg(camera: &CameraIntrinsics) -> Option<f64> {
    camera
        .model
        .radial_spline()
        .map(|(_, d_max, radial)| match radial {
            SplineRadial::IncidenceAngle => d_max.to_degrees(),
            SplineRadial::ImagePlaneRadius => d_max.atan().to_degrees(),
        })
}

/// Resize point `p`'s patch frame so it keeps the angular size it had.
///
/// The stored half-vectors are world-space extents on a finite point and
/// angular ones tangent to the direction sphere on a bearing, and the distance
/// that converts between them is the placement distance
/// ([`ImageTable::placement_scale`](super::data::ImageTable::placement_scale)).
/// `before` is that distance where the point had one and `None` where it was a
/// bearing; `after` is the point's new position where it has one and `None`
/// where it is now a bearing. So one call covers a point that merely moved, a
/// point that changed representation, and a point that did neither.
///
/// A degenerate distance leaves the frame alone: a patch at the camera-cloud
/// centroid has no angular size to preserve, and scaling by zero would destroy
/// the frame rather than resize it.
///
/// Crate-visible because every edit that moves a point has to keep its patch
/// the size it looked: this solve, and the re-triangulation
/// [`move_camera`](super::move_camera::move_camera) runs.
///
/// An edit that holds the frame as a loose [`PointRecord`](super::PointRecord)
/// rather than as rows of a value reads the same number out of
/// [`patch_frame_factor`] and applies it itself.
pub(crate) fn rescale_patch_frame(
    out: &mut SfmrReconstruction,
    p: usize,
    before: Option<f64>,
    after: Option<&Point3<f64>>,
) {
    let Some(factor) = patch_frame_factor(&out.image_table, before, after) else {
        return;
    };
    for column in [
        &mut out.point_set.patch_u_halfvec_xyz,
        &mut out.point_set.patch_v_halfvec_xyz,
    ] {
        if let Some(array) = column.as_mut() {
            for c in 0..3 {
                array[[p, c]] *= factor;
            }
        }
    }
}

/// How much a patch frame has to grow for its point to keep the angular size it
/// had, or `None` when the frame is to be left exactly as it is.
///
/// The arithmetic [`rescale_patch_frame`] applies, held apart from the rows it
/// applies it to, because an overlay edit carries the frame in a
/// [`PointRecord`](super::PointRecord) rather than in a column of a value, and
/// two spellings of one ratio could only disagree. `None` covers the two cases
/// that leave a frame alone: a degenerate placement distance on either side, and
/// a ratio of exactly one.
pub(crate) fn patch_frame_factor(
    table: &super::data::ImageTable,
    before: Option<f64>,
    after: Option<&Point3<f64>>,
) -> Option<f32> {
    let usable = |d: f64| (d.is_finite() && d > 0.0).then_some(d);
    let before = match before {
        Some(d) => usable(d)?,
        None => 1.0,
    };
    let after = match after {
        Some(position) => usable(table.placement_scale(position))?,
        None => 1.0,
    };
    let factor = (after / before) as f32;
    (factor.is_finite() && factor != 1.0).then_some(factor)
}

/// The median of the finite residuals of the points `keep` marks.
///
/// Over the survivors rather than over every row, because the deleted points'
/// observations describe geometry the value that comes back does not hold, and
/// a median that counted them would be reporting on a different reconstruction
/// from the one the caller now has.
fn median_residual(residuals: &[f64], obs_pt: &[u32], keep: &[bool]) -> f64 {
    let mut kept: Vec<f64> = residuals
        .iter()
        .zip(obs_pt)
        .filter(|(r, &p)| r.is_finite() && keep[p as usize])
        .map(|(&r, _)| r)
        .collect();
    if kept.is_empty() {
        return f64::NAN;
    }
    median_in_place(&mut kept)
}

/// The median of the residuals that projected, and how many did.
///
/// Deliberately not [`median_residual`], which is the report's figure and is
/// restricted to the points the solve kept: that population is not known until
/// the solve has finished, so a "before" measured over it could not be reported
/// while the stage that measured it was still open. This is over everything
/// that projected, which is what makes the before and after lines an operation
/// reports comparable with each other. Both say what they counted, so a reader
/// can see when they and the report are answering over different populations.
fn median_finite(residuals: &[f64]) -> (f64, usize) {
    let mut finite: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();
    let measured = finite.len();
    if finite.is_empty() {
        return (f64::NAN, 0);
    }
    (median_in_place(&mut finite), measured)
}

#[cfg(test)]
mod tests;
