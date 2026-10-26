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

use super::data::{ImageSet, PerImageAdjustments, SfmrReconstruction};

use super::outermost_keypoint::{outermost_keypoints, KeypointReach};
use crate::camera::distortion::bspline::MIN_BSPLINE_COEFFS;
use crate::camera::{CameraIntrinsics, CameraModel, IntrinsicAdjustment};
use crate::geometry::bundle_adjust::{
    BaCameras, BaSchedule, DistanceReference, FreePointDecision, FreePointPolicy, PointConstraints,
    PointConstraintsError, DEFAULT_PROTECTED_LOSS_SCALE, DEFAULT_SCHEDULE,
};
pub use crate::geometry::bundle_adjust::{
    CameraRelease, ImagePrior, ImagePriorOptions, ImagePriorWidths,
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

/// What the adjustment is allowed to move, and how hard it tries.
///
/// The defaults are the kernel's own, and hold every camera, so an adjustment
/// asked for with nothing stated is the one every other caller in this crate
/// runs.
#[derive(Debug, Clone)]
pub struct BundleAdjustOptions {
    /// What each camera may release, one entry per camera in the
    /// reconstruction's image set, in camera index order. A camera no posed image
    /// uses is not in the solve, and its entry is ignored. An empty list, the
    /// default, holds every camera; any other length than the camera count is
    /// refused ([`BundleAdjustError::ReleaseCount`]). Nothing is released by
    /// default, because a lens that moves is a different claim about the
    /// capture than a pose that does, and the caller should be the one making
    /// it.
    ///
    /// Each camera's entry is checked against its own model, and a release the
    /// model cannot take is refused naming the camera rather than silently
    /// dropped: a caller that asked for a focal solve and got a fixed-focal one
    /// back would have no way to tell.
    ///
    /// - [`CameraRelease::focal`] is accepted on the models the kernel's
    ///   analytic focal column is exact for -- `SIMPLE_PINHOLE`,
    ///   `EQUIDISTANT_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`, `SFMTOOL_FISHEYE` and
    ///   `SFMTOOL_PINHOLE` ([`focal_is_releasable`]).
    /// - [`CameraRelease::distortion`] is accepted where
    ///   [`distortion_is_releasable`] says the model has some: `k1` on
    ///   `SIMPLE_RADIAL_FISHEYE`, the radial spline on `SFMTOOL_FISHEYE` and
    ///   `SFMTOOL_PINHOLE`. It is released only together with that camera's
    ///   focal: neither `k1` nor the spline can change the scale at the centre
    ///   of the image (both leave the slope of the radial map on the axis at
    ///   one), and a distortion released against a held focal could only bend
    ///   the periphery around a scale it cannot fix.
    /// - [`CameraRelease::principal_point`] is accepted on every model, with or
    ///   without the focal: every model projects as `u = fx·x_d + cx` with a
    ///   distorted coordinate that does not read the principal point, so the
    ///   kernel's columns for it are exact everywhere. A step that would put
    ///   it off the image is rejected. It carries no prior: one pair of
    ///   numbers fitted from every image of the camera.
    /// - [`CameraRelease::image_focal`] releases each image's focal scale, on
    ///   the models the focal itself is accepted on, and
    ///   [`CameraRelease::image_principal_point`] each image's principal-point
    ///   offset, on every model. Either holds the camera's shared value of its
    ///   kind in the solve and moves it to the images' mean after every round,
    ///   so it is the same release with or without the shared one beside it,
    ///   and a per-image focal satisfies the distortion's need for a focal.
    ///   Each image is held toward that mean by the prior
    ///   [`Self::image_prior`] chooses.
    ///
    /// A camera with none of the five is held, so a rig can release some
    /// cameras and hold the rest.
    /// With every camera held the solve still runs, and refines the poses and
    /// the points alone.
    pub releases: Vec<CameraRelease>,
    /// The staged trim schedule, `(trim_px, loss_scale)` per round. Each
    /// round's `trim_px` is a positive number of pixels or `f64::INFINITY`,
    /// which trims nothing on the residual, and its `loss_scale` a positive
    /// finite number of pixels; [`check_schedule`] says which round breaks
    /// that.
    pub schedule: Vec<BaSchedule>,
    /// LM iteration budget per round.
    pub max_iters: usize,
    /// Trim survivors a point needs to stay in a round's solve. A point that
    /// falls below it is dropped from the result.
    pub min_track: usize,
    /// Trim survivors below which the round exits degenerate, which this call
    /// refuses.
    pub min_obs: usize,
    /// Whether free points cross between a position and a direction. The
    /// default, [`FreePointPolicy::CROSS`], solves every free point in inverse
    /// depth and stores it as the storage decision at the end of the solve
    /// says; [`FreePointPolicy::NO_CROSS`] keeps each point in the representation
    /// it has in the value. Ranged and held points keep theirs either way.
    pub free_points: FreePointPolicy,
    /// The widths of the prior on a per-image release: derived for each
    /// camera from the solve by default, or fixed.
    pub image_prior: ImagePriorOptions,
}

impl Default for BundleAdjustOptions {
    fn default() -> Self {
        Self {
            releases: Vec::new(),
            schedule: DEFAULT_SCHEDULE.to_vec(),
            max_iters: DEFAULT_MAX_ITERS,
            min_track: DEFAULT_MIN_TRACK,
            min_obs: DEFAULT_MIN_OBS,
            free_points: FreePointPolicy::default(),
            image_prior: ImagePriorOptions::default(),
        }
    }
}

impl BundleAdjustOptions {
    /// The defaults with `release` for each of `camera_count` cameras: the
    /// uniform case, where every camera of an image set is released alike. Pass the
    /// camera count, `recon.image_set.cameras.len()`.
    pub fn uniform(camera_count: usize, release: CameraRelease) -> Self {
        Self {
            releases: vec![release; camera_count],
            ..Self::default()
        }
    }
}

/// Whether `schedule` is one [`bundle_adjust`] runs: at least one round, each
/// with a trim that is a positive number of pixels or `f64::INFINITY` and a
/// loss scale that is a positive finite number of pixels.
///
/// The adjustment asks it first, and a caller that takes a schedule from a
/// person (a text box, a wire argument) asks it before starting one, so both
/// refuse the same schedules in the same words. A NaN in either field is
/// refused: it compares false against everything, so a NaN trim would trim
/// every observation and a NaN loss scale would make every cost NaN.
pub fn check_schedule(schedule: &[BaSchedule]) -> Result<(), BundleAdjustError> {
    if schedule.is_empty() {
        return Err(BundleAdjustError::EmptySchedule);
    }
    for (i, stage) in schedule.iter().enumerate() {
        if stage.trim_px.is_nan() || stage.trim_px <= 0.0 {
            return Err(BundleAdjustError::ScheduleTrim {
                round: i + 1,
                trim_px: stage.trim_px,
            });
        }
        if !stage.loss_scale.is_finite() || stage.loss_scale <= 0.0 {
            return Err(BundleAdjustError::ScheduleLossScale {
                round: i + 1,
                loss_scale: stage.loss_scale,
            });
        }
    }
    Ok(())
}

/// Why an adjustment produced nothing. Every variant names what did not hold,
/// because the caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum BundleAdjustError {
    /// The observations carry no pixel: a `sift_files` value without the
    /// format's optional inline keypoint column.
    NoKeypoints,
    /// The release list's length is neither zero nor the camera count.
    ReleaseCount {
        /// The entries given.
        releases: usize,
        /// The cameras the image set holds.
        cameras: usize,
    },
    /// A camera in the solve was asked to release its focal, and its model is
    /// one the kernel's focal column is not exact for.
    FocalNotReleasable {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
        /// Its model's name.
        camera_model: &'static str,
    },
    /// A camera in the solve was asked to release each image's focal scale,
    /// and its model is one the kernel's focal column is not exact for.
    ImageFocalNotReleasable {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
        /// Its model's name.
        camera_model: &'static str,
    },
    /// A camera in the solve was asked to release its distortion without its
    /// focal.
    DistortionWithoutFocal {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
    },
    /// A camera in the solve was asked to release its distortion, and its
    /// model has none the adjustment can release.
    DistortionNotReleasable {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
        /// Its model's name.
        camera_model: &'static str,
    },
    /// No image of the reconstruction carries a usable pose.
    NoPosedImages,
    /// No observation of a live point falls in a posed image.
    NoObservations,
    /// The schedule has no rounds, so there is nothing to run.
    EmptySchedule,
    /// A round's trim is not a positive number of pixels or `f64::INFINITY`.
    ScheduleTrim {
        /// The round, counted from 1.
        round: usize,
        /// Its trim.
        trim_px: f64,
    },
    /// A round's loss scale is not a positive finite number of pixels.
    ScheduleLossScale {
        /// The round, counted from 1.
        round: usize,
        /// Its loss scale.
        loss_scale: f64,
    },
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
            BundleAdjustError::ReleaseCount { releases, cameras } => write!(
                f,
                "the release list has {releases} entries and the reconstruction has \
                 {cameras} camera(s); give one entry per camera, or none to hold them all"
            ),
            BundleAdjustError::FocalNotReleasable {
                camera,
                camera_model,
            } => write!(
                f,
                "the focal cannot be released on camera {camera}, a {camera_model}; the \
                 adjustment's focal column is exact for SIMPLE_PINHOLE, \
                 EQUIDISTANT_FISHEYE, SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE and \
                 SFMTOOL_PINHOLE"
            ),
            BundleAdjustError::ImageFocalNotReleasable {
                camera,
                camera_model,
            } => write!(
                f,
                "the focal cannot be released per image on camera {camera}, a {camera_model}; \
                 the adjustment's focal-scale column is exact for SIMPLE_PINHOLE, \
                 EQUIDISTANT_FISHEYE, SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE and \
                 SFMTOOL_PINHOLE"
            ),
            BundleAdjustError::DistortionWithoutFocal { camera } => write!(
                f,
                "the lens distortion of camera {camera} is released only together with its \
                 focal length, shared or per image, because neither k1 nor the spline can \
                 change the scale at the centre of the image"
            ),
            BundleAdjustError::DistortionNotReleasable {
                camera,
                camera_model,
            } => write!(
                f,
                "camera {camera}, a {camera_model}, has no lens distortion the adjustment can \
                 release; it releases k1 on SIMPLE_RADIAL_FISHEYE and the spline on \
                 SFMTOOL_FISHEYE and SFMTOOL_PINHOLE"
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
            BundleAdjustError::ScheduleTrim { round, trim_px } => write!(
                f,
                "round {round} of the schedule has trim {trim_px}; a trim is a positive \
                 number of pixels, or inf for an unbounded trim"
            ),
            BundleAdjustError::ScheduleLossScale { round, loss_scale } => write!(
                f,
                "round {round} of the schedule has loss scale {loss_scale}; a loss scale is \
                 a positive finite number of pixels"
            ),
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
    /// One entry per camera in the solve, in camera index order.
    pub cameras: Vec<CameraAdjustment>,
    /// The storage decision of the free points: the noise level they were
    /// decided at, whether the final round converged, and how many crossed in
    /// each direction. `None` under [`FreePointPolicy::NO_CROSS`].
    pub free_point_decision: Option<FreePointDecision>,
}

/// What one adjustment did to one camera.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraAdjustment {
    /// The camera's index among the reconstruction's cameras.
    pub camera: usize,
    /// Posed images taken through it that were in the solve.
    pub images: usize,
    /// Its focal before the solve, the model's first where it carries two.
    pub focal_before: f64,
    /// Its focal after the solve, which is the one before it unless the focal
    /// was released, shared or per image.
    pub focal_after: f64,
    /// Whether its focal was released: shared, or per image, which moves the
    /// shared focal to the images' mean.
    pub focal_released: bool,
    /// Whether its lens distortion was released: `k1` on a
    /// `SIMPLE_RADIAL_FISHEYE`, the spline on a spline model.
    pub distortion_released: bool,
    /// Whether its principal point was released: shared, or per image, which
    /// moves the shared principal point to the images' mean.
    pub principal_point_released: bool,
    /// Whether each image's focal scale was released.
    pub image_focal_released: bool,
    /// Whether each image's principal-point offset was released.
    pub image_principal_point_released: bool,
    /// Its principal point before the solve, `[cx, cy]` in pixels.
    pub principal_point_before: [f64; 2],
    /// Its principal point after the solve, which is the one before it unless
    /// the principal point was released. The report states the shift rather
    /// than how well the data determine it: a principal point that moved far is
    /// one a person should look at.
    pub principal_point_after: [f64; 2],
    /// The outermost of the observations of its images, measured under the
    /// camera the solve returned. Only the observations: the adjustment reads
    /// nothing off disk, so the features detected in the images' `.sift` files
    /// are [`outermost_keypoints`]'s to report.
    pub outermost_observed: Option<KeypointReach>,
    /// What the per-image release did to the camera's images; `None` when the
    /// camera released no per-image parameter.
    pub image_adjustments: Option<ImageAdjustmentSpread>,
}

/// How a camera's images' intrinsic adjustments ended after a per-image
/// release, and the prior they were solved under.
///
/// The statistics are over the camera's posed images. After the re-centring
/// the mean is zero to rounding, so the standard deviations say how far the
/// camera's lens moved between photographs as the solve found it.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageAdjustmentSpread {
    /// The camera's posed images.
    pub images: usize,
    /// The mean of their focal scales, a fraction.
    pub focal_mean: f64,
    /// The standard deviation of their focal scales, a fraction.
    pub focal_std: f64,
    /// The mean of their principal-point offsets, `[x, y]` in pixels.
    pub principal_point_mean: [f64; 2],
    /// The standard deviation of their principal-point offsets, `[x, y]` in
    /// pixels.
    pub principal_point_std: [f64; 2],
    /// The widths the final round ran under: the focal scale's as a fraction,
    /// the principal point's in the short-side unit. `NaN` for a parameter not
    /// released per image.
    pub prior: ImagePriorWidths,
    /// Whether the widths were derived from the solve, rather than fixed.
    pub prior_derived: bool,
    /// The last derivation's spread before it was clamped, in the widths'
    /// units: a width at the floor or the loose width with a spread beyond it
    /// says the clamp decided it. `NaN` where nothing was derived.
    pub spread: ImagePriorWidths,
    /// The images whose adjustment ended more than two prior widths from the
    /// camera's mean, the ones a person should look at.
    pub near_prior: Vec<NearPrior>,
}

/// One image whose adjustment ended more than two prior widths from its
/// camera's mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearPrior {
    /// The image's index in the reconstruction.
    pub image: usize,
    /// Which of its adjustments.
    pub parameter: ImageAdjustmentParameter,
    /// How far from the mean, in prior widths: `|s − s̄|/σ` for the focal
    /// scale, `‖a − ā‖/σ` for the principal-point offset.
    pub widths: f64,
}

/// One of the parameters an intrinsic adjustment holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageAdjustmentParameter {
    /// The focal scale.
    Focal,
    /// The principal-point offset.
    PrincipalPoint,
}

impl ImageAdjustmentParameter {
    /// The parameter's name on every surface: `"focal"` or
    /// `"principal_point"`, the words a camera's `per_image_adjustments` list.
    pub fn name(self) -> &'static str {
        match self {
            ImageAdjustmentParameter::Focal => "focal",
            ImageAdjustmentParameter::PrincipalPoint => "principal_point",
        }
    }
}

/// Bundle-adjust `recon`, returning the adjusted value and a report.
///
/// Every posed image's pose, every live point's position and, as
/// [`BundleAdjustOptions::releases`] states camera by camera, each camera's
/// focal, its `k1` or radial spline, and its principal point are refined
/// together
/// against every observation that carries a pixel, by the staged robust solve in
/// [`crate::geometry::bundle_adjust()`]. The posed images may be taken through
/// any number of the image set's cameras; each keeps its own lens in the solve, and a
/// camera no posed image uses is not in it and comes back as it went in. A free
/// point goes in as the position or the direction it is and, under the default
/// [`BundleAdjustOptions::free_points`], comes back as whichever the
/// point-or-bearing test reads at the end of the solve, so a point can cross
/// between a bearing and a position here; [`FreePointPolicy::NO_CROSS`] honours
/// the value's representation for the whole solve instead.
/// [`BundleAdjustReport::free_point_decision`] says what the decision did. A
/// point's constraint -- free, ranged or held -- is the one its
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
    check_schedule(&options.schedule)?;
    // Every observation states a pixel, or there is nothing to reproject
    // against. The optional inline column answers for `sift_files` too.
    let keypoints = recon
        .point_set
        .keypoints_xy()
        .ok_or(BundleAdjustError::NoKeypoints)?;

    let image_set = &recon.image_set;
    // The solve's image list, and the map from the image set's indexes onto it. An
    // unposed image is not a camera the residuals can be written against, so it
    // sits the solve out and its observations do not enter.
    let mut old_to_new: Vec<Option<u32>> = vec![None; image_set.images.len()];
    let mut posed: Vec<usize> = Vec::with_capacity(image_set.images.len());
    for (i, image) in image_set.images.iter().enumerate() {
        if is_posed(&image.quaternion_wxyz, &image.translation_xyz) {
            old_to_new[i] = Some(posed.len() as u32);
            posed.push(i);
        }
    }
    if posed.is_empty() {
        return Err(BundleAdjustError::NoPosedImages);
    }

    // The cameras of the solve: the image set's shared cameras the posed images
    // use, in camera index order, and each posed image's index into that list.
    // A camera only unposed images use sits the solve out with them. Each posed
    // image's intrinsic adjustment goes in beside them, held, so the solve reads
    // every image through its own intrinsics while it releases the shared lens.
    let mut used: Vec<u32> = posed
        .iter()
        .map(|&i| image_set.images[i].camera_index)
        .collect();
    used.sort_unstable();
    used.dedup();
    let source_cameras: Vec<CameraIntrinsics> = used
        .iter()
        .map(|&c| image_set.cameras[c as usize].clone())
        .collect();
    let image_camera: Vec<u32> = posed
        .iter()
        .map(|&i| {
            used.binary_search(&image_set.images[i].camera_index)
                .expect("every posed image's camera is in the list") as u32
        })
        .collect();
    // Each camera of the solve's release, from the caller's list in camera
    // order. The release refuses rather than degrades: the kernel would hold a
    // parameter its model cannot release and move the rest, and a report
    // saying it was released would then be false of that camera.
    let camera_count = image_set.cameras.len();
    if !options.releases.is_empty() && options.releases.len() != camera_count {
        return Err(BundleAdjustError::ReleaseCount {
            releases: options.releases.len(),
            cameras: camera_count,
        });
    }
    let releases: Vec<CameraRelease> = used
        .iter()
        .map(|&c| {
            options
                .releases
                .get(c as usize)
                .copied()
                .unwrap_or(CameraRelease::HELD)
        })
        .collect();
    for ((&c, camera), release) in used.iter().zip(&source_cameras).zip(&releases) {
        let camera_index = c as usize;
        if release.distortion && !release.focal && !release.image_focal {
            return Err(BundleAdjustError::DistortionWithoutFocal {
                camera: camera_index,
            });
        }
        if release.focal && !focal_is_releasable(camera) {
            return Err(BundleAdjustError::FocalNotReleasable {
                camera: camera_index,
                camera_model: camera.model_name(),
            });
        }
        if release.image_focal && !focal_is_releasable(camera) {
            return Err(BundleAdjustError::ImageFocalNotReleasable {
                camera: camera_index,
                camera_model: camera.model_name(),
            });
        }
        if release.distortion && !distortion_is_releasable(camera) {
            return Err(BundleAdjustError::DistortionNotReleasable {
                camera: camera_index,
                camera_model: camera.model_name(),
            });
        }
    }
    // Every kernel flag on, and each camera's own release narrowing them: the
    // checks above leave only releases its model admits, so the kernel frees
    // exactly what the report below says it did.
    let image_adjustments: Vec<IntrinsicAdjustment> = posed
        .iter()
        .map(|&i| image_set.images[i].intrinsic_adjustment)
        .collect();
    let releases_image = releases
        .iter()
        .any(|r| r.image_focal || r.image_principal_point);
    let ba_cameras = BaCameras {
        cameras: &source_cameras,
        image_camera: image_camera.as_slice().into(),
        releases: Some(&releases),
        image_adjustments: (releases_image || image_adjustments.iter().any(|a| !a.is_zero()))
            .then(|| image_adjustments.as_slice().into()),
    };

    // The solve is all of the time; the other three walk arrays the size of the
    // reconstruction once. An estimate, as every set of weights is.
    let [p_gather, p_before, p_solve, p_write] = progress.split([0.04, 0.04, 0.90, 0.02]);

    let gather = p_gather.phase("gather arrays");
    // ── The arrays the kernel takes ────────────────────────────────────────
    let mut quats: Vec<UnitQuaternion<f64>> = posed
        .iter()
        .map(|&i| image_set.images[i].quaternion_wxyz)
        .collect();
    let mut trans: Vec<Vector3<f64>> = posed
        .iter()
        .map(|&i| image_set.images[i].translation_xyz)
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
        &ba_cameras,
        &mut quats.clone(),
        &mut trans.clone(),
        &mut points.clone(),
        &uv,
        &obs_img,
        &obs_pt,
        Some(&is_dir),
        constraints.as_ref(),
        // An empty schedule runs no round and decides nothing; the policy is
        // stated off so that this call only measures.
        FreePointPolicy::NO_CROSS,
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        false,
        false,
        false,
        false,
        false,
        false,
        None,
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
        options.free_points,
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        true,
        // One request, both kernel rungs: `k1` and the spline live on different
        // models, and the kernel decides each per camera.
        true,
        true,
        // The principal point and the per-image releases, which each camera's
        // release narrows like the rest.
        true,
        true,
        true,
        Some(&options.image_prior),
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
        out.image_set.images[i].quaternion_wxyz = quats[slot];
        out.image_set.images[i].translation_xyz = trans[slot];
    }
    // Each camera in the solve takes the one the kernel returned, which is
    // itself at its solved focal where its focal was released, with its solved
    // `k1` or spline where its distortion was, at its solved principal point
    // where that was, and itself unchanged otherwise: nothing else about any
    // lens moves.
    for (&c, solved_camera) in used.iter().zip(&solved.cameras) {
        out.image_set.cameras[c as usize] = solved_camera.clone();
    }
    // A camera with a per-image release: each posed image takes its solved
    // adjustment, each unposed image of the camera the shift its shared value
    // took, so that its intrinsics are as they were, and the camera records the
    // released parameters as varying per image.
    if releases_image {
        write_back_image_adjustments(
            &mut out.image_set,
            &posed,
            &used,
            &releases,
            &source_cameras,
            &solved.cameras,
            &solved.image_adjustments,
        );
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
            focal_released: releases[j].focal || releases[j].image_focal,
            distortion_released: releases[j].distortion,
            principal_point_released: releases[j].principal_point
                || releases[j].image_principal_point,
            image_focal_released: releases[j].image_focal,
            image_principal_point_released: releases[j].image_principal_point,
            principal_point_before: source_cameras[j].principal_point().into(),
            principal_point_after: solved.cameras[j].principal_point().into(),
            outermost_observed: outermost[j].observed,
            image_adjustments: solved.image_priors[j].map(|prior| {
                image_adjustment_spread(&out.image_set, c as usize, &posed, &releases[j], prior)
            }),
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
    // The frame rescale measures "before" against the input's poses and
    // "after" against the output's; each centroid is taken once, not per point.
    let centroid_before = recon.image_set.camera_centroid();
    let centroid_after = out.image_set.camera_centroid();
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
            &centroid_after,
            (!was_at_infinity)
                .then(|| ImageSet::placement_scale_from(&centroid_before, &before_position)),
            (!now_at_infinity).then_some(&after_position),
        );
        let point = &mut out.point_set.points[p];
        point.position = after_position;
        point.w = if now_at_infinity { 0.0 } else { 1.0 };
        point.rms_reprojection_error = (squares[p] / counts[p] as f64).sqrt() as f32;
    }

    let mut out = if points_deleted > 0 {
        out.filter_points_by_mask(&keep)
    } else {
        out
    };
    out.rebuild_derived_fields();
    // A point that crossed changed the count of points at infinity, which the
    // metadata states as well as the point set.
    out.metadata.infinity_point_count = out.point_set.infinity_point_count as u32;

    let report = BundleAdjustReport {
        images: posed.len(),
        points: points_in_solve,
        observations: uv.len(),
        points_deleted,
        median_residual_before: median_residual(&before.residual_norms, &obs_pt, &keep),
        median_residual_after: median_residual(&solved.residual_norms, &obs_pt, &keep),
        cameras: camera_reports,
        free_point_decision: solved.free_point_decision,
    };
    Ok((out, report))
}

/// Write a per-image release's answer into `image_set`, which already holds the
/// solved cameras: each posed image of a camera that released a per-image
/// parameter takes its solved adjustment from `solved` (one per posed image, in
/// `posed` order); each unposed image of such a camera takes the shift the
/// re-centring gave its camera's shared value, so that its intrinsics are the
/// ones it had; and the camera's `per_image_adjustments` gains the released
/// parameters. `used`, `releases`, `before` and `after` are per camera of the
/// solve.
fn write_back_image_adjustments(
    image_set: &mut ImageSet,
    posed: &[usize],
    used: &[u32],
    releases: &[CameraRelease],
    before: &[CameraIntrinsics],
    after: &[CameraIntrinsics],
    solved: &[IntrinsicAdjustment],
) {
    let camera_count = image_set.cameras.len();
    if image_set.per_image_adjustments.len() < camera_count {
        image_set
            .per_image_adjustments
            .resize(camera_count, PerImageAdjustments::default());
    }
    let mut is_posed = vec![false; image_set.images.len()];
    for (slot, &i) in posed.iter().enumerate() {
        is_posed[i] = true;
        let c = image_set.images[i].camera_index;
        let j = used
            .binary_search(&c)
            .expect("a posed image's camera is in the solve");
        let release = releases[j];
        let adjustment = &mut image_set.images[i].intrinsic_adjustment;
        if release.image_focal {
            adjustment.focal = solved[slot].focal;
        }
        if release.image_principal_point {
            adjustment.principal_point = solved[slot].principal_point;
        }
    }
    for (j, (&c, release)) in used.iter().zip(releases).enumerate() {
        if !(release.image_focal || release.image_principal_point) {
            continue;
        }
        let flags = &mut image_set.per_image_adjustments[c as usize];
        flags.focal |= release.image_focal;
        flags.principal_point |= release.image_principal_point;
        let m = f64::from(before[j].width.min(before[j].height));
        let (f_before, f_after) = (before[j].focal_lengths().0, after[j].focal_lengths().0);
        let (pp_before, pp_after) = (before[j].principal_point(), after[j].principal_point());
        for (i, image) in image_set.images.iter_mut().enumerate() {
            if is_posed[i] || image.camera_index != c {
                continue;
            }
            let adjustment = &mut image.intrinsic_adjustment;
            if release.image_focal {
                adjustment.focal = (1.0 + adjustment.focal) * f_before / f_after - 1.0;
            }
            if release.image_principal_point {
                adjustment.principal_point[0] -= (pp_after.0 - pp_before.0) / m;
                adjustment.principal_point[1] -= (pp_after.1 - pp_before.1) / m;
            }
        }
    }
}

/// The report of camera `camera`'s per-image release over its posed images in
/// `image_set`, which holds the written-back adjustments, under the prior the
/// kernel reports.
fn image_adjustment_spread(
    image_set: &ImageSet,
    camera: usize,
    posed: &[usize],
    release: &CameraRelease,
    prior: ImagePrior,
) -> ImageAdjustmentSpread {
    let m = {
        let cam = &image_set.cameras[camera];
        f64::from(cam.width.min(cam.height))
    };
    let images: Vec<usize> = posed
        .iter()
        .copied()
        .filter(|&i| image_set.images[i].camera_index as usize == camera)
        .collect();
    let adjustments: Vec<IntrinsicAdjustment> = images
        .iter()
        .map(|&i| image_set.images[i].intrinsic_adjustment)
        .collect();
    let n = adjustments.len().max(1) as f64;
    let mean_std = |value: &dyn Fn(&IntrinsicAdjustment) -> f64| {
        let mean = adjustments.iter().map(value).sum::<f64>() / n;
        let var = adjustments
            .iter()
            .map(|a| (value(a) - mean).powi(2))
            .sum::<f64>()
            / n;
        (mean, var.sqrt())
    };
    let (focal_mean, focal_std) = mean_std(&|a| a.focal);
    let (x_mean, x_std) = mean_std(&|a| a.principal_point[0]);
    let (y_mean, y_std) = mean_std(&|a| a.principal_point[1]);
    let mut near_prior = Vec::new();
    for (&i, a) in images.iter().zip(&adjustments) {
        if release.image_focal {
            let widths = (a.focal - focal_mean).abs() / prior.widths.focal;
            if widths > 2.0 {
                near_prior.push(NearPrior {
                    image: i,
                    parameter: ImageAdjustmentParameter::Focal,
                    widths,
                });
            }
        }
        if release.image_principal_point {
            let widths = (a.principal_point[0] - x_mean).hypot(a.principal_point[1] - y_mean)
                / prior.widths.principal_point;
            if widths > 2.0 {
                near_prior.push(NearPrior {
                    image: i,
                    parameter: ImageAdjustmentParameter::PrincipalPoint,
                    widths,
                });
            }
        }
    }
    ImageAdjustmentSpread {
        images: images.len(),
        focal_mean,
        focal_std,
        principal_point_mean: [x_mean * m, y_mean * m],
        principal_point_std: [x_std * m, y_std * m],
        prior: prior.widths,
        prior_derived: prior.derived,
        spread: prior.spread,
        near_prior,
    }
}

/// Whether a stored pose is one at all. Every `.sfmr` image row has the fields;
/// a non-finite one is a placeholder rather than a registration.
///
/// Crate-visible because the same question decides which images enter this
/// solve, whether
/// [`move_camera`](super::move_camera::move_camera) has a pose to replace,
/// which images a resection treats as already registered, and which centres
/// [`ImageSet::camera_centroid`] averages.
pub(crate) fn is_posed(rotation: &UnitQuaternion<f64>, translation: &Vector3<f64>) -> bool {
    rotation.coords.iter().all(|c| c.is_finite()) && translation.iter().all(|c| c.is_finite())
}

/// Whether the kernel's analytic focal column is exact for this camera, which is
/// what decides whether the focal can be released at all. The kernel degrades a
/// model it is not exact for to a fixed-focal solve;
/// [`bundle_adjust`] refuses instead, and a caller offering the release as a
/// choice asks this first so it can grey the choice rather than take it and
/// refuse. The answer depends on the model alone, and the list of models is
/// kept with the kernel in `geometry::bundle_adjust`, since it describes that
/// kernel's focal column rather than the camera.
pub fn focal_is_releasable(camera: &CameraIntrinsics) -> bool {
    crate::geometry::bundle_adjust::focal_is_releasable(&camera.model)
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

/// Resize point `p`'s patch frame so it keeps the angular size it had.
///
/// The stored half-vectors are world-space extents on a finite point and
/// angular ones tangent to the direction sphere on a bearing, and the distance
/// that converts between them is the placement distance
/// ([`ImageSet::placement_scale`](super::data::ImageSet::placement_scale)).
/// `before` is that distance where the point had one and `None` where it was a
/// bearing; `after` is the point's new position where it has one and `None`
/// where it is now a bearing, measured from `centroid`, the camera centroid of
/// the image set the point now stands in
/// ([`ImageSet::camera_centroid`]),
/// which the caller takes once for every point it writes. So one call covers a
/// point that merely moved, a point that changed representation, and a point
/// that did neither.
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
    centroid: &Point3<f64>,
    before: Option<f64>,
    after: Option<&Point3<f64>>,
) {
    let Some(factor) = patch_frame_factor(centroid, before, after) else {
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
/// `before` is the point's placement distance where it had one and `None`
/// where it was a bearing; `after` is its new position where it has one and
/// `None` where it is now a bearing, measured from `centroid`, the camera
/// centroid of the image set the point now stands in
/// ([`ImageSet::camera_centroid`]).
/// A caller writing many points takes the centroid once and measures each
/// `before` with
/// [`ImageSet::placement_scale_from`],
/// so the whole write costs one pass over the images.
///
/// The arithmetic every edit that moves a point applies to its frame (the
/// adjustment's and the camera move's write-back, and the retriangulation's),
/// held apart from the rows it applies it to, because an overlay edit carries
/// the frame in a [`PointRecord`](super::PointRecord) rather than in a column of
/// a value, and a caller drawing a point between two positions reads the size
/// the edit will write from it as well. Two spellings of one ratio could only
/// disagree. `None` covers the two cases that leave a frame alone: a degenerate
/// placement distance on either side, and a ratio of exactly one.
///
/// # Example
///
/// ```
/// use nalgebra::Point3;
/// use sfmtool_core::reconstruction::bundle_adjust::patch_frame_factor;
/// use sfmtool_core::reconstruction::ImageSet;
///
/// let centroid = Point3::origin();
/// let before = ImageSet::placement_scale_from(&centroid, &Point3::new(0.0, 0.0, 4.0));
/// // A point that halves its distance from the centroid halves its frame.
/// let factor = patch_frame_factor(&centroid, Some(before), Some(&Point3::new(0.0, 2.0, 0.0)));
/// assert_eq!(factor, Some(0.5));
/// // A point that becomes a bearing keeps the angular size it had.
/// assert_eq!(patch_frame_factor(&centroid, Some(before), None), Some(0.25));
/// ```
pub fn patch_frame_factor(
    centroid: &Point3<f64>,
    before: Option<f64>,
    after: Option<&Point3<f64>>,
) -> Option<f32> {
    let usable = |d: f64| (d.is_finite() && d > 0.0).then_some(d);
    let before = match before {
        Some(d) => usable(d)?,
        None => 1.0,
    };
    let after = match after {
        Some(position) => usable(ImageSet::placement_scale_from(centroid, position))?,
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
