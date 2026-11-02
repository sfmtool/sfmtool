// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Switch cameras of a reconstruction to another camera model.
//!
//! `specs/core/reconstruction/switch-camera-model.md` is the design. The
//! function here is pure: an [`SfmrReconstruction`] goes in, a new one and a
//! report come out. Each chosen camera drops the per-image parameters the
//! caller names, and is then replaced by the camera
//! [`refit_camera_intrinsics`](crate::camera::refit_intrinsics::refit_camera_intrinsics) fits to it, and every
//! observation of an image that uses the camera is measured before and after,
//! on the same set of observations. Poses, points, keypoints, patches and
//! tracks are not touched; the stored per-point errors of the points those
//! images observe are recomputed.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use super::bundle_adjust::ImageAdjustmentParameter;
use super::data::{observation_reprojection_error, ImageSet, SfmrReconstruction};
use super::drop_image_adjustments::{drop_image_adjustments, DropImageAdjustmentsError};
use super::outermost_keypoint::{outermost_keypoints, OutermostKeypoints};
use crate::camera::intrinsics::SplineRadial;
use crate::camera::refit_intrinsics::{
    domain_end_of, far_corner_domain_end, image_corner_deg, refit_camera_intrinsics_over,
    refit_radial_correction, refit_spline, CameraIntrinsicsRefit, RadialCorrectionRefit,
    RefitError, RefitOptions, RefitTarget, ThetaFitSource,
};
use crate::camera::report::trustworthy_max_theta_deg;
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::numeric::quantile_of_sorted;

/// Why a switch produced nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchCameraModelError {
    /// No camera was named.
    NoCameras,
    /// A camera index past the image set's cameras.
    UnknownCamera {
        /// The index given.
        camera: usize,
        /// The number of cameras the reconstruction has.
        count: usize,
    },
    /// The fit refused one camera.
    Refit {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
        /// The fit's refusal.
        error: RefitError,
    },
    /// The observation pixels could not be read: a `sift_files`
    /// reconstruction without inline keypoints whose `.sift` files are missing.
    Observations(String),
    /// The camera keeps a per-image parameter the switch cannot carry over
    /// ([`image_adjustments_carried`]): a radial correction or rolling-shutter
    /// rates on a target other than `SFMTOOL_PINHOLE`. The radial correction
    /// is a curve added to an `SFMTOOL_PINHOLE` camera's spline and the rates
    /// are defined on `SFMTOOL_PINHOLE` alone, so the switched camera would
    /// read them as something else. Dropping them
    /// ([`SwitchCameraModelOptions::drop_image_adjustments`]) lifts it.
    LensCorrections {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
        /// The target model's name.
        target: &'static str,
        /// The parameters kept that the switch cannot carry, in
        /// [`ImageAdjustmentParameter::ALL`] order.
        kept: Vec<ImageAdjustmentParameter>,
    },
    /// Dropping the named per-image parameters refused, before any fit.
    Drop(DropImageAdjustmentsError),
    /// One image's radial correction could not be refitted to the switched
    /// camera's new spline domain ([`refit_radial_corrections`]).
    RadialCorrection {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
        /// The image's index among the reconstruction's images.
        image: usize,
        /// The refit's refusal.
        error: RefitError,
    },
}

impl fmt::Display for SwitchCameraModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwitchCameraModelError::NoCameras => write!(f, "no camera was named to switch"),
            SwitchCameraModelError::UnknownCamera { camera, count } => write!(
                f,
                "camera {camera} does not exist; the reconstruction has {count} camera(s)"
            ),
            SwitchCameraModelError::Refit { camera, error } => {
                write!(f, "camera {camera}: {error}")
            }
            SwitchCameraModelError::Observations(e) => write!(
                f,
                "the observations' pixels could not be read to compare the models: {e}"
            ),
            SwitchCameraModelError::LensCorrections {
                camera,
                target,
                kept,
            } => {
                let names: Vec<&str> = kept.iter().map(|&p| correction_words(p)).collect();
                write!(
                    f,
                    "camera {camera} keeps its images' own {}, which a switch to {target} \
                     does not carry over (only an SFMTOOL_PINHOLE target carries the rates \
                     and the radial correction); drop them per image to switch",
                    names.join(" and ")
                )
            }
            SwitchCameraModelError::Drop(e) => write!(f, "{e}"),
            SwitchCameraModelError::RadialCorrection {
                camera,
                image,
                error,
            } => write!(
                f,
                "camera {camera}: the radial correction of image {image} could not be refitted \
                 to the new spline domain: {error}"
            ),
        }
    }
}

impl std::error::Error for SwitchCameraModelError {}

/// What a refusal calls a per-image parameter an image keeps.
fn correction_words(parameter: ImageAdjustmentParameter) -> &'static str {
    match parameter {
        ImageAdjustmentParameter::Focal => "focal scale",
        ImageAdjustmentParameter::PrincipalPoint => "principal-point offset",
        ImageAdjustmentParameter::RadialBspline => "radial correction",
        ImageAdjustmentParameter::RollingShutter => "rolling-shutter rates",
    }
}

/// One per-image parameter a camera varies, and whether a switch carries it
/// over; [`image_adjustments_carried`] lists them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CarriedImageAdjustment {
    /// The parameter.
    pub parameter: ImageAdjustmentParameter,
    /// Whether the switch keeps it, as it is or refitted (`refitted`).
    /// `false` is a parameter the switch refuses to keep
    /// ([`SwitchCameraModelError::LensCorrections`]), which has to be dropped
    /// for the switch to run.
    pub carried: bool,
    /// Whether the switch carries it by refitting each image's value to the
    /// switched camera rather than as it is: the radial correction of an
    /// `SFMTOOL_PINHOLE` camera whose spline domain end the switch moves
    /// ([`refit_radial_corrections`]). Never `true` where `carried` is
    /// `false`.
    pub refitted: bool,
}

/// The per-image parameters camera `camera` of `image_set` varies: those its
/// `per_image_adjustments` flags list, and any other one of its images holds a
/// non-zero value of, in [`ImageAdjustmentParameter::ALL`] order. These are
/// the parameters [`drop_image_adjustments`] changes something for.
pub fn varied_image_adjustments(
    image_set: &ImageSet,
    camera: usize,
) -> Vec<ImageAdjustmentParameter> {
    let flags = image_set.camera_per_image_adjustments(camera);
    let images = || {
        image_set
            .images
            .iter()
            .filter(move |im| im.camera_index as usize == camera)
            .map(|im| &im.intrinsic_adjustment)
    };
    ImageAdjustmentParameter::ALL
        .into_iter()
        .filter(|&p| match p {
            ImageAdjustmentParameter::Focal => flags.focal || images().any(|a| a.focal != 0.0),
            ImageAdjustmentParameter::PrincipalPoint => {
                flags.principal_point || images().any(|a| a.principal_point != [0.0; 2])
            }
            ImageAdjustmentParameter::RadialBspline => {
                flags.radial_bspline || images().any(|a| a.radial_bspline != [0.0; 2])
            }
            ImageAdjustmentParameter::RollingShutter => {
                flags.rolling_shutter || images().any(|a| a.rolling_shutter != [0.0; 3])
            }
        })
        .collect()
}

/// Each per-image parameter camera `camera` of `recon` varies
/// ([`varied_image_adjustments`]), and whether a switch to `target` with
/// `options` carries it over as it is.
///
/// The focal scale and the principal-point offset are relative to whatever
/// the camera becomes, so every switch carries them. The rolling-shutter rates
/// are defined on `SFMTOOL_PINHOLE` alone, so only an `SFMTOOL_PINHOLE` target
/// carries them. The radial correction is a curve on the camera's own spline
/// domain, `τ = ρ/ρ_max`, added to the camera's spline, so it too is carried
/// only by an `SFMTOOL_PINHOLE` target: as it is where the domain end stays
/// put (a refit of the spline as itself with neither
/// [`RefitOptions::theta_fit_deg`] nor [`RefitOptions::spline_domain_deg`]
/// given), and refitted to the new domain where the switch moves it
/// ([`CarriedImageAdjustment::refitted`], [`refit_radial_corrections`]). A
/// parameter that is not carried has to be dropped
/// ([`SwitchCameraModelOptions::drop_image_adjustments`]); kept, the switch
/// refuses ([`SwitchCameraModelError::LensCorrections`]). A caller offering
/// the choice, such as the viewer's camera model proposal, reads here which
/// drops a target forces.
///
/// # Errors
///
/// [`SwitchCameraModelError::UnknownCamera`] for an index past the image set's
/// cameras.
pub fn image_adjustments_carried(
    recon: &SfmrReconstruction,
    camera: usize,
    target: &RefitTarget,
    options: &RefitOptions,
) -> Result<Vec<CarriedImageAdjustment>, SwitchCameraModelError> {
    let image_set = &recon.image_set;
    let count = image_set.cameras.len();
    let Some(source) = image_set.cameras.get(camera) else {
        return Err(SwitchCameraModelError::UnknownCamera { camera, count });
    };
    let pinhole_target = matches!(target, RefitTarget::SfmtoolPinhole { .. });
    let moves_domain = pinhole_target && moves_domain_end(source, options);
    Ok(varied_image_adjustments(image_set, camera)
        .into_iter()
        .map(|parameter| {
            let (carried, refitted) = match parameter {
                ImageAdjustmentParameter::Focal | ImageAdjustmentParameter::PrincipalPoint => {
                    (true, false)
                }
                ImageAdjustmentParameter::RadialBspline => (pinhole_target, moves_domain),
                ImageAdjustmentParameter::RollingShutter => (pinhole_target, false),
            };
            CarriedImageAdjustment {
                parameter,
                carried,
                refitted,
            }
        })
        .collect())
}

/// Whether a switch of the `SFMTOOL_PINHOLE` camera `source` to
/// `SFMTOOL_PINHOLE` with `options` puts the spline's domain end anywhere but
/// where it is, bit for bit: a given domain end does unless it is the same
/// number, a fit over a given angle places it at the far image corner, and a
/// refit of the spline with neither keeps it. `false` for any other source.
fn moves_domain_end(source: &CameraIntrinsics, options: &RefitOptions) -> bool {
    let CameraModel::SfmtoolPinhole {
        bspline_rho_max, ..
    } = &source.model
    else {
        return false;
    };
    let radial = SplineRadial::ImagePlaneRadius;
    let placed = match (options.theta_fit_deg, options.spline_domain_deg) {
        (_, Some(deg)) => domain_end_of(radial, deg).ok(),
        (None, None) => Some(*bspline_rho_max),
        (Some(_), None) => Some(far_corner_domain_end(source, radial)),
    };
    placed != Some(*bspline_rho_max)
}

/// The median, 90th percentile and maximum of a set of reprojection errors,
/// in pixels. All `NaN` over an empty set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorSummary {
    /// The median.
    pub median_px: f64,
    /// The 90th percentile, linearly interpolated.
    pub p90_px: f64,
    /// The largest.
    pub max_px: f64,
}

impl ErrorSummary {
    fn of(values: &[f64]) -> Self {
        if values.is_empty() {
            return ErrorSummary {
                median_px: f64::NAN,
                p90_px: f64::NAN,
                max_px: f64::NAN,
            };
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        ErrorSummary {
            median_px: quantile_of_sorted(&sorted, 0.5),
            p90_px: quantile_of_sorted(&sorted, 0.9),
            max_px: sorted[sorted.len() - 1],
        }
    }
}

/// The reprojection errors of one camera's observations before and after the
/// switch, over one fixed set of observations.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationComparison {
    /// Observations of images that use the camera with a finite error both
    /// before and after: the set every figure below is over.
    pub observations: usize,
    /// Observations of those images left out because one of the two models
    /// gave them no pixel (a point behind the camera, an unposed image).
    pub unmeasured: usize,
    /// The largest incidence angle among the observations, in degrees, from
    /// the point's direction in the camera frame.
    pub max_theta_deg: f64,
    /// The errors under the source model.
    pub before: ErrorSummary,
    /// The errors under the fitted model.
    pub after: ErrorSummary,
    /// Observations whose error changed by more than a pixel.
    pub changed_over_1px: usize,
    /// The source's trusted bound in degrees, where it has one.
    pub trusted_deg: Option<f64>,
    /// Observations past that bound, which are the ones the switch is for.
    pub past_trusted: usize,
    /// Their errors under the source model.
    pub past_trusted_before: ErrorSummary,
    /// Their errors under the fitted model.
    pub past_trusted_after: ErrorSummary,
}

/// How a switch runs: the per-image parameters dropped first, the lens fit's
/// options, and what the report carries beside the fit.
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchCameraModelOptions {
    /// The per-image parameters each switched camera drops before its fit, by
    /// [`drop_image_adjustments`]: the flags cleared, the images' values zeroed
    /// and their mean folded into the camera, which is then the camera the
    /// target is fitted to. A parameter a camera does not vary is passed over.
    /// Empty by default, which keeps every image's adjustment.
    pub drop_image_adjustments: Vec<ImageAdjustmentParameter>,
    /// The fit's largest angle and the spline domain end, as the lens fit
    /// takes them.
    pub refit: RefitOptions,
    /// Whether each camera's entry carries its outermost keypoint
    /// ([`CameraSwitch::outermost`]). On by default. Off, the entry's
    /// `outermost` is `None` and no `.sift` file is read for it, for a caller
    /// that already holds the keypoint or does not show it. The observations'
    /// pixels are still read from the `.sift` files of a value without the
    /// inline keypoint column, since the comparison and the points' stored
    /// errors are measured against them.
    pub outermost_keypoint: bool,
}

impl Default for SwitchCameraModelOptions {
    fn default() -> Self {
        Self {
            drop_image_adjustments: Vec::new(),
            refit: RefitOptions::default(),
            outermost_keypoint: true,
        }
    }
}

/// What the switch did to one camera.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraSwitch {
    /// The camera's index among the reconstruction's cameras.
    pub camera: usize,
    /// The camera before the switch, before any drop.
    pub source: CameraIntrinsics,
    /// The per-image parameters the camera dropped before the fit: those of
    /// [`SwitchCameraModelOptions::drop_image_adjustments`] it varied, in
    /// [`ImageAdjustmentParameter::ALL`] order.
    pub dropped_image_adjustments: Vec<ImageAdjustmentParameter>,
    /// The fit, carrying the camera after the switch. It is fitted to the
    /// camera the drops left, so its distances are from that camera.
    pub refit: CameraIntrinsicsRefit,
    /// The images' radial corrections refitted to the camera's new spline
    /// domain ([`refit_radial_corrections`]), with how far each image's curve
    /// moved; `None` where the switch refitted none: the camera keeps its
    /// domain end, varies no radial correction, or dropped it.
    pub radial_corrections: Option<RefittedRadialCorrections>,
    /// Posed or not, the images that use the camera.
    pub images: usize,
    /// The observation comparison.
    pub observations: ObservationComparison,
    /// The outermost keypoint of the camera's images, observed and detected,
    /// measured under the camera after the switch: how far out the
    /// photographs reach, beside the spline domain the fit placed. `None`
    /// where [`SwitchCameraModelOptions::outermost_keypoint`] was off.
    pub outermost: Option<OutermostKeypoints>,
}

/// The radial corrections of one camera's images, refitted to the camera's
/// new spline domain by [`refit_radial_corrections`].
#[derive(Debug, Clone, PartialEq)]
pub struct RefittedRadialCorrections {
    /// Each image refitted, by its index among the reconstruction's images,
    /// ascending, with its new correction and its distance from its old curve.
    pub images: Vec<(usize, RadialCorrectionRefit)>,
    /// RMS pixel distance between each image's radial map before and after,
    /// over every refitted image's sampled angles.
    pub rms_px: f64,
    /// The largest such distance.
    pub max_px: f64,
    /// The image it belongs to.
    pub max_image: usize,
    /// How many images the monotonicity constraint changed.
    pub monotone_constrained_images: usize,
}

/// What one switch did, one entry per switched camera in camera index order.
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchCameraModelReport {
    /// One entry per switched camera.
    pub cameras: Vec<CameraSwitch>,
}

/// Switch `cameras` of `recon` to `target`, returning the switched value and a
/// report.
///
/// Each camera is fitted independently by
/// [`refit_camera_intrinsics`](crate::camera::refit_intrinsics::refit_camera_intrinsics),
/// with the fit's options in [`SwitchCameraModelOptions::refit`]. When
/// [`RefitOptions::theta_fit_deg`] is `None` the fit's largest angle is the
/// source's trusted bound, or, for a source without one, the largest incidence
/// angle among the observations of the camera's images (the far image corner
/// when it has none).
///
/// A spline camera switched to its own spline model -- an `SFMTOOL_FISHEYE`
/// to `SFMTOOL_FISHEYE`, an `SFMTOOL_PINHOLE` to `SFMTOOL_PINHOLE` -- is a
/// change of coefficient count or domain rather than of model, and with
/// [`RefitOptions::theta_fit_deg`] `None` it is fitted by [`refit_spline`]
/// instead: over the whole new domain, with the domain end kept exactly where
/// [`RefitOptions::spline_domain_deg`] is `None`, and constrained to stay
/// monotone. The entry's `refit` then reports `ThetaFitSource::SplineDomain`,
/// the distance from the old curve and where the monotonicity constraint
/// bound. With a `theta_fit_deg` given, the camera is fitted like any other
/// source, over that angle. A perspective target is refused for a camera one
/// of whose points lies 90° or more off its axis, naming the point
/// ([`RefitError::ObservationsPast90`]). The first camera refused refuses the
/// whole switch, and nothing changes.
///
/// Before any fit, each camera drops the per-image parameters
/// [`SwitchCameraModelOptions::drop_image_adjustments`] names that it varies,
/// by [`drop_image_adjustments`], and the target is fitted to the camera that
/// leaves. A camera that keeps a per-image parameter the switch cannot carry
/// ([`image_adjustments_carried`]) is refused
/// ([`SwitchCameraModelError::LensCorrections`]). Where an `SFMTOOL_PINHOLE`
/// camera's switch moves its spline's domain end, each image's radial
/// correction is refitted to the new domain after the camera's fit
/// ([`refit_radial_corrections`]), and the entry's `radial_corrections` says
/// how far each image's curve moved. The comparison's `before` is measured on
/// `recon` as it was handed in, so it covers the drops and the switch
/// together.
///
/// Poses, points, keypoints, patches and tracks are carried over unchanged. The
/// stored error of every point observed in an image that uses a switched
/// camera is recomputed as the root mean square of its reprojection errors, the convention
/// [`SfmrReconstruction::recompute_point_errors`] writes.
///
/// The observations' pixels come from the inline keypoint column where the
/// value carries one, and otherwise from each image's `.sift` file.
///
/// Each camera's entry also carries its outermost keypoint
/// ([`outermost_keypoints`]), observed and, where the images' `.sift` files can
/// be read, detected, measured under the switched camera. A file that cannot
/// be read leaves `detected` empty rather than refusing the switch. With
/// [`SwitchCameraModelOptions::outermost_keypoint`] off the entry carries none
/// and no file is read for it.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::camera::refit_intrinsics::RefitTarget;
/// use sfmtool_core::reconstruction::switch_camera_model::{
///     switch_camera_model, SwitchCameraModelOptions,
/// };
/// # fn run(recon: &sfmtool_core::SfmrReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8))?;
/// let options = SwitchCameraModelOptions::default();
/// let (switched, report) = switch_camera_model(recon, &[0], &target, &options)?;
/// let camera = &report.cameras[0];
/// println!(
///     "rms {:.3} px over θ ≤ {:.1}°; median error {:.3} -> {:.3} px",
///     camera.refit.rms_px,
///     camera.refit.theta_fit_deg,
///     camera.observations.before.median_px,
///     camera.observations.after.median_px,
/// );
/// # let _ = switched;
/// # Ok(())
/// # }
/// ```
pub fn switch_camera_model(
    recon: &SfmrReconstruction,
    cameras: &[usize],
    target: &RefitTarget,
    options: &SwitchCameraModelOptions,
) -> Result<(SfmrReconstruction, SwitchCameraModelReport), SwitchCameraModelError> {
    let refit_options = &options.refit;
    let count = recon.image_set.cameras.len();
    let mut chosen: Vec<usize> = cameras.to_vec();
    chosen.sort_unstable();
    chosen.dedup();
    if chosen.is_empty() {
        return Err(SwitchCameraModelError::NoCameras);
    }
    if let Some(&camera) = chosen.iter().find(|&&c| c >= count) {
        return Err(SwitchCameraModelError::UnknownCamera { camera, count });
    }

    let image_set = &recon.image_set;
    let tracks = &recon.point_set.tracks;
    // Which shared camera each observation's image belongs to.
    let camera_of_row =
        |row: usize| image_set.images[tracks[row].image_index as usize].camera_index;
    // Which slot of `chosen` each camera is, or `None` for a camera left alone.
    let mut slot_of: Vec<Option<usize>> = vec![None; count];
    for (slot, &c) in chosen.iter().enumerate() {
        slot_of[c] = Some(slot);
    }

    // The rows the switch measures (every observation of an image that uses a
    // switched camera) and the rows whose point's stored error it rewrites
    // (every observation of a point one of those rows observes).
    let n_pt = recon.point_set.points.len();
    let mut touched = vec![false; n_pt];
    let switched_rows: Vec<usize> = (0..tracks.len())
        .filter(|&row| slot_of[camera_of_row(row) as usize].is_some())
        .collect();
    for &row in &switched_rows {
        touched[tracks[row].point_index as usize] = true;
    }
    let touched_rows: Vec<usize> = (0..tracks.len())
        .filter(|&row| touched[tracks[row].point_index as usize])
        .collect();
    // The drops first, camera by camera: each camera's fit is of the camera
    // its drops leave, and the value they leave is the one written back.
    let drops = &options.drop_image_adjustments;
    let dropped: Vec<Vec<ImageAdjustmentParameter>> = chosen
        .iter()
        .map(|&c| {
            varied_image_adjustments(image_set, c)
                .into_iter()
                .filter(|p| drops.contains(p))
                .collect()
        })
        .collect();
    let mut after_drops: Option<SfmrReconstruction> = None;
    for (&c, parameters) in chosen.iter().zip(&dropped) {
        if !parameters.is_empty() {
            let from = after_drops.as_ref().unwrap_or(recon);
            after_drops = Some(
                drop_image_adjustments(from, c, parameters)
                    .map_err(SwitchCameraModelError::Drop)?,
            );
        }
    }
    let working = after_drops.as_ref().unwrap_or(recon);
    // The fits next: each is a few milliseconds, and a refused one refuses
    // the switch before any `.sift` file is read for the pixels below.
    let mut refits: Vec<CameraIntrinsicsRefit> = Vec::with_capacity(chosen.len());
    let mut corrections: Vec<Option<RefittedRadialCorrections>> = Vec::with_capacity(chosen.len());
    for &c in &chosen {
        let refit = fit_camera_switch(working, c, target, refit_options)?;
        corrections.push(refit_radial_corrections(working, c, &refit)?);
        refits.push(refit);
    }
    let pixels = observation_pixels(recon, &touched_rows)?;
    let theta: Vec<f64> = switched_rows
        .iter()
        .map(|&row| incidence_deg(recon, row))
        .collect();

    // ── The switched value ─────────────────────────────────────────────────
    let mut out = match after_drops {
        Some(value) => value,
        None => recon.clone_for_edit(),
    };
    for (&c, refit) in chosen.iter().zip(&refits) {
        out.image_set.cameras[c] = refit.camera.clone();
    }
    // Every image keeps what the drops left of its intrinsic adjustment as it
    // is, since the focal scale, the principal-point offset and the rates are
    // relative to whatever its camera is, except a radial correction written
    // on a domain end the switch moved, which takes its refit. The errors
    // below are read through each image's intrinsics, before against the
    // value as it was handed in, so the comparison covers the drops and the
    // switch together.
    for refitted in corrections.iter().flatten() {
        for (image, correction) in &refitted.images {
            out.image_set.images[*image]
                .intrinsic_adjustment
                .radial_bspline = correction.radial_bspline;
        }
    }
    let before_intrinsics = recon.image_set.all_image_intrinsics();
    let after_intrinsics = out.image_set.all_image_intrinsics();

    // Each touched point's stored error, over every one of its observations
    // under the cameras the value now holds.
    let mut sums = vec![0.0f64; n_pt];
    let mut counts = vec![0u32; n_pt];
    for &row in &touched_rows {
        if let Some(e) = row_error(&out, &after_intrinsics, row, pixels[row]) {
            let p = tracks[row].point_index as usize;
            sums[p] += e * e;
            counts[p] += 1;
        }
    }
    for p in 0..n_pt {
        if touched[p] {
            out.point_set.points[p].rms_reprojection_error = if counts[p] > 0 {
                (sums[p] / f64::from(counts[p])).sqrt() as f32
            } else {
                0.0
            };
        }
    }

    // ── The report, over one fixed set of observations per camera ──────────
    let mut rows_of: Vec<Vec<Row>> = (0..chosen.len()).map(|_| Vec::new()).collect();
    for (&row, &theta_deg) in switched_rows.iter().zip(&theta) {
        let slot = slot_of[camera_of_row(row) as usize].expect("a switched row");
        rows_of[slot].push(Row {
            theta_deg,
            before: row_error(recon, &before_intrinsics, row, pixels[row]),
            after: row_error(&out, &after_intrinsics, row, pixels[row]),
        });
    }
    // One entry per chosen camera, in the same ascending order, or none at all
    // where the caller left the keypoint out, so no `.sift` file is read for it.
    let outermost: Vec<Option<OutermostKeypoints>> = if options.outermost_keypoint {
        outermost_keypoints(&out, &chosen, true)
            .into_iter()
            .map(Some)
            .collect()
    } else {
        vec![None; chosen.len()]
    };
    let mut entries = Vec::with_capacity(chosen.len());
    for (((((&c, refit), radial_corrections), rows), outermost), dropped) in chosen
        .iter()
        .zip(refits)
        .zip(corrections)
        .zip(rows_of)
        .zip(outermost)
        .zip(dropped)
    {
        let images = image_set
            .images
            .iter()
            .filter(|image| image.camera_index as usize == c)
            .count();
        let observations = compare(&rows, refit.extent.source_trusted_deg);
        entries.push(CameraSwitch {
            camera: c,
            source: image_set.cameras[c].clone(),
            dropped_image_adjustments: dropped,
            refit,
            radial_corrections,
            images,
            observations,
            outermost,
        });
    }
    Ok((out, SwitchCameraModelReport { cameras: entries }))
}

/// The camera [`switch_camera_model`] would put in place of camera `camera`
/// of `recon`, and the fit's report, without building the switched value or
/// reading any pixel.
///
/// The fit a switch makes for one camera, with the same defaults and the same
/// refusals: the fit angle defaults to the source's trusted bound, or for a
/// source without one to the widest incidence angle among the observations of
/// the camera's images (the far image corner when it has none); a spline camera
/// switched to its own spline model with no fit angle is refitted by
/// [`refit_spline`]; and a perspective target is refused when one of the
/// camera's points lies 90° or more off its axis. It reads the poses and the
/// point positions in one pass over the observations, so a caller that shows a
/// fit before applying it, such as the viewer's camera model proposal, gets the
/// camera the switch would install for the same request.
///
/// It fits the camera as `recon` holds it, so a caller dropping per-image
/// parameters first ([`SwitchCameraModelOptions::drop_image_adjustments`])
/// calls [`drop_image_adjustments`] and hands this the value that returns, as
/// [`switch_camera_model`] does. It fits the shared camera alone; the images'
/// radial corrections the switch refits to a moved domain end are
/// [`refit_radial_corrections`] of the fit this returns.
///
/// # Errors
///
/// [`SwitchCameraModelError::UnknownCamera`] for an index past the image set's
/// cameras, [`SwitchCameraModelError::LensCorrections`] for a camera that
/// keeps a per-image parameter the switch cannot carry
/// ([`image_adjustments_carried`]), and [`SwitchCameraModelError::Refit`]
/// where the fit refuses.
pub fn fit_camera_switch(
    recon: &SfmrReconstruction,
    camera: usize,
    target: &RefitTarget,
    options: &RefitOptions,
) -> Result<CameraIntrinsicsRefit, SwitchCameraModelError> {
    let image_set = &recon.image_set;
    let count = image_set.cameras.len();
    // The shared camera: the switch refits the lens all its images share, and
    // each image's adjustment rides on whatever camera it becomes.
    let Some(source) = image_set.cameras.get(camera) else {
        return Err(SwitchCameraModelError::UnknownCamera { camera, count });
    };
    // The radial correction is a curve on the camera's own spline domain and
    // the rates are defined on `SFMTOOL_PINHOLE`, so each rides on the
    // switched camera only where that stays as it was.
    let kept: Vec<ImageAdjustmentParameter> =
        image_adjustments_carried(recon, camera, target, options)?
            .into_iter()
            .filter(|p| !p.carried)
            .map(|p| p.parameter)
            .collect();
    if !kept.is_empty() {
        return Err(SwitchCameraModelError::LensCorrections {
            camera,
            target: target.model_name(),
            kept,
        });
    }
    let tracks = &recon.point_set.tracks;
    let widest = widest_ray(recon, camera);
    let max_theta_deg = widest.map_or(f64::NAN, |(_, t)| t);
    if let Some((row, max_theta_deg)) = widest.filter(|_| target.is_perspective()) {
        if max_theta_deg >= 90.0 {
            return Err(SwitchCameraModelError::Refit {
                camera,
                error: RefitError::ObservationsPast90 {
                    max_theta_deg,
                    point: tracks[row].point_index,
                },
            });
        }
    }
    let (theta_fit_deg, theta_fit_source) = match options.theta_fit_deg {
        Some(theta) => (theta, ThetaFitSource::Given),
        None => fit_angle_default(source, max_theta_deg, camera)?,
    };
    match (spline_refit_count(source, target), options.theta_fit_deg) {
        (Some(count), None) => refit_spline(source, count, options.spline_domain_deg),
        _ => refit_camera_intrinsics_over(
            source,
            target,
            theta_fit_deg,
            theta_fit_source,
            options.spline_domain_deg,
        ),
    }
    .map_err(|error| SwitchCameraModelError::Refit { camera, error })
}

/// The radial corrections of camera `camera`'s images refitted to `refit`,
/// the camera's fit by [`fit_camera_switch`], where that fit moved an
/// `SFMTOOL_PINHOLE` spline's domain end; `None` where there is nothing to
/// refit.
///
/// A radial correction is a curve on the camera's own domain, `τ = ρ/ρ_max`,
/// so where `ρ_max` moves the same two coefficients would describe another
/// curve. Each image's correction is refitted the way the camera's curve was:
/// over the angles the camera's fit sampled, so that the new camera's spline
/// plus the new correction is as close as it can be to the old camera's
/// spline plus the old correction, linear tail included, under the same
/// monotonicity floor. The refit is two coefficients per image and one small
/// least-squares solve, a fraction of a millisecond, so it takes no
/// `Progress`. An image is refitted where the camera's
/// `per_image_adjustments` lists the radial correction, or where its own
/// correction is non-zero; a zero correction on a camera that does not list
/// it stays zero. The focal scale, the principal-point offset and the rates
/// are relative to whatever the camera becomes and are not refitted.
///
/// `None` where either camera is not an `SFMTOOL_PINHOLE`, the domain end is
/// the same number bit for bit (the corrections then mean what they did, and
/// are carried as they are), or no image has a correction to refit.
///
/// # Errors
///
/// [`SwitchCameraModelError::UnknownCamera`] for an index past the image set's
/// cameras, and [`SwitchCameraModelError::RadialCorrection`] naming the first
/// image whose refit is refused.
pub fn refit_radial_corrections(
    recon: &SfmrReconstruction,
    camera: usize,
    refit: &CameraIntrinsicsRefit,
) -> Result<Option<RefittedRadialCorrections>, SwitchCameraModelError> {
    let image_set = &recon.image_set;
    let count = image_set.cameras.len();
    let Some(source) = image_set.cameras.get(camera) else {
        return Err(SwitchCameraModelError::UnknownCamera { camera, count });
    };
    let (
        CameraModel::SfmtoolPinhole {
            bspline_rho_max: before,
            ..
        },
        CameraModel::SfmtoolPinhole {
            bspline_rho_max: after,
            ..
        },
    ) = (&source.model, &refit.camera.model)
    else {
        return Ok(None);
    };
    if before.to_bits() == after.to_bits() {
        return Ok(None);
    }
    let listed = image_set
        .camera_per_image_adjustments(camera)
        .radial_bspline;
    let mut images = Vec::new();
    for (index, image) in image_set.images.iter().enumerate() {
        let adjustment = &image.intrinsic_adjustment;
        if image.camera_index as usize != camera || !listed && adjustment.radial_bspline == [0.0; 2]
        {
            continue;
        }
        let refitted =
            refit_radial_correction(source, &refit.camera, adjustment, refit.theta_fit_deg)
                .map_err(|error| SwitchCameraModelError::RadialCorrection {
                    camera,
                    image: index,
                    error,
                })?;
        images.push((index, refitted));
    }
    let Some(&(max_image, worst)) = images
        .iter()
        .max_by(|a, b| a.1.max_px.total_cmp(&b.1.max_px))
    else {
        return Ok(None);
    };
    // Every image samples the same angles, so the mean of the images' mean
    // squares is the mean square over every sample.
    let mean_square =
        images.iter().map(|(_, r)| r.rms_px * r.rms_px).sum::<f64>() / images.len() as f64;
    let monotone_constrained_images = images
        .iter()
        .filter(|(_, r)| r.monotone_constraint.active)
        .count();
    Ok(Some(RefittedRadialCorrections {
        rms_px: mean_square.sqrt(),
        max_px: worst.max_px,
        max_image,
        monotone_constrained_images,
        images,
    }))
}

/// The fit angle a switch of camera `camera` of `recon` takes when none is
/// given, and where it comes from: the camera's trusted bound, or for a camera
/// without one the widest incidence angle among the observations of its
/// images, or the far image corner when it has none.
///
/// The angle [`fit_camera_switch`] and [`switch_camera_model`] fit over with
/// [`RefitOptions::theta_fit_deg`] `None`, whatever the target, except a spline
/// camera refitted as its own model, which is fitted over its whole domain. It
/// is reported whether or not the fit at that angle succeeds, so a caller can
/// show the default beside a refused fit.
///
/// # Errors
///
/// [`SwitchCameraModelError::UnknownCamera`] for an index past the camera
/// table, and [`SwitchCameraModelError::Refit`] for a camera with no image to
/// take a corner from.
pub fn default_fit_angle(
    recon: &SfmrReconstruction,
    camera: usize,
) -> Result<(f64, ThetaFitSource), SwitchCameraModelError> {
    let count = recon.image_set.cameras.len();
    let Some(source) = recon.image_set.cameras.get(camera) else {
        return Err(SwitchCameraModelError::UnknownCamera { camera, count });
    };
    let max_theta_deg = widest_ray(recon, camera).map_or(f64::NAN, |(_, t)| t);
    fit_angle_default(source, max_theta_deg, camera)
}

/// The widest ray among the observations of camera `camera`'s images, and the
/// row it belongs to, so a refusal can name the point: the angle is measured
/// to the point's triangulated position, not through the lens to its keypoint,
/// and a badly placed point can reach far past any keypoint the camera
/// detected.
fn widest_ray(recon: &SfmrReconstruction, camera: usize) -> Option<(usize, f64)> {
    let image_set = &recon.image_set;
    let mut widest: Option<(usize, f64)> = None;
    for (row, observation) in recon.point_set.tracks.iter().enumerate() {
        if image_set.images[observation.image_index as usize].camera_index as usize != camera {
            continue;
        }
        let t = incidence_deg(recon, row);
        if t.is_finite() && widest.is_none_or(|(_, best)| t > best) {
            widest = Some((row, t));
        }
    }
    widest
}

/// [`default_fit_angle`] for a source whose widest observed ray is
/// `max_theta_deg` (`NaN` with none).
fn fit_angle_default(
    source: &CameraIntrinsics,
    max_theta_deg: f64,
    camera: usize,
) -> Result<(f64, ThetaFitSource), SwitchCameraModelError> {
    Ok(match trustworthy_max_theta_deg(source) {
        Some(bound) => (bound, ThetaFitSource::TrustedBound),
        None if max_theta_deg > 0.0 => (max_theta_deg.min(180.0), ThetaFitSource::Observations),
        None => (
            image_corner_deg(source)
                .map_err(|error| SwitchCameraModelError::Refit { camera, error })?,
            ThetaFitSource::ImageCorner,
        ),
    })
}

/// The coefficient count `target` asks of `source` when `source` is a spline
/// camera and `target` is its own spline model, which makes the switch a
/// refit of its spline; `None` for a switch between models. With no count
/// stated the refit keeps the source's.
fn spline_refit_count(source: &CameraIntrinsics, target: &RefitTarget) -> Option<usize> {
    let (_, _, radial) = source.model.radial_spline()?;
    match (radial, target) {
        (SplineRadial::IncidenceAngle, RefitTarget::SfmtoolFisheye { .. })
        | (SplineRadial::ImagePlaneRadius, RefitTarget::SfmtoolPinhole { .. }) => {
            target.coeff_count_for(source)
        }
        _ => None,
    }
}

/// One observation of a switched camera's image: its incidence angle and its
/// error under each model, `None` where that model gave it no pixel.
struct Row {
    theta_deg: f64,
    before: Option<f64>,
    after: Option<f64>,
}

/// The observed pixel of each row in `rows`, indexed by row over the whole
/// track table (`None` for a row not asked for): from the inline keypoint
/// column where the value carries one, otherwise from each image's `.sift`
/// file, read once per image.
fn observation_pixels(
    recon: &SfmrReconstruction,
    rows: &[usize],
) -> Result<Vec<Option<[f64; 2]>>, SwitchCameraModelError> {
    let tracks = &recon.point_set.tracks;
    let mut out = vec![None; tracks.len()];
    if let Some(xy) = recon.keypoints_xy() {
        for &row in rows {
            out[row] = Some([f64::from(xy[[row, 0]]), f64::from(xy[[row, 1]])]);
        }
        return Ok(out);
    }
    let Some(features) = recon.feature_indexes() else {
        return Err(SwitchCameraModelError::Observations(
            "the observations carry neither inline keypoints nor feature indexes".to_string(),
        ));
    };
    let mut by_image: HashMap<usize, Vec<usize>> = HashMap::new();
    for &row in rows {
        by_image
            .entry(tracks[row].image_index as usize)
            .or_default()
            .push(row);
    }
    for (image, image_rows) in by_image {
        let read_count = image_rows
            .iter()
            .map(|&row| features[row] as usize + 1)
            .max()
            .unwrap_or(0);
        let path = recon.sift_path_for_image(image);
        let positions =
            sfmtool_sift_format::read_sift_positions(&path, read_count).map_err(|e| {
                SwitchCameraModelError::Observations(format!("{}: {e}", path.display()))
            })?;
        for row in image_rows {
            out[row] = positions
                .get(features[row] as usize)
                .map(|p| [f64::from(p[0]), f64::from(p[1])]);
        }
    }
    Ok(out)
}

/// Row `row`'s reprojection error through its image's intrinsics in `recon`,
/// read from `intrinsics` (the image set's `all_image_intrinsics`), or `None`
/// where there is no pixel to measure against or the model gives the point
/// none.
fn row_error(
    recon: &SfmrReconstruction,
    intrinsics: &[Cow<'_, CameraIntrinsics>],
    row: usize,
    pixel: Option<[f64; 2]>,
) -> Option<f64> {
    let observation = &recon.point_set.tracks[row];
    let image = &recon.image_set.images[observation.image_index as usize];
    let point = &recon.point_set.points[observation.point_index as usize];
    observation_reprojection_error(
        &image.quaternion_wxyz,
        &image.translation_xyz,
        &intrinsics[observation.image_index as usize],
        &point.position,
        point.is_at_infinity(),
        pixel?,
    )
    .filter(|e| e.is_finite())
}

/// The angle in degrees between row `row`'s camera axis and its point's
/// direction, from the pose and the point alone, so it does not depend on the
/// model being replaced. `NaN` for an unposed image.
fn incidence_deg(recon: &SfmrReconstruction, row: usize) -> f64 {
    let observation = &recon.point_set.tracks[row];
    let image = &recon.image_set.images[observation.image_index as usize];
    let p = &recon.point_set.points[observation.point_index as usize];
    let camera_frame = if p.is_at_infinity() {
        image.quaternion_wxyz * p.position.coords
    } else {
        image.quaternion_wxyz * p.position.coords + image.translation_xyz
    };
    // The camera looks along −Z.
    let lateral = camera_frame.x.hypot(camera_frame.y);
    lateral.atan2(-camera_frame.z).to_degrees()
}

/// The before/after comparison over the rows with an error under both models.
fn compare(rows: &[Row], trusted_deg: Option<f64>) -> ObservationComparison {
    let mut b = Vec::new();
    let mut a = Vec::new();
    let mut past_b = Vec::new();
    let mut past_a = Vec::new();
    let mut unmeasured = 0usize;
    let mut changed = 0usize;
    let mut max_theta: f64 = f64::NAN;
    for row in rows {
        let (Some(before), Some(after)) = (row.before, row.after) else {
            unmeasured += 1;
            continue;
        };
        b.push(before);
        a.push(after);
        if (after - before).abs() > 1.0 {
            changed += 1;
        }
        if row.theta_deg.is_finite() {
            max_theta = max_theta.max(row.theta_deg);
        }
        if trusted_deg.is_some_and(|bound| row.theta_deg > bound) {
            past_b.push(before);
            past_a.push(after);
        }
    }
    ObservationComparison {
        observations: b.len(),
        unmeasured,
        max_theta_deg: max_theta,
        before: ErrorSummary::of(&b),
        after: ErrorSummary::of(&a),
        changed_over_1px: changed,
        trusted_deg,
        past_trusted: past_b.len(),
        past_trusted_before: ErrorSummary::of(&past_b),
        past_trusted_after: ErrorSummary::of(&past_a),
    }
}

#[cfg(test)]
mod tests;
