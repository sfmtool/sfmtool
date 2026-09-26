// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Switch cameras of a reconstruction to another camera model.
//!
//! `specs/core/reconstruction/switch-camera-model.md` is the design. The
//! function here is pure: an [`SfmrReconstruction`] goes in, a new one and a
//! report come out. Each chosen camera is replaced by the camera
//! [`refit_camera_intrinsics`](crate::camera::refit_intrinsics::refit_camera_intrinsics) fits to it, and every
//! observation of an image that uses the camera is measured before and after,
//! on the same set of observations. Poses, points, keypoints, patches and
//! tracks are not touched; the stored per-point errors of the points those
//! images observe are recomputed.

use std::collections::HashMap;
use std::fmt;

use super::data::{observation_reprojection_error, SfmrReconstruction};
use super::outermost_keypoint::{outermost_keypoints, OutermostKeypoints};
use crate::camera::intrinsics::SplineRadial;
use crate::camera::refit_intrinsics::{
    image_corner_deg, refit_camera_intrinsics_over, refit_spline, CameraIntrinsicsRefit,
    RefitError, RefitOptions, RefitTarget, ThetaFitSource,
};
use crate::camera::report::trustworthy_max_theta_deg;
use crate::camera::CameraIntrinsics;
use crate::numeric::quantile_of_sorted;

/// Why a switch produced nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchCameraModelError {
    /// No camera was named.
    NoCameras,
    /// A camera index past the camera table.
    UnknownCamera {
        /// The index given.
        camera: usize,
        /// The number of cameras the reconstruction has.
        count: usize,
    },
    /// The fit refused one camera.
    Refit {
        /// The camera's index in the reconstruction's camera table.
        camera: usize,
        /// The fit's refusal.
        error: RefitError,
    },
    /// The observation pixels could not be read: a `sift_files`
    /// reconstruction without inline keypoints whose `.sift` files are missing.
    Observations(String),
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
        }
    }
}

impl std::error::Error for SwitchCameraModelError {}

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

/// What the switch did to one camera.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraSwitch {
    /// The camera's index in the reconstruction's camera table.
    pub camera: usize,
    /// The camera before the switch.
    pub source: CameraIntrinsics,
    /// The fit, carrying the camera after the switch.
    pub refit: CameraIntrinsicsRefit,
    /// Posed or not, the images that use the camera.
    pub images: usize,
    /// The observation comparison.
    pub observations: ObservationComparison,
    /// The outermost keypoint of the camera's images, observed and detected,
    /// measured under the camera after the switch: how far out the
    /// photographs reach, beside the spline domain the fit placed.
    pub outermost: OutermostKeypoints,
}

/// What one switch did, one entry per switched camera in table order.
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchCameraModelReport {
    /// One entry per switched camera.
    pub cameras: Vec<CameraSwitch>,
}

/// Switch `cameras` of `recon` to `target`, returning the switched value and a
/// report.
///
/// Each camera is fitted independently by
/// [`refit_camera_intrinsics`](crate::camera::refit_intrinsics::refit_camera_intrinsics). When
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
/// source, over that angle. A perspective target is refused for a camera observed at
/// 90° or more. The first camera refused refuses the whole switch, and nothing
/// changes.
///
/// Poses, points, keypoints, patches and tracks are carried over unchanged. The
/// stored error of every point observed in an image that uses a switched
/// camera is recomputed as the mean of its reprojection errors, the convention
/// [`SfmrReconstruction::recompute_point_errors`] writes.
///
/// The observations' pixels come from the inline keypoint column where the
/// value carries one, and otherwise from each image's `.sift` file.
///
/// Each camera's entry also carries its outermost keypoint
/// ([`outermost_keypoints`]), observed and, where the images' `.sift` files can
/// be read, detected, measured under the switched camera. A file that cannot
/// be read leaves `detected` empty rather than refusing the switch.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::camera::refit_intrinsics::{RefitOptions, RefitTarget};
/// use sfmtool_core::reconstruction::switch_camera_model::switch_camera_model;
/// # fn run(recon: &sfmtool_core::SfmrReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8))?;
/// let (switched, report) = switch_camera_model(recon, &[0], &target, &RefitOptions::default())?;
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
    options: &RefitOptions,
) -> Result<(SfmrReconstruction, SwitchCameraModelReport), SwitchCameraModelError> {
    let count = recon.image_table.cameras.len();
    let mut chosen: Vec<usize> = cameras.to_vec();
    chosen.sort_unstable();
    chosen.dedup();
    if chosen.is_empty() {
        return Err(SwitchCameraModelError::NoCameras);
    }
    if let Some(&camera) = chosen.iter().find(|&&c| c >= count) {
        return Err(SwitchCameraModelError::UnknownCamera { camera, count });
    }

    let table = &recon.image_table;
    let tracks = &recon.point_set.tracks;
    let camera_of_row = |row: usize| table.images[tracks[row].image_index as usize].camera_index;
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
    let pixels = observation_pixels(recon, &touched_rows)?;

    // ── The fits ────────────────────────────────────────────────────────────
    let theta: Vec<f64> = switched_rows
        .iter()
        .map(|&row| incidence_deg(recon, row))
        .collect();
    let mut refits: Vec<CameraIntrinsicsRefit> = Vec::with_capacity(chosen.len());
    for &c in &chosen {
        let source = &table.cameras[c];
        let max_theta_deg = switched_rows
            .iter()
            .zip(&theta)
            .filter(|(&row, t)| camera_of_row(row) as usize == c && t.is_finite())
            .map(|(_, &t)| t)
            .fold(f64::NAN, f64::max);
        if target.is_perspective() && max_theta_deg >= 90.0 {
            return Err(SwitchCameraModelError::Refit {
                camera: c,
                error: RefitError::ObservationsPast90 { max_theta_deg },
            });
        }
        let (theta_fit_deg, theta_fit_source) = match options.theta_fit_deg {
            Some(theta) => (theta, ThetaFitSource::Given),
            None => match trustworthy_max_theta_deg(source) {
                Some(bound) => (bound, ThetaFitSource::TrustedBound),
                None if max_theta_deg > 0.0 => {
                    (max_theta_deg.min(180.0), ThetaFitSource::Observations)
                }
                None => (
                    image_corner_deg(source)
                        .map_err(|error| SwitchCameraModelError::Refit { camera: c, error })?,
                    ThetaFitSource::ImageCorner,
                ),
            },
        };
        let refit = match (spline_refit_count(source, target), options.theta_fit_deg) {
            (Some(count), None) => refit_spline(source, count, options.spline_domain_deg),
            _ => refit_camera_intrinsics_over(
                source,
                target,
                theta_fit_deg,
                theta_fit_source,
                options.spline_domain_deg,
            ),
        }
        .map_err(|error| SwitchCameraModelError::Refit { camera: c, error })?;
        refits.push(refit);
    }

    // ── The switched value ─────────────────────────────────────────────────
    let mut out = recon.clone_for_edit();
    for (&c, refit) in chosen.iter().zip(&refits) {
        out.image_table.cameras[c] = refit.camera.clone();
    }

    // Each touched point's stored error, over every one of its observations
    // under the cameras the value now holds.
    let mut sums = vec![0.0f64; n_pt];
    let mut counts = vec![0u32; n_pt];
    for &row in &touched_rows {
        if let Some(e) = row_error(&out, row, pixels[row]) {
            let p = tracks[row].point_index as usize;
            sums[p] += e;
            counts[p] += 1;
        }
    }
    for p in 0..n_pt {
        if touched[p] {
            out.point_set.points[p].error = if counts[p] > 0 {
                (sums[p] / f64::from(counts[p])) as f32
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
            before: row_error(recon, row, pixels[row]),
            after: row_error(&out, row, pixels[row]),
        });
    }
    let outermost = outermost_keypoints(&out, &chosen, true);
    let mut entries = Vec::with_capacity(chosen.len());
    for (((&c, refit), rows), outermost) in chosen.iter().zip(refits).zip(rows_of).zip(outermost) {
        let images = table
            .images
            .iter()
            .filter(|image| image.camera_index as usize == c)
            .count();
        let observations = compare(&rows, refit.extent.source_trusted_deg);
        entries.push(CameraSwitch {
            camera: c,
            source: table.cameras[c].clone(),
            refit,
            images,
            observations,
            outermost,
        });
    }
    Ok((out, SwitchCameraModelReport { cameras: entries }))
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

/// Row `row`'s reprojection error under the cameras `recon` holds, or `None`
/// where there is no pixel to measure against or the model gives the point
/// none.
fn row_error(recon: &SfmrReconstruction, row: usize, pixel: Option<[f64; 2]>) -> Option<f64> {
    let observation = &recon.point_set.tracks[row];
    let image = &recon.image_table.images[observation.image_index as usize];
    let point = &recon.point_set.points[observation.point_index as usize];
    observation_reprojection_error(
        &image.quaternion_wxyz,
        &image.translation_xyz,
        &recon.image_table.cameras[image.camera_index as usize],
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
    let image = &recon.image_table.images[observation.image_index as usize];
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
