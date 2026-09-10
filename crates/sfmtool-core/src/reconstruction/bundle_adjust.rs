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
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::bundle_adjust::{
    BaSchedule, DistanceReference, FreePointPolicy, PointConstraints, PointConstraintsError,
    DEFAULT_PROTECTED_LOSS_SCALE, DEFAULT_SCHEDULE,
};
use crate::numeric::median_in_place;
use sfmr_format::{NO_REFERENCE_IMAGE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED};

/// LM iteration budget per round, the kernel's own default.
const DEFAULT_MAX_ITERS: usize = 60;
/// Trim survivors a point needs to stay in the solve, the kernel's own default.
const DEFAULT_MIN_TRACK: usize = 2;
/// Trim survivors below which a round exits degenerate, the kernel's own
/// default.
const DEFAULT_MIN_OBS: usize = 12;

/// What the adjustment is allowed to move, and how hard it tries.
///
/// The defaults are the kernel's own, so an adjustment asked for with nothing
/// stated is the one every other caller in this crate runs.
#[derive(Debug, Clone)]
pub struct BundleAdjustOptions {
    /// Release the shared focal length. Off by default, because a focal that
    /// moves is a different claim about the capture than a pose that does, and
    /// the caller should be the one making it.
    ///
    /// Only the models the kernel's analytic focal column is exact for accept
    /// it -- `SIMPLE_PINHOLE`, `EQUIDISTANT_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`,
    /// `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`. On any other model this is
    /// refused rather than silently ignored: a caller that asked for a focal
    /// solve and got a fixed-focal one back would have no way to tell.
    pub opt_f: bool,
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
    /// The posed images do not share one set of camera intrinsics.
    MixedCameras {
        /// How many the posed images between them name.
        cameras: usize,
    },
    /// The focal release was asked for on a model the kernel's focal column is
    /// not exact for.
    FocalNotReleasable(&'static str),
    /// No image of the reconstruction carries a usable pose.
    NoPosedImages,
    /// No observation of a live point falls in a posed image.
    NoObservations,
    /// The schedule has no rounds, so there is nothing to run.
    EmptySchedule,
    /// The value's constraint columns state something the adjustment cannot
    /// honour.
    Constraints(PointConstraintsError),
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
            BundleAdjustError::MixedCameras { cameras } => write!(
                f,
                "the adjustment solves one shared camera, and these images are taken \
                 through {cameras}"
            ),
            BundleAdjustError::FocalNotReleasable(model) => write!(
                f,
                "the focal cannot be released on a {model} camera; the adjustment's focal \
                 column is exact for SIMPLE_PINHOLE, EQUIDISTANT_FISHEYE, \
                 SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE and SFMTOOL_PINHOLE"
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
    /// The shared focal before the solve.
    pub focal_before: f64,
    /// The shared focal after it, which is the one before it unless the focal
    /// was released.
    pub focal_after: f64,
    /// Whether the focal was released.
    pub focal_released: bool,
}

/// Bundle-adjust `recon`, returning the adjusted value and a report.
///
/// Every posed image's pose, every live point's position and, under
/// [`BundleAdjustOptions::opt_f`], the shared focal are refined together against
/// every observation that carries a pixel, by the staged robust solve in
/// [`crate::geometry::bundle_adjust()`]. A point at infinity goes in as the
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
/// # Example
///
/// ```no_run
/// use sfmtool_core::reconstruction::bundle_adjust::{bundle_adjust, BundleAdjustOptions};
/// # fn run(recon: &sfmtool_core::SfmrReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let (next, report) = bundle_adjust(recon, &BundleAdjustOptions::default())?;
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

    // One camera for the whole solve: the kernel carries a single shared model,
    // and a value whose images disagree about the lens has to be told so rather
    // than silently adjusted through one of them.
    let camera_index = table.images[posed[0]].camera_index;
    let mut lenses: Vec<u32> = posed
        .iter()
        .map(|&i| table.images[i].camera_index)
        .collect();
    lenses.sort_unstable();
    lenses.dedup();
    if lenses.len() != 1 {
        return Err(BundleAdjustError::MixedCameras {
            cameras: lenses.len(),
        });
    }
    let camera = &table.cameras[camera_index as usize];
    if options.opt_f && !focal_is_releasable(camera) {
        return Err(BundleAdjustError::FocalNotReleasable(camera.model_name()));
    }

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

    // ── The two solves ────────────────────────────────────────────────────
    //
    // The first is the kernel over an **empty** schedule, which runs no round
    // and reports the residuals at the state it was handed: the "before" median
    // is then measured by the same projection the "after" one is, rather than by
    // a second spelling of it here.
    let before = crate::geometry::bundle_adjust::bundle_adjust(
        camera,
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
    );

    let solved = crate::geometry::bundle_adjust::bundle_adjust(
        camera,
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
        false,
        false,
        &options.schedule,
        options.max_iters,
        options.min_track,
        options.min_obs,
    );
    // Every residual infinite is the kernel's degenerate exit, which passes the
    // state through: read as a per-point verdict it would delete the whole
    // point set, so it is the call's refusal instead.
    if solved.residual_norms.iter().all(|r| !r.is_finite()) {
        return Err(BundleAdjustError::Degenerate {
            observations: uv.len(),
            min_obs: options.min_obs,
        });
    }

    // ── Writing the answer back ───────────────────────────────────────────
    let mut out = recon.clone();
    for (slot, &i) in posed.iter().enumerate() {
        out.image_table.images[i].quaternion_wxyz = quats[slot];
        out.image_table.images[i].translation_xyz = trans[slot];
    }
    let focal_before = camera.focal_lengths().0;
    if options.opt_f {
        out.image_table.cameras[camera_index as usize] = camera.with_focal(solved.focal);
    }

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
        points: in_solve.iter().filter(|&&s| s).count(),
        observations: uv.len(),
        points_deleted,
        median_residual_before: median_residual(&before.residual_norms, &obs_pt, &keep),
        median_residual_after: median_residual(&solved.residual_norms, &obs_pt, &keep),
        focal_before,
        focal_after: if options.opt_f {
            solved.focal
        } else {
            focal_before
        },
        focal_released: options.opt_f,
    };
    Ok((out, report))
}

/// Whether a stored pose is one at all. Every `.sfmr` image row has the fields;
/// a non-finite one is a placeholder rather than a registration.
fn is_posed(rotation: &UnitQuaternion<f64>, translation: &Vector3<f64>) -> bool {
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
fn rescale_patch_frame(
    out: &mut SfmrReconstruction,
    p: usize,
    before: Option<f64>,
    after: Option<&Point3<f64>>,
) {
    let usable = |d: f64| (d.is_finite() && d > 0.0).then_some(d);
    let before = match before {
        Some(d) => match usable(d) {
            Some(d) => d,
            None => return,
        },
        None => 1.0,
    };
    let after = match after {
        Some(position) => match usable(out.image_table.placement_scale(position)) {
            Some(d) => d,
            None => return,
        },
        None => 1.0,
    };
    let factor = (after / before) as f32;
    if !factor.is_finite() || factor == 1.0 {
        return;
    }
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

#[cfg(test)]
mod tests;
