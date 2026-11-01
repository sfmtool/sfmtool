// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Drop per-image parameters from one camera of a reconstruction, folding
//! their mean into the shared camera.
//!
//! `specs/core/reconstruction/drop-image-adjustments.md` is the design. The
//! function here is pure: an [`SfmrReconstruction`] goes in and a new one comes
//! out. Only the bundle adjustment turns a per-image parameter on, since it is
//! the one that solves it; this edit only turns one off.

use std::fmt;

use super::bundle_adjust::{is_posed, ImageAdjustmentParameter};
use super::data::SfmrReconstruction;
use crate::camera::distortion::bspline::bspline_is_monotone;
use crate::camera::{CameraIntrinsics, CameraModel, IntrinsicAdjustment};

/// Why a drop produced nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum DropImageAdjustmentsError {
    /// A camera index past the image set's cameras.
    UnknownCamera {
        /// The index given.
        camera: usize,
        /// The number of cameras the reconstruction has.
        count: usize,
    },
    /// The mean of the dropped parameters cannot be folded into the camera:
    /// the camera cannot take it as an adjustment, or the folded camera's focal
    /// or principal point is not finite, or its focal is not positive.
    Fold {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
        /// What is wrong.
        message: String,
    },
    /// The camera's spline with the mean radial correction folded in is not
    /// monotone, so its inverse is not defined everywhere.
    FoldedSplineNotMonotone {
        /// The camera's index among the reconstruction's cameras.
        camera: usize,
    },
    /// An image of the result cannot be read through its intrinsics
    /// ([`ImageSet::check_image_intrinsics`](super::ImageSet::check_image_intrinsics)).
    Intrinsics(String),
}

impl fmt::Display for DropImageAdjustmentsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DropImageAdjustmentsError::UnknownCamera { camera, count } => write!(
                f,
                "camera {camera} does not exist; the reconstruction has {count} camera(s)"
            ),
            DropImageAdjustmentsError::Fold { camera, message } => write!(
                f,
                "camera {camera}: the mean of the dropped per-image values cannot be \
                 folded into the camera: {message}"
            ),
            DropImageAdjustmentsError::FoldedSplineNotMonotone { camera } => write!(
                f,
                "camera {camera}: the spline with the mean radial correction folded in \
                 is not monotone"
            ),
            DropImageAdjustmentsError::Intrinsics(e) => {
                write!(f, "the result has an image its camera cannot read: {e}")
            }
        }
    }
}

impl std::error::Error for DropImageAdjustmentsError {}

/// Drop `parameters` from camera `camera` of `recon`: the camera stops varying
/// them per image, and their mean over the camera's images is folded into the
/// camera, so that an image whose value was the mean is read exactly as
/// before.
///
/// For each parameter named:
///
/// - **focal**: the camera's focal (both focals on a two-focal model) is
///   multiplied by the mean of `1 + s`;
/// - **principal point**: the camera's principal point moves by the mean
///   offset times the short side, `min(width, height)`;
/// - **radial correction**: the camera's spline coefficients `c` become
///   `c + K·ē`, or `ē` on an empty spline, and the result must be monotone;
/// - **rolling shutter**: nothing is folded, since a mean rate describes no
///   photograph. The camera's scanlines setting is kept: it states how the
///   sensor reads out, which does not change when the rates are dropped.
///
/// The fold is [`CameraIntrinsics::with_adjustment`] with the mean adjustment,
/// so it is exactly what an image with that adjustment was read through. The
/// mean is over the camera's posed images, the ones whose observations the
/// edit changes, or over all its images where none is posed. Every image of
/// the camera then has those parts of its adjustment set to zero, and the
/// camera's flags for them are cleared.
///
/// A parameter the camera does not vary, and whose values on its images are
/// all zero, is left as it is, so dropping it changes nothing; an empty
/// `parameters` is the reconstruction unchanged. Other cameras and their
/// images, the poses and the points are not touched: a caller that wants the
/// poses and points to follow the new lens refits or re-triangulates. The
/// stored error of every point the camera's images observe is recomputed from
/// the inline keypoints where the value carries them, as
/// [`SfmrReconstruction::recompute_point_errors`] scores it; a `sift_files`
/// value without them keeps its stored errors.
///
/// # Errors
///
/// [`DropImageAdjustmentsError::UnknownCamera`] for an index past the
/// cameras; [`DropImageAdjustmentsError::Fold`] and
/// [`DropImageAdjustmentsError::FoldedSplineNotMonotone`] where the folded
/// camera is not a valid lens; and [`DropImageAdjustmentsError::Intrinsics`]
/// where an image of the result cannot be read through its intrinsics.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::reconstruction::bundle_adjust::ImageAdjustmentParameter;
/// use sfmtool_core::reconstruction::drop_image_adjustments::drop_image_adjustments;
/// # fn run(recon: &sfmtool_core::SfmrReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let dropped = drop_image_adjustments(
///     recon,
///     0,
///     &[ImageAdjustmentParameter::Focal, ImageAdjustmentParameter::RollingShutter],
/// )?;
/// assert!(!dropped.image_set.camera_per_image_adjustments(0).focal);
/// # Ok(())
/// # }
/// ```
pub fn drop_image_adjustments(
    recon: &SfmrReconstruction,
    camera: usize,
    parameters: &[ImageAdjustmentParameter],
) -> Result<SfmrReconstruction, DropImageAdjustmentsError> {
    let image_set = &recon.image_set;
    let count = image_set.cameras.len();
    let Some(source) = image_set.cameras.get(camera) else {
        return Err(DropImageAdjustmentsError::UnknownCamera { camera, count });
    };
    let has = |p: ImageAdjustmentParameter| parameters.contains(&p);
    let drop_focal = has(ImageAdjustmentParameter::Focal);
    let drop_pp = has(ImageAdjustmentParameter::PrincipalPoint);
    let drop_radial = has(ImageAdjustmentParameter::RadialBspline);
    let drop_rates = has(ImageAdjustmentParameter::RollingShutter);

    let members: Vec<usize> = (0..image_set.images.len())
        .filter(|&i| image_set.images[i].camera_index as usize == camera)
        .collect();
    let flags = image_set.camera_per_image_adjustments(camera);
    let carries = |part: fn(&IntrinsicAdjustment) -> bool| {
        members
            .iter()
            .any(|&i| part(&image_set.images[i].intrinsic_adjustment))
    };
    let changes = (drop_focal && (flags.focal || carries(|a| a.focal != 0.0)))
        || (drop_pp && (flags.principal_point || carries(|a| a.principal_point != [0.0; 2])))
        || (drop_radial && (flags.radial_bspline || carries(|a| a.radial_bspline != [0.0; 2])))
        || (drop_rates && (flags.rolling_shutter || carries(|a| a.rolling_shutter != [0.0; 3])));
    if !changes {
        // Nothing to drop: the value is the one handed in, content hashes and
        // all.
        return Ok(recon.clone());
    }

    // The mean over the posed images, whose observations the fold is for, or
    // over every image of the camera where none is posed.
    let posed: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&i| {
            let image = &image_set.images[i];
            is_posed(&image.quaternion_wxyz, &image.translation_xyz)
        })
        .collect();
    let over = if posed.is_empty() { &members } else { &posed };
    let mut mean = IntrinsicAdjustment::default();
    if !over.is_empty() {
        let n = over.len() as f64;
        let adjustments = over
            .iter()
            .map(|&i| &image_set.images[i].intrinsic_adjustment);
        for a in adjustments {
            if drop_focal {
                mean.focal += a.focal;
            }
            if drop_pp {
                mean.principal_point[0] += a.principal_point[0];
                mean.principal_point[1] += a.principal_point[1];
            }
            if drop_radial {
                mean.radial_bspline[0] += a.radial_bspline[0];
                mean.radial_bspline[1] += a.radial_bspline[1];
            }
        }
        mean.focal /= n;
        mean.principal_point = mean.principal_point.map(|v| v / n);
        mean.radial_bspline = mean.radial_bspline.map(|v| v / n);
    }
    let folded = fold(source, &mean, camera)?;

    let mut out = recon.clone_for_edit();
    out.image_set.cameras[camera] = folded;
    for &i in &members {
        let adjustment = &mut out.image_set.images[i].intrinsic_adjustment;
        if drop_focal {
            adjustment.focal = 0.0;
        }
        if drop_pp {
            adjustment.principal_point = [0.0; 2];
        }
        if drop_radial {
            adjustment.radial_bspline = [0.0; 2];
        }
        if drop_rates {
            adjustment.rolling_shutter = [0.0; 3];
        }
    }
    if let Some(entry) = out.image_set.per_image_adjustments.get_mut(camera) {
        entry.focal &= !drop_focal;
        entry.principal_point &= !drop_pp;
        entry.radial_bspline &= !drop_radial;
        entry.rolling_shutter &= !drop_rates;
    }
    // The column is empty when no camera states anything, as a value read from
    // a file without adjustments has it.
    if !out
        .image_set
        .per_image_adjustments
        .iter()
        .any(|f| f.is_set())
    {
        out.image_set.per_image_adjustments.clear();
    }
    out.image_set
        .check_image_intrinsics()
        .map_err(DropImageAdjustmentsError::Intrinsics)?;

    // Every point the camera's images observe projects through the new lens
    // there.
    let mut observed = vec![false; out.point_set.points.len()];
    for t in &out.point_set.tracks {
        if out.image_set.images[t.image_index as usize].camera_index as usize == camera {
            observed[t.point_index as usize] = true;
        }
    }
    out.rescore_points_from_inline_keypoints(&observed);
    Ok(out)
}

/// `source` with the mean adjustment `mean` applied, refused where it is not a
/// valid lens.
fn fold(
    source: &CameraIntrinsics,
    mean: &IntrinsicAdjustment,
    camera: usize,
) -> Result<CameraIntrinsics, DropImageAdjustmentsError> {
    // The mean carries no rates, so no scanlines setting is read.
    source
        .check_adjustment(mean, None)
        .map_err(|message| DropImageAdjustmentsError::Fold { camera, message })?;
    let folded = source.with_adjustment(mean, None);
    let (fx, fy) = folded.focal_lengths();
    let (cx, cy) = folded.principal_point();
    if !(fx > 0.0 && fy > 0.0 && fx.is_finite() && fy.is_finite()) {
        return Err(DropImageAdjustmentsError::Fold {
            camera,
            message: format!("the folded focal is {fx}, {fy}, not positive and finite"),
        });
    }
    if !(cx.is_finite() && cy.is_finite()) {
        return Err(DropImageAdjustmentsError::Fold {
            camera,
            message: format!("the folded principal point is ({cx}, {cy}), not finite"),
        });
    }
    if mean.radial_bspline != [0.0; 2] {
        if let CameraModel::SfmtoolPinhole {
            bspline,
            bspline_rho_max,
            ..
        } = &folded.model
        {
            let valid = bspline.iter().all(|c| c.is_finite())
                && bspline_is_monotone(bspline, *bspline_rho_max, *bspline_rho_max);
            if !valid {
                return Err(DropImageAdjustmentsError::FoldedSplineNotMonotone { camera });
            }
        }
    }
    Ok(folded)
}

#[cfg(test)]
mod tests;
