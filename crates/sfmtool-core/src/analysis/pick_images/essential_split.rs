// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The essential split of a frame pair: the camera's rotation between the two
//! frames and the parallax of their matches once that rotation is taken out.
//! See `specs/core/analysis/frame-pair-matching.md` § "The essential split".
//!
//! Precision: `f64` throughout. The rays come from the camera's pixel-to-ray
//! map, which is `f64`, and the parallax of a distant point is a small
//! fraction of a degree, read as the angle between two nearly equal unit
//! vectors.

use nalgebra::Vector3;
use serde::Serialize;

use crate::camera::CameraIntrinsics;
use crate::geometry::relative_pose::{
    decompose_essential_rays, estimate_essential_rays, EpipolarSide, RayEssentialOptions,
    MAX_RAY_SAMPLES,
};
use crate::geometry::rotation::rotation_angle;
use crate::numeric::{quantile_in_place, quantile_of_sorted};

use super::frame_pair::FramePairError;

#[cfg(test)]
mod tests;

/// The fewest matches an essential matrix is fitted to: the linear
/// eight-point constraint's minimal sample.
const MIN_SAMPLE: usize = 8;

/// How [`measure_essential_split`] measures, and the camera it measures with.
#[derive(Clone, Debug, PartialEq)]
pub struct EssentialSplitOptions {
    /// The camera both frames were taken with, at the frames' image size. Its
    /// pixel-to-ray map turns each matched position into a ray.
    pub camera: CameraIntrinsics,
    /// The consensus bound, in pixels at the camera's focal: a match is an
    /// inlier when its rays are within `tolerance_px / f` radians of their
    /// epipolar planes, `f` the mean of the camera's two focal lengths.
    /// Positive and finite.
    pub tolerance_px: f64,
    /// The fewest inliers a split is measured from, and so the fewest
    /// matches. At least 8.
    pub min_inliers: usize,
    /// The minimal samples the consensus draws, from 1 to
    /// [`MAX_RAY_SAMPLES`] (65,536).
    pub samples: usize,
    /// The seed of the sampler.
    pub seed: u64,
}

impl EssentialSplitOptions {
    /// The options for `camera`, with a 3-pixel bound, 12 inliers at least,
    /// 512 samples and seed 0.
    pub fn new(camera: CameraIntrinsics) -> Self {
        Self {
            camera,
            tolerance_px: 3.0,
            min_inliers: 12,
            samples: 512,
            seed: 0,
        }
    }

    /// The camera's focal in pixels, the mean of its two focal lengths.
    fn focal(&self) -> f64 {
        let (fx, fy) = self.camera.focal_lengths();
        0.5 * (fx + fy)
    }

    /// The checks [`measure_essential_split`] makes on the options. `name`s
    /// are prefixed `essential_split.`, the field of [`super::FramePairOptions`]
    /// that holds them.
    pub(super) fn validate(&self) -> Result<(), FramePairError> {
        let refuse = |name: &'static str, reason: String| FramePairError::Options { name, reason };
        let (fx, fy) = self.camera.focal_lengths();
        if !(fx.is_finite() && fx > 0.0 && fy.is_finite() && fy > 0.0) {
            return Err(refuse(
                "essential_split.camera",
                format!("needs positive finite focal lengths, got {fx} and {fy}"),
            ));
        }
        let (cx, cy) = self.camera.principal_point();
        if !(cx.is_finite() && cy.is_finite()) {
            return Err(refuse(
                "essential_split.camera",
                format!("needs a finite principal point, got ({cx}, {cy})"),
            ));
        }
        self.camera
            .check_rolling_shutter()
            .map_err(|reason| refuse("essential_split.camera", reason))?;
        if !(self.tolerance_px.is_finite() && self.tolerance_px > 0.0) {
            return Err(refuse(
                "essential_split.tolerance_px",
                format!("must be positive and finite, got {}", self.tolerance_px),
            ));
        }
        if self.min_inliers < MIN_SAMPLE {
            return Err(refuse(
                "essential_split.min_inliers",
                format!("must be at least {MIN_SAMPLE}, got {}", self.min_inliers),
            ));
        }
        if !(1..=MAX_RAY_SAMPLES).contains(&self.samples) {
            return Err(refuse(
                "essential_split.samples",
                format!("must be from 1 to {MAX_RAY_SAMPLES}, got {}", self.samples),
            ));
        }
        Ok(())
    }
}

/// A measured essential split.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EssentialSplit {
    /// The matches considered: those whose two positions both map to a ray.
    pub matches: usize,
    /// The essential matrix's inliers among them, which the parallax is
    /// measured over.
    pub inliers: usize,
    /// The angle of the rotation between the two frames, in degrees.
    pub rotation_deg: f64,
    /// The median parallax of the inliers, in degrees: the angle between a
    /// match's first ray rotated into the second frame and its second ray,
    /// which is the triangulation angle of its point.
    pub parallax_p50_deg: f64,
    /// The 0.9 quantile of the inliers' parallax, in degrees.
    pub parallax_p90_deg: f64,
}

/// Why no essential split was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitNotMeasured {
    /// Fewer matches map to rays than
    /// [`EssentialSplitOptions::min_inliers`].
    TooFewMatches,
    /// No essential matrix has `min_inliers` inliers.
    NoConsensus,
    /// None of the four poses the essential matrix decomposes into puts an
    /// inlier in front of both cameras.
    NoPoseInFront,
}

/// The essential split of a pair, or why there is none.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EssentialSplitOutcome {
    Measured(EssentialSplit),
    NotMeasured {
        /// The matches considered, as [`EssentialSplit::matches`].
        matches: usize,
        reason: SplitNotMeasured,
    },
}

/// Measure the essential split of matched positions: `points_a[i]` in the
/// first frame matches `points_b[i]` in the second, both in pixels of
/// `options.camera`.
///
/// Each position is turned into a ray by the camera's `pixel_to_ray`; a match
/// either of whose positions gives no finite ray is not considered.
/// [`estimate_essential_rays`] fits the essential matrix with the angular
/// bound `tolerance_px / f`, and [`decompose_essential_rays`] keeps the pose
/// that puts most inliers in front of both cameras. The rotation angle is that
/// pose's; each inlier's parallax is the angle between `R·ray_a` and `ray_b`.
///
/// Too few matches, no consensus and no pose in front are outcomes, not
/// errors, so a caller measuring many pairs gets an answer for each.
///
/// # Errors
///
/// [`FramePairError::Options`] for an option out of range, and
/// [`FramePairError::FeatureArrays`] when the two slices differ in length.
pub fn measure_essential_split(
    points_a: &[[f64; 2]],
    points_b: &[[f64; 2]],
    options: &EssentialSplitOptions,
) -> Result<EssentialSplitOutcome, FramePairError> {
    options.validate()?;
    if points_a.len() != points_b.len() {
        return Err(FramePairError::FeatureArrays {
            reason: format!(
                "{} positions in the first frame match {} in the second",
                points_a.len(),
                points_b.len()
            ),
        });
    }
    let rays_a = options.camera.pixel_to_ray_batch(points_a);
    let rays_b = options.camera.pixel_to_ray_batch(points_b);
    let (r1, r2): (Vec<Vector3<f64>>, Vec<Vector3<f64>>) = rays_a
        .iter()
        .zip(&rays_b)
        .filter_map(|(a, b)| Some((unit(a)?, unit(b)?)))
        .unzip();
    let matches = r1.len();
    let not_measured = |reason| Ok(EssentialSplitOutcome::NotMeasured { matches, reason });
    if matches < options.min_inliers {
        return not_measured(SplitNotMeasured::TooFewMatches);
    }
    let essential = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: options.tolerance_px / options.focal(),
            min_inliers: options.min_inliers,
            samples: options.samples,
            seed: options.seed,
            side: EpipolarSide::Both,
        },
    );
    let Some(essential) = essential else {
        return not_measured(SplitNotMeasured::NoConsensus);
    };
    let Some(pose) = decompose_essential_rays(&essential.e_matrix, &r1, &r2, &essential.inliers)
    else {
        return not_measured(SplitNotMeasured::NoPoseInFront);
    };
    let mut parallax: Vec<f64> = (0..matches)
        .filter(|&i| essential.inliers[i])
        .map(|i| {
            let turned = pose.rotation * r1[i];
            turned
                .cross(&r2[i])
                .norm()
                .atan2(turned.dot(&r2[i]))
                .to_degrees()
        })
        .collect();
    let parallax_p90_deg = quantile_in_place(&mut parallax, 0.9);
    Ok(EssentialSplitOutcome::Measured(EssentialSplit {
        matches,
        inliers: parallax.len(),
        rotation_deg: rotation_angle(&pose.rotation).to_degrees(),
        parallax_p50_deg: quantile_of_sorted(&parallax, 0.5),
        parallax_p90_deg,
    }))
}

/// `ray` as a unit vector, or `None` when it is not finite or has no length.
fn unit(ray: &[f64; 3]) -> Option<Vector3<f64>> {
    let v = Vector3::from(*ray);
    if !v.iter().all(|x| x.is_finite()) {
        return None;
    }
    v.try_normalize(0.0)
}
