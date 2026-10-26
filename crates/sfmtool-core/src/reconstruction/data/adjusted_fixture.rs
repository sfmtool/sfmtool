// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A test helper for the consumers of an image's intrinsics: give a fixture's
//! images non-zero intrinsic adjustments and make its inline keypoints again
//! through them, so that a consumer reading the image's intrinsics sees exact
//! observations and one reading the shared camera does not.

use crate::camera::IntrinsicAdjustment;

use super::{ObservationSource, SfmrReconstruction};

/// Give image `i` of `recon` the adjustment `adjustment_of(i)`, and move every
/// inline keypoint to its point's projection through its image's intrinsics: a
/// finite point at `R·X + t`, a point at infinity at `R·d`. A keypoint whose
/// point does not project keeps its value.
///
/// # Panics
///
/// Panics when `recon` carries no inline keypoints.
pub(crate) fn adjust_and_reproject(
    recon: &mut SfmrReconstruction,
    adjustment_of: impl Fn(usize) -> IntrinsicAdjustment,
) {
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.intrinsic_adjustment = adjustment_of(i);
    }
    let intrinsics: Vec<_> = recon
        .image_set
        .all_image_intrinsics()
        .into_iter()
        .map(|c| c.into_owned())
        .collect();
    let images = recon.image_set.images.clone();
    let points = recon.point_set.points.clone();
    let tracks = recon.point_set.tracks.clone();
    let keypoints = match &mut recon.point_set.observations {
        ObservationSource::EmbeddedPatches { keypoints_xy, .. } => keypoints_xy,
        ObservationSource::SiftFiles { keypoints_xy, .. } => keypoints_xy
            .as_mut()
            .expect("the fixture has inline keypoints"),
    };
    for (row, observation) in tracks.iter().enumerate() {
        let i = observation.image_index as usize;
        let image = &images[i];
        let point = &points[observation.point_index as usize];
        let rotated = image.quaternion_wxyz * point.position.coords;
        let local = if point.is_at_infinity() {
            rotated
        } else {
            rotated + image.translation_xyz
        };
        if let Some((u, v)) = intrinsics[i].ray_to_pixel([local.x, local.y, local.z]) {
            keypoints[[row, 0]] = u as f32;
            keypoints[[row, 1]] = v as f32;
        }
    }
}

/// A small adjustment distinct for each image: a focal scale within ±1.5% and
/// a principal-point offset within ±1% of the short side on each axis, the
/// size the evidence of `specs/drafts/per-image-intrinsics.md` shows.
pub(crate) fn varied_adjustment(i: usize) -> IntrinsicAdjustment {
    let t = i as f64;
    IntrinsicAdjustment {
        focal: 0.015 * (0.9 * t + 0.3).sin(),
        principal_point: [0.01 * (1.3 * t + 0.5).cos(), 0.01 * (0.7 * t + 1.1).sin()],
    }
}
