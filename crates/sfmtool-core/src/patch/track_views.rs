// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A patch cloud's **track views** read from the reconstruction it was built
//! from: per patch, the image indices its source point's track observes, and
//! the stored keypoint of each. These are the `patch_views` (and
//! `patch_keypoints`) a batch driver evaluates each patch against when no view
//! selection has run, and the `track_views` (and `track_keypoints`) view
//! selection starts from.

use crate::patch::cloud::PatchCloud;
use crate::reconstruction::SfmrReconstruction;

/// For each patch of `cloud` (linked to `recon` via `point_indexes`), the image
/// indices observing its source 3D point, in track order.
///
/// The result can be passed as the `patch_views` of a batch driver, such as
/// [`localize_patch_cloud_keypoints`](super::keypoint_localize::localize_patch_cloud_keypoints)
/// or
/// [`validate_patch_cloud_member_coherence`](super::member_coherence::validate_patch_cloud_member_coherence),
/// with one [`ProjectedImage`](super::normal_refine::ProjectedImage) per
/// reconstruction image, or as the `track_views` of
/// [`select_patch_cloud_views`](super::view_selection::select_patch_cloud_views).
///
/// # Panics
///
/// Panics if `cloud.point_indexes` is not parallel to its patches.
pub fn track_views_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<u32>> {
    assert_eq!(
        cloud.point_indexes.len(),
        cloud.len(),
        "cloud must carry a point_index per patch"
    );
    cloud
        .point_indexes
        .iter()
        .map(|&p| {
            let p = p as usize;
            recon.point_set.tracks
                [recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1]]
                .iter()
                .map(|o| o.image_index)
                .collect()
        })
        .collect()
}

/// The stored keypoint of each track view returned by
/// [`track_views_from_reconstruction`], in the same order: the
/// `patch_keypoints` of a batch driver run over the track views, or the
/// `track_keypoints` of
/// [`select_patch_cloud_views`](super::view_selection::select_patch_cloud_views).
///
/// `None` for every view of a `sift_files` reconstruction (it carries feature
/// indexes rather than inline keypoints), which anchors those views at their
/// projections.
///
/// # Panics
///
/// Panics if `cloud.point_indexes` is not parallel to its patches.
pub fn track_keypoints_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<Option<[f64; 2]>>> {
    assert_eq!(
        cloud.point_indexes.len(),
        cloud.len(),
        "cloud must carry a point_index per patch"
    );
    let kxy = recon.keypoints_xy();
    cloud
        .point_indexes
        .iter()
        .map(|&p| {
            let p = p as usize;
            let range =
                recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1];
            match kxy {
                Some(k) => range
                    .map(|r| Some([k[[r, 0]] as f64, k[[r, 1]] as f64]))
                    .collect(),
                None => range.map(|_| None).collect(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
