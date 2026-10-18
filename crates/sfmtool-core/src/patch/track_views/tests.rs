// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::patch::cloud::{PatchExtent, PatchNormal};
use crate::progress::Progress;

/// A `sift_files` reconstruction stores no inline keypoints, so the helper's
/// documented behavior is an all-`None` table parallel to the track views —
/// those members anchor at their projections.
#[test]
fn track_keypoints_from_reconstruction_is_all_none_without_inline_keypoints() {
    let recon = crate::reconstruction::SfmrReconstruction::demo(6);
    let cloud = PatchCloud::from_reconstruction(
        &recon,
        PatchNormal::MeanViewing,
        PatchExtent::Fixed(0.05),
        true,
        &Progress::none(),
    )
    .unwrap();
    let tv = track_views_from_reconstruction(&recon, &cloud);
    let kps = track_keypoints_from_reconstruction(&recon, &cloud);
    assert_eq!(kps.len(), tv.len());
    for (k, v) in kps.iter().zip(&tv) {
        assert_eq!(k.len(), v.len(), "keypoints parallel to track views");
        assert!(
            k.iter().all(|e| e.is_none()),
            "sift_files reconstruction has no inline keypoints"
        );
    }
}
