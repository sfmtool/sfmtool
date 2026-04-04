// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point correspondence finding between SfM reconstructions.
//!
//! Finds corresponding 3D points between two reconstructions by analyzing
//! shared feature observations across images.

use std::collections::HashMap;

/// Result of finding point correspondences between two reconstructions.
pub struct PointCorrespondenceResult {
    /// Source point IDs for each correspondence.
    pub source_ids: Vec<u32>,
    /// Target point IDs for each correspondence.
    pub target_ids: Vec<u32>,
}

/// Find corresponding 3D points between two reconstructions via shared feature observations.
///
/// For each pair of shared images, finds features that appear in both, then maps
/// those features to their respective 3D points. Uses first-occurrence semantics
/// when a source point appears in multiple shared images.
///
/// # Arguments
/// * `source_track_image_indexes` - Image index for each source track observation
/// * `source_track_feature_indexes` - Feature index for each source track observation
/// * `source_track_point_ids` - Point ID for each source track observation
/// * `target_track_image_indexes` - Image index for each target track observation
/// * `target_track_feature_indexes` - Feature index for each target track observation
/// * `target_track_point_ids` - Point ID for each target track observation
/// * `shared_images_source` - Source image indices for shared image pairs
/// * `shared_images_target` - Target image indices for shared image pairs
///
/// # Returns
/// `PointCorrespondenceResult` with parallel `source_ids` and `target_ids` vectors.
#[allow(clippy::too_many_arguments)]
pub fn find_point_correspondences(
    source_track_image_indexes: &[u32],
    source_track_feature_indexes: &[u32],
    source_track_point_ids: &[u32],
    target_track_image_indexes: &[u32],
    target_track_feature_indexes: &[u32],
    target_track_point_ids: &[u32],
    shared_images_source: &[u32],
    shared_images_target: &[u32],
) -> PointCorrespondenceResult {
    let source_map = build_feature_to_point_map(
        source_track_image_indexes,
        source_track_feature_indexes,
        source_track_point_ids,
    );
    let target_map = build_feature_to_point_map(
        target_track_image_indexes,
        target_track_feature_indexes,
        target_track_point_ids,
    );

    let mut correspondences: HashMap<u32, u32> = HashMap::new();

    for (&src_img, &tgt_img) in shared_images_source.iter().zip(shared_images_target.iter()) {
        let Some(src_features) = source_map.get(&src_img) else {
            continue;
        };
        let Some(tgt_features) = target_map.get(&tgt_img) else {
            continue;
        };

        for (&feat_idx, &src_point_id) in src_features {
            if let Some(&tgt_point_id) = tgt_features.get(&feat_idx) {
                correspondences.entry(src_point_id).or_insert(tgt_point_id);
            }
        }
    }

    let mut source_ids = Vec::with_capacity(correspondences.len());
    let mut target_ids = Vec::with_capacity(correspondences.len());
    for (src_id, tgt_id) in &correspondences {
        source_ids.push(*src_id);
        target_ids.push(*tgt_id);
    }

    PointCorrespondenceResult {
        source_ids,
        target_ids,
    }
}

/// Build a nested map: image_idx -> (feature_idx -> point_id).
fn build_feature_to_point_map(
    image_indexes: &[u32],
    feature_indexes: &[u32],
    point_ids: &[u32],
) -> HashMap<u32, HashMap<u32, u32>> {
    let mut map: HashMap<u32, HashMap<u32, u32>> = HashMap::new();
    for i in 0..image_indexes.len() {
        map.entry(image_indexes[i])
            .or_default()
            .insert(feature_indexes[i], point_ids[i]);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_correspondences() {
        // Source reconstruction: image 0 has features 10,20; image 1 has feature 30
        let src_img = [0, 0, 1];
        let src_feat = [10, 20, 30];
        let src_pts = [100, 200, 300];

        // Target reconstruction: image 0 has features 10,20; image 1 has feature 30
        let tgt_img = [0, 0, 1];
        let tgt_feat = [10, 20, 30];
        let tgt_pts = [500, 600, 700];

        // Shared images: source img 0 <-> target img 0, source img 1 <-> target img 1
        let shared_src = [0, 1];
        let shared_tgt = [0, 1];

        let result = find_point_correspondences(
            &src_img,
            &src_feat,
            &src_pts,
            &tgt_img,
            &tgt_feat,
            &tgt_pts,
            &shared_src,
            &shared_tgt,
        );

        assert_eq!(result.source_ids.len(), 3);
        assert_eq!(result.target_ids.len(), 3);

        // Verify all correspondences are present (order may vary due to HashMap)
        let corr: HashMap<u32, u32> = result
            .source_ids
            .iter()
            .zip(result.target_ids.iter())
            .map(|(&s, &t)| (s, t))
            .collect();
        assert_eq!(corr[&100], 500);
        assert_eq!(corr[&200], 600);
        assert_eq!(corr[&300], 700);
    }

    #[test]
    fn test_no_shared_features() {
        // Source has feature 10 in image 0
        let src_img = [0];
        let src_feat = [10];
        let src_pts = [100];

        // Target has feature 20 in image 0 (different feature index)
        let tgt_img = [0];
        let tgt_feat = [20];
        let tgt_pts = [500];

        let shared_src = [0];
        let shared_tgt = [0];

        let result = find_point_correspondences(
            &src_img,
            &src_feat,
            &src_pts,
            &tgt_img,
            &tgt_feat,
            &tgt_pts,
            &shared_src,
            &shared_tgt,
        );

        assert!(result.source_ids.is_empty());
        assert!(result.target_ids.is_empty());
    }

    #[test]
    fn test_deduplication_first_occurrence_wins() {
        // Source: point 100 is observed as feature 10 in both image 0 and image 1
        let src_img = [0, 1];
        let src_feat = [10, 10];
        let src_pts = [100, 100];

        // Target: feature 10 maps to point 500 in image 0, point 600 in image 1
        let tgt_img = [0, 1];
        let tgt_feat = [10, 10];
        let tgt_pts = [500, 600];

        // Both pairs are shared
        let shared_src = [0, 1];
        let shared_tgt = [0, 1];

        let result = find_point_correspondences(
            &src_img,
            &src_feat,
            &src_pts,
            &tgt_img,
            &tgt_feat,
            &tgt_pts,
            &shared_src,
            &shared_tgt,
        );

        // Source point 100 should appear only once
        assert_eq!(result.source_ids.len(), 1);
        assert_eq!(result.source_ids[0], 100);
        // First occurrence wins: mapped to target point 500 (from first shared pair)
        assert_eq!(result.target_ids[0], 500);
    }

    #[test]
    fn test_missing_shared_image() {
        // Source has observations only in image 0
        let src_img = [0];
        let src_feat = [10];
        let src_pts = [100];

        // Target has observations only in image 0
        let tgt_img = [0];
        let tgt_feat = [10];
        let tgt_pts = [500];

        // Shared images reference image 5 which doesn't exist in either
        let shared_src = [5];
        let shared_tgt = [5];

        let result = find_point_correspondences(
            &src_img,
            &src_feat,
            &src_pts,
            &tgt_img,
            &tgt_feat,
            &tgt_pts,
            &shared_src,
            &shared_tgt,
        );

        assert!(result.source_ids.is_empty());
    }
}