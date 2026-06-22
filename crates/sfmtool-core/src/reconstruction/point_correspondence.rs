// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point correspondence finding between SfM reconstructions.
//!
//! Finds corresponding 3D points between two reconstructions by analyzing
//! shared feature observations across images.

use std::collections::{HashMap, HashSet};

use crate::SfmrReconstruction;

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

/// Result of merging 3D points and tracks from multiple reconstructions.
pub struct MergedPointsAndTracks {
    /// Averaged positions of merged 3D points, `[x, y, z]` per point.
    pub positions: Vec<[f64; 3]>,
    /// Averaged RGB colors per point.
    pub colors: Vec<[u8; 3]>,
    /// Averaged reprojection errors per point.
    pub errors: Vec<f32>,
    /// Image indexes for all track observations.
    pub track_image_indexes: Vec<u32>,
    /// Feature indexes for all track observations.
    pub track_feature_indexes: Vec<u32>,
    /// Point IDs for all track observations.
    pub track_point_ids: Vec<u32>,
}

/// Merge 3D points and tracks from multiple aligned reconstructions.
///
/// Points listed in `correspondence_groups` are merged by averaging their
/// positions/colors/errors. Remaining unique points are added as-is.
/// A union-find pass then merges any points (across both groups and unique
/// points) that share the same observation (merged_image_idx, feature_idx).
///
/// Uses `observations_for_point()` for O(1) per-point track lookup.
///
/// # Arguments
/// * `reconstructions` — Slice of reconstruction references.
/// * `correspondence_groups` — Groups of (recon_idx, point_id) that represent
///   the same physical 3D point.
/// * `reverse_image_mapping` — Per-reconstruction mapping from old image index
///   to merged image index.
///
/// # Errors
/// Returns an error if any input reconstruction is `embedded_patches`. Merge
/// keys observations on `(image, feature_index)` identity, which an
/// `embedded_patches` reconstruction has no feature index to provide; merging
/// those is not supported.
pub fn merge_points_and_tracks(
    reconstructions: &[&SfmrReconstruction],
    correspondence_groups: &[Vec<(usize, u32)>],
    reverse_image_mapping: &[HashMap<u32, u32>],
) -> Result<MergedPointsAndTracks, String> {
    // Merge identity is `(image, feature_index)`, a sift_files notion. Refuse
    // embedded_patches up front rather than silently substituting a placeholder
    // feature index (which would collapse distinct observations in an image).
    for (i, recon) in reconstructions.iter().enumerate() {
        if recon.feature_indexes().is_none() {
            return Err(format!(
                "merge_points_and_tracks is not supported for embedded_patches \
                 reconstructions (reconstruction {i} is {})",
                recon.feature_source()
            ));
        }
    }

    struct TempPoint {
        position: [f64; 3],
        color: [f64; 3],
        error: f64,
        observations: HashSet<(u32, u32)>,
    }

    let mut temp_points: Vec<TempPoint> = Vec::new();
    let mut obs_to_points: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut in_group: HashSet<(usize, u32)> = HashSet::new();

    // Helper to collect observations for a point, remapping image indexes.
    let collect_observations = |recon_idx: usize, point_id: u32| -> HashSet<(u32, u32)> {
        let recon = reconstructions[recon_idx];
        let mapping = &reverse_image_mapping[recon_idx];
        // sift_files-only: guaranteed `Some` by the embedded_patches check above.
        let feature_indexes = recon
            .feature_indexes()
            .expect("embedded_patches rejected above");
        let start = recon.observation_offsets[point_id as usize];
        let mut obs = HashSet::new();
        for (k, track) in recon
            .observations_for_point(point_id as usize)
            .iter()
            .enumerate()
        {
            if let Some(&merged_img_idx) = mapping.get(&track.image_index) {
                obs.insert((merged_img_idx, feature_indexes[start + k]));
            }
        }
        obs
    };

    // Step 1: Create temp points from correspondence groups.
    for group in correspondence_groups {
        let mut sum_pos = [0.0f64; 3];
        let mut sum_color = [0.0f64; 3];
        let mut sum_error = 0.0f64;
        let mut count = 0usize;
        let mut observations = HashSet::new();

        for &(recon_idx, point_id) in group {
            let point = &reconstructions[recon_idx].points[point_id as usize];
            sum_pos[0] += point.position.x;
            sum_pos[1] += point.position.y;
            sum_pos[2] += point.position.z;
            sum_color[0] += point.color[0] as f64;
            sum_color[1] += point.color[1] as f64;
            sum_color[2] += point.color[2] as f64;
            sum_error += point.error as f64;
            count += 1;
            in_group.insert((recon_idx, point_id));
            observations.extend(collect_observations(recon_idx, point_id));
        }

        let c = count as f64;
        let tp_id = temp_points.len();
        let tp = TempPoint {
            position: [sum_pos[0] / c, sum_pos[1] / c, sum_pos[2] / c],
            color: [sum_color[0] / c, sum_color[1] / c, sum_color[2] / c],
            error: sum_error / c,
            observations,
        };
        for &obs in &tp.observations {
            obs_to_points.entry(obs).or_default().push(tp_id);
        }
        temp_points.push(tp);
    }

    // Step 2: Add unique points (not in any correspondence group).
    for (recon_idx, recon) in reconstructions.iter().enumerate() {
        for point_id in 0..recon.points.len() {
            if in_group.contains(&(recon_idx, point_id as u32)) {
                continue;
            }
            let point = &recon.points[point_id];
            let observations = collect_observations(recon_idx, point_id as u32);

            let tp_id = temp_points.len();
            for &obs in &observations {
                obs_to_points.entry(obs).or_default().push(tp_id);
            }
            temp_points.push(TempPoint {
                position: [point.position.x, point.position.y, point.position.z],
                color: [
                    point.color[0] as f64,
                    point.color[1] as f64,
                    point.color[2] as f64,
                ],
                error: point.error as f64,
                observations,
            });
        }
    }

    // Step 3: Union-find — merge temp points that share observations.
    let n = temp_points.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path splitting
            x = parent[x];
        }
        x
    }

    for point_ids in obs_to_points.values() {
        if point_ids.len() > 1 {
            let root = find(&mut parent, point_ids[0]);
            for &id in &point_ids[1..] {
                let other = find(&mut parent, id);
                if root != other {
                    parent[other] = root;
                }
            }
        }
    }

    // Step 4: Group by root.
    let mut point_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        point_groups.entry(root).or_default().push(i);
    }

    // Step 5: Create final merged points.
    let mut result = MergedPointsAndTracks {
        positions: Vec::with_capacity(point_groups.len()),
        colors: Vec::with_capacity(point_groups.len()),
        errors: Vec::with_capacity(point_groups.len()),
        track_image_indexes: Vec::new(),
        track_feature_indexes: Vec::new(),
        track_point_ids: Vec::new(),
    };

    for group_ids in point_groups.values() {
        let merged_point_id = result.positions.len() as u32;
        let gc = group_ids.len() as f64;

        let mut avg_pos = [0.0f64; 3];
        let mut avg_color = [0.0f64; 3];
        let mut avg_error = 0.0f64;
        let mut all_observations: HashSet<(u32, u32)> = HashSet::new();

        for &tp_id in group_ids {
            let tp = &temp_points[tp_id];
            avg_pos[0] += tp.position[0];
            avg_pos[1] += tp.position[1];
            avg_pos[2] += tp.position[2];
            avg_color[0] += tp.color[0];
            avg_color[1] += tp.color[1];
            avg_color[2] += tp.color[2];
            avg_error += tp.error;
            all_observations.extend(&tp.observations);
        }

        result
            .positions
            .push([avg_pos[0] / gc, avg_pos[1] / gc, avg_pos[2] / gc]);
        result.colors.push([
            (avg_color[0] / gc).round() as u8,
            (avg_color[1] / gc).round() as u8,
            (avg_color[2] / gc).round() as u8,
        ]);
        result.errors.push((avg_error / gc) as f32);

        for (merged_img_idx, feat_idx) in all_observations {
            result.track_image_indexes.push(merged_img_idx);
            result.track_feature_indexes.push(feat_idx);
            result.track_point_ids.push(merged_point_id);
        }
    }

    Ok(result)
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

    #[test]
    fn test_merge_rejects_embedded_patches() {
        use crate::reconstruction::ObservationSource;

        // Merge keys on (image, feature_index), which embedded_patches lacks, so
        // the guard must refuse it up front rather than substitute a placeholder.
        let mut recon = SfmrReconstruction::demo(2);
        let n_obs = recon.tracks.len();
        let n_img = recon.images.len();
        recon.observations = ObservationSource::EmbeddedPatches {
            keypoints_xy: ndarray::Array2::zeros((n_obs, 2)),
            image_file_hashes: vec![[0u8; 16]; n_img],
        };

        let mapping = vec![HashMap::new()];
        // (MergedPointsAndTracks is not Debug, so match rather than expect_err.)
        let Err(err) = merge_points_and_tracks(&[&recon], &[], &mapping) else {
            panic!("merge must reject embedded_patches");
        };
        assert!(
            err.contains("embedded_patches"),
            "unexpected error message: {err}"
        );
    }
}
