// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image pair graph utilities for computing covisibility and frustum intersection pairs.
//!
//! This module provides functions for identifying which image pairs should be
//! matched based on either:
//! 1. Covisibility - images that share 3D points in the reconstruction
//! 2. Frustum intersection - images whose view frustums overlap in 3D space

use std::collections::HashMap;

use nalgebra::{Quaternion, UnitQuaternion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::frustum;

/// Compute camera viewing directions from quaternions.
///
/// For each quaternion, builds a rotation matrix `R_cam_from_world`, transposes to get
/// `R_world_from_cam`, then computes `direction = -R_world_from_cam[:, 2]` (the camera
/// looks down -Z in camera space).
///
/// # Arguments
/// * `quaternions_wxyz` - Flat slice of `N*4` values `[w0,x0,y0,z0, w1,...]`
/// * `num_images` - Number of images N
///
/// # Returns
/// Flat `Vec<f64>` of `N*3` normalized direction vectors.
pub fn compute_camera_directions(quaternions_wxyz: &[f64], num_images: usize) -> Vec<f64> {
    debug_assert_eq!(quaternions_wxyz.len(), num_images * 4);

    let mut directions = vec![0.0; num_images * 3];

    for img_idx in 0..num_images {
        let qo = img_idx * 4;
        let quat = UnitQuaternion::new_normalize(Quaternion::new(
            quaternions_wxyz[qo],
            quaternions_wxyz[qo + 1],
            quaternions_wxyz[qo + 2],
            quaternions_wxyz[qo + 3],
        ));

        let r_cam_from_world = quat.to_rotation_matrix();
        let r_world_from_cam = r_cam_from_world.transpose();

        // Camera looks down -Z in camera space, so direction = -R_world_from_cam * [0,0,1]
        // which is -column(2) of R_world_from_cam
        let col2 = r_world_from_cam.matrix().column(2);
        let mut dir = [-col2[0], -col2[1], -col2[2]];

        // Normalize
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if norm > 0.0 {
            dir[0] /= norm;
            dir[1] /= norm;
            dir[2] /= norm;
        }

        let do_ = img_idx * 3;
        directions[do_] = dir[0];
        directions[do_ + 1] = dir[1];
        directions[do_ + 2] = dir[2];
    }

    directions
}

/// Build covisibility pairs from track observations.
///
/// Finds all image pairs that share 3D points, filtered by camera viewing angle.
///
/// # Arguments
/// * `quaternions_wxyz` - Flat `N*4` slice of quaternions in WXYZ format
/// * `num_images` - Number of images
/// * `track_point_ids` - Point ID for each observation
/// * `track_image_indexes` - Image index for each observation
/// * `angle_threshold_deg` - Max angle between camera directions (90.0 typical)
///
/// # Returns
/// `Vec<(u32, u32, u32)>` of `(img_i, img_j, shared_count)`, sorted by
/// `shared_count` descending.
pub fn build_covisibility_pairs(
    quaternions_wxyz: &[f64],
    num_images: usize,
    track_point_ids: &[u32],
    track_image_indexes: &[u32],
    angle_threshold_deg: f64,
) -> Vec<(u32, u32, u32)> {
    debug_assert_eq!(track_point_ids.len(), track_image_indexes.len());

    // Step 1: Build mapping from point_id -> list of image_ids
    let mut point_to_images: HashMap<u32, Vec<u32>> = HashMap::new();
    for (obs_idx, &point_id) in track_point_ids.iter().enumerate() {
        let image_id = track_image_indexes[obs_idx];
        point_to_images.entry(point_id).or_default().push(image_id);
    }

    // Step 2: For each point, count covisibility between all pairs
    let mut covis: HashMap<(u32, u32), u32> = HashMap::new();

    for image_list in point_to_images.values() {
        let mut unique_images: Vec<u32> = image_list.clone();
        unique_images.sort_unstable();
        unique_images.dedup();

        for i in 0..unique_images.len() {
            for j in (i + 1)..unique_images.len() {
                let key = (unique_images[i], unique_images[j]);
                *covis.entry(key).or_insert(0) += 1;
            }
        }
    }

    // Step 3: Compute camera directions
    let directions = compute_camera_directions(quaternions_wxyz, num_images);

    // Step 4: Filter by angle and collect pairs
    let cos_threshold = angle_threshold_deg.to_radians().cos();

    let mut pairs: Vec<(u32, u32, u32)> = covis
        .into_iter()
        .filter(|&((img_i, img_j), _)| {
            let oi = img_i as usize * 3;
            let oj = img_j as usize * 3;
            let dot = directions[oi] * directions[oj]
                + directions[oi + 1] * directions[oj + 1]
                + directions[oi + 2] * directions[oj + 2];
            dot >= cos_threshold
        })
        .map(|((i, j), count)| (i, j, count))
        .collect();

    // Step 5: Sort by count descending
    pairs.sort_by(|a, b| b.2.cmp(&a.2));

    pairs
}

/// Estimate Z value at given percentile from histogram.
///
/// Computes the cumulative sum, finds the bin containing the target count
/// (`percentile / 100 * total`), and returns the bin center value via
/// linear interpolation.
///
/// # Arguments
/// * `hist_counts` - Histogram bin counts
/// * `min_z` - Minimum Z value (left edge of first bin)
/// * `max_z` - Maximum Z value (right edge of last bin)
/// * `percentile` - Percentile to estimate (0-100)
pub fn estimate_z_from_histogram(
    hist_counts: &[u32],
    min_z: f64,
    max_z: f64,
    percentile: f64,
) -> f64 {
    let total_count: u64 = hist_counts.iter().map(|&c| c as u64).sum();
    if total_count == 0 {
        return (min_z + max_z) / 2.0;
    }

    // Compute cumulative distribution and find bin containing target
    let target_count = (percentile / 100.0) * total_count as f64;

    let mut cumsum: u64 = 0;
    let mut bin_idx = hist_counts.len() - 1;
    for (i, &count) in hist_counts.iter().enumerate() {
        cumsum += count as u64;
        if cumsum as f64 >= target_count {
            bin_idx = i;
            break;
        }
    }

    // Linear interpolation at bin center
    let num_bins = hist_counts.len();
    let bin_width = (max_z - min_z) / num_bins as f64;
    min_z + (bin_idx as f64 + 0.5) * bin_width
}

/// Build frustum intersection pairs using Monte Carlo volume estimation.
///
/// Uses rayon for parallel pair comparison. Each thread gets a deterministic
/// RNG seeded from `seed + i`.
///
/// # Arguments
/// * `quaternions_wxyz` - Flat `N*4` slice of quaternions in WXYZ format
/// * `translations` - Flat `N*3` slice of camera translations
/// * `num_images` - Number of images N
/// * `fx`, `fy`, `cx`, `cy` - Per-image camera intrinsics (length N each)
/// * `widths`, `heights` - Per-image dimensions (length N each)
/// * `histogram_counts` - Flat `N*num_bins` slice of depth histogram counts
/// * `histogram_min_z`, `histogram_max_z` - Per-image (length N, `f64::NAN` for missing)
/// * `num_bins` - Number of histogram bins per image
/// * `near_percentile`, `far_percentile` - For Z estimation (5.0, 95.0 typical)
/// * `num_samples` - Monte Carlo samples per pair (100 typical)
/// * `angle_threshold_deg` - Max viewing angle (90.0 typical)
/// * `seed` - RNG seed for reproducibility
///
/// # Returns
/// `Vec<(u32, u32, f64)>` of `(img_i, img_j, intersection_volume)` sorted by
/// volume descending.
#[allow(clippy::too_many_arguments)]
pub fn build_frustum_intersection_pairs(
    quaternions_wxyz: &[f64],
    translations: &[f64],
    num_images: usize,
    fx: &[f64],
    fy: &[f64],
    cx: &[f64],
    cy: &[f64],
    widths: &[u32],
    heights: &[u32],
    histogram_counts: &[u32],
    histogram_min_z: &[f64],
    histogram_max_z: &[f64],
    num_bins: usize,
    near_percentile: f64,
    far_percentile: f64,
    num_samples: usize,
    angle_threshold_deg: f64,
    seed: u64,
) -> Vec<(u32, u32, f64)> {
    // Step 1: Compute camera directions for angle culling
    let directions = compute_camera_directions(quaternions_wxyz, num_images);
    let cos_threshold = angle_threshold_deg.to_radians().cos();

    // Step 2: Pre-compute frustum data for each image
    // Each entry is Option<(corners: [f64; 24], planes: [f64; 24], volume: f64)>
    #[allow(clippy::type_complexity)]
    let frustum_data: Vec<Option<([f64; 24], [f64; 24], f64)>> = (0..num_images)
        .map(|img_idx| {
            // Skip images without depth data
            if histogram_min_z[img_idx].is_nan() {
                return None;
            }

            let min_z = histogram_min_z[img_idx];
            let max_z = histogram_max_z[img_idx];

            // Estimate near/far Z from histogram percentiles
            let hist_slice = &histogram_counts[img_idx * num_bins..(img_idx + 1) * num_bins];

            let mut near_z = estimate_z_from_histogram(hist_slice, min_z, max_z, near_percentile);
            let mut far_z = estimate_z_from_histogram(hist_slice, min_z, max_z, far_percentile);

            // Ensure near < far
            if near_z >= far_z {
                near_z = min_z;
                far_z = max_z;
            }

            // Build rotation and compute camera center
            let qo = img_idx * 4;
            let quat = UnitQuaternion::new_normalize(Quaternion::new(
                quaternions_wxyz[qo],
                quaternions_wxyz[qo + 1],
                quaternions_wxyz[qo + 2],
                quaternions_wxyz[qo + 3],
            ));

            let r_cam_from_world = quat.to_rotation_matrix();
            let r_world_from_cam = r_cam_from_world.transpose();

            // Camera center: C = -R_world_from_cam @ translation
            let to = img_idx * 3;
            let t = nalgebra::Vector3::new(
                translations[to],
                translations[to + 1],
                translations[to + 2],
            );
            let camera_center = -(r_world_from_cam.matrix() * t);

            // Build R_world_from_cam as row-major [f64; 9]
            let rwc = r_world_from_cam.matrix();
            let r_flat: [f64; 9] = [
                rwc[(0, 0)],
                rwc[(0, 1)],
                rwc[(0, 2)],
                rwc[(1, 0)],
                rwc[(1, 1)],
                rwc[(1, 2)],
                rwc[(2, 0)],
                rwc[(2, 1)],
                rwc[(2, 2)],
            ];

            let center: [f64; 3] = [camera_center[0], camera_center[1], camera_center[2]];

            let corners = frustum::compute_frustum_corners(
                &center,
                &r_flat,
                fx[img_idx],
                fy[img_idx],
                cx[img_idx],
                cy[img_idx],
                widths[img_idx],
                heights[img_idx],
                near_z,
                far_z,
            );
            let planes = frustum::compute_frustum_planes(&center, &corners);
            let volume = frustum::compute_frustum_volume(
                widths[img_idx],
                heights[img_idx],
                fx[img_idx],
                fy[img_idx],
                near_z,
                far_z,
            );

            Some((corners, planes, volume))
        })
        .collect();

    // Step 3: Parallel pair loop with rayon
    let mut pairs: Vec<(u32, u32, f64)> = (0..num_images)
        .into_par_iter()
        .flat_map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            let mut local_pairs = Vec::new();

            let frustum_i = match &frustum_data[i] {
                Some(fd) => fd,
                None => return local_pairs,
            };

            let (corners_i, planes_i, volume_i) = frustum_i;

            #[allow(clippy::needless_range_loop)]
            for j in (i + 1)..num_images {
                let frustum_j = match &frustum_data[j] {
                    Some(fd) => fd,
                    None => continue,
                };

                let (corners_j, planes_j, volume_j) = frustum_j;

                // Angle culling
                let oi = i * 3;
                let oj = j * 3;
                let dot = directions[oi] * directions[oj]
                    + directions[oi + 1] * directions[oj + 1]
                    + directions[oi + 2] * directions[oj + 2];
                if dot < cos_threshold {
                    continue;
                }

                // Separating plane test
                if !frustum::frustums_can_intersect(corners_i, planes_i, corners_j, planes_j) {
                    continue;
                }

                // Monte Carlo volume estimate from both directions, averaged
                let vol_i_to_j = frustum::estimate_frustum_intersection_volume(
                    corners_i,
                    planes_i,
                    *volume_i,
                    planes_j,
                    num_samples,
                    &mut rng,
                );
                let vol_j_to_i = frustum::estimate_frustum_intersection_volume(
                    corners_j,
                    planes_j,
                    *volume_j,
                    planes_i,
                    num_samples,
                    &mut rng,
                );

                let intersection_vol = (vol_i_to_j + vol_j_to_i) / 2.0;

                if intersection_vol > 0.0 {
                    local_pairs.push((i as u32, j as u32, intersection_vol));
                }
            }

            local_pairs
        })
        .collect();

    // Step 4: Sort by volume descending
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    /// Identity quaternion: w=1, x=0, y=0, z=0.
    fn identity_quat() -> [f64; 4] {
        [1.0, 0.0, 0.0, 0.0]
    }

    /// Quaternion for 180-degree rotation around Y axis.
    fn quat_180_around_y() -> [f64; 4] {
        // 180° around Y: w=cos(90°)=0, x=0, y=sin(90°)=1, z=0
        [0.0, 0.0, 1.0, 0.0]
    }

    #[test]
    fn test_camera_directions_identity() {
        let quaternions = identity_quat();
        let dirs = compute_camera_directions(&quaternions, 1);

        assert_eq!(dirs.len(), 3);
        // Identity rotation: camera looks down -Z in camera space.
        // R_world_from_cam = I, so direction = -I[:, 2] = (0, 0, -1)
        assert_relative_eq!(dirs[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dirs[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dirs[2], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_camera_directions_rotated() {
        let quaternions = quat_180_around_y();
        let dirs = compute_camera_directions(&quaternions, 1);

        assert_eq!(dirs.len(), 3);
        // 180° around Y: the Z axis flips, so direction = (0, 0, 1)
        assert_relative_eq!(dirs[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dirs[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dirs[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_covisibility_simple() {
        // 3 images, all identity quaternions (all pointing same direction)
        let quaternions: Vec<f64> = [identity_quat(), identity_quat(), identity_quat()].concat();

        // Point 0 observed by images 0 and 1
        // Point 1 observed by images 0 and 2
        let track_point_ids = [0_u32, 0, 1, 1];
        let track_image_indexes = [0_u32, 1, 0, 2];

        let pairs = build_covisibility_pairs(
            &quaternions,
            3,
            &track_point_ids,
            &track_image_indexes,
            90.0,
        );

        assert_eq!(pairs.len(), 2);

        // Both pairs have count 1; check we have the right pairs
        let pair_set: std::collections::HashSet<(u32, u32)> =
            pairs.iter().map(|&(i, j, _)| (i, j)).collect();
        assert!(pair_set.contains(&(0, 1)));
        assert!(pair_set.contains(&(0, 2)));

        // All counts should be 1
        for &(_, _, count) in &pairs {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_covisibility_angle_culling() {
        // Two cameras pointing opposite directions: identity and 180° around Y
        let quaternions: Vec<f64> = [identity_quat(), quat_180_around_y()].concat();

        // Point 0 observed by both images
        let track_point_ids = [0_u32, 0];
        let track_image_indexes = [0_u32, 1];

        // Angle between directions is 180° which exceeds 90° threshold
        let pairs = build_covisibility_pairs(
            &quaternions,
            2,
            &track_point_ids,
            &track_image_indexes,
            90.0,
        );

        assert!(
            pairs.is_empty(),
            "Cameras pointing opposite directions should be filtered at 90° threshold"
        );
    }

    #[test]
    fn test_estimate_z_from_histogram() {
        // Uniform histogram: 4 bins each with 10 counts, min=0, max=4
        let hist_counts = [10_u32, 10, 10, 10];
        let z = estimate_z_from_histogram(&hist_counts, 0.0, 4.0, 50.0);

        // 50th percentile of uniform [0, 4] should be ~2.0
        // Bin width = 1.0, target count = 20 out of 40
        // cumsum = [10, 20, ...], bin_idx = 1, center = 0 + (1 + 0.5) * 1.0 = 1.5
        // This is the bin center which is an approximation
        assert_relative_eq!(z, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_estimate_z_from_histogram_empty() {
        let hist_counts = [0_u32, 0, 0, 0];
        let z = estimate_z_from_histogram(&hist_counts, 1.0, 5.0, 50.0);
        assert_relative_eq!(z, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_build_frustum_intersection_pairs_identical_cameras() {
        let num_images = 2;
        let quaternions: Vec<f64> = [identity_quat(), identity_quat()].concat();
        let translations = vec![0.0; num_images * 3]; // both at origin

        let fx_vals = vec![500.0; num_images];
        let fy_vals = vec![500.0; num_images];
        let cx_vals = vec![320.0; num_images];
        let cy_vals = vec![240.0; num_images];
        let widths = vec![640_u32; num_images];
        let heights = vec![480_u32; num_images];

        // Uniform depth histograms
        let num_bins = 10;
        let hist_counts: Vec<u32> = vec![10; num_images * num_bins];
        let histogram_min_z = vec![1.0; num_images];
        let histogram_max_z = vec![10.0; num_images];

        let pairs = build_frustum_intersection_pairs(
            &quaternions,
            &translations,
            num_images,
            &fx_vals,
            &fy_vals,
            &cx_vals,
            &cy_vals,
            &widths,
            &heights,
            &hist_counts,
            &histogram_min_z,
            &histogram_max_z,
            num_bins,
            5.0,
            95.0,
            1000,
            90.0,
            42,
        );

        assert_eq!(pairs.len(), 1, "Should find exactly one pair");
        assert_eq!(pairs[0].0, 0);
        assert_eq!(pairs[0].1, 1);

        // Identical cameras should have high intersection volume
        let volume = pairs[0].2;
        assert!(
            volume > 0.0,
            "Identical cameras should have positive intersection volume, got {}",
            volume
        );

        // Compare to analytical frustum volume
        // With 5th and 95th percentile on uniform [1, 10], near ~ bin center,
        // far ~ bin center. The intersection should be a large fraction of the volume.
        let expected_near = estimate_z_from_histogram(&hist_counts[..num_bins], 1.0, 10.0, 5.0);
        let expected_far = estimate_z_from_histogram(&hist_counts[..num_bins], 1.0, 10.0, 95.0);
        let full_volume =
            frustum::compute_frustum_volume(640, 480, 500.0, 500.0, expected_near, expected_far);

        // Monte Carlo estimate should be within 50% of full volume for identical cameras
        assert!(
            volume > full_volume * 0.5,
            "Intersection volume ({:.4}) should be at least 50% of full volume ({:.4})",
            volume,
            full_volume
        );
    }

    #[test]
    fn test_build_frustum_intersection_pairs_nan_skipped() {
        let num_images = 2;
        let quaternions: Vec<f64> = [identity_quat(), identity_quat()].concat();
        let translations = vec![0.0; num_images * 3];

        let fx_vals = vec![500.0; num_images];
        let fy_vals = vec![500.0; num_images];
        let cx_vals = vec![320.0; num_images];
        let cy_vals = vec![240.0; num_images];
        let widths = vec![640_u32; num_images];
        let heights = vec![480_u32; num_images];

        let num_bins = 4;
        let hist_counts = vec![10_u32; num_images * num_bins];
        // First image has NAN min_z -> should be skipped
        let histogram_min_z = vec![f64::NAN, 1.0];
        let histogram_max_z = vec![f64::NAN, 10.0];

        let pairs = build_frustum_intersection_pairs(
            &quaternions,
            &translations,
            num_images,
            &fx_vals,
            &fy_vals,
            &cx_vals,
            &cy_vals,
            &widths,
            &heights,
            &hist_counts,
            &histogram_min_z,
            &histogram_max_z,
            num_bins,
            5.0,
            95.0,
            100,
            90.0,
            42,
        );

        assert!(
            pairs.is_empty(),
            "Should find no pairs when one image has NAN depth stats"
        );
    }

    #[test]
    fn test_camera_directions_multiple() {
        // Test with multiple images
        let quaternions: Vec<f64> =
            [identity_quat(), identity_quat(), quat_180_around_y()].concat();

        let dirs = compute_camera_directions(&quaternions, 3);
        assert_eq!(dirs.len(), 9);

        // First two should be (0, 0, -1)
        assert_relative_eq!(dirs[2], -1.0, epsilon = 1e-10);
        assert_relative_eq!(dirs[5], -1.0, epsilon = 1e-10);

        // Third should be (0, 0, 1)
        assert_relative_eq!(dirs[8], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_camera_directions_90_around_y() {
        // 90° rotation around Y axis: w=cos(45°), x=0, y=sin(45°), z=0
        let half = (PI / 4.0).cos();
        let quaternions = [half, 0.0, half, 0.0];

        let dirs = compute_camera_directions(&quaternions, 1);

        // R_cam_from_world = R_y(90°), R_world_from_cam = R_y(90°)^T
        // direction = -col2(R_world_from_cam) = (1, 0, 0)
        // Verified against Python numpy-quaternion implementation.
        assert_relative_eq!(dirs[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dirs[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dirs[2], 0.0, epsilon = 1e-10);
    }
}
