// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Feature matching algorithms.
//!
//! This module provides descriptor distance computation, best-match
//! search, and sort-and-sweep matching for SIFT features.

pub mod descriptor;
pub mod geometric_filter;
pub mod polar;
pub mod sweep;

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

pub use descriptor::{
    descriptor_distance_l2, descriptor_distance_l2_squared, find_best_match,
    find_best_match_contiguous,
};
pub use geometric_filter::{
    two_stage_geometric_filter, CameraParams, GeometricFilterConfig, StereoPairGeometry,
};
pub use polar::{polar_mutual_best_match, polar_mutual_best_match_geometric};
pub use sweep::{
    match_one_way_sweep, match_one_way_sweep_geometric, mutual_best_match_sweep,
    mutual_best_match_sweep_geometric,
};

/// Match features between an image pair given camera poses and SIFT features.
///
/// Internally:
/// 1. Computes fundamental matrix and epipole
/// 2. Checks rectification safety (epipole outside image)
/// 3. If safe: computes stereo rectification, rectifies points, runs sweep matching
/// 4. If unsafe: runs polar sweep matching
/// 5. Optionally applies geometric filtering
#[allow(clippy::too_many_arguments)]
pub fn match_image_pair(
    // Camera intrinsics
    k1: &Matrix3<f64>,
    k2: &Matrix3<f64>,
    // Camera poses (cam_from_world)
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
    // Image dimensions (for rectification safety check)
    width1: u32,
    height1: u32,
    width2: u32,
    height2: u32,
    // Features (flat row-major arrays)
    positions1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    positions2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    // Matching parameters
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    rectification_margin: u32,
    // Optional geometric filtering
    affines1: Option<&[f64]>,
    affines2: Option<&[f64]>,
    geometric_config: Option<&GeometricFilterConfig>,
) -> Vec<(usize, usize, f64)> {
    use crate::{epipolar, rectification};

    let f = epipolar::compute_fundamental_matrix(k1, r1, t1, k2, r2, t2);
    // Check both epipoles: right epipole (in image 2) and left epipole (in image 1)
    let is_safe =
        rectification::check_rectification_safe_from_f(&f, width2, height2, rectification_margin)
            && rectification::check_rectification_safe_from_f(
                &f.transpose(),
                width1,
                height1,
                rectification_margin,
            );

    let use_geometric = affines1.is_some() && affines2.is_some() && geometric_config.is_some();

    if !is_safe {
        // Polar path
        let f_data = f.as_slice();
        let f_arr: [f64; 9] = [
            f_data[0], f_data[3], f_data[6], // row 0 (nalgebra is column-major)
            f_data[1], f_data[4], f_data[7], // row 1
            f_data[2], f_data[5], f_data[8], // row 2
        ];
        let min_radius = 10.0;

        let polar_result = if use_geometric {
            let geom = StereoPairGeometry::new(k1, k2, r1, r2, t1, t2);
            polar_mutual_best_match_geometric(
                positions1,
                descriptors1,
                n1,
                positions2,
                descriptors2,
                n2,
                affines1.unwrap(),
                affines2.unwrap(),
                desc_len,
                &f_arr,
                window_size,
                threshold,
                min_radius,
                &geom,
                geometric_config.unwrap(),
            )
        } else {
            polar_mutual_best_match(
                positions1,
                descriptors1,
                n1,
                positions2,
                descriptors2,
                n2,
                desc_len,
                &f_arr,
                window_size,
                threshold,
                min_radius,
            )
        };

        // If polar returns Some, use it; if None (epipoles at infinity), fall through to rectified path
        if let Some(matches) = polar_result {
            return matches;
        }
    }

    // Rectified sweep path
    let r_rel = r2 * r1.transpose();
    let t_rel = t2 - r_rel * t1;
    let rect = rectification::compute_stereo_rectification(k1, k2, &r_rel, &t_rel, width1, height1);

    let k1_inv = k1
        .try_inverse()
        .expect("Intrinsic matrix K1 must be invertible");
    let k2_inv = k2
        .try_inverse()
        .expect("Intrinsic matrix K2 must be invertible");

    let rectified1 =
        rectification::rectify_points(positions1, n1, &k1_inv, &rect.r1_rect, &rect.p1);
    let rectified2 =
        rectification::rectify_points(positions2, n2, &k2_inv, &rect.r2_rect, &rect.p2);

    if use_geometric {
        let geom = StereoPairGeometry::new(k1, k2, r1, r2, t1, t2);
        mutual_best_match_sweep_geometric(
            &rectified1,
            descriptors1,
            n1,
            &rectified2,
            descriptors2,
            n2,
            affines1.unwrap(),
            affines2.unwrap(),
            desc_len,
            window_size,
            threshold,
            &geom,
            geometric_config.unwrap(),
        )
    } else {
        mutual_best_match_sweep(
            &rectified1,
            descriptors1,
            n1,
            &rectified2,
            descriptors2,
            n2,
            desc_len,
            window_size,
            threshold,
        )
    }
}

/// Match features across multiple image pairs in parallel using Rayon.
///
/// Each pair is processed independently using `match_image_pair`.
///
/// # Parameters
///
/// * `pairs` - List of (image_i_index, image_j_index) pairs to match
/// * `intrinsics` - Intrinsic matrix for each camera (indexed by camera_index)
/// * `rotations` - Per-image rotation matrix (cam_from_world)
/// * `translations` - Per-image translation vector (cam_from_world)
/// * `camera_indices` - Maps image index to camera index in `intrinsics`
/// * `positions` - Per-image feature positions (flat N×2 arrays)
/// * `descriptors` - Per-image feature descriptors (flat N×128 arrays)
/// * `num_features` - Number of features per image
/// * `widths` - Per-camera image width in pixels (indexed by camera_index)
/// * `heights` - Per-camera image height in pixels (indexed by camera_index)
/// * All other params same as match_image_pair
///
/// # Returns
///
/// Vec of match results, one per pair, in the same order as `pairs`.
#[allow(clippy::too_many_arguments)]
pub fn match_image_pairs_batch(
    pairs: &[(usize, usize)],
    intrinsics: &[Matrix3<f64>],
    rotations: &[Matrix3<f64>],
    translations: &[Vector3<f64>],
    camera_indices: &[usize],
    positions: &[&[f64]],
    descriptors: &[&[u8]],
    num_features: &[usize],
    widths: &[u32],
    heights: &[u32],
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    rectification_margin: u32,
    affines: Option<&[&[f64]]>,
    geometric_config: Option<&GeometricFilterConfig>,
) -> Vec<Vec<(usize, usize, f64)>> {
    pairs
        .par_iter()
        .map(|&(i, j)| {
            let cam_idx_i = camera_indices[i];
            let cam_idx_j = camera_indices[j];
            let k1 = &intrinsics[cam_idx_i];
            let k2 = &intrinsics[cam_idx_j];
            let r1 = &rotations[i];
            let r2 = &rotations[j];
            let t1 = &translations[i];
            let t2 = &translations[j];

            let aff1 = affines.map(|a| a[i]);
            let aff2 = affines.map(|a| a[j]);

            match_image_pair(
                k1,
                k2,
                r1,
                t1,
                r2,
                t2,
                widths[cam_idx_i],
                heights[cam_idx_i],
                widths[cam_idx_j],
                heights[cam_idx_j],
                positions[i],
                descriptors[i],
                num_features[i],
                positions[j],
                descriptors[j],
                num_features[j],
                desc_len,
                window_size,
                threshold,
                rectification_margin,
                aff1,
                aff2,
                geometric_config,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    fn test_intrinsics() -> Matrix3<f64> {
        Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0)
    }

    /// Smoke test: match_image_pair works with per-image dimensions and
    /// forward motion. The actual both-epipoles safety logic is tested in
    /// rectification::tests::test_rectification_safe_requires_both_epipoles_outside.
    #[test]
    fn test_match_image_pair_forward_motion_asymmetric_dimensions() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(0.0, 0.0, 1.0);

        // 4 features at the image corners, each with a unique descriptor.
        // Avoid the epipole (320, 240) where polar matching is degenerate.
        let n = 4;
        let positions = [
            100.0, 80.0, // top-left
            500.0, 80.0, // top-right
            100.0, 400.0, // bottom-left
            500.0, 400.0, // bottom-right
        ];
        let mut descriptors = vec![0u8; n * 128];
        for i in 0..n {
            descriptors[i * 128] = (i * 50) as u8;
        }

        let matches = match_image_pair(
            &k,
            &k,
            &r,
            &t1,
            &r,
            &t2,
            640,
            480, // image 1: normal size
            10,
            10, // image 2: different size
            &positions,
            &descriptors,
            n,
            &positions,
            &descriptors,
            n,
            128,
            5,
            None,
            50,
            None,
            None,
            None,
        );

        assert_eq!(
            matches.len(),
            n,
            "All {n} features should match (got {})",
            matches.len()
        );
        for (idx1, idx2, dist) in &matches {
            assert_eq!(idx1, idx2, "Feature {idx1} should match itself");
            assert_eq!(*dist, 0.0);
        }
    }
}
