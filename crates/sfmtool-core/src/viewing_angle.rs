// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Compute which 3D points to keep based on minimum viewing angle threshold.
///
/// For each point, computes the maximum angle between any pair of viewing rays
/// from observing cameras. Points with max angle >= min_angle_rad are kept.
///
/// # Arguments
/// * `quaternions_wxyz` - Flat slice of camera quaternions [w0,x0,y0,z0, w1,...], length num_images*4
/// * `translations` - Flat slice of camera translations [tx0,ty0,tz0, ...], length num_images*3
/// * `num_images` - Number of camera images
/// * `positions` - Flat slice of 3D point positions [x0,y0,z0, ...], length num_points*3
/// * `num_points` - Number of 3D points
/// * `track_point_ids` - Point ID for each track observation
/// * `track_image_indexes` - Image index for each track observation
/// * `min_angle_rad` - Minimum angle threshold in radians
///
/// # Returns
/// Boolean vec of length num_points. true = keep, false = remove.
#[allow(clippy::too_many_arguments)]
pub fn compute_narrow_track_mask(
    quaternions_wxyz: &[f64],
    translations: &[f64],
    num_images: usize,
    positions: &[f64],
    num_points: usize,
    track_point_ids: &[u32],
    track_image_indexes: &[u32],
    min_angle_rad: f64,
) -> Vec<bool> {
    // Step 1: Compute camera centers for all images
    let camera_centers: Vec<Vector3<f64>> = (0..num_images)
        .map(|i| {
            let qo = i * 4;
            let to = i * 3;
            let quat = UnitQuaternion::new_normalize(Quaternion::new(
                quaternions_wxyz[qo],
                quaternions_wxyz[qo + 1],
                quaternions_wxyz[qo + 2],
                quaternions_wxyz[qo + 3],
            ));
            let t = Vector3::new(translations[to], translations[to + 1], translations[to + 2]);
            let rot = quat.to_rotation_matrix();
            // Camera center: C = -R^T @ t
            -(rot.transpose() * t)
        })
        .collect();

    // Step 2: Build point -> list of observing image indices
    let mut point_observations: Vec<Vec<u32>> = vec![Vec::new(); num_points];
    for (point_id, &img_idx) in track_point_ids.iter().zip(track_image_indexes.iter()) {
        let pid = *point_id as usize;
        if pid < num_points {
            point_observations[pid].push(img_idx);
        }
    }

    // Step 3: For each point, compute max viewing angle and decide keep/remove
    // Use cos_threshold optimization: comparing dot products avoids acos
    let cos_threshold = min_angle_rad.cos();

    let mut keep = vec![false; num_points];
    for point_id in 0..num_points {
        let obs = &point_observations[point_id];
        if obs.len() < 2 {
            continue;
        }

        let px = positions[point_id * 3];
        let py = positions[point_id * 3 + 1];
        let pz = positions[point_id * 3 + 2];
        let point_pos = Vector3::new(px, py, pz);

        // Compute normalized viewing rays
        let mut rays: Vec<Vector3<f64>> = Vec::with_capacity(obs.len());
        for &img_idx in obs {
            let center = &camera_centers[img_idx as usize];
            let ray = point_pos - center;
            let norm = ray.norm();
            if norm > 0.0 {
                rays.push(ray / norm);
            }
        }

        if rays.len() < 2 {
            continue;
        }

        // Find minimum dot product (= maximum angle) among all ray pairs
        // If min_dot <= cos_threshold, the max angle >= min_angle_rad
        let mut min_dot = 1.0_f64;
        'outer: for i in 0..rays.len() {
            for j in (i + 1)..rays.len() {
                let dot = rays[i].dot(&rays[j]);
                if dot < min_dot {
                    min_dot = dot;
                    // Early exit: if we already found a pair exceeding the threshold
                    if min_dot <= cos_threshold {
                        break 'outer;
                    }
                }
            }
        }

        if min_dot <= cos_threshold {
            keep[point_id] = true;
        }
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: create identity quaternion (w=1, x=0, y=0, z=0)
    fn identity_quat() -> [f64; 4] {
        [1.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn test_cameras_at_same_position_filtered() {
        // Two cameras at the same position looking at a point - 0 angle, should be filtered
        let quaternions = [identity_quat(), identity_quat()].concat();
        let translations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let positions = [0.0, 0.0, 5.0]; // point in front
        let track_point_ids = [0_u32, 0];
        let track_image_indexes = [0_u32, 1];

        let keep = compute_narrow_track_mask(
            &quaternions,
            &translations,
            2,
            &positions,
            1,
            &track_point_ids,
            &track_image_indexes,
            0.1, // any positive threshold
        );

        assert_eq!(keep.len(), 1);
        assert!(
            !keep[0],
            "Point observed from same position should be filtered"
        );
    }

    #[test]
    fn test_cameras_90_degrees_apart_kept() {
        // Camera 0: at origin, identity rotation -> center at (0,0,0)
        // Camera 1: rotated 90° around Y, translated so center is at (5,0,0)
        // Point at (0,0,5)
        //
        // Ray from cam0: (0,0,5) - (0,0,0) = (0,0,1)
        // Ray from cam1: (0,0,5) - (5,0,0) = (-5,0,5), normalized ~ (-0.707, 0, 0.707)
        // Dot product: 0.707, angle ~ 45°
        //
        // Use identity for both cameras but place them apart via translation
        // Camera center = -R^T @ t, so t = -R @ center
        // Cam0: center=(0,0,0), R=I, t=(0,0,0)
        // Cam1: center=(5,0,0), R=I, t=(-5,0,0)

        let quaternions = [identity_quat(), identity_quat()].concat();
        let translations = [0.0, 0.0, 0.0, -5.0, 0.0, 0.0];
        let positions = [0.0, 0.0, 5.0];
        let track_point_ids = [0_u32, 0];
        let track_image_indexes = [0_u32, 1];

        // Threshold at 30° (pi/6) - the actual angle is ~45°, so point should be kept
        let keep = compute_narrow_track_mask(
            &quaternions,
            &translations,
            2,
            &positions,
            1,
            &track_point_ids,
            &track_image_indexes,
            PI / 6.0,
        );

        assert!(
            keep[0],
            "Point with ~45° angle should be kept with 30° threshold"
        );

        // Threshold at 80° - the actual angle is ~45°, so point should be filtered
        let keep2 = compute_narrow_track_mask(
            &quaternions,
            &translations,
            2,
            &positions,
            1,
            &track_point_ids,
            &track_image_indexes,
            80.0_f64.to_radians(),
        );

        assert!(
            !keep2[0],
            "Point with ~45° angle should be filtered with 80° threshold"
        );
    }

    #[test]
    fn test_single_observation_removed() {
        let quaternions = identity_quat().to_vec();
        let translations = [0.0, 0.0, 0.0];
        let positions = [1.0, 2.0, 3.0];
        let track_point_ids = [0_u32];
        let track_image_indexes = [0_u32];

        let keep = compute_narrow_track_mask(
            &quaternions,
            &translations,
            1,
            &positions,
            1,
            &track_point_ids,
            &track_image_indexes,
            0.01,
        );

        assert!(
            !keep[0],
            "Point with single observation should always be removed"
        );
    }

    #[test]
    fn test_cos_threshold_matches_acos() {
        // Verify that the cos-threshold optimization gives the same result as computing acos
        // Set up cameras with known geometry
        let quaternions = [identity_quat(), identity_quat()].concat();
        let translations = [0.0, 0.0, 0.0, -10.0, 0.0, 0.0];
        let positions = [0.0, 0.0, 10.0];
        let track_point_ids = [0_u32, 0];
        let track_image_indexes = [0_u32, 1];

        // The actual angle: atan2(10, 10) = 45°
        // Test at thresholds just below and above
        let threshold_below = 44.0_f64.to_radians();
        let threshold_above = 46.0_f64.to_radians();

        let keep_below = compute_narrow_track_mask(
            &quaternions,
            &translations,
            2,
            &positions,
            1,
            &track_point_ids,
            &track_image_indexes,
            threshold_below,
        );
        let keep_above = compute_narrow_track_mask(
            &quaternions,
            &translations,
            2,
            &positions,
            1,
            &track_point_ids,
            &track_image_indexes,
            threshold_above,
        );

        assert!(
            keep_below[0],
            "Should keep with threshold below actual angle"
        );
        assert!(
            !keep_above[0],
            "Should filter with threshold above actual angle"
        );
    }

    #[test]
    fn test_multiple_points_mixed() {
        // 3 cameras, 3 points with different observation patterns
        let quaternions = [identity_quat(), identity_quat(), identity_quat()].concat();
        // Cam0 at origin, Cam1 at (10,0,0), Cam2 at (0,10,0)
        let translations = [0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, -10.0, 0.0];
        // Point 0 at (0,0,1) - very close, large angle from cam0+cam1
        // Point 1 at (0,0,100) - very far, small angle
        // Point 2 at (5,5,5) - moderate
        let positions = [0.0, 0.0, 1.0, 0.0, 0.0, 100.0, 5.0, 5.0, 5.0];

        let track_point_ids = [0_u32, 0, 1, 1, 2, 2];
        let track_image_indexes = [0_u32, 1, 0, 1, 0, 2];

        let keep = compute_narrow_track_mask(
            &quaternions,
            &translations,
            3,
            &positions,
            3,
            &track_point_ids,
            &track_image_indexes,
            10.0_f64.to_radians(),
        );

        assert_eq!(keep.len(), 3);
        // Point 0: angle ~ atan2(10, 1) ~ 84° - should be kept
        assert!(keep[0]);
        // Point 1: angle ~ atan2(10, 100) ~ 5.7° - should be filtered
        assert!(!keep[1]);
    }
}