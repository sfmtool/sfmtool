// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};

/// Normalized world-frame rays from each observing camera toward `point`, each
/// paired with the image index it came from.
///
/// A camera coincident with `point` yields a zero-length ray and is dropped, so
/// the result may be shorter than `observing_images`.
pub fn viewing_rays(
    point: Point3<f64>,
    camera_centers: &[Point3<f64>],
    observing_images: impl IntoIterator<Item = usize>,
) -> Vec<(usize, Vector3<f64>)> {
    observing_images
        .into_iter()
        .filter_map(|img| {
            let ray = point.coords - camera_centers[img].coords;
            let norm = ray.norm();
            (norm > 0.0).then_some((img, ray / norm))
        })
        .collect()
}

/// Maximum angle (radians) between any pair of viewing rays produced by
/// [`viewing_rays`]; `0.0` when fewer than two rays are supplied.
pub fn max_viewing_angle(rays: &[(usize, Vector3<f64>)]) -> f64 {
    let mut min_dot = 1.0_f64;
    for i in 0..rays.len() {
        for j in (i + 1)..rays.len() {
            min_dot = min_dot.min(rays[i].1.dot(&rays[j].1));
        }
    }
    min_dot.clamp(-1.0, 1.0).acos()
}

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
/// * `track_point_indexes` - Point ID for each track observation
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
    track_point_indexes: &[u32],
    track_image_indexes: &[u32],
    min_angle_rad: f64,
) -> Vec<bool> {
    // Step 1: Compute camera centers for all images
    let camera_centers: Vec<Point3<f64>> = (0..num_images)
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
            Point3::from(-(rot.transpose() * t))
        })
        .collect();

    // Step 2: Build point -> list of observing image indices
    let mut point_observations: Vec<Vec<u32>> = vec![Vec::new(); num_points];
    for (point_id, &img_idx) in track_point_indexes.iter().zip(track_image_indexes.iter()) {
        let pid = *point_id as usize;
        if pid < num_points {
            point_observations[pid].push(img_idx);
        }
    }

    // Step 3: For each point, keep it when its maximum viewing angle (the
    // largest angle between any pair of observation rays) meets the threshold.
    let mut keep = vec![false; num_points];
    for point_id in 0..num_points {
        let obs = &point_observations[point_id];
        if obs.len() < 2 {
            continue;
        }

        let point_pos = Point3::new(
            positions[point_id * 3],
            positions[point_id * 3 + 1],
            positions[point_id * 3 + 2],
        );
        let rays = viewing_rays(
            point_pos,
            &camera_centers,
            obs.iter().map(|&img_idx| img_idx as usize),
        );
        if rays.len() < 2 {
            continue;
        }

        if max_viewing_angle(&rays) >= min_angle_rad {
            keep[point_id] = true;
        }
    }

    keep
}

#[cfg(test)]
mod tests;
