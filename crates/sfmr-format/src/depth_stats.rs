// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Depth statistics computation for `.sfmr` reconstructions.
//!
//! Computes per-image depth statistics, depth histograms, and estimated
//! surface normals from camera poses, 3D points, and track observations.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use ndarray::{Array1, Array2};

use crate::types::{DepthStatistics, ImageDepthStats, ObservedDepthStats, SfmrError};

const NUM_HISTOGRAM_BUCKETS: u32 = 128;

/// Compute per-image camera centers in world coordinates from per-image camera extrinsics.
///
/// For world-to-camera transform `(R, t)`, camera center `C = -R^T * t`.
///
/// - `quaternions_wxyz`: `(N, 4)` per-image world-to-camera rotation quaternions
///   (WXYZ order).
/// - `translations_xyz`: `(N, 3)` per-image world-to-camera translation vectors.
///
/// Returns `(N, 3)` camera center positions in world coordinates.
fn compute_camera_centers(
    quaternions_wxyz: &Array2<f64>,
    translations_xyz: &Array2<f64>,
) -> Array2<f64> {
    let n = quaternions_wxyz.shape()[0];
    let mut centers = Array2::<f64>::zeros((n, 3));

    for i in 0..n {
        let qw = quaternions_wxyz[[i, 0]];
        let qx = quaternions_wxyz[[i, 1]];
        let qy = quaternions_wxyz[[i, 2]];
        let qz = quaternions_wxyz[[i, 3]];
        let q = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(qw, qx, qy, qz));
        let r = q.to_rotation_matrix();

        let t = Vector3::new(
            translations_xyz[[i, 0]],
            translations_xyz[[i, 1]],
            translations_xyz[[i, 2]],
        );
        let center = -(r.transpose() * t);
        centers[[i, 0]] = center.x;
        centers[[i, 1]] = center.y;
        centers[[i, 2]] = center.z;
    }

    centers
}

/// Compute estimated surface normals for 3D points from track observations.
///
/// For each 3D point, the normal is the average direction from the point
/// toward all cameras that observe it, normalized to a unit vector.
///
/// - `positions_xyz`: `(P, 3)` 3D point positions in world coordinates.
/// - `camera_centers`: `(N, 3)` per-image camera center in world coordinates,
///   derived from the per-image camera extrinsics (`C = -R^T * t`).
/// - `image_indexes`, `points3d_indexes`: parallel `(M,)` arrays representing
///   track observations. Each observation `i` means "image `image_indexes[i]`
///   sees point `points3d_indexes[i]`", where image indices index into
///   `camera_centers` and point indices index into `positions_xyz`.
fn compute_estimated_normals(
    positions_xyz: &Array2<f64>,
    camera_centers: &Array2<f64>,
    image_indexes: &Array1<u32>,
    points3d_indexes: &Array1<u32>,
) -> Array2<f32> {
    let num_points = positions_xyz.shape()[0];
    let num_obs = image_indexes.len();
    let mut normals = Array2::<f64>::zeros((num_points, 3));

    // Accumulate directions from each point toward its observing cameras
    for obs in 0..num_obs {
        let point_idx = points3d_indexes[obs] as usize;
        let image_idx = image_indexes[obs] as usize;

        // Direction from point toward camera
        for d in 0..3 {
            normals[[point_idx, d]] +=
                camera_centers[[image_idx, d]] - positions_xyz[[point_idx, d]];
        }
    }

    // Normalize to unit vectors
    let mut result = Array2::<f32>::zeros((num_points, 3));
    for i in 0..num_points {
        let nx = normals[[i, 0]];
        let ny = normals[[i, 1]];
        let nz = normals[[i, 2]];
        let norm = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-10);
        result[[i, 0]] = (nx / norm) as f32;
        result[[i, 1]] = (ny / norm) as f32;
        result[[i, 2]] = (nz / norm) as f32;
    }

    result
}

/// Compute z-depths of observed 3D points in a single camera's coordinate frame.
///
/// Returns only positive depths (points in front of the camera).
///
/// - `rotation_matrix`: `3x3` world-to-camera rotation for this image.
/// - `camera_center`: camera center in world coordinates for this image
///   (`C = -R^T * t`).
/// - `positions_xyz`: `(P, 3)` all 3D point positions in world coordinates.
/// - `observed_point_indices`: indices into `positions_xyz` for the points
///   observed by this image.
fn compute_observed_depths(
    rotation_matrix: &Matrix3<f64>,
    camera_center: &Vector3<f64>,
    positions_xyz: &Array2<f64>,
    observed_point_indices: &[u32],
) -> Vec<f64> {
    let mut depths = Vec::with_capacity(observed_point_indices.len());

    for &pt_idx in observed_point_indices {
        let pt_idx = pt_idx as usize;
        let px = positions_xyz[[pt_idx, 0]] - camera_center.x;
        let py = positions_xyz[[pt_idx, 1]] - camera_center.y;
        let pz = positions_xyz[[pt_idx, 2]] - camera_center.z;

        // z-component in camera coordinates = third row of R dot (point - center)
        let z = rotation_matrix[(2, 0)] * px
            + rotation_matrix[(2, 1)] * py
            + rotation_matrix[(2, 2)] * pz;

        if z > 0.0 {
            depths.push(z);
        }
    }

    depths
}

/// Compute a histogram of values within [min, max] using `num_buckets` bins.
///
/// Uses `np.linspace(min, max, num_buckets + 1)` equivalent bucket edges,
/// matching the Python implementation's `np.histogram` behavior.
fn compute_histogram(values: &[f64], min: f64, max: f64, num_buckets: usize) -> Vec<u32> {
    let mut counts = vec![0u32; num_buckets];

    if num_buckets == 0 || min >= max {
        return counts;
    }

    let range = max - min;
    for &v in values {
        let bucket = ((v - min) / range * num_buckets as f64) as usize;
        // Clamp to last bucket (values exactly at max go to the last bucket)
        let bucket = bucket.min(num_buckets - 1);
        counts[bucket] += 1;
    }

    counts
}

/// Compute median of a sorted slice.
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Result of depth statistics computation.
pub struct DepthStatsResult {
    /// Estimated surface normals `(P, 3)` float32.
    pub estimated_normals_xyz: Array2<f32>,
    /// Per-image depth statistics JSON structure.
    pub depth_statistics: DepthStatistics,
    /// Depth histogram counts `(N, num_buckets)` uint32.
    pub observed_depth_histogram_counts: Array2<u32>,
}

/// Compute depth statistics, normals, and histograms for a reconstruction.
///
/// From camera poses, 3D points, and track observations, computes:
/// - Estimated surface normals for each 3D point
/// - Per-image depth statistics (min/max/median/mean z-depth)
/// - Per-image depth histograms with [`NUM_HISTOGRAM_BUCKETS`] bins
///
/// - `quaternions_wxyz`: `(N, 4)` world-to-camera rotation quaternions (WXYZ order).
/// - `translations_xyz`: `(N, 3)` world-to-camera translation vectors.
/// - `positions_xyz`: `(P, 3)` 3D point positions in world coordinates.
/// - `image_indexes`, `points3d_indexes`: parallel `(M,)` arrays representing
///   track observations. Each observation `i` means "image `image_indexes[i]`
///   sees point `points3d_indexes[i]`", where image indices index into the
///   camera pose arrays and point indices index into `positions_xyz`.
pub fn compute_depth_statistics(
    quaternions_wxyz: &Array2<f64>,
    translations_xyz: &Array2<f64>,
    positions_xyz: &Array2<f64>,
    image_indexes: &Array1<u32>,
    points3d_indexes: &Array1<u32>,
) -> Result<DepthStatsResult, SfmrError> {
    let num_images = quaternions_wxyz.shape()[0];
    let num_points = positions_xyz.shape()[0];
    let num_buckets = NUM_HISTOGRAM_BUCKETS as usize;

    // Validate array shapes
    if quaternions_wxyz.shape()[1] != 4 {
        return Err(SfmrError::InvalidFormat(format!(
            "quaternions_wxyz shape {:?} doesn't match expected ({}, 4)",
            quaternions_wxyz.shape(),
            num_images
        )));
    }
    if translations_xyz.shape() != [num_images, 3] {
        return Err(SfmrError::InvalidFormat(format!(
            "translations_xyz shape {:?} doesn't match expected ({}, 3)",
            translations_xyz.shape(),
            num_images
        )));
    }
    if positions_xyz.shape()[1] != 3 {
        return Err(SfmrError::InvalidFormat(format!(
            "positions_xyz shape {:?} doesn't match expected ({}, 3)",
            positions_xyz.shape(),
            num_points
        )));
    }

    // Handle empty reconstructions
    if num_images == 0 || num_points == 0 {
        return Ok(DepthStatsResult {
            estimated_normals_xyz: Array2::zeros((num_points, 3)),
            depth_statistics: DepthStatistics {
                num_histogram_buckets: NUM_HISTOGRAM_BUCKETS,
                images: Vec::new(),
            },
            observed_depth_histogram_counts: Array2::zeros((num_images, num_buckets)),
        });
    }

    // Compute camera centers
    let camera_centers = compute_camera_centers(quaternions_wxyz, translations_xyz);

    // Compute estimated normals
    let estimated_normals_xyz = compute_estimated_normals(
        positions_xyz,
        &camera_centers,
        image_indexes,
        points3d_indexes,
    );

    // Build mapping: image_index -> set of observed point indices
    let mut image_to_points: Vec<Vec<u32>> = vec![Vec::new(); num_images];
    for obs in 0..image_indexes.len() {
        let img_idx = image_indexes[obs] as usize;
        let pt_idx = points3d_indexes[obs];
        // Use sorted insert to deduplicate (like Python's set)
        let points = &mut image_to_points[img_idx];
        if !points.contains(&pt_idx) {
            points.push(pt_idx);
        }
    }

    // Compute per-image statistics
    let mut observed_depth_histogram_counts = Array2::<u32>::zeros((num_images, num_buckets));
    let mut depth_stats_images = Vec::with_capacity(num_images);

    for img_idx in 0..num_images {
        let qw = quaternions_wxyz[[img_idx, 0]];
        let qx = quaternions_wxyz[[img_idx, 1]];
        let qy = quaternions_wxyz[[img_idx, 2]];
        let qz = quaternions_wxyz[[img_idx, 3]];
        let q = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(qw, qx, qy, qz));
        let r = q.to_rotation_matrix().into_inner();

        let center = Vector3::new(
            camera_centers[[img_idx, 0]],
            camera_centers[[img_idx, 1]],
            camera_centers[[img_idx, 2]],
        );

        let observed_points = &image_to_points[img_idx];
        let mut depths = compute_observed_depths(&r, &center, positions_xyz, observed_points);

        if depths.is_empty() {
            depth_stats_images.push(ImageDepthStats {
                histogram_min_z: None,
                histogram_max_z: None,
                observed: ObservedDepthStats {
                    count: 0,
                    min_z: None,
                    max_z: None,
                    median_z: None,
                    mean_z: None,
                },
            });
            continue;
        }

        // Sort for median computation
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = depths.len() as u32;
        let min_z = depths[0];
        let max_z = depths[depths.len() - 1];
        let median_z = median_sorted(&depths);
        let mean_z = depths.iter().sum::<f64>() / depths.len() as f64;

        // Compute histogram
        let histogram = compute_histogram(&depths, min_z, max_z, num_buckets);
        for (j, &c) in histogram.iter().enumerate() {
            observed_depth_histogram_counts[[img_idx, j]] = c;
        }

        depth_stats_images.push(ImageDepthStats {
            histogram_min_z: Some(min_z),
            histogram_max_z: Some(max_z),
            observed: ObservedDepthStats {
                count,
                min_z: Some(min_z),
                max_z: Some(max_z),
                median_z: Some(median_z),
                mean_z: Some(mean_z),
            },
        });
    }

    Ok(DepthStatsResult {
        estimated_normals_xyz,
        depth_statistics: DepthStatistics {
            num_histogram_buckets: NUM_HISTOGRAM_BUCKETS,
            images: depth_stats_images,
        },
        observed_depth_histogram_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_empty_reconstruction() {
        let q = Array2::<f64>::zeros((0, 4));
        let t = Array2::<f64>::zeros((0, 3));
        let p = Array2::<f64>::zeros((0, 3));
        let ii = Array1::<u32>::zeros(0);
        let pi = Array1::<u32>::zeros(0);

        let result = compute_depth_statistics(&q, &t, &p, &ii, &pi).unwrap();
        assert_eq!(result.estimated_normals_xyz.shape(), &[0, 3]);
        assert_eq!(result.depth_statistics.images.len(), 0);
        assert_eq!(result.observed_depth_histogram_counts.shape(), &[0, 128]);
    }

    #[test]
    fn test_single_camera_single_point() {
        // Camera at origin looking along +Z (identity rotation, zero translation)
        let mut q = Array2::<f64>::zeros((1, 4));
        q[[0, 0]] = 1.0; // w=1, identity quaternion
        let t = Array2::<f64>::zeros((1, 3));

        // Point at (0, 0, 5) - should be at depth 5
        let mut p = Array2::<f64>::zeros((1, 3));
        p[[0, 2]] = 5.0;

        let ii = Array1::from_vec(vec![0u32]);
        let pi = Array1::from_vec(vec![0u32]);

        let result = compute_depth_statistics(&q, &t, &p, &ii, &pi).unwrap();

        assert_eq!(result.estimated_normals_xyz.shape(), &[1, 3]);
        assert_eq!(result.depth_statistics.images.len(), 1);

        let stats = &result.depth_statistics.images[0];
        assert_eq!(stats.observed.count, 1);
        assert!((stats.observed.min_z.unwrap() - 5.0).abs() < 1e-10);
        assert!((stats.observed.max_z.unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_centers() {
        // Identity rotation, translation = [1, 2, 3]
        // Center = -R^T @ t = -[1, 2, 3]
        let mut q = Array2::<f64>::zeros((1, 4));
        q[[0, 0]] = 1.0;
        let mut t = Array2::<f64>::zeros((1, 3));
        t[[0, 0]] = 1.0;
        t[[0, 1]] = 2.0;
        t[[0, 2]] = 3.0;

        let centers = compute_camera_centers(&q, &t);
        assert!((centers[[0, 0]] - (-1.0)).abs() < 1e-10);
        assert!((centers[[0, 1]] - (-2.0)).abs() < 1e-10);
        assert!((centers[[0, 2]] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_uniform() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let hist = compute_histogram(&values, 0.0, 99.0, 10);
        // Each bucket should have 10 values
        assert_eq!(hist.len(), 10);
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_normals_point_between_cameras() {
        // Two cameras on opposite sides of a point
        let mut q = Array2::<f64>::zeros((2, 4));
        q[[0, 0]] = 1.0; // identity
        q[[1, 0]] = 1.0; // identity
        let mut t = Array2::<f64>::zeros((2, 3));
        // Camera 0 at x=-5, camera 1 at x=+5 (centers = -R^T @ t)
        t[[0, 0]] = 5.0; // center at (-5, 0, 0)
        t[[1, 0]] = -5.0; // center at (5, 0, 0)

        // Point at origin
        let p = Array2::<f64>::zeros((1, 3));
        let ii = Array1::from_vec(vec![0u32, 1]);
        let pi = Array1::from_vec(vec![0u32, 0]);

        let normals = compute_estimated_normals(&p, &compute_camera_centers(&q, &t), &ii, &pi);

        // Directions (-5,0,0)->(0,0,0) and (5,0,0)->(0,0,0) cancel out on x,
        // but the directions are from point TO camera, so (-5,0,0) and (5,0,0)
        // sum to (0,0,0) -> normalized to (0,0,0) with 1e-10 floor
        // Actually: camera centers are at (-5,0,0) and (5,0,0).
        // Direction from point(0,0,0) to camera(-5,0,0) = (-5,0,0)
        // Direction from point(0,0,0) to camera(5,0,0) = (5,0,0)
        // Sum = (0,0,0) -> near-zero, normalized to ~(0,0,0)
        let norm =
            (normals[[0, 0]].powi(2) + normals[[0, 1]].powi(2) + normals[[0, 2]].powi(2)).sqrt();
        // The result is a unit vector (or near-zero normalized)
        assert!(norm <= 1.0 + 1e-5);
    }
}