// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::Point3;
use rand::seq::SliceRandom;

use super::gpu_types::{FALLBACK_POINT_SIZE, NN_SUBSAMPLE_COUNT};

/// Compute an automatic point size from nearest-neighbor distances.
///
/// Builds a KD-tree of all points, then queries NN distances for a random
/// subsample of up to `NN_SUBSAMPLE_COUNT` points. Returns
/// `median_nn_distance * 0.5` as the splat radius.
pub(super) fn compute_auto_point_size(points: &[sfmtool_core::Point3D]) -> f32 {
    // Points at infinity store a unit direction, not a location, so they would
    // cluster on the unit sphere and skew NN distances — exclude them.
    let positions: Vec<[f32; 3]> = points
        .iter()
        .filter(|p| !p.is_at_infinity())
        .map(|p| {
            [
                p.position.x as f32,
                p.position.y as f32,
                p.position.z as f32,
            ]
        })
        .collect();
    if positions.len() < 2 {
        return FALLBACK_POINT_SIZE;
    }

    // Build KD-tree from finite points (f32 for speed)
    let mut tree: KdTree<f32, 3> = KdTree::with_capacity(positions.len());
    for (i, p) in positions.iter().enumerate() {
        tree.add(p, i as u64);
    }

    // Subsample indices for NN queries
    let mut rng = rand::rng();
    let query_indices: Vec<usize> = if positions.len() <= NN_SUBSAMPLE_COUNT {
        (0..positions.len()).collect()
    } else {
        let mut indices: Vec<usize> = (0..positions.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(NN_SUBSAMPLE_COUNT);
        indices
    };

    // Query nearest neighbor for each subsampled point (k=2: self + nearest)
    let mut nn_distances: Vec<f32> = Vec::with_capacity(query_indices.len());
    for &idx in &query_indices {
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&positions[idx], 2);
        // The first result is the point itself (distance 0); take the second
        if neighbors.len() >= 2 {
            let dist = neighbors[1].distance.sqrt();
            if dist > 0.0 {
                nn_distances.push(dist);
            }
        }
    }

    if nn_distances.is_empty() {
        return FALLBACK_POINT_SIZE;
    }

    // 40th percentile NN distance
    nn_distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p40 = nn_distances[nn_distances.len() * 2 / 5];

    let auto_size = p40 * 1.2;
    log::info!(
        "Auto point size: {:.4} (p40 NN dist: {:.4}, from {} queries over {} finite points)",
        auto_size,
        p40,
        nn_distances.len(),
        positions.len()
    );

    auto_size
}

/// Compute a characteristic inter-camera distance from nearest-neighbor distances.
///
/// Builds a KD-tree of all camera centers, queries the NN distance for each
/// camera, and returns the 90th percentile. The high percentile makes the
/// result robust to a few cameras that happen to sit on top of each other
/// (e.g. colocated rig cameras), which would otherwise pull the value to zero.
///
/// Returns `None` if there are fewer than 2 images.
pub(super) fn compute_camera_nn_scale(images: &[sfmtool_core::SfmrImage]) -> Option<f32> {
    if images.len() < 2 {
        return None;
    }

    let mut tree: KdTree<f32, 3> = KdTree::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        let c = img.camera_center();
        tree.add(&[c.x as f32, c.y as f32, c.z as f32], i as u64);
    }

    let mut nn_distances: Vec<f32> = Vec::with_capacity(images.len());
    for img in images {
        let c = img.camera_center();
        let query = [c.x as f32, c.y as f32, c.z as f32];
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&query, 2);
        if neighbors.len() >= 2 {
            let dist = neighbors[1].distance.sqrt();
            if dist > 0.0 {
                nn_distances.push(dist);
            }
        }
    }

    if nn_distances.is_empty() {
        return None;
    }

    nn_distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p90 = nn_distances[nn_distances.len() * 9 / 10];

    log::info!(
        "Camera NN scale: {:.4} (p90 of {} NN distances from {} cameras)",
        p90,
        nn_distances.len(),
        images.len()
    );

    Some(p90)
}

/// Compute the bounding sphere (center, radius) for a set of 3D points.
///
/// Uses component-wise median for a robust center, then 80th percentile
/// distance from center as a robust radius. Handles outliers gracefully
/// since percentile-based statistics ignore extreme values.
///
/// Returns `(origin, 1.0)` if fewer than 2 points.
pub(super) fn compute_scene_bounds(points: &[sfmtool_core::Point3D]) -> (Point3<f64>, f64) {
    // Exclude points at infinity: their `position` is a unit direction, not a
    // location, and would pull the center toward the origin and distort the
    // radius (and hence the adaptive clip planes that depend on these bounds).
    let finite: Vec<&sfmtool_core::Point3D> =
        points.iter().filter(|p| !p.is_at_infinity()).collect();
    if finite.len() < 2 {
        return (Point3::origin(), 1.0);
    }

    // Collect coordinates
    let mut xs: Vec<f64> = finite.iter().map(|p| p.position.x).collect();
    let mut ys: Vec<f64> = finite.iter().map(|p| p.position.y).collect();
    let mut zs: Vec<f64> = finite.iter().map(|p| p.position.z).collect();

    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    zs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n = xs.len();
    let center = Point3::new(xs[n / 2], ys[n / 2], zs[n / 2]);

    // Compute distances from center and take 80th percentile
    let mut dists: Vec<f64> = finite
        .iter()
        .map(|p| (p.position - center).norm())
        .collect();
    dists.sort_unstable_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    let p80 = dists[n * 4 / 5].max(0.1);

    log::info!(
        "Scene bounds: center=[{:.2}, {:.2}, {:.2}], radius={:.2} (from {} points)",
        center.x,
        center.y,
        center.z,
        p80,
        n
    );

    (center, p80)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3 as NPoint3, UnitQuaternion, Vector3};
    use sfmtool_core::{Point3D, SfmrImage};

    fn make_point(x: f64, y: f64, z: f64) -> Point3D {
        Point3D {
            position: NPoint3::new(x, y, z),
            w: 1.0,
            color: [255, 255, 255],
            error: 0.0,
            normal: nalgebra::Vector3::new(0.0, 0.0, 1.0),
        }
    }

    /// A point at infinity: `w = 0` with a unit-length direction in `position`.
    fn make_infinity_point(dx: f64, dy: f64, dz: f64) -> Point3D {
        let dir = nalgebra::Vector3::new(dx, dy, dz).normalize();
        Point3D {
            position: NPoint3::from(dir),
            w: 0.0,
            color: [255, 255, 255],
            error: 0.0,
            normal: nalgebra::Vector3::zeros(),
        }
    }

    #[test]
    fn test_scene_bounds_ignores_infinity_points() {
        // A finite cluster far from the origin.
        let mut points: Vec<Point3D> = (0..20)
            .map(|i| make_point(100.0 + i as f64 * 0.1, 100.0, 100.0))
            .collect();
        let (center_finite, radius_finite) = compute_scene_bounds(&points);

        // Infinity points are unit directions near the origin; including them
        // would drag the median center toward 0 and change the radius.
        for k in 0..50 {
            let a = k as f64;
            points.push(make_infinity_point(a.cos(), a.sin(), 0.3));
        }
        let (center, radius) = compute_scene_bounds(&points);

        assert!(
            (center - center_finite).norm() < 1e-9,
            "infinity points moved the center: {center:?} vs {center_finite:?}"
        );
        assert!(
            (radius - radius_finite).abs() < 1e-9,
            "infinity points changed the radius: {radius} vs {radius_finite}"
        );
        // Sanity: the center stayed at the finite cluster, not the origin.
        assert!(center.x > 50.0, "center.x={}", center.x);
    }

    #[test]
    fn test_auto_point_size_ignores_infinity_points() {
        // Regular grid of finite points (deterministic: count < subsample cap).
        // A unique z per point keeps the KD-tree from seeing too many items
        // sharing one axis value.
        let mut points: Vec<Point3D> = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                points.push(make_point(i as f64, j as f64, (i * 10 + j) as f64 * 0.01));
            }
        }
        let size_finite = compute_auto_point_size(&points);

        for k in 0..40 {
            let a = k as f64 * 0.3;
            points.push(make_infinity_point(a.cos(), a.sin(), 1.0));
        }
        let size = compute_auto_point_size(&points);

        assert!(
            (size - size_finite).abs() < 1e-6,
            "infinity points changed the auto size: {size} vs {size_finite}"
        );
    }

    #[test]
    fn test_scene_bounds_empty() {
        let (center, radius) = compute_scene_bounds(&[]);
        assert_eq!(center, Point3::origin());
        assert_eq!(radius, 1.0);
    }

    #[test]
    fn test_scene_bounds_single_point() {
        let points = vec![make_point(5.0, 5.0, 5.0)];
        let (center, radius) = compute_scene_bounds(&points);
        assert_eq!(center, Point3::origin());
        assert_eq!(radius, 1.0);
    }

    #[test]
    fn test_scene_bounds_uniform_cube() {
        // 1000 points in a [-10, 10]^3 cube
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let x = -10.0 + 20.0 * i as f64 / 9.0;
                    let y = -10.0 + 20.0 * j as f64 / 9.0;
                    let z = -10.0 + 20.0 * k as f64 / 9.0;
                    points.push(make_point(x, y, z));
                }
            }
        }

        let (center, radius) = compute_scene_bounds(&points);
        // Center should be near origin
        assert!(center.x.abs() < 2.5, "center.x={}", center.x);
        assert!(center.y.abs() < 2.5, "center.y={}", center.y);
        assert!(center.z.abs() < 2.5, "center.z={}", center.z);
        // Radius should be roughly 10-17 (80th percentile of distances in unit cube)
        assert!(radius > 5.0, "radius={}", radius);
        assert!(radius < 20.0, "radius={}", radius);
    }

    #[test]
    fn test_scene_bounds_with_outliers() {
        // Cluster of points near origin + a few extreme outliers
        let mut points: Vec<Point3D> = (0..100)
            .map(|i| {
                let t = i as f64 / 99.0;
                make_point(t - 0.5, t - 0.5, 0.0)
            })
            .collect();
        // Add outliers
        points.push(make_point(1000.0, 0.0, 0.0));
        points.push(make_point(-1000.0, 0.0, 0.0));

        let (center, radius) = compute_scene_bounds(&points);
        // Center should still be near 0 (median is robust to outliers)
        assert!(center.x.abs() < 1.0, "center.x={}", center.x);
        // Radius at 80th percentile should be small-ish (not 1000)
        assert!(radius < 10.0, "radius={}", radius);
    }

    /// Create a camera at the given world-space position (identity rotation, looking along +Z).
    fn make_image_at(x: f64, y: f64, z: f64) -> SfmrImage {
        let q = UnitQuaternion::identity();
        // For identity rotation, t = -R * C = -C
        let t = Vector3::new(-x, -y, -z);
        SfmrImage {
            name: String::new(),
            camera_index: 0,
            quaternion_wxyz: q,
            translation_xyz: t,
            feature_tool_hash: [0; 16],
            sift_content_hash: [0; 16],
        }
    }

    #[test]
    fn test_camera_nn_scale_too_few() {
        assert_eq!(compute_camera_nn_scale(&[]), None);
        assert_eq!(
            compute_camera_nn_scale(&[make_image_at(0.0, 0.0, 0.0)]),
            None
        );
    }

    #[test]
    fn test_camera_nn_scale_uniform_line() {
        // 10 cameras spaced 2.0 apart along X
        let images: Vec<SfmrImage> = (0..10)
            .map(|i| make_image_at(i as f64 * 2.0, 0.0, 0.0))
            .collect();
        let scale = compute_camera_nn_scale(&images).unwrap();
        // All NN distances are 2.0, so p90 should be 2.0
        assert!((scale - 2.0).abs() < 0.01, "scale={}", scale);
    }

    #[test]
    fn test_camera_nn_scale_with_colocated() {
        // 10 cameras spaced 1.0 apart, plus 2 colocated at origin
        let mut images: Vec<SfmrImage> =
            (0..10).map(|i| make_image_at(i as f64, 0.0, 0.0)).collect();
        // Add 2 cameras sitting right on camera 0
        images.push(make_image_at(0.0, 0.0, 0.0));
        images.push(make_image_at(0.0, 0.0, 0.0));
        let scale = compute_camera_nn_scale(&images).unwrap();
        // The colocated cameras have NN distance ~0 but p90 should still be ~1.0
        assert!(scale > 0.5, "scale={}", scale);
        assert!(scale < 1.5, "scale={}", scale);
    }

    #[test]
    fn test_grid_step_power_of_10() {
        // Verify the power-of-10 snapping logic used by the adaptive grid
        for &(length_scale, expected_step) in &[
            (0.001, 0.01),
            (0.01, 0.1),
            (0.1, 1.0),
            (0.5, 1.0),
            (1.0, 10.0),
            (5.0, 10.0),
            (10.0, 100.0),
            (100.0, 1000.0),
        ] {
            let raw: f64 = length_scale * 5.0;
            let step = 10.0_f64.powf(raw.log10().round());
            assert!(
                (step - expected_step).abs() < 0.001,
                "length_scale={}, expected step={}, got step={}",
                length_scale,
                expected_step,
                step
            );
        }
    }
}
