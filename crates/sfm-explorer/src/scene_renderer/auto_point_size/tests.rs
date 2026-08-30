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
    let mut images: Vec<SfmrImage> = (0..10).map(|i| make_image_at(i as f64, 0.0, 0.0)).collect();
    // Add 2 cameras sitting right on camera 0
    images.push(make_image_at(0.0, 0.0, 0.0));
    images.push(make_image_at(0.0, 0.0, 0.0));
    let scale = compute_camera_nn_scale(&images).unwrap();
    // The colocated cameras have NN distance ~0 but p90 should still be ~1.0
    assert!(scale > 0.5, "scale={}", scale);
    assert!(scale < 1.5, "scale={}", scale);
}

#[test]
fn test_camera_nn_scale_all_coincident() {
    // A rotation-only reconstruction stores every camera at the origin.
    // More than 32 identical centers used to overflow kiddo's fixed leaf
    // bucket and panic on insert; now they collapse to one entry and the
    // scale is None (no positive NN distance exists).
    let images: Vec<SfmrImage> = (0..148).map(|_| make_image_at(0.0, 0.0, 0.0)).collect();
    assert_eq!(compute_camera_nn_scale(&images), None);
}

#[test]
fn test_camera_nn_scale_many_colocated_plus_spread() {
    // 40 cameras on one spot (past the kiddo bucket size) plus a spaced line:
    // insertion must not panic and the spaced cameras still set the scale.
    let mut images: Vec<SfmrImage> = (0..40).map(|_| make_image_at(0.0, 0.0, 0.0)).collect();
    images.extend((1..=10).map(|i| make_image_at(i as f64, 0.0, 0.0)));
    let scale = compute_camera_nn_scale(&images).unwrap();
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

/// A cubic grid of `side^3` points spaced `spacing` apart: every point's
/// nearest neighbor is exactly `spacing` away.
fn make_grid(side: usize, spacing: f64) -> Vec<Point3D> {
    let mut points = Vec::with_capacity(side * side * side);
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                points.push(make_point(
                    i as f64 * spacing,
                    j as f64 * spacing,
                    k as f64 * spacing,
                ));
            }
        }
    }
    points
}

#[test]
fn test_auto_point_size_clean_grid() {
    // Every NN distance is the grid spacing, so nothing sits above the
    // isolation bar and the trim is a no-op: the size is SPLAT_RADIUS_FACTOR
    // times the spacing.
    let points = make_grid(6, 2.0);
    let size = compute_auto_point_size(&points);
    assert!((size - 2.2).abs() < 1e-4, "size={size}");
}

#[test]
fn test_auto_point_size_ignores_scattered_population() {
    // The motivating case: a dense grid plus a scattered sub-population whose
    // own spacing runs over two orders of magnitude (points strung out along
    // rays through empty space). The scattered points are the majority here,
    // so the plain median lands inside them, several times the grid spacing.
    let grid = make_grid(6, 2.0);
    let clean = compute_auto_point_size(&grid);

    let mut points = grid;
    let mut x = 1000.0_f64;
    let mut gap = 4.0_f64;
    for _ in 0..300 {
        points.push(make_point(x, 0.0, 0.0));
        x += gap;
        gap *= 1.02;
    }

    // What the plain median of the same distance set would have produced.
    let mut nn: Vec<f32> = vec![2.0; 216];
    let mut g = 4.0_f32;
    for _ in 0..300 {
        nn.push(g);
        g *= 1.02;
    }
    nn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let plain_median_size = nn[nn.len() / 2] * SPLAT_RADIUS_FACTOR;
    assert!(
        plain_median_size > 3.0 * clean,
        "the scattered population must inflate the plain median for this test \
         to mean anything: plain={plain_median_size}, clean={clean}"
    );

    let size = compute_auto_point_size(&points);
    assert!(
        (size - clean).abs() < 1e-5,
        "the scattered population moved the size: {size} vs {clean} (plain \
         median would give {plain_median_size})"
    );
}

#[test]
fn test_trimmed_median_no_op_on_tight_distribution() {
    let tight: Vec<f32> = (0..101).map(|i| 1.0 + i as f32 * 0.001).collect();
    let m = iteratively_trimmed_median(&tight);
    assert!((m - tight[50]).abs() < 1e-6, "m={m}");
}

#[test]
fn test_trimmed_median_reaches_a_stable_fixpoint() {
    // A dense block plus a broadly spread tail, the shape the trim exists for.
    let mut distances: Vec<f32> = vec![2.0; 216];
    let mut g = 4.0_f32;
    for _ in 0..300 {
        distances.push(g);
        g *= 1.02;
    }
    distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let m = iteratively_trimmed_median(&distances);
    assert!(
        (m - 2.0).abs() < 1e-6,
        "trim did not reach the dense block: {m}"
    );

    // Re-running the trim on the set it converged to changes nothing.
    let kept: Vec<f32> = distances
        .iter()
        .copied()
        .filter(|&d| d <= m * NN_ISOLATION_FACTOR)
        .collect();
    assert!(kept.len() < distances.len(), "nothing was trimmed at all");
    let m2 = iteratively_trimmed_median(&kept);
    assert_eq!(m, m2, "the fixpoint moved on a second application");
}

#[test]
fn test_auto_point_size_fallback_on_degenerate_input() {
    assert_eq!(compute_auto_point_size(&[]), FALLBACK_POINT_SIZE);
    assert_eq!(
        compute_auto_point_size(&[make_point(1.0, 2.0, 3.0)]),
        FALLBACK_POINT_SIZE
    );
    // Coincident points yield only zero NN distances, which are discarded.
    assert_eq!(
        compute_auto_point_size(&[make_point(1.0, 2.0, 3.0), make_point(1.0, 2.0, 3.0)]),
        FALLBACK_POINT_SIZE
    );
    // Points at infinity are excluded before the count check.
    assert_eq!(
        compute_auto_point_size(&[
            make_infinity_point(1.0, 0.0, 0.0),
            make_infinity_point(0.0, 1.0, 0.0),
            make_point(1.0, 2.0, 3.0),
        ]),
        FALLBACK_POINT_SIZE
    );
}
