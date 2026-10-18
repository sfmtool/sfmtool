// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

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

/// A rotation of `angle_deg` degrees about the unit `axis`.
fn axis_rotation(axis: Vector3<f64>, angle_deg: f64) -> Matrix3<f64> {
    *nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Unit::new_normalize(axis),
        angle_deg.to_radians(),
    )
    .matrix()
}

/// On the rectified path the sweep sorts by rectified positions, but the
/// geometric filter must read the raw positions: its `StereoPairGeometry`
/// holds the original cameras. Each feature pair here is an exact projection of
/// one 3D point, with affine shapes that agree with the poses, so every pair
/// passes the filter at the raw positions and `match_image_pair` must find
/// all of them under a tight size gate.
#[test]
fn test_match_image_pair_rectified_geometric_filter_reads_raw_positions() {
    let k = test_intrinsics();
    // Camera 1 at the origin; camera 2 one unit to the right, turned towards
    // the scene and rolled, so rectification moves every point well away from
    // its raw position.
    let r1 = Matrix3::identity();
    let t1 = Vector3::zeros();
    let r2 = axis_rotation(Vector3::new(0.0, 0.0, 1.0), 12.0)
        * axis_rotation(Vector3::new(0.0, 1.0, 0.0), -10.0);
    let c2 = Vector3::new(1.0, 0.0, 0.0);
    let t2 = -(r2 * c2);
    let geom = StereoPairGeometry::new(&k, &k, &r1, &r2, &t1, &t2);

    let mut points = Vec::new();
    for iy in 0..4 {
        for ix in 0..5 {
            let depth = 3.0 + 0.7 * ((ix + 2 * iy) % 5) as f64;
            let x = (ix as f64 - 2.0) * 0.25 * depth;
            let y = (iy as f64 - 1.5) * 0.25 * depth;
            points.push(Vector3::new(x / 2.0, y / 2.0, depth));
        }
    }
    let n = points.len();

    let project = |r: &Matrix3<f64>, t: &Vector3<f64>, p: &Vector3<f64>| {
        let x = k * (r * p + t);
        [x[0] / x[2], x[1] / x[2]]
    };
    let base_shape = [3.0, -1.0, 1.0, 3.0];
    let rotated = [
        geom.r_2d[0] * base_shape[0] + geom.r_2d[1] * base_shape[2],
        geom.r_2d[0] * base_shape[1] + geom.r_2d[1] * base_shape[3],
        geom.r_2d[2] * base_shape[0] + geom.r_2d[3] * base_shape[2],
        geom.r_2d[2] * base_shape[1] + geom.r_2d[3] * base_shape[3],
    ];
    let mut positions1 = Vec::new();
    let mut positions2 = Vec::new();
    let mut affines1 = Vec::new();
    let mut affines2 = Vec::new();
    let mut descriptors = vec![0u8; n * 128];
    for (i, p) in points.iter().enumerate() {
        positions1.extend_from_slice(&project(&r1, &t1, p));
        positions2.extend_from_slice(&project(&r2, &t2, p));
        let d1 = (r1 * p + t1)[2];
        let d2 = (r2 * p + t2)[2];
        assert!(d2 > 0.0);
        affines1.extend_from_slice(&base_shape);
        affines2.extend(rotated.iter().map(|a| a * d1 / d2));
        descriptors[i * 128 + i] = 200;
    }

    let config = GeometricFilterConfig {
        max_angle_difference: 5.0,
        min_triangulation_angle: 1.0,
        geometric_size_ratio_min: 0.98,
        geometric_size_ratio_max: 1.02,
    };

    // The filter's mask for each true pair: all pass at the raw positions, and
    // at least one fails at the rectified positions, so the end-to-end check
    // below can tell which positions the rectified path hands the filter.
    let pair_passes = |p1: &[f64], p2: &[f64], i: usize| {
        two_stage_geometric_filter(
            [p1[i * 2], p1[i * 2 + 1]],
            &affines1[i * 4..i * 4 + 4].try_into().unwrap(),
            &p2[i * 2..i * 2 + 2],
            &affines2[i * 4..i * 4 + 4],
            1,
            &geom,
            &config,
        )[0]
    };
    assert!((0..n).all(|i| pair_passes(&positions1, &positions2, i)));

    let f =
        crate::camera::epipolar::compute_fundamental_matrix(&k, &r1, &t1, &k, &r2, &t2).unwrap();
    assert!(
        crate::camera::rectification::check_rectification_safe_from_f(&f, 640, 480, 50)
            && crate::camera::rectification::check_rectification_safe_from_f(
                &f.transpose(),
                640,
                480,
                50
            ),
        "the pair must take the rectified path"
    );
    let r_rel = r2 * r1.transpose();
    let t_rel = t2 - r_rel * t1;
    let rect = crate::camera::rectification::compute_stereo_rectification(
        &k, &k, &r_rel, &t_rel, 640, 480,
    );
    let k_inv = k.try_inverse().unwrap();
    let rectified1 = crate::camera::rectification::rectify_points(
        &positions1,
        n,
        &k_inv,
        &rect.r1_rect,
        &rect.p1,
    );
    let rectified2 = crate::camera::rectification::rectify_points(
        &positions2,
        n,
        &k_inv,
        &rect.r2_rect,
        &rect.p2,
    );
    assert!((0..n).any(|i| !pair_passes(&rectified1, &rectified2, i)));

    let mut matches = match_image_pair(
        &k,
        &k,
        &r1,
        &t1,
        &r2,
        &t2,
        640,
        480,
        640,
        480,
        &positions1,
        &descriptors,
        n,
        &positions2,
        &descriptors,
        n,
        128,
        n,
        None,
        50,
        Some(&affines1),
        Some(&affines2),
        Some(&config),
    );
    matches.sort_by_key(|m| m.0);
    let pairs: Vec<(usize, usize)> = matches.iter().map(|m| (m.0, m.1)).collect();
    let expected: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    assert_eq!(pairs, expected);
}
