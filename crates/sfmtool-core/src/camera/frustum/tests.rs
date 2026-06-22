use super::*;
use approx::assert_relative_eq;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Identity rotation as row-major 3x3.
fn identity_rotation() -> [f64; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

/// Helper to build a test frustum with identity camera at origin.
fn make_test_frustum() -> ([f64; 24], [f64; 24], f64) {
    let center = [0.0, 0.0, 0.0];
    let r = identity_rotation();
    let (fx, fy, cx, cy) = (500.0, 500.0, 320.0, 240.0);
    let (w, h) = (640, 480);
    let (near, far) = (1.0, 10.0);

    let corners = compute_frustum_corners(&center, &r, fx, fy, cx, cy, w, h, near, far);
    let planes = compute_frustum_planes(&center, &corners);
    let volume = compute_frustum_volume(w, h, fx, fy, near, far);

    (corners, planes, volume)
}

#[test]
fn test_frustum_corners_identity_camera() {
    let center = [0.0, 0.0, 0.0];
    let r = identity_rotation();
    let (fx, fy, cx, cy) = (500.0, 500.0, 320.0, 240.0);
    let (w, h) = (640_u32, 480_u32);
    let (near, far) = (1.0, 10.0);

    let corners = compute_frustum_corners(&center, &r, fx, fy, cx, cy, w, h, near, far);

    // Near top-left corner (index 0): pixel (0,0)
    // x_norm = (0 - 320)/500 = -0.64, y_norm = (0 - 240)/500 = -0.48
    // dir = normalize([-0.64, -0.48, 1.0])
    // t_near = 1.0 / dir_z
    // At near plane z should be ~1.0
    let ntl = Vector3::new(corners[0], corners[1], corners[2]);
    assert_relative_eq!(ntl[2], 1.0, epsilon = 1e-10);

    // Near top-right corner (index 1): pixel (640, 0)
    // x_norm = (640 - 320)/500 = 0.64
    let ntr = Vector3::new(corners[3], corners[4], corners[5]);
    assert_relative_eq!(ntr[2], 1.0, epsilon = 1e-10);
    assert!(ntr[0] > 0.0, "Top-right x should be positive");
    assert!(ntl[0] < 0.0, "Top-left x should be negative");

    // Far corners should be at z ~ 10.0
    let ftl = Vector3::new(corners[12], corners[13], corners[14]);
    assert_relative_eq!(ftl[2], 10.0, epsilon = 1e-10);

    // Far corners should be 10x the near corners (linear scaling)
    assert_relative_eq!(ftl[0], ntl[0] * 10.0, epsilon = 1e-10);
    assert_relative_eq!(ftl[1], ntl[1] * 10.0, epsilon = 1e-10);
}

#[test]
fn test_frustum_planes_corners_inside() {
    let (corners, planes, _) = make_test_frustum();

    // All 8 corners should have non-negative signed distance to all 6 planes
    for ci in 0..8 {
        let px = corners[ci * 3];
        let py = corners[ci * 3 + 1];
        let pz = corners[ci * 3 + 2];

        for pi in 0..6 {
            let nx = planes[pi * 4];
            let ny = planes[pi * 4 + 1];
            let nz = planes[pi * 4 + 2];
            let d = planes[pi * 4 + 3];

            let dist = nx * px + ny * py + nz * pz + d;
            assert!(
                dist >= -1e-9,
                "Corner {} has negative distance {:.6e} to plane {}",
                ci,
                dist,
                pi
            );
        }
    }
}

#[test]
fn test_frustum_volume() {
    let (fx, fy) = (500.0, 500.0);
    let (w, h) = (640_u32, 480_u32);
    let (near, far) = (1.0, 10.0);

    let volume = compute_frustum_volume(w, h, fx, fy, near, far);

    // Manual calculation
    let near_w: f64 = (640.0 / 500.0) * 1.0;
    let near_h: f64 = (480.0 / 500.0) * 1.0;
    let far_w: f64 = (640.0 / 500.0) * 10.0;
    let far_h: f64 = (480.0 / 500.0) * 10.0;
    let a1 = near_w * near_h;
    let a2 = far_w * far_h;
    let expected = (9.0 / 3.0) * (a1 + a2 + (a1 * a2).sqrt());

    assert_relative_eq!(volume, expected, epsilon = 1e-10);
    assert!(volume > 0.0);
}

#[test]
fn test_points_in_frustum() {
    let (corners, planes, _) = make_test_frustum();

    // A point in the middle of the frustum should be inside
    // Camera looking along +Z, center at origin. Mid-z = 5.5
    let inside_point = [0.0, 0.0, 5.5];

    // A point far outside
    let outside_point = [100.0, 100.0, 100.0];

    let points = [
        inside_point[0],
        inside_point[1],
        inside_point[2],
        outside_point[0],
        outside_point[1],
        outside_point[2],
    ];

    let result = points_in_frustum(&points, 2, &planes);
    assert!(result[0], "Point at center of frustum should be inside");
    assert!(!result[1], "Point far away should be outside");

    // Also verify all corners are inside
    let corner_result = points_in_frustum(&corners, 8, &planes);
    for (i, &inside) in corner_result.iter().enumerate() {
        assert!(inside, "Corner {} should be inside the frustum", i);
    }
}

#[test]
fn test_frustums_can_intersect_identical() {
    let (corners, planes, _) = make_test_frustum();
    assert!(
        frustums_can_intersect(&corners, &planes, &corners, &planes),
        "Identical frustums must intersect"
    );
}

#[test]
fn test_frustums_can_intersect_separated() {
    let center_a = [0.0, 0.0, 0.0];
    let center_b = [1000.0, 0.0, 0.0];
    let r = identity_rotation();
    let (fx, fy, cx, cy) = (500.0, 500.0, 320.0, 240.0);
    let (w, h) = (640, 480);
    let (near, far) = (1.0, 10.0);

    let corners_a = compute_frustum_corners(&center_a, &r, fx, fy, cx, cy, w, h, near, far);
    let planes_a = compute_frustum_planes(&center_a, &corners_a);

    let corners_b = compute_frustum_corners(&center_b, &r, fx, fy, cx, cy, w, h, near, far);
    let planes_b = compute_frustum_planes(&center_b, &corners_b);

    assert!(
        !frustums_can_intersect(&corners_a, &planes_a, &corners_b, &planes_b),
        "Widely separated frustums should not intersect"
    );
}

#[test]
fn test_sample_points_all_inside() {
    let (corners, planes, _) = make_test_frustum();
    let mut rng = StdRng::seed_from_u64(42);

    let samples = sample_points_in_frustum(&corners, &planes, 500, &mut rng);
    let n = samples.len() / 3;
    assert!(n > 0, "Should have sampled some points");

    let inside = points_in_frustum(&samples, n, &planes);
    for (i, &is_in) in inside.iter().enumerate() {
        assert!(is_in, "Sampled point {} should be inside frustum", i);
    }
}

#[test]
fn test_intersection_volume_identical() {
    let (corners, planes, volume) = make_test_frustum();
    let mut rng = StdRng::seed_from_u64(123);

    let est =
        estimate_frustum_intersection_volume(&corners, &planes, volume, &planes, 10000, &mut rng);

    // Should be close to the full volume (within 30%)
    let lower = volume * 0.7;
    let upper = volume * 1.3;
    assert!(
        est >= lower && est <= upper,
        "Estimated volume {:.4} should be within 30% of actual volume {:.4}",
        est,
        volume
    );
}

// -----------------------------------------------------------------------
// Distorted frustum grid tests
// -----------------------------------------------------------------------

use crate::camera::intrinsics::{CameraIntrinsics, CameraModel};

#[test]
fn distorted_grid_pinhole_matches_corners() {
    // For a pinhole camera, the grid corners should match compute_frustum_corners
    let center = [0.0, 0.0, 0.0];
    let r = identity_rotation();
    let (fx, fy, cx, cy) = (500.0, 500.0, 320.0, 240.0);
    let far_z = 10.0;

    let camera = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: fx,
            focal_length_y: fy,
            principal_point_x: cx,
            principal_point_y: cy,
        },
        width: 640,
        height: 480,
    };

    let grid = compute_distorted_frustum_grid(&center, &r, &camera, far_z, 4);
    assert_eq!(grid.grid_size, 5);
    assert_eq!(grid.positions.len(), 5 * 5 * 3);

    let corners = compute_frustum_corners(&center, &r, fx, fy, cx, cy, 640, 480, 0.0, far_z);

    // Grid corner (0,0) = far TL = corners[12..15]
    let grid_tl = &grid.positions[0..3];
    // Grid corner (4,0) = far TR = corners[15..18]
    let grid_tr = &grid.positions[(4) * 3..(4) * 3 + 3];
    // Grid corner (4,4) = far BR = corners[18..21]
    let grid_br = &grid.positions[(4 * 5 + 4) * 3..(4 * 5 + 4) * 3 + 3];
    // Grid corner (0,4) = far BL = corners[21..24]
    let grid_bl = &grid.positions[(4 * 5) * 3..(4 * 5) * 3 + 3];

    // Note: compute_frustum_corners normalizes the direction, while
    // compute_distorted_frustum_grid uses (x*far_z, y*far_z, far_z).
    // The result differs slightly because frustum_corners normalizes first.
    // We compare the direction instead.
    for (grid_pt, corner_idx) in [
        (grid_tl, 4), // far TL
        (grid_tr, 5), // far TR
        (grid_br, 6), // far BR
        (grid_bl, 7), // far BL
    ] {
        let cx_pt = corners[corner_idx * 3];
        let cy_pt = corners[corner_idx * 3 + 1];
        let cz_pt = corners[corner_idx * 3 + 2];

        // Both should be along the same ray from center,
        // so their direction should match
        let grid_len =
            (grid_pt[0] * grid_pt[0] + grid_pt[1] * grid_pt[1] + grid_pt[2] * grid_pt[2]).sqrt();
        let corn_len = (cx_pt * cx_pt + cy_pt * cy_pt + cz_pt * cz_pt).sqrt();

        assert_relative_eq!(grid_pt[0] / grid_len, cx_pt / corn_len, epsilon = 1e-10);
        assert_relative_eq!(grid_pt[1] / grid_len, cy_pt / corn_len, epsilon = 1e-10);
        assert_relative_eq!(grid_pt[2] / grid_len, cz_pt / corn_len, epsilon = 1e-10);
    }
}

#[test]
fn distorted_grid_has_correct_size() {
    let center = [0.0, 0.0, 0.0];
    let r = identity_rotation();
    let camera = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };

    for subdivisions in [1, 4, 8, 16] {
        let grid = compute_distorted_frustum_grid(&center, &r, &camera, 5.0, subdivisions);
        let n = subdivisions + 1;
        assert_eq!(grid.grid_size, n);
        assert_eq!(grid.positions.len(), n * n * 3);
    }
}

#[test]
fn distorted_grid_all_at_far_z() {
    // For identity rotation pinhole camera, all grid z-coordinates should be far_z
    let center = [0.0, 0.0, 0.0];
    let r = identity_rotation();
    let far_z = 7.5;
    let camera = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };

    let grid = compute_distorted_frustum_grid(&center, &r, &camera, far_z, 4);
    let n = grid.grid_size;
    for j in 0..n {
        for i in 0..n {
            let z = grid.positions[(j * n + i) * 3 + 2];
            assert_relative_eq!(z, far_z, epsilon = 1e-10);
        }
    }
}

#[test]
fn distorted_grid_with_radial_distortion_differs() {
    // A camera with distortion should produce different corner positions
    // than a pinhole camera
    let center = [0.0, 0.0, 0.0];
    let r = identity_rotation();
    let far_z = 5.0;

    let pinhole = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };

    let distorted = CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
        },
        width: 640,
        height: 480,
    };

    let grid_pin = compute_distorted_frustum_grid(&center, &r, &pinhole, far_z, 4);
    let grid_dist = compute_distorted_frustum_grid(&center, &r, &distorted, far_z, 4);

    // Corner positions should differ due to distortion
    let n = grid_pin.grid_size;
    // Check corner (0,0) — top-left, off-center so distortion matters
    let pin_x = grid_pin.positions[0];
    let dist_x = grid_dist.positions[0];
    assert!(
        (pin_x - dist_x).abs() > 0.01,
        "Distorted grid corner should differ from pinhole: pin={pin_x}, dist={dist_x}"
    );

    // Center vertex should be similar (principal point → zero distortion)
    let mid = n / 2;
    // For this camera cx=320, width=640, so center of grid hits near principal point
    // but not exactly. The exact center grid vertex maps to pixel (320, 240) = principal point.
    let pin_center = &grid_pin.positions[(mid * n + mid) * 3..(mid * n + mid) * 3 + 3];
    let dist_center = &grid_dist.positions[(mid * n + mid) * 3..(mid * n + mid) * 3 + 3];
    // At principal point, distortion is identity
    assert_relative_eq!(pin_center[0], dist_center[0], epsilon = 0.1);
    assert_relative_eq!(pin_center[1], dist_center[1], epsilon = 0.1);
    assert_relative_eq!(pin_center[2], dist_center[2], epsilon = 1e-10);
}

#[test]
fn distorted_grid_fisheye_spherical_placement() {
    // For a fisheye camera, all grid vertices should be at distance far_z from center
    let center = [0.0, 0.0, 0.0];
    let r = identity_rotation();
    let far_z = 5.0;
    let camera = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 300.0,
            focal_length_y: 300.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.01,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        },
        width: 640,
        height: 480,
    };

    let grid = compute_distorted_frustum_grid(&center, &r, &camera, far_z, 8);
    let n = grid.grid_size;
    for j in 0..n {
        for i in 0..n {
            let idx = (j * n + i) * 3;
            let px = grid.positions[idx];
            let py = grid.positions[idx + 1];
            let pz = grid.positions[idx + 2];
            let dist = (px * px + py * py + pz * pz).sqrt();
            assert!(
                (dist - far_z).abs() < 1e-10,
                "Fisheye grid vertex ({i},{j}) should be at distance far_z from center, got {dist}"
            );
        }
    }
}

#[test]
fn distorted_grid_pinhole_not_spherical() {
    // For a pinhole camera, vertices should be on a flat plane at z = far_z (NOT spherical)
    let center = [0.0, 0.0, 0.0];
    let r = identity_rotation();
    let far_z = 5.0;
    let camera = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };

    let grid = compute_distorted_frustum_grid(&center, &r, &camera, far_z, 4);
    let n = grid.grid_size;
    // All z coordinates should be far_z (flat plane)
    for j in 0..n {
        for i in 0..n {
            let z = grid.positions[(j * n + i) * 3 + 2];
            assert_relative_eq!(z, far_z, epsilon = 1e-10);
        }
    }
    // Corner vertices should NOT all be at the same distance from center
    // (because they're on a flat plane, not a sphere)
    let corner_dist =
        (grid.positions[0].powi(2) + grid.positions[1].powi(2) + grid.positions[2].powi(2)).sqrt();
    let center_dist = {
        let mid = n / 2;
        let idx = (mid * n + mid) * 3;
        (grid.positions[idx].powi(2)
            + grid.positions[idx + 1].powi(2)
            + grid.positions[idx + 2].powi(2))
        .sqrt()
    };
    // Corner should be farther than center on a flat plane
    assert!(
            corner_dist > center_dist + 0.01,
            "Pinhole flat plane: corner dist ({corner_dist}) should be greater than center dist ({center_dist})"
        );
}
