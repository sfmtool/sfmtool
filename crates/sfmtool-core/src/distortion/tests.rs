use super::*;
use approx::assert_relative_eq;

// -----------------------------------------------------------------------
// Test camera constructors (reused from camera_intrinsics tests)
// -----------------------------------------------------------------------

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

fn simple_pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

fn simple_radial() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
        },
        width: 640,
        height: 480,
    }
}

fn radial() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Radial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
        },
        width: 640,
        height: 480,
    }
}

fn opencv() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCV {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.002,
        },
        width: 640,
        height: 480,
    }
}

fn opencv_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
            radial_distortion_k3: 0.01,
            radial_distortion_k4: -0.005,
        },
        width: 640,
        height: 480,
    }
}

fn full_opencv() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::FullOpenCV {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.002,
            radial_distortion_k3: 0.01,
            radial_distortion_k4: -0.005,
            radial_distortion_k5: 0.002,
            radial_distortion_k6: -0.001,
        },
        width: 640,
        height: 480,
    }
}

fn simple_radial_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
        },
        width: 640,
        height: 480,
    }
}

fn radial_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::RadialFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.02,
        },
        width: 640,
        height: 480,
    }
}

fn thin_prism_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::ThinPrismFisheye {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.01,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.001,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
            thin_prism_sx1: 0.002,
            thin_prism_sy1: -0.001,
        },
        width: 640,
        height: 480,
    }
}

fn rad_tan_thin_prism_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::RadTanThinPrismFisheye {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k0: 0.03,
            radial_distortion_k1: -0.01,
            radial_distortion_k2: 0.005,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
            radial_distortion_k5: 0.0,
            tangential_distortion_p0: 0.001,
            tangential_distortion_p1: -0.001,
            thin_prism_s0: 0.001,
            thin_prism_s1: 0.0,
            thin_prism_s2: -0.001,
            thin_prism_s3: 0.0,
        },
        width: 640,
        height: 480,
    }
}

fn equirectangular() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Equirectangular {
            focal_length_x: 640.0 / (2.0 * std::f64::consts::PI),
            focal_length_y: 480.0 / std::f64::consts::PI,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

fn all_cameras() -> Vec<CameraIntrinsics> {
    vec![
        pinhole(),
        simple_pinhole(),
        simple_radial(),
        radial(),
        opencv(),
        opencv_fisheye(),
        simple_radial_fisheye(),
        radial_fisheye(),
        thin_prism_fisheye(),
        rad_tan_thin_prism_fisheye(),
        full_opencv(),
        equirectangular(),
    ]
}

// -----------------------------------------------------------------------
// Pinhole: distort/undistort are identity
// -----------------------------------------------------------------------

#[test]
fn pinhole_distort_is_identity() {
    for cam in [pinhole(), simple_pinhole()] {
        let (xd, yd) = cam.model.distort(0.3, -0.4);
        assert_relative_eq!(xd, 0.3, epsilon = 1e-15);
        assert_relative_eq!(yd, -0.4, epsilon = 1e-15);
    }
}

#[test]
fn pinhole_undistort_is_identity() {
    for cam in [pinhole(), simple_pinhole()] {
        let (x, y) = cam.model.undistort(0.3, -0.4);
        assert_relative_eq!(x, 0.3, epsilon = 1e-15);
        assert_relative_eq!(y, -0.4, epsilon = 1e-15);
    }
}

// -----------------------------------------------------------------------
// Origin: all models should be identity at (0, 0)
// -----------------------------------------------------------------------

#[test]
fn distort_at_origin_is_identity() {
    for cam in all_cameras() {
        let (xd, yd) = cam.model.distort(0.0, 0.0);
        assert_relative_eq!(xd, 0.0, epsilon = 1e-15);
        assert_relative_eq!(yd, 0.0, epsilon = 1e-15);
    }
}

#[test]
fn undistort_at_origin_is_identity() {
    for cam in all_cameras() {
        let (x, y) = cam.model.undistort(0.0, 0.0);
        assert_relative_eq!(x, 0.0, epsilon = 1e-15);
        assert_relative_eq!(y, 0.0, epsilon = 1e-15);
    }
}

// -----------------------------------------------------------------------
// Round-trip: undistort(distort(x, y)) ≈ (x, y) for all models
// -----------------------------------------------------------------------

/// Test points spanning a range of distances from the optical axis.
fn test_points() -> Vec<[f64; 2]> {
    vec![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [-0.2, 0.15],
        [0.3, -0.2],
        [-0.1, -0.3],
        [0.5, 0.5],
        [-0.4, 0.3],
        [0.05, -0.05],
    ]
}

#[test]
fn round_trip_distort_then_undistort() {
    for cam in all_cameras() {
        for &[x, y] in &test_points() {
            let (xd, yd) = cam.model.distort(x, y);
            let (x_rt, y_rt) = cam.model.undistort(xd, yd);
            assert_relative_eq!(x_rt, x, epsilon = 1e-8,);
            assert_relative_eq!(y_rt, y, epsilon = 1e-8,);
        }
    }
}

#[test]
fn round_trip_undistort_then_distort() {
    for cam in all_cameras() {
        for &[xd, yd] in &test_points() {
            let (x, y) = cam.model.undistort(xd, yd);
            let (xd_rt, yd_rt) = cam.model.distort(x, y);
            assert_relative_eq!(xd_rt, xd, epsilon = 1e-8);
            assert_relative_eq!(yd_rt, yd, epsilon = 1e-8);
        }
    }
}

// -----------------------------------------------------------------------
// SimpleRadial: verify distort formula directly
// -----------------------------------------------------------------------

#[test]
fn simple_radial_distort_formula() {
    let cam = simple_radial();
    let (x, y) = (0.3, 0.4);
    let r2 = x * x + y * y; // 0.25
    let k1 = 0.1;
    let expected_scale = 1.0 + k1 * r2; // 1.025
    let (xd, yd) = cam.model.distort(x, y);
    assert_relative_eq!(xd, x * expected_scale, epsilon = 1e-15);
    assert_relative_eq!(yd, y * expected_scale, epsilon = 1e-15);
}

// -----------------------------------------------------------------------
// Radial: verify distort formula directly
// -----------------------------------------------------------------------

#[test]
fn radial_distort_formula() {
    let cam = radial();
    let (x, y) = (0.3, 0.4);
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let (k1, k2) = (0.1, -0.05);
    let expected_scale = 1.0 + k1 * r2 + k2 * r4;
    let (xd, yd) = cam.model.distort(x, y);
    assert_relative_eq!(xd, x * expected_scale, epsilon = 1e-15);
    assert_relative_eq!(yd, y * expected_scale, epsilon = 1e-15);
}

// -----------------------------------------------------------------------
// Distortion changes coordinates (non-zero distortion should differ)
// -----------------------------------------------------------------------

#[test]
fn distortion_is_not_identity_for_distorted_models() {
    let point = (0.3, 0.4);
    for cam in [
        simple_radial(),
        radial(),
        opencv(),
        opencv_fisheye(),
        simple_radial_fisheye(),
        radial_fisheye(),
        thin_prism_fisheye(),
        rad_tan_thin_prism_fisheye(),
        full_opencv(),
    ] {
        let (xd, yd) = cam.model.distort(point.0, point.1);
        let differs = (xd - point.0).abs() > 1e-10 || (yd - point.1).abs() > 1e-10;
        assert!(
            differs,
            "{} distort should modify off-center points",
            cam.model_name()
        );
    }
}

// -----------------------------------------------------------------------
// Pixel-space project/unproject round-trip
// -----------------------------------------------------------------------

#[test]
fn project_unproject_round_trip() {
    for cam in all_cameras() {
        for &[x, y] in &test_points() {
            let (u, v) = cam.project(x, y);
            let (x_rt, y_rt) = cam.unproject(u, v);
            assert_relative_eq!(x_rt, x, epsilon = 1e-8);
            assert_relative_eq!(y_rt, y, epsilon = 1e-8);
        }
    }
}

#[test]
fn project_pinhole_matches_intrinsic_matrix() {
    let cam = pinhole();
    let (x, y) = (0.3, -0.2);
    let (u, v) = cam.project(x, y);
    // For pinhole: u = fx * x + cx, v = fy * y + cy
    assert_relative_eq!(u, 500.0 * 0.3 + 320.0, epsilon = 1e-12);
    assert_relative_eq!(v, 502.0 * -0.2 + 240.0, epsilon = 1e-12);
}

#[test]
fn unproject_pinhole_at_principal_point() {
    let cam = pinhole();
    let (x, y) = cam.unproject(320.0, 240.0);
    assert_relative_eq!(x, 0.0, epsilon = 1e-15);
    assert_relative_eq!(y, 0.0, epsilon = 1e-15);
}

// -----------------------------------------------------------------------
// Batch variants
// -----------------------------------------------------------------------

#[test]
fn distort_batch_matches_single() {
    for cam in all_cameras() {
        let pts = test_points();
        let batch_result = cam.model.distort_batch(&pts);
        for (i, &[x, y]) in pts.iter().enumerate() {
            let (xd, yd) = cam.model.distort(x, y);
            assert_relative_eq!(batch_result[i][0], xd, epsilon = 1e-15);
            assert_relative_eq!(batch_result[i][1], yd, epsilon = 1e-15);
        }
    }
}

#[test]
fn undistort_batch_matches_single() {
    for cam in all_cameras() {
        let pts = test_points();
        let batch_result = cam.model.undistort_batch(&pts);
        for (i, &[x_d, y_d]) in pts.iter().enumerate() {
            let (x, y) = cam.model.undistort(x_d, y_d);
            assert_relative_eq!(batch_result[i][0], x, epsilon = 1e-15);
            assert_relative_eq!(batch_result[i][1], y, epsilon = 1e-15);
        }
    }
}

#[test]
fn project_batch_matches_single() {
    for cam in all_cameras() {
        let pts = test_points();
        let batch_result = cam.project_batch(&pts);
        for (i, &[x, y]) in pts.iter().enumerate() {
            let (u, v) = cam.project(x, y);
            assert_relative_eq!(batch_result[i][0], u, epsilon = 1e-15);
            assert_relative_eq!(batch_result[i][1], v, epsilon = 1e-15);
        }
    }
}

#[test]
fn unproject_batch_matches_single() {
    for cam in all_cameras() {
        let pixels: Vec<[f64; 2]> = test_points()
            .iter()
            .map(|&[x, y]| {
                let (u, v) = cam.project(x, y);
                [u, v]
            })
            .collect();
        let batch_result = cam.unproject_batch(&pixels);
        for (i, &[u, v]) in pixels.iter().enumerate() {
            let (x, y) = cam.unproject(u, v);
            assert_relative_eq!(batch_result[i][0], x, epsilon = 1e-15);
            assert_relative_eq!(batch_result[i][1], y, epsilon = 1e-15);
        }
    }
}

// -----------------------------------------------------------------------
// Fisheye: specific behavior tests
// -----------------------------------------------------------------------

#[test]
fn fisheye_distort_at_origin() {
    let cam = opencv_fisheye();
    let (xd, yd) = cam.model.distort(0.0, 0.0);
    assert_relative_eq!(xd, 0.0, epsilon = 1e-15);
    assert_relative_eq!(yd, 0.0, epsilon = 1e-15);
}

#[test]
fn fisheye_round_trip_wide_angle() {
    // Test at wider angles where fisheye diverges most from pinhole
    let cam = opencv_fisheye();
    for &[x, y] in &[[0.8, 0.0], [0.0, 0.8], [0.6, 0.6], [-0.7, 0.5]] {
        let (xd, yd) = cam.model.distort(x, y);
        let (x_rt, y_rt) = cam.model.undistort(xd, yd);
        assert_relative_eq!(x_rt, x, epsilon = 1e-8);
        assert_relative_eq!(y_rt, y, epsilon = 1e-8);
    }
}

// -----------------------------------------------------------------------
// undistort_to_ray tests
// -----------------------------------------------------------------------

#[test]
fn undistort_to_ray_at_origin_is_optical_axis() {
    for cam in all_cameras() {
        let ray = cam.model.undistort_to_ray(0.0, 0.0);
        assert_relative_eq!(ray[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(ray[1], 0.0, epsilon = 1e-15);
        assert_relative_eq!(ray[2], 1.0, epsilon = 1e-15);
    }
}

#[test]
fn undistort_to_ray_produces_unit_vectors() {
    for cam in all_cameras() {
        for &[x_d, y_d] in &test_points() {
            let ray = cam.model.undistort_to_ray(x_d, y_d);
            let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
            assert_relative_eq!(len, 1.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn undistort_to_ray_agrees_with_undistort_for_perspective() {
    // For perspective models, undistort_to_ray should give the same
    // direction as normalize(undistort(x_d, y_d), 1)
    for cam in [
        pinhole(),
        simple_pinhole(),
        simple_radial(),
        radial(),
        opencv(),
        full_opencv(),
    ] {
        for &[x_d, y_d] in &test_points() {
            let ray = cam.model.undistort_to_ray(x_d, y_d);
            let (x, y) = cam.model.undistort(x_d, y_d);
            let len = (x * x + y * y + 1.0).sqrt();
            assert_relative_eq!(ray[0], x / len, epsilon = 1e-10);
            assert_relative_eq!(ray[1], y / len, epsilon = 1e-10);
            assert_relative_eq!(ray[2], 1.0 / len, epsilon = 1e-10);
        }
    }
}

#[test]
fn undistort_to_ray_agrees_with_undistort_for_small_angles() {
    // For fisheye models at small angles, undistort_to_ray should agree
    // with normalize(undistort(x_d, y_d), 1) since tan(theta) ≈ theta
    let small_points = [[0.01, 0.0], [0.0, 0.01], [0.01, 0.01], [-0.02, 0.015]];
    let fisheye_cameras = vec![
        opencv_fisheye(),
        simple_radial_fisheye(),
        radial_fisheye(),
        thin_prism_fisheye(),
        rad_tan_thin_prism_fisheye(),
    ];
    for cam in fisheye_cameras {
        for &[x_d, y_d] in &small_points {
            let ray = cam.model.undistort_to_ray(x_d, y_d);
            let (x, y) = cam.model.undistort(x_d, y_d);
            let len = (x * x + y * y + 1.0).sqrt();
            assert_relative_eq!(ray[0], x / len, epsilon = 1e-6);
            assert_relative_eq!(ray[1], y / len, epsilon = 1e-6);
            assert_relative_eq!(ray[2], 1.0 / len, epsilon = 1e-6);
        }
    }
}

#[test]
fn undistort_to_ray_fisheye_beyond_90_degrees() {
    // For a pure equidistant fisheye (no distortion coefficients),
    // a distorted radius of π/2 corresponds to theta = 90°,
    // and beyond that the ray should point backward (z < 0).
    let cam = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        },
        width: 640,
        height: 480,
    };

    // At exactly 90°: theta = π/2, r_d = π/2 in normalized coords
    let r_d_90 = std::f64::consts::FRAC_PI_2;
    let ray = cam.model.undistort_to_ray(r_d_90, 0.0);
    assert_relative_eq!(ray[2], 0.0, epsilon = 1e-10); // z ≈ 0 at 90°
    assert!(ray[0] > 0.0); // pointing rightward

    // Beyond 90°: theta > π/2, z should be negative
    let r_d_120 = std::f64::consts::FRAC_PI_3 * 2.0; // 120° = 2π/3
    let ray = cam.model.undistort_to_ray(r_d_120, 0.0);
    assert!(
        ray[2] < 0.0,
        "Ray beyond 90° should have negative z, got {}",
        ray[2]
    );
    assert!(ray[0] > 0.0, "Ray should still point rightward");
    let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    assert_relative_eq!(len, 1.0, epsilon = 1e-10);
}

#[test]
fn thin_prism_fisheye_undistort_to_ray_wide_angle() {
    // Verify undistort_to_ray for thin prism fisheye with nonzero distortion.
    // Exact round-trip is tested below 80° (before the blend to undistorted
    // kicks in). Above 80°, we just verify unit-length and no NaNs.
    let (k1, k2, p1, p2, k3, k4, sx1, sy1) =
        (0.01, -0.0001, 0.001, -0.001, 0.0, 0.0, 0.002, -0.001);
    let cam = CameraModel::ThinPrismFisheye {
        focal_length_x: 500.0,
        focal_length_y: 500.0,
        principal_point_x: 0.0,
        principal_point_y: 0.0,
        radial_distortion_k1: k1,
        radial_distortion_k2: k2,
        tangential_distortion_p1: p1,
        tangential_distortion_p2: p2,
        radial_distortion_k3: k3,
        radial_distortion_k4: k4,
        thin_prism_sx1: sx1,
        thin_prism_sy1: sy1,
    };

    for deg in (0..=360).step_by(5) {
        let theta = (deg as f64).to_radians();
        let uu = theta * 0.8_f64.cos();
        let vv = theta * 0.8_f64.sin();

        // Forward distort in equidistant space
        let theta2 = uu * uu + vv * vv;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;
        let radial_val = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
        let duu =
            uu * radial_val + 2.0 * p1 * uu * vv + p2 * (theta2 + 2.0 * uu * uu) + sx1 * theta2;
        let dvv =
            vv * radial_val + 2.0 * p2 * uu * vv + p1 * (theta2 + 2.0 * vv * vv) + sy1 * theta2;
        let x_d = uu + duu;
        let y_d = vv + dvv;

        let ray = cam.undistort_to_ray(x_d, y_d);
        let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();

        assert!(
            !ray[0].is_nan() && !ray[1].is_nan() && !ray[2].is_nan(),
            "ThinPrism: NaN at {deg}°"
        );
        assert_relative_eq!(len, 1.0, epsilon = 1e-6);

        // Exact round-trip only below the blend range (80°)
        if deg < 90 {
            let expected = equidistant_to_ray(uu, vv);
            let err = ((ray[0] - expected[0]).powi(2)
                + (ray[1] - expected[1]).powi(2)
                + (ray[2] - expected[2]).powi(2))
            .sqrt();
            assert!(
                err < 1e-6,
                "ThinPrism: ray error {err:.2e} at {deg}° (ray={ray:?}, expected={expected:?})"
            );
        }
    }
}

#[test]
fn rad_tan_thin_prism_fisheye_undistort_to_ray_wide_angle() {
    // Same test for RadTanThinPrismFisheye with small distortion.
    // Exact round-trip below 80°; unit-length and no NaNs everywhere.
    let (k0, k1, k2, k3, k4, k5) = (0.01, -0.0001, 0.0, 0.0, 0.0, 0.0);
    let (p0, p1) = (0.001, -0.001);
    let (s0, s1, s2, s3) = (0.001, 0.0, -0.001, 0.0);
    let cam = CameraModel::RadTanThinPrismFisheye {
        focal_length_x: 500.0,
        focal_length_y: 500.0,
        principal_point_x: 0.0,
        principal_point_y: 0.0,
        radial_distortion_k0: k0,
        radial_distortion_k1: k1,
        radial_distortion_k2: k2,
        radial_distortion_k3: k3,
        radial_distortion_k4: k4,
        radial_distortion_k5: k5,
        tangential_distortion_p0: p0,
        tangential_distortion_p1: p1,
        thin_prism_s0: s0,
        thin_prism_s1: s1,
        thin_prism_s2: s2,
        thin_prism_s3: s3,
    };

    for deg in (0..=360).step_by(5) {
        let theta = (deg as f64).to_radians();
        let uu = theta * 0.8_f64.cos();
        let vv = theta * 0.8_f64.sin();

        // Forward distort: radial scaling then tangential+thin prism
        let th2 = uu * uu + vv * vv;
        let th4 = th2 * th2;
        let th6 = th4 * th2;
        let th8 = th4 * th4;
        let th10 = th8 * th2;
        let th12 = th8 * th4;
        let th_radial = 1.0 + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8 + k4 * th10 + k5 * th12;
        let uu_r = uu * th_radial;
        let vv_r = vv * th_radial;
        let uu_r2 = uu_r * uu_r;
        let vv_r2 = vv_r * vv_r;
        let r2 = uu_r2 + vv_r2;
        let r4 = r2 * r2;
        let duu = 2.0 * p1 * uu_r * vv_r + p0 * (r2 + 2.0 * uu_r2) + s0 * r2 + s1 * r4;
        let dvv = p1 * (r2 + 2.0 * vv_r2) + 2.0 * p0 * uu_r * vv_r + s2 * r2 + s3 * r4;
        let x_d = uu_r + duu;
        let y_d = vv_r + dvv;

        let ray = cam.undistort_to_ray(x_d, y_d);
        let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();

        assert!(
            !ray[0].is_nan() && !ray[1].is_nan() && !ray[2].is_nan(),
            "RadTanThinPrism: NaN at {deg}°"
        );
        assert_relative_eq!(len, 1.0, epsilon = 1e-6);

        if deg < 90 {
            let expected = equidistant_to_ray(uu, vv);
            let err = ((ray[0] - expected[0]).powi(2)
                + (ray[1] - expected[1]).powi(2)
                + (ray[2] - expected[2]).powi(2))
            .sqrt();
            assert!(
                err < 1e-6,
                "RadTanThinPrism: ray error {err:.2e} at {deg}° (ray={ray:?}, expected={expected:?})"
            );
        }
    }
}

#[test]
fn recover_theta_equidistant_out_of_range() {
    // Fisheye camera with distortion coefficients from a real 360 camera.
    // The distortion function f(theta) peaks at ~106° and then decreases,
    // so r_d values beyond ~1.878 have no valid inverse. Previously Newton's
    // method would diverge, producing garbage theta values (e.g. 2800°).
    let k1 = 0.04338287031606894;
    let k2 = -0.010311408690860134;
    let k3 = 0.00890875030327529;
    let k4 = -0.0026965936602161068;

    // In-range: should converge to a valid theta
    let (theta, converged) = recover_theta_equidistant(1.5, k1, k2, k3, k4);
    assert!(converged, "should converge for in-range r_d");
    assert!(theta > 0.0 && theta < std::f64::consts::PI, "theta={theta}");

    // Out-of-range (corner pixel): should NOT produce garbage
    let (theta, converged) = recover_theta_equidistant(2.636, k1, k2, k3, k4);
    assert!(!converged, "should not converge for out-of-range r_d");
    assert!(
        theta > 0.0 && theta <= std::f64::consts::PI,
        "Out-of-range r_d should produce bounded theta, got {theta} ({} degrees)",
        theta.to_degrees()
    );

    // The ray from an out-of-range theta must still be a valid unit vector
    let cam = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 1033.0,
            focal_length_y: 1027.0,
            principal_point_x: 1920.0,
            principal_point_y: 1920.0,
            radial_distortion_k1: k1,
            radial_distortion_k2: k2,
            radial_distortion_k3: k3,
            radial_distortion_k4: k4,
        },
        width: 3840,
        height: 3840,
    };
    // Corner pixel — beyond valid distortion range
    let ray = cam.pixel_to_ray(3840.0, 3840.0);
    let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    assert!(
        (len - 1.0).abs() < 0.01,
        "Ray should be approximately unit length, got {len}"
    );
    assert!(
        ray[2] > -1.1,
        "Ray z component should be reasonable, got {}",
        ray[2]
    );
}

// -----------------------------------------------------------------------
// pixel_to_ray tests
// -----------------------------------------------------------------------

#[test]
fn pixel_to_ray_at_principal_point() {
    for cam in all_cameras() {
        let (cx, cy) = cam.principal_point();
        let ray = cam.pixel_to_ray(cx, cy);
        assert_relative_eq!(ray[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(ray[1], 0.0, epsilon = 1e-15);
        assert_relative_eq!(ray[2], 1.0, epsilon = 1e-15);
    }
}

#[test]
fn pixel_to_ray_produces_unit_vectors() {
    for cam in all_cameras() {
        let pixels = [[0.0, 0.0], [320.0, 240.0], [639.0, 479.0], [100.0, 200.0]];
        for &[u, v] in &pixels {
            let ray = cam.pixel_to_ray(u, v);
            let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
            assert_relative_eq!(len, 1.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn pixel_to_ray_batch_matches_single() {
    for cam in all_cameras() {
        let pixels = [[0.0, 0.0], [320.0, 240.0], [639.0, 479.0], [100.0, 200.0]];
        let batch = cam.pixel_to_ray_batch(&pixels);
        for (i, &[u, v]) in pixels.iter().enumerate() {
            let ray = cam.pixel_to_ray(u, v);
            assert_relative_eq!(batch[i][0], ray[0], epsilon = 1e-15);
            assert_relative_eq!(batch[i][1], ray[1], epsilon = 1e-15);
            assert_relative_eq!(batch[i][2], ray[2], epsilon = 1e-15);
        }
    }
}

/// Round-trip: project a 3D direction to pixels, then pixel_to_ray should
/// recover the same direction. Tests all fisheye camera models.
#[test]
fn pixel_to_ray_round_trip_fisheye() {
    let cameras = vec![simple_radial_fisheye(), radial_fisheye(), opencv_fisheye()];
    // Undistorted normalized coords → 3D directions
    let test_dirs: Vec<[f64; 3]> = [
        [0.1, 0.0],
        [0.0, 0.1],
        [0.2, 0.15],
        [-0.1, 0.3],
        [0.4, -0.2],
        [0.05, 0.05],
    ]
    .iter()
    .map(|&[x, y]: &[f64; 2]| {
        let len = (x * x + y * y + 1.0).sqrt();
        [x / len, y / len, 1.0 / len]
    })
    .collect();

    for cam in &cameras {
        for &dir in &test_dirs {
            // Normalized coords from direction
            let x: f64 = dir[0] / dir[2];
            let y: f64 = dir[1] / dir[2];
            // Project to pixel
            let (u, v) = cam.project(x, y);
            // Recover ray
            let ray = cam.pixel_to_ray(u, v);
            let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
            assert_relative_eq!(len, 1.0, epsilon = 1e-10);
            assert_relative_eq!(ray[0], dir[0], epsilon = 1e-8);
            assert_relative_eq!(ray[1], dir[1], epsilon = 1e-8);
            assert_relative_eq!(ray[2], dir[2], epsilon = 1e-8);
        }
    }
}

/// Test pixel_to_ray at image corners for SimpleRadialFisheye.
/// Verifies no NaN/Inf and that rays are sane at extreme pixels.
#[test]
fn pixel_to_ray_simple_radial_fisheye_corners() {
    // Camera with various k values (positive, zero, negative)
    for k in [-0.2, -0.1, 0.0, 0.05, 0.1] {
        let cam = CameraIntrinsics {
            model: CameraModel::SimpleRadialFisheye {
                focal_length: 300.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: k,
            },
            width: 640,
            height: 480,
        };
        let corners = [
            [0.0, 0.0],
            [640.0, 0.0],
            [0.0, 480.0],
            [640.0, 480.0],
            [320.0, 240.0],
        ];
        for &[u, v] in &corners {
            let ray = cam.pixel_to_ray(u, v);
            let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
            assert!(
                len.is_finite(),
                "k={k}, pixel=({u},{v}): ray is not finite: {ray:?}"
            );
            assert_relative_eq!(len, 1.0, epsilon = 1e-10,);
            // z should be positive for this camera (half-diagonal FoV ~53°)
            assert!(
                ray[2] > 0.0,
                "k={k}, pixel=({u},{v}): ray z should be positive, got {}",
                ray[2]
            );
        }
    }
}

/// Wide-angle SimpleRadialFisheye: pixels beyond 90° from the optical
/// axis should produce backward-facing rays (z < 0), matching the
/// equidistant projection model.
#[test]
fn pixel_to_ray_simple_radial_fisheye_wide_angle() {
    // Pure equidistant (k=0) with focal length chosen so corners
    // exceed 90°: half-diagonal r_d = sqrt(500²+500²)/300 ≈ 2.36 rad ≈ 135°.
    let cam = CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: 300.0,
            principal_point_x: 500.0,
            principal_point_y: 500.0,
            radial_distortion_k1: 0.0,
        },
        width: 1000,
        height: 1000,
    };

    // Principal point → straight ahead
    let ray = cam.pixel_to_ray(500.0, 500.0);
    assert_relative_eq!(ray[2], 1.0, epsilon = 1e-10);

    // Edge midpoint: r_d = 500/300 ≈ 1.667 rad ≈ 95° → just past 90°
    let ray = cam.pixel_to_ray(1000.0, 500.0);
    let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    assert_relative_eq!(len, 1.0, epsilon = 1e-10);
    assert!(ray[0] > 0.0, "should point rightward");
    assert!(
        ray[2] < 0.0,
        "edge at ~95° should have z < 0, got {}",
        ray[2]
    );

    // Corner: r_d ≈ 2.36 rad ≈ 135° → well past 90°
    let ray = cam.pixel_to_ray(1000.0, 1000.0);
    let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    assert_relative_eq!(len, 1.0, epsilon = 1e-10);
    assert!(ray[0] > 0.0, "corner should point right");
    assert!(ray[1] > 0.0, "corner should point down");
    assert!(
        ray[2] < 0.0,
        "corner at ~135° should have z < 0, got {}",
        ray[2]
    );

    // Verify the angle is approximately correct: theta ≈ r_d for k=0
    let theta = ray[2].acos();
    let expected_theta = (500.0_f64 * 2.0_f64.sqrt()) / 300.0;
    assert_relative_eq!(theta, expected_theta, epsilon = 1e-6);
}

/// SimpleRadialFisheye with small positive k: the distortion function
/// is monotonic, so recovery should converge even at wide angles.
/// Verifies round-trip at 100° and 110°.
#[test]
fn pixel_to_ray_simple_radial_fisheye_wide_angle_with_distortion() {
    let cam = CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: 300.0,
            principal_point_x: 500.0,
            principal_point_y: 500.0,
            radial_distortion_k1: 0.02,
        },
        width: 1000,
        height: 1000,
    };

    // Test round-trip at various angles including beyond 90°
    for theta_deg in [30.0, 60.0, 80.0, 100.0, 110.0] {
        let theta = theta_deg * std::f64::consts::PI / 180.0;
        // Undistorted normalized coords for this angle along x-axis
        let r = theta.tan();
        let x = r;
        let y = 0.0;

        // Skip angles where tan() is very large (near 90°)
        if r.abs() > 100.0 {
            continue;
        }

        let (u, v) = cam.project(x, y);

        // Only test if the projected pixel is within a reasonable range
        if u < -1000.0 || u > 2000.0 {
            continue;
        }

        let ray = cam.pixel_to_ray(u, v);
        let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
        assert_relative_eq!(len, 1.0, epsilon = 1e-10);

        // Expected ray direction: normalize(x, 0, 1)
        let expected_len = (x * x + 1.0).sqrt();
        let expected = [x / expected_len, 0.0, 1.0 / expected_len];

        assert_relative_eq!(ray[0], expected[0], epsilon = 1e-6,);
        assert_relative_eq!(ray[1], expected[1], epsilon = 1e-6,);
        assert_relative_eq!(ray[2], expected[2], epsilon = 1e-6,);
    }
}

// -----------------------------------------------------------------------
// ray_to_pixel: round-trip pixel_to_ray → ray_to_pixel for all models
// -----------------------------------------------------------------------

#[test]
fn ray_to_pixel_round_trip_all_models() {
    for cam in all_cameras() {
        let (cx, cy) = cam.principal_point();

        // Test at center and nearby positions. Avoid extreme corners where
        // fisheye blending in pixel_to_ray causes larger round-trip errors.
        let test_points = vec![
            [cx, cy],        // center
            [cx + 10.0, cy], // right of center
            [cx, cy + 10.0], // below center
            [cx - 10.0, cy - 10.0],
            [cx + 50.0, cy + 30.0],
            [cx - 30.0, cy + 50.0],
        ];

        for pt in &test_points {
            let ray = cam.pixel_to_ray(pt[0], pt[1]);
            if let Some((u, v)) = cam.ray_to_pixel(ray) {
                // Thin prism / rad-tan fisheye models have larger round-trip
                // errors due to blending in pixel_to_ray at moderate angles.
                let tol = if cam.model.is_fisheye() { 0.5 } else { 0.01 };
                assert!(
                    (u - pt[0]).abs() < tol && (v - pt[1]).abs() < tol,
                    "ray_to_pixel round-trip failed for {} at ({}, {}): got ({u}, {v}), tol={tol}",
                    cam.model_name(),
                    pt[0],
                    pt[1],
                );
            }
        }
    }
}

#[test]
fn ray_to_pixel_returns_none_behind_camera_perspective() {
    let cam = pinhole();
    // Ray pointing backward
    assert!(cam.ray_to_pixel([0.0, 0.0, -1.0]).is_none());
    assert!(cam.ray_to_pixel([0.5, 0.3, -0.1]).is_none());
}

#[test]
fn ray_to_pixel_batch_matches_single() {
    let cam = opencv();
    let rays = vec![[0.0, 0.0, 1.0], [0.1, 0.2, 1.0], [-0.3, 0.1, 1.0]];
    let batch = cam.ray_to_pixel_batch(&rays);
    for (ray, result) in rays.iter().zip(batch.iter()) {
        let single = cam.ray_to_pixel(*ray);
        match (single, result) {
            (Some((u, v)), Some([bu, bv])) => {
                assert_relative_eq!(u, bu, epsilon = 1e-10);
                assert_relative_eq!(v, bv, epsilon = 1e-10);
            }
            (None, None) => {}
            _ => panic!("mismatch"),
        }
    }
}

// -----------------------------------------------------------------------
// Equirectangular model tests
// -----------------------------------------------------------------------

#[test]
fn equirectangular_pixel_to_ray_center() {
    let cam = equirectangular();
    let (cx, cy) = cam.principal_point();
    let ray = cam.pixel_to_ray(cx, cy);
    // Center should point along +Z
    assert_relative_eq!(ray[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[2], 1.0, epsilon = 1e-10);
}

#[test]
fn equirectangular_pixel_to_ray_right_edge() {
    // Right edge is at longitude = π (pointing along -Z)
    let cam = equirectangular();
    let ray = cam.pixel_to_ray(cam.width as f64, cam.height as f64 / 2.0);
    assert_relative_eq!(ray[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[2], -1.0, epsilon = 1e-10);
}

#[test]
fn equirectangular_pixel_to_ray_top() {
    // Top is at latitude = +π/2 (pointing along +Y)
    let cam = equirectangular();
    let ray = cam.pixel_to_ray(cam.width as f64 / 2.0, 0.0);
    assert_relative_eq!(ray[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(ray[2], 0.0, epsilon = 1e-6);
}

#[test]
fn equirectangular_round_trip() {
    let cam = equirectangular();
    let test_pixels = vec![
        [100.0, 50.0],
        [320.0, 160.0],
        [500.0, 100.0],
        [1.0, 1.0],
        [639.0, 319.0],
    ];
    for pt in &test_pixels {
        let ray = cam.pixel_to_ray(pt[0], pt[1]);
        let (u, v) = cam.ray_to_pixel(ray).unwrap();
        assert_relative_eq!(u, pt[0], epsilon = 1e-8);
        assert_relative_eq!(v, pt[1], epsilon = 1e-8);
    }
}

#[test]
fn equirectangular_ray_to_pixel_always_valid() {
    // Equirectangular can represent any direction
    let cam = equirectangular();
    let rays = vec![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.577, 0.577, 0.577],
    ];
    for ray in &rays {
        assert!(cam.ray_to_pixel(*ray).is_some(), "failed for ray {ray:?}");
    }
}

#[test]
fn equirectangular_distort_undistort_identity() {
    let cam = equirectangular();
    let (x, y) = (0.5, -0.3);
    let (xd, yd) = cam.model.distort(x, y);
    assert_relative_eq!(xd, x);
    assert_relative_eq!(yd, y);
    let (xu, yu) = cam.model.undistort(x, y);
    assert_relative_eq!(xu, x);
    assert_relative_eq!(yu, y);
}

// -----------------------------------------------------------------------
// ray_to_pixel for fisheye at wide angles
// -----------------------------------------------------------------------

#[test]
fn ray_to_pixel_fisheye_wide_angle() {
    let cam = opencv_fisheye();
    // 80° from optical axis
    let theta = 80.0_f64.to_radians();
    let ray = [theta.sin(), 0.0, theta.cos()];
    let result = cam.ray_to_pixel(ray);
    assert!(result.is_some(), "80° should be valid for fisheye");

    // 89° should also work
    let theta = 89.0_f64.to_radians();
    let ray = [theta.sin(), 0.0, theta.cos()];
    let result = cam.ray_to_pixel(ray);
    assert!(result.is_some(), "89° should be valid for fisheye");
}

#[test]
fn ray_to_pixel_fisheye_round_trip_wide_angle() {
    let cam = opencv_fisheye();
    for angle_deg in [10.0_f64, 30.0, 60.0, 80.0, 85.0] {
        let theta = angle_deg.to_radians();
        let ray_in = [theta.sin(), 0.0, theta.cos()];
        if let Some((u, v)) = cam.ray_to_pixel(ray_in) {
            let ray_out = cam.pixel_to_ray(u, v);
            let len = (ray_out[0] * ray_out[0] + ray_out[1] * ray_out[1] + ray_out[2] * ray_out[2])
                .sqrt();
            assert_relative_eq!(len, 1.0, epsilon = 1e-10);
            assert_relative_eq!(ray_out[0], ray_in[0], epsilon = 1e-4);
            assert_relative_eq!(ray_out[1], ray_in[1], epsilon = 1e-4);
            assert_relative_eq!(ray_out[2], ray_in[2], epsilon = 1e-4);
        }
    }
}

// -----------------------------------------------------------------------
// best_fit_inside_pinhole / best_fit_outside_pinhole
// -----------------------------------------------------------------------

#[test]
fn best_fit_inside_pinhole_simple_radial() {
    let cam = simple_radial();
    let result = cam.best_fit_inside_pinhole(640, 480).unwrap();
    assert_eq!(result.width, 640);
    assert_eq!(result.height, 480);

    // The inside pinhole must map every boundary pixel to a valid
    // source location.
    let boundary = CameraIntrinsics::boundary_samples(640, 480);
    let (cx, cy) = result.principal_point();
    let (fx, _fy) = result.focal_lengths();
    for &(u, v) in &boundary {
        let x = (u - cx) / fx;
        let y = (v - cy) / fx;
        let (sx, sy) = cam.project(x, y);
        assert!(
            sx >= 0.0 && sy >= 0.0 && sx < 640.0 && sy < 480.0,
            "boundary point ({u}, {v}) maps outside source at ({sx}, {sy})"
        );
    }

    // The focal length should be larger than the source (narrower FoV
    // to avoid black borders with positive barrel distortion k1=0.1).
    assert!(
        fx > 500.0,
        "expected focal > 500 for barrel distortion, got {fx}"
    );
}

#[test]
fn best_fit_outside_pinhole_simple_radial() {
    let cam = simple_radial();
    let result = cam.best_fit_outside_pinhole(640, 480).unwrap();
    assert_eq!(result.width, 640);
    assert_eq!(result.height, 480);

    // The outside pinhole must cover every source boundary pixel.
    let src_boundary = CameraIntrinsics::boundary_samples(640, 480);
    let (cx, cy) = result.principal_point();
    let (fx, _fy) = result.focal_lengths();
    for &(u, v) in &src_boundary {
        let (x, y) = cam.unproject(u, v);
        let px = fx * x + cx;
        let py = fx * y + cy;
        assert!(
            px >= 0.0 && py >= 0.0 && px < 640.0 && py < 480.0,
            "source boundary ({u}, {v}) maps outside dst at ({px}, {py})"
        );
    }
}

#[test]
fn best_fit_inside_larger_than_outside() {
    // For barrel distortion, inside focal > outside focal
    // (inside is narrower FoV, outside is wider).
    let cam = simple_radial();
    let inside = cam.best_fit_inside_pinhole(640, 480).unwrap();
    let outside = cam.best_fit_outside_pinhole(640, 480).unwrap();
    let (fi, _) = inside.focal_lengths();
    let (fo, _) = outside.focal_lengths();
    assert!(
        fi > fo,
        "inside focal ({fi}) should be > outside focal ({fo})"
    );
}

#[test]
fn best_fit_pinhole_different_resolution() {
    let cam = simple_radial();
    let result = cam.best_fit_inside_pinhole(1280, 960).unwrap();
    assert_eq!(result.width, 1280);
    assert_eq!(result.height, 960);

    let (cx, cy) = result.principal_point();
    assert_relative_eq!(cx, 640.0, epsilon = 1e-10);
    assert_relative_eq!(cy, 480.0, epsilon = 1e-10);
}

#[test]
fn best_fit_pinhole_rejects_fisheye() {
    let cam = simple_radial_fisheye();
    assert!(cam.best_fit_inside_pinhole(640, 480).is_err());
    assert!(cam.best_fit_outside_pinhole(640, 480).is_err());
}

#[test]
fn best_fit_pinhole_rejects_equirectangular() {
    let cam = equirectangular();
    assert!(cam.best_fit_inside_pinhole(640, 480).is_err());
    assert!(cam.best_fit_outside_pinhole(640, 480).is_err());
}

#[test]
fn best_fit_pinhole_no_distortion_returns_square_pixels() {
    // Source pinhole has non-square pixels (fx=500, fy=502).
    let cam = pinhole();
    let result = cam.best_fit_inside_pinhole(640, 480).unwrap();
    let (fx, fy) = result.focal_lengths();
    // Output must have square pixels.
    assert_relative_eq!(fx, fy, epsilon = 1e-6);
    // The focal length should be close to the source focal lengths.
    assert!((fx - 500.0).abs() < 5.0, "expected focal ~500, got {fx}");

    // Inside and outside should agree for a no-distortion camera
    // (both converge on the same focal length to map identical FoV).
    let outside = cam.best_fit_outside_pinhole(640, 480).unwrap();
    let (fox, foy) = outside.focal_lengths();
    assert_relative_eq!(fox, foy, epsilon = 1e-6);
}

#[test]
fn best_fit_pinhole_opencv_model() {
    let cam = opencv();
    let inside = cam.best_fit_inside_pinhole(640, 480).unwrap();
    let outside = cam.best_fit_outside_pinhole(640, 480).unwrap();
    let (fi, _) = inside.focal_lengths();
    let (fo, _) = outside.focal_lengths();
    // Both should succeed and the inside focal should be larger.
    assert!(fi > fo);
}

#[test]
fn best_fit_pinhole_radial_model() {
    let cam = radial();
    let inside = cam.best_fit_inside_pinhole(640, 480).unwrap();
    let outside = cam.best_fit_outside_pinhole(640, 480).unwrap();
    let (fi, _) = inside.focal_lengths();
    let (fo, _) = outside.focal_lengths();
    assert!(fi > fo);
}
