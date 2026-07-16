use super::*;
use approx::assert_relative_eq;

// -----------------------------------------------------------------------
// Test camera constructors (reused from camera::intrinsics tests)
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
    // For pinhole: u = fx * x + cx, v = fy * (−y) + cy — the canonical
    // image-plane y is up, pixel v is down.
    assert_relative_eq!(u, 500.0 * 0.3 + 320.0, epsilon = 1e-12);
    assert_relative_eq!(v, 502.0 * 0.2 + 240.0, epsilon = 1e-12);
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
    // The canonical optical axis is −Z.
    for cam in all_cameras() {
        let ray = cam.model.undistort_to_ray(0.0, 0.0);
        assert_relative_eq!(ray[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(ray[1], 0.0, epsilon = 1e-15);
        assert_relative_eq!(ray[2], -1.0, epsilon = 1e-15);
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
    // For perspective models, undistort_to_ray should give the S-mapped
    // (canonical) direction of normalize(undistort(x_d, y_d), 1):
    // (x, −y, −1) / len.
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
            assert_relative_eq!(ray[1], -y / len, epsilon = 1e-10);
            assert_relative_eq!(ray[2], -1.0 / len, epsilon = 1e-10);
        }
    }
}

#[test]
fn undistort_to_ray_agrees_with_undistort_for_small_angles() {
    // For fisheye models at small angles, undistort_to_ray should agree
    // with the S-mapped normalize(undistort(x_d, y_d), 1) — i.e.
    // (x, −y, −1)/len — since tan(theta) ≈ theta.
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
            assert_relative_eq!(ray[1], -y / len, epsilon = 1e-6);
            assert_relative_eq!(ray[2], -1.0 / len, epsilon = 1e-6);
        }
    }
}

#[test]
fn undistort_to_ray_fisheye_beyond_90_degrees() {
    // For a pure equidistant fisheye (no distortion coefficients),
    // a distorted radius of π/2 corresponds to theta = 90° off the −Z
    // optical axis, and beyond that the ray should point backward
    // (canonical z > 0).
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

    // Beyond 90°: theta > π/2, canonical z should be positive (behind)
    let r_d_120 = std::f64::consts::FRAC_PI_3 * 2.0; // 120° = 2π/3
    let ray = cam.model.undistort_to_ray(r_d_120, 0.0);
    assert!(
        ray[2] > 0.0,
        "Ray beyond 90° should have positive canonical z, got {}",
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

        // Exact round-trip only below the blend range (80°). The kernel
        // helper produces optical-frame rays; the public API returns the
        // canonical S-mapped ray (x, −y, −z).
        if deg < 90 {
            let e = equidistant_to_ray(uu, vv);
            let expected = [e[0], -e[1], -e[2]];
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
            // Kernel helper is optical-frame; the public API is canonical.
            let e = equidistant_to_ray(uu, vv);
            let expected = [e[0], -e[1], -e[2]];
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

/// kerry_park is a real ~180° FOV OPENCV_FISHEYE rig (test-data/images/kerry_park).
/// Sweep a radial line of pixels from the principal point out to the corner and
/// assert that the recovered ray varies continuously: no kink where the
/// undistortion solver hands off to the small-angle blend at 90°–100°.
#[test]
fn kerry_park_pixel_to_ray_smooth_across_blend_region() {
    let cam = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 129.1499937015594,
            focal_length_y: 129.2573627423474,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.038113353966529886,
            radial_distortion_k2: -0.00800851799065643,
            radial_distortion_k3: 0.008329720504707577,
            radial_distortion_k4: -0.0026901578801066814,
        },
        width: 480,
        height: 480,
    };

    // Walk in 1-pixel steps along +x from the principal point to the image
    // corner. r_d (= radial pixel distance / f, capped at π in the
    // unprojection path) crosses both the 90° (≈ f·π/2 = 203 px) and 100°
    // (≈ f·100°/57.3 = 225 px) boundaries within the 240 px half-width.
    let (cx, cy) = (240.0, 240.0);
    let mut rays = Vec::with_capacity(240);
    for du in 0..240 {
        rays.push(cam.pixel_to_ray(cx + du as f64, cy));
    }

    // 1. Every ray must be a finite unit vector. The blend exists precisely
    //    so that out-of-range pixels don't produce NaN/inf or garbage.
    for (i, r) in rays.iter().enumerate() {
        assert!(
            r[0].is_finite() && r[1].is_finite() && r[2].is_finite(),
            "non-finite ray at offset {i}: {r:?}",
        );
        let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        assert_relative_eq!(len, 1.0, epsilon = 1e-10);
    }

    // 2. The ray sequence must be continuous. A discontinuity at the blend
    //    boundary would show up as a single large step between adjacent
    //    1-pixel samples. Each step is small in well-behaved regions
    //    (~0.005 rad/px) and must not spike past a generous threshold.
    for i in 1..rays.len() {
        let dx = rays[i][0] - rays[i - 1][0];
        let dy = rays[i][1] - rays[i - 1][1];
        let dz = rays[i][2] - rays[i - 1][2];
        let step = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            step < 0.05,
            "ray discontinuity at pixel offset {i}: step={step}, \
             rays[{}]={:?}, rays[{i}]={:?}",
            i - 1,
            rays[i - 1],
            rays[i],
        );
    }

    // 3. The ray at the principal point must be (0, 0, −1) exactly — the
    //    canonical optical axis.
    assert_relative_eq!(rays[0][0], 0.0, epsilon = 1e-15);
    assert_relative_eq!(rays[0][1], 0.0, epsilon = 1e-15);
    assert_relative_eq!(rays[0][2], -1.0, epsilon = 1e-15);

    // 4. By the image edge (240 px out, well past the 100° blend end) the
    //    ray must be pointing strongly sideways: z component small, x large.
    let edge = rays.last().unwrap();
    assert!(
        edge[0] > 0.7,
        "edge ray should swing nearly perpendicular, got x={}",
        edge[0],
    );
    assert!(
        edge[2].abs() < 0.7,
        "edge ray z should be small (near 90° off-axis), got z={}",
        edge[2],
    );
}

// -----------------------------------------------------------------------
// pixel_to_ray tests
// -----------------------------------------------------------------------

#[test]
fn pixel_to_ray_at_principal_point() {
    // The principal point looks down the canonical optical axis, −Z.
    for cam in all_cameras() {
        let (cx, cy) = cam.principal_point();
        let ray = cam.pixel_to_ray(cx, cy);
        assert_relative_eq!(ray[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(ray[1], 0.0, epsilon = 1e-15);
        assert_relative_eq!(ray[2], -1.0, epsilon = 1e-15);
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
    // Undistorted canonical (y-up) image-plane coords → canonical 3D
    // directions (x, y, −1)/len, in front of the −Z-forward camera.
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
        [x / len, y / len, -1.0 / len]
    })
    .collect();

    for cam in &cameras {
        for &dir in &test_dirs {
            // Canonical normalized coords from direction: divide by −z.
            let x: f64 = dir[0] / -dir[2];
            let y: f64 = dir[1] / -dir[2];
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
            // z should be negative (in front of the canonical −Z-forward
            // camera) — half-diagonal FoV is only ~53°.
            assert!(
                ray[2] < 0.0,
                "k={k}, pixel=({u},{v}): ray z should be negative, got {}",
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

    // Principal point → straight ahead (canonical −Z)
    let ray = cam.pixel_to_ray(500.0, 500.0);
    assert_relative_eq!(ray[2], -1.0, epsilon = 1e-10);

    // Edge midpoint: r_d = 500/300 ≈ 1.667 rad ≈ 95° → just past 90°,
    // so the ray points behind the camera (canonical z > 0).
    let ray = cam.pixel_to_ray(1000.0, 500.0);
    let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    assert_relative_eq!(len, 1.0, epsilon = 1e-10);
    assert!(ray[0] > 0.0, "should point rightward");
    assert!(
        ray[2] > 0.0,
        "edge at ~95° should have canonical z > 0, got {}",
        ray[2]
    );

    // Corner: r_d ≈ 2.36 rad ≈ 135° → well past 90°
    let ray = cam.pixel_to_ray(1000.0, 1000.0);
    let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    assert_relative_eq!(len, 1.0, epsilon = 1e-10);
    assert!(ray[0] > 0.0, "corner should point right");
    assert!(
        ray[1] < 0.0,
        "corner (below image centre) should point down (canonical −Y)"
    );
    assert!(
        ray[2] > 0.0,
        "corner at ~135° should have canonical z > 0, got {}",
        ray[2]
    );

    // Verify the angle is approximately correct: theta ≈ r_d for k=0,
    // measured off the canonical −Z optical axis.
    let theta = (-ray[2]).acos();
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
        if !(-1000.0..=2000.0).contains(&u) {
            continue;
        }

        let ray = cam.pixel_to_ray(u, v);
        let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
        assert_relative_eq!(len, 1.0, epsilon = 1e-10);

        // Expected canonical ray direction: normalize(x, 0, −1)
        let expected_len = (x * x + 1.0).sqrt();
        let expected = [x / expected_len, 0.0, -1.0 / expected_len];

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
    // Rays pointing backward (canonical camera looks down −Z, so +z rays
    // are behind).
    assert!(cam.ray_to_pixel([0.0, 0.0, 1.0]).is_none());
    assert!(cam.ray_to_pixel([0.5, 0.3, 0.1]).is_none());
}

#[test]
fn ray_to_pixel_with_jacobian_none_behind_camera() {
    let cam = pinhole();
    // Behind the camera: no projection, no Jacobian (same domain as
    // ray_to_pixel).
    assert!(cam.ray_to_pixel_with_jacobian([0.0, 0.0, 1.0]).is_none());
    assert!(cam.ray_to_pixel_with_jacobian([0.5, 0.3, 0.1]).is_none());
}

/// The analytic `ray_to_pixel_with_jacobian` agrees with a central-difference
/// of `ray_to_pixel` across every perspective model and a wide sweep of ray
/// directions and depths. Fisheye / equirectangular models report no analytic
/// Jacobian. This pins the derivation and guards against regressions in either
/// the projection math or the Jacobian.
#[test]
fn ray_to_pixel_jacobian_matches_central_difference() {
    let h = 1e-6;
    for cam in all_cameras() {
        if !cam.model.supports_pixel_jacobian() {
            // Ray-path models (fisheye / equirectangular): a forward ray still
            // projects, but there is no analytic Jacobian yet.
            assert!(
                cam.ray_to_pixel_with_jacobian([0.0, 0.0, -1.0]).is_none(),
                "{} should report no analytic Jacobian",
                cam.model_name(),
            );
            continue;
        }

        let (cx, cy) = cam.principal_point();
        let mut samples = 0;
        // In-image pixels → in-domain rays; several depths exercise the
        // perspective-divide (1/rz) columns.
        for du in [-60.0, -25.0, 0.0, 25.0, 60.0] {
            for dv in [-60.0, -25.0, 0.0, 25.0, 60.0] {
                let base = cam.pixel_to_ray(cx + du, cy + dv);
                for scale in [0.5_f64, 1.0, 2.5] {
                    let ray = [base[0] * scale, base[1] * scale, base[2] * scale];
                    let Some((_, jac)) = cam.ray_to_pixel_with_jacobian(ray) else {
                        continue;
                    };
                    for c in 0..3 {
                        let mut rp = ray;
                        let mut rm = ray;
                        rp[c] += h;
                        rm[c] -= h;
                        let (Some((up, vp)), Some((um, vm))) =
                            (cam.ray_to_pixel(rp), cam.ray_to_pixel(rm))
                        else {
                            continue;
                        };
                        let fd_u = (up - um) / (2.0 * h);
                        let fd_v = (vp - vm) / (2.0 * h);
                        assert!(
                            (jac[0][c] - fd_u).abs() <= 1e-4 * (1.0 + jac[0][c].abs()),
                            "{} ∂u/∂r[{c}]: analytic {} vs central-diff {}",
                            cam.model_name(),
                            jac[0][c],
                            fd_u,
                        );
                        assert!(
                            (jac[1][c] - fd_v).abs() <= 1e-4 * (1.0 + jac[1][c].abs()),
                            "{} ∂v/∂r[{c}]: analytic {} vs central-diff {}",
                            cam.model_name(),
                            jac[1][c],
                            fd_v,
                        );
                        samples += 1;
                    }
                }
            }
        }
        assert!(samples > 0, "no in-domain samples for {}", cam.model_name());
    }
}

/// SimpleRadial with strongly-negative k1 has a non-monotonic forward
/// distortion polynomial — past the inflection radius the polynomial
/// folds and produces ghost projections at the opposite side of the
/// image. `ray_to_pixel` must reject rays in that regime.
///
/// k1 = -0.6563 reproduces the seoul_bull-style refined-intrinsics bug
/// where wide-angle equirect rays were mapping to mirror copies inside
/// the source image.
#[test]
fn ray_to_pixel_rejects_folded_simple_radial() {
    let cam = CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 497.08,
            principal_point_x: 135.0,
            principal_point_y: 240.0,
            radial_distortion_k1: -0.6563,
        },
        width: 270,
        height: 480,
    };

    // A ray at lon ≈ 53° (rx/(−rz) ≈ 1.35) is well outside the camera's
    // physical FOV but, with the folded polynomial, would project to a
    // pixel near x = 7 — visibly inside the source rectangle. This is
    // the spurious mirror; the gate must reject it.
    let folded_ray = [0.803_f64, 0.0, -0.596];
    assert!(
        cam.ray_to_pixel(folded_ray).is_none(),
        "expected None for ray in distortion fold-over region"
    );

    // A ray well inside the monotonic regime (small angle) must still
    // project successfully — the gate isn't allowed to over-reject.
    let on_axis = [0.0_f64, 0.0, -1.0];
    let pix = cam.ray_to_pixel(on_axis).expect("on-axis ray must project");
    assert_relative_eq!(pix.0, 135.0, epsilon = 1e-6);
    assert_relative_eq!(pix.1, 240.0, epsilon = 1e-6);
}

/// Pinhole / SimplePinhole have no distortion; the new gate must be a
/// no-op for them, accepting any ray with rz > 0.
#[test]
fn ray_to_pixel_pinhole_accepts_wide_rays() {
    let cam = pinhole();
    // Ray well off-axis but still in front (canonical rz < 0); pinhole has
    // no distortion to fold, so projection lands far outside the image
    // rectangle but the function must still return Some(_) — it is then up
    // to the caller to do bounds checking.
    let result = cam.ray_to_pixel([0.9_f64, 0.0, -0.4]);
    assert!(
        result.is_some(),
        "pinhole ray_to_pixel must not reject wide-angle rays (no distortion)"
    );
}

/// Radial (k1, k2) with negative coefficients can also fold; the gate
/// catches it via the same closed-form det(J) test as SimpleRadial.
#[test]
fn ray_to_pixel_rejects_folded_radial() {
    let cam = CameraIntrinsics {
        model: CameraModel::Radial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: -0.7,
            radial_distortion_k2: 0.0,
        },
        width: 640,
        height: 480,
    };
    // For k1 = -0.7, det(J) = (1 - 0.7 r²)(1 - 2.1 r²) goes negative
    // for r² in (1/2.1, 1/0.7) ≈ (0.476, 1.429). Pick a ray squarely
    // in the fold-zone (r² between the roots), in front of the canonical
    // camera (rz < 0).
    let folded = [0.5_f64, 0.5, -0.6]; // |x|=|y|=0.833, r²=1.39 in fold zone
    assert!(
        cam.ray_to_pixel(folded).is_none(),
        "expected None for radial fold-over"
    );
}

#[test]
fn ray_to_pixel_batch_matches_single() {
    let cam = opencv();
    let rays = vec![[0.0, 0.0, -1.0], [0.1, 0.2, -1.0], [-0.3, 0.1, -1.0]];
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
    // Center (longitude 0) points along the canonical forward axis, −Z
    assert_relative_eq!(ray[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[2], -1.0, epsilon = 1e-10);
}

#[test]
fn equirectangular_pixel_to_ray_right_edge() {
    // Right edge is at longitude = π (pointing backward, canonical +Z)
    let cam = equirectangular();
    let ray = cam.pixel_to_ray(cam.width as f64, cam.height as f64 / 2.0);
    assert_relative_eq!(ray[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(ray[2], 1.0, epsilon = 1e-10);
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
    // 80° from the canonical −Z optical axis
    let theta = 80.0_f64.to_radians();
    let ray = [theta.sin(), 0.0, -theta.cos()];
    let result = cam.ray_to_pixel(ray);
    assert!(result.is_some(), "80° should be valid for fisheye");

    // 89° should also work
    let theta = 89.0_f64.to_radians();
    let ray = [theta.sin(), 0.0, -theta.cos()];
    let result = cam.ray_to_pixel(ray);
    assert!(result.is_some(), "89° should be valid for fisheye");
}

#[test]
fn ray_to_pixel_fisheye_round_trip_wide_angle() {
    let cam = opencv_fisheye();
    for angle_deg in [10.0_f64, 30.0, 60.0, 80.0, 85.0] {
        let theta = angle_deg.to_radians();
        let ray_in = [theta.sin(), 0.0, -theta.cos()];
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

// -----------------------------------------------------------------------
// ray_to_pixel_grid: perspective exactness + fisheye coarse-grid bound
// -----------------------------------------------------------------------

/// Affine ray basis for an `r×r` grid spanning camera-frame image-plane coords
/// `(x, y) ∈ [x0, x0+span]²` at unit depth in front of the canonical camera
/// (z = −1), optionally tilted in depth by `tilt` across columns so `z` varies
/// (exercising foreshortening). Mirrors the basis `WarpMap::from_patch` hands
/// to `ray_to_pixel_grid`.
fn grid_basis(x0: f64, y0: f64, span: f64, r: u32, tilt: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let step = span / r as f64;
    let origin = [x0 + 0.5 * step, y0 + 0.5 * step, -1.0];
    let col_step = [step, 0.0, tilt * step];
    let row_step = [0.0, step, 0.0];
    (origin, col_step, row_step)
}

#[test]
fn ray_to_pixel_grid_perspective_matches_scalar() {
    // The perspective path is exact: every node equals scalar ray_to_pixel with
    // the same in-frame test, bit-for-bit (same f64 math, same f32 cast).
    let r = 48u32;
    for cam in [
        pinhole(),
        simple_pinhole(),
        simple_radial(),
        radial(),
        opencv(),
    ] {
        let (o, cs, rs) = grid_basis(-0.4, -0.3, 0.9, r, 0.2);
        let mut out = vec![0f32; 2 * (r * r) as usize];
        cam.ray_to_pixel_grid(o, cs, rs, r, r, &mut out);
        let (w, h) = (cam.width as f64, cam.height as f64);
        for row in 0..r {
            for col in 0..r {
                let ray = [
                    o[0] + col as f64 * cs[0] + row as f64 * rs[0],
                    o[1] + col as f64 * cs[1] + row as f64 * rs[1],
                    o[2] + col as f64 * cs[2] + row as f64 * rs[2],
                ];
                let expect = match cam.ray_to_pixel(ray) {
                    Some((px, py)) if px >= 0.0 && py >= 0.0 && px < w && py < h => {
                        [px as f32, py as f32]
                    }
                    _ => [f32::NAN, f32::NAN],
                };
                let i = 2 * (row * r + col) as usize;
                for k in 0..2 {
                    if expect[k].is_nan() {
                        assert!(out[i + k].is_nan(), "expected NaN at ({col},{row})");
                    } else {
                        assert_eq!(out[i + k], expect[k], "mismatch at ({col},{row})");
                    }
                }
            }
        }
    }
}

#[test]
fn coarse_grid_error_within_bound() {
    // Fisheye/equirect take the coarse sub-grid + bilinear path. Compare it to
    // the exact per-node projection over a spread of placements/spans — including
    // wide-angle off-center tiles (worst-case curvature) and depth tilt — and
    // bound both the photometric (sub-pixel) error and the validity disagreement.
    let r = 48u32;
    // Gentle, realistic patch tiles (small source span) where interpolation is
    // accepted, plus aggressive wide-angle tiles (worst-case curvature) that the
    // probe demotes to exact.
    let configs = [
        (-0.05, -0.05, 0.10, 0.0),
        (0.10, -0.05, 0.12, 0.1),
        (-0.20, 0.10, 0.20, 0.0),
        (-0.30, -0.30, 0.60, 0.3),
        (0.50, 0.30, 0.90, 0.2),
        (-0.90, -0.20, 1.20, 0.0),
        (0.80, 0.80, 1.00, 0.4),
    ];
    let mut max_err = 0f32;
    let mut sse = 0f64;
    let mut n_both = 0u64;
    let mut n_disagree = 0u64;
    let mut n_total = 0u64;
    let mut interp_cells = 0usize;
    let mut total_cells = 0usize;
    // All `needs_ray_path` models take the coarse path, including equirectangular.
    for cam in [
        simple_radial_fisheye(),
        radial_fisheye(),
        opencv_fisheye(),
        equirectangular(),
    ] {
        for &(x0, y0, span, tilt) in &configs {
            let (o, cs, rs) = grid_basis(x0, y0, span, r, tilt);
            let mut coarse = vec![0f32; 2 * (r * r) as usize];
            let mut exact = vec![0f32; 2 * (r * r) as usize];
            let (ic, tc) = cam.ray_to_pixel_grid_coarse(o, cs, rs, r, r, &mut coarse);
            interp_cells += ic;
            total_cells += tc;
            cam.ray_to_pixel_grid_exact(o, cs, rs, r, r, &mut exact);
            for i in (0..coarse.len()).step_by(2) {
                n_total += 1;
                let (cf, ef) = (coarse[i].is_finite(), exact[i].is_finite());
                if cf != ef {
                    n_disagree += 1;
                    continue;
                }
                if ef {
                    let d = ((coarse[i] - exact[i]).powi(2)
                        + (coarse[i + 1] - exact[i + 1]).powi(2))
                    .sqrt();
                    max_err = max_err.max(d);
                    sse += (d as f64).powi(2);
                    n_both += 1;
                }
            }
        }
    }
    let rms = (sse / n_both.max(1) as f64).sqrt();
    let disagree_frac = n_disagree as f64 / n_total as f64;
    let interp_frac = interp_cells as f64 / total_cells as f64;
    eprintln!(
        "[coarse-grid] stride={COARSE_GRID_STRIDE} tol={COARSE_GRID_TOL_PX} r={r} valid_px={n_both} \
         max_err={max_err:.4}px rms_err={rms:.5}px \
         validity_disagree={n_disagree}/{n_total} ({:.3}%) \
         interpolated_cells={interp_cells}/{total_cells} ({:.1}%)",
        100.0 * disagree_frac,
        100.0 * interp_frac,
    );
    // The fast path must actually be exercised (else the test is vacuous).
    assert!(interp_cells > 0, "no cells were interpolated");
    // The per-cell probe demotes any cell that would exceed COARSE_GRID_TOL_PX to
    // exact, so the worst-case error tracks the tolerance (a hair above it from
    // non-probe interior points of accepted cells). Validity disagreements are
    // confined to a sub-pixel band at the frame/domain edge, so they stay rare.
    assert!(
        max_err < 2.0 * COARSE_GRID_TOL_PX,
        "coarse-grid max error {max_err} px exceeds 2x tol ({}px)",
        2.0 * COARSE_GRID_TOL_PX,
    );
    assert!(
        disagree_frac < 0.01,
        "coarse-grid validity disagreement {disagree_frac} exceeds 1%"
    );
}

#[test]
fn coarse_grid_jacobian_degradation() {
    // Numeric analysis: how much does the piecewise-bilinear coarse warp degrade
    // the central-difference Jacobian (and the SVD derived from it) vs the exact
    // per-node map? Compares J/sigma/anisotropy/major-dir pixel-by-pixel, split by
    // cell-seam (central diff straddles a stride-8 node) vs cell-interior pixels.
    use crate::camera::warp_map::WarpMap;
    const MAX_ANISOTROPY: f32 = 16.0; // mirrors keypoint_subpixel.rs
    let r = 48u32;
    // Gentle/realistic tiles where interpolation is actually accepted (so seams
    // exist), plus a couple of moderately curved ones.
    let configs = [
        (-0.05, -0.05, 0.10, 0.0),
        (0.10, -0.05, 0.12, 0.1),
        (-0.20, 0.10, 0.20, 0.0),
        (0.15, 0.15, 0.25, 0.15),
        (-0.30, -0.10, 0.35, 0.0),
        // Strongly oblique, small span: anisotropic Jacobian at low curvature
        // (cells stay interpolated) — stresses sigma/anisotropy/major-dir.
        (-0.05, -0.05, 0.10, 1.5),
        (0.08, -0.04, 0.12, 2.5),
        (-0.15, 0.05, 0.14, 3.5),
    ];
    let fin = |d: &[f32], c: u32, rr: u32| d[2 * (rr * r + c) as usize].is_finite();
    let nb_ok = |d: &[f32], c: u32, rr: u32| {
        fin(d, c, rr)
            && fin(d, c - 1, rr)
            && fin(d, c + 1, rr)
            && fin(d, c, rr - 1)
            && fin(d, c, rr + 1)
    };

    let (mut jac_abs_max, mut jac_rel_max) = (0f32, 0f32);
    let (mut jac_sse, mut jmag_sse, mut n) = (0f64, 0f64, 0u64);
    let (mut sig_maj_rel_max, mut aniso_abs_max, mut ang_max) = (0f32, 0f32, 0f32);
    let mut aniso_max_e = 1f32;
    let mut aniso_cross = 0u64;
    let (mut seam_sse, mut seam_n, mut int_sse, mut int_n) = (0f64, 0u64, 0f64, 0u64);

    for cam in [simple_radial_fisheye(), radial_fisheye(), opencv_fisheye()] {
        for &(x0, y0, span, tilt) in &configs {
            let (o, cs, rs) = grid_basis(x0, y0, span, r, tilt);
            let mut cd = vec![0f32; 2 * (r * r) as usize];
            let mut ed = vec![0f32; 2 * (r * r) as usize];
            cam.ray_to_pixel_grid(o, cs, rs, r, r, &mut cd); // coarse (fisheye)
            cam.ray_to_pixel_grid_exact(o, cs, rs, r, r, &mut ed); // exact ground truth
            let mut cw = WarpMap::new(r, r, cd.clone());
            cw.compute_svd();
            let mut ew = WarpMap::new(r, r, ed.clone());
            ew.compute_svd();
            for row in 1..r - 1 {
                for col in 1..r - 1 {
                    if !(nb_ok(&cd, col, row) && nb_ok(&ed, col, row)) {
                        continue;
                    }
                    let jc = cw.get_jacobian(col, row);
                    let je = ew.get_jacobian(col, row);
                    let mut df = 0f32;
                    let mut ef = 0f32;
                    for i in 0..2 {
                        for j in 0..2 {
                            df += (jc[i][j] - je[i][j]).powi(2);
                            ef += je[i][j].powi(2);
                        }
                    }
                    let df = df.sqrt();
                    let ef = ef.sqrt().max(1e-12);
                    let rel = df / ef;
                    jac_abs_max = jac_abs_max.max(df);
                    jac_rel_max = jac_rel_max.max(rel);
                    jac_sse += (df as f64).powi(2);
                    jmag_sse += (ef as f64).powi(2);
                    n += 1;

                    let (smaj_c, smin_c, vx_c, vy_c) = cw.get_svd(col, row);
                    let (smaj_e, smin_e, vx_e, vy_e) = ew.get_svd(col, row);
                    sig_maj_rel_max =
                        sig_maj_rel_max.max((smaj_c - smaj_e).abs() / smaj_e.max(1e-6));
                    let an_c = smaj_c / smin_c.max(1e-6);
                    let an_e = smaj_e / smin_e.max(1e-6);
                    aniso_max_e = aniso_max_e.max(an_e);
                    aniso_abs_max = aniso_abs_max.max((an_c - an_e).abs());
                    if (an_c >= MAX_ANISOTROPY) != (an_e >= MAX_ANISOTROPY) {
                        aniso_cross += 1;
                    }
                    // Major-direction angle error (only meaningful when anisotropic).
                    if an_e > 1.2 {
                        let cross = vx_c * vy_e - vy_c * vx_e;
                        let dot = vx_c * vx_e + vy_c * vy_e;
                        ang_max = ang_max.max(cross.atan2(dot).abs().to_degrees());
                    }

                    let seam = col % 8 == 0 || row % 8 == 0;
                    if seam {
                        seam_sse += (rel as f64).powi(2);
                        seam_n += 1;
                    } else {
                        int_sse += (rel as f64).powi(2);
                        int_n += 1;
                    }
                }
            }
        }
    }
    let jac_rms = (jac_sse / n.max(1) as f64).sqrt();
    let jmag_rms = (jmag_sse / n.max(1) as f64).sqrt();
    eprintln!("[jac-degradation] pixels={n}  |J|_F rms={jmag_rms:.3} (scale of the Jacobian)");
    eprintln!(
        "[jac-degradation] dJ_F: abs_max={jac_abs_max:.4} rms={jac_rms:.4}  rel_max={:.2}% rel_rms={:.3}%",
        100.0 * jac_rel_max,
        100.0 * jac_rms / jmag_rms,
    );
    eprintln!(
        "[jac-degradation] sigma_major rel_max={:.3}%  anisotropy abs_max={aniso_abs_max:.4}  max_anisotropy_observed={aniso_max_e:.3} (clamp={MAX_ANISOTROPY})  crossings={aniso_cross}/{n}  major-dir angle_max={ang_max:.3}deg",
        100.0 * sig_maj_rel_max,
    );
    let seam_rms = (seam_sse / seam_n.max(1) as f64).sqrt();
    let int_rms = (int_sse / int_n.max(1) as f64).sqrt();
    eprintln!(
        "[jac-degradation] dJ rel-RMS  seam(col|row%8==0)={:.3}% ({seam_n}px)  interior={:.3}% ({int_n}px)  ratio={:.2}x",
        100.0 * seam_rms,
        100.0 * int_rms,
        seam_rms / int_rms.max(1e-12),
    );

    // The position probe (COARSE_GRID_TOL_PX) bounds the warp, not its
    // derivative, so guard the central-difference Jacobian that compute_svd /
    // compute_jacobians feed to the anisotropic sampler and the GN gradient.
    // Empirically the degradation is deep sub-percent and the kink at stride
    // boundaries is harmless (central differencing averages across it, so seam
    // pixels are no worse than cell interiors).
    assert!(
        jac_rel_max < 0.01,
        "coarse-grid Jacobian worst-case rel error {jac_rel_max} exceeds 1%"
    );
    assert!(
        jac_rms / jmag_rms < 0.005,
        "coarse-grid Jacobian rel-RMS exceeds 0.5%"
    );
    assert_eq!(
        aniso_cross, 0,
        "coarse grid flipped a pixel across the MAX_ANISOTROPY clamp"
    );
    assert!(
        ang_max < 1.0,
        "major-direction error {ang_max} deg exceeds 1 deg"
    );
    assert!(
        seam_rms <= int_rms * 2.0,
        "seam Jacobian error unexpectedly dominates interior (kink not averaged out)"
    );
}
