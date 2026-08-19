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

fn equidistant_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 500.0,
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
        equidistant_fisheye(),
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
            // Multi-coefficient fisheye / equirectangular: a forward ray still
            // projects, but there is no analytic Jacobian yet. (The
            // one-coefficient equidistant pair — EQUIDISTANT_FISHEYE and
            // SIMPLE_RADIAL_FISHEYE — takes the analytic path below.)
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

// -----------------------------------------------------------------------
// Equidistant seed model: `SimpleRadialFisheye { k1 = 0 }`
//
// The fisheye-seed campaign (`scripts/notes-fisheye-seed.md`, Phase 1)
// represents the single-parameter equidistant map `θ = r/f` as
// `SimpleRadialFisheye` with `k1 = 0`. These pin the two facts every
// geometric kernel downstream leans on: the map is EXACTLY `θ = r/f` (no
// small-angle or `z = 1` approximation anywhere in it), and it stays exact
// for `θ ≥ 90°` — the backward-of-image-plane rays a >180° capture really
// observes.
// -----------------------------------------------------------------------

/// The Phase-1 seed camera: one focal, centred principal point, `k1 = 0`.
fn equidistant_seed(f: f64, w: u32, h: u32) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: f,
            principal_point_x: w as f64 / 2.0,
            principal_point_y: h as f64 / 2.0,
            radial_distortion_k1: 0.0,
        },
        width: w,
        height: h,
    }
}

/// Canonical-frame unit ray at incidence angle `theta` off the `−Z` axis,
/// azimuth `phi` measured in the canonical (y-up) image plane.
fn ray_at(theta: f64, phi: f64) -> [f64; 3] {
    [
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        -theta.cos(),
    ]
}

#[test]
fn equidistant_seed_is_exactly_theta_over_f() {
    let f = 130.0;
    let cam = equidistant_seed(f, 480, 480);
    let (cx, cy) = cam.principal_point();
    // Sweep well past 90°: a 211° Insta360 capture reaches θ ≈ 105°.
    for deg in [
        0.0f64, 1.0, 30.0, 60.0, 89.0, 90.0, 91.0, 105.0, 130.0, 179.0,
    ] {
        let theta = deg.to_radians();
        for phi_deg in [0.0f64, 37.0, 90.0, 180.0, 263.0] {
            let phi = phi_deg.to_radians();
            let ray = ray_at(theta, phi);
            let (u, v) = cam
                .ray_to_pixel(ray)
                .unwrap_or_else(|| panic!("ray_to_pixel None at θ={deg}°, φ={phi_deg}°"));
            // Forward: r = f·θ, and the pixel azimuth is the ray azimuth
            // (pixel v grows DOWN, so the canonical +y ray lands at −v).
            let r = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt();
            assert_relative_eq!(r, f * theta, epsilon = 1e-9);
            if theta > 1e-9 {
                assert_relative_eq!(u - cx, f * theta * phi.cos(), epsilon = 1e-9);
                assert_relative_eq!(v - cy, -f * theta * phi.sin(), epsilon = 1e-9);
            }
            // Inverse: back to the same unit ray, exactly.
            let back = cam.pixel_to_ray(u, v);
            for c in 0..3 {
                assert_relative_eq!(back[c], ray[c], epsilon = 1e-9);
            }
        }
    }
}

#[test]
fn equidistant_seed_round_trips_over_the_whole_sensor() {
    // Pixel → ray → pixel over a dense grid of a 480² fisheye circle at a
    // focal whose image circle (θ = 105°) is inscribed in the frame.
    let f = 480.0 / 2.0 / 105.0_f64.to_radians();
    let cam = equidistant_seed(f, 480, 480);
    let mut worst = 0.0f64;
    let mut n_past_90 = 0usize;
    for iy in 0..48 {
        for ix in 0..48 {
            let (u, v) = (5.0 + 10.0 * ix as f64, 5.0 + 10.0 * iy as f64);
            let ray = cam.pixel_to_ray(u, v);
            assert_relative_eq!(
                (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt(),
                1.0,
                epsilon = 1e-12
            );
            if ray[2] > 0.0 {
                n_past_90 += 1; // canonical z > 0 ⇒ θ > 90°, behind the image plane
            }
            let (u2, v2) = cam.ray_to_pixel(ray).expect("ray_to_pixel None on-sensor");
            worst = worst.max((u2 - u).hypot(v2 - v));
        }
    }
    assert!(worst < 1e-9, "worst pixel round-trip {worst}");
    assert!(
        n_past_90 > 100,
        "grid did not exercise θ > 90° (only {n_past_90} samples)"
    );
}

#[test]
fn equidistant_seed_pixel_jacobian_is_the_analytic_one() {
    // The finite-difference fallback in `pose_refine` / `bundle_adjust` is
    // selected by this flag. `SimpleRadialFisheye` carries the same
    // closed-form `θ_d = θ·(1 + k1·θ²)` derivative as the native model (k1 = 0
    // here), so it takes the analytic path — including past 90°, where the
    // perspective path would return None on rz ≤ 0. The MULTI-COEFFICIENT
    // fisheye family still has no analytic form.
    let cam = equidistant_seed(130.0, 480, 480);
    assert!(cam.model.supports_pixel_jacobian());
    let native = equidistant_native(130.0, 480, 480);
    for deg in [30.0f64, 89.0, 91.0, 130.0] {
        let ray = ray_at(deg.to_radians(), 0.4);
        let (_, seed_j) = cam.ray_to_pixel_with_jacobian(ray).unwrap();
        let (_, native_j) = native.ray_to_pixel_with_jacobian(ray).unwrap();
        // `k1 = 0` is not merely close to the distortion-free map — the
        // shared kernel collapses to its exact arithmetic.
        for row in 0..2 {
            for c in 0..3 {
                assert_eq!(
                    seed_j[row][c].to_bits(),
                    native_j[row][c].to_bits(),
                    "k1 = 0 Jacobian [{row}][{c}] at θ={deg}° is not the equidistant one",
                );
            }
        }
    }
}

#[test]
fn equidistant_seed_central_difference_jacobian_survives_past_90_degrees() {
    // What the `project_with_jac` fallback does at a backward ray — the path
    // the multi-coefficient fisheye models still take, exercised here on the
    // map whose analytic derivative is known: the ±h probes must all stay
    // in-domain, and the numeric derivative must be stable.
    let f = 130.0;
    let cam = equidistant_seed(f, 480, 480);
    let h = 1e-6;
    for deg in [95.0f64, 105.0, 130.0] {
        let d = ray_at(deg.to_radians(), 0.7);
        // Place a point at range 3 along that direction.
        let p = [3.0 * d[0], 3.0 * d[1], 3.0 * d[2]];
        for c in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[c] += h;
            pm[c] -= h;
            assert!(
                cam.ray_to_pixel(pp).is_some() && cam.ray_to_pixel(pm).is_some(),
                "central-difference probe left the domain at θ={deg}°"
            );
        }
        // ∂u/∂p against the closed form: u = f·θ·cos φ + cx with
        // θ = atan2(√(x²+y²), −z) in canonical components.
        let (u0, v0) = cam.ray_to_pixel(p).unwrap();
        let mut pp = p;
        pp[0] += h;
        let (u1, v1) = cam.ray_to_pixel(pp).unwrap();
        let du = (u1 - u0) / h;
        let dv = (v1 - v0) / h;
        // Numeric-vs-numeric at a 10× coarser step: a stable derivative.
        let mut pc = p;
        pc[0] += 10.0 * h;
        let (u2, v2) = cam.ray_to_pixel(pc).unwrap();
        assert_relative_eq!(du, (u2 - u0) / (10.0 * h), epsilon = 1e-3);
        assert_relative_eq!(dv, (v2 - v0) / (10.0 * h), epsilon = 1e-3);
        assert!(du.is_finite() && dv.is_finite());
    }
}

#[test]
fn equidistant_ray_at_the_exact_antipode_aliases_the_principal_point() {
    // KNOWN, DOCUMENTED domain edge (audit finding, Phase 1): a ray exactly
    // along +Z (θ = π, r_xy = 0) hits `distort_ray_equidistant`'s r_xy
    // early-return and projects to the principal point — the same pixel as
    // θ = 0. It is a measure-zero direction 75° outside any real capture's
    // FOV, and the surrounding neighbourhood is correct, so nothing in the
    // seed path can reach it; this test pins the behaviour so a future
    // fisheye-native stage does not discover it by surprise.
    let cam = equidistant_seed(130.0, 480, 480);
    let (cx, cy) = cam.principal_point();
    let (u, v) = cam.ray_to_pixel([0.0, 0.0, 1.0]).unwrap();
    assert_relative_eq!(u, cx, epsilon = 1e-12);
    assert_relative_eq!(v, cy, epsilon = 1e-12);
    // One micro-radian off the antipode the map is already correct.
    let near = ray_at(std::f64::consts::PI - 1e-6, 0.0);
    let (u2, _v2) = cam.ray_to_pixel(near).unwrap();
    assert_relative_eq!(
        u2 - cx,
        130.0 * (std::f64::consts::PI - 1e-6),
        epsilon = 1e-6
    );
}

#[test]
fn equidistant_seed_batch_maps_match_the_scalar_ones_past_90_degrees() {
    // `pixel_to_ray_batch` / `ray_to_pixel_batch` are what the seed scripts
    // call; assert they are the scalar maps, including for backward rays.
    let cam = equidistant_seed(130.0, 480, 480);
    let mut pixels = Vec::new();
    let mut rays = Vec::new();
    for deg in [0.0f64, 45.0, 90.0, 100.0, 140.0] {
        for phi_deg in [0.0f64, 120.0, 240.0] {
            let r = ray_at(deg.to_radians(), phi_deg.to_radians());
            rays.push(r);
            let (u, v) = cam.ray_to_pixel(r).unwrap();
            pixels.push([u, v]);
        }
    }
    let back = cam.pixel_to_ray_batch(&pixels);
    let fwd = cam.ray_to_pixel_batch(&rays);
    for i in 0..rays.len() {
        let s = cam.pixel_to_ray(pixels[i][0], pixels[i][1]);
        for c in 0..3 {
            assert_relative_eq!(back[i][c], s[c], epsilon = 0.0);
            assert_relative_eq!(back[i][c], rays[i][c], epsilon = 1e-9);
        }
        let p = fwd[i].expect("ray_to_pixel_batch None on an in-domain ray");
        assert_relative_eq!(p[0], pixels[i][0], epsilon = 0.0);
        assert_relative_eq!(p[1], pixels[i][1], epsilon = 0.0);
    }
}

// -----------------------------------------------------------------------
// EQUIDISTANT_FISHEYE — the native `θ = r/f` model
//
// The tests above pin the same map carried as `SimpleRadialFisheye { k1 = 0 }`
// (the pre-Phase-3a convention); these pin the native model, its exactness at
// and past 90°, and its analytic pixel Jacobian.
// -----------------------------------------------------------------------

/// The native model at focal `f`, principal point centred — the exact
/// counterpart of `equidistant_seed`'s `SimpleRadialFisheye { k1 = 0 }`.
fn equidistant_native(f: f64, w: u32, h: u32) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: f,
            principal_point_x: w as f64 / 2.0,
            principal_point_y: h as f64 / 2.0,
        },
        width: w,
        height: h,
    }
}

#[test]
fn equidistant_native_is_exactly_theta_over_f_both_ways() {
    let f = 130.0;
    let cam = equidistant_native(f, 480, 480);
    let (cx, cy) = cam.principal_point();
    for deg in [
        0.0f64, 1.0, 30.0, 60.0, 89.0, 90.0, 91.0, 105.0, 130.0, 179.0,
    ] {
        let theta = deg.to_radians();
        for phi_deg in [0.0f64, 37.0, 90.0, 180.0, 263.0] {
            let phi = phi_deg.to_radians();
            let ray = ray_at(theta, phi);
            let (u, v) = cam
                .ray_to_pixel(ray)
                .unwrap_or_else(|| panic!("ray_to_pixel None at θ={deg}°, φ={phi_deg}°"));
            // Forward: r = f·θ, pixel azimuth = ray azimuth (v grows DOWN).
            let r = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt();
            assert_relative_eq!(r, f * theta, epsilon = 1e-9);
            if theta > 1e-9 {
                assert_relative_eq!(u - cx, f * theta * phi.cos(), epsilon = 1e-9);
                assert_relative_eq!(v - cy, -f * theta * phi.sin(), epsilon = 1e-9);
            }
            // Inverse: exactly back to the same unit ray — no Newton, no blend.
            let back = cam.pixel_to_ray(u, v);
            for c in 0..3 {
                assert_relative_eq!(back[c], ray[c], epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn equidistant_native_agrees_with_the_k1_zero_convention() {
    // The cross-check the seed scripts also run: `SimpleRadialFisheye` with
    // k1 = 0 parameterizes the identical map, so both representations must
    // project and unproject to the same numbers. They are bit-identical
    // everywhere except in the polynomial family's 90°–100° wide-angle blend
    // band, where `SimpleRadialFisheye` lerps two identical rays and
    // renormalizes — a round-off of order 1e-16, so 1e-12 is the tolerance.
    let f = 130.0;
    let native = equidistant_native(f, 480, 480);
    let legacy = equidistant_seed(f, 480, 480);
    for deg in [0.0f64, 15.0, 60.0, 89.0, 90.0, 91.0, 95.0, 105.0, 130.0] {
        for phi_deg in [0.0f64, 71.0, 200.0] {
            let ray = ray_at(deg.to_radians(), phi_deg.to_radians());
            let (un, vn) = native.ray_to_pixel(ray).unwrap();
            let (ul, vl) = legacy.ray_to_pixel(ray).unwrap();
            assert_relative_eq!(un, ul, epsilon = 1e-12);
            assert_relative_eq!(vn, vl, epsilon = 1e-12);
            let rn = native.pixel_to_ray(un, vn);
            let rl = legacy.pixel_to_ray(ul, vl);
            for c in 0..3 {
                assert_relative_eq!(rn[c], rl[c], epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn equidistant_native_ray_at_the_exact_antipode_aliases_the_principal_point() {
    // Same documented domain edge as the k1 = 0 convention: θ = π with
    // r_xy = 0 hits the on-axis early return and projects to the principal
    // point, aliasing θ = 0. Measure-zero, 75° outside any real capture.
    let cam = equidistant_native(130.0, 480, 480);
    let (cx, cy) = cam.principal_point();
    let (u, v) = cam.ray_to_pixel([0.0, 0.0, 1.0]).unwrap();
    assert_relative_eq!(u, cx, epsilon = 1e-12);
    assert_relative_eq!(v, cy, epsilon = 1e-12);
    // One micro-radian off the antipode the map is already correct.
    let near = ray_at(std::f64::consts::PI - 1e-6, 0.0);
    let (u2, _) = cam.ray_to_pixel(near).unwrap();
    assert_relative_eq!(
        u2 - cx,
        130.0 * (std::f64::consts::PI - 1e-6),
        epsilon = 1e-6
    );
    // The projection is defined there; the DERIVATIVE is not — θ/ρ diverges.
    assert!(cam.ray_to_pixel_with_jacobian([0.0, 0.0, 1.0]).is_none());
}

/// The analytic Jacobian against a central difference over a whole synthetic
/// sensor: a 480² fisheye frame whose image circle reaches θ = 130°, plus
/// explicit θ bands straddling 90°. Nothing here is allowed to be rejected —
/// `rz ≤ 0` (θ ≥ 90°) is the periphery this model exists to carry.
#[test]
fn equidistant_native_jacobian_matches_central_difference_over_the_sensor() {
    let f = 480.0 / 2.0 / 130.0_f64.to_radians();
    let cam = equidistant_native(f, 480, 480);
    let (cx, cy) = cam.principal_point();
    let h = 1e-6;
    let mut samples = 0usize;
    let mut past_90 = 0usize;
    let mut worst = 0.0f64;

    let check = |ray: [f64; 3], past: &mut usize, worst: &mut f64| -> usize {
        let (uv, jac) = cam
            .ray_to_pixel_with_jacobian(ray)
            .expect("analytic Jacobian None on an in-domain equidistant ray");
        let direct = cam.ray_to_pixel(ray).unwrap();
        assert_relative_eq!(uv.0, direct.0, epsilon = 1e-12);
        assert_relative_eq!(uv.1, direct.1, epsilon = 1e-12);
        if ray[2] > 0.0 {
            *past += 1;
        }
        let mut n = 0;
        for c in 0..3 {
            let mut rp = ray;
            let mut rm = ray;
            rp[c] += h;
            rm[c] -= h;
            let (up, vp) = cam.ray_to_pixel(rp).unwrap();
            let (um, vm) = cam.ray_to_pixel(rm).unwrap();
            let fd_u = (up - um) / (2.0 * h);
            let fd_v = (vp - vm) / (2.0 * h);
            for (a, fd) in [(jac[0][c], fd_u), (jac[1][c], fd_v)] {
                let rel = (a - fd).abs() / (1.0 + a.abs());
                *worst = worst.max(rel);
                assert!(
                    rel <= 1e-6,
                    "equidistant ∂/∂r[{c}]: analytic {a} vs central-diff {fd} (rel {rel})",
                );
            }
            n += 1;
        }
        n
    };

    // (a) The whole sensor: every pixel of a 24×24 grid, at three ray scales
    // (the map is degree-0 homogeneous, so the Jacobian scales as 1/‖r‖).
    for iy in 0..24 {
        for ix in 0..24 {
            let u = (ix as f64 + 0.5) * 480.0 / 24.0;
            let v = (iy as f64 + 0.5) * 480.0 / 24.0;
            if (u - cx).hypot(v - cy) > 239.0 {
                continue; // outside the image circle
            }
            let base = cam.pixel_to_ray(u, v);
            for scale in [0.5f64, 1.0, 3.0] {
                let ray = [base[0] * scale, base[1] * scale, base[2] * scale];
                samples += check(ray, &mut past_90, &mut worst);
            }
        }
    }

    // (b) Explicit θ bands straddling 90°, at several azimuths.
    for deg in [60.0f64, 89.0, 91.0, 105.0, 130.0] {
        for phi_deg in [0.0f64, 45.0, 137.0, 250.0, 330.0] {
            let ray = ray_at(deg.to_radians(), phi_deg.to_radians());
            samples += check(ray, &mut past_90, &mut worst);
        }
    }

    assert!(samples > 1000, "thin coverage: only {samples} samples");
    assert!(past_90 > 20, "grid did not exercise θ > 90° ({past_90})");
    eprintln!("[equidistant-jac] {samples} samples, worst rel error {worst:.3e}");
}

#[test]
fn equidistant_native_jacobian_on_axis_is_the_pinhole_limit() {
    // φ is undefined on the optical axis, but the limit is not: θ/ρ → 1/rz,
    // the off-diagonal factor vanishes and the third column goes to zero,
    // leaving diag(f/rz, −f/rz) — the small-angle pinhole Jacobian. Assert
    // both the exact-axis branch and its continuity from nearby rays.
    let f = 130.0;
    let cam = equidistant_native(f, 480, 480);
    let (cx, cy) = cam.principal_point();
    let ((u, v), jac) = cam.ray_to_pixel_with_jacobian([0.0, 0.0, -2.0]).unwrap();
    assert_relative_eq!(u, cx, epsilon = 1e-15);
    assert_relative_eq!(v, cy, epsilon = 1e-15);
    // rz (optical) = 2, so ∂u/∂x = f/2 and ∂v/∂y = −f/2.
    assert_relative_eq!(jac[0][0], f / 2.0, epsilon = 1e-12);
    assert_relative_eq!(jac[1][1], -f / 2.0, epsilon = 1e-12);
    for c in [1usize, 2] {
        assert_relative_eq!(jac[0][c], 0.0, epsilon = 1e-15);
    }
    assert_relative_eq!(jac[1][0], 0.0, epsilon = 1e-15);
    assert_relative_eq!(jac[1][2], 0.0, epsilon = 1e-15);

    // Continuity: approaching the axis from an arbitrary azimuth converges to
    // the same matrix, direction-independently.
    for eps in [1e-4f64, 1e-6, 1e-9] {
        for phi_deg in [0.0f64, 61.0, 233.0] {
            let phi = phi_deg.to_radians();
            let ray = [2.0 * eps * phi.cos(), 2.0 * eps * phi.sin(), -2.0];
            let (_, j) = cam.ray_to_pixel_with_jacobian(ray).unwrap();
            assert_relative_eq!(j[0][0], f / 2.0, epsilon = 1e-6);
            assert_relative_eq!(j[1][1], -f / 2.0, epsilon = 1e-6);
            assert_relative_eq!(j[0][1], 0.0, epsilon = 1e-6);
            assert_relative_eq!(j[1][0], 0.0, epsilon = 1e-6);
        }
    }
}

#[test]
fn equidistant_native_jacobian_is_scale_invariant_in_the_ray() {
    // The map is degree-0 homogeneous, so J·r = 0 (Euler) and J(s·r) = J(r)/s.
    let cam = equidistant_native(130.0, 480, 480);
    for deg in [20.0f64, 90.0, 120.0] {
        let ray = ray_at(deg.to_radians(), 0.9);
        let (_, j1) = cam.ray_to_pixel_with_jacobian(ray).unwrap();
        for row in &j1 {
            let dot = (0..3).map(|c| row[c] * ray[c]).sum::<f64>();
            assert_relative_eq!(dot, 0.0, epsilon = 1e-10);
        }
        let s = 4.0;
        let scaled = [ray[0] * s, ray[1] * s, ray[2] * s];
        let (_, j2) = cam.ray_to_pixel_with_jacobian(scaled).unwrap();
        for row in 0..2 {
            for c in 0..3 {
                assert_relative_eq!(j2[row][c], j1[row][c] / s, epsilon = 1e-10);
            }
        }
    }
}

// -----------------------------------------------------------------------
// SIMPLE_RADIAL_FISHEYE — the one-coefficient equidistant map
//
// `θ_d = θ·(1 + k1·θ²)` shares the native model's closed-form derivative
// (`k1 = 0` recovers it exactly). The BA's `opt_k1` rung linearizes through
// this Jacobian, so it is pinned here the same way the native one is:
// against a central difference over a field that runs past 90°, with the
// domain and the on-axis limit checked alongside.
// -----------------------------------------------------------------------

/// A `SIMPLE_RADIAL_FISHEYE` at focal `f` and coefficient `k1`, principal
/// point centred in a 480² frame.
fn simple_radial_fisheye_at(f: f64, k1: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: k1,
        },
        width: 480,
        height: 480,
    }
}

#[test]
fn simple_radial_fisheye_jacobian_matches_central_difference_past_90_degrees() {
    let f = 130.0;
    let h = 1e-6;
    let mut samples = 0usize;
    let mut past_90 = 0usize;
    let mut worst = 0.0f64;
    // Both signs of curvature, plus the k1 = 0 degeneracy through the same
    // code path. |k1| here keeps `1 + 3·k1·θ²` positive out to θ = 170°.
    for &k1 in &[0.0f64, 0.1, 0.05, -0.02] {
        let cam = simple_radial_fisheye_at(f, k1);
        for ti in 0..18 {
            let theta = (5.0 + 10.0 * ti as f64).to_radians();
            for phi_deg in [0.0f64, 43.0, 137.0, 250.0, 331.0] {
                let base = ray_at(theta, phi_deg.to_radians());
                // Degree-0 homogeneous: the Jacobian must scale as 1/‖r‖.
                for scale in [0.4f64, 1.0, 6.0] {
                    let ray = [base[0] * scale, base[1] * scale, base[2] * scale];
                    let (uv, jac) = cam
                        .ray_to_pixel_with_jacobian(ray)
                        .expect("analytic Jacobian None on an in-domain radial-fisheye ray");
                    // The projection the Jacobian belongs to is the projection
                    // the rest of the pipeline uses.
                    let direct = cam.ray_to_pixel(ray).unwrap();
                    assert_relative_eq!(uv.0, direct.0, epsilon = 1e-12);
                    assert_relative_eq!(uv.1, direct.1, epsilon = 1e-12);
                    if ray[2] > 0.0 {
                        past_90 += 1;
                    }
                    for c in 0..3 {
                        let mut rp = ray;
                        let mut rm = ray;
                        rp[c] += h;
                        rm[c] -= h;
                        let (up, vp) = cam.ray_to_pixel(rp).unwrap();
                        let (um, vm) = cam.ray_to_pixel(rm).unwrap();
                        let fd_u = (up - um) / (2.0 * h);
                        let fd_v = (vp - vm) / (2.0 * h);
                        for (a, fd) in [(jac[0][c], fd_u), (jac[1][c], fd_v)] {
                            let rel = (a - fd).abs() / (1.0 + a.abs());
                            worst = worst.max(rel);
                            assert!(
                                rel <= 1e-6,
                                "k1={k1} ∂/∂r[{c}] at θ={:.0}°: analytic {a} vs \
                                 central-diff {fd} (rel {rel})",
                                theta.to_degrees(),
                            );
                        }
                        samples += 1;
                    }
                }
            }
        }
    }
    assert!(samples > 1000, "thin coverage: only {samples} samples");
    assert!(past_90 > 100, "grid did not exercise θ > 90° ({past_90})");
    eprintln!("[radial-fisheye-jac] {samples} samples, worst rel error {worst:.3e}");
}

#[test]
fn simple_radial_fisheye_k1_zero_jacobian_is_bitwise_the_equidistant_one() {
    // The shared kernel is the equidistant one with `θ_d = θ·(1 + 0·θ²)` and
    // `dθ_d/dθ = 1 + 3·0·θ²`: every multiplication by the unit collapses, so
    // this is bit-identical rather than merely close. That is what lets the
    // BA promote EQUIDISTANT_FISHEYE → SIMPLE_RADIAL_FISHEYE(k1 = 0) without
    // moving the geometry.
    let f = 137.5;
    let seed = simple_radial_fisheye_at(f, 0.0);
    let native = equidistant_native(f, 480, 480);
    let mut samples = 0usize;
    for ti in 0..24 {
        let theta = (2.0 + 7.0 * ti as f64).to_radians();
        for phi_deg in [0.0f64, 29.0, 91.0, 188.0, 300.0] {
            let ray = ray_at(theta, phi_deg.to_radians());
            let ((us, vs), js) = seed.ray_to_pixel_with_jacobian(ray).unwrap();
            let ((un, vn), jn) = native.ray_to_pixel_with_jacobian(ray).unwrap();
            assert_eq!(us.to_bits(), un.to_bits());
            assert_eq!(vs.to_bits(), vn.to_bits());
            for row in 0..2 {
                for c in 0..3 {
                    assert_eq!(
                        js[row][c].to_bits(),
                        jn[row][c].to_bits(),
                        "[{row}][{c}] at θ={:.0}°",
                        theta.to_degrees(),
                    );
                }
            }
            samples += 1;
        }
    }
    assert!(samples >= 100);
    // The two shared domain edges agree too: the on-axis forward limit is
    // finite, the antipode is not.
    for cam in [&seed, &native] {
        let (_, j) = cam.ray_to_pixel_with_jacobian([0.0, 0.0, -2.0]).unwrap();
        assert_relative_eq!(j[0][0], f / 2.0, epsilon = 1e-12);
        assert!(cam.ray_to_pixel_with_jacobian([0.0, 0.0, 1.0]).is_none());
    }
}

#[test]
fn simple_radial_fisheye_jacobian_shares_the_projection_domain() {
    // A `k1` strong enough to fold `θ_d` non-positive takes the projection out
    // of domain; the derivative must go with it (there is nothing to
    // differentiate past the fold). Inside the fold both are `Some`.
    let cam = simple_radial_fisheye_at(130.0, -0.35);
    let mut folded = 0usize;
    let mut fine = 0usize;
    for deg in [10.0f64, 45.0, 90.0, 100.0, 120.0, 150.0, 175.0] {
        let ray = ray_at(deg.to_radians(), 0.6);
        match cam.ray_to_pixel(ray) {
            Some(_) => {
                assert!(
                    cam.ray_to_pixel_with_jacobian(ray).is_some(),
                    "no Jacobian at an in-domain θ={deg}°"
                );
                fine += 1;
            }
            None => {
                assert!(
                    cam.ray_to_pixel_with_jacobian(ray).is_none(),
                    "Jacobian past the θ_d fold at θ={deg}°"
                );
                folded += 1;
            }
        }
    }
    // `1 + k1·θ²` crosses zero at θ = 1/√0.35 ≈ 1.69 rad ≈ 97°.
    assert!(
        fine >= 3 && folded >= 3,
        "{fine} in-domain, {folded} folded"
    );
}

#[test]
fn simple_radial_fisheye_pixel_to_ray_inverts_the_projection_past_90_degrees() {
    // `pixel_to_ray` is the map the bundle adjustment's retriangulation and
    // direction re-estimation read, so it has to be the true inverse where
    // the observations are — including the rim of a >180° capture. The
    // wide-angle blend used to hand back the identity (`θ = r_d`) ray past
    // 90°, dropping `k1` exactly where `k1·θ³` is largest: at θ = 105° and
    // k1 = 0.02 that is 0.11 rad ≈ 6° of ray error.
    let f = 130.0;
    let mut worst_rad = 0.0f64;
    let mut worst_px = 0.0f64;
    let mut past_90 = 0usize;
    for &k1 in &[0.02f64, 0.05, -0.02] {
        let cam = simple_radial_fisheye_at(f, k1);
        for ti in 0..27 {
            let theta = (2.0 + 5.0 * ti as f64).to_radians();
            for phi_deg in [0.0f64, 47.0, 133.0, 271.0] {
                let ray = ray_at(theta, phi_deg.to_radians());
                let (u, v) = cam.ray_to_pixel(ray).unwrap();
                let back = cam.pixel_to_ray(u, v);
                let dot = (0..3).map(|c| back[c] * ray[c]).sum::<f64>();
                worst_rad = worst_rad.max(dot.clamp(-1.0, 1.0).acos());
                let (u2, v2) = cam.ray_to_pixel(back).unwrap();
                worst_px = worst_px.max((u2 - u).hypot(v2 - v));
                if theta > std::f64::consts::FRAC_PI_2 {
                    past_90 += 1;
                }
            }
        }
    }
    assert!(past_90 >= 100, "not enough periphery: {past_90}");
    // The floor is the Newton recovery's own step tolerance (`UNDISTORT_EPS`,
    // 1e-10 on the step, a few 1e-8 rad on θ near the wide end) — 3e-6 px at
    // this focal. The pixel round trip closes to rounding.
    assert!(
        worst_rad < 1e-7 && worst_px < 1e-6,
        "round-trip error {worst_rad} rad / {worst_px} px"
    );
}

#[test]
fn simple_radial_fisheye_jacobian_on_axis_is_the_pinhole_limit_for_any_k1() {
    // On the optical axis `θ_d/ρ → 1/rz` and `dθ_d/dθ → 1` whatever `k1` is:
    // the curvature term is O(θ²) and vanishes. Both the exact-axis branch and
    // the approach from a generic azimuth.
    let f = 130.0;
    for &k1 in &[0.0f64, 0.2, -0.1] {
        let cam = simple_radial_fisheye_at(f, k1);
        let (_, jac) = cam.ray_to_pixel_with_jacobian([0.0, 0.0, -2.0]).unwrap();
        assert_relative_eq!(jac[0][0], f / 2.0, epsilon = 1e-12);
        assert_relative_eq!(jac[1][1], -f / 2.0, epsilon = 1e-12);
        for eps in [1e-4f64, 1e-6, 1e-9] {
            for phi_deg in [0.0f64, 61.0, 233.0] {
                let phi = phi_deg.to_radians();
                let ray = [2.0 * eps * phi.cos(), 2.0 * eps * phi.sin(), -2.0];
                let (_, j) = cam.ray_to_pixel_with_jacobian(ray).unwrap();
                assert_relative_eq!(j[0][0], f / 2.0, epsilon = 1e-6);
                assert_relative_eq!(j[1][1], -f / 2.0, epsilon = 1e-6);
                assert_relative_eq!(j[0][1], 0.0, epsilon = 1e-6);
                assert_relative_eq!(j[1][0], 0.0, epsilon = 1e-6);
            }
        }
    }
}

// -----------------------------------------------------------------------
// SFMTOOL_FISHEYE — equidistant base + monotone radial spline
//
// `θ_d(θ) = θ + δ(θ)` with `δ` a cubic open-uniform B-spline on
// `[0, θ_max]` whose first two basis functions are omitted (`δ(0) = 0`,
// `δ'(0) = 0` — the center-anchored gauge) and which is held constant past
// `θ_max`. Zero coefficients ≡ EQUIDISTANT_FISHEYE bit for bit.
// -----------------------------------------------------------------------

/// A gently flattening 8-coefficient spline out to `θ_max = 2.0` rad
/// (≈114.6°) — the shape of a wide lens that departs from equidistant toward
/// the rim, comfortably inside the monotonicity invariant.
const FLATTENING_BSPLINE: [f64; 8] = [-0.001, -0.004, -0.01, -0.02, -0.03, -0.05, -0.07, -0.09];

/// The [`FLATTENING_BSPLINE`]'s spline domain end.
const BSPLINE_THETA_MAX: f64 = 2.0;

/// An `SFMTOOL_FISHEYE` at focal `f` with the given coefficients, principal
/// point centred in a 480² frame — the spline sibling of
/// `simple_radial_fisheye_at`.
fn sfmtool_fisheye_at(f: f64, coeffs: Vec<f64>) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: BSPLINE_THETA_MAX,
            bspline: coeffs,
        },
        width: 480,
        height: 480,
    }
}

/// Assert `cam` reproduces `native` **bit for bit** on every public map —
/// `ray_to_pixel`, its Jacobian, `pixel_to_ray` and the tangent-plane
/// `project`/`unproject` pair — over a field running out to 163°.
///
/// This is the short-circuit contract shared by every inactive spline: it is
/// the exact `EQUIDISTANT_FISHEYE` arithmetic that runs, not an equivalent
/// evaluation of a spline that happens to sum to zero.
fn assert_bitwise_equidistant(cam: &CameraIntrinsics, native: &CameraIntrinsics) {
    let mut samples = 0usize;
    for ti in 0..24 {
        let theta = (2.0 + 7.0 * ti as f64).to_radians(); // out to 163°
        for phi_deg in [0.0f64, 29.0, 91.0, 188.0, 300.0] {
            let ray = ray_at(theta, phi_deg.to_radians());
            // ray_to_pixel and its Jacobian.
            let ((us, vs), js) = cam.ray_to_pixel_with_jacobian(ray).unwrap();
            let ((un, vn), jn) = native.ray_to_pixel_with_jacobian(ray).unwrap();
            assert_eq!(us.to_bits(), un.to_bits());
            assert_eq!(vs.to_bits(), vn.to_bits());
            // The plain projection path, compared like for like (it rounds
            // `θ·rx/ρ` in a different association than the Jacobian kernel's
            // `θ·(rx/ρ)`, so plain vs plain).
            let (du, dv) = cam.ray_to_pixel(ray).unwrap();
            let (dnu, dnv) = native.ray_to_pixel(ray).unwrap();
            assert_eq!(du.to_bits(), dnu.to_bits());
            assert_eq!(dv.to_bits(), dnv.to_bits());
            for row in 0..2 {
                for c in 0..3 {
                    assert_eq!(
                        js[row][c].to_bits(),
                        jn[row][c].to_bits(),
                        "[{row}][{c}] at θ={:.0}°",
                        theta.to_degrees(),
                    );
                }
            }
            // pixel_to_ray from the shared pixel.
            let rs = cam.pixel_to_ray(us, vs);
            let rn = native.pixel_to_ray(un, vn);
            for c in 0..3 {
                assert_eq!(rs[c].to_bits(), rn[c].to_bits());
            }
            samples += 1;
        }
    }
    assert!(samples >= 100);
    // The tangent-plane pair (project / unproject) short-circuits too;
    // meaningful below 90°, where the tangent plane exists.
    for &[x, y] in &test_points() {
        let (us, vs) = cam.project(x, y);
        let (un, vn) = native.project(x, y);
        assert_eq!(us.to_bits(), un.to_bits());
        assert_eq!(vs.to_bits(), vn.to_bits());
        let (xs, ys) = cam.unproject(us, vs);
        let (xn, yn) = native.unproject(un, vn);
        assert_eq!(xs.to_bits(), xn.to_bits());
        assert_eq!(ys.to_bits(), yn.to_bits());
    }
}

#[test]
fn sfmtool_fisheye_zero_bspline_is_bitwise_the_equidistant_model() {
    // The promotion contract: an empty OR all-zero spline short-circuits to
    // the exact EQUIDISTANT_FISHEYE arithmetic on every public map, so
    // promoting a solved equidistant camera into this model moves nothing.
    // Bitwise, not merely close — the same standard as the
    // `SimpleRadialFisheye { k1 = 0 }` convention.
    let f = 137.5;
    let native = equidistant_native(f, 480, 480);
    for coeffs in [vec![], vec![0.0; 8]] {
        assert_bitwise_equidistant(&sfmtool_fisheye_at(f, coeffs), &native);
    }
}

#[test]
fn sfmtool_fisheye_degenerate_theta_max_is_bitwise_the_equidistant_model() {
    // A domain end that is not positive and finite leaves the basis no
    // interval to live on, so the map is the identity however live the
    // coefficients are — and it takes the SAME short-circuit as a zero
    // spline. Not a nicety: `+∞` puts every knot at infinity, and the Cox–de
    // Boor recurrence then computes `inf · 0` and hands back NaN through
    // every projection, so this is also the no-NaN gate.
    let f = 137.5;
    let native = equidistant_native(f, 480, 480);
    for theta_max in [0.0f64, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let cam = CameraIntrinsics {
            model: CameraModel::SfmtoolFisheye {
                focal_length: f,
                principal_point_x: 240.0,
                principal_point_y: 240.0,
                bspline_theta_max: theta_max,
                bspline: FLATTENING_BSPLINE.to_vec(),
            },
            width: 480,
            height: 480,
        };
        // Live coefficients that cannot reach the map are not distortion.
        assert!(
            !cam.has_distortion(),
            "θ_max = {theta_max} reported distortion"
        );
        assert_bitwise_equidistant(&cam, &native);
    }
}

#[test]
fn sfmtool_fisheye_round_trips_with_a_live_bspline_past_90_degrees() {
    // Forward/inverse consistency with a non-trivial flattening spline,
    // from the axis out past 90° and beyond θ_max (where the map continues
    // linearly with slope f and the inverse is closed-form).
    let f = 130.0;
    let cam = sfmtool_fisheye_at(f, FLATTENING_BSPLINE.to_vec());
    let mut worst_px = 0.0f64;
    let mut worst_rad = 0.0f64;
    let mut past_90 = 0usize;
    let mut past_theta_max = 0usize;
    for ti in 0..27 {
        let theta = (2.0 + 5.0 * ti as f64).to_radians(); // out to 132°
        for phi_deg in [0.0f64, 47.0, 133.0, 271.0] {
            let ray = ray_at(theta, phi_deg.to_radians());
            let (u, v) = cam.ray_to_pixel(ray).unwrap();
            let back = cam.pixel_to_ray(u, v);
            let dot = (0..3).map(|c| back[c] * ray[c]).sum::<f64>();
            worst_rad = worst_rad.max(dot.clamp(-1.0, 1.0).acos());
            let (u2, v2) = cam.ray_to_pixel(back).unwrap();
            worst_px = worst_px.max((u2 - u).hypot(v2 - v));
            if theta > std::f64::consts::FRAC_PI_2 {
                past_90 += 1;
            }
            if theta > BSPLINE_THETA_MAX {
                past_theta_max += 1;
            }
        }
    }
    assert!(past_90 >= 30, "not enough periphery: {past_90}");
    assert!(
        past_theta_max >= 10,
        "held-constant region unexercised: {past_theta_max}"
    );
    eprintln!("[sfmtool-fisheye-rt] worst {worst_rad:.3e} rad / {worst_px:.3e} px");
    assert!(
        worst_px < 1e-9,
        "pixel round-trip {worst_px} exceeds 1e-9 px"
    );
    // The angle floor is acos() conditioning at dot ≈ 1 (√ε ≈ 1.5e-8), not
    // inverse error — the pixel round trip above is the sharp gate.
    assert!(worst_rad < 1e-7, "ray round-trip {worst_rad} rad");
}

#[test]
fn sfmtool_fisheye_jacobian_matches_central_difference_past_90_degrees() {
    // The analytic Jacobian substitutes (θ_d, θ_d') = (θ + δ, 1 + δ') into
    // the radial template; pin it against a central difference over a field
    // running past 90° and across the θ_max seam, at several ray scales
    // (degree-0 homogeneity). Same bar as the k1-family test above.
    let f = 130.0;
    let cam = sfmtool_fisheye_at(f, FLATTENING_BSPLINE.to_vec());
    let h = 1e-6;
    let mut samples = 0usize;
    let mut past_90 = 0usize;
    let mut worst = 0.0f64;
    for ti in 0..18 {
        let theta = (5.0 + 7.0 * ti as f64).to_radians(); // out to 124°
        for phi_deg in [0.0f64, 43.0, 137.0, 250.0, 331.0] {
            let base = ray_at(theta, phi_deg.to_radians());
            for scale in [0.4f64, 1.0, 6.0] {
                let ray = [base[0] * scale, base[1] * scale, base[2] * scale];
                let (uv, jac) = cam
                    .ray_to_pixel_with_jacobian(ray)
                    .expect("analytic Jacobian None on an in-domain spline ray");
                let direct = cam.ray_to_pixel(ray).unwrap();
                assert_relative_eq!(uv.0, direct.0, epsilon = 1e-12);
                assert_relative_eq!(uv.1, direct.1, epsilon = 1e-12);
                if ray[2] > 0.0 {
                    past_90 += 1;
                }
                for c in 0..3 {
                    let mut rp = ray;
                    let mut rm = ray;
                    rp[c] += h;
                    rm[c] -= h;
                    let (up, vp) = cam.ray_to_pixel(rp).unwrap();
                    let (um, vm) = cam.ray_to_pixel(rm).unwrap();
                    let fd_u = (up - um) / (2.0 * h);
                    let fd_v = (vp - vm) / (2.0 * h);
                    for (a, fd) in [(jac[0][c], fd_u), (jac[1][c], fd_v)] {
                        let rel = (a - fd).abs() / (1.0 + a.abs());
                        worst = worst.max(rel);
                        assert!(
                            rel <= 1e-6,
                            "∂/∂r[{c}] at θ={:.0}°: analytic {a} vs central-diff {fd} (rel {rel})",
                            theta.to_degrees(),
                        );
                    }
                    samples += 1;
                }
            }
        }
    }
    assert!(samples > 500, "thin coverage: only {samples} samples");
    assert!(past_90 > 50, "grid did not exercise θ > 90° ({past_90})");
    eprintln!("[sfmtool-fisheye-jac] {samples} samples, worst rel error {worst:.3e}");
}

#[test]
fn sfmtool_fisheye_jacobian_on_axis_is_the_pinhole_limit() {
    // The gauge pins δ(0) = 0 and δ'(0) = 0, so on the axis θ_d/ρ → 1/rz and
    // θ_d' → 1 whatever the coefficients — the same pinhole limit as the
    // k-family. The approach is only LINEAR in θ, though (the gauge does not
    // pin δ''(0), so δ' ~ δ''(0)·θ, versus the k-family's O(k1·θ²)), hence
    // the θ-proportional tolerance on the continuity sweep.
    let f = 130.0;
    let cam = sfmtool_fisheye_at(f, FLATTENING_BSPLINE.to_vec());
    let (_, jac) = cam.ray_to_pixel_with_jacobian([0.0, 0.0, -2.0]).unwrap();
    assert_relative_eq!(jac[0][0], f / 2.0, epsilon = 1e-12);
    assert_relative_eq!(jac[1][1], -f / 2.0, epsilon = 1e-12);
    for eps in [1e-4f64, 1e-6, 1e-9] {
        let tol = (f * eps).max(1e-10);
        for phi_deg in [0.0f64, 61.0, 233.0] {
            let phi = phi_deg.to_radians();
            let ray = [2.0 * eps * phi.cos(), 2.0 * eps * phi.sin(), -2.0];
            let (_, j) = cam.ray_to_pixel_with_jacobian(ray).unwrap();
            assert_relative_eq!(j[0][0], f / 2.0, epsilon = tol);
            assert_relative_eq!(j[1][1], -f / 2.0, epsilon = tol);
        }
    }
    // The antipode stays the one direction with no derivative.
    assert!(cam.ray_to_pixel_with_jacobian([0.0, 0.0, 1.0]).is_none());
}

#[test]
fn sfmtool_fisheye_folded_bspline_projects_none_past_the_fold() {
    // A spline violating the monotonicity invariant hard enough to drive
    // θ_d non-positive: the forward map is gated (None past the fold, like
    // the polynomial family's θ_d ≤ 0 gate), the Jacobian shares that
    // domain, and the monotonicity check reports the violation.
    let coeffs = vec![-0.05, -0.2, -0.8, -2.0, -3.5, -4.5, -5.0, -5.0];
    assert!(!bspline::bspline_is_monotone(
        &coeffs,
        BSPLINE_THETA_MAX,
        BSPLINE_THETA_MAX
    ));
    let cam = sfmtool_fisheye_at(130.0, coeffs);
    let mut folded = 0usize;
    let mut fine = 0usize;
    for deg in [5.0f64, 15.0, 30.0, 60.0, 90.0, 105.0, 114.0] {
        let ray = ray_at(deg.to_radians(), 0.6);
        match cam.ray_to_pixel(ray) {
            Some(_) => {
                assert!(
                    cam.ray_to_pixel_with_jacobian(ray).is_some(),
                    "no Jacobian at an in-domain θ={deg}°"
                );
                fine += 1;
            }
            None => {
                assert!(
                    cam.ray_to_pixel_with_jacobian(ray).is_none(),
                    "Jacobian past the θ_d fold at θ={deg}°"
                );
                folded += 1;
            }
        }
    }
    assert!(
        fine >= 2 && folded >= 2,
        "{fine} in-domain, {folded} folded"
    );
    // The gently flattening spline stays inside the invariant.
    assert!(bspline::bspline_is_monotone(
        &FLATTENING_BSPLINE,
        BSPLINE_THETA_MAX,
        BSPLINE_THETA_MAX
    ));
}

// -----------------------------------------------------------------------
// The radial spline itself (`distortion::bspline`): gauge anchoring,
// derivative correctness, the held-constant tail, and partition of unity.
// -----------------------------------------------------------------------

#[test]
fn bspline_delta_is_center_anchored() {
    // δ(0) = 0 and δ'(0) = 0 for ANY coefficients: the two omitted basis
    // functions are the only ones live at the origin.
    for coeffs in [
        FLATTENING_BSPLINE.to_vec(),
        vec![0.7, -0.3],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    ] {
        let (d, dp) = bspline::delta_and_deriv(&coeffs, BSPLINE_THETA_MAX, 0.0);
        assert_eq!(d, 0.0);
        assert_eq!(dp, 0.0);
    }
}

#[test]
fn bspline_equal_coefficients_plateau_once_the_anchored_pair_dies() {
    // With every coefficient equal to c, partition of unity gives δ = c
    // exactly wherever the two anchored (zero) basis functions have no
    // support: θ ≥ 2·h with h the knot spacing.
    let c = 0.37;
    let coeffs = vec![c; 8];
    let m = coeffs.len() + 2;
    let h = BSPLINE_THETA_MAX / (m - 3) as f64;
    for frac in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
        let theta = 2.0 * h + frac * (BSPLINE_THETA_MAX - 2.0 * h);
        let (d, dp) = bspline::delta_and_deriv(&coeffs, BSPLINE_THETA_MAX, theta);
        assert_relative_eq!(d, c, epsilon = 1e-14);
        assert_relative_eq!(dp, 0.0, epsilon = 1e-13);
    }
    // Below 2h the anchored pair still bites and δ < c.
    assert!(bspline::delta(&coeffs, BSPLINE_THETA_MAX, h) < c);
}

#[test]
fn bspline_derivative_matches_central_difference() {
    let coeffs = FLATTENING_BSPLINE.to_vec();
    let h = 1e-7;
    for i in 0..=100 {
        let theta = BSPLINE_THETA_MAX * i as f64 / 100.0;
        if theta < h || theta > BSPLINE_THETA_MAX - h {
            continue; // clamping would bias the difference at the ends
        }
        let (_, dp) = bspline::delta_and_deriv(&coeffs, BSPLINE_THETA_MAX, theta);
        let fd = (bspline::delta(&coeffs, BSPLINE_THETA_MAX, theta + h)
            - bspline::delta(&coeffs, BSPLINE_THETA_MAX, theta - h))
            / (2.0 * h);
        assert_relative_eq!(dp, fd, epsilon = 1e-6);
    }
}

#[test]
fn bspline_is_held_constant_beyond_theta_max() {
    let coeffs = FLATTENING_BSPLINE.to_vec();
    let (end, _) = bspline::delta_and_deriv(&coeffs, BSPLINE_THETA_MAX, BSPLINE_THETA_MAX);
    for theta in [
        BSPLINE_THETA_MAX + 1e-9,
        BSPLINE_THETA_MAX + 0.5,
        std::f64::consts::PI,
    ] {
        let (d, dp) = bspline::delta_and_deriv(&coeffs, BSPLINE_THETA_MAX, theta);
        assert_eq!(d.to_bits(), end.to_bits());
        assert_eq!(dp, 0.0);
    }
}

#[test]
fn bspline_basis_is_a_partition_of_unity() {
    // The FULL basis (anchored pair included) sums to 1 with derivative sum 0
    // at every θ — the property that makes the equal-coefficient plateau and
    // the clamped endpoint values exact.
    let n_coeffs = 8;
    for i in 0..=64 {
        let theta = BSPLINE_THETA_MAX * i as f64 / 64.0;
        let (_, values, derivs) = bspline::basis_at(n_coeffs, BSPLINE_THETA_MAX, theta);
        assert_relative_eq!(values.iter().sum::<f64>(), 1.0, epsilon = 1e-13);
        assert_relative_eq!(derivs.iter().sum::<f64>(), 0.0, epsilon = 1e-12);
        assert!(values.iter().all(|&v| v >= 0.0));
    }
}

#[test]
fn bspline_below_minimum_length_is_the_identity() {
    for coeffs in [vec![], vec![0.4]] {
        assert!(bspline::bspline_is_identity(&coeffs));
        assert_eq!(
            bspline::delta_and_deriv(&coeffs, BSPLINE_THETA_MAX, 1.0),
            (0.0, 0.0)
        );
        assert!(bspline::bspline_is_monotone(
            &coeffs,
            BSPLINE_THETA_MAX,
            BSPLINE_THETA_MAX
        ));
    }
    // A live coefficient is not the identity.
    assert!(!bspline::bspline_is_identity(&[0.0, 1e-30]));
}

#[test]
fn bspline_degenerate_theta_max_is_inactive_and_reports_the_identity_map() {
    // `bspline_is_identity` stays a coefficient-only test; `bspline_is_inactive`
    // is the one the kernels ask, and it also fails a domain end that is not
    // positive and finite. `+∞` is the case that used to slip through: it
    // passed a `<= 0 || is_nan` guard, put every knot at infinity, and made
    // the basis recurrence produce NaN.
    let folded = [-0.05, -0.2, -0.8, -2.0, -3.5, -4.5, -5.0, -5.0];
    for theta_max in [0.0f64, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for coeffs in [&FLATTENING_BSPLINE, &folded] {
            assert!(bspline::bspline_is_inactive(coeffs, theta_max));
            assert!(!bspline::bspline_is_identity(coeffs));
            for theta in [0.0f64, 0.5, 1.0, 2.0, 3.0] {
                assert_eq!(
                    bspline::delta_and_deriv(coeffs, theta_max, theta),
                    (0.0, 0.0),
                    "θ_max = {theta_max} at θ = {theta}"
                );
            }
            // The monotonicity report is now GROUNDED in that identity: the
            // map really is `θ_d = θ`. It used to be vacuous for `+∞` — a
            // `true` about a map that was NaN at every angle.
            assert!(bspline::bspline_is_monotone(coeffs, theta_max, 2.0));
        }
    }
    // On a real domain the coefficients are back in charge, and the folded
    // spline is reported for what it is.
    assert!(!bspline::bspline_is_inactive(
        &FLATTENING_BSPLINE,
        BSPLINE_THETA_MAX
    ));
    assert!(bspline::bspline_is_inactive(&[0.0, 0.0], BSPLINE_THETA_MAX));
    assert!(bspline::bspline_is_inactive(&[0.4], BSPLINE_THETA_MAX));
    assert!(!bspline::bspline_is_monotone(
        &folded,
        BSPLINE_THETA_MAX,
        BSPLINE_THETA_MAX
    ));
}

// -----------------------------------------------------------------------
// Local pixel scale and patch sizing (`min_pixel_scale`,
// `pixel_radius_to_world`)
// -----------------------------------------------------------------------

/// A camera-frame point at incidence angle `deg` off the −Z axis and range
/// `range`, at a generic azimuth so no expression is exercised only on an axis.
fn point_at(deg: f64, range: f64) -> [f64; 3] {
    let r = ray_at(deg.to_radians(), 0.37);
    [r[0] * range, r[1] * range, r[2] * range]
}

/// `σ_min(∂(u, v)/∂p_cam)` computed independently of the production path:
/// central-difference `ray_to_pixel`, then the smaller eigenvalue of the 2×2
/// Gram matrix by the difference form (production uses `det/λ_max`).
fn numeric_min_pixel_scale(cam: &CameraIntrinsics, p: [f64; 3]) -> f64 {
    let h = 1e-6 * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let mut j = [[0.0f64; 3]; 2];
    for col in 0..3 {
        let (mut plus, mut minus) = (p, p);
        plus[col] += h;
        minus[col] -= h;
        let (up, vp) = cam.ray_to_pixel(plus).unwrap();
        let (um, vm) = cam.ray_to_pixel(minus).unwrap();
        j[0][col] = (up - um) / (2.0 * h);
        j[1][col] = (vp - vm) / (2.0 * h);
    }
    let a = j[0][0] * j[0][0] + j[0][1] * j[0][1] + j[0][2] * j[0][2];
    let b = j[0][0] * j[1][0] + j[0][1] * j[1][1] + j[0][2] * j[1][2];
    let c = j[1][0] * j[1][0] + j[1][1] * j[1][1] + j[1][2] * j[1][2];
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    (0.5 * (a + c - disc)).max(0.0).sqrt()
}

#[test]
fn pinhole_closed_form_scale_equals_the_numeric_min_singular_value() {
    // σ_min = f/|z| is an identity, not an approximation: the two tangent
    // scales are f·sec²θ/R (radial) and f·secθ/R (azimuthal), and the smaller
    // is f·secθ/R = f/|z| at every θ.
    let cam = simple_pinhole();
    let f = cam.focal_lengths().0;
    for &deg in &[0.0f64, 30.0, 60.0, 75.0, 89.0] {
        for &range in &[0.02f64, 1.0, 9.5, 3100.0] {
            let p = point_at(deg, range);
            let closed = f / p[2].abs();
            assert_relative_eq!(cam.min_pixel_scale(p).unwrap(), closed, max_relative = 1e-9);
            assert_relative_eq!(
                numeric_min_pixel_scale(&cam, p),
                closed,
                max_relative = 1e-9
            );
        }
    }
}

#[test]
fn equidistant_closed_form_scale_equals_the_numeric_min_singular_value() {
    // σ_min = f/R, likewise exact: the tangent scales are f·(θ/sin θ)/R
    // (azimuthal) and f/R (radial), and θ/sin θ ≥ 1 for every θ in (0, π).
    let cam = equidistant_fisheye();
    let f = cam.focal_lengths().0;
    for &deg in &[0.0f64, 30.0, 60.0, 75.0, 89.0, 95.0, 110.0, 130.0] {
        for &range in &[0.02f64, 1.0, 9.5, 3100.0] {
            let p = point_at(deg, range);
            let closed = f / range;
            assert_relative_eq!(cam.min_pixel_scale(p).unwrap(), closed, max_relative = 1e-9);
            assert_relative_eq!(
                numeric_min_pixel_scale(&cam, p),
                closed,
                max_relative = 1e-9
            );
        }
    }
}

#[test]
fn pixel_radius_closed_forms_are_bit_identical_to_the_depth_and_range_forms() {
    // The two fast paths must leave every existing caller untouched, so pin
    // them against the literal pre-change expressions — `|z|` for the pinhole
    // patch sizing, `‖p_cam‖` for the ray-path one — including the 1e-6 floor
    // and the nalgebra norm the patch cloud computed it with.
    use nalgebra::Vector3;
    let pinhole_iso = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };
    let perspective = [simple_pinhole(), pinhole_iso];
    let fisheye = equidistant_fisheye();
    for &deg in &[0.0f64, 17.0, 30.0, 60.0, 75.0, 89.0, 95.0, 130.0, 179.0] {
        for &range in &[0.0f64, 1e-9, 0.05, 1.0, 7.5, 1234.0] {
            let p = point_at(deg, range);
            let v = Vector3::new(p[0], p[1], p[2]);
            for &radius_px in &[1.0f64, 4.0, 12.5] {
                for cam in &perspective {
                    let old = radius_px * v.z.abs().max(1e-6) / cam.focal_lengths().0;
                    assert_eq!(
                        cam.pixel_radius_to_world(p, radius_px).to_bits(),
                        old.to_bits(),
                        "perspective fast path moved at θ={deg}°, R={range}"
                    );
                }
                let old = radius_px * v.norm().max(1e-6) / fisheye.focal_lengths().0;
                assert_eq!(
                    fisheye.pixel_radius_to_world(p, radius_px).to_bits(),
                    old.to_bits(),
                    "ray-path fast path moved at θ={deg}°, R={range}"
                );
            }
        }
    }
}

#[test]
fn anisotropic_pinhole_sizes_by_the_smaller_focal_not_fx() {
    // The fast path is gated on `fx == fy` because σ_min is `min(fx, fy)/|z|`
    // on axis, not `fx/|z|` — an fy < fx camera resolves less vertically, so
    // the same pixel budget buys a LARGER patch than the old fx expression.
    let cam = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 400.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };
    let p = point_at(0.0, 6.0);
    let old = 4.0 * p[2].abs() / 500.0;
    let new = cam.pixel_radius_to_world(p, 4.0);
    assert_relative_eq!(new, 4.0 * p[2].abs() / 400.0, max_relative = 1e-12);
    assert!(
        new > old,
        "expected the smaller focal to win: {new} vs {old}"
    );
}

#[test]
fn simple_radial_sizes_through_the_local_distortion_scale() {
    // `k1 > 0` magnifies off axis, so the local pixel scale exceeds the
    // undistorted pinhole `f/|z|` and the patch is correspondingly SMALLER
    // than the old `|z|/f` expression gave. The exact tangent scales for a
    // radially symmetric model at `r = tan θ` are
    //   radial    f·sec²θ·(1 + 3k₁r²)/R
    //   azimuthal f·sec θ·(1 + k₁r²)/R
    // and σ_min is whichever is smaller.
    let cam = simple_radial();
    let f = cam.focal_lengths().0;
    let (range, radius_px, k1) = (6.0f64, 4.0, 0.1);
    for &deg in &[15.0f64, 35.0, 50.0] {
        let theta = deg.to_radians();
        let p = point_at(deg, range);
        let r2 = theta.tan() * theta.tan();
        let sec = 1.0 / theta.cos();
        let expect = (f / range) * (sec * sec * (1.0 + 3.0 * k1 * r2)).min(sec * (1.0 + k1 * r2));
        let new = cam.pixel_radius_to_world(p, radius_px);
        assert_relative_eq!(new, radius_px / expect, max_relative = 1e-9);
        assert_relative_eq!(
            new,
            radius_px / numeric_min_pixel_scale(&cam, p),
            max_relative = 1e-8
        );
        let old = radius_px * p[2].abs() / f;
        assert!(
            new < old,
            "expected a smaller patch at θ={deg}°: {new} vs {old}"
        );
    }
    // On axis the distortion is inert, so the rule reproduces the old value.
    let axis = point_at(0.0, range);
    assert_relative_eq!(
        cam.pixel_radius_to_world(axis, radius_px),
        radius_px * axis[2].abs() / f,
        max_relative = 1e-12
    );
}

#[test]
fn simple_radial_fisheye_sizes_through_dr_dtheta_not_f() {
    // The polynomial fisheye family maps θ to `r_d = θ·(1 + k₁θ²)`, so its
    // radial pixel scale is `f·(1 + 3k₁θ²)/R`, not `f/R`; the azimuthal one is
    // `f·θ(1 + k₁θ²)/(R·sin θ)`. With `k₁ > 0` both exceed `f/R`, so the patch
    // is smaller than the plain range expression gave.
    let cam = simple_radial_fisheye();
    let f = cam.focal_lengths().0;
    let (range, radius_px, k1) = (6.0f64, 4.0, 0.05);
    for &deg in &[20.0f64, 50.0, 85.0] {
        let theta = deg.to_radians();
        let p = point_at(deg, range);
        let radial = 1.0 + 3.0 * k1 * theta * theta;
        let azimuthal = theta * (1.0 + k1 * theta * theta) / theta.sin();
        let expect = (f / range) * radial.min(azimuthal);
        let new = cam.pixel_radius_to_world(p, radius_px);
        assert_relative_eq!(new, radius_px / expect, max_relative = 1e-6);
        let old = radius_px * range / f;
        assert!(
            new < old,
            "expected a smaller patch at θ={deg}°: {new} vs {old}"
        );
    }
}

#[test]
fn min_pixel_scale_is_defined_for_every_model_on_a_visible_ray() {
    // The rule must not have holes: every model this crate supports has a
    // Jacobian (analytic or differenced) on a ray it can actually image, and
    // the numeric reading agrees with it.
    for cam in all_cameras() {
        let p = point_at(20.0, 5.0);
        let scale = cam
            .min_pixel_scale(p)
            .unwrap_or_else(|| panic!("no pixel scale for {}", cam.model_name()));
        assert!(
            scale.is_finite() && scale > 0.0,
            "{}: σ_min = {scale}",
            cam.model_name()
        );
        assert_relative_eq!(scale, numeric_min_pixel_scale(&cam, p), max_relative = 1e-6);
    }
}

#[test]
fn equidistant_angular_radius_is_bit_identical_to_radius_over_f() {
    // The one model for which the naive angular reading is exact: the map is
    // angle-linear, so `‖ray‖·σ_min = f` at every θ and the angle is
    // `radius_px/f` outright. Pinned bit-for-bit — this is what every infinity
    // patch used to get, regardless of model.
    let cam = equidistant_fisheye();
    let f = cam.focal_lengths().0;
    for &deg in &[0.0f64, 17.0, 30.0, 60.0, 75.0, 89.0, 95.0, 130.0, 179.0] {
        for &range in &[0.02f64, 1.0, 9.5, 3100.0] {
            for &radius_px in &[1.0f64, 4.0, 12.5] {
                assert_eq!(
                    cam.pixel_radius_to_angle(point_at(deg, range), radius_px)
                        .to_bits(),
                    (radius_px / f).to_bits(),
                    "equidistant angular radius moved at θ={deg}°"
                );
            }
        }
    }
}

#[test]
fn pinhole_angular_radius_falls_off_as_cos_theta() {
    // `‖ray‖·σ_min = f·secθ` for a pinhole, so a pixel budget buys `cosθ` LESS
    // angle off axis. `radius_px/f` is the on-axis value only — using it at
    // every θ oversizes a peripheral infinity patch by `1/cosθ` (2× at 60°).
    let cams = [
        simple_pinhole(),
        CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: 500.0,
                focal_length_y: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
            },
            width: 640,
            height: 480,
        },
    ];
    for cam in &cams {
        let f = cam.focal_lengths().0;
        for &deg in &[0.0f64, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0] {
            let theta = deg.to_radians();
            for &range in &[0.02f64, 1.0, 9.5, 3100.0] {
                let p = point_at(deg, range);
                let got = cam.pixel_radius_to_angle(p, 6.0);
                // Closed form, built independently of the implementation.
                assert_relative_eq!(got, 6.0 * theta.cos() / f, max_relative = 1e-12);
                // And the general rule it is a closed form OF: `radius_px`
                // over the range-free pixels-per-radian `R·σ_min`.
                let numeric = range * numeric_min_pixel_scale(cam, p);
                assert_relative_eq!(got, 6.0 / numeric, max_relative = 1e-9);
            }
        }
        // On axis the old `radius_px/f` reading is recovered exactly.
        assert_relative_eq!(
            cam.pixel_radius_to_angle(point_at(0.0, 5.0), 6.0),
            6.0 / f,
            max_relative = 1e-15
        );
    }
}

#[test]
fn angular_radius_is_range_free_and_defined_for_every_model() {
    // `σ_min ∝ 1/R`, so `R·σ_min` — and therefore the angle — depends only on
    // the DIRECTION. Every model must produce it on a ray it can image.
    for cam in all_cameras() {
        let mut prev: Option<f64> = None;
        for &range in &[0.05f64, 1.0, 250.0] {
            let a = cam.pixel_radius_to_angle(point_at(25.0, range), 6.0);
            assert!(
                a.is_finite() && a > 0.0,
                "{}: angular radius {a}",
                cam.model_name()
            );
            if let Some(p) = prev {
                assert_relative_eq!(a, p, max_relative = 1e-6);
            }
            prev = Some(a);
        }
    }
}

#[test]
fn an_off_axis_pinhole_infinity_patch_subtends_the_requested_pixel_radius() {
    // The end-to-end statement: size a tangent-plane patch at an off-axis
    // bearing by the angular rule, project its corners, and the pixel footprint
    // around the keypoint is `radius_px` in the least-magnified direction. The
    // old `radius_px/f` angle overshoots by `1/cosθ`.
    let cam = simple_pinhole();
    let radius_px = 6.0;
    for &deg in &[30.0f64, 60.0] {
        let d = ray_at(deg.to_radians(), 0.37);
        let angle = cam.pixel_radius_to_angle(d, radius_px);
        // Orthonormal tangent basis at the bearing.
        let a = if d[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let dot = a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
        let mut u = [a[0] - dot * d[0], a[1] - dot * d[1], a[2] - dot * d[2]];
        let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        u = [u[0] / un, u[1] / un, u[2] / un];
        let v = [
            d[1] * u[2] - d[2] * u[1],
            d[2] * u[0] - d[0] * u[2],
            d[0] * u[1] - d[1] * u[0],
        ];
        let (cu, cv) = cam.ray_to_pixel(d).unwrap();
        // Walk the tangent circle; the CLOSEST corner is the least-magnified
        // direction, and that is what the rule pins to `radius_px`.
        let mut min_r = f64::INFINITY;
        for k in 0..64 {
            let phi = (k as f64) * std::f64::consts::TAU / 64.0;
            let (cs, sn) = (phi.cos() * angle, phi.sin() * angle);
            let corner = [
                d[0] + cs * u[0] + sn * v[0],
                d[1] + cs * u[1] + sn * v[1],
                d[2] + cs * u[2] + sn * v[2],
            ];
            let (x, y) = cam.ray_to_pixel(corner).unwrap();
            min_r = min_r.min(((x - cu).powi(2) + (y - cv).powi(2)).sqrt());
        }
        // Second-order in the tangent step, so a loose relative bar.
        assert_relative_eq!(min_r, radius_px, max_relative = 2e-3);
        // The pre-change angle would have overshot by 1/cos θ.
        let old_angle = radius_px / cam.focal_lengths().0;
        assert_relative_eq!(
            old_angle / angle,
            1.0 / deg.to_radians().cos(),
            max_relative = 1e-12
        );
    }
}
