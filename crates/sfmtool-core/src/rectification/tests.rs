use super::*;
use approx::assert_relative_eq;

fn test_intrinsics() -> Matrix3<f64> {
    Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0)
}

// ---- Rectification safety tests ----

#[test]
fn test_rectification_safe_lateral_motion() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);

    let safe = check_rectification_safe(&k, &r, &t1, &k, &r, &t2, 640, 480, 50);
    assert!(safe, "Lateral motion should be safe for rectification");
}

#[test]
fn test_rectification_unsafe_forward_motion() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.0, 0.0, 1.0);

    let safe = check_rectification_safe(&k, &r, &t1, &k, &r, &t2, 640, 480, 50);
    assert!(
        !safe,
        "Forward motion should be unsafe for rectification (epipole at principal point)"
    );
}

#[test]
fn test_rectification_safe_epipole_far_outside() {
    // Diagonal motion where the epipole projects far outside the image
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    // Mostly lateral, tiny forward component
    let t2 = Vector3::new(10.0, 0.0, 0.01);

    let safe = check_rectification_safe(&k, &r, &t1, &k, &r, &t2, 640, 480, 50);
    assert!(
        safe,
        "Epipole far outside image should be safe for rectification"
    );
}

// ---- Stereo rectification tests ----

#[test]
fn test_stereo_rectification_identity_rotation() {
    // No rotation between cameras, pure lateral translation
    let k = test_intrinsics();
    let r_rel = Matrix3::identity();
    let t_rel = Vector3::new(1.0, 0.0, 0.0);

    let result = compute_stereo_rectification(&k, &k, &r_rel, &t_rel, 640, 480);

    // With identity rotation, rectification rotations should be close to identity
    // (only need to align baseline with x-axis, which it already is)
    assert_relative_eq!(result.r1_rect, Matrix3::identity(), epsilon = 1e-10);
    assert_relative_eq!(result.r2_rect, Matrix3::identity(), epsilon = 1e-10);
}

#[test]
fn test_stereo_rectification_epipolar_alignment() {
    // Create a pair with some rotation
    let k = test_intrinsics();

    // Small rotation around Y axis (5 degrees)
    let angle = 5.0_f64.to_radians();
    let r_rel = Matrix3::new(
        angle.cos(),
        0.0,
        angle.sin(),
        0.0,
        1.0,
        0.0,
        -angle.sin(),
        0.0,
        angle.cos(),
    );
    let t_rel = Vector3::new(1.0, 0.0, 0.1);

    let result = compute_stereo_rectification(&k, &k, &r_rel, &t_rel, 640, 480);

    // Test key property: for corresponding points, rectified Y coordinates must be equal.
    // Create a 3D point, project into both cameras, then rectify.
    let k_inv = k.try_inverse().expect("K must be invertible");

    // 3D point at (2, 1, 15) in camera 1 frame
    let pt_3d_cam1 = Vector3::new(2.0, 1.0, 15.0);

    // Project into camera 1
    let proj1 = k * pt_3d_cam1;
    let p1_pixel = [proj1[0] / proj1[2], proj1[1] / proj1[2]];

    // Transform to camera 2 and project
    let pt_3d_cam2 = r_rel * pt_3d_cam1 + t_rel;
    let proj2 = k * pt_3d_cam2;
    let p2_pixel = [proj2[0] / proj2[2], proj2[1] / proj2[2]];

    // Rectify both points
    let rect1 = rectify_points(&p1_pixel, 1, &k_inv, &result.r1_rect, &result.p1);
    let rect2 = rectify_points(&p2_pixel, 1, &k_inv, &result.r2_rect, &result.p2);

    // Y coordinates should be equal (epipolar lines are horizontal)
    assert_relative_eq!(rect1[1], rect2[1], epsilon = 1e-6);
}

#[test]
fn test_stereo_rectification_multiple_points_epipolar() {
    let k = test_intrinsics();

    // Rotation around Y axis (10 degrees)
    let angle = 10.0_f64.to_radians();
    let r_rel = Matrix3::new(
        angle.cos(),
        0.0,
        angle.sin(),
        0.0,
        1.0,
        0.0,
        -angle.sin(),
        0.0,
        angle.cos(),
    );
    let t_rel = Vector3::new(2.0, 0.0, 0.3);

    let result = compute_stereo_rectification(&k, &k, &r_rel, &t_rel, 640, 480);
    let k_inv = k.try_inverse().unwrap();

    // Test multiple 3D points
    let points_3d = [
        Vector3::new(0.0, 0.0, 10.0),
        Vector3::new(3.0, -2.0, 20.0),
        Vector3::new(-1.5, 1.0, 8.0),
        Vector3::new(5.0, 3.0, 30.0),
    ];

    for pt in &points_3d {
        let proj1 = k * pt;
        let p1 = [proj1[0] / proj1[2], proj1[1] / proj1[2]];

        let pt_cam2 = r_rel * pt + t_rel;
        let proj2 = k * pt_cam2;
        let p2 = [proj2[0] / proj2[2], proj2[1] / proj2[2]];

        let rect1 = rectify_points(&p1, 1, &k_inv, &result.r1_rect, &result.p1);
        let rect2 = rectify_points(&p2, 1, &k_inv, &result.r2_rect, &result.p2);

        assert_relative_eq!(rect1[1], rect2[1], epsilon = 1e-5,);
    }
}

#[test]
fn test_rectification_safe_45_degree_motion() {
    // 45° motion (equal lateral and forward components) should still be safe
    // because the epipole projects far outside the image
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 1.0); // 45 degrees between lateral and forward

    let safe = check_rectification_safe(&k, &r, &t1, &k, &r, &t2, 640, 480, 50);
    // For t=(1,0,1), the epipole projects to K*(1/1, 0/1, 1) = (820, 240)
    // That's x=820 which is > 640+50=690, so it's outside the image with margin
    assert!(
        safe,
        "45° motion should be safe (epipole at x=820, outside 640+50)"
    );
}

#[test]
fn test_rectification_custom_margins() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    // t = (1, 0, 0.5): epipole at K*(1/0.5, 0, 1) = K*(2, 0, 1) = (500*2+320, 240, 1) = (1320, 240)
    let t2 = Vector3::new(1.0, 0.0, 0.5);

    // Small margin: should be safe (epipole at x=1320, far from 640+10=650)
    assert!(check_rectification_safe(
        &k, &r, &t1, &k, &r, &t2, 640, 480, 10
    ));

    // Large margin: should still be safe (epipole at x=1320, > 640+200=840)
    assert!(check_rectification_safe(
        &k, &r, &t1, &k, &r, &t2, 640, 480, 200
    ));

    // Very large margin: should still be safe (epipole at x=1320, > 640+500=1140)
    assert!(check_rectification_safe(
        &k, &r, &t1, &k, &r, &t2, 640, 480, 500
    ));

    // Forward motion: epipole at principal point (320, 240), always unsafe
    let t_fwd = Vector3::new(0.0, 0.0, 1.0);
    assert!(!check_rectification_safe(
        &k, &r, &t1, &k, &r, &t_fwd, 640, 480, 10
    ));
    assert!(!check_rectification_safe(
        &k, &r, &t1, &k, &r, &t_fwd, 640, 480, 200
    ));
}

#[test]
fn test_rectification_safe_requires_both_epipoles_outside() {
    // Forward motion: both epipoles near the principal point (320, 240).
    // Image 1 is 640x480 → epipole inside → unsafe.
    // Image 2 is 10x10  → epipole at (320, 240) is far outside → safe alone.
    //
    // A correct implementation must check both epipoles and reject this
    // pair. Checking only the image-2 epipole would incorrectly pass.
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.0, 0.0, 1.0);

    let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

    // Epipole in image 2 is outside the tiny image
    assert!(check_rectification_safe_from_f(&f, 10, 10, 50));
    // Epipole in image 1 is inside the normal-sized image
    assert!(!check_rectification_safe_from_f(
        &f.transpose(),
        640,
        480,
        50
    ));

    // Combined check: must be unsafe because image 1's epipole is inside
    let both_safe = check_rectification_safe_from_f(&f, 10, 10, 50)
        && check_rectification_safe_from_f(&f.transpose(), 640, 480, 50);
    assert!(
        !both_safe,
        "Must be unsafe when either epipole is inside its image"
    );
}

// ---- Point rectification tests ----

#[test]
fn test_rectify_points_identity() {
    // With identity K, R, and P=[I|0], points should pass through unchanged
    let k_inv = Matrix3::identity();
    let r_rect = Matrix3::identity();
    let mut p_rect = Matrix3x4::zeros();
    for i in 0..3 {
        p_rect[(i, i)] = 1.0;
    }

    let points = [100.0, 200.0, 300.0, 400.0];
    let result = rectify_points(&points, 2, &k_inv, &r_rect, &p_rect);

    assert_eq!(result.len(), 4);
    assert_relative_eq!(result[0], 100.0, epsilon = 1e-10);
    assert_relative_eq!(result[1], 200.0, epsilon = 1e-10);
    assert_relative_eq!(result[2], 300.0, epsilon = 1e-10);
    assert_relative_eq!(result[3], 400.0, epsilon = 1e-10);
}

#[test]
fn test_rectify_points_with_intrinsics() {
    // Verify that K_inv removes intrinsics and P re-applies them
    let k = test_intrinsics();
    let k_inv = k.try_inverse().unwrap();
    let r_rect = Matrix3::identity();
    let mut p_rect = Matrix3x4::zeros();
    for i in 0..3 {
        for j in 0..3 {
            p_rect[(i, j)] = k[(i, j)];
        }
    }

    // Principal point should map to itself
    let points = [320.0, 240.0];
    let result = rectify_points(&points, 1, &k_inv, &r_rect, &p_rect);
    assert_relative_eq!(result[0], 320.0, epsilon = 1e-10);
    assert_relative_eq!(result[1], 240.0, epsilon = 1e-10);
}

#[test]
fn test_rectify_points_batch() {
    let k = test_intrinsics();
    let k_inv = k.try_inverse().unwrap();
    let r_rect = Matrix3::identity();
    let mut p_rect = Matrix3x4::zeros();
    for i in 0..3 {
        for j in 0..3 {
            p_rect[(i, j)] = k[(i, j)];
        }
    }

    let points = [320.0, 240.0, 400.0, 300.0, 100.0, 50.0];
    let result = rectify_points(&points, 3, &k_inv, &r_rect, &p_rect);

    assert_eq!(result.len(), 6);
    // With identity R_rect and P = [K|0], should get back original points
    assert_relative_eq!(result[0], 320.0, epsilon = 1e-10);
    assert_relative_eq!(result[1], 240.0, epsilon = 1e-10);
    assert_relative_eq!(result[2], 400.0, epsilon = 1e-10);
    assert_relative_eq!(result[3], 300.0, epsilon = 1e-10);
    assert_relative_eq!(result[4], 100.0, epsilon = 1e-10);
    assert_relative_eq!(result[5], 50.0, epsilon = 1e-10);
}

#[test]
fn test_rectify_points_manual_computation() {
    // Manually verify the computation for a single point
    let k = Matrix3::new(400.0, 0.0, 200.0, 0.0, 400.0, 150.0, 0.0, 0.0, 1.0);
    let k_inv = k.try_inverse().unwrap();

    // 90 degree rotation around Z
    let r_rect = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let mut p_rect = Matrix3x4::zeros();
    for i in 0..3 {
        for j in 0..3 {
            p_rect[(i, j)] = k[(i, j)];
        }
    }

    // Point at principal point (200, 150)
    // K_inv * [200, 150, 1]^T = [0, 0, 1]^T
    // R_rect * [0, 0, 1]^T = [0, 0, 1]^T
    // P * [0, 0, 1, 1]^T = K * [0, 0, 1]^T + K * [0,0,0]^T = [200, 150, 1]^T
    let points = [200.0, 150.0];
    let result = rectify_points(&points, 1, &k_inv, &r_rect, &p_rect);
    assert_relative_eq!(result[0], 200.0, epsilon = 1e-10);
    assert_relative_eq!(result[1], 150.0, epsilon = 1e-10);

    // Off-center point: (600, 150)
    // K_inv * [600, 150, 1]^T = [1, 0, 1]^T
    // R_rect * [1, 0, 1]^T = [0, 1, 1]^T
    // P * [0, 1, 1, 1]^T = K * [0, 1, 1]^T = [200, 550, 1]^T
    let points2 = [600.0, 150.0];
    let result2 = rectify_points(&points2, 1, &k_inv, &r_rect, &p_rect);
    assert_relative_eq!(result2[0], 200.0, epsilon = 1e-10);
    assert_relative_eq!(result2[1], 550.0, epsilon = 1e-10);
}

#[test]
fn test_rectify_points_empty() {
    let k_inv = Matrix3::identity();
    let r_rect = Matrix3::identity();
    let p_rect = Matrix3x4::zeros();

    let result = rectify_points(&[], 0, &k_inv, &r_rect, &p_rect);
    assert!(result.is_empty());
}
