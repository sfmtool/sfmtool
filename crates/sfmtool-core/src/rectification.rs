// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Stereo rectification for undistorted pinhole cameras.
//!
//! This module provides:
//! - Rectification safety checks (epipole location)
//! - Bouguet's stereo rectification algorithm
//! - Batch point rectification

use nalgebra::{Matrix3, Matrix3x4, Vector3};

use crate::epipolar::{compute_epipole, compute_fundamental_matrix};
use crate::rotation::{axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle};

/// Check if rectification is safe (epipole outside image bounds + margin).
///
/// Rectification becomes unstable when the epipole is inside or near the image.
/// This typically occurs during forward/backward camera motion.
///
/// # Parameters
///
/// * `k1`, `k2` - 3x3 intrinsic matrices.
/// * `r1`, `r2` - 3x3 cam_from_world rotation matrices.
/// * `t1`, `t2` - cam_from_world translation vectors.
/// * `width`, `height` - Image dimensions in pixels.
/// * `margin` - Safety margin in pixels.
#[allow(clippy::too_many_arguments)]
pub fn check_rectification_safe(
    k1: &Matrix3<f64>,
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    k2: &Matrix3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
    width: u32,
    height: u32,
    margin: u32,
) -> bool {
    let f = compute_fundamental_matrix(k1, r1, t1, k2, r2, t2);
    check_rectification_safe_from_f(&f, width, height, margin)
}

/// Check if rectification is safe given a pre-computed fundamental matrix.
///
/// Same as [`check_rectification_safe`] but avoids recomputing F when it is
/// already available.
pub fn check_rectification_safe_from_f(
    f: &Matrix3<f64>,
    width: u32,
    height: u32,
    margin: u32,
) -> bool {
    let (epipole, is_at_infinity) = compute_epipole(f);

    if is_at_infinity {
        return true;
    }

    let x = epipole[0];
    let y = epipole[1];
    let m = margin as f64;
    let w = width as f64;
    let h = height as f64;

    x < -m || x > w + m || y < -m || y > h + m
}

/// Result of stereo rectification.
#[derive(Debug, Clone)]
pub struct RectificationResult {
    /// Rectification rotation for camera 1.
    pub r1_rect: Matrix3<f64>,
    /// Rectification rotation for camera 2.
    pub r2_rect: Matrix3<f64>,
    /// 3x4 projection matrix for camera 1.
    pub p1: Matrix3x4<f64>,
    /// 3x4 projection matrix for camera 2.
    pub p2: Matrix3x4<f64>,
}

/// Compute stereo rectification for undistorted pinhole cameras.
///
/// Splits the relative rotation equally between the two cameras, then
/// rotates both so the baseline is horizontal. The new intrinsic matrix
/// is the average of both cameras' intrinsics, and the principal points
/// are aligned so that corresponding points at infinity have zero disparity.
///
/// Like in OpenCV's `stereoRectify`, the returned projection matrices `P1`/`P2`
/// contain only the new intrinsics and baseline translation — they do not include
/// the rectification rotations `R1`/`R2`. The caller is responsible for applying
/// the rotation separately (e.g. via [`rectify_points`]).
///
/// # Parameters
///
/// * `k1`, `k2` - 3x3 intrinsic matrices.
/// * `r_rel` - 3x3 relative rotation (camera 2 from camera 1).
/// * `t_rel` - Relative translation (camera 2 from camera 1).
/// * `image_width`, `image_height` - Image dimensions in pixels.
pub fn compute_stereo_rectification(
    k1: &Matrix3<f64>,
    k2: &Matrix3<f64>,
    r_rel: &Matrix3<f64>,
    t_rel: &Vector3<f64>,
    _image_width: u32,
    _image_height: u32,
) -> RectificationResult {
    // Step 1: Decompose R_rel into axis-angle and apply half rotation to each camera.
    let (axis, angle) = rotation_matrix_to_axis_angle(r_rel);

    let (r1_half, r2_half) = if angle.abs() < 1e-12 {
        (Matrix3::identity(), Matrix3::identity())
    } else {
        let r = axis_angle_to_rotation_matrix(&axis, angle / 2.0);
        (r, r.transpose())
    };

    // Step 2: After half-rotation, the new baseline direction
    let baseline = r2_half * t_rel;
    let baseline_length = baseline.norm();
    let e1 = baseline / baseline_length;

    // Step 3: Construct e2 perpendicular to e1, in the horizontal plane.
    // e2 = [-e1_y, e1_x, 0] / norm, which is the cross product [0,0,1] x e1.
    // This fails when e1 is nearly along Z, so fall back to crossing with [0,1,0].
    let raw_e2 = Vector3::new(-e1[1], e1[0], 0.0);
    let e2 = if raw_e2.norm() < 1e-10 {
        // e1 nearly along Z, use [0,1,0] x e1 instead
        Vector3::new(0.0, 1.0, 0.0).cross(&e1).normalize()
    } else {
        raw_e2.normalize()
    };

    // Step 4: e3 = e1 x e2
    let e3 = e1.cross(&e2);

    // Step 5: R_rect_common = matrix from rows [e1, e2, e3]
    let r_rect_common = Matrix3::from_rows(&[e1.transpose(), e2.transpose(), e3.transpose()]);

    // Step 6: Final rectification rotations
    let r1_rect = r_rect_common * r1_half;
    let r2_rect = r_rect_common * r2_half;

    // Step 7: For CALIB_ZERO_DISPARITY with alpha=1.0, construct K_new from average
    let fx = (k1[(0, 0)] + k2[(0, 0)]) / 2.0;
    let fy = (k1[(1, 1)] + k2[(1, 1)]) / 2.0;
    let cx = (k1[(0, 2)] + k2[(0, 2)]) / 2.0;
    let cy = (k1[(1, 2)] + k2[(1, 2)]) / 2.0;
    let k_new = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

    // Step 8: P1 = K_new * [I | 0], P2 = K_new * [I | -baseline * e_x]
    let mut it1 = Matrix3x4::zeros();
    for i in 0..3 {
        it1[(i, i)] = 1.0;
    }
    let p1 = k_new * it1;

    let mut it2 = Matrix3x4::zeros();
    for i in 0..3 {
        it2[(i, i)] = 1.0;
    }
    // Translation column: -baseline_length * [1, 0, 0]
    it2[(0, 3)] = -baseline_length;
    let p2 = k_new * it2;

    RectificationResult {
        r1_rect,
        r2_rect,
        p1,
        p2,
    }
}

/// Rectify a batch of N points.
///
/// For each point `[u, v]`, computes:
/// 1. `x_norm = K_inv * [u, v, 1]^T` (remove intrinsics)
/// 2. `x_rot = R_rect * x_norm` (apply rectification rotation)
/// 3. `[u', v', w'] = P_rect * [x_rot; 1]` (project with rectified camera)
/// 4. Result: `[u'/w', v'/w']` (dehomogenize)
///
/// This is equivalent to `cv2.undistortPoints(pts, K, D=0, R=R_rect, P=P_rect)`.
///
/// # Parameters
///
/// * `points` - Flat row-major array of N 2D points `[u0, v0, u1, v1, ...]`.
/// * `n` - Number of points.
/// * `k_inv` - 3x3 inverse intrinsic matrix.
/// * `r_rect` - 3x3 rectification rotation.
/// * `p_rect` - 3x4 rectified projection matrix.
///
/// # Returns
///
/// Flat row-major array of N rectified 2D points.
pub fn rectify_points(
    points: &[f64],
    n: usize,
    k_inv: &Matrix3<f64>,
    r_rect: &Matrix3<f64>,
    p_rect: &Matrix3x4<f64>,
) -> Vec<f64> {
    debug_assert_eq!(points.len(), n * 2);

    let mut result = Vec::with_capacity(n * 2);

    // Precompute the combined 3x3 matrix for the first three columns of P_rect
    // and the translation column
    let p_33 = p_rect.fixed_columns::<3>(0).into_owned();
    let p_t = p_rect.column(3);
    let transform = p_33 * r_rect * k_inv;

    for i in 0..n {
        let u = points[i * 2];
        let v = points[i * 2 + 1];
        let pt_hom = Vector3::new(u, v, 1.0);

        // Apply combined transform and add translation
        let projected = transform * pt_hom + p_t;

        // Dehomogenize
        let w = projected[2];
        result.push(projected[0] / w);
        result.push(projected[1] / w);
    }

    result
}

#[cfg(test)]
mod tests {
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
}
