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
    // A singular intrinsic matrix yields no fundamental matrix; treat such a
    // degenerate pair as unsafe to rectify rather than panicking.
    match compute_fundamental_matrix(k1, r1, t1, k2, r2, t2) {
        Some(f) => check_rectification_safe_from_f(&f, width, height, margin),
        None => false,
    }
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
mod tests;
