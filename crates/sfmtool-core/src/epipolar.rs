// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Epipolar geometry primitives.
//!
//! Shared utilities for computing epipoles from fundamental matrices,
//! used by both stereo rectification and polar sweep matching.

use nalgebra::{Matrix3, Vector3};

use crate::rotation::skew_symmetric;

/// Compute the fundamental matrix F from two camera poses and intrinsics.
///
/// `F = K2^{-T} [t_rel]_x R_rel K1^{-1}`
///
/// where `R_rel = R2 @ R1^T` and `t_rel = t2 - R_rel @ t1`.
///
/// The fundamental matrix relates corresponding points in pixel coordinates:
/// `p2^T F p1 = 0`.
///
/// # Parameters
///
/// * `k1`, `k2` - 3x3 intrinsic matrices.
/// * `r1`, `r2` - 3x3 cam_from_world rotation matrices.
/// * `t1`, `t2` - cam_from_world translation vectors.
#[allow(clippy::too_many_arguments)]
pub fn compute_fundamental_matrix(
    k1: &Matrix3<f64>,
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    k2: &Matrix3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
) -> Matrix3<f64> {
    let r_rel = r2 * r1.transpose();
    let t_rel = t2 - r_rel * t1;
    let t_skew = skew_symmetric(&t_rel);
    let e = t_skew * r_rel;

    let k2_inv = k2
        .try_inverse()
        .expect("Intrinsic matrix K2 must be invertible");
    let k1_inv = k1
        .try_inverse()
        .expect("Intrinsic matrix K1 must be invertible");

    k2_inv.transpose() * e * k1_inv
}

/// Compute a single epipole from a fundamental matrix via SVD null space.
///
/// Finds the null space of `F^T` (the right epipole, i.e. the projection
/// of camera 1's center into image 2). To get the left epipole (null space
/// of `F`), pass `&f.transpose()`.
///
/// Returns `(epipole_xy, is_at_infinity)` where `is_at_infinity` is true
/// when `|w| < 1e-10`. The `epipole_xy` is the dehomogenized `[x/w, y/w]`
/// coordinates (undefined when at infinity, returns `[0.0, 0.0]`).
pub fn compute_epipole(f: &Matrix3<f64>) -> ([f64; 2], bool) {
    let ft = f.transpose();
    let svd = ft.svd(true, true);
    let v_t = svd.v_t.expect("SVD failed to compute V^T");

    // Null space is the last row of V^T (smallest singular value)
    let w = v_t[(2, 2)];
    let is_at_infinity = w.abs() < 1e-10;

    if is_at_infinity {
        ([0.0, 0.0], true)
    } else {
        ([v_t[(2, 0)] / w, v_t[(2, 1)] / w], false)
    }
}

/// Compute both epipoles from a 3x3 fundamental matrix.
///
/// The epipoles are the null spaces of `F` and `F^T` respectively:
/// - `e1`: null space of `F` (left epipole, projection of camera 2's center into image 1)
/// - `e2`: null space of `F^T` (right epipole, projection of camera 1's center into image 2)
///
/// Returns `Some((e1, e2))` where each is `[x, y]` in pixel coordinates, or
/// `None` if either epipole is at infinity (homogeneous w ≈ 0).
pub fn compute_epipole_pair(f: &Matrix3<f64>) -> Option<([f64; 2], [f64; 2])> {
    let (e2, e2_inf) = compute_epipole(f);
    if e2_inf {
        return None;
    }

    let (e1, e1_inf) = compute_epipole(&f.transpose());
    if e1_inf {
        return None;
    }

    Some((e1, e2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fundamental_matrix_identity_cameras() {
        // Two cameras at same pose: R_rel = I, t_rel = 0
        // E = [0]_x * I = 0, so F = 0
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t = Vector3::zeros();

        let f = compute_fundamental_matrix(&k, &r, &t, &k, &r, &t);
        assert_relative_eq!(f, Matrix3::zeros(), epsilon = 1e-10);
    }

    #[test]
    fn test_fundamental_matrix_lateral_baseline() {
        // Camera 1 at origin, camera 2 translated along X axis
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        // F should not be zero
        assert!(f.norm() > 1e-10);

        // F should be rank 2 (det = 0)
        assert_relative_eq!(f.determinant(), 0.0, epsilon = 1e-10);

        // Epipolar constraint: for a 3D point, its projections satisfy p2^T F p1 = 0
        // Point at (0, 0, 10): projects to (320, 240) in cam1, (370, 240) in cam2
        let p1 = Vector3::new(320.0, 240.0, 1.0);
        let p2 = Vector3::new(370.0, 240.0, 1.0);
        let epipolar_constraint = p2.transpose() * f * p1;
        assert_relative_eq!(epipolar_constraint[(0, 0)], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_fundamental_matrix_vertical_baseline() {
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(0.0, 1.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        assert!(f.norm() > 1e-10);
        assert_relative_eq!(f.determinant(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_epipole_at_infinity() {
        // F = [[0,0,0],[0,0,-1],[0,1,0]]
        // Null space of F is [1,0,0] (at infinity), so pair should return None.
        let f = Matrix3::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
        assert!(compute_epipole_pair(&f).is_none());
    }

    #[test]
    fn test_epipole_from_pure_translation() {
        // Pure translation: P1 = [I|0], P2 = [I|t] with t = (2, 3, 1).
        // F = [t]_x is skew-symmetric, so both null(F) and null(F^T) are t.
        // Both epipoles dehomogenize to t/t_z = (2, 3).
        let f = Matrix3::from_row_slice(&[0.0, -1.0, 3.0, 1.0, 0.0, -2.0, -3.0, 2.0, 0.0]);

        let (e1, e2) = compute_epipole_pair(&f).expect("both epipoles should be finite");
        assert!((e1[0] - 2.0).abs() < 1e-6, "e1.x = {}", e1[0]);
        assert!((e1[1] - 3.0).abs() < 1e-6, "e1.y = {}", e1[1]);
        assert!((e2[0] - 2.0).abs() < 1e-6, "e2.x = {}", e2[0]);
        assert!((e2[1] - 3.0).abs() < 1e-6, "e2.y = {}", e2[1]);
    }

    #[test]
    fn test_epipole_from_diagonal_translation() {
        // Pure translation with t = (5, -4, 2).
        // Both epipoles = t/t_z = (2.5, -2).
        let f = Matrix3::from_row_slice(&[0.0, -2.0, -4.0, 2.0, 0.0, -5.0, 4.0, 5.0, 0.0]);

        let (e1, e2) = compute_epipole_pair(&f).expect("both epipoles should be finite");
        assert!((e1[0] - 2.5).abs() < 1e-6, "e1.x = {}", e1[0]);
        assert!((e1[1] - (-2.0)).abs() < 1e-6, "e1.y = {}", e1[1]);
        assert!((e2[0] - 2.5).abs() < 1e-6, "e2.x = {}", e2[0]);
        assert!((e2[1] - (-2.0)).abs() < 1e-6, "e2.y = {}", e2[1]);
    }

    #[test]
    fn test_single_epipole_lateral_motion() {
        // For pure lateral motion, epipole is at infinity.
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        let (_epipole, is_at_infinity) = compute_epipole(&f);
        assert!(is_at_infinity);
    }

    #[test]
    fn test_fundamental_equals_essential_when_k_identity() {
        // When K = I, F = E = [t]_x R_rel
        let k = Matrix3::identity();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        // E = [t_rel]_x * R_rel
        //   = [t2]_x * I
        //   = skew(t2)
        let e = skew_symmetric(&t2);

        // F should equal E when K = I
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (f[(i, j)] - e[(i, j)]).abs() < 1e-10,
                    "F[{i},{j}] = {} != E[{i},{j}] = {}",
                    f[(i, j)],
                    e[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_essential_matrix_epipolar_constraint() {
        // Verify p2^T E p1 = 0 for a known 3D point
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r1 = Matrix3::identity();
        let t1 = Vector3::zeros();
        let r2 = Matrix3::identity();
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r1, &t1, &k, &r2, &t2);

        // 3D point at (2, 3, 10):
        // cam1: K * (2, 3, 10) / 10 = K * (0.2, 0.3, 1) = (420, 390, 1) (homogeneous)
        // cam2: K * (2+1, 3, 10) / 10 = K * (0.3, 0.3, 1) = (470, 390, 1)
        let p1 = Vector3::new(500.0 * 0.2 + 320.0, 500.0 * 0.3 + 240.0, 1.0);
        let p2 = Vector3::new(500.0 * 0.3 + 320.0, 500.0 * 0.3 + 240.0, 1.0);

        let constraint = p2.transpose() * f * p1;
        assert!(
            constraint[(0, 0)].abs() < 1e-6,
            "Epipolar constraint p2^T F p1 should be ≈ 0, got {}",
            constraint[(0, 0)]
        );
    }

    #[test]
    fn test_epipole_from_90_degree_rotation() {
        // Camera 2 rotated 90° around Y from camera 1
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r1 = Matrix3::identity();
        let t1 = Vector3::zeros();

        let angle = std::f64::consts::FRAC_PI_2;
        let r2 = Matrix3::new(
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
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r1, &t1, &k, &r2, &t2);

        // F should be rank 2
        assert!(f.norm() > 1e-10);
        assert!(f.determinant().abs() < 1e-6, "F should be rank 2");
    }

    #[test]
    fn test_single_epipole_forward_motion() {
        // For forward motion along Z, epipole is at principal point.
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(0.0, 0.0, 1.0);
        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        let (epipole, is_at_infinity) = compute_epipole(&f);
        assert!(!is_at_infinity);
        assert!((epipole[0] - 320.0).abs() < 1.0);
        assert!((epipole[1] - 240.0).abs() < 1.0);
    }
}
