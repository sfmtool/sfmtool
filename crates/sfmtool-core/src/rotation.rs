// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rotation primitives: skew-symmetric matrices and axis-angle conversions.

use nalgebra::{Matrix3, Vector3};

/// Build a 3x3 skew-symmetric matrix from a 3D vector.
///
/// The skew-symmetric matrix `[v]_x` satisfies `[v]_x @ u = v x u` (cross product).
pub(crate) fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

/// Decompose a 3x3 rotation matrix into axis-angle representation.
///
/// Returns `(axis, angle)` where `axis` is a unit vector and `angle` is in
/// radians `[0, π]`. For identity (angle ≈ 0), returns `([1, 0, 0], 0.0)`.
pub(crate) fn rotation_matrix_to_axis_angle(r: &Matrix3<f64>) -> (Vector3<f64>, f64) {
    let angle = ((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos();

    if angle.abs() < 1e-12 {
        return (Vector3::x(), 0.0);
    }

    let sin_angle = angle.sin();
    let axis = if sin_angle.abs() > 1e-10 {
        // Standard case: extract axis from skew-symmetric part (R - R^T) / (2 sin θ)
        let ax = Vector3::new(
            r[(2, 1)] - r[(1, 2)],
            r[(0, 2)] - r[(2, 0)],
            r[(1, 0)] - r[(0, 1)],
        ) / (2.0 * sin_angle);
        ax.normalize()
    } else {
        // angle ≈ π: sin θ ≈ 0, extract axis from (R + I) columns
        let r_plus_i = r + Matrix3::identity();
        let mut best_col = 0;
        let mut best_norm = 0.0;
        for col in 0..3 {
            let n = r_plus_i.column(col).norm();
            if n > best_norm {
                best_norm = n;
                best_col = col;
            }
        }
        r_plus_i.column(best_col).normalize().into_owned()
    };

    (axis, angle)
}

/// Build a 3x3 rotation matrix from axis-angle via the Rodrigues formula.
///
/// `R = I + sin(θ) [axis]_× + (1 − cos θ) [axis]_×²`
///
/// `axis` must be a unit vector.
pub(crate) fn axis_angle_to_rotation_matrix(axis: &Vector3<f64>, angle: f64) -> Matrix3<f64> {
    let k = skew_symmetric(axis);
    Matrix3::identity() + angle.sin() * k + (1.0 - angle.cos()) * k * k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: verify R is a valid rotation matrix (R^T R = I, det = 1).
    fn assert_rotation_matrix(r: &Matrix3<f64>) {
        let rtr = r.transpose() * r;
        assert!(
            (rtr - Matrix3::identity()).norm() < 1e-10,
            "R^T R should be identity, got {rtr}"
        );
        assert!(
            (r.determinant() - 1.0).abs() < 1e-10,
            "det(R) should be 1, got {}",
            r.determinant()
        );
    }

    #[test]
    fn test_axis_angle_to_rotation_90_around_z() {
        // 90° around Z: (1,0,0) → (0,1,0)
        let r = axis_angle_to_rotation_matrix(&Vector3::z(), std::f64::consts::FRAC_PI_2);
        assert_rotation_matrix(&r);

        let result = r * Vector3::new(1.0, 0.0, 0.0);
        assert!((result - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_axis_angle_to_rotation_180_around_x() {
        // 180° around X: (0,1,0) → (0,-1,0)
        let r = axis_angle_to_rotation_matrix(&Vector3::x(), std::f64::consts::PI);
        assert_rotation_matrix(&r);

        let result = r * Vector3::new(0.0, 1.0, 0.0);
        assert!((result - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_axis_angle_to_rotation_90_around_y() {
        // 90° around Y: (1,0,0) → (0,0,-1)
        let r = axis_angle_to_rotation_matrix(&Vector3::y(), std::f64::consts::FRAC_PI_2);
        assert_rotation_matrix(&r);

        let result = r * Vector3::new(1.0, 0.0, 0.0);
        assert!((result - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-10);
    }

    #[test]
    fn test_axis_angle_to_rotation_zero_is_identity() {
        let r = axis_angle_to_rotation_matrix(&Vector3::new(1.0, 2.0, 3.0).normalize(), 0.0);
        assert!((r - Matrix3::identity()).norm() < 1e-10);
    }

    #[test]
    fn test_axis_angle_to_rotation_120_around_111() {
        // 120° around (1,1,1)/√3 cycles x→y→z→x
        let axis = Vector3::new(1.0, 1.0, 1.0).normalize();
        let r = axis_angle_to_rotation_matrix(&axis, 2.0 * std::f64::consts::FRAC_PI_3);
        assert_rotation_matrix(&r);

        let result = r * Vector3::new(1.0, 0.0, 0.0);
        assert!(
            (result - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-10,
            "(1,0,0) should map to (0,1,0), got {result}"
        );

        let result2 = r * Vector3::new(0.0, 1.0, 0.0);
        assert!(
            (result2 - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-10,
            "(0,1,0) should map to (0,0,1), got {result2}"
        );
    }

    #[test]
    fn test_rotation_matrix_to_axis_angle_identity() {
        let (_, angle) = rotation_matrix_to_axis_angle(&Matrix3::identity());
        assert!(angle.abs() < 1e-12);
    }

    #[test]
    fn test_rotation_matrix_to_axis_angle_90_around_z() {
        let r = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let (axis, angle) = rotation_matrix_to_axis_angle(&r);

        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        // Axis should be ±Z
        assert!((axis.dot(&Vector3::z())).abs() > 0.999);
    }

    #[test]
    fn test_rotation_matrix_to_axis_angle_180_around_x() {
        // 180° around X: diag(1, -1, -1)
        let r = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
        let (axis, angle) = rotation_matrix_to_axis_angle(&r);

        assert!((angle - std::f64::consts::PI).abs() < 1e-10);
        assert!((axis.dot(&Vector3::x())).abs() > 0.999);
    }

    #[test]
    fn test_axis_angle_round_trip() {
        // Build a rotation from axis-angle, decompose it, should recover the same values.
        let original_axis = Vector3::new(1.0, -2.0, 0.5).normalize();
        let original_angle = 1.23;

        let r = axis_angle_to_rotation_matrix(&original_axis, original_angle);
        let (recovered_axis, recovered_angle) = rotation_matrix_to_axis_angle(&r);

        assert!(
            (recovered_angle - original_angle).abs() < 1e-10,
            "angle: expected {original_angle}, got {recovered_angle}"
        );
        // Axis may be flipped (axis, angle) == (-axis, -angle), but angle ∈ [0,π]
        // so for angle > 0 the axes should agree in sign.
        assert!(
            (recovered_axis - original_axis).norm() < 1e-10,
            "axis: expected {original_axis}, got {recovered_axis}"
        );
    }

    #[test]
    fn test_axis_angle_round_trip_near_pi() {
        // Near-π rotation: the tricky case for decomposition
        let original_axis = Vector3::y();
        let original_angle = 3.0; // close to π

        let r = axis_angle_to_rotation_matrix(&original_axis, original_angle);
        assert_rotation_matrix(&r);
        let (recovered_axis, recovered_angle) = rotation_matrix_to_axis_angle(&r);

        assert!(
            (recovered_angle - original_angle).abs() < 1e-10,
            "angle: expected {original_angle}, got {recovered_angle}"
        );
        assert!(
            (recovered_axis - original_axis).norm() < 1e-10
                || (recovered_axis + original_axis).norm() < 1e-10,
            "axis: expected ±{original_axis}, got {recovered_axis}"
        );
    }
}