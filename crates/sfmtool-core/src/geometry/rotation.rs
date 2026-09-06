// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rotation primitives: skew-symmetric matrices, axis-angle conversions, the
//! rotation angle, and the two projections of a near-rotation onto SO(3).

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
    let angle = rotation_angle(r);

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

/// Rotation angle of `r` in radians.
pub(crate) fn rotation_angle(r: &Matrix3<f64>) -> f64 {
    (((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)).acos()
}

/// Nearest rotation to `m` by polar decomposition (`U Vᵀ` from the SVD).
///
/// The conjugate-homography and pose-verification callers recover `R` only up
/// to scale *including sign*, so `M ≈ −R` must come back as `R`, not as the
/// (distant) proper projection of `−R`. `None` for a non-finite or degenerate
/// input.
pub(crate) fn polar_rotation(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let p = u * v_t;
    if !p.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(if p.determinant() < 0.0 { -p } else { p })
}

/// Nearest rotation to `m` by polar decomposition, **preserving orientation**.
///
/// The sibling of [`polar_rotation`], and the choice between them is the sign
/// convention, not the algorithm — both take `U Vᵀ` from the SVD and differ
/// only in what they do when that product reflects:
///
/// - [`polar_rotation`] negates the whole product, because its callers recover
///   `R` only up to scale *including sign*, so `M ≈ −R` must come back as `R`.
/// - this one folds the sign into the last singular direction
///   (`U · diag(1, 1, det) · Vᵀ`), which is the *proper* projection of `m`
///   itself. Its caller has a matrix that is already the right rotation to
///   within accumulated rounding — a Kabsch fit — and wants the nearest
///   rotation to that, not to its negation.
///
/// Returns `m` unchanged when the SVD does not produce both factors.
pub(crate) fn orthonormalized(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => {
            let d = (u * v_t).determinant().signum();
            u * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d) * v_t
        }
        _ => *m,
    }
}

#[cfg(test)]
mod tests;
