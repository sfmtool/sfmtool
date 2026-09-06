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

/// The angle is the one the axis-angle decomposition reports, which is what
/// lets that conversion call this rather than re-derive it.
#[test]
fn test_rotation_angle_agrees_with_axis_angle() {
    for angle in [0.0, 0.3, 1.0, std::f64::consts::FRAC_PI_2, 3.0] {
        let axis = Vector3::new(1.0, -2.0, 0.5).normalize();
        let r = axis_angle_to_rotation_matrix(&axis, angle);
        assert!((rotation_angle(&r) - angle).abs() < 1e-10, "angle {angle}");
        assert!((rotation_matrix_to_axis_angle(&r).1 - rotation_angle(&r)).abs() < 1e-12);
    }
}

/// The whole reason the two projections are separate functions: given `−R`,
/// `polar_rotation` returns `R` (its callers know the sign only up to scale),
/// while `orthonormalized` returns the *proper* projection of `−R` itself,
/// which is a different rotation.
#[test]
fn test_polar_rotation_and_orthonormalized_differ_on_a_reflection() {
    let r = axis_angle_to_rotation_matrix(&Vector3::z(), 0.7);
    let polar = polar_rotation(&(-r)).expect("finite input");
    assert_rotation_matrix(&polar);
    assert!(
        (polar - r).norm() < 1e-12,
        "polar_rotation should undo the sign"
    );

    let proper = orthonormalized(&(-r));
    assert_rotation_matrix(&proper);
    assert!(
        (proper - r).norm() > 1e-3,
        "orthonormalized should project −R, not negate it"
    );
}

/// A near-rotation carrying accumulated rounding — the Kabsch case
/// `orthonormalized` exists for — comes back as an exact rotation close to the
/// input, and `polar_rotation` agrees there because no reflection is involved.
#[test]
fn test_orthonormalized_cleans_a_perturbed_rotation() {
    let r = axis_angle_to_rotation_matrix(&Vector3::new(0.3, 0.4, 0.5).normalize(), 1.1);
    let mut m = r;
    m[(0, 1)] += 1e-6;
    m[(2, 0)] -= 2e-6;

    let cleaned = orthonormalized(&m);
    assert_rotation_matrix(&cleaned);
    assert!((cleaned - r).norm() < 1e-5);
    assert!((cleaned - polar_rotation(&m).expect("finite input")).norm() < 1e-12);
}

/// A non-finite input has no polar decomposition to report.
#[test]
fn test_polar_rotation_rejects_non_finite() {
    let mut m = Matrix3::identity();
    m[(1, 1)] = f64::NAN;
    assert!(polar_rotation(&m).is_none());
}
