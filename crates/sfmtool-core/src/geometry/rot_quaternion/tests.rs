use super::*;
use approx::assert_relative_eq;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI};

#[test]
fn identity_quaternion() {
    let q = RotQuaternion::identity();
    assert_relative_eq!(q.w(), 1.0, epsilon = 1e-12);
    assert_relative_eq!(q.x(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(q.y(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(q.z(), 0.0, epsilon = 1e-12);

    let mat = q.to_rotation_matrix();
    assert_relative_eq!(mat, Matrix3::identity(), epsilon = 1e-12);
}

#[test]
fn new_normalizes_input() {
    let q = RotQuaternion::new(2.0, 0.0, 0.0, 0.0);
    assert_relative_eq!(q.w(), 1.0, epsilon = 1e-12);
    assert_relative_eq!(q.x(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(q.y(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(q.z(), 0.0, epsilon = 1e-12);

    // Non-trivial normalization
    let q2 = RotQuaternion::new(1.0, 1.0, 1.0, 1.0);
    let norm = (q2.w().powi(2) + q2.x().powi(2) + q2.y().powi(2) + q2.z().powi(2)).sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
}

#[test]
fn from_wxyz_array_matches_new() {
    let q1 = RotQuaternion::new(0.5, 0.5, 0.5, 0.5);
    let q2 = RotQuaternion::from_wxyz_array([0.5, 0.5, 0.5, 0.5]);
    assert_eq!(q1, q2);
}

#[test]
fn axis_angle_90_around_z() {
    let q = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let mat = q.to_rotation_matrix();

    // 90° around Z: (1,0,0) → (0,1,0)
    let result = mat * Vector3::new(1.0, 0.0, 0.0);
    assert_relative_eq!(result, Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-10);
}

#[test]
fn axis_angle_180_around_x() {
    let q = RotQuaternion::from_axis_angle(Vector3::x(), PI).unwrap();
    let mat = q.to_rotation_matrix();

    // 180° around X: (0,1,0) → (0,-1,0)
    let result = mat * Vector3::new(0.0, 1.0, 0.0);
    assert_relative_eq!(result, Vector3::new(0.0, -1.0, 0.0), epsilon = 1e-10);
}

#[test]
fn axis_angle_120_around_111() {
    let axis = Vector3::new(1.0, 1.0, 1.0);
    let q = RotQuaternion::from_axis_angle(axis, 2.0 * FRAC_PI_3).unwrap();
    let mat = q.to_rotation_matrix();

    // 120° around (1,1,1)/√3 cycles x→y→z→x
    let result = mat * Vector3::new(1.0, 0.0, 0.0);
    assert_relative_eq!(result, Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-10);
}

#[test]
fn zero_axis_rejected() {
    let result = RotQuaternion::from_axis_angle(Vector3::zeros(), 1.0);
    assert!(result.is_err());
}

#[test]
fn rotation_matrix_round_trip() {
    let q_orig = RotQuaternion::from_axis_angle(Vector3::new(1.0, -2.0, 0.5), 1.23).unwrap();

    let mat = q_orig.to_rotation_matrix();
    let q_back = RotQuaternion::from_rotation_matrix(mat);

    // Compare rotation matrices since quaternion sign may flip
    let mat_back = q_back.to_rotation_matrix();
    assert_relative_eq!(mat, mat_back, epsilon = 1e-10);
}

#[test]
fn conjugate_gives_identity() {
    let q = RotQuaternion::from_axis_angle(Vector3::new(1.0, 2.0, 3.0), 0.7).unwrap();
    let product = &q * &q.conjugate();
    assert_eq!(product, RotQuaternion::identity());
}

#[test]
fn inverse_gives_identity() {
    let q = RotQuaternion::from_axis_angle(Vector3::new(-1.0, 0.5, 2.0), 1.5).unwrap();
    let product = &q * &q.inverse();
    assert_eq!(product, RotQuaternion::identity());
}

#[test]
fn multiplication_composes_rotations() {
    // Two 90° rotations around Z should give 180° around Z
    let q90 = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let q180 = &q90 * &q90;

    let expected = RotQuaternion::from_axis_angle(Vector3::z(), PI).unwrap();
    let mat_result = q180.to_rotation_matrix();
    let mat_expected = expected.to_rotation_matrix();
    assert_relative_eq!(mat_result, mat_expected, epsilon = 1e-10);
}

#[test]
fn multiplication_order() {
    // Verify q1 * q2 applies q2 first, then q1 (nalgebra convention).
    // q1 = 90° around Z, q2 = 90° around X
    let q1 = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let q2 = RotQuaternion::from_axis_angle(Vector3::x(), FRAC_PI_2).unwrap();

    let composed = &q1 * &q2;
    let mat = composed.to_rotation_matrix();

    // Apply q2 first (90° around X), then q1 (90° around Z) to (1, 0, 0):
    // q2: (1,0,0) → (1,0,0)  (X rotation doesn't affect X axis)
    // q1: (1,0,0) → (0,1,0)
    let result = mat * Vector3::new(1.0, 0.0, 0.0);
    assert_relative_eq!(result, Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-10);

    // Apply to (0, 1, 0):
    // q2: (0,1,0) → (0,0,1)
    // q1: (0,0,1) → (0,0,1)
    let result2 = mat * Vector3::new(0.0, 1.0, 0.0);
    assert_relative_eq!(result2, Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-10);
}

#[test]
fn component_accessors() {
    let q = RotQuaternion::new(1.0, 0.0, 0.0, 0.0);
    assert_relative_eq!(q.w(), 1.0, epsilon = 1e-12);
    assert_relative_eq!(q.x(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(q.y(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(q.z(), 0.0, epsilon = 1e-12);

    // Non-trivial: 90° around Z has w=cos(45°), z=sin(45°)
    let q2 = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let half = FRAC_PI_2 / 2.0;
    assert_relative_eq!(q2.w(), half.cos(), epsilon = 1e-12);
    assert_relative_eq!(q2.x(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(q2.y(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(q2.z(), half.sin(), epsilon = 1e-12);
}

#[test]
fn wxyz_array_round_trip() {
    let q_orig = RotQuaternion::from_axis_angle(Vector3::new(1.0, -0.5, 2.0), 0.9).unwrap();
    let arr = q_orig.to_wxyz_array();
    let q_back = RotQuaternion::from_wxyz_array(arr);
    assert_eq!(q_orig, q_back);
}

#[test]
fn euler_angles_known_values() {
    // Identity → all zeros
    let q_id = RotQuaternion::identity();
    let (roll, pitch, yaw) = q_id.to_euler_angles();
    assert_relative_eq!(roll, 0.0, epsilon = 1e-12);
    assert_relative_eq!(pitch, 0.0, epsilon = 1e-12);
    assert_relative_eq!(yaw, 0.0, epsilon = 1e-12);

    // 90° around Z → yaw = π/2 (nalgebra euler_angles returns roll, pitch, yaw)
    let q_z90 = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let (roll, pitch, yaw) = q_z90.to_euler_angles();
    assert_relative_eq!(roll, 0.0, epsilon = 1e-10);
    assert_relative_eq!(pitch, 0.0, epsilon = 1e-10);
    assert_relative_eq!(yaw, FRAC_PI_2, epsilon = 1e-10);
}

#[test]
fn partial_eq_sign_ambiguity() {
    let q = RotQuaternion::from_axis_angle(Vector3::new(1.0, 2.0, 3.0), 1.0).unwrap();
    let neg_q = RotQuaternion::new(-q.w(), -q.x(), -q.y(), -q.z());

    // q and -q represent the same rotation
    assert_eq!(q, neg_q);
}

#[test]
fn default_is_identity() {
    let q: RotQuaternion = Default::default();
    assert_eq!(q, RotQuaternion::identity());
}

#[test]
fn display_format() {
    let q = RotQuaternion::identity();
    let s = format!("{q}");
    assert!(s.contains("RotQuaternion"));
    assert!(s.contains("w="));
}

#[test]
fn owned_multiplication() {
    let q1 = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let q2 = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let q1_clone = q1.clone();
    let q2_clone = q2.clone();

    let ref_result = &q1 * &q2;
    let owned_result = q1_clone * q2_clone;

    assert_eq!(ref_result, owned_result);
}
