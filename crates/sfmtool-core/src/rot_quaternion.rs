// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rotation quaternion wrapper for 3D rotations in structure-from-motion.

use std::fmt;
use std::ops::Mul;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

/// Rotation quaternion representing a 3D rotation, stored in WXYZ order.
///
/// This is a thin wrapper around nalgebra's [`nalgebra::UnitQuaternion<f64>`],
/// providing a domain-specific API for 3D rotations used in
/// structure-from-motion.
///
/// The quaternion is always normalized (unit length), guaranteeing that the
/// corresponding rotation matrix has `det(R) = +1` (proper rotation).
#[derive(Debug, Clone)]
pub struct RotQuaternion {
    inner: UnitQuaternion<f64>,
}

impl RotQuaternion {
    /// Create a quaternion from individual components, normalizing to unit length.
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: UnitQuaternion::new_normalize(nalgebra::Quaternion::new(w, x, y, z)),
        }
    }

    /// Create the identity quaternion (no rotation).
    pub fn identity() -> Self {
        Self {
            inner: UnitQuaternion::identity(),
        }
    }

    /// Create a quaternion from a WXYZ array, normalizing to unit length.
    pub fn from_wxyz_array(wxyz: [f64; 4]) -> Self {
        Self::new(wxyz[0], wxyz[1], wxyz[2], wxyz[3])
    }

    /// Create a quaternion from a 3x3 rotation matrix.
    pub fn from_rotation_matrix(mat: Matrix3<f64>) -> Self {
        let rot = nalgebra::Rotation3::from_matrix_unchecked(mat);
        Self {
            inner: UnitQuaternion::from_rotation_matrix(&rot),
        }
    }

    /// Create a quaternion from an axis and angle (radians).
    ///
    /// Returns an error if the axis vector is zero (or near-zero).
    pub fn from_axis_angle(axis: Vector3<f64>, angle: f64) -> Result<Self, &'static str> {
        let norm = axis.norm();
        if norm < 1e-12 {
            return Err("axis vector must be non-zero");
        }
        let unit_axis = nalgebra::Unit::new_normalize(axis);
        Ok(Self {
            inner: UnitQuaternion::from_axis_angle(&unit_axis, angle),
        })
    }

    /// Convert to a 3x3 rotation matrix.
    pub fn to_rotation_matrix(&self) -> Matrix3<f64> {
        *self.inner.to_rotation_matrix().matrix()
    }

    /// Extract as a [w, x, y, z] array.
    pub fn to_wxyz_array(&self) -> [f64; 4] {
        let q = self.inner.quaternion();
        [q.w, q.i, q.j, q.k]
    }

    /// Convert to Euler angles (roll, pitch, yaw) in radians.
    pub fn to_euler_angles(&self) -> (f64, f64, f64) {
        self.inner.euler_angles()
    }

    /// Return the conjugate quaternion (same as inverse for rotation quaternions).
    pub fn conjugate(&self) -> Self {
        Self {
            inner: self.inner.conjugate(),
        }
    }

    /// Return the inverse rotation (equivalent to conjugate for unit quaternions).
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }

    /// The scalar (w) component.
    pub fn w(&self) -> f64 {
        self.inner.quaternion().w
    }

    /// The first imaginary (x) component.
    pub fn x(&self) -> f64 {
        self.inner.quaternion().i
    }

    /// The second imaginary (y) component.
    pub fn y(&self) -> f64 {
        self.inner.quaternion().j
    }

    /// The third imaginary (z) component.
    pub fn z(&self) -> f64 {
        self.inner.quaternion().k
    }

    /// Rotate a 3D vector by this quaternion.
    pub fn rotate_vector(&self, v: &Vector3<f64>) -> Vector3<f64> {
        self.inner.transform_vector(v)
    }

    /// Compute the camera center in world coordinates from a world-to-camera pose.
    ///
    /// Given a world-to-camera transform where `p_camera = R * p_world + t`,
    /// returns the camera center `C = -R^T * t` (the camera position in world coords).
    pub fn camera_center(&self, translation: &Vector3<f64>) -> Vector3<f64> {
        -self.inverse().rotate_vector(translation)
    }

    /// Access the inner nalgebra `UnitQuaternion` for interop with other Rust code.
    pub fn as_nalgebra(&self) -> &UnitQuaternion<f64> {
        &self.inner
    }

    /// Create from an existing nalgebra `UnitQuaternion`.
    ///
    /// This is the primary constructor for internal Rust code that already
    /// works with nalgebra types.
    pub fn from_nalgebra(uq: UnitQuaternion<f64>) -> Self {
        Self { inner: uq }
    }
}

impl Mul<&RotQuaternion> for &RotQuaternion {
    type Output = RotQuaternion;

    fn mul(self, rhs: &RotQuaternion) -> RotQuaternion {
        RotQuaternion {
            inner: self.inner * rhs.inner,
        }
    }
}

impl Mul<RotQuaternion> for RotQuaternion {
    type Output = RotQuaternion;

    fn mul(self, rhs: RotQuaternion) -> RotQuaternion {
        RotQuaternion {
            inner: self.inner * rhs.inner,
        }
    }
}

impl Default for RotQuaternion {
    fn default() -> Self {
        Self::identity()
    }
}

impl PartialEq for RotQuaternion {
    /// Approximate equality with epsilon = 1e-12, accounting for the sign
    /// ambiguity where q and -q represent the same rotation.
    fn eq(&self, other: &Self) -> bool {
        let a = self.to_wxyz_array();
        let b = other.to_wxyz_array();

        let diff_pos = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).abs())
            .fold(0.0_f64, f64::max);

        let diff_neg = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai + bi).abs())
            .fold(0.0_f64, f64::max);

        diff_pos < 1e-12 || diff_neg < 1e-12
    }
}

impl fmt::Display for RotQuaternion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RotQuaternion(w={:.6}, x={:.6}, y={:.6}, z={:.6})",
            self.w(),
            self.x(),
            self.y(),
            self.z()
        )
    }
}

#[cfg(test)]
mod tests {
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
}
