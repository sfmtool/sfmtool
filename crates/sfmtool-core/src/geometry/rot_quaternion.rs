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

    /// The rotation angle in radians (always non-negative, in [0, π]).
    pub fn angle(&self) -> f64 {
        self.inner.angle()
    }

    /// Spherical linear interpolation between `self` and `other` at parameter `t`.
    ///
    /// `t=0` returns `self`, `t=1` returns `other`. Values outside [0, 1]
    /// extrapolate beyond the two rotations.
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        Self {
            inner: self.inner.slerp(&other.inner, t),
        }
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
mod tests;
