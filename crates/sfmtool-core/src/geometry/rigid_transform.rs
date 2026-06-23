// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rigid body transformation (rotation + translation, no scale).
//!
//! A [`RigidTransform`] represents an element of the special Euclidean group
//! SE(3): a proper rotation (det(R) = +1) composed with a translation. It
//! preserves distances and handedness.
//!
//! This complements [`crate::Se3Transform`], which extends rigid transforms
//! with a uniform scale factor (a similarity transformation).

use std::fmt;

use nalgebra::{Matrix3, Point3, Vector3};

use crate::RotQuaternion;

/// Rigid body transformation: rotation + translation, no scale.
///
/// Transforms a point via: `p' = R * p + t`
///
/// The rotation is stored as a unit quaternion, which guarantees:
/// - `det(R) = +1` (proper rotation, no reflections)
/// - `R^T R = I` (orthogonal)
/// - Distance-preserving: `|R*a - R*b| = |a - b|`
///
/// This is the appropriate type for camera extrinsics (world-to-camera),
/// relative poses between frames, and any rigid body motion. For transforms
/// that also include uniform scaling, use [`crate::Se3Transform`].
#[derive(Debug, Clone)]
pub struct RigidTransform {
    /// Rotation component (unit quaternion, so det(R) = +1).
    pub rotation: RotQuaternion,
    /// Translation component.
    pub translation: Vector3<f64>,
}

impl RigidTransform {
    /// Create a new rigid transform from rotation and translation components.
    pub fn new(rotation: RotQuaternion, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Create the identity transform (no rotation, no translation).
    pub fn identity() -> Self {
        Self {
            rotation: RotQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Create a rotation-only transform from an axis and angle (radians).
    ///
    /// Returns an error if the axis vector is zero (or near-zero).
    pub fn from_axis_angle(axis: Vector3<f64>, angle: f64) -> Result<Self, &'static str> {
        let rotation = RotQuaternion::from_axis_angle(axis, angle)?;
        Ok(Self {
            rotation,
            translation: Vector3::zeros(),
        })
    }

    /// Create from raw WXYZ quaternion and translation arrays.
    ///
    /// Convenient for numpy interop where quaternions and translations
    /// arrive as flat arrays.
    pub fn from_wxyz_translation(q: [f64; 4], t: [f64; 3]) -> Self {
        Self {
            rotation: RotQuaternion::from_wxyz_array(q),
            translation: Vector3::new(t[0], t[1], t[2]),
        }
    }

    /// Convert the rotation to a 3x3 rotation matrix.
    pub fn to_rotation_matrix(&self) -> Matrix3<f64> {
        self.rotation.to_rotation_matrix()
    }

    /// Compute the inverse translation origin in the source frame: `-R^T * t`.
    ///
    /// When used as a world-to-camera transform, this gives the camera center
    /// in world coordinates.
    pub fn inverse_translation_origin(&self) -> Point3<f64> {
        Point3::from(-self.rotation.inverse().rotate_vector(&self.translation))
    }

    /// Apply the transform to a point: `R * p + t`.
    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        let r = self.rotation.to_rotation_matrix();
        Point3::from(r * point.coords + self.translation)
    }
}

impl Default for RigidTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl PartialEq for RigidTransform {
    /// Approximate equality: rotation uses RotQuaternion's PartialEq (which handles
    /// sign ambiguity), translation uses component-wise epsilon of 1e-12.
    fn eq(&self, other: &Self) -> bool {
        if self.rotation != other.rotation {
            return false;
        }
        let dt = self.translation - other.translation;
        dt.x.abs() < 1e-12 && dt.y.abs() < 1e-12 && dt.z.abs() < 1e-12
    }
}

impl fmt::Display for RigidTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RigidTransform(rotation={}, translation=[{:.6}, {:.6}, {:.6}])",
            self.rotation, self.translation.x, self.translation.y, self.translation.z
        )
    }
}

#[cfg(test)]
mod tests;
