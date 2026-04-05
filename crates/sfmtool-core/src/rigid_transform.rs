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
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn identity_transform() {
        let t = RigidTransform::identity();

        assert_eq!(t.rotation, RotQuaternion::identity());

        assert_relative_eq!(t.translation.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(t.translation.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(t.translation.z, 0.0, epsilon = 1e-12);

        let origin = t.inverse_translation_origin();
        assert_relative_eq!(origin.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(origin.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(origin.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn from_axis_angle_rotation_only() {
        let rt = RigidTransform::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
        // Translation should be zero
        assert_relative_eq!(rt.translation.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(rt.translation.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(rt.translation.z, 0.0, epsilon = 1e-12);
        // Rotating (1,0,0) by 90° around Z should give (0,1,0)
        let result = rt.transform_point(&Point3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn from_axis_angle_zero_axis_error() {
        assert!(RigidTransform::from_axis_angle(Vector3::zeros(), 1.0).is_err());
    }

    #[test]
    fn from_wxyz_translation_round_trip() {
        let q = [0.5, 0.5, 0.5, 0.5];
        let t = [1.0, 2.0, 3.0];
        let rt = RigidTransform::from_wxyz_translation(q, t);

        let q_out = rt.rotation.to_wxyz_array();
        assert_relative_eq!(q_out[0], q[0], epsilon = 1e-12);
        assert_relative_eq!(q_out[1], q[1], epsilon = 1e-12);
        assert_relative_eq!(q_out[2], q[2], epsilon = 1e-12);
        assert_relative_eq!(q_out[3], q[3], epsilon = 1e-12);

        assert_relative_eq!(rt.translation.x, t[0], epsilon = 1e-12);
        assert_relative_eq!(rt.translation.y, t[1], epsilon = 1e-12);
        assert_relative_eq!(rt.translation.z, t[2], epsilon = 1e-12);
    }

    #[test]
    fn inverse_translation_origin_identity_rotation() {
        // Identity rotation, t = [0, 0, 5]
        // origin = -R^T * t = -I * [0,0,5] = [0, 0, -5]
        let rt = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.0, 0.0, 5.0));
        let origin = rt.inverse_translation_origin();
        assert_relative_eq!(origin.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(origin.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(origin.z, -5.0, epsilon = 1e-12);
    }

    #[test]
    fn inverse_translation_origin_with_rotation() {
        // 90 degrees around Y: R rotates X->Z, Z->-X
        // R = [[0,0,1],[0,1,0],[-1,0,0]]
        // With t = [1, 0, 0]:
        // origin = -R^T * t = -[[0,0,-1],[0,1,0],[1,0,0]] * [1,0,0] = -[0,0,1] = [0,0,-1]
        let rot = RotQuaternion::from_axis_angle(Vector3::y(), FRAC_PI_2).unwrap();
        let rt = RigidTransform::new(rot, Vector3::new(1.0, 0.0, 0.0));
        let origin = rt.inverse_translation_origin();
        assert_relative_eq!(origin.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(origin.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(origin.z, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn transform_point_identity_preserves() {
        let rt = RigidTransform::identity();
        let point = Point3::new(5.0, -3.0, 7.0);
        let result = rt.transform_point(&point);
        assert_relative_eq!(result.x, point.x, epsilon = 1e-12);
        assert_relative_eq!(result.y, point.y, epsilon = 1e-12);
        assert_relative_eq!(result.z, point.z, epsilon = 1e-12);
    }

    #[test]
    fn transform_point_with_rotation_and_translation() {
        // 90 degrees around Z: (1,0,0) -> (0,1,0)
        // Then add translation (10, 20, 30)
        // Result: (0+10, 1+20, 0+30) = (10, 21, 30)
        let rot = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
        let rt = RigidTransform::new(rot, Vector3::new(10.0, 20.0, 30.0));
        let result = rt.transform_point(&Point3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(result.x, 10.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 21.0, epsilon = 1e-10);
        assert_relative_eq!(result.z, 30.0, epsilon = 1e-10);
    }

    #[test]
    fn partial_eq_same() {
        let rt1 = RigidTransform::new(
            RotQuaternion::from_axis_angle(Vector3::z(), 1.0).unwrap(),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let rt2 = rt1.clone();
        assert_eq!(rt1, rt2);
    }

    #[test]
    fn partial_eq_different() {
        let rt1 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(1.0, 2.0, 3.0));
        let rt2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(1.0, 2.0, 4.0));
        assert_ne!(rt1, rt2);
    }

    #[test]
    fn partial_eq_quaternion_sign_ambiguity() {
        let q = RotQuaternion::from_axis_angle(Vector3::new(1.0, 2.0, 3.0), 1.0).unwrap();
        let neg_q = RotQuaternion::new(-q.w(), -q.x(), -q.y(), -q.z());
        let t = Vector3::new(1.0, 2.0, 3.0);

        let rt1 = RigidTransform::new(q, t);
        let rt2 = RigidTransform::new(neg_q, t);
        assert_eq!(rt1, rt2);
    }

    #[test]
    fn default_is_identity() {
        let rt: RigidTransform = Default::default();
        assert_eq!(rt, RigidTransform::identity());
    }

    #[test]
    fn display_format() {
        let rt = RigidTransform::identity();
        let s = format!("{rt}");
        assert!(s.contains("RigidTransform"));
        assert!(s.contains("rotation="));
        assert!(s.contains("translation="));
    }

    #[test]
    fn to_rotation_matrix_delegates() {
        let rot = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
        let rt = RigidTransform::new(rot.clone(), Vector3::zeros());
        let mat_rt = rt.to_rotation_matrix();
        let mat_quat = rot.to_rotation_matrix();
        assert_relative_eq!(mat_rt, mat_quat, epsilon = 1e-12);
    }
}
