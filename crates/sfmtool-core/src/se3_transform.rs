// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! SE(3) similarity transformation: rotation, translation, and uniform scale.
//!
//! This module provides [`Se3Transform`] for representing and manipulating
//! similarity transformations used in structure-from-motion reconstruction.

use nalgebra::{Point3, Vector3};

use crate::rigid_transform::RigidTransform;
use crate::rot_quaternion::RotQuaternion;

/// SE(3) similarity transformation: rotation, translation, and uniform scale.
///
/// Applies as: `p' = scale * (R * p) + t`
///
/// This is used to align reconstructions, transform coordinate frames, and
/// compose sequences of geometric transforms in SfM pipelines.
#[derive(Debug, Clone)]
pub struct Se3Transform {
    /// Rotation component.
    pub rotation: RotQuaternion,
    /// Translation component.
    pub translation: Vector3<f64>,
    /// Uniform scale factor.
    pub scale: f64,
}

impl Se3Transform {
    /// Create a new SE(3) transform from components.
    pub fn new(rotation: RotQuaternion, translation: Vector3<f64>, scale: f64) -> Self {
        Self {
            rotation,
            translation,
            scale,
        }
    }

    /// Create the identity transform (no rotation, no translation, scale=1).
    pub fn identity() -> Self {
        Self {
            rotation: RotQuaternion::identity(),
            translation: Vector3::zeros(),
            scale: 1.0,
        }
    }

    /// Create a rotation-only transform from an axis and angle (radians).
    ///
    /// Returns an error if the axis vector is zero (or near-zero).
    pub fn from_axis_angle(axis: Vector3<f64>, angle: f64) -> Result<Self, &'static str> {
        let rigid = RigidTransform::from_axis_angle(axis, angle)?;
        Ok(Self {
            rotation: rigid.rotation,
            translation: rigid.translation,
            scale: 1.0,
        })
    }

    /// Apply the transform to a single 3D point: `scale * (R * p) + t`.
    pub fn apply_to_point(&self, point: &Point3<f64>) -> Point3<f64> {
        let rot_matrix = self.rotation.to_rotation_matrix();
        let rotated = rot_matrix * point.coords;
        Point3::from(self.scale * rotated + self.translation)
    }

    /// Apply the transform to a slice of 3D points.
    pub fn apply_to_points(&self, points: &[Point3<f64>]) -> Vec<Point3<f64>> {
        let rot_matrix = self.rotation.to_rotation_matrix();
        points
            .iter()
            .map(|p| {
                let rotated = rot_matrix * p.coords;
                Point3::from(self.scale * rotated + self.translation)
            })
            .collect()
    }

    /// Transform a single camera pose (world-to-camera convention).
    ///
    /// Given a camera pose `(q_cam, t_cam)` where `p_camera = R_cam * p_world + t_cam`,
    /// compute the new pose after applying this SE(3) world transform.
    ///
    /// The math:
    /// 1. Camera center: `C = -R_cam^T * t_cam`
    /// 2. Transform center: `C' = scale * (R_world * C) + t_world`
    /// 3. New rotation: `q_new = q_cam * conjugate(q_world)`
    /// 4. New translation: `t_new = -R_new * C'`
    pub fn apply_to_camera_pose(
        &self,
        cam_rot: &RotQuaternion,
        cam_trans: &Vector3<f64>,
    ) -> (RotQuaternion, Vector3<f64>) {
        // Camera center in world coords: C = -R_cam^T * t_cam
        let camera_center = -cam_rot.inverse().rotate_vector(cam_trans);

        // Transform camera center as world point: C' = scale * (R_world * C) + t_world
        let new_center =
            self.scale * self.rotation.rotate_vector(&camera_center) + self.translation;

        // New camera rotation: q_new = q_cam * conj(q_world)
        let new_cam_quat = cam_rot * &self.rotation.inverse();

        // New translation: t_new = -R_new * C'
        let new_cam_trans = -new_cam_quat.rotate_vector(&new_center);

        (new_cam_quat, new_cam_trans)
    }

    /// Batch-transform camera poses using flat f64 slices (for numpy interop).
    ///
    /// RotQuaternions are packed as `[w0,x0,y0,z0, w1,x1,y1,z1, ...]` and
    /// translations as `[tx0,ty0,tz0, tx1,ty1,tz1, ...]`.
    pub fn apply_to_camera_poses_flat(
        &self,
        q_in: &[f64],
        t_in: &[f64],
        q_out: &mut [f64],
        t_out: &mut [f64],
    ) {
        let n = q_in.len() / 4;
        debug_assert_eq!(q_in.len(), n * 4);
        debug_assert_eq!(t_in.len(), n * 3);
        debug_assert_eq!(q_out.len(), n * 4);
        debug_assert_eq!(t_out.len(), n * 3);

        for i in 0..n {
            let q_off = i * 4;
            let t_off = i * 3;

            let cam_rot = RotQuaternion::from_wxyz_array([
                q_in[q_off],
                q_in[q_off + 1],
                q_in[q_off + 2],
                q_in[q_off + 3],
            ]);
            let cam_trans = Vector3::new(t_in[t_off], t_in[t_off + 1], t_in[t_off + 2]);

            let (new_rot, new_trans) = self.apply_to_camera_pose(&cam_rot, &cam_trans);

            let q = new_rot.to_wxyz_array();
            q_out[q_off] = q[0];
            q_out[q_off + 1] = q[1];
            q_out[q_off + 2] = q[2];
            q_out[q_off + 3] = q[3];

            t_out[t_off] = new_trans.x;
            t_out[t_off + 1] = new_trans.y;
            t_out[t_off + 2] = new_trans.z;
        }
    }

    /// Compose two transforms: apply `self` first, then `other`.
    ///
    /// Result satisfies: `composed.apply(p) == other.apply(self.apply(p))`
    pub fn compose(&self, other: &Self) -> Self {
        // combined_rotation = other.rotation * self.rotation
        let combined_rotation = &other.rotation * &self.rotation;

        // combined_translation = other.scale * (other.rotation * self.translation) + other.translation
        let combined_translation =
            other.scale * other.rotation.rotate_vector(&self.translation) + other.translation;

        // combined_scale = self.scale * other.scale
        let combined_scale = self.scale * other.scale;

        Self {
            rotation: combined_rotation,
            translation: combined_translation,
            scale: combined_scale,
        }
    }

    /// Compute the inverse transform.
    ///
    /// Returns an error if scale is zero.
    pub fn inverse(&self) -> Result<Self, &'static str> {
        if self.scale == 0.0 {
            return Err("transform with scale=0 cannot be inverted");
        }

        let inv_rotation = self.rotation.inverse();

        // inv_translation = -(inv_rotation * self.translation) / self.scale
        let inv_translation = -inv_rotation.rotate_vector(&self.translation) / self.scale;

        let inv_scale = 1.0 / self.scale;

        Ok(Self {
            rotation: inv_rotation,
            translation: inv_translation,
            scale: inv_scale,
        })
    }
}

impl Default for Se3Transform {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests;
