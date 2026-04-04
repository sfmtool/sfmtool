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
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4};

    // ── Helpers ──────────────────────────────────────────────────────────

    fn p(x: f64, y: f64, z: f64) -> Point3<f64> {
        Point3::new(x, y, z)
    }

    /// Extract camera center from a world-to-camera pose.
    fn camera_center(q: &RotQuaternion, t: &Vector3<f64>) -> Vector3<f64> {
        q.camera_center(t)
    }

    // ── 1. Identity preserves points ────────────────────────────────────

    #[test]
    fn identity_preserves_points() {
        let t = Se3Transform::identity();
        let point = p(5.0, -3.0, 7.0);
        let result = t.apply_to_point(&point);
        assert_relative_eq!(result.x, point.x, epsilon = 1e-12);
        assert_relative_eq!(result.y, point.y, epsilon = 1e-12);
        assert_relative_eq!(result.z, point.z, epsilon = 1e-12);
    }

    // ── 2. Scale only ───────────────────────────────────────────────────

    #[test]
    fn scale_only_doubles_distances() {
        let t = Se3Transform::new(RotQuaternion::identity(), Vector3::zeros(), 2.0);
        let result = t.apply_to_point(&p(1.0, 2.0, 3.0));
        assert_relative_eq!(result.x, 2.0, epsilon = 1e-12);
        assert_relative_eq!(result.y, 4.0, epsilon = 1e-12);
        assert_relative_eq!(result.z, 6.0, epsilon = 1e-12);
    }

    // ── 3. Translation only ─────────────────────────────────────────────

    #[test]
    fn translation_only_shifts_points() {
        let t = Se3Transform::new(
            RotQuaternion::identity(),
            Vector3::new(10.0, -5.0, 3.0),
            1.0,
        );
        let result = t.apply_to_point(&p(1.0, 2.0, 3.0));
        assert_relative_eq!(result.x, 11.0, epsilon = 1e-12);
        assert_relative_eq!(result.y, -3.0, epsilon = 1e-12);
        assert_relative_eq!(result.z, 6.0, epsilon = 1e-12);
    }

    // ── 4. Rotation only (90° around Z) ─────────────────────────────────

    #[test]
    fn rotation_90_around_z() {
        let t = Se3Transform::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
        let result = t.apply_to_point(&p(1.0, 0.0, 0.0));
        assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.z, 0.0, epsilon = 1e-10);
    }

    // ── 5. Full transform (rotation + translation + scale) ──────────────

    #[test]
    fn full_transform() {
        // identity rotation, translation (1,2,3), scale 2
        let t = Se3Transform::new(RotQuaternion::identity(), Vector3::new(1.0, 2.0, 3.0), 2.0);
        // 2 * (I * (1,1,1)) + (1,2,3) = (3,4,5)
        let result = t.apply_to_point(&p(1.0, 1.0, 1.0));
        assert_relative_eq!(result.x, 3.0, epsilon = 1e-12);
        assert_relative_eq!(result.y, 4.0, epsilon = 1e-12);
        assert_relative_eq!(result.z, 5.0, epsilon = 1e-12);
    }

    // ── 6. Compose translations ─────────────────────────────────────────

    #[test]
    fn compose_translations() {
        let t1 = Se3Transform::new(RotQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0), 1.0);
        let t2 = Se3Transform::new(RotQuaternion::identity(), Vector3::new(0.0, 2.0, 0.0), 1.0);
        let composed = t1.compose(&t2);
        let result = composed.apply_to_point(&p(0.0, 0.0, 0.0));
        assert_relative_eq!(result.x, 1.0, epsilon = 1e-12);
        assert_relative_eq!(result.y, 2.0, epsilon = 1e-12);
        assert_relative_eq!(result.z, 0.0, epsilon = 1e-12);
    }

    // ── 7. Compose scales ───────────────────────────────────────────────

    #[test]
    fn compose_scales() {
        let t1 = Se3Transform::new(RotQuaternion::identity(), Vector3::zeros(), 2.0);
        let t2 = Se3Transform::new(RotQuaternion::identity(), Vector3::zeros(), 3.0);
        let composed = t1.compose(&t2);
        assert_relative_eq!(composed.scale, 6.0, epsilon = 1e-12);
    }

    // ── 8. Compose rotations ────────────────────────────────────────────

    #[test]
    fn compose_rotations_two_90_gives_180() {
        let t1 = Se3Transform::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
        let t2 = Se3Transform::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
        let composed = t1.compose(&t2);

        // (1,0,0) rotated 180° around Z -> (-1,0,0)
        let result = composed.apply_to_point(&p(1.0, 0.0, 0.0));
        assert_relative_eq!(result.x, -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.z, 0.0, epsilon = 1e-10);
    }

    // ── 9. Compose equals sequential apply ──────────────────────────────

    #[test]
    fn compose_equals_sequential_apply() {
        let t1 = Se3Transform::new(
            RotQuaternion::from_axis_angle(Vector3::x(), FRAC_PI_4).unwrap(),
            Vector3::new(1.0, 2.0, 3.0),
            1.5,
        );
        let t2 = Se3Transform::new(
            RotQuaternion::from_axis_angle(Vector3::y(), FRAC_PI_3).unwrap(),
            Vector3::new(-1.0, 0.0, 2.0),
            0.8,
        );

        let composed = t1.compose(&t2);
        let point = p(5.0, -3.0, 2.0);

        let sequential = t2.apply_to_point(&t1.apply_to_point(&point));
        let composed_result = composed.apply_to_point(&point);

        assert_relative_eq!(sequential.x, composed_result.x, epsilon = 1e-10);
        assert_relative_eq!(sequential.y, composed_result.y, epsilon = 1e-10);
        assert_relative_eq!(sequential.z, composed_result.z, epsilon = 1e-10);
    }

    // ── 10. Inverse round-trip ──────────────────────────────────────────

    #[test]
    fn inverse_round_trip() {
        let t = Se3Transform::new(
            RotQuaternion::from_axis_angle(Vector3::new(1.0, 2.0, 3.0), 0.7).unwrap(),
            Vector3::new(5.0, -3.0, 2.0),
            2.0,
        );
        let inv = t.inverse().unwrap();

        let point = p(7.0, -2.0, 8.0);
        let transformed = t.apply_to_point(&point);
        let recovered = inv.apply_to_point(&transformed);

        assert_relative_eq!(recovered.x, point.x, epsilon = 1e-10);
        assert_relative_eq!(recovered.y, point.y, epsilon = 1e-10);
        assert_relative_eq!(recovered.z, point.z, epsilon = 1e-10);
    }

    // ── 11. Zero-scale inverse error ────────────────────────────────────

    #[test]
    fn zero_scale_inverse_error() {
        let t = Se3Transform::new(RotQuaternion::identity(), Vector3::zeros(), 0.0);
        assert!(t.inverse().is_err());
    }

    // ── 12. Compose with inverse is identity ────────────────────────────

    #[test]
    fn compose_with_inverse_is_identity() {
        let t = Se3Transform::new(
            RotQuaternion::from_axis_angle(Vector3::new(1.0, 1.0, 0.0), 0.5).unwrap(),
            Vector3::new(1.0, 2.0, 3.0),
            1.5,
        );
        let inv = t.inverse().unwrap();
        let composed = t.compose(&inv);

        let point = p(5.0, -3.0, 7.0);
        let result = composed.apply_to_point(&point);

        assert_relative_eq!(result.x, point.x, epsilon = 1e-10);
        assert_relative_eq!(result.y, point.y, epsilon = 1e-10);
        assert_relative_eq!(result.z, point.z, epsilon = 1e-10);
    }

    // ── 13. Apply to camera pose — center moves ─────────────────────────

    #[test]
    fn apply_to_camera_pose_center_moves() {
        // Translation-only SE3 transform: shift by (10, 20, 30)
        let se3 = Se3Transform::new(
            RotQuaternion::identity(),
            Vector3::new(10.0, 20.0, 30.0),
            1.0,
        );

        // Identity camera at origin
        let cam_rot = RotQuaternion::identity();
        let cam_trans = Vector3::zeros();

        let (new_rot, new_trans) = se3.apply_to_camera_pose(&cam_rot, &cam_trans);

        let new_center = camera_center(&new_rot, &new_trans);
        assert_relative_eq!(new_center.x, 10.0, epsilon = 1e-10);
        assert_relative_eq!(new_center.y, 20.0, epsilon = 1e-10);
        assert_relative_eq!(new_center.z, 30.0, epsilon = 1e-10);
    }

    // ── 14. Apply to camera pose — rotation updates ─────────────────────

    #[test]
    fn apply_to_camera_pose_rotation_updates() {
        // 90° around Z world transform
        let se3 = Se3Transform::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();

        // Identity camera at origin
        let cam_rot = RotQuaternion::identity();
        let cam_trans = Vector3::zeros();

        let (new_rot, _new_trans) = se3.apply_to_camera_pose(&cam_rot, &cam_trans);

        // q_new = q_cam * conj(q_world) = identity * conj(90°Z) = -90°Z
        let expected = RotQuaternion::from_axis_angle(Vector3::z(), -FRAC_PI_2).unwrap();
        let new_mat = new_rot.to_rotation_matrix();
        let expected_mat = expected.to_rotation_matrix();
        assert_relative_eq!(new_mat, expected_mat, epsilon = 1e-10);
    }

    // ── 15. Apply to camera pose round-trip ─────────────────────────────

    #[test]
    fn apply_to_camera_pose_round_trip() {
        let se3 = Se3Transform::new(
            RotQuaternion::from_axis_angle(Vector3::new(1.0, 1.0, 1.0), 0.7).unwrap(),
            Vector3::new(1.5, -2.3, 0.7),
            2.5,
        );
        let inv = se3.inverse().unwrap();

        // Non-trivial camera pose
        let cam_rot = RotQuaternion::from_axis_angle(Vector3::x(), 1.2).unwrap();
        let cam_trans = Vector3::new(3.0, -1.0, 5.0);

        // Forward
        let (fwd_rot, fwd_trans) = se3.apply_to_camera_pose(&cam_rot, &cam_trans);
        // Inverse
        let (back_rot, back_trans) = inv.apply_to_camera_pose(&fwd_rot, &fwd_trans);

        // Compare rotation matrices (sign ambiguity)
        let orig_mat = cam_rot.to_rotation_matrix();
        let back_mat = back_rot.to_rotation_matrix();
        assert_relative_eq!(orig_mat, back_mat, epsilon = 1e-10);

        assert_relative_eq!(back_trans.x, cam_trans.x, epsilon = 1e-10);
        assert_relative_eq!(back_trans.y, cam_trans.y, epsilon = 1e-10);
        assert_relative_eq!(back_trans.z, cam_trans.z, epsilon = 1e-10);
    }

    // ── apply_to_camera_poses_flat tests (matching transform.rs tests) ──

    #[test]
    fn flat_identity_preserves_poses() {
        let se3 = Se3Transform::identity();

        let half_angle = FRAC_PI_4 / 2.0;
        let cam_q = [half_angle.cos(), half_angle.sin(), 0.0, 0.0];
        let cam_t = [0.5, -1.0, 2.0];

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        se3.apply_to_camera_poses_flat(&cam_q, &cam_t, &mut out_q, &mut out_t);

        for (a, b) in cam_q.iter().zip(out_q.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        for (a, b) in cam_t.iter().zip(out_t.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn flat_pure_translation_shifts_center() {
        let se3 = Se3Transform::new(
            RotQuaternion::identity(),
            Vector3::new(10.0, 20.0, 30.0),
            1.0,
        );

        let cam_q = [1.0, 0.0, 0.0, 0.0];
        let cam_t = [0.0, 0.0, 0.0];

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        se3.apply_to_camera_poses_flat(&cam_q, &cam_t, &mut out_q, &mut out_t);

        assert_relative_eq!(out_q[0], 1.0, epsilon = 1e-12);

        // Camera center should have shifted
        let center = camera_center(
            &RotQuaternion::from_wxyz_array([out_q[0], out_q[1], out_q[2], out_q[3]]),
            &Vector3::new(out_t[0], out_t[1], out_t[2]),
        );
        assert_relative_eq!(center.x, 10.0, epsilon = 1e-12);
        assert_relative_eq!(center.y, 20.0, epsilon = 1e-12);
        assert_relative_eq!(center.z, 30.0, epsilon = 1e-12);
    }

    #[test]
    fn flat_scale_triples_distances() {
        let se3 = Se3Transform::new(RotQuaternion::identity(), Vector3::zeros(), 3.0);

        // Camera at (2,0,0): identity rotation, t = -R*C = (-2,0,0)
        let cam_q = [1.0, 0.0, 0.0, 0.0];
        let cam_t = [-2.0, 0.0, 0.0];

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        se3.apply_to_camera_poses_flat(&cam_q, &cam_t, &mut out_q, &mut out_t);

        let center = camera_center(
            &RotQuaternion::from_wxyz_array([out_q[0], out_q[1], out_q[2], out_q[3]]),
            &Vector3::new(out_t[0], out_t[1], out_t[2]),
        );
        assert_relative_eq!(center.x, 6.0, epsilon = 1e-10);
        assert_relative_eq!(center.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(center.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn flat_batch_multiple_cameras() {
        let se3 = Se3Transform::new(RotQuaternion::identity(), Vector3::new(1.0, 2.0, 3.0), 2.0);

        let cam_q = [
            1.0, 0.0, 0.0, 0.0, // camera 0: identity
            1.0, 0.0, 0.0, 0.0, // camera 1: identity
        ];
        let cam_t = [
            0.0, 0.0, 0.0, // camera 0 at origin
            -1.0, 0.0, 0.0, // camera 1 at (1,0,0)
        ];

        let mut out_q = [0.0; 8];
        let mut out_t = [0.0; 6];

        se3.apply_to_camera_poses_flat(&cam_q, &cam_t, &mut out_q, &mut out_t);

        // Camera 0: center (0,0,0) -> 2*(0,0,0) + (1,2,3) = (1,2,3)
        let c0 = camera_center(
            &RotQuaternion::from_wxyz_array([out_q[0], out_q[1], out_q[2], out_q[3]]),
            &Vector3::new(out_t[0], out_t[1], out_t[2]),
        );
        assert_relative_eq!(c0.x, 1.0, epsilon = 1e-12);
        assert_relative_eq!(c0.y, 2.0, epsilon = 1e-12);
        assert_relative_eq!(c0.z, 3.0, epsilon = 1e-12);

        // Camera 1: center (1,0,0) -> 2*(1,0,0) + (1,2,3) = (3,2,3)
        let c1 = camera_center(
            &RotQuaternion::from_wxyz_array([out_q[4], out_q[5], out_q[6], out_q[7]]),
            &Vector3::new(out_t[3], out_t[4], out_t[5]),
        );
        assert_relative_eq!(c1.x, 3.0, epsilon = 1e-12);
        assert_relative_eq!(c1.y, 2.0, epsilon = 1e-12);
        assert_relative_eq!(c1.z, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn flat_round_trip_recovers_original() {
        let se3 = Se3Transform::new(
            RotQuaternion::from_axis_angle(Vector3::new(1.0, 1.0, 1.0), 0.7).unwrap(),
            Vector3::new(1.5, -2.3, 0.7),
            2.5,
        );
        let inv = se3.inverse().unwrap();

        // Non-trivial camera pose
        let cam_q = {
            let q = RotQuaternion::from_axis_angle(Vector3::x(), 1.2).unwrap();
            q.to_wxyz_array()
        };
        let cam_t = [3.0, -1.0, 5.0];

        // Forward
        let mut fwd_q = [0.0; 4];
        let mut fwd_t = [0.0; 3];
        se3.apply_to_camera_poses_flat(&cam_q, &cam_t, &mut fwd_q, &mut fwd_t);

        // Inverse
        let mut back_q = [0.0; 4];
        let mut back_t = [0.0; 3];
        inv.apply_to_camera_poses_flat(&fwd_q, &fwd_t, &mut back_q, &mut back_t);

        // Account for quaternion sign ambiguity
        let sign = if back_q[0] * cam_q[0] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for j in 0..4 {
            assert_relative_eq!(sign * back_q[j], cam_q[j], epsilon = 1e-10);
        }
        for j in 0..3 {
            assert_relative_eq!(back_t[j], cam_t[j], epsilon = 1e-10);
        }
    }

    // ── Default trait ───────────────────────────────────────────────────

    #[test]
    fn default_is_identity() {
        let t: Se3Transform = Default::default();
        let point = p(5.0, -3.0, 7.0);
        let result = t.apply_to_point(&point);
        assert_relative_eq!(result.x, point.x, epsilon = 1e-12);
        assert_relative_eq!(result.y, point.y, epsilon = 1e-12);
        assert_relative_eq!(result.z, point.z, epsilon = 1e-12);
    }

    // ── from_axis_angle error ───────────────────────────────────────────

    #[test]
    fn from_axis_angle_zero_axis_error() {
        assert!(Se3Transform::from_axis_angle(Vector3::zeros(), 1.0).is_err());
    }

    // ── apply_to_points batch ───────────────────────────────────────────

    #[test]
    fn apply_to_points_matches_individual() {
        let t = Se3Transform::new(
            RotQuaternion::from_axis_angle(Vector3::new(1.0, 2.0, 3.0), 0.5).unwrap(),
            Vector3::new(1.0, -1.0, 2.0),
            1.5,
        );

        let points = vec![p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0), p(0.0, 0.0, 1.0)];

        let batch = t.apply_to_points(&points);

        for (i, pt) in points.iter().enumerate() {
            let individual = t.apply_to_point(pt);
            assert_relative_eq!(batch[i].x, individual.x, epsilon = 1e-12);
            assert_relative_eq!(batch[i].y, individual.y, epsilon = 1e-12);
            assert_relative_eq!(batch[i].z, individual.z, epsilon = 1e-12);
        }
    }
}