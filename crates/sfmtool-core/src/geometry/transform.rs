// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! SE3 similarity transform applied to batched camera poses.

use nalgebra::Vector3;

use crate::rot_quaternion::RotQuaternion;
use crate::se3_transform::Se3Transform;

/// Apply an SE3 similarity transform to batched camera poses.
///
/// Given a world transform T = (R_world, t_world, scale), transform each
/// camera pose (q_cam, t_cam) representing a world-to-camera transform.
///
/// The transform maps world coordinates: p_new = scale * R_world * p_old + t_world
/// Camera poses must be updated so that C_new = C_old * T^{-1}
///
/// # Arguments
/// * `rotation_wxyz` - Transform rotation as [w, x, y, z]
/// * `translation` - Transform translation as [tx, ty, tz]
/// * `scale` - Transform uniform scale factor
/// * `quaternions_wxyz` - Flat slice of camera quaternions [w0,x0,y0,z0, w1,...], length n*4
/// * `translations_xyz` - Flat slice of camera translations [tx0,ty0,tz0, ...], length n*3
/// * `out_quaternions_wxyz` - Output buffer for transformed quaternions (same layout)
/// * `out_translations_xyz` - Output buffer for transformed translations (same layout)
pub fn apply_se3_to_camera_poses(
    rotation_wxyz: [f64; 4],
    translation: [f64; 3],
    scale: f64,
    quaternions_wxyz: &[f64],
    translations_xyz: &[f64],
    out_quaternions_wxyz: &mut [f64],
    out_translations_xyz: &mut [f64],
) {
    let se3 = Se3Transform::new(
        RotQuaternion::from_wxyz_array(rotation_wxyz),
        Vector3::new(translation[0], translation[1], translation[2]),
        scale,
    );
    se3.apply_to_camera_poses_flat(
        quaternions_wxyz,
        translations_xyz,
        out_quaternions_wxyz,
        out_translations_xyz,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_4;

    /// Helper: extract camera center from pose (q_wxyz, t_xyz).
    fn camera_center(q_wxyz: &[f64; 4], t_xyz: &[f64; 3]) -> Vector3<f64> {
        let q = RotQuaternion::from_wxyz_array(*q_wxyz);
        let t = Vector3::new(t_xyz[0], t_xyz[1], t_xyz[2]);
        let r = q.to_rotation_matrix();
        -(r.transpose() * t)
    }

    #[test]
    fn identity_transform_preserves_poses() {
        let rot = [1.0, 0.0, 0.0, 0.0]; // identity quaternion
        let trans = [0.0, 0.0, 0.0];
        let scale = 1.0;

        // Camera with 45 deg rotation around X
        let half_angle = FRAC_PI_4 / 2.0;
        let cam_q = [half_angle.cos(), half_angle.sin(), 0.0, 0.0]; // 22.5 deg half-angle
        let cam_t = [0.5, -1.0, 2.0];

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        apply_se3_to_camera_poses(rot, trans, scale, &cam_q, &cam_t, &mut out_q, &mut out_t);

        for (a, b) in cam_q.iter().zip(out_q.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        for (a, b) in cam_t.iter().zip(out_t.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn pure_translation_shifts_center() {
        let rot = [1.0, 0.0, 0.0, 0.0];
        let trans = [10.0, 20.0, 30.0];
        let scale = 1.0;

        // Identity camera at origin: q=identity, t=0
        let cam_q = [1.0, 0.0, 0.0, 0.0];
        let cam_t = [0.0, 0.0, 0.0];

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        apply_se3_to_camera_poses(rot, trans, scale, &cam_q, &cam_t, &mut out_q, &mut out_t);

        // Orientation should be unchanged
        assert_relative_eq!(out_q[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(out_q[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out_q[2], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out_q[3], 0.0, epsilon = 1e-12);

        // Camera center should have shifted by the translation
        let new_center = camera_center(&out_q, &[out_t[0], out_t[1], out_t[2]]);
        assert_relative_eq!(new_center.x, 10.0, epsilon = 1e-12);
        assert_relative_eq!(new_center.y, 20.0, epsilon = 1e-12);
        assert_relative_eq!(new_center.z, 30.0, epsilon = 1e-12);
    }

    #[test]
    fn pure_rotation_composes_correctly() {
        // World transform: 90 deg around Z
        let angle = std::f64::consts::FRAC_PI_2;
        let half = (angle / 2.0).sin();
        let cos_half = (angle / 2.0).cos();
        let world_q = [cos_half, 0.0, 0.0, half]; // 90 deg around Z

        let trans = [0.0, 0.0, 0.0];
        let scale = 1.0;

        // Camera with identity rotation at origin
        let cam_q = [1.0, 0.0, 0.0, 0.0];
        let cam_t = [0.0, 0.0, 0.0];

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        apply_se3_to_camera_poses(
            world_q, trans, scale, &cam_q, &cam_t, &mut out_q, &mut out_t,
        );

        // New camera quat = cam_q * conj(world_q) = identity * conj(90 deg Z)
        // = conjugate of 90 deg Z = -90 deg Z
        let expected_q = RotQuaternion::new(cos_half, 0.0, 0.0, -half);
        let expected_arr = expected_q.to_wxyz_array();
        assert_relative_eq!(out_q[0], expected_arr[0], epsilon = 1e-12);
        assert_relative_eq!(out_q[1], expected_arr[1], epsilon = 1e-12);
        assert_relative_eq!(out_q[2], expected_arr[2], epsilon = 1e-12);
        assert_relative_eq!(out_q[3], expected_arr[3], epsilon = 1e-12);
    }

    #[test]
    fn round_trip_recovers_original() {
        // Non-trivial transform
        let angle: f64 = 0.7;
        let half = (angle / 2.0).sin();
        let cos_half = (angle / 2.0).cos();
        // Rotation around normalized (1,1,1)
        let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
        let world_q = [
            cos_half,
            half * inv_sqrt3,
            half * inv_sqrt3,
            half * inv_sqrt3,
        ];
        let world_trans = [1.5, -2.3, 0.7];
        let scale = 2.5;

        // Non-trivial camera pose
        let cam_angle: f64 = 1.2;
        let cam_half = (cam_angle / 2.0).sin();
        let cam_cos = (cam_angle / 2.0).cos();
        let cam_q = [cam_cos, cam_half, 0.0, 0.0]; // rotation around X
        let cam_t = [3.0, -1.0, 5.0];

        // Forward transform
        let mut fwd_q = [0.0; 4];
        let mut fwd_t = [0.0; 3];
        apply_se3_to_camera_poses(
            world_q,
            world_trans,
            scale,
            &cam_q,
            &cam_t,
            &mut fwd_q,
            &mut fwd_t,
        );

        // Compute inverse transform using Se3Transform
        let se3 = crate::se3_transform::Se3Transform::new(
            RotQuaternion::from_wxyz_array(world_q),
            Vector3::new(world_trans[0], world_trans[1], world_trans[2]),
            scale,
        );
        let inv = se3.inverse().unwrap();
        let inv_q_arr = inv.rotation.to_wxyz_array();
        let inv_q = [inv_q_arr[0], inv_q_arr[1], inv_q_arr[2], inv_q_arr[3]];
        let inv_t = [inv.translation.x, inv.translation.y, inv.translation.z];
        let inv_scale = inv.scale;

        // Inverse transform
        let mut back_q = [0.0; 4];
        let mut back_t = [0.0; 3];
        apply_se3_to_camera_poses(
            inv_q,
            inv_t,
            inv_scale,
            &fwd_q,
            &fwd_t,
            &mut back_q,
            &mut back_t,
        );

        // Account for possible sign flip in quaternion (q and -q represent same rotation)
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

    #[test]
    fn scale_transform_changes_distances() {
        // Scale = 3.0 with identity rotation should triple all distances from origin
        let rot = [1.0, 0.0, 0.0, 0.0];
        let trans = [0.0, 0.0, 0.0];
        let scale = 3.0;

        // Camera at (2, 0, 0): identity rotation, t = -R*C = (-2, 0, 0)
        let cam_q = [1.0, 0.0, 0.0, 0.0];
        let cam_t = [-2.0, 0.0, 0.0];

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        apply_se3_to_camera_poses(rot, trans, scale, &cam_q, &cam_t, &mut out_q, &mut out_t);

        // New center should be at (6, 0, 0) = 3 * (2, 0, 0)
        let new_center = camera_center(&out_q, &[out_t[0], out_t[1], out_t[2]]);
        assert_relative_eq!(new_center.x, 6.0, epsilon = 1e-10);
        assert_relative_eq!(new_center.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(new_center.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn compose_rotation_and_translation() {
        // Apply both rotation (90° around Z) and translation (10, 0, 0)
        let angle = std::f64::consts::FRAC_PI_2;
        let half = (angle / 2.0).sin();
        let cos_half = (angle / 2.0).cos();
        let world_q = [cos_half, 0.0, 0.0, half]; // 90° around Z
        let trans = [10.0, 0.0, 0.0];
        let scale = 1.0;

        // Camera at (1, 0, 0) with identity rotation
        let cam_q = [1.0, 0.0, 0.0, 0.0];
        let cam_t = [-1.0, 0.0, 0.0]; // t = -R*C = -I*(1,0,0) = (-1,0,0)

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        apply_se3_to_camera_poses(
            world_q, trans, scale, &cam_q, &cam_t, &mut out_q, &mut out_t,
        );

        // Center = R*(1,0,0) + t = 90°Z*(1,0,0) + (10,0,0) = (0,1,0) + (10,0,0) = (10,1,0)
        let new_center = camera_center(&out_q, &[out_t[0], out_t[1], out_t[2]]);
        assert_relative_eq!(new_center.x, 10.0, epsilon = 1e-10);
        assert_relative_eq!(new_center.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(new_center.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn arbitrary_axis_rotation() {
        // Rotation around (1,1,0)/sqrt(2) by 180°
        let angle: f64 = std::f64::consts::PI;
        let half = (angle / 2.0).sin();
        let cos_half = (angle / 2.0).cos();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let world_q = [cos_half, half * inv_sqrt2, half * inv_sqrt2, 0.0];
        let trans = [0.0, 0.0, 0.0];
        let scale = 1.0;

        // Camera at (1, 0, 0)
        let cam_q = [1.0, 0.0, 0.0, 0.0];
        let cam_t = [-1.0, 0.0, 0.0];

        let mut out_q = [0.0; 4];
        let mut out_t = [0.0; 3];

        apply_se3_to_camera_poses(
            world_q, trans, scale, &cam_q, &cam_t, &mut out_q, &mut out_t,
        );

        // 180° rotation around (1,1,0)/sqrt(2) maps (1,0,0) to (0,1,0)
        let new_center = camera_center(&out_q, &[out_t[0], out_t[1], out_t[2]]);
        assert_relative_eq!(new_center.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(new_center.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(new_center.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn compose_forward_then_inverse_is_identity() {
        // Apply a transform, then its inverse — should recover original pose
        let world_q = [0.9239, 0.0, 0.3827, 0.0]; // 45° around Y
        let world_trans = [5.0, -3.0, 2.0];
        let scale = 1.5;

        let cam_q = [1.0, 0.0, 0.0, 0.0];
        let cam_t = [-2.0, 1.0, -3.0];

        // Forward
        let mut fwd_q = [0.0; 4];
        let mut fwd_t = [0.0; 3];
        apply_se3_to_camera_poses(
            world_q,
            world_trans,
            scale,
            &cam_q,
            &cam_t,
            &mut fwd_q,
            &mut fwd_t,
        );

        // Compute inverse using Se3Transform
        let se3 = crate::se3_transform::Se3Transform::new(
            RotQuaternion::from_wxyz_array(world_q),
            Vector3::new(world_trans[0], world_trans[1], world_trans[2]),
            scale,
        );
        let inv = se3.inverse().unwrap();
        let inv_q_arr = inv.rotation.to_wxyz_array();
        let inv_q = [inv_q_arr[0], inv_q_arr[1], inv_q_arr[2], inv_q_arr[3]];
        let inv_t = [inv.translation.x, inv.translation.y, inv.translation.z];
        let inv_scale = inv.scale;

        // Inverse
        let mut back_q = [0.0; 4];
        let mut back_t = [0.0; 3];
        apply_se3_to_camera_poses(
            inv_q,
            inv_t,
            inv_scale,
            &fwd_q,
            &fwd_t,
            &mut back_q,
            &mut back_t,
        );

        // Should recover original
        let sign = if back_q[0] * cam_q[0] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for j in 0..4 {
            assert_relative_eq!(sign * back_q[j], cam_q[j], epsilon = 1e-8);
        }
        for j in 0..3 {
            assert_relative_eq!(back_t[j], cam_t[j], epsilon = 1e-8);
        }
    }

    #[test]
    fn batch_multiple_cameras() {
        let rot = [1.0, 0.0, 0.0, 0.0];
        let trans = [1.0, 2.0, 3.0];
        let scale = 2.0;

        // Two cameras
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

        apply_se3_to_camera_poses(rot, trans, scale, &cam_q, &cam_t, &mut out_q, &mut out_t);

        // Both cameras should have identity rotation (no world rotation applied)
        for cam_idx in 0..2 {
            assert_relative_eq!(out_q[cam_idx * 4], 1.0, epsilon = 1e-12);
            assert_relative_eq!(out_q[cam_idx * 4 + 1], 0.0, epsilon = 1e-12);
            assert_relative_eq!(out_q[cam_idx * 4 + 2], 0.0, epsilon = 1e-12);
            assert_relative_eq!(out_q[cam_idx * 4 + 3], 0.0, epsilon = 1e-12);
        }

        // Camera 0: center (0,0,0) -> scale*(0,0,0) + (1,2,3) = (1,2,3)
        // t_new = -R_new * C_new = -(1,2,3)
        let c0 = camera_center(
            &[out_q[0], out_q[1], out_q[2], out_q[3]],
            &[out_t[0], out_t[1], out_t[2]],
        );
        assert_relative_eq!(c0.x, 1.0, epsilon = 1e-12);
        assert_relative_eq!(c0.y, 2.0, epsilon = 1e-12);
        assert_relative_eq!(c0.z, 3.0, epsilon = 1e-12);

        // Camera 1: center (1,0,0) -> scale*(1,0,0) + (1,2,3) = (3,2,3)
        let c1 = camera_center(
            &[out_q[4], out_q[5], out_q[6], out_q[7]],
            &[out_t[3], out_t[4], out_t[5]],
        );
        assert_relative_eq!(c1.x, 3.0, epsilon = 1e-12);
        assert_relative_eq!(c1.y, 2.0, epsilon = 1e-12);
        assert_relative_eq!(c1.z, 3.0, epsilon = 1e-12);
    }
}
