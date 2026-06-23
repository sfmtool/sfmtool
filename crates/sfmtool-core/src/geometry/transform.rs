// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! SE3 similarity transform applied to batched camera poses.

use nalgebra::Vector3;

use crate::geometry::RotQuaternion;
use crate::geometry::Se3Transform;

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
mod tests;
