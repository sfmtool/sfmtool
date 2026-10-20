// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Triangulation numerics: per-observation reprojection error and ray angle,
//! and whole-track diagnostics for one 3D point.
//!
//! Nothing here touches egui state: every function is a pure computation over
//! the point and image table handed to it, which is what lets the display code
//! that quotes them stay thin. A point arrives as a
//! [`PointView`](sfmtool_core::PointView), so the numbers describe the version
//! on screen rather than the base under it. They live at the crate root rather
//! than under a panel because three different surfaces read the same numbers:
//! Track View's header quotes them, the Image Detail overlay colours features
//! by them, and the MCP `get_point` tool reports them to an agent. A figure an
//! agent is told and a figure the human beside it reads off a panel have to be
//! the same figure, and that is easier to keep true when there is one
//! definition and no panel owns it.
//!
//! The whole-track numbers (the observation rays, the widest angle between
//! them, the condition number and the inverse-depth z) are point columns, so
//! their one definition is core's `analysis::point_columns`, re-exported here.

use nalgebra::Vector3;
use sfmtool_core::Point3D;

#[cfg(test)]
mod tests;

/// Compute per-observation reprojection error and ray angle for one observation.
///
/// Returns `(reproj_error_px, ray_angle_deg)`. Both are defined for a point at
/// infinity too: its stored direction rotates into camera space without
/// translating and then projects like any homogeneous coordinate. The point is
/// projected as a ray through the camera's own model, so a fisheye observation
/// more than 90° off the axis has an error like any other. If the camera model
/// has no pixel for the ray (a perspective model and a point behind the camera,
/// or a ray past a model's valid domain), returns `(NaN, NaN)`.
///
/// Read by the MCP surface's point track (`mcp::read::get_point`), which is
/// why it goes unused in a build without the `mcp` feature.
#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
pub(crate) fn compute_observation_metrics(
    point: &Point3D,
    image: &sfmtool_core::SfmrImage,
    camera: &sfmtool_core::CameraIntrinsics,
    feature_xy: [f32; 2],
) -> (f32, f32) {
    // Transform into camera space. A finite point takes the full rigid
    // transform `R * p + t`; a point at infinity is a pure direction, which
    // rotates but does not translate.
    let r = image.quaternion_wxyz.to_rotation_matrix();
    let p_cam = if point.is_at_infinity() {
        r * point.position.coords
    } else {
        r * point.position.coords + image.translation_xyz
    };

    // Project the ray, not the image-plane point `p / (-z)`: that division is
    // only meaningful in front of the camera, and a fisheye sees past 90° off
    // the axis, where `z >= 0`. The model decides which rays have a pixel.
    let Some(point_dir) = p_cam.try_normalize(0.0) else {
        return (f32::NAN, f32::NAN);
    };
    let Some((u_proj, v_proj)) = camera.ray_to_pixel([point_dir.x, point_dir.y, point_dir.z])
    else {
        return (f32::NAN, f32::NAN);
    };

    // Reprojection error in pixels
    let du = u_proj - feature_xy[0] as f64;
    let dv = v_proj - feature_xy[1] as f64;
    let reproj_error = (du * du + dv * dv).sqrt() as f32;

    // Ray angle: angle between the observation ray and the actual point direction
    // Both computed in camera space.
    let obs_ray = camera.pixel_to_ray(feature_xy[0] as f64, feature_xy[1] as f64);
    let obs_ray = Vector3::new(obs_ray[0], obs_ray[1], obs_ray[2]);

    let dot = obs_ray.dot(&point_dir).clamp(-1.0, 1.0);
    let ray_angle_deg = dot.acos().to_degrees() as f32;

    (reproj_error, ray_angle_deg)
}

// The whole-track numbers, defined in core as point columns so that the
// panels and the Python `point_columns` binding read one figure.
pub(crate) use sfmtool_core::analysis::point_columns::{
    compute_max_pairwise_angle, compute_max_ray_angle_deg, compute_point_diagnostics,
    compute_position_diagnostics, observation_rays,
};
