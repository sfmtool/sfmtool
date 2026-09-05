// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Triangulation numerics: per-observation reprojection error and ray angle,
//! and whole-track diagnostics for one 3D point.
//!
//! Nothing here touches egui state — every function is a pure computation over
//! a [`SfmrReconstruction`], which is what lets the display code that quotes
//! them stay thin. They live at the crate root rather than under a panel
//! because three different surfaces read the same numbers: the Point Track
//! Detail table tabulates them, the Image Detail overlay colours features by
//! them, and the MCP `get_point` tool reports them to an agent. A figure an
//! agent is told and a figure the human beside it reads off a panel have to be
//! the same figure, and that is easier to keep true when there is one
//! definition and no panel owns it.

use nalgebra::Vector3;
use sfmtool_core::SfmrReconstruction;

#[cfg(test)]
mod tests;

/// Compute per-observation reprojection error and ray angle for one observation.
///
/// Returns `(reproj_error_px, ray_angle_deg)`. If the point is behind the
/// camera, returns `(NaN, NaN)`.
///
/// Crate-visible because the MCP surface reports the same number in a point
/// track (`mcp::read::get_point`), and an agent told one figure while the human
/// beside it reads another off this panel is the failure that boundary exists
/// to avoid.
pub(crate) fn compute_observation_metrics(
    point_pos: &nalgebra::Point3<f64>,
    image: &sfmtool_core::SfmrImage,
    camera: &sfmtool_core::CameraIntrinsics,
    feature_xy: [f32; 2],
) -> (f32, f32) {
    // Transform point from world to camera space: p_cam = R * p_world + t
    let r = image.quaternion_wxyz.to_rotation_matrix();
    let p_cam = r * point_pos.coords + image.translation_xyz;

    // Canonical cameras look down -Z, so in-front points have z < 0 and depth
    // is -z. Point behind camera — return NaN to signal invalid.
    let depth = -p_cam.z;
    if depth <= 0.0 {
        return (f32::NAN, f32::NAN);
    }

    // Project to image plane (undistorted normalized canonical coords, p/(-z))
    let x = p_cam.x / depth;
    let y = p_cam.y / depth;

    // Apply distortion + intrinsics to get pixel coordinates
    let (u_proj, v_proj) = camera.project(x, y);

    // Reprojection error in pixels
    let du = u_proj - feature_xy[0] as f64;
    let dv = v_proj - feature_xy[1] as f64;
    let reproj_error = (du * du + dv * dv).sqrt() as f32;

    // Ray angle: angle between the observation ray and the actual point direction
    // Both computed in camera space.
    let obs_ray = camera.pixel_to_ray(feature_xy[0] as f64, feature_xy[1] as f64);
    let obs_ray = Vector3::new(obs_ray[0], obs_ray[1], obs_ray[2]);

    let point_dir = p_cam.normalize();

    let dot = obs_ray.dot(&point_dir).clamp(-1.0, 1.0);
    let ray_angle_deg = dot.acos().to_degrees() as f32;

    (reproj_error, ray_angle_deg)
}

/// Triangulation observability diagnostics for a 3D point, computed from the
/// rays from each observing camera to the *stored* point (no `.sift` reads):
/// `(condition_number, inverse_depth_z)`. Returns `(NaN, NaN)` for points at
/// infinity, missing points, or fewer than two usable rays. The per-ray angular
/// noise is `max(reproj_error, 1px) / f`, matching the classifier's policy.
pub(crate) fn compute_point_diagnostics(
    recon: &SfmrReconstruction,
    point_idx: usize,
) -> (f32, f32) {
    use sfmtool_core::reconstruction::triangulation::{depth_uncertainty_batch, triangulate_batch};

    let Some(pt) = recon.points.get(point_idx) else {
        return (f32::NAN, f32::NAN);
    };
    if pt.is_at_infinity() {
        return (f32::NAN, f32::NAN);
    }
    let observations = recon.observations_for_point(point_idx);
    let noise = (pt.error as f64).max(1.0);
    let mut dirs = Vec::with_capacity(observations.len());
    let mut centers = Vec::with_capacity(observations.len());
    let mut sigma = Vec::with_capacity(observations.len());
    for obs in observations {
        let img_idx = obs.image_index as usize;
        let Some(image) = recon.images.get(img_idx) else {
            continue;
        };
        let center = image.camera_center();
        let dir = pt.position - center;
        let len = dir.norm();
        if len > 1e-12 {
            dirs.push(dir / len);
            centers.push(center);
            let (fx, fy) = recon.cameras[image.camera_index as usize].focal_lengths();
            sigma.push(noise / fx.max(fy));
        }
    }
    if dirs.len() < 2 {
        return (f32::NAN, f32::NAN);
    }
    let offsets = [0usize, dirs.len()];
    let tris = triangulate_batch(&dirs, &centers, &offsets);
    let dus = depth_uncertainty_batch(&tris, &dirs, &centers, &offsets, &sigma);
    (
        tris[0].condition_number as f32,
        dus[0].inverse_depth_z as f32,
    )
}

/// Compute the maximum angle (in degrees) between any pair of world-space rays.
pub(crate) fn compute_max_pairwise_angle(rays: &[[f64; 3]]) -> f32 {
    let mut min_dot = 1.0f64;
    for i in 0..rays.len() {
        for j in (i + 1)..rays.len() {
            let dot = rays[i][0] * rays[j][0] + rays[i][1] * rays[j][1] + rays[i][2] * rays[j][2];
            if dot < min_dot {
                min_dot = dot;
            }
        }
    }
    min_dot.clamp(-1.0, 1.0).acos().to_degrees() as f32
}
