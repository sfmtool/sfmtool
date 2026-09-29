// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Triangulation numerics: per-observation reprojection error and ray angle,
//! and whole-track diagnostics for one 3D point.
//!
//! Nothing here touches egui state — every function is a pure computation over
//! the point and image table handed to it, which is what lets the display code
//! that quotes them stay thin. A point arrives as a [`PointView`], so the
//! numbers describe the version on screen rather than the base under it.
//! They live at the crate root rather than under a panel
//! because three different surfaces read the same numbers: Track View's
//! header quotes them, the Image Detail overlay colours features by
//! them, and the MCP `get_point` tool reports them to an agent. A figure an
//! agent is told and a figure the human beside it reads off a panel have to be
//! the same figure, and that is easier to keep true when there is one
//! definition and no panel owns it.

use nalgebra::{Point3, Vector3};
use sfmtool_core::{ImageTable, Point3D, PointView};

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

/// Triangulation observability diagnostics for a 3D point, computed from the
/// rays from each observing camera to the *stored* point (no `.sift` reads):
/// `(condition_number, inverse_depth_z)`. Returns `(NaN, NaN)` for points at
/// infinity, missing points, or fewer than two usable rays. The per-ray angular
/// noise is `max(reproj_error, 1px) / f`, matching the classifier's policy.
pub(crate) fn compute_point_diagnostics(
    image_table: &ImageTable,
    view: &PointView<'_>,
) -> (f32, f32) {
    let pt = view.point();
    if pt.is_at_infinity() {
        return (f32::NAN, f32::NAN);
    }
    let images = view
        .observations()
        .iter()
        .map(|obs| obs.image_index as usize);
    compute_position_diagnostics(image_table, &pt.position, f64::from(pt.error), images)
}

/// [`compute_point_diagnostics`] for a finite `position` seen from `images`,
/// with `error_px` the reprojection error the per-ray noise is scaled by
/// (floored at one pixel): what Track View's header reads a bench track's own
/// position and kept observations by, where there is no stored point to hand
/// in. Returns `(NaN, NaN)` with fewer than two usable rays.
pub(crate) fn compute_position_diagnostics(
    image_table: &ImageTable,
    position: &Point3<f64>,
    error_px: f64,
    images: impl IntoIterator<Item = usize>,
) -> (f32, f32) {
    use sfmtool_core::reconstruction::triangulation::{depth_uncertainty_batch, triangulate_batch};

    let noise = if error_px.is_finite() {
        error_px.max(1.0)
    } else {
        1.0
    };
    let mut dirs = Vec::new();
    let mut centers = Vec::new();
    let mut sigma = Vec::new();
    for img_idx in images {
        let Some(image) = image_table.images.get(img_idx) else {
            continue;
        };
        let center = image.camera_center();
        let dir = position - center;
        let len = dir.norm();
        if len > 1e-12 {
            dirs.push(dir / len);
            centers.push(center);
            let (fx, fy) = image_table.cameras[image.camera_index as usize].focal_lengths();
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

/// The unit rays from each of `images`' camera centres to `position`, in
/// world space, which [`compute_max_pairwise_angle`] reads. A point at
/// infinity is seen along the same stored direction from every camera, so its
/// rays are that direction each time and the angle between them is zero.
pub(crate) fn observation_rays(
    image_table: &ImageTable,
    position: &Point3<f64>,
    at_infinity: bool,
    images: impl IntoIterator<Item = usize>,
) -> Vec<[f64; 3]> {
    images
        .into_iter()
        .filter_map(|img_idx| {
            if at_infinity {
                let d = position.coords;
                return Some([d.x, d.y, d.z]);
            }
            let image = image_table.images.get(img_idx)?;
            let dir = position - image.camera_center();
            let len = dir.norm();
            (len > 1e-12).then(|| [dir.x / len, dir.y / len, dir.z / len])
        })
        .collect()
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
