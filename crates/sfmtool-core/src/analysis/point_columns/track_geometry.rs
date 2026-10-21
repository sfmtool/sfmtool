// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The triangulation numbers of one track read from its stored point: the
//! rays from each observing camera to it, the widest angle between two of
//! them, and the condition number and inverse-depth z of its triangulation.
//!
//! These are the definitions of the `max_ray_angle_deg`, `condition_number`
//! and `inverse_depth_z` point columns, and the viewer's Track View header and
//! Image Detail heat maps call them too, so every surface reads one number
//! for one point. The bench's nearby-track finder
//! measures its candidates' ray angle with [`compute_max_pairwise_angle`] as
//! well.

use nalgebra::Point3;

use crate::reconstruction::triangulation::{depth_uncertainty_batch, triangulate_batch};
use crate::reconstruction::{ImageTable, PointView};

/// The unit rays from each of `images`' camera centres to `position`, in
/// world space, which [`compute_max_pairwise_angle`] reads.
///
/// A point at infinity is seen along the same stored direction from every
/// camera, so its rays are that direction each time and the angle between
/// them is zero. An image index past the table, or a camera centre on the
/// point itself, gives no ray.
pub fn observation_rays(
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

/// The largest angle between any two of `rays`, in degrees: `0` for fewer
/// than two rays. The rays are unit vectors; a pair whose dot product is not
/// a number is skipped.
pub fn compute_max_pairwise_angle(rays: &[[f64; 3]]) -> f64 {
    let mut min_dot = 1.0f64;
    for i in 0..rays.len() {
        for j in (i + 1)..rays.len() {
            let dot = rays[i][0] * rays[j][0] + rays[i][1] * rays[j][1] + rays[i][2] * rays[j][2];
            if dot < min_dot {
                min_dot = dot;
            }
        }
    }
    min_dot.clamp(-1.0, 1.0).acos().to_degrees()
}

/// The largest angle between two of the viewing rays of the point `view`, in
/// degrees: the `max_ray_angle_deg` point column.
///
/// **A point at infinity is a bearing and every sighting of it casts the same
/// ray**, so its widest pair is zero degrees whatever the baseline. The three
/// stored numbers are that direction rather than a place, and subtracting a
/// camera centre from them would measure the spread of rays to a point one
/// unit from the world origin. A track with fewer than two usable rays is `0`.
pub fn compute_max_ray_angle_deg(image_table: &ImageTable, view: &PointView<'_>) -> f32 {
    let point = view.point();
    if point.is_at_infinity() {
        return 0.0;
    }
    let images = view
        .observations()
        .iter()
        .map(|obs| obs.image_index as usize);
    let rays = observation_rays(image_table, &point.position, false, images);
    compute_max_pairwise_angle(&rays) as f32
}

/// Triangulation observability diagnostics for the point `view`, computed
/// from the rays from each observing camera to the *stored* point (no
/// `.sift` reads): `(condition_number, inverse_depth_z)`.
///
/// Returns `(NaN, NaN)` for a point at infinity, which has no depth to be
/// uncertain about, and for fewer than two usable rays. The per-ray angular
/// noise is `max(error, 1 px) / f`, with `error` the point's stored
/// reprojection error ([`Point3D::rms_reprojection_error`](crate::reconstruction::Point3D::rms_reprojection_error)).
pub fn compute_point_diagnostics(image_table: &ImageTable, view: &PointView<'_>) -> (f32, f32) {
    let pt = view.point();
    if pt.is_at_infinity() {
        return (f32::NAN, f32::NAN);
    }
    let images = view
        .observations()
        .iter()
        .map(|obs| obs.image_index as usize);
    compute_position_diagnostics(
        image_table,
        &pt.position,
        f64::from(pt.rms_reprojection_error),
        images,
    )
}

/// [`compute_point_diagnostics`] for a finite `position` seen from `images`,
/// with `error_px` the reprojection error the per-ray noise is scaled by
/// (floored at one pixel, and one pixel where it is not a number): for a
/// track that has a position and observations but no stored point, such as a
/// bench track. Returns `(NaN, NaN)` with fewer than two usable rays.
pub fn compute_position_diagnostics(
    image_table: &ImageTable,
    position: &Point3<f64>,
    error_px: f64,
    images: impl IntoIterator<Item = usize>,
) -> (f32, f32) {
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
