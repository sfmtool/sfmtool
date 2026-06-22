// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use crate::camera::intrinsics::CameraIntrinsics;
use nalgebra::{Matrix3, Vector3};
use rand::RngExt;

/// Compute 8 corners of a camera frustum in world coordinates.
///
/// # Arguments
/// * `camera_center` - Camera center in world coordinates
/// * `r_world_from_cam` - Row-major 3x3 rotation matrix (camera to world)
/// * `fx`, `fy` - Focal lengths
/// * `cx`, `cy` - Principal point
/// * `width`, `height` - Image dimensions in pixels
/// * `near_z`, `far_z` - Near and far plane Z distances
///
/// # Returns
/// `[f64; 24]` containing 8 corners x 3 coordinates.
/// Corner order: near (TL, TR, BR, BL), far (TL, TR, BR, BL).
#[allow(clippy::too_many_arguments)]
pub fn compute_frustum_corners(
    camera_center: &[f64; 3],
    r_world_from_cam: &[f64; 9],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    width: u32,
    height: u32,
    near_z: f64,
    far_z: f64,
) -> [f64; 24] {
    let center = Vector3::new(camera_center[0], camera_center[1], camera_center[2]);

    // nalgebra Matrix3::new takes arguments in row-major order
    let r = Matrix3::new(
        r_world_from_cam[0],
        r_world_from_cam[1],
        r_world_from_cam[2],
        r_world_from_cam[3],
        r_world_from_cam[4],
        r_world_from_cam[5],
        r_world_from_cam[6],
        r_world_from_cam[7],
        r_world_from_cam[8],
    );

    let w = width as f64;
    let h = height as f64;

    // Image corners in pixel coordinates: TL, TR, BR, BL
    let pixel_corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];

    let mut corners = [0.0_f64; 24];

    for (i, &(px, py)) in pixel_corners.iter().enumerate() {
        let x_norm = (px - cx) / fx;
        let y_norm = (py - cy) / fy;

        let dir_cam = Vector3::new(x_norm, y_norm, 1.0).normalize();

        let t_near = near_z / dir_cam[2];
        let t_far = far_z / dir_cam[2];

        let point_near_cam = dir_cam * t_near;
        let point_far_cam = dir_cam * t_far;

        let corner_near = center + r * point_near_cam;
        let corner_far = center + r * point_far_cam;

        // Near corners at indices 0..4, far corners at indices 4..8
        corners[i * 3] = corner_near[0];
        corners[i * 3 + 1] = corner_near[1];
        corners[i * 3 + 2] = corner_near[2];

        corners[(i + 4) * 3] = corner_far[0];
        corners[(i + 4) * 3 + 1] = corner_far[1];
        corners[(i + 4) * 3 + 2] = corner_far[2];
    }

    corners
}

/// Compute 6 inward-facing plane equations from frustum corners and camera center.
///
/// # Arguments
/// * `camera_center` - Camera center in world coordinates
/// * `corners` - 8 corners x 3 coordinates as returned by [`compute_frustum_corners`]
///
/// # Returns
/// `[f64; 24]` containing 6 planes x 4 coefficients `[nx, ny, nz, d]`.
/// Convention: `dot(normal, point) + d >= 0` means inside.
/// Plane order: near, far, left, right, top, bottom.
pub fn compute_frustum_planes(_camera_center: &[f64; 3], corners: &[f64; 24]) -> [f64; 24] {
    // Extract corners as Vector3
    let c = |idx: usize| -> Vector3<f64> {
        Vector3::new(corners[idx * 3], corners[idx * 3 + 1], corners[idx * 3 + 2])
    };

    // Near: 0=ntl, 1=ntr, 2=nbr, 3=nbl
    // Far:  4=ftl, 5=ftr, 6=fbr, 7=fbl
    let ntl = c(0);
    let ntr = c(1);
    let nbr = c(2);
    let nbl = c(3);
    let ftl = c(4);
    let ftr = c(5);
    let _fbr = c(6);
    let fbl = c(7);

    // Plane definitions: (p0, p1, p2, inside_reference)
    #[allow(clippy::type_complexity)]
    let plane_defs: [(Vector3<f64>, Vector3<f64>, Vector3<f64>, Vector3<f64>); 6] = [
        (ntl, ntr, nbr, ftr),  // near
        (ftl, c(6), ftr, ntr), // far  (ftl, fbr, ftr)
        (ntl, nbl, fbl, ntr),  // left
        (ntr, ftr, c(6), ntl), // right (ntr, ftr, fbr)
        (ntl, ftr, ntr, nbl),  // top
        (nbl, nbr, c(6), ntl), // bottom (nbl, nbr, fbr)
    ];

    let mut planes = [0.0_f64; 24];

    for (i, (p0, p1, p2, ref_inside)) in plane_defs.iter().enumerate() {
        let v1 = p1 - p0;
        let v2 = p2 - p0;
        let mut normal = v1.cross(&v2).normalize();
        let mut d = -normal.dot(p0);

        // Ensure normal points inward
        if normal.dot(ref_inside) + d < 0.0 {
            normal = -normal;
            d = -d;
        }

        planes[i * 4] = normal[0];
        planes[i * 4 + 1] = normal[1];
        planes[i * 4 + 2] = normal[2];
        planes[i * 4 + 3] = d;
    }

    planes
}

/// Compute frustum volume using the truncated pyramid formula.
///
/// `V = (h/3) * (A1 + A2 + sqrt(A1 * A2))`
///
/// where `h = far_z - near_z`, and `A1`, `A2` are the near and far cross-section areas.
pub fn compute_frustum_volume(
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    near_z: f64,
    far_z: f64,
) -> f64 {
    let w = width as f64;
    let h = height as f64;

    let near_width = (w / fx) * near_z;
    let near_height = (h / fy) * near_z;
    let far_width = (w / fx) * far_z;
    let far_height = (h / fy) * far_z;

    let near_area = near_width * near_height;
    let far_area = far_width * far_height;

    let depth = far_z - near_z;
    (depth / 3.0) * (near_area + far_area + (near_area * far_area).sqrt())
}

/// Test which points are inside a frustum defined by 6 planes.
///
/// # Arguments
/// * `points` - Flat slice of N*3 coordinates
/// * `num_points` - Number of points
/// * `planes` - 6 planes x 4 coefficients `[nx, ny, nz, d]`
///
/// # Returns
/// `Vec<bool>` of length N. `true` if the point is inside all 6 planes.
pub fn points_in_frustum(points: &[f64], num_points: usize, planes: &[f64; 24]) -> Vec<bool> {
    debug_assert_eq!(points.len(), num_points * 3);

    let tolerance = -1e-10;
    let mut result = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let px = points[i * 3];
        let py = points[i * 3 + 1];
        let pz = points[i * 3 + 2];

        let mut inside = true;
        for j in 0..6 {
            let nx = planes[j * 4];
            let ny = planes[j * 4 + 1];
            let nz = planes[j * 4 + 2];
            let d = planes[j * 4 + 3];

            let dist = nx * px + ny * py + nz * pz + d;
            if dist < tolerance {
                inside = false;
                break;
            }
        }
        result.push(inside);
    }

    result
}

/// Fast separating plane test for frustum-frustum intersection.
///
/// Conservative: may return `true` for non-intersecting frustums,
/// but never `false` for intersecting ones.
pub fn frustums_can_intersect(
    corners_a: &[f64; 24],
    planes_a: &[f64; 24],
    corners_b: &[f64; 24],
    planes_b: &[f64; 24],
) -> bool {
    // Check if all corners of B are outside any single plane of A
    for p in 0..6 {
        let nx = planes_a[p * 4];
        let ny = planes_a[p * 4 + 1];
        let nz = planes_a[p * 4 + 2];
        let d = planes_a[p * 4 + 3];

        let mut all_outside = true;
        for c in 0..8 {
            let dist =
                nx * corners_b[c * 3] + ny * corners_b[c * 3 + 1] + nz * corners_b[c * 3 + 2] + d;
            if dist >= 0.0 {
                all_outside = false;
                break;
            }
        }
        if all_outside {
            return false;
        }
    }

    // Check if all corners of A are outside any single plane of B
    for p in 0..6 {
        let nx = planes_b[p * 4];
        let ny = planes_b[p * 4 + 1];
        let nz = planes_b[p * 4 + 2];
        let d = planes_b[p * 4 + 3];

        let mut all_outside = true;
        for c in 0..8 {
            let dist =
                nx * corners_a[c * 3] + ny * corners_a[c * 3 + 1] + nz * corners_a[c * 3 + 2] + d;
            if dist >= 0.0 {
                all_outside = false;
                break;
            }
        }
        if all_outside {
            return false;
        }
    }

    true
}

/// Rejection sample uniform random points inside a frustum.
///
/// Computes the AABB of the frustum corners, oversamples 4x, and filters
/// via [`points_in_frustum`]. Returns up to `num_samples` points.
///
/// # Returns
/// Flat `Vec<f64>` of M*3 values where M <= `num_samples`.
pub fn sample_points_in_frustum(
    corners: &[f64; 24],
    planes: &[f64; 24],
    num_samples: usize,
    rng: &mut impl rand::Rng,
) -> Vec<f64> {
    // Compute AABB
    let mut min_coords = [f64::INFINITY; 3];
    let mut max_coords = [f64::NEG_INFINITY; 3];
    for i in 0..8 {
        for d in 0..3 {
            let v = corners[i * 3 + d];
            if v < min_coords[d] {
                min_coords[d] = v;
            }
            if v > max_coords[d] {
                max_coords[d] = v;
            }
        }
    }

    let oversample = 4;
    let candidate_count = num_samples * oversample;

    // Generate random points in AABB
    let mut candidates = Vec::with_capacity(candidate_count * 3);
    for _ in 0..candidate_count {
        for d in 0..3 {
            candidates.push(rng.random_range(min_coords[d]..=max_coords[d]));
        }
    }

    // Filter to points inside frustum
    let inside_mask = points_in_frustum(&candidates, candidate_count, planes);

    let mut result = Vec::new();
    let mut count = 0;
    for (i, &is_inside) in inside_mask.iter().enumerate() {
        if is_inside {
            result.push(candidates[i * 3]);
            result.push(candidates[i * 3 + 1]);
            result.push(candidates[i * 3 + 2]);
            count += 1;
            if count >= num_samples {
                break;
            }
        }
    }

    result
}

/// Monte Carlo estimate of the intersection volume between two frustums.
///
/// Samples points uniformly in frustum A and counts how many are also inside
/// frustum B. The intersection volume is estimated as `fraction * volume_a`.
pub fn estimate_frustum_intersection_volume(
    corners_a: &[f64; 24],
    planes_a: &[f64; 24],
    volume_a: f64,
    planes_b: &[f64; 24],
    num_samples: usize,
    rng: &mut impl rand::Rng,
) -> f64 {
    let points_in_a = sample_points_in_frustum(corners_a, planes_a, num_samples, rng);

    let n_points = points_in_a.len() / 3;
    if n_points == 0 {
        return 0.0;
    }

    let in_both = points_in_frustum(&points_in_a, n_points, planes_b);
    let count_in_both = in_both.iter().filter(|&&v| v).count();

    let fraction = count_in_both as f64 / n_points as f64;
    fraction * volume_a
}

/// Result of computing a tessellated frustum grid for distorted cameras.
///
/// Contains world-space vertex positions in a row-major N×N grid, where
/// N = `grid_size`. Vertex `(i, j)` is at index `j * grid_size + i` in
/// `positions` (i.e., `positions[(j * grid_size + i) * 3..]`).
pub struct DistortedFrustumGrid {
    /// World-space positions, row-major: `positions[(j * N + i) * 3..]` = vertex at (i, j).
    /// Length: `grid_size * grid_size * 3`.
    pub positions: Vec<f64>,
    /// Grid dimension (N = subdivisions + 1).
    pub grid_size: usize,
}

/// Compute a tessellated grid of world-space positions for a camera's
/// far-plane image quad, accounting for lens distortion.
///
/// Each grid vertex `(i, j)` maps to a pixel position on the image boundary
/// and interior, is unprojected through the camera's distortion model to get
/// the true ray direction, then placed at the frustum's far plane distance.
///
/// For pinhole cameras (no distortion), the result is a flat quad identical to
/// `compute_frustum_corners()`.
///
/// # Arguments
/// * `camera_center` - Camera center in world coordinates
/// * `r_world_from_cam` - Row-major 3×3 rotation matrix (camera to world)
/// * `camera` - Camera intrinsics (includes model, width, height)
/// * `far_z` - Far plane Z distance
/// * `subdivisions` - Number of subdivisions per edge (N-1); grid has N×N vertices
pub fn compute_distorted_frustum_grid(
    camera_center: &[f64; 3],
    r_world_from_cam: &[f64; 9],
    camera: &CameraIntrinsics,
    far_z: f64,
    subdivisions: usize,
) -> DistortedFrustumGrid {
    let n = subdivisions + 1;
    let center = Vector3::new(camera_center[0], camera_center[1], camera_center[2]);
    let r = Matrix3::new(
        r_world_from_cam[0],
        r_world_from_cam[1],
        r_world_from_cam[2],
        r_world_from_cam[3],
        r_world_from_cam[4],
        r_world_from_cam[5],
        r_world_from_cam[6],
        r_world_from_cam[7],
        r_world_from_cam[8],
    );

    let w = camera.width as f64;
    let h = camera.height as f64;

    let mut positions = Vec::with_capacity(n * n * 3);

    for j in 0..n {
        for i in 0..n {
            // Pixel coordinates for this grid vertex
            let u = (i as f64 / (n - 1) as f64) * w;
            let v = (j as f64 / (n - 1) as f64) * h;

            // Get unit ray direction (works correctly for fisheye beyond 180°)
            let ray = camera.pixel_to_ray(u, v);
            let dir = Vector3::new(ray[0], ray[1], ray[2]);

            // Camera-space point on far surface
            let p_cam = if camera.model.is_fisheye() {
                // Spherical: place at distance far_z along ray
                dir * far_z
            } else {
                // Flat: perspective projection, intersect ray with z = far_z plane
                dir * (far_z / dir[2])
            };

            // Transform to world space
            let p_world = center + r * p_cam;

            positions.push(p_world[0]);
            positions.push(p_world[1]);
            positions.push(p_world[2]);
        }
    }

    DistortedFrustumGrid {
        positions,
        grid_size: n,
    }
}

#[cfg(test)]
mod tests;
