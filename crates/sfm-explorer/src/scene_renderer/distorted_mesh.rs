// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use sfmtool_core::camera_intrinsics::CameraIntrinsics;

use super::gpu_types::BgDistortedVertex;

/// Generate a tessellated mesh for the distorted background image.
///
/// Vertex positions are ray directions transformed to world space via the
/// provided camera-to-world rotation matrix `r_world_from_cam` (row-major 3×3).
/// This matches the coordinate convention used by frustum wireframes and image
/// quads, allowing the BG shader to use the same `view_proj = projection * view`
/// transform pipeline.
///
/// Rays are computed via [`CameraIntrinsics::pixel_to_ray`], which works
/// correctly for fisheye cameras with field of view at and beyond 180°.
/// Beyond ~90° from the optical axis, the ray blends smoothly to the
/// undistorted (identity) model, so all rays are finite and well-behaved.
///
/// The mesh depends on camera intrinsics and the camera's orientation, so it
/// is rebuilt when switching cameras.
pub(super) fn generate_bg_distorted_mesh(
    camera: &CameraIntrinsics,
    r_world_from_cam: &[f64; 9],
    subdivisions: usize,
) -> (Vec<BgDistortedVertex>, Vec<u32>) {
    let n = subdivisions + 1;
    let w = camera.width as f64;
    let h = camera.height as f64;

    let mut vertices = Vec::with_capacity(n * n);
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);

    for j in 0..n {
        for i in 0..n {
            let s = i as f64 / (n - 1) as f64;
            let t = j as f64 / (n - 1) as f64;
            let ray = camera.pixel_to_ray(s * w, t * h);

            // Transform ray direction from camera space to world space
            let rx = r_world_from_cam[0] * ray[0]
                + r_world_from_cam[1] * ray[1]
                + r_world_from_cam[2] * ray[2];
            let ry = r_world_from_cam[3] * ray[0]
                + r_world_from_cam[4] * ray[1]
                + r_world_from_cam[5] * ray[2];
            let rz = r_world_from_cam[6] * ray[0]
                + r_world_from_cam[7] * ray[1]
                + r_world_from_cam[8] * ray[2];

            vertices.push(BgDistortedVertex {
                position: [rx as f32, ry as f32, rz as f32],
                uv: [s as f32, t as f32],
            });
        }
    }

    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let tl = (j * n + i) as u32;
            let tr = tl + 1;
            let bl = ((j + 1) * n + i) as u32;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr]);
            indices.extend_from_slice(&[tr, bl, br]);
        }
    }

    (vertices, indices)
}
