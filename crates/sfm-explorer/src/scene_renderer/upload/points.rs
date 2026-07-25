// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point cloud instance buffer upload.

use super::super::auto_point_size::{
    compute_auto_point_size, compute_camera_nn_scale, compute_scene_bounds,
};
use super::super::gpu_types::PointInstance;
use super::super::SceneRenderer;
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

impl SceneRenderer {
    /// Upload point cloud data to the GPU.
    ///
    /// Converts positions from f64 to f32, packs colors into u32, and
    /// computes the auto point size from nearest-neighbor distances.
    pub fn upload_points(&mut self, device: &wgpu::Device, recon: &SfmrReconstruction) {
        let instances: Vec<PointInstance> = recon
            .points
            .iter()
            .map(|p| {
                // For an infinity point `position` holds a unit direction; the
                // shader detects it via alpha = 0 and transforms it with w = 0.
                let alpha: u32 = if p.is_at_infinity() { 0 } else { 255 };
                PointInstance {
                    position: [
                        p.position.x as f32,
                        p.position.y as f32,
                        p.position.z as f32,
                    ],
                    color: (p.color[0] as u32)
                        | ((p.color[1] as u32) << 8)
                        | ((p.color[2] as u32) << 16)
                        | (alpha << 24),
                }
            })
            .collect();

        // Compute auto point size from nearest-neighbor distances
        self.auto_point_size = compute_auto_point_size(&recon.points);

        // Compute characteristic inter-camera distance
        self.camera_nn_scale = compute_camera_nn_scale(&recon.images);

        // Compute scene bounding sphere for adaptive clip planes
        let (center, radius) = compute_scene_bounds(&recon.points);
        self.scene_center = center;
        self.scene_radius = radius;

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        self.instance_buffer = Some(buffer);
        self.point_count = instances.len() as u32;

        log::info!(
            "Uploaded {} points to GPU (auto point size: {:.4})",
            self.point_count,
            self.auto_point_size
        );
    }
}
