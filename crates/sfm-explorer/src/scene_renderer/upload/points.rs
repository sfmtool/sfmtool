// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point cloud instance buffer upload.

use super::super::auto_point_size::{
    compute_auto_point_size, compute_camera_nn_scale, compute_scene_bounds,
};
use super::super::gpu_types::PointInstance;
use super::super::SceneRenderer;
use crate::scene::ReconId;
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

impl SceneRenderer {
    /// Upload one reconstruction's point cloud into its resource bundle.
    ///
    /// Converts positions from f64 to f32, packs colors into u32, and
    /// computes the node's auto point size from nearest-neighbor distances.
    pub fn upload_points(
        &mut self,
        device: &wgpu::Device,
        id: ReconId,
        recon: &SfmrReconstruction,
    ) {
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

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        self.ensure_recon(device, id);
        let bundle = self.recons.get_mut(&id).expect("just ensured");

        // Auto point size, inter-camera distance and bounding sphere are all
        // per-recon: they describe this node's data, not the scene.
        bundle.auto_point_size = compute_auto_point_size(&recon.points);
        bundle.camera_nn_scale = compute_camera_nn_scale(&recon.images);
        bundle.bounds = Some(compute_scene_bounds(&recon.points));
        bundle.point_instance_buffer = Some(buffer);
        bundle.point_count = instances.len() as u32;

        let (count, size) = (bundle.point_count, bundle.auto_point_size);
        // The point count moved, so the global pick index space has to be
        // re-cut. Only the per-recon uniform blocks change; no instance buffer
        // carries a base.
        self.assign_pick_bases();

        log::info!("Uploaded {count} points to GPU (auto point size: {size:.4})");
    }
}
