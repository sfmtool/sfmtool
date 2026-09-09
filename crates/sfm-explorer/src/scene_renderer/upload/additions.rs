// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The overlay's **additions** on the GPU: a second set of instance buffers,
//! drawn after the base's in the same passes.
//!
//! See `specs/gui/document-model.md`, "Change detection by identity". The base's
//! buffers are what a run of point edits shares, so a point the overlay adds
//! cannot go into them without rebuilding a million instances. It goes into its
//! own pair of buffers instead, and its patch into its own small atlas, both
//! rebuilt only when the addition set moves.

use wgpu::util::DeviceExt;

use super::super::gpu_types::PointInstance;
use super::super::recon::AdditionResources;
use super::super::SceneRenderer;
use crate::scene::ReconId;
use sfmtool_core::EditedReconstruction;

impl SceneRenderer {
    /// Whether `id`'s addition buffers were built from a different addition set.
    ///
    /// The signal is the set's `(point count, observation count)`, which is
    /// complete because additions are append-only along a node's history and
    /// that history is linear: a new edit discards the redo tail, so two
    /// versions of one base whose addition sets agree on both counts hold the
    /// same additions.
    pub fn additions_changed(&self, id: ReconId, edited: &EditedReconstruction) -> bool {
        let signature = addition_signature(edited);
        match self.recons.get(&id) {
            Some(bundle) => bundle.uploaded_additions != signature,
            None => signature != (0, 0),
        }
    }

    /// Rebuild `id`'s addition buffers from `edited`'s addition set.
    ///
    /// Leaves the base's buffers, its atlas and its deleted mask untouched: this
    /// is the half of the node's GPU state the overlay owns. A version with no
    /// additions clears them.
    pub fn upload_additions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ReconId,
        edited: &EditedReconstruction,
    ) {
        self.ensure_recon(device, id);
        let signature = addition_signature(edited);
        {
            let bundle = self.recons.get_mut(&id).expect("just ensured");
            bundle.additions = None;
            bundle.uploaded_additions = signature;
        }
        let added = &edited.added;
        if added.points.is_empty() {
            // The pick ranges are cut from the instance counts, and one just
            // went to zero.
            self.assign_pick_bases();
            return;
        }

        let instances: Vec<PointInstance> = added
            .points
            .iter()
            .map(|p| {
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
        let point_instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("addition point instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // An addition that has itself been deleted or replaced keeps its row so
        // the indexes after it do not move, so the mask starts from the set
        // rather than all-alive as the base's does.
        let base_count = edited.base_point_count() as u32;
        let alive: Vec<u32> = (0..instances.len() as u32)
            .map(|k| u32::from(!edited.is_deleted(base_count + k)))
            .collect();
        let point_alive_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("addition point liveness"),
            contents: bytemuck::cast_slice(&alive),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // The additions' surfels index the base's point set nowhere, so their
        // atlas is their own; `index_offset` makes each instance carry its
        // *edited* index, which is what the pick id and the mask are keyed on.
        let patch = self.build_patch_resources(device, queue, id, added, base_count);

        // The additions' own `ReconUniforms`, which differ from the node's in
        // one field: the pick base is shifted past the base's instances, so an
        // addition's global pick id is still `point_pick_base + edited index`.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("addition recon uniforms"),
            size: std::mem::size_of::<super::super::gpu_types::ReconUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (Some(layout), Some(global)) = (
            self.point_bind_group_layout.as_ref(),
            self.point_uniform_buffer.as_ref(),
        ) else {
            return;
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("addition point bind group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            layout,
        });

        let bundle = self.recons.get_mut(&id).expect("just ensured");
        bundle.additions = Some(AdditionResources {
            uniform_buffer,
            bind_group,
            point_instance_buffer,
            point_alive_buffer,
            point_count: instances.len() as u32,
            patch,
        });
        // The node's pick range grew by the additions, so the global index
        // space has to be re-cut.
        self.assign_pick_bases();
    }
}

/// What identifies an addition set for change detection.
fn addition_signature(edited: &EditedReconstruction) -> (usize, usize) {
    (edited.added.points.len(), edited.added.tracks.len())
}
