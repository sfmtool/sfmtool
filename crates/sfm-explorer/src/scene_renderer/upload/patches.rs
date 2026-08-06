// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch surfel instance buffer + bitmap atlas upload.

use super::super::gpu_types::{PatchInstance, PatchUniforms};
use super::super::recon::PatchResources;
use super::super::SceneRenderer;
use crate::scene::ReconId;
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

impl SceneRenderer {
    /// Upload embedded patch surfels into a GPU instance buffer + texture atlas.
    ///
    /// Walks the per-point patch frame arrays, skipping points without a patch
    /// (all-zero `u` row), and packs each point's `(R, R, 4)` RGBA bitmap into a
    /// 2D texture array atlas with page-grid packing (mirroring the thumbnail
    /// atlas), so the patch count can exceed the GPU array-layer limit.
    ///
    /// v1 renders textured patches only: a reconstruction that carries patch
    /// frames but no bitmaps uploads nothing (flat-shaded fallback is deferred).
    pub fn upload_patches(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ReconId,
        recon: &SfmrReconstruction,
    ) {
        // The bind group below needs the patch pipeline's layout and the
        // node's bundle, neither of which may exist yet.
        self.ensure_recon(device, id);

        // Reset so reloading a reconstruction without patches clears the old ones.
        self.recons.get_mut(&id).expect("just ensured").patch = None;

        let (Some(u_halfvecs), Some(v_halfvecs)) =
            (&recon.patch_u_halfvec_xyz, &recon.patch_v_halfvec_xyz)
        else {
            return;
        };
        let Some(bitmaps) = &recon.patch_bitmaps_y_x_rgba else {
            return;
        };
        // Tiles must be square and fit the GPU's 2D texture limit; on-disk files
        // are shape-verified, but an in-memory recon (e.g. built in Python) may
        // not be, so guard rather than trip a wgpu validation error.
        let resolution = bitmaps.shape()[1] as u32;
        let tile_cols = bitmaps.shape()[2] as u32;
        if resolution == 0 {
            return;
        }
        if tile_cols != resolution {
            log::warn!("patch bitmaps are non-square ({resolution}×{tile_cols}); skipping patches");
            return;
        }
        let max_texture_dim = device.limits().max_texture_dimension_2d;
        let max_array_layers = device.limits().max_texture_array_layers;
        if resolution > max_texture_dim {
            log::warn!(
                "patch bitmap resolution {resolution} exceeds the GPU texture limit \
                 {max_texture_dim}; skipping patches",
            );
            return;
        }

        // Collect the points that carry a patch: a point with no patch is an
        // all-zero `u` row. Bound the scan by every parallel array's length so a
        // short frame/bitmap array can't index out of range. The instance/atlas
        // buffers are compacted, so an instance's atlas slot is not its point
        // index.
        let n_rows = recon
            .points
            .len()
            .min(bitmaps.shape()[0])
            .min(u_halfvecs.nrows())
            .min(v_halfvecs.nrows());
        let point_indices: Vec<usize> = (0..n_rows)
            .filter(|&i| (0..3).any(|k| u_halfvecs[[i, k]] != 0.0))
            .collect();
        let patch_count = point_indices.len() as u32;
        if patch_count == 0 {
            return;
        }

        // Atlas grid dimensions: each layer ("page") holds a cols×rows grid of
        // patch tiles, respecting GPU texture size limits.
        let max_cells_per_axis = (max_texture_dim / resolution).max(1);
        let cols = ((patch_count as f32).sqrt().ceil() as u32).clamp(1, max_cells_per_axis);
        let rows_per_page = max_cells_per_axis;
        let patches_per_page = cols * rows_per_page;
        let num_pages = patch_count.div_ceil(patches_per_page).min(max_array_layers);
        let max_patches = patches_per_page * num_pages;
        let patch_count_clamped = patch_count.min(max_patches);
        if patch_count_clamped < patch_count {
            log::warn!(
                "GPU limits can only fit {patch_count_clamped} of {patch_count} patches \
                 in {num_pages} atlas pages; extra patches will not be displayed",
            );
        }
        // Shrink the last page's row count so the texture isn't larger than needed
        let total_rows = patch_count_clamped.div_ceil(cols);
        let actual_rows_per_page = total_rows.min(rows_per_page);
        let atlas_width = cols * resolution;
        let atlas_height = actual_rows_per_page * resolution;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("patch atlas"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: num_pages,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Write each patch's RGBA tile into its atlas cell and build the
        // corresponding instance.
        let mut instances: Vec<PatchInstance> = Vec::with_capacity(patch_count_clamped as usize);
        for (slot, &i) in point_indices
            .iter()
            .enumerate()
            .take(patch_count_clamped as usize)
        {
            let tile = bitmaps.index_axis(ndarray::Axis(0), i);
            let page = slot as u32 / patches_per_page;
            let idx_in_page = slot as u32 % patches_per_page;
            let col = idx_in_page % cols;
            let row = idx_in_page / cols;

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: col * resolution,
                        y: row * resolution,
                        z: page,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                tile.as_slice().unwrap(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(resolution * 4),
                    rows_per_image: Some(resolution),
                },
                wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 1,
                },
            );

            let p = &recon.points[i];
            instances.push(PatchInstance {
                center: [
                    p.position.x as f32,
                    p.position.y as f32,
                    p.position.z as f32,
                ],
                w: p.w as f32,
                u_halfvec: [u_halfvecs[[i, 0]], u_halfvecs[[i, 1]], u_halfvecs[[i, 2]]],
                _pad0: 0.0,
                v_halfvec: [v_halfvecs[[i, 0]], v_halfvecs[[i, 1]], v_halfvecs[[i, 2]]],
                atlas_layer: slot as u32,
                point_index: i as u32,
            });
        }

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Per-recon uniform buffer: `PatchUniforms` carries this atlas's grid.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("patch uniforms"),
            size: std::mem::size_of::<PatchUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = self.patch_bind_group_layout.as_ref();
        let sampler = self.patch_sampler.as_ref();
        let bundle = self.recons.get_mut(&id).expect("just ensured");
        let (Some(layout), Some(sampler)) = (layout, sampler) else {
            return;
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("patch bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bundle.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        bundle.patch = Some(PatchResources {
            instance_buffer,
            atlas_texture: texture,
            uniform_buffer,
            bind_group,
            count: patch_count_clamped,
            atlas_cols: cols,
            atlas_rows: actual_rows_per_page,
            patches_per_page,
        });

        let atlas_bytes = atlas_width as u64 * atlas_height as u64 * 4 * num_pages as u64;
        log::info!(
            "Uploaded {} patches ({}×{} px) as {}×{} × {} page(s) atlas ({:.1} MiB)",
            patch_count_clamped,
            resolution,
            resolution,
            atlas_width,
            atlas_height,
            num_pages,
            atlas_bytes as f64 / (1024.0 * 1024.0),
        );
    }
}
