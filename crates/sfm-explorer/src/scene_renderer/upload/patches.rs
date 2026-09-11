// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch surfel instance buffer + bitmap atlas upload.

use std::sync::Arc;

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
        // The atlas is the expensive half of this upload by a wide margin: one
        // texture allocation and one `write_texture` per patch, which is tens of
        // thousands of them on a real embedded-patches node. The instances are a
        // single buffer write. A bulk edit moves poses, positions and patch
        // frames and does not touch a single texel, so when the tiles and the
        // packing are the same the atlas is kept and only the instances are
        // rewritten -- which is what makes stepping through a node's history
        // cost the edit rather than the node.
        if self.repack_patches(device, id, &recon.point_set) {
            return;
        }
        // Reset so reloading a reconstruction without patches clears the old ones.
        self.recons.get_mut(&id).expect("just ensured").patch = None;
        let patch = self.build_patch_resources(device, queue, id, &recon.point_set, 0);
        self.recons.get_mut(&id).expect("just ensured").patch = patch;
    }

    /// Rewrite the base's patch instances over the atlas the node already holds,
    /// when that atlas is still the right one. `false` when it is not, and the
    /// caller must rebuild.
    ///
    /// Reusable means the tiles are the same pixels (the bitmap column, by
    /// pointer) and the same points pack into the same slots -- both of which
    /// hold across every edit the viewer has, and neither of which is assumed:
    /// a `false` here is a correct full rebuild, not a failure.
    fn repack_patches(
        &mut self,
        device: &wgpu::Device,
        id: ReconId,
        point_set: &sfmtool_core::PointSet,
    ) -> bool {
        let Some(bundle) = self.recons.get_mut(&id) else {
            return false;
        };
        let Some(patch) = bundle.patch.as_mut() else {
            return false;
        };
        let (Some(u_halfvecs), Some(v_halfvecs)) = (
            &point_set.patch_u_halfvec_xyz,
            &point_set.patch_v_halfvec_xyz,
        ) else {
            return false;
        };
        let Some(bitmaps) = point_set.patch_bitmaps_y_x_rgba.as_ref() else {
            return false;
        };
        if !Arc::ptr_eq(bitmaps, &patch.uploaded_bitmaps) {
            return false;
        }
        let packed = packed_points(point_set, u_halfvecs, v_halfvecs, bitmaps);
        // The slot a tile sits in is its position in this list, so an atlas
        // built from a different list addresses different tiles.
        if packed != patch.packed_points {
            return false;
        }

        let instances: Vec<PatchInstance> = packed
            .iter()
            .enumerate()
            .map(|(slot, &point)| {
                let i = point as usize;
                let p = &point_set.points[i];
                PatchInstance {
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
                    point_index: point,
                }
            })
            .collect();
        patch.instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // A new base carries no overlay, so every patch starts alive; the frame's
        // mask write puts the version's deleted set back over the top.
        let alive = vec![1u32; instances.len().max(1)];
        patch.alive_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch liveness"),
            contents: bytemuck::cast_slice(&alive),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        true
    }

    /// The surfel instances and bitmap atlas for one point set, with each
    /// instance's `point_index` offset by `index_offset`.
    ///
    /// Shared by the base's upload and the overlay's additions, which differ
    /// only in which point set they read and where its indexes start. The
    /// additions get their own `PatchResources`, and so their own atlas: the
    /// base's atlas is what a run of point edits shares, and appending to it
    /// would mean rebuilding it.
    pub(super) fn build_patch_resources(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ReconId,
        point_set: &sfmtool_core::PointSet,
        index_offset: u32,
    ) -> Option<PatchResources> {
        let (Some(u_halfvecs), Some(v_halfvecs)) = (
            &point_set.patch_u_halfvec_xyz,
            &point_set.patch_v_halfvec_xyz,
        ) else {
            return None;
        };
        let bitmaps = point_set.patch_bitmaps_y_x_rgba.as_ref()?;
        // Tiles must be square and fit the GPU's 2D texture limit; on-disk files
        // are shape-verified, but an in-memory recon (e.g. built in Python) may
        // not be, so guard rather than trip a wgpu validation error.
        let resolution = bitmaps.shape()[1] as u32;
        let tile_cols = bitmaps.shape()[2] as u32;
        if resolution == 0 {
            return None;
        }
        if tile_cols != resolution {
            log::warn!("patch bitmaps are non-square ({resolution}×{tile_cols}); skipping patches");
            return None;
        }
        let max_texture_dim = device.limits().max_texture_dimension_2d;
        let max_array_layers = device.limits().max_texture_array_layers;
        if resolution > max_texture_dim {
            log::warn!(
                "patch bitmap resolution {resolution} exceeds the GPU texture limit \
                 {max_texture_dim}; skipping patches",
            );
            return None;
        }

        let point_indices = packed_points(point_set, u_halfvecs, v_halfvecs, bitmaps);
        let patch_count = point_indices.len() as u32;
        if patch_count == 0 {
            return None;
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
        for (slot, &point) in point_indices
            .iter()
            .enumerate()
            .take(patch_count_clamped as usize)
        {
            let i = point as usize;
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

            let p = &point_set.points[i];
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
                point_index: index_offset + point,
            });
        }

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // The compaction above is what makes this map necessary: a patch's slot
        // is not its point index, so clearing a deleted point's surfel needs
        // the slot it was packed into.
        let slot_of_point: std::collections::HashMap<u32, u32> = instances
            .iter()
            .enumerate()
            .map(|(slot, instance)| (instance.point_index, slot as u32))
            .collect();
        let alive = vec![1u32; instances.len().max(1)];
        let alive_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch liveness"),
            contents: bytemuck::cast_slice(&alive),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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
        let bundle = self
            .recons
            .get_mut(&id)
            .expect("the caller ensured the bundle");
        let (Some(layout), Some(sampler)) = (layout, sampler) else {
            return None;
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

        let resources = PatchResources {
            instance_buffer,
            alive_buffer,
            slot_of_point,
            uploaded_bitmaps: Arc::clone(bitmaps),
            // The clamped list, so a node whose patches did not all fit is
            // compared against what the atlas actually holds.
            packed_points: point_indices
                .into_iter()
                .take(patch_count_clamped as usize)
                .collect(),
            atlas_texture: texture,
            uniform_buffer,
            bind_group,
            count: patch_count_clamped,
            atlas_cols: cols,
            atlas_rows: actual_rows_per_page,
            patches_per_page,
        };

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
        Some(resources)
    }
}

/// The points that carry a patch, ascending: a point with no patch is an
/// all-zero `u` row.
///
/// The atlas and the instance buffer are both compacted over this list, so a
/// patch's slot is its position here and not its point index. The scan is
/// bounded by every parallel array's length, so a short frame or bitmap array
/// cannot index out of range.
///
/// Shared by the build and the repack because it *is* the packing: the repack is
/// only sound if it produces the same list the atlas was written from, and two
/// copies of this filter could drift apart without either one looking wrong.
fn packed_points(
    point_set: &sfmtool_core::PointSet,
    u_halfvecs: &ndarray::Array2<f32>,
    v_halfvecs: &ndarray::Array2<f32>,
    bitmaps: &ndarray::Array4<u8>,
) -> Vec<u32> {
    let n_rows = point_set
        .points
        .len()
        .min(bitmaps.shape()[0])
        .min(u_halfvecs.nrows())
        .min(v_halfvecs.nrows());
    (0..n_rows)
        .filter(|&i| (0..3).any(|k| u_halfvecs[[i, k]] != 0.0))
        .map(|i| i as u32)
        .collect()
}
