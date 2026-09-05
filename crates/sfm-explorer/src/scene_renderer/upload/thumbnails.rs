// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Embedded camera thumbnail atlas upload.

use super::super::gpu_types::{ImageQuadUniforms, MAX_ATLAS_COLS, THUMBNAIL_SIZE};
use super::super::SceneRenderer;
use crate::scene::ReconId;
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

impl SceneRenderer {
    /// Upload one reconstruction's embedded camera thumbnails into a GPU 2D
    /// texture atlas of its own.
    ///
    /// Packs all 128×128 RGB thumbnails into a single large 2D texture arranged
    /// as a grid, avoiding the 256-layer limit of texture arrays. Also creates
    /// the node's image quad uniform buffer.
    pub fn upload_thumbnails(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ReconId,
        recon: &SfmrReconstruction,
    ) {
        let image_count = recon.images.len() as u32;
        if image_count == 0 {
            return;
        }
        self.ensure_recon(device, id);

        // Compute atlas grid dimensions, respecting GPU texture size limits.
        // Images are packed into a 2D texture array: each layer ("page") holds a
        // cols×rows grid of thumbnails, and we add as many layers as needed.
        let max_texture_dim = device.limits().max_texture_dimension_2d;
        let max_array_layers = device.limits().max_texture_array_layers;
        let max_cells_per_axis = max_texture_dim / THUMBNAIL_SIZE;
        let cols = ((image_count as f32).sqrt().ceil() as u32)
            .min(MAX_ATLAS_COLS)
            .min(max_cells_per_axis);
        let rows_per_page = max_cells_per_axis;
        let images_per_page = cols * rows_per_page;
        let num_pages = image_count.div_ceil(images_per_page).min(max_array_layers);
        let max_images = images_per_page * num_pages;
        let image_count_clamped = image_count.min(max_images);
        if image_count_clamped < image_count {
            log::warn!(
                "GPU limits can only fit {image_count_clamped} of {image_count} thumbnails \
                 in {num_pages} atlas pages; extra thumbnails will not be displayed",
            );
        }
        // Shrink the last page's row count so the texture isn't larger than needed
        let total_rows = image_count_clamped.div_ceil(cols);
        let actual_rows_per_page = total_rows.min(rows_per_page);
        let atlas_width = cols * THUMBNAIL_SIZE;
        let atlas_height = actual_rows_per_page * THUMBNAIL_SIZE;

        // Create 2D texture array atlas
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("thumbnail atlas"),
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

        // Upload each embedded thumbnail to its grid cell (RGB → RGBA)
        for i in 0..image_count_clamped as usize {
            let rgb_slice = recon.thumbnails_y_x_rgb.index_axis(ndarray::Axis(0), i);
            let mut rgba_data = Vec::with_capacity((THUMBNAIL_SIZE * THUMBNAIL_SIZE * 4) as usize);
            for pixel in rgb_slice.as_slice().unwrap().as_chunks::<3>().0.iter() {
                rgba_data.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
            }

            let page = i as u32 / images_per_page;
            let idx_in_page = i as u32 % images_per_page;
            let col = idx_in_page % cols;
            let row = idx_in_page / cols;

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: col * THUMBNAIL_SIZE,
                        y: row * THUMBNAIL_SIZE,
                        z: page,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(THUMBNAIL_SIZE * 4),
                    rows_per_image: Some(THUMBNAIL_SIZE),
                },
                wgpu::Extent3d {
                    width: THUMBNAIL_SIZE,
                    height: THUMBNAIL_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Per-recon uniform buffer for this atlas's grid parameters. (The
        // sampler is shared: every atlas is sampled the same way.)
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("image quad uniforms"),
            contents: bytemuck::bytes_of(&ImageQuadUniforms {
                view_proj: [[0.0; 4]; 4],
                atlas_cols: cols,
                atlas_rows: actual_rows_per_page,
                images_per_page,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Store texture and view; the bind group is created by
        // rebuild_frustum_bind_group once the color buffer exists.
        let bundle = self.recons.get_mut(&id).expect("just ensured");
        bundle.atlas_cols = cols;
        bundle.atlas_rows = actual_rows_per_page;
        bundle.images_per_page = images_per_page;
        bundle.thumbnail_view = Some(texture_view);
        bundle.image_quad_uniform_buffer = Some(uniform_buf);
        bundle.thumbnail_texture = Some(texture);
        log::info!(
            "Uploaded {} thumbnails as {}×{} × {} page(s) atlas ({}×{} grid per page)",
            image_count_clamped,
            atlas_width,
            atlas_height,
            num_pages,
            cols,
            actual_rows_per_page,
        );
    }
}
