// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Full-resolution background image upload for camera view mode.

use super::super::distorted_mesh::generate_bg_distorted_mesh;
use super::super::gpu_types::{
    BG_DISTORTION_SUBDIVISIONS, BG_FISHEYE_SUBDIVISIONS, BG_PINHOLE_SUBDIVISIONS,
};
use super::super::SceneRenderer;
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

impl SceneRenderer {
    /// Load a full-resolution camera image for the background in camera view mode.
    ///
    /// Creates a single 2D texture at the image's native resolution and rebuilds
    /// the background image bind group. Skips reloading if the same image index
    /// is already loaded.
    pub fn upload_bg_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        recon: &SfmrReconstruction,
        image_index: usize,
    ) {
        if self.bg_image_loaded_index == Some(image_index) {
            return; // already loaded
        }

        let Some(img) = recon.images.get(image_index) else {
            return;
        };
        let image_path = recon.workspace_dir.join(&img.name);
        let dyn_image = match image::open(&image_path) {
            Ok(img) => img,
            Err(e) => {
                log::warn!("Failed to load bg image {}: {}", image_path.display(), e);
                return;
            }
        };

        let rgba = dyn_image.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bg image"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&Default::default());

        // Rebuild bind group with the new texture
        if let (Some(layout), Some(uniform_buf), Some(sampler)) = (
            &self.bg_image_bind_group_layout,
            &self.bg_image_uniform_buffer,
            &self.bg_image_sampler,
        ) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg image bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.bg_image_bind_group = Some(bind_group);
        }

        self.bg_image_texture = Some(texture);
        self.bg_image_loaded_index = Some(image_index);
        log::info!("Loaded bg image {} ({}×{})", image_path.display(), w, h);

        // Generate tessellated mesh with world-space ray directions.
        // Uses the same camera-to-world rotation as frustum wireframes, so the
        // BG shader can use the standard view_proj = projection * view pipeline.
        let camera = &recon.cameras[img.camera_index as usize];
        let r = img.camera_to_world_rotation_flat();
        let subdivisions = if camera.model.is_fisheye() {
            BG_FISHEYE_SUBDIVISIONS
        } else if camera.has_distortion() {
            BG_DISTORTION_SUBDIVISIONS
        } else {
            BG_PINHOLE_SUBDIVISIONS
        };
        let (vertices, indices) = generate_bg_distorted_mesh(camera, &r, subdivisions);
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bg distorted vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bg distorted indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.bg_image_distorted_vertex_buffer = Some(vbuf);
        self.bg_image_distorted_index_buffer = Some(ibuf);
        self.bg_image_distorted_index_count = indices.len() as u32;
    }

    /// Clear the background image when leaving camera view mode.
    pub fn clear_bg_image(&mut self) {
        self.bg_image_bind_group = None;
        self.bg_image_texture = None;
        self.bg_image_loaded_index = None;
        self.bg_image_distorted_vertex_buffer = None;
        self.bg_image_distorted_index_buffer = None;
        self.bg_image_distorted_index_count = 0;
    }
}
