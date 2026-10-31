// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Full-resolution background image upload for camera view mode.

use super::super::distorted_mesh::generate_bg_distorted_mesh;
use super::super::gpu_types::{
    BgDistortedVertex, BG_DISTORTION_SUBDIVISIONS, BG_FISHEYE_SUBDIVISIONS, BG_PINHOLE_SUBDIVISIONS,
};
use super::super::SceneRenderer;
use crate::scene::ImageRef;
use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

/// The background photograph's mesh for image `image_index`: a tessellated
/// grid of world-space ray directions through the image's own intrinsics (its
/// camera with its intrinsic adjustment applied), so the photograph lies where
/// its pixels' rays point.
///
/// Uses the same camera-to-world rotation as the frustum wireframes, so the
/// background shader can use the standard `view_proj = projection * view`
/// pipeline. Split out from [`SceneRenderer::upload_bg_image`] so the mesh is
/// assertable without a photograph on disk or a GPU buffer readback.
///
/// # Panics
///
/// Panics when `image_index` is past the images.
pub(super) fn bg_image_mesh(
    recon: &SfmrReconstruction,
    image_index: usize,
) -> (Vec<BgDistortedVertex>, Vec<u32>) {
    let img = &recon.image_set.images[image_index];
    let camera = recon.image_set.image_intrinsics(image_index);
    let r = img.camera_to_world_rotation_flat();
    let subdivisions = if camera.model.is_fisheye() {
        BG_FISHEYE_SUBDIVISIONS
    } else if camera.has_distortion() {
        BG_DISTORTION_SUBDIVISIONS
    } else {
        BG_PINHOLE_SUBDIVISIONS
    };
    generate_bg_distorted_mesh(&camera, &r, subdivisions)
}

/// What one background mesh was built from: the image, its intrinsics (its
/// camera with its intrinsic adjustment applied) and its camera-to-world
/// rotation. [`bg_image_mesh`] reads exactly these, so a mesh whose key still
/// matches is the one a rebuild would make.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BgMeshKey {
    pub(crate) image: ImageRef,
    pub(crate) intrinsics: CameraIntrinsics,
    pub(crate) rotation: [f64; 9],
}

impl SceneRenderer {
    /// Load a full-resolution camera image for the background in camera view mode.
    ///
    /// Creates a single 2D texture at the image's native resolution and rebuilds
    /// the background image bind group, skipping the reload when the same image
    /// is already loaded. The mesh is rebuilt whenever what it is built from
    /// changed, so a version that gives the image other intrinsics or another
    /// pose redraws the photograph where its rays now point without decoding it
    /// again.
    pub(crate) fn upload_bg_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        recon: &SfmrReconstruction,
        image: ImageRef,
    ) {
        let image_index = image.index();
        let Some(img) = recon.image_set.images.get(image_index) else {
            return;
        };
        if self.bg_image_loaded != Some(image) && !self.load_bg_texture(device, queue, recon, image)
        {
            return;
        }
        let intrinsics = recon.image_set.image_intrinsics(image_index);
        let rotation = img.camera_to_world_rotation_flat();
        if self.bg_image_mesh_key.as_ref().is_some_and(|key| {
            key.image == image && key.rotation == rotation && key.intrinsics == *intrinsics
        }) {
            return; // the mesh is the one a rebuild would make
        }
        let (vertices, indices) = bg_image_mesh(recon, image_index);
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
        self.bg_image_mesh_key = Some(BgMeshKey {
            image,
            intrinsics: intrinsics.into_owned(),
            rotation,
        });
    }

    /// Decode `image`'s photograph into the background texture and rebuild
    /// the bind group around it. Returns whether it did: a photograph that
    /// cannot be read leaves every field as it was.
    fn load_bg_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        recon: &SfmrReconstruction,
        image: ImageRef,
    ) -> bool {
        let img = &recon.image_set.images[image.index()];
        let image_path = recon.workspace_dir.join(&img.name);
        // The photograph decoder every reader shares: the contents choose the
        // format, and the EXIF orientation is ignored, so the texture lines up
        // with the camera's width and height.
        let photograph = match ImageU8::read_rgba(&image_path) {
            Ok(img) => img,
            Err(e) => {
                log::warn!("Failed to load bg image {}: {}", image_path.display(), e);
                return false;
            }
        };

        let (w, h) = (photograph.width(), photograph.height());
        let rgba = photograph.into_data();

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
        self.bg_image_loaded = Some(image);
        log::info!("Loaded bg image {} ({}×{})", image_path.display(), w, h);
        true
    }

    /// Clear the background image when leaving camera view mode.
    pub(crate) fn clear_bg_image(&mut self) {
        self.bg_image_bind_group = None;
        self.bg_image_texture = None;
        self.bg_image_loaded = None;
        self.bg_image_mesh_key = None;
        self.bg_image_distorted_vertex_buffer = None;
        self.bg_image_distorted_index_buffer = None;
        self.bg_image_distorted_index_count = 0;
    }
}
