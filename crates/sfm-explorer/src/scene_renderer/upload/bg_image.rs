// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The photograph behind the 3D view in camera view mode.
//!
//! The background is one texture of one level of the photograph's pyramid:
//! the coarsest level that still covers the viewport, and no finer than the
//! GPU's texture limit allows ([`background_level`]). The pixels come from the
//! viewer's photograph cache through two closures the caller passes,
//! `state::display_layout` and `state::display_level`, which read the layout
//! and the level on the rayon pool and never on the calling thread; the
//! renderer calls them only when the level it wants is not the one it holds.

use super::super::distorted_mesh::generate_bg_distorted_mesh;
use super::super::gpu_types::{
    BgDistortedVertex, BG_DISTORTION_SUBDIVISIONS, BG_FISHEYE_SUBDIVISIONS, BG_PINHOLE_SUBDIVISIONS,
};
use super::super::SceneRenderer;
use crate::scene::ImageRef;
use crate::state::{DisplayLayout, DisplayLevel};
use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::camera::{CameraIntrinsics, CoarsestLevel, PhotographTile, PyramidLayout};
use sfmtool_core::SfmrReconstruction;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// The background photograph's mesh for image `image_index`: a tessellated
/// grid of world-space ray directions through the image's own intrinsics (its
/// camera with its intrinsic adjustment applied), so the photograph lies where
/// its pixels' rays point. Its texture coordinates run from 0 to 1 over the
/// whole photograph, multiplied by `uv_scale`, the [`level_uv_scale`] of the
/// level the texture holds.
///
/// Uses the same camera-to-world rotation as the frustum wireframes, so the
/// background shader can use the standard `view_proj = projection * view`
/// pipeline. Split out from [`SceneRenderer::upload_bg_image`] so the mesh is
/// assertable without a photograph or a GPU buffer readback.
///
/// # Panics
///
/// Panics when `image_index` is past the images.
pub(super) fn bg_image_mesh(
    recon: &SfmrReconstruction,
    image_index: usize,
    uv_scale: [f32; 2],
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
    let (mut vertices, indices) = generate_bg_distorted_mesh(&camera, &r, subdivisions);
    for vertex in &mut vertices {
        vertex.uv = [vertex.uv[0] * uv_scale[0], vertex.uv[1] * uv_scale[1]];
    }
    (vertices, indices)
}

/// The pyramid level the background draws a photograph at, from the sizes of
/// its levels, level 0 first and each half the one before.
///
/// The coarsest level whose longer side is still at least `viewport_long_px`,
/// the viewport's longer side in physical pixels, but no finer than the first
/// level whose longer side fits `max_texture_dimension`, the device's
/// `max_texture_dimension_2d`. Level 0 when even it is shorter than the
/// viewport. `None` when no level fits the limit, which only a pyramid that
/// stops before its long side is short enough can give: one of a photograph a
/// pixel or two tall.
pub(super) fn background_level(
    level_sizes: &[(u32, u32)],
    viewport_long_px: u32,
    max_texture_dimension: u32,
) -> Option<usize> {
    let long = |&(w, h): &(u32, u32)| w.max(h);
    let first_fit = level_sizes
        .iter()
        .position(|size| long(size) <= max_texture_dimension)?;
    let covering = level_sizes
        .iter()
        .rposition(|size| long(size) >= viewport_long_px)
        .unwrap_or(0);
    Some(covering.max(first_fit))
}

/// The factors the background mesh's texture coordinates are multiplied by
/// when the texture holds level `level`, of `level_size`, of a photograph of
/// `size`: `width / (w_l·2^l)` and `height / (h_l·2^l)`.
///
/// Each halving drops an odd last column or row, so level `l` covers only the
/// level-0 columns `0` to `w_l·2^l` and the rows `0` to `h_l·2^l`. The mesh's
/// texture coordinates run from 0 to 1 over the whole photograph; scaled, each
/// level-0 pixel lands on the texel that covers it, and the dropped columns and
/// rows, fewer than `2^l`, fall past the texture's edge, where the sampler
/// clamps, instead of the level being stretched over them.
pub(super) fn level_uv_scale(size: (u32, u32), level_size: (u32, u32), level: usize) -> [f32; 2] {
    let covered = |n: u32| f64::from(n) * 2f64.powi(level as i32);
    [
        (f64::from(size.0) / covered(level_size.0)) as f32,
        (f64::from(size.1) / covered(level_size.1)) as f32,
    ]
}

/// The level of a photograph the background texture holds, with the sizes of
/// every level of that photograph, kept so the level can be chosen again when
/// the viewport changes without asking the cache, which may have dropped the
/// photograph since.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LoadedBgLevel {
    pub(crate) image: ImageRef,
    pub(crate) level: usize,
    pub(crate) level_sizes: Vec<(u32, u32)>,
}

impl LoadedBgLevel {
    /// [`level_uv_scale`] for the level held.
    fn uv_scale(&self) -> [f32; 2] {
        level_uv_scale(
            self.level_sizes[0],
            self.level_sizes[self.level],
            self.level,
        )
    }
}

/// What one background mesh was built from: the image, its intrinsics (its
/// camera with its intrinsic adjustment applied), its camera-to-world rotation
/// and the texture coordinates' scale for the level the texture holds.
/// [`bg_image_mesh`] and the scale read exactly these, so a mesh whose key
/// still matches is the one a rebuild would make.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BgMeshKey {
    pub(crate) image: ImageRef,
    pub(crate) intrinsics: CameraIntrinsics,
    pub(crate) rotation: [f64; 9],
    pub(crate) uv_scale: [f32; 2],
}

/// The core of `tile`, a photograph tile with its one-pixel border: the
/// pixels of its level it covers.
fn tile_core(tile: &PhotographTile) -> ImageU8 {
    let [w, h] = tile.size;
    let c = tile.pixels.channels() as usize;
    let stride = tile.pixels.width() as usize * c;
    let span = w as usize * c;
    let data = tile.pixels.data();
    let mut core = Vec::with_capacity(span * h as usize);
    for y in 0..h as usize {
        let from = (y + 1) * stride + c;
        core.extend_from_slice(&data[from..from + span]);
    }
    ImageU8::new(w, h, tile.pixels.channels(), core)
}

/// `level` as RGBA rows for an `Rgba8UnormSrgb` texture, with alpha 255 where
/// the level has none. The cache's levels are RGB; the background pipeline
/// draws with no blending, so the alpha has no visible effect.
fn rgba_of(level: &ImageU8) -> Vec<u8> {
    let data = level.data();
    match level.channels() {
        4 => data.to_vec(),
        3 => data
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect(),
        2 => data
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[g, a]| [g, g, g, a])
            .collect(),
        _ => data.iter().flat_map(|&g| [g, g, g, 255]).collect(),
    }
}

impl SceneRenderer {
    /// Show `image`'s photograph behind the 3D view in camera view mode.
    ///
    /// Only when the texture does not already hold this image at the level
    /// the viewport needs, on a change of image or a change of viewport size
    /// that changes the level, it calls `layout` for the photograph's layout,
    /// chooses the level with [`background_level`] from its level sizes, the
    /// viewport's size (`current_size`, in physical pixels) and the device's
    /// texture limit, and calls `level` for that level's pixels. The app
    /// passes `state::display_layout` and `state::display_level`, which read
    /// on the rayon pool and never on the calling thread. What they answer
    /// decides what is drawn:
    ///
    /// - **Ready**: the level is copied into a new texture, and the bind group
    ///   rebuilt around it.
    /// - **Building** with the coarsest level a build of the photograph's
    ///   mipmap has published: that level is drawn, level `K`, the one tile a
    ///   build publishes first, until the photograph is known and its own
    ///   level lands. It is copied into a texture once, not on every frame
    ///   that asks while the build runs.
    /// - **Loading**, the layout's or the level's, or **Building** before the
    ///   coarsest level is published: the texture keeps what it
    ///   holds, so while the next image of a camera switch or an animation
    ///   loads, the previous one stays on its own mesh. A background of
    ///   another node is cleared, since the frame's node transform is not its
    ///   node's. Nothing is drawn when nothing was shown.
    /// - **Unreadable**: nothing is drawn. The failure is remembered, so later
    ///   frames ask again without decoding the file.
    ///
    /// The mesh is then rebuilt, for the image whose photograph the texture
    /// holds, whenever what it is built from changed, so a version that gives
    /// the image other intrinsics or another pose redraws the photograph where
    /// its rays now point without asking for the pixels again.
    pub(crate) fn upload_bg_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        recon: &SfmrReconstruction,
        image: ImageRef,
        layout: impl FnOnce() -> DisplayLayout,
        level: impl FnOnce(&Arc<PyramidLayout>, usize) -> DisplayLevel,
    ) {
        if image.index() >= recon.image_set.images.len() {
            self.clear_bg_image();
            return;
        }
        let viewport_long_px = self.current_size.0.max(self.current_size.1);
        let max_dimension = device.limits().max_texture_dimension_2d;
        let held = self.bg_image_loaded.as_ref().is_some_and(|loaded| {
            loaded.image == image
                && background_level(&loaded.level_sizes, viewport_long_px, max_dimension)
                    == Some(loaded.level)
        });
        if !held {
            match layout() {
                DisplayLayout::Known(layout) => {
                    let level_sizes: Vec<(u32, u32)> =
                        (0..layout.levels()).map(|l| layout.level_size(l)).collect();
                    match background_level(&level_sizes, viewport_long_px, max_dimension) {
                        Some(chosen) => match level(&layout, chosen) {
                            DisplayLevel::Ready(pixels) => self.load_bg_texture(
                                device,
                                queue,
                                LoadedBgLevel {
                                    image,
                                    level: chosen,
                                    level_sizes,
                                },
                                &pixels,
                            ),
                            DisplayLevel::Loading => self.keep_bg_image_while_loading(image),
                            DisplayLevel::Unreadable => self.clear_bg_image(),
                        },
                        None => self.clear_bg_image(),
                    }
                }
                // A photograph waiting for its mipmap draws its coarsest level
                // once the build has published it, and until then keeps the
                // previous image's photograph on its own mesh, as one being
                // read does.
                DisplayLayout::Building(Some(coarsest)) => {
                    self.load_coarsest_level(device, queue, image, &coarsest)
                }
                DisplayLayout::Loading | DisplayLayout::Building(None) => {
                    self.keep_bg_image_while_loading(image)
                }
                DisplayLayout::Unreadable | DisplayLayout::WorkspaceFileError(_) => {
                    self.clear_bg_image()
                }
            }
        }
        self.update_bg_mesh(device, recon);
    }

    /// Draw `coarsest`, the coarsest level a build of `image`'s mipmap has
    /// published, unless the texture holds that level of `image` already: its
    /// first tile's core, level `K` whole, with the texture coordinates
    /// scaled for level `K` of the photograph's own size, so it lies where the
    /// photograph will.
    fn load_coarsest_level(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: ImageRef,
        coarsest: &CoarsestLevel,
    ) {
        let level = coarsest.level();
        if self
            .bg_image_loaded
            .as_ref()
            .is_some_and(|loaded| loaded.image == image && loaded.level == level)
        {
            return;
        }
        let Some(pixels) = coarsest.tiles.first().map(|tile| tile_core(tile)) else {
            self.keep_bg_image_while_loading(image);
            return;
        };
        let layout = &coarsest.layout;
        let level_sizes = (0..layout.levels()).map(|l| layout.level_size(l)).collect();
        self.load_bg_texture(
            device,
            queue,
            LoadedBgLevel {
                image,
                level,
                level_sizes,
            },
            &pixels,
        );
    }

    /// While `image`'s photograph loads, keep the one the texture holds, on
    /// its own mesh, unless it is of another node, whose transform is not the
    /// frame's.
    fn keep_bg_image_while_loading(&mut self, image: ImageRef) {
        if self
            .bg_image_loaded
            .as_ref()
            .is_some_and(|loaded| loaded.image.recon != image.recon)
        {
            self.clear_bg_image();
        }
    }

    /// Rebuild the background mesh for the image whose photograph the texture
    /// holds, unless the mesh already is the one a rebuild would make.
    fn update_bg_mesh(&mut self, device: &wgpu::Device, recon: &SfmrReconstruction) {
        let Some(loaded) = &self.bg_image_loaded else {
            return;
        };
        let (image, uv_scale) = (loaded.image, loaded.uv_scale());
        let Some(img) = recon.image_set.images.get(image.index()) else {
            self.clear_bg_image();
            return;
        };
        let intrinsics = recon.image_set.image_intrinsics(image.index());
        let rotation = img.camera_to_world_rotation_flat();
        if self.bg_image_mesh_key.as_ref().is_some_and(|key| {
            key.image == image
                && key.rotation == rotation
                && key.uv_scale == uv_scale
                && key.intrinsics == *intrinsics
        }) {
            return; // the mesh is the one a rebuild would make
        }
        let (vertices, indices) = bg_image_mesh(recon, image.index(), uv_scale);
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
            uv_scale,
        });
    }

    /// Copy `pixels`, the level `loaded` names, into a new background texture
    /// and rebuild the bind group around it.
    fn load_bg_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        loaded: LoadedBgLevel,
        pixels: &ImageU8,
    ) {
        let (w, h) = (pixels.width(), pixels.height());
        let rgba = rgba_of(pixels);

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

        log::info!(
            "Loaded bg image {} at level {} ({}×{})",
            loaded.image.index(),
            loaded.level,
            w,
            h
        );
        self.bg_image_texture = Some(texture);
        self.bg_image_loaded = Some(loaded);
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
