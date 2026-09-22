// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera thumbnail atlas upload, from the node's display column.

use std::sync::Arc;

use ndarray::Array4;

use super::super::gpu_types::{ImageQuadUniforms, MAX_ATLAS_COLS, THUMBNAIL_SIZE};
use super::super::SceneRenderer;
use super::atlas::{write_band, Band};
use super::Uploaded;
use crate::display_thumbnails::{row_for, DisplayThumbnails, PLACEHOLDER_GREY};
use crate::scene::ReconId;
use sfmtool_core::progress::Progress;
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

/// What a node's thumbnail atlas was written from, which is what says whether
/// it is still the right one.
pub struct UploadedThumbnails {
    /// The node's display column, by pointer.
    display: Option<Arc<DisplayThumbnails>>,
    /// The value's own thumbnail column, by pointer: the fallback for an image
    /// the display column was not built for.
    column: Option<Arc<Array4<u8>>>,
    /// The image names in atlas order, the value's image order.
    names: Vec<String>,
    /// Which cells hold a final row rather than the placeholder.
    written: Vec<bool>,
    /// The display column's ready count when the cells were last filled.
    ready_seen: usize,
}

impl UploadedThumbnails {
    /// How many cells hold a final row.
    #[cfg(test)]
    pub(crate) fn final_cells(&self) -> usize {
        self.written.iter().filter(|&&w| w).count()
    }

    /// Whether this atlas was built for `display`, `column` and the images of
    /// `recon`, in `recon`'s order.
    fn matches(
        &self,
        display: Option<&Arc<DisplayThumbnails>>,
        recon: &SfmrReconstruction,
    ) -> bool {
        let same_display = match (&self.display, display) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        let same_column = match (&self.column, &recon.image_table.thumbnails_y_x_rgb) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        same_display
            && same_column
            && self.names.len() == recon.image_table.images.len()
            && self
                .names
                .iter()
                .zip(&recon.image_table.images)
                .all(|(name, image)| *name == image.name)
    }
}

/// One cell's pixels: the row when it is final, the flat placeholder until
/// then.
fn cell<'a>(
    display: Option<&'a DisplayThumbnails>,
    recon: &'a SfmrReconstruction,
    index: usize,
    placeholder: &'a [u8],
) -> (std::borrow::Cow<'a, [u8]>, bool) {
    match row_for(display, recon, index) {
        Some(view) => match view.to_slice() {
            Some(slice) => (std::borrow::Cow::Borrowed(slice), true),
            None => (
                std::borrow::Cow::Owned(view.iter().copied().collect()),
                true,
            ),
        },
        None => (std::borrow::Cow::Borrowed(placeholder), false),
    }
}

impl SceneRenderer {
    /// Upload one reconstruction's camera thumbnails into a GPU 2D texture
    /// atlas of its own, from the node's display column.
    ///
    /// Packs all 128×128 RGB thumbnails into a single large 2D texture arranged
    /// as a grid, avoiding the 256-layer limit of texture arrays. Also creates
    /// the node's image quad uniform buffer. A cell whose row is still being
    /// built from its photograph holds a flat grey placeholder, which
    /// [`Self::refresh_thumbnails`] overwrites once the row is final.
    ///
    /// A node with no display column and a value with no thumbnail column of
    /// its own gets no atlas, and its frustums draw outlines without image
    /// quads.
    ///
    /// [`Uploaded::Reused`] when the atlas the node already holds is still the
    /// right one, which is the answer on every edit that leaves the image list
    /// alone, and what the frame's phase note says as `reused`.
    ///
    /// `progress` is the frame's `thumbnails` phase; the stages under it are
    /// [`Progress::detail_phase`]s, and divide the cost into allocating the
    /// atlas and filling it one thumbnail at a time.
    pub fn upload_thumbnails(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ReconId,
        recon: &SfmrReconstruction,
        display: Option<&Arc<DisplayThumbnails>>,
        progress: &Progress<'_>,
    ) -> Uploaded {
        let image_count = recon.image_table.images.len() as u32;
        self.ensure_recon(device, id, progress);
        if image_count == 0 || (display.is_none() && recon.image_table.thumbnails_y_x_rgb.is_none())
        {
            let bundle = self.recons.get_mut(&id).expect("just ensured");
            bundle.thumbnail_texture = None;
            bundle.thumbnail_view = None;
            bundle.uploaded_thumbnails = None;
            return Uploaded::Built(0);
        }
        // The atlas is a function of the display column, the value's own
        // column and the image list, and of nothing else. An edit that leaves
        // the image list alone leaves it correct, so the node keeps the one it
        // has rather than paying a texture allocation to arrive at the same
        // pixels; rows that finished since are written by `refresh_thumbnails`.
        let bundle = self.recons.get(&id).expect("just ensured");
        let reusable = bundle
            .uploaded_thumbnails
            .as_ref()
            .is_some_and(|uploaded| uploaded.matches(display, recon))
            && bundle.thumbnail_view.is_some();
        if reusable {
            return Uploaded::Reused;
        }

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
        let atlas_phase = progress.detail_phase("atlas");
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
        drop(atlas_phase);

        // Fill the atlas one row of cells at a time and upload each row in a
        // single call, expanding RGB to RGBA on the way in. The same shape as
        // the patch atlas and for the same reason ([`super::atlas`]).
        let tiles_phase = progress.detail_phase("tiles");
        let placeholder = vec![PLACEHOLDER_GREY; (THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3) as usize];
        let ready_seen = display.map_or(0, |d| d.ready());
        let mut written = vec![false; image_count as usize];
        let mut band = Band::new(atlas_width, THUMBNAIL_SIZE);
        for i in 0..image_count_clamped {
            let (tile, final_row) = cell(display.map(|d| &**d), recon, i as usize, &placeholder);
            written[i as usize] = final_row;
            let idx_in_page = i % images_per_page;
            let col = idx_in_page % cols;
            band.place_rgb(col, &tile);

            if col + 1 == cols || i + 1 == image_count_clamped {
                band.blank_from(col + 1);
                write_band(
                    queue,
                    &texture,
                    &band,
                    i / images_per_page,
                    idx_in_page / cols,
                    THUMBNAIL_SIZE,
                );
            }
        }
        drop(tiles_phase);

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
        bundle.uploaded_thumbnails = Some(UploadedThumbnails {
            display: display.cloned(),
            column: recon.image_table.thumbnails_y_x_rgb.clone(),
            names: recon
                .image_table
                .images
                .iter()
                .map(|image| image.name.clone())
                .collect(),
            written,
            ready_seen,
        });
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
        Uploaded::Built(image_count_clamped as usize)
    }

    /// Write the cells whose rows have become final since the atlas was
    /// filled, and nothing else.
    ///
    /// Runs every frame for every node, and costs a counter comparison unless
    /// synthesis has finished rows since the last look. Returns how many cells
    /// it wrote.
    pub fn refresh_thumbnails(
        &mut self,
        queue: &wgpu::Queue,
        id: ReconId,
        recon: &SfmrReconstruction,
    ) -> usize {
        let Some(bundle) = self.recons.get_mut(&id) else {
            return 0;
        };
        let (Some(uploaded), Some(texture)) = (
            bundle.uploaded_thumbnails.as_mut(),
            bundle.thumbnail_texture.as_ref(),
        ) else {
            return 0;
        };
        let Some(display) = uploaded.display.clone() else {
            return 0;
        };
        let ready = display.ready();
        if ready == uploaded.ready_seen {
            return 0;
        }
        uploaded.ready_seen = ready;
        let (cols, images_per_page) = (bundle.atlas_cols, bundle.images_per_page);
        let placeholder: [u8; 0] = [];
        let mut count = 0;
        let mut band = Band::new(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
        for (i, written) in uploaded.written.iter_mut().enumerate() {
            if *written || i as u32 >= images_per_page * texture.depth_or_array_layers() {
                continue;
            }
            let (tile, final_row) = cell(Some(&display), recon, i, &placeholder);
            if !final_row {
                continue;
            }
            *written = true;
            count += 1;
            // One cell at its own origin: a band one tile wide.
            band.place_rgb(0, &tile);
            let idx_in_page = i as u32 % images_per_page;
            write_cell(
                queue,
                texture,
                &band,
                i as u32 / images_per_page,
                idx_in_page % cols,
                idx_in_page / cols,
            );
        }
        count
    }
}

/// Upload a one-tile band into cell (`col`, `row`) of page `page`.
fn write_cell(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    band: &Band,
    page: u32,
    col: u32,
    row: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: col * THUMBNAIL_SIZE,
                y: row * THUMBNAIL_SIZE,
                z: page,
            },
            aspect: wgpu::TextureAspect::All,
        },
        band.bytes(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(band.bytes_per_row()),
            rows_per_image: Some(THUMBNAIL_SIZE),
        },
        wgpu::Extent3d {
            width: THUMBNAIL_SIZE,
            height: THUMBNAIL_SIZE,
            depth_or_array_layers: 1,
        },
    );
}
