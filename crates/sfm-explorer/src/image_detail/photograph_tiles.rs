// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The photograph as Image Detail draws it: photograph tiles of the pyramid
//! level that matches the zoom, never one texture of the whole photograph.
//!
//! A photograph tile is a 512 x 512 block of one level of the photograph's
//! pyramid with a one-pixel border of its neighbours' pixels, uploaded as one
//! egui texture of at most 514 x 514. No texture is ever larger than that, so
//! a photograph of any size fits the GPU device's 8192 px texture limit, and
//! the GUI thread copies only the tiles it uploads, never the whole level 0.
//! See `specs/gui/multi-panel-image-browser.md` § "Image Detail".
//!
//! Three rules shape what a frame draws:
//!
//! - **The level.** [`level_for_scale`]: the level whose pixels are about one
//!   physical screen pixel each at the current zoom.
//! - **The tiles.** Those of that level whose core meets the panel's visible
//!   part of the image ([`Layout::tiles_in`]). Each is drawn over the level-0
//!   pixels its core covers, with texture coordinates inset by the border
//!   texel, so linear filtering blends across a tile edge from the neighbour's
//!   own pixels and no seam shows.
//! - **The fallback.** A tile that is not uploaded yet is drawn from the
//!   nearest coarser level whose tiles over its area all are. At most
//!   [`MAX_UPLOADS_PER_FRAME`] tiles are uploaded in one frame, the overview
//!   level's first ([`Layout::overview_level`]), so a large jump draws coarse
//!   pixels at once and sharpens over the next frames, which the panel
//!   repaints for. At most [`MAX_TILE_TEXTURES`] textures are kept, the least
//!   recently drawn evicted first.

use std::collections::HashMap;

use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};

use crate::scene::{ImageRef, ReconId};

#[cfg(test)]
mod tests;

/// The side of a photograph tile's core, in pixels of its level.
pub(super) const TILE_SIZE: u32 = 512;

/// The most tile textures the panel keeps: about 270 MB of GPU memory at
/// 514 x 514 RGBA each.
pub(super) const MAX_TILE_TEXTURES: usize = 256;

/// The most tile textures one frame uploads, about 17 MB, so a jump to a new
/// place spreads its uploads over a few frames instead of stalling one.
pub(super) const MAX_UPLOADS_PER_FRAME: usize = 16;

/// The pyramid level drawn at `screen_px_per_source_px` physical screen
/// pixels per level-0 pixel: `round(log2(1 / s))`, clamped to the
/// photograph's `num_levels`.
///
/// A drawn texel then covers between 0.71 and 1.41 screen pixels wherever the
/// pyramid is deep enough, and at `s ≥ 0.71` the level is 0, full resolution.
/// A scale that is not a positive number draws level 0 (`NaN`) or the
/// coarsest level (zero).
pub(super) fn level_for_scale(screen_px_per_source_px: f32, num_levels: usize) -> usize {
    let coarsest = num_levels.saturating_sub(1);
    let level = (1.0 / screen_px_per_source_px).log2().round();
    if level.is_nan() || level <= 0.0 {
        0
    } else if level >= coarsest as f32 {
        coarsest
    } else {
        level as usize
    }
}

/// The size of each level of one photograph's pyramid, which is all the
/// arithmetic of where its tiles go needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    /// `[width, height]` of each level, level 0 first.
    sizes: Vec<[u32; 2]>,
}

impl Layout {
    /// The layout of `pyramid`.
    pub(super) fn of(pyramid: &ImageU8Pyramid) -> Self {
        Self {
            sizes: (0..pyramid.num_levels())
                .map(|level| {
                    let image = pyramid.level(level);
                    [image.width(), image.height()]
                })
                .collect(),
        }
    }

    /// The photograph's size: level 0's, in its own pixels.
    pub(super) fn size(&self) -> [u32; 2] {
        self.sizes[0]
    }

    /// How many levels the pyramid has.
    pub(super) fn num_levels(&self) -> usize {
        self.sizes.len()
    }

    /// The level the panel uploads first, so every later frame has something
    /// to fall back on: the finest level that is one tile, or the coarsest
    /// level when none is.
    pub(super) fn overview_level(&self) -> usize {
        self.sizes
            .iter()
            .position(|&[w, h]| w <= TILE_SIZE && h <= TILE_SIZE)
            .unwrap_or(self.sizes.len() - 1)
    }

    /// The tile's core, `[x0, y0, x1, y1]` in level-0 pixels: column `col`
    /// covers level-0 pixels `col·512·2^l` to `(col+1)·512·2^l`, clipped to
    /// the level's own width times `2^l`, and the same for rows.
    ///
    /// A level's width times `2^l` can fall short of level 0's width by less
    /// than one of its texels (each halving drops an odd last column), so the
    /// right and bottom edges of a coarse level end up to that much early.
    pub(super) fn core_rect(&self, level: usize, col: u32, row: u32) -> [f32; 4] {
        let [w, h] = self.sizes[level];
        let factor = (1u64 << level) as f32;
        let x0 = col * TILE_SIZE;
        let y0 = row * TILE_SIZE;
        let x1 = (x0 + TILE_SIZE).min(w);
        let y1 = (y0 + TILE_SIZE).min(h);
        [
            x0 as f32 * factor,
            y0 as f32 * factor,
            x1 as f32 * factor,
            y1 as f32 * factor,
        ]
    }

    /// The `(col, row)` of every tile of `level` whose core meets `rect`,
    /// `[x0, y0, x1, y1]` in level-0 pixels, row by row. A core that only
    /// touches the rectangle's edge does not meet it.
    pub(super) fn tiles_in(&self, level: usize, rect: [f32; 4]) -> Vec<(u32, u32)> {
        let [w, h] = self.sizes[level];
        let factor = (1u64 << level) as f32;
        let x0 = (rect[0] / factor).max(0.0);
        let y0 = (rect[1] / factor).max(0.0);
        let x1 = (rect[2] / factor).min(w as f32);
        let y1 = (rect[3] / factor).min(h as f32);
        if x1 <= x0 || y1 <= y0 {
            return Vec::new();
        }
        let tile = TILE_SIZE as f32;
        let (c0, c1) = ((x0 / tile).floor() as u32, (x1 / tile).ceil() as u32);
        let (r0, r1) = ((y0 / tile).floor() as u32, (y1 / tile).ceil() as u32);
        (r0..r1)
            .flat_map(|row| (c0..c1).map(move |col| (col, row)))
            .collect()
    }
}

/// Cut tile `(col, row)` out of `level` as the texture it is uploaded as: its
/// core and a one-pixel border all round, `(core + 2)` pixels on each side,
/// so at most 514 x 514.
///
/// The border holds the neighbouring tiles' pixels, and at the edge of the
/// level the edge pixel repeated, which is what a texture of the whole level
/// would read there with clamp-to-edge sampling.
pub(super) fn cut_tile(level: &ImageU8, col: u32, row: u32) -> egui::ColorImage {
    let (w, h) = (level.width(), level.height());
    let channels = level.channels() as usize;
    let x0 = col * TILE_SIZE;
    let y0 = row * TILE_SIZE;
    let x1 = (x0 + TILE_SIZE).min(w);
    let y1 = (y0 + TILE_SIZE).min(h);
    let (tw, th) = ((x1 - x0 + 2) as usize, (y1 - y0 + 2) as usize);
    let clamp = |v: i64, n: u32| v.clamp(0, i64::from(n) - 1) as usize;
    let data = level.data();
    let mut pixels = Vec::with_capacity(tw * th);
    for ty in 0..th {
        let sy = clamp(i64::from(y0) + ty as i64 - 1, h);
        let row_start = sy * w as usize * channels;
        for tx in 0..tw {
            let sx = clamp(i64::from(x0) + tx as i64 - 1, w);
            let p = &data[row_start + sx * channels..][..channels];
            pixels.push(match channels {
                1 => egui::Color32::from_gray(p[0]),
                2 => egui::Color32::from_rgba_unmultiplied(p[0], p[0], p[0], p[1]),
                3 => egui::Color32::from_rgb(p[0], p[1], p[2]),
                _ => egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]),
            });
        }
    }
    egui::ColorImage::new([tw, th], pixels)
}

/// One photograph tile, by the image it is of, its level, column and row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct PhotographTileKey {
    pub(super) image: ImageRef,
    pub(super) level: usize,
    pub(super) col: u32,
    pub(super) row: u32,
}

/// An uploaded tile.
struct Resident {
    texture: egui::TextureHandle,
    /// The frame that last drew it, for the least-recently-drawn eviction.
    last_drawn: u64,
}

/// The tile textures the panel holds, and the layout of the photograph it
/// shows.
pub(super) struct PhotographTiles {
    /// The layout of the image last drawn, kept so that its uploaded tiles can
    /// still be drawn on a frame the cache no longer holds its pyramid.
    layout: Option<(ImageRef, Layout)>,
    /// Every uploaded tile, of any image, at most [`MAX_TILE_TEXTURES`].
    resident: HashMap<PhotographTileKey, Resident>,
    /// Counts [`Self::draw`] calls, the clock the eviction reads.
    frame: u64,
}

impl PhotographTiles {
    pub(super) fn new() -> Self {
        Self {
            layout: None,
            resident: HashMap::new(),
            frame: 0,
        }
    }

    /// The layout of `image`: `pyramid`'s when the cache handed one over,
    /// otherwise the one kept from the last frame that drew `image`, or `None`
    /// when there is neither.
    ///
    /// A pyramid whose sizes differ from the kept layout of the same image
    /// (the file was replaced by one of another size) drops that image's
    /// tiles, which were cut from the old pixels.
    pub(super) fn layout(
        &mut self,
        image: ImageRef,
        pyramid: Option<&ImageU8Pyramid>,
    ) -> Option<&Layout> {
        if let Some(pyramid) = pyramid {
            let layout = Layout::of(pyramid);
            let changed = self
                .layout
                .as_ref()
                .is_some_and(|(kept, old)| *kept == image && *old != layout);
            if changed {
                self.resident.retain(|key, _| key.image != image);
            }
            self.layout = Some((image, layout));
        }
        self.layout
            .as_ref()
            .filter(|(kept, _)| *kept == image)
            .map(|(_, layout)| layout)
    }

    /// Draw `image` into `image_rect`, clipped by `painter`, from the tiles of
    /// the level [`level_for_scale`] picks at `scale` panel points per level-0
    /// pixel and `pixels_per_point`, uploading what is missing from `pyramid`
    /// when there is one.
    ///
    /// Needs a layout from [`Self::layout`] for `image` this frame, and draws
    /// nothing without one. Asks for a repaint while a visible tile is still
    /// to be uploaded from a pyramid at hand.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw(
        &mut self,
        ctx: &egui::Context,
        painter: &egui::Painter,
        image: ImageRef,
        pyramid: Option<&ImageU8Pyramid>,
        image_rect: egui::Rect,
        scale: f32,
        pixels_per_point: f32,
    ) {
        self.frame += 1;
        let Some(layout) = self
            .layout
            .as_ref()
            .filter(|(kept, _)| *kept == image)
            .map(|(_, layout)| layout.clone())
        else {
            return;
        };
        let visible = painter.clip_rect().intersect(image_rect);
        if !visible.is_positive() || scale.is_nan() || scale <= 0.0 {
            return;
        }
        let level = level_for_scale(scale * pixels_per_point, layout.num_levels());
        let to_source = |p: egui::Pos2| (p - image_rect.min) / scale;
        let (min, max) = (to_source(visible.min), to_source(visible.max));
        let visible = [min.x, min.y, max.x, max.y];
        let key = |level: usize, (col, row): (u32, u32)| PhotographTileKey {
            image,
            level,
            col,
            row,
        };

        if let Some(pyramid) = pyramid {
            let overview = layout.overview_level();
            let levels: &[usize] = if overview > level {
                &[overview, level]
            } else {
                &[level]
            };
            let mut uploads = 0;
            'upload: for &l in levels {
                for tile in layout.tiles_in(l, visible) {
                    if uploads == MAX_UPLOADS_PER_FRAME {
                        break 'upload;
                    }
                    let key = key(l, tile);
                    if !self.resident.contains_key(&key) {
                        self.upload(ctx, pyramid, key);
                        uploads += 1;
                    }
                }
            }
        }

        let mut pending = false;
        for tile in layout.tiles_in(level, visible) {
            let key = key(level, tile);
            let core = layout.core_rect(level, tile.0, tile.1);
            if self.resident.contains_key(&key) {
                self.paint(painter, key, core, image_rect, scale);
            } else {
                pending = true;
                self.paint_fallback(painter, &layout, key, core, image_rect, scale);
            }
        }
        if pending && pyramid.is_some() {
            ctx.request_repaint();
        }
    }

    /// Cut `key`'s tile from `pyramid` and upload it, evicting the least
    /// recently drawn tile first when the panel holds its limit.
    fn upload(&mut self, ctx: &egui::Context, pyramid: &ImageU8Pyramid, key: PhotographTileKey) {
        while self.resident.len() >= MAX_TILE_TEXTURES {
            let Some(oldest) = self
                .resident
                .iter()
                .min_by_key(|(_, resident)| resident.last_drawn)
                .map(|(key, _)| *key)
            else {
                break;
            };
            self.resident.remove(&oldest);
        }
        let pixels = cut_tile(pyramid.level(key.level), key.col, key.row);
        let texture = ctx.load_texture(
            format!(
                "detail_tile_{}_{}_{}_{}",
                key.image.index(),
                key.level,
                key.col,
                key.row
            ),
            pixels,
            egui::TextureOptions::LINEAR,
        );
        self.resident.insert(
            key,
            Resident {
                texture,
                last_drawn: self.frame,
            },
        );
    }

    /// Paint the part `area` (level-0 pixels, inside the tile's core) of the
    /// resident tile `key`.
    fn paint(
        &mut self,
        painter: &egui::Painter,
        key: PhotographTileKey,
        area: [f32; 4],
        image_rect: egui::Rect,
        scale: f32,
    ) {
        let Some(resident) = self.resident.get_mut(&key) else {
            return;
        };
        resident.last_drawn = self.frame;
        let [tw, th] = resident.texture.size();
        let factor = (1u64 << key.level) as f32;
        let (tile_x0, tile_y0) = ((key.col * TILE_SIZE) as f32, (key.row * TILE_SIZE) as f32);
        // Level-0 pixels to texture coordinates: into the level's pixels, then
        // into the tile's, then past the one-texel border.
        let uv = |x: f32, y: f32| {
            egui::pos2(
                (1.0 + x / factor - tile_x0) / tw as f32,
                (1.0 + y / factor - tile_y0) / th as f32,
            )
        };
        let screen = |x: f32, y: f32| image_rect.min + egui::vec2(x, y) * scale;
        painter.image(
            resident.texture.id(),
            egui::Rect::from_min_max(screen(area[0], area[1]), screen(area[2], area[3])),
            egui::Rect::from_min_max(uv(area[0], area[1]), uv(area[2], area[3])),
            egui::Color32::WHITE,
        );
    }

    /// Paint the core `core` of the missing tile `missing` from the nearest
    /// coarser level whose tiles over it are all resident, each clipped to
    /// the core. Paints nothing when no coarser level has the whole area.
    fn paint_fallback(
        &mut self,
        painter: &egui::Painter,
        layout: &Layout,
        missing: PhotographTileKey,
        core: [f32; 4],
        image_rect: egui::Rect,
        scale: f32,
    ) {
        for level in missing.level + 1..layout.num_levels() {
            let keys: Vec<PhotographTileKey> = layout
                .tiles_in(level, core)
                .into_iter()
                .map(|(col, row)| PhotographTileKey {
                    image: missing.image,
                    level,
                    col,
                    row,
                })
                .collect();
            if keys.is_empty() || !keys.iter().all(|key| self.resident.contains_key(key)) {
                continue;
            }
            for key in keys {
                let tile = layout.core_rect(level, key.col, key.row);
                let area = [
                    tile[0].max(core[0]),
                    tile[1].max(core[1]),
                    tile[2].min(core[2]),
                    tile[3].min(core[3]),
                ];
                self.paint(painter, key, area, image_rect, scale);
            }
            return;
        }
    }

    /// Drop the tiles and the layout of a reconstruction that has left the
    /// scene.
    pub(super) fn forget_recon(&mut self, id: ReconId) {
        self.resident.retain(|key, _| key.image.recon != id);
        if self
            .layout
            .as_ref()
            .is_some_and(|(image, _)| image.recon == id)
        {
            self.layout = None;
        }
    }

    /// Drop every tile and the layout.
    pub(super) fn clear(&mut self) {
        self.resident.clear();
        self.layout = None;
    }

    /// The tiles uploaded now, with their textures, in no order.
    #[cfg(test)]
    fn resident(&self) -> Vec<(PhotographTileKey, egui::TextureId)> {
        self.resident
            .iter()
            .map(|(key, resident)| (*key, resident.texture.id()))
            .collect()
    }

    /// Drop one uploaded tile, as an eviction would.
    #[cfg(test)]
    fn evict(&mut self, key: PhotographTileKey) {
        self.resident.remove(&key);
    }
}
