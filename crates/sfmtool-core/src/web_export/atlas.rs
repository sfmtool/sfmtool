// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Atlas packing: square RGB tiles laid out in a grid over one or more pages,
//! each tile ringed by a one-texel border copied from its own edge.

use image::codecs::jpeg::JpegEncoder;
use image::ExtendedColorType;

/// Where the tiles of one atlas go: the same grid on every page, the last page
/// cut short to the rows it uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasLayout {
    /// Tile edge without the border, in texels.
    pub size: usize,
    /// Tile edge with the border, `size + 2`.
    pub tile: usize,
    /// Tiles per row, on every page.
    pub cols: usize,
    /// Rows on a full page.
    pub rows_per_page: usize,
    /// Tiles in the atlas.
    pub count: usize,
}

/// Where one tile landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasCell {
    /// Page index.
    pub page: usize,
    /// Cell index within the page, row major over [`AtlasLayout::cols`].
    pub cell: usize,
}

impl AtlasLayout {
    /// Lay `count` tiles of `size` texels (plus the border) over pages at most
    /// `max_page` texels on a side.
    ///
    /// A set that fits one page is laid out close to square, so a small scene
    /// writes a small page. A larger one fills full pages of
    /// `max_page / tile` columns and rows. `None` when there is nothing to lay
    /// out or one bordered tile is larger than a page.
    pub fn new(count: usize, size: usize, max_page: usize) -> Option<Self> {
        let tile = size + 2;
        let across = max_page / tile;
        if count == 0 || size == 0 || across == 0 {
            return None;
        }
        let (cols, rows_per_page) = if count <= across * across {
            let cols = (count as f64).sqrt().ceil() as usize;
            let cols = cols.clamp(1, across);
            (cols, count.div_ceil(cols))
        } else {
            (across, across)
        };
        Some(Self {
            size,
            tile,
            cols,
            rows_per_page,
            count,
        })
    }

    /// Tiles on a full page.
    pub fn per_page(&self) -> usize {
        self.cols * self.rows_per_page
    }

    /// Pages the atlas takes.
    pub fn page_count(&self) -> usize {
        self.count.div_ceil(self.per_page())
    }

    /// Width and height of page `page`, in texels.
    pub fn page_dims(&self, page: usize) -> (usize, usize) {
        let on_page = (self.count - page * self.per_page()).min(self.per_page());
        (
            self.cols * self.tile,
            on_page.div_ceil(self.cols) * self.tile,
        )
    }

    /// Where tile `k` goes.
    pub fn place(&self, k: usize) -> AtlasCell {
        AtlasCell {
            page: k / self.per_page(),
            cell: k % self.per_page(),
        }
    }

    /// Texel of the top-left corner of `cell`'s bordered tile on its page.
    pub fn origin(&self, cell: usize) -> (usize, usize) {
        (
            (cell % self.cols) * self.tile,
            (cell / self.cols) * self.tile,
        )
    }

    /// The RGB pages, with tile `k` taken from `tile_rgb(k)`, `size * size * 3`
    /// bytes row major. Texels no tile covers are black.
    pub fn pack(&self, tile_rgb: impl Fn(usize) -> Vec<u8>) -> Vec<RgbPage> {
        let mut pages: Vec<RgbPage> = (0..self.page_count())
            .map(|page| {
                let (width, height) = self.page_dims(page);
                RgbPage {
                    width,
                    height,
                    pixels: vec![0; width * height * 3],
                }
            })
            .collect();
        for k in 0..self.count {
            let AtlasCell { page, cell } = self.place(k);
            let (x0, y0) = self.origin(cell);
            let rgb = tile_rgb(k);
            write_bordered_tile(&mut pages[page], x0, y0, &rgb, self.size);
        }
        pages
    }
}

/// One atlas page before encoding.
#[derive(Debug, Clone)]
pub struct RgbPage {
    /// Width in texels.
    pub width: usize,
    /// Height in texels.
    pub height: usize,
    /// `height` rows of `width` RGB texels.
    pub pixels: Vec<u8>,
}

impl RgbPage {
    /// The page as a baseline JPEG at `quality` (1 to 100).
    pub fn to_jpeg(&self, quality: u8) -> Result<Vec<u8>, image::ImageError> {
        let mut bytes = Vec::new();
        JpegEncoder::new_with_quality(&mut bytes, quality.clamp(1, 100)).encode(
            &self.pixels,
            self.width as u32,
            self.height as u32,
            ExtendedColorType::Rgb8,
        )?;
        Ok(bytes)
    }
}

/// Copy a `size` x `size` RGB tile into `page` with its top-left bordered
/// corner at `(x0, y0)`: the tile at `(x0 + 1, y0 + 1)`, and the ring around it
/// a copy of the tile's nearest edge texel, so bilinear sampling at the tile's
/// edge reads the tile and never its neighbour.
fn write_bordered_tile(page: &mut RgbPage, x0: usize, y0: usize, rgb: &[u8], size: usize) {
    debug_assert_eq!(rgb.len(), size * size * 3);
    let tile = size + 2;
    for ty in 0..tile {
        let sy = ty.saturating_sub(1).min(size - 1);
        let row = (y0 + ty) * page.width;
        for tx in 0..tile {
            let sx = tx.saturating_sub(1).min(size - 1);
            let src = (sy * size + sx) * 3;
            let dst = (row + x0 + tx) * 3;
            page.pixels[dst..dst + 3].copy_from_slice(&rgb[src..src + 3]);
        }
    }
}
