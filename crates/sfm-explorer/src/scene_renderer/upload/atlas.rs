// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Filling a texture atlas, one row of cells at a time.
//!
//! The tiles an atlas is built from are **tile-major**: one patch bitmap or one
//! thumbnail is a contiguous run of bytes. The atlas is **grid-major**, so a
//! tile's bytes belong to `resolution` different rows of the page, one stride
//! apart. The two layouts agree nowhere.
//!
//! Handing each tile to `wgpu::Queue::write_texture` at its own origin lets the
//! driver do that scatter. It is correct, and it is one call per tile. The call
//! is what an upload of small tiles costs rather than the bytes, so a 2 KiB
//! patch tile costs about what a 64 KiB thumbnail does, and an atlas is priced
//! by how many tiles it holds rather than how large they are. Tens of thousands
//! of patches is tens of thousands of calls.
//!
//! Doing the scatter here makes one row of cells a single contiguous image, and
//! its upload a single call: one per `cols` tiles rather than one per tile.
//!
//! **A row of cells rather than a whole page**, because a page is bounded only
//! by the GPU's 2D texture limit and can be most of a gigabyte, while a band is
//! `atlas_width` by `resolution` and stays small whatever the atlas does. The
//! calls a whole-page buffer would additionally save are the ones that stopped
//! mattering: a square-ish atlas of *n* tiles is `sqrt(n)` bands.

/// One row of atlas cells, filled tile by tile and uploaded in one call.
///
/// Reused across the bands of an atlas: [`Self::blank_from`] is what keeps the
/// cells no tile was placed in from showing the previous band's pixels.
pub(super) struct Band {
    /// `stride * resolution` bytes of RGBA, the full width of the atlas.
    pixels: Vec<u8>,
    /// Bytes per row of the atlas, which is this band's own row pitch.
    stride: usize,
    /// The tile edge in pixels. Tiles are square in both atlases.
    resolution: usize,
}

impl Band {
    /// A zeroed band across an atlas `atlas_width` pixels wide.
    pub(super) fn new(atlas_width: u32, resolution: u32) -> Self {
        let stride = atlas_width as usize * 4;
        Band {
            pixels: vec![0; stride * resolution as usize],
            stride,
            resolution: resolution as usize,
        }
    }

    /// Place one RGBA tile in cell `col`, scattering its rows across the band.
    ///
    /// `tile` is `resolution * resolution * 4` bytes, which is what an atlas's
    /// bitmap column hands over for one row of it.
    pub(super) fn place_rgba(&mut self, col: u32, tile: &[u8]) {
        let row_bytes = self.resolution * 4;
        debug_assert_eq!(tile.len(), row_bytes * self.resolution);
        let left = col as usize * row_bytes;
        for y in 0..self.resolution {
            let to = y * self.stride + left;
            let from = y * row_bytes;
            self.pixels[to..to + row_bytes].copy_from_slice(&tile[from..from + row_bytes]);
        }
    }

    /// The same for a tile stored as RGB, opaque in the alpha it gains.
    ///
    /// The expansion happens on the way into the band rather than into a buffer
    /// of its own, so a thumbnail atlas costs no per-tile allocation.
    pub(super) fn place_rgb(&mut self, col: u32, tile: &[u8]) {
        debug_assert_eq!(tile.len(), self.resolution * self.resolution * 3);
        let row_bytes = self.resolution * 4;
        let left = col as usize * row_bytes;
        for y in 0..self.resolution {
            let to = y * self.stride + left;
            let from = y * self.resolution * 3;
            let source = &tile[from..from + self.resolution * 3];
            let target = &mut self.pixels[to..to + row_bytes];
            for (pixel, out) in source
                .as_chunks::<3>()
                .0
                .iter()
                .zip(target.as_chunks_mut::<4>().0.iter_mut())
            {
                *out = [pixel[0], pixel[1], pixel[2], 255];
            }
        }
    }

    /// Zero the cells from `col` to the end of the band.
    ///
    /// Called once per band, with the first cell no tile was placed in, which
    /// is what a partly-filled last band leaves behind. Zeroing only the tail
    /// rather than the whole band before filling it keeps the memset off the
    /// full bands, which are all of them but one.
    pub(super) fn blank_from(&mut self, col: u32) {
        let left = col as usize * self.resolution * 4;
        if left >= self.stride {
            return;
        }
        for y in 0..self.resolution {
            let start = y * self.stride;
            self.pixels[start + left..start + self.stride].fill(0);
        }
    }

    /// The band's pixels, to hand to one `write_texture`.
    pub(super) fn bytes(&self) -> &[u8] {
        &self.pixels
    }

    /// The row pitch `write_texture` should read them at.
    pub(super) fn bytes_per_row(&self) -> u32 {
        self.stride as u32
    }
}

/// Upload one filled band into row `row` of page `page` of an atlas texture.
///
/// The band spans the atlas, so the copy always starts at x = 0 and is as wide
/// as the texture; only the row within the page varies.
pub(super) fn write_band(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    band: &Band,
    page: u32,
    row: u32,
    resolution: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: row * resolution,
                z: page,
            },
            aspect: wgpu::TextureAspect::All,
        },
        band.bytes(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(band.bytes_per_row()),
            rows_per_image: Some(resolution),
        },
        wgpu::Extent3d {
            width: texture.width(),
            height: resolution,
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests;
