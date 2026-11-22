// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a kernel reads a photograph through: [`TiledPhotograph`], its
//! [`PhotographTile`]s, and [`PhotographReadError`], why a read failed.
//!
//! A tiled photograph answers its layout and channels with no pixel read, and
//! gives its pixels a photograph tile or a rectangle of one level at a time.
//! The render entry points of [`sampler`](crate::camera::sampler) are what
//! read it; see `specs/core/camera/view-region-pyramids.md`.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::mipmap::{PyramidLayout, PHOTOGRAPH_TILE_SIZE};
use crate::camera::remap::HeldLevel;
use crate::progress::Cancelled;

/// Which photograph tile: its level, and its column and row in that level's
/// grid of tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub level: u8,
    pub col: u32,
    pub row: u32,
}

/// One photograph tile: a block of one level, with a one-pixel border of its
/// neighbours' pixels, or the edge pixel repeated at the image's edge.
pub struct PhotographTile {
    pub key: TileKey,
    /// The core's top-left pixel in level pixels.
    pub origin: [u32; 2],
    /// The core's size in level pixels.
    pub size: [u32; 2],
    /// `size + 2` on each axis: the core with its one-pixel border.
    pub pixels: ImageU8,
}

/// Why a kernel could not read the photographs it was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhotographReadError {
    Cancelled,
    /// A level, rectangle or tile the photograph does not have, or an empty
    /// one, was asked for; `path` is `None` for a photograph made in memory.
    OutOfRange {
        path: Option<PathBuf>,
        why: String,
    },
}

impl fmt::Display for PhotographReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(f, "{Cancelled}"),
            Self::OutOfRange {
                path: Some(path),
                why,
            } => write!(f, "{}: {why}", path.display()),
            Self::OutOfRange { path: None, why } => write!(f, "a photograph in memory: {why}"),
        }
    }
}

impl std::error::Error for PhotographReadError {}

impl From<Cancelled> for PhotographReadError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// What a kernel reads a photograph through.
///
/// It is made in memory: from an image, whose pyramid it builds, or from a
/// whole pyramid already built, read in place. A tile read copies the tile,
/// with its border, from its level; a rectangle read copies the rectangle
/// straight from its level, refused where a read of the tiles it covers
/// would be; and the photograph keeps no tile.
pub struct TiledPhotograph {
    pyramid: Arc<ImageU8Pyramid>,
    layout: PyramidLayout,
    channels: u32,
}

impl TiledPhotograph {
    /// Tiles cut from the image's pyramid of
    /// [`ImageU8Pyramid::full_levels`] depth, built by
    /// [`ImageU8Pyramid::from_image`], so the pixels the Python bindings and
    /// the kernels read today.
    pub fn from_image(image: ImageU8) -> Self {
        let levels = ImageU8Pyramid::full_levels(image.width(), image.height());
        Self::from_pyramid(Arc::new(ImageU8Pyramid::from_image(image, levels)))
    }

    /// A whole pyramid, read in place. Every read goes through its tiles, so
    /// a read of a tile, with its border, whose pixels a pyramid made by
    /// [`ImageU8Pyramid::from_regions`] does not hold is refused with
    /// [`PhotographReadError::OutOfRange`]. Its levels are the layout's: level `l`
    /// is level 0's size halved `l` times, as [`ImageU8::downsample_2x`]
    /// halves it.
    pub fn from_pyramid(pyramid: Arc<ImageU8Pyramid>) -> Self {
        let [width, height] = pyramid.level_full_size(0);
        let layout =
            PyramidLayout::in_memory(width, height, pyramid.num_levels(), PHOTOGRAPH_TILE_SIZE);
        let channels = pyramid.level(0).channels();
        Self {
            pyramid,
            layout,
            channels,
        }
    }

    /// The same photograph cut into tiles of `tile_size` px (a size of 0 is
    /// taken as 1); the tests cut small tiles, so that a small image crosses
    /// many of them.
    #[cfg(test)]
    pub(crate) fn with_tile_size(mut self, tile_size: u32) -> Self {
        let [width, height] = self.pyramid.level_full_size(0);
        self.layout = PyramidLayout::in_memory(width, height, self.pyramid.num_levels(), tile_size);
        self
    }

    /// Every level's size and the number of levels, read with no pixel.
    pub fn layout(&self) -> &PyramidLayout {
        &self.layout
    }

    /// The channels of every level's pixels.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// The pyramid the photograph is read from, for the tests that compare a
    /// read with the whole pyramid's; every kernel reads through the render
    /// entry points.
    #[cfg(test)]
    pub(crate) fn held_pyramid(&self) -> &ImageU8Pyramid {
        &self.pyramid
    }

    /// The tile `key`, with its border.
    pub(crate) fn tile(&self, key: TileKey) -> Result<Arc<PhotographTile>, PhotographReadError> {
        let level = key.level as usize;
        let (cols, rows) = self.layout.tiles(level);
        if key.col >= cols || key.row >= rows {
            return Err(out_of_range(format!(
                "level {level} has no tile at column {}, row {}",
                key.col, key.row
            )));
        }
        let ts = self.layout.tile_size();
        let (w, h) = self.layout.level_size(level);
        let origin = [key.col * ts, key.row * ts];
        let size = [ts.min(w - origin[0]), ts.min(h - origin[1])];
        let pixels = self.copy(
            level,
            [i64::from(origin[0]) - 1, i64::from(origin[1]) - 1],
            [size[0] + 2, size[1] + 2],
        )?;
        Ok(Arc::new(PhotographTile {
            key,
            origin,
            size,
            pixels,
        }))
    }

    /// The pixels `x0..x1` by `y0..y1` of level `level`: the pixels of the
    /// photograph tiles whose cores the rectangle covers, copied row by row
    /// straight from the level the photograph holds, so no tile is cut. A
    /// rectangle is refused where one of those tiles, with its border, would
    /// be, so a region read and a tile read refuse the same pixels.
    pub(crate) fn region(
        &self,
        level: usize,
        rect: crate::camera::sampler::LevelRect,
    ) -> Result<ImageU8, PhotographReadError> {
        check_rect(&self.layout, level, rect)?;
        // The tiles the rectangle covers, with their borders.
        let ts = self.layout.tile_size();
        let (w, h) = self.layout.level_size(level);
        let span = |lo: u32, hi: u32, full: u32| {
            let first = lo / ts * ts;
            let end = ((hi - 1) / ts * ts + ts).min(full);
            (i64::from(first) - 1, end - first + 2)
        };
        let (sx, nx) = span(rect.x0, rect.x1, w);
        let (sy, ny) = span(rect.y0, rect.y1, h);
        let held = self.held(level, [sx, sy], [nx, ny])?;
        let img = held.image();
        let [ox, oy] = held.origin().map(|o| o as u32);
        let c = self.channels as usize;
        let stride = img.width() as usize * c;
        let row_len = rect.width() as usize * c;
        let mut data = Vec::with_capacity(row_len * rect.height() as usize);
        for y in rect.y0..rect.y1 {
            let from = (y - oy) as usize * stride + (rect.x0 - ox) as usize * c;
            data.extend_from_slice(&img.data()[from..from + row_len]);
        }
        Ok(ImageU8::new(
            rect.width(),
            rect.height(),
            self.channels,
            data,
        ))
    }

    /// Level `level` as the pyramid holds it, once it is known to hold every
    /// pixel the `size` pixels from `start` reach with each address clamped
    /// to the level. Refuses pixels the pyramid does not hold, as for a
    /// pyramid made by [`ImageU8Pyramid::from_regions`] handed to
    /// [`from_pyramid`](Self::from_pyramid), rather than read other pixels.
    fn held(
        &self,
        level: usize,
        start: [i64; 2],
        size: [u32; 2],
    ) -> Result<HeldLevel<'_>, PhotographReadError> {
        let held = HeldLevel::of(&self.pyramid, level);
        let img = held.image();
        let [ox, oy] = held.origin().map(i64::from);
        let (w, h) = self.layout.level_size(level);
        // The span of full-level pixels the clamped addresses reach on one
        // axis lies inside the held rectangle.
        let holds = |start: i64, size: u32, origin: i64, full: u32, held: u32| {
            let last = i64::from(full) - 1;
            let lo = start.clamp(0, last);
            let hi = (start + i64::from(size) - 1).clamp(0, last);
            origin <= lo && hi < origin + i64::from(held)
        };
        if self.pyramid.level_full_size(level) != [w, h]
            || !holds(start[0], size[0], ox, w, img.width())
            || !holds(start[1], size[1], oy, h, img.height())
        {
            return Err(out_of_range(format!(
                "pixels of level {level} the pyramid does not hold"
            )));
        }
        Ok(held)
    }

    /// The `size` pixels from `start` of level `level`, each address clamped
    /// to the level, refused as [`Self::held`] refuses them. The columns that
    /// need no clamping are copied as one span a row.
    fn copy(
        &self,
        level: usize,
        start: [i64; 2],
        size: [u32; 2],
    ) -> Result<ImageU8, PhotographReadError> {
        let held = self.held(level, start, size)?;
        let img = held.image();
        let [ox, oy] = held.origin().map(i64::from);
        let (w, h) = self.layout.level_size(level);
        let c = self.channels as usize;
        let stride = img.width() as usize * c;
        let clamp = |v: i64, origin: i64, full: u32, held: u32| {
            (v.clamp(0, i64::from(full) - 1) - origin).clamp(0, i64::from(held) - 1) as usize
        };
        let cols: Vec<usize> = (0..i64::from(size[0]))
            .map(|i| clamp(start[0] + i, ox, w, img.width()) * c)
            .collect();
        // The run of columns whose held addresses follow one another, copied
        // with one slice copy a row; the clamped columns either side of it
        // pixel by pixel.
        let first = cols
            .windows(2)
            .position(|p| p[1] == p[0] + c)
            .unwrap_or(cols.len());
        let mut last = first;
        while last + 1 < cols.len() && cols[last + 1] == cols[last] + c {
            last += 1;
        }
        let mut data = Vec::with_capacity(size[0] as usize * size[1] as usize * c);
        for j in 0..i64::from(size[1]) {
            let row = &img.data()[clamp(start[1] + j, oy, h, img.height()) * stride..][..stride];
            for &col in &cols[..first.min(cols.len())] {
                data.extend_from_slice(&row[col..col + c]);
            }
            if first < cols.len() {
                data.extend_from_slice(&row[cols[first]..cols[last] + c]);
                for &col in &cols[last + 1..] {
                    data.extend_from_slice(&row[col..col + c]);
                }
            }
        }
        Ok(ImageU8::new(size[0], size[1], self.channels, data))
    }
}

/// Refuse a level the layout does not have, or an empty rectangle or one
/// that reaches past its level.
pub(crate) fn check_rect(
    layout: &PyramidLayout,
    level: usize,
    rect: crate::camera::sampler::LevelRect,
) -> Result<(), PhotographReadError> {
    if level >= layout.levels() {
        return Err(out_of_range(format!(
            "level {level} of a pyramid of {} levels",
            layout.levels()
        )));
    }
    let (w, h) = layout.level_size(level);
    if rect.x0 >= rect.x1 || rect.y0 >= rect.y1 || rect.x1 > w || rect.y1 > h {
        return Err(out_of_range(format!(
            "the rectangle {}..{} by {}..{} of level {level}, which is {w} x {h}",
            rect.x0, rect.x1, rect.y0, rect.y1
        )));
    }
    Ok(())
}

/// [`PhotographReadError::OutOfRange`] for a photograph made in memory.
pub(crate) fn out_of_range(why: String) -> PhotographReadError {
    PhotographReadError::OutOfRange { path: None, why }
}
