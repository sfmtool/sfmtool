// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a kernel reads a photograph through: [`TiledPhotograph`], its
//! [`PhotographTile`]s, and [`PhotographReadError`], why a read failed.
//!
//! A tiled photograph answers its layout and channels with no pixel read, and
//! gives its pixels a photograph tile or a rectangle of one level at a time.
//! It reads them from a whole pyramid, or from the photograph tiles it holds
//! and no others ([`TiledPhotograph::from_tiles`]). The render entry points of
//! [`sampler`](crate::camera::sampler) are what read it; see
//! `specs/core/camera/view-region-pyramids.md`.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::mipmap::{PyramidLayout, PHOTOGRAPH_TILE_SIZE};
use crate::camera::remap::HeldLevel;
use crate::camera::sampler::{LevelRect, LevelRegions};
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
    /// The read needs the photograph tiles `keys`, and the photograph, made
    /// by [`TiledPhotograph::from_tiles`], does not hold every one of them.
    /// `keys` are all the tiles the read needs, held or not, so a caller that
    /// makes the photograph again with those tiles can read it.
    NeedsTiles {
        keys: Vec<TileKey>,
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
            Self::NeedsTiles { keys } => write!(
                f,
                "the read needs {} photograph tiles, not all of which the photograph holds",
                keys.len()
            ),
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
/// would be; and the photograph keeps no tile. Or it is made of photograph
/// tiles ([`from_tiles`](Self::from_tiles)), which it holds and reads, and
/// holds no other pixel.
pub struct TiledPhotograph {
    pixels: Pixels,
    layout: PyramidLayout,
    channels: u32,
}

/// Where a [`TiledPhotograph`]'s pixels are.
enum Pixels {
    /// A whole pyramid, read in place.
    Pyramid(Arc<ImageU8Pyramid>),
    /// The photograph tiles held, by key; no other pixel.
    Tiles(HashMap<TileKey, Arc<PhotographTile>>),
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
            pixels: Pixels::Pyramid(pyramid),
            layout,
            channels,
        }
    }

    /// The photograph of `layout` made of the photograph tiles `tiles`, held
    /// while it lives, and no other pixel: what a viewer panel renders from
    /// on the GUI thread, made of the tiles it peeked in the photograph cache,
    /// so a tile the cache evicts after the peek is still read, and the
    /// render never decodes.
    ///
    /// Every read works out the tiles it needs before it reads a pixel, and a
    /// read that needs a tile the photograph does not hold is refused with
    /// [`PhotographReadError::NeedsTiles`], naming all the tiles it needs. A
    /// render entry point's read needs the tiles under its regions grown by
    /// [`REGION_REUSE_MARGIN`](crate::camera::sampler::REGION_REUSE_MARGIN),
    /// whichever way it then reads them. A read of tiles it holds gives the
    /// values the whole pyramid gives, bit for bit.
    ///
    /// Its channels are its tiles', so it refuses no tiles and tiles of
    /// different channels, and it refuses a tile the layout does not have or
    /// does not place and size as the tile says, each with
    /// [`PhotographReadError::OutOfRange`].
    pub fn from_tiles(
        layout: PyramidLayout,
        tiles: impl IntoIterator<Item = Arc<PhotographTile>>,
    ) -> Result<Self, PhotographReadError> {
        let mut held = HashMap::new();
        let mut channels = None;
        for tile in tiles {
            check_tile(&layout, &tile)?;
            let c = tile.pixels.channels();
            let first = *channels.get_or_insert(c);
            if first != c {
                return Err(out_of_range(format!(
                    "tiles of {first} and of {c} channels"
                )));
            }
            held.insert(tile.key, tile);
        }
        let Some(channels) = channels else {
            return Err(out_of_range("no tiles, so no channels".to_string()));
        };
        Ok(Self {
            pixels: Pixels::Tiles(held),
            layout,
            channels,
        })
    }

    /// The same photograph cut into tiles of `tile_size` px (a size of 0 is
    /// taken as 1): the photograph cache's tile size, and in the tests small
    /// tiles, so that a small image crosses many of them.
    pub(crate) fn with_tile_size(mut self, tile_size: u32) -> Self {
        if let Pixels::Pyramid(pyramid) = &self.pixels {
            let [width, height] = pyramid.level_full_size(0);
            self.layout = PyramidLayout::in_memory(width, height, pyramid.num_levels(), tile_size);
        }
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
    /// entry points. A photograph made of tiles has none.
    #[cfg(test)]
    pub(crate) fn held_pyramid(&self) -> &ImageU8Pyramid {
        match &self.pixels {
            Pixels::Pyramid(pyramid) => pyramid,
            Pixels::Tiles(_) => panic!("a photograph made of tiles holds no pyramid"),
        }
    }

    /// All the tiles a read of `regions` needs, when the photograph is made of
    /// tiles and does not hold every one of them; `None` when it can read
    /// them.
    pub(crate) fn missing_tiles(&self, regions: &LevelRegions) -> Option<Vec<TileKey>> {
        let Pixels::Tiles(held) = &self.pixels else {
            return None;
        };
        let keys: Vec<TileKey> = (0..regions.levels())
            .filter_map(|level| Some((level, regions.region(level)?)))
            .flat_map(|(level, rect)| covering(&self.layout, level, rect))
            .collect();
        (!keys.iter().all(|key| held.contains_key(key))).then_some(keys)
    }

    /// All the tiles a read of `rect` of level `level`, a rectangle inside its
    /// level, needs, when the photograph is made of tiles and does not hold
    /// every one of them; `None` when it can read them.
    pub(crate) fn missing_rect_tiles(&self, level: usize, rect: LevelRect) -> Option<Vec<TileKey>> {
        let Pixels::Tiles(held) = &self.pixels else {
            return None;
        };
        let keys: Vec<TileKey> = covering(&self.layout, level, rect).collect();
        (!keys.iter().all(|key| held.contains_key(key))).then_some(keys)
    }

    /// The tile `key`, with its border.
    pub(crate) fn tile(&self, key: TileKey) -> Result<Arc<PhotographTile>, PhotographReadError> {
        if let Pixels::Tiles(held) = &self.pixels {
            return held
                .get(&key)
                .cloned()
                .ok_or(PhotographReadError::NeedsTiles { keys: vec![key] });
        }
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
        rect: LevelRect,
    ) -> Result<ImageU8, PhotographReadError> {
        check_rect(&self.layout, level, rect)?;
        if let Pixels::Tiles(held) = &self.pixels {
            return self.region_of_tiles(held, level, rect);
        }
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

    /// [`Self::region`] of a photograph made of tiles: the part of each held
    /// tile's core inside `rect`, copied to its place.
    fn region_of_tiles(
        &self,
        held: &HashMap<TileKey, Arc<PhotographTile>>,
        level: usize,
        rect: LevelRect,
    ) -> Result<ImageU8, PhotographReadError> {
        if let Some(keys) = self.missing_rect_tiles(level, rect) {
            return Err(PhotographReadError::NeedsTiles { keys });
        }
        let c = self.channels as usize;
        let row_len = rect.width() as usize * c;
        let mut data = vec![0; row_len * rect.height() as usize];
        for key in covering(&self.layout, level, rect) {
            let tile = &held[&key];
            let [ox, oy] = tile.origin;
            let (x0, x1) = (rect.x0.max(ox), rect.x1.min(ox + tile.size[0]));
            let (y0, y1) = (rect.y0.max(oy), rect.y1.min(oy + tile.size[1]));
            let stride = tile.pixels.width() as usize * c;
            let span = (x1 - x0) as usize * c;
            for y in y0..y1 {
                // The core starts one row and one column inside the border.
                let from = (y - oy + 1) as usize * stride + (x0 - ox + 1) as usize * c;
                let to = (y - rect.y0) as usize * row_len + (x0 - rect.x0) as usize * c;
                data[to..to + span].copy_from_slice(&tile.pixels.data()[from..from + span]);
            }
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
        let Pixels::Pyramid(pyramid) = &self.pixels else {
            return Err(out_of_range(
                "a photograph made of tiles holds no pyramid".to_string(),
            ));
        };
        let held = HeldLevel::of(pyramid, level);
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
        if pyramid.level_full_size(level) != [w, h]
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

/// The keys of the tiles of level `level` of `layout` whose cores `rect`
/// covers, row by row; `rect` lies inside its level.
fn covering(
    layout: &PyramidLayout,
    level: usize,
    rect: LevelRect,
) -> impl Iterator<Item = TileKey> {
    let ts = layout.tile_size();
    let cols = rect.x0 / ts..=(rect.x1 - 1) / ts;
    (rect.y0 / ts..=(rect.y1 - 1) / ts).flat_map(move |row| {
        cols.clone().map(move |col| TileKey {
            level: level as u8,
            col,
            row,
        })
    })
}

/// Refuse a tile `layout` does not have, or one whose place, size or pixels
/// are not the ones the layout gives its key.
fn check_tile(layout: &PyramidLayout, tile: &PhotographTile) -> Result<(), PhotographReadError> {
    let TileKey { level, col, row } = tile.key;
    let level = usize::from(level);
    let refuse = |why: String| {
        Err(out_of_range(format!(
            "the tile at column {col}, row {row} of level {level} {why}"
        )))
    };
    if level >= layout.levels() {
        return refuse(format!("of a pyramid of {} levels", layout.levels()));
    }
    let (cols, rows) = layout.tiles(level);
    if col >= cols || row >= rows {
        return refuse(format!("of a level of {cols} x {rows} tiles"));
    }
    let ts = layout.tile_size();
    let (w, h) = layout.level_size(level);
    let origin = [col * ts, row * ts];
    let size = [ts.min(w - origin[0]), ts.min(h - origin[1])];
    if tile.origin != origin || tile.size != size {
        return refuse("is not placed and sized as the layout places it".to_string());
    }
    if [tile.pixels.width(), tile.pixels.height()] != [size[0] + 2, size[1] + 2]
        || tile.pixels.channels() == 0
    {
        return refuse("does not hold its core and its border".to_string());
    }
    Ok(())
}

/// Refuse a level the layout does not have, or an empty rectangle or one
/// that reaches past its level.
pub(crate) fn check_rect(
    layout: &PyramidLayout,
    level: usize,
    rect: LevelRect,
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
