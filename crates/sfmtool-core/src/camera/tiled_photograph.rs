// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a kernel reads a photograph through: [`TiledPhotograph`], its
//! [`PhotographTile`]s, and [`PhotographReadError`], why a read failed.
//!
//! A tiled photograph answers its layout and channels with no pixel read, and
//! gives its pixels a photograph tile or a rectangle of one level at a time.
//! It reads them from the photograph tiles it holds: every tile of an image's
//! pyramid ([`TiledPhotograph::from_image`]), or the tiles it was made of
//! ([`TiledPhotograph::from_tiles`]); or, for one the photograph cache's
//! `get` answers, from the cache, which decodes the photograph again for a
//! tile it has evicted. The render entry points of
//! [`sampler`](crate::camera::sampler) are what read it; see
//! `specs/core/camera/view-region-pyramids.md`.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::mipmap::{PyramidLayout, MAX_JPEG_SIDE, PHOTOGRAPH_TILE_SIZE};
use crate::camera::photograph_cache::{CachedTiles, GetError, TileError};
use crate::camera::remap::HeldLevel;
use crate::camera::sampler::{LevelRect, LevelRegions};
use crate::progress::{Cancelled, Progress};

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
    /// The file of a photograph the cache answered is no longer the one its
    /// tiles were cut from (its length, modification time or size differ),
    /// found when a tile evicted after the answer was read again; or, for an
    /// out-of-core photograph read from its mipmap, a mip level file it
    /// reads is missing, not current, or holds another record than the one
    /// the cache read first. `path` names the file. A kernel never goes on
    /// with pixels of two versions of one photograph.
    Changed {
        path: PathBuf,
    },
    /// The photograph's file does not exist ([`GetError::NotFound`]): for a
    /// photograph the cache answered, it was removed before a tile evicted
    /// since the answer was read again.
    NotFound {
        path: PathBuf,
    },
    /// A side over [`MAX_JPEG_SIDE`] px, read from the header
    /// ([`GetError::TooLarge`]).
    TooLarge {
        path: PathBuf,
        width: u32,
        height: u32,
    },
    /// The photograph's file could not be read or decoded, or its workspace
    /// file could not be read or holds an invalid `mipmap_options`
    /// ([`GetError::Workspace`]); `why` names the file. `path` is `None` when
    /// the error came from a [`GetError`], whose message alone names it.
    Unreadable {
        path: Option<PathBuf>,
        why: String,
    },
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
            Self::Changed { path } => write!(
                f,
                "{} changed while it was being read; run it again",
                path.display()
            ),
            Self::NotFound { path } => write!(f, "{}: no such file", path.display()),
            Self::TooLarge {
                path,
                width,
                height,
            } => write!(
                f,
                "{} is {width} x {height}, and a photograph with a side over \
                 {MAX_JPEG_SIDE} px cannot be read",
                path.display()
            ),
            Self::Unreadable { why, .. } => write!(f, "{why}"),
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

/// A failed tile, region or level read of the cache as a kernel reports it:
/// `Changed`, `Unreadable` and `OutOfRange` keep their path.
impl From<TileError> for PhotographReadError {
    fn from(e: TileError) -> Self {
        match e {
            TileError::Cancelled => Self::Cancelled,
            TileError::Changed { path } => Self::Changed { path },
            TileError::Unreadable { path, why } => Self::Unreadable {
                path: Some(path),
                why,
            },
            TileError::OutOfRange { path, why } => Self::OutOfRange {
                path: Some(path),
                why,
            },
        }
    }
}

/// A failed `get` as a kernel reports it; `Unreadable` has no path, since
/// the message of [`GetError::Unreadable`] alone names the file, and that of
/// [`GetError::Workspace`] the workspace file.
impl From<GetError> for PhotographReadError {
    fn from(e: GetError) -> Self {
        match e {
            GetError::Cancelled => Self::Cancelled,
            GetError::NotFound(path) => Self::NotFound { path },
            GetError::TooLarge {
                path,
                width,
                height,
            } => Self::TooLarge {
                path,
                width,
                height,
            },
            GetError::Unreadable(why) => Self::Unreadable { path: None, why },
            GetError::Workspace(e) => Self::Unreadable {
                path: None,
                why: e.to_string(),
            },
        }
    }
}

/// What a kernel reads a photograph through.
///
/// It reads photograph tiles, and holds no other pixel: every tile of an
/// image's pyramid, cut when it is made ([`from_image`](Self::from_image)),
/// the tiles it is given ([`from_tiles`](Self::from_tiles)), or, for one
/// [`PhotographCache::get`](crate::camera::PhotographCache::get) or
/// [`PhotographCache::get_many`](crate::camera::PhotographCache::get_many)
/// answers, none: it holds its layout and the cache's handle on its tiles,
/// and reads each tile through the cache, which decodes the photograph again
/// for a tile it has evicted. Such a read can fail as a read of the file can
/// ([`PhotographReadError::NotFound`], [`PhotographReadError::Changed`] and
/// the rest), and polls the reader's `Progress` while it decodes.
pub struct TiledPhotograph {
    /// The photograph tiles held, by key.
    held: HashMap<TileKey, Arc<PhotographTile>>,
    /// The tiles held are every tile of the layout, so no read checks for a
    /// tile the photograph lacks.
    every: bool,
    layout: PyramidLayout,
    channels: u32,
    /// The cache the photograph reads its tiles through, for one the cache
    /// answered; it then holds no tile.
    through: Option<CachedTiles>,
}

/// Every photograph tile of an image's pyramid, as [`cut_image`] cuts them.
pub(crate) struct CutImage {
    /// The pyramid's layout: its own levels, and no stored level.
    pub(crate) layout: PyramidLayout,
    /// Every tile of the layout, each with its border.
    pub(crate) tiles: HashMap<TileKey, Arc<PhotographTile>>,
}

/// Every photograph tile, each with its border, of the pyramid of
/// [`ImageU8Pyramid::full_levels`] depth that [`ImageU8Pyramid::from_image`]
/// builds of `image`, cut at `tile_size` px (a size of 0 is taken as 1):
/// what [`TiledPhotograph::from_image`] holds, and what the photograph cache
/// keeps of a decode. A read of the tiles gives the values the whole pyramid
/// gives, bit for bit.
///
/// The pyramid is never held whole: each level is cut and dropped, and the
/// next is made from the tiles just cut, with the 2 x 2 box filter of
/// [`ImageU8::downsample_2x`], so the call holds at most the tiles cut so far
/// and one level, about 2.0 times the image's bytes at its peak, as it cuts
/// level 0 (the image and its tiles), less than the pyramid and its tiles
/// together.
pub(crate) fn cut_image(image: ImageU8, tile_size: u32) -> CutImage {
    let full = ImageU8Pyramid::full_levels(image.width(), image.height());
    // The levels `ImageU8Pyramid::from_image` builds: it stops before a
    // level that would have a side of 0.
    let (mut w, mut h, mut levels) = (image.width(), image.height(), 1);
    while levels < full && w >= 2 && h >= 2 {
        (w, h, levels) = (w / 2, h / 2, levels + 1);
    }
    let layout = PyramidLayout::in_memory(image.width(), image.height(), levels, tile_size);
    // One level at a time: cut it and drop it, then make the next from its
    // tiles, so no more than one level is held beside the tiles.
    let mut tiles = HashMap::new();
    let mut current = image;
    for level in 0..levels {
        cut_level(&layout, level, &current, &mut tiles);
        if level + 1 < levels {
            let channels = current.channels();
            drop(current);
            current = downsample_tiles(&layout, level, channels, &tiles);
        }
    }
    CutImage { layout, tiles }
}

/// Level `level + 1` of `layout`'s pyramid, made from the tiles of level
/// `level` in `tiles` as [`ImageU8::downsample_2x`] makes it from the whole
/// level, bit for bit: each output pixel is the rounded mean of the 2 x 2
/// block at twice its position. The block's top-left pixel lies in one tile's
/// core, and the rest of it in that core or the tile's one-pixel border,
/// which holds the neighbouring tiles' pixels; it never reaches past the
/// level, since `downsample_2x` drops an odd last row and column. So any
/// tile size, odd ones included, gives the same pixels.
fn downsample_tiles(
    layout: &PyramidLayout,
    level: usize,
    channels: u32,
    tiles: &HashMap<TileKey, Arc<PhotographTile>>,
) -> ImageU8 {
    let (w, h) = layout.level_size(level);
    let (out_w, out_h) = (w / 2, h / 2);
    let c = channels as usize;
    let out_stride = out_w as usize * c;
    let mut data = vec![0u8; out_stride * out_h as usize];
    let (cols, rows) = layout.tiles(level);
    for row in 0..rows {
        for col in 0..cols {
            let key = TileKey {
                level: level as u8,
                col,
                row,
            };
            let Some(tile) = tiles.get(&key) else {
                continue;
            };
            let [x0, y0] = tile.origin;
            let [sw, sh] = tile.size;
            // The output pixels whose block starts in this tile's core.
            let (ox0, ox1) = (x0.div_ceil(2), ((x0 + sw).div_ceil(2)).min(out_w));
            let (oy0, oy1) = (y0.div_ceil(2), ((y0 + sh).div_ceil(2)).min(out_h));
            let stride = tile.pixels.width() as usize * c;
            let src = tile.pixels.data();
            for oy in oy0..oy1 {
                // One row and one column of border before the core.
                let ly = (2 * oy - y0 + 1) as usize;
                let row0 = &src[ly * stride..][..stride];
                let row1 = &src[(ly + 1) * stride..][..stride];
                let out_row = &mut data[oy as usize * out_stride..][..out_stride];
                for ox in ox0..ox1 {
                    let lx = (2 * ox - x0 + 1) as usize;
                    for ch in 0..c {
                        let v00 = u16::from(row0[lx * c + ch]);
                        let v10 = u16::from(row0[(lx + 1) * c + ch]);
                        let v01 = u16::from(row1[lx * c + ch]);
                        let v11 = u16::from(row1[(lx + 1) * c + ch]);
                        out_row[ox as usize * c + ch] = ((v00 + v10 + v01 + v11 + 2) / 4) as u8;
                    }
                }
            }
        }
    }
    ImageU8::new(out_w, out_h, channels, data)
}

impl TiledPhotograph {
    /// Every photograph tile of the image's pyramid of
    /// [`ImageU8Pyramid::full_levels`] depth, the pyramid
    /// [`ImageU8Pyramid::from_image`] builds, cut at [`PHOTOGRAPH_TILE_SIZE`]
    /// and held. A read gives the values the whole pyramid gives, bit for
    /// bit.
    ///
    /// The pyramid is never held whole: each level is cut and dropped, and
    /// the next made from its tiles as [`ImageU8::downsample_2x`] makes it
    /// from the level, so the call holds at most the tiles cut so far and one
    /// level, about 2.0 times the image's bytes at its peak, less than the
    /// pyramid and its tiles together.
    pub fn from_image(image: ImageU8) -> Self {
        let channels = image.channels();
        let CutImage { layout, tiles } = cut_image(image, PHOTOGRAPH_TILE_SIZE);
        Self {
            held: tiles,
            every: true,
            layout,
            channels,
            through: None,
        }
    }

    /// Every photograph tile of `pyramid`, a pyramid of whole levels such as
    /// [`ImageU8Pyramid::from_image`] builds, cut at `tile_size` px (a size
    /// of 0 is taken as 1) and held, as [`Self::from_tiles`] holds them: for
    /// the tests that build a pyramid of fewer levels than the full depth,
    /// and the test that holds [`Self::from_image`]'s tiles, cut level by
    /// level, to the whole pyramid's.
    #[cfg(test)]
    pub(crate) fn cut(pyramid: &ImageU8Pyramid, tile_size: u32) -> Self {
        debug_assert!(
            (0..pyramid.num_levels()).all(|l| pyramid.level_origin(l) == [0, 0]
                && pyramid.level_full_size(l)
                    == [pyramid.level(l).width(), pyramid.level(l).height()]),
            "a pyramid of whole levels"
        );
        let levels: Vec<&ImageU8> = (0..pyramid.num_levels())
            .map(|l| pyramid.level(l))
            .collect();
        Self::cut_levels(&levels, tile_size)
    }

    /// Every photograph tile of the whole levels `levels`, each level 0's
    /// size halved, cut at `tile_size` px and held.
    #[cfg(test)]
    fn cut_levels(levels: &[&ImageU8], tile_size: u32) -> Self {
        let (width, height) = (levels[0].width(), levels[0].height());
        let layout = PyramidLayout::in_memory(width, height, levels.len(), tile_size);
        let channels = levels[0].channels();
        let mut held = HashMap::new();
        for (level, image) in levels.iter().enumerate() {
            cut_level(&layout, level, image, &mut held);
        }
        Self {
            held,
            every: true,
            layout,
            channels,
            through: None,
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
        let every = held.len() == tile_count(&layout);
        Ok(Self {
            held,
            every,
            layout,
            channels,
            through: None,
        })
    }

    /// The photograph of `layout` and `channels` that holds no tile and reads
    /// each through `through`: what the photograph cache's `get` answers.
    pub(crate) fn read_through(layout: PyramidLayout, channels: u32, through: CachedTiles) -> Self {
        Self {
            held: HashMap::new(),
            every: false,
            layout,
            channels,
            through: Some(through),
        }
    }

    /// The number of photograph tiles the photograph holds: none for one
    /// that reads through the cache.
    #[cfg(test)]
    pub(crate) fn held_tiles(&self) -> usize {
        self.held.len()
    }

    /// The same photograph cut into tiles of `tile_size` px (a size of 0 is
    /// taken as 1): for the tests, small tiles, so that a small image
    /// crosses many of them. A photograph that
    /// holds every tile of its layout has its levels put together from its
    /// tiles and cut again; one made by [`Self::from_tiles`] that lacks a
    /// tile, or one that reads through the cache, is left as it is.
    #[cfg(test)]
    pub(crate) fn with_tile_size(self, tile_size: u32) -> Self {
        if !self.every {
            return self;
        }
        let Ok(levels) = (0..self.layout.levels())
            .map(|level| self.whole_level(level))
            .collect::<Result<Vec<_>, _>>()
        else {
            // An empty photograph has no tile to cut again.
            return self;
        };
        Self::cut_levels(&levels.iter().collect::<Vec<_>>(), tile_size)
    }

    /// The whole of level `level`, put together from the tiles that cover it:
    /// for [`Self::with_tile_size`], and for the tests that need a level's
    /// pixels and do not compare a read with them.
    #[cfg(test)]
    pub(crate) fn whole_level(&self, level: usize) -> Result<ImageU8, PhotographReadError> {
        let (w, h) = self.layout.level_size(level);
        let rect = LevelRect {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        };
        self.region(level, rect, &Progress::none())
    }

    /// Every level's size and the number of levels, read with no pixel.
    pub fn layout(&self) -> &PyramidLayout {
        &self.layout
    }

    /// The channels of every level's pixels.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// The tiles held, when the photograph may lack one a read needs; `None`
    /// when it holds every tile of its layout or reads them through the
    /// cache.
    fn lacking(&self) -> Option<&HashMap<TileKey, Arc<PhotographTile>>> {
        (!self.every && self.through.is_none()).then_some(&self.held)
    }

    /// All the tiles a read of `regions` needs, when the photograph does not
    /// hold every one of them; `None` when it can read them.
    pub(crate) fn missing_tiles(&self, regions: &LevelRegions) -> Option<Vec<TileKey>> {
        let held = self.lacking()?;
        let keys: Vec<TileKey> = (0..regions.levels())
            .filter_map(|level| Some((level, regions.region(level)?)))
            .flat_map(|(level, rect)| covering(&self.layout, level, rect))
            .collect();
        (!keys.iter().all(|key| held.contains_key(key))).then_some(keys)
    }

    /// Whether the photograph does not hold every one of `keys`.
    pub(crate) fn missing_keys(&self, keys: &[TileKey]) -> bool {
        self.lacking()
            .is_some_and(|held| !keys.iter().all(|key| held.contains_key(key)))
    }

    /// Whether the photograph may lack a tile a read needs, made by
    /// [`Self::from_tiles`] of fewer than every tile, so the read has to work
    /// out the tiles it needs before it reads a pixel.
    pub(crate) fn may_lack_tiles(&self) -> bool {
        self.lacking().is_some()
    }

    /// All the tiles a read of `rect` of level `level`, a rectangle inside its
    /// level, needs, when the photograph does not hold every one of them;
    /// `None` when it can read them.
    pub(crate) fn missing_rect_tiles(&self, level: usize, rect: LevelRect) -> Option<Vec<TileKey>> {
        let held = self.lacking()?;
        let keys: Vec<TileKey> = covering(&self.layout, level, rect).collect();
        (!keys.iter().all(|key| held.contains_key(key))).then_some(keys)
    }

    /// The tile `key`, with its border; [`PhotographReadError::NeedsTiles`]
    /// naming it when the photograph does not hold it and does not read
    /// through the cache.
    ///
    /// `progress` is the caller's, which a read through the cache that has
    /// to decode the photograph again polls for cancellation; a photograph
    /// that holds its tiles decodes nothing and never polls it. Every read of
    /// a tiled photograph takes it, so that a render entry point passes its
    /// caller's down to the read.
    pub(crate) fn tile(
        &self,
        key: TileKey,
        progress: &Progress<'_>,
    ) -> Result<Arc<PhotographTile>, PhotographReadError> {
        if let Some(tile) = self.held.get(&key) {
            return Ok(Arc::clone(tile));
        }
        match &self.through {
            Some(through) => through
                .read(&[key], progress)?
                .into_iter()
                .next()
                .ok_or(PhotographReadError::NeedsTiles { keys: vec![key] }),
            None => Err(PhotographReadError::NeedsTiles { keys: vec![key] }),
        }
    }

    /// The tiles `keys`, by key: those held, or, for a photograph that reads
    /// through the cache, all of them in one read, which decodes the
    /// photograph at most once. [`PhotographReadError::NeedsTiles`] naming
    /// them all when the photograph does not hold every one and does not
    /// read through the cache. `progress` is as for [`Self::tile`].
    pub(crate) fn tiles(
        &self,
        keys: &[TileKey],
        progress: &Progress<'_>,
    ) -> Result<HashMap<TileKey, Arc<PhotographTile>>, PhotographReadError> {
        if let Some(through) = &self.through {
            let tiles = through.read(keys, progress)?;
            return Ok(keys.iter().copied().zip(tiles).collect());
        }
        keys.iter()
            .map(|&key| match self.held.get(&key) {
                Some(tile) => Ok((key, Arc::clone(tile))),
                None => Err(PhotographReadError::NeedsTiles {
                    keys: keys.to_vec(),
                }),
            })
            .collect()
    }

    /// The pixels `x0..x1` by `y0..y1` of level `level`: the part of each
    /// photograph tile's core inside the rectangle, copied to its place.
    /// Refused with [`PhotographReadError::NeedsTiles`] when the photograph
    /// does not hold every tile the rectangle covers and does not read
    /// through the cache. `progress` is as for [`Self::tile`].
    pub(crate) fn region(
        &self,
        level: usize,
        rect: LevelRect,
        progress: &Progress<'_>,
    ) -> Result<ImageU8, PhotographReadError> {
        check_rect(&self.layout, level, rect)?;
        if let Some(keys) = self.missing_rect_tiles(level, rect) {
            return Err(PhotographReadError::NeedsTiles { keys });
        }
        let keys: Vec<TileKey> = covering(&self.layout, level, rect).collect();
        let tiles = self.tiles(&keys, progress)?;
        Ok(self.region_of(level, rect, &tiles))
    }

    /// The pixels of `rect`, a rectangle inside level `level`, copied from
    /// `tiles`, which hold every tile it covers.
    pub(crate) fn region_of(
        &self,
        level: usize,
        rect: LevelRect,
        tiles: &HashMap<TileKey, Arc<PhotographTile>>,
    ) -> ImageU8 {
        let c = self.channels as usize;
        let row_len = rect.width() as usize * c;
        let mut data = vec![0; row_len * rect.height() as usize];
        for key in covering(&self.layout, level, rect) {
            let tile = &tiles[&key];
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
        ImageU8::new(rect.width(), rect.height(), self.channels, data)
    }
}

/// The `size` pixels from `start` of a level of size `full`, each address
/// clamped to the level, of which `held` holds every pixel the clamped
/// addresses reach: a tile with its border, as [`cut_level`] cuts it. The
/// columns that need no clamping are copied as one span a row.
fn copy_clamped(held: HeldLevel<'_>, full: (u32, u32), start: [i64; 2], size: [u32; 2]) -> ImageU8 {
    let img = held.image();
    let [ox, oy] = held.origin().map(i64::from);
    let (w, h) = full;
    let c = img.channels() as usize;
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
    ImageU8::new(size[0], size[1], img.channels(), data)
}

/// The photograph tile `key` of `layout`, with its border, from `region`, a
/// rectangle of its level whose top-left pixel is `origin` and which holds
/// the tile's core and every border pixel inside the level: what the cache
/// cuts from a rectangle it decoded from a mip level file or an original. The
/// border repeats the edge pixel at the level's edge, as [`cut_level`] cuts
/// it from the whole level, so the two give the same tile.
pub(crate) fn tile_from_region(
    layout: &PyramidLayout,
    key: TileKey,
    region: &ImageU8,
    origin: [u32; 2],
) -> PhotographTile {
    let ts = layout.tile_size();
    let (w, h) = layout.level_size(usize::from(key.level));
    let core = [key.col * ts, key.row * ts];
    let size = [ts.min(w - core[0]), ts.min(h - core[1])];
    // A level's side is under 2^16, so its origin fits.
    let held = HeldLevel::at(region, [origin[0] as i32, origin[1] as i32]);
    let pixels = copy_clamped(
        held,
        (w, h),
        [i64::from(core[0]) - 1, i64::from(core[1]) - 1],
        [size[0] + 2, size[1] + 2],
    );
    PhotographTile {
        key,
        origin: core,
        size,
        pixels,
    }
}

/// Every photograph tile of level `level` of `layout`, whose whole pixels are
/// `image`, each with its border, into `held`.
pub(crate) fn cut_level(
    layout: &PyramidLayout,
    level: usize,
    image: &ImageU8,
    held: &mut HashMap<TileKey, Arc<PhotographTile>>,
) {
    let ts = layout.tile_size();
    let (cols, rows) = layout.tiles(level);
    let (w, h) = layout.level_size(level);
    for row in 0..rows {
        for col in 0..cols {
            let key = TileKey {
                level: level as u8,
                col,
                row,
            };
            let origin = [col * ts, row * ts];
            let size = [ts.min(w - origin[0]), ts.min(h - origin[1])];
            let pixels = copy_clamped(
                HeldLevel::whole(image),
                (w, h),
                [i64::from(origin[0]) - 1, i64::from(origin[1]) - 1],
                [size[0] + 2, size[1] + 2],
            );
            let tile = PhotographTile {
                key,
                origin,
                size,
                pixels,
            };
            held.insert(key, Arc::new(tile));
        }
    }
}

/// The number of tiles of every level of `layout`.
pub(crate) fn tile_count(layout: &PyramidLayout) -> usize {
    (0..layout.levels())
        .map(|level| {
            let (cols, rows) = layout.tiles(level);
            cols as usize * rows as usize
        })
        .sum()
}

/// The keys of the tiles of level `level` of `layout` whose cores `rect`
/// covers, row by row; `rect` lies inside its level.
pub(crate) fn covering(
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

#[cfg(test)]
mod tests;
