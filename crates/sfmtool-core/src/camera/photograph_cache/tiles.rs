// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cache's layout and tile requests: a photograph's layout, its
//! photograph tiles, a view-region pyramid or one level whole, each cut from
//! the whole pyramid the cache holds when it is asked for.
//!
//! A panel asks [`PhotographCache::layout`] and [`PhotographCache::peek_tile`]
//! on the GUI thread, which never read a file or wait, and asks on the rayon
//! pool for what is missing: [`PhotographCache::request_layout`] reads the
//! header, and [`PhotographCache::try_decode_tiles`] decodes the photograph
//! without waiting on its slot or on the build count. A kernel's reads,
//! [`PhotographCache::get_tile`], [`PhotographCache::view_region_pyramid`]
//! and [`PhotographCache::get_level`], decode as [`PhotographCache::get`]
//! does. See the spec `specs/core/camera/photograph-cache.md` § "Layouts and
//! tiles".

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{
    header, whole_decode_bytes, GetError, Making, PhotographCache, Share, Slot, SlotState,
};
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::mipmap::PyramidLayout;
use crate::camera::sampler::{copy_regions, LevelRect, LevelRegions};
use crate::camera::tiled_photograph::{
    PhotographReadError, PhotographTile, TileKey, TiledPhotograph,
};
use crate::progress::{Cancelled, Progress};

#[cfg(test)]
mod tests;

/// What [`PhotographCache::layout`] knows of a photograph's layout, read
/// without reading a file or waiting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PeekedLayout {
    /// The header has been read, or the photograph decoded: its tiles may be
    /// asked for.
    Known(Arc<PyramidLayout>),
    /// The photograph cannot be read; the message names the file and why.
    Unreadable(String),
    /// Not read yet: [`PhotographCache::request_layout`] reads it.
    NotKnown,
}

/// What [`PhotographCache::peek_tile`] found for a tile.
pub enum PeekedTile {
    /// The tile, with its border.
    Ready(Arc<PhotographTile>),
    /// The photograph cannot be read, or has no such tile: asking for it
    /// would fail.
    Unreadable,
    /// Not decoded yet, or being decoded.
    NotReady,
}

/// Why a tile, a region or a level could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileError {
    /// The photograph does not exist, cannot be read or decoded, or has a
    /// side over [`MAX_JPEG_SIDE`](crate::camera::mipmap::MAX_JPEG_SIDE);
    /// `why` is the [`GetError`]'s message.
    Unreadable { path: PathBuf, why: String },
    /// A level, tile or rectangle the photograph does not have, or an empty
    /// rectangle, was asked for.
    OutOfRange { path: PathBuf, why: String },
    /// The caller's `Progress` was cancelled while it decoded the photograph
    /// or waited for another caller's decode of it. Never remembered: the
    /// next call decodes or waits again.
    Cancelled,
}

impl fmt::Display for TileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { why, .. } => write!(f, "{why}"),
            Self::OutOfRange { path, why } => write!(f, "{}: {why}", path.display()),
            Self::Cancelled => write!(f, "{Cancelled}"),
        }
    }
}

impl std::error::Error for TileError {}

/// What [`PhotographCache::try_decode_tiles`] did.
pub enum TryDecode {
    /// The tiles asked for, in the order of the keys: decoded now, or cut
    /// from the photograph already held.
    Decoded(Vec<Arc<PhotographTile>>),
    /// Nothing was decoded: another caller is decoding the photograph, or
    /// its decode does not fit under the build budget now. Ask again later.
    Busy,
}

/// A made slot's value: the pyramid, or the failure remembered for the file.
type Made = Result<Arc<ImageU8Pyramid>, GetError>;

/// What a panel's decode found its slot holding when it tried to claim it.
enum Claim<'s, 'r> {
    /// Made, with the pyramid or the remembered failure.
    Made(Result<Arc<ImageU8Pyramid>, GetError>),
    /// Being made by another caller.
    Busy,
    /// Empty, and now marked as being made by this caller.
    Making(Making<'s, 'r>),
}

/// Mark `slot` as being made if it is empty, without waiting on it.
fn claim<'s, 'r>(slot: &'s Slot, share: Share<'r>) -> Claim<'s, 'r> {
    let mut state = slot.lock();
    match &*state {
        SlotState::Made(value) => Claim::Made(value.clone()),
        SlotState::Making => Claim::Busy,
        SlotState::Empty => {
            *state = SlotState::Making;
            Claim::Making(Making {
                slot,
                share,
                finished: false,
            })
        }
    }
}

/// The [`TileError`] for a [`GetError`]: [`TileError::Cancelled`] for a
/// cancelled call, and [`TileError::Unreadable`] for every other.
fn unreadable(path: &Path, e: GetError) -> TileError {
    match e {
        GetError::Cancelled => TileError::Cancelled,
        e => TileError::Unreadable {
            path: path.to_path_buf(),
            why: e.to_string(),
        },
    }
}

/// [`TileError::OutOfRange`] for a read of a photograph made in memory.
fn out_of_range(path: &Path, e: PhotographReadError) -> TileError {
    let why = match e {
        PhotographReadError::OutOfRange { why, .. } => why,
        other => other.to_string(),
    };
    TileError::OutOfRange {
        path: path.to_path_buf(),
        why,
    }
}

/// Refuse a key whose level, column or row `layout` does not have.
fn check_keys(path: &Path, layout: &PyramidLayout, keys: &[TileKey]) -> Result<(), TileError> {
    for key in keys {
        let (cols, rows) = layout.tiles(usize::from(key.level));
        if key.col >= cols || key.row >= rows {
            return Err(TileError::OutOfRange {
                path: path.to_path_buf(),
                why: format!(
                    "level {} has no tile at column {}, row {}",
                    key.level, key.col, key.row
                ),
            });
        }
    }
    Ok(())
}

impl PhotographCache {
    /// The layout of a `width` x `height` photograph as the cache cuts its
    /// tiles: in core, of [`ImageU8Pyramid::full_levels`] depth.
    fn layout_for(&self, width: u32, height: u32) -> PyramidLayout {
        PyramidLayout::in_memory(
            width,
            height,
            ImageU8Pyramid::full_levels(width, height),
            self.options.tile_size,
        )
    }

    /// `pyramid` read in place, cut into the cache's tiles.
    fn tiled(&self, pyramid: &Arc<ImageU8Pyramid>) -> TiledPhotograph {
        TiledPhotograph::from_pyramid(Arc::clone(pyramid)).with_tile_size(self.options.tile_size)
    }

    /// The tile `key` of `pyramid`, the photograph at `path`.
    fn cut(
        &self,
        path: &Path,
        pyramid: &Arc<ImageU8Pyramid>,
        key: TileKey,
    ) -> Result<Arc<PhotographTile>, TileError> {
        self.tiled(pyramid)
            .tile(key)
            .map_err(|e| out_of_range(path, e))
    }

    /// `path`'s slot and its value when it is made, without a stat of the
    /// file and without waiting on a decode in flight.
    fn made_slot(&self, path: &Path) -> Option<(Arc<Slot>, Made)> {
        let slot = self.lock().map.get(path).map(Arc::clone)?;
        let value = slot.made()?;
        Some((slot, value))
    }

    /// What is known of `path`'s layout, without reading anything or
    /// waiting: from the photograph held or its remembered failure, and
    /// otherwise from [`Self::request_layout`]'s answer.
    ///
    /// Like [`Self::peek_state`] it does not stat the file, so a layout or a
    /// failure is answered until the photograph is decoded again, the layout
    /// is asked for again, or the path is forgotten.
    ///
    /// The layout of a held photograph is kept as `request_layout` keeps
    /// one, so a panel that asks every frame is answered the same `Arc`
    /// while the held pyramid's size stays the same, with nothing allocated.
    pub fn layout(&self, path: &Path) -> PeekedLayout {
        if let Some((_, value)) = self.made_slot(path) {
            return match value {
                Ok(pyramid) => {
                    let [width, height] = pyramid.level_full_size(0);
                    let held = PyramidLayout::in_memory(
                        width,
                        height,
                        pyramid.num_levels(),
                        self.options.tile_size,
                    );
                    let mut inner = self.lock();
                    if let Some(Ok(kept)) = inner.layouts.get(path) {
                        if **kept == held {
                            return PeekedLayout::Known(Arc::clone(kept));
                        }
                    }
                    let held = Arc::new(held);
                    inner
                        .layouts
                        .insert(path.to_path_buf(), Ok(Arc::clone(&held)));
                    PeekedLayout::Known(held)
                }
                Err(e) => PeekedLayout::Unreadable(e.to_string()),
            };
        }
        match self.lock().layouts.get(path) {
            Some(Ok(layout)) => PeekedLayout::Known(Arc::clone(layout)),
            Some(Err(why)) => PeekedLayout::Unreadable(why.clone()),
            None => PeekedLayout::NotKnown,
        }
    }

    /// Read `path`'s header and keep the layout it gives, or why it could
    /// not be read, for [`Self::layout`] to answer, whatever the budget.
    /// Reads only the header, decodes nothing and never waits on a slot;
    /// a panel calls it on the rayon pool.
    pub fn request_layout(&self, path: &Path) {
        let layout = header(path)
            .map(|(width, height)| Arc::new(self.layout_for(width, height)))
            .map_err(|e| e.to_string());
        self.lock().layouts.insert(path.to_path_buf(), layout);
    }

    /// The tile `key` of `path` if the photograph is decoded, cut from the
    /// pyramid held, without reading a file and without waiting on a decode
    /// in flight. What a panel calls every frame.
    ///
    /// Like [`Self::peek`] it does not stat the file, and counts as a use for
    /// the eviction order but not as a hit.
    pub fn peek_tile(&self, path: &Path, key: TileKey) -> PeekedTile {
        let Some((slot, value)) = self.made_slot(path) else {
            return PeekedTile::NotReady;
        };
        self.touch(&slot);
        match value.map(|pyramid| self.cut(path, &pyramid, key)) {
            Ok(Ok(tile)) => PeekedTile::Ready(tile),
            Ok(Err(_)) | Err(_) => PeekedTile::Unreadable,
        }
    }

    /// The tile `key` of `path`, decoding the photograph on a miss as
    /// [`Self::get`] does: reserving with [`Self::reserve_builds`] and waiting
    /// while another caller decodes it. A kernel's read; a panel asks with
    /// [`Self::try_decode_tiles`], which never waits.
    ///
    /// A whole decode can take over a second, so it takes the caller's
    /// `progress`, which it polls for cancellation as `get` does and reports
    /// nothing through, and answers [`TileError::Cancelled`] when it is
    /// cancelled, remembering nothing and holding no reservation.
    pub fn get_tile(
        &self,
        path: &Path,
        key: TileKey,
        progress: &Progress<'_>,
    ) -> Result<Arc<PhotographTile>, TileError> {
        let pyramid = self.get(path, progress).map_err(|e| unreadable(path, e))?;
        self.cut(path, &pyramid, key)
    }

    /// A view-region pyramid of `path`: the region of each level `regions`
    /// names, copied from the photograph, decoded on a miss as
    /// [`Self::get_tile`] decodes it, polling `progress` as it does. A level
    /// or region the photograph does not have, or no region at all, is
    /// [`TileError::OutOfRange`].
    pub fn view_region_pyramid(
        &self,
        path: &Path,
        regions: &LevelRegions,
        progress: &Progress<'_>,
    ) -> Result<ImageU8Pyramid, TileError> {
        let pyramid = self.get(path, progress).map_err(|e| unreadable(path, e))?;
        copy_regions(&self.tiled(&pyramid), regions).map_err(|e| out_of_range(path, e))
    }

    /// Level `level` of `path` whole, decoded on a miss as
    /// [`Self::get_tile`] decodes it, polling `progress` as it does. A level
    /// the photograph does not have is [`TileError::OutOfRange`].
    pub fn get_level(
        &self,
        path: &Path,
        level: usize,
        progress: &Progress<'_>,
    ) -> Result<Arc<ImageU8>, TileError> {
        let pyramid = self.get(path, progress).map_err(|e| unreadable(path, e))?;
        let tiled = self.tiled(&pyramid);
        let (x1, y1) = tiled.layout().level_size(level);
        let rect = LevelRect {
            x0: 0,
            y0: 0,
            x1,
            y1,
        };
        tiled
            .region(level, rect)
            .map(Arc::new)
            .map_err(|e| out_of_range(path, e))
    }

    /// A panel's decode of `path`, answering the tiles `keys` names.
    ///
    /// Answers the tiles at once when the photograph is held. Otherwise it
    /// reads the header, refuses a key the photograph does not have, and
    /// reserves the decode with [`Self::try_reserve_build`]; then it claims
    /// the slot only when no other caller is decoding in it, giving the
    /// reservation back if one is, and decodes in the slot as [`Self::get`]
    /// does. It answers [`TryDecode::Busy`], decoding nothing, when the
    /// reservation is refused or another caller holds the slot. So it never
    /// waits on a slot or on the build count, and never takes the count past
    /// the build budget.
    ///
    /// Polls `progress` for cancellation after its decode, as `get` does, and
    /// answers [`TileError::Cancelled`] when it is cancelled: the slot is left
    /// empty, nothing is remembered and the reservation is given back. It
    /// reports nothing through `progress`; a panel's task passes its own, or
    /// `&Progress::none()`.
    ///
    /// A cache with a budget of 0 keeps nothing, so there the tiles it
    /// answers are the only ones the caller gets.
    pub fn try_decode_tiles(
        &self,
        path: &Path,
        keys: &[TileKey],
        progress: &Progress<'_>,
    ) -> Result<TryDecode, TileError> {
        let slot = (self.options.budget_bytes > 0).then(|| self.slot_for(path));
        if let Some(slot) = &slot {
            // Read with the slot's lock released before the tiles are cut.
            let made = match &*slot.lock() {
                SlotState::Made(value) => Some(value.clone()),
                SlotState::Making => return Ok(TryDecode::Busy),
                SlotState::Empty => None,
            };
            if let Some(value) = made {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return self.answer(path, value, keys);
            }
        }
        // A file whose header cannot be read is made at once, as the failure
        // remembered in its slot.
        let size = header(path);
        let mut share = Share::None;
        if let Ok((width, height)) = size {
            check_keys(path, &self.layout_for(width, height), keys)?;
            match self.try_reserve_build(whole_decode_bytes(width, height)) {
                Some(reservation) => share = Share::Own(reservation),
                None => return Ok(TryDecode::Busy),
            }
        }
        #[cfg(test)]
        self.slot_event(super::SlotEvent::PanelReserved);
        let value = match &slot {
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                // The reservation is given back when `share` drops, whatever
                // the outcome.
                size.and_then(|_| self.decode(path, progress, &mut share))
            }
            Some(slot) => {
                let mut making = match claim(slot, share) {
                    Claim::Made(value) => {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                        return self.answer(path, value, keys);
                    }
                    // Dropping the share gives the reservation back.
                    Claim::Busy => return Ok(TryDecode::Busy),
                    Claim::Making(making) => making,
                };
                self.misses.fetch_add(1, Ordering::Relaxed);
                let value = size.and_then(|_| self.decode(path, progress, &mut making.share));
                if let Err(GetError::Cancelled) = value {
                    // Dropping the hold gives the share back, empties the slot
                    // and wakes its waiters.
                    drop(making);
                    return Err(TileError::Cancelled);
                }
                making.finish(value.clone());
                if let Ok(pyramid) = &value {
                    self.charge(path, slot, pyramid.byte_len() as u64);
                }
                value
            }
        };
        self.answer(path, value, keys)
    }

    /// The tiles `keys` of a made photograph, or its failure.
    fn answer(
        &self,
        path: &Path,
        value: Result<Arc<ImageU8Pyramid>, GetError>,
        keys: &[TileKey],
    ) -> Result<TryDecode, TileError> {
        let pyramid = value.map_err(|e| unreadable(path, e))?;
        keys.iter()
            .map(|&key| self.cut(path, &pyramid, key))
            .collect::<Result<Vec<_>, _>>()
            .map(TryDecode::Decoded)
    }
}
