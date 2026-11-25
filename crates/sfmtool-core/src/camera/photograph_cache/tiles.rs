// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cache's layout and tile requests: a photograph's layout, its
//! photograph tiles, a view-region pyramid or one level whole, each from the
//! tiles the cache keeps, decoding the photograph when a tile asked for is
//! not kept.
//!
//! A panel asks [`PhotographCache::layout`] and [`PhotographCache::peek_tile`]
//! on the GUI thread, which never read a file or wait, and asks on the rayon
//! pool for what is missing: [`PhotographCache::request_layout`] reads the
//! header, and [`PhotographCache::try_decode_tiles`] decodes the photograph
//! without waiting on its slot or on the build count. A kernel's reads,
//! [`PhotographCache::get_tile`], [`PhotographCache::view_region_pyramid`]
//! and [`PhotographCache::get_level`], decode as [`PhotographCache::get`]
//! does, but never make a mipmap: for an out-of-core photograph whose files
//! are not current they answer [`TileError::Changed`]. See the spec
//! `specs/core/camera/photograph-cache.md` § "Layouts and tiles".

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::build::Make;
use super::files::{is_out_of_core, Failure, Tier, Unmapped};
use super::{
    answer_of, has_key, header, whole_decode_bytes, Asked, GetError, Got, Making, PhotographCache,
    Share, Slot,
};
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::mipmap::PyramidLayout;
use crate::camera::sampler::{copy_regions, LevelRect, LevelRegions};
use crate::camera::tiled_photograph::TiledPhotograph;
use crate::camera::tiled_photograph::{PhotographReadError, PhotographTile, TileKey};
use crate::progress::{Cancelled, Progress};

#[cfg(test)]
mod tests;

/// What [`PhotographCache::layout`] knows of a photograph's layout, read
/// without reading a file or waiting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PeekedLayout {
    /// The header has been read, or the photograph decoded, and for an
    /// out-of-core photograph its mipmap found current or made: its tiles
    /// may be asked for.
    Known(Arc<PyramidLayout>),
    /// An out-of-core photograph whose mipmap is missing, stale or being
    /// made: no tile of it is asked for until a `get` or `get_many` makes
    /// the mipmap, which a panel never does. `coarsest` is the mipmap's
    /// coarsest level once a build has encoded it, which a panel draws while
    /// it waits; `None` before.
    Building {
        coarsest: Option<Arc<CoarsestLevel>>,
    },
    /// The photograph cannot be read; the message names the file and why.
    Unreadable(String),
    /// Not read yet: [`PhotographCache::request_layout`] reads it.
    NotKnown,
}

/// The name of the [`Progress::value`] a build of the cache reports when it
/// has published a photograph's coarsest level ([`CoarsestLevel`]), so a
/// caller that draws can repaint: the value is the level, `K`. Reported once
/// a build, through `get`'s `Progress` and every `get_many` member's.
pub const COARSEST_PUBLISHED: &str = "coarsest_published";

/// The coarsest level of an out-of-core photograph's mipmap, published by
/// the build that is making it as soon as that level is encoded, before the
/// rest is made: what a panel draws while the photograph is
/// [`PeekedLayout::Building`].
///
/// `tiles` are the tiles of level `K` and of every level below it, one
/// each, cut from level `K` and the levels box-filtered from it, as the
/// finished mipmap's coarse tiles are, so they are the tiles a panel is
/// answered once the photograph is `Known`. A panel draws them over level-0
/// pixels with `layout`, the layout the finished mipmap is read under, whose
/// level 0 is the photograph's own size, so what it draws in level-0 pixels
/// over them lines up.
pub struct CoarsestLevel {
    /// The photograph's out-of-core layout.
    pub layout: Arc<PyramidLayout>,
    /// The tiles of levels `K` to the last, in level order, each `(level,
    /// 0, 0)` with its border.
    pub tiles: Vec<Arc<PhotographTile>>,
}

impl CoarsestLevel {
    /// `K`, the level of the first tile.
    pub fn level(&self) -> usize {
        self.layout.stored_levels()
    }
}

impl fmt::Debug for CoarsestLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoarsestLevel")
            .field("layout", &self.layout)
            .field(
                "tiles",
                &self.tiles.iter().map(|tile| tile.key).collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// The same layout and the same tiles, not copies of them: what a panel
/// compares to know it has uploaded these already.
impl PartialEq for CoarsestLevel {
    fn eq(&self, other: &Self) -> bool {
        self.layout == other.layout
            && self.tiles.len() == other.tiles.len()
            && self
                .tiles
                .iter()
                .zip(&other.tiles)
                .all(|(a, b)| Arc::ptr_eq(a, b))
    }
}

impl Eq for CoarsestLevel {}

/// What [`PhotographCache::peek_tile`] found for a tile.
pub enum PeekedTile {
    /// The tile, with its border.
    Ready(Arc<PhotographTile>),
    /// The photograph is [`PeekedLayout::Building`]: asking for the tile
    /// would answer [`TileError::Changed`].
    Building,
    /// The photograph cannot be read, or has no such tile: asking for it
    /// would fail.
    Unreadable,
    /// Not decoded yet, or being decoded.
    NotReady,
}

/// Why a tile, a region or a level could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileError {
    /// A file of an out-of-core photograph's mipmap, or its original, is
    /// missing, not current, or not the one the cache first read for it:
    /// `path` names the file, or the photograph when its mipmap was already
    /// found not current. The photograph is [`PeekedLayout::Building`] from
    /// then on, until a [`PhotographCache::get`] or
    /// [`PhotographCache::get_many`] makes its mipmap; a tile read never
    /// makes one.
    Changed { path: PathBuf },
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
            Self::Changed { path } => write!(
                f,
                "{} changed while it was being read; run it again",
                path.display()
            ),
            Self::Unreadable { why, .. } => write!(f, "{why}"),
            Self::OutOfRange { path, why } => write!(f, "{}: {why}", path.display()),
            Self::Cancelled => write!(f, "{Cancelled}"),
        }
    }
}

impl std::error::Error for TileError {}

/// What [`PhotographCache::try_decode_tiles`] did.
pub enum TryDecode {
    /// The tiles asked for, in the order of the keys: decoded now, or kept
    /// by the cache already.
    Decoded(Vec<Arc<PhotographTile>>),
    /// Nothing was decoded: another caller is decoding the photograph, or
    /// its decode does not fit under the build budget now. Ask again later.
    Busy,
}

/// What a panel's decode found its slot holding when it tried to claim it.
enum Claim<'s, 'r> {
    /// The tiles asked for, every one kept, or the remembered failure.
    Made(Result<Got, GetError>),
    /// Being made by another caller.
    Busy,
    /// Not holding every tile asked for, and now marked as being made by
    /// this caller.
    Making(Making<'s, 'r>),
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

/// The [`TileError`] for a request's failure: [`TileError::Changed`] naming
/// the file found not current, and otherwise as [`unreadable`] says.
fn failed(path: &Path, e: Failure) -> TileError {
    match e {
        Failure::Get(e) => unreadable(path, e),
        Failure::Changed(file) => TileError::Changed { path: file },
        Failure::NotHeld => unreadable(path, Failure::not_held(path)),
    }
}

/// The [`TileError`] for a failed read of a photograph the cache answered:
/// [`TileError::Cancelled`] for a cancelled read, [`TileError::Changed`] for
/// a file found changed, [`TileError::OutOfRange`] for a level or region the
/// photograph does not have, and
/// [`TileError::Unreadable`] for a read of its file again that failed.
fn read_failed(path: &Path, e: PhotographReadError) -> TileError {
    let path = path.to_path_buf();
    match e {
        PhotographReadError::Cancelled => TileError::Cancelled,
        PhotographReadError::Changed { path: file } => TileError::Changed { path: file },
        PhotographReadError::OutOfRange { why, .. } => TileError::OutOfRange { path, why },
        PhotographReadError::NeedsTiles { .. } => TileError::OutOfRange {
            path,
            why: e.to_string(),
        },
        e => TileError::Unreadable {
            path,
            why: e.to_string(),
        },
    }
}

/// Refuse a key whose level, column or row `layout` does not have.
fn check_keys(path: &Path, layout: &PyramidLayout, keys: &[TileKey]) -> Result<(), TileError> {
    for &key in keys {
        if !has_key(layout, key) {
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
    pub(super) fn layout_for(&self, width: u32, height: u32) -> PyramidLayout {
        PyramidLayout::in_memory(
            width,
            height,
            ImageU8Pyramid::full_levels(width, height),
            self.shared.options.tile_size,
        )
    }

    /// Mark `slot` as being made if it does not keep every tile `keys`
    /// names, without waiting on it.
    fn claim<'s, 'r>(
        &self,
        slot: &'s Arc<Slot>,
        share: Share<'r>,
        keys: &[TileKey],
    ) -> Claim<'s, 'r> {
        let mut data = slot.lock();
        if data.making {
            return Claim::Busy;
        }
        if let Some(Err(e)) = &data.made {
            return Claim::Made(Err(e.clone()));
        }
        if let Some(got) = data.kept(Asked::Keys(keys), self.next_tick(), slot) {
            return Claim::Made(Ok(got));
        }
        data.making = true;
        Claim::Making(Making {
            slot,
            share,
            finished: false,
        })
    }

    /// What is known of `path`'s layout, without reading anything or
    /// waiting: from the photograph's decode or its remembered failure, and
    /// otherwise from [`Self::request_layout`]'s answer.
    ///
    /// Like [`Self::peek_state`] it does not stat the file, so a layout or a
    /// failure is answered until the photograph is decoded again, the layout
    /// is asked for again, or the path is forgotten.
    ///
    /// The layout of a decoded photograph is the one its slot keeps, so a
    /// panel that asks every frame is answered the same `Arc`, with nothing
    /// allocated, while the photograph's layout stays the same.
    ///
    /// An out-of-core photograph whose slot found its mipmap missing or
    /// stale, or is making it, is [`PeekedLayout::Building`] until the
    /// mipmap is made, or [`PeekedLayout::Unreadable`] when its build
    /// failed. It carries the coarsest level once a build has published it:
    /// the slot's under a budget, and under a budget of 0, which keeps no
    /// slot, the one the build put in the layout [`Self::request_layout`]
    /// kept, which is how it is handed to the panel that asked.
    pub fn layout(&self, path: &Path) -> PeekedLayout {
        if let Some(slot) = self.mapped(path) {
            let data = slot.lock();
            match (&data.made, &data.tier) {
                (Some(Err(e)), _) => return PeekedLayout::Unreadable(e.to_string()),
                (_, Tier::NotCurrent) => {
                    return PeekedLayout::Building {
                        coarsest: data.coarsest.clone(),
                    }
                }
                (Some(Ok(layout)), _) => return PeekedLayout::Known(Arc::clone(layout)),
                (None, _) => {}
            }
        }
        self.lock()
            .layouts
            .get(path)
            .cloned()
            .unwrap_or(PeekedLayout::NotKnown)
    }

    /// Read `path`'s header and keep what it says of the layout its tiles are
    /// served under, or why it could not be read, for [`Self::layout`] to
    /// answer, whatever the budget: an in-core photograph's in-core layout,
    /// an out-of-core photograph's out-of-core layout when its mipmap is
    /// current, and [`PeekedLayout::Building`] when its mipmap is missing or
    /// stale. For an out-of-core photograph it checks the coarsest mip level
    /// file, as the photograph's first request does (with a budget, in the
    /// slot, which keeps the answer). Decodes no pixel and never waits on a
    /// slot; a panel calls it on the rayon pool.
    ///
    /// Under a budget of 0, which keeps no build, an out-of-core photograph
    /// whose last build could not write its files, under the original as it
    /// is now, is answered with its in-core layout, since a panel then reads
    /// it from a decode in core ([`Self::try_decode_tiles`]).
    pub fn request_layout(&self, path: &Path) {
        let setting = self.setting_for(path);
        let layout = match header(path, &setting) {
            Err(e) => PeekedLayout::Unreadable(e.to_string()),
            Ok((width, height)) => {
                let in_core = || PeekedLayout::Known(Arc::new(self.layout_for(width, height)));
                if !is_out_of_core(&setting, width, height) {
                    in_core()
                } else if self.shared.options.budget_bytes == 0 {
                    match self.unmapped_files_slot(path, &setting) {
                        Unmapped::Files(_, files) => PeekedLayout::Known(Arc::clone(&files.layout)),
                        Unmapped::InCore => in_core(),
                        Unmapped::NotCurrent if self.held_in_memory(path) => in_core(),
                        Unmapped::NotCurrent => PeekedLayout::Building { coarsest: None },
                    }
                } else {
                    let slot = self.slot_for(path);
                    match self.tier_of(path, &slot) {
                        Tier::Files(files) => PeekedLayout::Known(Arc::clone(&files.layout)),
                        Tier::NotCurrent => match &slot.lock().made {
                            Some(Err(e)) => PeekedLayout::Unreadable(e.to_string()),
                            _ => PeekedLayout::Building { coarsest: None },
                        },
                        Tier::Unchecked | Tier::InCore => in_core(),
                    }
                }
            }
        };
        let mut inner = self.lock();
        let kept = inner
            .layouts
            .entry(path.to_path_buf())
            .or_insert(layout.clone());
        // A coarsest level a build in progress has published stays.
        if !matches!(
            (&*kept, &layout),
            (
                PeekedLayout::Building { coarsest: Some(_) },
                PeekedLayout::Building { .. }
            )
        ) {
            *kept = layout;
        }
    }

    /// The tile `key` of `path` if the cache keeps it, without reading a
    /// file and without waiting on a decode in flight. What a panel calls
    /// every frame.
    ///
    /// Like [`Self::peek`] it does not stat the file. It counts as a use for
    /// the eviction order, marking the tile as used, but not as a hit. Any
    /// tile of a photograph [`Self::layout`] answers as
    /// [`PeekedLayout::Building`] from its slot is [`PeekedTile::Building`].
    pub fn peek_tile(&self, path: &Path, key: TileKey) -> PeekedTile {
        let Some(slot) = self.mapped(path) else {
            return PeekedTile::NotReady;
        };
        let tick = self.next_tick();
        let mut guard = slot.lock();
        let data = &mut *guard;
        match &data.made {
            Some(Err(_)) => PeekedTile::Unreadable,
            _ if matches!(data.tier, Tier::NotCurrent) => PeekedTile::Building,
            None => PeekedTile::NotReady,
            Some(Ok(layout)) if !has_key(layout, key) => PeekedTile::Unreadable,
            Some(Ok(_)) => match data.tiles.get_mut(&key) {
                Some(kept) => {
                    (kept.used, kept.tick) = (true, tick);
                    PeekedTile::Ready(Arc::clone(&kept.tile))
                }
                None => PeekedTile::NotReady,
            },
        }
    }

    /// The tile `key` of `path`, decoding the photograph when the cache does
    /// not keep it, as [`Self::get`] decodes: reserving with
    /// [`Self::reserve_builds`] and waiting while another caller decodes it.
    /// A decode keeps the tile as used and the rest of the photograph as
    /// unused. A kernel's read; a panel asks with [`Self::try_decode_tiles`],
    /// which never waits.
    ///
    /// An out-of-core photograph whose mipmap is current has the tile decoded
    /// from its file instead, without waiting, and a file found missing, not
    /// current or changed since the slot read it is [`TileError::Changed`],
    /// naming it. It never makes a mipmap: an out-of-core photograph whose
    /// files are missing or stale, found so now or before, is
    /// [`TileError::Changed`] naming the photograph, decoding nothing, until
    /// [`Self::get`] makes its mipmap.
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
        let (value, _) = self.fetch(
            path,
            progress,
            Share::None,
            Asked::Keys(&[key]),
            Make::Never,
        );
        let got = value.map_err(|e| failed(path, e))?;
        check_keys(path, &got.layout, &[key])?;
        got.tiles
            .into_iter()
            .next()
            .ok_or_else(|| TileError::OutOfRange {
                path: path.to_path_buf(),
                why: format!("the tile {key:?} was not cut"),
            })
    }

    /// A view-region pyramid of `path`: the region of each level `regions`
    /// names, copied from the photograph, decoded on a miss as
    /// [`Self::get_tile`] decodes it, polling `progress` as it does, and like
    /// it never making a mipmap. A level or region the photograph does not
    /// have, or no region at all, is [`TileError::OutOfRange`].
    pub fn view_region_pyramid(
        &self,
        path: &Path,
        regions: &LevelRegions,
        progress: &Progress<'_>,
    ) -> Result<ImageU8Pyramid, TileError> {
        let photograph = self.read_photograph(path, progress)?;
        copy_regions(&photograph, regions, progress).map_err(|e| read_failed(path, e))
    }

    /// Level `level` of `path` whole, decoded on a miss as
    /// [`Self::get_tile`] decodes it, polling `progress` as it does, and like
    /// it never making a mipmap. A level the photograph does not have is
    /// [`TileError::OutOfRange`].
    pub fn get_level(
        &self,
        path: &Path,
        level: usize,
        progress: &Progress<'_>,
    ) -> Result<Arc<ImageU8>, TileError> {
        let tiled = self.read_photograph(path, progress)?;
        let (x1, y1) = tiled.layout().level_size(level);
        let rect = LevelRect {
            x0: 0,
            y0: 0,
            x1,
            y1,
        };
        tiled
            .region(level, rect, progress)
            .map(Arc::new)
            .map_err(|e| read_failed(path, e))
    }

    /// The photograph at `path` as [`Self::get`] answers it, for a kernel's
    /// read that never makes a mipmap: an out-of-core photograph whose files
    /// are missing or stale is [`TileError::Changed`].
    fn read_photograph(
        &self,
        path: &Path,
        progress: &Progress<'_>,
    ) -> Result<Arc<TiledPhotograph>, TileError> {
        let (value, _) = self.fetch(path, progress, Share::None, Asked::Photograph, Make::Never);
        let got = value.map_err(|e| failed(path, e))?;
        got.photograph(self, path).map_err(|e| unreadable(path, e))
    }

    /// A panel's decode of `path`, answering the tiles `keys` names.
    ///
    /// Answers the tiles at once when the cache keeps every one of them.
    /// Otherwise it reads the header, refuses a key the photograph does not
    /// have, and reserves the decode with [`Self::try_reserve_build`]; then it
    /// claims the slot only when no other caller is decoding in it, giving the
    /// reservation back if one is, and decodes in the slot as [`Self::get`]
    /// does, keeping the tiles `keys` names as used and the rest as unused. It
    /// answers [`TryDecode::Busy`], decoding nothing, when the reservation is
    /// refused or another caller holds the slot. So it never waits on a slot
    /// or on the build count, and takes the count past the build budget only
    /// by one decode that runs alone, as [`Self::try_reserve_build`] grants
    /// it.
    ///
    /// Polls `progress` for cancellation after its decode, as `get` does, and
    /// answers [`TileError::Cancelled`] when it is cancelled: the slot is left
    /// as it was, nothing is remembered and the reservation is given back. It
    /// reports nothing through `progress`; a panel's task passes its own, or
    /// `&Progress::none()`.
    ///
    /// A cache with a budget of 0 keeps nothing, so there the tiles it
    /// answers are the only ones the caller gets.
    ///
    /// An out-of-core photograph whose mipmap is current is read from its
    /// files, with no reservation. It never makes a mipmap and never reads
    /// an out-of-core photograph whole in core: one whose files are missing
    /// or stale, being made, or turn out missing, not current or changed
    /// since the slot read them, is [`TileError::Changed`], decoding
    /// nothing, and [`PeekedLayout::Building`] from then on, until a `get` or
    /// `get_many` makes its mipmap. Two cases are decoded whole in core, as
    /// above, and kept by no slot: a level of a mipmap held in memory that
    /// the slot no longer keeps, and, under a budget of 0, which keeps no
    /// build, a photograph whose last build could not write its files.
    pub fn try_decode_tiles(
        &self,
        path: &Path,
        keys: &[TileKey],
        progress: &Progress<'_>,
    ) -> Result<TryDecode, TileError> {
        let asked = Asked::Keys(keys);
        let slot = (self.shared.options.budget_bytes > 0).then(|| self.slot_for(path));
        // Found once for a call with no slot, which both reads below use.
        let unslotted;
        let setting = match &slot {
            Some(slot) => &slot.setting,
            None => {
                unslotted = self.setting_for(path);
                &unslotted
            }
        };
        let mut slot = slot.as_ref();
        // An out-of-core photograph whose mipmap is current is read from its
        // files, which never waits on a slot or the build count. One whose
        // mipmap is not current, found so now or before, is `Changed` and
        // `Building` from then on, decoding nothing.
        let read_from_files = match slot {
            Some(slot) => match self.tier_of(path, slot) {
                Tier::Files(files) => Some((Arc::clone(slot), files)),
                Tier::NotCurrent => {
                    let remembered = match &slot.lock().made {
                        Some(Err(e)) => Some(e.clone()),
                        _ => None,
                    };
                    return Err(match remembered {
                        Some(e) => unreadable(path, e),
                        None => TileError::Changed {
                            path: path.to_path_buf(),
                        },
                    });
                }
                Tier::Unchecked | Tier::InCore => None,
            },
            None => match self.unmapped_files_slot(path, setting) {
                Unmapped::Files(slot, files) => Some((slot, files)),
                Unmapped::InCore => None,
                // A build that could not write its files is not kept under
                // a budget of 0: the tiles are decoded whole in core.
                Unmapped::NotCurrent if self.held_in_memory(path) => None,
                Unmapped::NotCurrent => {
                    // A coarsest level a build in progress has published
                    // stays.
                    self.lock()
                        .layouts
                        .entry(path.to_path_buf())
                        .and_modify(|kept| {
                            if !matches!(kept, PeekedLayout::Building { .. }) {
                                *kept = PeekedLayout::Building { coarsest: None };
                            }
                        })
                        .or_insert(PeekedLayout::Building { coarsest: None });
                    return Err(TileError::Changed {
                        path: path.to_path_buf(),
                    });
                }
            },
        };
        if let Some((files_slot, files)) = read_from_files {
            check_keys(path, &files.layout, keys)?;
            match self.read_files(path, &files_slot, asked, false, progress) {
                Ok(got) => return Ok(TryDecode::Decoded(got.tiles)),
                Err(Failure::Get(e)) => return Err(unreadable(path, e)),
                // `read_files` has marked the photograph `Building`.
                Err(Failure::Changed(file)) => return Err(TileError::Changed { path: file }),
                // A level of a mipmap held in memory was evicted, which only
                // a build makes again: the tiles are decoded whole in core
                // and kept by no slot, which goes on reading its mipmap.
                Err(Failure::NotHeld) => slot = None,
            }
        }
        if let Some(slot) = slot {
            let tick = self.next_tick();
            let mut data = slot.lock();
            if data.making {
                return Ok(TryDecode::Busy);
            }
            match &data.made {
                Some(Err(e)) => {
                    let e = e.clone();
                    drop(data);
                    self.shared.hits.fetch_add(1, Ordering::Relaxed);
                    return Err(unreadable(path, e));
                }
                Some(Ok(layout)) => {
                    check_keys(path, layout, keys)?;
                    if let Some(got) = data.kept(asked, tick, slot) {
                        drop(data);
                        self.shared.hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(TryDecode::Decoded(got.tiles));
                    }
                }
                None => {}
            }
        }
        // A file whose header cannot be read, or whose workspace file cannot,
        // is made at once, as the failure remembered in its slot.
        let size = header(path, setting);
        let mut share = Share::None;
        if let Ok((width, height)) = size {
            check_keys(path, &self.layout_for(width, height), keys)?;
            match self.try_reserve_build(whole_decode_bytes(width, height)) {
                Some(reservation) => share = Share::Own(reservation),
                None => return Ok(TryDecode::Busy),
            }
        }
        #[cfg(test)]
        self.slot_event(super::SlotEvent::PanelReserved, path);
        let value = match &slot {
            None => {
                self.shared.misses.fetch_add(1, Ordering::Relaxed);
                // The reservation is given back when `share` drops, whatever
                // the outcome.
                size.and_then(|_| self.decode(path, progress, &mut share, setting))
                    .map(|cut| answer_of(cut, asked))
            }
            Some(slot) => {
                let mut making = match self.claim(slot, share, keys) {
                    Claim::Made(value) => {
                        self.shared.hits.fetch_add(1, Ordering::Relaxed);
                        return answer(path, value);
                    }
                    // Dropping the share gives the reservation back.
                    Claim::Busy => return Ok(TryDecode::Busy),
                    Claim::Making(making) => making,
                };
                self.shared.misses.fetch_add(1, Ordering::Relaxed);
                let value =
                    size.and_then(|_| self.decode(path, progress, &mut making.share, setting));
                if let Err(GetError::Cancelled) = value {
                    // Dropping the hold gives the share back, leaves the slot
                    // as it was and wakes its waiters.
                    drop(making);
                    return Err(TileError::Cancelled);
                }
                self.keep(path, making, value, asked)
            }
        };
        answer(path, value)
    }
}

/// A panel's decode's answer: the tiles asked for, or the failure.
fn answer(path: &Path, value: Result<Got, GetError>) -> Result<TryDecode, TileError> {
    value
        .map(|got| TryDecode::Decoded(got.tiles))
        .map_err(|e| unreadable(path, e))
}
