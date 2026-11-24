// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reading an out-of-core photograph from its mip level files: whether its
//! mipmap is current, the file entries kept under the cache's budget, and the
//! photograph tiles decoded from them.
//!
//! A slot decides its [`Tier`] the first time a request needs the
//! photograph. An out-of-core photograph whose coarsest mip level file is
//! current is read from its files: levels 1 to `K` a tile at a time from the
//! level's file, the levels below `K` cut from one decode of level `K`, and
//! level 0 from the original when it is region-readable or from its `-mip0`.
//! Each finer file is checked when it is first opened. An out-of-core
//! photograph whose files are missing or stale has its mipmap made in its
//! slot by `get` (`build.rs`); every other photograph is read whole in core.
//! See the spec `specs/core/camera/photograph-cache.md` § "Out-of-core
//! photographs".

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, MutexGuard};

use super::{
    file_stamp, has_key, tile_bytes, Asked, FileStamp, GetError, Got, Inner, Kept, PeekedLayout,
    PhotographCache, Slot, SlotData, WorkspaceSetting,
};
use crate::camera::image::{image_dimensions, ImageU8};
use crate::camera::mipmap::{
    mipmap_location, ExpectedMipLevel, MipLevelFile, MipLevelRecord, MipmapLocation, MipmapSource,
    OriginalRegions, PyramidLayout,
};
use crate::camera::tiled_photograph::{cut_level, tile_from_region, PhotographTile, TileKey};
use crate::progress::{Cancelled, Progress};

/// How a slot reads its photograph's pixels.
#[derive(Clone, Default)]
pub(super) enum Tier {
    /// Not decided yet: no request has needed the photograph.
    #[default]
    Unchecked,
    /// Whole, from a decode of the original: an in-core photograph, or an
    /// out-of-core one too thin to store a reduced level.
    InCore,
    /// Out of core, its mipmap missing or stale when the slot checked it, or
    /// found not current since. `get` makes the mipmap in the slot; until it
    /// is made a panel's decode reads the photograph whole in core, and a
    /// kernel's tile read answers `Changed`.
    NotCurrent,
    /// From its mip level files, current when the slot checked them or made
    /// by the slot's build, or from the build's files held in memory.
    Files(Arc<MipmapFiles>),
}

impl Tier {
    /// The files, when the slot reads from them.
    pub(super) fn files(&self) -> Option<&Arc<MipmapFiles>> {
        match self {
            Tier::Files(files) => Some(files),
            Tier::Unchecked | Tier::InCore | Tier::NotCurrent => None,
        }
    }

    /// Whether the slot reads from `files`, the ones a request started with.
    pub(super) fn is(&self, files: &Arc<MipmapFiles>) -> bool {
        self.files().is_some_and(|f| Arc::ptr_eq(f, files))
    }
}

/// An out-of-core photograph's mipmap, found current or just made: its
/// layout, where its files are, the original its records are checked
/// against, and whether its level 0 is read from the original.
pub(super) struct MipmapFiles {
    /// The out-of-core layout, of the workspace's `mipmap_options`.
    pub(super) layout: Arc<PyramidLayout>,
    pub(super) location: MipmapLocation,
    pub(super) source: Arc<MipmapSource>,
    /// The region answer of the coarsest file's record: level 0 is read
    /// from the original when `true`, and from `-mip0` otherwise.
    pub(super) region_readable: bool,
    /// Whether the files are a build's that could not be written, the
    /// in-memory equivalent: a slot in the map keeps them as file entries,
    /// evicted after every other entry but the coarse tiles, and a slot
    /// outside the map holds them ([`SlotData::held`]). A kernel's read of
    /// a level no longer kept makes the build again in memory
    /// ([`PhotographCache::rebuild_in_memory`]).
    pub(super) in_memory: bool,
}

impl MipmapFiles {
    /// `K`, the coarsest level the mipmap stores.
    pub(super) fn coarsest(&self) -> usize {
        self.layout.stored_levels()
    }

    /// Whether `key` is a tile of level `K` or of a level below it, all cut
    /// from one decode of level `K`'s file.
    fn is_coarse(&self, key: TileKey) -> bool {
        usize::from(key.level) >= self.coarsest()
    }

    /// The file level `level`'s tiles are decoded from: the original for
    /// level 0 of a region-readable original, and the level's mip level file
    /// otherwise.
    pub(super) fn file_of(&self, photograph: &Path, level: usize) -> PathBuf {
        if level == 0 && self.region_readable {
            return photograph.to_path_buf();
        }
        let quality = self
            .layout
            .options()
            .map_or(0, |options| options.quality_for_level(level));
        self.location.level_path(level, quality)
    }

    /// The mip level file of `level`, read and checked against what its name
    /// and the layout say.
    fn open(&self, photograph: &Path, level: usize) -> Option<MipLevelFile> {
        let expected = ExpectedMipLevel::new(&self.layout, self.source.relative(), level)?;
        MipLevelFile::open(&self.file_of(photograph, level), &expected).ok()
    }

    /// Whether `record`, of a file of this mipmap, is current for the
    /// original and agrees with the coarsest file's: the same number of
    /// stored levels and the same region answer.
    fn is_current(&self, record: &MipLevelRecord) -> bool {
        self.source.check(record).is_ok()
            && record.stored_levels as usize == self.coarsest()
            && record.region_readable_original == self.region_readable
    }
}

/// What a level's tiles are decoded from: a mip level file, or a
/// region-readable original.
pub(super) enum LevelSource {
    Mip(MipLevelFile),
    Original(OriginalRegions),
}

impl LevelSource {
    /// The bytes it holds, what its file entry counts against the budget.
    fn held_bytes(&self) -> u64 {
        match self {
            LevelSource::Mip(file) => file.held_bytes(),
            LevelSource::Original(original) => original.held_bytes(),
        }
    }

    /// Pixels `x0..x1` by `y0..y1` of the level, as RGB; `Err` when the
    /// intervals they need do not decode.
    fn decode_region(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> Result<ImageU8, String> {
        match self {
            LevelSource::Mip(file) => file
                .decode_region(x0, y0, x1, y1)
                .map_err(|e| e.to_string()),
            LevelSource::Original(original) => original
                .decode_region(x0, y0, x1, y1)
                .map_err(|e| e.to_string()),
        }
    }
}

/// What [`PhotographCache::check_files`] found for a photograph.
pub(super) enum Checked {
    /// Read in core: in a workspace without `mipmap_options`, under the
    /// threshold, too thin to store a reduced level, or with a header that
    /// cannot be read (whose decode then reports it).
    InCore,
    /// Out of core, and its coarsest mip level file missing or stale, or a
    /// file of its mipmap found not current by a read ([`Stale`]).
    NotCurrent,
    /// Out of core with its coarsest file current: the mipmap, and that file.
    Current(Arc<MipmapFiles>, Box<MipLevelFile>),
}

/// A file of an out-of-core photograph's mipmap that a read found missing,
/// not current or not decoding, which a check of the coarsest file alone
/// does not show (a finer file, `-mip0`, or the original read as level 0).
///
/// While the original is the one it was found under and no build has made
/// the mipmap since, [`PhotographCache::check_files`] answers the mipmap not
/// current, so the next `get` or `get_many` makes it, rewriting every level
/// whose file is not current, rather than take the coarsest file's word and
/// read the same file again. A build mends a file that is missing or whose
/// record is not current by writing it, so such a file found again, gone or
/// with another length or modification time (deleted again, or rewritten by
/// another process), is a first find once more. A read that finds the same
/// file not current again after that build, still there with the length and
/// modification time it had when first found, has found one the build left
/// as it was because its record is current, though its data does not
/// decode: no build mends it, and the photograph is then remembered as
/// unreadable until its original changes or the cache forgets it, so
/// neither a panel nor *Build mipmaps* asks for it again and again.
pub(super) struct Stale {
    /// The original's length and modification time when the file was found.
    stamp: FileStamp,
    /// The file found not current.
    file: PathBuf,
    /// That file's own length and modification time when it was found,
    /// `None` when it was missing.
    file_stamp: FileStamp,
    /// Whether a build has made the mipmap since.
    rebuilt: bool,
}

/// How a slot outside the map reads its photograph, under a budget of 0,
/// as [`PhotographCache::unmapped_files_slot`] decides it.
pub(super) enum Unmapped {
    /// Whole in core.
    InCore,
    /// Out of core, its mipmap missing or stale.
    NotCurrent,
    /// From its files, through the slot made to read them.
    Files(Arc<Slot>, Arc<MipmapFiles>),
}

/// A file entry: a level's source, kept under the cache's budget while tiles
/// of it are read.
pub(super) struct FileEntry {
    pub(super) source: Arc<LevelSource>,
    /// What it counts against the budget.
    pub(super) bytes: u64,
    /// The cache's tick when it was kept or last read.
    pub(super) tick: u64,
}

/// Why a request could not answer the tiles it was asked for.
pub(super) enum Failure {
    /// The photograph could not be read, or the caller was cancelled.
    Get(GetError),
    /// A file of the photograph's mipmap is missing, not current or not the
    /// one first read: the named file.
    Changed(PathBuf),
    /// A level of a mipmap held in memory is no longer kept, and the request
    /// is a panel's, which never makes a mipmap: it decodes the photograph
    /// in core instead.
    NotHeld,
}

impl Failure {
    /// The [`GetError`] a caller that never sees [`Failure::NotHeld`] answers
    /// for one, rather than panic on a broken invariant.
    pub(super) fn not_held(path: &Path) -> GetError {
        GetError::Unreadable(format!(
            "{}: a level of its mipmap held in memory was not kept",
            path.display()
        ))
    }
}

impl From<GetError> for Failure {
    fn from(e: GetError) -> Self {
        Failure::Get(e)
    }
}

impl From<Cancelled> for Failure {
    fn from(_: Cancelled) -> Self {
        Failure::Get(GetError::Cancelled)
    }
}

/// Whether a photograph of `width` x `height` read under `setting` is out of
/// core, so its tier is decided from its files.
pub(super) fn is_out_of_core(setting: &WorkspaceSetting, width: u32, height: u32) -> bool {
    matches!(&setting.options, Ok(Some(o)) if width.max(height) > o.out_of_core_long_side)
}

/// Every tile key of `layout`.
fn every_key(layout: &PyramidLayout) -> Vec<TileKey> {
    (0..layout.levels())
        .flat_map(|level| {
            let (cols, rows) = layout.tiles(level);
            (0..rows).flat_map(move |row| {
                (0..cols).map(move |col| TileKey {
                    level: level as u8,
                    col,
                    row,
                })
            })
        })
        .collect()
}

/// The photograph tile `key` of `layout`, with its border, decoded from
/// `source`, the file of its level: the tile's core and the border pixels
/// inside the level, decoded together, and the edge pixel repeated at the
/// level's edge.
fn tile_of(
    source: &LevelSource,
    layout: &PyramidLayout,
    key: TileKey,
) -> Result<PhotographTile, String> {
    let ts = layout.tile_size();
    let (w, h) = layout.level_size(usize::from(key.level));
    let core = [key.col * ts, key.row * ts];
    let size = [ts.min(w - core[0]), ts.min(h - core[1])];
    let (x0, y0) = (core[0].saturating_sub(1), core[1].saturating_sub(1));
    let (x1, y1) = (
        (core[0] + size[0] + 1).min(w),
        (core[1] + size[1] + 1).min(h),
    );
    let region = source.decode_region(x0, y0, x1, y1)?;
    Ok(tile_from_region(layout, key, &region, [x0, y0]))
}

impl PhotographCache {
    /// The facts of the original at `path`, whose mipmap is at `location`, as
    /// its records are checked against: the ones kept for this spelling of
    /// its path while its length and modification time stay the same, and
    /// otherwise read again and kept. `None` when it cannot be read.
    pub(super) fn source_for(
        &self,
        path: &Path,
        location: &MipmapLocation,
    ) -> Option<Arc<MipmapSource>> {
        let stamp = file_stamp(path);
        let mut sources = self
            .shared
            .sources
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some((kept, source)) = sources.get(path) {
            if *kept == stamp {
                return Some(Arc::clone(source));
            }
        }
        let source = Arc::new(MipmapSource::read(path, location).ok()?);
        sources.insert(path.to_path_buf(), (stamp, Arc::clone(&source)));
        Some(source)
    }

    /// Whether a read has found a file of `path`'s mipmap not current
    /// ([`Stale`]) under the original as it is now, with no build since. An
    /// entry found under another original is dropped.
    fn found_stale(&self, path: &Path) -> bool {
        let stamp = file_stamp(path);
        let mut stale = self.shared.stale.lock().unwrap_or_else(|e| e.into_inner());
        match stale.get(path) {
            Some(found) if found.stamp != stamp => {
                stale.remove(path);
                false
            }
            Some(found) => !found.rebuilt,
            None => false,
        }
    }

    /// Note that a build has made `path`'s mipmap, so a file a read found
    /// not current before is made again or found current
    /// ([`Stale::rebuilt`]).
    pub(super) fn made_since_stale(&self, path: &Path) {
        let mut stale = self.shared.stale.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(found) = stale.get_mut(path) {
            found.rebuilt = true;
        }
    }

    /// Note that a read found `file`, a file of `path`'s mipmap, not current,
    /// and answer whether it found the same file so after a build made the
    /// mipmap again under the same original, the file still there and
    /// unchanged since it was first found: a file a build does not mend.
    fn note_stale(&self, path: &Path, file: &Path) -> bool {
        let stamp = file_stamp(path);
        let found_stamp = file_stamp(file);
        let mut stale = self.shared.stale.lock().unwrap_or_else(|e| e.into_inner());
        let again = stale.get(path).is_some_and(|found| {
            found.rebuilt
                && found.stamp == stamp
                && found.file == file
                && found_stamp.is_some()
                && found.file_stamp == found_stamp
        });
        if !again {
            stale.insert(
                path.to_path_buf(),
                Stale {
                    stamp,
                    file: file.to_path_buf(),
                    file_stamp: found_stamp,
                    rebuilt: false,
                },
            );
        }
        again
    }

    /// Whether `path`'s mipmap was last made by a build that could not write
    /// its files, under the original as it is now.
    pub(super) fn held_in_memory(&self, path: &Path) -> bool {
        let stamp = file_stamp(path);
        let not_written = self
            .shared
            .not_written
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        not_written.get(path) == Some(&stamp)
    }

    /// Whether the photograph at `path`, read under `setting`, is out of core
    /// and whether its coarsest mip level file is current, with the mipmap
    /// and that file when it is. Reads the header, and for an out-of-core
    /// photograph the coarsest file and the original's length and time,
    /// hashing the original only when its time is not the record's.
    pub(super) fn check_files(&self, path: &Path, setting: &WorkspaceSetting) -> Checked {
        let Ok(Some(options)) = &setting.options else {
            return Checked::InCore;
        };
        let Ok((width, height)) = image_dimensions(path) else {
            return Checked::InCore;
        };
        if !is_out_of_core(setting, width, height) {
            return Checked::InCore;
        }
        let tile_size = self.shared.options.tile_size;
        let Some(layout) = PyramidLayout::for_size(width, height, Some(options), tile_size) else {
            return Checked::InCore;
        };
        // A photograph a pixel thin stores no reduced level, and is read in
        // core.
        let k = layout.stored_levels();
        let Some(location) = mipmap_location(path).filter(|_| k > 0) else {
            return Checked::InCore;
        };
        if self.found_stale(path) {
            return Checked::NotCurrent;
        }
        let current = || {
            let source = self.source_for(path, &location)?;
            let expected = ExpectedMipLevel::new(&layout, source.relative(), k)?;
            let file = location.level_path(k, options.jpeg_quality);
            let coarsest = MipLevelFile::open(&file, &expected).ok()?;
            let record = coarsest.record();
            if source.check(record).is_err() || record.stored_levels as usize != k {
                return None;
            }
            let region_readable = record.region_readable_original;
            Some((source, region_readable, coarsest))
        };
        match current() {
            Some((source, region_readable, coarsest)) => {
                let files = MipmapFiles {
                    layout: Arc::new(layout),
                    location,
                    source,
                    region_readable,
                    in_memory: false,
                };
                Checked::Current(Arc::new(files), Box::new(coarsest))
            }
            None => Checked::NotCurrent,
        }
    }

    /// How `slot` reads its photograph, deciding its tier the first time it
    /// is needed, so never [`Tier::Unchecked`]: from its files when the
    /// photograph is out of core and its coarsest file was current then, that
    /// file then being the slot's first file entry.
    pub(super) fn tier_of(&self, path: &Path, slot: &Arc<Slot>) -> Tier {
        let tier = slot.lock().tier.clone();
        if !matches!(tier, Tier::Unchecked) {
            return tier;
        }
        // Checked with no lock held; another caller may decide first, and
        // its answer is then the slot's.
        let checked = self.check_files(path, &slot.setting);
        let mut inner = self.lock();
        let mapped = inner
            .map
            .get(path)
            .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
        let mut data = slot.lock();
        if !matches!(data.tier, Tier::Unchecked) {
            return data.tier.clone();
        }
        match checked {
            Checked::InCore => data.tier = Tier::InCore,
            Checked::NotCurrent => data.tier = Tier::NotCurrent,
            Checked::Current(files, coarsest) => {
                let tier = Tier::Files(Arc::clone(&files));
                self.read_from(path, &mut inner, mapped, slot, data, files, vec![*coarsest]);
                return tier;
            }
        }
        data.tier.clone()
    }

    /// Make `slot`, whose data `data` is, read its photograph from `files`:
    /// its tier, its layout and the record of each of `opened`, the files a
    /// check or a build has read or made. When the slot is in the map
    /// (`mapped`), each of `opened` is kept as a file entry before entries are
    /// dropped down to the budget; outside the map, files held in memory are
    /// held by the slot, and others dropped, to be read from disk again. What
    /// the slot kept before, the tiles of a whole decode in core included, is
    /// dropped. The layout [`Self::request_layout`] kept for `path`, if any,
    /// becomes the files' one.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_from(
        &self,
        path: &Path,
        inner: &mut Inner,
        mapped: bool,
        slot: &Arc<Slot>,
        mut data: MutexGuard<'_, SlotData>,
        files: Arc<MipmapFiles>,
        opened: Vec<MipLevelFile>,
    ) {
        let freed = data.drop_entries();
        if mapped {
            inner.bytes -= freed;
        }
        data.records.clear();
        data.made = Some(Ok(Arc::clone(&files.layout)));
        data.channels = 3;
        for file in &opened {
            let record = file.record();
            data.records.insert(record.level as usize, record.clone());
        }
        if mapped {
            let added = self.keep_entries(&mut data, opened);
            inner.bytes += added;
        } else if files.in_memory {
            for file in opened {
                let level = file.record().level as usize;
                data.held.insert(level, Arc::new(LevelSource::Mip(file)));
            }
        }
        if let Some(kept) = inner.layouts.get_mut(path) {
            *kept = PeekedLayout::Known(Arc::clone(&files.layout));
        }
        data.tier = Tier::Files(files);
        drop(data);
        if mapped {
            self.evict(inner, slot, Asked::Keys(&[]), None);
        }
    }

    /// Keep each of `opened` as a file entry of `data`, a slot in the map,
    /// unless it keeps one for that level already, answering the bytes added,
    /// which the caller adds to [`Inner::bytes`].
    pub(super) fn keep_entries(&self, data: &mut SlotData, opened: Vec<MipLevelFile>) -> u64 {
        let mut added = 0;
        for file in opened {
            let level = file.record().level as usize;
            if data.files.contains_key(&level) {
                continue;
            }
            let source = Arc::new(LevelSource::Mip(file));
            let bytes = source.held_bytes();
            let tick = self.next_tick();
            data.files.insert(
                level,
                FileEntry {
                    source,
                    bytes,
                    tick,
                },
            );
            added += bytes;
        }
        data.bytes += added;
        added
    }

    /// For a cache of budget 0, which keeps no slot: how the photograph at
    /// `path`, read under `setting`, is read now, with a slot outside the map
    /// that reads its files when they are current. Each request decides so
    /// from the files as they are. Files found current make the layout
    /// [`Self::request_layout`] kept for `path`, if any, theirs.
    pub(super) fn unmapped_files_slot(&self, path: &Path, setting: &WorkspaceSetting) -> Unmapped {
        // A workspace without `mipmap_options` has no out-of-core
        // photograph: nothing is read for it, not even the stamp.
        if !matches!(setting.options, Ok(Some(_))) {
            return Unmapped::InCore;
        }
        let stamp = file_stamp(path);
        match self.check_files(path, setting) {
            Checked::InCore => Unmapped::InCore,
            Checked::NotCurrent => Unmapped::NotCurrent,
            Checked::Current(files, coarsest) => {
                let slot = Arc::new(Slot::new(stamp, setting.clone()));
                {
                    let mut data = slot.lock();
                    data.tier = Tier::Files(Arc::clone(&files));
                    data.made = Some(Ok(Arc::clone(&files.layout)));
                    data.records
                        .insert(files.coarsest(), coarsest.record().clone());
                }
                if let Some(kept) = self.lock().layouts.get_mut(path) {
                    *kept = PeekedLayout::Known(Arc::clone(&files.layout));
                }
                Unmapped::Files(slot, files)
            }
        }
    }

    /// The tiles `asked` names of the photograph at `path`, read from the
    /// files `slot` reads it from: those the slot keeps, and the rest decoded
    /// from their files, kept as used beside the other coarse tiles their
    /// decode cut, which are kept as unused. For [`Asked::Photograph`], the
    /// photograph reading through `slot`, with nothing read. A slot outside
    /// the map keeps nothing.
    ///
    /// Fails with [`Failure::Changed`] when the slot no longer reads from its
    /// files, or a file it reads is missing, not current, does not decode, or
    /// holds another record than the one the slot first read, and then marks
    /// the slot's mipmap not current ([`Self::mark_not_current`]), so the
    /// photograph's next `get` makes it again; and with
    /// [`GetError::Cancelled`] when `progress` is cancelled between files.
    ///
    /// A level of a mipmap held in memory that the slot no longer keeps is
    /// made again in memory when `may_build` (a kernel's read), and is
    /// otherwise [`Failure::NotHeld`] (a panel's), the slot left as it is.
    pub(super) fn read_files(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        asked: Asked<'_>,
        may_build: bool,
        progress: &Progress<'_>,
    ) -> Result<Got, Failure> {
        let changed = || Failure::Changed(path.to_path_buf());
        let files = slot.lock().tier.files().cloned().ok_or_else(changed)?;
        let layout = &files.layout;
        let keys: Vec<TileKey> = match asked {
            Asked::Photograph => {
                self.shared.hits.fetch_add(1, Ordering::Relaxed);
                #[cfg(test)]
                self.shared.reserved_reading_files.fetch_max(
                    self.shared.build_reserved.load(Ordering::SeqCst),
                    Ordering::SeqCst,
                );
                return Ok(Got {
                    layout: Arc::clone(layout),
                    tiles: Vec::new(),
                    channels: 3,
                    slot: Some(Arc::clone(slot)),
                });
            }
            Asked::Every => every_key(layout),
            Asked::Keys(keys) | Asked::Read(keys) => keys
                .iter()
                .copied()
                .filter(|&key| has_key(layout, key))
                .collect(),
        };
        let mut found: HashMap<TileKey, Arc<PhotographTile>> = HashMap::new();
        {
            let tick = self.next_tick();
            let mut data = slot.lock();
            if !data.tier.is(&files) {
                return Err(changed());
            }
            for &key in &keys {
                if let Some(kept) = data.tiles.get_mut(&key) {
                    (kept.used, kept.tick) = (true, tick);
                    found.insert(key, Arc::clone(&kept.tile));
                }
            }
        }
        let mut missing: Vec<TileKey> = keys
            .iter()
            .copied()
            .filter(|key| !found.contains_key(key))
            .collect();
        missing.sort_unstable_by_key(|key| (key.level, key.row, key.col));
        missing.dedup();
        if missing.is_empty() {
            self.shared.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.shared.misses.fetch_add(1, Ordering::Relaxed);
            let decoded = self.decode_missing(path, slot, &files, &missing, may_build, progress);
            let made = match decoded {
                Ok(made) => made,
                Err(Failure::Changed(file)) => {
                    // The mipmap is not current: from now on the slot treats
                    // the photograph as one whose files are stale, or, for a
                    // file a build did not mend, as unreadable.
                    return Err(match self.mark_not_current(path, slot, &file) {
                        Some(e) => Failure::Get(e),
                        None => Failure::Changed(file),
                    });
                }
                Err(e) => return Err(e),
            };
            if !self.keep_tiles(path, slot, &files, &made, &keys) {
                return Err(changed());
            }
            found.extend(made);
        }
        let tiles = keys
            .iter()
            .map(|key| found.get(key).map(Arc::clone).ok_or_else(changed))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Got {
            layout: Arc::clone(layout),
            tiles,
            channels: 3,
            slot: None,
        })
    }

    /// The tiles `missing` of `path`, decoded from `files`, the files `slot`
    /// reads: the coarse ones from one decode of level `K`, the others each
    /// from its level's file, polling `progress` between files.
    fn decode_missing(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        files: &Arc<MipmapFiles>,
        missing: &[TileKey],
        may_build: bool,
        progress: &Progress<'_>,
    ) -> Result<HashMap<TileKey, Arc<PhotographTile>>, Failure> {
        let mut made = HashMap::new();
        if missing.iter().any(|&key| files.is_coarse(key)) {
            progress.check_cancel()?;
            made = self.coarse_tiles(path, slot, files, may_build, progress)?;
        }
        let mut level = None;
        let mut source = None;
        for &key in missing.iter().filter(|&&key| !files.is_coarse(key)) {
            if level != Some(key.level) {
                progress.check_cancel()?;
                let l = usize::from(key.level);
                source = Some(self.level_source(path, slot, files, l, may_build, progress)?);
                level = Some(key.level);
            }
            let Some(source) = &source else {
                continue;
            };
            let tile = tile_of(source, &files.layout, key)
                .map_err(|_| Failure::Changed(files.file_of(path, usize::from(key.level))))?;
            made.insert(key, Arc::new(tile));
        }
        Ok(made)
    }

    /// Keep `made`, tiles of `path` decoded from `files`, in `slot` when it
    /// is in the map, those `asked` names as used and the rest as unused,
    /// then drop entries down to the budget, never one `asked` names. `false`
    /// when the slot no longer reads from `files`, so the tiles are of
    /// another version than its own.
    fn keep_tiles(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        files: &Arc<MipmapFiles>,
        made: &HashMap<TileKey, Arc<PhotographTile>>,
        asked: &[TileKey],
    ) -> bool {
        let mut inner = self.lock();
        let mapped = inner
            .map
            .get(path)
            .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
        let mut data = slot.lock();
        if !data.tier.is(files) {
            return false;
        }
        if !mapped {
            return true;
        }
        let tick = self.next_tick();
        let mut added = 0;
        for (&key, tile) in made {
            let used = asked.contains(&key);
            let kept = data.tiles.entry(key).or_insert_with(|| {
                added += tile_bytes(tile);
                Kept {
                    tile: Arc::clone(tile),
                    used: false,
                    tick,
                }
            });
            kept.tick = tick;
            kept.used |= used;
        }
        data.bytes += added;
        inner.bytes += added;
        drop(data);
        self.evict(&mut inner, slot, Asked::Keys(asked), None);
        true
    }

    /// The source of `level`'s tiles of `path`, whose slot reads it from
    /// `files`: the slot's file entry when it keeps one, and otherwise the
    /// file read, checked and kept as an entry. A mip level file read for the
    /// first time is checked as current and its record kept; read again,
    /// after its bytes were evicted, its record is compared with that one. A
    /// region-readable original is read when it is the one the slot was made
    /// from, and indexed, a failure of the index failing the read. A level of
    /// a mipmap held in memory is the one the slot holds or keeps, and one
    /// it no longer keeps is made again ([`Self::rebuild_in_memory`]) when
    /// `may_build`, and is [`Failure::NotHeld`] otherwise.
    pub(super) fn level_source(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        files: &Arc<MipmapFiles>,
        level: usize,
        may_build: bool,
        progress: &Progress<'_>,
    ) -> Result<Arc<LevelSource>, Failure> {
        {
            let tick = self.next_tick();
            let mut data = slot.lock();
            if !data.tier.is(files) {
                return Err(Failure::Changed(path.to_path_buf()));
            }
            if let Some(source) = data.kept_level(level, tick) {
                return Ok(source);
            }
        }
        // A build's files that could not be written are made again.
        if files.in_memory && !(level == 0 && files.region_readable) {
            if !may_build {
                return Err(Failure::NotHeld);
            }
            return self.rebuild_in_memory(path, slot, files, level, progress);
        }
        let file = files.file_of(path, level);
        let changed = || Failure::Changed(file.clone());
        let source = if level == 0 && files.region_readable {
            let bytes = std::fs::read(path).map_err(|_| changed())?;
            if file_stamp(path) != slot.stamp {
                return Err(changed());
            }
            LevelSource::Original(OriginalRegions::from_bytes(bytes.into()).map_err(|_| changed())?)
        } else {
            let opened = files.open(path, level).ok_or_else(changed)?;
            let first = !slot.lock().records.contains_key(&level);
            // The first read's check, which can hash the original, runs with
            // no lock held.
            if first && !files.is_current(opened.record()) {
                return Err(changed());
            }
            let mut data = slot.lock();
            let recorded = data
                .records
                .entry(level)
                .or_insert_with(|| opened.record().clone());
            if *recorded != *opened.record() {
                return Err(changed());
            }
            LevelSource::Mip(opened)
        };
        let source = Arc::new(source);
        self.keep_file(path, slot, files, level, &source);
        Ok(source)
    }

    /// Keep `source` as `slot`'s file entry for `level` when the slot is in
    /// the map and reads from `files`, then drop entries down to the budget.
    fn keep_file(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        files: &Arc<MipmapFiles>,
        level: usize,
        source: &Arc<LevelSource>,
    ) {
        let mut inner = self.lock();
        let mapped = inner
            .map
            .get(path)
            .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
        let mut data = slot.lock();
        if !mapped || !data.tier.is(files) || data.files.contains_key(&level) {
            return;
        }
        let bytes = source.held_bytes();
        let tick = self.next_tick();
        data.files.insert(
            level,
            FileEntry {
                source: Arc::clone(source),
                bytes,
                tick,
            },
        );
        data.bytes += bytes;
        inner.bytes += bytes;
        drop(data);
        self.evict(&mut inner, slot, Asked::Keys(&[]), None);
    }

    /// Every tile of level `K` and the levels below it, cut from one decode
    /// of level `K`'s file and the levels box-filtered from it, as
    /// [`ImageU8::downsample_2x`] makes them.
    fn coarse_tiles(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        files: &Arc<MipmapFiles>,
        may_build: bool,
        progress: &Progress<'_>,
    ) -> Result<HashMap<TileKey, Arc<PhotographTile>>, Failure> {
        let k = files.coarsest();
        let changed = || Failure::Changed(files.file_of(path, k));
        let source = self.level_source(path, slot, files, k, may_build, progress)?;
        let LevelSource::Mip(file) = &*source else {
            return Err(changed());
        };
        let layout = &files.layout;
        let mut level = file.decode().map_err(|_| changed())?;
        if (level.width(), level.height()) != layout.level_size(k) {
            return Err(changed());
        }
        let mut tiles = HashMap::new();
        for l in k..layout.levels() {
            cut_level(layout, l, &level, &mut tiles);
            if l + 1 < layout.levels() {
                level = level.downsample_2x();
            }
        }
        Ok(tiles)
    }

    /// Mark `slot`'s mipmap not current, a read having found `file`, a file
    /// of it, missing, not current or not decoding: its tiles and file
    /// entries are dropped, the layout `request_layout` kept for `path`
    /// becomes [`PeekedLayout::Building`], the file is noted ([`Stale`]), and
    /// from now on `get` makes the mipmap again and a panel's decode and a
    /// kernel's tile read answer `Changed`.
    ///
    /// When the same file was found so after a build made the mipmap again
    /// under the same original, a build does not mend it: the photograph is
    /// remembered as unreadable instead, with the error answered here, and
    /// the kept layout says so, so nothing asks for it again until the
    /// original changes. `None` otherwise, and when the slot no longer reads
    /// from its files (another request marked it first).
    pub(super) fn mark_not_current(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        file: &Path,
    ) -> Option<GetError> {
        // Only a slot that reads its files is marked, the first request to
        // find one of them not current.
        slot.lock().tier.files()?;
        // Noted with no other lock held.
        let again = self.note_stale(path, file).then(|| {
            GetError::Unreadable(format!(
                "{}, a file of the mipmap of {}, could not be read after the mipmap was \
                 made again",
                file.display(),
                path.display()
            ))
        });
        let mut inner = self.lock();
        let mapped = inner
            .map
            .get(path)
            .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
        {
            let mut data = slot.lock();
            data.tier.files()?;
            data.tier = Tier::NotCurrent;
            data.made = again.clone().map(Err);
            data.records.clear();
            let freed = data.drop_entries();
            if mapped {
                inner.bytes -= freed;
            }
        }
        if let Some(kept) = inner.layouts.get_mut(path) {
            *kept = match &again {
                Some(e) => PeekedLayout::Unreadable(e.to_string()),
                None => PeekedLayout::Building,
            };
        }
        again
    }
}
