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
//! Each finer file is checked when it is first opened. Every other
//! photograph, an out-of-core one whose files are missing or stale included,
//! is read whole in core. See the spec `specs/core/camera/photograph-cache.md`
//! § "Out-of-core photographs".

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{
    file_stamp, has_key, tile_bytes, Asked, GetError, Got, Kept, PhotographCache, Slot,
    WorkspaceSetting,
};
use crate::camera::image::{image_dimensions, ImageU8};
use crate::camera::mipmap::{
    mipmap_location, ExpectedMipLevel, MipLevelFile, MipLevelRecord, MipmapLocation, MipmapSource,
    OriginalRegions, PyramidLayout,
};
use crate::camera::tiled_photograph::{cut_level, tile_from_region, PhotographTile, TileKey};
use crate::progress::{Cancelled, Progress};

/// How a slot reads its photograph's pixels.
#[derive(Default)]
pub(super) enum Tier {
    /// Not decided yet: no request has needed the photograph.
    #[default]
    Unchecked,
    /// Whole, from a decode of the original: an in-core photograph, an
    /// out-of-core one whose mipmap was missing or stale when the slot
    /// checked it, or one whose files a panel's request found not current
    /// since.
    InCore,
    /// From its mip level files, current when the slot checked them.
    Files(Arc<MipmapFiles>),
}

impl Tier {
    /// The files, when the slot reads from them.
    pub(super) fn files(&self) -> Option<&Arc<MipmapFiles>> {
        match self {
            Tier::Files(files) => Some(files),
            Tier::Unchecked | Tier::InCore => None,
        }
    }

    /// Whether the slot reads from `files`, the ones a request started with.
    fn is(&self, files: &Arc<MipmapFiles>) -> bool {
        self.files().is_some_and(|f| Arc::ptr_eq(f, files))
    }
}

/// An out-of-core photograph's mipmap, found current: its layout, where its
/// files are, the original its records are checked against, and whether its
/// level 0 is read from the original.
pub(super) struct MipmapFiles {
    /// The out-of-core layout, of the workspace's `mipmap_options`.
    pub(super) layout: Arc<PyramidLayout>,
    location: MipmapLocation,
    source: Arc<MipmapSource>,
    /// The region answer of the coarsest file's record: level 0 is read
    /// from the original when `true`, and from `-mip0` otherwise.
    region_readable: bool,
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
    fn file_of(&self, photograph: &Path, level: usize) -> PathBuf {
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

/// A file entry: a level's source, kept under the cache's budget while tiles
/// of it are read.
pub(super) struct FileEntry {
    source: Arc<LevelSource>,
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
    fn source_for(&self, path: &Path, location: &MipmapLocation) -> Option<Arc<MipmapSource>> {
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

    /// The mipmap of the photograph at `path`, read under `setting`, when the
    /// photograph is out of core and its coarsest mip level file is current,
    /// with that file; `None` for every other photograph, which is read in
    /// core. Reads the header, and for an out-of-core photograph the coarsest
    /// file and the original's length and time, hashing the original only
    /// when its time is not the record's.
    fn check_files(
        &self,
        path: &Path,
        setting: &WorkspaceSetting,
    ) -> Option<(Arc<MipmapFiles>, MipLevelFile)> {
        let Ok(Some(options)) = &setting.options else {
            return None;
        };
        let (width, height) = image_dimensions(path).ok()?;
        if !is_out_of_core(setting, width, height) {
            return None;
        }
        let layout =
            PyramidLayout::for_size(width, height, Some(options), self.shared.options.tile_size)?;
        // A photograph a pixel thin stores no reduced level, and is read in
        // core.
        let k = layout.stored_levels();
        if k == 0 {
            return None;
        }
        let location = mipmap_location(path)?;
        let source = self.source_for(path, &location)?;
        let expected = ExpectedMipLevel::new(&layout, source.relative(), k)?;
        let coarsest =
            MipLevelFile::open(&location.level_path(k, options.jpeg_quality), &expected).ok()?;
        let record = coarsest.record();
        if source.check(record).is_err() || record.stored_levels as usize != k {
            return None;
        }
        let files = MipmapFiles {
            layout: Arc::new(layout),
            location,
            source,
            region_readable: record.region_readable_original,
        };
        Some((Arc::new(files), coarsest))
    }

    /// The mipmap `slot` reads its photograph from, deciding its tier the
    /// first time it is needed: `Some` when the photograph is out of core and
    /// its files were current then, `None` when it is read in core. The
    /// coarsest file, read for the check, is the slot's first file entry.
    pub(super) fn tier_of(&self, path: &Path, slot: &Arc<Slot>) -> Option<Arc<MipmapFiles>> {
        match &slot.lock().tier {
            Tier::Unchecked => {}
            Tier::InCore => return None,
            Tier::Files(files) => return Some(Arc::clone(files)),
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
        match &data.tier {
            Tier::Unchecked => {}
            Tier::InCore => return None,
            Tier::Files(files) => return Some(Arc::clone(files)),
        }
        let Some((files, coarsest)) = checked else {
            data.tier = Tier::InCore;
            return None;
        };
        let k = files.coarsest();
        data.tier = Tier::Files(Arc::clone(&files));
        data.made = Some(Ok(Arc::clone(&files.layout)));
        data.records.insert(k, coarsest.record().clone());
        if mapped {
            let source = Arc::new(LevelSource::Mip(coarsest));
            let bytes = source.held_bytes();
            let tick = self.next_tick();
            data.files.insert(
                k,
                FileEntry {
                    source,
                    bytes,
                    tick,
                },
            );
            data.bytes += bytes;
            inner.bytes += bytes;
            drop(data);
            self.evict(&mut inner, slot, Asked::Keys(&[]));
        }
        Some(files)
    }

    /// For a cache of budget 0, which keeps no slot: a slot outside the map
    /// for the photograph at `path`, read under `setting`, with its mipmap,
    /// when its files are current now; `None` when it is read in core. Each
    /// request decides so from the files as they are.
    pub(super) fn unmapped_files_slot(
        &self,
        path: &Path,
        setting: &WorkspaceSetting,
    ) -> Option<(Arc<Slot>, Arc<MipmapFiles>)> {
        // A workspace without `mipmap_options` has no out-of-core
        // photograph: nothing is read for it, not even the stamp.
        if !matches!(setting.options, Ok(Some(_))) {
            return None;
        }
        let stamp = file_stamp(path);
        let (files, coarsest) = self.check_files(path, setting)?;
        let slot = Arc::new(Slot::new(stamp, setting.clone()));
        {
            let mut data = slot.lock();
            data.tier = Tier::Files(Arc::clone(&files));
            data.made = Some(Ok(Arc::clone(&files.layout)));
            data.records
                .insert(files.coarsest(), coarsest.record().clone());
        }
        Some((slot, files))
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
    /// photograph's next requests read it in core; and with
    /// [`GetError::Cancelled`] when `progress` is cancelled between files.
    pub(super) fn read_files(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        asked: Asked<'_>,
        progress: &Progress<'_>,
    ) -> Result<Got, Failure> {
        let changed = || Failure::Changed(path.to_path_buf());
        let files = slot.lock().tier.files().cloned().ok_or_else(changed)?;
        let layout = &files.layout;
        let keys: Vec<TileKey> = match asked {
            Asked::Photograph => {
                self.shared.hits.fetch_add(1, Ordering::Relaxed);
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
            let made = match self.decode_missing(path, slot, &files, &missing, progress) {
                Ok(made) => made,
                Err(Failure::Changed(file)) => {
                    // The mipmap is not current: from now on the slot reads
                    // the photograph in core, as one whose files are stale.
                    self.mark_not_current(path, slot);
                    return Err(Failure::Changed(file));
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
        progress: &Progress<'_>,
    ) -> Result<HashMap<TileKey, Arc<PhotographTile>>, Failure> {
        let mut made = HashMap::new();
        if missing.iter().any(|&key| files.is_coarse(key)) {
            progress.check_cancel()?;
            made = self.coarse_tiles(path, slot, files)?;
        }
        let mut level = None;
        let mut source = None;
        for &key in missing.iter().filter(|&&key| !files.is_coarse(key)) {
            if level != Some(key.level) {
                progress.check_cancel()?;
                let l = usize::from(key.level);
                source = Some(self.level_source(path, slot, files, l)?);
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
        self.evict(&mut inner, slot, Asked::Keys(asked));
        true
    }

    /// The source of `level`'s tiles of `path`, whose slot reads it from
    /// `files`: the slot's file entry when it keeps one, and otherwise the
    /// file read, checked and kept as an entry. A mip level file read for the
    /// first time is checked as current and its record kept; read again,
    /// after its bytes were evicted, its record is compared with that one. A
    /// region-readable original is read when it is the one the slot was made
    /// from, and indexed, a failure of the index failing the read.
    fn level_source(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        files: &Arc<MipmapFiles>,
        level: usize,
    ) -> Result<Arc<LevelSource>, Failure> {
        {
            let tick = self.next_tick();
            let mut data = slot.lock();
            if !data.tier.is(files) {
                return Err(Failure::Changed(path.to_path_buf()));
            }
            if let Some(entry) = data.files.get_mut(&level) {
                entry.tick = tick;
                return Ok(Arc::clone(&entry.source));
            }
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
        self.evict(&mut inner, slot, Asked::Keys(&[]));
    }

    /// Every tile of level `K` and the levels below it, cut from one decode
    /// of level `K`'s file and the levels box-filtered from it, as
    /// [`ImageU8::downsample_2x`] makes them.
    fn coarse_tiles(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        files: &Arc<MipmapFiles>,
    ) -> Result<HashMap<TileKey, Arc<PhotographTile>>, Failure> {
        let k = files.coarsest();
        let changed = || Failure::Changed(files.file_of(path, k));
        let source = self.level_source(path, slot, files, k)?;
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

    /// Mark `slot`'s mipmap not current: it reads its photograph whole in
    /// core from now on, its tiles and file entries from the files dropped,
    /// and the layout `request_layout` kept for `path` becomes the in-core
    /// one. What a request that finds a file not current does: a kernel's
    /// read then fails with `Changed`, and a panel's decodes the tiles in
    /// core, so it never answers `Changed`.
    pub(super) fn mark_not_current(&self, path: &Path, slot: &Arc<Slot>) {
        let mut inner = self.lock();
        let mapped = inner
            .map
            .get(path)
            .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
        let (width, height) = {
            let mut data = slot.lock();
            let Some(files) = data.tier.files() else {
                return;
            };
            let size = files.layout.level_size(0);
            data.tier = Tier::InCore;
            data.made = None;
            data.records.clear();
            let freed = data.drop_entries();
            if mapped {
                inner.bytes -= freed;
            }
            size
        };
        if let Some(Ok(layout)) = inner.layouts.get(path) {
            if layout.is_out_of_core() {
                let in_core = Arc::new(self.layout_for(width, height));
                inner.layouts.insert(path.to_path_buf(), Ok(in_core));
            }
        }
    }
}
