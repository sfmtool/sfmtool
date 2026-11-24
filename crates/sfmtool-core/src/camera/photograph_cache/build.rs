// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Making an out-of-core photograph's mipmap in its slot: what `get` and
//! `get_many` do for a photograph whose mip level files are missing or
//! stale, before they read it from the files the build returned.
//!
//! The build runs on the caller's thread while the slot is marked as being
//! made, reserved on the build count as a whole decode is, so other callers
//! asking for the same path wait on the slot and then read what it made. A
//! cancelled maker leaves the slot as it was, and the waiter that takes the
//! build over makes only the levels whose files are not current, since
//! `build_mipmaps` reads a level whose file is current rather than make it
//! again. See the spec `specs/core/camera/photograph-cache.md` § "Making a
//! mipmap".

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::files::{cut_coarse, Checked, Failure, LevelSource, MipmapFiles, Tier};
use super::{
    file_stamp, header, whole_decode_bytes, Asked, CoarsestLevel, Found, GetError, Got, Making,
    PeekedLayout, PhotographCache, Share, Slot, WorkspaceSetting, COARSEST_PUBLISHED,
};
use crate::camera::mipmap::{
    build_mipmaps, mipmap_location, BuiltMipmap, MipLevelFile, MipmapError, PyramidLayout,
};
use crate::camera::tiled_photograph::TileKey;
use crate::progress::{Event, Progress};
use crate::progress_warn;

/// Whether a request makes an out-of-core photograph's mipmap when its files
/// are missing or stale.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Make {
    /// Never: a kernel's tile read ([`PhotographCache::get_tile`], a read
    /// through a photograph `get` answered, `view_region_pyramid` and
    /// `get_level`), which answers `Changed` instead.
    Never,
    /// `get`'s: makes it, reporting the build's phases through the caller's
    /// `Progress`.
    Reporting,
    /// A `get_many` member's: makes it, passing on only the build's
    /// messages, since the phases of members running in parallel would
    /// overwrite one another in the caller's display.
    Quietly,
}

/// Where a build publishes the photograph's coarsest level once it is
/// encoded, for a panel to draw while it waits ([`PeekedLayout::Building`]).
#[derive(Clone, Copy)]
pub(super) enum Publish<'s> {
    /// Nowhere: a build made again in memory for a kernel's read of a
    /// photograph already read from its files.
    Nowhere,
    /// In the slot the build is made in.
    Slot(&'s Arc<Slot>),
    /// Under a budget of 0, which keeps no slot: in the layout
    /// [`PhotographCache::request_layout`] kept for the photograph, when a
    /// panel asked for one and it is `Building`.
    Layouts,
}

/// The coarsest level of `layout`, a photograph's out-of-core layout, cut
/// from `file`, its level `K`'s file: the tile of level `K` and one of each
/// level below it. `None` when the file does not decode to level `K`'s size
/// or a level is more than one tile.
fn coarsest_level(layout: &PyramidLayout, file: &MipLevelFile) -> Option<CoarsestLevel> {
    let mut cut = cut_coarse(layout, file)?;
    let tiles = (layout.stored_levels()..layout.levels())
        .map(|level| {
            cut.remove(&TileKey {
                level: level as u8,
                col: 0,
                row: 0,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    cut.is_empty().then(|| CoarsestLevel {
        layout: Arc::new(layout.clone()),
        tiles,
    })
}

/// The [`GetError`] a failed build of `path`'s mipmap is answered and
/// remembered as.
fn build_error(path: &Path, e: MipmapError) -> GetError {
    match e {
        MipmapError::Cancelled => GetError::Cancelled,
        MipmapError::Workspace(e) => GetError::Workspace(e),
        MipmapError::TooLarge {
            path,
            width,
            height,
        } => GetError::TooLarge {
            path,
            width,
            height,
        },
        MipmapError::Io { error, .. } if error.kind() == std::io::ErrorKind::NotFound => {
            GetError::NotFound(path.to_path_buf())
        }
        e => GetError::Unreadable(format!(
            "the mipmap of {} could not be made: {e}",
            path.display()
        )),
    }
}

/// The photograph `slot` answers once its mipmap is made, which reads its
/// tiles from the files through the slot: no tile, and the slot. Fails with
/// [`Failure::Changed`] when another request has found the files not current
/// since.
fn photograph_of(path: &Path, slot: &Arc<Slot>) -> Result<Got, Failure> {
    let data = slot.lock();
    let files = data
        .tier
        .files()
        .ok_or_else(|| Failure::Changed(path.to_path_buf()))?;
    Ok(Got {
        layout: Arc::clone(&files.layout),
        tiles: Vec::new(),
        channels: 3,
        slot: Some(Arc::clone(slot)),
    })
}

impl PhotographCache {
    /// The photograph at `path`, an out-of-core photograph whose `slot` found
    /// its mipmap not current, its mipmap made by this request as `make`
    /// says: waiting while another caller makes the slot, polling `progress`
    /// for cancellation every [`super::SLOT_WAIT_POLL`], and reading the
    /// files it made; or, when the slot is not being made, marking it so and
    /// making the mipmap, unless its files have been made current since.
    /// `share` is a `get_many` member's share of the build count, or
    /// [`Share::None`] for a `get`, which reserves its one build when it
    /// marks the slot.
    pub(super) fn make_in_slot<'r>(
        &'r self,
        path: &Path,
        slot: &Arc<Slot>,
        progress: &Progress<'_>,
        share: Share<'r>,
        make: Make,
    ) -> (Result<Got, Failure>, Found) {
        #[cfg(test)]
        let mut told_waiting = false;
        let mut making = {
            let mut data = slot.lock();
            loop {
                if !data.making {
                    match &data.tier {
                        Tier::Files(_) => {
                            drop(data);
                            let asked = Asked::Photograph;
                            let read = self.read_files(path, slot, asked, true, progress);
                            return (read, Found::Reused);
                        }
                        Tier::NotCurrent => {
                            if let Some(Err(e)) = &data.made {
                                self.shared.hits.fetch_add(1, Ordering::Relaxed);
                                return (Err(Failure::Get(e.clone())), Found::Reused);
                            }
                            data.making = true;
                            break Making {
                                slot,
                                share,
                                finished: false,
                            };
                        }
                        // A slot's tier leaves `NotCurrent` only for
                        // `Files`; this reads it as its tier says rather
                        // than panic on a broken invariant.
                        Tier::Unchecked | Tier::InCore => {
                            drop(data);
                            let (value, found) =
                                self.fetch_in_core(path, slot, progress, share, Asked::Photograph);
                            return (value.map_err(Failure::Get), found);
                        }
                    }
                }
                if progress.is_cancelled() {
                    return (Err(Failure::Get(GetError::Cancelled)), Found::Reused);
                }
                #[cfg(test)]
                if !std::mem::replace(&mut told_waiting, true) {
                    self.slot_event(super::SlotEvent::WaitStarted, path);
                }
                data = slot
                    .ready
                    .wait_timeout(data, super::SLOT_WAIT_POLL)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        };
        // Another process, or a build of this one that finished before the
        // slot found its files stale, may have made them current since.
        let made = match self.check_files(path, &slot.setting) {
            Checked::Current(files, coarsest) => {
                self.shared.hits.fetch_add(1, Ordering::Relaxed);
                Ok((files, vec![*coarsest], Found::FromFiles))
            }
            Checked::InCore | Checked::NotCurrent => {
                self.shared.misses.fetch_add(1, Ordering::Relaxed);
                let publish = Publish::Slot(slot);
                self.build(
                    path,
                    progress,
                    &mut making.share,
                    &slot.setting,
                    make,
                    publish,
                )
                .map(|(files, opened)| (files, opened, Found::Made))
            }
        };
        if let Err(GetError::Cancelled) = made {
            // Dropping the hold gives the share back, leaves the slot as it
            // was and wakes its waiters, the first of which makes the
            // mipmap in this one's place.
            drop(making);
            return (Err(Failure::Get(GetError::Cancelled)), Found::Decoded);
        }
        making.finish();
        let answer = {
            let mut inner = self.lock();
            let mapped = inner
                .map
                .get(path)
                .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
            let mut data = slot.lock();
            data.making = false;
            match made {
                Ok((files, opened, found)) => {
                    self.read_from(path, &mut inner, mapped, slot, data, files, opened);
                    Ok(found)
                }
                Err(e) => {
                    let freed = data.drop_entries();
                    if mapped {
                        inner.bytes -= freed;
                    }
                    data.made = Some(Err(e.clone()));
                    Err(e)
                }
            }
        };
        slot.ready.notify_all();
        match answer {
            Ok(found) => (photograph_of(path, slot), found),
            Err(e) => (Err(Failure::Get(e)), Found::Decoded),
        }
    }

    /// For a cache of budget 0, which keeps no slot: make the mipmap of
    /// `path`, an out-of-core photograph whose files are missing or stale,
    /// read under `setting`, and answer the photograph, read through a slot
    /// outside the map that reads the files the build made.
    pub(super) fn make_unmapped<'r>(
        &'r self,
        path: &Path,
        progress: &Progress<'_>,
        mut share: Share<'r>,
        setting: WorkspaceSetting,
        make: Make,
    ) -> (Result<Got, Failure>, Found) {
        self.shared.misses.fetch_add(1, Ordering::Relaxed);
        let stamp = file_stamp(path);
        let built = self.build(path, progress, &mut share, &setting, make, Publish::Layouts);
        share.release();
        let (files, opened) = match built {
            Ok(built) => built,
            Err(e) => {
                // A failed build's coarsest level is not drawn; a cancelled
                // one's is, until the next build publishes its own.
                if !matches!(e, GetError::Cancelled) {
                    if let Some(PeekedLayout::Building { coarsest }) =
                        self.lock().layouts.get_mut(path)
                    {
                        *coarsest = None;
                    }
                }
                return (Err(Failure::Get(e)), Found::Decoded);
            }
        };
        let slot = Arc::new(Slot::new(stamp, setting));
        {
            let mut inner = self.lock();
            self.read_from(path, &mut inner, false, &slot, slot.lock(), files, opened);
        }
        (photograph_of(path, &slot), Found::Made)
    }

    /// Make the mipmap of the photograph at `path`, read under `setting`,
    /// with [`build_mipmaps`]: the mipmap its files are read from, and every
    /// file it made. A build whose files could not be written is read from
    /// those files in memory ([`MipmapFiles::in_memory`]), with a warning
    /// through `progress` the first time for each photograph, not again when
    /// a level evicted since is made again.
    /// Reads the header first, refusing the photograph as [`header`] does,
    /// and when `share` holds nothing reserves the build as a whole decode
    /// with [`Self::reserve_builds`] before reading the pixels.
    ///
    /// Reports the phases `decode`, `reduce` and `encode` through `progress`
    /// for [`Make::Reporting`], and only messages for [`Make::Quietly`], and
    /// polls it for cancellation between the build's stages, answering
    /// [`GetError::Cancelled`] with the levels already written left current.
    ///
    /// Once the coarsest level is encoded, or found current, it is cut into a
    /// [`CoarsestLevel`] and published where `publish` says, and
    /// [`COARSEST_PUBLISHED`] reported through `progress` whatever `make`
    /// says, so a caller that draws can repaint: one value a build.
    ///
    /// Must not use rayon, for the reason [`Self::decode`] gives: it runs
    /// while its slot is marked as being made. `build_mipmaps`, the decoder
    /// and the encoder use none.
    fn build<'r>(
        &'r self,
        path: &Path,
        progress: &Progress<'_>,
        share: &mut Share<'r>,
        setting: &WorkspaceSetting,
        make: Make,
        publish: Publish<'_>,
    ) -> Result<(Arc<MipmapFiles>, Vec<MipLevelFile>), GetError> {
        let (width, height) = header(path, setting)?;
        let not_out_of_core = || {
            GetError::Unreadable(format!(
                "{} has no mipmap to make: it is not out of core",
                path.display()
            ))
        };
        let Ok(Some(options)) = &setting.options else {
            return Err(not_out_of_core());
        };
        if let Share::None = share {
            *share = Share::Own(self.reserve_builds(whole_decode_bytes(width, height), 1));
        }
        #[cfg(test)]
        self.slot_event(super::SlotEvent::DecodeStarted, path);
        let messages = |event: Event<'_>| match event {
            Event::Message { .. }
            | Event::Value {
                name: COARSEST_PUBLISHED,
                ..
            } => progress.forward(event),
            _ => {}
        };
        let quiet = progress.tapped(&messages);
        let reporting = match make {
            Make::Quietly => &quiet,
            Make::Never | Make::Reporting => progress,
        };
        let tile_size = self.shared.options.tile_size;
        // No layout, nothing published: a build made again in memory.
        let layout = PyramidLayout::for_size(width, height, Some(options), tile_size)
            .filter(|_| !matches!(publish, Publish::Nowhere));
        let mut published = false;
        let mut on_level = |file: &MipLevelFile| {
            let Some(layout) = &layout else {
                return;
            };
            let k = layout.stored_levels();
            if published || file.record().level as usize != k {
                return;
            }
            published = true;
            if let Some(coarsest) = coarsest_level(layout, file) {
                self.publish_coarsest(path, publish, Arc::new(coarsest));
                #[cfg(test)]
                self.slot_event(super::SlotEvent::CoarsestPublished, path);
                reporting.value(COARSEST_PUBLISHED, k as f64);
            }
        };
        let built = build_mipmaps(path, None, options, tile_size, reporting, &mut on_level)
            .map_err(|e| build_error(path, e))?;
        self.shared.mipmaps_made.fetch_add(1, Ordering::Relaxed);
        let location = mipmap_location(path).ok_or_else(not_out_of_core)?;
        let source = self.source_for(path, &location).ok_or_else(|| {
            GetError::Unreadable(format!("{}: it could not be read again", path.display()))
        })?;
        let BuiltMipmap {
            layout,
            region_readable,
            files,
            not_written,
        } = built;
        self.made_since_stale(path);
        let stamp = file_stamp(path);
        let first = {
            let mut held = self
                .shared
                .not_written
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            match &not_written {
                Some(_) => held.insert(path.to_path_buf(), stamp).is_none(),
                None => {
                    held.remove(path);
                    false
                }
            }
        };
        if let Some(why) = &not_written {
            if first {
                progress_warn!(
                    reporting,
                    "The mipmap of {} is held in memory, since its files could not be \
                     written: {why}",
                    path.display()
                );
            }
        }
        let made = MipmapFiles {
            layout: Arc::new(layout),
            location,
            source,
            region_readable,
            in_memory: not_written.is_some(),
        };
        Ok((Arc::new(made), files))
    }

    /// Publish `coarsest`, the coarsest level a build of `path` has encoded,
    /// where `publish` says: in the slot, which answers it to
    /// [`Self::layout`] while the photograph is not current, or under a
    /// budget of 0 in the layout [`Self::request_layout`] kept for `path`
    /// when it is `Building`, which a panel then takes from
    /// [`Self::layout`]. Replaces one a build published before.
    fn publish_coarsest(&self, path: &Path, publish: Publish<'_>, coarsest: Arc<CoarsestLevel>) {
        match publish {
            Publish::Nowhere => {}
            Publish::Slot(slot) => slot.lock().coarsest = Some(coarsest),
            Publish::Layouts => {
                if let Some(PeekedLayout::Building { coarsest: kept }) =
                    self.lock().layouts.get_mut(path)
                {
                    *kept = Some(coarsest);
                }
            }
        }
    }

    /// For a kernel's read of `level` of `path`, whose `slot` reads its
    /// mipmap from `files`, a build's files held in memory, and no longer
    /// keeps that level: make the build again in memory, as a build is made,
    /// reserved on the build count, and answer the level. While another
    /// caller makes it in the slot this waits, polling `progress`, and reads
    /// what it made. The build is deterministic, so each level's record is
    /// the one the slot first read, and one that is not fails the read with
    /// [`Failure::Changed`] naming the file. Every level made is kept, as an
    /// entry of a slot in the map or held by one outside it, and the slot's
    /// own eviction spares the level asked for whatever its size, and every
    /// level of the photograph while they are under half the budget, so one
    /// build then serves the rest of the read.
    pub(super) fn rebuild_in_memory(
        &self,
        path: &Path,
        slot: &Arc<Slot>,
        files: &Arc<MipmapFiles>,
        level: usize,
        progress: &Progress<'_>,
    ) -> Result<Arc<LevelSource>, Failure> {
        let changed = || Failure::Changed(path.to_path_buf());
        let mut making = {
            let mut data = slot.lock();
            loop {
                if !data.tier.is(files) {
                    return Err(changed());
                }
                if let Some(source) = data.kept_level(level, self.next_tick()) {
                    return Ok(source);
                }
                if !data.making {
                    data.making = true;
                    break Making {
                        slot,
                        share: Share::None,
                        finished: false,
                    };
                }
                progress.check_cancel()?;
                data = slot
                    .ready
                    .wait_timeout(data, super::SLOT_WAIT_POLL)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        };
        self.shared.misses.fetch_add(1, Ordering::Relaxed);
        let built = self.build(
            path,
            progress,
            &mut making.share,
            &slot.setting,
            Make::Quietly,
            Publish::Nowhere,
        );
        // Dropping the hold on a failure gives the share back, leaves the
        // slot as it was and wakes its waiters.
        let (_, opened) = built?;
        making.finish();
        let answer = {
            let mut inner = self.lock();
            let mapped = inner
                .map
                .get(path)
                .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
            let mut data = slot.lock();
            data.making = false;
            let differs = opened.iter().find(|file| {
                let record = file.record();
                data.records.get(&(record.level as usize)) != Some(record)
            });
            if !data.tier.is(files) {
                Err(changed())
            } else if let Some(file) = differs {
                let level = file.record().level as usize;
                Err(Failure::Changed(files.file_of(path, level)))
            } else if mapped {
                let added = self.keep_entries(&mut data, opened);
                inner.bytes += added;
                let tick = self.next_tick();
                let source = data.kept_level(level, tick);
                drop(data);
                // Every level is spared while under half the budget, and the
                // one asked for whatever its size.
                self.evict(&mut inner, slot, Asked::Keys(&[]), Some(level));
                source.ok_or_else(changed)
            } else {
                for file in opened {
                    let level = file.record().level as usize;
                    data.held.insert(level, Arc::new(LevelSource::Mip(file)));
                }
                data.kept_level(level, self.next_tick()).ok_or_else(changed)
            }
        };
        slot.ready.notify_all();
        answer
    }
}
