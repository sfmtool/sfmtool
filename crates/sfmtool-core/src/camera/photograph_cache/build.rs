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

use super::files::{Checked, Failure, LevelSource, MipmapFiles, Tier};
use super::{
    file_stamp, header, whole_decode_bytes, Asked, Found, GetError, Got, Making, PhotographCache,
    Share, Slot, WorkspaceSetting,
};
use crate::camera::mipmap::{
    build_mipmaps, mipmap_location, BuiltMipmap, MipLevelFile, MipmapError,
};
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
                            let read = self.read_files(path, slot, Asked::Photograph, progress);
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
                Ok((files, vec![*coarsest], Found::Reused))
            }
            Checked::InCore | Checked::NotCurrent => {
                self.shared.misses.fetch_add(1, Ordering::Relaxed);
                self.build(path, progress, &mut making.share, &slot.setting, make)
                    .map(|(files, opened)| (files, opened, Found::Decoded))
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
                    self.read_from(&mut inner, mapped, slot, data, files, opened);
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
        let built = self.build(path, progress, &mut share, &setting, make);
        share.release();
        let (files, opened) = match built {
            Ok(built) => built,
            Err(e) => return (Err(Failure::Get(e)), Found::Decoded),
        };
        let slot = Arc::new(Slot::new(stamp, setting));
        {
            let mut inner = self.lock();
            self.read_from(&mut inner, false, &slot, slot.lock(), files, opened);
        }
        (photograph_of(path, &slot), Found::Decoded)
    }

    /// Make the mipmap of the photograph at `path`, read under `setting`,
    /// with [`build_mipmaps`]: the mipmap its files are read from, and the
    /// files it wrote, or for a build whose files could not be written the
    /// mipmap read from them in memory, with a warning through `progress`.
    /// Reads the header first, refusing the photograph as [`header`] does,
    /// and when `share` holds nothing reserves the build as a whole decode
    /// with [`Self::reserve_builds`] before reading the pixels.
    ///
    /// Reports the phases `decode`, `reduce` and `encode` through `progress`
    /// for [`Make::Reporting`], and only messages for [`Make::Quietly`], and
    /// polls it for cancellation between the build's stages, answering
    /// [`GetError::Cancelled`] with the levels already written left current.
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
        let messages = |event: Event<'_>| {
            if let Event::Message { .. } = event {
                progress.forward(event);
            }
        };
        let quiet = progress.tapped(&messages);
        let reporting = match make {
            Make::Quietly => &quiet,
            Make::Never | Make::Reporting => progress,
        };
        let tile_size = self.shared.options.tile_size;
        let built = build_mipmaps(path, None, options, tile_size, reporting)
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
        let (memory, opened) = match not_written {
            None => (None, files),
            Some(why) => {
                progress_warn!(
                    reporting,
                    "The mipmap of {} is held in memory, since its files could not be \
                     written: {why}",
                    path.display()
                );
                let memory = files
                    .into_iter()
                    .map(|file| {
                        let level = file.record().level as usize;
                        (level, Arc::new(LevelSource::Mip(file)))
                    })
                    .collect();
                (Some(memory), Vec::new())
            }
        };
        let files = MipmapFiles {
            layout: Arc::new(layout),
            location,
            source,
            region_readable,
            memory,
        };
        Ok((Arc::new(files), opened))
    }
}
