// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A photograph's layout and pixels as a caller outside the kernels reads
//! them, blocking: [`PhotographCache::read_layout`], and
//! [`PhotographCache::read_level`], [`PhotographCache::read_tile`] and
//! [`PhotographCache::read_region`], which read the pixels every kernel
//! samples for the photograph, making its mipmap first as
//! [`PhotographCache::get`] does. What the Python `PhotographCache`'s
//! `layout`, `read_level`, `read_tile` and `read_region` call. See the spec
//! `specs/core/camera/photograph-cache.md` § "Layouts and pixels for a
//! script".

use std::path::Path;
use std::sync::Arc;

use super::build::Make;
use super::files::Checked;
use super::tiles::check_keys;
use super::{header, Found, GetError, PhotographCache, Share, TileError};
use crate::camera::image::ImageU8;
use crate::camera::mipmap::{region_layout, PyramidLayout};
use crate::camera::sampler::LevelRect;
use crate::camera::tiled_photograph::{PhotographReadError, TileKey};
use crate::progress::Progress;

#[cfg(test)]
mod tests;

/// The name of the [`Progress::value`] the reads report when they have made
/// the photograph's mipmap: the value is 1. `get_many` reports the same name
/// with its running count.
const MIPMAPS_MADE: &str = "mipmaps_made";

/// [`PhotographCache::read_layout`]'s answer: the layout every read of the
/// photograph is served under, and whether its original is region-readable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhotographLayout {
    /// The level sizes, the tile grid, the tier and the stored levels: the
    /// out-of-core layout for an out-of-core photograph, and the in-core one
    /// otherwise.
    pub layout: Arc<PyramidLayout>,
    /// `None` in core; out of core, whether level 0 is read from the original
    /// (`true`) or from its `-mip0` (`false`).
    pub region_readable: Option<bool>,
}

impl PhotographCache {
    /// The layout of the photograph at `path`, the one [`Self::get`] answers
    /// it under, and out of core whether its original is region-readable.
    /// Reads the header, the workspace's `mipmap_options` and, out of core,
    /// the region answer of the coarsest mip level file's record when that
    /// file is current, or otherwise the region test of the original, as a
    /// build decides it. Makes no mipmap and decodes no pixel: the blocking
    /// form of [`Self::layout`] and [`Self::request_layout`], kept by no
    /// slot.
    ///
    /// # Errors
    ///
    /// The [`GetError`] a `get` would answer from the header:
    /// [`GetError::NotFound`], [`GetError::Unreadable`] for a header or an
    /// original that cannot be read, [`GetError::Workspace`] for an invalid
    /// `mipmap_options`, and [`GetError::TooLarge`].
    pub fn read_layout(&self, path: &Path) -> Result<PhotographLayout, GetError> {
        let setting = self.setting_for(path);
        let (width, height) = header(path, &setting)?;
        self.note_large_in_core(path, &setting, width, height);
        let layout = match self.check_files(path, &setting) {
            Checked::InCore => PhotographLayout {
                layout: Arc::new(self.layout_for(width, height)),
                region_readable: None,
            },
            Checked::Current(files, _) => PhotographLayout {
                layout: Arc::clone(&files.layout),
                region_readable: Some(files.region_readable),
            },
            Checked::NotCurrent => {
                let options = setting.options.as_ref().ok().and_then(Option::as_ref);
                let layout =
                    PyramidLayout::for_size(width, height, options, self.shared.options.tile_size)
                        .ok_or_else(|| GetError::TooLarge {
                            path: path.to_path_buf(),
                            width,
                            height,
                        })?;
                let bytes = std::fs::read(path).map_err(|e| match e.kind() {
                    std::io::ErrorKind::NotFound => GetError::NotFound(path.to_path_buf()),
                    _ => GetError::Unreadable(format!("{}: {e}", path.display())),
                })?;
                PhotographLayout {
                    layout: Arc::new(layout),
                    region_readable: Some(region_layout(&bytes).is_ok()),
                }
            }
        };
        Ok(layout)
    }

    /// Level `level` of the photograph at `path` whole, as
    /// [`Self::read_region`] reads it.
    pub fn read_level(
        &self,
        path: &Path,
        level: usize,
        progress: &Progress<'_>,
    ) -> Result<ImageU8, TileError> {
        self.read_pixels(path, progress, |layout| {
            let (x1, y1) = layout.level_size(level);
            Ok((
                level,
                LevelRect {
                    x0: 0,
                    y0: 0,
                    x1,
                    y1,
                },
            ))
        })
    }

    /// The core of the photograph tile `key` of the photograph at `path`,
    /// without its border: the rectangle of its level the tile covers, as
    /// [`Self::read_region`] reads it. A key the photograph does not have is
    /// [`TileError::OutOfRange`].
    pub fn read_tile(
        &self,
        path: &Path,
        key: TileKey,
        progress: &Progress<'_>,
    ) -> Result<ImageU8, TileError> {
        self.read_pixels(path, progress, |layout| {
            check_keys(path, layout, &[key])?;
            let level = usize::from(key.level);
            let ts = layout.tile_size();
            let (w, h) = layout.level_size(level);
            let (x0, y0) = (key.col * ts, key.row * ts);
            Ok((
                level,
                LevelRect {
                    x0,
                    y0,
                    x1: (x0 + ts).min(w),
                    y1: (y0 + ts).min(h),
                },
            ))
        })
    }

    /// The pixels `rect` of level `level` of the photograph at `path`, as
    /// every kernel samples them: in core the exact levels of its decode;
    /// out of core level 0 from the original or its `-mip0`, levels 1 to `K`
    /// from its mip level files and the levels below `K` box-filtered from
    /// level `K`. Reads the photograph as [`Self::get`] answers it, making
    /// its mipmap first when its files are missing or stale and reporting
    /// the value `mipmaps_made` (1) through `progress` when it did, then
    /// copies the rectangle from the tiles it covers, reading only those.
    ///
    /// Polls `progress` for cancellation as `get` does and while it reads
    /// the tiles, and answers [`TileError::Cancelled`] when it is cancelled.
    ///
    /// # Errors
    ///
    /// [`TileError::OutOfRange`] for a level or rectangle the photograph
    /// does not have, or an empty rectangle; [`TileError::Unreadable`] with
    /// the [`GetError`] `get` answered, or that a read of a tile evicted
    /// since met; [`TileError::Changed`] naming the file when a file changed
    /// during the read.
    pub fn read_region(
        &self,
        path: &Path,
        level: usize,
        rect: LevelRect,
        progress: &Progress<'_>,
    ) -> Result<ImageU8, TileError> {
        self.read_pixels(path, progress, |_| Ok((level, rect)))
    }

    /// The photograph at `path` as `get` answers it, and the rectangle of the
    /// level that `which` picks from its layout, copied from its tiles.
    fn read_pixels(
        &self,
        path: &Path,
        progress: &Progress<'_>,
        which: impl FnOnce(&PyramidLayout) -> Result<(usize, LevelRect), TileError>,
    ) -> Result<ImageU8, TileError> {
        let (photograph, found) = self.get_found(path, progress, Share::None, Make::Reporting);
        let photograph = photograph.map_err(|e| unreadable(path, e))?;
        if let Found::Made = found {
            progress.value(MIPMAPS_MADE, 1.0);
        }
        let (level, rect) = which(photograph.layout())?;
        photograph
            .region(level, rect, progress)
            .map_err(|e| read_error(path, e))
    }
}

/// The [`TileError`] for a `get` that failed.
fn unreadable(path: &Path, e: GetError) -> TileError {
    match e {
        GetError::Cancelled => TileError::Cancelled,
        why => TileError::Unreadable {
            path: path.to_path_buf(),
            why,
        },
    }
}

/// The [`TileError`] for a failed read of the photograph `get` answered,
/// keeping what [`GetError`] a failed read of its file again met.
fn read_error(path: &Path, e: PhotographReadError) -> TileError {
    let why = match e {
        PhotographReadError::Cancelled => return TileError::Cancelled,
        PhotographReadError::Changed { path } => return TileError::Changed { path },
        PhotographReadError::OutOfRange { why, .. } => {
            return TileError::OutOfRange {
                path: path.to_path_buf(),
                why,
            }
        }
        PhotographReadError::NotFound { path } => GetError::NotFound(path),
        PhotographReadError::TooLarge {
            path,
            width,
            height,
        } => GetError::TooLarge {
            path,
            width,
            height,
        },
        e @ (PhotographReadError::Unreadable { .. } | PhotographReadError::NeedsTiles { .. }) => {
            GetError::Unreadable(e.to_string())
        }
    };
    TileError::Unreadable {
        path: path.to_path_buf(),
        why,
    }
}
