// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Whether the mipmaps of many photographs are current, read from their
//! files without decoding a photograph or writing anything: what
//! `sfm ws mipmaps --check` reports for the photographs under its paths.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use super::build::{check_mipmap, photograph_layout};
use super::record::MipmapSource;
use super::MipmapError;
use crate::progress::{Cancelled, Progress};

/// What [`check_mipmaps`] found for each photograph it was given, each path
/// in one list, as given and in the order given.
#[derive(Debug, Default)]
pub struct MipmapCheckReport {
    /// Out-of-core photographs whose mipmap is current.
    pub current: Vec<PathBuf>,
    /// Out-of-core photographs with no file at their coarsest mip level's
    /// name: no mipmap was made, or it was made at another quality.
    pub missing: Vec<PathBuf>,
    /// Out-of-core photographs whose mipmap has files that are not current,
    /// each with the first file found so and why, naming the file by its name
    /// alone.
    pub stale: Vec<(PathBuf, String)>,
    /// Photographs in core, which have no mipmap: in a workspace without
    /// `mipmap_options`, outside every workspace, or not over the threshold.
    pub in_core: Vec<PathBuf>,
    /// Photographs that could not be read, or whose workspace file is
    /// invalid, with why.
    pub failed: Vec<(PathBuf, MipmapError)>,
}

/// What one photograph's check found.
enum Found {
    Current,
    Missing,
    Stale(String),
    InCore,
    Failed(MipmapError),
}

/// Check the mipmap on disk of each of `photographs`, with photograph tiles
/// of `tile_size` ([`PHOTOGRAPH_TILE_SIZE`](super::PHOTOGRAPH_TILE_SIZE)
/// outside tests), as [`check_mipmap`] checks one, writing nothing.
///
/// Each photograph's header and workspace are read first, so a photograph in
/// core is not read further. An out-of-core photograph is **missing** when
/// there is no file at its coarsest mip level's name, and otherwise
/// **current** or **stale** by [`check_mipmap`], which reads each stored
/// level's file and hashes the original only when its length matches a
/// record and its modification time does not. The photographs are checked
/// on the rayon pool, `progress` counting them as `images` and asked between
/// them whether to stop.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled.
///
/// # Example
///
/// ```
/// use sfmtool_core::camera::mipmap::{check_mipmaps, PHOTOGRAPH_TILE_SIZE};
/// use sfmtool_core::Progress;
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("a.png");
/// image::RgbImage::new(8, 6).save(&path)?;
/// // Outside every workspace, so in core.
/// let report = check_mipmaps(&[path.as_path()], PHOTOGRAPH_TILE_SIZE, &Progress::none())?;
/// assert_eq!(report.in_core, vec![path]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn check_mipmaps(
    photographs: &[&Path],
    tile_size: u32,
    progress: &Progress<'_>,
) -> Result<MipmapCheckReport, Cancelled> {
    let total = photographs.len() as u64;
    let done = AtomicU64::new(0);
    let found: Vec<Option<Found>> = photographs
        .par_iter()
        .map(|path| {
            if progress.is_cancelled() {
                return None;
            }
            let found = check_one(path, tile_size);
            progress.count(
                done.fetch_add(1, Ordering::Relaxed) + 1,
                Some(total),
                "images",
            );
            Some(found)
        })
        .collect();
    progress.check_cancel()?;
    let mut report = MipmapCheckReport::default();
    for (path, found) in photographs.iter().zip(found) {
        let path = path.to_path_buf();
        match found.ok_or(Cancelled)? {
            Found::Current => report.current.push(path),
            Found::Missing => report.missing.push(path),
            Found::Stale(why) => report.stale.push((path, why)),
            Found::InCore => report.in_core.push(path),
            Found::Failed(e) => report.failed.push((path, e)),
        }
    }
    Ok(report)
}

fn check_one(path: &Path, tile_size: u32) -> Found {
    let (layout, location) = match photograph_layout(path, tile_size) {
        Ok(found) => found,
        Err(e) => return Found::Failed(e),
    };
    let (Some(location), Some(options)) = (location, layout.options()) else {
        return Found::InCore;
    };
    let coarsest = location.level_path(layout.stored_levels(), options.jpeg_quality);
    if std::fs::symlink_metadata(&coarsest).is_err() {
        return Found::Missing;
    }
    let source = match MipmapSource::read(path, &location) {
        Ok(source) => source,
        Err(e) => return Found::Failed(e),
    };
    match check_mipmap(&source, &location, &layout) {
        Ok(_) => Found::Current,
        Err(MipmapError::Stale { path: file, why }) => {
            let name = file.file_name().unwrap_or_default().to_string_lossy();
            Found::Stale(format!("{name} is not current: {why}"))
        }
        Err(e) => Found::Failed(e),
    }
}

#[cfg(test)]
mod tests;
