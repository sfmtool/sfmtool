// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Deleting the orphan files of a workspace's mipmap directory: what
//! `sfm ws mipmaps --prune` does, and what `--check` lists without deleting.
//!
//! A file is judged only when it is a regular file inside a mipmap directory,
//! reached from the directory without passing through a link, and its name is
//! a mip level file's or a build's temporary file's. Every other file is left
//! as it is, and no directory is deleted. See the spec
//! `specs/cli/workspace/ws-mipmaps-command.md` § "Orphans".

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::build::photograph_layout;
use super::file::{ExpectedMipLevel, MipLevelFile};
use super::location::MipmapLocation;
use super::record::MipmapSource;
use super::{MipmapError, MipmapOptions, PyramidLayout, MIPMAP_DIR_NAME, PHOTOGRAPH_TILE_SIZE};
use crate::progress::{Cancelled, Progress};
use crate::workspace::{find_workspace_for_path, read_mipmap_options, WORKSPACE_FILE_NAME};

/// How old a build's temporary file must be, by its modification time,
/// before a prune deletes it: a build writes and renames one within seconds,
/// so one this old was left by a build that stopped.
pub const TEMPORARY_FILE_MAX_AGE: Duration = Duration::from_secs(60 * 60);

/// How [`prune_mipmaps`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PruneMipmapsOptions {
    /// Judge every file and delete none; the report then lists the files a
    /// prune would delete.
    pub dry_run: bool,
    /// The photograph tile size the stored levels are counted with:
    /// [`PHOTOGRAPH_TILE_SIZE`] outside tests.
    pub tile_size: u32,
}

impl Default for PruneMipmapsOptions {
    fn default() -> Self {
        Self {
            dry_run: false,
            tile_size: PHOTOGRAPH_TILE_SIZE,
        }
    }
}

/// What [`prune_mipmaps`] found and did, each list sorted by path.
#[derive(Debug, Default)]
pub struct MipmapPruneReport {
    /// The orphan files, deleted (in a dry run, to be deleted), each with
    /// why it is an orphan.
    pub orphans: Vec<(PathBuf, String)>,
    /// Mip level files current for their photographs, and temporary files
    /// too new to delete, left as they are.
    pub kept: Vec<PathBuf>,
    /// Files in the searched part of a mipmap directory whose names are not a
    /// mip level file's or a build's temporary file's, such as
    /// `CACHEDIR.TAG`, left as they are.
    pub other_files: Vec<PathBuf>,
    /// Links (symbolic links, and junctions on Windows) met in a mipmap
    /// directory, or a mipmap directory that is one, neither entered nor
    /// deleted.
    pub links: Vec<PathBuf>,
    /// Paths that could not be searched, files that could not be judged or
    /// deleted, and workspaces whose `mipmap_options` is invalid, with why.
    /// Nothing of them is deleted.
    pub failed: Vec<(PathBuf, String)>,
    /// Paths in no workspace, which have no mipmap directory to prune.
    pub outside_workspaces: Vec<PathBuf>,
}

/// Delete the orphan files of the mipmap directories that hold the mipmaps of
/// the photographs under `paths`: the part of the enclosing workspace's
/// mipmap directory that mirrors each path, and the whole mipmap directory of
/// each workspace nested below a path that is a directory.
///
/// A file there is an **orphan** when its name is a mip level file's and the
/// workspace's file has no `mipmap_options`, its quality is not the one the
/// object gives its level, its photograph no longer exists, is in core or is
/// too large, the photograph's mipmap is kept elsewhere (in a nested
/// workspace), the mipmap stores no such level, or the file is not current
/// for the photograph; or when its name is a build's temporary file's and it
/// is older than [`TEMPORARY_FILE_MAX_AGE`]. Each file is judged on its own,
/// so the current files a stopped build left are kept for the next build. A
/// mip level file of a photograph that is in no reconstruction is judged as
/// any other. A path may name a photograph or a directory that no longer
/// exists, whose mip level files are then all orphans.
///
/// Only regular files are deleted, each checked again just before, to be a
/// regular file whose directory is inside the mipmap directory on disk. A
/// link is never entered nor deleted. A workspace whose `mipmap_options` is
/// invalid is not searched. `progress` counts the files judged as `files`
/// and is asked before each whether to stop.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled; the files deleted before
/// that stay deleted.
///
/// # Example
///
/// ```
/// use sfmtool_core::camera::mipmap::{prune_mipmaps, PruneMipmapsOptions};
/// use sfmtool_core::Progress;
///
/// let dir = tempfile::tempdir()?;
/// std::fs::write(dir.path().join(".sfm-workspace.json"), r#"{"version": 2}"#)?;
/// let stray = dir.path().join(".sfm-mipmaps").join("gone.jpg-mip1-q95.jpg");
/// std::fs::create_dir_all(stray.parent().unwrap())?;
/// std::fs::write(&stray, b"")?;
/// // A workspace without `mipmap_options` keeps no mip level file.
/// let report = prune_mipmaps(&[dir.path()], &PruneMipmapsOptions::default(), &Progress::none())?;
/// assert_eq!(report.orphans.len(), 1);
/// assert!(!stray.exists());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn prune_mipmaps(
    paths: &[&Path],
    options: &PruneMipmapsOptions,
    progress: &Progress<'_>,
) -> Result<MipmapPruneReport, Cancelled> {
    let mut report = MipmapPruneReport::default();
    // Each workspace root, with the parts of its mipmap directory to search,
    // relative to it ("" is the whole directory).
    let mut scopes: BTreeMap<PathBuf, BTreeSet<PathBuf>> = BTreeMap::new();
    for path in paths {
        progress.check_cancel()?;
        find_scopes(path, &mut scopes, &mut report);
    }

    let mut candidates: Vec<(PathBuf, Option<MipmapOptions>, PathBuf)> = Vec::new();
    for (root, parts) in &scopes {
        let options = match read_mipmap_options(root) {
            Ok(options) => options,
            Err(e) => {
                report.failed.push((
                    root.join(WORKSPACE_FILE_NAME),
                    format!("its mipmap directory was not searched: {e}"),
                ));
                continue;
            }
        };
        let mipmap_dir = root.join(MIPMAP_DIR_NAME);
        match real_dir(&mipmap_dir) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(Linked) => {
                report.links.push(mipmap_dir);
                continue;
            }
        }
        let mut files = BTreeSet::new();
        for part in parts {
            progress.check_cancel()?;
            collect(&mipmap_dir, part, &mut files, &mut report);
        }
        for file in files {
            candidates.push((root.clone(), options, file));
        }
    }

    let total = candidates.len() as u64;
    let mut judge = Judge::new(options.tile_size);
    for (done, (root, workspace_options, file)) in candidates.into_iter().enumerate() {
        progress.check_cancel()?;
        let mipmap_dir = root.join(MIPMAP_DIR_NAME);
        match judge.file(&root, &mipmap_dir, workspace_options.as_ref(), &file) {
            Verdict::Orphan(why) => {
                if options.dry_run {
                    report.orphans.push((file, why));
                } else {
                    match delete(&mipmap_dir, &file) {
                        Ok(()) => report.orphans.push((file, why)),
                        Err(e) => report
                            .failed
                            .push((file, format!("it could not be deleted: {e}"))),
                    }
                }
            }
            Verdict::Keep => report.kept.push(file),
            Verdict::NotAMipmap => report.other_files.push(file),
            Verdict::Failed(why) => report.failed.push((file, why)),
        }
        progress.count(done as u64 + 1, Some(total), "files");
    }
    report.orphans.sort();
    report.kept.sort();
    report.other_files.sort();
    report.links.sort();
    report.failed.sort_by(|a, b| a.0.cmp(&b.0));
    report.outside_workspaces.sort();
    Ok(report)
}

/// A path met where only a real directory is entered: a link, or a junction.
struct Linked;

/// Whether `path` is a directory that is not a link: `Ok(false)` when it does
/// not exist or is not a directory, `Err(Linked)` when it is a link.
fn real_dir(path: &Path) -> Result<bool, Linked> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(Linked),
        Ok(meta) => Ok(meta.is_dir()),
        Err(_) => Ok(false),
    }
}

/// The parts of mipmap directories that hold the mipmaps of the photographs
/// under `path`, added to `scopes`.
fn find_scopes(
    path: &Path,
    scopes: &mut BTreeMap<PathBuf, BTreeSet<PathBuf>>,
    report: &mut MipmapPruneReport,
) {
    let absolute: PathBuf = match std::path::absolute(path) {
        Ok(absolute) => absolute.components().collect(),
        Err(e) => {
            report.failed.push((path.to_path_buf(), e.to_string()));
            return;
        }
    };
    let mut found = false;
    if let Some(root) = find_workspace_for_path(&absolute) {
        found = true;
        let relative = absolute.strip_prefix(&root).ok().filter(|relative| {
            relative
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
        });
        if let Some(relative) = relative {
            let relative = relative.to_path_buf();
            scopes.entry(root).or_default().insert(relative);
        } else {
            report.failed.push((
                path.to_path_buf(),
                "it leads out of its workspace through `..`; give it without".to_string(),
            ));
        }
    }
    if matches!(real_dir(&absolute), Ok(true)) {
        let mut nested = Vec::new();
        nested_workspaces(&absolute, &mut nested, report);
        for root in nested {
            found = true;
            scopes.entry(root).or_default().insert(PathBuf::new());
        }
    }
    if !found {
        report.outside_workspaces.push(path.to_path_buf());
    }
}

/// Every workspace root strictly below `dir`, not entering a link or a
/// mipmap directory.
fn nested_workspaces(dir: &Path, out: &mut Vec<PathBuf>, report: &mut MipmapPruneReport) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            report
                .failed
                .push((dir.to_path_buf(), format!("it could not be listed: {e}")));
            return;
        }
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() || file_type.is_symlink() || entry.file_name() == MIPMAP_DIR_NAME {
            continue;
        }
        let path = entry.path();
        if path.join(WORKSPACE_FILE_NAME).is_file() {
            out.push(path.clone());
        }
        nested_workspaces(&path, out, report);
    }
}

/// The candidate files of the part `part` of `mipmap_dir`: every regular file
/// under the directory `part` names, and the files beside it whose names
/// start with its name and `-mip`, as a photograph's mip level files and
/// their temporary files do. Each directory on the way down must be a real
/// one; a link is listed in the report instead.
fn collect(
    mipmap_dir: &Path,
    part: &Path,
    files: &mut BTreeSet<PathBuf>,
    report: &mut MipmapPruneReport,
) {
    let mut dir = mipmap_dir.to_path_buf();
    let components: Vec<&OsStr> = part.iter().collect();
    for (i, name) in components.iter().enumerate() {
        let last = i + 1 == components.len();
        if last {
            // The photographs' files beside the path, and below it when it
            // is a directory.
            if let Some(entries) = list(&dir, report) {
                for (path, kind) in entries {
                    let file_name = path.file_name().unwrap_or_default();
                    if !names_files_of(file_name, name) {
                        continue;
                    }
                    match kind {
                        Kind::File => {
                            files.insert(path);
                        }
                        Kind::Link => report.links.push(path),
                        Kind::Dir | Kind::Other => {}
                    }
                }
            }
        }
        dir.push(name);
        match real_dir(&dir) {
            Ok(true) => {}
            Ok(false) => return,
            Err(Linked) => {
                report.links.push(dir);
                return;
            }
        }
    }
    walk(&dir, files, report);
}

/// Whether `file_name` is that of a mip level file, or a build's temporary
/// file, of the photograph named `photograph`.
fn names_files_of(file_name: &OsStr, photograph: &OsStr) -> bool {
    let (Some(file_name), Some(photograph)) = (file_name.to_str(), photograph.to_str()) else {
        return false;
    };
    let file_name = file_name.strip_prefix('.').unwrap_or(file_name);
    file_name
        .strip_prefix(photograph)
        .is_some_and(|rest| rest.starts_with("-mip"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    File,
    Dir,
    Link,
    Other,
}

/// The entries of `dir`, each classified without following a link.
fn list(dir: &Path, report: &mut MipmapPruneReport) -> Option<Vec<(PathBuf, Kind)>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            report
                .failed
                .push((dir.to_path_buf(), format!("it could not be listed: {e}")));
            return None;
        }
    };
    let mut out = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                report
                    .failed
                    .push((dir.to_path_buf(), format!("it could not be listed: {e}")));
                return None;
            }
        };
        let kind = match entry.file_type() {
            Ok(t) if t.is_symlink() => Kind::Link,
            Ok(t) if t.is_dir() => Kind::Dir,
            Ok(t) if t.is_file() => Kind::File,
            _ => Kind::Other,
        };
        out.push((entry.path(), kind));
    }
    Some(out)
}

/// Every regular file under `dir`, not entering a link.
fn walk(dir: &Path, files: &mut BTreeSet<PathBuf>, report: &mut MipmapPruneReport) {
    let Some(entries) = list(dir, report) else {
        return;
    };
    for (path, kind) in entries {
        match kind {
            Kind::File => {
                files.insert(path);
            }
            Kind::Dir => walk(&path, files, report),
            Kind::Link => report.links.push(path),
            Kind::Other => {}
        }
    }
}

/// What a file's name says it is.
#[derive(Debug, PartialEq, Eq)]
enum Named<'a> {
    /// `<photograph>-mip<level>-q<quality>.jpg`.
    MipLevel {
        photograph: &'a str,
        level: usize,
        quality: u8,
    },
    /// `.<mip level file name>.<process id>-<count>.tmp`.
    Temporary,
    /// Anything else.
    Other,
}

/// A decimal number as a mip level file's name writes one: digits with no
/// leading zero, or `0`.
fn decimal(text: &str) -> Option<u64> {
    let canonical = !text.is_empty()
        && text.bytes().all(|b| b.is_ascii_digit())
        && (text == "0" || !text.starts_with('0'));
    canonical.then(|| text.parse().ok()).flatten()
}

fn parse_name(name: &str) -> Named<'_> {
    if let Some(inner) = name
        .strip_prefix('.')
        .and_then(|rest| rest.strip_suffix(".tmp"))
    {
        let temporary = inner.rsplit_once('.').is_some_and(|(target, stamp)| {
            stamp
                .split_once('-')
                .is_some_and(|(pid, count)| decimal(pid).is_some() && decimal(count).is_some())
                && matches!(parse_name(target), Named::MipLevel { .. })
        });
        return if temporary {
            Named::Temporary
        } else {
            Named::Other
        };
    }
    let Some(rest) = name.strip_suffix(".jpg") else {
        return Named::Other;
    };
    let Some((rest, quality)) = rest.rsplit_once("-q") else {
        return Named::Other;
    };
    let Some((photograph, level)) = rest.rsplit_once("-mip") else {
        return Named::Other;
    };
    match (
        decimal(level).and_then(|l| usize::try_from(l).ok()),
        decimal(quality).and_then(|q| u8::try_from(q).ok()),
    ) {
        (Some(level), Some(quality))
            if !photograph.is_empty() && (1..=100).contains(&quality) && level < 64 =>
        {
            Named::MipLevel {
                photograph,
                level,
                quality,
            }
        }
        _ => Named::Other,
    }
}

/// What to do with one file.
enum Verdict {
    Orphan(String),
    Keep,
    NotAMipmap,
    Failed(String),
}

/// What one photograph's layout says, read once for all its files.
enum Photograph {
    /// No file at the photograph's path.
    Gone,
    /// The photograph has no mipmap, for the reason given.
    NoMipmap(String),
    /// Its header or workspace could not be read.
    Unreadable(String),
    OutOfCore(Box<OutOfCore>),
}

/// An out-of-core photograph's layout, where its mipmap is, and its
/// original's facts, or why they could not be read.
struct OutOfCore {
    layout: PyramidLayout,
    location: MipmapLocation,
    source: Result<MipmapSource, String>,
}

/// The judgement of files, keeping each photograph's layout and original's
/// facts for the rest of its files, so the original is hashed at most once.
struct Judge {
    tile_size: u32,
    photographs: HashMap<PathBuf, Photograph>,
}

impl Judge {
    fn new(tile_size: u32) -> Self {
        Self {
            tile_size,
            photographs: HashMap::new(),
        }
    }

    fn file(
        &mut self,
        root: &Path,
        mipmap_dir: &Path,
        options: Option<&MipmapOptions>,
        file: &Path,
    ) -> Verdict {
        let Some(name) = file.file_name().and_then(OsStr::to_str) else {
            return Verdict::NotAMipmap;
        };
        let (photograph_name, level, quality) = match parse_name(name) {
            Named::Other => return Verdict::NotAMipmap,
            Named::Temporary => return temporary(file),
            Named::MipLevel {
                photograph,
                level,
                quality,
            } => (photograph, level, quality),
        };
        let Some(options) = options else {
            return Verdict::Orphan("the workspace's file has no mipmap_options".to_string());
        };
        let wanted = options.quality_for_level(level);
        if quality != wanted {
            let key = if level == 0 {
                "full_resolution_jpeg_quality"
            } else {
                "jpeg_quality"
            };
            return Verdict::Orphan(format!(
                "its quality {quality} is not the workspace's {key}, {wanted}"
            ));
        }
        // The photograph the name and the directory mirror.
        let Some(parent) = file.parent().and_then(|p| p.strip_prefix(mipmap_dir).ok()) else {
            return Verdict::NotAMipmap;
        };
        let photograph = root.join(parent).join(photograph_name);
        let relative = parent.join(photograph_name);
        let relative = relative.to_string_lossy().replace('\\', "/");
        let tile_size = self.tile_size;
        let state = self
            .photographs
            .entry(photograph.clone())
            .or_insert_with(|| read_photograph(&photograph, tile_size));
        match state {
            Photograph::Gone => {
                Verdict::Orphan(format!("its photograph {relative} no longer exists"))
            }
            Photograph::NoMipmap(why) => {
                Verdict::Orphan(format!("its photograph {relative} {why}"))
            }
            Photograph::Unreadable(why) => Verdict::Failed(format!(
                "its photograph {relative} could not be read, so it is kept: {why}"
            )),
            Photograph::OutOfCore(out_of_core) => {
                let OutOfCore {
                    layout,
                    location,
                    source,
                } = &**out_of_core;
                if location.level_path(level, quality) != file {
                    let elsewhere = location.dir().strip_prefix(root).unwrap_or(location.dir());
                    return Verdict::Orphan(format!(
                        "the mipmap of {relative} is kept in {}",
                        elsewhere.display()
                    ));
                }
                let k = layout.stored_levels();
                if level > k {
                    return Verdict::Orphan(format!(
                        "the mipmap of {relative} stores levels 1 to {k}"
                    ));
                }
                let source = match source {
                    Ok(source) => source,
                    Err(why) => {
                        return Verdict::Failed(format!(
                            "its photograph {relative} could not be read, so it is kept: {why}"
                        ))
                    }
                };
                current(file, layout, location, source, level)
            }
        }
    }
}

/// A build's temporary file is an orphan once it is older than
/// [`TEMPORARY_FILE_MAX_AGE`].
fn temporary(file: &Path) -> Verdict {
    let age = std::fs::symlink_metadata(file)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok());
    match age {
        Some(age) if age > TEMPORARY_FILE_MAX_AGE => Verdict::Orphan(format!(
            "it is a temporary file a build left {} minutes ago",
            age.as_secs() / 60
        )),
        _ => Verdict::Keep,
    }
}

fn read_photograph(photograph: &Path, tile_size: u32) -> Photograph {
    match std::fs::metadata(photograph) {
        Ok(meta) if meta.is_file() => {}
        Ok(_) => return Photograph::NoMipmap("is not a file".to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Photograph::Gone,
        Err(e) => return Photograph::Unreadable(e.to_string()),
    }
    match photograph_layout(photograph, tile_size) {
        Ok((layout, Some(location))) => {
            let source = MipmapSource::read(photograph, &location).map_err(|e| e.to_string());
            Photograph::OutOfCore(Box::new(OutOfCore {
                layout,
                location,
                source,
            }))
        }
        Ok((_, None)) => Photograph::NoMipmap("is in core".to_string()),
        Err(MipmapError::TooLarge { width, height, .. }) => Photograph::NoMipmap(format!(
            "is {width} x {height}, too large to be read, so it has no mipmap"
        )),
        Err(e) => Photograph::Unreadable(e.to_string()),
    }
}

/// Whether the mip level file of `level` is current for its photograph: its
/// own record and frame (tests 1 to 4 of when a mip level file is current),
/// the number of levels the mipmap stores, and, for `-mip0`, a record saying
/// the original is not region-readable.
fn current(
    file: &Path,
    layout: &PyramidLayout,
    location: &MipmapLocation,
    source: &MipmapSource,
    level: usize,
) -> Verdict {
    let Some(expected) = ExpectedMipLevel::new(layout, location.relative(), level) else {
        return Verdict::Orphan("its photograph is in core".to_string());
    };
    let opened = match MipLevelFile::open(file, &expected) {
        Ok(opened) => opened,
        Err(MipmapError::Stale { why, .. }) => {
            return Verdict::Orphan(format!("it is not current: {why}"))
        }
        Err(e) => return Verdict::Failed(format!("it could not be read, so it is kept: {e}")),
    };
    let record = opened.record();
    if let Err(why) = source.check(record) {
        return Verdict::Orphan(format!("it is not current: {why}"));
    }
    let k = layout.stored_levels();
    if record.stored_levels as usize != k {
        return Verdict::Orphan(format!(
            "it is not current: its record says the mipmap stores {} levels, not {k}",
            record.stored_levels
        ));
    }
    if level == 0 && record.region_readable_original {
        return Verdict::Orphan(
            "its original is region-readable, so the mipmap has no -mip0".to_string(),
        );
    }
    Verdict::Keep
}

/// Delete `file`, a candidate found in `mipmap_dir`, after checking again
/// that it is a regular file, not a link, and that its directory on disk is
/// inside the mipmap directory on disk.
fn delete(mipmap_dir: &Path, file: &Path) -> std::io::Result<()> {
    let refuse = |why: &str| Err(std::io::Error::other(why.to_string()));
    let meta = std::fs::symlink_metadata(file)?;
    if !meta.file_type().is_file() || meta.file_type().is_symlink() {
        return refuse("it is no longer a regular file");
    }
    let (Some(parent), Some(name)) = (file.parent(), file.file_name()) else {
        return refuse("it has no directory");
    };
    let real_dir = std::fs::canonicalize(mipmap_dir)?;
    let real_parent = std::fs::canonicalize(parent)?;
    if !real_parent.starts_with(&real_dir) {
        return refuse("its directory is not inside the mipmap directory");
    }
    std::fs::remove_file(real_parent.join(name))
}

#[cfg(test)]
mod tests;
