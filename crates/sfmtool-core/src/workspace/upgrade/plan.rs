// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Phases 1 and 2 of the workspace upgrade: find the files and the version 1
//! features folders, and map each version 1 record to its new folder.
//!
//! Nothing here writes. A dry run is these two phases and the report they
//! give; a run goes on from the [`Plan`] to the phases in the parent module.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use rayon::prelude::*;
use sfmtool_archive_io::{parse_hash_bytes, WorkspaceContents};

use super::{
    SiftConflict, UpgradeFileKind, UpgradeOutcome, UpgradeWorkspaceError, WORKSPACE_FILE_NAME,
};
use crate::features::sift::version_1_sift_files;
use crate::progress::Progress;
use crate::workspace::{
    feature_prefix_dir, find_workspace_for_path, has_image_to_gray, upgrade_feature_options,
    FeatureOptionsError,
};

/// What a run does to one `.sfmr`, `.matches`, `.camrig` or `.kdf` file.
#[derive(Debug, Clone)]
pub(super) enum Action {
    /// Read it and write it back as the current version, at its own path.
    Rewrite {
        /// The converted record its `workspace.contents` becomes, when it
        /// records a version 1 one.
        new_contents: Option<WorkspaceContents>,
        /// Whether its observations name `.sift` files by content hash, whose
        /// new hashes phase 3 gives.
        names_sift_files: bool,
    },
    /// Delete it: a `.kdf` index of an old folder.
    Delete,
    /// Leave it as it is, with this outcome.
    Leave(UpgradeOutcome),
}

/// One file the walk found, and what the run does to it.
#[derive(Debug, Clone)]
pub(super) struct FilePlan {
    pub(super) path: PathBuf,
    pub(super) kind: UpgradeFileKind,
    pub(super) action: Action,
}

/// One version 1 features folder, and what the run does with it.
#[derive(Debug, Clone)]
pub(super) struct OldFolder {
    pub(super) path: PathBuf,
    /// The folder its files are converted into, or `None` when no file of
    /// the workspace that can be converted names it.
    pub(super) new_folder: Option<PathBuf>,
    /// The directory the record's `feature_prefix_dir` is relative to, where
    /// the images are, when a record names the folder.
    pub(super) image_dir: Option<PathBuf>,
    /// Its version 1 `.sift` files, sorted.
    pub(super) sift_files: Vec<PathBuf>,
    /// Why phase 6 keeps it, or `None` when phase 6 deletes it.
    pub(super) kept_because: Option<String>,
}

/// One version 1 `.sift` file to convert.
#[derive(Debug, Clone)]
pub(super) struct SiftJob {
    pub(super) source: PathBuf,
    pub(super) image: PathBuf,
    /// Index into [`Plan::new_folders`].
    pub(super) folder: usize,
}

/// What phases 1 and 2 found.
#[derive(Debug)]
pub(super) struct Plan {
    /// The workspace root, as the upward search found it.
    pub(super) root: PathBuf,
    /// Its real path.
    pub(super) root_canonical: PathBuf,
    /// `.sfm-workspace.json`.
    pub(super) workspace_file: PathBuf,
    /// The version 2 workspace file to write, or `None` when it is version 2
    /// already.
    pub(super) new_workspace_file: Option<serde_json::Value>,
    pub(super) folders: Vec<OldFolder>,
    pub(super) new_folders: Vec<PathBuf>,
    pub(super) jobs: Vec<SiftJob>,
    pub(super) files: Vec<FilePlan>,
    pub(super) nested_workspaces: Vec<PathBuf>,
    pub(super) linked_directories: Vec<PathBuf>,
    /// Why phase 6 deletes nothing at all, when a folder it would delete lies
    /// outside the workspace.
    pub(super) deletion_refused: Option<String>,
}

/// `path` with its components joined again, so two spellings of one path
/// (`a/b` and `a\b` on Windows, a trailing separator) compare and hash equal.
pub(super) fn normalized(path: &Path) -> PathBuf {
    path.components().collect()
}

/// Whether `path` is relative and holds only plain names: no root, drive or
/// `..`, so joining it to a directory stays below that directory.
fn is_plain_relative(path: &Path) -> bool {
    path.components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

/// The `.sift` path image `name` (workspace-relative, `/`-separated) has
/// under `prefix` in the workspace at `root`:
/// `<root>/<name's directory>/<prefix>/<name's file name>.sift`.
pub(super) fn sift_path(root: &Path, name: &str, prefix: &str) -> PathBuf {
    let image = Path::new(name);
    let parent = image.parent().unwrap_or(Path::new(""));
    let file_name = image.file_name().unwrap_or_default().to_string_lossy();
    normalized(
        &root
            .join(parent)
            .join(prefix)
            .join(format!("{file_name}.sift")),
    )
}

/// Whether an error from reading a `.sfmr` or `.matches` file is the file
/// system's (the file could not be opened, read or was locked) rather than
/// the reader refusing what the file holds. An I/O error of kind
/// `InvalidData` or `UnexpectedEof` is the content's: the decompressor and a
/// truncated archive report through `std::io::Error` too.
pub(super) fn is_file_system_error(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(e) = current {
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            return !matches!(
                io.kind(),
                std::io::ErrorKind::InvalidData | std::io::ErrorKind::UnexpectedEof
            );
        }
        current = e.source();
    }
    false
}

/// Why a file the current reader refuses is not converted.
fn refused(error: &str) -> String {
    format!("the current reader refuses it, so it is left as it is: {error}")
}

/// A folder of version 1 `.sift` files, and each file with its content hash.
type HashedFolder = (PathBuf, Vec<(PathBuf, [u8; 16])>);

/// The files and folders the walk found, before any is read.
#[derive(Default)]
struct Walk {
    sfmr: Vec<PathBuf>,
    matches: Vec<PathBuf>,
    camrig: Vec<PathBuf>,
    kdf: Vec<PathBuf>,
    sift_dirs: Vec<PathBuf>,
    nested_workspaces: Vec<PathBuf>,
    linked_directories: Vec<PathBuf>,
}

/// Walk the workspace at `root` for the files the upgrade reads. It does not
/// descend into a nested workspace, a directory below `root` that holds its
/// own `.sfm-workspace.json`, nor into a symbolic link or junction to a
/// directory, whose target may be outside the workspace; it lists both.
fn walk(root: &Path, progress: &Progress<'_>) -> Result<Walk, UpgradeWorkspaceError> {
    let mut found = Walk::default();
    let mut stack = vec![root.to_path_buf()];
    let mut listed = 0u64;
    while let Some(dir) = stack.pop() {
        progress.check_cancel()?;
        let list_error = |e: std::io::Error| UpgradeWorkspaceError::ListFolder {
            folder: dir.clone(),
            problem: e.to_string(),
        };
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&dir).map_err(list_error)? {
            let entry = entry.map_err(list_error)?;
            // The entry's own type, not its target's: a symbolic link, or on
            // Windows a junction, is reported as a link.
            let file_type = entry.file_type().map_err(list_error)?;
            entries.push((entry.path(), file_type));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut holds_sift = false;
        for (path, file_type) in entries {
            if file_type.is_symlink() {
                if path.is_dir() {
                    found.linked_directories.push(path);
                }
                continue;
            }
            if file_type.is_dir() {
                if path.join(WORKSPACE_FILE_NAME).is_file() {
                    found.nested_workspaces.push(path);
                } else {
                    stack.push(path);
                }
                continue;
            }
            match path.extension().and_then(|e| e.to_str()) {
                Some("sfmr") => found.sfmr.push(path),
                Some("matches") => found.matches.push(path),
                Some("camrig") => found.camrig.push(path),
                Some("kdf") => found.kdf.push(path),
                Some("sift") => holds_sift = true,
                _ => {}
            }
        }
        if holds_sift {
            found.sift_dirs.push(normalized(&dir));
        }
        listed += 1;
        progress.count(listed, None, "folder");
    }
    for list in [
        &mut found.sfmr,
        &mut found.matches,
        &mut found.camrig,
        &mut found.kdf,
        &mut found.sift_dirs,
        &mut found.nested_workspaces,
        &mut found.linked_directories,
    ] {
        list.sort();
    }
    Ok(found)
}

/// The record a file or the workspace file names, as the key two equal
/// records share.
fn record_key(contents: &WorkspaceContents) -> String {
    format!(
        "{}\u{0}{}\u{0}{}\u{0}{}",
        contents.feature_tool,
        contents.feature_type,
        contents.feature_prefix_dir,
        contents.feature_options
    )
}

/// A version 1 record, the files that name it, and the record it becomes.
struct Record {
    old: WorkspaceContents,
    new: Option<WorkspaceContents>,
    /// The workspace file and the files of the workspace that can be read
    /// and name it.
    named_by: Vec<PathBuf>,
    /// Of those, how many will be rewritten, the workspace file among them
    /// when it is version 1.
    rewritten_by: usize,
    /// The files that name it and that the current reader refuses.
    refused_by: usize,
}

/// What phase 1 read of a `.sfmr` or `.matches` file: whether it is in this
/// workspace, and if so the record it holds.
enum Recorded {
    /// It records this workspace: its version, its record, whether its
    /// observations name `.sift` files, and for a cluster `.matches` file
    /// below version 6, which the reader refuses, `true`.
    Here {
        version: u32,
        contents: WorkspaceContents,
        names_sift_files: bool,
        unreadable_clusters: bool,
    },
    Leave(UpgradeOutcome),
}

/// Whether `dir`, a workspace directory a file records, is the one at `root`.
fn same_dir(dir: &Path, root_canonical: &Path) -> bool {
    std::fs::canonicalize(dir).is_ok_and(|d| d == root_canonical)
}

fn elsewhere(dir: Option<&Path>) -> UpgradeOutcome {
    UpgradeOutcome::NotConverted(match dir {
        Some(dir) => format!(
            "it records the workspace {}, not this one, so that workspace's upgrade converts it",
            dir.display()
        ),
        None => "it records no workspace that can be found".to_string(),
    })
}

fn read_sfmr_record(path: &Path, root_canonical: &Path) -> Recorded {
    let metadata = match sfmtool_sfmr_format::read_sfmr_metadata(path) {
        Ok(m) => m,
        Err(e) => {
            return Recorded::Leave(UpgradeOutcome::Failed(format!("it cannot be read: {e}")))
        }
    };
    match sfmtool_sfmr_format::resolve_workspace_dir(path, &metadata) {
        Ok(dir) if same_dir(&dir, root_canonical) => Recorded::Here {
            version: metadata.version,
            names_sift_files: metadata.feature_source
                == sfmtool_sfmr_format::FEATURE_SOURCE_SIFT_FILES,
            contents: metadata.workspace.contents,
            unreadable_clusters: false,
        },
        Ok(dir) => Recorded::Leave(elsewhere(Some(&dir))),
        Err(_) => Recorded::Leave(elsewhere(None)),
    }
}

fn read_matches_record(path: &Path, root_canonical: &Path) -> Recorded {
    let metadata = match sfmtool_matches_format::read_matches_metadata(path) {
        Ok(m) => m,
        Err(e) => {
            return Recorded::Leave(UpgradeOutcome::Failed(format!("it cannot be read: {e}")))
        }
    };
    match sfmtool_matches_format::resolve_workspace_dir(path, &metadata) {
        Some(dir) if same_dir(&dir, root_canonical) => Recorded::Here {
            version: metadata.version,
            unreadable_clusters: metadata.has_clusters && metadata.version < 6,
            contents: metadata.workspace.contents,
            names_sift_files: true,
        },
        Some(dir) => Recorded::Leave(elsewhere(Some(&dir))),
        None => Recorded::Leave(elsewhere(None)),
    }
}

/// What reading a whole `.sfmr` or `.matches` file of this workspace gave.
enum Contents {
    /// The image names it records and, when it names `.sift` files, their
    /// content hashes.
    Read {
        names: Vec<String>,
        hashes: Option<Vec<[u8; 16]>>,
    },
    /// The current reader refuses it.
    Refused(String),
    /// The file system would not give it up.
    Unreadable(String),
}

/// Read the file at `path` as the run will: a `.sfmr` into a reconstruction,
/// which checks what the format reader leaves to it, such as each camera's
/// parameters.
fn read_contents(path: &Path, kind: UpgradeFileKind) -> Contents {
    fn classify<E: std::error::Error + 'static>(e: E) -> Contents {
        if is_file_system_error(&e) {
            Contents::Unreadable(e.to_string())
        } else {
            Contents::Refused(e.to_string())
        }
    }
    match kind {
        UpgradeFileKind::Sfmr => {
            let data = match sfmtool_sfmr_format::read_sfmr(path) {
                Ok(d) => d,
                Err(e) => return classify(e),
            };
            let names = data.image_names.clone();
            let hashes = data.sift_content_hashes.clone();
            match crate::SfmrReconstruction::from_sfmr_data(data) {
                Ok(_) => Contents::Read { names, hashes },
                Err(e) => classify(e),
            }
        }
        _ => match sfmtool_matches_format::read_matches(path) {
            Ok(data) => Contents::Read {
                names: data.image_names,
                hashes: Some(data.sift_content_hashes),
            },
            Err(e) => classify(e),
        },
    }
}

/// Read the workspace file: the version 2 file to write in its place, or
/// `None` when it is version 2 already, and the version 1 record it holds.
fn read_workspace_file(
    path: &Path,
) -> Result<(Option<serde_json::Value>, Option<WorkspaceContents>), UpgradeWorkspaceError> {
    let invalid = |problem: String| UpgradeWorkspaceError::WorkspaceFile {
        path: path.to_path_buf(),
        problem,
    };
    let text = std::fs::read_to_string(path).map_err(|e| invalid(e.to_string()))?;
    let config: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| invalid(format!("it is not valid JSON: {e}")))?;
    let Some(map) = config.as_object() else {
        return Err(invalid("it is not a JSON object".into()));
    };
    match map.get("version").and_then(serde_json::Value::as_u64) {
        Some(2) => return Ok((None, None)),
        Some(1) => {}
        Some(v) => {
            return Err(invalid(format!(
                "it is version {v}, and this build upgrades version 1 files to version 2"
            )))
        }
        None => return Err(invalid("it has no integer 'version'".into())),
    }
    let text_of = |key: &str| -> Result<String, UpgradeWorkspaceError> {
        map.get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| invalid(format!("it has no string '{key}'")))
    };
    let contents = WorkspaceContents {
        feature_tool: text_of("feature_tool")?,
        feature_type: text_of("feature_type")?,
        feature_options: map
            .get("feature_options")
            .cloned()
            .ok_or_else(|| invalid("it has no 'feature_options'".into()))?,
        feature_prefix_dir: text_of("feature_prefix_dir")?,
    };
    // A version 1 file whose options already carry `image_to_gray` mixes the
    // versions; `upgrade_feature_options` refuses such options, and so does
    // the upgrade.
    if has_image_to_gray(&contents.feature_options) {
        return Err(UpgradeWorkspaceError::FeatureOptions {
            files: vec![path.to_path_buf()],
            error: FeatureOptionsError::AlreadyVersion2,
        });
    }
    Ok((Some(config), Some(contents)))
}

/// The version 1 workspace file of a workspace that encloses the one at
/// `root`, if any. Its upgrade converts the folders its files name inside
/// this workspace, so it goes first.
pub(super) fn enclosing_version_1_workspace(root: &Path) -> Option<PathBuf> {
    let mut start = root.parent().map(Path::to_path_buf);
    while let Some(dir) = start {
        let enclosing = find_workspace_for_path(&dir)?;
        let file = enclosing.join(WORKSPACE_FILE_NAME);
        let version = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|config| config.get("version").and_then(serde_json::Value::as_u64));
        if version == Some(1) {
            return Some(file);
        }
        start = enclosing.parent().map(Path::to_path_buf);
    }
    None
}

/// The directory `folder` is in when it is `<image_dir>/<prefix>`, or `None`
/// when it does not end with `prefix`'s components.
fn image_dir_of(folder: &Path, root: &Path, prefix: &str) -> Option<PathBuf> {
    let relative = folder.strip_prefix(root).ok()?;
    let folder_parts: Vec<OsString> = relative
        .components()
        .map(|c| c.as_os_str().to_owned())
        .collect();
    let prefix_parts: Vec<OsString> = Path::new(prefix)
        .components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .map(|c| c.as_os_str().to_owned())
        .collect();
    if prefix_parts.is_empty() || folder_parts.len() < prefix_parts.len() {
        return None;
    }
    let split = folder_parts.len() - prefix_parts.len();
    if folder_parts[split..] != prefix_parts[..] {
        return None;
    }
    Some(normalized(
        &folder_parts[..split]
            .iter()
            .fold(root.to_path_buf(), |dir, part| dir.join(part)),
    ))
}

/// Where a folder lies relative to the workspace.
enum Location {
    /// In this workspace.
    Here,
    /// In the nested workspace at this directory.
    Nested(PathBuf),
    /// Outside the workspace, through a link.
    Outside(PathBuf),
}

fn location(folder: &Path, root_canonical: &Path) -> Location {
    let real = std::fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf());
    if !real.starts_with(root_canonical) {
        return Location::Outside(real);
    }
    match find_workspace_for_path(folder) {
        Some(ws) if !same_dir(&ws, root_canonical) => Location::Nested(ws),
        _ => Location::Here,
    }
}

/// Run phases 1 and 2 on the workspace at `root`, whose canonical path is
/// `root_canonical`. Every `.sfmr` and `.matches` file that the run would
/// rewrite is read here, so that a file the reader refuses, a file that
/// cannot be read and a file whose `.sift` files are not on disk are known
/// before anything is written.
pub(super) fn plan(
    root: &Path,
    root_canonical: &Path,
    progress: &Progress<'_>,
) -> Result<Plan, UpgradeWorkspaceError> {
    let workspace_file = root.join(WORKSPACE_FILE_NAME);
    let (new_workspace_file, workspace_record) = read_workspace_file(&workspace_file)?;

    let found = {
        let phase = progress.phase("find files");
        walk(root, &phase)?
    };

    let phase = progress.phase("plan");

    // Each `.sfmr` and `.matches` file: whether it is in this workspace, and
    // for those the run would rewrite, what reading it whole gives.
    let current_version = |kind| match kind {
        UpgradeFileKind::Sfmr => sfmtool_sfmr_format::SFMR_FORMAT_VERSION,
        _ => sfmtool_matches_format::MATCHES_FORMAT_VERSION,
    };
    let candidates: Vec<(UpgradeFileKind, &PathBuf)> = found
        .sfmr
        .iter()
        .map(|p| (UpgradeFileKind::Sfmr, p))
        .chain(found.matches.iter().map(|p| (UpgradeFileKind::Matches, p)))
        .collect();
    let read: Vec<(Recorded, Option<Contents>)> = candidates
        .par_iter()
        .map(|(kind, path)| {
            let recorded = match kind {
                UpgradeFileKind::Sfmr => read_sfmr_record(path, root_canonical),
                _ => read_matches_record(path, root_canonical),
            };
            let contents = match &recorded {
                Recorded::Here {
                    version,
                    contents,
                    unreadable_clusters: false,
                    ..
                } if *version < current_version(*kind)
                    || !has_image_to_gray(&contents.feature_options) =>
                {
                    Some(read_contents(path, *kind))
                }
                _ => None,
            };
            (recorded, contents)
        })
        .collect();
    phase.check_cancel()?;

    // The folders to check for version 1 files: each the walk found, and each
    // a readable file's record names for its images, found by its path on
    // disk even inside a nested workspace.
    let mut sift_dirs: BTreeSet<PathBuf> = found.sift_dirs.iter().cloned().collect();
    for ((_, _), (recorded, contents)) in candidates.iter().zip(&read) {
        if let (
            Recorded::Here {
                contents: record, ..
            },
            Some(Contents::Read { names, .. }),
        ) = (recorded, contents)
        {
            if has_image_to_gray(&record.feature_options)
                || !is_plain_relative(Path::new(&record.feature_prefix_dir))
            {
                continue;
            }
            for name in names {
                let parent = Path::new(name).parent().unwrap_or(Path::new(""));
                if is_plain_relative(parent) {
                    let dir = normalized(&root.join(parent).join(&record.feature_prefix_dir));
                    if dir.is_dir() {
                        sift_dirs.insert(dir);
                    }
                }
            }
        }
    }

    // The version 1 `.sift` files of each of those folders, by the check
    // every writer of version 2 files makes.
    let listed: Vec<Result<Option<HashedFolder>, UpgradeWorkspaceError>> = sift_dirs
        .par_iter()
        .map(|dir| {
            let files = version_1_sift_files(dir)
                .map_err(|e| UpgradeWorkspaceError::ListFolder {
                    folder: dir.clone(),
                    problem: e.to_string(),
                })?
                .map(|found| found.files)
                .unwrap_or_default();
            if files.is_empty() {
                return Ok(None);
            }
            // A file `version_1_sift_files` counted read as version 1, so its
            // hashes read too unless it changed since; one that no longer
            // reads keeps no hash and so matches no record.
            let hashed = files
                .into_iter()
                .map(|path| {
                    let hash = sfmtool_sift_format::read_sift_metadata(&path)
                        .ok()
                        .and_then(|(_, _, h)| parse_hash_bytes(&h.content_xxh128).ok())
                        .unwrap_or([0; 16]);
                    (path, hash)
                })
                .collect();
            Ok(Some((dir.clone(), hashed)))
        })
        .collect();
    let mut version_1_folders = Vec::new();
    let mut version_1_hashes = HashMap::new();
    for item in listed {
        if let Some((dir, files)) = item? {
            let mut paths = Vec::with_capacity(files.len());
            for (path, hash) in files {
                version_1_hashes.insert(normalized(&path), hash);
                paths.push(path);
            }
            version_1_folders.push((dir, paths));
        }
    }
    phase.check_cancel()?;

    // Why a readable file that names a version 1 folder cannot be
    // converted: a `.sift` file it records is not on disk, or has changed.
    let missing = |names: &[String], hashes: &[[u8; 16]], prefix: &str| -> Option<String> {
        for (name, hash) in names.iter().zip(hashes) {
            let path = sift_path(root, name, prefix);
            if version_1_hashes.get(&path) == Some(hash) {
                continue;
            }
            return Some(if path.is_file() {
                format!(
                    "the .sift file {} it records for {name} is not the version 1 file it \
                     records: it has changed",
                    path.display()
                )
            } else {
                format!(
                    "the .sift file {} it records for {name} is not on disk",
                    path.display()
                )
            });
        }
        None
    };
    let reasons: Vec<Option<String>> = read
        .iter()
        .map(|(recorded, contents)| match (recorded, contents) {
            (
                Recorded::Here {
                    contents: record,
                    names_sift_files: true,
                    ..
                },
                Some(Contents::Read { names, hashes }),
            ) if !has_image_to_gray(&record.feature_options) => missing(
                names,
                hashes.as_deref().unwrap_or_default(),
                &record.feature_prefix_dir,
            ),
            _ => None,
        })
        .collect();

    // The records: the workspace file's when it is version 1, and each one a
    // file of this workspace holds.
    let mut records: BTreeMap<String, Record> = BTreeMap::new();
    let mut note_record =
        |contents: &WorkspaceContents, path: &Path, rewritten: bool, is_refused: bool| {
            if has_image_to_gray(&contents.feature_options) {
                return;
            }
            let record = records
                .entry(record_key(contents))
                .or_insert_with(|| Record {
                    old: contents.clone(),
                    new: None,
                    named_by: Vec::new(),
                    rewritten_by: 0,
                    refused_by: 0,
                });
            if is_refused {
                record.refused_by += 1;
            } else {
                record.named_by.push(path.to_path_buf());
                record.rewritten_by += usize::from(rewritten);
            }
        };
    if let Some(contents) = &workspace_record {
        note_record(contents, &workspace_file, true, false);
    }
    for (((_, path), (recorded, contents)), reason) in candidates.iter().zip(&read).zip(&reasons) {
        if let Recorded::Here {
            contents: record, ..
        } = recorded
        {
            match contents {
                Some(Contents::Refused(_)) => note_record(record, path, false, true),
                Some(Contents::Read { .. }) => note_record(record, path, reason.is_none(), false),
                // A cluster file the reader refuses names the folder like any
                // file the run does not rewrite, and so does one that cannot
                // be read now: neither may lose its folder.
                Some(Contents::Unreadable(_)) | None => note_record(record, path, false, false),
            }
        }
    }

    // Phase 2: each record's new folder, by the one folder-hash
    // implementation. A record that cannot be converted, or whose folder
    // would not stay below each image's directory, stops the run here,
    // before anything is written.
    for record in records.values_mut() {
        let old = &record.old;
        let mut files = record.named_by.clone();
        if files.is_empty() {
            // Only files the reader refuses name it; it is not converted.
            continue;
        }
        files.sort();
        if !is_plain_relative(Path::new(&old.feature_prefix_dir))
            || old.feature_prefix_dir.is_empty()
        {
            return Err(UpgradeWorkspaceError::FeaturePrefixDir {
                files,
                feature_prefix_dir: old.feature_prefix_dir.clone(),
            });
        }
        let options = upgrade_feature_options(&old.feature_tool, &old.feature_options)
            .map_err(|error| UpgradeWorkspaceError::FeatureOptions { files, error })?;
        record.new = Some(WorkspaceContents {
            feature_prefix_dir: feature_prefix_dir(&old.feature_tool, &old.feature_type, &options),
            feature_tool: old.feature_tool.clone(),
            feature_type: old.feature_type.clone(),
            feature_options: options,
        });
    }
    let new_record = |contents: &WorkspaceContents| {
        records
            .get(&record_key(contents))
            .and_then(|r| r.new.clone())
    };
    // What each folder becomes: converted when the workspace file or a file
    // the run rewrites names it; kept when
    // a file the reader refuses names it or it lies in a nested workspace;
    // and deleted otherwise, an orphan included.
    let mut folders = Vec::new();
    let mut new_folders: Vec<PathBuf> = Vec::new();
    let mut new_folder_index: HashMap<PathBuf, usize> = HashMap::new();
    let mut destinations: BTreeMap<PathBuf, (PathBuf, [u8; 16])> = BTreeMap::new();
    let mut conflicts = Vec::new();
    let mut jobs = Vec::new();
    let mut deletion_refused = None;
    for (dir, sift_files) in version_1_folders {
        let place = location(&dir, root_canonical);
        let naming: Vec<(&Record, PathBuf)> = records
            .values()
            .filter_map(|record| {
                image_dir_of(&dir, root, &record.old.feature_prefix_dir).map(|d| (record, d))
            })
            .collect();
        let refused_by: usize = naming.iter().map(|(r, _)| r.refused_by).sum();
        // Converted only for a file this run rewrites: a file it leaves
        // cannot read the new folder, and a nested workspace's own run sees
        // to its folders for its own files.
        let converter = naming
            .iter()
            .find(|(record, _)| record.new.is_some() && record.rewritten_by > 0);
        let image_dir = naming.first().map(|(_, d)| d.clone());
        let kept_because = match &place {
            Location::Nested(ws) => Some(format!(
                "it is inside the nested workspace {}, whose own upgrade deletes it",
                ws.display()
            )),
            _ if refused_by > 0 => Some(format!(
                "kept: named by {refused_by} file(s) the current reader refuses"
            )),
            Location::Outside(real) => {
                deletion_refused.get_or_insert_with(|| {
                    format!(
                        "the old folder {} lies outside the workspace, at {}, so nothing was \
                         deleted; delete the old folders by hand once you have checked them",
                        dir.display(),
                        real.display()
                    )
                });
                Some(format!(
                    "it lies outside the workspace, at {}",
                    real.display()
                ))
            }
            Location::Here => None,
        };
        let Some((record, image_dir)) = converter else {
            folders.push(OldFolder {
                path: dir,
                new_folder: None,
                image_dir,
                sift_files,
                kept_because,
            });
            continue;
        };
        let new = record
            .new
            .as_ref()
            .ok_or_else(|| UpgradeWorkspaceError::WorkspaceFile {
                path: workspace_file.clone(),
                problem: "a record was not mapped to a new folder".into(),
            })?;
        let new_folder = normalized(&image_dir.join(&new.feature_prefix_dir));
        let index = *new_folder_index
            .entry(new_folder.clone())
            .or_insert_with(|| {
                new_folders.push(new_folder.clone());
                new_folders.len() - 1
            });
        for source in &sift_files {
            let file_name = source.file_name().unwrap_or_default();
            let destination = new_folder.join(file_name);
            let hash = version_1_hashes
                .get(&normalized(source))
                .copied()
                .unwrap_or_default();
            match destinations.get(&destination) {
                Some((first, first_hash)) if *first_hash != hash => {
                    conflicts.push(SiftConflict {
                        new_file: destination.clone(),
                        first: first.clone(),
                        second: source.clone(),
                    });
                    continue;
                }
                // The same file in two old folders is converted once.
                Some(_) => continue,
                None => {}
            }
            destinations.insert(destination, (source.clone(), hash));
            let name = file_name.to_string_lossy();
            let image_name = name.strip_suffix(".sift").unwrap_or(&name);
            jobs.push(SiftJob {
                source: source.clone(),
                image: image_dir.join(image_name),
                folder: index,
            });
        }
        folders.push(OldFolder {
            path: dir,
            new_folder: Some(new_folder),
            image_dir: Some(image_dir.clone()),
            sift_files,
            kept_because,
        });
    }
    if !conflicts.is_empty() {
        return Err(UpgradeWorkspaceError::Conflicts(conflicts));
    }

    let new_workspace_file = match (new_workspace_file, &workspace_record) {
        (Some(mut config), Some(contents)) => {
            let new = new_record(contents).ok_or_else(|| UpgradeWorkspaceError::WorkspaceFile {
                path: workspace_file.clone(),
                problem: "its record was not mapped to a new folder".into(),
            })?;
            if let Some(map) = config.as_object_mut() {
                map.insert("version".into(), crate::workspace::WORKSPACE_VERSION.into());
                map.insert("feature_options".into(), new.feature_options);
                map.insert(
                    "feature_prefix_dir".into(),
                    new.feature_prefix_dir.clone().into(),
                );
            }
            Some(config)
        }
        _ => None,
    };

    // What the run does to each file.
    let mut files = Vec::new();
    for (((kind, path), (recorded, contents)), reason) in candidates.iter().zip(read).zip(reasons) {
        let action = match (recorded, contents) {
            (Recorded::Leave(outcome), _) => Action::Leave(outcome),
            (
                Recorded::Here {
                    unreadable_clusters: true,
                    ..
                },
                _,
            ) => Action::Leave(UpgradeOutcome::NotConverted(
                "it is a cluster .matches file below version 6, which the reader refuses; \
                 regenerate it with 'sfm match --cluster'"
                    .into(),
            )),
            (Recorded::Here { .. }, None) => Action::Leave(UpgradeOutcome::AlreadyCurrent),
            (Recorded::Here { .. }, Some(Contents::Refused(e))) => {
                Action::Leave(UpgradeOutcome::NotConverted(refused(&e)))
            }
            (Recorded::Here { .. }, Some(Contents::Unreadable(e))) => {
                Action::Leave(UpgradeOutcome::Failed(format!("it cannot be read: {e}")))
            }
            (
                Recorded::Here {
                    contents: record,
                    names_sift_files,
                    ..
                },
                Some(Contents::Read { .. }),
            ) => {
                let new_contents = new_record(&record);
                let names_sift_files = names_sift_files && new_contents.is_some();
                match reason {
                    Some(reason) => Action::Leave(UpgradeOutcome::NotConverted(reason)),
                    None => Action::Rewrite {
                        new_contents,
                        names_sift_files,
                    },
                }
            }
        };
        files.push(FilePlan {
            path: (*path).clone(),
            kind: *kind,
            action,
        });
    }
    for path in &found.camrig {
        let action = match sfmtool_camrig_format::read_camrig_metadata(path) {
            Err(e) => Action::Leave(UpgradeOutcome::Failed(format!("it cannot be read: {e}"))),
            Ok((metadata, _))
                if metadata.version >= sfmtool_camrig_format::CAMRIG_FORMAT_VERSION =>
            {
                Action::Leave(UpgradeOutcome::AlreadyCurrent)
            }
            Ok(_) => Action::Rewrite {
                new_contents: None,
                names_sift_files: false,
            },
        };
        files.push(FilePlan {
            path: path.clone(),
            kind: UpgradeFileKind::Camrig,
            action,
        });
    }
    for path in &found.kdf {
        let action = match sfmtool_kdf_format::read_kdf_workspace(path) {
            Err(e) => Action::Leave(UpgradeOutcome::NotConverted(format!(
                "it cannot be read: {e}"
            ))),
            Ok(Some(workspace))
                if !has_image_to_gray(&workspace.contents.feature_options)
                    && sfmtool_archive_io::resolve_workspace_dir(
                        path,
                        &workspace.relative_path,
                        &workspace.absolute_path,
                    )
                    .is_some_and(|dir| same_dir(&dir, root_canonical)) =>
            {
                Action::Delete
            }
            Ok(_) => Action::Leave(UpgradeOutcome::AlreadyCurrent),
        };
        files.push(FilePlan {
            path: path.clone(),
            kind: UpgradeFileKind::Kdf,
            action,
        });
    }
    drop(phase);
    Ok(Plan {
        root: root.to_path_buf(),
        root_canonical: root_canonical.to_path_buf(),
        workspace_file,
        new_workspace_file,
        folders,
        new_folders,
        jobs,
        files,
        nested_workspaces: found.nested_workspaces,
        linked_directories: found.linked_directories,
        deletion_refused,
    })
}
