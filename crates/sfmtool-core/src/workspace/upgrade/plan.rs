// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Phases 1 and 2 of the workspace upgrade: find the workspaces and files of
//! a tree, and decide what the run does to each file.
//!
//! Nothing here writes. A dry run is these two phases and the report they
//! give; a run goes on from the [`Plan`] to the rewrite in the parent module.

use std::path::{Component, Path, PathBuf};

use rayon::prelude::*;

use super::{UpgradeFileKind, UpgradeOutcome, UpgradeWorkspaceError, WORKSPACE_FILE_NAME};
use crate::progress::Progress;
use crate::workspace::WORKSPACE_VERSION;

/// What a run does to one `.sfmr`, `.matches`, `.camrig` or `.kdf` file.
#[derive(Debug, Clone)]
pub(super) enum Action {
    /// Read it and write it back as the current version, at its own path.
    Rewrite,
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

/// What phases 1 and 2 found.
#[derive(Debug)]
pub(super) struct Plan {
    /// The workspaces of the tree, the root first, then the nested ones
    /// sorted by path.
    pub(super) workspaces: Vec<PathBuf>,
    /// Every `.sfmr`, then `.matches`, then `.camrig`, then `.kdf` file, each
    /// kind sorted by path.
    pub(super) files: Vec<FilePlan>,
    pub(super) linked_directories: Vec<PathBuf>,
}

/// `path` with its components joined again, so two spellings of one path
/// (`a/b` and `a\b` on Windows, a trailing separator) compare equal.
pub(super) fn normalized(path: &Path) -> PathBuf {
    path.components().collect()
}

/// `path` with each `..` taking off the name before it, without touching
/// the file system.
pub(super) fn lexically_resolved(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Whether an error from reading a file is the file system's (the file could
/// not be opened, read or was locked) rather than the reader refusing what
/// the file holds. An I/O error of kind `InvalidData` or `UnexpectedEof` is
/// the content's: the decompressor and a truncated archive report through
/// `std::io::Error` too.
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

/// The outcome of a file whose read failed with `error`: a file the file
/// system will not give up fails; one whose content the current reader
/// refuses is left as it is.
pub(super) fn read_failure<E: std::error::Error + 'static>(error: E) -> UpgradeOutcome {
    if is_file_system_error(&error) {
        UpgradeOutcome::Failed(format!("it cannot be read: {error}"))
    } else {
        UpgradeOutcome::NotConverted(format!(
            "the current reader refuses it, so it is left as it is: {error}"
        ))
    }
}

/// The outcome of a `.sfmr` file the run's save refuses with `error`.
pub(super) fn write_failure(error: &sfmtool_sfmr_format::SfmrError) -> UpgradeOutcome {
    UpgradeOutcome::Failed(format!("it could not be written: {error}"))
}

/// Refuse a workspace file this build does not upgrade: one that cannot be
/// read or parsed, a version 1 file, which only sfmtool 0.4.1 and earlier
/// upgrade, and any version other than [`WORKSPACE_VERSION`].
pub(super) fn check_workspace_file(path: &Path) -> Result<(), UpgradeWorkspaceError> {
    let invalid = |problem: String| UpgradeWorkspaceError::WorkspaceFile {
        path: path.to_path_buf(),
        problem,
    };
    let text = std::fs::read_to_string(path).map_err(|e| invalid(e.to_string()))?;
    let config: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| invalid(format!("is not valid JSON: {e}")))?;
    match config.get("version").and_then(serde_json::Value::as_u64) {
        Some(1) => Err(UpgradeWorkspaceError::Version1Workspace {
            path: path.to_path_buf(),
        }),
        Some(v) if v == u64::from(WORKSPACE_VERSION) => Ok(()),
        Some(v) => Err(invalid(format!(
            "is version {v}, and this build upgrades the files of a version \
             {WORKSPACE_VERSION} workspace"
        ))),
        None => Err(invalid("has no integer 'version'".into())),
    }
}

/// The files and folders the walk found, before any is read.
#[derive(Default)]
struct Walk {
    sfmr: Vec<PathBuf>,
    matches: Vec<PathBuf>,
    camrig: Vec<PathBuf>,
    kdf: Vec<PathBuf>,
    nested_workspaces: Vec<PathBuf>,
    linked_directories: Vec<PathBuf>,
}

/// Walk the tree at `root` for the files the upgrade reads, nested
/// workspaces included, which it lists. It does not enter a symbolic link or
/// junction to a directory, whose target may be outside the tree, or lead
/// back into it; it lists each.
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
        for (path, file_type) in entries {
            if file_type.is_symlink() {
                if path.is_dir() {
                    found.linked_directories.push(path);
                }
                continue;
            }
            if file_type.is_dir() {
                if path.join(WORKSPACE_FILE_NAME).is_file() {
                    found.nested_workspaces.push(path.clone());
                }
                stack.push(path);
                continue;
            }
            match path.extension().and_then(|e| e.to_str()) {
                Some("sfmr") => found.sfmr.push(path),
                Some("matches") => found.matches.push(path),
                Some("camrig") => found.camrig.push(path),
                Some("kdf") => found.kdf.push(path),
                _ => {}
            }
        }
        listed += 1;
        progress.count(listed, None, "folder");
    }
    for list in [
        &mut found.sfmr,
        &mut found.matches,
        &mut found.camrig,
        &mut found.kdf,
        &mut found.nested_workspaces,
        &mut found.linked_directories,
    ] {
        list.sort();
    }
    Ok(found)
}

/// What the run does to the `.sfmr` file at `path`: rewrite it when it is
/// below the current version and loads as the run will load it.
///
/// A version 16 `sift_files` file without `tracks/keypoints_xy` whose `.sift`
/// files the load could not fill the column from fails here, with the reason
/// the run's save gives, so a dry run reports what a run does.
fn plan_sfmr(path: &Path) -> Action {
    match sfmtool_sfmr_format::read_sfmr_metadata(path) {
        Err(e) => Action::Leave(read_failure(e)),
        Ok(metadata) if metadata.version >= sfmtool_sfmr_format::SFMR_FORMAT_VERSION => {
            Action::Leave(UpgradeOutcome::AlreadyCurrent)
        }
        Ok(_) => match crate::SfmrReconstruction::load(path, &Progress::none()) {
            Ok(recon) if recon.keypoints_xy().is_some() => Action::Rewrite,
            Ok(recon) => match recon.keypoints_for_save(&Progress::none()) {
                Ok(_) => Action::Rewrite,
                Err(e) => Action::Leave(write_failure(&e)),
            },
            Err(e) => Action::Leave(read_failure(e)),
        },
    }
}

/// What the run does to the `.matches` file at `path`: rewrite it when it is
/// below the current version and the reader reads it whole.
fn plan_matches(path: &Path) -> Action {
    match sfmtool_matches_format::read_matches_metadata(path) {
        Err(e) => Action::Leave(read_failure(e)),
        Ok(metadata) if metadata.version >= sfmtool_matches_format::MATCHES_FORMAT_VERSION => {
            Action::Leave(UpgradeOutcome::AlreadyCurrent)
        }
        Ok(_) => match sfmtool_matches_format::read_matches(path) {
            Ok(_) => Action::Rewrite,
            Err(e) => Action::Leave(read_failure(e)),
        },
    }
}

/// What the run does to the `.camrig` file at `path`: rewrite it when it is
/// below the current version and the reader reads it whole.
fn plan_camrig(path: &Path) -> Action {
    match sfmtool_camrig_format::read_camrig_metadata(path) {
        Err(e) => Action::Leave(read_failure(e)),
        Ok((metadata, _)) if metadata.version >= sfmtool_camrig_format::CAMRIG_FORMAT_VERSION => {
            Action::Leave(UpgradeOutcome::AlreadyCurrent)
        }
        Ok(_) => match sfmtool_camrig_format::read_camrig(path) {
            Ok(_) => Action::Rewrite,
            Err(e) => Action::Leave(read_failure(e)),
        },
    }
}

/// What the run does to the `.kdf` index at `path`: rewrite it when it is
/// below the current version. Reading its version checks the file as an open
/// does, which is what the rewrite reads.
fn plan_kdf(path: &Path) -> Action {
    match sfmtool_kdf_format::read_kdf_version(path) {
        Err(e) => Action::Leave(read_failure(e)),
        Ok(version) if version >= sfmtool_kdf_format::KDF_FORMAT_VERSION => {
            Action::Leave(UpgradeOutcome::AlreadyCurrent)
        }
        Ok(_) => Action::Rewrite,
    }
}

/// Phases 1 and 2 for the tree at `root`, whose own workspace file the caller
/// has checked: walk it, refuse a nested workspace file this build does not
/// upgrade, and read each file to decide what the run does to it. A file
/// below the current version is read whole, as the run will read it, so a
/// dry run tells a file the current reader refuses from one it rewrites.
pub(super) fn plan(root: &Path, progress: &Progress<'_>) -> Result<Plan, UpgradeWorkspaceError> {
    let found = {
        let phase = progress.phase("find files");
        walk(root, &phase)?
    };
    for dir in &found.nested_workspaces {
        check_workspace_file(&dir.join(WORKSPACE_FILE_NAME))?;
    }

    let phase = progress.phase("plan");
    let candidates: Vec<(UpgradeFileKind, &PathBuf)> = found
        .sfmr
        .iter()
        .map(|p| (UpgradeFileKind::Sfmr, p))
        .chain(found.matches.iter().map(|p| (UpgradeFileKind::Matches, p)))
        .chain(found.camrig.iter().map(|p| (UpgradeFileKind::Camrig, p)))
        .chain(found.kdf.iter().map(|p| (UpgradeFileKind::Kdf, p)))
        .collect();
    let files: Vec<FilePlan> = candidates
        .par_iter()
        .map(|(kind, path)| {
            let action = if phase.is_cancelled() {
                // Not looked at; the cancellation below stops the run.
                Action::Leave(UpgradeOutcome::AlreadyCurrent)
            } else {
                match kind {
                    UpgradeFileKind::Sfmr => plan_sfmr(path),
                    UpgradeFileKind::Matches => plan_matches(path),
                    UpgradeFileKind::Camrig => plan_camrig(path),
                    UpgradeFileKind::Kdf => plan_kdf(path),
                }
            };
            FilePlan {
                path: (*path).clone(),
                kind: *kind,
                action,
            }
        })
        .collect();
    phase.check_cancel()?;

    let mut workspaces = vec![root.to_path_buf()];
    workspaces.extend(found.nested_workspaces);
    Ok(Plan {
        workspaces,
        files,
        linked_directories: found.linked_directories,
    })
}
