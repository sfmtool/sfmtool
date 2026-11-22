// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bring a workspace's files to the current versions: [`upgrade_workspace`].
//!
//! The upgrade rewrites each `.sfmr`, `.matches` and `.camrig` file of a
//! version 2 workspace that is below the current version and that the reader
//! reads, at its own path as the current version, by reading it and writing it
//! back. The readers of all three read only the current version, so a run
//! rewrites no file until one of the formats next changes version; a file
//! below the current version is one the reader refuses, which the run leaves
//! as it is. A version 1 `.sfm-workspace.json` is refused:
//! sfmtool 0.4.1 is the last release that upgrades one, since it is the last
//! that reads the older `.sift`, `.sfmr`, `.matches` and `.camrig` versions
//! such a workspace holds. See `specs/cli/workspace/ws-upgrade-command.md`,
//! whose phases 1 to 3 the function runs in order.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::{
    find_workspace_for_path, WORKSPACE_FILE_NAME, WORKSPACE_LAST_RELEASE_READING_VERSION_1,
};
use crate::progress::{Cancelled, Progress};
use crate::SfmrReconstruction;

mod plan;

use plan::{lexically_resolved, normalized, read_failure, Action};

/// The zstd level the upgrade writes `.matches` and `.camrig` files at, the
/// level their bindings write by default.
const ZSTD_LEVEL: i32 = 3;

/// How [`upgrade_workspace`] runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpgradeWorkspaceOptions {
    /// Run phases 1 and 2 only, find the files and decide what to do with
    /// each, and write nothing. The report then gives the outcome each file
    /// would have.
    pub dry_run: bool,
}

/// The kinds of file the upgrade reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpgradeFileKind {
    /// A `.sfmr` reconstruction.
    Sfmr,
    /// A `.matches` file.
    Matches,
    /// A `.camrig` file.
    Camrig,
}

/// What the upgrade did to a file, or in a dry run what it would do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", content = "reason", rename_all = "snake_case")]
pub enum UpgradeOutcome {
    /// Rewritten at its own path as the current version.
    Converted,
    /// Already the current version, so not written.
    AlreadyCurrent,
    /// Left as it is for the reason given: the current reader refuses it.
    NotConverted(String),
    /// Could not be read or rewritten, for the reason given.
    Failed(String),
}

/// One file and its outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpgradedFile {
    /// The file.
    pub path: PathBuf,
    /// Its kind.
    pub kind: UpgradeFileKind,
    /// What was done to it.
    #[serde(flatten)]
    pub outcome: UpgradeOutcome,
}

/// What [`upgrade_workspace`] did, or in a dry run what it would do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpgradeWorkspaceReport {
    /// The root of the tree: the workspace found from the path the run was
    /// given.
    pub workspace_dir: PathBuf,
    /// The workspaces of the tree, the root first, then each nested one
    /// sorted by path.
    pub workspaces: Vec<PathBuf>,
    /// Whether this was a dry run, which wrote nothing.
    pub dry_run: bool,
    /// Every `.sfmr`, then `.matches`, then `.camrig` file the walk found,
    /// each kind sorted by path.
    pub files: Vec<UpgradedFile>,
    /// The symbolic links and junctions to directories the walk found. It
    /// does not enter them, since their targets may be outside the tree.
    pub linked_directories: Vec<PathBuf>,
    /// Notes for the user, such as that point IDs taken from a rewritten
    /// `.sfmr` before the upgrade no longer resolve.
    pub notes: Vec<String>,
}

impl UpgradeWorkspaceReport {
    /// Whether there was nothing to do: no file was converted or failed.
    pub fn nothing_to_do(&self) -> bool {
        self.files.iter().all(|f| {
            matches!(
                f.outcome,
                UpgradeOutcome::AlreadyCurrent | UpgradeOutcome::NotConverted(_)
            )
        })
    }
}

/// Why [`upgrade_workspace`] stopped. Each is raised before any file is
/// written, except [`Cancelled`](Self::Cancelled), which can leave some files
/// rewritten and the rest as they were, a state a rerun finishes.
#[derive(Debug)]
pub enum UpgradeWorkspaceError {
    /// No `.sfm-workspace.json` at or above the path.
    NoWorkspace {
        /// The path searched from.
        path: PathBuf,
    },
    /// A workspace file of the tree is version 1, which only sfmtool
    /// [`WORKSPACE_LAST_RELEASE_READING_VERSION_1`] and earlier upgrade.
    Version1Workspace {
        /// The workspace file.
        path: PathBuf,
    },
    /// A workspace file of the tree cannot be read or is of a version this
    /// build does not know.
    WorkspaceFile {
        /// The workspace file.
        path: PathBuf,
        /// What is wrong.
        problem: String,
    },
    /// A directory of the tree could not be listed, so a file in it could be
    /// missed.
    ListFolder {
        /// The directory.
        folder: PathBuf,
        /// What went wrong.
        problem: String,
    },
    /// The caller asked the upgrade to stop.
    Cancelled,
}

impl fmt::Display for UpgradeWorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoWorkspace { path } => write!(
                f,
                "no workspace file {WORKSPACE_FILE_NAME} was found in {} or any directory \
                 above it",
                path.display()
            ),
            Self::Version1Workspace { path } => write!(
                f,
                "the workspace file {} is version 1, and this sfmtool upgrades a version 2 \
                 workspace. sfmtool {WORKSPACE_LAST_RELEASE_READING_VERSION_1} is the last \
                 release that reads the older .sift, .sfmr, .matches and .camrig versions a \
                 version 1 workspace holds: install it, run `sfm ws upgrade` on the workspace, \
                 then use the workspace with this release.",
                path.display()
            ),
            Self::WorkspaceFile { path, problem } => {
                write!(f, "the workspace file {} {problem}", path.display())
            }
            Self::ListFolder { folder, problem } => {
                write!(f, "failed to list {}: {problem}", folder.display())
            }
            Self::Cancelled => write!(
                f,
                "the upgrade was cancelled; run it again to finish it from where it stopped"
            ),
        }
    }
}

impl std::error::Error for UpgradeWorkspaceError {}

impl From<Cancelled> for UpgradeWorkspaceError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// Bring the tree of workspaces whose root is the workspace at or above
/// `path` to the current file versions, and report every file and what
/// became of it.
///
/// The tree is that workspace and every workspace nested in it. The run
/// refuses a version 1 workspace file before it reads anything else, the
/// root's first, then each nested one's, and runs the phases of
/// `specs/cli/workspace/ws-upgrade-command.md`:
///
/// 1. Walk the tree, nested workspaces included but no symbolic link or
///    junction, for `.sfmr`, `.matches` and `.camrig` files.
/// 2. Read each file's metadata, and read whole each one below the current
///    version, as the run will read it. A file the file system will not give
///    up fails; one the current reader refuses is left as it is.
/// 3. Rewrite each file below the current version at its own path: a `.sfmr`
///    by [`SfmrReconstruction::load`] and [`SfmrReconstruction::save`]; a
///    `.matches` or `.camrig` file by reading and writing it.
///
/// A run that stops part way, by cancellation, can be run again to finish.
///
/// It reads and writes every file of a tree, minutes of work on a large
/// one, so it takes a `Progress`: it opens the phases `find files`, `plan`
/// and `rewrite files`, counts folders and files in them, and checks for
/// cancellation between files.
///
/// ```no_run
/// use sfmtool_core::workspace::{upgrade_workspace, UpgradeWorkspaceOptions};
/// use sfmtool_core::Progress;
///
/// let options = UpgradeWorkspaceOptions { dry_run: true };
/// let report = upgrade_workspace("my_workspace".as_ref(), &options, &Progress::none())?;
/// for file in &report.files {
///     println!("{}: {:?}", file.path.display(), file.outcome);
/// }
/// # Ok::<(), sfmtool_core::workspace::UpgradeWorkspaceError>(())
/// ```
///
/// # Errors
///
/// [`UpgradeWorkspaceError`], each named there. Every one but `Cancelled` is
/// found in phases 1 and 2, before anything is written, so a dry run gives it
/// too.
pub fn upgrade_workspace(
    path: &Path,
    options: &UpgradeWorkspaceOptions,
    progress: &Progress<'_>,
) -> Result<UpgradeWorkspaceReport, UpgradeWorkspaceError> {
    // The workspace is spelled with each `..` resolved, so the paths the
    // report gives read plainly. Where a `..` follows a symbolic link, the
    // path can name a different directory on disk than it does with each `..`
    // resolved by name; the workspace is then spelled by its real path, so
    // the run upgrades the workspace that was found.
    let identity = |dir: &Path| std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    let root = find_workspace_for_path(path)
        .map(|found| {
            let resolved = normalized(&lexically_resolved(&found));
            if identity(&resolved) == identity(&found) {
                resolved
            } else {
                identity(&found)
            }
        })
        .ok_or_else(|| UpgradeWorkspaceError::NoWorkspace {
            path: path.to_path_buf(),
        })?;
    plan::check_workspace_file(&root.join(WORKSPACE_FILE_NAME))?;
    let plan = plan::plan(&root, progress)?;

    let mut report = UpgradeWorkspaceReport {
        workspace_dir: root,
        workspaces: plan.workspaces,
        dry_run: options.dry_run,
        files: Vec::with_capacity(plan.files.len()),
        linked_directories: plan.linked_directories,
        notes: Vec::new(),
    };

    if options.dry_run {
        for file in plan.files {
            report.files.push(UpgradedFile {
                path: file.path,
                kind: file.kind,
                outcome: match file.action {
                    Action::Rewrite => UpgradeOutcome::Converted,
                    Action::Leave(outcome) => outcome,
                },
            });
        }
    } else {
        let phase = progress.phase("rewrite files");
        let total = plan.files.len() as u64;
        for (done, file) in plan.files.into_iter().enumerate() {
            phase.check_cancel()?;
            let outcome = match file.action {
                Action::Leave(outcome) => outcome,
                Action::Rewrite => match file.kind {
                    UpgradeFileKind::Sfmr => rewrite_sfmr(&file.path, &phase)?,
                    UpgradeFileKind::Matches => rewrite_matches(&file.path),
                    UpgradeFileKind::Camrig => rewrite_camrig(&file.path),
                },
            };
            report.files.push(UpgradedFile {
                path: file.path,
                kind: file.kind,
                outcome,
            });
            phase.count(done as u64 + 1, Some(total), "file");
        }
    }
    if report
        .files
        .iter()
        .any(|f| f.kind == UpgradeFileKind::Sfmr && f.outcome == UpgradeOutcome::Converted)
    {
        report.notes.push(
            "Saving a .sfmr file changes its content_xxh128, so a point ID \
             (pt3d_<hash>_<index>) written down from a rewritten .sfmr before the upgrade no \
             longer resolves."
                .into(),
        );
    }
    Ok(report)
}

/// Phase 3 for one `.sfmr` file. Phase 2 loaded it, so an error on loading
/// it again means it was locked or changed since.
fn rewrite_sfmr(path: &Path, progress: &Progress<'_>) -> Result<UpgradeOutcome, Cancelled> {
    let recon = match SfmrReconstruction::load(path, progress) {
        Ok(r) => r,
        Err(e) => return Ok(read_failure(e)),
    };
    match recon.save(path, progress) {
        Ok(_) => Ok(UpgradeOutcome::Converted),
        Err(sfmtool_sfmr_format::SfmrError::Io(e))
            if e.kind() == std::io::ErrorKind::Interrupted =>
        {
            Err(Cancelled)
        }
        Err(e) => Ok(plan::write_failure(&e)),
    }
}

/// Phase 3 for one `.matches` file.
fn rewrite_matches(path: &Path) -> UpgradeOutcome {
    let data = match sfmtool_matches_format::read_matches(path) {
        Ok(d) => d,
        Err(e) => return read_failure(e),
    };
    match sfmtool_matches_format::write_matches(path, &data, ZSTD_LEVEL) {
        Ok(()) => UpgradeOutcome::Converted,
        Err(e) => UpgradeOutcome::Failed(format!("it could not be written: {e}")),
    }
}

/// Phase 3 for one `.camrig` file.
fn rewrite_camrig(path: &Path) -> UpgradeOutcome {
    let data = match sfmtool_camrig_format::read_camrig(path) {
        Ok(d) => d,
        Err(e) => return read_failure(e),
    };
    match sfmtool_camrig_format::write_camrig(path, &data, ZSTD_LEVEL) {
        Ok(()) => UpgradeOutcome::Converted,
        Err(e) => UpgradeOutcome::Failed(format!("it could not be written: {e}")),
    }
}

#[cfg(test)]
mod tests;
