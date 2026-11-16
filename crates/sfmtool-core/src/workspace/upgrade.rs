// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bring a workspace's files to the current versions: [`upgrade_workspace`].
//!
//! A workspace made by sfmtool 0.3 or a development build holds a version 1
//! `.sfm-workspace.json`, version 1 `.sift` files in features folders named
//! from version 1 options, and `.sfmr`, `.matches` and `.camrig` files of
//! older versions, many of which name those folders and record the `.sift`
//! files' content hashes. The upgrade converts every version 1 `.sift` file
//! to version 2 without extracting it again, rewrites each file that records
//! an old folder so that it records the new one and the new files' hashes,
//! rewrites the workspace file as version 2, and deletes the old folders. See
//! `specs/drafts/ws-upgrade-files-amendment.md`, whose phases 1 to 6 the
//! function runs in order.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use serde::Serialize;
use sfmtool_archive_io::{parse_hash_bytes, WorkspaceContents};
use sfmtool_sift_format::{SiftData, SIFT_READINGS_VERSION};

use super::{find_workspace_for_path, FeatureOptionsError, WORKSPACE_FILE_NAME};
use crate::features::sift::Version2SiftFolder;
use crate::progress::{Cancelled, Progress};
use crate::ObservationSource;
use crate::SfmrReconstruction;

mod plan;

use plan::{missing_sift_file, sift_path, Action, Plan};

/// The zstd level the upgrade writes `.matches` and `.camrig` files at, the
/// level their bindings write by default.
const ZSTD_LEVEL: i32 = 3;

/// How many items an error's message names before it gives the count of the
/// rest.
const NAMED_ITEMS: usize = 10;

/// How [`upgrade_workspace`] runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpgradeWorkspaceOptions {
    /// Run phases 1 and 2 only, find the files and plan their conversion,
    /// and write nothing. The report then gives the outcome each file and
    /// folder would have.
    pub dry_run: bool,
}

/// The kinds of file the upgrade reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpgradeFileKind {
    /// `.sfm-workspace.json`.
    WorkspaceFile,
    /// A `.sfmr` reconstruction.
    Sfmr,
    /// A `.matches` file.
    Matches,
    /// A `.camrig` file.
    Camrig,
    /// A `.kdf` index.
    Kdf,
}

/// What the upgrade did to a file, or in a dry run what it would do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", content = "reason", rename_all = "snake_case")]
pub enum UpgradeOutcome {
    /// Rewritten at its own path as the current version, naming the new
    /// folder where it named an old one.
    Converted,
    /// Already the current version and naming no old folder, so not written.
    AlreadyCurrent,
    /// Left as it is for the reason given. It does not stop the old folders
    /// from being deleted, since no current reader reads it through them.
    NotConverted(String),
    /// Could not be read or rewritten, for the reason given. Any such file
    /// stops the old folders from being deleted.
    Failed(String),
    /// Deleted: a `.kdf` index built from an old folder.
    Deleted,
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

/// One version 1 features folder and what became of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpgradedFolder {
    /// The version 1 folder.
    pub old_folder: PathBuf,
    /// The folder its `.sift` files were converted into, or `None` for an
    /// orphan, a folder that neither the workspace file nor any `.sfmr` or
    /// `.matches` file of the workspace names, whose files are deleted
    /// without being converted.
    pub new_folder: Option<PathBuf>,
    /// How many version 1 `.sift` files it held.
    pub sift_file_count: usize,
    /// Whether its version 1 `.sift` files were deleted (in a dry run,
    /// whether they would be).
    pub deleted: bool,
    /// The files the deletion left in the folder because they are not
    /// version 1 `.sift` files, which keep the folder from being removed.
    pub kept_files: Vec<PathBuf>,
}

/// What [`upgrade_workspace`] did, or in a dry run what it would do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpgradeWorkspaceReport {
    /// The workspace directory.
    pub workspace_dir: PathBuf,
    /// Whether this was a dry run, which wrote nothing.
    pub dry_run: bool,
    /// The workspace file first, then every `.sfmr`, `.matches`, `.camrig`
    /// and `.kdf` file the walk found, each kind sorted by path.
    pub files: Vec<UpgradedFile>,
    /// Every version 1 features folder the walk found, sorted by path.
    pub folders: Vec<UpgradedFolder>,
    /// How many version 1 `.sift` files were converted (or would be).
    pub converted_sift_file_count: usize,
    /// The directories below the workspace that hold a `.sfm-workspace.json`
    /// of their own. They are other workspaces, so the walk did not enter
    /// them and nothing in them was changed.
    pub nested_workspaces: Vec<PathBuf>,
    /// Whether the version 1 folders were deleted (or would be): only when
    /// no file [`Failed`](UpgradeOutcome::Failed).
    pub old_folders_deleted: bool,
    /// Notes for the user, such as that point IDs taken from a rewritten
    /// `.sfmr` before the upgrade no longer resolve.
    pub notes: Vec<String>,
}

impl UpgradeWorkspaceReport {
    /// Whether there was nothing to do: the workspace file is version 2, no
    /// file was converted or deleted, and no version 1 folder was found.
    pub fn nothing_to_do(&self) -> bool {
        self.folders.is_empty()
            && self.files.iter().all(|f| {
                matches!(
                    f.outcome,
                    UpgradeOutcome::AlreadyCurrent | UpgradeOutcome::NotConverted(_)
                )
            })
    }
}

/// Two version 1 `.sift` files that would both be converted to one new file
/// and differ, from two old folders whose records convert to the same new
/// one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiftConflict {
    /// The new file both would become.
    pub new_file: PathBuf,
    /// One version 1 file.
    pub first: PathBuf,
    /// The other.
    pub second: PathBuf,
}

/// One version 1 `.sift` file that could not be converted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiftFileFailure {
    /// The version 1 file.
    pub sift_path: PathBuf,
    /// Why.
    pub problem: String,
}

/// Why [`upgrade_workspace`] stopped. Each is raised before any `.sfmr`,
/// `.matches`, `.camrig` or `.kdf` file or the workspace file is written,
/// except [`Cancelled`](Self::Cancelled), which leaves the workspace in one
/// of the states a rerun finishes.
#[derive(Debug)]
pub enum UpgradeWorkspaceError {
    /// No `.sfm-workspace.json` at or above the path.
    NoWorkspace {
        /// The path searched from.
        path: PathBuf,
    },
    /// The workspace file cannot be read or is not one this build upgrades.
    WorkspaceFile {
        /// The workspace file.
        path: PathBuf,
        /// What is wrong.
        problem: String,
    },
    /// A directory of the workspace could not be listed, so a version 1
    /// folder in it could be missed.
    ListFolder {
        /// The directory.
        folder: PathBuf,
        /// What went wrong.
        problem: String,
    },
    /// A version 1 record cannot be converted to version 2: an
    /// `sfmtool-undistort` record, a feature tool this build does not know,
    /// or a `gray_formula` that does not parse. Its folder could be neither
    /// converted nor deleted.
    FeatureOptions {
        /// The files that hold the record.
        files: Vec<PathBuf>,
        /// Why it cannot be converted.
        error: FeatureOptionsError,
    },
    /// Two old folders map to one new folder and hold different files for
    /// one image.
    Conflicts(Vec<SiftConflict>),
    /// Version 1 `.sift` files that could not be converted, every one listed:
    /// an image missing, unreadable or changed, or a new file already in the
    /// new folder that is not this file's conversion. Other `.sift` files may
    /// have been converted; nothing else was written.
    SiftFiles(Vec<SiftFileFailure>),
    /// The caller asked the upgrade to stop.
    Cancelled,
}

fn write_list<T>(
    f: &mut fmt::Formatter<'_>,
    items: &[T],
    item: impl Fn(&mut fmt::Formatter<'_>, &T) -> fmt::Result,
) -> fmt::Result {
    for x in items.iter().take(NAMED_ITEMS) {
        write!(f, "\n  ")?;
        item(f, x)?;
    }
    if items.len() > NAMED_ITEMS {
        write!(f, "\n  and {} more", items.len() - NAMED_ITEMS)?;
    }
    Ok(())
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
            Self::WorkspaceFile { path, problem } => {
                write!(f, "the workspace file {} {problem}", path.display())
            }
            Self::ListFolder { folder, problem } => {
                write!(f, "failed to list {}: {problem}", folder.display())
            }
            Self::FeatureOptions { files, error } => {
                write!(
                    f,
                    "a version 1 feature record cannot be converted to version 2, so its \
                     features folder cannot be upgraded: {error}; it is recorded by:"
                )?;
                write_list(f, files, |f, p| write!(f, "{}", p.display()))
            }
            Self::Conflicts(conflicts) => {
                write!(
                    f,
                    "two old features folders convert to the same new folder and hold \
                     different .sift files for the same image; remove the one that is not \
                     wanted and run again:"
                )?;
                write_list(f, conflicts, |f, c| {
                    write!(
                        f,
                        "{} and {} would both become {}",
                        c.first.display(),
                        c.second.display(),
                        c.new_file.display()
                    )
                })
            }
            Self::SiftFiles(failures) => {
                write!(
                    f,
                    "{} version 1 .sift file(s) could not be converted, so no other file was \
                     rewritten and nothing was deleted; fix them and run again:",
                    failures.len()
                )?;
                write_list(f, failures, |f, x| {
                    write!(f, "{}: {}", x.sift_path.display(), x.problem)
                })
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

/// The old and new hashes of one converted `.sift` file.
#[derive(Debug, Clone, Copy)]
struct NewHashes {
    content: [u8; 16],
    feature_tool: [u8; 16],
}

/// Bring the workspace found at or above `path` to the current file
/// versions, and report every file and folder and what became of it.
///
/// It runs the phases of `specs/drafts/ws-upgrade-files-amendment.md`:
///
/// 1. Walk the workspace for `.sfmr`, `.matches`, `.camrig` and `.kdf`
///    files and for folders of version 1 `.sift` files
///    ([`version_1_sift_files`](crate::features::sift::version_1_sift_files)),
///    without entering a nested workspace (listed in the report).
/// 2. Map each distinct version 1 record, the workspace file's and each one a
///    `.sfmr` or `.matches` file of this workspace holds, to its new folder:
///    [`upgrade_feature_options`](super::upgrade_feature_options) and
///    [`feature_prefix_dir`](super::feature_prefix_dir). A version 1 folder
///    is named by a record when it is `<image dir>/<feature_prefix_dir>`;
///    one no record names is an orphan.
/// 3. Convert every version 1 `.sift` file of a folder that is not an orphan
///    into its new folder ([`Version2SiftFolder`]), in parallel, the image
///    being `<image dir>/<name>` for `<image dir>/<feature_prefix_dir>/<name>.sift`.
///    Any failure stops the run here, listing every file that failed.
/// 4. Rewrite each `.sfmr` and `.matches` file that names an old folder or
///    is below the current version, each `.camrig` below version 2, at its
///    own path, and delete each `.kdf` built from an old folder.
/// 5. Write the workspace file as version 2.
/// 6. When no file failed, delete the version 1 `.sift` files of every
///    version 1 folder, orphans included, and the directories that leaves
///    empty.
///
/// A file whose record resolves to another workspace, a cluster `.matches`
/// file below version 6 and a `.sfmr` or `.matches` file whose recorded
/// `.sift` files are not the version 1 files in an old folder are left as
/// they are, listed as not converted. A run that stops part way, by
/// cancellation or a failure, can be run again to finish.
///
/// It reads and writes every file of a workspace, minutes of work on a large
/// one, so it takes a `Progress`: it opens the phases `find files`, `plan`,
/// `convert .sift files`, `rewrite files`, `workspace file` and `delete old
/// folders`, counts folders, `.sift` files and files in them, and checks for
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
/// [`UpgradeWorkspaceError`], each named there. Every one but `SiftFiles` and
/// `Cancelled` is found in phases 1 and 2, before anything is written, so a
/// dry run gives it too.
pub fn upgrade_workspace(
    path: &Path,
    options: &UpgradeWorkspaceOptions,
    progress: &Progress<'_>,
) -> Result<UpgradeWorkspaceReport, UpgradeWorkspaceError> {
    let root = find_workspace_for_path(path).ok_or_else(|| UpgradeWorkspaceError::NoWorkspace {
        path: path.to_path_buf(),
    })?;
    let root_canonical =
        std::fs::canonicalize(&root).map_err(|e| UpgradeWorkspaceError::ListFolder {
            folder: root.clone(),
            problem: e.to_string(),
        })?;
    let plan = plan::plan(&root, &root_canonical, options.dry_run, progress)?;

    let mut report = UpgradeWorkspaceReport {
        workspace_dir: root.clone(),
        dry_run: options.dry_run,
        files: Vec::with_capacity(plan.files.len() + 1),
        folders: Vec::new(),
        converted_sift_file_count: plan.jobs.len(),
        nested_workspaces: plan.nested_workspaces.clone(),
        old_folders_deleted: false,
        notes: Vec::new(),
    };
    report.files.push(UpgradedFile {
        path: plan.workspace_file.clone(),
        kind: UpgradeFileKind::WorkspaceFile,
        outcome: if plan.new_workspace_file.is_some() {
            UpgradeOutcome::Converted
        } else {
            UpgradeOutcome::AlreadyCurrent
        },
    });

    if options.dry_run {
        for file in &plan.files {
            report.files.push(UpgradedFile {
                path: file.path.clone(),
                kind: file.kind,
                outcome: match &file.action {
                    Action::Rewrite { .. } => UpgradeOutcome::Converted,
                    Action::Delete => UpgradeOutcome::Deleted,
                    Action::Leave(outcome) => outcome.clone(),
                },
            });
        }
        report.old_folders_deleted = !any_failed(&report.files);
        report.folders = folder_reports(&plan, report.old_folders_deleted);
        finish_notes(&mut report);
        return Ok(report);
    }

    // Phase 3.
    let table = convert_sift_files(&plan, progress)?;

    // Phase 4.
    {
        let phase = progress.phase("rewrite files");
        let total = plan.files.len() as u64;
        for (done, file) in plan.files.iter().enumerate() {
            phase.check_cancel()?;
            let outcome = match &file.action {
                Action::Leave(outcome) => outcome.clone(),
                Action::Delete => match std::fs::remove_file(&file.path) {
                    Ok(()) => UpgradeOutcome::Deleted,
                    Err(e) => UpgradeOutcome::Failed(format!("it could not be deleted: {e}")),
                },
                Action::Rewrite {
                    new_contents,
                    old_prefix,
                    names_sift_files,
                } => {
                    let rewrite = Rewrite {
                        plan: &plan,
                        table: &table,
                        new_contents: new_contents.as_ref(),
                        old_prefix,
                        names_sift_files: *names_sift_files,
                    };
                    match file.kind {
                        UpgradeFileKind::Sfmr => rewrite.sfmr(&file.path, &phase)?,
                        UpgradeFileKind::Matches => rewrite.matches(&file.path),
                        _ => rewrite_camrig(&file.path),
                    }
                }
            };
            report.files.push(UpgradedFile {
                path: file.path.clone(),
                kind: file.kind,
                outcome,
            });
            phase.count(done as u64 + 1, Some(total), "file");
        }
    }

    // Phase 5.
    if let Some(config) = &plan.new_workspace_file {
        let phase = progress.phase("workspace file");
        phase.check_cancel()?;
        let text = serde_json::to_string_pretty(config).expect("a JSON value serializes");
        if let Err(e) = sfmtool_archive_io::write_atomically(
            &plan.workspace_file,
            |file| -> std::io::Result<()> {
                use std::io::Write;
                file.write_all(text.as_bytes())
            },
        ) {
            report.files[0].outcome =
                UpgradeOutcome::Failed(format!("it could not be written: {e}"));
        }
    }

    // Phase 6.
    report.old_folders_deleted = !any_failed(&report.files);
    report.folders = folder_reports(&plan, false);
    if report.old_folders_deleted {
        let phase = progress.phase("delete old folders");
        let total = plan.folders.len() as u64;
        for (done, (folder, folder_report)) in
            plan.folders.iter().zip(&mut report.folders).enumerate()
        {
            phase.check_cancel()?;
            delete_old_folder(&plan.root, folder, folder_report);
            phase.count(done as u64 + 1, Some(total), "folder");
        }
        report.old_folders_deleted = report.folders.iter().all(|f| f.deleted);
    }
    finish_notes(&mut report);
    Ok(report)
}

fn any_failed(files: &[UpgradedFile]) -> bool {
    files
        .iter()
        .any(|f| matches!(f.outcome, UpgradeOutcome::Failed(_)))
}

fn folder_reports(plan: &Plan, deleted: bool) -> Vec<UpgradedFolder> {
    plan.folders
        .iter()
        .map(|folder| UpgradedFolder {
            old_folder: folder.path.clone(),
            new_folder: folder.new_folder.clone(),
            sift_file_count: folder.sift_files.len(),
            deleted,
            kept_files: Vec::new(),
        })
        .collect()
}

fn finish_notes(report: &mut UpgradeWorkspaceReport) {
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
    if !report.old_folders_deleted && !report.folders.is_empty() {
        report.notes.push(
            "The version 1 features folders were not deleted, because a file failed; fix the \
             files listed as failed and run the upgrade again."
                .into(),
        );
    }
}

/// Phase 3: convert every `.sift` file of the plan's jobs, and return the
/// table from each old file's content hash to the new file's hashes.
fn convert_sift_files(
    plan: &Plan,
    progress: &Progress<'_>,
) -> Result<HashMap<[u8; 16], NewHashes>, UpgradeWorkspaceError> {
    let phase = progress.phase("convert .sift files");
    let mut failures = Vec::new();
    let mut folders = Vec::with_capacity(plan.new_folders.len());
    for folder in &plan.new_folders {
        match Version2SiftFolder::create(folder) {
            Ok(f) => folders.push(Some(f)),
            Err(e) => {
                failures.push(SiftFileFailure {
                    sift_path: folder.clone(),
                    problem: e.to_string(),
                });
                folders.push(None);
            }
        }
    }
    if !failures.is_empty() {
        return Err(UpgradeWorkspaceError::SiftFiles(failures));
    }
    let total = plan.jobs.len() as u64;
    let done = AtomicU64::new(0);
    let results: Vec<Result<([u8; 16], NewHashes), SiftFileFailure>> = plan
        .jobs
        .par_iter()
        .map(|job| {
            let failure = |problem: String| SiftFileFailure {
                sift_path: job.source.clone(),
                problem,
            };
            if phase.is_cancelled() {
                return Err(failure(String::new()));
            }
            let folder = folders[job.folder].as_ref().expect("every folder was made");
            let result = convert_one(folder, &job.source, &job.image, &phase).map_err(failure);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            phase.count(n, Some(total), "file");
            result
        })
        .collect();
    phase.check_cancel()?;
    let mut table = HashMap::with_capacity(results.len());
    for result in results {
        match result {
            Ok((old, new)) => {
                table.insert(old, new);
            }
            Err(failure) => failures.push(failure),
        }
    }
    if !failures.is_empty() {
        return Err(UpgradeWorkspaceError::SiftFiles(failures));
    }
    Ok(table)
}

/// Convert `source` into `folder`, or take the file already there when it is
/// `source`'s conversion, as a rerun finds it. A file there that is not is
/// refused rather than replaced: it may be an extraction other files record.
fn convert_one(
    folder: &Version2SiftFolder,
    source: &Path,
    image: &Path,
    progress: &Progress<'_>,
) -> Result<([u8; 16], NewHashes), String> {
    let hashes = |old: &str, content: &str, feature_tool: &str| -> Result<_, String> {
        let parse = |h: &str| parse_hash_bytes(h).map_err(|e| e.to_string());
        Ok((
            parse(old)?,
            NewHashes {
                content: parse(content)?,
                feature_tool: parse(feature_tool)?,
            },
        ))
    };
    let destination = folder.path().join(source.file_name().unwrap_or_default());
    if destination.is_file() {
        let old = sfmtool_sift_format::read_sift(source).map_err(|e| e.to_string())?;
        let new = sfmtool_sift_format::read_sift(&destination)
            .map_err(|e| format!("the file {} cannot be read: {e}", destination.display()))?;
        if !is_conversion_of(&new, &old) {
            return Err(format!(
                "the new folder already holds {}, which is not this file's conversion; move it \
                 away if it is not wanted, or keep it and remove this file",
                destination.display()
            ));
        }
        return hashes(
            &old.content_hash.content_xxh128,
            &new.content_hash.content_xxh128,
            &new.content_hash.feature_tool_xxh128,
        );
    }
    let converted = folder
        .convert_version_1_sift_file(source, image, progress)
        .map_err(|e| e.to_string())?;
    hashes(
        &converted.source_content_hash.content_xxh128,
        &converted.content_hash.content_xxh128,
        &converted.content_hash.feature_tool_xxh128,
    )
}

/// Whether `new` is what converting the version 1 `old` writes: a version 2
/// file of the same tool, type, image and features, with `old`'s options
/// upgraded.
fn is_conversion_of(new: &SiftData, old: &SiftData) -> bool {
    let (n, o) = (&new.metadata, &old.metadata);
    let options = crate::workspace::upgrade_feature_options(
        &old.feature_tool_metadata.feature_tool,
        &old.feature_tool_metadata.feature_options,
    );
    n.version >= SIFT_READINGS_VERSION
        && o.version < SIFT_READINGS_VERSION
        && new.feature_tool_metadata.feature_tool == old.feature_tool_metadata.feature_tool
        && new.feature_tool_metadata.feature_type == old.feature_tool_metadata.feature_type
        && options.is_ok_and(|o| o == new.feature_tool_metadata.feature_options)
        && n.image_name == o.image_name
        && n.image_file_xxh128 == o.image_file_xxh128
        && n.image_file_size == o.image_file_size
        && (n.image_width, n.image_height) == (o.image_width, o.image_height)
        && n.feature_count == o.feature_count
        && new.positions_xy == old.positions_xy
        && new.affine_shapes == old.affine_shapes
        && new.descriptors == old.descriptors
        && new.thumbnail_y_x_rgb == old.thumbnail_y_x_rgb
}

/// The phase 4 rewrite of one `.sfmr` or `.matches` file.
struct Rewrite<'a> {
    plan: &'a Plan,
    table: &'a HashMap<[u8; 16], NewHashes>,
    new_contents: Option<&'a WorkspaceContents>,
    old_prefix: &'a str,
    names_sift_files: bool,
}

impl Rewrite<'_> {
    /// The new hashes of the `.sift` files `names` and `hashes` record, or
    /// why the file is not converted.
    fn new_hashes(
        &self,
        names: &[String],
        hashes: &[[u8; 16]],
    ) -> Result<Vec<NewHashes>, UpgradeOutcome> {
        if let Some(reason) = missing_sift_file(self.plan, names, hashes, self.old_prefix) {
            return Err(UpgradeOutcome::NotConverted(reason));
        }
        hashes
            .iter()
            .map(|h| {
                self.table.get(h).copied().ok_or_else(|| {
                    UpgradeOutcome::Failed("a .sift file it records was not converted".to_string())
                })
            })
            .collect()
    }

    fn sfmr(&self, path: &Path, progress: &Progress<'_>) -> Result<UpgradeOutcome, Cancelled> {
        let mut recon = match SfmrReconstruction::load(path, progress) {
            Ok(r) => r,
            Err(e) => return Ok(UpgradeOutcome::NotConverted(plan::refused(&e.to_string()))),
        };
        if let Some(new_contents) = self.new_contents {
            let names: Vec<String> = recon
                .image_set
                .images
                .iter()
                .map(|i| i.name.clone())
                .collect();
            if let ObservationSource::SiftFiles {
                feature_tool_hashes,
                sift_content_hashes,
                ..
            } = &mut recon.point_set.observations
            {
                if self.names_sift_files {
                    let new = match self.new_hashes(&names, sift_content_hashes) {
                        Ok(new) => new,
                        Err(outcome) => return Ok(outcome),
                    };
                    for (i, hashes) in new.into_iter().enumerate() {
                        sift_content_hashes[i] = hashes.content;
                        feature_tool_hashes[i] = hashes.feature_tool;
                    }
                }
            }
            // Switched before the save, so a file below version 15 takes its
            // readings from the new `.sift` files rather than the photographs.
            recon.metadata.workspace.contents = new_contents.clone();
        }
        match recon.save(path, progress) {
            Ok(_) => Ok(UpgradeOutcome::Converted),
            Err(sfmtool_sfmr_format::SfmrError::Io(e))
                if e.kind() == std::io::ErrorKind::Interrupted =>
            {
                Err(Cancelled)
            }
            Err(e) => Ok(UpgradeOutcome::Failed(format!(
                "it could not be written: {e}"
            ))),
        }
    }

    fn matches(&self, path: &Path) -> UpgradeOutcome {
        let mut data = match sfmtool_matches_format::read_matches(path) {
            Ok(d) => d,
            Err(e) => return UpgradeOutcome::NotConverted(plan::refused(&e.to_string())),
        };
        if let Some(new_contents) = self.new_contents {
            let new = match self.new_hashes(&data.image_names, &data.sift_content_hashes) {
                Ok(new) => new,
                Err(outcome) => return outcome,
            };
            for (i, hashes) in new.into_iter().enumerate() {
                data.sift_content_hashes[i] = hashes.content;
                data.feature_tool_hashes[i] = hashes.feature_tool;
            }
            data.metadata.workspace.contents = new_contents.clone();
        }
        // A file of version 3 or older stores no image sizes; each is taken
        // from the image's `.sift` file in the folder it now records.
        if data.image_dims.is_none() {
            let prefix = &data.metadata.workspace.contents.feature_prefix_dir;
            let mut dims = Vec::with_capacity(2 * data.image_names.len());
            for name in &data.image_names {
                let sift = sift_path(&self.plan.root, name, prefix);
                match sfmtool_sift_format::read_sift_metadata(&sift) {
                    Ok((_, metadata, _)) => {
                        dims.extend([metadata.image_width, metadata.image_height]);
                    }
                    Err(e) => {
                        return UpgradeOutcome::Failed(format!(
                            "it stores no image sizes, and the .sift file {} to take them from \
                             cannot be read: {e}",
                            sift.display()
                        ))
                    }
                }
            }
            data.image_dims =
                ndarray::Array2::from_shape_vec((data.image_names.len(), 2), dims).ok();
        }
        match sfmtool_matches_format::write_matches(path, &data, ZSTD_LEVEL) {
            Ok(()) => UpgradeOutcome::Converted,
            Err(e) => UpgradeOutcome::Failed(format!("it could not be written: {e}")),
        }
    }
}

fn rewrite_camrig(path: &Path) -> UpgradeOutcome {
    let data = match sfmtool_camrig_format::read_camrig(path) {
        Ok(d) => d,
        Err(e) => return UpgradeOutcome::Failed(format!("it cannot be read: {e}")),
    };
    match sfmtool_camrig_format::write_camrig(path, &data, ZSTD_LEVEL) {
        Ok(()) => UpgradeOutcome::Converted,
        Err(e) => UpgradeOutcome::Failed(format!("it could not be written: {e}")),
    }
}

/// Phase 6 for one folder: delete its version 1 `.sift` files, then the
/// directories that leaves empty, up to but not including the image
/// directory. An orphan's image directory is not known, so the folder and,
/// when it is inside a `features` folder, that folder are removed when empty.
fn delete_old_folder(root: &Path, folder: &plan::OldFolder, report: &mut UpgradedFolder) {
    for file in &folder.sift_files {
        if let Err(e) = std::fs::remove_file(file) {
            if e.kind() != std::io::ErrorKind::NotFound {
                report.kept_files.push(file.clone());
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(&folder.path) {
        let mut kept: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        kept.sort();
        for path in kept {
            if !report.kept_files.contains(&path) {
                report.kept_files.push(path);
            }
        }
    }
    report.deleted = folder
        .sift_files
        .iter()
        .all(|f| !report.kept_files.contains(f));
    if !report.kept_files.is_empty() {
        return;
    }
    let stop = match &folder.image_dir {
        Some(dir) => dir.clone(),
        None => match folder.path.parent() {
            Some(parent) if parent.file_name().is_some_and(|n| n == "features") => {
                parent.parent().unwrap_or(root).to_path_buf()
            }
            Some(parent) => parent.to_path_buf(),
            None => root.to_path_buf(),
        },
    };
    let stop = plan::normalized(&stop);
    let root = plan::normalized(root);
    let mut dir = plan::normalized(&folder.path);
    while dir != stop && dir != root && dir.starts_with(&root) {
        if std::fs::remove_dir(&dir).is_err() {
            break;
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
}

#[cfg(test)]
mod tests;
