// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Phases 1 and 2 of the workspace upgrade: find the files and the version 1
//! features folders of a tree of workspaces, and map each version 1 record to
//! its new folder.
//!
//! Nothing here writes. A dry run is these two phases and the report they
//! give; a run goes on from the [`Plan`] to the phases in the parent module.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use rayon::prelude::*;
use sfmtool_archive_io::{parse_hash_bytes, WorkspaceContents};

use super::{
    SiftConflict, SiftFileFailure, UpgradeFileKind, UpgradeOutcome, UpgradeWorkspaceError,
    WORKSPACE_FILE_NAME,
};
use crate::features::sift::{scan_sift_folder, ScannedSiftFile};
use crate::progress::Progress;
use crate::workspace::{
    feature_prefix_dir, feature_type_for_tool, has_image_to_gray, upgrade_feature_options,
    FeatureOptionsError,
};

/// What a run does to one `.sfmr`, `.matches`, `.camrig` or `.kdf` file.
#[derive(Debug, Clone)]
pub(super) enum Action {
    /// Read it and write it back as the current version, at its own path.
    Rewrite {
        /// The directory of the workspace its record resolves to, which its
        /// image names are relative to.
        workspace: PathBuf,
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
    /// The folder its files are converted into, or `None` when neither a
    /// version 1 workspace file nor a file the run rewrites names it.
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

/// One workspace of the tree, and its workspace file.
#[derive(Debug, Clone)]
pub(super) struct WorkspacePlan {
    /// The workspace directory.
    pub(super) dir: PathBuf,
    /// Its `.sfm-workspace.json`.
    pub(super) file: PathBuf,
    /// The version 2 workspace file to write, or `None` when it is version 2
    /// already.
    pub(super) new_file: Option<serde_json::Value>,
}

/// What phases 1 and 2 found.
#[derive(Debug)]
pub(super) struct Plan {
    /// The root of the tree, the outermost workspace.
    pub(super) root: PathBuf,
    /// Its real path.
    pub(super) root_canonical: PathBuf,
    /// The workspaces of the tree, the root first, then the nested ones
    /// sorted by path.
    pub(super) workspaces: Vec<WorkspacePlan>,
    pub(super) folders: Vec<OldFolder>,
    pub(super) new_folders: Vec<PathBuf>,
    pub(super) jobs: Vec<SiftJob>,
    pub(super) files: Vec<FilePlan>,
    pub(super) linked_directories: Vec<PathBuf>,
    /// Why phase 6 deletes nothing at all, when a folder it would delete lies
    /// outside the tree.
    pub(super) deletion_refused: Option<String>,
}

/// `path` with its components joined again, so two spellings of one path
/// (`a/b` and `a\b` on Windows, a trailing separator) compare and hash equal.
pub(super) fn normalized(path: &Path) -> PathBuf {
    path.components().collect()
}

/// `path` with each `..` taking off the name before it, without touching
/// the file system.
fn lexically_resolved(path: &Path) -> PathBuf {
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

/// Whether `path` is relative and holds only plain names: no root, drive or
/// `..`, so joining it to a directory stays below that directory.
fn is_plain_relative(path: &Path) -> bool {
    path.components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

/// The directory image `name` (workspace-relative, `/`-separated) is in, in
/// the workspace at `workspace`, with each `..` of the name resolved by name,
/// so that it is spelled as the walk spells the directories it finds.
fn image_dir_of_name(workspace: &Path, name: &str) -> PathBuf {
    let parent = Path::new(name).parent().unwrap_or(Path::new(""));
    normalized(&lexically_resolved(&workspace.join(parent)))
}

/// The `.sift` path image `name` (workspace-relative, `/`-separated) has
/// under `prefix` in the workspace at `workspace`:
/// `<workspace>/<name's directory>/<prefix>/<name's file name>.sift`, with
/// each `..` resolved by name.
pub(super) fn sift_path(workspace: &Path, name: &str, prefix: &str) -> PathBuf {
    let file_name = Path::new(name)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    normalized(
        &image_dir_of_name(workspace, name)
            .join(prefix)
            .join(format!("{file_name}.sift")),
    )
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

/// Why a file the current reader refuses is not converted.
fn refused(error: &str) -> String {
    format!("the current reader refuses it, so it is left as it is: {error}")
}

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

/// Walk the tree at `root` for the files the upgrade reads, nested
/// workspaces included, which it lists. It does not enter a symbolic link or
/// junction to a directory, whose target may be outside the tree; it lists
/// each.
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

/// The record a file or a workspace file names, in workspace `workspace`, as
/// the key two equal records share.
fn record_key(workspace: usize, contents: &WorkspaceContents) -> String {
    format!(
        "{workspace}\u{0}{}\u{0}{}\u{0}{}\u{0}{}",
        contents.feature_tool,
        contents.feature_type,
        contents.feature_prefix_dir,
        contents.feature_options
    )
}

/// A version 1 record of one workspace, and the files that hold it.
struct Record {
    old: WorkspaceContents,
    /// The workspace file and the files that can be read and hold it.
    named_by: Vec<PathBuf>,
}

/// What the run does with a `.sfmr` or `.matches` file of the tree, as far as
/// the folders it names are concerned.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Naming {
    /// The run rewrites it, so the folders it names are converted.
    Rewritten,
    /// The current reader refuses it, so the folders it names are kept.
    Refused,
    /// The run leaves it for another reason.
    Left,
}

/// What phase 1 read of a `.sfmr` or `.matches` file from its metadata and
/// image names alone: which workspace of the tree it is in, if any, and the
/// record it holds. `names` is `None` when the names entry does not read as
/// the format holds it.
enum Recorded {
    /// Its record resolves to the tree's workspace `workspace`: its version,
    /// its record, whether its observations name `.sift` files, and for a
    /// cluster `.matches` file below version 6, which the reader refuses,
    /// `true`.
    Here {
        workspace: usize,
        version: u32,
        contents: WorkspaceContents,
        names: Option<Vec<String>>,
        names_sift_files: bool,
        unreadable_clusters: bool,
    },
    /// Its record resolves to the workspace at `dir`, outside the tree.
    Elsewhere {
        dir: PathBuf,
        contents: WorkspaceContents,
        names: Option<Vec<String>>,
    },
    Leave(UpgradeOutcome),
}

/// The index of the workspace of the tree at `dir`, by real path.
fn workspace_index(dir: &Path, canonical: &[PathBuf]) -> Option<usize> {
    let real = std::fs::canonicalize(dir).ok()?;
    canonical.iter().position(|c| *c == real)
}

fn elsewhere(dir: &Path) -> UpgradeOutcome {
    UpgradeOutcome::NotConverted(format!(
        "it records the workspace {}, which is outside this tree, so that workspace's upgrade \
         converts it",
        dir.display()
    ))
}

fn no_workspace() -> UpgradeOutcome {
    UpgradeOutcome::NotConverted("it records no workspace that can be found".to_string())
}

fn cannot_read(e: impl std::fmt::Display) -> Recorded {
    Recorded::Leave(UpgradeOutcome::Failed(format!("it cannot be read: {e}")))
}

fn read_sfmr_record(path: &Path, canonical: &[PathBuf]) -> Recorded {
    // The names entry is read with the metadata, by the cheap reader; a file
    // the reader would refuse whole still names its images there. A file the
    // file system will not give up fails, wherever its record resolves.
    let (metadata, names) = match sfmtool_sfmr_format::read_sfmr_image_names(path) {
        Ok((metadata, names)) => (metadata, Some(names)),
        Err(e) if is_file_system_error(&e) => return cannot_read(e),
        Err(_) => match sfmtool_sfmr_format::read_sfmr_metadata(path) {
            Ok(metadata) => (metadata, None),
            Err(e) => return cannot_read(e),
        },
    };
    match sfmtool_sfmr_format::resolve_workspace_dir(path, &metadata) {
        Ok(dir) => match workspace_index(&dir, canonical) {
            Some(workspace) => Recorded::Here {
                workspace,
                version: metadata.version,
                names_sift_files: metadata.feature_source
                    == sfmtool_sfmr_format::FEATURE_SOURCE_SIFT_FILES,
                contents: metadata.workspace.contents,
                names,
                unreadable_clusters: false,
            },
            None => Recorded::Elsewhere {
                dir,
                contents: metadata.workspace.contents,
                names,
            },
        },
        Err(_) => Recorded::Leave(no_workspace()),
    }
}

fn read_matches_record(path: &Path, canonical: &[PathBuf]) -> Recorded {
    let (metadata, names) = match sfmtool_matches_format::read_matches_image_names(path) {
        Ok((metadata, names)) => (metadata, Some(names)),
        Err(e) if is_file_system_error(&e) => return cannot_read(e),
        Err(_) => match sfmtool_matches_format::read_matches_metadata(path) {
            Ok(metadata) => (metadata, None),
            Err(e) => return cannot_read(e),
        },
    };
    match sfmtool_matches_format::resolve_workspace_dir(path, &metadata) {
        Some(dir) => match workspace_index(&dir, canonical) {
            Some(workspace) => Recorded::Here {
                workspace,
                version: metadata.version,
                unreadable_clusters: metadata.has_clusters && metadata.version < 6,
                contents: metadata.workspace.contents,
                names,
                names_sift_files: true,
            },
            None => Recorded::Elsewhere {
                dir,
                contents: metadata.workspace.contents,
                names,
            },
        },
        None => Recorded::Leave(no_workspace()),
    }
}

/// What reading a whole `.sfmr` or `.matches` file gave.
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

/// Read a workspace file: the version 2 file to write in its place, or
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

/// The directory `folder` is in when it is `<image_dir>/<prefix>` below
/// `workspace`, or `None` when it is not.
fn image_dir_of(folder: &Path, workspace: &Path, prefix: &str) -> Option<PathBuf> {
    let relative = folder.strip_prefix(workspace).ok()?;
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
            .fold(workspace.to_path_buf(), |dir, part| dir.join(part)),
    ))
}

/// One spelling for each path of the tree, so that two spellings of one
/// directory or file on disk (`Images` and `images` on a file system that
/// ignores case, a prefix typed as `Features/...`) are one key.
///
/// A path that exists and whose real path ([`std::fs::canonicalize`], which
/// gives each name as it is on disk) lies in the tree is spelled as the root
/// was given followed by the real path below the root's real path, which is
/// how the walk spells the directories and files it lists. Any other path,
/// one that does not exist or lies outside the tree, keeps its spelling with
/// `..` resolved by name: a path that does not exist cannot collide with one
/// that does, and the run deletes nothing outside the tree. Spellings are
/// cached, since every image name of every file is spelled.
struct TreePaths<'a> {
    root: &'a Path,
    root_canonical: &'a Path,
    cache: std::cell::RefCell<HashMap<PathBuf, PathBuf>>,
}

impl<'a> TreePaths<'a> {
    fn new(root: &'a Path, root_canonical: &'a Path) -> Self {
        Self {
            root,
            root_canonical,
            cache: Default::default(),
        }
    }

    /// The tree's one spelling of `path`.
    fn spell(&self, path: &Path) -> PathBuf {
        let lexical = normalized(&lexically_resolved(path));
        if let Some(spelled) = self.cache.borrow().get(&lexical) {
            return spelled.clone();
        }
        let spelled = match std::fs::canonicalize(&lexical) {
            Ok(real) => match real.strip_prefix(self.root_canonical) {
                Ok(below) => normalized(&self.root.join(below)),
                Err(_) => lexical.clone(),
            },
            Err(_) => lexical.clone(),
        };
        self.cache.borrow_mut().insert(lexical, spelled.clone());
        spelled
    }

    /// Whether `path` lies in the tree: by its real path when it exists, and
    /// otherwise by its spelling with `..` resolved.
    fn is_inside(&self, path: &Path) -> bool {
        match std::fs::canonicalize(path) {
            Ok(real) => real.starts_with(self.root_canonical),
            Err(_) => normalized(&lexically_resolved(path))
                .starts_with(normalized(&lexically_resolved(self.root))),
        }
    }

    /// The directory `folder` (spelled by [`spell`](Self::spell)) is in when
    /// it is `<image_dir>/<prefix>` below `workspace`, the prefix matched by
    /// what it names on disk rather than by its spelling, or `None` when it
    /// is not.
    fn image_dir_of(&self, folder: &Path, workspace: &Path, prefix: &str) -> Option<PathBuf> {
        let count = Path::new(prefix)
            .components()
            .filter(|c| matches!(c, Component::Normal(_)))
            .count();
        let relative = folder.strip_prefix(workspace).ok()?;
        let parts: Vec<_> = relative.components().collect();
        if count == 0 || parts.len() < count {
            return None;
        }
        let image_dir = normalized(
            &parts[..parts.len() - count]
                .iter()
                .fold(workspace.to_path_buf(), |dir, part| dir.join(part)),
        );
        if let Some(found) = image_dir_of(folder, workspace, prefix) {
            return Some(found);
        }
        (self.spell(&image_dir.join(prefix)) == folder).then_some(image_dir)
    }
}

/// The real path of `folder` when it lies outside the tree at
/// `root_canonical`, reached through a link.
fn outside_tree(folder: &Path, root_canonical: &Path) -> Option<PathBuf> {
    let real = std::fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf());
    (!real.starts_with(root_canonical)).then_some(real)
}

/// The converted record of the folder a version 1 record of a workspace
/// names when the folder's `.sift` files decide it: the tool's workspace
/// `feature_type` and the files' own options, converted.
fn record_from_sift_files(
    folder: &Path,
    files: &[ScannedSiftFile],
) -> Result<WorkspaceContents, UpgradeWorkspaceError> {
    let first = &files[0];
    for other in &files[1..] {
        if other.feature_tool.feature_tool != first.feature_tool.feature_tool
            || other.feature_tool.feature_options != first.feature_tool.feature_options
        {
            return Err(UpgradeWorkspaceError::FolderFeatureOptions {
                folder: folder.to_path_buf(),
                first: first.path.clone(),
                second: other.path.clone(),
            });
        }
    }
    let tool = &first.feature_tool.feature_tool;
    let options =
        upgrade_feature_options(tool, &first.feature_tool.feature_options).map_err(|error| {
            UpgradeWorkspaceError::FeatureOptions {
                files: vec![first.path.clone()],
                error,
            }
        })?;
    let feature_type = feature_type_for_tool(tool, &options);
    Ok(WorkspaceContents {
        feature_prefix_dir: feature_prefix_dir(tool, &feature_type, &options),
        feature_tool: tool.clone(),
        feature_type,
        feature_options: options,
    })
}

/// Run phases 1 and 2 on the tree of workspaces at `root`, whose canonical
/// path is `root_canonical`. Every `.sfmr` and `.matches` file that the run
/// would rewrite is read here, and the metadata of every `.sift` file in a
/// folder it examines, so that a file the reader refuses, a file that cannot
/// be read and a file whose `.sift` files are not on disk are known before
/// anything is written.
pub(super) fn plan(
    root: &Path,
    root_canonical: &Path,
    progress: &Progress<'_>,
) -> Result<Plan, UpgradeWorkspaceError> {
    let found = {
        let phase = progress.phase("find files");
        walk(root, &phase)?
    };
    let phase = progress.phase("plan");

    // The workspaces of the tree and their workspace files.
    let mut workspace_dirs = vec![root.to_path_buf()];
    workspace_dirs.extend(found.nested_workspaces.iter().cloned());
    let mut workspaces = Vec::with_capacity(workspace_dirs.len());
    let mut workspace_records = Vec::with_capacity(workspace_dirs.len());
    for dir in &workspace_dirs {
        let file = dir.join(WORKSPACE_FILE_NAME);
        let (new_file, record) = read_workspace_file(&file)?;
        workspaces.push(WorkspacePlan {
            dir: dir.clone(),
            file,
            new_file,
        });
        workspace_records.push(record);
    }
    let canonical: Vec<PathBuf> = workspace_dirs
        .iter()
        .map(|d| std::fs::canonicalize(d).unwrap_or_else(|_| d.clone()))
        .collect();

    // Each `.sfmr` and `.matches` file: which workspace of the tree it is
    // in, with the image names it records, and what reading it whole gives
    // for those the run would rewrite.
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
                UpgradeFileKind::Sfmr => read_sfmr_record(path, &canonical),
                _ => read_matches_record(path, &canonical),
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

    // The folders each file names, found from its record and image names
    // alone, by their paths on disk, so that a file the reader refuses names
    // them too, each with the directory of the images it names it for. An
    // image name's `..` is resolved by name, as the root is spelled. A folder
    // inside the tree that a file of a workspace outside the tree names is
    // kept whatever else names it, and so is one that a file of the tree
    // names when another of its image names leaves the tree's root through
    // `..`: the run does not convert that file, since it does not convert
    // the folders outside the tree it also names.
    // Every folder is keyed by the tree's one spelling of it, so that a file
    // naming `Images/<image>` and the walk finding `images` name one folder.
    let paths = TreePaths::new(root, root_canonical);
    let mut sift_dirs: BTreeSet<PathBuf> = found.sift_dirs.iter().cloned().collect();
    let mut kept_for: HashMap<PathBuf, String> = HashMap::new();
    let mut file_dirs: Vec<Vec<(PathBuf, PathBuf)>> = Vec::with_capacity(read.len());
    let mut leaves_tree: Vec<Option<PathBuf>> = Vec::with_capacity(read.len());
    for ((_, path), (recorded, _)) in candidates.iter().zip(&read) {
        let (workspace, record, names, from_outside) = match recorded {
            Recorded::Here {
                workspace,
                contents,
                names,
                ..
            } => (workspace_dirs[*workspace].clone(), contents, names, false),
            Recorded::Elsewhere {
                dir,
                contents,
                names,
            } => (dir.clone(), contents, names, true),
            Recorded::Leave(_) => {
                file_dirs.push(Vec::new());
                leaves_tree.push(None);
                continue;
            }
        };
        let mut dirs = BTreeMap::new();
        let mut outside = None;
        if !has_image_to_gray(&record.feature_options)
            && is_plain_relative(Path::new(&record.feature_prefix_dir))
        {
            for name in names.iter().flatten() {
                let image_dir = image_dir_of_name(&workspace, name);
                let parent = Path::new(name).parent().unwrap_or(Path::new(""));
                if !from_outside && !is_plain_relative(parent) && !paths.is_inside(&image_dir) {
                    outside.get_or_insert(image_dir);
                    continue;
                }
                let image_dir = paths.spell(&image_dir);
                let dir = paths.spell(&image_dir.join(&record.feature_prefix_dir));
                if !dir.is_dir() {
                    continue;
                }
                if from_outside && outside_tree(&dir, root_canonical).is_some() {
                    continue;
                }
                dirs.entry(dir).or_insert(image_dir);
            }
        }
        if from_outside {
            for dir in dirs.keys() {
                kept_for.entry(dir.clone()).or_insert_with(|| {
                    format!(
                        "kept: named by a file of the workspace {}, outside this tree, whose \
                         other files this run cannot see",
                        workspace.display()
                    )
                });
            }
        } else if let Some(outside) = &outside {
            for dir in dirs.keys() {
                kept_for.entry(dir.clone()).or_insert_with(|| {
                    format!(
                        "kept: named by {}, which also names images in {}, outside this tree, \
                         and is not converted",
                        path.display(),
                        outside.display()
                    )
                });
            }
        }
        sift_dirs.extend(dirs.keys().cloned());
        file_dirs.push(dirs.into_iter().collect());
        leaves_tree.push(outside);
    }

    // The version 1 `.sift` files of each of those folders, by the rule every
    // writer of version 2 files follows. A `.sift` file the file system will
    // not give up (locked, or a placeholder that cannot be fetched) could
    // hold features a file needs, so it stops the run here.
    let scans: Vec<_> = sift_dirs
        .par_iter()
        .map(|dir| (dir, scan_sift_folder(dir)))
        .collect();
    let mut version_1_folders: Vec<(PathBuf, Vec<ScannedSiftFile>)> = Vec::new();
    let mut version_1_hashes: HashMap<PathBuf, [u8; 16]> = HashMap::new();
    let mut rejected: BTreeSet<PathBuf> = BTreeSet::new();
    let mut unreadable = Vec::new();
    for (dir, scan) in scans {
        let scan = scan.map_err(|e| UpgradeWorkspaceError::ListFolder {
            folder: dir.clone(),
            problem: e.to_string(),
        })?;
        for (path, error) in scan.unreadable {
            if is_file_system_error(&error) {
                unreadable.push(SiftFileFailure {
                    sift_path: path,
                    problem: error.to_string(),
                });
            } else {
                rejected.insert(normalized(&path));
            }
        }
        let mut files = Vec::with_capacity(scan.version_1.len());
        for file in scan.version_1 {
            // A content hash that is not 32 hex digits is the content's
            // fault, like any other the reader rejects.
            match parse_hash_bytes(&file.content_hash.content_xxh128) {
                Ok(hash) => {
                    version_1_hashes.insert(normalized(&file.path), hash);
                    files.push(file);
                }
                Err(_) => {
                    rejected.insert(normalized(&file.path));
                }
            }
        }
        if !files.is_empty() {
            version_1_folders.push((dir.clone(), files));
        }
    }
    if !unreadable.is_empty() {
        return Err(UpgradeWorkspaceError::UnreadableSiftFiles(unreadable));
    }
    phase.check_cancel()?;

    // Why a readable file that names a version 1 folder cannot be
    // converted: a `.sift` file it records is not on disk, is not readable as
    // a `.sift` file, or is not the version 1 file it records.
    let missing =
        |workspace: &Path, names: &[String], hashes: &[[u8; 16]], prefix: &str| -> Option<String> {
            for (name, hash) in names.iter().zip(hashes) {
                // Spelled as the scan spells it only when the name's spelling
                // differs from the disk's, which saves a lookup per file.
                let path = sift_path(workspace, name, prefix);
                let path = if version_1_hashes.contains_key(&path) || rejected.contains(&path) {
                    path
                } else {
                    paths.spell(&path)
                };
                if version_1_hashes.get(&path) == Some(hash) {
                    continue;
                }
                return Some(if rejected.contains(&path) {
                    format!(
                        "the .sift file {} it records for {name} cannot be read as the version 1 \
                     file it records",
                        path.display()
                    )
                } else if path.is_file() {
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
        .zip(&leaves_tree)
        .map(
            |((recorded, contents), outside)| match (recorded, contents) {
                (Recorded::Here { .. }, Some(Contents::Read { .. })) if outside.is_some() => {
                    outside.as_ref().map(|outside| {
                        format!(
                            "it names images in {}, outside this tree, whose .sift files this run \
                         does not convert; the old folders it names in the tree are kept",
                            outside.display()
                        )
                    })
                }
                (
                    Recorded::Here {
                        workspace,
                        contents: record,
                        names_sift_files: true,
                        ..
                    },
                    Some(Contents::Read { names, hashes }),
                ) if !has_image_to_gray(&record.feature_options) => missing(
                    &workspace_dirs[*workspace],
                    names,
                    hashes.as_deref().unwrap_or_default(),
                    &record.feature_prefix_dir,
                ),
                _ => None,
            },
        )
        .collect();

    // The records: each version 1 workspace file's, and each one a file of
    // the tree holds, by the workspace it belongs to.
    // What the run does with each file, as far as the folders it names are
    // concerned. A cluster file the reader refuses names its folders like
    // any file the run does not rewrite, and so does one that cannot be read
    // now: neither may lose its folder.
    let naming_of: Vec<Naming> = read
        .iter()
        .zip(&reasons)
        .map(
            |((recorded, contents), reason)| match (recorded, contents) {
                (Recorded::Here { contents, .. }, Some(Contents::Read { .. }))
                    if reason.is_none() && !has_image_to_gray(&contents.feature_options) =>
                {
                    Naming::Rewritten
                }
                (Recorded::Here { .. }, Some(Contents::Refused(_))) => Naming::Refused,
                _ => Naming::Left,
            },
        )
        .collect();

    let mut records: BTreeMap<String, Record> = BTreeMap::new();
    let mut note_record = |workspace: usize, contents: &WorkspaceContents, path: &Path| {
        if has_image_to_gray(&contents.feature_options) {
            return;
        }
        records
            .entry(record_key(workspace, contents))
            .or_insert_with(|| Record {
                old: contents.clone(),
                named_by: Vec::new(),
            })
            .named_by
            .push(path.to_path_buf());
    };
    for (i, (record, workspace)) in workspace_records.iter().zip(&workspaces).enumerate() {
        if let Some(contents) = record {
            note_record(i, contents, &workspace.file);
        }
    }
    for (((_, path), (recorded, _)), naming) in candidates.iter().zip(&read).zip(&naming_of) {
        if let Recorded::Here {
            workspace,
            contents: record,
            ..
        } = recorded
        {
            // A file the reader refuses is not converted, so its record need
            // not convert.
            if *naming != Naming::Refused {
                note_record(*workspace, record, path);
            }
        }
    }

    // Phase 2, first the records: one that cannot be converted, or whose
    // folder would not stay below each image's directory, stops the run
    // here, before anything is written.
    for record in records.values() {
        let old = &record.old;
        let mut files = record.named_by.clone();
        files.sort();
        if !is_plain_relative(Path::new(&old.feature_prefix_dir))
            || old.feature_prefix_dir.is_empty()
        {
            return Err(UpgradeWorkspaceError::FeaturePrefixDir {
                files,
                feature_prefix_dir: old.feature_prefix_dir.clone(),
            });
        }
        upgrade_feature_options(&old.feature_tool, &old.feature_options)
            .map_err(|error| UpgradeWorkspaceError::FeatureOptions { files, error })?;
    }
    // Each version 1 workspace file's converted record, by
    // `upgrade_feature_options` and the one folder-hash implementation.
    let mut workspace_new: Vec<Option<WorkspaceContents>> = Vec::with_capacity(workspaces.len());
    for (record, workspace) in workspace_records.iter().zip(&workspaces) {
        workspace_new.push(match record {
            None => None,
            Some(old) => {
                let options = upgrade_feature_options(&old.feature_tool, &old.feature_options)
                    .map_err(|error| UpgradeWorkspaceError::FeatureOptions {
                        files: vec![workspace.file.clone()],
                        error,
                    })?;
                Some(WorkspaceContents {
                    feature_prefix_dir: feature_prefix_dir(
                        &old.feature_tool,
                        &old.feature_type,
                        &options,
                    ),
                    feature_tool: old.feature_tool.clone(),
                    feature_type: old.feature_type.clone(),
                    feature_options: options,
                })
            }
        });
    }

    // Then the folders: each physical old folder maps to exactly one new
    // folder and one converted record. When a version 1 workspace file names
    // the folder, the deepest such workspace's converted record decides it,
    // as `sfm ws upgrade` names it; otherwise the folder's own `.sift` files
    // do, which must agree. It is converted when a version 1 workspace file
    // or a file the run rewrites names it; kept when a file the reader
    // refuses, a file of a workspace outside the tree names it, or the tree
    // holds a link; and deleted otherwise, an orphan included, unless it lies
    // outside the tree.
    let mut folders = Vec::new();
    let mut targets: HashMap<PathBuf, WorkspaceContents> = HashMap::new();
    let mut new_folders: Vec<PathBuf> = Vec::new();
    let mut new_folder_index: HashMap<PathBuf, usize> = HashMap::new();
    let mut destinations: BTreeMap<PathBuf, (PathBuf, [u8; 16])> = BTreeMap::new();
    let mut conflicts = Vec::new();
    let mut jobs = Vec::new();
    let mut deletion_refused = None;
    // The files of the tree that name each folder, with the directory of the
    // images they name it for. A file whose image names cannot be read names
    // the folders its record's prefix names below its workspace instead, so
    // that a file the reader refuses keeps them all the same.
    let mut named_by_files: HashMap<&Path, Vec<(Naming, &Path)>> = HashMap::new();
    let mut named_by_prefix: Vec<(&Path, &str, Naming)> = Vec::new();
    for (((recorded, _), dirs), naming) in read.iter().zip(&file_dirs).zip(&naming_of) {
        let Recorded::Here {
            workspace,
            contents,
            names,
            ..
        } = recorded
        else {
            continue;
        };
        if names.is_none() {
            named_by_prefix.push((
                &workspace_dirs[*workspace],
                &contents.feature_prefix_dir,
                *naming,
            ));
        }
        for (dir, image_dir) in dirs {
            named_by_files
                .entry(dir)
                .or_default()
                .push((*naming, image_dir));
        }
    }
    // The deepest workspace of the tree that `dir` lies in.
    let deepest_workspace = |dir: &Path| -> Option<usize> {
        workspace_dirs
            .iter()
            .enumerate()
            .filter(|(_, w)| dir.starts_with(w))
            .max_by_key(|(_, w)| w.components().count())
            .map(|(i, _)| i)
    };

    for (dir, sift_files) in version_1_folders {
        let mut naming: Vec<(Naming, PathBuf)> = named_by_files
            .get(dir.as_path())
            .map(|v| v.iter().map(|(n, d)| (*n, d.to_path_buf())).collect())
            .unwrap_or_default();
        for (workspace, prefix, n) in &named_by_prefix {
            if let Some(image_dir) = paths.image_dir_of(&dir, workspace, prefix) {
                naming.push((*n, image_dir));
            }
        }
        // A workspace file names the folder when its workspace is the deepest
        // one the folder lies in, the folder is its record's prefix below a
        // directory of that workspace, and that directory is where the
        // images of the files that name the folder are, if any do.
        let by_workspace = deepest_workspace(&dir).and_then(|i| {
            let old = workspace_records[i].as_ref()?;
            let new = workspace_new[i].as_ref()?;
            let image_dir =
                paths.image_dir_of(&dir, &workspaces[i].dir, &old.feature_prefix_dir)?;
            (naming.is_empty() || naming.iter().any(|(_, d)| *d == image_dir))
                .then(|| (image_dir, new.clone()))
        });
        let refused_by = naming.iter().filter(|(n, _)| *n == Naming::Refused).count();
        // Converted only for a version 1 workspace file or a file this run
        // rewrites: a file it leaves cannot read the new folder.
        let needed = by_workspace.is_some() || naming.iter().any(|(n, _)| *n == Naming::Rewritten);
        let image_dir = by_workspace
            .as_ref()
            .map(|(d, _)| d.clone())
            .or_else(|| naming.iter().map(|(_, d)| d.clone()).min());
        let kept_because = if refused_by > 0 {
            Some(format!(
                "kept: named by {refused_by} file(s) the current reader refuses"
            ))
        } else if let Some(reason) = kept_for.get(&dir) {
            Some(reason.clone())
        } else if let Some(real) = outside_tree(&dir, root_canonical) {
            deletion_refused.get_or_insert_with(|| {
                format!(
                    "the old folder {} lies outside the tree, at {}, so nothing was deleted; \
                     delete the old folders by hand once you have checked them",
                    dir.display(),
                    real.display()
                )
            });
            Some(format!("it lies outside the tree, at {}", real.display()))
        } else if !found.linked_directories.is_empty() {
            deletion_refused.get_or_insert_with(|| {
                "the tree holds linked directories, whose files this run does not examine, so \
                 no old folder was deleted; delete them by hand once you have checked that no \
                 file behind a link needs them"
                    .to_string()
            });
            Some("kept: the tree holds linked directories, whose files are not examined".into())
        } else {
            None
        };
        let (Some(image_dir), true) = (image_dir.clone(), needed) else {
            folders.push(OldFolder {
                path: dir,
                new_folder: None,
                image_dir,
                sift_files: sift_files.into_iter().map(|f| f.path).collect(),
                kept_because,
            });
            continue;
        };
        let target = match by_workspace {
            Some((_, new)) => new,
            None => record_from_sift_files(&dir, &sift_files)?,
        };
        let new_folder = paths.spell(&image_dir.join(&target.feature_prefix_dir));
        targets.insert(dir.clone(), target);
        let index = *new_folder_index
            .entry(new_folder.clone())
            .or_insert_with(|| {
                new_folders.push(new_folder.clone());
                new_folders.len() - 1
            });
        for file in &sift_files {
            let source = &file.path;
            let Some(&hash) = version_1_hashes.get(&normalized(source)) else {
                continue;
            };
            let file_name = source.file_name().unwrap_or_default();
            let destination = new_folder.join(file_name);
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
            image_dir: Some(image_dir),
            sift_files: sift_files.into_iter().map(|f| f.path).collect(),
            kept_because,
        });
    }
    if !conflicts.is_empty() {
        return Err(UpgradeWorkspaceError::Conflicts(conflicts));
    }

    // Each version 1 workspace file's version 2 form.
    for (workspace, new) in workspaces.iter_mut().zip(&workspace_new) {
        let (Some(config), Some(new)) = (workspace.new_file.as_mut(), new) else {
            continue;
        };
        if let Some(map) = config.as_object_mut() {
            map.insert("version".into(), crate::workspace::WORKSPACE_VERSION.into());
            map.insert("feature_options".into(), new.feature_options.clone());
            map.insert(
                "feature_prefix_dir".into(),
                new.feature_prefix_dir.clone().into(),
            );
        }
    }

    // The converted record a file that names version 1 folders takes: the
    // one record of the folders it names. With no folder on disk, it is its
    // workspace file's converted record when that names the same folder
    // prefix, and otherwise its own record converted, under the tool's
    // workspace `feature_type`.
    let converted_record = |file: &Path,
                            workspace: usize,
                            record: &WorkspaceContents,
                            dirs: &[(PathBuf, PathBuf)]|
     -> Result<WorkspaceContents, UpgradeWorkspaceError> {
        let mut found: Vec<(&PathBuf, &WorkspaceContents)> = dirs
            .iter()
            .filter_map(|(d, _)| targets.get(d).map(|t| (d, t)))
            .collect();
        let mut seen = BTreeSet::new();
        found.retain(|(_, t)| seen.insert(record_key(0, t)));
        match found.as_slice() {
            [(_, target)] => Ok((*target).clone()),
            [] => {
                if let (Some(old), Some(new)) =
                    (&workspace_records[workspace], &workspace_new[workspace])
                {
                    if old.feature_prefix_dir == record.feature_prefix_dir {
                        return Ok(new.clone());
                    }
                }
                let options =
                    upgrade_feature_options(&record.feature_tool, &record.feature_options)
                        .map_err(|error| UpgradeWorkspaceError::FeatureOptions {
                            files: vec![file.to_path_buf()],
                            error,
                        })?;
                let feature_type = feature_type_for_tool(&record.feature_tool, &options);
                Ok(WorkspaceContents {
                    feature_prefix_dir: feature_prefix_dir(
                        &record.feature_tool,
                        &feature_type,
                        &options,
                    ),
                    feature_tool: record.feature_tool.clone(),
                    feature_type,
                    feature_options: options,
                })
            }
            _ => Err(UpgradeWorkspaceError::FolderTargets {
                file: file.to_path_buf(),
                folders: found.iter().map(|(d, _)| (*d).clone()).collect(),
            }),
        }
    };

    // What the run does to each file.
    let mut files = Vec::new();
    for ((((kind, path), (recorded, contents)), reason), dirs) in
        candidates.iter().zip(read).zip(reasons).zip(&file_dirs)
    {
        let action = match (recorded, contents) {
            (Recorded::Leave(outcome), _) => Action::Leave(outcome),
            (Recorded::Elsewhere { dir, .. }, _) => Action::Leave(elsewhere(&dir)),
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
                    workspace,
                    contents: record,
                    names_sift_files,
                    ..
                },
                Some(Contents::Read { .. }),
            ) => match reason {
                Some(reason) => Action::Leave(UpgradeOutcome::NotConverted(reason)),
                None => {
                    let new_contents = if has_image_to_gray(&record.feature_options) {
                        None
                    } else {
                        Some(converted_record(path, workspace, &record, dirs)?)
                    };
                    Action::Rewrite {
                        workspace: workspace_dirs[workspace].clone(),
                        names_sift_files: names_sift_files && new_contents.is_some(),
                        new_contents,
                    }
                }
            },
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
                workspace: root.to_path_buf(),
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
                    .is_some_and(|dir| workspace_index(&dir, &canonical).is_some()) =>
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
        workspaces,
        folders,
        new_folders,
        jobs,
        files,
        linked_directories: found.linked_directories,
        deletion_refused,
    })
}
