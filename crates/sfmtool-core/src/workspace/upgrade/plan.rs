// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Phases 1 and 2 of the workspace upgrade: find the files and the version 1
//! features folders, and map each version 1 record to its new folder.
//!
//! Nothing here writes. A dry run is these two phases and the report they
//! give; a run goes on from the [`Plan`] to the phases in the parent module.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use rayon::prelude::*;
use sfmtool_archive_io::{parse_hash_bytes, WorkspaceContents};

use super::{
    SiftConflict, UpgradeFileKind, UpgradeOutcome, UpgradeWorkspaceError, WORKSPACE_FILE_NAME,
};
use crate::features::sift::version_1_sift_files;
use crate::progress::Progress;
use crate::workspace::{feature_prefix_dir, has_image_to_gray, upgrade_feature_options};

/// What a run does to one `.sfmr`, `.matches`, `.camrig` or `.kdf` file.
#[derive(Debug, Clone)]
pub(super) enum Action {
    /// Read it and write it back as the current version, at its own path.
    Rewrite {
        /// The converted record its `workspace.contents` becomes, when it
        /// records a version 1 one.
        new_contents: Option<WorkspaceContents>,
        /// The `feature_prefix_dir` of the record it holds now, from which
        /// its version 1 `.sift` files are found.
        old_prefix: String,
        /// Whether its observations name `.sift` files by content hash, which
        /// must be the version 1 files in an old folder for it to convert.
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

/// One version 1 features folder, and the new folder its files go to.
#[derive(Debug, Clone)]
pub(super) struct OldFolder {
    pub(super) path: PathBuf,
    /// `None` for an orphan, which no record names.
    pub(super) new_folder: Option<PathBuf>,
    /// The directory the record's `feature_prefix_dir` is relative to, where
    /// the images are; `None` for an orphan.
    pub(super) image_dir: Option<PathBuf>,
    /// Its version 1 `.sift` files, sorted.
    pub(super) sift_files: Vec<PathBuf>,
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
    /// Each version 1 `.sift` file found, by its path with separators made
    /// uniform ([`normalized`]), to its content hash.
    pub(super) version_1_hashes: HashMap<PathBuf, [u8; 16]>,
}

/// `path` with its components joined again, so two spellings of one path
/// (`a/b` and `a\b` on Windows, a trailing separator) compare and hash equal.
pub(super) fn normalized(path: &Path) -> PathBuf {
    path.components().collect()
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

/// Why the `.sift` files `names` and `hashes` record under `prefix` are not
/// all version 1 files of an old folder, naming the first that is not, or
/// `None` when they are.
pub(super) fn missing_sift_file(
    plan: &Plan,
    names: &[String],
    hashes: &[[u8; 16]],
    prefix: &str,
) -> Option<String> {
    for (name, hash) in names.iter().zip(hashes) {
        let path = sift_path(&plan.root, name, prefix);
        if plan.version_1_hashes.get(&path) != Some(hash) {
            return Some(format!(
                "the .sift file {} it records for {name} is missing, or is not the version 1 \
                 file it records",
                path.display()
            ));
        }
    }
    None
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
}

/// Walk the workspace at `root` for the files the upgrade reads, without
/// descending into a nested workspace, a directory below `root` that holds
/// its own `.sfm-workspace.json`.
fn walk(root: &Path, progress: &Progress<'_>) -> Result<Walk, UpgradeWorkspaceError> {
    let mut found = Walk::default();
    let mut visited = HashSet::new();
    let mut stack = vec![root.to_path_buf()];
    let mut listed = 0u64;
    while let Some(dir) = stack.pop() {
        progress.check_cancel()?;
        // A link to a directory is followed, as a workspace whose image folder
        // is a link to storage outside it holds those images; the real path
        // is remembered so a link back up the tree is not followed forever.
        if !visited.insert(std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone())) {
            continue;
        }
        let list_error = |e: std::io::Error| UpgradeWorkspaceError::ListFolder {
            folder: dir.clone(),
            problem: e.to_string(),
        };
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(list_error)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<_, _>>()
            .map_err(list_error)?;
        entries.sort();
        let mut holds_sift = false;
        for path in entries {
            if path.is_dir() {
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
            found.sift_dirs.push(dir.clone());
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
    named_by: Vec<PathBuf>,
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
    Ok((Some(config), Some(contents)))
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

/// Run phases 1 and 2 on the workspace at `root`, whose canonical path is
/// `root_canonical`. `check_sift_files` reads every `.sfmr` and `.matches`
/// file that records a version 1 folder to check its `.sift` files there,
/// which a dry run does to list those that cannot be converted; a run checks
/// them as it rewrites each.
pub(super) fn plan(
    root: &Path,
    root_canonical: &Path,
    check_sift_files: bool,
    progress: &Progress<'_>,
) -> Result<Plan, UpgradeWorkspaceError> {
    let workspace_file = root.join(WORKSPACE_FILE_NAME);
    let (new_workspace_file, workspace_record) = read_workspace_file(&workspace_file)?;

    let found = {
        let phase = progress.phase("find files");
        walk(root, &phase)?
    };

    let phase = progress.phase("plan");
    // The version 1 `.sift` files of every folder that holds `.sift` files,
    // by the check every writer of version 2 files makes.
    let listed: Vec<Result<Option<HashedFolder>, UpgradeWorkspaceError>> = found
        .sift_dirs
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
            // A file `version_1_sift_files` counted read as version 1, so
            // its hashes read too unless it changed since; one that no
            // longer reads keeps no hash and so matches no record.
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

    // The records: the workspace file's when it is version 1, and each one a
    // file of this workspace holds.
    let mut records: BTreeMap<String, Record> = BTreeMap::new();
    let mut note_record = |contents: &WorkspaceContents, path: &Path| {
        if has_image_to_gray(&contents.feature_options) {
            return;
        }
        records
            .entry(record_key(contents))
            .or_insert_with(|| Record {
                old: contents.clone(),
                new: None,
                named_by: Vec::new(),
            })
            .named_by
            .push(path.to_path_buf());
    };
    if let Some(contents) = &workspace_record {
        note_record(contents, &workspace_file);
    }
    let sfmr_records: Vec<Recorded> = found
        .sfmr
        .par_iter()
        .map(|p| read_sfmr_record(p, root_canonical))
        .collect();
    let matches_records: Vec<Recorded> = found
        .matches
        .par_iter()
        .map(|p| read_matches_record(p, root_canonical))
        .collect();
    for (path, recorded) in found
        .sfmr
        .iter()
        .zip(&sfmr_records)
        .chain(found.matches.iter().zip(&matches_records))
    {
        if let Recorded::Here { contents, .. } = recorded {
            note_record(contents, path);
        }
    }

    // Phase 2: each record's new folder, by the one folder-hash
    // implementation. A record that cannot be converted stops the run here,
    // before anything is written, since its folder could be neither
    // converted nor deleted.
    for record in records.values_mut() {
        let old = &record.old;
        let options =
            upgrade_feature_options(&old.feature_tool, &old.feature_options).map_err(|error| {
                UpgradeWorkspaceError::FeatureOptions {
                    files: record.named_by.clone(),
                    error,
                }
            })?;
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

    // Each old folder's new folder: the first record whose prefix it ends
    // with. A folder no record names is an orphan.
    let mut folders = Vec::new();
    let mut new_folders: Vec<PathBuf> = Vec::new();
    let mut new_folder_index: HashMap<PathBuf, usize> = HashMap::new();
    // The source of each new file, to find two old files that would both
    // become it.
    let mut destinations: BTreeMap<PathBuf, (PathBuf, [u8; 16])> = BTreeMap::new();
    let mut conflicts = Vec::new();
    let mut jobs = Vec::new();
    for (dir, sift_files) in version_1_folders {
        let named = records.values().find_map(|record| {
            let image_dir = image_dir_of(&dir, root, &record.old.feature_prefix_dir)?;
            let new = record.new.as_ref()?;
            Some((image_dir.clone(), image_dir.join(&new.feature_prefix_dir)))
        });
        let Some((image_dir, new_folder)) = named else {
            folders.push(OldFolder {
                path: dir,
                new_folder: None,
                image_dir: None,
                sift_files,
            });
            continue;
        };
        let new_folder = normalized(&new_folder);
        let index = *new_folder_index
            .entry(new_folder.clone())
            .or_insert_with(|| {
                new_folders.push(new_folder.clone());
                new_folders.len() - 1
            });
        for source in &sift_files {
            let file_name = source.file_name().unwrap_or_default();
            let destination = new_folder.join(file_name);
            let hash = version_1_hashes[&normalized(source)];
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
            sift_files,
        });
    }
    if !conflicts.is_empty() {
        return Err(UpgradeWorkspaceError::Conflicts(conflicts));
    }

    let mut plan = Plan {
        root: root.to_path_buf(),
        workspace_file,
        new_workspace_file: None,
        folders,
        new_folders,
        jobs,
        files: Vec::new(),
        nested_workspaces: found.nested_workspaces,
        version_1_hashes,
    };
    plan.new_workspace_file = match (new_workspace_file, &workspace_record) {
        (Some(mut config), Some(contents)) => {
            let new = new_record(contents).expect("every version 1 record was mapped");
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

    let mut files = Vec::new();
    for (kind, paths, recorded) in [
        (UpgradeFileKind::Sfmr, &found.sfmr, sfmr_records),
        (UpgradeFileKind::Matches, &found.matches, matches_records),
    ] {
        let current = match kind {
            UpgradeFileKind::Sfmr => sfmtool_sfmr_format::SFMR_FORMAT_VERSION,
            _ => sfmtool_matches_format::MATCHES_FORMAT_VERSION,
        };
        for (path, recorded) in paths.iter().zip(recorded) {
            let action = match recorded {
                Recorded::Leave(outcome) => Action::Leave(outcome),
                Recorded::Here {
                    unreadable_clusters: true,
                    ..
                } => Action::Leave(UpgradeOutcome::NotConverted(
                    "it is a cluster .matches file below version 6, which the reader refuses; \
                     regenerate it with 'sfm match --cluster'"
                        .into(),
                )),
                Recorded::Here {
                    version,
                    contents,
                    names_sift_files,
                    ..
                } => {
                    let new_contents = new_record(&contents);
                    if new_contents.is_none() && version >= current {
                        Action::Leave(UpgradeOutcome::AlreadyCurrent)
                    } else {
                        Action::Rewrite {
                            names_sift_files: names_sift_files && new_contents.is_some(),
                            new_contents,
                            old_prefix: contents.feature_prefix_dir,
                        }
                    }
                }
            };
            files.push(FilePlan {
                path: path.clone(),
                kind,
                action,
            });
        }
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
                old_prefix: String::new(),
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
    plan.files = files;

    if check_sift_files {
        let checked: Vec<Option<UpgradeOutcome>> = plan
            .files
            .par_iter()
            .map(|file| match &file.action {
                Action::Rewrite {
                    names_sift_files,
                    old_prefix,
                    ..
                } if file.kind != UpgradeFileKind::Camrig => {
                    match recorded_sift_files(&file.path, file.kind) {
                        Err(e) => Some(UpgradeOutcome::NotConverted(refused(&e))),
                        Ok(_) if !names_sift_files => None,
                        Ok((names, hashes)) => {
                            missing_sift_file(&plan, &names, &hashes, old_prefix)
                                .map(UpgradeOutcome::NotConverted)
                        }
                    }
                }
                _ => None,
            })
            .collect();
        for (file, outcome) in plan.files.iter_mut().zip(checked) {
            if let Some(outcome) = outcome {
                file.action = Action::Leave(outcome);
            }
        }
    }
    drop(phase);
    Ok(plan)
}

/// Why a file the current reader refuses is not converted: no reader can read
/// it through an old folder or any other, so it does not stop the deletion.
pub(super) fn refused(error: &str) -> String {
    format!("the current reader refuses it, so it is left as it is: {error}")
}

/// The image names and `.sift` content hashes a `.sfmr` or `.matches` file
/// records, after reading it as the run will: a `.sfmr` into a
/// reconstruction, which checks what the format reader leaves to it, such as
/// each camera's parameters.
fn recorded_sift_files(
    path: &Path,
    kind: UpgradeFileKind,
) -> Result<(Vec<String>, Vec<[u8; 16]>), String> {
    match kind {
        UpgradeFileKind::Sfmr => {
            let data = sfmtool_sfmr_format::read_sfmr(path).map_err(|e| e.to_string())?;
            let names = data.image_names.clone();
            let hashes = data.sift_content_hashes.clone().unwrap_or_default();
            crate::SfmrReconstruction::from_sfmr_data(data).map_err(|e| e.to_string())?;
            Ok((names, hashes))
        }
        _ => {
            let data = sfmtool_matches_format::read_matches(path).map_err(|e| e.to_string())?;
            Ok((data.image_names, data.sift_content_hashes))
        }
    }
}
