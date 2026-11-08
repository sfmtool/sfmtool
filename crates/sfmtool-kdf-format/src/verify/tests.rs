// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How source verification finds the workspace a `.kdf` records.

use std::path::{Path, PathBuf};

use super::resolve_workspace;
use crate::{KdfError, KdfWorkspaceContents, KdfWorkspaceMetadata};

/// The workspace record of a `.kdf`, with the two paths given.
fn recorded(relative_path: &str, absolute_path: &Path) -> KdfWorkspaceMetadata {
    KdfWorkspaceMetadata {
        absolute_path: absolute_path.to_string_lossy().into_owned(),
        relative_path: relative_path.into(),
        contents: KdfWorkspaceContents {
            feature_tool: "test".into(),
            feature_type: "sift".into(),
            feature_options: serde_json::json!({}),
            feature_prefix_dir: "features/sift".into(),
        },
    }
}

/// `root/index/x.kdf` beside a directory `root/plain` that has no marker and a
/// workspace `root/ws` that has one. The `.kdf` itself need not exist: the
/// lookup reads only the path.
fn layout() -> (tempfile::TempDir, PathBuf, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let plain = dir.path().join("plain");
    let workspace = dir.path().join("ws");
    std::fs::create_dir_all(&plain).unwrap();
    std::fs::create_dir_all(dir.path().join("index")).unwrap();
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(workspace.join(".sfm-workspace.json"), "{}").unwrap();
    let kdf = dir.path().join("index").join("x.kdf");
    (dir, kdf, plain, workspace)
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap()
}

/// A recorded relative directory that exists but holds no marker is not the
/// workspace: the lookup goes on to the absolute path, which does.
#[test]
fn a_relative_directory_without_the_marker_is_skipped() {
    let (_dir, kdf, plain, workspace) = layout();
    assert!(plain.is_dir());

    let found = resolve_workspace(&kdf, &recorded("../plain", &workspace)).unwrap();

    assert_eq!(canonical(&found), canonical(&workspace));
}

/// A recorded relative directory that holds the marker is the workspace, ahead
/// of an absolute path that names another directory.
#[test]
fn a_relative_directory_with_the_marker_resolves() {
    let (_dir, kdf, plain, workspace) = layout();

    let found = resolve_workspace(&kdf, &recorded("../ws", &plain)).unwrap();

    assert_eq!(canonical(&found), canonical(&workspace));
}

/// With neither recorded directory holding the marker and none above the
/// `.kdf`, there is no workspace, and the error names the `.kdf`'s directory.
#[test]
fn directories_without_the_marker_resolve_to_no_workspace() {
    let (_dir, kdf, plain, _) = layout();

    let error = resolve_workspace(&kdf, &recorded("../plain", &plain)).unwrap_err();

    assert!(
        matches!(&error, KdfError::MissingSource(dir) if dir == kdf.parent().unwrap()),
        "{error}"
    );
}
