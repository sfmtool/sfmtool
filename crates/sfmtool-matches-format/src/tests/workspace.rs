// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How a `.matches` file's recorded workspace is found
//! (`resolve_workspace_dir`).

use std::path::{Path, PathBuf};

use super::*;

/// A temporary directory holding a workspace `ws` with the marker, inside it
/// a directory `ws/matches`, a second workspace `other` with the marker, and
/// a directory `plain` and a directory `out` without one. The `.matches`
/// files the tests name need not exist: the lookup reads only the path.
fn layout() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for sub in ["ws/matches", "other", "plain", "out"] {
        std::fs::create_dir_all(dir.path().join(sub)).unwrap();
    }
    for workspace in ["ws", "other"] {
        std::fs::write(dir.path().join(workspace).join(".sfm-workspace.json"), "{}").unwrap();
    }
    dir
}

/// The test fixture's metadata, recording the two given paths.
fn recorded(relative_path: &str, absolute_path: &Path) -> MatchesMetadata {
    let mut metadata = make_test_data().metadata;
    metadata.workspace.relative_path = relative_path.into();
    metadata.workspace.absolute_path = absolute_path.to_string_lossy().into_owned();
    metadata
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap()
}

/// A relative path that names a marked directory is the workspace, ahead of
/// an absolute path that names another marked one.
#[test]
fn the_relative_path_comes_first() {
    let dir = layout();
    let file = dir.path().join("out").join("x.matches");

    let found = resolve_workspace_dir(&file, &recorded("../ws", &dir.path().join("other")));

    assert_eq!(
        found.map(|f| canonical(&f)),
        Some(canonical(&dir.path().join("ws")))
    );
}

/// A relative path that names a directory without the marker is skipped, and
/// the absolute path is tried next.
#[test]
fn an_unmarked_relative_directory_falls_back_to_the_absolute_path() {
    let dir = layout();
    let file = dir.path().join("out").join("x.matches");
    let workspace = dir.path().join("other");

    let found = resolve_workspace_dir(&file, &recorded("../plain", &workspace));

    assert_eq!(found, Some(workspace));
}

/// With neither recorded directory holding the marker, the nearest marked
/// directory above the file is the workspace.
#[test]
fn unmarked_recorded_directories_fall_back_to_the_enclosing_workspace() {
    let dir = layout();
    let file = dir.path().join("ws").join("matches").join("x.matches");

    let found = resolve_workspace_dir(&file, &recorded("../../plain", &dir.path().join("plain")));

    assert_eq!(
        found.map(|f| canonical(&f)),
        Some(canonical(&dir.path().join("ws")))
    );
}

/// With neither recorded directory holding the marker and none above the
/// file, there is no workspace.
#[test]
fn no_marked_directory_is_no_workspace() {
    let dir = layout();
    let file = dir.path().join("out").join("x.matches");

    let found = resolve_workspace_dir(&file, &recorded("../plain", &dir.path().join("plain")));

    assert_eq!(found, None);
}

/// The search upward starts from the real location of the file's directory,
/// not from the path it was reached by: a file reached through a link whose
/// target is inside a workspace resolves to that workspace even though no
/// directory above the link holds the marker.
#[test]
fn the_upward_search_follows_a_linked_directory_to_its_target() {
    let dir = layout();
    let link = dir.path().join("out").join("link");
    let target = dir.path().join("ws").join("matches");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &link).unwrap();
    #[cfg(windows)]
    if let Err(error) = std::os::windows::fs::symlink_dir(&target, &link) {
        // Creating a symbolic link on Windows needs Developer Mode or an
        // administrator; without either there is nothing to test.
        eprintln!("skipped: cannot create a directory symlink: {error}");
        return;
    }
    let file = link.join("x.matches");

    let found = resolve_workspace_dir(&file, &recorded("", Path::new("")));

    assert_eq!(
        found.map(|f| canonical(&f)),
        Some(canonical(&dir.path().join("ws")))
    );
}
