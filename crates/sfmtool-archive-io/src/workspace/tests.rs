// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The search upward from a path for the workspace marker.

use std::path::Path;

use super::{find_workspace_for_path, find_workspace_from, WORKSPACE_MARKER};

/// A temporary directory holding a workspace `ws` with an image
/// `ws/images/a.jpg` and a directory `ws/sub`, and a directory `out` with no
/// marker.
fn layout() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for sub in ["ws/images", "ws/sub", "out"] {
        std::fs::create_dir_all(dir.path().join(sub)).unwrap();
    }
    std::fs::write(dir.path().join("ws").join(WORKSPACE_MARKER), "{}").unwrap();
    std::fs::write(dir.path().join("ws/images/a.jpg"), "").unwrap();
    dir
}

/// A directory symlink at `link` to `target`, or `None` where the platform
/// does not let this process make one (Windows without Developer Mode or an
/// administrator).
fn symlink_dir(target: &Path, link: &Path) -> Option<()> {
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, link).unwrap();
    #[cfg(windows)]
    if let Err(error) = std::os::windows::fs::symlink_dir(target, link) {
        eprintln!("skipped: cannot create a directory symlink: {error}");
        return None;
    }
    Some(())
}

/// An existing file, a directory and a path that does not exist yet all find
/// the workspace around them, spelled as the absolute path given.
#[test]
fn a_file_a_directory_and_a_missing_path_find_the_enclosing_workspace() {
    let dir = layout();
    let ws = dir.path().join("ws");

    for path in [
        ws.join("images/a.jpg"),
        ws.join("sub"),
        ws.join("sfmr/new/out.sfmr"),
    ] {
        assert_eq!(find_workspace_for_path(&path), Some(ws.clone()), "{path:?}");
    }
}

/// With no marker at or above the path there is no workspace.
#[test]
fn a_path_with_no_marker_above_it_has_no_workspace() {
    let dir = layout();

    assert_eq!(
        find_workspace_for_path(&dir.path().join("out/x.matches")),
        None
    );
}

/// A relative path, a bare file name and the empty path are taken against the
/// current directory, and the search goes on above it.
#[test]
fn a_relative_path_is_searched_above_the_current_directory() {
    let dir = layout();
    let ws = dir.path().join("ws");
    let current_dir = ws.join("sub");

    for path in ["x.matches", "", ".", "deeper/x.sfmr"] {
        assert_eq!(
            find_workspace_from(Path::new(path), || Some(current_dir.clone())),
            Some(ws.clone()),
            "{path:?}"
        );
    }
}

/// A directory reached through a link belongs to the workspace around the
/// link, not to one around its target: a link inside `ws` to a directory with
/// no marker above it is in `ws`, and a link in `out` to a directory inside
/// `ws` is in no workspace.
#[test]
fn a_linked_directory_belongs_to_the_workspace_around_the_link() {
    let dir = layout();
    let ws = dir.path().join("ws");
    let outside = dir.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    let Some(()) = symlink_dir(&outside, &ws.join("linked")) else {
        return;
    };
    symlink_dir(&ws.join("images"), &dir.path().join("out/link")).unwrap();

    assert_eq!(
        find_workspace_for_path(&ws.join("linked/a.jpg")),
        Some(ws.clone())
    );
    assert_eq!(
        find_workspace_for_path(&dir.path().join("out/link/a.jpg")),
        None
    );
}

/// A `..` in the path stays as written: the walk goes up the path's text, so
/// the directory found is spelled as given, and an image path built from the
/// same `..` path is measured against it as `images/a.jpg`.
#[test]
fn a_path_with_parent_components_finds_the_ancestor_spelled_as_given() {
    let dir = layout();
    let ws = dir.path().join("ws");
    let image_dir = ws.join("sub").join("..").join("images");
    let expected = ws.join("sub").join("..");

    let found = find_workspace_for_path(&image_dir);

    assert_eq!(found.as_deref(), Some(expected.as_path()));
    assert_eq!(
        image_dir.join("a.jpg").strip_prefix(&expected).unwrap(),
        Path::new("images").join("a.jpg")
    );
}

/// The current directory is read only for a relative path: an absolute path is
/// searched where it cannot be read.
#[test]
fn an_absolute_path_needs_no_current_directory() {
    let dir = layout();
    let ws = dir.path().join("ws");

    assert_eq!(
        find_workspace_from(&ws.join("images/a.jpg"), || None),
        Some(ws.clone())
    );
    assert_eq!(find_workspace_from(Path::new("x.matches"), || None), None);
}
