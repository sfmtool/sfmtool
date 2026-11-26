// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The notice is given once for two large photographs of a workspace without
//! `mipmap_options`, and not for a workspace with the object. The
//! photographs are 1537 x 1 PNGs, since a layout read reads only the header.

use std::path::{Path, PathBuf};

use sfmtool_core::workspace::WORKSPACE_FILE_NAME;

use super::PHOTOGRAPH_TIER_NOTICE;
use crate::action_log::{Actor, Kind};
use crate::state::AppState;

/// A temporary workspace whose file is `text`, holding two 1537 x 1 PNGs.
fn workspace(text: &str) -> (tempfile::TempDir, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(WORKSPACE_FILE_NAME), text).unwrap();
    let paths = ["a.png", "b.png"]
        .iter()
        .map(|name| {
            let path = dir.path().join(name);
            image::RgbImage::new(1537, 1).save(&path).unwrap();
            path
        })
        .collect();
    (dir, paths)
}

/// Read each photograph's layout as a panel does, giving the notice after
/// each read as a frame would, and answer the rows the notice wrote.
fn notices_after_reading(paths: &[PathBuf]) -> Vec<(Actor, Kind, String)> {
    let mut state = AppState::new();
    state.action_log.clear();
    for path in paths {
        state.photographs.request_layout(Path::new(path));
        state.give_photograph_tier_notice();
        state.give_photograph_tier_notice();
    }
    state
        .action_log
        .entries()
        .filter(|entry| entry.text == PHOTOGRAPH_TIER_NOTICE)
        .map(|entry| (entry.actor, entry.kind, entry.text.clone()))
        .collect()
}

#[test]
fn the_notice_is_given_once_for_two_large_photographs_of_a_workspace_without_mipmap_options() {
    let (_dir, paths) = workspace(r#"{"version": 2}"#);
    assert_eq!(
        notices_after_reading(&paths),
        [(
            Actor::Viewer,
            Kind::Session,
            PHOTOGRAPH_TIER_NOTICE.to_owned()
        )]
    );
}

#[test]
fn the_notice_is_not_given_for_a_workspace_with_mipmap_options() {
    let (_dir, paths) = workspace(
        r#"{"version": 2, "mipmap_options": {"out_of_core_long_side": 1536,
        "jpeg_quality": 95, "full_resolution_jpeg_quality": 95}}"#,
    );
    assert_eq!(notices_after_reading(&paths), []);
}
