// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The notice is given once for two large photographs of a workspace without
//! `mipmap_options`, and not for a workspace with the object. The
//! photographs are 1537 x 1 PNGs, since a layout read reads only the header,
//! except for the open's, which renders patch bitmaps from them.

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

/// A saved file with patch frames and no bitmaps over photographs 1537 px
/// wide, one more than a new workspace's threshold, in a workspace without
/// `mipmap_options`, so an open renders the bitmaps from them. The demo's
/// camera is widened to match, keeping its focal length and principal
/// point, so every point still projects where its keypoints are.
fn saved_with_large_photographs(dir: &Path) -> PathBuf {
    let mut recon = crate::state::edits::tests::projected_embedded_demo(12);
    recon.workspace_dir = dir.to_path_buf();
    for camera in &mut recon.image_set.cameras {
        camera.width = 1537;
    }
    let height = recon.image_set.cameras[0].height;
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("photo_{i:02}.png");
        image::RgbImage::from_fn(1537, height, |x, y| {
            image::Rgb([(x % 9 * 14) as u8, (y % 7 * 18) as u8, i as u8])
        })
        .save(dir.join(&image.name))
        .unwrap();
    }
    std::fs::write(dir.join(WORKSPACE_FILE_NAME), r#"{"version": 2}"#).unwrap();
    let path = dir.join("recon.sfmr");
    recon
        .save(&path, &sfmtool_core::progress::Progress::none())
        .unwrap();
    path
}

/// An open that renders the patch bitmaps decodes every photograph into the
/// cache, so a panel's layout read afterwards is answered from the cache and
/// reads no header: the notice still comes, from the open's own reads.
#[test]
fn the_notice_is_given_after_an_open_that_decodes_the_large_photographs() {
    let tmp = tempfile::tempdir().unwrap();
    let path = saved_with_large_photographs(tmp.path());

    let mut state = AppState::new();
    let id = state.open_now(&path).expect("the file opens");
    let opened = state.node(id).unwrap().recon();
    assert!(
        opened.point_set.patch_bitmaps_y_x_rgba.is_some(),
        "the open rendered the patch bitmaps"
    );
    state.give_photograph_tier_notice();
    let notices = state
        .action_log
        .entries()
        .filter(|entry| entry.text == PHOTOGRAPH_TIER_NOTICE)
        .count();
    assert_eq!(notices, 1);
}
