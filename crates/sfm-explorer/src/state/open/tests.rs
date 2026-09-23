// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The open as a background task: its phases, the columns it fills in for
//! display and where they live, what a save of the node then writes, and the
//! minimal copy.

use std::path::{Path, PathBuf};

use sfmtool_core::SfmrReconstruction;

use crate::display_thumbnails::tests::{
    colour_of, is_flat, sift_grey_of, workspace_with_photographs, write_sift_thumbnails,
};
use crate::display_thumbnails::PLACEHOLDER_GREY;
use crate::scene::{PointRef, ReconId};
use crate::state::AppState;
use crate::test_support::{phase_note, phase_rows};

/// Mark `dir` as a workspace and write `recon` into it as `name`.
fn save_into(dir: &Path, recon: &SfmrReconstruction, name: &str) -> PathBuf {
    std::fs::write(dir.join(".sfm-workspace.json"), "{}").unwrap();
    let path = dir.join(name);
    recon.save(&path).unwrap();
    path
}

/// A saved `embedded_patches` file with patch frames and inline keypoints but
/// neither thumbnails nor bitmaps, and a textured photograph of each camera's
/// size for every image, so an open has both columns to fill in.
///
/// Crate-visible: the background tests start the real open over it.
pub(crate) fn saved_with_photographs(dir: &Path) -> PathBuf {
    let mut recon = crate::state::edits::tests::projected_embedded_demo(12);
    recon.workspace_dir = dir.to_path_buf();
    std::fs::create_dir_all(dir.join("images")).unwrap();
    let camera = recon.image_table.cameras[0].clone();
    let (w, h) = (camera.width, camera.height);
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.name = format!("images/photo_{i:02}.png");
        // A pattern rather than a flat field, as the bench fixture's: a fuse
        // of a flat field is a flat tile, which says nothing about where it
        // was sampled.
        let photo = image::RgbImage::from_fn(w, h, |x, y| {
            let v = ((x % 9) * 14 + (y % 7) * 18 + i as u32 * 5) as u8;
            image::Rgb([v, v.wrapping_add(40), 255 - v])
        });
        photo.save(dir.join(&image.name)).unwrap();
    }
    assert!(recon.image_table.thumbnails_y_x_rgb.is_none());
    assert!(recon.point_set.patch_bitmaps_y_x_rgba.is_none());
    save_into(dir, &recon, "recon.sfmr")
}

/// A directory of this test's own, emptied first.
fn temp_dir(name: &str) -> PathBuf {
    crate::display_thumbnails::tests::temp_dir(&format!("open_{name}"))
}

/// The newest Action Log entry.
fn newest(state: &AppState) -> &crate::action_log::Entry {
    state.action_log.entries().next_back().expect("an entry")
}

#[test]
fn opening_builds_every_thumbnail_from_the_sift_first_then_the_photographs() {
    let dir = temp_dir("thumbnails");
    let mut recon = workspace_with_photographs(&dir, &[2, 5]);
    write_sift_thumbnails(&mut recon, &[0, 1, 2], &[0, 1, 2]);
    let n = recon.image_table.images.len();
    let path = save_into(&dir, &recon, "recon.sfmr");

    let mut state = AppState::new();
    let id = state.open_now(&path).expect("the file opens");
    let node = state.node(id).unwrap();
    let display = node.display_thumbnails.as_ref().expect("rows were built");
    let row = |i: usize| display.row(&format!("images/photo_{i:02}.png")).unwrap();
    for i in [0, 1, 2] {
        let g = sift_grey_of(i);
        assert!(is_flat(row(i), [g, g, g]), "row {i} is its .sift copy");
    }
    assert!(is_flat(row(3), colour_of(3)), "row 3 is its photograph");
    assert!(is_flat(row(5), [PLACEHOLDER_GREY; 3]), "row 5 has neither");
    // Display data on the node: the value is the file's, without thumbnails.
    assert!(node.recon().image_table.thumbnails_y_x_rgb.is_none());

    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert_eq!(entry.text, format!("Opened recon from {}", path.display()));
    assert_eq!(
        phase_note(&entry.detail, "thumbnails"),
        Some(format!(
            "3 from .sift files, {} from photographs, 1 placeholders",
            n - 4
        )),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn opening_renders_the_patch_bitmaps_a_file_does_not_carry() {
    let dir = temp_dir("bitmaps");
    let path = saved_with_photographs(&dir);

    let mut state = AppState::new();
    let id = state.open_now(&path).expect("the file opens");
    let recon = state.node(id).unwrap().recon();
    let bitmaps = recon
        .point_set
        .patch_bitmaps_y_x_rgba
        .as_ref()
        .expect("the open rendered the column");
    assert!(recon.point_set.patch_bitmaps_for_display);
    assert_eq!(bitmaps.shape()[0], recon.point_count());
    // A point two photographs see gets a tile; the demo sees point p from
    // p % 3 + 1 images, so point 1 has two views and point 0 one.
    let tile = |p: usize| bitmaps.index_axis(ndarray::Axis(0), p);
    assert!(tile(1).iter().any(|&b| b != 0), "a two-view point is fused");
    assert!(tile(0).iter().all(|&b| b == 0), "a one-view point is not");

    let entry = newest(&state);
    let rows = phase_rows(&entry.detail);
    for stage in ["patch bitmaps", "decode photographs", "fuse", "thumbnails"] {
        assert!(
            rows.iter().any(|(name, _, _)| *name == stage),
            "no {stage} stage in {rows:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// What the open filled in is not part of the reconstruction: the value keeps
/// the file's content hash, which every point id is minted against, and a save
/// writes the columns the file had.
#[test]
fn the_fill_in_keeps_the_files_identity_and_a_save_writes_its_own_columns() {
    let dir = temp_dir("identity");
    let path = saved_with_photographs(&dir);
    let file_hash = sfmtool_sfmr_format::read_sfmr(&path)
        .unwrap()
        .content_hash
        .content_xxh128;

    let mut state = AppState::new();
    let id = state.open_now(&path).expect("the file opens");
    let node = state.node(id).unwrap();
    assert_eq!(node.recon().content_hash.content_xxh128, file_hash);
    assert_eq!(
        node.recon().content_xxh128().unwrap().content_xxh128,
        file_hash,
        "the display column is outside the hash"
    );
    assert_eq!(
        node.edited().base_content_hash().unwrap().content_xxh128,
        file_hash
    );

    // A plain Save with nothing to fold writes the file's own columns back.
    state.save_node(id).expect("a writable path");
    let saved = sfmtool_sfmr_format::read_sfmr(&path).unwrap();
    assert!(saved.patch_bitmaps_y_x_rgba.is_none());
    assert!(saved.thumbnails_y_x_rgb.is_none());
    assert_eq!(saved.content_hash.content_xxh128, file_hash);

    // A save that folds an edit mints a version whose hash is the new file's.
    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    state.save_node(id).expect("a writable path");
    let saved = sfmtool_sfmr_format::read_sfmr(&path).unwrap();
    assert!(saved.patch_bitmaps_y_x_rgba.is_none());
    let node = state.node(id).unwrap();
    assert_eq!(
        node.edited().base_content_hash().unwrap().content_xxh128,
        saved.content_hash.content_xxh128
    );
    // The session goes on drawing the rendered tiles of the points it kept.
    assert!(node.recon().point_set.patch_bitmaps_for_display);
    assert_eq!(
        node.recon()
            .point_set
            .patch_bitmaps_y_x_rgba
            .as_ref()
            .unwrap()
            .shape()[0],
        node.recon().point_count()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_path_that_is_not_a_file_is_refused_before_anything_starts() {
    let dir = temp_dir("missing");
    let mut state = AppState::new();
    let error = state
        .start_open(vec![dir.join("there-is-no-such-file.sfmr")])
        .expect_err("refused");
    assert!(error.starts_with("Failed to load "), "{error}");
    assert!(state.background_task().is_none());
    assert_eq!(state.action_log.entries().count(), 0, "nothing is logged");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_open_is_refused_while_another_operation_runs() {
    let dir = temp_dir("busy");
    let recon = workspace_with_photographs(&dir, &[]);
    let path = save_into(&dir, &recon, "recon.sfmr");
    let mut state = AppState::new();
    state.start_open(vec![path.clone()]).expect("nothing runs");
    let error = state.start_open(vec![path]).expect_err("one at a time");
    assert_eq!(error, "Open is still running on recon.");
    state.finish_background_task();
    assert_eq!(state.scene.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn several_files_open_in_order_as_one_task() {
    let dir = temp_dir("several");
    let recon = workspace_with_photographs(&dir, &[]);
    let first = save_into(&dir, &recon, "first.sfmr");
    let second = save_into(&dir, &recon, "second.sfmr");
    let mut state = AppState::new();
    state.open_files(vec![first.clone(), dir.join("gone.sfmr"), second.clone()]);
    state.finish_background_task();

    let labels: Vec<&str> = state.scene.iter().map(|n| n.label.as_str()).collect();
    assert_eq!(labels, ["first", "second"]);
    let rows: Vec<(bool, &str)> = state
        .action_log
        .entries()
        .map(|entry| (entry.failed, entry.text.as_str()))
        .collect();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert!(
        rows[0].0 && rows[0].1.starts_with("Failed to load "),
        "{rows:?}"
    );
    assert_eq!(
        rows[1],
        (
            false,
            format!(
                "Opened first from {}; second from {}",
                first.display(),
                second.display()
            )
            .as_str()
        )
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Open the fixture with both columns filled in, and delete a point so the
/// node has an overlay a minimal copy has to fold.
fn opened_and_edited(dir: &Path) -> (AppState, ReconId, PathBuf) {
    let path = saved_with_photographs(dir);
    let mut state = AppState::new();
    let id = state.open_now(&path).expect("the file opens");
    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    (state, id, path)
}

#[test]
fn save_as_minimal_writes_the_minimal_file_and_leaves_the_node() {
    let dir = temp_dir("minimal");
    let (mut state, id, path) = opened_and_edited(&dir);
    let (versions, disk, points) = {
        let node = state.node(id).unwrap();
        (
            node.history.versions().len(),
            node.history.disk_serial(),
            node.point_count(),
        )
    };
    let out = dir.join("published").join("recon.sfmr");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    state
        .save_minimal_copy(id, &out, None)
        .expect("a writable path");

    let written = sfmtool_sfmr_format::read_sfmr(&out).unwrap();
    assert!(written.thumbnails_y_x_rgb.is_none());
    assert!(written.patch_bitmaps_y_x_rgba.is_none());
    assert!(written.patch_u_halfvec_xyz.is_some(), "the frames stay");
    assert!(written.metadata.lineage.is_empty());
    assert!(written.metadata.workspace.absolute_path.is_empty());
    assert_eq!(written.metadata.workspace.relative_path, "..");
    assert_eq!(written.metadata.operation, "minimal");
    assert_eq!(written.metadata.tool, "sfm-explorer");
    assert!(written.metadata.tool_options.is_empty());
    assert_eq!(written.metadata.point_count as usize, points);

    // The node is exactly where it was: an export, not a save of it.
    let node = state.node(id).unwrap();
    assert_eq!(node.path.as_deref(), Some(path.as_path()));
    assert_eq!(node.label, "recon");
    assert_eq!(node.history.versions().len(), versions);
    assert_eq!(node.history.disk_serial(), disk);
    assert!(node.is_dirty());
    let entry = newest(&state);
    assert!(!entry.failed);
    assert!(
        entry.text.starts_with("Saved a minimal copy of recon at ")
            && entry.text.ends_with(&out.display().to_string()),
        "{}",
        entry.text
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_minimal_copy_over_the_nodes_own_file_is_refused() {
    let dir = temp_dir("minimal_own");
    let (mut state, id, path) = opened_and_edited(&dir);
    let before = std::fs::read(&path).unwrap();
    let error = state
        .save_minimal_copy(id, &path, None)
        .expect_err("refused");
    assert!(
        error.contains("cannot replace the file it came from"),
        "{error}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "the file is untouched"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
