// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Display thumbnails: the file's own column shared, rows built from the
//! photographs for a file without one, and none of it reaching a save.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::Array4;
use sfmtool_core::{SfmrReconstruction, THUMBNAIL_SIZE};

use super::{row_for, DisplayThumbnails, PLACEHOLDER_GREY};
use crate::scene::SceneNode;
use crate::state::AppState;

/// A directory of this test's own under the system temp dir, emptied first.
pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sfm_explorer_display_thumbnails_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a writable temp dir");
    dir
}

/// The flat colour photograph `image` is written as.
pub(crate) fn colour_of(image: usize) -> [u8; 3] {
    [
        (image * 30 % 256) as u8,
        200,
        (255 - image * 20 % 256) as u8,
    ]
}

/// The demo reconstruction in workspace `dir`, with no thumbnails, and a flat
/// colour PNG photograph under `images/` for every image not in `missing`.
pub(crate) fn workspace_with_photographs(dir: &Path, missing: &[usize]) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(16);
    recon.workspace_dir = dir.to_path_buf();
    std::fs::create_dir_all(dir.join("images")).unwrap();
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.name = format!("images/photo_{i:02}.png");
        if missing.contains(&i) {
            continue;
        }
        let [r, g, b] = colour_of(i);
        let photo = image::RgbImage::from_pixel(96, 54, image::Rgb([r, g, b]));
        photo.save(dir.join(&image.name)).unwrap();
    }
    assert!(recon.image_table.thumbnails_y_x_rgb.is_none());
    recon
}

fn is_flat(row: ndarray::ArrayView3<'_, u8>, rgb: [u8; 3]) -> bool {
    row.shape() == [THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3]
        && row
            .as_slice()
            .expect("a contiguous row")
            .chunks(3)
            .all(|p| p == rgb)
}

#[test]
fn a_file_with_thumbnails_shares_its_own_column() {
    let mut recon = SfmrReconstruction::demo(16);
    let n = recon.image_table.images.len();
    let column = Arc::new(Array4::<u8>::from_shape_fn(
        (n, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        |(i, _, _, c)| (i * 10 + c) as u8,
    ));
    recon.image_table.thumbnails_y_x_rgb = Some(Arc::clone(&column));

    let node = SceneNode::demo(recon);
    let display = node.display_thumbnails.as_ref().expect("the file's column");
    assert!(!display.is_synthesized());
    assert!(display.is_complete());
    // Shared rather than copied: the row is a view into the table's own array.
    let row = display.row("image_003.jpg").expect("a final row");
    assert_eq!(
        row.as_ptr(),
        column.index_axis(ndarray::Axis(0), 3).as_ptr()
    );
}

#[test]
fn a_node_opened_without_thumbnails_builds_rows_and_saves_none() {
    let dir = temp_dir("builds");
    let recon = workspace_with_photographs(&dir, &[]);
    let n = recon.image_table.images.len();
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::from_path(&dir.join("recon.sfmr"), recon));

    let display = state.node(id).unwrap().display_thumbnails.clone();
    let display = display.expect("photographs to build from");
    assert!(display.is_synthesized());
    display.wait();
    assert_eq!(display.ready(), n);
    for i in 0..n {
        let row = display.row(&format!("images/photo_{i:02}.png")).unwrap();
        assert!(
            is_flat(row, colour_of(i)),
            "row {i} is its photograph's resize"
        );
    }

    // The value never saw any of it.
    let node = state.node(id).unwrap();
    assert!(node.recon().image_table.thumbnails_y_x_rgb.is_none());

    // One Action Log line when it finished, and only one.
    state.report_display_thumbnails();
    state.report_display_thumbnails();
    let lines: Vec<String> = state
        .action_log
        .entries()
        .map(|entry| entry.text.clone())
        .filter(|text| text.contains("display thumbnail"))
        .collect();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with(&format!("Built {n} display thumbnails")));

    // And a save writes a file without thumbnails.
    let out = dir.join("saved.sfmr");
    state.save_node_as(id, &out).expect("a save");
    let saved = sfmtool_sfmr_format::read_sfmr(&out).expect("a readable file");
    assert!(saved.thumbnails_y_x_rgb.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_photograph_that_cannot_be_read_gets_a_grey_row() {
    let dir = temp_dir("unreadable");
    let recon = workspace_with_photographs(&dir, &[2]);
    let display = DisplayThumbnails::synthesize(&recon, None).expect("some photographs");
    display.wait();

    let grey = [PLACEHOLDER_GREY; 3];
    assert!(is_flat(display.row("images/photo_02.png").unwrap(), grey));
    assert!(is_flat(
        display.row("images/photo_01.png").unwrap(),
        colour_of(1)
    ));
    let line = display.take_finished_line("recon").expect("finished");
    assert!(line.contains("1 could not be read"), "{line}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_node_whose_photographs_are_absent_has_no_display_column() {
    let dir = temp_dir("absent");
    let all: Vec<usize> = (0..8).collect();
    let recon = workspace_with_photographs(&dir, &all);
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::from_path(&dir.join("recon.sfmr"), recon));

    let node = state.node(id).unwrap();
    assert!(node.display_thumbnails.is_none());
    // Every panel falls back to its placeholder.
    assert!(row_for(None, node.recon(), 0).is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rows_follow_the_image_name_across_a_renumbering() {
    let dir = temp_dir("renumber");
    let recon = workspace_with_photographs(&dir, &[]);
    let display = DisplayThumbnails::synthesize(&recon, None).expect("photographs");
    display.wait();

    // Delete Image renumbers the table: what was image 3 is image 2 now, and
    // the column built for the first version still draws it.
    let subset = recon
        .subset_by_image_indices(&[0, 1, 3, 4, 5, 6, 7], false)
        .unwrap();
    let row = row_for(Some(&display), &subset, 2).expect("a final row");
    assert!(is_flat(row, colour_of(3)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_image_the_column_was_not_built_for_falls_back_to_the_values_column() {
    let dir = temp_dir("fallback");
    let recon = workspace_with_photographs(&dir, &[]);
    let display = DisplayThumbnails::synthesize(&recon, None).expect("photographs");
    display.wait();

    let mut other = recon.clone();
    other.image_table.images[5].name = "elsewhere/new.png".into();
    let n = other.image_table.images.len();
    other.image_table.thumbnails_y_x_rgb = Some(Arc::new(Array4::from_elem(
        (n, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        7u8,
    )));
    assert!(is_flat(
        row_for(Some(&display), &other, 5).unwrap(),
        [7, 7, 7]
    ));
    // The images the column knows are still drawn from it.
    assert!(is_flat(
        row_for(Some(&display), &other, 4).unwrap(),
        colour_of(4)
    ));
    let _ = std::fs::remove_dir_all(&dir);
}
