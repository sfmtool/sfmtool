// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The display thumbnail rows an open leaves awaiting a mipmap, filled in once
//! *Build mipmaps* has answered their photographs. The reconstruction is
//! `saved_out_of_core_recon`'s: images 0 to 4 are 300 x 200 PNGs with no
//! mipmap in a workspace whose `mipmap_options` puts a photograph with a side
//! over 64 px out of core, image 5 is in core, image 6 has a current mipmap
//! and image 7 has no file. The reads of the rows run on a queue the test
//! runs by hand.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::thumbnail::{
    photograph_thumbnail_row, thumbnail_from_rgb, NotCurrentMipmap, ThumbnailSource,
};

use crate::display_thumbnails::tests::is_flat;
use crate::display_thumbnails::PLACEHOLDER_GREY;
use crate::scene::ReconId;
use crate::state::build_mipmaps::tests::{
    cache_for, run_until_empty, saved_out_of_core_recon, write_out_of_core_workspace_file,
};
use crate::state::{AppState, PhotographRequests, TaskQueue};

/// A state whose photograph cache keeps `budget_bytes` and makes one build at
/// a time, whose reads run on the returned queue, with `saved` open; and the
/// paths of its eight images as the opened reconstruction spells them.
fn opened(saved: &Path, budget_bytes: u64) -> (AppState, ReconId, TaskQueue, Vec<PathBuf>) {
    let mut state = AppState::new();
    state.photographs = cache_for(budget_bytes, 1);
    let (requests, tasks) = PhotographRequests::queued();
    state.photograph_requests = requests;
    let id = state.open_now(saved).expect("the file opens");
    let recon = state.node(id).expect("the opened node").recon();
    let paths = recon
        .image_set
        .images
        .iter()
        .map(|image| recon.workspace_dir.join(&image.name))
        .collect();
    (state, id, tasks, paths)
}

/// The display row of image `i` of node `id`, as bytes.
fn row(state: &AppState, id: ReconId, i: usize) -> Vec<u8> {
    let node = state.node(id).expect("the node");
    let display = node.display_thumbnails.as_ref().expect("a display column");
    let name = &node.recon().image_set.images[i].name;
    display.row(name).expect("a row").iter().copied().collect()
}

/// The row of the photograph at `path` decoded whole.
fn whole_decode_row(path: &Path) -> Vec<u8> {
    let image = ImageU8::read_rgb(path).unwrap();
    thumbnail_from_rgb(
        image.data(),
        image.width() as usize,
        image.height() as usize,
    )
}

/// The open draws the five photographs without a current mipmap as the grey
/// placeholder, decoding none of them; once *Build mipmaps* has made their
/// mipmaps, one read each on the rayon pool fills every row in from its
/// coarsest mip level file, the frame after it lands. The value, its
/// history and what a save writes are unchanged, the node has no unsaved
/// changes, and the next frames start no read and change nothing.
#[test]
fn a_row_awaiting_its_mipmap_is_filled_once_the_task_has_made_it() {
    let (_dir, saved, _) = saved_out_of_core_recon();
    let (mut state, id, tasks, paths) = opened(&saved, 1 << 30);
    let grey = vec![PLACEHOLDER_GREY; 128 * 128 * 3];
    for i in 0..5 {
        assert!(row(&state, id, i) == grey, "image {i} awaits its mipmap");
    }
    assert!(row(&state, id, 5) == whole_decode_row(&paths[5]));
    assert!(row(&state, id, 6) != grey, "image 6's mipmap is current");
    assert_eq!(state.photographs.stats().decodes, 0);
    let entry = state.action_log.entries().next_back().unwrap();
    assert_eq!(
        crate::test_support::phase_note(&entry.detail, "thumbnails").as_deref(),
        Some("0 from .sift files, 2 from photographs, 1 placeholders, 5 awaiting their mipmaps")
    );
    let before = Arc::clone(state.node(id).unwrap().display_thumbnails.as_ref().unwrap());

    // Nothing is answered yet: no read starts.
    assert!(!state.drive_thumbnail_fills());
    assert_eq!(tasks.len(), 0);

    run_until_empty(&state.photographs, &state.photograph_requests);
    assert!(!state.drive_thumbnail_fills(), "the reads have not landed");
    assert_eq!(tasks.run_off_this_thread(), 5);
    assert!(state.drive_thumbnail_fills());

    // Each row is its coarsest mip level file's, as a fresh cache reads it.
    let fresh = cache_for(1 << 30, 1);
    for (i, path) in paths.iter().enumerate().take(5) {
        let (expected, source) = photograph_thumbnail_row(
            path,
            &fresh,
            NotCurrentMipmap::Placeholder,
            &Progress::none(),
        )
        .unwrap();
        assert_eq!(source, ThumbnailSource::Photograph);
        assert!(row(&state, id, i) == expected, "image {i}");
        assert!(expected != whole_decode_row(path));
    }
    let node = state.node(id).unwrap();
    let display = node.display_thumbnails.as_ref().unwrap();
    assert_eq!(display.awaiting().count(), 0);
    assert!(before.same_build(display), "the other rows are shared");
    assert_eq!(state.photographs.stats().decodes, 0);
    assert_eq!(state.photographs.stats().mipmaps_made, 5);
    assert_eq!(node.history.versions().len(), 1);
    assert!(node.recon().image_set.thumbnails_y_x_rgb.is_none());
    assert!(!state.is_dirty(id));

    assert!(!state.drive_thumbnail_fills());
    assert_eq!(tasks.len(), 0);
    assert!(state.photograph_requests.mipmaps_waiting().is_empty());
}

/// With a budget of 0 and a directory where each mip level file of the five
/// photographs would go, so none of their files can be written and no level
/// is kept, the row of each photograph *Build mipmaps* answered is read once,
/// decoded whole, and nothing is queued again: no read, no build and no
/// repaint follows.
#[test]
fn a_row_whose_mipmap_is_not_current_once_answered_is_decoded_whole_once() {
    let (_dir, saved, _) = saved_out_of_core_recon();
    let (mut state, id, tasks, paths) = opened(&saved, 0);
    for path in &paths[..5] {
        let location = sfmtool_core::camera::mipmap::mipmap_location(path).unwrap();
        for level in 0..8 {
            std::fs::create_dir_all(location.level_path(level, 95)).unwrap();
        }
    }
    for i in 0..5 {
        assert!(is_flat(
            ndarray::ArrayView3::from_shape((128, 128, 3), &row(&state, id, i)).unwrap(),
            [PLACEHOLDER_GREY; 3]
        ));
    }

    run_until_empty(&state.photographs, &state.photograph_requests);
    assert!(!state.drive_thumbnail_fills(), "the reads have not landed");
    assert_eq!(tasks.run_off_this_thread(), 5);
    assert!(state.drive_thumbnail_fills());
    for (i, path) in paths.iter().enumerate().take(5) {
        assert!(row(&state, id, i) == whole_decode_row(path), "image {i}");
    }
    let made = state.photographs.stats().mipmaps_made;
    for _ in 0..3 {
        assert!(!state.drive_thumbnail_fills());
        assert!(!state.drive_build_mipmaps());
    }
    assert_eq!(tasks.len(), 0);
    assert!(state.photograph_requests.mipmaps_waiting().is_empty());
    assert_eq!(state.photographs.stats().mipmaps_made, made);
}

/// A node closed while its rows are being read gets none of them: the reads
/// land, change nothing and start nothing. A photograph *Build mipmaps*
/// answers whose row no node awaits, as for one a panel queued, starts no
/// read.
#[test]
fn a_read_that_lands_after_its_node_closed_changes_nothing() {
    let (_dir, saved, _) = saved_out_of_core_recon();
    let (mut state, id, tasks, paths) = opened(&saved, 1 << 30);
    run_until_empty(&state.photographs, &state.photograph_requests);
    assert!(!state.drive_thumbnail_fills());
    assert_eq!(tasks.len(), 5);
    state.close_node(id).expect("the node closes");
    assert_eq!(tasks.run_off_this_thread(), 5);
    assert!(!state.drive_thumbnail_fills(), "no node awaits the rows");
    assert!(!state.drive_thumbnail_fills());
    assert_eq!(tasks.len(), 0);

    // Opened again, its photographs' mipmaps current, so no row awaits.
    state.open_now(&saved).expect("the file opens again");
    crate::state::build_mipmaps::lock(&state.photograph_requests.mipmaps).queue(&paths[0]);
    run_until_empty(&state.photographs, &state.photograph_requests);
    assert!(!state.drive_thumbnail_fills());
    assert_eq!(tasks.len(), 0, "no row awaits the photograph answered");
}

/// A row the thumbnails left awaiting a mipmap that the open's patch bitmap
/// render then made is filled in by the open itself, from the coarsest mip
/// level file: no photograph is left on *Build mipmaps*' list for it to
/// answer.
#[test]
fn the_open_fills_the_rows_whose_mipmaps_its_patch_bitmaps_made() {
    let tmp = tempfile::tempdir().unwrap();
    let saved = crate::state::open::tests::saved_with_photographs(tmp.path());
    write_out_of_core_workspace_file(tmp.path());
    let (state, id, tasks, paths) = opened(&saved, 1 << 30);
    let n = paths.len();
    let entry = state.action_log.entries().next_back().unwrap();
    assert_eq!(
        crate::test_support::phase_note(&entry.detail, "thumbnails"),
        Some(format!(
            "0 from .sift files, 0 from photographs, 0 placeholders, {n} awaiting their mipmaps"
        ))
    );
    assert_eq!(
        crate::test_support::phase_note(&entry.detail, "mipmaps"),
        Some(format!("0 of {n} photographs need their mipmaps made"))
    );
    let display = state.node(id).unwrap().display_thumbnails.as_ref().unwrap();
    assert_eq!(display.awaiting().count(), 0);
    let fresh = cache_for(1 << 30, 1);
    for (i, path) in paths.iter().enumerate() {
        let (expected, source) = photograph_thumbnail_row(
            path,
            &fresh,
            NotCurrentMipmap::Placeholder,
            &Progress::none(),
        )
        .unwrap();
        assert_eq!(source, ThumbnailSource::Photograph, "image {i}");
        assert!(row(&state, id, i) == expected, "image {i}");
    }
    assert_eq!(tasks.len(), 0);
}
