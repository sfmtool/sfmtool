// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The SIFT index of a node: where it goes, what opening one adopts, what a
//! build makes, and what says an index is out of date.
//!
//! The fixture is a node saved as `demo.sfmr` in a temporary directory that is
//! also its workspace, with a real `.sift` file per image and one patch planted
//! in three of them under warps the tests state. That is what lets the search
//! test beside it ([`crate::bench::tests`]) assert where a candidate landed
//! rather than only that one appeared.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use ndarray::{Array2, Array3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use sfmtool_core::progress::{Event, Progress};

use super::{index_path, SiftIndexState, INDEX_FILE_SUFFIX};
use crate::action_log::Kind;
use crate::background::Finished;
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::edits::tests::projected_embedded_demo;
use crate::state::AppState;

/// Descriptor width, which is SIFT's.
const DIM: usize = 128;
/// The feature directory the fixture's workspace uses.
const PREFIX: &str = "features/sift-test";
/// Features of the planted patch, and so the inlier count a found image
/// reaches.
pub(crate) const PLANTED: usize = 24;
/// Features an unrelated image holds: too few to reach the inlier bar however
/// their positions fall.
const UNRELATED: usize = 6;
/// The point the bench tests put on the bench. It is observed in images 0, 1
/// and 2, so image 1 is an image a search finds and the track already holds,
/// and image 3 is one it finds and the track does not.
pub(crate) const POINT: u32 = 2;
/// The image a search runs from: the one observation 0 of [`POINT`] is in.
pub(crate) const QUERY_IMAGE: u32 = 0;
/// The image the track already holds that the patch is also planted in.
pub(crate) const HELD_IMAGE: u32 = 1;
/// The image the patch is planted in that the track does not hold.
pub(crate) const FOUND_IMAGE: u32 = 3;
/// The image left without a `.sift` file, so a build has to carry a row for an
/// image that contributes no descriptor.
pub(crate) const UNINDEXED_IMAGE: u32 = 7;

/// The warp planted between the query image and [`HELD_IMAGE`].
pub(crate) fn warp_to_held(p: [f64; 2]) -> [f64; 2] {
    [
        0.9 * p[0] - 0.1 * p[1] + 60.0,
        0.1 * p[0] + 0.9 * p[1] - 30.0,
    ]
}

/// The warp planted between the query image and [`FOUND_IMAGE`].
pub(crate) fn warp_to_found(p: [f64; 2]) -> [f64; 2] {
    [
        1.1 * p[0] + 0.05 * p[1] - 90.0,
        -0.05 * p[0] + 1.1 * p[1] + 40.0,
    ]
}

/// The 2x2 linear part of a planted warp, differenced off the warp itself so
/// the expected shape comes from the same function the positions do.
pub(crate) fn linear_of(warp: fn([f64; 2]) -> [f64; 2]) -> [[f64; 2]; 2] {
    let origin = warp([0.0, 0.0]);
    let x = warp([1.0, 0.0]);
    let y = warp([0.0, 1.0]);
    [
        [x[0] - origin[0], y[0] - origin[0]],
        [x[1] - origin[1], y[1] - origin[1]],
    ]
}

/// The `.sfmr` the fixture's node is saved as, whose stem the index path takes.
const SFMR_NAME: &str = "demo.sfmr";

/// A state holding one `embedded_patches` node saved in `dir`, which is also
/// its workspace.
///
/// No `.sift` file is written: this is the node a refusal is asserted against,
/// and [`with_sift_files`] is what gives it features.
pub(crate) fn state_in(dir: &Path) -> (AppState, ReconId) {
    let mut recon = projected_embedded_demo(12);
    recon.workspace_dir = dir.to_path_buf();
    recon.metadata.workspace.absolute_path = dir.display().to_string();
    recon.metadata.workspace.relative_path = ".".into();
    recon.metadata.workspace.contents.feature_prefix_dir = PREFIX.into();
    let mut node = SceneNode::demo(recon);
    node.path = Some(dir.join(SFMR_NAME));
    let mut state = AppState::new();
    state.append_node(node);
    let id = state.selected_recon.expect("a selected reconstruction");
    (state, id)
}

/// The same node, never saved: the one a build has nowhere to write for.
fn unsaved_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    (state, id)
}

/// Where the fixture's index goes.
fn index_of(dir: &Path) -> PathBuf {
    dir.join(format!("demo{INDEX_FILE_SUFFIX}"))
}

/// Write a `.sift` file for every image but [`UNINDEXED_IMAGE`], with one patch
/// planted around `center` in the query image and carried into the two images
/// the warps name.
pub(crate) fn with_sift_files(state: &AppState, id: ReconId, center: [f64; 2]) {
    let node = state.node(id).expect("loaded");
    let recon = node.recon();
    let mut rng = StdRng::seed_from_u64(19);
    let patch: Vec<Vec<u8>> = (0..PLANTED)
        .map(|_| (0..DIM).map(|_| rng.random_range(0..=u8::MAX)).collect())
        .collect();
    // Inside a box that stays on a 1920 x 1080 sensor wherever the centre is,
    // so a warp of it does not run off the frame either.
    let at: Vec<[f64; 2]> = (0..PLANTED)
        .map(|_| {
            [
                center[0] + f64::from(rng.random_range(-200.0..200.0_f32)),
                center[1] + f64::from(rng.random_range(-200.0..200.0_f32)),
            ]
        })
        .collect();

    for image in 0..recon.image_table.images.len() as u32 {
        if image == UNINDEXED_IMAGE {
            continue;
        }
        let (descriptors, positions) = match image {
            QUERY_IMAGE => (patch.clone(), at.clone()),
            HELD_IMAGE => (patch.clone(), at.iter().map(|&p| warp_to_held(p)).collect()),
            FOUND_IMAGE => (
                patch.clone(),
                at.iter().map(|&p| warp_to_found(p)).collect(),
            ),
            _ => (
                (0..UNRELATED)
                    .map(|_| (0..DIM).map(|_| rng.random_range(0..=u8::MAX)).collect())
                    .collect(),
                (0..UNRELATED)
                    .map(|_| {
                        [
                            f64::from(rng.random_range(0.0..1920.0_f32)),
                            f64::from(rng.random_range(0.0..1080.0_f32)),
                        ]
                    })
                    .collect(),
            ),
        };
        let path = recon.sift_path_for_image(image as usize);
        std::fs::create_dir_all(path.parent().expect("a feature directory")).unwrap();
        write_sift(
            &path,
            &recon.image_table.images[image as usize].name,
            &descriptors,
            &positions,
        );
    }
}

/// One `.sift` file with the given descriptors and keypoint positions, all at
/// the same isotropic shape.
///
/// Shared with the conversion fixture in `state::edits::tests`, which wants
/// the companions rather than the descriptors in them: what a `.sift` file has
/// to carry to be read at all is the same question either way.
pub(crate) fn write_sift(
    path: &Path,
    image_name: &str,
    descriptors: &[Vec<u8>],
    positions: &[[f64; 2]],
) {
    let count = descriptors.len();
    let mut positions_xy = Array2::<f32>::zeros((count, 2));
    let mut affine = Array3::<f32>::zeros((count, 2, 2));
    let mut rows = Array2::<u8>::zeros((count, DIM));
    for row in 0..count {
        positions_xy[[row, 0]] = positions[row][0] as f32;
        positions_xy[[row, 1]] = positions[row][1] as f32;
        affine[[row, 0, 0]] = 2.0;
        affine[[row, 1, 1]] = 2.0;
        for column in 0..DIM {
            rows[[row, column]] = descriptors[row][column];
        }
    }
    let data = sfmtool_sift_format::SiftData {
        feature_tool_metadata: sfmtool_sift_format::FeatureToolMetadata {
            feature_tool: "test".into(),
            feature_type: "sift".into(),
            feature_options: serde_json::json!({}),
        },
        metadata: sfmtool_sift_format::SiftMetadata {
            version: sfmtool_sift_format::SIFT_FORMAT_VERSION,
            image_name: image_name.to_string(),
            image_file_xxh128: "0".repeat(32),
            image_file_size: 1,
            image_width: 1920,
            image_height: 1080,
            feature_count: count as u32,
        },
        content_hash: sfmtool_sift_format::SiftContentHash::default(),
        positions_xy,
        affine_shapes: affine,
        descriptors: rows,
        thumbnail_y_x_rgb: Array3::<u8>::zeros((128, 128, 3)),
    };
    sfmtool_sift_format::write_sift(path, &data, 3).unwrap();
}

/// A state whose node has `.sift` files and a built, open SIFT index,
/// with [`POINT`] on the bench under the label it took.
///
/// The centre the patch is planted around is observation 0's own keypoint, read
/// off the track after it is put on, so a search from that observation looks at
/// the patch this planted.
pub(crate) fn searchable(dir: &Path) -> (AppState, ReconId, String) {
    let (mut state, id) = state_in(dir);
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize))
        .expect("a live point");
    let center = {
        let track = state.bench_track(id, &label).expect("just put on");
        let keypoint = track.observations[0]
            .track
            .as_ref()
            .and_then(|m| m.keypoint)
            .expect("a committed track carries its keypoints");
        assert_eq!(track.observations[0].image, QUERY_IMAGE);
        [f64::from(keypoint[0]), f64::from(keypoint[1])]
    };
    with_sift_files(&state, id, center);
    state
        .start_build_sift_index(id, None)
        .expect("a node with .sift files can be indexed");
    state.finish_background_task();
    (state, id, label)
}

// ── Where the file goes ─────────────────────────────────────────────────

/// The index takes its name from the `.sfmr`'s stem and sits beside it, so two
/// reconstructions saved in one directory have two indexes.
#[test]
fn the_index_path_is_the_sfmr_s_stem_beside_the_sfmr() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id) = state_in(dir.path());
    let expected = index_of(dir.path());
    assert_eq!(
        index_path(state.node(id).expect("loaded")),
        Some(expected.clone())
    );
    assert_eq!(state.sift_index_path(id), Some(expected));
}

/// The path is spelled in one convention, whatever the session was handed.
///
/// A `.sfmr` path can arrive spelled either way, and a name joined onto it
/// reads in a hover text, a reply and a log row as two conventions arguing.
#[test]
fn the_index_path_is_spelled_with_one_kind_of_separator() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    {
        let node = state.scene.first_mut().expect("one node");
        // A path with the separator this platform does not use in it, which is
        // what a `.sfmr` written elsewhere and opened here can look like.
        node.path = Some(PathBuf::from(format!(
            "{}/runs/demo.sfmr",
            dir.path().display()
        )));
    }
    let shown = state
        .sift_index_path(id)
        .expect("a saved node has an index path")
        .display()
        .to_string();
    // The separator this platform does not use, written as its code point so
    // the test carries no escape of its own.
    let backslash = char::from(0x5c_u8);
    let foreign = if std::path::MAIN_SEPARATOR == '/' {
        backslash
    } else {
        '/'
    };
    assert!(
        !shown.contains(foreign),
        "one convention, this platform's: {shown}"
    );
    assert!(shown.ends_with(INDEX_FILE_SUFFIX), "{shown}");
}

/// A node that has never been saved has nowhere to put an index, and is told
/// which thing to do about it.
#[test]
fn a_node_with_no_path_on_disk_cannot_have_an_index_built() {
    let (mut state, id) = unsaved_state();
    let why = state
        .build_sift_index_refusal(id)
        .expect("nowhere to write it");
    assert_eq!(
        why,
        "Save demo first: the SIFT index is written beside the .sfmr file."
    );
    let refused = state
        .start_build_sift_index(id, None)
        .expect_err("the step asks the same question the menu does");
    assert_eq!(refused, why);
    assert!(state.sift_index_path(id).is_none());
}

/// A caller may name the file; it may not name a file outside the `.sfmr`'s own
/// directory.
///
/// Naming it is worth keeping -- a second index over the same capture, under a
/// name of its own, is a reasonable thing to ask for -- and what is not worth
/// keeping is a build that writes wherever the string points.
#[test]
fn a_build_outside_the_sfmr_s_directory_is_refused_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    with_sift_files(&state, id, [900.0, 500.0]);

    let elsewhere = tempfile::tempdir().unwrap();
    let outside = elsewhere.path().join("stolen.kdf");
    let why = state
        .start_build_sift_index(id, Some(outside.clone()))
        .expect_err("that is not beside the .sfmr it indexes");
    assert!(why.contains("is outside"), "{why}");
    assert!(!outside.exists(), "nothing was written there");
    assert!(state.background_task().is_none(), "no task was started");

    // A climb out of the directory is the same refusal, spelled differently.
    let climbing = dir.path().join("..").join("stolen.kdf");
    let why = state
        .start_build_sift_index(id, Some(climbing))
        .expect_err("a .. that leaves the tree leaves the tree");
    assert!(why.contains("is outside"), "{why}");

    // Inside it, under a name of the caller's own, is built and opened.
    let mine = dir.path().join("second.kdf");
    state
        .start_build_sift_index(id, Some(mine.clone()))
        .expect("a path inside the .sfmr's directory");
    state.finish_background_task();
    assert!(mine.is_file(), "the build wrote the file it was given");
    assert_eq!(state.sift_index(id).expect("it opened it").path, mine);
}

#[test]
fn a_node_with_no_sift_files_cannot_have_an_index_built() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    let why = state
        .build_sift_index_refusal(id)
        .expect("nothing to index");
    assert!(why.contains("No .sift file"), "{why}");
    let refused = state
        .start_build_sift_index(id, None)
        .expect_err("the step asks the same question the menu does");
    assert_eq!(refused, why);
    assert!(
        state
            .action_log
            .entries()
            .any(|entry| entry.kind == Kind::Bench && entry.text.contains("No .sift file")),
        "the refusal is a row of its own"
    );
    assert!(state.sift_index(id).is_none());
}

// ── What a build makes ──────────────────────────────────────────────────

#[test]
fn a_build_indexes_every_sift_file_and_opens_what_it_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    let index = state.sift_index(id).expect("the build opened it");
    assert_eq!(index.path, index_of(dir.path()));
    assert!(index.path.is_file());
    // Every image but the one with no `.sift` file contributed: the two planted
    // copies and the query image, plus the unrelated ones.
    let unrelated = 8 - 3 - 1;
    assert_eq!(index.feature_count(), 3 * PLANTED + unrelated * UNRELATED);
    // The corpus carries a row for every image of the node, the one with no
    // features included, which is what makes a corpus image index a node image
    // index.
    let names = index
        .forest
        .image_table()
        .expect("a readable table")
        .expect("the build wrote one")
        .names
        .clone();
    let recon = state.node(id).expect("loaded").recon();
    let expected: Vec<String> = recon
        .image_table
        .images
        .iter()
        .map(|image| image.name.clone())
        .collect();
    assert_eq!(names, expected);
    assert_eq!(index.images, expected.len());
    assert!(
        state
            .action_log
            .entries()
            .any(|entry| entry.text.contains("Built the SIFT index")),
        "the build writes one row saying what it made"
    );
}

/// What the build is for: an index it has just written, over the node it read,
/// is current, and a search runs.
#[test]
fn a_built_index_is_current_over_an_embedded_patches_node() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    assert_eq!(state.sift_index_state(id), SiftIndexState::Current);
    assert_eq!(
        state.sift_index(id).expect("open").stale_reason(),
        None,
        "the hashes it recorded are the hashes the .sift files carry"
    );
    assert_eq!(state.sift_index_search_refusal(id), None);
}

/// The same over a `sift_files` node, whose observations name features in the
/// very files the index was built from.
#[test]
fn a_built_index_is_current_over_a_sift_files_node() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = crate::state::edits::tests::convertible_state(dir.path());
    state
        .start_build_sift_index(id, None)
        .expect("the fixture has a .sift file per image");
    state.finish_background_task();
    assert_eq!(state.sift_index_state(id), SiftIndexState::Current);
    assert_eq!(
        state.sift_index(id).expect("open").path,
        dir.path().join(format!("run_a{INDEX_FILE_SUFFIX}"))
    );
}

// ── Current or stale ────────────────────────────────────────────────────

/// An index over other photographs opens all the same, stays open and stays
/// named, and says which image it first disagrees on.
#[test]
fn an_index_over_other_images_is_stale_naming_the_first_one_it_disagrees_on() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    let path = state.sift_index(id).expect("built above").path.clone();

    let other = tempfile::tempdir().unwrap();
    let (mut renamed, other_id) = state_in(other.path());
    {
        let node = renamed.scene.first_mut().expect("one node");
        let recon = Arc::make_mut(&mut node.history.current_mut().base);
        recon.image_table.images[0].name = "somewhere_else.jpg".into();
    }
    renamed
        .open_sift_index(other_id, Some(path.clone()))
        .expect("a file that opens is adopted whether or not it fits");
    assert_eq!(renamed.sift_index_state(other_id), SiftIndexState::Stale);
    let why = renamed
        .sift_index(other_id)
        .expect("still open")
        .stale_reason()
        .expect("stale")
        .to_string();
    assert!(why.contains("somewhere_else.jpg"), "{why}");
    assert_eq!(
        renamed.sift_index(other_id).expect("still open").path,
        path,
        "a stale index stays named, so the person sees what is there"
    );
    // And it answers no search.
    let refusal = renamed
        .sift_index_search_refusal(other_id)
        .expect("a stale index answers nothing");
    assert!(refusal.contains(&why), "{refusal}");
    assert!(
        renamed
            .action_log
            .entries()
            .any(|entry| entry.text.contains("out of date")),
        "the open says so in the log rather than refusing"
    );
}

/// An index over a **superset** of the node's images is stale, for the reason
/// any other discrepancy is: a forest query answers across the whole corpus.
#[test]
fn an_index_over_more_images_than_the_node_has_is_stale() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    let path = state.sift_index(id).expect("built above").path.clone();
    let indexed = state.sift_index(id).expect("built above").images;

    state
        .delete_image(ImageRef::new(id, 5))
        .expect("an image the node can lose");
    state
        .open_sift_index(id, Some(path))
        .expect("it still opens");
    assert_eq!(state.sift_index_state(id), SiftIndexState::Stale);
    let why = state
        .sift_index(id)
        .expect("open")
        .stale_reason()
        .expect("stale")
        .to_string();
    assert!(
        why.contains(&format!("indexes {indexed} images")),
        "the sentence names the counts: {why}"
    );
}

/// Features extracted again after the index was built make it stale, and the
/// sentence says so in those words.
#[test]
fn re_extracted_features_make_the_index_stale() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    let path = state.sift_index(id).expect("built above").path.clone();

    let name = {
        let recon = state.node(id).expect("loaded").recon();
        let sift = recon.sift_path_for_image(2);
        // The same image, extracted again: different descriptors, so a
        // different content hash in the file's own metadata.
        write_sift(
            &sift,
            &recon.image_table.images[2].name,
            &vec![vec![7u8; DIM]; 4],
            &[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]],
        );
        recon.image_table.images[2].name.clone()
    };
    state.open_sift_index(id, Some(path)).expect("it opens");
    assert_eq!(state.sift_index_state(id), SiftIndexState::Stale);
    assert_eq!(
        state.sift_index(id).expect("open").stale_reason(),
        Some(
            format!("The features of {name} were extracted again after this index was built.")
                .as_str()
        )
    );
}

/// A `.sift` file that has appeared since the build is a discrepancy, and so is
/// one that has gone.
#[test]
fn a_sift_file_that_appeared_or_vanished_makes_the_index_stale() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    let path = state.sift_index(id).expect("built above").path.clone();

    // The one image the fixture left without features now has some.
    let (appeared, vanished) = {
        let recon = state.node(id).expect("loaded").recon();
        let sift = recon.sift_path_for_image(UNINDEXED_IMAGE as usize);
        std::fs::create_dir_all(sift.parent().expect("a feature directory")).unwrap();
        write_sift(
            &sift,
            &recon.image_table.images[UNINDEXED_IMAGE as usize].name,
            &vec![vec![3u8; DIM]; 2],
            &[[1.0, 2.0], [3.0, 4.0]],
        );
        (
            recon.image_table.images[UNINDEXED_IMAGE as usize]
                .name
                .clone(),
            recon.sift_path_for_image(0),
        )
    };
    state
        .open_sift_index(id, Some(path.clone()))
        .expect("it opens");
    assert_eq!(
        state.sift_index(id).expect("open").stale_reason(),
        Some(
            format!("{appeared} has a .sift file now, and this index was built without one.")
                .as_str()
        )
    );

    // And the other direction: image 0's features are gone. Its row comes
    // first, so it is the discrepancy the sentence names.
    let gone = {
        let recon = state.node(id).expect("loaded").recon();
        recon.image_table.images[0].name.clone()
    };
    std::fs::remove_file(&vanished).unwrap();
    state.open_sift_index(id, Some(path)).expect("it opens");
    assert_eq!(
        state.sift_index(id).expect("open").stale_reason(),
        Some(format!("{gone} has no .sift file now, and this index was built from one.").as_str())
    );
}

/// The state is derived again when a version moves the image table, and left
/// alone when one does not -- which is what keeps the `.sift` reads off every
/// edit.
#[test]
fn a_version_re_derives_the_state_only_when_it_moves_the_image_table() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, label) = searchable(dir.path());
    assert_eq!(state.sift_index_state(id), SiftIndexState::Current);

    // A `.sift` file rewritten underneath: a re-derivation would find it and
    // call the index stale, so the verdict after a version that leaves the
    // image table alone says whether one happened.
    {
        let recon = state.node(id).expect("loaded").recon();
        let sift = recon.sift_path_for_image(2);
        write_sift(
            &sift,
            &recon.image_table.images[2].name,
            &vec![vec![9u8; DIM]; 3],
            &[[11.0, 21.0], [31.0, 41.0], [51.0, 61.0]],
        );
    }
    state
        .set_bench_verdict(id, &label, 0, sfmtool_core::bench::Verdict::Out)
        .expect("a verdict is a version");
    state.refresh_sift_index(id);
    assert_eq!(
        state.sift_index_state(id),
        SiftIndexState::Current,
        "a version that moves no image re-reads nothing"
    );

    // One that does move the table re-derives, and now finds the rewritten
    // file as well as the missing row.
    state
        .delete_image(ImageRef::new(id, 5))
        .expect("an image the node can lose");
    state.refresh_sift_index(id);
    assert_eq!(state.sift_index_state(id), SiftIndexState::Stale);
}

// ── Opening on sight, and letting go ────────────────────────────────────

#[test]
fn looking_for_an_index_that_is_not_there_is_silent_and_remembered() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    state.refresh_sift_index(id);
    assert!(state.sift_index(id).is_none());
    assert_eq!(state.sift_index_state(id), SiftIndexState::None);
    assert!(
        !state
            .action_log
            .entries()
            .any(|entry| entry.kind == Kind::Bench),
        "an index that is not there is the ordinary state, not a refusal"
    );
    // Remembered, so the tree's next frame does not stat the same absent file.
    assert!(state.sift_indexes.contains_key(&id));
}

/// Closing lets go of the forest, leaves the file, and greys the search.
#[test]
fn closing_lets_go_of_the_index_and_leaves_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    let path = state.sift_index(id).expect("built above").path.clone();

    state.close_sift_index(id).expect("one is open");
    assert_eq!(state.sift_index_state(id), SiftIndexState::None);
    assert!(path.is_file(), "the file stays where it is");
    assert!(
        state
            .action_log
            .entries()
            .any(|entry| entry.text.starts_with("Closed the SIFT index")),
        "closing is a row of its own"
    );
    let why = state
        .sift_index_search_refusal(id)
        .expect("nothing to search");
    assert!(why.contains("No SIFT index is open"), "{why}");

    // A second close has nothing to do and says so, and the look is remembered
    // rather than re-opening the file the next frame.
    let refused = state.close_sift_index(id).expect_err("none is open");
    assert!(refused.contains("No SIFT index is open"), "{refused}");
    state.refresh_sift_index(id);
    assert_eq!(state.sift_index_state(id), SiftIndexState::None);
}

// ── What the build reports, and what a cancel leaves ────────────────────

/// Every fraction `job` reported, and what it came back with.
fn fractions_of(job: crate::background::Job, cancel: Option<&AtomicBool>) -> (Vec<f32>, Finished) {
    let seen = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Fraction { of_whole } = event {
            seen.lock().unwrap().push(of_whole);
        }
    };
    let progress = Progress::to(&sink);
    let progress = match cancel {
        Some(flag) => progress.cancelled_by(flag),
        None => progress,
    };
    let finished = job(&progress);
    (seen.into_inner().unwrap(), finished)
}

/// The bar moves through all three phases. The read is the first quarter, so a
/// fraction above it is the forest saying something, and one above a half is
/// the write: a build that only reported its read would leave the bar standing
/// still for the ten seconds that matter.
#[test]
fn a_build_reports_through_the_forest_and_the_write() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id) = state_in(dir.path());
    with_sift_files(&state, id, [900.0, 500.0]);
    let job = state
        .build_sift_index_job(id, None)
        .expect("a node with .sift files can be indexed");

    let (fractions, finished) = fractions_of(job, None);
    assert!(
        matches!(finished, Finished::SiftIndex { .. }),
        "the build did not produce an index"
    );
    // Not asserted in order: the forest's reports come from rayon's workers,
    // so two of them can reach one sink in the other order from the one they
    // were computed in. Making the bar monotone is the collector's, which
    // clamps to the greatest it has seen
    // (`specs/gui/operation-progress.md`).
    assert!(
        fractions.iter().all(|f| (0.0..=1.0).contains(f)),
        "{fractions:?}"
    );
    assert!(
        fractions.iter().any(|f| *f > 0.25 && *f < 0.5),
        "the forest build reported nothing: {fractions:?}"
    );
    assert!(
        fractions.iter().any(|f| *f > 0.5 && *f < 1.0),
        "the write reported nothing: {fractions:?}"
    );
    assert_eq!(
        fractions.last().copied(),
        Some(1.0),
        "a finished build ends at its end"
    );
}

/// A cancelled rebuild leaves the index that is there exactly as it was, and
/// leaves nothing of its own beside it.
#[test]
fn a_cancelled_rebuild_leaves_the_index_that_is_there() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    let path = index_of(dir.path());
    let before = std::fs::read(&path).expect("the first build wrote it");

    let job = state
        .build_sift_index_job(id, None)
        .expect("a node with .sift files can be indexed");
    let flag = AtomicBool::new(true);
    let (_, finished) = fractions_of(job, Some(&flag));
    assert!(
        matches!(finished, Finished::Cancelled),
        "the build did not stop"
    );
    assert_eq!(
        std::fs::read(&path).expect("still there"),
        before,
        "a cancelled rebuild replaced the index it was rebuilding"
    );
    // And no half-written file under the name a build writes into.
    let leftovers: Vec<PathBuf> = std::fs::read_dir(dir.path())
        .expect("the directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|entry| entry.extension().is_some_and(|e| e == "building"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    // The node still has its index open, and it is still the good one.
    state.refresh_sift_index(id);
    assert_eq!(state.sift_index_state(id), SiftIndexState::Current);
}
