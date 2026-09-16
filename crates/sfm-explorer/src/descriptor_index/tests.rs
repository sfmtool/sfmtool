// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The descriptor index beside a node: where it goes by default, what opening
//! one refuses, and what a build makes.
//!
//! The fixture is a demo node whose workspace is a temporary directory with a
//! real `.sift` file per image, with one patch planted in three of them under
//! warps the tests state. That is what lets the search test beside it
//! ([`crate::bench::tests`]) assert where a candidate landed rather than only
//! that one appeared.

use std::path::Path;
use std::sync::Arc;

use ndarray::{Array2, Array3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::{default_index_path, INDEX_FILE_NAME};
use crate::action_log::Kind;
use crate::scene::{PointRef, ReconId, SceneNode};
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

/// A state holding one `embedded_patches` node whose workspace is `dir`.
///
/// No `.sift` file is written: this is the node a refusal is asserted against,
/// and [`with_sift_files`] is what gives it features.
pub(crate) fn state_in(dir: &Path) -> (AppState, ReconId) {
    let mut recon = projected_embedded_demo(12);
    recon.workspace_dir = dir.to_path_buf();
    recon.metadata.workspace.absolute_path = dir.display().to_string();
    recon.metadata.workspace.relative_path = ".".into();
    recon.metadata.workspace.contents.feature_prefix_dir = PREFIX.into();
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.selected_recon.expect("a selected reconstruction");
    (state, id)
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
fn write_sift(path: &Path, image_name: &str, descriptors: &[Vec<u8>], positions: &[[f64; 2]]) {
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

/// A state whose node has `.sift` files and a built, open descriptor index,
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
        .start_build_descriptor_index(id, None)
        .expect("a node with .sift files can be indexed");
    state.finish_background_task();
    (state, id, label)
}

#[test]
fn the_default_path_is_index_kdf_beside_the_first_image_s_sift_file() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id) = state_in(dir.path());
    let recon = state.node(id).expect("loaded").recon();
    let expected = recon
        .sift_path_for_image(0)
        .parent()
        .expect("a feature directory")
        .join(INDEX_FILE_NAME);
    assert_eq!(default_index_path(recon), Some(expected.clone()));
    assert_eq!(state.default_descriptor_index_path(id), Some(expected));
}

#[test]
fn a_node_with_no_sift_files_cannot_have_an_index_built() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    let why = state
        .build_descriptor_index_refusal(id)
        .expect("nothing to index");
    assert!(why.contains("No .sift file"), "{why}");
    let refused = state
        .start_build_descriptor_index(id, None)
        .expect_err("the step asks the same question the button does");
    assert_eq!(refused, why);
    assert!(
        state
            .action_log
            .entries()
            .any(|entry| entry.kind == Kind::Bench && entry.text.contains("No .sift file")),
        "the refusal is a row of its own"
    );
    assert!(state.descriptor_index(id).is_none());
}

#[test]
fn a_build_indexes_every_sift_file_and_opens_what_it_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    let index = state.descriptor_index(id).expect("the build opened it");
    assert_eq!(
        Some(index.path.clone()),
        state.default_descriptor_index_path(id)
    );
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
    assert!(
        state
            .action_log
            .entries()
            .any(|entry| entry.text.contains("Built the descriptor index")),
        "the build writes one row saying what it made"
    );
}

#[test]
fn an_index_over_other_images_is_refused_naming_the_first_one_it_disagrees_on() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    let path = state
        .descriptor_index(id)
        .expect("built above")
        .path
        .clone();

    // A second node whose images are named differently: the same file, opened
    // beside it, is not an index of its photographs.
    let other = tempfile::tempdir().unwrap();
    let (mut renamed, other_id) = state_in(other.path());
    {
        let node = renamed.scene.first_mut().expect("one node");
        let recon = Arc::make_mut(&mut node.history.current_mut().base);
        recon.image_table.images[0].name = "somewhere_else.jpg".into();
    }
    let why = renamed
        .open_descriptor_index(other_id, Some(path))
        .expect_err("the image tables disagree");
    assert!(why.contains("somewhere_else.jpg"), "{why}");
    assert!(renamed.descriptor_index(other_id).is_none());
}

#[test]
fn looking_for_a_default_that_is_not_there_is_silent_and_remembered() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    state.open_default_descriptor_index(id);
    assert!(state.descriptor_index(id).is_none());
    assert!(
        !state
            .action_log
            .entries()
            .any(|entry| entry.kind == Kind::Bench),
        "an index that is not there is the ordinary state, not a refusal"
    );
    // Remembered, so the panel's next frame does not stat the same absent file.
    assert!(state.descriptor_indexes.contains_key(&id));
}
