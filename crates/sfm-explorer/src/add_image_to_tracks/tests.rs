// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `Add Image to Tracks` in the viewer: the plane capture with one image's
//! observations taken out, added back on a worker, landed as one version
//! with its Action Log row, and taken back out by Undo; and the refusals the
//! greyed entry and the step share.

use std::sync::Arc;

use ndarray::Array2;
use sfmtool_core::{ObservationSource, SfmrReconstruction, TrackObservation};

use super::*;
use crate::bench::track_at_pixel::tests::{cache_photographs, plane_recon};
use crate::scene::{ReconId, SceneNode};

/// `recon` with every observation `image` makes taken out.
fn without_image(mut recon: SfmrReconstruction, image: u32) -> SfmrReconstruction {
    let set = &mut recon.point_set;
    let keep: Vec<usize> = (0..set.tracks.len())
        .filter(|&row| set.tracks[row].image_index != image)
        .collect();
    let ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes,
    } = &set.observations
    else {
        panic!("the plane capture is embedded_patches");
    };
    let mut kps = Array2::<f32>::zeros((keep.len(), 2));
    for (k, &row) in keep.iter().enumerate() {
        kps[[k, 0]] = keypoints_xy[[row, 0]];
        kps[[k, 1]] = keypoints_xy[[row, 1]];
    }
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: kps,
        image_file_hashes: image_file_hashes.clone(),
    };
    let tracks: Vec<TrackObservation> = keep.iter().map(|&row| set.tracks[row]).collect();
    let mut counts = vec![0u32; set.points.len()];
    for t in &tracks {
        counts[t.point_index as usize] += 1;
    }
    set.tracks = tracks;
    set.observation_counts = counts;
    recon.metadata.observation_count = recon.point_set.tracks.len() as u32;
    recon.rebuild_derived_fields();
    recon
}

/// A state holding the plane capture with image 0's observations taken out,
/// its photographs cached.
pub(crate) fn untracked_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(without_image(plane_recon(), 0)));
    let id = state.selected_recon.expect("a selected reconstruction");
    cache_photographs(&mut state, id);
    (state, id)
}

fn observations_of(state: &AppState, id: ReconId, image: u32) -> usize {
    state
        .node(id)
        .expect("loaded")
        .recon()
        .point_set
        .tracks
        .iter()
        .filter(|t| t.image_index == image)
        .count()
}

fn newest(state: &AppState) -> &crate::action_log::Entry {
    state.action_log.entries().next_back().expect("an entry")
}

#[test]
fn the_image_is_added_back_to_the_tracks_it_sees_and_undo_takes_it_out() {
    let (mut state, id) = untracked_state();
    let points = state.node(id).unwrap().edited().point_count();
    let versions = state.node(id).unwrap().history.versions().len();
    assert_eq!(observations_of(&state, id, 0), 0);

    state
        .start_add_image_to_tracks(ImageRef::new(id, 0))
        .expect("a posed image of an embedded_patches node with frames");
    assert_eq!(
        state.background_task().map(|task| task.operation.name),
        Some(Operation::ADD_IMAGE_TO_TRACKS.name),
        "the step runs on a worker"
    );
    state.finish_background_task();

    let node = state.node(id).unwrap();
    assert_eq!(node.history.versions().len(), versions + 1, "one version");
    assert_eq!(
        node.edited().point_count(),
        points,
        "no point created or lost"
    );
    let added = observations_of(&state, id, 0);
    assert!(added >= points - 2, "{added} of {points} tracks rejoined");
    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert!(
        entry
            .text
            .starts_with(&format!("Added image_0.jpg to {added} tracks (")),
        "{}",
        entry.text
    );
    assert!(entry.text.contains("candidates refused"), "{}", entry.text);

    state.undo(id).expect("a version to undo");
    assert_eq!(observations_of(&state, id, 0), 0);
}

#[test]
fn an_image_already_in_every_track_adds_nothing_and_pushes_no_version() {
    let (mut state, id) = crate::bench::track_at_pixel::tests::plane_state();
    let versions = state.node(id).unwrap().history.versions().len();
    state
        .start_add_image_to_tracks(ImageRef::new(id, 0))
        .expect("the step can run");
    state.finish_background_task();
    assert_eq!(state.node(id).unwrap().history.versions().len(), versions);
    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert!(
        entry
            .text
            .starts_with("Added image_0.jpg to 0 tracks (0 candidates refused)"),
        "{}",
        entry.text
    );
}

#[test]
fn the_refusals_are_the_greyed_entrys_sentences() {
    // An unposed image.
    let mut recon = without_image(plane_recon(), 0);
    recon.image_table.images[0].translation_xyz.x = f64::NAN;
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.scene[0].id;
    cache_photographs(&mut state, id);
    assert_eq!(
        state
            .add_image_to_tracks_refusal(ImageRef::new(id, 0))
            .as_deref(),
        Some(NOT_POSED_HINT)
    );
    let err = state
        .start_add_image_to_tracks(ImageRef::new(id, 0))
        .unwrap_err();
    assert!(err.ends_with(NOT_POSED_HINT), "{err}");
    assert!(newest(&state).failed);

    // No patch frames.
    let mut recon = without_image(plane_recon(), 0);
    recon.point_set.patch_u_halfvec_xyz = None;
    recon.point_set.patch_v_halfvec_xyz = None;
    recon.point_set.patch_bitmaps_y_x_rgba = None;
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.scene[0].id;
    cache_photographs(&mut state, id);
    assert_eq!(
        state
            .add_image_to_tracks_refusal(ImageRef::new(id, 0))
            .as_deref(),
        Some(NO_FRAMES_HINT)
    );

    // A photograph neither cached nor on disk.
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(without_image(plane_recon(), 0)));
    let id = state.scene[0].id;
    assert_eq!(
        state
            .add_image_to_tracks_refusal(ImageRef::new(id, 0))
            .as_deref(),
        Some(NO_PHOTOGRAPH_HINT)
    );

    // A node with no points.
    let mut recon = plane_recon();
    recon.point_set.points.clear();
    recon.point_set.tracks.clear();
    recon.point_set.observation_counts.clear();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::zeros((0, 2)),
        image_file_hashes: vec![[0u8; 16]; 3],
    };
    recon.point_set.patch_u_halfvec_xyz = Some(Array2::zeros((0, 3)));
    recon.point_set.patch_v_halfvec_xyz = Some(Array2::zeros((0, 3)));
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(ndarray::Array4::zeros((0, 8, 8, 4))));
    recon.metadata.observation_count = 0;
    recon.rebuild_derived_fields();
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.scene[0].id;
    cache_photographs(&mut state, id);
    assert_eq!(
        state
            .add_image_to_tracks_refusal(ImageRef::new(id, 0))
            .as_deref(),
        Some(NO_POINTS_HINT)
    );
}

#[test]
fn a_node_with_an_operation_running_refuses_with_the_busy_sentence() {
    let (mut state, id) = untracked_state();
    state
        .start_add_image_to_tracks(ImageRef::new(id, 0))
        .expect("the step can run");
    let busy = state.busy_refusal(id).expect("the node is locked");
    assert_eq!(
        state.add_image_to_tracks_refusal(ImageRef::new(id, 1)),
        Some(busy)
    );
    state.finish_background_task();
}

#[test]
fn the_outcome_names_the_refusals_most_first() {
    use sfmtool_core::reconstruction::add_image_to_tracks::{CandidateReport, Refusal};
    let candidate = |point: u32, refusal: Option<Refusal>| {
        let mut c = CandidateReport {
            point,
            refusal,
            projection: None,
            references: Vec::new(),
            reference_loo_zncc: Vec::new(),
            reference_pair_zncc: Vec::new(),
            search_keypoint: None,
            keypoint: None,
            offset_px: f64::NAN,
            sigma_pos: f64::NAN,
            peak_zncc: f64::NAN,
            zncc: f64::NAN,
            pair_zncc: Vec::new(),
            judged: f64::NAN,
            bar: f64::NAN,
        };
        c.point = point;
        c
    };
    let report = AddImageToTracksReport {
        image: 0,
        candidates: vec![
            candidate(0, None),
            candidate(1, Some(Refusal::NotInFrame)),
            candidate(2, Some(Refusal::TooFar)),
            candidate(3, Some(Refusal::NotInFrame)),
        ],
        accepted: 1,
        observations_before: 10,
        observations_after: 11,
        pooled_bar: None,
        position_bound_px: None,
    };
    assert_eq!(
        outcome_text("frame_22.jpg", &report),
        "Added frame_22.jpg to 1 tracks (3 candidates refused: 2 not in frame, 1 too far)"
    );
}
