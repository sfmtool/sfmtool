// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two edits end to end: what each does to the node's value, what the
//! selection does across it and back, and what the Action Log says.
//!
//! The GPU consequences are asserted separately, in
//! `scene_renderer/upload/tests.rs`, against a real `wgpu` device.

use std::sync::Arc;

use sfmtool_core::SfmrReconstruction;

use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;

/// A state holding one demo node, selected.
fn state() -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(64)));
    state
}

/// The id of the one node.
fn node(state: &AppState) -> ReconId {
    state.selected_recon.expect("a selected reconstruction")
}

/// The Action Log's texts, oldest first.
fn texts(state: &AppState) -> Vec<String> {
    state
        .action_log
        .entries()
        .map(|entry| entry.text.clone())
        .collect()
}

// ── Delete point: the point edit ────────────────────────────────────────

#[test]
fn deleting_a_point_leaves_the_base_alone_and_costs_one_index() {
    let mut state = state();
    let id = node(&state);
    let before = Arc::clone(&state.scene[0].edited().base);
    let count = state.scene[0].point_count();

    state.selected_point = Some(PointRef::new(id, 7));
    state
        .delete_point(PointRef::new(id, 7))
        .expect("a live point");

    let node = &state.scene[0];
    assert!(
        Arc::ptr_eq(&before, &node.edited().base),
        "a point edit wrote through the base"
    );
    assert_eq!(node.point_count(), count - 1);
    assert!(node.is_point_deleted(7));
    assert!(!node.is_point_deleted(8), "an index after the hole moved");
    assert_eq!(node.history.versions().len(), 2);
}

#[test]
fn deleting_the_selected_point_clears_the_selection_and_an_undo_does_not_guess() {
    let mut state = state();
    let id = node(&state);
    state.selected_point = Some(PointRef::new(id, 3));

    state.delete_selected_point().expect("a live point");
    assert_eq!(
        state.selected_point, None,
        "a deleted point stayed selected"
    );

    state.undo(id).expect("one edit to undo");
    assert!(!state.scene[0].is_point_deleted(3));
    // The selection is not restored: the map says where an index went, not
    // what the user was looking at.
    assert_eq!(state.selected_point, None);
}

#[test]
fn a_surviving_selection_keeps_its_index_across_a_point_edit() {
    let mut state = state();
    let id = node(&state);
    state.selected_point = Some(PointRef::new(id, 9));
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");
    assert_eq!(state.selected_point, Some(PointRef::new(id, 9)));
}

#[test]
fn deleting_a_point_twice_is_refused_and_leaves_the_history_alone() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    let versions = state.scene[0].history.versions().len();
    assert!(state.delete_point(PointRef::new(id, 1)).is_err());
    assert_eq!(state.scene[0].history.versions().len(), versions);
}

// ── Delete image: the bulk edit ─────────────────────────────────────────

#[test]
fn deleting_an_image_produces_a_new_base_with_an_empty_overlay() {
    let mut state = state();
    let id = node(&state);
    let before = Arc::clone(&state.scene[0].edited().base);
    let images = state.scene[0].image_count();
    let name = state.scene[0].recon().image_table.images[1].name.clone();

    state.delete_image(ImageRef::new(id, 1)).expect("an image");

    let node = &state.scene[0];
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "the base is the same value"
    );
    assert!(node.edited().deleted_points.is_empty());
    assert!(node.edited().added.points.is_empty());
    assert_eq!(node.image_count(), images - 1);
    assert!(node
        .recon()
        .image_table
        .images
        .iter()
        .all(|image| image.name != name));
    assert_eq!(node.id, id, "a node's identity did not survive an edit");
}

#[test]
fn deleting_an_image_after_a_point_edit_folds_the_overlay_in() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 0))
        .expect("a live point");
    let after_point_edit = state.scene[0].point_count();

    state.delete_image(ImageRef::new(id, 0)).expect("an image");

    let node = &state.scene[0];
    assert!(node.edited().deleted_points.is_empty());
    // The materialisation closed the hole the point edit made, so the new
    // base holds at most what the overlay did.
    assert!(node.point_count() <= after_point_edit);
    assert_eq!(node.history.versions().len(), 3);
}

#[test]
fn the_selection_follows_the_renumbering_and_comes_back_through_the_undo() {
    let mut state = state();
    let id = node(&state);
    // A point the deleted image does not orphan, so it survives the edit.
    let survivor = surviving_point(&state, 0);
    let position = state.scene[0].recon().point_set.points[survivor as usize].position;
    state.selected_point = Some(PointRef::new(id, survivor as usize));

    state.delete_image(ImageRef::new(id, 0)).expect("an image");
    let moved = state.selected_point.expect("the point survived");
    assert_eq!(
        state.scene[0].recon().point_set.points[moved.index()].position,
        position,
        "the selection followed the map onto a different point"
    );

    state.undo(id).expect("one edit to undo");
    assert_eq!(
        state.selected_point,
        Some(PointRef::new(id, survivor as usize)),
        "the selection did not come back through the map"
    );
}

#[test]
fn the_last_image_cannot_be_deleted() {
    let mut state = AppState::new();
    let mut recon = SfmrReconstruction::demo(16);
    recon = recon
        .subset_by_image_indices(&[0], false)
        .expect("a one-image subset");
    state.append_node(SceneNode::demo(recon));
    let id = node(&state);
    assert!(state.delete_image(ImageRef::new(id, 0)).is_err());
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

/// A point of `state`'s node that image `image` does not orphan: it has an
/// observation somewhere else.
fn surviving_point(state: &AppState, image: u32) -> u32 {
    let recon = state.scene[0].recon();
    let mut elsewhere = vec![false; recon.point_count()];
    for observation in &recon.point_set.tracks {
        if observation.image_index != image {
            elsewhere[observation.point_index as usize] = true;
        }
    }
    elsewhere
        .iter()
        .position(|&alive| alive)
        .expect("some point is seen by another image") as u32
}

// ── Undo, redo and the log ──────────────────────────────────────────────

#[test]
fn undo_and_redo_walk_the_versions_and_refuse_at_the_ends() {
    let mut state = state();
    let id = node(&state);
    assert!(!state.can_undo(id) && !state.can_redo(id));

    state
        .delete_point(PointRef::new(id, 4))
        .expect("a live point");
    assert!(state.can_undo(id) && !state.can_redo(id));

    state.undo(id).expect("one edit to undo");
    assert!(!state.can_undo(id) && state.can_redo(id));
    assert!(state.undo(id).is_err());
    assert!(!state.scene[0].is_point_deleted(4));

    state.redo(id).expect("a redo tail");
    assert!(state.scene[0].is_point_deleted(4));
    assert!(state.redo(id).is_err());
}

#[test]
fn every_edit_undo_and_redo_writes_one_log_entry_naming_the_node_and_the_serials() {
    let mut state = state();
    let id = node(&state);
    state.action_log.clear();

    state
        .delete_point(PointRef::new(id, 5))
        .expect("a live point");
    state.undo(id).expect("one edit to undo");
    state.redo(id).expect("a redo tail");

    let texts = texts(&state);
    assert_eq!(texts.len(), 3, "{texts:?}");
    assert!(
        texts[0].starts_with("Deleted point 5 in demo ("),
        "{}",
        texts[0]
    );
    assert!(
        texts[1].starts_with("Undo: Deleted point 5 in demo ("),
        "{}",
        texts[1]
    );
    assert!(
        texts[2].starts_with("Redo: Deleted point 5 in demo ("),
        "{}",
        texts[2]
    );
    // Each entry names the step it took, as `vN → vM`.
    for text in &texts {
        assert!(text.contains(" → v"), "{text}");
    }
    assert!(state
        .action_log
        .entries()
        .all(|entry| entry.kind == crate::action_log::Kind::Edit));
}

#[test]
fn an_edit_after_an_undo_discards_the_redo_tail() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");
    state.undo(id).expect("one edit to undo");
    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");

    assert!(!state.can_redo(id));
    let node = &state.scene[0];
    assert!(node.is_point_deleted(1));
    assert!(!node.is_point_deleted(2), "the discarded version came back");
    assert!(node.is_point_deleted(3));
}

// ── Jumping to a version ────────────────────────────────────────────────

/// The serial of the node's version at `position`.
fn serial_at(state: &AppState, position: usize) -> crate::document::VersionSerial {
    state.scene[0].history.versions()[position].serial
}

#[test]
fn a_jump_back_lands_on_the_version_a_run_of_undos_would_have() {
    let mut state = state();
    let id = node(&state);
    for point in [1u32, 2, 3] {
        state
            .delete_point(PointRef::new(id, point as usize))
            .expect("a live point");
    }
    let first = serial_at(&state, 1);

    state.jump_to_version(id, first).expect("a live version");

    let node = &state.scene[0];
    assert_eq!(node.history.current_version().serial, first);
    assert!(node.is_point_deleted(1));
    assert!(!node.is_point_deleted(2));
    assert!(!node.is_point_deleted(3));
    // The versions themselves did not move: a jump walks the cursor.
    assert_eq!(node.history.versions().len(), 4);
    assert_eq!(node.history.cursor(), 1);
}

#[test]
fn a_jump_forward_returns_to_the_version_it_came_from() {
    let mut state = state();
    let id = node(&state);
    for point in [1u32, 2, 3] {
        state
            .delete_point(PointRef::new(id, point as usize))
            .expect("a live point");
    }
    let last = serial_at(&state, 3);
    let first = serial_at(&state, 1);
    state.jump_to_version(id, first).expect("a live version");

    state.jump_to_version(id, last).expect("a live version");

    let node = &state.scene[0];
    assert_eq!(node.history.current_version().serial, last);
    for point in [1u32, 2, 3] {
        assert!(node.is_point_deleted(point));
    }
}

#[test]
fn a_jump_takes_the_selection_through_every_step_it_passes() {
    let mut state = state();
    let id = node(&state);
    let survivor = surviving_point(&state, 0);
    let position = state.scene[0].recon().point_set.points[survivor as usize].position;
    state.selected_point = Some(PointRef::new(id, survivor as usize));
    let loaded = serial_at(&state, 0);

    // A point edit and then a bulk edit, so the jump back composes a
    // stable-index step with a renumbering one.
    state
        .delete_point(PointRef::new(id, survivor as usize + 1))
        .expect("a live point");
    state.delete_image(ImageRef::new(id, 0)).expect("an image");
    let moved = state.selected_point.expect("the point survived");
    assert_eq!(
        state.scene[0].recon().point_set.points[moved.index()].position,
        position
    );

    state.jump_to_version(id, loaded).expect("a live version");

    assert_eq!(
        state.selected_point,
        Some(PointRef::new(id, survivor as usize)),
        "the jump did not compose the maps a run of undos would have"
    );
}

#[test]
fn a_jump_writes_one_log_entry_naming_the_two_serials() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");
    let from = serial_at(&state, 2);
    let to = serial_at(&state, 0);
    state.action_log.clear();

    state.jump_to_version(id, to).expect("a live version");

    let texts = texts(&state);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(texts[0], format!("Go to: Opened demo ({from} → {to})"));
    assert!(state
        .action_log
        .entries()
        .all(|entry| entry.kind == crate::action_log::Kind::Edit));
}

#[test]
fn a_jump_onto_a_released_version_is_refused_whole() {
    let mut state = state();
    let id = node(&state);
    for point in [1u32, 2, 3] {
        state
            .delete_point(PointRef::new(id, point as usize))
            .expect("a live point");
    }
    let released = serial_at(&state, 1);
    let earliest = serial_at(&state, 0);
    state.scene[0].history.versions_mut_for_test()[1].value = None;
    state.action_log.clear();

    // The released version itself, and any version the walk would pass
    // through it to reach.
    assert!(state.jump_to_version(id, released).is_err());
    assert!(state.jump_to_version(id, earliest).is_err());
    assert_eq!(
        state.scene[0].history.cursor(),
        3,
        "the cursor moved anyway"
    );
    assert_eq!(texts(&state), Vec::<String>::new());
}

#[test]
fn a_jump_to_where_the_cursor_already_is_or_to_no_version_is_refused() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    let here = serial_at(&state, 1);
    assert!(state.jump_to_version(id, here).is_err());

    // A serial of another node's history is not one of this node's versions.
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(8)));
    let other = state.scene[1].history.current_version().serial;
    assert!(state.jump_to_version(id, other).is_err());
}

// ── Add observation: the point edit that creates structure ──────────────

/// A demo node whose value is `embedded_patches`, which is the only mode this
/// edit is defined on. No images behind it, so the fit itself is exercised in
/// `sfmtool-core` and through the Python binding rather than here.
fn embedded_state() -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(embedded_demo(64)));
    state
}

/// The demo reconstruction with its observations rewritten as inline keypoints
/// and a patch frame per point.
///
/// Built here rather than through `to_embedded_patches`, which reads the `.sift`
/// companions the demo value has no files for.
pub(crate) fn embedded_demo(points: usize) -> SfmrReconstruction {
    use ndarray::Array2;
    use sfmtool_core::ObservationSource;

    let mut recon = SfmrReconstruction::demo(points);
    let images = recon.image_count();
    let observations = recon.point_set.tracks.len();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::<f32>::from_shape_fn((observations, 2), |(i, c)| {
            (i % 64) as f32 + c as f32
        }),
        image_file_hashes: vec![[0u8; 16]; images],
    };
    let p = recon.point_count();
    let mut u = Array2::<f32>::zeros((p, 3));
    let mut v = Array2::<f32>::zeros((p, 3));
    for i in 0..p {
        u[[i, 0]] = 0.02;
        v[[i, 1]] = 0.02;
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.metadata.feature_source = "embedded_patches".to_string();
    recon.rebuild_derived_fields();
    recon
}

#[test]
fn a_sift_files_node_refuses_the_edit() {
    let mut state = state();
    let id = node(&state);
    let why = state
        .add_observation_at(PointRef::new(id, 3), ImageRef::new(id, 5), [10.0, 10.0])
        .expect_err("a sift_files node has no room for a featureless observation");
    assert!(why.contains("embedded_patches"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

#[test]
fn an_image_already_in_the_track_refuses_the_edit() {
    let mut state = embedded_state();
    let id = node(&state);
    let seen = state.scene[0].edited().track_image_indices(3)[0];
    let why = state
        .add_observation_at(PointRef::new(id, 3), ImageRef::new(id, seen), [1.0, 1.0])
        .expect_err("that image already observes the point");
    assert!(why.contains("already observes"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

#[test]
fn the_edit_refuses_without_a_selected_point() {
    let state = embedded_state();
    let edited = state.scene[0].edited();
    let entry = crate::image_detail::add_observation_entry(edited, 0, None)
        .expect("an embedded_patches node offers the entry");
    assert!(entry.is_err(), "the entry should be greyed with no point");
}

#[test]
fn the_entry_is_absent_on_a_sift_files_node() {
    let state = state();
    let edited = state.scene[0].edited();
    assert!(
        crate::image_detail::add_observation_entry(edited, 0, Some(3)).is_none(),
        "the entry has no meaning on a sift_files node"
    );
}

#[test]
fn the_entry_is_offered_for_an_image_outside_the_track() {
    let state = embedded_state();
    let edited = state.scene[0].edited();
    let seen: Vec<usize> = edited.track_image_indices(3);
    let unseen = (0..edited.image_count())
        .find(|i| !seen.contains(i))
        .expect("the demo track does not span every image");
    assert_eq!(
        crate::image_detail::add_observation_entry(edited, unseen, Some(3)),
        Some(Ok(())),
    );
}

// ── Create point: the point edit that creates a point ───────────────────

/// The image every create-point test points into, and the pixel it points at.
const CREATE_IMAGE: usize = 0;
const CREATE_PIXEL: [f32; 2] = [10.0, 12.0];

/// [`embedded_state`] with a synthetic photograph cached for `CREATE_IMAGE`, so
/// the edit's decode finds pixels without a file on disk.
fn creatable_state() -> (AppState, ReconId) {
    let mut state = embedded_state();
    let id = node(&state);
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    let data = (0..(w * h * 3))
        .map(|i| (i % 251) as u8)
        .collect::<Vec<u8>>();
    state.full_res_cache.insert(
        ImageRef::new(id, CREATE_IMAGE),
        Some(sfmtool_core::camera::remap::ImageU8::new(w, h, 3, data)),
    );
    (state, id)
}

#[test]
fn creating_a_point_appends_a_bearing_and_selects_it() {
    let (mut state, id) = creatable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    let count = state.scene[0].point_count();

    state
        .create_point(ImageRef::new(id, CREATE_IMAGE), CREATE_PIXEL, 6.0)
        .expect("a pixel on the sensor of a decodable image");

    let node = &state.scene[0];
    assert!(
        Arc::ptr_eq(&before, &node.edited().base),
        "a point edit wrote through the base"
    );
    assert_eq!(node.point_count(), count + 1);
    assert_eq!(node.history.versions().len(), 2);

    let selected = state.selected_point.expect("the created point is selected");
    assert_eq!(selected.recon, id);
    let view = state.scene[0]
        .edited()
        .point(selected.point)
        .expect("the created point");
    assert_eq!(view.point().w, 0.0, "one sighting fixes no distance");
    assert_eq!(view.observations().len(), 1);
    assert_eq!(view.observations()[0].image_index, CREATE_IMAGE as u32);

    let log = texts(&state);
    let last = log.last().expect("one entry per edit");
    assert!(last.contains("Created point"), "{last}");
    assert!(last.contains("radius 6.0 px"), "{last}");
}

#[test]
fn a_sift_files_node_refuses_creating_a_point() {
    let mut state = state();
    let id = node(&state);
    let why = state
        .create_point(ImageRef::new(id, 0), CREATE_PIXEL, 6.0)
        .expect_err("a sift_files node has no room for a featureless observation");
    assert!(why.contains("embedded_patches"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

#[test]
fn the_created_points_id_is_minted_against_the_point_edit() {
    let (mut state, id) = creatable_state();
    let base_prefix =
        crate::point_ids::base_hash_prefix(state.scene[0].edited()).expect("a hashable base");
    state
        .create_point(ImageRef::new(id, CREATE_IMAGE), CREATE_PIXEL, 6.0)
        .expect("a pixel on the sensor of a decodable image");

    let index = state.selected_point.expect("selected").point;
    let node = &state.scene[0];
    let minted = crate::point_ids::mint(node, index).expect("the created point has an id");
    let rest = minted.strip_prefix("pt3d_").expect("the id's one form");
    let (hash, k) = rest.split_once('_').expect("hash and index");
    assert_ne!(
        hash, base_prefix,
        "a point no base holds cannot be named by a base's hash"
    );
    assert_eq!(k, "0", "it is the edit's first creation");
    assert_eq!(
        crate::point_ids::resolve(node, hash, 0),
        Ok(index),
        "the id resolves back to the point it names"
    );
}

#[test]
fn an_undo_drops_the_created_point_and_the_selection() {
    let (mut state, id) = creatable_state();
    let count = state.scene[0].point_count();
    state
        .create_point(ImageRef::new(id, CREATE_IMAGE), CREATE_PIXEL, 6.0)
        .expect("a pixel on the sensor of a decodable image");

    state.undo(id).expect("one edit to undo");
    assert_eq!(state.scene[0].point_count(), count);
    assert_eq!(
        state.selected_point, None,
        "the selection sat on a point this version does not hold"
    );
}

#[test]
fn the_prompts_radius_is_the_median_the_reconstruction_already_uses() {
    let (mut state, id) = creatable_state();
    let image = ImageRef::new(id, CREATE_IMAGE);
    let derived = state.create_point_default_radius(image);
    assert!(
        derived > 0.0 && derived != super::FALLBACK_PATCH_RADIUS_PX,
        "a node with patch frames derives its own radius, got {derived}"
    );

    // Once a point has been created by hand, the radius that made it is what
    // the next prompt offers.
    state
        .create_point(image, CREATE_PIXEL, 3.5)
        .expect("a pixel on the sensor of a decodable image");
    assert_eq!(state.create_point_radius, Some(3.5));
    assert!(state.create_point_prompt.is_none(), "the prompt is closed");
}

#[test]
fn a_node_with_no_patch_frames_falls_back_to_the_named_radius() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(8)));
    let id = node(&state);
    assert_eq!(
        state.create_point_default_radius(ImageRef::new(id, 0)),
        super::FALLBACK_PATCH_RADIUS_PX
    );
}

// ── Remove observation: the point edit that shortens a track ────────────

/// [`embedded_demo`] with every keypoint the exact projection of its point, and
/// track lengths of one, two and three cycling by point index.
///
/// The projections are what make the re-triangulation well-conditioned, and the
/// three lengths are the three outcomes: point 0 is deleted by the edit, point 1
/// becomes a bearing, point 2 stays finite and is re-solved.
fn projected_embedded_demo(points: usize) -> SfmrReconstruction {
    use ndarray::Array2;
    use sfmtool_core::{ObservationSource, TrackObservation};

    let mut recon = embedded_demo(points);
    let tracks: Vec<TrackObservation> = (0..points)
        .flat_map(|p| {
            (0..=(p % 3)).map(move |image| TrackObservation {
                image_index: image as u32,
                point_index: p as u32,
            })
        })
        .collect();
    let mut keypoints = Array2::<f32>::zeros((tracks.len(), 2));
    for (row, track) in tracks.iter().enumerate() {
        let image = &recon.image_table.images[track.image_index as usize];
        let point = recon.point_set.points[track.point_index as usize].position;
        let cam = image.quaternion_wxyz.to_rotation_matrix() * point.coords + image.translation_xyz;
        let (x, y) = recon.image_table.cameras[image.camera_index as usize]
            .ray_to_pixel([cam.x, cam.y, cam.z])
            .expect("every demo camera sees every demo point");
        keypoints[[row, 0]] = x as f32;
        keypoints[[row, 1]] = y as f32;
    }
    let images = recon.image_count();
    recon.point_set.observation_counts = (0..points).map(|p| (p % 3) as u32 + 1).collect();
    recon.point_set.tracks = tracks;
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0u8; 16]; images],
    };
    recon.rebuild_derived_fields();
    recon
}

/// A state holding one node of [`projected_embedded_demo`], selected.
fn removable_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    (state, id)
}

#[test]
fn removing_from_a_longer_track_moves_the_selection_with_the_point() {
    let (mut state, id) = removable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    state.selected_point = Some(PointRef::new(id, 2));
    state.action_log.clear();

    state
        .remove_observation(PointRef::new(id, 2), ImageRef::new(id, 1))
        .expect("image 1 observes point 2");

    let node = &state.scene[0];
    assert!(
        Arc::ptr_eq(&before, &node.edited().base),
        "a point edit wrote through the base"
    );
    assert_eq!(node.history.versions().len(), 2);
    let moved = state.selected_point.expect("the point is still selected");
    assert_ne!(moved.point, 2, "a modification takes a new index");
    let view = node.edited().point(moved.point).expect("the moved point");
    assert_eq!(view.observations().len(), 2);
    assert_eq!(view.point().w, 1.0, "two rays still state a depth");

    let label = &node.history.current_version().label;
    assert!(
        label.starts_with("Removed observation of point 2 in image_001.jpg"),
        "{label}"
    );
    let texts = texts(&state);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert!(texts[0].contains("2 observations left"), "{}", texts[0]);
}

#[test]
fn removing_down_to_one_view_leaves_a_bearing_under_the_selection() {
    let (mut state, id) = removable_state();
    state.selected_point = Some(PointRef::new(id, 1));
    state.action_log.clear();

    state
        .remove_observation(PointRef::new(id, 1), ImageRef::new(id, 1))
        .expect("image 1 observes point 1");

    let moved = state.selected_point.expect("the point is still selected");
    let view = state.scene[0]
        .edited()
        .point(moved.point)
        .expect("the moved point");
    assert_eq!(view.observations().len(), 1);
    assert_eq!(view.point().w, 0.0, "one sighting fixes a bearing");
    assert!(
        texts(&state)[0].contains("bearing at infinity"),
        "{:?}",
        texts(&state)
    );
}

#[test]
fn removing_the_last_observation_deletes_the_point_and_clears_the_selection() {
    let (mut state, id) = removable_state();
    let count = state.scene[0].point_count();
    state.selected_point = Some(PointRef::new(id, 0));
    state.action_log.clear();

    state
        .remove_observation(PointRef::new(id, 0), ImageRef::new(id, 0))
        .expect("image 0 is point 0's only observation");

    assert_eq!(
        state.selected_point, None,
        "a deleted point stayed selected"
    );
    let node = &state.scene[0];
    assert!(node.is_point_deleted(0));
    assert_eq!(node.point_count(), count - 1);
    assert!(
        texts(&state)[0].contains("is deleted"),
        "{:?}",
        texts(&state)
    );
}

#[test]
fn an_undo_puts_the_observation_back() {
    let (mut state, id) = removable_state();
    let before = state.scene[0]
        .edited()
        .point(2)
        .expect("a live point")
        .point()
        .position;

    state
        .remove_observation(PointRef::new(id, 2), ImageRef::new(id, 1))
        .expect("image 1 observes point 2");
    state.undo(id).expect("one edit to undo");

    let view = state.scene[0].edited().point(2).expect("the point is back");
    assert_eq!(view.observations().len(), 3);
    assert_eq!(view.point().position, before);
    assert_eq!(state.scene[0].history.cursor(), 0);
}

#[test]
fn a_sift_files_node_allows_the_edit() {
    let mut state = state();
    let id = node(&state);
    let seen = state.scene[0].edited().track_image_indices(3)[0];
    state.selected_point = Some(PointRef::new(id, 3));

    state
        .remove_observation(PointRef::new(id, 3), ImageRef::new(id, seen))
        .expect("taking a row out invents no feature");
    assert_eq!(state.scene[0].history.versions().len(), 2);
}

#[test]
fn the_entry_is_greyed_without_a_point_and_for_an_image_outside_the_track() {
    let (state, _) = removable_state();
    let edited = state.scene[0].edited();
    assert!(crate::image_detail::remove_observation_entry(edited, 0, None).is_err());
    let outside = (0..edited.image_count())
        .find(|i| !edited.track_image_indices(2).contains(i))
        .expect("the track does not span every image");
    assert!(crate::image_detail::remove_observation_entry(edited, outside, Some(2)).is_err());
    assert_eq!(
        crate::image_detail::remove_observation_entry(edited, 1, Some(2)),
        Ok(()),
        "image 1 observes point 2"
    );
}

#[test]
fn removing_an_observation_that_is_not_there_is_refused_and_leaves_the_history_alone() {
    let (mut state, id) = removable_state();
    let outside = (0..state.scene[0].edited().image_count())
        .find(|i| !state.scene[0].edited().track_image_indices(2).contains(i))
        .expect("the track does not span every image");
    let why = state
        .remove_observation(PointRef::new(id, 2), ImageRef::new(id, outside))
        .expect_err("that image does not observe the point");
    assert!(why.contains("does not observe"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

// ── Resect in place: the bulk edit that re-poses one image ──────────────

/// A state holding one node a resection can run on, with image 1's stored pose
/// pushed off the truth its own keypoints were computed at.
///
/// The fixture is the Scene Graph tests' own, because "a node a resection can
/// run on" is one thing and there is no second answer to it: the demo camera
/// ring with every observation recomputed as the pixel that camera projects the
/// point to.
fn resectable_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    let image = &mut state.scene[0].recon_mut().image_table.images[1];
    image.translation_xyz += nalgebra::Vector3::new(0.30, -0.20, 0.15);
    (state, id)
}

/// How far image `index`'s camera centre stands from `centre`.
fn centre_offset(state: &AppState, index: usize, centre: nalgebra::Point3<f64>) -> f64 {
    (state.scene[0].recon().image_table.images[index].camera_center() - centre).norm()
}

#[test]
fn resecting_in_place_pushes_a_version_whose_base_is_new_and_whose_images_stay_put() {
    let (mut state, id) = resectable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    let images = state.scene[0].image_count();
    let truth = crate::scene_graph::tests::resectable_node("/runs/truth.sfmr")
        .recon()
        .image_table
        .images[1]
        .camera_center();
    let moved = centre_offset(&state, 1, truth);

    state
        .resect_image_in_place(id, 1, crate::resect::ResectFrom::Observations)
        .expect("the ring corroborates image 1");

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert!(
        node.edited().deleted_points.is_empty() && node.edited().added.points.is_empty(),
        "the new version arrived with an overlay"
    );
    assert_eq!(node.image_count(), images, "the image table moved");
    let recovered = centre_offset(&state, 1, truth);
    assert!(
        recovered < moved * 0.1,
        "the resection left the camera {recovered} from the truth, having started {moved}"
    );
}

#[test]
fn the_action_log_carries_one_entry_naming_the_image_and_the_version() {
    let (mut state, id) = resectable_state();
    let entries = texts(&state).len();
    state
        .resect_image_in_place(id, 1, crate::resect::ResectFrom::Observations)
        .expect("the ring corroborates image 1");
    let logged = texts(&state);
    assert_eq!(logged.len(), entries + 1, "{logged:?}");
    let last = logged.last().expect("one entry");
    assert!(
        last.starts_with("Resected image_001.jpg in place (run_a): 120 pts, inliers "),
        "{last}"
    );
    let serials = state.scene[0].history.versions();
    assert!(
        last.ends_with(&format!(
            "re-triangulated ({} → {})",
            serials[0].serial, serials[1].serial
        )),
        "{last}"
    );
    assert_eq!(
        state.scene[0].history.current_version().label,
        "Resected image_001.jpg in place (run_a)"
    );
}

#[test]
fn an_undo_puts_the_stored_pose_and_the_selection_back() {
    let (mut state, id) = resectable_state();
    let stored = state.scene[0].recon().image_table.images[1].camera_center();
    state.selected_point = Some(PointRef::new(id, 9));

    state
        .resect_image_in_place(id, 1, crate::resect::ResectFrom::Observations)
        .expect("the ring corroborates image 1");
    let selected = state.selected_point.expect("the point survived the edit");
    assert_eq!(selected.recon, id);
    assert!(
        (state.scene[0].recon().image_table.images[1].camera_center() - stored).norm() > 1e-6,
        "the pose did not move"
    );

    state.undo(id).expect("one version to undo");
    assert_eq!(
        state.scene[0].recon().image_table.images[1].camera_center(),
        stored
    );
    assert_eq!(state.selected_point, Some(PointRef::new(id, 9)));
}

#[test]
fn a_resection_that_cannot_be_attempted_pushes_no_version_and_logs_a_failure() {
    let (mut state, id) = resectable_state();
    state.scene[0].recon_mut().image_table.images[1].translation_xyz =
        nalgebra::Vector3::new(f64::NAN, 0.0, 0.0);

    let why = state
        .resect_image_in_place(id, 1, crate::resect::ResectFrom::Observations)
        .expect_err("an unposed target has no pose to re-estimate");

    assert!(why.contains("refused"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    let last = state.action_log.entries().last().expect("one entry");
    assert!(last.failed, "the refusal was logged as a success");
    assert_eq!(last.text, why);
}

#[test]
fn a_refused_estimate_pushes_no_version_and_logs_a_failure() {
    let (mut state, id) = resectable_state();
    // Image 1's keypoints now agree with nothing: the estimate finds
    // correspondences and no consensus among them, which is the refusal that
    // still produces a derived node and must not produce a version.
    {
        let recon = state.scene[0].recon_mut();
        let rows: Vec<usize> = recon
            .point_set
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, o)| o.image_index == 1)
            .map(|(row, _)| row)
            .collect();
        let sfmtool_core::ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
            &mut recon.point_set.observations
        else {
            panic!("the fixture is embedded_patches");
        };
        for (k, row) in rows.iter().enumerate() {
            keypoints_xy[[*row, 0]] = (k % 97) as f32 * 13.0;
            keypoints_xy[[*row, 1]] = (k % 53) as f32 * 11.0;
        }
    }

    let why = state
        .resect_image_in_place(id, 1, crate::resect::ResectFrom::Observations)
        .expect_err("nothing corroborates that pose");

    assert!(why.contains("refused"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!(
        state.action_log.entries().last().expect("one entry").failed,
        "the refusal was logged as a success"
    );
}

// ── Bundle adjust: the bulk edit that moves everything ──────────────────

/// The resectable node with image 1 pushed a few pixels off the pose its own
/// keypoints were computed at.
///
/// A few pixels rather than the resection fixture's tens: the adjustment is a
/// local refinement whose first round trims what disagrees by more than 50 px,
/// so a camera perturbed past that is one whose observations leave the solve
/// rather than one it pulls back.
fn adjustable_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    let image = &mut state.scene[0].recon_mut().image_table.images[1];
    image.translation_xyz += nalgebra::Vector3::new(0.02, -0.015, 0.01);
    (state, id)
}

#[test]
fn bundle_adjusting_pushes_a_version_with_a_new_base_and_keeps_the_images() {
    let (mut state, id) = adjustable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    let images = state.scene[0].image_count();
    let truth = crate::scene_graph::tests::resectable_node("/runs/truth.sfmr")
        .recon()
        .image_table
        .images[1]
        .camera_center();
    let moved = centre_offset(&state, 1, truth);
    state.selected_point = Some(PointRef::new(id, 11));

    state
        .bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("the fixture is well posed");

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert_eq!(node.image_count(), images, "the image table moved");
    assert_eq!(
        node.history.current_version().label,
        "Bundle adjusted run_a"
    );
    // The perturbed camera came back toward the pose its own observations were
    // computed at.
    assert!(centre_offset(&state, 1, truth) < moved, "{moved}");
    // The selection followed the map onto a live point.
    let selected = state.selected_point.expect("the point survived");
    assert_eq!(selected.recon, id);
    assert!(state.scene[0].edited().point(selected.point).is_some());
}

#[test]
fn the_log_entry_carries_the_counts_and_the_residuals() {
    let (mut state, id) = adjustable_state();
    let entries = texts(&state).len();

    state
        .bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("well posed");

    let logged = texts(&state);
    assert_eq!(logged.len(), entries + 1, "{logged:?}");
    let last = logged.last().expect("one entry");
    assert!(
        last.starts_with("Bundle adjusted run_a: 8 images, "),
        "{last}"
    );
    assert!(last.contains("median residual "), "{last}");
    assert!(!last.contains("focal"), "the focal was held: {last}");
    let serials = state.scene[0].history.versions();
    assert!(
        last.ends_with(&format!("({} → {})", serials[0].serial, serials[1].serial)),
        "{last}"
    );
}

#[test]
fn a_released_focal_is_named_in_the_label_and_the_entry() {
    let (mut state, id) = adjustable_state();
    // The demo camera is a two-focal PINHOLE, which the adjustment's focal
    // column is not exact for; a single-focal one of the same geometry is.
    {
        let camera = &mut state.scene[0].recon_mut().image_table.cameras[0];
        let (fx, _) = camera.focal_lengths();
        let (cx, cy) = camera.principal_point();
        camera.model = sfmtool_core::CameraModel::SimplePinhole {
            focal_length: fx,
            principal_point_x: cx,
            principal_point_y: cy,
        };
    }
    let options = sfmtool_core::BundleAdjustOptions {
        opt_f: true,
        ..sfmtool_core::BundleAdjustOptions::default()
    };

    state.bundle_adjust(id, &options).expect("well posed");

    assert_eq!(
        state.scene[0].history.current_version().label,
        "Bundle adjusted run_a, focal released"
    );
    let last = texts(&state).last().expect("one entry").clone();
    assert!(last.contains(", focal "), "{last}");
}

#[test]
fn an_undo_puts_every_pose_back() {
    let (mut state, id) = adjustable_state();
    let before: Vec<nalgebra::Point3<f64>> = state.scene[0]
        .recon()
        .image_table
        .images
        .iter()
        .map(|i| i.camera_center())
        .collect();

    state
        .bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("well posed");
    state.undo(id).expect("one version to undo");

    let after: Vec<nalgebra::Point3<f64>> = state.scene[0]
        .recon()
        .image_table
        .images
        .iter()
        .map(|i| i.camera_center())
        .collect();
    assert_eq!(before, after);
}

#[test]
fn a_node_with_no_inline_keypoints_is_refused_before_anything_is_solved() {
    let mut state = state();
    let id = node(&state);

    let why = state
        .bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect_err("a sift_files demo carries no keypoints");

    assert!(why.contains("inline keypoints"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    let last = state.action_log.entries().last().expect("one entry");
    assert!(last.failed, "the refusal was logged as a success");
    // The menu entry says the same thing, in the same call.
    assert!(crate::bundle_adjust_prompt::refusal(state.scene[0].edited()).is_some());
}

#[test]
fn a_node_whose_images_disagree_about_the_lens_is_refused() {
    let (mut state, id) = adjustable_state();
    {
        let recon = state.scene[0].recon_mut();
        let second = recon.image_table.cameras[0].clone();
        recon.image_table.cameras.push(second);
        recon.image_table.images[2].camera_index = 1;
    }

    let why = state
        .bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect_err("two lenses, one solve");

    assert!(why.contains("one shared camera"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!(crate::bundle_adjust_prompt::refusal(state.scene[0].edited()).is_some());
}

#[test]
fn the_gate_and_the_focal_gate_pass_on_a_node_that_can_be_adjusted() {
    let (state, _) = adjustable_state();
    let edited = state.scene[0].edited();
    assert_eq!(crate::bundle_adjust_prompt::refusal(edited), None);
    // The demo camera is a two-focal PINHOLE, whose focal the adjustment cannot
    // release: the checkbox is greyed and says so.
    assert!(crate::bundle_adjust_prompt::focal_refusal(edited).is_some());
}

// ── Move camera: the bulk edit that moves one pose ──────────────────────

#[test]
fn moving_a_camera_pushes_a_new_base_and_leaves_the_image_table_where_it_was() {
    let (mut state, id) = adjustable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    let images = state.scene[0].image_count();
    let points = state.scene[0].point_count();
    state.selected_point = Some(PointRef::new(id, 11));
    // Back to the pose image 1's own keypoints were computed at, which is what
    // `adjustable_state` moved it off.
    let truth = crate::scene_graph::tests::resectable_node("/runs/truth.sfmr");
    let pose = sfmtool_core::reconstruction::move_camera::pose_of(truth.recon(), 1);

    state
        .move_camera(ImageRef::new(id, 1), &pose)
        .expect("a posed image and a finite pose");

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert_eq!(node.image_count(), images, "the image table moved");
    assert_eq!(node.point_count(), points, "the move deleted a point");
    let label = node.history.current_version().label.clone();
    assert!(label.starts_with("Moved camera "), "{label}");
    assert!(label.contains("points re-solved"), "{label}");
    // The identity map: the selection stays on the point it was on, and the
    // value still holds it.
    let selected = state.selected_point.expect("the point survived");
    assert_eq!(selected, PointRef::new(id, 11));
    assert!(state.scene[0].edited().point(11).is_some());
}

#[test]
fn a_move_of_an_image_that_is_not_there_pushes_no_version_and_logs_a_failure() {
    let (mut state, id) = adjustable_state();
    let pose = sfmtool_core::reconstruction::move_camera::pose_of(state.scene[0].recon(), 0);

    let why = state
        .move_camera(ImageRef::new(id, 99), &pose)
        .expect_err("there is no image 99");

    assert!(why.contains("no longer in the reconstruction"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!(
        state.action_log.entries().last().expect("one entry").failed,
        "the refusal was logged as a success"
    );
}

#[test]
fn an_undo_of_a_move_puts_the_pose_back() {
    let (mut state, id) = adjustable_state();
    let before = state.scene[0].recon().image_table.images[1].camera_center();
    let truth = crate::scene_graph::tests::resectable_node("/runs/truth.sfmr");
    let pose = sfmtool_core::reconstruction::move_camera::pose_of(truth.recon(), 1);

    state
        .move_camera(ImageRef::new(id, 1), &pose)
        .expect("a posed image and a finite pose");
    let moved = state.scene[0].recon().image_table.images[1].camera_center();
    assert!((moved - before).norm() > 1e-6, "the move moved nothing");

    state.undo(id).expect("one edit to undo");
    let restored = state.scene[0].recon().image_table.images[1].camera_center();
    assert!((restored - before).norm() < 1e-12);
}
