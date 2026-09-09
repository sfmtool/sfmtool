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
