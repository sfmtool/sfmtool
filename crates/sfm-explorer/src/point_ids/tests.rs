// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two walks: which content a point's id names as the node is edited and
//! saved, and which point an id lands on after an undo, an edit over it, or a
//! materialisation.

use std::path::PathBuf;

use sfmtool_core::SfmrReconstruction;

use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;

use super::*;

/// A state holding one demo node, selected.
fn state() -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(64)));
    state
}

/// A directory of this test's own under the system temp dir, emptied first so a
/// rerun does not read a previous run's file.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sfm_explorer_ids_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a writable temp dir");
    dir
}

/// A state holding one node that came from `dir/recon.sfmr`, selected, with its
/// cursor at the version the disk serial names.
fn state_from_file(dir: &std::path::Path) -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        &dir.join("recon.sfmr"),
        SfmrReconstruction::demo(64),
    ));
    state
}

/// The id of the one node.
fn id(state: &AppState) -> ReconId {
    state.selected_recon.expect("a selected reconstruction")
}

/// The hash of the base the node's disk serial names, eight digits.
fn disk_hash(state: &AppState) -> String {
    let node = &state.scene[0];
    let serial = node.history.disk_serial();
    let value = node
        .history
        .versions()
        .iter()
        .find(|v| v.serial == serial)
        .and_then(|v| v.value.as_ref())
        .expect("a version the budget has not released");
    base_hash_prefix(value).expect("a hashable reconstruction")
}

/// The hash the ids of `state`'s node are minted against at the moment of the
/// call, which is its current base's.
fn current_hash(state: &AppState) -> String {
    base_hash_prefix(state.scene[0].edited()).expect("a hashable demo reconstruction")
}

#[test]
fn demo_data_has_a_hash_like_any_other_reconstruction() {
    // The whole point of hashing the value rather than reading the file's
    // stored hash: a node that came from no file is not in the `00000000`
    // state, so its ids are ids.
    let state = state();
    let hash = current_hash(&state);
    assert_eq!(hash.len(), HASH_PREFIX_LEN);
    assert_ne!(hash, "00000000");
    assert!(hash.bytes().all(|b| b.is_ascii_hexdigit()));
}

#[test]
fn an_id_is_a_hash_and_a_row_and_nothing_else() {
    // Nothing in an id names a node: a point's identity is its content hash and
    // its row there, which is the same pair in every node holding that content.
    let state = state();
    let minted = mint(&state.scene[0], 7).expect("a live point");
    assert_eq!(minted, format!("pt3d_{}_7", current_hash(&state)));
}

#[test]
fn the_disk_version_is_what_an_id_names_across_a_point_edit() {
    // A point edit leaves indexes alone, so the id is unchanged in both halves
    // — but it is the *content* it is minted against that must not move, and
    // that is what the assertion is about.
    let mut state = state();
    let node_id = id(&state);
    let before = mint(&state.scene[0], 9).expect("a live point");

    state
        .delete_point(PointRef::new(node_id, 3))
        .expect("a live point");

    assert_eq!(mint(&state.scene[0], 9).as_deref(), Some(before.as_str()));
    assert_eq!(resolve(&state.scene[0], &current_hash(&state), 9), Ok(9));
}

#[test]
fn an_unsaved_materialisation_that_renumbers_keeps_the_loaded_id() {
    // Delete an early point, then a bulk edit, which materialises and shifts
    // every point after the deletion down by one. Nothing was saved, so the
    // version on disk is still the loaded one, and the id shown for the shifted
    // point stays the one it had in that file's own numbering.
    let mut state = state();
    let node_id = id(&state);
    let first_hash = current_hash(&state);
    let before = mint(&state.scene[0], 40).expect("a live point");
    assert!(before.starts_with(&format!("pt3d_{first_hash}_40")));

    state
        .delete_point(PointRef::new(node_id, 3))
        .expect("a live point");
    state
        .delete_image(ImageRef::new(node_id, 0))
        .expect("a deletable image");

    // The base has changed, so a coordinate in it would be a different number.
    let now_hash = current_hash(&state);
    assert_ne!(now_hash, first_hash);
    // Wherever the point ended up, the id it is shown under is the old one.
    let moved = resolve(&state.scene[0], &first_hash, 40).expect("the point survived");
    assert_eq!(
        mint(&state.scene[0], moved).as_deref(),
        Some(before.as_str())
    );
}

#[test]
fn an_id_resolves_through_a_branch_an_undo_discarded() {
    // Copy an id, delete the point's neighbour, undo, make a different edit —
    // the id was minted on a version that is now on no line of descent, and it
    // still lands on its point.
    let mut state = state();
    let node_id = id(&state);
    let hash = current_hash(&state);

    state
        .delete_point(PointRef::new(node_id, 3))
        .expect("a live point");
    state.undo(node_id).expect("one edit to undo");
    state
        .delete_point(PointRef::new(node_id, 5))
        .expect("a live point");

    assert_eq!(resolve(&state.scene[0], &hash, 9), Ok(9));
}

#[test]
fn resolving_a_deleted_point_names_the_version_it_stopped_at() {
    let mut state = state();
    let node_id = id(&state);
    let hash = current_hash(&state);

    state
        .delete_point(PointRef::new(node_id, 3))
        .expect("a live point");

    let error = resolve(&state.scene[0], &hash, 3).expect_err("the point is gone");
    assert!(error.contains("v"), "{error}");
    assert!(error.contains("not in"), "{error}");
}

#[test]
fn an_unknown_hash_says_so_rather_than_resolving_somewhere_else() {
    let state = state();
    let error = resolve(&state.scene[0], "0badf00d", 0).expect_err("no such content");
    assert!(error.contains("never held content"), "{error}");
    assert!(!holds_hash(&state.scene[0], "0badf00d"));
    assert!(holds_hash(&state.scene[0], &current_hash(&state)));
}

// ── Which content an id names: the disk version, then the earliest ──────

#[test]
fn a_save_moves_the_id_onto_the_file_that_was_just_written() {
    // Loaded, edited, saved. The point is a row of the file now on disk, so its
    // id is that file's hash and that row: a reader of the saved file resolves
    // it with nothing else to hand.
    let dir = temp_dir("after_save");
    let mut state = state_from_file(&dir);
    let node_id = id(&state);
    let loaded_hash = disk_hash(&state);
    assert_eq!(
        mint(&state.scene[0], 40).as_deref(),
        Some(format!("pt3d_{loaded_hash}_40").as_str())
    );

    // Deleting an earlier point moves this one, so the save changes both halves
    // of its id and neither can be right by accident.
    state
        .delete_point(PointRef::new(node_id, 3))
        .expect("a live point");
    state.save_node(node_id).expect("a writable path");

    let saved_hash = disk_hash(&state);
    assert_ne!(saved_hash, loaded_hash);
    assert_eq!(
        mint(&state.scene[0], 39).as_deref(),
        Some(format!("pt3d_{saved_hash}_39").as_str())
    );
    // And the id taken before the save still resolves, through the lineage the
    // save recorded.
    assert_eq!(resolve(&state.scene[0], &loaded_hash, 40), Ok(39));
}

#[test]
fn an_undo_past_a_save_puts_the_id_back_on_the_loaded_file() {
    // The cursor is on a version the saved one is not an ancestor of, so the
    // point's identity does not reach the disk version and the earliest rule
    // takes over: the file the node was loaded from.
    let dir = temp_dir("undo_past_save");
    let mut state = state_from_file(&dir);
    let node_id = id(&state);
    let loaded_hash = disk_hash(&state);

    state
        .delete_point(PointRef::new(node_id, 3))
        .expect("a live point");
    state.save_node(node_id).expect("a writable path");
    let saved_hash = disk_hash(&state);
    assert_eq!(
        mint(&state.scene[0], 39).as_deref(),
        Some(format!("pt3d_{saved_hash}_39").as_str())
    );

    // Back to the value before the save materialised, where indexes are the
    // loaded file's again.
    state.undo(node_id).expect("the save's materialisation");

    assert_eq!(
        mint(&state.scene[0], 40).as_deref(),
        Some(format!("pt3d_{loaded_hash}_40").as_str())
    );
}

#[test]
fn a_point_created_since_the_last_save_is_named_by_the_edit_that_made_it() {
    // It is a row of no base, the one on disk included, so there is nothing for
    // the disk rule to name it with and the creating edit's own hash is the
    // earliest content its identity reaches.
    let dir = temp_dir("created");
    let mut state = state_from_file(&dir);
    let record = state.scene[0]
        .edited()
        .point(5)
        .expect("a live point")
        .to_record();
    let edit_hash = state.scene[0]
        .edited()
        .point_edit_hash(std::slice::from_ref(&record))
        .expect("a hashable base");

    // The shape a point-creating edit pushes: the same base, an overlay one
    // point bigger, indexes stable across the step (so nothing stopped
    // resolving), and the created index named on the version.
    let node = &mut state.scene[0];
    let mut next = node.history.current().clone();
    let added = next.add_point(record).expect("a well-formed record");
    node.history.push_creating(
        next,
        crate::document::PointMap::Removed(Vec::new()),
        "Added a point",
        Some(crate::document::CreatedPoints {
            hash: edit_hash.clone(),
            indexes: vec![added],
        }),
    );

    // Position zero among that edit's creations, under the edit's hash.
    assert_eq!(
        mint(&state.scene[0], added).as_deref(),
        Some(format!("pt3d_{}_0", &edit_hash[..HASH_PREFIX_LEN]).as_str())
    );
    // A point that *is* on disk is unaffected: the two rules coexist per point.
    assert_eq!(
        mint(&state.scene[0], 40).as_deref(),
        Some(format!("pt3d_{}_40", disk_hash(&state)).as_str())
    );
    // And the edit hash resolves back to the point it named.
    assert_eq!(
        resolve(&state.scene[0], &edit_hash[..HASH_PREFIX_LEN], 0),
        Ok(added)
    );
}
