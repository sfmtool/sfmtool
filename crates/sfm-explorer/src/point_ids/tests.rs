// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two walks: which id a point is shown under after an edit, and which
//! point an id lands on after an undo, an edit over it, or a materialisation.

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

/// The id of the one node.
fn id(state: &AppState) -> ReconId {
    state.selected_recon.expect("a selected reconstruction")
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
fn an_id_is_the_session_form_and_carries_the_node() {
    let state = state();
    let node = &state.scene[0];
    let minted = mint(node, 7).expect("a live point");
    assert_eq!(
        minted,
        format!("pt3d_{}_7_n{}", current_hash(&state), node.id.raw())
    );
}

#[test]
fn the_earliest_rule_keeps_an_id_across_a_point_edit() {
    // A point edit leaves indexes alone, so the id is unchanged in both halves
    // — but it is the *base* it is minted against that must not move, and that
    // is what the assertion is about.
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
fn the_earliest_rule_survives_a_materialisation_that_renumbers() {
    // Delete an early point, then a bulk edit, which materialises and shifts
    // every point after the deletion down by one. The id shown for the shifted
    // point stays the one it had in the file's own numbering.
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

#[test]
fn a_node_suffix_parses_and_a_bad_one_does_not() {
    assert_eq!(parse_node_suffix("n3"), Some(3));
    assert_eq!(parse_node_suffix("n0"), Some(0));
    assert_eq!(parse_node_suffix("3"), None);
    assert_eq!(parse_node_suffix("nx"), None);
}
