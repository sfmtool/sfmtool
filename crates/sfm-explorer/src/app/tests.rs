// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a frame decides before it draws. Headless: these are questions about
//! the state, asked the way the frame asks them.

use super::track_ray_source;
use crate::scene::PointRef;
use crate::state::edits::tests as edits;
use crate::state::AppState;

/// Run the adjustment the way the viewer does, and wait for it.
fn adjust(state: &mut AppState, id: crate::scene::ReconId) {
    state
        .start_bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("the fixture is well posed");
    state.finish_background_task();
}

/// A node with a point worth selecting, and its id.
fn selected(index: u32) -> (AppState, PointRef) {
    let (mut state, id) = edits::adjustable_state();
    let point = PointRef::new(id, index as usize);
    state.select_point(point);
    (state, point)
}

/// The rays are built from a point *and* the value it was read out of, so a
/// new version of the same point is a different source.
///
/// This is the bug it is here for: a bundle adjustment that renumbers nothing
/// leaves the selection exactly where it was and replaces every position under
/// it. Comparing the selection alone said nothing had changed, and the rays
/// stayed drawn from the version before.
#[test]
fn a_new_version_of_the_selected_point_is_a_new_ray_source() {
    let (mut state, point) = selected(0);
    let before = track_ray_source(&state).expect("a selection with a live point");
    assert_eq!(before.0, point);

    let id = point.recon;
    adjust(&mut state, id);

    let after = track_ray_source(&state).expect("the point survived");
    assert_ne!(
        before, after,
        "the rays would have been left on the version before",
    );
    assert_eq!(
        after.0, before.0,
        "this case is the one where the selection does not move",
    );
}

/// Undo is the same question in reverse, and was the same bug.
#[test]
fn undoing_back_to_a_version_is_a_new_ray_source_again() {
    let (mut state, point) = selected(0);
    let id = point.recon;
    adjust(&mut state, id);
    let adjusted = track_ray_source(&state).expect("a live point");

    state.undo(id).expect("there is something to undo");
    let undone = track_ray_source(&state).expect("the point is back");
    assert_ne!(adjusted, undone, "undo left the rays where they were");

    state.redo(id).expect("there is something to redo");
    let redone = track_ray_source(&state).expect("a live point");
    assert_ne!(undone, redone, "redo left the rays where they were");
    assert_eq!(
        adjusted, redone,
        "the same version is the same source, so nothing is rebuilt twice",
    );
}

/// Nothing to draw is a source of its own, so the rays are cleared rather than
/// left behind: no selection, and a point this version does not have.
#[test]
fn a_point_that_is_not_there_has_no_ray_source() {
    let (mut state, point) = selected(0);
    assert!(track_ray_source(&state).is_some());

    state.deselect_point();
    assert_eq!(track_ray_source(&state), None, "a cleared selection");

    state.select_point(point);
    state.delete_point(point).expect("it deletes");
    assert_eq!(
        track_ray_source(&state),
        None,
        "a deleted point still draws rays",
    );
}

/// A node that is not on screen draws no rays.
///
/// `render_track_rays` draws whatever the buffer holds and has no node of its
/// own in the draw loop, so nothing else stops a hidden node's rays hanging in
/// the air over the node that is still shown.
#[test]
fn a_hidden_node_has_no_ray_source() {
    // The second node first: a node arriving clears the point selection, which
    // is a file taking focus rather than anything this is about.
    let (mut state, _) = edits::adjustable_state();
    let other = state.append_node(crate::scene::SceneNode::demo(
        sfmtool_core::SfmrReconstruction::demo(8),
    ));
    let point = PointRef::new(state.scene[0].id, 0);
    state.select_point(point);
    assert!(track_ray_source(&state).is_some(), "the point is selected");

    // The node's own eye.
    state.scene[0].visible = false;
    assert_eq!(track_ray_source(&state), None, "the eye was closed");
    state.scene[0].visible = true;
    assert!(track_ray_source(&state).is_some(), "and opened again");

    // Solo somewhere else, which hides this one without touching its eye.
    state.solo = Some(other);
    assert_eq!(track_ray_source(&state), None, "solo is elsewhere");
    state.solo = None;
    assert!(track_ray_source(&state).is_some(), "and released");
}
