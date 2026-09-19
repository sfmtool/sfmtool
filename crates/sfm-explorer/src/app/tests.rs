// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a frame decides before it draws. Headless: these are questions about
//! the state, asked the way the frame asks them.

use super::{selected_point_source, track_ray_source};
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

/// The frustums of the images that observe the selected point are lit from that
/// point's **track**, which is the version's rather than the index's, so they
/// are gated on the same source the rays are.
///
/// The adjustment is the case that makes the difference visible: it renumbers
/// nothing, so a comparison of the selection alone reads as no change at all and
/// the colours would be left describing the version before.
#[test]
fn a_new_version_under_the_selection_is_a_new_point_source() {
    let (mut state, point) = selected(0);
    let before = selected_point_source(&state).expect("a selection in a loaded node");

    adjust(&mut state, point.recon);

    assert_eq!(
        state.selected_point,
        Some(point),
        "this case is the one where the selection does not move",
    );
    let after = selected_point_source(&state).expect("the node is still loaded");
    assert_ne!(before, after, "the colours would have been left behind");
}

/// A commit puts the rays on the point it wrote, wherever they were.
///
/// Two changes at once, and each on its own would leave them wrong: the
/// selection moves to the written point, and the version under it is the one the
/// commit pushed.
#[test]
fn a_commit_puts_the_ray_source_on_the_point_it_wrote() {
    let (mut state, id) = crate::bench::tests::state();
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    // Somewhere other than the track's origin, which is where the map would
    // have left it.
    state.select_point(PointRef::new(id, 0));
    let before = track_ray_source(&state).expect("a live point");

    let written = state.commit_bench_track(id, &label).expect("a track stage");

    let after = track_ray_source(&state).expect("the written point is live");
    assert_ne!(before, after, "the rays were left on the point before");
    assert_eq!(
        after.0,
        PointRef::new(id, written.point as usize),
        "the rays are not on what the commit wrote",
    );

    // Undo takes the written point away again, and the rays with it: the
    // selection lands back on the point the commit replaced.
    state.undo(id).expect("the commit");
    let undone = track_ray_source(&state).expect("the replaced point is back");
    assert_ne!(undone, after);
    assert_eq!(
        undone.0,
        PointRef::new(id, written.replaced.expect("this one replaces") as usize),
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
