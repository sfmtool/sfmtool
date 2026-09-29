// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The viewed track: built for the viewed point as a put builds a bench track,
//! kept in a small cache keyed by the point and the document serial, evaluated
//! live ahead of the bench's tracks without touching a version, and the
//! read-only bars carried onto the bench by a put of the viewed point.
//!
//! Over the bench fixture of [`crate::bench::tests`], whose photographs are
//! cached for every image, so the evaluation runs its real kernels.

use std::sync::Arc;

use sfmtool_core::bench::{EditableTrack, Thresholds, Verdict};

use crate::bench::live::Evaluation;
use crate::bench::tests::{put_on_bench, state, POINT};
use crate::scene::{PointRef, ReconId};
use crate::state::edits::PointGesture;
use crate::state::AppState;

/// How many versions the node holds.
fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

/// The label of the version at the node's cursor.
fn version_label(state: &AppState, id: ReconId) -> String {
    state
        .node(id)
        .expect("loaded")
        .history
        .current_version()
        .label
        .clone()
}

/// A live point of the fixture other than [`POINT`].
fn other_point(state: &AppState, id: ReconId) -> u32 {
    let node = state.node(id).expect("loaded");
    (0..node.edited().point_count() as u32)
        .find(|&p| p != POINT && node.edited().point(p).is_some())
        .expect("the fixture has more than one point")
}

/// Select `point` on `id` and ask for its viewed track, as the dock does when
/// it draws Track View.
fn view(state: &mut AppState, id: ReconId, point: u32) {
    state.select_point(PointRef::new(id, point as usize));
    state.refresh_viewed_track();
}

/// The current viewed track's value.
fn viewed(state: &AppState) -> Arc<EditableTrack> {
    Arc::clone(&state.viewed_track().expect("a viewed track").track)
}

/// Bars that differ from the defaults in the minimum ZNCC alone.
fn strict_bars() -> Thresholds {
    Thresholds {
        min_zncc: 0.83,
        ..Thresholds::default()
    }
}

#[test]
fn the_viewed_track_is_the_selected_point_as_a_put_builds_it() {
    let (mut state, id) = state();
    let before = versions(&state, id);
    view(&mut state, id, POINT);

    let viewed = state.viewed_track().expect("a viewed track");
    let node = state.node(id).expect("loaded");
    assert_eq!(
        viewed.label,
        crate::scene::point_id(node, POINT as usize),
        "labelled with the point's portable ID"
    );
    assert_eq!((viewed.node, viewed.point), (id, POINT));
    assert!(!viewed.track.observations.is_empty());
    assert!(
        viewed
            .track
            .observations
            .iter()
            .all(|o| o.verdict == Verdict::In && o.pinned),
        "every row arrives in and pinned"
    );
    assert_eq!(viewed.track.thresholds, Thresholds::default());
    assert_eq!(viewed.evaluation, Evaluation::Evaluating);
    assert!(
        state.bench(id).expect("a bench").is_empty(),
        "nothing put on"
    );
    assert_eq!(versions(&state, id), before, "no version pushed");
}

#[test]
fn there_is_no_viewed_track_while_an_item_is_focused_or_for_a_deleted_point() {
    let (mut state, id) = state();
    view(&mut state, id, POINT);
    assert!(state.viewed_track().is_some());

    // Putting the point on focuses its item, which leaves no viewed point.
    put_on_bench(&mut state, id);
    assert_eq!(state.viewed_point(), None);
    state.refresh_viewed_track();
    assert!(state.viewed_track().is_none());

    // Unfocus, delete the point, and name it again.
    state.unfocus_bench_item();
    state
        .delete_point(PointRef::new(id, POINT as usize))
        .expect("a live point");
    state.selected_point = Some(PointRef::new(id, POINT as usize));
    state.refresh_viewed_track();
    assert!(
        state.viewed_track().is_none(),
        "a deleted point gives no viewed track"
    );
}

#[test]
fn a_hidden_panel_leaves_no_current_viewed_track_and_keeps_the_cache() {
    let (mut state, id) = state();
    view(&mut state, id, POINT);
    let built = viewed(&state);
    state.hide_viewed_track();
    assert!(state.viewed_track().is_none());
    state.refresh_viewed_track();
    assert!(Arc::ptr_eq(&viewed(&state), &built));
}

#[test]
fn the_cache_returns_the_same_track_and_a_document_edit_rebuilds_it() {
    let (mut state, id) = state();
    let other = other_point(&state, id);
    view(&mut state, id, POINT);
    let first = viewed(&state);

    view(&mut state, id, other);
    assert!(!Arc::ptr_eq(&viewed(&state), &first));
    view(&mut state, id, POINT);
    assert!(
        Arc::ptr_eq(&viewed(&state), &first),
        "a point returned to is taken from the cache"
    );

    // A document edit moves the document serial, which is part of the key.
    let third = (0..state.scene[0].edited().point_count() as u32)
        .find(|&p| p != POINT && p != other && state.scene[0].edited().point(p).is_some())
        .expect("the fixture has a third point");
    state
        .delete_point(PointRef::new(id, third as usize))
        .expect("a live point");
    view(&mut state, id, POINT);
    let edited = viewed(&state);
    assert!(
        !Arc::ptr_eq(&edited, &first),
        "a document edit rebuilds the viewed track"
    );

    // An undo across the edit lands on the first document serial again.
    state.undo(id).expect("the deletion");
    view(&mut state, id, POINT);
    assert!(Arc::ptr_eq(&viewed(&state), &first));
}

#[test]
fn the_viewed_track_is_evaluated_first_with_no_version_and_no_row() {
    let (mut state, id) = state();
    // A bench track waiting for its evaluation, and not focused.
    let other = other_point(&state, id);
    let label = state
        .put_point_on_bench(PointRef::new(id, other as usize), None)
        .expect("a live point");
    view(&mut state, id, POINT);
    assert!(state.focused_item().is_none());
    let before = versions(&state, id);
    state.action_log.clear();

    state.drive_bench_evaluation();
    assert!(
        state.viewed_evaluation_running(),
        "the viewed track was not evaluated first"
    );
    assert!(!state.bench_evaluation_running(id, &label));

    state.settle_bench_evaluation();
    let viewed = state.viewed_track().expect("a viewed track");
    assert_eq!(viewed.evaluation, Evaluation::Current);
    assert!(
        viewed
            .track
            .observations
            .iter()
            .any(|o| o.track.as_ref().is_some_and(|m| m.seed_shift_px.is_some())),
        "the evaluation measured nothing"
    );
    assert!(
        viewed
            .track
            .observations
            .iter()
            .all(|o| o.verdict == Verdict::In && o.pinned),
        "the evaluation moved a verdict of the viewed track"
    );
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current),
        "the bench track was evaluated after it"
    );
    assert_eq!(
        versions(&state, id),
        before,
        "an evaluation pushed a version"
    );
    assert_eq!(
        state.action_log.entries().count(),
        0,
        "an evaluation wrote an Action Log row"
    );
}

#[test]
fn a_selection_change_cancels_the_viewed_evaluation_and_a_revisit_evaluates_nothing() {
    let (mut state, id) = state();
    let other = other_point(&state, id);
    view(&mut state, id, POINT);
    state.drive_bench_evaluation();
    let read = state
        .running_evaluation_track()
        .expect("the evaluation started");
    assert!(Arc::ptr_eq(&read, &viewed(&state)));

    view(&mut state, id, other);
    state.drive_bench_evaluation();
    assert!(
        state.running_evaluation_cancelled(),
        "the selection change did not cancel the evaluation"
    );
    assert!(!state.viewed_evaluation_running());
    state.land_running_evaluation();
    state.settle_bench_evaluation();
    assert_eq!(
        state.viewed_track().expect("viewed").evaluation,
        Evaluation::Current
    );

    // The cancelled point was never evaluated, so going back evaluates it.
    view(&mut state, id, POINT);
    assert_eq!(
        state.viewed_track().expect("viewed").evaluation,
        Evaluation::Evaluating
    );
    state.settle_bench_evaluation();

    // Both are current now: moving between them starts nothing.
    view(&mut state, id, other);
    assert_eq!(
        state.viewed_track().expect("viewed").evaluation,
        Evaluation::Current
    );
    state.drive_bench_evaluation();
    assert!(
        state.running_evaluation_track().is_none(),
        "a cached current evaluation was run again"
    );
}

#[test]
fn moved_bars_carry_onto_the_bench_when_the_edit_box_puts_the_viewed_point() {
    let (mut state, id) = state();
    view(&mut state, id, POINT);
    state.set_viewed_thresholds(strict_bars());
    let before = versions(&state, id);

    state.set_editing(id, true).expect("a put");
    assert_eq!(versions(&state, id), before + 1, "one version");
    let label = state.focused_item_label(id).expect("focused").to_string();
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(track.thresholds, strict_bars());
    assert!(
        track
            .observations
            .iter()
            .all(|o| o.verdict == Verdict::In && o.pinned),
        "the rows arrive in and pinned"
    );
    assert_eq!(
        version_label(&state, id),
        format!("Put point {POINT} on the bench as {label}, with min ZNCC 83%")
    );
    assert_eq!(
        state.viewed_thresholds,
        strict_bars(),
        "the boxes keep their bars for the next point"
    );
}

#[test]
fn moved_bars_carry_onto_the_bench_through_edit_on_bench() {
    let (mut state, id) = state();
    // Another point is on screen; Edit on Bench selects this one first.
    let other = other_point(&state, id);
    view(&mut state, id, other);
    state.set_viewed_thresholds(strict_bars());

    state.apply_point_gesture(PointGesture::EditOnBench(PointRef::new(id, POINT as usize)));
    let label = state.focused_item_label(id).expect("focused").to_string();
    assert_eq!(
        state.bench_track(id, &label).expect("on").thresholds,
        strict_bars()
    );
    assert!(version_label(&state, id).ends_with(", with min ZNCC 83%"));
}

#[test]
fn unmoved_bars_put_the_point_on_at_the_defaults_with_no_suffix() {
    let (mut state, id) = state();
    view(&mut state, id, POINT);
    state.set_editing(id, true).expect("a put");
    let label = state.focused_item_label(id).expect("focused").to_string();
    assert_eq!(
        state.bench_track(id, &label).expect("on").thresholds,
        Thresholds::default()
    );
    assert_eq!(
        version_label(&state, id),
        format!("Put point {POINT} on the bench as {label}")
    );
}

#[test]
fn another_point_and_an_existing_item_take_no_carried_bars() {
    let (mut state, id) = state();
    let other = other_point(&state, id);
    view(&mut state, id, POINT);
    state.set_viewed_thresholds(strict_bars());

    // A put of a point that is not the viewed point.
    let put = state
        .put_point_on_bench(PointRef::new(id, other as usize), None)
        .expect("a live point");
    assert_eq!(
        state.bench_track(id, &put).expect("on").thresholds,
        Thresholds::default()
    );
    assert!(!version_label(&state, id).contains("with"));

    // A point that already has an item focuses it, keeping its own bars.
    view(&mut state, id, other);
    let versions_before = versions(&state, id);
    state.set_editing(id, true).expect("a focus");
    assert_eq!(state.focused_item_label(id), Some(put.as_str()));
    assert_eq!(versions(&state, id), versions_before, "no version");
    assert_eq!(
        state.bench_track(id, &put).expect("on").thresholds,
        Thresholds::default()
    );
}

#[test]
fn setting_the_read_only_bars_pushes_nothing_and_changes_no_verdict() {
    let (mut state, id) = state();
    view(&mut state, id, POINT);
    let before = versions(&state, id);
    let track = viewed(&state);
    state.action_log.clear();

    state.set_viewed_thresholds(strict_bars());
    assert_eq!(versions(&state, id), before);
    assert_eq!(state.action_log.entries().count(), 0);
    assert!(Arc::ptr_eq(&viewed(&state), &track));
    assert_eq!(viewed(&state).thresholds, Thresholds::default());
}

#[test]
fn the_read_only_bars_judge_the_rows_by_verdicts_if_unpinned() {
    let (mut state, id) = state();
    view(&mut state, id, POINT);
    state.settle_bench_evaluation();
    let rows = viewed(&state).observations.len();

    let lenient = state.viewed_verdicts().expect("a viewed track");
    assert_eq!(lenient.len(), rows);

    // A bar no reading can reach turns every measured row out.
    state.set_viewed_thresholds(Thresholds {
        min_zncc: 1.1,
        ..Thresholds::default()
    });
    let strict = state.viewed_verdicts().expect("a viewed track");
    assert!(!strict.contains(&Some(Verdict::In)));
    assert!(strict.contains(&Some(Verdict::Out)));
}
