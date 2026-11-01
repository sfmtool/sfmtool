// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The live evaluation: a track put on the bench is evaluated first, every
//! change to an input of the evaluation evaluates it again, and an answer that
//! comes back for inputs the track has moved on from is never shown as current.
//!
//! Over the bench fixture of [`crate::bench::tests`]: a point with three
//! observations and a textured photograph cached for every image, so the
//! evaluation runs its real kernels.

use std::sync::Arc;

use sfmtool_core::bench::{BenchItem, Stage, Verdict};

use crate::background::{Finished, Operation, Queue, TaskState};
use crate::bench::live::Evaluation;
use crate::bench::tests::{put_on_bench, state};
use crate::scene::{PointRef, ReconId};
use crate::state::AppState;

/// How many versions the node holds.
fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

/// The track called `label`, as the node's bench holds it now.
fn track(state: &AppState, id: ReconId, label: &str) -> Arc<sfmtool_core::bench::EditableTrack> {
    Arc::clone(state.bench_track(id, label).expect("on the bench"))
}

#[test]
fn a_track_put_on_the_bench_is_evaluated_first_with_no_version_and_no_row() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let before = versions(&state, id);
    state.action_log.clear();

    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating),
        "a track that has just arrived has no evaluation of its inputs"
    );
    let arrived = track(&state, id, &label);
    state.drive_bench_evaluation();
    assert!(
        state.bench_evaluation_running(id, &label),
        "the frame after the track arrived did not start its evaluation"
    );

    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );
    let measured = track(&state, id, &label);
    assert!(
        !Arc::ptr_eq(&arrived, &measured),
        "the evaluation installed nothing"
    );
    assert!(
        measured
            .observations
            .iter()
            .any(|o| o.track.as_ref().is_some_and(|m| m.seed_shift_px.is_some())),
        "the evaluation measured nothing"
    );
    assert_eq!(
        versions(&state, id),
        before,
        "the evaluation pushed a version"
    );
    assert_eq!(
        state.action_log.entries().count(),
        0,
        "the evaluation wrote an Action Log row"
    );
}

/// The focused item is evaluated before every other track, on any node, since
/// it is what Track View shows.
#[test]
fn the_focused_item_is_evaluated_first() {
    let (mut state, a) = state();
    let on_a = put_on_bench(&mut state, a);
    let b = state.append_node(crate::scene::SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let on_b = put_on_bench(&mut state, b);

    // Node a comes first in the scene, and node b's item is focused.
    state.drive_bench_evaluation();
    assert!(
        state.bench_evaluation_running(b, &on_b),
        "the focused item on the second node was not evaluated first"
    );
    state.settle_bench_evaluation();

    state.focus_bench_item(a, &on_a).expect("on a's bench");
    state
        .set_bench_verdict(b, &on_b, 1, Verdict::Out)
        .expect("observation 1 exists");
    state
        .set_bench_verdict(a, &on_a, 1, Verdict::Out)
        .expect("observation 1 exists");
    state.drive_bench_evaluation();
    assert!(state.bench_evaluation_running(a, &on_a));
    state.settle_bench_evaluation();
}

/// A rename keeps the item's ID, which is what the evaluation is keyed by, so
/// an evaluated track stays current under its new label.
#[test]
fn a_rename_keeps_the_evaluation_current() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    state
        .rename_bench_item(id, &label, "renamed")
        .expect("a free label");
    assert_eq!(
        state.bench_evaluation(id, "renamed"),
        Some(Evaluation::Current)
    );
}

#[test]
fn a_step_that_changes_an_input_starts_a_new_evaluation() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );

    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating),
        "a verdict is an input of the evaluation"
    );
    state.drive_bench_evaluation();
    assert!(state.bench_evaluation_running(id, &label));
    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );
    assert_eq!(
        track(&state, id, &label).observations[1].verdict,
        Verdict::Out,
        "the evaluation moved a verdict"
    );
}

#[test]
fn an_answer_for_inputs_that_have_moved_on_is_not_shown_as_current() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    // Held, so the worker cannot answer before the next frame's drive, which
    // would land the stale answer and queue the next evaluation in turn.
    let open = state.hold_next_evaluation();
    state.drive_bench_evaluation();
    let read = state
        .running_evaluation_track()
        .expect("the evaluation started");

    // A step lands while the worker is reading the track as it was.
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    let edited = track(&state, id, &label);
    assert!(!Arc::ptr_eq(&read, &edited));

    // The next frame asks the stale evaluation to stop and starts nothing
    // beside it: one worker at a time, however many steps land.
    state.drive_bench_evaluation();
    assert!(
        Arc::ptr_eq(
            &state
                .running_evaluation_track()
                .expect("still running until it answers"),
            &read
        ),
        "a second evaluation started beside the stale one"
    );
    assert!(
        state.running_evaluation_cancelled(),
        "the stale evaluation was not asked to stop"
    );
    assert!(!state.bench_evaluation_running(id, &label));

    // Whatever it answers -- cancelled, or measured before it saw the flag --
    // is dropped, and the track stays as the step left it.
    open.send(()).expect("the evaluation is held");
    state.land_running_evaluation();
    assert!(
        Arc::ptr_eq(&track(&state, id, &label), &edited),
        "the stale answer was installed over the step"
    );
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating)
    );

    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );
    assert_eq!(
        track(&state, id, &label).observations[1].verdict,
        Verdict::Out
    );
}

#[test]
fn undo_the_reconstruction_under_the_track_and_the_search_radius_are_inputs() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    state.settle_bench_evaluation();
    // The fixture's photographs are only in memory. The photograph cache is
    // keyed by path, so the cursor move and the document edit below keep them.

    // Undo lands on another version of the track.
    state.undo(id).expect("the verdict");
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating),
        "undo is a change of inputs"
    );
    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );

    // A document edit leaves the track alone and moves the reconstruction the
    // track is read against.
    let other = (0..state.scene[0].edited().point_count())
        .find(|&p| p != 2)
        .expect("the fixture has more than one point");
    state
        .delete_point(PointRef::new(id, other))
        .expect("a live point");
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating),
        "the reconstruction under the track is an input"
    );
    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );

    // The search radius is the track's shift bar, so moving the bar is a
    // step on the track like any other, and evaluates it again.
    let bars = sfmtool_core::bench::Thresholds {
        max_shift_px: 9.0,
        ..track(&state, id, &label).thresholds.clone()
    };
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("on the bench");
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating),
        "the shift bar, which is the search radius, is an input"
    );
    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );
}

#[test]
fn a_track_that_cannot_be_evaluated_says_why_and_starts_nothing() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let mut without_patch_frame = (*track(&state, id, &label)).clone();
    match &mut without_patch_frame.stage {
        Stage::Track(payload) => payload.patch_frame = None,
        Stage::Cluster(_) => panic!("a point goes on the bench at the track stage"),
    }
    let history = &mut state.scene[0].history;
    let bench = history
        .current_bench()
        .replace(&label, BenchItem::Track(Arc::new(without_patch_frame)))
        .expect("on the bench");
    history.push_bench(Arc::new(bench), "Dropped the patch frame");

    let evaluation = state.bench_evaluation(id, &label).expect("a track");
    match &evaluation {
        Evaluation::Refused(why) => {
            assert!(
                why.starts_with(&format!("Cannot evaluate {label}")),
                "{why}"
            )
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    state.drive_bench_evaluation();
    assert!(
        state.running_evaluation_track().is_none(),
        "a refused track started an evaluation"
    );
}

#[test]
fn an_evaluation_waits_while_an_operation_holds_the_node() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state
        .start_background_task(
            Operation::BENCH_FIT,
            id,
            Box::new(|_| Finished::NoChange("Nothing to fit".to_string())),
        )
        .expect("nothing else is running");

    state.drive_bench_evaluation();
    assert!(
        state.running_evaluation_track().is_none(),
        "an evaluation started under an operation whose answer replaces its inputs"
    );
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating)
    );

    state.finish_background_tasks();
    state.drive_bench_evaluation();
    assert!(state.bench_evaluation_running(id, &label));
}

#[test]
fn a_tilt_drops_the_bitmap_and_the_evaluation_fuses_it_again_where_the_patch_now_faces() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    let payload = |track: &sfmtool_core::bench::EditableTrack| match &track.stage {
        Stage::Track(payload) => payload.clone(),
        Stage::Cluster(_) => panic!("the point came onto the bench as a cluster"),
    };
    let before = payload(&track(&state, id, &label));
    let was = before.patch_frame.clone().expect("the point has a patch");
    assert!(before.bitmap.is_some(), "the point came with no bitmap");

    let (sin, cos) = 10f64.to_radians().sin_cos();
    let normal = was.normal() * cos + was.u_axis * sin;
    state
        .edit_bench_patch(
            id,
            &label,
            &crate::bench::PatchEdit::Tilt {
                normal: normal.into(),
            },
        )
        .expect("a finite direction");
    let tilted = payload(&track(&state, id, &label));
    assert!(
        tilted.bitmap.is_none(),
        "the tilt kept the bitmap it turned away from"
    );
    let versions_after_tilt = versions(&state, id);

    state.drive_bench_evaluation();
    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );
    let fused = payload(&track(&state, id, &label));
    assert!(fused.bitmap.is_some(), "the evaluation rendered no bitmap");
    assert_eq!(
        fused.patch_frame, tilted.patch_frame,
        "rendering the bitmap moved the patch"
    );
    assert!((fused.patch_frame.expect("placed").normal() - normal).norm() < 1e-9);
    assert_eq!(
        versions(&state, id),
        versions_after_tilt,
        "rendering the bitmap pushed a version"
    );
}

#[test]
fn an_evaluation_whose_repaint_moved_a_verdict_is_read_once_more_and_then_settles() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    // Hand every row to the bars, and set bars no reading clears while nothing
    // is measured, so the first evaluation's repaint turns every row out.
    let rows: Vec<usize> = (0..track(&state, id, &label).observations.len()).collect();
    state
        .unpin_bench_verdicts(id, &label, &rows)
        .expect("the rows exist");
    let mut bars = track(&state, id, &label).thresholds.clone();
    bars.min_zncc = 1.1;
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("a track on the bench");
    let before = versions(&state, id);

    state.drive_bench_evaluation();
    state.land_running_evaluation();
    let repainted = track(&state, id, &label);
    assert_eq!(
        repainted.verdict_counts().0,
        0,
        "the repaint turned them out"
    );
    assert!(repainted.repainted());
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating),
        "readings taken under the old verdicts are not current"
    );

    state.drive_bench_evaluation();
    assert!(state.bench_evaluation_running(id, &label));
    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );
    let settled = track(&state, id, &label);
    assert!(!settled.repainted());
    assert_eq!(settled.verdict_counts().0, 0);
    assert_eq!(
        versions(&state, id),
        before,
        "no evaluation pushed a version"
    );
}

// -- As a quiet task in the fast queue -------------------------------------

/// A gated job a bench step's place in the fast queue is held by, which
/// produces nothing.
fn held_step(gate: std::sync::mpsc::Receiver<()>) -> crate::background::Job {
    Box::new(move |_| {
        let _ = gate.recv();
        Finished::NoChange("Nothing to fit".to_string())
    })
}

/// How many quiet tasks wait in the fast queue.
fn quiet_waiting(state: &AppState) -> usize {
    state
        .queue(Queue::Fast)
        .waiting
        .iter()
        .filter(|task| task.operation.quiet)
        .count()
}

/// A bench step asked for while an evaluation runs waits for it to stop,
/// starts once it has reported, and the evaluation it displaced is queued
/// again behind it and lands once it has run.
#[test]
fn a_bench_step_displaces_a_running_evaluation_which_runs_after_it() {
    let (mut state, a) = state();
    put_on_bench(&mut state, a);
    let b = state.append_node(crate::scene::SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let on_b = put_on_bench(&mut state, b);
    state.drive_bench_evaluation();
    assert!(state.bench_evaluation_running(b, &on_b));

    let queued = std::time::Instant::now();
    let step = state
        .start_background_task(
            Operation::BENCH_FIT,
            a,
            Box::new(|_| Finished::NoChange("Nothing to fit".to_string())),
        )
        .expect("a busy fast queue queues the step");
    assert!(
        matches!(state.task(step), Some(TaskState::Waiting(_))),
        "the step started beside the evaluation"
    );
    assert!(
        state.running_evaluation_cancelled(),
        "the step did not ask the evaluation to stop"
    );
    // The answer is read and the evaluation's task lands, which starts the
    // step in the same poll.
    state.land_running_evaluation();
    let waited = match state.task(step) {
        Some(TaskState::Running(task)) => task.started - queued,
        _ => panic!("the step did not start once the evaluation had stopped"),
    };
    eprintln!("the step waited {waited:?} for the evaluation to stop");

    // Unless it finished before it read the flag, the evaluation is
    // unevaluated still, and queued again it waits behind the step.
    if state.bench_evaluation(b, &on_b) == Some(Evaluation::Evaluating) {
        state.drive_bench_evaluation();
        assert!(
            state.running_evaluation_track().is_some(),
            "not queued again"
        );
        assert!(
            !state.bench_evaluation_running(b, &on_b),
            "the evaluation ran beside the step"
        );
    }
    state.finish_background_task(step);
    state.settle_bench_evaluation();
    assert_eq!(state.bench_evaluation(b, &on_b), Some(Evaluation::Current));
}

/// A locking task queued on a node cancels a running evaluation of that node,
/// and no evaluation lands into a locked node: not the cancelled one, and not
/// one that had finished before the task was queued, whose answer comes back
/// as one that finished before it read the flag would.
#[test]
fn no_evaluation_lands_into_a_locked_node() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let arrived = track(&state, id, &label);

    // An evaluation that has finished, its answer not yet read.
    state.drive_bench_evaluation();
    state.land_quiet_tasks();
    let (open, gate) = std::sync::mpsc::channel();
    let solve = state
        .start_background_task(Operation::RETRIANGULATE_ALL_POINTS, id, held_step(gate))
        .expect("the slow queue is idle");
    state.action_log.clear();
    state.drive_bench_evaluation();
    assert!(
        Arc::ptr_eq(&track(&state, id, &label), &arrived),
        "an evaluation landed into a locked node"
    );
    assert_eq!(state.action_log.entries().count(), 0);
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating)
    );
    assert!(
        state.running_evaluation_track().is_none(),
        "an evaluation was queued on a locked node"
    );
    open.send(()).expect("the task is waiting");
    state.finish_background_task(solve);

    // A running evaluation, cancelled by the task queued on its node.
    state.drive_bench_evaluation();
    assert!(state.bench_evaluation_running(id, &label));
    let (open, gate) = std::sync::mpsc::channel();
    let solve = state
        .start_background_task(Operation::RETRIANGULATE_ALL_POINTS, id, held_step(gate))
        .expect("the slow queue is idle");
    assert!(
        state.running_evaluation_cancelled(),
        "a task queued on the node did not cancel its evaluation"
    );
    state.land_running_evaluation();
    assert!(Arc::ptr_eq(&track(&state, id, &label), &arrived));
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating)
    );
    open.send(()).expect("the task is waiting");
    state.finish_background_task(solve);

    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );
}

/// An evaluation that waits in the fast queue and is replaced by a newer
/// request, or dropped when it would start because its node became locked,
/// never ran: neither is recorded as failed, the driver keeps one entry
/// waiting, and the track is evaluated once its node is free.
#[test]
fn a_replaced_or_dropped_evaluation_is_not_recorded_as_failed() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let other = state.append_node(crate::scene::SceneNode::demo(
        sfmtool_core::SfmrReconstruction::demo(8),
    ));
    let (open_step, gate) = std::sync::mpsc::channel();
    let step = state
        .start_background_task(Operation::BENCH_FIT, other, held_step(gate))
        .expect("the fast queue is idle");
    state.drive_bench_evaluation();
    let first = state.queued_evaluation_task().expect("queued");
    assert!(
        !state.bench_evaluation_running(id, &label),
        "it ran beside the step"
    );

    // Replaced: a step moves its input while it waits.
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    state.drive_bench_evaluation();
    assert_ne!(state.queued_evaluation_task(), Some(first), "not replaced");
    assert_eq!(
        quiet_waiting(&state),
        1,
        "the source has two entries waiting"
    );
    state.drive_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating)
    );

    // Dropped at start: its node is locked while it waits.
    let (open_solve, gate) = std::sync::mpsc::channel();
    let solve = state
        .start_background_task(Operation::RETRIANGULATE_ALL_POINTS, id, held_step(gate))
        .expect("the slow queue is idle");
    open_step.send(()).expect("the step is waiting");
    state.finish_background_task(step);
    assert_eq!(quiet_waiting(&state), 0);
    assert!(
        state.queue(Queue::Fast).running.is_none(),
        "it started on a locked node"
    );
    state.drive_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating),
        "a dropped evaluation was recorded as failed"
    );
    assert!(state.running_evaluation_track().is_none());

    open_solve.send(()).expect("the task is waiting");
    state.finish_background_task(solve);
    state.settle_bench_evaluation();
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Current)
    );
}
