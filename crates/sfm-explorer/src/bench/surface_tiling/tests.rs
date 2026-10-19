// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Surface Tiling Mode: the duplicate that unpins the copy and then drops its
//! `out` rows, and the sequence a move starts, run step by step as the frame
//! runs it, with the background task and the live evaluation driven to their
//! ends in between.

use sfmtool_core::bench::Verdict;

use super::{search_row, TilingStep, TILING_STEPS};
use crate::action_log::Kind;
use crate::bench::tests::{put_on_bench, state};
use crate::bench::SplitSettings;
use crate::scene::ReconId;
use crate::state::AppState;

/// The Action Log's entries, oldest first, as `(kind, failed, text)`.
fn rows(state: &AppState) -> Vec<(Kind, bool, String)> {
    state
        .action_log
        .entries()
        .map(|entry| (entry.kind, entry.failed, entry.text.clone()))
        .collect()
}

/// How many versions the node holds.
fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

/// Run frames until the sequence under way has ended, ending each step's
/// task and settling the evaluation after it as the frames between would.
/// Returns the steps the sequence was at, in the order it started them.
fn run_sequence(state: &mut AppState) -> Vec<TilingStep> {
    let mut started = Vec::new();
    for _ in 0..64 {
        state.settle_bench_evaluation();
        let Some(waiting) = state.surface_tiling.as_ref().map(|s| s.waiting_on) else {
            return started;
        };
        state.drive_surface_tiling();
        match state.surface_tiling.as_ref() {
            // A step on the worker that was not there before the frame.
            Some(s) if s.waiting_on.is_some() && s.waiting_on != waiting => {
                started.push(s.step());
            }
            Some(_) => {}
            // The sequence ended in the frame; with no failed row, its commit
            // is what ended it.
            None if rows(state).iter().all(|(_, failed, _)| !failed) => {
                started.push(TilingStep::Commit);
            }
            None => {}
        }
        if state.running_tasks().next().is_some() {
            state.finish_background_tasks();
        }
    }
    panic!("the sequence did not end: {:?}", rows(state));
}

/// Run frames until a duplicate's drop has ended, settling the copy's
/// evaluation before each as the frames between would.
fn finish_drop(state: &mut AppState) {
    for _ in 0..8 {
        if state.surface_tiling.is_none() {
            return;
        }
        state.settle_bench_evaluation();
        state.drive_surface_tiling();
    }
    panic!("the drop did not end: {:?}", rows(state));
}

/// The mode's duplicate hands the copy's pins back to the thresholds first,
/// and drops what is `out` only once the copy's evaluation has settled, so a
/// row the thresholds turn `out` is dropped with the rest.
#[test]
fn a_tiling_duplicate_unpins_the_copy_and_then_drops_what_the_bars_turn_out() {
    use crate::bench::live::Evaluation;
    let (mut state, id) = state();
    let (label, refused) = crate::bench::tests::put_with_rows_the_bars_refuse(
        &mut state,
        id,
        crate::bench::tests::Refused::Some,
    );
    state.action_log.clear();
    let before = versions(&state, id);

    let copy = state
        .duplicate_for_surface_tiling(id, &label)
        .expect("the copy is made");

    // The duplicate and the unpin are taken at once; the drop waits.
    assert_eq!(state.focused_item_label(id), Some(copy.as_str()));
    assert_eq!(versions(&state, id), before + 2, "{:?}", rows(&state));
    assert_eq!(
        state.surface_tiling_status(id, &copy).as_deref(),
        Some("Step 1 of 1: Drop the out observations")
    );
    assert_eq!(
        state.bench_evaluation(id, &copy),
        Some(Evaluation::Evaluating)
    );
    state.drive_surface_tiling();
    assert_eq!(
        versions(&state, id),
        before + 2,
        "the drop ran before the evaluation settled"
    );

    finish_drop(&mut state);

    let logged = rows(&state);
    assert_eq!(versions(&state, id), before + 3, "{logged:?}");
    assert!(state.surface_tiling.is_none());
    let kept = 3 - refused.len();
    let track = state.bench_track(id, &copy).expect("on the bench");
    assert_eq!(track.observations.len(), kept, "{logged:?}");
    assert!(
        track
            .observations
            .iter()
            .all(|o| !refused.contains(&o.image)),
        "a row the bars turned out was kept: {logged:?}"
    );
    assert!(
        track
            .observations
            .iter()
            .all(|o| !o.pinned && o.verdict == Verdict::In),
        "{logged:?}"
    );
    assert_eq!(
        track.origin, None,
        "the copy creates a point when committed"
    );
    // The original is left as it was.
    let original = state.bench_track(id, &label).expect("still on the bench");
    assert_eq!(original.observations.len(), 3);
    assert!(original
        .observations
        .iter()
        .all(|o| o.pinned && o.verdict == Verdict::In));

    assert!(logged.iter().all(|(_, failed, _)| !failed), "{logged:?}");
    let texts: Vec<String> = logged
        .iter()
        .filter(|(kind, _, _)| *kind == Kind::Bench)
        .map(|(_, _, t)| t.clone())
        .collect();
    assert!(
        texts[0].starts_with(&format!("Duplicated {label} as {copy}")),
        "{texts:?}"
    );
    let n = refused.len();
    let observations = |n: usize| {
        if n == 1 {
            "observation"
        } else {
            "observations"
        }
    };
    assert!(
        texts[1].starts_with(&format!(
            "Handed 3 verdicts back to the thresholds in {copy}: {n} turned out"
        )),
        "{texts:?}"
    );
    let dropped = format!(
        "Dropped {n} out {} of {copy}, leaving {kept} in",
        observations(n)
    );
    assert!(
        texts.iter().skip(2).any(|t| t.starts_with(&dropped)),
        "{texts:?}"
    );
}

#[test]
fn a_tiling_duplicate_skips_the_steps_with_nothing_to_do() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let pinned: Vec<usize> = (0..3).collect();
    state
        .unpin_bench_verdicts(id, &label, &pinned)
        .expect("an unpin step");
    state.settle_bench_evaluation();
    let has_out = state
        .bench_track(id, &label)
        .expect("on the bench")
        .observations
        .iter()
        .any(|o| o.verdict == Verdict::Out);
    state.action_log.clear();
    let before = versions(&state, id);

    state
        .duplicate_for_surface_tiling(id, &label)
        .expect("the copy is made");
    finish_drop(&mut state);

    // No row is pinned, so only the duplicate and, when the bars turned a
    // row out, the drop are steps.
    let expected = if has_out { 2 } else { 1 };
    assert_eq!(
        versions(&state, id),
        before + expected,
        "{:?}",
        rows(&state)
    );
    assert!(
        rows(&state).iter().all(|(_, failed, _)| !failed),
        "{:?}",
        rows(&state)
    );
}

/// Where the thresholds turn every row of the copy `out`, the drop is
/// refused, the copy is left with its rows, and the refusal and the stop are
/// each a failed row.
#[test]
fn a_tiling_duplicate_whose_bars_turn_every_row_out_keeps_them() {
    let (mut state, id) = state();
    let (label, _) = crate::bench::tests::put_with_rows_the_bars_refuse(
        &mut state,
        id,
        crate::bench::tests::Refused::Every,
    );
    state.action_log.clear();
    let before = versions(&state, id);

    let copy = state
        .duplicate_for_surface_tiling(id, &label)
        .expect("the copy is made");
    finish_drop(&mut state);

    let logged = rows(&state);
    assert_eq!(versions(&state, id), before + 2, "{logged:?}");
    let track = state.bench_track(id, &copy).expect("on the bench");
    assert_eq!(track.observations.len(), 3, "{logged:?}");
    assert!(track.observations.iter().all(|o| o.verdict == Verdict::Out));
    let stop = stop_row(&state);
    assert!(
        stop.starts_with(&format!(
            "Surface tiling of {copy} stopped at Drop the out observations (step 1 of 1): \
             Cannot drop the out observations of {copy}: every one of its 3 observations \
             is out"
        )),
        "{stop}"
    );
    let (_, failed, refusal) = &logged[logged.len() - 2];
    assert!(*failed && refusal.starts_with("Cannot drop"), "{logged:?}");
}

/// A move of the copy while its drop waits for the evaluation ends the drop
/// with a row, and the move's own sequence starts.
#[test]
fn a_move_while_the_drop_waits_ends_it() {
    use crate::bench::PatchEdit;
    let (mut state, id) = state();
    let (label, _) = crate::bench::tests::put_with_rows_the_bars_refuse(
        &mut state,
        id,
        crate::bench::tests::Refused::Some,
    );
    let copy = state
        .duplicate_for_surface_tiling(id, &label)
        .expect("the copy is made");

    state
        .edit_bench_patch_by_gesture(
            id,
            &copy,
            &PatchEdit::Translate {
                by: [0.01, 0.0, 0.0],
            },
            Some(SplitSettings::default()),
        )
        .expect("a move in the mode");

    let logged = rows(&state);
    let (_, failed, stop) = &logged[logged.len() - 2];
    assert!(*failed, "{logged:?}");
    assert_eq!(
        stop,
        &format!(
            "Surface tiling of {copy} stopped at Drop the out observations (step 1 of 1): \
             the patch was moved before its evaluation settled, so its out observations \
             were left on it"
        )
    );
    let (_, _, started) = logged.last().expect("a row");
    assert!(
        started.starts_with(&format!("Surface tiling {copy} after the move")),
        "{logged:?}"
    );
    assert_eq!(
        state.surface_tiling.as_ref().map(|s| s.step()),
        Some(TilingStep::Fit)
    );
}

#[test]
fn dropping_every_row_is_refused_and_dropping_none_is_no_effect() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let before = versions(&state, id);
    state.action_log.clear();

    state
        .drop_out_bench_observations(id, &label)
        .expect("nothing to drop is not a refusal");
    assert_eq!(versions(&state, id), before);
    assert_eq!(
        rows(&state)[0].2,
        format!("Dropped the out observations of {label}: no effect, none is out")
    );

    for row in 0..3 {
        state
            .set_bench_verdict(id, &label, row, Verdict::Out)
            .expect("a verdict step");
    }
    let before = versions(&state, id);
    let refused = state
        .drop_out_bench_observations(id, &label)
        .expect_err("a track needs an observation");
    assert!(
        refused.contains("every one of its 3 observations is out"),
        "{refused}"
    );
    assert_eq!(versions(&state, id), before);
}

#[test]
fn a_move_runs_fit_search_fit_normal_fit_and_commit_in_order() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    state.action_log.clear();

    state.start_surface_tiling(id, &label, SplitSettings::default());
    assert_eq!(
        state.surface_tiling_status(id, &label).as_deref(),
        Some("Step 1 of 6: Fit")
    );
    let started = run_sequence(&mut state);

    let logged = rows(&state);
    assert!(state.surface_tiling.is_none(), "{logged:?}");
    assert_eq!(started, TILING_STEPS.to_vec(), "{logged:?}");
    assert!(
        logged.iter().all(|(_, failed, _)| !failed),
        "a step failed: {logged:?}"
    );
    // Each step wrote the row it writes when its button is pressed, in the
    // sequence's order, and the last is the commit's edit row.
    let find = |from: usize, kind: Kind, prefix: &str| {
        logged[from..]
            .iter()
            .position(|(k, _, text)| *k == kind && text.starts_with(prefix))
            .map(|i| from + i)
            .unwrap_or_else(|| panic!("no {prefix:?} row after {from}: {logged:?}"))
    };
    let fit = format!("Fitted {label}");
    let search = format!("{label}: Geometry search from observation");
    let normal = format!("Grid-plane normal of {label}");
    let first = find(0, Kind::Bench, &fit);
    let searched = find(first + 1, Kind::Bench, &search);
    let second = find(searched + 1, Kind::Bench, &fit);
    let turned = find(second + 1, Kind::Bench, &normal);
    let third = find(turned + 1, Kind::Bench, &fit);
    let committed = find(third + 1, Kind::Edit, "Committed ");
    assert!(
        logged[committed + 1..]
            .iter()
            .all(|(kind, _, _)| *kind == Kind::Selection),
        "a step ran after the commit: {logged:?}"
    );
    // The commit left the item focused, seated on the point it wrote, so the
    // next move's commit replaces that point.
    assert_eq!(state.focused_item_label(id), Some(label.as_str()));
    assert!(state
        .bench_track(id, &label)
        .expect("still on the bench")
        .origin
        .is_some());
}

#[test]
fn unfocusing_the_item_stops_the_sequence_with_a_row() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    state.start_surface_tiling(id, &label, SplitSettings::default());
    state.drive_surface_tiling();
    assert!(
        state.running_tasks().next().is_some(),
        "the first fit started"
    );

    state.unfocus_bench_item();
    state.drive_surface_tiling();

    assert!(state.surface_tiling.is_none());
    let last = rows(&state).pop().expect("a row");
    assert!(last.1, "the stop is a failed row: {last:?}");
    assert_eq!(
        last.2,
        format!("Surface tiling of {label} stopped at Fit (step 1 of 6): {label} is no longer the focused item")
    );
    // The fit it started still lands, as a pressed Fit does.
    state.finish_background_tasks();
}

#[test]
fn a_version_the_sequence_did_not_push_stops_it() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    state.start_surface_tiling(id, &label, SplitSettings::default());

    // A verdict set by hand before the first step starts.
    state
        .set_bench_verdict(id, &label, 0, Verdict::Out)
        .expect("a verdict step");
    state.settle_bench_evaluation();
    state.drive_surface_tiling();

    assert!(state.surface_tiling.is_none());
    assert!(
        state.running_tasks().next().is_none(),
        "a step started anyway"
    );
    let last = rows(&state).pop().expect("a row");
    assert!(
        last.2
            .ends_with("was changed by a step the sequence did not take"),
        "{last:?}"
    );
}

#[test]
fn a_second_move_starts_the_sequence_again_from_the_first_fit() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    state.start_surface_tiling(id, &label, SplitSettings::default());
    state.drive_surface_tiling();
    state.finish_background_tasks();
    state.settle_bench_evaluation();
    state.drive_surface_tiling();
    state.finish_background_tasks();
    assert_eq!(
        state.surface_tiling.as_ref().map(|s| s.step()),
        Some(TilingStep::SearchGeometry)
    );

    state.start_surface_tiling(id, &label, SplitSettings::default());

    assert_eq!(
        state.surface_tiling.as_ref().map(|s| s.step()),
        Some(TilingStep::Fit)
    );
    let last = rows(&state).pop().expect("a row");
    assert!(
        last.2.starts_with(&format!(
            "Surface tiling {label} again, from the start, after the move"
        )),
        "{last:?}"
    );
}

#[test]
fn the_search_starts_from_the_row_the_fit_rendered_the_patch_from() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state
        .start_bench_fit(id, &label)
        .expect("a framed track fits");
    state.finish_background_tasks();
    let track = state.bench_track(id, &label).expect("on the bench");
    let reference = crate::bench::reference_in_use(track).expect("the fit rendered a bitmap");
    assert_eq!(search_row(track), Some(reference));
}

/// A duplicate's first sequence creates a point, and a second move's sequence
/// replaces that point rather than adding another: the commit keeps the track
/// seated on the point it wrote, as a second press of *Commit* does.
#[test]
fn a_second_move_of_a_duplicate_replaces_the_point_its_first_wrote() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    // A plain duplicate, its rows still pinned `in`, so the fixture's bars
    // cannot leave the copy with too few rows to fit.
    let copy = state
        .duplicate_bench_item(id, &label)
        .expect("the copy is made");
    state.settle_bench_evaluation();
    let points = |state: &AppState| state.node(id).expect("loaded").point_count();
    let before = points(&state);

    state.start_surface_tiling(id, &copy, SplitSettings::default());
    run_sequence(&mut state);
    assert_eq!(points(&state), before + 1, "{:?}", rows(&state));

    state.action_log.clear();
    state.start_surface_tiling(id, &copy, SplitSettings::default());
    run_sequence(&mut state);
    let logged = rows(&state);
    assert_eq!(points(&state), before + 1, "{logged:?}");
    assert!(
        logged
            .iter()
            .any(|(kind, _, text)| *kind == Kind::Edit && text.contains("replacing point")),
        "{logged:?}"
    );
}

/// A move in the 3D viewer starts the sequence only in Surface Tiling Mode,
/// and only when it changed the patch.
#[test]
fn only_a_move_that_changed_the_patch_in_the_mode_starts_the_sequence() {
    use crate::bench::PatchEdit;
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let split = Some(SplitSettings::default());

    let moved = PatchEdit::Translate {
        by: [0.01, 0.0, 0.0],
    };
    state
        .edit_bench_patch_by_gesture(id, &label, &moved, None)
        .expect("a move with the mode off");
    assert!(state.surface_tiling.is_none(), "the mode is off");

    let still = PatchEdit::Translate { by: [0.0; 3] };
    state
        .edit_bench_patch_by_gesture(id, &label, &still, split)
        .expect("a move that changed nothing");
    assert!(state.surface_tiling.is_none(), "nothing moved");

    state
        .edit_bench_patch_by_gesture(id, &label, &moved, split)
        .expect("a move in the mode");
    assert_eq!(
        state.surface_tiling.as_ref().map(|s| s.step()),
        Some(TilingStep::Fit)
    );
}

// ── The ways a sequence stops ──────────────────────────────────────────

/// The last Action Log row, which must be the sequence's failed stop row.
fn stop_row(state: &AppState) -> String {
    let (kind, failed, text) = rows(state).pop().expect("a row");
    assert_eq!(kind, Kind::Bench, "{text}");
    assert!(failed, "the stop is a failed row: {text}");
    text
}

/// A step whose task fails stops the sequence, after the step's own row.
#[test]
fn a_step_whose_task_fails_stops_the_sequence() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    state.start_surface_tiling(id, &label, SplitSettings::default());
    // With no photograph to read, the fit's worker fails.
    state.photographs.clear();
    state.drive_surface_tiling();
    assert!(
        state.running_tasks().next().is_some(),
        "the first fit started"
    );
    state.finish_background_tasks();

    state.drive_surface_tiling();

    assert!(state.surface_tiling.is_none());
    let text = stop_row(&state);
    assert!(
        text.starts_with(&format!(
            "Surface tiling of {label} stopped at Fit (step 1 of 6): it did not finish: "
        )),
        "{text}"
    );
    assert!(text.contains("Cannot read"), "{text}");
}

/// A step its starter refuses stops the sequence, and starts no task.
#[test]
fn a_step_its_starter_refuses_stops_the_sequence() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    // One row left in: the fit needs two.
    for row in [1, 2] {
        state
            .set_bench_verdict(id, &label, row, Verdict::Out)
            .expect("a verdict step");
    }
    state.settle_bench_evaluation();
    state.start_surface_tiling(id, &label, SplitSettings::default());

    state.drive_surface_tiling();

    assert!(state.surface_tiling.is_none());
    assert!(state.running_tasks().next().is_none(), "a task started");
    let text = stop_row(&state);
    assert!(
        text.starts_with(&format!(
            "Surface tiling of {label} stopped at Fit (step 1 of 6): Cannot fit {label}: "
        )),
        "{text}"
    );
}

/// A refused commit writes the Commit button's own failed Edit row, then the
/// sequence's stop row, and pushes no version.
#[test]
fn a_refused_commit_stops_the_sequence_at_its_last_step() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    // One row left in: a commit needs two.
    for row in [1, 2] {
        state
            .set_bench_verdict(id, &label, row, Verdict::Out)
            .expect("a verdict step");
    }
    state.settle_bench_evaluation();
    state.start_surface_tiling(id, &label, SplitSettings::default());
    // As if the five steps before it had run.
    state.surface_tiling.as_mut().expect("started").next = TILING_STEPS.len() - 1;
    let before = versions(&state, id);

    state.drive_surface_tiling();

    assert!(state.surface_tiling.is_none());
    assert_eq!(
        versions(&state, id),
        before,
        "the refused commit pushed one"
    );
    let logged = rows(&state);
    let (kind, failed, refusal) = &logged[logged.len() - 2];
    assert_eq!((*kind, *failed), (Kind::Edit, true), "{logged:?}");
    assert!(
        refusal.starts_with(&format!("Cannot commit {label}: ")),
        "{refusal}"
    );
    assert_eq!(
        stop_row(&state),
        format!("Surface tiling of {label} stopped at Commit (step 6 of 6): {refusal}")
    );
}

/// A step waits for the item's live evaluation: with it unsettled a frame
/// starts nothing, and once it has landed the next frame starts the step.
#[test]
fn a_step_waits_until_the_evaluation_has_caught_up() {
    use crate::bench::live::Evaluation;
    use crate::bench::PatchEdit;
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    // The move gives the track new inputs, which nothing has evaluated yet.
    state
        .edit_bench_patch_by_gesture(
            id,
            &label,
            &PatchEdit::Translate {
                by: [0.01, 0.0, 0.0],
            },
            Some(SplitSettings::default()),
        )
        .expect("a move in the mode");
    assert_eq!(
        state.bench_evaluation(id, &label),
        Some(Evaluation::Evaluating)
    );

    state.drive_surface_tiling();
    assert!(
        state.running_tasks().next().is_none(),
        "the fit did not wait"
    );
    let sequence = state.surface_tiling.as_ref().expect("still under way");
    assert_eq!(
        (sequence.step(), sequence.waiting_on),
        (TilingStep::Fit, None)
    );

    state.settle_bench_evaluation();
    state.drive_surface_tiling();
    assert!(
        state.running_tasks().next().is_some(),
        "the fit did not start"
    );
    let sequence = state.surface_tiling.as_ref().expect("still under way");
    assert!(sequence.waiting_on.is_some());
    state.finish_background_tasks();
}
