// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The turn loop's acceptance and stopping rules, on a state that is a
//! number: each measurement scores the state, and each proposal is the state
//! one step on with a movement the test chooses.

use super::*;

/// A loop over states `0, 1, 2, …` whose scores are `scores[state]` and whose
/// turn from state `s` moves the cells by `moves[s]`. The proposal past the
/// last move asks for no turn.
fn run(scores: &[f64], moves: &[f64], max_iterations: u8) -> LoopEnd<usize, f64> {
    run_under(
        scores,
        moves,
        max_iterations,
        GridPlaneAcceptance::WindowedZncc,
    )
}

/// [`run`] under the acceptance rule `acceptance`.
fn run_under(
    scores: &[f64],
    moves: &[f64],
    max_iterations: u8,
    acceptance: GridPlaneAcceptance,
) -> LoopEnd<usize, f64> {
    let result: Result<_, ()> = turn_loop(
        0usize,
        max_iterations,
        0.05,
        acceptance,
        |&s| Ok(scores[s]),
        |&m| m,
        |&s, _| Ok(moves.get(s).map(|&m| (s + 1, m))),
    );
    result.expect("the closures do not fail")
}

#[test]
fn a_turn_that_lowers_the_score_is_not_kept() {
    let end = run(&[0.9, 0.5], &[1.0], 5);
    assert_eq!(end.stop, LoopStop::Rejected);
    assert_eq!(end.state, 0, "the starting state is kept");
    assert_eq!(end.measured, 0.9, "with its own measurement");
    assert_eq!(
        end.iterations, 2,
        "the rejected turn was rendered to score it"
    );
}

#[test]
fn the_tolerance_does_not_accumulate_below_the_start() {
    // The second turn falls by less than the tolerance from the first, but
    // below where the loop started.
    let tol = ACCEPT_ZNCC_TOLERANCE;
    let end = run(&[0.9, 0.9 + 0.5 * tol, 0.9 - 0.25 * tol], &[1.0, 0.5], 5);
    assert_eq!(end.stop, LoopStop::Rejected);
    assert_eq!(end.state, 1, "the floor is the start's score");

    // Within the tolerance and above the start, a fall is crossed.
    let end = run(&[0.9, 0.95, 0.95 - 0.5 * tol], &[1.0, 0.5, 0.25], 3);
    assert_eq!((end.stop, end.state), (LoopStop::Cap, 2));
}

#[test]
fn a_turn_under_the_tolerance_is_taken_without_a_render() {
    let end = run(&[0.7, 0.8, f64::NAN], &[1.0, 0.01], 5);
    assert_eq!(end.stop, LoopStop::Converged);
    assert_eq!(end.state, 2, "the last, small turn is taken");
    assert_eq!(
        end.measured, 0.8,
        "and the last render is the one before it"
    );
    assert_eq!(end.iterations, 2);
    assert_eq!(end.start_score, 0.7);
}

#[test]
fn a_turn_that_does_not_shrink_ends_on_the_better_frame() {
    // The second turn moves as far as the first: the loop is alternating. Its
    // score falls by less than the acceptance tolerance, so it is not
    // rejected, and the better of the two patch frames is kept.
    let lower = 0.8 - 0.5 * ACCEPT_ZNCC_TOLERANCE;
    let end = run(&[0.7, 0.8, lower], &[1.0, 1.0], 5);
    assert_eq!(end.stop, LoopStop::Oscillation);
    assert_eq!((end.state, end.measured), (1, 0.8), "the better of 1 and 2");

    let end = run(&[0.7, 0.8, 0.85], &[1.0, 1.0], 5);
    assert_eq!(end.stop, LoopStop::Oscillation);
    assert_eq!((end.state, end.measured), (2, 0.85));
}

#[test]
fn the_cap_leaves_the_last_turn_unscored_and_untaken() {
    let end = run(&[0.5, 0.6, 0.7, 0.8], &[1.0, 0.5, 0.25], 3);
    assert_eq!(end.stop, LoopStop::Cap);
    assert_eq!((end.state, end.iterations), (2, 3));
}

#[test]
fn no_turn_asked_and_no_score_stop_where_they_are() {
    let end = run(&[0.5], &[], 5);
    assert_eq!(
        (end.stop, end.state, end.iterations),
        (LoopStop::Measured, 0, 1)
    );

    let end = run(&[f64::NAN, 0.9], &[1.0], 5);
    assert_eq!(
        (end.stop, end.state, end.iterations),
        (LoopStop::NotRun, 0, 1)
    );
}

#[test]
fn without_acceptance_a_turn_that_lowers_the_score_is_taken() {
    // The ZNCC rule rejects the first turn; with no acceptance test it is
    // taken, and the loop goes on to converge.
    let scores = [0.9, 0.5, 0.4];
    let moves = [1.0, 0.5, 0.01];
    let end = run(&scores, &moves, 5);
    assert_eq!((end.stop, end.state), (LoopStop::Rejected, 0));
    assert_eq!(end.taken, vec![false]);

    let end = run_under(&scores, &moves, 5, GridPlaneAcceptance::None);
    assert_eq!((end.stop, end.state), (LoopStop::Converged, 3));
    assert_eq!(end.taken, vec![true, true, true]);
}

#[test]
fn without_acceptance_an_oscillation_ends_on_the_later_state() {
    let lower = 0.8 - 0.5 * ACCEPT_ZNCC_TOLERANCE;
    let end = run_under(
        &[0.7, 0.8, lower],
        &[1.0, 1.0],
        5,
        GridPlaneAcceptance::None,
    );
    assert_eq!((end.stop, end.state), (LoopStop::Oscillation, 2));
    assert_eq!(end.taken, vec![true, true]);
}

#[test]
fn the_loop_reports_which_proposals_it_took() {
    // Two turns kept, the third left untaken at the cap.
    let end = run(&[0.5, 0.6, 0.7, 0.8], &[1.0, 0.5, 0.25], 3);
    assert_eq!(end.taken, vec![true, true, false]);
    // A turn under the tolerance is taken without a render.
    let end = run(&[0.7, 0.8, f64::NAN], &[1.0, 0.01], 5);
    assert_eq!(end.taken, vec![true, true]);
}

#[test]
fn the_working_side_follows_the_reference_footprint_within_r_and_2r() {
    // A patch whose `u` axis spans 1.5 photograph px per grid px and whose
    // `v` axis spans less: the longer side, 24 · 1.5 = 36 px, rounded up.
    let j = [[1.5, 0.0], [0.0, 0.8]];
    assert_eq!(working_side(Some(j), 24), 36);
    // The longer side is read along either axis, at any angle in the image.
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert_eq!(working_side(Some([[0.3, 1.2 * s], [0.1, 1.2 * s]]), 20), 24);
    // A fraction of a pixel rounds up.
    assert_eq!(working_side(Some([[1.01, 0.0], [0.0, 1.0]]), 24), 25);
    // A whole number of pixels up to the Jacobian's error keeps that number:
    // 32.0000005 px is 32, as the tests' scene reads its 32 px patch.
    let over = 32.000_000_5 / 24.0;
    assert_eq!(working_side(Some([[over, 0.0], [0.0, over]]), 24), 32);
    // More than 2R px clamps to 2R; fewer than R clamps to R.
    assert_eq!(working_side(Some([[5.0, 0.0], [0.0, 5.0]]), 24), 48);
    assert_eq!(working_side(Some([[0.4, 0.0], [0.0, 0.6]]), 24), 24);
    // No Jacobian, or one that is not finite, is R.
    assert_eq!(working_side(None, 24), 24);
    assert_eq!(working_side(Some([[f64::NAN, 0.0], [0.0, 1.0]]), 24), 24);
}

#[test]
fn each_loop_ending_maps_to_its_outcome() {
    for (stop, outcome) in [
        (LoopStop::Converged, GridPlaneOutcome::Turned),
        (LoopStop::NotRun, GridPlaneOutcome::Undetermined),
        (LoopStop::Measured, GridPlaneOutcome::Undetermined),
        (LoopStop::Rejected, GridPlaneOutcome::Unstable),
        (LoopStop::Oscillation, GridPlaneOutcome::Unstable),
        (LoopStop::Cap, GridPlaneOutcome::Unstable),
    ] {
        assert_eq!(GridPlaneOutcome::of(stop), outcome, "{stop:?}");
    }
}

/// The step's cell ZNCC bar is its own 0.6; the piecewise refinement of
/// cluster patches keeps 0.8.
#[test]
fn the_step_and_the_piecewise_refinement_keep_separate_cell_zncc_bars() {
    assert_eq!(BENCH_MIN_CELL_ZNCC, 0.6);
    assert_eq!(GridPlaneNormalOptions::default().min_cell_zncc, 0.6);
    assert_eq!(crate::patch::cluster_refine::DEFAULT_MIN_CELL_ZNCC, 0.8);
    assert_eq!(PiecewiseParams::default().min_cell_zncc, 0.8);
}
