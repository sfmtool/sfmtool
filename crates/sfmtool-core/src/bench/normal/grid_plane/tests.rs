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
    let result: Result<_, ()> = turn_loop(
        0usize,
        max_iterations,
        0.05,
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
    // rejected, and the better of the two frames is kept.
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
