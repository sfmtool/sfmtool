// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pilot vote: a sequence with no camera walked with the camera a focal
//! vote on its first chosen frames finds, a camera given in its place, no
//! pilot at a maximum gain of 1, and each reason the vote's camera is not
//! used.

use super::capture::FOCAL;
use super::density::{camera, choices, panning, run_camera};
use super::*;
use crate::analysis::pick_images::pilot::{verdict_camera, MAX_POOL_SPREAD, PILOT_FRAMES};
use crate::geometry::focal_vote::CameraModel as VoteModel;

fn pilot_options() -> PickImagesOptions {
    PickImagesOptions {
        pilot_vote: true,
        ..options()
    }
}

/// What a sequence's walk chose and how: its choices, steps, swaps and
/// local density term, without the pairs and frames the run read.
fn walked(walk: &SequenceReport) -> serde_json::Value {
    serde_json::json!({
        "chosen": walk.chosen,
        "steps": walk.steps,
        "swaps": walk.swaps,
        "density_term": walk.density_term,
    })
}

#[test]
fn a_sequence_with_no_camera_is_walked_with_the_pilot_vote_s_camera() {
    let (synth, off) = panning();
    let run = run_camera(synth, None, &pilot_options());
    assert_read_is_asked(&run);
    let walk = run.walk(0);
    let vote = walk.pilot_vote.as_ref().expect("a pilot ran");
    // The vote is on the first frames the walk without the term chooses.
    let chosen = &off.walk(0).chosen;
    assert_eq!(vote.frames, chosen[..chosen.len().min(PILOT_FRAMES)]);
    assert!(vote.frames_read >= vote.frames.len());
    assert_eq!(vote.outcome, PilotOutcome::Used);
    assert_eq!(vote.camera_model, Some("Pinhole"));
    let focal = vote.focal_px.expect("a consensus");
    assert!((focal / FOCAL - 1.0).abs() < 0.02, "{focal}");
    assert!(vote.pool_spread.expect("a consensus") <= MAX_POOL_SPREAD);
    assert!(vote.features > 0 && vote.clusters > 0 && vote.n_pool >= 2);
    // The walk starts again from the first frame with the vote's camera, and
    // chooses what the walk given that camera chooses.
    assert_eq!(walk.density_term, DensityTerm::On { focal_px: focal });
    let given = run_camera(
        synth,
        Some(verdict_camera(
            Some(VoteModel::Pinhole),
            focal,
            capture::WIDTH,
            capture::HEIGHT,
        )),
        &options(),
    );
    assert_eq!(walked(walk), walked(given.walk(0)));
    assert!(walk.steps.iter().any(|s| s.density.gain > 1.0));
}

#[test]
fn a_camera_given_wins_and_a_gain_of_1_runs_no_pilot() {
    let (synth, off) = panning();
    let given = run_camera(synth, Some(camera(1.0)), &pilot_options());
    let without = run_camera(synth, Some(camera(1.0)), &options());
    assert!(given.walk(0).pilot_vote.is_none());
    assert_eq!(decisions(&given), decisions(&without));
    let one = PickImagesOptions {
        max_density_gain: 1.0,
        ..pilot_options()
    };
    let run = run_camera(synth, None, &one);
    assert!(run.walk(0).pilot_vote.is_none());
    assert_eq!(run.waits, off.waits);
    assert_eq!(choices(&run), choices(off));
}

/// Run `synth` with a pilot and without, and check that the term is off,
/// the walk chooses what it chooses without a pilot, and return the vote.
fn unused_vote(synth: &Synth) -> PilotVote {
    let off = run(&[synth]);
    let piloted = run_with(&[synth], &pilot_options(), Fault::None, None);
    assert_read_is_asked(&piloted);
    let walk = piloted.walk(0);
    assert_eq!(
        walk.density_term,
        DensityTerm::Off(DensityTermOff::NoCamera)
    );
    assert_eq!(walked(walk), walked(off.walk(0)));
    let vote = walk.pilot_vote.clone().expect("a pilot ran");
    let chosen = &off.walk(0).chosen;
    assert_eq!(vote.frames, chosen[..chosen.len().min(PILOT_FRAMES)]);
    vote
}

#[test]
fn a_vote_whose_pool_spreads_wide_is_not_used() {
    // A camera moving sideways with no turn: the epipolar votes of pure
    // sideways motion do not observe the focal, and scatter. The walk
    // chooses more than the pilot's frames, so the pilot walk stops short of
    // the last frame.
    let synth = Synth::steady(42, 150, 3.0 * ONE_PERCENT);
    let vote = unused_vote(&synth);
    assert_eq!(vote.frames.len(), 30);
    assert_eq!(
        vote.outcome,
        PilotOutcome::WideSpread {
            max_pool_spread: MAX_POOL_SPREAD
        }
    );
    assert!(vote.pool_spread.expect("a consensus") > MAX_POOL_SPREAD);
    assert!(vote.focal_px.is_some());
}

#[test]
fn a_vote_with_no_consensus_is_not_used() {
    // Frames that match nothing: walked one at a time, and clustered into
    // no tracks a vote reads a focal from.
    let mut synth = Synth::steady(43, 8, ONE_PERCENT);
    for shot in &mut synth.shots {
        shot.content = Content::Junk;
    }
    let vote = unused_vote(&synth);
    assert_eq!(vote.outcome, PilotOutcome::NoConsensus);
    assert_eq!(vote.frames, (0..8).collect::<Vec<_>>());
    assert_eq!((vote.focal_px, vote.pool_spread), (None, None));
    assert!(vote.n_pool < 2);
}

#[test]
fn a_pilot_of_too_few_frames_runs_no_vote() {
    let vote = unused_vote(&Synth::steady(44, 2, ONE_PERCENT));
    assert_eq!(vote.outcome, PilotOutcome::TooFewFrames { min_frames: 3 });
    assert_eq!(vote.frames, vec![0, 1]);
    assert_eq!((vote.features, vote.n_pool, vote.focal_px), (0, 0, None));
}
