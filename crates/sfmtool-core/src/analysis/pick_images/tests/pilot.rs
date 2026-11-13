// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pilot vote: a sequence with no camera walked with the camera a focal
//! vote on its first chosen frames finds, a camera given in its place, no
//! pilot at a maximum gain of 1, and each reason the vote's camera is not
//! used.

use std::time::Duration;

use super::capture::FOCAL;
use super::density::{
    assert_splits_are_those_of_a_second_match, camera, choices, panning, run_camera,
};
use super::*;
use crate::analysis::pick_images::pilot::{verdict_camera, MAX_POOL_SPREAD, PILOT_FRAMES};
use crate::geometry::focal_vote::CameraModel as VoteModel;

/// The default options, whose pilot vote is on, with the match floor of
/// [`options`].
fn pilot_options() -> PickImagesOptions {
    PickImagesOptions {
        min_step_matches: FLOOR,
        ..PickImagesOptions::default()
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

/// With the default options, which do not name the pilot vote, a sequence
/// with no camera runs it.
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
    let camera = verdict_camera(
        Some(VoteModel::Pinhole),
        focal,
        capture::WIDTH,
        capture::HEIGHT,
    );
    let given = run_camera(synth, Some(camera.clone()), &options());
    assert_eq!(walked(walk), walked(given.walk(0)));
    // The steps the walk takes again from the pilot's pairs are measured
    // from the positions the pilot kept.
    assert_splits_are_those_of_a_second_match(&run, &pilot_options(), &camera);
    assert!(walk.steps.iter().any(|s| s.density.gain > 1.0));
    // The run did every kind of work its times count, and they add up.
    assert_times_add_up(&run);
    let times = &run.report().times;
    for (name, seconds) in walk_thread_times(times) {
        assert!(
            name == "other" || name == "stopping_source" || seconds > 0.0,
            "{name}: {seconds}"
        );
    }
    assert!(vote.seconds > 0.0 && times.source.seconds > 0.0);
}

/// A source that takes `DELAY` per frame on one worker makes the pilot walk
/// last at least `DELAY` per frame it read; the vote's own time is the
/// clustering and the vote alone, and the walk's thread spends the delay
/// waiting for frames. The frames match nothing, so the pilot walk reads
/// every frame and the vote, on few features, is quick next to the delay.
/// With no prefetch the worker starts a frame only when the walk waits for
/// it, so each wait lasts at least `DELAY` however long the walk's own reads
/// take.
#[test]
fn the_pilot_vote_s_time_leaves_out_the_extraction_of_its_frames() {
    const DELAY: Duration = Duration::from_millis(25);
    let mut synth = Synth::steady(43, 8, ONE_PERCENT);
    for shot in &mut synth.shots {
        shot.content = Content::Junk;
    }
    let dir = tempfile::tempdir().expect("a temporary folder");
    let sequence = synth.sequence(&dir.path().join("seq0"));
    let mut source = SynthSource::slowed(&[&synth], 1, Fault::None, DELAY);
    let report = pick_images(
        std::slice::from_ref(&sequence),
        &mut source,
        &PickImagesOptions {
            prefetch: false,
            ..pilot_options()
        },
        &Progress::none(),
    )
    .expect("the walk runs");
    let vote = report.sequences[0]
        .pilot_vote
        .as_ref()
        .expect("a pilot ran");
    assert!(vote.features > 0, "a vote ran");
    // The one worker wrote each frame the pilot read after a delay of its
    // own, started by the walk's wait for it, so the pilot walk waited at
    // least this long.
    let pilot_extraction = vote.frames_read as f64 * DELAY.as_secs_f64();
    assert!(
        vote.seconds < 0.5 * pilot_extraction,
        "the vote took {} s, the pilot's frames at least {pilot_extraction} s",
        vote.seconds
    );
    assert!((report.times.pilot_vote_seconds - vote.seconds).abs() < 1e-9);
    assert!(
        report.times.waiting_for_frames_seconds >= pilot_extraction,
        "{:?}",
        report.times
    );
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
    assert_eq!(vote.seconds, 0.0);
}
