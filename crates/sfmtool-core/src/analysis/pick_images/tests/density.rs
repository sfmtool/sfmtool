// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The walk with the local density term: a capture with a near foreground
//! in front of a distant view walked with shorter steps, the gain each step
//! is walked with read back from the steps before it, and the walk with the
//! term at a gain of 1 or off choosing what the walk without it chooses.

use super::capture::{FOCAL, HEIGHT, WIDTH};
use super::*;
use crate::camera::{CameraIntrinsics, CameraModel};

/// The synthetic captures' camera, at `scale` times their size.
pub(super) fn camera(scale: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: FOCAL * scale,
            principal_point_x: f64::from(WIDTH) / 2.0 * scale,
            principal_point_y: f64::from(HEIGHT) / 2.0 * scale,
        },
        width: (f64::from(WIDTH) * scale) as u32,
        height: (f64::from(HEIGHT) * scale) as u32,
        rolling_shutter: None,
    }
}

/// Run `synth` as one sequence with `camera`.
pub(super) fn run_camera(
    synth: &Synth,
    camera: Option<CameraIntrinsics>,
    options: &PickImagesOptions,
) -> Run {
    let dir = tempfile::tempdir().expect("a temporary folder");
    let mut sequence = synth.sequence(&dir.path().join("seq0"));
    sequence.camera = camera;
    let mut source = SynthSource::new(&[synth], THREADS, Fault::None);
    let result = pick_images(
        std::slice::from_ref(&sequence),
        &mut source,
        options,
        &Progress::none(),
    );
    let Asked { waits, prefetched } = source.into_asked();
    Run {
        result,
        asked: waits.iter().flatten().copied().collect(),
        waits,
        prefetched,
        sequences: vec![sequence],
        dir,
    }
}

/// [`decisions`] without what the local density term reports: each step's
/// density and the sequence's term.
pub(super) fn choices(run: &Run) -> serde_json::Value {
    let mut value = decisions(run);
    for sequence in value["sequences"].as_array_mut().expect("sequences") {
        sequence["density_term"] = serde_json::Value::Null;
        for step in sequence["steps"].as_array_mut().expect("steps") {
            step["density"] = serde_json::Value::Null;
        }
    }
    value
}

/// Every step's split, of a run with `options`, is the one a second match
/// of its pair, with the split's options for `camera` set, measures: the walk
/// measures it from its own match of the pair, which finds the same positions
/// in the same order.
pub(super) fn assert_splits_are_those_of_a_second_match(
    run: &Run,
    options: &PickImagesOptions,
    camera: &CameraIntrinsics,
) {
    let split = crate::analysis::pick_images::density::split_options(
        Some(camera),
        WIDTH,
        HEIGHT,
        options.max_density_gain,
    )
    .expect("the term is on");
    let pair_options = FramePairOptions {
        essential_split: Some(split),
        ..options.pair.clone()
    };
    let features = |frame: usize| {
        let path = &run.sequences[0].frames[frame].images[0].sift_path;
        let (_, metadata, _) = sfmtool_sift_format::read_sift_metadata(path).expect("metadata");
        let sift = sfmtool_sift_format::read_sift_features(path).expect("features");
        FrameFeatures::from_sift(sift, &metadata, options.forest).expect("a forest")
    };
    let steps = &run.walk(0).steps;
    assert!(!steps.is_empty());
    for step in steps {
        let stats = match_frame_pair(&features(step.from), &features(step.to), &pair_options)
            .expect("a match");
        assert!(stats.essential_split.is_some());
        assert_eq!(
            step.density.essential_split, stats.essential_split,
            "{} -> {}",
            step.from, step.to
        );
    }
}

/// Every step was walked with the base target and measured nothing.
fn assert_base_target_only(walk: &SequenceReport) {
    let base = options().target_step_percent;
    for step in &walk.steps {
        assert_eq!(
            step.density,
            StepDensity {
                target_percent: base,
                gain: 1.0,
                median_measure: 1.0,
                essential_split: None,
                measure: 1.0,
            },
            "{} -> {}",
            step.from,
            step.to
        );
    }
}

/// The panning capture with no camera, its walk shared by the tests that
/// compare with it.
pub(super) fn panning() -> &'static (Synth, Run) {
    static PANNING: OnceLock<(Synth, Run)> = OnceLock::new();
    PANNING.get_or_init(|| {
        let synth = Synth::panning(41, 120);
        let run = run_camera(&synth, None, &options());
        (synth, run)
    })
}

#[test]
fn a_near_foreground_shortens_the_steps_by_the_gain_of_the_last_three() {
    let (synth, off) = panning();
    let on = run_camera(synth, Some(camera(1.0)), &options());
    assert_read_is_asked(&on);
    assert_splits_are_those_of_a_second_match(&on, &options(), &camera(1.0));
    let walk = on.walk(0);
    assert_eq!(walk.density_term, DensityTerm::On { focal_px: FOCAL });
    let base = options().target_step_percent;
    let (_, high) = band();
    let mut gained = 0;
    for (k, step) in walk.steps.iter().enumerate() {
        let density = &step.density;
        // Every step's split is measured once it is accepted.
        let split = density.essential_split.as_ref().expect("measured");
        assert!(
            matches!(split, EssentialSplitOutcome::Measured(_)),
            "{split:?}"
        );
        // Each step is walked with the gain of the median of the measures
        // of the last three steps before it.
        let before: Vec<f64> = walk.steps[k.saturating_sub(3)..k]
            .iter()
            .map(|s| s.density.measure)
            .collect();
        let m = if before.is_empty() {
            1.0
        } else {
            median(&before)
        };
        assert_eq!(density.median_measure, m, "step {k}");
        assert_eq!(density.gain, (m / 2.5).clamp(1.0, 3.0), "step {k}");
        assert_eq!(density.target_percent, base / density.gain, "step {k}");
        // A step in the band is in its own target's band.
        if step.kind == StepKind::InBand {
            let t = density.target_percent;
            assert!(
                (0.7 * t..=1.3 * t).contains(&step.stats.step_percent),
                "step {k}: {} against {t}",
                step.stats.step_percent
            );
        }
        if density.gain > 1.0 {
            gained += 1;
            // Shorter than the base band allows.
            assert!(step.stats.step_percent < high, "step {k}");
        }
        // The prediction aims at the step's own target from rates measured
        // at another, so after the first step each step in the band is
        // reached by its first try.
        if k > 0 && step.kind == StepKind::InBand {
            assert_eq!(tries(&on, step.from), vec![step.to], "step {k}");
        }
    }
    // The near foreground reads as one: past the first steps the gain is at
    // its maximum, and the walk takes far more frames than without the term.
    assert_eq!(walk.steps[0].density.gain, 1.0);
    assert!(
        walk.steps[0].density.measure > 7.5,
        "{:?}",
        walk.steps[0].density
    );
    assert!(
        gained + 1 >= walk.steps.len(),
        "{gained} of {}",
        walk.steps.len()
    );
    let without = off.walk(0);
    assert_base_target_only(without);
    assert!(
        walk.chosen.len() >= 2 * without.chosen.len(),
        "{} against {}",
        walk.chosen.len(),
        without.chosen.len()
    );
    // The pairs matched to decide carry no split: only the accepted steps
    // are measured.
    assert!(on
        .report()
        .pairs
        .iter()
        .all(|p| p.stats.essential_split.is_none()));
}

#[test]
fn a_gain_of_1_chooses_what_the_walk_without_a_camera_chooses() {
    let (synth, off) = panning();
    let one = PickImagesOptions {
        max_density_gain: 1.0,
        ..options()
    };
    let run = run_camera(synth, Some(camera(1.0)), &one);
    assert_eq!(choices(&run), choices(off));
    assert_eq!(run.waits, off.waits);
    let walk = run.walk(0);
    assert_eq!(
        walk.density_term,
        DensityTerm::Off(DensityTermOff::MaxGainOne)
    );
    assert_base_target_only(walk);
    assert_eq!(
        off.walk(0).density_term,
        DensityTerm::Off(DensityTermOff::NoCamera)
    );
}

#[test]
fn steps_that_never_raise_the_gain_choose_what_the_walk_without_the_term_does() {
    // A wall seen while moving sideways: every step's parallax is wide, so
    // no measure counts, and the term, on, leaves every target at the base.
    let synth = eventful();
    let off = run(&[&synth]);
    let on = run_camera(&synth, Some(camera(1.0)), &options());
    assert_eventful(&on);
    let walk = on.walk(0);
    assert_eq!(walk.density_term, DensityTerm::On { focal_px: FOCAL });
    assert!(walk
        .steps
        .iter()
        .all(|s| s.density.essential_split.is_some()
            && s.density.gain == 1.0
            && s.density.target_percent == options().target_step_percent));
    assert_eq!(choices(&on), choices(&off));
    assert_eq!(on.waits, off.waits);
}

#[test]
fn a_camera_of_another_size_is_scaled_and_one_of_another_shape_turns_the_term_off() {
    let (synth, _) = panning();
    // Twice the frames' size: scaled to them, the walk is the one at their
    // size, measured with the same camera.
    let at_size = run_camera(synth, Some(camera(1.0)), &options());
    let double = run_camera(synth, Some(camera(2.0)), &options());
    assert_eq!(decisions(&double), decisions(&at_size));
    // Another aspect: off, saying why, and walked as without a camera.
    let mut tall = camera(1.0);
    (tall.width, tall.height) = (HEIGHT, WIDTH);
    let run = run_camera(synth, Some(tall), &options());
    assert_eq!(
        run.walk(0).density_term,
        DensityTerm::Off(DensityTermOff::CameraShape {
            camera: [HEIGHT, WIDTH],
            frames: [WIDTH, HEIGHT],
        })
    );
    assert_base_target_only(run.walk(0));
    assert_eq!(choices(&run), choices(&panning().1));
}

#[test]
fn a_camera_the_split_refuses_turns_the_term_off_saying_why() {
    let synth = Synth::steady(42, 12, ONE_PERCENT);
    let mut flat = camera(1.0);
    flat.model = CameraModel::SimplePinhole {
        focal_length: 0.0,
        principal_point_x: 320.0,
        principal_point_y: 240.0,
    };
    let run = run_camera(&synth, Some(flat), &options());
    match &run.walk(0).density_term {
        DensityTerm::Off(DensityTermOff::CameraRefused { problem }) => {
            assert!(problem.contains("focal"), "{problem}")
        }
        other => panic!("{other:?}"),
    }
    assert_base_target_only(run.walk(0));
    // A lone frame reports the term too.
    let lone = Synth::steady(43, 1, ONE_PERCENT);
    let run = run_camera(&lone, Some(camera(1.0)), &options());
    assert_eq!(
        run.walk(0).density_term,
        DensityTerm::On { focal_px: FOCAL }
    );
    assert!(run.walk(0).steps.is_empty());
}
