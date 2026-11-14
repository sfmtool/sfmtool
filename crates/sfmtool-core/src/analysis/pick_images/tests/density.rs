// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The walk with the local density term: a capture with a near foreground
//! in front of a distant view walked with shorter steps, the gain each step
//! is walked with read back from the steps before it, and the walk with the
//! term at a gain of 1 or off choosing what the walk without it chooses.

use super::capture::{FOCAL, HEIGHT, WIDTH};
use super::*;
use crate::camera::{CameraIntrinsics, CameraModel, IntrinsicAdjustment};

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

/// Every step's split, and every reference pair's, of a run with `options`
/// is the one a second match of its pair measures with the split's options
/// for `camera` set, at the focal the focal search kept: the walk measures
/// it from its own match of the pair, which finds the same positions in the
/// same order.
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
    let features = |frame: usize| {
        let path = &run.sequences[0].frames[frame].images[0].sift_path;
        let (_, metadata, _) = sfmtool_sift_format::read_sift_metadata(path).expect("metadata");
        let sift = sfmtool_sift_format::read_sift_positions_descriptors(path).expect("features");
        FrameFeatures::from_sift(sift, &metadata, options.forest).expect("a forest")
    };
    // The split of `a` with `b` a second match measures at `factor` times
    // the camera's focal, the factor `1 + 0.03 i` the search measures at.
    let second = |a: usize, b: usize, factor: f64| {
        let i = ((factor - 1.0) / 0.03).round();
        let adjustment = IntrinsicAdjustment {
            focal: 0.03 * i,
            ..IntrinsicAdjustment::default()
        };
        let camera = if i == 0.0 {
            split.camera.clone()
        } else {
            split.camera.with_adjustment(&adjustment, None)
        };
        let pair_options = FramePairOptions {
            essential_split: Some(EssentialSplitOptions {
                camera,
                ..split.clone()
            }),
            ..options.pair.clone()
        };
        let stats = match_frame_pair(&features(a), &features(b), &pair_options).expect("a match");
        stats.essential_split.expect("measured")
    };
    let steps = &run.walk(0).steps;
    assert!(!steps.is_empty());
    for step in steps {
        let density = &step.density;
        assert_eq!(
            density.essential_split,
            Some(second(step.from, step.to, density.focal_factor)),
            "{} -> {}",
            step.from,
            step.to
        );
        if let Some(DensityPair::Reference {
            to,
            essential_split,
            focal_factor,
            ..
        }) = &density.measured_pair
        {
            assert_eq!(
                *essential_split,
                second(step.from, *to, *focal_factor),
                "{} -> {to}",
                step.from
            );
        }
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
                focal_factor: 1.0,
                measured_pair: None,
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
        // The measure is the one of the pair the step reports, at the gate
        // it reports: the step itself, at the step gate, up to 1.3 times the
        // reference step, and otherwise a reference pair, or none, at the
        // reference gate.
        let reference = base / 3.0;
        let mut matched_for_measure = None;
        match density.measured_pair.as_ref().expect("a pair") {
            DensityPair::Step { p50_gate_deg } => {
                let gate = if step.stats.step_percent <= 1.3 * reference {
                    0.5
                } else {
                    0.2
                };
                assert_eq!(*p50_gate_deg, gate, "step {k}");
                assert_eq!(
                    density.measure,
                    crate::analysis::pick_images::density::pair_measure(split, *p50_gate_deg),
                    "step {k}"
                );
            }
            DensityPair::Reference {
                to,
                matched_for_measure: matched,
                essential_split,
                p50_gate_deg,
                ..
            } => {
                assert!(step.stats.step_percent > 1.3 * reference, "step {k}");
                assert!((step.from + 1..=step.to).contains(to), "step {k}");
                assert_eq!(*p50_gate_deg, 0.2, "step {k}");
                assert_eq!(
                    density.measure,
                    crate::analysis::pick_images::density::pair_measure(essential_split, 0.2),
                    "step {k}"
                );
                if *matched {
                    matched_for_measure = Some(*to);
                }
            }
            DensityPair::WideParallax { .. } => assert_eq!(density.measure, 1.0, "step {k}"),
        }
        // The prediction aims at the step's own target from rates measured
        // at another, so after the first step each step in the band is
        // reached by its first try; a reference pair matched for the measure
        // is matched after it.
        if k > 0 && step.kind == StepKind::InBand {
            let expected: Vec<usize> = std::iter::once(step.to)
                .chain(matched_for_measure)
                .collect();
            assert_eq!(tries(&on, step.from), expected, "step {k}");
        }
    }
    // The near foreground reads as one: past the first steps the gain is at
    // its maximum, and the walk takes far more frames than without the term.
    // Near the end of the capture a step of three frames has a parallax P90
    // just under the 0.2° its measure needs, so the last step may be walked
    // with two such measures of the last three, and a gain of 1.
    assert_eq!(walk.steps[0].density.gain, 1.0);
    assert!(
        walk.steps[0].density.measure > 7.5,
        "{:?}",
        walk.steps[0].density
    );
    let middle = &walk.steps[1..walk.steps.len() - 1];
    assert!(
        middle.iter().all(|s| s.density.gain > 1.0),
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

/// A panning capture whose near ring holds two fifths of the points, which
/// keep their appearance over the first `steady` frames and after that
/// change it every 8 frames: a step of 3 frames matches most of them again,
/// and one of 8 frames or more, about the base target's length, none, so
/// that a step reads the near foreground only when it is short.
fn near_fading(frames: usize, steady: usize) -> Synth {
    let mut synth = Synth::panning_with_near(44, frames, 0.4);
    for shot in &mut synth.shots[steady..] {
        shot.content = Content::NearRenewed { lifetime: 8 };
    }
    synth
}

#[test]
fn the_gain_a_walk_settles_at_does_not_depend_on_where_it_starts() {
    // From the first frame the steady stretch raises the gain; from frame 40
    // the walk starts in the stretch whose long steps read no near
    // foreground. Measuring a pair of the reference step's length, both
    // walks find the near foreground there and shorten their steps.
    let synth = near_fading(130, 30);
    let gains_past = |offset: usize, frame: usize| -> Vec<f64> {
        let mut part = synth.clone();
        part.shots.drain(..offset);
        let run = run_camera(&part, Some(camera(1.0)), &options());
        run.walk(0)
            .steps
            .iter()
            .filter(|s| s.from + offset >= frame)
            .map(|s| s.density.gain)
            .collect()
    };
    for offset in [0, 40] {
        let gains = gains_past(offset, 60);
        assert!(gains.iter().all(|&g| g > 2.0), "from {offset}: {gains:?}");
        assert!(gains.len() >= 10, "from {offset}: {gains:?}");
    }
}
