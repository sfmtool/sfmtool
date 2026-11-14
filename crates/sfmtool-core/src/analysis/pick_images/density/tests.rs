// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::analysis::pick_images::essential_split::{
    measure_essential_split, EssentialSplit, SplitNotMeasured,
};
use crate::camera::{CameraModel, RollingShutter, RollingShutterScanlines};
use crate::geometry::rotation::axis_angle_to_rotation_matrix;

fn split(inliers: usize, p50: f64, p90: f64) -> EssentialSplitOutcome {
    EssentialSplitOutcome::Measured(EssentialSplit {
        matches: inliers + 10,
        inliers,
        rotation_deg: 3.0,
        parallax_p50_deg: p50,
        parallax_p90_deg: p90,
    })
}

#[test]
fn a_pair_counts_only_with_enough_inliers_a_wide_p90_and_a_p50_under_its_gate() {
    let (step, reference) = (MAX_STEP_PARALLAX_P50_DEG, MAX_REFERENCE_PARALLAX_P50_DEG);
    assert_eq!((step, reference), (0.5, 0.2));
    // Counts: its ratio.
    assert_eq!(pair_measure(&split(50, 0.125, 1.5), step), 12.0);
    assert_eq!(pair_measure(&split(400, 0.25, 0.375), step), 1.5);
    assert_eq!(pair_measure(&split(50, 0.125, 1.5), reference), 12.0);
    // Each condition at its edge, then just past it.
    assert_eq!(pair_measure(&split(49, 0.1, 1.2), step), 1.0);
    assert_eq!(pair_measure(&split(80, 0.1, 0.2), step), 2.0);
    assert_eq!(pair_measure(&split(80, 0.05, 0.199), step), 1.0);
    assert_eq!(pair_measure(&split(80, 0.499, 0.998), step), 2.0);
    assert_eq!(pair_measure(&split(80, 0.5, 5.0), step), 1.0);
    // The reference gate: a P50 the step's gate passes and the reference
    // pair's does not.
    assert_eq!(pair_measure(&split(80, 0.199, 0.398), reference), 2.0);
    assert_eq!(pair_measure(&split(80, 0.2, 1.0), reference), 1.0);
    assert_eq!(pair_measure(&split(80, 0.2, 1.0), step), 5.0);
    assert_eq!(pair_measure(&split(49, 0.1, 1.2), reference), 1.0);
    assert_eq!(pair_measure(&split(80, 0.05, 0.199), reference), 1.0);
    // A P50 of 0 has no ratio, and a P90 that is not a number fails its floor.
    assert_eq!(pair_measure(&split(80, 0.0, 1.0), step), 1.0);
    assert_eq!(pair_measure(&split(80, 0.1, f64::NAN), step), 1.0);
    // Not measured.
    for reason in [
        SplitNotMeasured::TooFewMatches,
        SplitNotMeasured::NoConsensus,
        SplitNotMeasured::NoPoseInFront,
    ] {
        let outcome = EssentialSplitOutcome::NotMeasured {
            matches: 300,
            reason,
        };
        assert_eq!(pair_measure(&outcome, step), 1.0);
        assert_eq!(pair_measure(&outcome, reference), 1.0);
    }
}

/// The reference step at the default base target and maximum gain.
const R: f64 = 10.0 / 3.0;

#[test]
fn a_step_near_the_reference_step_is_measured_itself() {
    let narrow = split(400, 0.1, 1.0);
    let step = MeasurePlan::Step {
        p50_gate_deg: MAX_STEP_PARALLAX_P50_DEG,
    };
    // Up to 1.3 times the reference step, whatever was matched.
    let matched = [(105, 3.0), (110, 4.0)];
    for s in [0.5, R, 1.3 * R] {
        assert_eq!(measure_plan(100, 110, s, R, &narrow, &matched), step, "{s}");
    }
    // A step that is not finite is measured itself too.
    for s in [f64::NAN, f64::INFINITY] {
        assert_eq!(measure_plan(100, 110, s, R, &narrow, &matched), step);
    }
    // Just past it, a reference pair is.
    assert_ne!(
        measure_plan(100, 110, 1.3 * R * 1.001, R, &narrow, &matched),
        step
    );
}

#[test]
fn a_long_step_is_measured_by_a_matched_pair_near_the_reference_step_or_the_frame_its_rate_puts_there(
) {
    let narrow = split(400, 0.1, 1.0);
    // A step of 10% over 30 frames: its rate puts the reference step 10
    // frames on. Of the pairs matched, those within 0.7 to 1.3 times the
    // reference step qualify, and the nearest in ratio is measured.
    let matched = [(104, 1.5), (108, 2.5), (112, 4.0), (115, 5.0), (130, 10.0)];
    assert_eq!(
        measure_plan(100, 130, 10.0, R, &narrow, &matched),
        MeasurePlan::Reference {
            to: 112,
            matched: true
        }
    );
    // The band's ends qualify, and the nearer in ratio is taken: 1.3 times
    // the reference step is nearer than 0.7 times.
    for (n, step) in [(105, 0.7 * R), (106, 1.3 * R)] {
        assert_eq!(
            measure_plan(100, 130, 10.0, R, &narrow, &[(n, step)]),
            MeasurePlan::Reference {
                to: n,
                matched: true
            }
        );
    }
    let ends = [(105, 0.7 * R), (106, 1.3 * R)];
    assert_eq!(
        measure_plan(100, 130, 10.0, R, &narrow, &ends),
        MeasurePlan::Reference {
            to: 106,
            matched: true
        }
    );
    // Of two as near, the first.
    let tie = [(107, 3.0), (109, 3.0)];
    assert_eq!(
        measure_plan(100, 130, 10.0, R, &narrow, &tie),
        MeasurePlan::Reference {
            to: 107,
            matched: true
        }
    );
    // With none within the band, the frame the step's rate puts there.
    let outside = [(104, 0.69 * R), (115, 1.31 * R), (130, 10.0)];
    assert_eq!(
        measure_plan(100, 130, 10.0, R, &narrow, &outside),
        MeasurePlan::Reference {
            to: 110,
            matched: false
        }
    );
    assert_eq!(
        measure_plan(100, 130, 10.0, R, &narrow, &[]),
        MeasurePlan::Reference {
            to: 110,
            matched: false
        }
    );
    // A step of two frames whose rate puts the reference step one frame on.
    assert_eq!(
        measure_plan(100, 102, 6.0, R, &narrow, &[]),
        MeasurePlan::Reference {
            to: 101,
            matched: false
        }
    );
    // A step of one frame, or of two whose rate rounds to both, has no frame
    // nearer the reference step: it is measured itself, at the reference
    // gate, unless a matched pair qualifies.
    let own = MeasurePlan::Step {
        p50_gate_deg: MAX_REFERENCE_PARALLAX_P50_DEG,
    };
    assert_eq!(measure_plan(100, 101, 6.0, R, &narrow, &[]), own);
    assert_eq!(measure_plan(100, 102, 4.4, R, &narrow, &[]), own);
    assert_eq!(
        measure_plan(100, 101, 6.0, R, &narrow, &[(103, 3.5)]),
        MeasurePlan::Reference {
            to: 103,
            matched: true
        }
    );
}

#[test]
fn a_long_step_whose_own_parallax_rules_out_the_reference_gate_measures_no_pair() {
    // Scaled to the reference pair's 10 frames of 30, a P50 of 0.6° is at the
    // reference gate, whatever was matched.
    let matched = [(110, R)];
    assert_eq!(
        measure_plan(100, 130, 10.0, R, &split(50, 0.6, 3.0), &matched),
        MeasurePlan::WideParallax { to: 110 }
    );
    // Just under it, or with too few inliers, the reference pair is measured.
    let reference = MeasurePlan::Reference {
        to: 110,
        matched: true,
    };
    assert_eq!(
        measure_plan(100, 130, 10.0, R, &split(50, 0.599, 3.0), &matched),
        reference
    );
    assert_eq!(
        measure_plan(100, 130, 10.0, R, &split(49, 3.0, 9.0), &matched),
        reference
    );
    let lost = EssentialSplitOutcome::NotMeasured {
        matches: 30,
        reason: SplitNotMeasured::NoConsensus,
    };
    assert_eq!(measure_plan(100, 130, 10.0, R, &lost, &matched), reference);
}

#[test]
fn the_measure_is_read_from_the_pair_planned_at_its_gate() {
    // A split the step gate passes and the reference gate does not.
    let between = split(400, 0.3, 3.0);
    let reference = |to: usize, matched: bool| {
        assert_eq!((to, matched), (110, false));
        Ok::<_, ()>(ReferenceSplit {
            essential_split: between.clone(),
            focal_factor: 1.06,
            step_percent: 3.5,
        })
    };
    let unused = |_: usize, _: bool| -> Result<ReferenceSplit, ()> { panic!("no reference pair") };
    // The step itself, at the gate its plan names.
    for (gate, measure) in [(0.5, 10.0), (0.2, 1.0)] {
        assert_eq!(
            read_measure(MeasurePlan::Step { p50_gate_deg: gate }, &between, unused),
            Ok((measure, DensityPair::Step { p50_gate_deg: gate }))
        );
    }
    // A reference pair, at the reference gate, with what it was measured at.
    let plan = MeasurePlan::Reference {
        to: 110,
        matched: false,
    };
    let reported = DensityPair::Reference {
        to: 110,
        step_percent: 3.5,
        matched_for_measure: true,
        essential_split: between.clone(),
        focal_factor: 1.06,
        p50_gate_deg: 0.2,
    };
    assert_eq!(
        read_measure(plan, &split(400, 0.1, 3.0), reference),
        Ok((1.0, reported))
    );
    let narrow = split(400, 0.125, 1.5);
    let matched = |to: usize, matched: bool| {
        assert_eq!((to, matched), (112, true));
        Ok::<_, ()>(ReferenceSplit {
            essential_split: narrow.clone(),
            focal_factor: 1.0,
            step_percent: 3.0,
        })
    };
    let plan = MeasurePlan::Reference {
        to: 112,
        matched: true,
    };
    let (measure, reported) = read_measure(plan, &between, matched).expect("read");
    assert_eq!(measure, 12.0);
    assert!(matches!(
        reported,
        DensityPair::Reference {
            to: 112,
            matched_for_measure: false,
            ..
        }
    ));
    // No pair: a measure of 1 at the reference gate.
    assert_eq!(
        read_measure(MeasurePlan::WideParallax { to: 110 }, &narrow, unused),
        Ok((
            1.0,
            DensityPair::WideParallax {
                to: 110,
                p50_gate_deg: 0.2
            }
        ))
    );
    // An error measuring the reference pair is the reading's.
    let failed = |_: usize, _: bool| -> Result<ReferenceSplit, &str> { Err("cancelled") };
    assert_eq!(read_measure(plan, &narrow, failed), Err("cancelled"));
}

fn turned(inliers: usize, rotation_deg: f64, p50: f64) -> EssentialSplitOutcome {
    EssentialSplitOutcome::Measured(EssentialSplit {
        matches: inliers + 10,
        inliers,
        rotation_deg,
        parallax_p50_deg: p50,
        parallax_p90_deg: 1.0,
    })
}

#[test]
fn the_focal_search_runs_only_where_a_focal_could_bring_the_pair_under_a_gate() {
    assert!(focal_search_runs(&turned(400, 0.5, 0.1)));
    assert!(!focal_search_runs(&turned(400, 0.499, 0.1)));
    // A focal error within the search's 12% adds at most 0.12° of P50 per
    // degree of rotation.
    assert!(focal_search_runs(&turned(400, 10.0, 1.699)));
    assert!(!focal_search_runs(&turned(400, 10.0, 1.7)));
    // Inliers do not decide it, a split that was not measured does.
    assert!(focal_search_runs(&turned(10, 3.0, 0.1)));
    let lost = EssentialSplitOutcome::NotMeasured {
        matches: 30,
        reason: SplitNotMeasured::NoConsensus,
    };
    assert!(!focal_search_runs(&lost));
}

/// The factor of `options`' focal over 500.
fn factor_of(options: &EssentialSplitOptions) -> f64 {
    options.camera.focal_lengths().0 / 500.0
}

#[test]
fn the_focal_search_keeps_the_smallest_p50_of_a_split_with_enough_inliers() {
    let split = EssentialSplitOptions::new(pinhole(640, 480, 500.0));
    // The P50 at each factor: 0.05 per step from 1.06, and at 1.03 a P50 of
    // 0 from too few inliers.
    let run = |at: &dyn Fn(i32) -> EssentialSplitOutcome| {
        let mut asked = Vec::new();
        let kept = focal_searched_split(&split, |options| {
            let i = ((factor_of(options) - 1.0) / FOCAL_SEARCH_STEP).round() as i32;
            asked.push(i);
            Ok::<_, ()>(at(i))
        })
        .expect("no error");
        (kept, asked)
    };
    let p50 = |i: i32| 0.01 + 0.05 * f64::from((i - 2).abs());
    let ((outcome, factor), asked) = run(&|i| {
        if i == 1 {
            turned(49, 3.0, 0.0)
        } else {
            turned(400, 3.0, p50(i))
        }
    });
    assert_eq!(asked, vec![0, -4, -3, -2, -1, 1, 2, 3, 4]);
    assert_eq!(factor, 1.06);
    assert_eq!(outcome, turned(400, 3.0, 0.01));
    // Of two as small, the first; the camera's own is first of all.
    let ((_, factor), _) = run(&|i| turned(400, 3.0, if i >= 2 { 0.01 } else { 0.2 }));
    assert_eq!(factor, 1.06);
    let ((_, factor), _) = run(&|i| turned(400, 3.0, if i == 0 || i == 3 { 0.01 } else { 0.2 }));
    assert_eq!(factor, 1.0);
    // With too few inliers at the camera's focal, another focal is kept
    // whatever its P50.
    let ((_, factor), _) = run(&|i| {
        if i == 0 {
            turned(40, 3.0, 0.01)
        } else {
            turned(400, 3.0, 0.3 + 0.01 * f64::from(i))
        }
    });
    assert_eq!(factor, 0.88);
    // Where the search does not run, the camera's split alone is measured.
    let ((outcome, factor), asked) = run(&|_| turned(400, 0.4, 0.01));
    assert_eq!((asked, factor), (vec![0], 1.0));
    assert_eq!(outcome, turned(400, 0.4, 0.01));
}

/// Matched positions of a camera of focal 500 that turns by `turn_deg`
/// about the vertical and moves `travel` sideways, in front of far points,
/// 200 to 400 away, and a share `near` of near points, 4 to 8 away.
fn near_and_far(turn_deg: f64, travel: f64, near: f64) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let camera = pinhole(640, 480, 500.0);
    let mut rng = StdRng::seed_from_u64(7);
    let rotation = axis_angle_to_rotation_matrix(&Vector3::y(), turn_deg.to_radians());
    let translation = Vector3::new(travel, 0.0, 0.0);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    while a.len() < 400 {
        let pixel = [rng.random_range(0.0..640.0), rng.random_range(0.0..480.0)];
        let ray = Vector3::from(camera.pixel_to_ray(pixel[0], pixel[1]));
        let depth = if rng.random_range(0.0..1.0) < near {
            rng.random_range(4.0..8.0)
        } else {
            rng.random_range(200.0..400.0)
        };
        let seen = rotation * (ray * depth) + translation;
        let Some((u, v)) = camera.ray_to_pixel([seen.x, seen.y, seen.z]) else {
            continue;
        };
        if (0.0..640.0).contains(&u) && (0.0..480.0).contains(&v) {
            a.push(pixel);
            b.push([u, v]);
        }
    }
    (a, b)
}

#[test]
fn the_focal_search_finds_the_focal_a_turning_pair_was_taken_with() {
    // A turn of 6° with a fifth of the points near: off the true focal, the
    // turn leaves parallax on the far points.
    let (a, b) = near_and_far(6.0, 0.05, 0.2);
    let search = |focal: f64| {
        let split = EssentialSplitOptions::new(pinhole(640, 480, focal));
        focal_searched_split(&split, |options| measure_essential_split(&a, &b, options))
            .expect("measured")
    };
    let p50 = |outcome: &EssentialSplitOutcome| match outcome {
        EssentialSplitOutcome::Measured(s) => s.parallax_p50_deg,
        other => panic!("{other:?}"),
    };
    let (at_truth, factor) = search(500.0);
    assert_eq!(factor, 1.0);
    for (given, expected) in [
        (500.0 / 1.06, 1.06),
        (500.0 / 0.91, 0.91),
        (500.0 / 1.12, 1.12),
    ] {
        let (outcome, factor) = search(given);
        assert_eq!(factor, expected, "given {given}");
        assert!(
            (p50(&outcome) - p50(&at_truth)).abs() < 1e-6,
            "given {given}: {outcome:?} against {at_truth:?}"
        );
    }
}

#[test]
fn the_gain_is_the_median_measure_over_its_unit_kept_within_its_bounds() {
    // Before the first step.
    assert_eq!(gain(&[], 3.0), (1.0, 1.0));
    // One or two measures: their median.
    assert_eq!(gain(&[5.0], 3.0), (5.0, 2.0));
    assert_eq!(gain(&[1.0, 6.0], 3.0), (3.5, 1.4));
    // Three: the middle one, whatever their order.
    assert_eq!(gain(&[20.0, 1.0, 5.0], 3.0), (5.0, 2.0));
    // Below the unit the gain stays 1, above the maximum it is the maximum.
    assert_eq!(gain(&[2.0, 2.4, 1.0], 3.0), (2.0, 1.0));
    assert_eq!(gain(&[30.0, 40.0, 1.0], 3.0), (30.0, 3.0));
    assert_eq!(gain(&[30.0, 40.0, 1.0], 1.5), (30.0, 1.5));
    assert_eq!(gain(&[30.0, 40.0, 1.0], 1.0), (30.0, 1.0));
}

fn pinhole(width: u32, height: u32, f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: f,
            principal_point_x: f64::from(width) / 2.0,
            principal_point_y: f64::from(height) / 2.0,
        },
        width,
        height,
        rolling_shutter: None,
    }
}

#[test]
fn the_camera_is_scaled_to_the_frames_or_the_term_says_why_it_is_off() {
    let camera = pinhole(1920, 1080, 1500.0);
    // At the frames' size it is used as it is.
    let same = split_options(Some(&camera), 1920, 1080, 3.0).unwrap();
    assert_eq!(same.camera, camera);
    assert_eq!(same, EssentialSplitOptions::new(camera.clone()));
    // At half the size, scaled by a half.
    let half = split_options(Some(&camera), 960, 540, 3.0).unwrap();
    assert_eq!(half.camera, pinhole(960, 540, 750.0));
    // A gain of 1 measures nothing, with or without a camera.
    assert_eq!(
        split_options(Some(&camera), 1920, 1080, 1.0),
        Err(DensityTermOff::MaxGainOne)
    );
    assert_eq!(
        split_options(None, 1920, 1080, 3.0),
        Err(DensityTermOff::NoCamera)
    );
    // Another aspect cannot be scaled to.
    assert_eq!(
        split_options(Some(&camera), 1080, 1920, 3.0),
        Err(DensityTermOff::CameraShape {
            camera: [1920, 1080],
            frames: [1080, 1920],
        })
    );
    assert_eq!(
        split_options(Some(&pinhole(0, 1080, 1500.0)), 1920, 1080, 3.0),
        Err(DensityTermOff::CameraShape {
            camera: [0, 1080],
            frames: [1920, 1080],
        })
    );
    // A camera the split refuses, named by its problem.
    let mut shutter = camera.clone();
    shutter.rolling_shutter = Some(RollingShutter {
        scanlines: RollingShutterScanlines::Rows,
        rates: [0.01, 0.0, 0.0],
    });
    match split_options(Some(&shutter), 1920, 1080, 3.0) {
        Err(DensityTermOff::CameraRefused { problem }) => {
            assert!(problem.contains("SFMTOOL_PINHOLE"), "{problem}")
        }
        other => panic!("{other:?}"),
    }
    match split_options(Some(&pinhole(1920, 1080, 0.0)), 1920, 1080, 3.0) {
        Err(DensityTermOff::CameraRefused { problem }) => {
            assert!(problem.contains("focal"), "{problem}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_equidistant_fisheye_is_scaled_and_measured_with() {
    let camera = CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 300.0,
            principal_point_x: 482.0,
            principal_point_y: 478.0,
        },
        width: 960,
        height: 960,
        rolling_shutter: None,
    };
    let options = split_options(Some(&camera), 480, 480, 3.0).unwrap();
    assert_eq!(
        options.camera.model,
        CameraModel::EquidistantFisheye {
            focal_length: 150.0,
            principal_point_x: 241.0,
            principal_point_y: 239.0,
        }
    );
}

#[test]
fn the_term_serializes_with_its_status_and_reason() {
    let json = |term: &DensityTerm| serde_json::to_value(term).unwrap();
    assert_eq!(
        json(&DensityTerm::On { focal_px: 750.0 }),
        serde_json::json!({"status": "on", "focal_px": 750.0})
    );
    assert_eq!(
        json(&DensityTerm::Off(DensityTermOff::NoCamera)),
        serde_json::json!({"status": "off", "reason": "no_camera"})
    );
    assert_eq!(
        json(&DensityTerm::Off(DensityTermOff::CameraShape {
            camera: [4, 3],
            frames: [3, 4],
        })),
        serde_json::json!({"status": "off", "reason": "camera_shape",
                           "camera": [4, 3], "frames": [3, 4]})
    );
}
