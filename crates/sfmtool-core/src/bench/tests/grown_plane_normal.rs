// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The grown plane normal, decided against synthetic planes seen by pinholes
//! that look at the point from around it: a textured plane seen out to 50
//! degrees, one view whose top cells see something else, views all within a
//! few degrees, a view that sees another plane, repeating texture seen from
//! poses that are off with keypoints on the content, keypoints off the
//! content in views whose poses are further off, views on opposite sides of
//! the reference whose pairs sharpen the plane, a selection that matches a
//! few of many views, a flat wall, and large turns on weak and on strong
//! evidence, and the ladder's selection near to far: its order, a refused
//! rung retried from the starting plane, its end at the widest view that
//! passes, and a view at the reference's centre.

use std::sync::atomic::{AtomicBool, Ordering};

use super::scene::{fixture_of, Scene, PLANE_Z, WORLD};
use super::*;
use crate::bench::{
    GrownPlaneNormalOptions, GrownPlaneOutcome, GrownPlaneRefusal, GrownPlaneSequence,
    GrownPlaneStart, GrownPlaneTiltStart, GrownPlaneTryOutcome, GrownPlaneView,
};
use crate::camera::CameraModel;

/// A half-length that projects to 16 px in a view facing the plane from the
/// scenes' distance, as in the grid plane normal's tests.
const HALF: f64 = 0.4;

/// Every camera's distance from the point.
const DISTANCE: f64 = 4.0;

fn camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 160.0,
            focal_length_y: 160.0,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
        },
        width: 128,
        height: 128,
    }
}

/// A camera centre `DISTANCE` from the point, `from` degrees off the plane's
/// normal toward the azimuth `toward` degrees.
fn centre(from: f64, toward: f64) -> [f64; 3] {
    let (t, p) = (from.to_radians(), toward.to_radians());
    [
        WORLD.x + DISTANCE * t.sin() * p.cos(),
        WORLD.y + DISTANCE * t.sin() * p.sin(),
        WORLD.z - DISTANCE * t.cos(),
    ]
}

/// Where the ray from `c` along `d` meets the plane `z = depth`.
fn hit(c: [f64; 3], d: [f64; 3], depth: f64) -> (f64, f64) {
    let lambda = (depth - c[2]) / d[2];
    (c[0] + lambda * d[0], c[1] + lambda * d[1])
}

/// Value noise: smooth steps between pseudo-random levels on a lattice
/// `spacing` apart, in `128 ± amplitude`, so that no shift of it resembles
/// another. `seed` picks the levels.
fn noise(x: f64, y: f64, spacing: f64, amplitude: f64, seed: u64) -> f64 {
    let level = |i: i64, j: i64| {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ seed.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        h ^= h >> 29;
        h = h.wrapping_mul(0x1656_67B1_9E37_79F9);
        h ^= h >> 32;
        (h % 1024) as f64 / 1023.0 * 2.0 - 1.0
    };
    let (u, v) = (x / spacing, y / spacing);
    let (i, j) = (u.floor() as i64, v.floor() as i64);
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let (s, t) = (smooth(u - u.floor()), smooth(v - v.floor()));
    let top = level(i, j) * (1.0 - s) + level(i + 1, j) * s;
    let bottom = level(i, j + 1) * (1.0 - s) + level(i + 1, j + 1) * s;
    128.0 + amplitude * (top * (1.0 - t) + bottom * t)
}

/// The plane's texture: value noise about 2.5 px a cell in a facing view.
fn plane_texture(x: f64, y: f64) -> f64 {
    noise(x, y, 0.06, 60.0, 1)
}

/// What a ray sees of the textured plane `z = PLANE_Z`.
fn textured_plane(c: [f64; 3], d: [f64; 3]) -> f64 {
    let (x, y) = hit(c, d, PLANE_Z);
    plane_texture(x, y)
}

/// A version holding one point at [`WORLD`], seen by every image of `scene`,
/// on a patch facing the cameras.
fn edited_of(scene: &Scene) -> EditedReconstruction {
    let patch = OrientedPatch::from_center_normal(
        WORLD,
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [HALF, HALF],
    );
    let observing: Vec<u32> = (0..scene.len() as u32).collect();
    EditedReconstruction::new(Arc::new(fixture_of(scene, WORLD, 1.0, &observing, &patch)))
}

/// The plane's outward normal, toward the cameras.
fn truth() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -1.0)
}

fn degrees(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    a.normalize()
        .dot(&b.normalize())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// The fixture's track, its normal turned `by` degrees about the unit `axis`
/// off the plane. The turn keeps every keypoint at the point's projection,
/// since each sighting's offset from the centre is zero.
fn tilted_track(edited: &EditedReconstruction, axis: Vector3<f64>, by: f64) -> EditableTrack {
    let (bench, label) = bench_with_point(edited, 0);
    let track = track_of(&bench, &label);
    let turn =
        nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), by.to_radians());
    let (tilted, report) = tilt_patch(&track, edited, turn * truth()).expect("a finite track");
    assert!(report.changed && report.stopped.is_none());
    tilted
}

fn run(
    scene: &Scene,
    edited: &EditedReconstruction,
    track: &EditableTrack,
    options: &GrownPlaneNormalOptions,
) -> (EditableTrack, NormalReport) {
    grown_plane_normal(track, edited, &scene.views(), options, &Progress::none())
        .expect("the step runs on a finite track")
}

fn sequence_of(report: &NormalReport) -> &GrownPlaneSequence {
    match &report.estimate {
        NormalEstimate::GrownPlane(sequence) => sequence,
        other => panic!("a grown plane estimate, not {other:?}"),
    }
}

/// The keypoint `track` holds for `row`.
fn keypoint_of(track: &EditableTrack, row: usize) -> Option<[f64; 2]> {
    track.observations[row]
        .track
        .as_ref()
        .and_then(|m| m.keypoint)
        .map(|k| [f64::from(k[0]), f64::from(k[1])])
}

/// The track's row in image `image`.
fn row_of(track: &EditableTrack, image: u32) -> usize {
    track
        .observations
        .iter()
        .position(|o| o.image == image)
        .expect("every image observes the point")
}

fn distance_px(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Six views of the textured plane, from facing it to 50 degrees off.
fn wide_centres() -> Vec<[f64; 3]> {
    vec![
        centre(0.0, 0.0),
        centre(8.0, 30.0),
        centre(15.0, 150.0),
        centre(25.0, 260.0),
        centre(38.0, 60.0),
        centre(50.0, 200.0),
    ]
}

/// From 25 degrees off, the patch is turned to within a degree of the plane.
/// The plane's `σ` stays over the target, so the selection matches all five
/// views; every one joins on its growth with no search, each written
/// keypoint is at the point's projection and the reference keeps its own, and
/// each joined view's score at the final plane is its held score. The two
/// pairs are kept, and halving their information raises `σ`.
#[test]
fn a_textured_plane_started_far_off_is_turned_onto_it() {
    let centres = wide_centres();
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::new(1.0, 0.5, 0.0), 25.0);
    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);

    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3} deg off: {report}");
    assert_eq!(sequence.matched(), centres.len() - 1, "{sequence:#?}");
    assert_eq!(sequence.picks.len(), centres.len() - 1, "{sequence:#?}");
    assert_eq!(sequence.joined(), centres.len() - 1, "{report}");
    assert_eq!(sequence.left_out(), 0, "{report}");
    assert_eq!(sequence.shift_only(), 0, "{report}");
    for entry in &sequence.trace {
        assert_eq!(entry.attempts.len(), 1, "{entry:#?}");
        assert_eq!(entry.attempts[0].start, GrownPlaneStart::NoSearch);
    }
    let precision = sequence.precision.expect("views joined");
    assert!(precision.sigma_weak_deg >= precision.sigma_strong_deg);
    assert!(precision.discounted_sigma_weak_deg > precision.sigma_weak_deg);
    assert!(!precision.low_confidence, "{report}");
    // The centre does not move, and the track is read back.
    assert!((patch_frame_of(&turned).center - WORLD).norm() < 1e-9);
    assert!(turned.track().expect("track stage").bitmap.is_some());

    let reference = sequence.reference.expect("a reference");
    assert_eq!(
        keypoint_of(&turned, reference),
        keypoint_of(&track, reference),
        "the reference keeps its keypoint"
    );
    for (row, view) in sequence.views.iter().enumerate() {
        if let GrownPlaneView::Joined {
            keypoint,
            grid_zncc,
            held_zncc,
            ..
        } = view
        {
            // The score at the final plane is the last held score.
            assert_eq!(Some(*grid_zncc), *held_zncc, "row {row}");
            let written = keypoint.expect("on the photograph");
            let truth = scene.project(track.observations[row].image as usize, WORLD);
            assert!(
                distance_px(written, truth) < 0.5,
                "row {row}: {written:?} vs {truth:?}"
            );
            let held = keypoint_of(&turned, row).expect("written");
            assert!(distance_px(held, written) < 1e-3);
        }
    }
    assert_eq!(sequence.written, centres.len(), "{report}");
    assert_eq!(sequence.pairs.len(), 2, "{sequence:#?}");
    assert_eq!(sequence.pairs_kept(), 2, "{sequence:#?}");
    let sentence = report.to_string();
    assert!(
        sentence.starts_with("5 of 5 views joined, 2 of 2 pairs kept; "),
        "{sentence}"
    );
    assert!(sentence.contains("6 keypoints written"), "{sentence}");
}

/// A view whose top cells see another surface still joins under the robust
/// cell score, and the plane is not pulled off by it. At the plane the step
/// ends on, that view's robust score is 0.997 where its ZNCC over the whole
/// grid is 0.19, under the 0.5 a view needs to join.
#[test]
fn a_view_whose_top_cells_see_other_texture_still_joins() {
    let centres = wide_centres();
    let occluded = centres[3];
    let scene = Scene::looking_at(&centres, WORLD, camera(), |c, d| {
        let (x, y) = hit(c, d, PLANE_Z);
        // Something in front of the plane's top third, seen by one camera
        // only: black and white blocks the reference never sees there.
        if c == occluded && y > HALF / 3.0 + 0.03 {
            if noise(x, y, 0.08, 1.0, 9) > 128.0 {
                250.0
            } else {
                5.0
            }
        } else {
            plane_texture(x, y)
        }
    });
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::x(), 15.0);
    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3} deg off: {report}");
    let row = row_of(&track, 3);
    assert!(
        matches!(sequence.views[row], GrownPlaneView::Joined { .. }),
        "{report}: {:#?}",
        sequence.views[row]
    );
}

/// Views all within three degrees of the reference.
fn narrow_centres() -> Vec<[f64; 3]> {
    vec![
        centre(0.0, 0.0),
        centre(2.0, 0.0),
        centre(3.0, 120.0),
        centre(2.5, 240.0),
    ]
}

/// Views all within three degrees of the reference turn the patch, with a
/// low confidence. The turn is under three times its `σ`, weak evidence, but
/// under the 70 degrees the weak-turn rule starts at, so it is made.
#[test]
fn narrow_views_turn_the_patch_with_low_confidence() {
    let centres = narrow_centres();
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::y(), 20.0);
    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);
    assert!(sequence.turned(), "{report}");
    assert_eq!(sequence.outcome, GrownPlaneOutcome::Turned);
    assert!(report.tilt.degrees > 0.0);
    let precision = sequence.precision.expect("views joined");
    assert!(precision.low_confidence, "{report}");
    assert!(precision.sigma_weak_deg > 2.0, "{report}");
    assert!(report.to_string().contains("low confidence"), "{report}");
    // A small turn on weak evidence: the rule does not apply under 70 degrees.
    let turn = sequence.turn_deg.expect("views joined");
    assert!(turn < 70.0, "{report}");
    assert!(
        sequence.turn_sigma_ratio().expect("views joined") < 3.0,
        "{report}"
    );
    assert!((turn - report.tilt.degrees).abs() < 1e-9, "{report}");
    let _ = turned;
}

/// The narrow views' turn of about 20 degrees on a `σ` of about 10, with the
/// weak-turn rule starting at 10 degrees: a large turn on weak evidence. The
/// step keeps the starting normal and every keypoint, fits no view's shift
/// alone, and still reports the plane it found, its turn, its precision and
/// the views that joined.
#[test]
fn a_large_turn_on_weak_evidence_keeps_the_start() {
    let centres = narrow_centres();
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::y(), 20.0);
    let options = GrownPlaneNormalOptions {
        weak_turn_deg: 10.0,
        ..Default::default()
    };
    let (kept, report) = run(&scene, &edited, &track, &options);
    let sequence = sequence_of(&report);
    assert_eq!(sequence.outcome, GrownPlaneOutcome::WeakTurn, "{report}");
    assert!(!sequence.turned());
    assert_eq!(report.tilt.degrees, 0.0, "{report}");
    let (a, b) = (patch_frame_of(&track), patch_frame_of(&kept));
    assert_eq!(
        (a.u_axis, a.v_axis, a.center),
        (b.u_axis, b.v_axis, b.center)
    );
    for row in 0..track.observations.len() {
        assert_eq!(
            keypoint_of(&kept, row),
            keypoint_of(&track, row),
            "row {row}"
        );
    }
    assert_eq!(sequence.written, 0);
    assert_eq!(sequence.shift_only(), 0, "{report}");

    // What was found is still reported: the plane, near the truth, its turn
    // from the start, and its precision, weak.
    let found = Vector3::from(sequence.normal.expect("a plane was found"));
    assert!(degrees(found, truth()) < 2.0, "{report}");
    let turn = sequence.turn_deg.expect("a plane was found");
    assert!(
        (turn - degrees(found, patch_frame_of(&track).normal())).abs() < 1e-6,
        "{report}"
    );
    assert!(turn >= options.weak_turn_deg, "{report}");
    let precision = sequence.precision.expect("a plane was found");
    let ratio = sequence.turn_sigma_ratio().expect("a plane was found");
    assert!((ratio - turn / precision.sigma_weak_deg).abs() < 1e-12);
    assert!(ratio < options.min_turn_sigma_ratio, "{report}");
    assert_eq!(sequence.joined(), centres.len() - 1, "{report}");
    assert!(!sequence.picks.is_empty());
    for view in &sequence.views {
        if let GrownPlaneView::Joined { keypoint, .. } = view {
            assert_eq!(*keypoint, None);
        }
    }
    assert_eq!(
        sequence.views[sequence.reference.expect("a reference")],
        GrownPlaneView::Reference { keypoint: None }
    );
    let sentence = report.to_string();
    assert!(
        sentence.contains("normal left unchanged: large turn on weak evidence"),
        "{sentence}"
    );
    assert!(sentence.contains("turned 0.0\u{b0}"), "{sentence}");

    // With the rule's ratio at 0 it never applies, and the same scene turns.
    let never = GrownPlaneNormalOptions {
        min_turn_sigma_ratio: 0.0,
        ..options.clone()
    };
    let (_, report) = run(&scene, &edited, &track, &never);
    assert_eq!(sequence_of(&report).outcome, GrownPlaneOutcome::Turned);
}

/// The wide views' turn of 25 degrees, with the weak-turn rule starting at
/// 10: a large turn, but on strong evidence, many times its `σ`, so it is
/// made.
#[test]
fn a_large_turn_on_strong_evidence_is_made() {
    let centres = wide_centres();
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::new(1.0, 0.5, 0.0), 25.0);
    let options = GrownPlaneNormalOptions {
        weak_turn_deg: 10.0,
        ..Default::default()
    };
    let (turned, report) = run(&scene, &edited, &track, &options);
    let sequence = sequence_of(&report);
    assert_eq!(sequence.outcome, GrownPlaneOutcome::Turned, "{report}");
    let turn = sequence.turn_deg.expect("views joined");
    assert!(turn >= options.weak_turn_deg, "{report}");
    assert!(
        sequence.turn_sigma_ratio().expect("views joined") >= 3.0 * options.min_turn_sigma_ratio,
        "{report}"
    );
    assert!((turn - report.tilt.degrees).abs() < 1e-9, "{report}");
    assert!(
        degrees(patch_frame_of(&turned).normal(), truth()) < 1.0,
        "{report}"
    );
    assert_eq!(sequence.written, centres.len(), "{report}");
}

/// A view that sees the texture on a plane turned 25 degrees from the one
/// the others see is left out, and the plane is the others'. The selection
/// matches every view; this one passes the floor on its own, the combination
/// leaves it out as too far, its search lands on its keypoint-implied shift,
/// and it stays out.
#[test]
fn a_view_whose_own_plane_disagrees_is_left_out() {
    let centres = wide_centres();
    let scene = disagreeing_scene();
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::y(), 10.0);
    let row = row_of(&track, 4);
    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3} deg off: {report}");
    assert_eq!(sequence.matched(), centres.len() - 1, "{sequence:#?}");
    match &sequence.views[row] {
        GrownPlaneView::LeftOut {
            reason, distance, ..
        } => {
            assert_eq!(*reason, GrownPlaneRefusal::TooFar);
            assert!(distance.expect("a combined plane") > 3.0, "{distance:?}");
        }
        other => panic!("left out, not {other:?}: {report}"),
    }
    assert_eq!(sequence.joined(), centres.len() - 2, "{report}");
    // No kept pair holds it.
    for pair in &sequence.pairs {
        assert!(pair.refusal.is_some() || !pair.observations.contains(&row));
    }
    let tries: Vec<_> = sequence
        .trace
        .iter()
        .filter(|t| t.observation == row)
        .collect();
    let outcomes: Vec<_> = tries.iter().map(|t| t.outcome).collect();
    assert_eq!(
        outcomes,
        [GrownPlaneTryOutcome::Matched, GrownPlaneTryOutcome::LeftOut],
        "{tries:#?}"
    );
    assert_eq!(tries[0].attempts.len(), 1);
    assert_eq!(tries[0].attempts[0].start, GrownPlaneStart::NoSearch);
    assert!(tries[1].attempts.is_empty(), "{tries:#?}");
}

/// A checker four image px a square, so that a 7×7 window matches equally
/// one period of eight px apart.
fn checker(x: f64, y: f64) -> f64 {
    // 0.2 world is 8 px facing the plane from DISTANCE.
    let period = 0.2;
    let cell = |t: f64| ((t / period).rem_euclid(1.0) < 0.5) as i32;
    if cell(x) ^ cell(y) == 1 {
        190.0
    } else {
        70.0
    }
}

/// On repeating texture, with each view's pose turned so that its content
/// sits 6 to 7 px from where the pose projects the point and its keypoint on
/// the content, every view grows from its keypoint-implied shift with no
/// search and joins on its first growth, written on the content, and does so
/// too where its search would widen at once about the keypoint, which on
/// this checker takes another peak. The same views searched instead, as a
/// view with no keypoint is, land on other peaks of the checker (here up to
/// 11.2 px off).
#[test]
fn the_keypoint_implied_shift_takes_the_right_peak_on_repeating_texture() {
    let centres = vec![
        centre(0.0, 0.0),
        centre(20.0, 0.0),
        centre(25.0, 120.0),
        centre(30.0, 240.0),
    ];
    let mut scene = Scene::looking_at(&centres, WORLD, camera(), |c, d| {
        let (x, y) = hit(c, d, PLANE_Z);
        checker(x, y)
    });
    // Where each photograph shows the point, before the poses are turned.
    let content: Vec<[f64; 2]> = (0..scene.len()).map(|i| scene.project(i, WORLD)).collect();
    // About 6.5 px at a focal length of 160.
    let angle = 6.5f64 / 160.0;
    scene.misplace_camera(1, angle, 0.0);
    scene.misplace_camera(2, -0.6 * angle, 0.8 * angle);
    scene.misplace_camera(3, 0.0, -angle);
    let edited = edited_of(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    for observation in &mut track.observations {
        let p = content[observation.image as usize];
        observation.track = Some(TrackMeasurement {
            keypoint: Some([p[0] as f32, p[1] as f32]),
            ..TrackMeasurement::default()
        });
    }
    for (i, &at) in content.iter().enumerate().skip(1) {
        assert!(distance_px(at, scene.project(i, WORLD)) > 5.0);
    }
    // The reference is the camera whose pose is right.
    let (track, _) = set_reference(&track, row_of(&track, 0)).expect("an in row");

    let written_off = |track: &EditableTrack, options: &GrownPlaneNormalOptions| {
        let (turned, report) = run(&scene, &edited, track, options);
        let sequence = sequence_of(&report).clone();
        let mut worst: f64 = 0.0;
        for (row, view) in sequence.views.iter().enumerate() {
            if let GrownPlaneView::Joined { .. } = view {
                let truth = content[track.observations[row].image as usize];
                let held = keypoint_of(&turned, row).expect("written");
                worst = worst.max(distance_px(held, truth));
            }
        }
        (sequence, worst)
    };

    // A search bound of nothing about the keypoint always widens to the
    // full search about it, which here lands on another peak; the growth
    // with no search comes first and does not use it.
    let widened = GrownPlaneNormalOptions {
        keypoint_search_px: 0.0,
        ..Default::default()
    };
    let mut sequence = None;
    for options in [GrownPlaneNormalOptions::default(), widened] {
        let (run_sequence, worst) = written_off(&track, &options);
        assert_eq!(run_sequence.joined(), 3, "{run_sequence:#?}");
        for entry in &run_sequence.trace {
            assert_eq!(entry.outcome, GrownPlaneTryOutcome::Matched, "{entry:#?}");
            assert_eq!(entry.attempts.len(), 1, "{entry:#?}");
            assert_eq!(entry.attempts[0].start, GrownPlaneStart::NoSearch);
            // Stage 1 reads the window at the implied shift.
            assert!(entry.attempts[0].stage_zncc[0].is_finite(), "{entry:#?}");
        }
        assert!(worst < 0.75, "{worst:.2} px: {run_sequence:#?}");
        sequence.get_or_insert(run_sequence);
    }
    let sequence = sequence.expect("a run");

    // With no keypoint but the reference's, every view searches.
    let mut searched = track.clone();
    let reference = sequence.reference.expect("a reference");
    for (row, observation) in searched.observations.iter_mut().enumerate() {
        if row != reference {
            observation.track = None;
        }
    }
    let (sequence, worst) = written_off(&searched, &GrownPlaneNormalOptions::default());
    for view in &sequence.views {
        if let GrownPlaneView::Joined { start, .. } = view {
            assert_eq!(*start, GrownPlaneStart::NoKeypoint, "{sequence:#?}");
        }
    }
    assert!(worst > 4.0, "{worst:.2} px: {sequence:#?}");
}

/// With each view's pose turned so that its content sits about 16 px from
/// where the pose projects the point, past the widened search's 12 px about
/// zero shift, and each keypoint 6 px off the content, past what the fit
/// recovers and past the 3 px search about the keypoint, every view grown
/// from its keypoint-implied shift with no search is refused; the search
/// then widens about that shift, finds the content, and the view joins and
/// is written on it.
#[test]
fn the_widened_search_is_centred_on_the_keypoint() {
    let centres = vec![
        centre(0.0, 0.0),
        centre(20.0, 0.0),
        centre(25.0, 120.0),
        centre(30.0, 240.0),
    ];
    let mut scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    // Where each photograph shows the point, before the poses are turned.
    let content: Vec<[f64; 2]> = (0..scene.len()).map(|i| scene.project(i, WORLD)).collect();
    // About 16 px at a focal length of 160.
    let angle = 16.0f64 / 160.0;
    scene.misplace_camera(1, angle, 0.0);
    scene.misplace_camera(2, 0.0, angle);
    scene.misplace_camera(3, -angle, 0.0);
    // Each keypoint 6 px off the content along one axis and 2 px along the
    // other.
    let keypoint_offsets = [[0.0, 0.0], [6.0, 2.0], [-2.0, 6.0], [-6.0, -2.0]];
    let edited = edited_of(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    for observation in &mut track.observations {
        let i = observation.image as usize;
        let (p, o) = (content[i], keypoint_offsets[i]);
        observation.track = Some(TrackMeasurement {
            keypoint: Some([(p[0] + o[0]) as f32, (p[1] + o[1]) as f32]),
            ..TrackMeasurement::default()
        });
    }
    // The content's shift from the pose's projection is past the widened
    // search's bound about zero along some axis, so a search centred on zero
    // shift could not reach it.
    for (i, &at) in content.iter().enumerate().skip(1) {
        let projected = scene.project(i, WORLD);
        let off = (at[0] - projected[0])
            .abs()
            .max((at[1] - projected[1]).abs());
        assert!(off > 13.0, "image {i}: {off:.2} px");
    }
    // The reference is the camera whose pose is right.
    let (track, _) = set_reference(&track, row_of(&track, 0)).expect("an in row");

    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);
    assert_eq!(sequence.joined(), 3, "{sequence:#?}");
    // Grown from the keypoint-implied shift, 6 px off, a view is refused,
    // from the combined plane and from the starting plane alike; the search
    // that follows from the starting plane is what joins.
    for (k, entry) in sequence.trace.iter().enumerate() {
        let starts: Vec<_> = entry
            .attempts
            .iter()
            .map(|a| (a.tilt_start, a.start))
            .collect();
        let searched = (GrownPlaneTiltStart::Starting, GrownPlaneStart::Widened);
        let unsearched = (GrownPlaneTiltStart::Starting, GrownPlaneStart::NoSearch);
        let expected = if k == 0 {
            vec![unsearched, searched]
        } else {
            let combined = (GrownPlaneTiltStart::Combined, GrownPlaneStart::NoSearch);
            vec![combined, unsearched, searched]
        };
        assert_eq!(starts, expected, "{entry:#?}");
        let (last, refused) = entry.attempts.split_last().expect("growths");
        assert!(refused.iter().all(|a| a.refusal.is_some()));
        assert!(last.refusal.is_none());
    }
    for (row, view) in sequence.views.iter().enumerate() {
        if let GrownPlaneView::Joined { start, .. } = view {
            assert_eq!(*start, GrownPlaneStart::Widened, "{sequence:#?}");
            let truth = content[track.observations[row].image as usize];
            let held = keypoint_of(&turned, row).expect("written");
            assert!(
                distance_px(held, truth) < 0.75,
                "row {row}: {held:?} vs {truth:?}: {sequence:#?}"
            );
        }
    }
}

/// A flat wall: no view joins, and the normal and every keypoint are left as
/// given, with no precision.
#[test]
fn a_flat_wall_joins_no_view_and_changes_nothing() {
    let centres = wide_centres();
    let scene = Scene::looking_at(&centres, WORLD, camera(), |_, _| 127.5);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::x(), 15.0);
    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);
    assert_eq!(sequence.joined(), 0, "{report}");
    assert!(!sequence.turned());
    assert_eq!(sequence.outcome, GrownPlaneOutcome::NoneJoined);
    assert_eq!((sequence.normal, sequence.turn_deg), (None, None));
    assert!(sequence.precision.is_none());
    assert_eq!(sequence.written, 0);
    assert_eq!(report.tilt.degrees, 0.0);
    let (a, b) = (patch_frame_of(&track), patch_frame_of(&turned));
    assert_eq!(
        (a.u_axis, a.v_axis, a.center),
        (b.u_axis, b.v_axis, b.center)
    );
    for row in 0..track.observations.len() {
        assert_eq!(
            keypoint_of(&turned, row),
            keypoint_of(&track, row),
            "row {row}"
        );
    }
    assert!(
        report.to_string().contains("normal left unchanged"),
        "{report}"
    );
    assert!(!report.to_string().contains("weak evidence"), "{report}");
}

#[test]
fn the_grown_plane_normal_refuses_a_bearing_bad_options_and_stops_when_cancelled() {
    let centres = wide_centres();
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let options = GrownPlaneNormalOptions::default();

    let bearing = as_bearing(&track, &edited);
    assert_eq!(
        grown_plane_normal(
            &bearing,
            &edited,
            &scene.views(),
            &options,
            &Progress::none()
        )
        .err(),
        Some(NormalError::AtInfinity)
    );

    let bad = GrownPlaneNormalOptions {
        spot_search_step_px: 0.0,
        ..Default::default()
    };
    assert_eq!(
        grown_plane_normal(&track, &edited, &scene.views(), &bad, &Progress::none()).err(),
        Some(NormalError::BadGrownPlaneOption("spot_search_step_px"))
    );

    // The weak-turn rule's two settings: a bound of at least zero, infinite
    // allowed, and a finite ratio of at least zero.
    for weak_turn_deg in [f64::NAN, -1.0] {
        let bad = GrownPlaneNormalOptions {
            weak_turn_deg,
            ..Default::default()
        };
        assert_eq!(
            bad.check(),
            Err(NormalError::BadGrownPlaneOption("weak_turn_deg")),
            "{weak_turn_deg}"
        );
    }
    for min_turn_sigma_ratio in [f64::NAN, f64::INFINITY, -0.5] {
        let bad = GrownPlaneNormalOptions {
            min_turn_sigma_ratio,
            ..Default::default()
        };
        assert_eq!(
            bad.check(),
            Err(NormalError::BadGrownPlaneOption("min_turn_sigma_ratio")),
            "{min_turn_sigma_ratio}"
        );
    }
    let edges = GrownPlaneNormalOptions {
        weak_turn_deg: f64::INFINITY,
        min_turn_sigma_ratio: 0.0,
        ..Default::default()
    };
    assert_eq!(edges.check(), Ok(()));
    let zero = GrownPlaneNormalOptions {
        weak_turn_deg: 0.0,
        ..Default::default()
    };
    assert_eq!(zero.check(), Ok(()));

    // The ladder's settings: a first rung of finite angles at least zero with
    // its aim within its bounds, and a finite ratio above 1.
    for (first_rung_min_deg, first_rung_max_deg) in [(f64::NAN, 20.0), (-1.0, 20.0), (15.0, 11.0)] {
        let bad = GrownPlaneNormalOptions {
            first_rung_min_deg,
            first_rung_max_deg,
            ..Default::default()
        };
        assert_eq!(
            bad.check(),
            Err(NormalError::BadGrownPlaneOption("first_rung_min_deg")),
            "{first_rung_min_deg} {first_rung_max_deg}"
        );
    }
    for first_rung_deg in [9.0, 21.0, f64::NAN] {
        let bad = GrownPlaneNormalOptions {
            first_rung_deg,
            ..Default::default()
        };
        assert_eq!(
            bad.check(),
            Err(NormalError::BadGrownPlaneOption("first_rung_deg")),
            "{first_rung_deg}"
        );
    }
    for rung_ratio in [1.0, 0.5, f64::INFINITY, f64::NAN] {
        let bad = GrownPlaneNormalOptions {
            rung_ratio,
            ..Default::default()
        };
        assert_eq!(
            bad.check(),
            Err(NormalError::BadGrownPlaneOption("rung_ratio")),
            "{rung_ratio}"
        );
    }

    let cancel = AtomicBool::new(true);
    let cancelled = Progress::none().cancelled_by(&cancel);
    assert_eq!(
        grown_plane_normal(&track, &edited, &scene.views(), &options, &cancelled).err(),
        Some(NormalError::Cancelled)
    );
    assert!(cancel.load(Ordering::Relaxed));
}

/// The scene of [`a_view_whose_own_plane_disagrees_is_left_out`]: one view at
/// 38 degrees sees the texture on a plane turned 25 degrees from the others'.
fn disagreeing_scene() -> Scene {
    let centres = wide_centres();
    let odd = centres[4];
    let turn = nalgebra::Rotation3::from_axis_angle(&Vector3::x_axis(), 25f64.to_radians());
    let other_normal = turn * truth();
    Scene::looking_at(&centres, WORLD, camera(), move |c, d| {
        if c != odd {
            let (x, y) = hit(c, d, PLANE_Z);
            return plane_texture(x, y);
        }
        let (c, d) = (Vector3::from(c), Vector3::from(d));
        let lambda = (WORLD.coords - c).dot(&other_normal) / d.dot(&other_normal);
        let p = c + d * lambda - WORLD.coords;
        let (u, v) = (turn * Vector3::x(), turn * Vector3::y());
        plane_texture(p.dot(&u), p.dot(&v))
    })
}

/// Detail phases change nothing: the step's track and report are the same
/// with no sink, with a sink at the overview level and with a detailed one.
/// The overview level records the step and one phase per try; the detailed
/// one adds every stage under them, here including the searches of a view
/// whose keypoints are removed, a view left out by the means, the pairs, the
/// picks and the shift-only fits. No joint fit is recorded.
#[test]
fn the_grown_plane_normals_detail_phases_change_nothing() {
    use crate::progress::Event;
    use std::sync::Mutex;

    let scene = disagreeing_scene();
    let edited = edited_of(&scene);
    let mut track = tilted_track(&edited, Vector3::y(), 10.0);
    // One view other than the reference with no keypoint, so that its
    // growth comes from the search.
    let row = row_of(&track, 1);
    track.observations[row].track = None;
    let views = scene.views();

    // Four picks, so that a view is fitted for its shift alone, and
    // the default, under which every view is matched.
    for max_selected_views in [4, 6] {
        let options = GrownPlaneNormalOptions {
            max_selected_views,
            ..Default::default()
        };
        let quiet = grown_plane_normal(&track, &edited, &views, &options, &Progress::none())
            .expect("the step runs");
        let record = |detailed: bool| {
            let seen: Mutex<Vec<(&'static str, u8, Option<String>)>> = Mutex::new(Vec::new());
            let sink = |event: Event<'_>| {
                if let Event::Leave {
                    phase, depth, note, ..
                } = event
                {
                    seen.lock()
                        .unwrap()
                        .push((phase, depth, note.map(str::to_string)));
                }
            };
            let progress = Progress::to(&sink).detailed(detailed);
            let result = grown_plane_normal(&track, &edited, &views, &options, &progress)
                .expect("the step runs");
            (result, seen.into_inner().unwrap())
        };
        let (overview, overview_phases) = record(false);
        let (detailed, detailed_phases) = record(true);
        for other in [&overview, &detailed] {
            assert_eq!(format!("{:?}", quiet.0), format!("{:?}", other.0));
            assert_eq!(format!("{:?}", quiet.1), format!("{:?}", other.1));
        }

        let sequence = sequence_of(&quiet.1);
        let tries = sequence.trace.len();
        let names = |phases: &[(&'static str, u8, Option<String>)], name: &str| {
            phases.iter().filter(|(n, _, _)| *n == name).count()
        };
        // The overview: the step and one phase per try, nothing finer.
        assert_eq!(names(&overview_phases, "grown plane normal"), 1);
        assert_eq!(names(&overview_phases, "grown plane view"), tries);
        for detail in [
            "growth",
            "window stage",
            "judge",
            "mean",
            "pairs",
            "selection",
        ] {
            assert_eq!(names(&overview_phases, detail), 0, "{detail}");
        }

        // The detail: every stage, each noted.
        let growths: usize = sequence.trace.iter().map(|t| t.attempts.len()).sum();
        assert!(names(&detailed_phases, "growth") >= growths);
        let windows = sequence.window_half_widths.len() - 1;
        let shift_only = sequence.shift_only();
        assert_eq!(names(&detailed_phases, "window stage"), growths * windows);
        assert_eq!(names(&detailed_phases, "whole-grid stage"), growths);
        assert_eq!(names(&detailed_phases, "judge"), growths);
        assert_eq!(names(&detailed_phases, "joint fit"), 0);
        assert_eq!(names(&detailed_phases, "precision"), 0);
        assert!(names(&detailed_phases, "mean") >= 1);
        assert_eq!(names(&detailed_phases, "pairs"), 1);
        assert!(!sequence.pairs.is_empty(), "{sequence:#?}");
        assert_eq!(names(&detailed_phases, "pair fit"), sequence.pairs.len());
        assert_eq!(names(&detailed_phases, "selection"), sequence.picks.len());
        assert_eq!(names(&detailed_phases, "shift-only fit"), shift_only);
        let picked = max_selected_views.min(sequence.views.len() - 1);
        assert_eq!(sequence.picks.len(), picked, "{sequence:#?}");
        assert_eq!(
            shift_only,
            sequence.views.len() - 1 - picked,
            "{sequence:#?}"
        );
        assert!(
            names(&detailed_phases, "widened search") >= 1,
            "{:#?}",
            sequence.trace
        );
        for once in ["setup", "turn and write", "grown plane normal"] {
            assert_eq!(names(&detailed_phases, once), 1, "{once}");
        }
        let own = [
            "grown plane view",
            "setup",
            "growth",
            "keypoint search",
            "widened search",
            "window stage",
            "whole-grid stage",
            "judge",
            "mean",
            "pairs",
            "pair fit",
            "selection",
            "shift-only fit",
            "turn and write",
        ];
        for (name, _, note) in &detailed_phases {
            assert!(!own.contains(name) || note.is_some(), "{name} has no note");
        }
        let depth_of = |name: &str| {
            detailed_phases
                .iter()
                .find(|(n, _, _)| *n == name)
                .map(|(_, d, _)| *d)
                .expect("recorded")
        };
        let step = depth_of("grown plane normal");
        assert_eq!(depth_of("grown plane view"), step + 1);
        assert_eq!(depth_of("growth"), depth_of("grown plane view") + 1);
        assert_eq!(depth_of("window stage"), depth_of("growth") + 1);
        // A search under a growth where a growth searched; under the
        // selection, also under a shift-only fit.
        let searched = sequence
            .trace
            .iter()
            .flat_map(|t| &t.attempts)
            .any(|a| a.start != GrownPlaneStart::NoSearch);
        assert!(
            !searched
                || detailed_phases
                    .iter()
                    .any(|(n, d, _)| *n == "widened search" && *d == depth_of("growth") + 1)
        );
        assert_eq!(depth_of("mean"), step + 1);
        assert_eq!(depth_of("selection"), step + 1);
        assert_eq!(depth_of("pair fit"), depth_of("pairs") + 1);
        if max_selected_views == 3 {
            assert_eq!(depth_of("shift-only fit"), step + 1);
        } else {
            // Every view is matched, and the means leave the disagreeing
            // one out.
            let refused = detailed_phases
                .iter()
                .filter(|(n, _, note)| {
                    let note = note.as_deref().unwrap_or("");
                    let words: Vec<&str> = note.split_whitespace().collect();
                    *n == "mean" && words.len() > 2 && words[0] != words[2]
                })
                .count();
            assert!(refused >= 1, "{detailed_phases:#?}");
        }
    }
}

/// The information-weighted plane of the views on opposite sides of a
/// reference that faces the plane: each view is 20 degrees from it, and the
/// two pairs, the views opposite each other, are 40 degrees apart. The
/// selection matches all four, the pairs are matched widest first and both
/// are kept; with them the weakest `σ` is under that of the reference matches
/// alone (with `max_pairs` 0), and still is with each pair's information
/// halved, and the plane stays on the truth.
#[test]
fn wide_pairs_sharpen_the_plane_over_reference_matches_alone() {
    let centres = vec![
        centre(0.0, 0.0),
        centre(20.0, 0.0),
        centre(20.0, 180.0),
        centre(20.0, 90.0),
        centre(20.0, 270.0),
    ];
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::new(1.0, 0.3, 0.0), 15.0);
    let (track, _) = set_reference(&track, row_of(&track, 0)).expect("an in row");

    let without = GrownPlaneNormalOptions {
        max_pairs: 0,
        ..Default::default()
    };
    let (alone, alone_report) = run(&scene, &edited, &track, &without);
    let (paired, paired_report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let (alone_seq, paired_seq) = (sequence_of(&alone_report), sequence_of(&paired_report));
    assert_eq!(alone_seq.joined(), 4, "{alone_seq:#?}");
    assert_eq!(paired_seq.joined(), 4, "{paired_seq:#?}");
    assert!(alone_seq.pairs.is_empty());
    assert_eq!(paired_seq.pairs.len(), 2, "{paired_seq:#?}");
    assert_eq!(paired_seq.pairs_kept(), 2, "{paired_seq:#?}");
    for pair in &paired_seq.pairs {
        assert!((pair.angle_deg - 40.0).abs() < 0.5, "{pair:?}");
        let images: Vec<u32> = pair
            .observations
            .iter()
            .map(|&r| track.observations[r].image)
            .collect();
        assert!(
            images == [1, 2] || images == [2, 1] || images == [3, 4] || images == [4, 3],
            "opposite views: {images:?}"
        );
    }
    let a = alone_seq.precision.expect("views joined");
    let p = paired_seq.precision.expect("views joined");
    assert!(
        p.sigma_weak_deg < a.sigma_weak_deg,
        "{paired_report} vs {alone_report}"
    );
    assert!(
        p.discounted_sigma_weak_deg < a.sigma_weak_deg,
        "{paired_report} vs {alone_report}"
    );
    assert!(p.discounted_sigma_weak_deg > p.sigma_weak_deg);
    for turned in [&alone, &paired] {
        let after = degrees(patch_frame_of(turned).normal(), truth());
        assert!(after < 1.0, "{after:.3}: {paired_report}");
    }
}

/// Sixteen views around the point, in rings at 10 to 40 degrees: the
/// selection matches a view of its first rung and then, from the combined
/// plane, the views of the widest ring that most raise the weakest predicted
/// information, and stops once the combined
/// plane's `σ` reaches the target, having matched far fewer views than there
/// are. Every other view is fitted for its shift alone at the final plane and
/// written at the point's projection.
#[test]
fn the_selection_reaches_the_target_sigma_matching_few_views() {
    let mut centres = vec![centre(0.0, 0.0)];
    for (ring, from) in [10.0, 20.0, 30.0, 40.0].into_iter().enumerate() {
        for k in 0..4 {
            centres.push(centre(from, 90.0 * k as f64 + 22.5 * ring as f64));
        }
    }
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::new(0.4, 1.0, 0.0), 15.0);
    let (track, _) = set_reference(&track, row_of(&track, 0)).expect("an in row");
    let target = 1.0;
    let options = GrownPlaneNormalOptions {
        target_sigma_deg: target,
        max_selected_views: 16,
        ..Default::default()
    };
    let (turned, report) = run(&scene, &edited, &track, &options);
    let sequence = sequence_of(&report);
    let others = centres.len() - 1;
    assert!(
        sequence.matched() <= 3,
        "{} of {others}: {report}: {:#?}",
        sequence.matched(),
        sequence.picks
    );
    assert_eq!(sequence.picks.len(), sequence.matched());
    let last = sequence.picks.last().expect("a pick");
    assert!(last.sigma_after_deg <= target, "{:#?}", sequence.picks);
    // The first pick is from the first rung, 10 to 20 degrees, and the
    // later ones from the widest ring, the last rung, grown from the
    // combined plane.
    let first = sequence.picks[0].angle_deg;
    assert!((10.0..=20.0).contains(&first), "{:#?}", sequence.picks);
    for pick in &sequence.picks[1..] {
        assert!((pick.angle_deg - 40.0).abs() < 0.5, "{:#?}", sequence.picks);
    }
    for entry in &sequence.trace[1..sequence.picks.len()] {
        assert_eq!(entry.attempts[0].tilt_start, GrownPlaneTiltStart::Combined);
    }
    // The two wide picks are not along one offset.
    let d0 = sequence.picks[1].offset_direction;
    let d1 = sequence.picks[2].offset_direction;
    assert!(
        (d0[0] * d1[0] + d0[1] * d1[1]).abs() < 0.9,
        "{:#?}",
        sequence.picks
    );
    let precision = sequence.precision.expect("views joined");
    assert!(precision.discounted_sigma_weak_deg <= target, "{report}");
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3} deg off: {report}");
    assert_eq!(
        sequence.shift_only(),
        others - sequence.matched(),
        "{report}"
    );
    for (row, view) in sequence.views.iter().enumerate() {
        if let GrownPlaneView::ShiftOnly { keypoint, .. } = view {
            let written = keypoint.expect("passed the floor, on the photograph");
            let truth = scene.project(track.observations[row].image as usize, WORLD);
            assert!(
                distance_px(written, truth) < 0.5,
                "row {row}: {written:?} vs {truth:?}"
            );
        }
    }
    assert_eq!(sequence.written, centres.len(), "{report}");
}

/// The selection's settings, with its first rung at 12 degrees and a target
/// `σ` and a budget the scenes below never reach, so that it runs until no
/// view is left.
fn ladder() -> GrownPlaneNormalOptions {
    GrownPlaneNormalOptions {
        first_rung_deg: 12.0,
        target_sigma_deg: 1e-3,
        max_selected_views: 16,
        ..Default::default()
    }
}

/// A view at each `(angle, azimuth)` of `views`, degrees, about a reference
/// facing the plane, seeing the plane's texture, except the views at the
/// angles in `unrelated`, which see unrelated texture that no plane
/// matches.
fn ladder_scene(views: &[(f64, f64)], unrelated: &[f64]) -> Scene {
    let mut centres = vec![centre(0.0, 0.0)];
    centres.extend(views.iter().map(|&(a, z)| centre(a, z)));
    let odd: Vec<[f64; 3]> = views
        .iter()
        .zip(&centres[1..])
        .filter(|((a, _), _)| unrelated.contains(a))
        .map(|(_, &c)| c)
        .collect();
    Scene::looking_at(&centres, WORLD, camera(), move |c, d| {
        let (x, y) = hit(c, d, PLANE_Z);
        if odd.contains(&c) {
            noise(x, y, 0.06, 60.0, 7)
        } else {
            plane_texture(x, y)
        }
    })
}

/// A view at each of `angles` degrees, each at its own azimuth.
fn spread(angles: &[f64]) -> Vec<(f64, f64)> {
    angles
        .iter()
        .enumerate()
        .map(|(k, &a)| (a, 30.0 + 100.0 * k as f64))
        .collect()
}

/// The ladder on `scene`, from 20 degrees off, with view 0 the reference.
fn run_ladder(scene: &Scene) -> (EditableTrack, EditableTrack, NormalReport) {
    let edited = edited_of(scene);
    let track = tilted_track(&edited, Vector3::new(1.0, 0.4, 0.0), 20.0);
    let (track, _) = set_reference(&track, row_of(&track, 0)).expect("an in row");
    let (turned, report) = run(scene, &edited, &track, &ladder());
    (track, turned, report)
}

/// Views at 5 to 50 degrees: the ladder picks the view nearest 12 degrees
/// first, grown from the starting plane, then one view a rung, each about
/// 1.6 times wider than the widest before it and grown from the combined
/// plane, out to the widest view, and last the view under its first rung.
#[test]
fn the_ladder_visits_views_near_to_far() {
    let scene = ladder_scene(&spread(&[5.0, 12.0, 19.0, 31.0, 50.0]), &[]);
    let (_, turned, report) = run_ladder(&scene);
    let sequence = sequence_of(&report);
    let angles: Vec<f64> = sequence.picks.iter().map(|p| p.angle_deg).collect();
    let expected = [12.0, 19.0, 31.0, 50.0, 5.0];
    assert_eq!(angles.len(), expected.len(), "{:#?}", sequence.picks);
    for (a, e) in angles.iter().zip(expected) {
        assert!((a - e).abs() < 0.5, "{angles:?}");
    }
    let rungs: Vec<u32> = sequence.picks.iter().map(|p| p.rung).collect();
    assert_eq!(rungs, [0, 1, 2, 3, 4]);
    assert!(sequence.picks.iter().all(|p| p.matched), "{report}");
    // The first pick grows from the starting plane, every later one from
    // the combined plane, and passes there.
    let first: Vec<GrownPlaneTiltStart> = sequence
        .trace
        .iter()
        .take(sequence.picks.len())
        .map(|t| t.attempts[0].tilt_start)
        .collect();
    assert_eq!(first[0], GrownPlaneTiltStart::Starting);
    assert!(
        first[1..]
            .iter()
            .all(|&s| s == GrownPlaneTiltStart::Combined),
        "{first:?}"
    );
    for entry in sequence.trace.iter().take(sequence.picks.len()) {
        assert_eq!(entry.attempts.len(), 1, "{entry:#?}");
    }
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3}: {report}");
}

/// A view in the second rung that sees unrelated texture is refused from
/// the combined plane and grown again from the starting plane, where it is
/// refused too; the rung waits, and its other view is picked next.
#[test]
fn a_refused_rung_is_retried_from_the_start() {
    // The unrelated view is across the first pick's offset and the wider of
    // the second rung's two, so the rung's prediction picks it first.
    let views = [(12.0, 0.0), (17.0, 20.0), (22.0, 90.0), (35.0, 200.0)];
    let scene = ladder_scene(&views, &[22.0]);
    let (_, turned, report) = run_ladder(&scene);
    let sequence = sequence_of(&report);
    let angles: Vec<f64> = sequence.picks.iter().map(|p| p.angle_deg).collect();
    let bad = sequence
        .picks
        .iter()
        .position(|p| (p.angle_deg - 22.0).abs() < 0.5)
        .expect("the unrelated view is picked");
    let pick = &sequence.picks[bad];
    assert!(!pick.matched, "{:#?}", sequence.picks);
    assert_eq!(pick.rung, 1, "{:#?}", sequence.picks);
    let entry = &sequence.trace[bad];
    assert_eq!(entry.outcome, GrownPlaneTryOutcome::LeftOut);
    assert_eq!(entry.attempts[0].tilt_start, GrownPlaneTiltStart::Combined);
    assert!(entry.attempts.len() >= 2, "{entry:#?}");
    assert!(
        entry.attempts[1..]
            .iter()
            .all(|a| a.tilt_start == GrownPlaneTiltStart::Starting),
        "{entry:#?}"
    );
    // The rung waits for its other view before the ladder moves out.
    let next = &sequence.picks[bad + 1];
    assert_eq!(next.rung, 1, "{angles:?}");
    assert!((next.angle_deg - 17.0).abs() < 0.5, "{angles:?}");
    assert!(next.matched);
    assert!(
        sequence.picks[bad + 2..].iter().all(|p| p.rung > 1),
        "{:#?}",
        sequence.picks
    );
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3}: {report}");
}

/// The widest view sees unrelated texture: the ladder's last rung tries it
/// and leaves it out, and ends at the widest view that passes, which joins.
#[test]
fn the_ladder_ends_at_the_widest_view_that_passes() {
    let scene = ladder_scene(&spread(&[12.0, 20.0, 32.0, 48.0, 56.0]), &[56.0]);
    let (track, turned, report) = run_ladder(&scene);
    let sequence = sequence_of(&report);
    let widest = sequence
        .picks
        .iter()
        .position(|p| (p.angle_deg - 56.0).abs() < 0.5)
        .expect("the widest view is picked");
    assert!(!sequence.picks[widest].matched, "{:#?}", sequence.picks);
    let joined: Vec<f64> = sequence
        .picks
        .iter()
        .filter(|p| p.matched)
        .map(|p| p.angle_deg)
        .collect();
    let widest_joined = joined.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!((widest_joined - 48.0).abs() < 0.5, "{joined:?}");
    // The two widest views are the last rung's, picked after every
    // narrower one.
    let last = sequence.picks[widest].rung;
    let at48 = sequence
        .picks
        .iter()
        .find(|p| (p.angle_deg - 48.0).abs() < 0.5)
        .expect("picked");
    assert_eq!(at48.rung, last, "{:#?}", sequence.picks);
    for p in &sequence.picks {
        if p.angle_deg < 40.0 {
            assert!(p.rung < last, "{:#?}", sequence.picks);
        }
    }
    let row = row_of(&track, 4);
    assert!(
        matches!(sequence.views[row], GrownPlaneView::Joined { .. }),
        "{:?}",
        sequence.views[row]
    );
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3}: {report}");
}

/// A view at the reference's own centre, 0 degrees from it, is the first
/// rung's pick when the only other view is far wider; once it is kept the
/// ladder's aim is zero, and the selection goes on to the wider view rather
/// than aiming at zero again.
#[test]
fn a_view_at_the_references_centre_ends_the_ladder() {
    let scene = ladder_scene(&[(0.0, 0.0), (35.0, 30.0)], &[]);
    let (_, _, report) = run_ladder(&scene);
    let sequence = sequence_of(&report);
    let angles: Vec<f64> = sequence.picks.iter().map(|p| p.angle_deg).collect();
    assert_eq!(angles.len(), 2, "{:#?}", sequence.picks);
    assert!(angles[0].abs() < 0.5, "{angles:?}");
    assert!((angles[1] - 35.0).abs() < 0.5, "{angles:?}");
}
