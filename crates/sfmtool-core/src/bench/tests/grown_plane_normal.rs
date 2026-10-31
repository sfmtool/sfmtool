// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The grown plane normal, decided against synthetic planes seen by pinholes
//! that look at the point from around it: a textured plane seen out to 50
//! degrees, one view whose top cells see something else, views all within a
//! few degrees, a view that sees another plane, repeating texture seen from
//! poses that are off with keypoints on the content, views with no keypoint,
//! a view whose keypoint is off by more than its growth recovers, views on
//! opposite sides of the reference whose pairs sharpen the plane, a selection
//! that matches a few of many views, a flat wall, and large turns on weak and
//! on strong evidence, and the ladder's selection near to far: its order, a
//! refused pick on the second rung grown again from the starting plane while
//! its rung waits for its other view, a refused pick past it grown once, its
//! end at the widest view that passes, and a view at the reference's centre;
//! a growth stopped early, keypoints off the content in views fitted for
//! their shift alone, and a shift-only fit from a keypoint too far off.

use std::sync::atomic::{AtomicBool, Ordering};

use super::scene::{fixture_of, Scene, PLANE_Z, WORLD};
use super::*;
use crate::bench::{
    GrownPlaneNormalOptions, GrownPlaneOutcome, GrownPlaneRefusal, GrownPlaneSequence,
    GrownPlaneTiltStart, GrownPlaneTryOutcome, GrownPlaneView,
};
use crate::camera::CameraModel;

/// A half-length that projects to 16 px in a view facing the plane from the
/// scenes' distance, as in the grid plane normal's tests.
pub(super) const HALF: f64 = 0.4;

/// Every camera's distance from the point.
const DISTANCE: f64 = 4.0;

pub(super) fn camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 160.0,
            focal_length_y: 160.0,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
        },
        width: 128,
        height: 128,
        rolling_shutter: None,
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
pub(super) fn textured_plane(c: [f64; 3], d: [f64; 3]) -> f64 {
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
pub(super) fn truth() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -1.0)
}

pub(super) fn degrees(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
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

/// Every row of `after` holds the keypoint `before` holds, bit for bit.
fn assert_keypoints_kept(before: &EditableTrack, after: &EditableTrack) {
    assert_eq!(before.observations.len(), after.observations.len());
    for (row, (a, b)) in before
        .observations
        .iter()
        .zip(&after.observations)
        .enumerate()
    {
        let keypoint = |o: &crate::bench::Observation| o.track.as_ref().and_then(|m| m.keypoint);
        assert_eq!(keypoint(a), keypoint(b), "row {row}");
    }
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

/// The defaults with a target `σ` of 0.5 degrees, which the scenes of up to
/// six views below never reach, so that the selection matches every view.
fn every_view() -> GrownPlaneNormalOptions {
    GrownPlaneNormalOptions {
        target_sigma_deg: 0.5,
        ..Default::default()
    }
}

/// Six views of the textured plane, from facing it to 50 degrees off.
pub(super) fn wide_centres() -> Vec<[f64; 3]> {
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
/// views; every one joins on its one growth with a shift under half a pixel,
/// every keypoint is kept as given, and each joined view's score at the final
/// plane is its held score. The two
/// pairs are kept, and halving their information raises `σ`.
#[test]
fn a_textured_plane_started_far_off_is_turned_onto_it() {
    let centres = wide_centres();
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::new(1.0, 0.5, 0.0), 25.0);
    let (turned, report) = run(&scene, &edited, &track, &every_view());
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
    }
    let precision = sequence.precision.expect("views joined");
    assert!(precision.sigma_weak_deg >= precision.sigma_strong_deg);
    assert!(precision.discounted_sigma_weak_deg > precision.sigma_weak_deg);
    assert!(!precision.low_confidence, "{report}");
    // The centre does not move, and the track is read back.
    assert!((patch_frame_of(&turned).center - WORLD).norm() < 1e-9);
    assert!(turned.track().expect("track stage").bitmap.is_some());

    // The patch turned 25 degrees and no keypoint moved, the reference's
    // included.
    assert!(report.tilt.degrees > 20.0, "{report}");
    assert_keypoints_kept(&track, &turned);
    for (row, view) in sequence.views.iter().enumerate() {
        if let GrownPlaneView::Joined {
            shift,
            grid_zncc,
            held_zncc,
            ..
        } = view
        {
            // The score at the final plane is the last held score.
            assert_eq!(Some(*grid_zncc), *held_zncc, "row {row}");
            // Each keypoint is at the point's projection, so the fit's shift
            // is near zero.
            assert!(distance_px(*shift, [0.0; 2]) < 0.5, "row {row}: {shift:?}");
        }
    }
    assert_eq!(sequence.pairs.len(), 2, "{sequence:#?}");
    assert_eq!(sequence.pairs_kept(), 2, "{sequence:#?}");
    let sentence = report.to_string();
    assert!(
        sentence.starts_with("5 of 5 views joined, 2 of 2 pairs kept; "),
        "{sentence}"
    );
    assert!(!sentence.contains("written"), "{sentence}");
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
    let (turned, report) = run(&scene, &edited, &track, &every_view());
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

/// The narrow views' turn of about 20 degrees on a `σ` of about 13, with the
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
    assert_keypoints_kept(&track, &kept);
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
    // The agreement is the joined views' at the plane found, not turned to.
    let agreement = sequence.agreement();
    assert!(agreement > 0.5 && agreement <= 1.0, "{report}");
    assert!(!sequence.picks.is_empty());
    assert_eq!(
        sequence.views[sequence.reference.expect("a reference")],
        GrownPlaneView::Reference
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
    assert_keypoints_kept(&track, &turned);
}

/// A view that sees the texture on a plane turned 25 degrees from the one
/// the others see is left out, and the plane is the others'. The selection
/// matches every view; this one passes the floor on its own, the combination
/// leaves it out as too far, and it is not grown again.
#[test]
fn a_view_whose_own_plane_disagrees_is_left_out() {
    let centres = wide_centres();
    let scene = disagreeing_scene();
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::y(), 10.0);
    let row = row_of(&track, 4);
    let (turned, report) = run(&scene, &edited, &track, &every_view());
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
    assert_eq!(tries.len(), 1, "{tries:#?}");
    assert_eq!(tries[0].outcome, GrownPlaneTryOutcome::Matched);
    assert_eq!(tries[0].attempts.len(), 1);
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
/// the content, every view grows from its keypoint-implied shift and joins on
/// its first growth, with a shift near zero that keeps it on the content, and
/// keeps its keypoint. With the views' keypoints
/// removed, no view has a shift to grow from: each is left out as having no
/// keypoint, never tried, and the normal is left as given.
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

    let (turned, report) = run(&scene, &edited, &track, &every_view());
    let sequence = sequence_of(&report).clone();
    assert_keypoints_kept(&track, &turned);
    let mut worst: f64 = 0.0;
    for view in &sequence.views {
        if let GrownPlaneView::Joined { shift, .. } = view {
            worst = worst.max(distance_px(*shift, [0.0; 2]));
        }
    }
    assert_eq!(sequence.joined(), 3, "{sequence:#?}");
    for entry in &sequence.trace {
        assert_eq!(entry.outcome, GrownPlaneTryOutcome::Matched, "{entry:#?}");
        assert_eq!(entry.attempts.len(), 1, "{entry:#?}");
        // Stage 1 reads the window at the implied shift.
        assert!(entry.attempts[0].stage_zncc[0].is_finite(), "{entry:#?}");
    }
    assert!(worst < 0.75, "{worst:.2} px: {sequence:#?}");

    // With no keypoint but the reference's, no view is grown.
    let mut bare = track.clone();
    let reference = sequence.reference.expect("a reference");
    for (row, observation) in bare.observations.iter_mut().enumerate() {
        if row != reference {
            observation.track = None;
        }
    }
    let (turned, report) = run(&scene, &edited, &bare, &every_view());
    let sequence = sequence_of(&report);
    assert_eq!(sequence.reference, Some(reference));
    assert!(sequence.trace.is_empty(), "{sequence:#?}");
    assert!(sequence.picks.is_empty(), "{sequence:#?}");
    assert_eq!(sequence.outcome, GrownPlaneOutcome::NoneJoined);
    assert_eq!(sequence.left_out(), 3, "{sequence:#?}");
    for (row, view) in sequence.views.iter().enumerate() {
        if row != reference {
            assert_eq!(
                *view,
                GrownPlaneView::LeftOut {
                    reason: GrownPlaneRefusal::NoKeypoint,
                    distance: None,
                    held_zncc: None,
                },
                "row {row}"
            );
        }
    }
    assert_eq!(
        patch_frame_of(&turned).normal(),
        patch_frame_of(&bare).normal()
    );
    assert!(report
        .to_string()
        .contains("0 of 3 views joined, 3 left out"));
}

/// Three views at 20 to 30 degrees, two with their keypoints on the point's
/// projection and the third with its keypoint 6 px off along one axis and
/// 2 px along the other, past what the growing windows recover. The step
/// trusts the keypoint: the third view is grown from the shift it implies,
/// from the combined plane and, on the second rung, again from the starting
/// plane at the same shift, both refused, and it is left out with no other
/// growth. No phase searches for a shift, and the plane is the other two
/// views'.
#[test]
fn a_view_whose_keypoint_is_off_past_the_growth_is_left_out() {
    use crate::progress::Event;
    use std::sync::Mutex;

    let centres = vec![
        centre(0.0, 0.0),
        centre(15.0, 0.0),
        centre(25.0, 120.0),
        centre(30.0, 240.0),
    ];
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::new(1.0, 0.4, 0.0), 15.0);
    let (mut track, _) = set_reference(&track, row_of(&track, 0)).expect("an in row");
    let off = row_of(&track, 3);
    let p = scene.project(3, WORLD);
    track.observations[off].track = Some(TrackMeasurement {
        keypoint: Some([(p[0] - 6.0) as f32, (p[1] - 2.0) as f32]),
        ..TrackMeasurement::default()
    });

    // A target under what two views reach, so that all three are picked.
    let options = GrownPlaneNormalOptions {
        target_sigma_deg: 0.5,
        ..Default::default()
    };
    let seen: Mutex<Vec<(&'static str, u8)>> = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Leave { phase, depth, .. } = event {
            seen.lock().unwrap().push((phase, depth));
        }
    };
    let progress = Progress::to(&sink).detailed(true);
    let (turned, report) = grown_plane_normal(&track, &edited, &scene.views(), &options, &progress)
        .expect("the step runs");
    let phases = seen.into_inner().unwrap();
    let sequence = sequence_of(&report);

    assert_eq!(sequence.picks.len(), 3, "{:#?}", sequence.picks);
    assert_eq!(sequence.joined(), 2, "{sequence:#?}");
    let entry = sequence
        .trace
        .iter()
        .find(|t| t.observation == off)
        .expect("the view is tried");
    assert_eq!(entry.outcome, GrownPlaneTryOutcome::LeftOut, "{entry:#?}");
    let pick = sequence
        .picks
        .iter()
        .find(|p| p.observation == off)
        .expect("picked");
    let starts: Vec<_> = entry.attempts.iter().map(|a| a.tilt_start).collect();
    // On the second rung, so a refusal from the combined plane is grown
    // once more from the starting plane.
    assert_eq!(pick.rung, 1, "{:#?}", sequence.picks);
    assert_eq!(
        starts,
        [GrownPlaneTiltStart::Combined, GrownPlaneTiltStart::Starting],
        "{entry:#?}"
    );
    assert!(
        entry.attempts.iter().all(|a| a.refusal.is_some()),
        "{entry:#?}"
    );
    assert!(
        matches!(sequence.views[off], GrownPlaneView::LeftOut { .. }),
        "{:?}",
        sequence.views[off]
    );
    // One growth phase and one judge per growth recorded, and no phase
    // searches.
    let growths: usize = sequence.trace.iter().map(|t| t.attempts.len()).sum();
    let count = |name: &str| phases.iter().filter(|(n, _)| *n == name).count();
    assert_eq!(count("growth"), growths);
    assert_eq!(count("judge"), growths);
    for (name, _) in &phases {
        assert!(!name.contains("search"), "{name}");
    }
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3} deg off: {report}");
}

/// A view the selection does not pick, holding a keypoint 6 px off the
/// content along one axis and 2 px along the other, has its shift alone
/// fitted from that keypoint, through the patch slid onto it, and from
/// nowhere else: the fit stays near the keypoint, under the floor, and the
/// view keeps its keypoint, as every view does.
#[test]
fn a_shift_only_fit_keeps_the_keypoint_implied_shift() {
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
    let options = GrownPlaneNormalOptions {
        target_sigma_deg: 2.0,
        max_selected_views: 16,
        ..Default::default()
    };
    // Which views the selection leaves for the shift-only fit: a view not
    // picked does not change the picks.
    let (_, before) = run(&scene, &edited, &track, &options);
    let picked = |report: &NormalReport| -> Vec<usize> {
        sequence_of(report)
            .picks
            .iter()
            .map(|p| p.observation)
            .collect()
    };
    let off = sequence_of(&before)
        .views
        .iter()
        .position(|v| matches!(v, GrownPlaneView::ShiftOnly { .. }))
        .expect("a view fitted for its shift alone");

    let mut moved = track.clone();
    let image = moved.observations[off].image as usize;
    let p = scene.project(image, WORLD);
    let implied = [-6.0, -2.0];
    moved.observations[off].track = Some(TrackMeasurement {
        keypoint: Some([(p[0] + implied[0]) as f32, (p[1] + implied[1]) as f32]),
        ..TrackMeasurement::default()
    });
    let (turned, report) = run(&scene, &edited, &moved, &options);
    let sequence = sequence_of(&report);
    assert!(sequence.turned(), "{report}");
    assert_eq!(picked(&report), picked(&before));
    match &sequence.views[off] {
        GrownPlaneView::ShiftOnly { shift, grid_zncc } => {
            assert!(*grid_zncc < options.min_held_zncc, "{grid_zncc}");
            // The shift is read from the view's keypoint, where the patch
            // slid onto it is centred, so the content sits at minus the
            // keypoint's offset. The fit stays nearer the keypoint.
            let from_keypoint = distance_px(*shift, [0.0; 2]);
            let from_content = distance_px(*shift, [-implied[0], -implied[1]]);
            assert!(
                from_keypoint < from_content,
                "{shift:?}: {from_keypoint:.2} px from the keypoint"
            );
        }
        other => panic!("shift only, not {other:?}"),
    }
    assert_keypoints_kept(&moved, &turned);
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
    assert_eq!(report.tilt.degrees, 0.0);
    let (a, b) = (patch_frame_of(&track), patch_frame_of(&turned));
    assert_eq!(
        (a.u_axis, a.v_axis, a.center),
        (b.u_axis, b.v_axis, b.center)
    );
    assert_keypoints_kept(&track, &turned);
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
        zncc_tolerance: 0.0,
        ..Default::default()
    };
    assert_eq!(
        grown_plane_normal(&track, &edited, &scene.views(), &bad, &Progress::none()).err(),
        Some(NormalError::BadGrownPlaneOption("zncc_tolerance"))
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
/// one adds every stage under them, here including a view left out by the
/// means, the pairs, the picks and the shift-only fits. A view whose keypoint
/// is removed is left out with no try. No joint fit is recorded.
#[test]
fn the_grown_plane_normals_detail_phases_change_nothing() {
    use crate::progress::Event;
    use std::sync::Mutex;

    let scene = disagreeing_scene();
    let edited = edited_of(&scene);
    let mut track = tilted_track(&edited, Vector3::y(), 10.0);
    // One view other than the reference with no keypoint, which is left out
    // with no try.
    let row = row_of(&track, 1);
    track.observations[row].track = None;
    let views = scene.views();

    // Three picks, so that a view is fitted for its shift alone, and
    // the default six, under which every view with a keypoint is matched.
    for max_selected_views in [3, 6] {
        let options = GrownPlaneNormalOptions {
            max_selected_views,
            ..every_view()
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
        assert_eq!(
            sequence.views[row],
            GrownPlaneView::LeftOut {
                reason: GrownPlaneRefusal::NoKeypoint,
                distance: None,
                held_zncc: None,
            }
        );
        assert!(sequence.trace.iter().all(|t| t.observation != row));
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
        assert_eq!(names(&detailed_phases, "growth"), growths);
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
        // The reference and the view with no keypoint are not candidates.
        let candidates = sequence.views.len() - 2;
        let picked = max_selected_views.min(candidates);
        assert_eq!(sequence.picks.len(), picked, "{sequence:#?}");
        assert_eq!(shift_only, candidates - picked, "{sequence:#?}");
        for once in ["setup", "turn", "grown plane normal"] {
            assert_eq!(names(&detailed_phases, once), 1, "{once}");
        }
        let own = [
            "grown plane view",
            "setup",
            "growth",
            "window stage",
            "whole-grid stage",
            "judge",
            "mean",
            "pairs",
            "pair fit",
            "selection",
            "shift-only fit",
            "turn",
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
/// are. Every other view is fitted for its shift alone at the final plane,
/// with a shift near zero, and keeps its keypoint.
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
    let target = 2.0;
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
        if let GrownPlaneView::ShiftOnly { shift, grid_zncc } = view {
            assert!(*grid_zncc >= options.min_held_zncc, "row {row}: {view:?}");
            assert!(distance_px(*shift, [0.0; 2]) < 0.5, "row {row}: {shift:?}");
        }
    }
    assert_keypoints_kept(&track, &turned);
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
/// the combined plane, its growth stopped at a window stage, and grown once
/// again from the starting plane, at the same keypoint-implied shift, where
/// it is refused too; the rung waits, and its other view is picked next.
#[test]
fn a_refused_second_rung_pick_is_retried_from_the_start() {
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
    let [first, retry] = entry.attempts.as_slice() else {
        panic!("two growths: {entry:#?}");
    };
    assert_eq!(first.tilt_start, GrownPlaneTiltStart::Combined);
    assert_eq!(
        first.refusal,
        Some(GrownPlaneRefusal::StoppedEarly),
        "{entry:#?}"
    );
    assert_eq!(retry.tilt_start, GrownPlaneTiltStart::Starting);
    assert!(retry.refusal.is_some(), "{entry:#?}");
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

/// A view in the third rung that sees unrelated texture is refused from the
/// combined plane and is not grown again from the starting plane: past the
/// second rung a refused pick is left out after its one growth.
#[test]
fn a_refused_pick_past_the_second_rung_is_not_retried() {
    let scene = ladder_scene(&spread(&[12.0, 20.0, 32.0]), &[32.0]);
    let (track, turned, report) = run_ladder(&scene);
    let sequence = sequence_of(&report);
    let rungs: Vec<u32> = sequence.picks.iter().map(|p| p.rung).collect();
    assert_eq!(rungs, [0, 1, 2], "{:#?}", sequence.picks);
    assert_eq!(
        sequence.picks.iter().map(|p| p.matched).collect::<Vec<_>>(),
        [true, true, false],
        "{:#?}",
        sequence.picks
    );
    let row = row_of(&track, 3);
    let entry = sequence
        .trace
        .iter()
        .find(|t| t.observation == row)
        .expect("the unrelated view is tried");
    assert_eq!(entry.outcome, GrownPlaneTryOutcome::LeftOut, "{entry:#?}");
    assert_eq!(entry.attempts.len(), 1, "{entry:#?}");
    assert_eq!(entry.attempts[0].tilt_start, GrownPlaneTiltStart::Combined);
    assert!(entry.attempts[0].refusal.is_some(), "{entry:#?}");
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

/// The first rung's view sees unrelated texture: each of its growths stops
/// at a window stage, refused as stopped early, with no whole-grid score or
/// `σ`, and the view is left out for it. The views that pass are grown
/// through every stage, and the patch is still turned onto the plane.
#[test]
fn a_growth_that_cannot_reach_the_floor_stops_early() {
    let scene = ladder_scene(&spread(&[12.0, 20.0, 32.0]), &[12.0]);
    let (track, turned, report) = run_ladder(&scene);
    let sequence = sequence_of(&report);
    let stages = sequence.window_half_widths.len() + 1;
    let row = row_of(&track, 1);
    let entry = sequence
        .trace
        .iter()
        .find(|t| t.observation == row)
        .expect("the unrelated view is tried");
    assert_eq!(entry.outcome, GrownPlaneTryOutcome::LeftOut, "{entry:#?}");
    for attempt in &entry.attempts {
        assert_eq!(
            attempt.refusal,
            Some(GrownPlaneRefusal::StoppedEarly),
            "{entry:#?}"
        );
        let n = attempt.stage_zncc.len();
        assert!((3..stages).contains(&n), "{entry:#?}");
        assert!(attempt.stage_zncc[n - 1] < 0.25, "{entry:#?}");
        assert!(attempt.grid_zncc.is_nan() && attempt.sigma_deg.is_nan());
    }
    assert!(
        matches!(
            sequence.views[row],
            GrownPlaneView::LeftOut {
                reason: GrownPlaneRefusal::StoppedEarly,
                ..
            }
        ),
        "{:?}",
        sequence.views[row]
    );
    for t in sequence.trace.iter().filter(|t| t.observation != row) {
        let attempt = t.attempts.last().expect("a growth");
        assert!(attempt.refusal.is_none(), "{t:#?}");
        assert_eq!(attempt.stage_zncc.len(), stages, "{t:#?}");
        assert!(attempt.grid_zncc >= 0.5, "{t:#?}");
    }
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3}: {report}");
}

/// The views the selection does not pick hold keypoints 1.5 px off the
/// content along each axis. Each one's shift-only fit, over a mid-size
/// window and then briefly over the whole grid, finds the shift that puts
/// the patch back on the content, and the step reports that shift and keeps
/// every keypoint where it was given, though the patch turns.
#[test]
fn the_shift_only_fit_finds_the_shift_to_the_content_and_keeps_the_keypoints() {
    let mut centres = vec![centre(0.0, 0.0)];
    for (ring, from) in [10.0, 20.0, 30.0, 40.0].into_iter().enumerate() {
        for k in 0..4 {
            centres.push(centre(from, 90.0 * k as f64 + 22.5 * ring as f64));
        }
    }
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::new(0.4, 1.0, 0.0), 15.0);
    let (mut track, _) = set_reference(&track, row_of(&track, 0)).expect("an in row");
    let reference = row_of(&track, 0);
    for (row, observation) in track.observations.iter_mut().enumerate() {
        if row == reference {
            continue;
        }
        let p = scene.project(observation.image as usize, WORLD);
        observation.track = Some(TrackMeasurement {
            keypoint: Some([(p[0] + 1.5) as f32, (p[1] - 1.5) as f32]),
            ..TrackMeasurement::default()
        });
    }
    let options = GrownPlaneNormalOptions {
        target_sigma_deg: 2.0,
        max_selected_views: 16,
        ..Default::default()
    };
    let (turned, report) = run(&scene, &edited, &track, &options);
    let sequence = sequence_of(&report);
    assert!(sequence.turned(), "{report}");
    let shift_only = sequence.shift_only();
    assert!(shift_only >= 10, "{report}");
    assert!(report.tilt.degrees > 10.0, "{report}");
    let mut fitted = 0;
    for (row, view) in sequence.views.iter().enumerate() {
        if let GrownPlaneView::ShiftOnly { shift, grid_zncc } = view {
            let truth = scene.project(track.observations[row].image as usize, WORLD);
            let given = keypoint_of(&track, row).expect("a keypoint before the step");
            assert!(distance_px(given, truth) > 2.0);
            assert!(*grid_zncc >= 0.5, "row {row}: {view:?}");
            // The shift carries the patch from the keypoint to the content.
            let to_content = [truth[0] - given[0], truth[1] - given[1]];
            assert!(
                distance_px(*shift, to_content) < 0.5,
                "row {row}: {shift:?} vs {to_content:?}"
            );
            fitted += 1;
        }
    }
    assert_eq!(fitted, shift_only);
    // Not written: every keypoint, 1.5 px off the content or not, is the one
    // the step was given.
    assert_keypoints_kept(&track, &turned);

    // The agreement counts the views fitted for their shift alone beside the
    // joined ones.
    let scores: Vec<f64> = sequence
        .views
        .iter()
        .filter_map(|v| match v {
            GrownPlaneView::Joined { grid_zncc, .. }
            | GrownPlaneView::ShiftOnly { grid_zncc, .. } => Some(*grid_zncc),
            _ => None,
        })
        .collect();
    assert_eq!(scores.len(), sequence.joined() + shift_only);
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    assert!((sequence.agreement() - mean).abs() < 1e-12, "{report}");
}

/// The agreement is the mean score of the joined and shift-only views, the
/// reference and the views left out not counted and a `NaN` score left out;
/// with no score it is `NaN`.
#[test]
fn the_agreement_averages_the_scored_views_only() {
    let sequence = |views: Vec<GrownPlaneView>| GrownPlaneSequence {
        reference: Some(0),
        working_resolution: 16,
        window_half_widths: Vec::new(),
        views,
        trace: Vec::new(),
        picks: Vec::new(),
        pairs: Vec::new(),
        outcome: GrownPlaneOutcome::Turned,
        normal: None,
        turn_deg: None,
        precision: None,
    };
    let joined = |z: f64| GrownPlaneView::Joined {
        shift: [0.0; 2],
        grid_zncc: z,
        distance: None,
        held_zncc: Some(z),
    };
    let left_out = GrownPlaneView::LeftOut {
        reason: GrownPlaneRefusal::UnderFloor,
        distance: None,
        held_zncc: Some(0.1),
    };
    let scored = sequence(vec![
        GrownPlaneView::Reference,
        joined(0.9),
        GrownPlaneView::ShiftOnly {
            shift: [0.0; 2],
            grid_zncc: 0.6,
        },
        joined(f64::NAN),
        left_out.clone(),
        GrownPlaneView::NotAttempted,
    ]);
    assert!((scored.agreement() - 0.75).abs() < 1e-12);
    let none = sequence(vec![
        GrownPlaneView::Reference,
        left_out,
        GrownPlaneView::NotAttempted,
    ]);
    assert!(none.agreement().is_nan());
    assert!(sequence(Vec::new()).agreement().is_nan());
}
