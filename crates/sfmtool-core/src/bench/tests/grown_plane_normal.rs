// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The grown plane normal, decided against synthetic planes seen by pinholes
//! that look at the point from around it: a textured plane seen out to 50
//! degrees, one view whose top cells see something else, views all within a
//! few degrees, a view that sees another plane, repeating texture seen from
//! poses that are off with keypoints on the content, keypoints off the
//! content in views whose poses are further off, and a flat wall.

use std::sync::atomic::{AtomicBool, Ordering};

use super::scene::{fixture_of, Scene, PLANE_Z, WORLD};
use super::*;
use crate::bench::{
    GrownPlaneNormalOptions, GrownPlaneRefusal, GrownPlaneSequence, GrownPlaneStart,
    GrownPlaneTryOutcome, GrownPlaneView,
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

/// From 25 degrees off, every view joins and the patch is turned to within a
/// degree of the plane; each joined view's keypoint is written at the point's
/// projection and the reference keeps its own.
#[test]
fn a_textured_plane_started_far_off_is_turned_onto_it() {
    let centres = wide_centres();
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::new(1.0, 0.5, 0.0), 25.0);
    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);

    let after = degrees(placement_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3} deg off: {report}");
    assert_eq!(
        sequence.joined(),
        centres.len() - 1,
        "{report}: {sequence:#?}"
    );
    assert_eq!(sequence.left_out(), 0);
    let precision = sequence.precision.expect("views joined");
    assert!(!precision.low_confidence, "{report}");
    assert!(precision.sigma_weak_deg >= precision.sigma_strong_deg);
    // The centre does not move, and the track is read back.
    assert!((placement_of(&turned).center - WORLD).norm() < 1e-9);
    assert!(turned.track().expect("track stage").bitmap.is_some());

    let reference = sequence.reference.expect("a reference");
    assert_eq!(
        keypoint_of(&turned, reference),
        keypoint_of(&track, reference),
        "the reference keeps its keypoint"
    );
    for (row, view) in sequence.views.iter().enumerate() {
        if let GrownPlaneView::Joined { keypoint, .. } = view {
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
    assert_eq!(sequence.written, centres.len());
    let sentence = report.to_string();
    assert!(sentence.starts_with("5 of 5 views joined; "), "{sentence}");
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
    let after = degrees(placement_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3} deg off: {report}");
    let row = row_of(&track, 3);
    assert!(
        matches!(sequence.views[row], GrownPlaneView::Joined { .. }),
        "{report}: {:#?}",
        sequence.views[row]
    );
}

/// Views all within three degrees of the reference turn the patch, with a
/// low confidence.
#[test]
fn narrow_views_turn_the_patch_with_low_confidence() {
    let centres = vec![
        centre(0.0, 0.0),
        centre(2.0, 0.0),
        centre(3.0, 120.0),
        centre(2.5, 240.0),
    ];
    let scene = Scene::looking_at(&centres, WORLD, camera(), textured_plane);
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::y(), 20.0);
    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);
    assert!(sequence.turned(), "{report}");
    assert!(report.tilt.degrees > 0.0);
    let precision = sequence.precision.expect("views joined");
    assert!(precision.low_confidence, "{report}");
    assert!(precision.sigma_weak_deg > 2.0, "{report}");
    assert!(report.to_string().contains("low confidence"), "{report}");
    let _ = turned;
}

/// A view that sees the texture on a plane turned 25 degrees from the one
/// the others see is left out, and the plane is the others'. Its own plane
/// is 8.7 σ from the shared one, with a held score of 0.49.
#[test]
fn a_view_whose_own_plane_disagrees_is_left_out() {
    let centres = wide_centres();
    let odd = centres[4];
    // The other plane through the point, turned about world `x`.
    let turn = nalgebra::Rotation3::from_axis_angle(&Vector3::x_axis(), 25f64.to_radians());
    let other_normal = turn * truth();
    let scene = Scene::looking_at(&centres, WORLD, camera(), move |c, d| {
        if c != odd {
            let (x, y) = hit(c, d, PLANE_Z);
            return plane_texture(x, y);
        }
        let (c, d) = (Vector3::from(c), Vector3::from(d));
        let lambda = (WORLD.coords - c).dot(&other_normal) / d.dot(&other_normal);
        let p = c + d * lambda - WORLD.coords;
        // Coordinates in the other plane, so its texture is the plane's
        // turned with it.
        let (u, v) = (turn * Vector3::x(), turn * Vector3::y());
        plane_texture(p.dot(&u), p.dot(&v))
    });
    let edited = edited_of(&scene);
    let track = tilted_track(&edited, Vector3::y(), 10.0);
    let (turned, report) = run(&scene, &edited, &track, &GrownPlaneNormalOptions::default());
    let sequence = sequence_of(&report);
    let after = degrees(placement_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after:.3} deg off: {report}");
    let row = row_of(&track, 4);
    match &sequence.views[row] {
        GrownPlaneView::LeftOut {
            reason, distance, ..
        } => {
            assert_eq!(*reason, GrownPlaneRefusal::TooFar);
            assert!(distance.expect("a shared plane") > 3.0, "{distance:?}");
        }
        other => panic!("left out, not {other:?}: {report}"),
    }
    assert_eq!(sequence.joined(), centres.len() - 2, "{report}");
    // The views are tried nearest the reference first; the odd view is
    // grown again from the starting plane, deferred once and then left out.
    let first_pass = &sequence.trace[..centres.len() - 1];
    assert!(first_pass
        .windows(2)
        .all(|w| w[0].angle_deg <= w[1].angle_deg));
    let tries: Vec<_> = sequence
        .trace
        .iter()
        .filter(|t| t.observation == row)
        .collect();
    assert_eq!(
        tries.iter().map(|t| t.outcome).collect::<Vec<_>>(),
        [
            GrownPlaneTryOutcome::Deferred,
            GrownPlaneTryOutcome::LeftOut
        ]
    );
    // From each plane it grows from its keypoint-implied shift; the search
    // after it lands on that same shift, so it is not grown again.
    for t in &tries {
        let starts: Vec<bool> = t.attempts.iter().map(|a| a.from_start).collect();
        assert_eq!(starts, [false, true], "{t:?}");
        assert!(
            t.attempts
                .iter()
                .all(|a| a.start == GrownPlaneStart::NoSearch),
            "{t:?}"
        );
    }
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
/// search and joins on its first growth, written on the content. The same
/// views searched instead, as a view with no keypoint is, land on other
/// peaks of the checker (here up to 11.2 px off).
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

    let written_off = |track: &EditableTrack| {
        let (turned, report) = run(&scene, &edited, track, &GrownPlaneNormalOptions::default());
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

    let (sequence, worst) = written_off(&track);
    assert_eq!(sequence.joined(), 3, "{sequence:#?}");
    for entry in &sequence.trace {
        assert_eq!(entry.outcome, GrownPlaneTryOutcome::Joined, "{entry:#?}");
        assert_eq!(entry.attempts.len(), 1, "{entry:#?}");
        assert_eq!(entry.attempts[0].start, GrownPlaneStart::NoSearch);
        // Stage 1 reads the window at the implied shift.
        assert!(entry.attempts[0].stage_zncc[0].is_finite(), "{entry:#?}");
    }
    assert!(worst < 0.75, "{worst:.2} px: {sequence:#?}");

    // With no keypoint but the reference's, every view searches.
    let mut searched = track.clone();
    let reference = sequence.reference.expect("a reference");
    for (row, observation) in searched.observations.iter_mut().enumerate() {
        if row != reference {
            observation.track = None;
        }
    }
    let (sequence, worst) = written_off(&searched);
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
    // Grown from the keypoint-implied shift, 6 px off, a view is refused; the
    // search that follows it is what joins.
    for entry in &sequence.trace {
        let starts: Vec<_> = entry.attempts.iter().map(|a| a.start).collect();
        assert_eq!(
            starts,
            [GrownPlaneStart::NoSearch, GrownPlaneStart::Widened],
            "{entry:#?}"
        );
        assert!(entry.attempts[0].refusal.is_some());
        assert!(entry
            .attempts
            .iter()
            .all(|a| a.from_start == entry.attempts[0].from_start));
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
    assert!(sequence.precision.is_none());
    assert_eq!(sequence.written, 0);
    assert_eq!(report.tilt.degrees, 0.0);
    let (a, b) = (placement_of(&track), placement_of(&turned));
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

    let cancel = AtomicBool::new(true);
    let cancelled = Progress::none().cancelled_by(&cancel);
    assert_eq!(
        grown_plane_normal(&track, &edited, &scene.views(), &options, &cancelled).err(),
        Some(NormalError::Cancelled)
    );
    assert!(cancel.load(Ordering::Relaxed));
}
