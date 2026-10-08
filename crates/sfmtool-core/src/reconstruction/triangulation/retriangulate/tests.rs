// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Re-solving the points of a whole reconstruction: what the value that comes
//! back holds, what it leaves alone, and every refusal.
//!
//! The fixture is a synthetic scene whose truth is known -- three images on a
//! short arc looking at a handful of points, every observation the exact
//! projection -- so a retriangulation run from a copy whose points have been
//! nudged has somewhere to converge to. The tests of several cameras use a
//! four-image variant whose images alternate between a pinhole and a fisheye.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::Array2;

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::progress::{Event, Progress};
use crate::reconstruction::data::{
    ObservationSource, Point3D, PointConstraintColumns, SfmrImage, SfmrReconstruction,
    TrackObservation,
};
use crate::reconstruction::edited::{EditedReconstruction, PointMap};
use crate::reconstruction::triangulation::{
    retriangulate_points, GeometryChange, LikelihoodRule, PointVerdict, RetriangulateError,
    RetriangulateOptions, RetriangulateOutcome, RetriangulateReport, RetriangulateWhich,
};

use sfmtool_sfmr_format::{POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED};

const IMG_W: u32 = 640;
const IMG_H: u32 = 480;
const IMAGES: usize = 3;
/// How far a re-solved point may stand from the truth. The pixels are an
/// `f32` column, so the rays they state are the truth's to about a part in
/// ten million of the frame, and a place read off them cannot be nearer
/// than that.
const TRUTH_TOLERANCE: f64 = 1e-3;
const POINTS: usize = 5;

/// Deterministic pseudo-random in `[-1, 1]` from an index.
fn jitter(i: usize, salt: u64) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    ((z % 20001) as f64 / 10000.0) - 1.0
}

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

/// Where `world` lands in image `i`, or `None` when it is outside the frame.
fn project(recon: &SfmrReconstruction, i: usize, world: Point3<f64>) -> Option<[f32; 2]> {
    let image = &recon.image_table.images[i];
    let camera = &recon.image_table.cameras[image.camera_index as usize];
    let local = image.quaternion_wxyz * world.coords + image.translation_xyz;
    let (u, v) = camera.ray_to_pixel([local.x, local.y, local.z])?;
    if u < 0.0 || v < 0.0 || u >= camera.width as f64 || v >= camera.height as f64 {
        return None;
    }
    Some([u as f32, v as f32])
}

/// The truth: cameras on an arc at radius 8 around the origin, points in a
/// box inside it, every observation the exact projection, and one square
/// patch frame per point so a rescale is measurable.
fn truth() -> SfmrReconstruction {
    scene(vec![pinhole()], &[0; IMAGES], |_, _| true)
}

/// The truth's construction over any camera table: one image per entry of
/// `camera_of`, taken through the camera it names, on the same arc, with point
/// `p` observed in image `i` wherever `sees(p, i)` says so. Every observation a
/// track is meant to carry has to land inside its frame.
fn scene(
    cameras: Vec<CameraIntrinsics>,
    camera_of: &[u32],
    sees: impl Fn(usize, usize) -> bool,
) -> SfmrReconstruction {
    let images = camera_of.len();
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_table.cameras = cameras;
    recon.image_table.images = (0..images)
        .map(|i| {
            let angle = 0.3 * (i as f64 - (images as f64 - 1.0) / 2.0);
            let centre = Vector3::new(8.0 * angle.sin(), 0.0, 8.0 * angle.cos());
            let rotation = UnitQuaternion::face_towards(&centre, &Vector3::y()).inverse();
            SfmrImage {
                name: format!("image_{i:03}.jpg"),
                camera_index: camera_of[i],
                quaternion_wxyz: rotation,
                translation_xyz: -(rotation * centre),
            }
        })
        .collect();
    let stats = recon.image_table.depth_statistics.images[0].clone();
    recon.image_table.depth_statistics.images = vec![stats; images];
    recon.image_table.depth_histogram_counts =
        vec![recon.image_table.depth_histogram_counts[0].clone(); images];

    recon.point_set.points = (0..POINTS)
        .map(|p| Point3D {
            position: Point3::new(1.5 * jitter(p, 1), 1.5 * jitter(p, 2), 1.0 * jitter(p, 3)),
            w: 1.0,
            color: [10, 20, 30],
            error: 0.25,
            normal: Vector3::new(0.0, 0.0, -1.0),
        })
        .collect();

    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    let mut expected = Vec::new();
    for p in 0..POINTS {
        let mut count = 0u32;
        expected.push((0..images).filter(|&i| sees(p, i)).count() as u32);
        for i in (0..images).filter(|&i| sees(p, i)) {
            let Some(pixel) = project(&recon, i, recon.point_set.points[p].position) else {
                continue;
            };
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            keypoints.push(pixel);
            count += 1;
        }
        counts.push(count);
    }
    assert_eq!(counts, expected, "the fixture is degenerate");
    let mut keypoints_xy = Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, uv) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    let set = &mut recon.point_set;
    set.tracks = tracks;
    set.observation_counts = counts;
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes: vec![[0u8; 16]; images],
    };
    let mut u = Array2::<f32>::zeros((POINTS, 3));
    let mut v = Array2::<f32>::zeros((POINTS, 3));
    for p in 0..POINTS {
        u[[p, 0]] = 0.05;
        v[[p, 1]] = 0.05;
    }
    set.patch_u_halfvec_xyz = Some(u);
    set.patch_v_halfvec_xyz = Some(v);
    set.point_constraints = Some(PointConstraintColumns::all_free(POINTS));
    recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// The truth with every point nudged off it, its pixels left where the truth
/// put them: what a retriangulation has to find its way back from.
fn nudged() -> SfmrReconstruction {
    nudge(truth())
}

/// `recon` with every point nudged off where it stands.
fn nudge(mut recon: SfmrReconstruction) -> SfmrReconstruction {
    for (p, point) in recon.point_set.points.iter_mut().enumerate() {
        point.position += Vector3::new(
            0.2 * jitter(p, 21),
            0.2 * jitter(p, 22),
            0.2 * jitter(p, 23),
        );
    }
    recon
}

fn edited(recon: SfmrReconstruction) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(recon))
}

/// How far point `p` of `recon` stands from the truth's.
fn error(recon: &SfmrReconstruction, truth: &SfmrReconstruction, p: usize) -> f64 {
    (recon.point_set.points[p].position - truth.point_set.points[p].position).norm()
}

/// Set point `p`'s constraint triple.
fn constrain(recon: &mut SfmrReconstruction, p: usize, kind: u8, distance: f64, reference: u32) {
    let columns = recon
        .point_set
        .point_constraints
        .as_mut()
        .expect("the fixture carries the columns");
    columns.point_constraints[p] = kind;
    columns.constraint_distances[p] = distance;
    columns.constraint_reference_images[p] = reference;
}

/// Check that a report's statuses, its counts and its census tell one story:
/// the statuses are one per index and ascending, every count is the number of
/// statuses of its kind, and the census's buckets are the statuses' verdicts.
fn assert_statuses_agree(report: &RetriangulateReport) {
    assert!(
        report.points.windows(2).all(|w| w[0].index < w[1].index),
        "the statuses are not one per index, ascending"
    );
    let c = &report.census;
    assert_eq!(report.read() + report.held(), report.points.len());
    assert_eq!(c.seen, report.read());
    assert_eq!(c.few, report.kept());
    for (count, verdict) in [
        (c.finite, PointVerdict::Finite),
        (c.finite_pruned, PointVerdict::FinitePruned),
        (c.marked, PointVerdict::Marked),
        (c.ranged, PointVerdict::Ranged),
        (c.thin, PointVerdict::Thin),
        (c.no_depth, PointVerdict::NoDepth),
        (c.behind, PointVerdict::Behind),
        (c.over_bar, PointVerdict::OverBar),
        (c.few, PointVerdict::Few),
    ] {
        assert_eq!(count, report.with_verdict(verdict), "{verdict:?}");
    }
    let mut pruned = 0usize;
    for point in &report.points {
        if let RetriangulateOutcome::Solved {
            verdict,
            pruned: dropped,
            ..
        } = point.outcome
        {
            assert_ne!(verdict, PointVerdict::Few, "a kept point reads as solved");
            assert!(
                dropped == 0 || verdict == PointVerdict::FinitePruned,
                "{dropped} observations pruned under {verdict:?}"
            );
            pruned += dropped as usize;
        }
    }
    assert_eq!(c.pruned_obs, pruned);
    let moved = report.points.iter().filter(|p| p.outcome.moved()).count();
    assert_eq!(report.moved(), moved);
    assert!(report.crossed() <= report.moved());
}

/// Check that each status names its point in both values: the map forwards
/// its `index` to its `new_index`, a live point stands there, and a point whose
/// geometry did not change stands there with the geometry it had.
fn assert_statuses_follow_the_map(
    report: &RetriangulateReport,
    map: &PointMap,
    before: &EditedReconstruction,
    after: &EditedReconstruction,
) {
    for status in &report.points {
        assert_eq!(
            map.forward(status.index),
            Some(status.new_index),
            "{status:?}"
        );
        assert_eq!(map.inverse(status.new_index), Some(status.index));
        let was = before.point(status.index).expect("a live index");
        let now = after.point(status.new_index).expect("a live index");
        let was = (was.point().position, was.point().w);
        let now = (now.point().position, now.point().w);
        if status.outcome.moved() {
            assert_ne!(now, was, "{status:?}");
        } else {
            assert_eq!(now, was, "{status:?}");
        }
    }
}

/// The outcome of the one point a report holds at `index`.
fn outcome_of(report: &RetriangulateReport, index: u32) -> RetriangulateOutcome {
    report
        .point(index)
        .expect("the point was asked about")
        .outcome
}

#[test]
fn every_point_lands_back_on_the_truth_its_pixels_state() {
    let truth = truth();
    let start = edited(nudged());
    let before: Vec<f64> = (0..POINTS).map(|p| error(&start.base, &truth, p)).collect();
    assert!(before.iter().all(|&e| e > 0.01), "{before:?}");

    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.read(), POINTS);
    assert_eq!(report.observations, POINTS * IMAGES);
    assert_eq!(report.moved(), POINTS);
    assert_eq!(report.crossed(), 0);
    assert_eq!(report.kept(), 0);
    assert_eq!(report.held(), 0);
    assert_eq!(report.census.finite, POINTS);
    for p in 0..POINTS {
        assert!(
            error(&next.base, &truth, p) < TRUTH_TOLERANCE,
            "point {p} landed at {:?}",
            next.base.point_set.points[p].position
        );
    }
    // Nothing was deleted and nothing created, so every index still means
    // what it meant.
    for p in 0..POINTS as u32 {
        assert_eq!(map.forward(p), Some(p));
    }
    assert!(report.median_shift() > 0.0, "{}", report.median_shift());

    // One status per point, each a finite position that moved, by as far as
    // the nudge put it off the truth.
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
    assert_eq!(report.points.len(), POINTS);
    for (p, status) in report.points.iter().enumerate() {
        assert_eq!((status.index, status.new_index), (p as u32, p as u32));
        let RetriangulateOutcome::Solved {
            verdict: PointVerdict::Finite,
            pruned: 0,
            change: GeometryChange::Moved { shift: Some(shift) },
        } = status.outcome
        else {
            panic!("point {p}: {:?}", status.outcome);
        };
        assert!((shift - before[p]).abs() < TRUTH_TOLERANCE, "point {p}");
    }
    assert_eq!(
        outcome_of(&report, 0).to_string(),
        "finite",
        "a moved finite point reads as its verdict alone"
    );
}

#[test]
fn the_input_is_left_exactly_as_it_was() {
    let start = edited(nudged());
    let before = start.base.point_set.points.clone();
    let _ = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(start.base.point_set.points, before);
}

#[test]
fn a_held_point_is_never_read_and_never_moves() {
    let mut recon = nudged();
    constrain(&mut recon, 2, POINT_CONSTRAINT_HELD, f64::NAN, u32::MAX);
    let held_at = recon.point_set.points[2].position;
    let start = edited(recon);

    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.held(), 1);
    assert_eq!(report.read(), POINTS - 1);
    assert_eq!(report.moved(), POINTS - 1);
    assert_eq!(next.base.point_set.points[2].position, held_at);

    // The held point has a status like every other, and it says it was not
    // read.
    assert_eq!(report.points.len(), POINTS);
    assert_eq!(outcome_of(&report, 2), RetriangulateOutcome::Held);
    assert_eq!(report.point(2).expect("asked about").new_index, 2);
    assert_statuses_agree(&report);
}

#[test]
fn a_held_point_among_others_named_has_its_own_status() {
    let mut recon = nudged();
    constrain(&mut recon, 2, POINT_CONSTRAINT_HELD, f64::NAN, u32::MAX);
    let start = edited(recon);
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[4, 2, 4]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("point 4 is solved");
    // Each index named once, ascending, the held one included.
    let indexes: Vec<u32> = report.points.iter().map(|p| p.index).collect();
    assert_eq!(indexes, [2, 4]);
    assert_eq!(outcome_of(&report, 2), RetriangulateOutcome::Held);
    assert!(outcome_of(&report, 4).moved());
    assert_eq!(report.point(0), None, "point 0 was not asked about");
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
}

#[test]
fn a_held_point_on_its_own_leaves_nothing_to_solve() {
    let mut recon = nudged();
    constrain(&mut recon, 1, POINT_CONSTRAINT_HELD, f64::NAN, u32::MAX);
    let start = edited(recon);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::These(&[1]),
            &RetriangulateOptions::default(),
            &Progress::none(),
        )
        .expect_err("a held point is not solved"),
        RetriangulateError::NothingToSolve
    );
}

#[test]
fn a_ranged_point_keeps_the_distance_the_value_states() {
    let truth = truth();
    let centre = truth.image_table.images[0].camera_center();
    let range = (truth.point_set.points[3].position - centre).norm() + 1.0;
    let mut recon = nudged();
    constrain(&mut recon, 3, POINT_CONSTRAINT_RANGED, range, 0);
    let start = edited(recon);

    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.census.ranged, 1);
    let got = (next.base.point_set.points[3].position - centre).norm();
    assert!((got - range).abs() < 1e-9 * range, "{got} is not {range}");
    // And the free points beside it are unaffected.
    assert!(error(&next.base, &truth, 0) < TRUTH_TOLERANCE);

    assert!(matches!(
        outcome_of(&report, 3),
        RetriangulateOutcome::Solved {
            verdict: PointVerdict::Ranged,
            pruned: 0,
            change: GeometryChange::Moved { shift: Some(_) },
        }
    ));
    assert_statuses_agree(&report);
}

#[test]
fn a_ranged_point_naming_no_reference_is_solved_free() {
    let truth = truth();
    let mut recon = nudged();
    constrain(
        &mut recon,
        3,
        POINT_CONSTRAINT_RANGED,
        2.0,
        sfmtool_sfmr_format::NO_REFERENCE_IMAGE,
    );
    let start = edited(recon);
    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.census.ranged, 0);
    assert!(error(&next.base, &truth, 3) < TRUTH_TOLERANCE);
}

#[test]
fn a_cancelled_retriangulation_writes_nothing() {
    let start = edited(nudged());
    let flag = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::All,
            &RetriangulateOptions::default(),
            &progress,
        )
        .expect_err("a cancelled call has no answer"),
        RetriangulateError::Cancelled
    );
}

#[test]
fn a_cancel_that_lands_while_the_arrays_are_gathered_still_stops_it() {
    // The flag is set by the first thing the gather reports, so it is
    // already set when the stage boundary after it reads it.
    let start = edited(nudged());
    let flag = AtomicBool::new(false);
    let sink = |_: Event<'_>| flag.store(true, std::sync::atomic::Ordering::Relaxed);
    let progress = Progress::to(&sink).cancelled_by(&flag);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::All,
            &RetriangulateOptions::default(),
            &progress,
        )
        .expect_err("a cancelled call has no answer"),
        RetriangulateError::Cancelled
    );
}

#[test]
fn one_point_is_an_overlay_edit_over_the_very_same_base() {
    let truth = truth();
    let start = edited(nudged());
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[1]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.read(), 1);
    assert_eq!(report.moved(), 1);
    assert_eq!(report.observations, IMAGES);
    assert!(
        Arc::ptr_eq(&next.base, &start.base),
        "the base was rewritten"
    );
    assert_eq!(next.point_count(), POINTS);

    // Delete-and-re-add: the point took a new index, and the map says so.
    let to = map.forward(1).expect("the point survived");
    assert_ne!(to, 1);
    assert!(matches!(map, PointMap::Replaced(_)));
    let moved = next.point(to).expect("the replacement is live");
    assert!(
        (moved.point().position - truth.point_set.points[1].position).norm() < TRUTH_TOLERANCE,
        "{:?}",
        moved.point().position
    );
    // And nothing else moved.
    assert_eq!(map.forward(0), Some(0));
    assert_eq!(
        next.point(0).expect("still live").point().position,
        start.base.point_set.points[0].position
    );

    // The one status names both indexes, and no other point has one.
    assert_eq!(report.points.len(), 1);
    let status = report.point(1).expect("point 1 was asked about");
    assert_eq!(status.new_index, to);
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
}

#[test]
fn a_point_that_does_not_move_takes_no_new_index() {
    // The first pass settles the point on what its pixels state; the second
    // reads the same pixels at the same poses, so it has nothing to do and
    // the point keeps the index the first pass gave it.
    let start = edited(truth());
    let (once, first, _) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[1]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let settled = first.forward(1).expect("the point survived");

    let (twice, map, report) = retriangulate_points(
        &once,
        RetriangulateWhich::These(&[settled]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.read(), 1);
    assert_eq!(report.moved(), 0);
    assert_eq!(map.forward(settled), Some(settled));
    assert_eq!(twice.point_count(), POINTS);
    assert_eq!(twice.added.points.len(), once.added.points.len());

    // Solved, and the answer is the geometry it had, so it keeps its index.
    let status = report.point(settled).expect("asked about");
    assert_eq!(status.new_index, settled);
    assert_eq!(
        status.outcome,
        RetriangulateOutcome::Solved {
            verdict: PointVerdict::Finite,
            pruned: 0,
            change: GeometryChange::Unchanged,
        }
    );
    assert_eq!(status.outcome.to_string(), "finite, unchanged");
    assert!(report.median_shift().is_nan());
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &once, &twice);
}

#[test]
fn a_point_too_few_observations_place_keeps_where_it_stood() {
    let mut recon = nudged();
    // Point 4 keeps one sighting, which states a direction and no place.
    let first = recon
        .point_set
        .tracks
        .iter()
        .position(|o| o.point_index == 4)
        .expect("point 4 is seen");
    let rows: Vec<usize> = (0..recon.point_set.tracks.len())
        .filter(|&row| recon.point_set.tracks[row].point_index != 4 || row == first)
        .collect();
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut recon.point_set.observations
    else {
        panic!("the fixture is embedded_patches");
    };
    *keypoints_xy = keypoints_xy.select(ndarray::Axis(0), &rows);
    recon.point_set.tracks = rows
        .iter()
        .map(|&row| recon.point_set.tracks[row])
        .collect();
    recon.point_set.observation_counts[4] = 1;
    recon.rebuild_derived_fields();
    let stood_at = recon.point_set.points[4].position;

    let start = edited(recon);
    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.kept(), 1);
    assert_eq!(report.census.few, 1);
    assert_eq!(next.base.point_set.points[4].position, stood_at);
    assert_eq!(next.base.point_set.points[4].w, 1.0);

    let kept = report.point(4).expect("asked about");
    assert_eq!(kept.outcome, RetriangulateOutcome::Kept);
    assert_eq!(kept.outcome.verdict(), Some(PointVerdict::Few));
    assert_eq!(kept.new_index, 4);
    assert_eq!(kept.outcome.to_string(), PointVerdict::Few.label());
    assert_statuses_agree(&report);
}

#[test]
fn a_floor_wide_enough_turns_every_point_into_a_direction() {
    let start = edited(nudged());
    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions {
            // Wider than the arc, so no pair of rays opens past it.
            floor_rad: Some(1.0),
            ..RetriangulateOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.census.thin, POINTS);
    assert_eq!(report.crossed(), POINTS);
    assert_eq!(report.moved(), POINTS);
    assert!(next
        .base
        .point_set
        .points
        .iter()
        .all(|p| p.is_at_infinity()));
    assert_eq!(next.base.point_set.infinity_point_count, POINTS);
    // A crossing has no distance in the scene, so none is reported.
    assert!(report.median_shift().is_nan());

    for status in &report.points {
        assert_eq!(
            status.outcome,
            RetriangulateOutcome::Solved {
                verdict: PointVerdict::Thin,
                pruned: 0,
                change: GeometryChange::Crossed,
            }
        );
    }
    assert_statuses_agree(&report);
}

#[test]
fn a_direction_is_read_as_one_and_stays_one() {
    let mut recon = nudged();
    recon.point_set.points[0].w = 0.0;
    recon.point_set.points[0].position = Point3::new(0.0, 0.0, -1.0);
    recon.rebuild_derived_fields();
    let start = edited(recon);
    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.census.marked, 1);
    assert_eq!(report.crossed(), 0);
    assert!(next.base.point_set.points[0].is_at_infinity());

    // The direction turned to the one its rays agree on, which is a move with
    // no distance.
    assert_eq!(
        outcome_of(&report, 0),
        RetriangulateOutcome::Solved {
            verdict: PointVerdict::Marked,
            pruned: 0,
            change: GeometryChange::Moved { shift: None },
        }
    );
    assert_statuses_agree(&report);
}

#[test]
fn the_patch_frame_of_a_moved_point_keeps_its_angular_size() {
    let start = edited(nudged());
    let before = start
        .base
        .point_set
        .patch_u_halfvec_xyz
        .as_ref()
        .expect("the fixture carries frames")[[0, 0]];
    let (next, _, _) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let after = next
        .base
        .point_set
        .patch_u_halfvec_xyz
        .as_ref()
        .expect("the frames survive")[[0, 0]];
    let ratio = f64::from(after / before);
    let was = start
        .base
        .image_table
        .placement_scale(&start.base.point_set.points[0].position);
    let now = next
        .base
        .image_table
        .placement_scale(&next.base.point_set.points[0].position);
    assert!(
        (ratio - now / was).abs() < 1e-5,
        "{ratio} is not {}",
        now / was
    );
}

/// A fisheye camera whose focal and model both differ from [`pinhole`], so a
/// pixel read through the wrong one of the two states a different ray.
fn fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 300.0,
            principal_point_x: IMG_W as f64 / 2.0 + 7.0,
            principal_point_y: IMG_H as f64 / 2.0 - 5.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

/// Four images on the arc, alternating between [`pinhole`] (camera 0) and
/// [`fisheye`] (camera 1). Points 0 and 1 are seen through the pinhole alone,
/// points 2 and 3 through the fisheye alone, and point 4 through both.
fn two_camera_truth() -> SfmrReconstruction {
    scene(vec![pinhole(), fisheye()], &[0, 1, 0, 1], |p, i| match p {
        0 | 1 => i % 2 == 0,
        2 | 3 => i % 2 == 1,
        _ => true,
    })
}

#[test]
fn each_track_is_solved_through_the_cameras_that_saw_it() {
    let truth = two_camera_truth();
    let start = edited(nudge(two_camera_truth()));
    let before: Vec<f64> = (0..POINTS).map(|p| error(&start.base, &truth, p)).collect();
    assert!(before.iter().all(|&e| e > 0.01), "{before:?}");

    // A bar a fraction of a pixel wide: it reads every observation's residual
    // through that observation's own camera, so it passes only where each
    // one was projected through the lens that took it.
    let options = RetriangulateOptions {
        bar_px: Some(0.01),
        ..RetriangulateOptions::default()
    };
    let (next, _, report) =
        retriangulate_points(&start, RetriangulateWhich::All, &options, &Progress::none())
            .expect("a value taken through two cameras retriangulates");

    assert_eq!(report.read(), POINTS);
    assert_eq!(report.observations, 2 + 2 + 2 + 2 + 4);
    assert_eq!(report.census.finite, POINTS, "{:?}", report.census);
    assert_eq!(report.moved(), POINTS);
    for p in 0..POINTS {
        assert!(
            error(&next.base, &truth, p) < TRUTH_TOLERANCE,
            "point {p} landed at {:?}",
            next.base.point_set.points[p].position
        );
    }
}

#[test]
fn a_track_read_through_the_wrong_camera_does_not_land_on_the_truth() {
    // The control for the test above: the same pixels with the table saying
    // every image was taken through the pinhole. The fisheye's pixels then
    // state the wrong rays, so the tracks they belong to miss the truth, and
    // the pinhole-only tracks are untouched by the change.
    let truth = two_camera_truth();
    let mut recon = nudge(two_camera_truth());
    recon.image_table.cameras[1] = pinhole();
    let (next, _, _) = retriangulate_points(
        &edited(recon),
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    for p in 0..2 {
        assert!(error(&next.base, &truth, p) < TRUTH_TOLERANCE, "point {p}");
    }
    for p in 2..POINTS {
        assert!(
            !next.base.point_set.points[p].is_at_infinity()
                && error(&next.base, &truth, p) > 100.0 * TRUTH_TOLERANCE,
            "point {p} landed on the truth through the wrong lens"
        );
    }
}

/// [`two_camera_truth`], nudged, with point 3 (seen through the fisheye
/// alone) and point 4 (seen through both cameras) ranged at their true
/// distance from image 1, a fisheye image. Returns the value and that image's
/// centre.
fn two_camera_ranged() -> (SfmrReconstruction, Point3<f64>) {
    let truth = two_camera_truth();
    let centre = truth.image_table.images[1].camera_center();
    let mut recon = nudge(two_camera_truth());
    for p in [3, 4] {
        let range = (truth.point_set.points[p].position - centre).norm();
        constrain(&mut recon, p, POINT_CONSTRAINT_RANGED, range, 1);
    }
    (recon, centre)
}

#[test]
fn a_ranged_point_seen_through_the_second_camera_lands_on_the_truth() {
    let truth = two_camera_truth();
    let (recon, centre) = two_camera_ranged();
    let (next, _, report) = retriangulate_points(
        &edited(recon),
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("a value taken through two cameras retriangulates");

    assert_eq!(report.census.ranged, 2, "{:?}", report.census);
    for p in [3, 4] {
        let range = (truth.point_set.points[p].position - centre).norm();
        let got = (next.base.point_set.points[p].position - centre).norm();
        assert!((got - range).abs() < 1e-9 * range, "{got} is not {range}");
        assert!(
            error(&next.base, &truth, p) < TRUTH_TOLERANCE,
            "point {p} landed at {:?}",
            next.base.point_set.points[p].position
        );
    }
}

#[test]
fn a_ranged_point_read_through_the_wrong_camera_does_not_land_on_the_truth() {
    // The control for the test above: the distance alone does not place the
    // point, so with the table saying every image was taken through the
    // pinhole, the fisheye's pixels state the wrong rays and the ranged points
    // miss the truth while still keeping their distance.
    let truth = two_camera_truth();
    let (mut recon, centre) = two_camera_ranged();
    recon.image_table.cameras[1] = pinhole();
    let (next, _, report) = retriangulate_points(
        &edited(recon),
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.census.ranged, 2, "{:?}", report.census);
    for p in [3, 4] {
        let range = (truth.point_set.points[p].position - centre).norm();
        let got = (next.base.point_set.points[p].position - centre).norm();
        assert!((got - range).abs() < 1e-9 * range, "{got} is not {range}");
        assert!(
            error(&next.base, &truth, p) > 100.0 * TRUTH_TOLERANCE,
            "point {p} landed on the truth through the wrong lens"
        );
    }
}

/// `recon` with point `p` moved to `world` and each of its observations
/// re-projected through its own image's camera.
fn move_point(recon: &mut SfmrReconstruction, p: usize, world: Point3<f64>) {
    recon.point_set.points[p].position = world;
    let rows: Vec<(usize, usize)> = recon
        .point_set
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.point_index as usize == p)
        .map(|(row, t)| (row, t.image_index as usize))
        .collect();
    let pixels: Vec<[f32; 2]> = rows
        .iter()
        .map(|&(_, i)| project(recon, i, world).expect("the moved point is in every frame"))
        .collect();
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut recon.point_set.observations
    else {
        unreachable!("the fixture embeds its keypoints");
    };
    for (&(row, _), uv) in rows.iter().zip(&pixels) {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    recon.rebuild_derived_fields();
}

/// [`two_camera_truth`] with point 4, the one seen through both cameras,
/// moved out to `FAR` units in front of the arc, so its four rays cross at an
/// angle far below what a fraction of a pixel of noise can resolve. Returns
/// the value and the direction the far point lies in.
fn two_camera_far() -> (SfmrReconstruction, Vector3<f64>) {
    const FAR: f64 = 1.0e5;
    let mut recon = two_camera_truth();
    let far = Point3::new(30.0, -20.0, -FAR);
    move_point(&mut recon, 4, far);
    (recon, far.coords.normalize())
}

#[test]
fn a_far_track_seen_through_two_cameras_becomes_its_bearing() {
    // The likelihood rule on a value taken through two cameras: the far track,
    // seen through the pinhole and the fisheye, has no depth at a fifth of a
    // pixel and becomes the bearing its rays share, while the near tracks --
    // through the pinhole alone, the fisheye alone, or both -- keep a depth
    // and land on the truth.
    let (recon, direction) = two_camera_far();
    let truth = recon.clone();
    let options = RetriangulateOptions {
        likelihood: Some(LikelihoodRule::at(0.2)),
        ..RetriangulateOptions::default()
    };
    let start = edited(nudge(recon));
    let (next, map, report) =
        retriangulate_points(&start, RetriangulateWhich::All, &options, &Progress::none())
            .expect("a value taken through two cameras retriangulates");

    assert_eq!(report.census.no_depth, 1, "{:?}", report.census);
    assert_eq!(report.census.finite, POINTS - 1, "{:?}", report.census);
    assert_eq!(report.crossed(), 1);
    for p in 0..4 {
        assert!(error(&next.base, &truth, p) < TRUTH_TOLERANCE, "point {p}");
    }
    let far = &next.base.point_set.points[4];
    assert!(far.is_at_infinity());
    let off = far.position.coords.normalize().angle(&direction);
    assert!(off < 1e-5, "the bearing is {off} rad off the far point");

    // The far point's status names the rule that made it a direction, and it
    // crossed from the position it was stored at; the near ones moved as
    // positions.
    assert_eq!(
        outcome_of(&report, 4),
        RetriangulateOutcome::Solved {
            verdict: PointVerdict::NoDepth,
            pruned: 0,
            change: GeometryChange::Crossed,
        }
    );
    assert_eq!(
        outcome_of(&report, 4).to_string(),
        PointVerdict::NoDepth.label()
    );
    for p in 0..4 {
        assert!(
            matches!(
                outcome_of(&report, p),
                RetriangulateOutcome::Solved {
                    verdict: PointVerdict::Finite,
                    pruned: 0,
                    change: GeometryChange::Moved { shift: Some(_) },
                }
            ),
            "point {p}: {:?}",
            outcome_of(&report, p)
        );
    }
    assert_eq!(report.with_verdict(PointVerdict::NoDepth), 1);
    assert_eq!(report.with_verdict(PointVerdict::Finite), POINTS - 1);
    assert_eq!(report.moved(), POINTS);
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
}

#[test]
fn a_far_point_named_alone_is_rewritten_as_its_bearing_and_says_why() {
    // The likelihood rule under `These`: the far point is the one asked about,
    // it becomes its bearing, takes a new index, and its status says the
    // likelihood rule decided it.
    let (recon, direction) = two_camera_far();
    let options = RetriangulateOptions {
        likelihood: Some(LikelihoodRule::at(0.2)),
        ..RetriangulateOptions::default()
    };
    let start = edited(recon);
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[4]),
        &options,
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.points.len(), 1);
    let status = *report.point(4).expect("point 4 was asked about");
    assert_eq!(
        status.outcome,
        RetriangulateOutcome::Solved {
            verdict: PointVerdict::NoDepth,
            pruned: 0,
            change: GeometryChange::Crossed,
        }
    );
    assert_ne!(status.new_index, 4, "a rewritten point takes a new index");
    let far = next.point(status.new_index).expect("live").point().clone();
    assert!(far.is_at_infinity());
    assert!(far.position.coords.normalize().angle(&direction) < 1e-5);
    assert_eq!(report.census.no_depth, 1);
    assert_eq!(report.crossed(), 1);
    assert!(report.median_shift().is_nan(), "a crossing has no distance");
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
}

#[test]
fn a_far_track_read_through_the_wrong_camera_misses_its_bearing() {
    // The control for the test above: with the table saying every image was
    // taken through the pinhole, the far track's fisheye pixels state rays
    // that point elsewhere, and whatever the rule makes of the track, it is
    // not the direction the far point lies in.
    let (mut recon, direction) = two_camera_far();
    recon.image_table.cameras[1] = pinhole();
    let options = RetriangulateOptions {
        likelihood: Some(LikelihoodRule::at(0.2)),
        ..RetriangulateOptions::default()
    };
    let (next, _, _) = retriangulate_points(
        &edited(recon),
        RetriangulateWhich::All,
        &options,
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let far = &next.base.point_set.points[4];
    let off = far.position.coords.normalize().angle(&direction);
    assert!(
        !far.is_at_infinity() || off > 1e-3,
        "the far point found its bearing through the wrong lens ({off} rad off)"
    );
}

/// Whether point 2 of [`two_camera_truth`], moved 1000 units out and so seen
/// through the fisheye alone at a parallax near what the noise resolves, comes
/// back as a direction under the likelihood rule at `sigma_px`.
fn fisheye_track_is_a_direction_at(sigma_px: f64) -> bool {
    let mut recon = two_camera_truth();
    move_point(&mut recon, 2, Point3::new(2.0, 1.0, -1000.0));
    let options = RetriangulateOptions {
        likelihood: Some(LikelihoodRule::at(sigma_px)),
        ..RetriangulateOptions::default()
    };
    let (next, _, _) = retriangulate_points(
        &edited(recon),
        RetriangulateWhich::All,
        &options,
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    next.base.point_set.points[2].is_at_infinity()
}

#[test]
fn the_likelihood_rule_weighs_each_ray_through_its_own_camera() {
    // A ray's noise weight is the pixel noise carried through its own lens's
    // derivative. The fisheye resolves 300 px per radian at its centre and the
    // pinhole 500, so a fisheye ray weighted as if the pinhole took it would
    // claim (500 / 300)^2, about 2.8 times, the depth evidence it carries.
    // This track's verdict changes between sigma 0.15 and 0.25 px, a factor of
    // 2.8 in the score, so at 0.25 px it is a direction only when each ray is
    // weighted through the camera that took it.
    assert!(!fisheye_track_is_a_direction_at(0.15));
    assert!(fisheye_track_is_a_direction_at(0.25));
}

#[test]
fn a_camera_no_image_names_changes_nothing() {
    let (one, _, one_report) = retriangulate_points(
        &edited(nudged()),
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let mut recon = nudged();
    recon.image_table.cameras.push(fisheye());
    let (two, _, two_report) = retriangulate_points(
        &edited(recon),
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(one.base.point_set.points, two.base.point_set.points);
    assert_eq!(one_report, two_report);
}

#[test]
fn a_value_with_no_pose_is_refused() {
    let mut recon = nudged();
    for image in recon.image_table.images.iter_mut() {
        image.translation_xyz = Vector3::new(f64::NAN, f64::NAN, f64::NAN);
    }
    let start = edited(recon);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::All,
            &RetriangulateOptions::default(),
            &Progress::none(),
        )
        .expect_err("an unposed value is refused"),
        RetriangulateError::NoPosedImages
    );
}

#[test]
fn an_index_that_names_no_live_point_is_refused() {
    let start = edited(nudged());
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::These(&[99]),
            &RetriangulateOptions::default(),
            &Progress::none(),
        )
        .expect_err("index 99 names nothing"),
        RetriangulateError::NoSuchPoint(99)
    );
}

#[test]
fn an_overlay_edit_is_folded_in_before_every_point_is_re_solved() {
    let mut start = edited(nudged());
    start.delete_point(0).expect("point 0 is live");
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.read(), POINTS - 1);
    assert_eq!(next.base.point_count(), POINTS - 1);
    assert!(next.deleted_points.is_empty());
    // The deleted index resolves to nothing, and the rest shift down.
    assert_eq!(map.forward(0), None);
    assert_eq!(map.forward(1), Some(0));

    // The statuses are in the caller's indexing and name the new base's rows.
    let pairs: Vec<(u32, u32)> = report
        .points
        .iter()
        .map(|p| (p.index, p.new_index))
        .collect();
    assert_eq!(pairs, [(1, 0), (2, 1), (3, 2), (4, 3)]);
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
}

#[test]
fn the_census_of_one_point_names_its_one_verdict() {
    let start = edited(nudged());
    let (_, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[2]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.census.sole_verdict(), Some(PointVerdict::Finite));

    let (_, _, whole) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(whole.census.sole_verdict(), None);
}

#[test]
fn a_free_constraint_column_leaves_every_point_free() {
    let mut recon = nudged();
    for p in 0..POINTS {
        constrain(&mut recon, p, POINT_CONSTRAINT_FREE, f64::NAN, u32::MAX);
    }
    let start = edited(recon);
    let (_, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.held(), 0);
    assert_eq!(report.census.ranged, 0);
    assert_eq!(report.census.finite, POINTS);
}

#[test]
fn the_stages_are_named_for_a_reader_watching() {
    let seen = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Leave { phase, .. } = event {
            seen.lock().unwrap().push(phase);
        }
    };
    let start = edited(nudged());
    let _ = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::to(&sink),
    )
    .expect("the fixture retriangulates");
    let seen = seen.lock().unwrap();
    assert_eq!(
        *seen,
        ["gather observations", "retriangulate", "write back"],
        "{seen:?}"
    );
}

// ── Per-point statuses ──────────────────────────────────────────────────

/// Four images: three on the arc that see every point, and a fourth that
/// stands beyond the points looking away from them and sees point `p` alone.
/// Its pixel states the ray whose backward extension passes through `p`'s
/// truth, so every line of `p`'s track meets there and the truth lies behind
/// the fourth camera. Returns the truth and that truth nudged.
fn backward_sighting(p: usize) -> (SfmrReconstruction, SfmrReconstruction) {
    let mut truth = scene(vec![pinhole()], &[0; 4], |q, i| i < 3 || q == p);
    let centre = Vector3::new(0.0, 0.0, -8.0);
    let rotation = UnitQuaternion::face_towards(&(-centre), &Vector3::y()).inverse();
    truth.image_table.images[3].quaternion_wxyz = rotation;
    truth.image_table.images[3].translation_xyz = -(rotation * centre);
    let along = (centre - truth.point_set.points[p].position.coords).normalize();
    let local = rotation * along;
    let (u, v) = pinhole()
        .ray_to_pixel([local.x, local.y, local.z])
        .expect("the ray is in front of the fourth camera");
    let row = truth
        .point_set
        .tracks
        .iter()
        .position(|o| o.point_index == p as u32 && o.image_index == 3)
        .expect("the fourth image sees the point");
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut truth.point_set.observations
    else {
        panic!("the fixture is embedded_patches");
    };
    keypoints_xy[[row, 0]] = u as f32;
    keypoints_xy[[row, 1]] = v as f32;
    let nudged = nudge(truth.clone());
    (truth, nudged)
}

#[test]
fn a_point_solved_on_the_observations_that_agree_names_how_many_were_left_out() {
    let (truth, recon) = backward_sighting(2);
    let start = edited(recon);
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert!(
        matches!(
            outcome_of(&report, 2),
            RetriangulateOutcome::Solved {
                verdict: PointVerdict::FinitePruned,
                pruned: 1,
                change: GeometryChange::Moved { shift: Some(_) },
            }
        ),
        "{:?}",
        outcome_of(&report, 2)
    );
    assert!(error(&next.base, &truth, 2) < TRUTH_TOLERANCE);
    assert_eq!(report.census.pruned_obs, 1);
    assert_eq!(
        outcome_of(&report, 2).to_string(),
        "finite, on the observations that agree \
         (1 observation that sees it behind was left out)"
    );
    // The rest were not touched by the prune.
    for p in [0, 1, 3, 4] {
        assert!(matches!(
            outcome_of(&report, p),
            RetriangulateOutcome::Solved {
                verdict: PointVerdict::Finite,
                pruned: 0,
                ..
            }
        ));
    }
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
}

#[test]
fn a_point_behind_a_camera_that_sees_it_becomes_a_direction_and_says_why() {
    let (_, recon) = backward_sighting(2);
    let start = edited(recon);
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions {
            prune_behind: false,
            ..RetriangulateOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(
        outcome_of(&report, 2),
        RetriangulateOutcome::Solved {
            verdict: PointVerdict::Behind,
            pruned: 0,
            change: GeometryChange::Crossed,
        }
    );
    assert!(next.base.point_set.points[2].is_at_infinity());
    assert_eq!(report.crossed(), 1);
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
}

#[test]
fn a_point_past_the_reprojection_bar_becomes_a_direction_and_says_why() {
    let mut recon = nudged();
    // One of point 3's pixels moved well off the truth, so no place reprojects
    // within half a pixel of all three.
    let row = recon
        .point_set
        .tracks
        .iter()
        .position(|o| o.point_index == 3)
        .expect("point 3 is seen");
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut recon.point_set.observations
    else {
        panic!("the fixture is embedded_patches");
    };
    keypoints_xy[[row, 0]] += 8.0;
    let start = edited(recon);
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions {
            bar_px: Some(0.5),
            ..RetriangulateOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(
        outcome_of(&report, 3),
        RetriangulateOutcome::Solved {
            verdict: PointVerdict::OverBar,
            pruned: 0,
            change: GeometryChange::Crossed,
        }
    );
    assert_eq!(report.with_verdict(PointVerdict::Finite), POINTS - 1);
    assert!(next.base.point_set.points[3].is_at_infinity());
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &start, &next);
}

#[test]
fn statuses_name_each_point_through_an_overlay_that_reassigns_indexes() {
    let truth = truth();
    let start = edited(nudged());
    // An overlay in which point 1 has been rewritten already, so it lives at
    // an index past the base and stands on its truth.
    let (layered, first, _) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[1]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let one = first.forward(1).expect("the point survived");
    assert!(one >= POINTS as u32, "{one}");

    // Points 0 and 3 are still nudged, so they move and take new indexes;
    // point `one` is already settled, so it keeps the index it has.
    let (next, map, report) = retriangulate_points(
        &layered,
        RetriangulateWhich::These(&[one, 3, 0]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let indexes: Vec<u32> = report.points.iter().map(|p| p.index).collect();
    assert_eq!(indexes, [0, 3, one]);
    for (index, truth_index) in [(0, 0), (3, 3), (one, 1)] {
        let status = report.point(index).expect("asked about");
        let landed = next.point(status.new_index).expect("live").point().position;
        assert!(
            (landed - truth.point_set.points[truth_index].position).norm() < TRUTH_TOLERANCE,
            "point {index} at {} landed at {landed:?}",
            status.new_index
        );
    }
    for index in [0, 3] {
        let status = report.point(index).expect("asked about");
        assert_ne!(
            status.new_index, index,
            "a rewritten point takes a new index"
        );
        assert!(status.outcome.moved());
    }
    let settled = report.point(one).expect("asked about");
    assert_eq!(settled.new_index, one);
    assert!(!settled.outcome.moved());
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &layered, &next);
}

#[test]
fn statuses_of_every_point_are_read_back_through_the_fold() {
    let truth = truth();
    let mut recon = nudged();
    constrain(&mut recon, 3, POINT_CONSTRAINT_HELD, f64::NAN, u32::MAX);
    let start = edited(recon);
    // Point 1 rewritten into the overlay, and point 2 deleted from it: the
    // fold puts the rewrite back in its base row and closes up the deletion,
    // so the caller's indexes and the new base's rows differ.
    let (mut layered, first, _) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[1]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let one = first.forward(1).expect("the point survived");
    layered.delete_point(2).expect("point 2 is live");

    let (next, map, report) = retriangulate_points(
        &layered,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let pairs: Vec<(u32, u32)> = report
        .points
        .iter()
        .map(|p| (p.index, p.new_index))
        .collect();
    assert_eq!(pairs, [(0, 0), (3, 2), (4, 3), (one, 1)]);
    assert_eq!(outcome_of(&report, 3), RetriangulateOutcome::Held);
    assert_eq!(report.held(), 1);
    assert_eq!(report.read(), 3);
    for (index, truth_index) in [(0, 0), (4, 4), (one, 1)] {
        let status = report.point(index).expect("asked about");
        let landed = next.point(status.new_index).expect("live").point().position;
        assert!(
            (landed - truth.point_set.points[truth_index].position).norm() < TRUTH_TOLERANCE,
            "point {index} landed at {landed:?}"
        );
    }
    assert_statuses_agree(&report);
    assert_statuses_follow_the_map(&report, &map, &layered, &next);
}

/// A reference only the display render picked stays display-only through a
/// retriangulation, whether the point is rewritten on its own (through a whole
/// record) or with every other point (in place), so both save it as `-1`.
#[test]
fn a_display_pick_stays_display_only_whether_one_point_or_all_are_retriangulated() {
    let mut recon = nudged();
    let mut refs = vec![sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION; POINTS];
    refs[1] = 0;
    recon.point_set.reference_observations = Some(refs);
    let mut marks = vec![false; POINTS];
    marks[1] = true;
    recon.point_set.display_only_references = Some(marks);
    let start = edited(recon);

    let saved = |next: &EditedReconstruction, map: &PointMap| -> (i32, Option<i32>) {
        let to = map.forward(1).expect("the point survived");
        let view = next.point(to).expect("live");
        assert!(view.display_only_reference(), "the mark was lost");
        let (value, rows) = next.materialize();
        let row = rows
            .forward(to)
            .expect("the point is in the materialised value") as usize;
        let written = value
            .point_set
            .saved_reference_observations()
            .expect("the column")[row];
        (written, view.reference_observation())
    };

    let mut results = Vec::new();
    for which in [RetriangulateWhich::These(&[1]), RetriangulateWhich::All] {
        let (next, map, report) = retriangulate_points(
            &start,
            which,
            &RetriangulateOptions::default(),
            &Progress::none(),
        )
        .expect("the fixture retriangulates");
        assert!(report.moved() >= 1);
        results.push(saved(&next, &map));
    }
    assert_eq!(results[0], (-1, Some(0)));
    assert_eq!(results[0], results[1]);
}
