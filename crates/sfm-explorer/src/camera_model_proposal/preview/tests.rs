// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What the preview does with each point, the targets and counts it builds
//! from a report, and the blend's motion with its pending-answer swap.

use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use sfmtool_core::reconstruction::bundle_adjust::patch_frame_factor;
use sfmtool_core::reconstruction::triangulation::{PointCensus, PointVerdict};
use sfmtool_core::reconstruction::{
    GeometryChange, ImageTable, Point3D, RetriangulateOutcome, RetriangulateReport,
    RetriangulatedPoint,
};
use sfmtool_core::{RowMap, SfmrReconstruction};

use super::*;

fn finite(x: f64, y: f64, z: f64) -> Point3D {
    Point3D {
        position: Point3::new(x, y, z),
        w: 1.0,
        color: [10, 20, 30],
        error: 0.5,
        normal: Vector3::zeros(),
    }
}

fn direction(x: f64, y: f64, z: f64) -> Point3D {
    let unit = Vector3::new(x, y, z).normalize();
    Point3D {
        position: Point3::from(unit),
        w: 0.0,
        ..finite(0.0, 0.0, 0.0)
    }
}

fn solved(verdict: PointVerdict, change: GeometryChange) -> RetriangulateOutcome {
    RetriangulateOutcome::Solved {
        verdict,
        pruned: 0,
        change,
    }
}

const MOVED: GeometryChange = GeometryChange::Moved { shift: Some(1.0) };

/// Each outcome of the proposed run, against each kind of stored point.
#[test]
fn every_outcome_decides_what_the_point_does() {
    let centroid = Point3::origin();
    let stored = finite(0.0, 0.0, 4.0);
    let answer = finite(0.0, 0.0, 2.0);

    // Held and kept points do not move, whatever the answer holds.
    for outcome in [RetriangulateOutcome::Held, RetriangulateOutcome::Kept] {
        let preview = point_preview(&stored, &outcome, &answer, &centroid);
        assert_eq!(preview.flag(), FLAG_STILL, "{outcome:?}");
    }
    assert_eq!(
        point_preview(&stored, &RetriangulateOutcome::Held, &answer, &centroid),
        PointPreview::Held
    );
    assert_eq!(
        point_preview(&stored, &RetriangulateOutcome::Kept, &answer, &centroid),
        PointPreview::Kept
    );

    // A finite point answered finite moves to the answer, its surfel scaled
    // by the factor the retriangulation rescales the frame by.
    let outcome = solved(PointVerdict::Finite, MOVED);
    match point_preview(&stored, &outcome, &answer, &centroid) {
        PointPreview::Moves {
            target,
            at_infinity,
            size_factor,
            changed,
        } => {
            assert_eq!(target, answer.position);
            assert!(!at_infinity && changed);
            let before = ImageTable::placement_scale_from(&centroid, &stored.position);
            let expected = patch_frame_factor(&centroid, Some(before), Some(&answer.position));
            assert_eq!(Some(size_factor), expected);
            assert_eq!(size_factor, 0.5);
        }
        other => panic!("{other:?}"),
    }
    // A ranged answer at a finite distance and a pruned one move the same way.
    for verdict in [PointVerdict::Ranged, PointVerdict::FinitePruned] {
        let preview = point_preview(&stored, &solved(verdict, MOVED), &answer, &centroid);
        assert_eq!(preview.flag(), FLAG_MOVES, "{verdict:?}");
    }

    // A finite point answered as a direction loses its position: it stays
    // and takes the failure colour, with the verdict as its reason.
    let lost = point_preview(
        &stored,
        &solved(PointVerdict::Behind, GeometryChange::Crossed),
        &direction(0.0, 1.0, 0.0),
        &centroid,
    );
    assert_eq!(lost, PointPreview::Loses(PointVerdict::Behind));
    assert_eq!(lost.flag(), FLAG_LOSES);

    // A point at infinity answered as a direction turns to the proposed
    // bearing, a unit vector, and keeps its size.
    let bearing = direction(1.0, 1.0, 0.0);
    match point_preview(
        &direction(1.0, 0.0, 0.0),
        &solved(PointVerdict::Marked, GeometryChange::Moved { shift: None }),
        &bearing,
        &centroid,
    ) {
        PointPreview::Moves {
            target,
            at_infinity,
            size_factor,
            ..
        } => {
            assert!(at_infinity);
            assert!((target.coords.norm() - 1.0).abs() < 1e-12);
            assert_eq!(target, bearing.position);
            assert_eq!(size_factor, 1.0);
        }
        other => panic!("{other:?}"),
    }

    // A direction the run places at a position is written by Apply, but is
    // not drawn travelling: it cannot move from a direction to a place along
    // a line.
    let placed = point_preview(
        &direction(1.0, 0.0, 0.0),
        &solved(PointVerdict::Ranged, GeometryChange::Crossed),
        &answer,
        &centroid,
    );
    assert_eq!(placed, PointPreview::Placed);
    assert_eq!(placed.flag(), FLAG_STILL);

    // An unchanged answer is a move to where the point already is.
    let still = point_preview(
        &stored,
        &solved(PointVerdict::Finite, GeometryChange::Unchanged),
        &stored,
        &centroid,
    );
    assert!(matches!(still, PointPreview::Moves { changed: false, .. }));
}

/// A report over a value with a deleted point: the targets land in the rows
/// the version gives each point, the deleted one's row does not move, and the
/// counts name the moves, the kept, the held and the lost, largest reason
/// first.
#[test]
fn the_targets_and_counts_are_read_off_the_report() {
    let mut stored = SfmrReconstruction::demo(8);
    for point in stored.point_set.points.iter_mut() {
        point.position = Point3::new(0.0, 0.0, 5.0);
        point.w = 1.0;
    }
    // The version has a ninth base point, row 3, that was deleted; the value
    // the run read is the materialisation, so its row 3 is the version's 4.
    let rows = RowMap::by_removal(9, &[3]);
    let mut answer = stored.clone();
    let status = |index: u32, outcome| RetriangulatedPoint {
        index,
        new_index: index,
        outcome,
        released: false,
    };
    let shifts = [1.0, 2.0, 3.0];
    for (k, row) in [0usize, 1, 2].into_iter().enumerate() {
        answer.point_set.points[row].position.z = 5.0 - shifts[k];
    }
    for row in [4usize, 5] {
        answer.point_set.points[row] = direction(0.0, 1.0, 0.0);
    }
    answer.point_set.points[6] = direction(1.0, 0.0, 0.0);
    let points = vec![
        status(
            0,
            solved(
                PointVerdict::Finite,
                GeometryChange::Moved { shift: Some(1.0) },
            ),
        ),
        status(
            1,
            solved(
                PointVerdict::Finite,
                GeometryChange::Moved { shift: Some(2.0) },
            ),
        ),
        status(
            2,
            solved(
                PointVerdict::Finite,
                GeometryChange::Moved { shift: Some(3.0) },
            ),
        ),
        status(3, RetriangulateOutcome::Kept),
        status(4, solved(PointVerdict::Behind, GeometryChange::Crossed)),
        status(5, solved(PointVerdict::Behind, GeometryChange::Crossed)),
        status(6, solved(PointVerdict::OverBar, GeometryChange::Crossed)),
        status(7, RetriangulateOutcome::Held),
    ];
    let report = RetriangulateReport {
        points,
        observations: 0,
        census: PointCensus::default(),
    };
    let (targets, counts) = PreviewTargets::build(7, &stored, Some(&rows), 9, 0, &report, &answer);

    assert_eq!(targets.generation, 7);
    assert_eq!(targets.base.len(), 9);
    assert!(targets.additions.is_empty());
    // Materialised rows 0..3 are the version's 0..3, and 3.. its 4...
    assert_eq!(targets.base[0], [0.0, 0.0, 4.0, FLAG_MOVES]);
    assert_eq!(targets.base[2][3], FLAG_MOVES);
    assert_eq!(targets.base[3][3], FLAG_STILL, "the deleted row moved");
    assert_eq!(targets.base[4][3], FLAG_STILL, "a kept point moved");
    assert_eq!(targets.base[5][3], FLAG_LOSES);
    assert_eq!(targets.base[7][3], FLAG_LOSES);
    assert_eq!(targets.base[8][3], FLAG_STILL, "a held point moved");
    let centroid = answer.image_table.camera_centroid();
    let before = ImageTable::placement_scale_from(&centroid, &stored.point_set.points[0].position);
    let expected = patch_frame_factor(
        &centroid,
        Some(before),
        Some(&answer.point_set.points[0].position),
    );
    assert_eq!(Some(targets.base_size[0]), expected);
    assert!(targets.touches(0), "a moving point's rays are not hidden");
    // A point that loses its position stands where its rays end.
    assert!(!targets.touches(5) && !targets.touches(3) && !targets.touches(8));

    assert_eq!(counts.tracks, 8);
    assert_eq!(counts.moves, 3);
    assert_eq!(counts.kept, 1);
    assert_eq!(counts.held, 1);
    assert_eq!(counts.lost(), 3);
    assert_eq!(
        counts.lost_by_verdict,
        vec![(PointVerdict::Behind, 2), (PointVerdict::OverBar, 1)]
    );
    let lost: Vec<u32> = counts.lost_points.iter().map(|p| p.index).collect();
    assert_eq!(lost, vec![5, 6, 7]);
    assert_eq!(counts.median_shift, Some(2.0));
    assert!((counts.p90_shift.unwrap() - 2.8).abs() < 1e-12);
    assert_eq!(
        lost_reason(PointVerdict::Behind),
        "behind a camera that sees them"
    );
}

/// Targets for `generation`, with one moving row.
fn targets(generation: u64) -> Arc<PreviewTargets> {
    Arc::new(PreviewTargets {
        generation,
        base: vec![[0.0, 0.0, 1.0, FLAG_MOVES]],
        base_size: vec![1.0],
        additions: Vec::new(),
        additions_size: Vec::new(),
        touched: vec![0],
    })
}

const DT: f64 = 1.0 / 60.0;

/// Ticking with an answer in hand writes its targets at once and runs `s` to
/// 1 in about 0.7 s; clearing runs it back to 0.
#[test]
fn ticking_runs_the_blend_up_and_clearing_runs_it_back() {
    let mut motion = PreviewMotion::default();
    let answer = targets(1);
    // Nothing to draw before the box is ticked.
    assert!(!motion.step(false, Some(&answer), true, DT));
    assert!(motion.drawn.is_none());

    let mut frames = 0;
    while motion.step(true, Some(&answer), true, DT) {
        frames += 1;
        assert_eq!(motion.drawn.as_ref().unwrap().generation, 1);
    }
    let seconds = (frames + 1) as f64 * DT;
    assert!((0.6..=0.8).contains(&seconds), "{seconds} s");
    assert_eq!(motion.s, 1.0);

    while motion.step(false, Some(&answer), true, DT) {}
    assert_eq!(motion.s, 0.0);
    // The buffers keep the targets, so ticking again reuses them.
    assert_eq!(motion.drawn.as_ref().unwrap().generation, 1);
}

/// A new answer at `s = 1` runs `s` down to 0 on the old targets without a
/// jump, takes the new targets only at 0, and runs back up on them. An
/// answer for no generation, as after a control changed, runs it down and
/// keeps the old targets.
#[test]
fn a_new_answer_swaps_the_targets_only_at_zero() {
    let mut motion = PreviewMotion::default();
    let (old, new) = (targets(1), targets(2));
    while motion.step(true, Some(&old), true, DT) {}
    assert_eq!(motion.s, 1.0);

    let mut last = motion.s;
    let mut swapped_at = None;
    for frame in 0..200 {
        let moving = motion.step(true, Some(&new), true, DT);
        assert!(
            (motion.s - last).abs() <= BLEND_MAX_SPEED * DT + 1e-9,
            "s jumped"
        );
        let generation = motion.drawn.as_ref().unwrap().generation;
        if generation == 2 && swapped_at.is_none() {
            assert_eq!(motion.s, 0.0, "the targets were swapped above 0");
            swapped_at = Some(frame);
        }
        last = motion.s;
        if !moving {
            break;
        }
    }
    assert!(swapped_at.is_some());
    assert_eq!(motion.s, 1.0);
    assert_eq!(motion.drawn.as_ref().unwrap().generation, 2);

    // A change of control: no answer for the new generation yet. The old one
    // is not moved to.
    while motion.step(true, None, true, DT) {}
    assert_eq!(motion.s, 0.0);
    assert_eq!(motion.drawn.as_ref().unwrap().generation, 2);
}

/// Under `animate: false` the blend is set to its goal: it stays at 0 until
/// the answer lands, takes 1 in that frame, and a new answer replaces the
/// targets at once.
#[test]
fn without_animation_the_blend_jumps_to_its_goal() {
    let mut motion = PreviewMotion::default();
    assert!(!motion.step(true, None, false, DT));
    assert_eq!(motion.s, 0.0);
    motion.step(true, Some(&targets(1)), false, DT);
    assert_eq!(motion.s, 1.0);
    motion.step(true, Some(&targets(2)), false, DT);
    assert_eq!(motion.s, 1.0);
    assert_eq!(motion.drawn.as_ref().unwrap().generation, 2);
    motion.step(false, Some(&targets(2)), false, DT);
    assert_eq!(motion.s, 0.0);
}
