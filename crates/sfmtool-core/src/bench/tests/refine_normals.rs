// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Refining every finite point's normal: each turned point's answer is the
//! bench's Grown Plane Normal and Commit of it, its centre and its keypoints
//! do not move, a point at infinity is skipped and counted, a point the step
//! leaves unchanged
//! or refuses keeps every column it had, held and ranged points are refined
//! and keep their constraint, and the run reports its points, can be
//! cancelled and refuses what it cannot refine.
//!
//! The scenes are the grown plane normal's: a textured plane seen from six
//! views out to 50 degrees, and a flat wall seen from the same views.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use sfmtool_sfmr_format::{
    NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED,
};

use super::grown_plane_normal::{camera, degrees, textured_plane, truth, wide_centres, HALF};
use super::scene::{fixture_points, with_columns, Scene, PLANE_Z, WORLD};
use super::*;
use crate::progress::Event;
use crate::reconstruction::data::PointConstraintColumns;

/// The bitmap side the value stores.
const BITMAP_R: usize = 8;

/// Three points on the textured plane.
fn three_points() -> Vec<Point3<f64>> {
    vec![
        WORLD,
        Point3::new(0.15, 0.1, PLANE_Z),
        Point3::new(-0.2, 0.05, PLANE_Z),
    ]
}

/// A value of `points` over `scene`, every one seen by every image at its
/// exact projection, each on a patch of half-length [`HALF`] whose normal is
/// turned `by` degrees off the plane's, with a bitmap column.
fn tilted_value(scene: &Scene, points: &[Point3<f64>], by: f64) -> SfmrReconstruction {
    let mut recon = fixture_points(scene, points);
    let turn = nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.5, 0.0)),
        by.to_radians(),
    );
    let patch = OrientedPatch::from_center_normal(
        points[0],
        turn * truth(),
        Vector3::new(0.0, 1.0, 0.0),
        [HALF, HALF],
    );
    let halfvec = |axis: Vector3<f64>, half: f64| {
        let v = axis * half;
        [v.x as f32, v.y as f32, v.z as f32]
    };
    let set = &mut recon.point_set;
    let n = points.len();
    let rows = |v: [f32; 3]| ndarray::Array2::from_shape_vec((n, 3), v.repeat(n)).expect("(P, 3)");
    set.patch_u_halfvec_xyz = Some(rows(halfvec(patch.u_axis, patch.half_extent[0])));
    set.patch_v_halfvec_xyz = Some(rows(halfvec(patch.v_axis, patch.half_extent[1])));
    let normal = patch.normal();
    for point in &mut set.points {
        point.normal = Vector3::new(normal.x as f32, normal.y as f32, normal.z as f32);
    }
    with_columns(recon, BITMAP_R)
}

/// The textured plane's scene.
fn plane_scene() -> Scene {
    Scene::looking_at(&wide_centres(), WORLD, camera(), textured_plane)
}

fn refine_over(
    scene: &Scene,
    recon: &SfmrReconstruction,
) -> (SfmrReconstruction, RefineNormalsReport) {
    refine_normals(
        recon,
        &scene.views(),
        &RefineNormalsOptions::default(),
        &Progress::none(),
    )
    .expect("the run refines")
}

/// Point `p` of `recon` as a whole record.
fn record_of(recon: &SfmrReconstruction, p: u32) -> PointRecord {
    EditedReconstruction::new(Arc::new(recon.clone()))
        .point(p)
        .expect("a live point")
        .to_record()
}

fn outcome_of(report: &RefineNormalsReport, p: u32) -> &RefineNormalsOutcome {
    &report.point(p).expect("a status per point").outcome
}

/// The outward normal point `p`'s patch frame states.
fn frame_normal(recon: &SfmrReconstruction, p: u32) -> Vector3<f64> {
    EditedReconstruction::new(Arc::new(recon.clone()))
        .point(p)
        .and_then(|point| point.patch_frame())
        .expect("a framed point")
        .normal()
}

/// Give every point of `recon` the constraint triple `triples[p]`.
fn constrain(recon: &mut SfmrReconstruction, triples: &[(u8, f64, u32)]) {
    assert_eq!(triples.len(), recon.point_count());
    recon.point_set.point_constraints = Some(PointConstraintColumns {
        point_constraints: triples.iter().map(|t| t.0).collect(),
        constraint_distances: triples.iter().map(|t| t.1).collect(),
        constraint_reference_images: triples.iter().map(|t| t.2).collect(),
    });
}

const FREE: (u8, f64, u32) = (POINT_CONSTRAINT_FREE, f64::NAN, NO_REFERENCE_IMAGE);

/// Make point `p` of `recon` a bearing along its direction from the origin,
/// with the zero normal a `w = 0` row carries.
fn to_infinity(recon: &mut SfmrReconstruction, p: usize) {
    let point = &mut recon.point_set.points[p];
    point.position = Point3::from(point.position.coords.normalize());
    point.w = 0.0;
    point.normal = Vector3::zeros();
    recon.rebuild_derived_fields();
}

// ---- A refinement of one point is the bench's --------------------------------

#[test]
fn every_finite_point_is_turned_onto_the_plane_as_the_bench_turns_and_commits_it() {
    let scene = plane_scene();
    let recon = tilted_value(&scene, &three_points(), 25.0);
    let (out, report) = refine_over(&scene, &recon);
    assert_eq!(report.read(), 3, "{report:?}");
    assert_eq!(report.turned(), 3, "{report:?}");
    assert_eq!(report.unchanged(), 0, "{report:?}");
    assert_eq!(report.at_infinity(), 0, "{report:?}");
    assert!(report.median_turn_deg() > 20.0, "{report:?}");
    // The patches turned and no keypoint moved.
    assert_eq!(out.point_set.keypoints_xy(), recon.point_set.keypoints_xy());

    // No index moves, and no observation is added or removed.
    assert_eq!(out.point_count(), recon.point_count());
    assert_eq!(out.point_set.tracks, recon.point_set.tracks);
    let edited = EditedReconstruction::new(Arc::new(recon.clone()));
    for p in 0..3u32 {
        let before = degrees(frame_normal(&recon, p), truth());
        let after = degrees(frame_normal(&out, p), truth());
        assert!(
            before > 20.0 && after < 2.0,
            "point {p}: {before} -> {after}"
        );
        // The centre does not move.
        assert_eq!(
            out.point_set.points[p as usize].position, recon.point_set.points[p as usize].position,
            "point {p}"
        );
        assert_eq!(out.point_set.points[p as usize].w, 1.0);

        // The record written is the one the bench's Grown Plane Normal and
        // Commit write for the point alone: the turned frame, the keypoints
        // as given, the bitmap it rendered and the readings it took.
        let (bench, created) =
            create_track(&Bench::new(), &edited, p, &CreateTrackOptions::default())
                .expect("a live point");
        let track = bench.track(&created.label).expect("just put on");
        let (turned, _) = grown_plane_normal(
            track,
            &edited,
            &scene.views(),
            &GrownPlaneNormalOptions::default(),
            &Progress::none(),
        )
        .expect("the step runs");
        let (committed, written) = commit(&edited, &turned).expect("the track commits");
        let bench_record = committed
            .point(written.point)
            .expect("the committed point")
            .to_record();
        let refined = record_of(&out, p);
        assert!(
            refined.agrees_with(&bench_record),
            "point {p}: the run wrote {refined:?} and the bench {bench_record:?}"
        );
        assert!(
            refined
                .patch_bitmap
                .as_ref()
                .is_some_and(|b| b.iter().any(|&v| v != 0)),
            "point {p}: the bitmap was rendered again"
        );
    }
}

/// Every keypoint but the first of each point pushed `nudge` px off its
/// content.
fn nudged(recon: &SfmrReconstruction, nudge: [f32; 2]) -> SfmrReconstruction {
    let mut out = recon.clone();
    let offsets = out.point_set.observation_offsets.clone();
    let crate::reconstruction::data::ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
        &mut out.point_set.observations
    else {
        panic!("embedded_patches");
    };
    for p in 0..recon.point_count() {
        for row in offsets[p] + 1..offsets[p + 1] {
            keypoints_xy[[row, 0]] += nudge[0];
            keypoints_xy[[row, 1]] += nudge[1];
        }
    }
    out
}

#[test]
fn keypoints_off_their_content_stay_where_they_were_given_through_two_runs() {
    // Pushed off the content by less than the step's growth recovers, so the
    // views still join and the shifts the fit solves for are not zero: the
    // step used to write those shifts into the keypoints.
    let scene = plane_scene();
    let recon = nudged(&tilted_value(&scene, &three_points(), 25.0), [0.6, -0.4]);
    let (first, report) = refine_over(&scene, &recon);
    assert_eq!(report.turned(), 3, "{report:?}");
    assert!(report.median_turn_deg() > 20.0, "{report:?}");
    assert_eq!(
        first.point_set.keypoints_xy(),
        recon.point_set.keypoints_xy()
    );

    // A second run starts from the keypoints the first was given, and moves
    // none of them either.
    let (second, again) = refine_over(&scene, &first);
    assert_eq!(again.read(), 3, "{again:?}");
    assert_eq!(
        second.point_set.keypoints_xy(),
        recon.point_set.keypoints_xy()
    );
}

// ---- Points at infinity ------------------------------------------------------

#[test]
fn a_point_at_infinity_is_skipped_counted_and_left_as_it_was() {
    let scene = plane_scene();
    let mut recon = tilted_value(&scene, &three_points(), 25.0);
    to_infinity(&mut recon, 1);
    let (out, report) = refine_over(&scene, &recon);
    assert_eq!(outcome_of(&report, 1), &RefineNormalsOutcome::AtInfinity);
    assert_eq!(report.at_infinity(), 1, "{report:?}");
    assert_eq!(report.read(), 2, "{report:?}");
    assert_eq!(report.turned(), 2, "{report:?}");
    let (stored, kept) = (record_of(&recon, 1), record_of(&out, 1));
    assert!(stored.agrees_with(&kept), "{stored:?} became {kept:?}");
    assert_eq!(out.point_set.points[1].w, 0.0);
    assert_eq!(out.point_set.infinity_point_count, 1);
}

// ---- Constraints ------------------------------------------------------------

#[test]
fn held_and_ranged_points_are_turned_and_keep_their_constraint_and_centre() {
    let scene = plane_scene();
    let mut recon = tilted_value(&scene, &three_points(), 25.0);
    let distance =
        (recon.point_set.points[2].position - recon.image_set.images[0].camera_center()).norm();
    let triples = [
        (POINT_CONSTRAINT_HELD, f64::NAN, NO_REFERENCE_IMAGE),
        FREE,
        (POINT_CONSTRAINT_RANGED, distance, 0),
    ];
    constrain(&mut recon, &triples);
    let (out, report) = refine_over(&scene, &recon);
    assert_eq!(report.turned(), 3, "{report:?}");
    let columns = out.point_set.point_constraints.as_ref().expect("kept");
    for (p, triple) in triples.iter().enumerate() {
        assert_eq!(columns.point_constraints[p], triple.0, "point {p}");
        assert_eq!(
            columns.constraint_reference_images[p], triple.2,
            "point {p}"
        );
        assert!(
            columns.constraint_distances[p] == triple.1
                || (columns.constraint_distances[p].is_nan() && triple.1.is_nan()),
            "point {p}"
        );
        assert_eq!(
            out.point_set.points[p].position, recon.point_set.points[p].position,
            "point {p}"
        );
        assert!(
            degrees(frame_normal(&out, p as u32), truth()) < 2.0,
            "point {p}"
        );
    }
}

// ---- Points the step does not turn -------------------------------------------

#[test]
fn a_flat_wall_turns_nothing_and_every_point_keeps_its_columns() {
    let scene = Scene::looking_at(&wide_centres(), WORLD, camera(), |_, _| 127.5);
    let recon = tilted_value(&scene, &three_points(), 15.0);
    let (out, report) = refine_over(&scene, &recon);
    assert_eq!(report.turned(), 0, "{report:?}");
    assert_eq!(report.none_joined(), 3, "{report:?}");
    assert_eq!(out.point_set.keypoints_xy(), recon.point_set.keypoints_xy());
    assert!(report.median_turn_deg().is_nan());
    for p in 0..3u32 {
        assert_eq!(
            outcome_of(&report, p),
            &RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::NoneJoined)
        );
        assert!(
            record_of(&recon, p).agrees_with(&record_of(&out, p)),
            "point {p}"
        );
    }
    // Nothing was written, so no readings column was added.
    assert!(out.point_set.observation_readings.is_none());
}

#[test]
fn a_point_with_one_observation_is_refused_and_left_as_it_was() {
    let scene = plane_scene();
    let mut recon = tilted_value(&scene, &three_points(), 25.0);
    // Point 0 keeps only its first observation.
    let n = scene.len();
    let keep: Vec<usize> = (0..recon.point_set.tracks.len())
        .filter(|&row| row == 0 || row >= n)
        .collect();
    let set = &mut recon.point_set;
    set.tracks = keep.iter().map(|&row| set.tracks[row]).collect();
    let crate::reconstruction::data::ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
        &mut set.observations
    else {
        panic!("embedded_patches");
    };
    *keypoints_xy = keypoints_xy.select(ndarray::Axis(0), &keep);
    set.observation_counts[0] = 1;
    set.observation_confidence = Some(vec![200; keep.len()]);
    recon.rebuild_derived_fields();

    let (out, report) = refine_over(&scene, &recon);
    assert_eq!(
        outcome_of(&report, 0),
        &RefineNormalsOutcome::NotRefined(NormalError::TooFewObservations(1))
    );
    assert_eq!(report.kept(), 1);
    assert_eq!(
        report.with_error(|e| matches!(e, NormalError::TooFewObservations(_))),
        1
    );
    assert!(record_of(&recon, 0).agrees_with(&record_of(&out, 0)));
    assert_eq!(report.turned(), 2, "{report:?}");
}

// ---- The run -----------------------------------------------------------------

#[test]
fn the_run_reports_its_points_and_stops_when_cancelled() {
    let scene = plane_scene();
    let recon = tilted_value(&scene, &three_points(), 25.0);
    let before = recon.content_xxh128().expect("hashes").content_xxh128;

    // The stages and the count of points.
    let phases = Mutex::new(Vec::new());
    let counted = AtomicU64::new(0);
    let sink = |event: Event<'_>| match event {
        Event::Enter { phase, .. } => phases.lock().unwrap().push(phase),
        Event::Count { done, total, unit } => {
            assert_eq!((total, unit), (Some(3), "points"));
            counted.fetch_max(done, Ordering::Relaxed);
        }
        _ => {}
    };
    refine_normals(
        &recon,
        &scene.views(),
        &RefineNormalsOptions::default(),
        &Progress::to(&sink),
    )
    .expect("the run refines");
    assert_eq!(*phases.lock().unwrap(), ["refine normals", "write back"]);
    assert_eq!(counted.load(Ordering::Relaxed), 3);

    // Cancelled before it started.
    let stop = AtomicBool::new(true);
    let refused = refine_normals(
        &recon,
        &scene.views(),
        &RefineNormalsOptions::default(),
        &Progress::none().cancelled_by(&stop),
    )
    .err()
    .expect("cancelled before it started");
    assert_eq!(refused, RefineNormalsError::Cancelled);

    // Part way: on one thread the first point counted asks the run to stop,
    // and no point after it is refined.
    let stop = AtomicBool::new(false);
    let counted = AtomicU64::new(0);
    let sink = |event: Event<'_>| {
        if let Event::Count { done, .. } = event {
            counted.fetch_max(done, Ordering::Relaxed);
            stop.store(true, Ordering::Relaxed);
        }
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool");
    let refused = pool
        .install(|| {
            refine_normals(
                &recon,
                &scene.views(),
                &RefineNormalsOptions::default(),
                &Progress::to(&sink).cancelled_by(&stop),
            )
        })
        .err()
        .expect("cancelled part way");
    assert_eq!(refused, RefineNormalsError::Cancelled);
    assert_eq!(counted.load(Ordering::Relaxed), 1);
    assert_eq!(
        recon.content_xxh128().expect("hashes").content_xxh128,
        before
    );
}

#[test]
fn one_thread_and_the_default_pool_give_the_same_value() {
    let scene = plane_scene();
    let recon = tilted_value(&scene, &three_points(), 25.0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool");
    let (one, one_report) = pool.install(|| refine_over(&scene, &recon));
    let (many, many_report) = refine_over(&scene, &recon);
    assert_eq!(
        one.content_xxh128().expect("hashes").content_xxh128,
        many.content_xxh128().expect("hashes").content_xxh128
    );
    assert_eq!(format!("{one_report:?}"), format!("{many_report:?}"));
}

#[test]
fn the_run_refuses_a_value_it_cannot_refine() {
    let scene = plane_scene();
    let views = scene.views();
    let recon = tilted_value(&scene, &three_points(), 25.0);
    let run = |recon: &SfmrReconstruction,
               views: &[crate::patch::normal_refine::ProjectedImage<'_>],
               options: &RefineNormalsOptions| {
        refine_normals(recon, views, options, &Progress::none())
            .err()
            .expect("refused")
    };
    let defaults = RefineNormalsOptions::default();

    let sift = SfmrReconstruction::demo(4);
    assert_eq!(run(&sift, &[], &defaults), RefineNormalsError::NoKeypoints);

    let mut without_patch_frame = recon.clone();
    without_patch_frame.point_set.patch_u_halfvec_xyz = None;
    without_patch_frame.point_set.patch_v_halfvec_xyz = None;
    assert_eq!(
        run(&without_patch_frame, &views, &defaults),
        RefineNormalsError::NoPatchFrames
    );

    let mut unposed = recon.clone();
    for row in &mut unposed.image_set.images {
        row.quaternion_wxyz = nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
            f64::NAN,
            0.0,
            0.0,
            0.0,
        ));
        row.translation_xyz = Vector3::new(f64::NAN, f64::NAN, f64::NAN);
    }
    assert_eq!(
        run(&unposed, &views, &defaults),
        RefineNormalsError::NoPosedImage
    );

    assert_eq!(
        run(&recon, &views[..2], &defaults),
        RefineNormalsError::ViewsMissing {
            got: 2,
            expected: scene.len()
        }
    );

    let bad = RefineNormalsOptions {
        normal: GrownPlaneNormalOptions {
            spot_window: 0,
            ..GrownPlaneNormalOptions::default()
        },
    };
    assert!(
        run(&recon, &views, &bad) == RefineNormalsError::BadOption("spot_window"),
        "a first window of no samples is refused"
    );
}
