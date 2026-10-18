// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Refitting every point of a reconstruction: each point's answer is the
//! bench's Fit and Commit of it, a held point is left out, a ranged one is
//! read at its distance, and the run is the same for any thread count.
//!
//! Most of these run on the synthetic textured-plane scene; one runs on the
//! seoul_bull ground truth and its photographs.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use sfmtool_sfmr_format::{
    NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED,
};

use super::reference_view::GroundTruth;
use super::scene::fixture_points;
use super::*;
use crate::progress::Event;
use crate::reconstruction::data::{
    ObservationReading, ObservationReadingOptions, ObservationReadings, ObservationSource,
    PointConstraintColumns,
};
use crate::reconstruction::triangulation::{
    retriangulate_points, triangulate_points_from_observations, ObservationSet, PointDistance,
    PointRules, RetriangulateOptions, RetriangulateWhich,
};

/// Refit options at [`TEST_SIGMA_PX`], the level the fit tests classify at.
fn refit_options() -> RefitPointsOptions {
    RefitPointsOptions {
        fit: test_fit_options(),
        ..RefitPointsOptions::default()
    }
}

/// Refit `recon` over `scene`'s photographs at [`refit_options`].
fn refit_over(
    scene: &Scene,
    recon: &SfmrReconstruction,
) -> (SfmrReconstruction, RefitPointsReport) {
    refit_points(recon, &scene.views(), &refit_options(), &Progress::none())
        .expect("the run refits")
}

/// Point `p` of `recon` as a whole record.
fn record_of(recon: &SfmrReconstruction, p: u32) -> PointRecord {
    EditedReconstruction::new(Arc::new(recon.clone()))
        .point(p)
        .expect("a live point")
        .to_record()
}

/// The keypoint of observation `k` of point `p`.
fn keypoint_of(recon: &SfmrReconstruction, p: usize, k: usize) -> [f32; 2] {
    let row = recon.point_set.observation_offsets[p] + k;
    let kp = recon.point_set.keypoints_xy().expect("inline keypoints");
    [kp[[row, 0]], kp[[row, 1]]]
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

/// Take image `image`'s pose away in the value's table: its row is a
/// placeholder, which is what an image with no pose stores.
fn unpose(recon: &mut SfmrReconstruction, image: usize) {
    let row = &mut recon.image_table.images[image];
    row.quaternion_wxyz =
        nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(f64::NAN, 0.0, 0.0, 0.0));
    row.translation_xyz = Vector3::new(f64::NAN, f64::NAN, f64::NAN);
}

/// Image `image`'s camera centre at the value's pose.
fn center_of(recon: &SfmrReconstruction, image: usize) -> Vector3<f64> {
    let c = recon.image_table.images[image].camera_center();
    Vector3::new(c.x, c.y, c.z)
}

/// The refit outcome of point `p`.
fn outcome_of(report: &RefitPointsReport, p: u32) -> &RefitOutcome {
    &report.point(p).expect("a status per point").outcome
}

/// Three points on the near plane, every one seen by all three images.
fn three_points() -> Vec<Point3<f64>> {
    vec![
        WORLD,
        Point3::new(0.15, 0.1, PLANE_Z),
        Point3::new(-0.2, 0.05, PLANE_Z),
    ]
}

/// The far scene's bearing, seen by the eight near cameras and by the ninth,
/// whose sighting sits where the far plane's own point projects: a bearing
/// whose rays now have a baseline.
fn far_bearing_with_baseline(scene: &Scene) -> SfmrReconstruction {
    let frame = OrientedPatch::from_infinity_direction(
        far_direction(),
        Vector3::new(0.0, 1.0, 0.0),
        [far_angular_half(), far_angular_half()],
    );
    let observing: Vec<u32> = (0..=FAR_OFFSET_IMAGE).collect();
    let mut recon = with_columns(
        fixture_of(scene, far_direction(), 0.0, &observing, &frame),
        BITMAP_R,
    );
    let pixel = scene.project(FAR_OFFSET_IMAGE as usize, far_world());
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut recon.point_set.observations
    else {
        panic!("the fixture is embedded_patches");
    };
    let row = FAR_OFFSET_IMAGE as usize;
    keypoints_xy[[row, 0]] = pixel[0] as f32;
    keypoints_xy[[row, 1]] = pixel[1] as f32;
    recon
}

// ---- One point's refit is the bench's ---------------------------------------

/// The options the fit reads `recon`'s point `p` with.
fn reading_options_of(
    scene: &Scene,
    recon: &SfmrReconstruction,
    p: u32,
) -> ObservationReadingOptions {
    let edited = EditedReconstruction::new(Arc::new(recon.clone()));
    let track = track_of_point(&edited, p, None).expect("a live point");
    let (fitted, _) = fit_over(scene, &edited, &track).expect("three sightings in");
    fitted.observations[0]
        .track
        .as_ref()
        .and_then(|m| m.reading_options)
        .expect("the fit reads its rows")
}

#[test]
fn a_refit_of_a_point_writes_the_record_the_bench_s_fit_and_commit_write() {
    let scene = Scene::new();
    let bare = with_columns(fixture_points(&scene, &three_points()), BITMAP_R);
    let options = reading_options_of(&scene, &bare, 0);
    // The value as it is, with no readings column, and with the column under
    // the options the fit reads with: in both, the rows the two write are
    // compared as measured rows.
    let mut with_readings = bare.clone();
    with_readings.point_set.observation_readings = Some(ObservationReadings::not_measured(
        bare.point_set.tracks.len(),
        options,
    ));

    for recon in [bare, with_readings] {
        let (out, report) = refit_over(&scene, &recon);
        assert_eq!(report.refit(), 3, "{report:?}");
        assert_eq!(report.unchanged(), 0, "{report:?}");
        assert_eq!(
            out.point_set
                .observation_readings
                .as_ref()
                .map(|r| r.options),
            Some(options),
            "the refit's column stands under the fit's options"
        );
        let edited = EditedReconstruction::new(Arc::new(recon.clone()));
        for p in 0..3u32 {
            let (bench, created) =
                create_track(&Bench::new(), &edited, p, &CreateTrackOptions::default())
                    .expect("a live point");
            let track = bench.track(&created.label).expect("just put on");
            let (fitted, _) = fit_over(&scene, &edited, track).expect("three sightings in");
            let (committed, written) = commit(&edited, &fitted).expect("the track commits");
            assert_eq!(committed.observation_reading_options(), Some(options));
            let bench_record = committed
                .point(written.point)
                .expect("the committed point")
                .to_record();
            let refit_record = record_of(&out, p);
            assert!(
                refit_record.agrees_with(&bench_record),
                "point {p}: the refit wrote {refit_record:?} and the bench {bench_record:?}"
            );
            assert!(
                refit_record
                    .observations
                    .iter()
                    .all(|o| o.reading.is_some_and(|r| r.is_measured())),
                "point {p}: every row is written as read"
            );
        }
    }
}

#[test]
fn a_refit_that_adds_the_readings_column_leaves_the_rows_it_does_not_write_not_measured() {
    let scene = Scene::new();
    let mut recon = with_columns(fixture_points(&scene, &three_points()), BITMAP_R);
    constrain(
        &mut recon,
        &[
            (POINT_CONSTRAINT_HELD, f64::NAN, NO_REFERENCE_IMAGE),
            FREE,
            FREE,
        ],
    );
    assert!(recon.point_set.observation_readings.is_none());
    let (out, report) = refit_over(&scene, &recon);
    assert_eq!(outcome_of(&report, 0), &RefitOutcome::Held);
    assert_eq!(report.refit(), 2, "{report:?}");

    // The same points committed one after another, in point order, through
    // the overlay: the column a run of commits adds.
    let mut edited = EditedReconstruction::new(Arc::new(recon.clone()));
    for p in 1..3u32 {
        let (bench, created) =
            create_track(&Bench::new(), &edited, p, &CreateTrackOptions::default())
                .expect("a live point");
        let track = bench.track(&created.label).expect("just put on");
        let (fitted, _) = fit_over(&scene, &edited, track).expect("three sightings in");
        let (next, written) = commit(&edited, &fitted).expect("the track commits");
        assert!(written.changed);
        edited = next;
    }
    let (committed, _) = edited.materialize();

    let refit = out.point_set.observation_readings.as_ref().expect("added");
    let commits = committed
        .point_set
        .observation_readings
        .as_ref()
        .expect("added");
    assert_eq!(refit.options, commits.options);
    assert_eq!(refit.rows, commits.rows);
    // The held point's rows are not measured, and every other row is.
    let offsets = &out.point_set.observation_offsets;
    for (row, reading) in refit.rows.iter().enumerate() {
        let held = row < offsets[1];
        assert_eq!(reading.is_measured(), !held, "row {row}");
        if held {
            assert_eq!(*reading, ObservationReading::NOT_MEASURED);
        }
    }
}

#[test]
fn a_refit_moves_no_index_and_keeps_every_track() {
    let scene = Scene::new();
    let recon = with_columns(fixture_points(&scene, &three_points()), BITMAP_R);
    let (out, report) = refit_over(&scene, &recon);
    assert_eq!(out.point_count(), recon.point_count());
    assert_eq!(out.point_set.tracks, recon.point_set.tracks);
    assert_eq!(
        out.point_set.observation_counts,
        recon.point_set.observation_counts
    );
    let indexes: Vec<u32> = report.points.iter().map(|p| p.index).collect();
    assert_eq!(indexes, vec![0, 1, 2]);
    // The keypoints were read again from the photographs, and stayed on the
    // content the exact projections already sat on.
    for p in 0..3 {
        for k in 0..3 {
            let (a, b) = (keypoint_of(&out, p, k), keypoint_of(&recon, p, k));
            let d = f64::from(a[0] - b[0]).hypot(f64::from(a[1] - b[1]));
            assert!(d < 0.5, "point {p} row {k} moved {d} px");
        }
    }
}

// ---- Photographs: keypoints put back on their content ----------------------

#[test]
fn keypoints_nudged_off_their_content_come_back_on_the_ground_truth() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let stored = &truth.recon;
    let offsets = &stored.point_set.observation_offsets;
    // The ground truth stores no reference observation, so each point is given
    // its first observation as one: the row the others are aligned to, which
    // is not moved. Every other row is pushed off its content.
    const NUDGE: [f32; 2] = [1.5, -1.0];
    const REFERENCE: usize = 0;
    let mut nudged = stored.clone_for_edit();
    nudged.point_set.reference_observations = Some(vec![REFERENCE as i32; stored.point_count()]);
    let mut moved_rows = Vec::new();
    {
        let ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
            &mut nudged.point_set.observations
        else {
            panic!("the ground truth is embedded_patches");
        };
        for p in 0..stored.point_count() {
            for k in 0..offsets[p + 1] - offsets[p] {
                if k == REFERENCE {
                    continue;
                }
                let row = offsets[p] + k;
                keypoints_xy[[row, 0]] += NUDGE[0];
                keypoints_xy[[row, 1]] += NUDGE[1];
                moved_rows.push(row);
            }
        }
    }
    // The points re-solved from the nudged keypoints, so they stand off the
    // ground truth too.
    let (nudged, _, _) = retriangulate_points(
        &EditedReconstruction::new(Arc::new(nudged)),
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the nudged value retriangulates");
    let nudged = (*nudged.base).clone();

    let (out, report) = refit_points(
        &nudged,
        &views,
        &RefitPointsOptions::default(),
        &Progress::none(),
    )
    .expect("the run refits");
    assert!(
        report.refit() * 10 >= report.read() * 9,
        "most points refit: {report:?}"
    );

    let row_error = |recon: &SfmrReconstruction| -> f64 {
        let truth_kp = stored.point_set.keypoints_xy().unwrap();
        let kp = recon.point_set.keypoints_xy().unwrap();
        let mut errors: Vec<f64> = moved_rows
            .iter()
            .map(|&r| {
                f64::from(kp[[r, 0]] - truth_kp[[r, 0]])
                    .hypot(f64::from(kp[[r, 1]] - truth_kp[[r, 1]]))
            })
            .collect();
        crate::numeric::median_in_place(&mut errors)
    };
    let before = row_error(&nudged);
    let after = row_error(&out);
    assert!(
        after < 0.25 * before && after < 0.5,
        "the median nudged keypoint sat {before} px off its content and {after} px after the refit"
    );

    let point_error = |recon: &SfmrReconstruction| -> f64 {
        let mut errors: Vec<f64> = (0..stored.point_count())
            .filter(|&p| !stored.point_set.points[p].is_at_infinity())
            .filter(|&p| !recon.point_set.points[p].is_at_infinity())
            .map(|p| {
                (recon.point_set.points[p].position - stored.point_set.points[p].position).norm()
            })
            .collect();
        crate::numeric::median_in_place(&mut errors)
    };
    let before = point_error(&nudged);
    let after = point_error(&out);
    assert!(
        after < 0.5 * before,
        "the median point stood {before} from the ground truth and {after} after the refit"
    );
}

// ---- Points at infinity ------------------------------------------------------

#[test]
fn a_point_at_infinity_is_refit_as_a_bearing() {
    let scene = far_scene();
    let recon = (*far_bearing_edited(&scene).base).clone();
    let (out, report) = refit_over(&scene, &recon);
    assert!(
        matches!(
            outcome_of(&report, 0),
            RefitOutcome::Refit {
                crossed: false,
                shift: None,
                range: RefitRange::Free,
                ..
            }
        ),
        "{report:?}"
    );
    let point = &out.point_set.points[0];
    assert_eq!(point.w, 0.0, "a bearing is written as a bearing");
    assert!((point.position.coords.norm() - 1.0).abs() < 1e-12);
    assert_eq!(point.normal, Vector3::zeros());
    assert_eq!(out.point_set.normal_confidence.as_deref(), Some(&[0u8][..]));
    assert_eq!(report.crossed(), 0);
}

#[test]
fn a_bearing_whose_rays_now_have_a_baseline_crosses_to_a_point() {
    let scene = far_scene();
    let recon = far_bearing_with_baseline(&scene);
    let (out, report) = refit_over(&scene, &recon);
    assert!(
        matches!(
            outcome_of(&report, 0),
            RefitOutcome::Refit {
                crossed: true,
                shift: None,
                ..
            }
        ),
        "{report:?}"
    );
    assert_eq!(report.crossed(), 1);
    let point = &out.point_set.points[0];
    assert_eq!(point.w, 1.0, "a crossing to finite writes a place");
    assert!((point.position - far_world()).norm() < 2.0, "{point:?}");
    assert_eq!(out.point_set.infinity_point_count, 0);
}

// ---- Constraints ------------------------------------------------------------

#[test]
fn a_held_point_is_left_out_and_comes_back_as_it_was() {
    let scene = Scene::new();
    let mut recon = with_columns(fixture_points(&scene, &three_points()), BITMAP_R);
    constrain(
        &mut recon,
        &[
            (POINT_CONSTRAINT_HELD, f64::NAN, NO_REFERENCE_IMAGE),
            FREE,
            FREE,
        ],
    );
    let (out, report) = refit_over(&scene, &recon);
    assert_eq!(outcome_of(&report, 0), &RefitOutcome::Held);
    assert_eq!(report.held(), 1);
    assert_eq!(report.read(), 2);
    let (held, kept) = (record_of(&recon, 0), record_of(&out, 0));
    assert!(held.agrees_with(&kept), "{held:?} became {kept:?}");
    assert!(matches!(outcome_of(&report, 1), RefitOutcome::Refit { .. }));
}

#[test]
fn a_ranged_point_is_read_at_its_distance_and_stays_ranged() {
    let scene = Scene::new();
    let base = fixture_with_columns(&scene, WORLD, BITMAP_R);
    let origin = center_of(&base, 0);
    let distance = 0.9 * (WORLD.coords - origin).norm();
    let mut ranged = base.clone();
    constrain(&mut ranged, &[(POINT_CONSTRAINT_RANGED, distance, 0)]);
    let mut free = base.clone();
    constrain(&mut free, &[FREE]);

    let (out, report) = refit_over(&scene, &ranged);
    let (free_out, _) = refit_over(&scene, &free);
    assert!(
        matches!(
            outcome_of(&report, 0),
            RefitOutcome::Refit {
                range: RefitRange::Applied,
                classification: None,
                crossed: false,
                ..
            }
        ),
        "{report:?}"
    );
    assert_eq!(report.ranged(), 1);

    let point = &out.point_set.points[0];
    assert_eq!(point.w, 1.0);
    let at = (point.position.coords - origin).norm();
    assert!(
        (at - distance).abs() < 1e-9 * distance,
        "the point stands {at} from its reference camera and is held at {distance}"
    );
    let triple = record_of(&out, 0).constraint.expect("the columns");
    assert_eq!(
        (triple.0, triple.1, triple.2),
        (POINT_CONSTRAINT_RANGED, distance, 0)
    );

    // The keypoints were moved as a free point's are: the alignment happens
    // before the coordinate is read.
    for k in 0..2 {
        assert_eq!(keypoint_of(&out, 0, k), keypoint_of(&free_out, 0, k));
    }
    // The coordinate is the distance rule's over those keypoints.
    let table = &out.image_table;
    let uv: Vec<f64> = (0..2)
        .flat_map(|k| keypoint_of(&out, 0, k).map(f64::from))
        .collect();
    let quats: Vec<f64> = table
        .images
        .iter()
        .flat_map(|i| {
            let q = i.quaternion_wxyz;
            [q.w, q.i, q.j, q.k]
        })
        .collect();
    let translations: Vec<f64> = table
        .images
        .iter()
        .flat_map(|i| {
            let t = i.translation_xyz;
            [t.x, t.y, t.z]
        })
        .collect();
    let entry = PointDistance {
        distance,
        origin: [origin.x, origin.y, origin.z],
    };
    let rule = triangulate_points_from_observations(
        &table.cameras[0],
        ObservationSet {
            uv: &uv,
            obs_image: &[0, 1],
            obs_point: &[0, 0],
            quats_wxyz: &quats,
            translations: &translations,
            n_tracks: 1,
        },
        None,
        PointRules {
            distance: Some(std::slice::from_ref(&entry)),
            ..PointRules::default()
        },
    );
    let expected = rule.xyzw[0];
    assert!(
        (point.position.coords - Vector3::new(expected[0], expected[1], expected[2])).norm() < 1e-9,
        "the refit wrote {} and the rule gives {expected:?}",
        point.position
    );
    // And the range was applied: the same point's free fit lands elsewhere.
    let free_point = &free_out.point_set.points[0];
    assert!(
        (free_point.position - point.position).norm() > 0.1,
        "free at {}, ranged at {}",
        free_point.position,
        point.position
    );
}

#[test]
fn a_point_ranged_at_infinity_stays_a_bearing_at_the_mean_of_its_rays() {
    let scene = far_scene();
    let mut recon = far_bearing_with_baseline(&scene);
    constrain(
        &mut recon,
        &[(POINT_CONSTRAINT_RANGED, f64::INFINITY, NO_REFERENCE_IMAGE)],
    );
    let (out, report) = refit_over(&scene, &recon);
    assert!(
        matches!(
            outcome_of(&report, 0),
            RefitOutcome::Refit {
                range: RefitRange::Applied,
                crossed: false,
                ..
            }
        ),
        "the rays have a baseline and the distance still holds: {report:?}"
    );
    assert_eq!(report.crossed(), 0);
    let point = &out.point_set.points[0];
    assert_eq!(point.w, 0.0);

    let table = &out.image_table;
    let mut sum = Vector3::zeros();
    let observing = out.point_set.observations_for_point(0);
    for (k, observation) in observing.iter().enumerate() {
        let image = &table.images[observation.image_index as usize];
        let [u, v] = keypoint_of(&out, 0, k);
        let d = table.cameras[image.camera_index as usize].pixel_to_ray(f64::from(u), f64::from(v));
        sum += image.quaternion_wxyz.inverse() * Vector3::new(d[0], d[1], d[2]);
    }
    let mean = (sum / observing.len() as f64).normalize();
    assert!(
        (point.position.coords - mean).norm() < 1e-9,
        "the bearing is {} and the mean of its rays {mean}",
        point.position
    );
    let triple = record_of(&out, 0).constraint.expect("the columns");
    assert_eq!(triple.0, POINT_CONSTRAINT_RANGED);
    assert_eq!(triple.1, f64::INFINITY);
    assert_eq!(triple.2, NO_REFERENCE_IMAGE);
}

#[test]
fn a_point_ranged_from_an_image_with_no_pose_is_fitted_free_and_keeps_its_range() {
    let scene = Scene::new();
    let mut recon = fixture_with_columns(&scene, WORLD, BITMAP_R);
    // Image 2 does not observe the point.
    unpose(&mut recon, 2);
    let mut free = recon.clone();
    constrain(&mut free, &[FREE]);
    constrain(&mut recon, &[(POINT_CONSTRAINT_RANGED, 3.0, 2)]);

    let (out, report) = refit_over(&scene, &recon);
    let (free_out, _) = refit_over(&scene, &free);
    assert!(
        matches!(
            outcome_of(&report, 0),
            RefitOutcome::Refit {
                range: RefitRange::Unresolved { released: false },
                classification: Some(_),
                ..
            }
        ),
        "{report:?}"
    );
    assert_eq!(report.released(), 0);
    assert_eq!(
        out.point_set.points[0].position,
        free_out.point_set.points[0].position
    );
    let triple = record_of(&out, 0).constraint.expect("the columns");
    assert_eq!(
        (triple.0, triple.1, triple.2),
        (POINT_CONSTRAINT_RANGED, 3.0, 2)
    );
}

#[test]
fn a_point_ranged_from_an_image_with_no_pose_that_becomes_a_direction_is_released() {
    let scene = far_scene();
    let mut recon = (*far_finite_edited(&scene).base).clone();
    // The ninth image does not observe the point.
    unpose(&mut recon, FAR_OFFSET_IMAGE as usize);
    constrain(
        &mut recon,
        &[(POINT_CONSTRAINT_RANGED, FAR_Z, FAR_OFFSET_IMAGE)],
    );

    let (out, report) = refit_over(&scene, &recon);
    assert!(
        matches!(
            outcome_of(&report, 0),
            RefitOutcome::Refit {
                range: RefitRange::Unresolved { released: true },
                crossed: true,
                ..
            }
        ),
        "{report:?}"
    );
    assert_eq!(report.released(), 1);
    assert_eq!(out.point_set.points[0].w, 0.0);
    let triple = record_of(&out, 0).constraint.expect("the columns");
    assert_eq!(triple.0, POINT_CONSTRAINT_FREE);
    assert!(triple.1.is_nan());
    assert_eq!(triple.2, NO_REFERENCE_IMAGE);
}

// ---- Points the fit cannot place ---------------------------------------------

#[test]
fn a_point_with_one_observation_is_left_as_it_was() {
    let scene = Scene::new();
    let patch = patch_frame_of_fixture(&scene);
    let recon = with_columns(fixture_of(&scene, WORLD, 1.0, &[0], &patch), BITMAP_R);
    let (out, report) = refit_over(&scene, &recon);
    assert_eq!(
        outcome_of(&report, 0),
        &RefitOutcome::NotFitted(FitError::TooFewObservations(1))
    );
    assert_eq!(report.kept(), 1);
    assert_eq!(
        report.with_error(|e| matches!(e, FitError::TooFewObservations(_))),
        1
    );
    assert!(record_of(&recon, 0).agrees_with(&record_of(&out, 0)));
}

#[test]
fn a_point_with_no_eligible_reference_in_a_value_with_bitmaps_is_left_as_it_was() {
    let scene = Scene::new();
    // A patch facing away from every camera: the reference-view rule finds no
    // sighting it can render, and the point stores no reference.
    let away = OrientedPatch::from_center_normal(
        WORLD,
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.12, 0.12],
    );
    let recon = with_columns(fixture_of(&scene, WORLD, 1.0, &[0, 1], &away), BITMAP_R);
    let (out, report) = refit_over(&scene, &recon);
    assert_eq!(
        outcome_of(&report, 0),
        &RefitOutcome::NoBitmap,
        "{report:?}"
    );
    assert_eq!(report.kept(), 1);
    assert!(record_of(&recon, 0).agrees_with(&record_of(&out, 0)));
}

#[test]
fn a_track_stored_out_of_image_order_is_written_back_at_its_own_rows() {
    let scene = Scene::new();
    let mut sorted = with_columns(fixture_points(&scene, &three_points()), BITMAP_R);
    // Each point's reference is its first row, the observation in image 0,
    // which the reversal moves to the last row.
    sorted.point_set.reference_observations = Some(vec![0; sorted.point_count()]);
    // The same value with point 0's rows reversed, every per-observation
    // column with them and its reference renumbered to the same observation:
    // what rows replaced by hand leave.
    let mut reversed = sorted.clone();
    let n = sorted.point_set.observation_offsets[1];
    {
        let set = &mut reversed.point_set;
        set.tracks[..n].reverse();
        if let Some(c) = set.observation_confidence.as_mut() {
            c[..n].reverse();
        }
        if let Some(r) = set.observation_readings.as_mut() {
            r.rows[..n].reverse();
        }
        let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut set.observations else {
            panic!("the fixture is embedded_patches");
        };
        for k in 0..n / 2 {
            for c in 0..2 {
                keypoints_xy.swap([k, c], [n - 1 - k, c]);
            }
        }
        let references = set.reference_observations.as_mut().expect("a reference");
        references[0] = n as i32 - 1 - references[0];
    }

    let (want, want_report) = refit_over(&scene, &sorted);
    let (got, got_report) = refit_over(&scene, &reversed);
    assert!(matches!(
        outcome_of(&got_report, 0),
        RefitOutcome::Refit { .. }
    ));
    assert_eq!(got_report.refit(), want_report.refit());
    assert_eq!(got.point_set.points, want.point_set.points);
    // Each row keeps its image, and carries the answer for that image.
    for k in 0..n {
        let r = n - 1 - k;
        assert_eq!(
            got.point_set.tracks[k].image_index,
            want.point_set.tracks[r].image_index
        );
        assert_eq!(keypoint_of(&got, 0, k), keypoint_of(&want, 0, r));
    }
    let reference = |recon: &SfmrReconstruction| {
        let r = recon.point_set.reference_observations.as_ref().unwrap()[0];
        recon.point_set.tracks[r as usize].image_index
    };
    assert_eq!(reference(&got), reference(&want));
    // The stored point the run compares its record with, to say whether it is
    // unchanged, is the reversed rows put back in record order, reference
    // and all: the sorted value's record.
    assert!(!record_of(&reversed, 0).agrees_with(&record_of(&sorted, 0)));
    assert!(
        crate::bench::refit_points::stored_in_record_order(record_of(&reversed, 0))
            .agrees_with(&record_of(&sorted, 0))
    );
}

/// The patch the scene's fixture stands its point on.
fn patch_frame_of_fixture(scene: &Scene) -> OrientedPatch {
    EditedReconstruction::new(Arc::new(fixture_with_columns(scene, WORLD, BITMAP_R)))
        .point(0)
        .and_then(|p| p.patch_frame())
        .expect("the fixture's frame")
}

// ---- The run -----------------------------------------------------------------

#[test]
fn a_cancelled_run_writes_nothing() {
    let scene = Scene::new();
    let points: Vec<Point3<f64>> = (0..6)
        .map(|i| Point3::new(-0.25 + 0.1 * i as f64, 0.05 * i as f64, PLANE_Z))
        .collect();
    let recon = with_columns(fixture_points(&scene, &points), BITMAP_R);
    let before = recon.clone();

    let stop = AtomicBool::new(true);
    let refused = refit_points(
        &recon,
        &scene.views(),
        &refit_options(),
        &Progress::none().cancelled_by(&stop),
    )
    .err()
    .expect("cancelled before it started");
    assert_eq!(refused, RefitPointsError::Cancelled);

    // Part way: the first point counted asks the run to stop. On one thread
    // the points are fitted one after another, so the check in front of each
    // point's fit leaves every point after the first unfitted, and the count
    // never passes one. Without that check every point would be fitted and
    // counted before the check after the loop refused the run.
    let stop = AtomicBool::new(false);
    let counted = AtomicU64::new(0);
    let sink = |event: Event<'_>| {
        if let Event::Count { done, .. } = event {
            counted.fetch_max(done, Ordering::Relaxed);
            if done >= 1 {
                stop.store(true, Ordering::Relaxed);
            }
        }
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool");
    let refused = pool
        .install(|| {
            refit_points(
                &recon,
                &scene.views(),
                &refit_options(),
                &Progress::to(&sink).cancelled_by(&stop),
            )
        })
        .err()
        .expect("cancelled part way");
    assert_eq!(refused, RefitPointsError::Cancelled);
    assert_eq!(
        counted.load(Ordering::Relaxed),
        1,
        "the run fitted points after it was asked to stop"
    );
    assert_eq!(
        recon.content_xxh128().expect("hashes").content_xxh128,
        before.content_xxh128().expect("hashes").content_xxh128
    );
}

#[test]
fn one_thread_and_the_default_pool_give_the_same_value() {
    let scene = Scene::new();
    let points: Vec<Point3<f64>> = (0..6)
        .map(|i| Point3::new(-0.25 + 0.1 * i as f64, 0.05 * i as f64, PLANE_Z))
        .collect();
    let recon = with_columns(fixture_points(&scene, &points), BITMAP_R);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool");
    let (one, one_report) = pool.install(|| refit_over(&scene, &recon));
    let (many, many_report) = refit_over(&scene, &recon);
    assert_eq!(
        one.content_xxh128().expect("hashes").content_xxh128,
        many.content_xxh128().expect("hashes").content_xxh128
    );
    assert_eq!(format!("{one_report:?}"), format!("{many_report:?}"));
}

#[test]
fn the_run_refuses_a_value_it_cannot_refit() {
    let scene = Scene::new();
    let views = scene.views();
    let recon = fixture_with_columns(&scene, WORLD, BITMAP_R);
    let run = |recon: &SfmrReconstruction,
               views: &[crate::patch::normal_refine::ProjectedImage<'_>],
               options: &RefitPointsOptions| {
        refit_points(recon, views, options, &Progress::none())
            .err()
            .expect("refused")
    };

    // A sift_files value.
    let sift = SfmrReconstruction::demo(4);
    assert_eq!(
        run(&sift, &[], &refit_options()),
        RefitPointsError::NoKeypoints
    );

    // No patch frames.
    let mut frameless = recon.clone();
    frameless.point_set.patch_u_halfvec_xyz = None;
    frameless.point_set.patch_v_halfvec_xyz = None;
    assert_eq!(
        run(&frameless, &views, &refit_options()),
        RefitPointsError::NoPatchFrames
    );

    // No posed image.
    let mut unposed = recon.clone();
    for image in 0..unposed.image_count() {
        unpose(&mut unposed, image);
    }
    assert_eq!(
        run(&unposed, &views, &refit_options()),
        RefitPointsError::NoPosedImage
    );

    // Too few views.
    assert_eq!(
        run(&recon, &views[..2], &refit_options()),
        RefitPointsError::ViewsMissing {
            got: 2,
            expected: 3
        }
    );

    // A track naming an image past the table, which a value whose tracks were
    // replaced by hand can carry: refused up front, even with a view for it,
    // rather than panicking where a step indexes the table by it.
    let mut corrupt = with_columns(fixture_points(&scene, &three_points()), BITMAP_R);
    let row = corrupt.point_set.observation_offsets[1] - 1;
    corrupt.point_set.tracks[row].image_index = 3;
    let mut extra = scene.views();
    extra.extend(scene.views().into_iter().take(1));
    for views in [&views[..], &extra[..]] {
        assert_eq!(
            run(&corrupt, views, &refit_options()),
            RefitPointsError::ImageOutOfRange {
                point: 0,
                image: 3,
                image_count: 3
            }
        );
    }

    // No noise level: none given, and a value with no finite point to measure
    // one from; or one given that weights nothing.
    let far = far_scene();
    let bearings = (*far_bearing_edited(&far).base).clone();
    let refused = run(&bearings, &far.views(), &RefitPointsOptions::default());
    assert!(
        matches!(&refused, RefitPointsError::NoNoiseLevel(why) if why.contains("no observation of a finite point")),
        "{refused}"
    );
    let zero = RefitPointsOptions {
        fit: FitOptions {
            sigma_px: Some(0.0),
            ..FitOptions::default()
        },
        ..RefitPointsOptions::default()
    };
    assert!(matches!(
        run(&recon, &views, &zero),
        RefitPointsError::NoNoiseLevel(_)
    ));
}
