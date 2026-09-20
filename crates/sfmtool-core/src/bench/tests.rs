// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench and the editable track: the labels, the verdicts, the split, and
//! what a commit writes.
//!
//! The reconstruction under the commit tests is the synthetic textured-plane
//! scene [`mod@scene`] builds -- pinhole cameras looking down world
//! `+z` at a plane, wrapped in an `embedded_patches` value whose stored
//! keypoints are the exact projections -- so what a commit should have written
//! is known to the pixel. Nothing here decodes a photograph: every step in this
//! slice is decided by what the reconstruction and the person already say.

mod scene;

use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use ndarray::Array3;

use crate::camera::CameraIntrinsics;
use crate::geometry::RigidTransform;

use crate::patch::cloud::OrientedPatch;
use crate::patch::cluster_refine::{sample_member_grid, ClusterRefineParams, MemberStatus};
use crate::patch::keypoint_localize::{localize_patch_keypoints, KeypointLocalization};
use crate::patch::keypoint_subpixel::{refine_patch_keypoints, KeypointRefinement};
use crate::progress::Progress;
use crate::reconstruction::data::Point3D;
use crate::reconstruction::edited::{EditedReconstruction, PointMap, PointRecord};
use crate::reconstruction::SfmrReconstruction;

use scene::{
    edited as edited_fixture, fixture_of, fixture_with_columns, with_columns, Scene, IMG_H, IMG_W,
    WORLD,
};

use super::*;

/// The bitmap edge the column fixture is built with.
const BITMAP_R: usize = 8;

/// The half-width, in source-image px, the hand-placed cluster tests ask for.
/// Around the size the fixture's own patch projects to (`0.12` world at depth
/// `4.0` through a focal of `160` is `4.8` px), so the template covers a piece
/// of plane the scene's texture actually varies over.
const PIXEL_SEED_RADIUS_PX: f64 = 5.0;

/// The fixture wrapped as a version, with the optional columns a full commit
/// has to fill in.
fn edited_with_columns(scene: &Scene, world: Point3<f64>) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(fixture_with_columns(scene, world, BITMAP_R)))
}

/// A bench holding the track put on from point `point`, and that track's label.
fn bench_with_point(edited: &EditedReconstruction, point: u32) -> (Bench, String) {
    let (bench, report) =
        create_track(&Bench::new(), edited, point, &CreateTrackOptions::default())
            .expect("the fixture holds this point");
    (bench, report.label)
}

/// The track called `label`, as an owned value to edit.
fn track_of(bench: &Bench, label: &str) -> EditableTrack {
    (**bench.track(label).expect("a track by that label")).clone()
}

/// `bench` with `track` in the place of the item called `label`.
fn install(bench: &Bench, label: &str, track: EditableTrack) -> Bench {
    bench
        .replace(label, BenchItem::Track(Arc::new(track)))
        .expect("a track by that label")
}

/// A cluster seed at a pixel of `image`, named as that image's stem.
fn pixel_seed(image: u32, pixel: [f64; 2]) -> ClusterSeed {
    ClusterSeed::from_pixel(image, format!("IMG_{image:04}"), pixel, 7.5)
}

// ---- Putting a point on the bench ------------------------------------------

#[test]
fn a_point_put_on_the_bench_is_a_track_with_every_observation_in() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = bench.track(&label).expect("just put on");

    assert_eq!(track.stage_kind(), StageKind::Track);
    assert_eq!(track.observations.len(), 2);
    assert_eq!(track.verdict_counts(), (2, 0, 0));
    assert!(track.observations.iter().all(|o| !o.pinned));
    assert!(track
        .observations
        .iter()
        .all(|o| o.provenance == Provenance::Origin));
    assert_eq!(track.origin.map(|o| o.point), Some(0));

    // The keypoints are the stored ones, carried across rather than refitted.
    let view = edited.point(0).expect("a live point");
    for (k, observation) in track.observations.iter().enumerate() {
        assert_eq!(observation.image, view.observations()[k].image_index);
        assert_eq!(
            observation.track.as_ref().expect("a track slot").keypoint,
            view.keypoint_xy(k)
        );
    }
    // The payload stands where the point stands.
    let payload = track.track().expect("the track stage");
    assert_eq!(payload.position, Some(view.point().position));
    assert!(payload.frame.is_some(), "the fixture stores a patch frame");
}

#[test]
fn the_stored_confidence_is_carried_as_the_leave_one_out_score() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = bench.track(&label).expect("just put on");
    for observation in &track.observations {
        let zncc = observation
            .track
            .as_ref()
            .and_then(|m| m.zncc)
            .expect("the column is carried");
        assert!((zncc - 200.0 / 255.0).abs() < 1e-9);
    }
}

#[test]
fn a_sift_files_point_is_put_on_the_bench_for_inspection() {
    // Inspecting a track is allowed everywhere; it is the commit that refuses.
    let edited = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(4)));
    let (bench, label) = bench_with_point(&edited, 0);
    let track = bench.track(&label).expect("just put on");
    assert_eq!(track.stage_kind(), StageKind::Track);
    assert_eq!(track.verdict_counts(), (2, 0, 0));
}

#[test]
fn a_point_that_is_not_live_is_refused() {
    let scene = Scene::new();
    let mut edited = edited_fixture(&scene, WORLD);
    edited.delete_point(0).expect("a live point");
    let err = create_track(&Bench::new(), &edited, 0, &CreateTrackOptions::default())
        .expect_err("point 0 was deleted");
    assert_eq!(err, CreateTrackError::NoSuchPoint(0));
}

// ---- Labels ----------------------------------------------------------------

#[test]
fn a_track_from_a_point_is_labelled_as_the_caller_asks() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (_, report) = create_track(
        &Bench::new(),
        &edited,
        0,
        &CreateTrackOptions {
            version: 3,
            label: Some("pt3d_a1b2c3d4_1207".to_string()),
        },
    )
    .expect("a live point");
    assert_eq!(report.label, "pt3d_a1b2c3d4_1207");
}

#[test]
fn a_point_with_no_id_named_is_labelled_by_the_content_it_is_a_row_of() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (_, label) = bench_with_point(&edited, 0);
    let hash = &edited
        .base_content_hash()
        .expect("an in-memory base hashes")
        .content_xxh128;
    assert_eq!(label, format!("pt3d_{}_0", &hash[..8]));
}

#[test]
fn a_cluster_from_a_pixel_is_labelled_by_its_image_and_pixel() {
    let (bench, report) =
        create_cluster(&Bench::new(), &pixel_seed(42, [142.0, 197.5])).expect("a usable seed");
    assert_eq!(report.label, "IMG_0042@142,198");
    assert_eq!(report.observation_count, 1);
    let track = bench.track(&report.label).expect("just put on");
    assert_eq!(track.stage_kind(), StageKind::Cluster);
    assert_eq!(track.verdict_counts(), (1, 0, 0));
    assert_eq!(track.cluster().expect("a cluster").reference, 0);
    assert!(
        track.cluster().expect("a cluster").template.is_none(),
        "cutting the template reads pixels, which this step does not"
    );
    assert!(track.origin.is_none());
}

#[test]
fn a_cluster_from_a_feature_is_labelled_by_its_feature_index() {
    let seed = ClusterSeed::from_feature(
        42,
        "IMG_0042",
        847,
        [142.0, 197.5],
        [[6.0, 0.0], [0.0, 6.0]],
    );
    let (bench, report) = create_cluster(&Bench::new(), &seed).expect("a usable seed");
    assert_eq!(report.label, "IMG_0042#847");
    let track = bench.track(&report.label).expect("just put on");
    assert_eq!(
        track.observations[0].provenance,
        Provenance::Descriptor { feature: 847 }
    );
}

#[test]
fn a_second_cluster_from_the_same_pixel_takes_a_suffix() {
    let seed = pixel_seed(42, [142.0, 197.5]);
    let (bench, first) = create_cluster(&Bench::new(), &seed).expect("a usable seed");
    let (bench, second) = create_cluster(&bench, &seed).expect("a usable seed");
    let (bench, third) = create_cluster(&bench, &seed).expect("a usable seed");
    assert_eq!(first.label, "IMG_0042@142,198");
    assert_eq!(second.label, "IMG_0042@142,198 (2)");
    assert_eq!(third.label, "IMG_0042@142,198 (3)");
    assert_eq!(bench.len(), 3);
    assert_eq!(
        bench.active_label(ItemKind::Track),
        Some(third.label.as_str())
    );
}

#[test]
fn a_degenerate_seed_shape_is_refused() {
    let mut seed = pixel_seed(0, [10.0, 10.0]);
    seed.shape = [[1.0, 2.0], [2.0, 4.0]];
    let err = create_cluster(&Bench::new(), &seed).expect_err("the columns are parallel");
    assert!(matches!(err, CreateClusterError::DegenerateShape(_)));
}

#[test]
fn a_rename_frees_the_old_label() {
    let seed = pixel_seed(42, [142.0, 197.5]);
    let (bench, report) = create_cluster(&Bench::new(), &seed).expect("a usable seed");
    let bench = bench
        .rename(&report.label, "bull-nose")
        .expect("a live item");
    assert!(bench.get(&report.label).is_none());
    assert!(bench.track("bull-nose").is_some());
    assert_eq!(bench.active_label(ItemKind::Track), Some("bull-nose"));
    // The old label names nothing now, so it mints again without a suffix.
    let (bench, again) = create_cluster(&bench, &seed).expect("a usable seed");
    assert_eq!(again.label, report.label);
    assert_eq!(bench.len(), 2);
}

#[test]
fn a_rename_onto_a_taken_label_is_refused() {
    let (bench, first) =
        create_cluster(&Bench::new(), &pixel_seed(1, [10.0, 10.0])).expect("a usable seed");
    let (bench, second) =
        create_cluster(&bench, &pixel_seed(2, [20.0, 20.0])).expect("a usable seed");
    let err = bench
        .rename(&second.label, &first.label)
        .expect_err("the first holds that label");
    assert_eq!(err, BenchError::LabelTaken(first.label));
}

#[test]
fn a_label_that_names_nothing_is_refused_by_name() {
    let err = Bench::new()
        .activate("bull-nose")
        .expect_err("an empty bench holds nothing");
    assert_eq!(err, BenchError::NoSuchItem("bull-nose".to_string()));
    assert_eq!(
        err.to_string(),
        "nothing on the bench is called `bull-nose`"
    );
}

// ---- The list, the activation and the sharing ------------------------------

#[test]
fn a_step_on_one_item_leaves_every_other_the_same_arc() {
    let (bench, first) =
        create_cluster(&Bench::new(), &pixel_seed(1, [10.0, 10.0])).expect("a usable seed");
    let held = Arc::clone(bench.track(&first.label).expect("just put on"));

    let (bench, second) =
        create_cluster(&bench, &pixel_seed(2, [20.0, 20.0])).expect("a usable seed");
    assert!(Arc::ptr_eq(
        &held,
        bench.track(&first.label).expect("still on")
    ));

    let bench = bench.activate(&first.label).expect("a live item");
    assert!(Arc::ptr_eq(
        &held,
        bench.track(&first.label).expect("still on")
    ));

    let bench = bench.discard(&second.label).expect("a live item");
    assert!(Arc::ptr_eq(
        &held,
        bench.track(&first.label).expect("still on")
    ));
    assert_eq!(bench.len(), 1);
}

#[test]
fn discarding_the_active_item_activates_the_one_before_it() {
    let (bench, first) =
        create_cluster(&Bench::new(), &pixel_seed(1, [10.0, 10.0])).expect("a usable seed");
    let (bench, second) =
        create_cluster(&bench, &pixel_seed(2, [20.0, 20.0])).expect("a usable seed");
    assert_eq!(
        bench.active_label(ItemKind::Track),
        Some(second.label.as_str())
    );

    let bench = bench.discard(&second.label).expect("a live item");
    assert_eq!(
        bench.active_label(ItemKind::Track),
        Some(first.label.as_str())
    );

    let bench = bench.discard(&first.label).expect("a live item");
    assert!(bench.is_empty());
    assert_eq!(bench.active_label(ItemKind::Track), None);
}

// ---- Verdicts --------------------------------------------------------------

#[test]
fn an_added_observation_joins_as_an_unruled_candidate() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (track, report) = add_observation(
        &track_of(&bench, &label),
        &ObservationSeed::at_pixel(2, [64.0, 64.0]),
    )
    .expect("a finite pixel");

    assert_eq!(report.observation, 2);
    assert_eq!(track.observations.len(), 3);
    let added = &track.observations[2];
    assert_eq!(added.verdict, Verdict::Candidate);
    assert!(!added.pinned);
    assert_eq!(added.provenance, Provenance::Pixel);
    assert_eq!(
        added.cluster.as_ref().expect("a seed").seed_position,
        [64.0, 64.0]
    );
}

#[test]
fn two_observations_in_one_image_cannot_both_be_in() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    // A second sighting in image 0, which observation 0 already holds.
    let (track, _) = add_observation(
        &track_of(&bench, &label),
        &ObservationSeed::at_pixel(0, [50.0, 50.0]),
    )
    .expect("a finite pixel");

    let err = set_verdict(&track, 2, Verdict::In).expect_err("image 0 is spoken for");
    assert_eq!(
        err,
        TrackEditError::ImageAlreadyIn {
            image: 0,
            observation: 0
        }
    );

    // Turning the first one out frees the image.
    let (track, _) = set_verdict(&track, 0, Verdict::Out).expect("a live observation");
    let (track, report) = set_verdict(&track, 2, Verdict::In).expect("image 0 is free now");
    assert!(report.changed);
    assert_eq!(track.in_observation_of_image(0), Some(2));
    assert_eq!(track.verdict_counts(), (2, 0, 1));
}

#[test]
fn a_verdict_set_by_hand_is_pinned_and_reports_what_it_moved() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (track, report) =
        set_verdict(&track_of(&bench, &label), 1, Verdict::Out).expect("a live observation");
    assert_eq!(report.was, Verdict::In);
    assert_eq!(report.is, Verdict::Out);
    assert!(report.changed);
    assert!(track.observations[1].pinned);

    // Setting the verdict it already has changes nothing but pins it.
    let (again, report) = set_verdict(&track, 1, Verdict::Out).expect("a live observation");
    assert!(!report.changed);
    assert!(again.observations[1].pinned);
}

#[test]
fn an_observation_past_the_end_is_refused() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let err = set_verdict(&track_of(&bench, &label), 9, Verdict::Out)
        .expect_err("the track holds two observations");
    assert_eq!(
        err,
        TrackEditError::NoSuchObservation {
            observation: 9,
            observation_count: 2
        }
    );
}

// ---- The threshold painting ------------------------------------------------

/// The fixture's track with the two stored observations turned into candidates
/// carrying track-stage scores.
fn scored_track(zncc: [f64; 2]) -> EditableTrack {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    for (observation, score) in track.observations.iter_mut().zip(zncc) {
        observation.verdict = Verdict::Candidate;
        let measurement = observation.track.as_mut().expect("a track slot");
        measurement.zncc = Some(score);
        measurement.seed_shift_px = Some(0.5);
        measurement.localizability = Some(0.1);
    }
    track
}

#[test]
fn the_painting_proposes_verdicts_from_the_stored_measurements() {
    let track = scored_track([0.95, 0.40]);
    let (painted, report) = apply_thresholds(&track);
    assert_eq!(painted.observations[0].verdict, Verdict::In);
    assert_eq!(painted.observations[1].verdict, Verdict::Out);
    assert_eq!(report.turned_in, 1);
    assert_eq!(report.turned_out, 1);
    // Painting is not deciding: nothing it touched is pinned.
    assert!(painted.observations.iter().all(|o| !o.pinned));
}

#[test]
fn the_painting_leaves_a_pinned_verdict_alone() {
    let track = scored_track([0.95, 0.40]);
    // The person says the low-scoring one belongs, and the high-scoring one
    // does not.
    let (track, _) = set_verdict(&track, 1, Verdict::In).expect("a live observation");
    let (track, _) = set_verdict(&track, 0, Verdict::Out).expect("a live observation");
    let (painted, report) = apply_thresholds(&track);
    assert_eq!(painted.observations[0].verdict, Verdict::Out);
    assert_eq!(painted.observations[1].verdict, Verdict::In);
    assert_eq!(report.pinned, 2);
    assert_eq!((report.turned_in, report.turned_out), (0, 0));
}

#[test]
fn the_painting_leaves_an_unmeasured_observation_where_it_is() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (track, _) = add_observation(
        &track_of(&bench, &label),
        &ObservationSeed::at_pixel(2, [64.0, 64.0]),
    )
    .expect("a finite pixel");
    let (painted, report) = apply_thresholds(&track);
    assert_eq!(painted.observations[2].verdict, Verdict::Candidate);
    assert_eq!(report.unmeasured, 3, "no observation carries a score yet");
}

#[test]
fn the_painting_gives_one_image_one_in() {
    let mut track = scored_track([0.95, 0.99]);
    // Both sightings are now of image 0, and both clear the bar.
    track.observations[1].image = 0;
    let (painted, _) = apply_thresholds(&track);
    assert_eq!(
        painted.observations[1].verdict,
        Verdict::In,
        "the better-scoring sighting takes the image"
    );
    assert_eq!(painted.observations[0].verdict, Verdict::Candidate);
    assert_eq!(painted.verdict_counts(), (1, 1, 0));
}

// ---- Splitting -------------------------------------------------------------

/// A four-observation cluster on a fresh bench, and its label.
fn four_observation_cluster() -> (Bench, String) {
    let (bench, report) =
        create_cluster(&Bench::new(), &pixel_seed(0, [10.0, 10.0])).expect("a usable seed");
    let mut track = track_of(&bench, &report.label);
    for image in 1..4u32 {
        let (next, _) = add_observation(
            &track,
            &ObservationSeed::at_pixel(image, [10.0 * f64::from(image), 20.0]),
        )
        .expect("a finite pixel");
        track = next;
    }
    (install(&bench, &report.label, track), report.label)
}

#[test]
fn a_split_takes_exactly_the_named_observations() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = four_observation_cluster();
    let images: Vec<u32> = bench
        .track(&label)
        .expect("just built")
        .observations
        .iter()
        .map(|o| o.image)
        .collect();
    assert_eq!(images, [0, 1, 2, 3]);

    let (bench, report) = split(&bench, &edited, &label, &[1, 3]).expect("two of four");
    assert_eq!(report.label, format!("{label}-split"));
    assert_eq!((report.moved, report.kept), (2, 2));

    let first: Vec<u32> = bench
        .track(&label)
        .expect("still on")
        .observations
        .iter()
        .map(|o| o.image)
        .collect();
    let second: Vec<u32> = bench
        .track(&report.label)
        .expect("just put on")
        .observations
        .iter()
        .map(|o| o.image)
        .collect();
    assert_eq!(first, [0, 2]);
    assert_eq!(second, [1, 3]);
    // The second half is a point of its own, so a commit of it creates.
    assert!(bench
        .track(&report.label)
        .expect("just put on")
        .origin
        .is_none());
}

#[test]
fn a_split_whose_reference_moved_reseats_both_halves() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = four_observation_cluster();
    // Observation 0 is the reference; move it to the second half.
    let (bench, report) = split(&bench, &edited, &label, &[0]).expect("one of four");
    let first = bench.track(&label).expect("still on");
    let second = bench.track(&report.label).expect("just put on");
    assert_eq!(first.cluster().expect("a cluster").reference, 0);
    assert_eq!(first.observations[0].image, 1);
    assert_eq!(second.cluster().expect("a cluster").reference, 0);
    assert_eq!(second.observations[0].image, 0);
}

/// The rows a person splits off are usually the ones the thresholds just turned
/// out, so a half with no `in` observation in it is the ordinary case rather
/// than a refusal.
///
/// The half is put down to the cluster stage, and a cluster needs an
/// observation to cut its template around. That reference is a seed and not a
/// judgement: it falls back past the verdicts to whichever of the split rows
/// shows the patch largest, and only a half carrying no seed at all is refused.
#[test]
fn a_split_of_rows_the_thresholds_turned_out_still_splits() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (track, _) = set_verdict(&track, 1, Verdict::Out).expect("a live row");
    assert!(
        track.observations[1].verdict == Verdict::Out && track.in_observations() == vec![0],
        "the row being split off is the rejected one"
    );
    let bench = install(&bench, &label, track);

    let (bench, report) = split(&bench, &edited, &label, &[1]).expect("the out row splits off");
    let second = bench.track(&report.label).expect("just put on");
    assert_eq!(second.stage_kind(), StageKind::Cluster);
    assert_eq!(
        second.cluster().expect("a cluster").reference,
        0,
        "the one row it has is what its template is cut around"
    );
    assert_eq!(
        second.observations[0].verdict,
        Verdict::Out,
        "the verdicts travel with the rows"
    );
    assert!(
        second.observations[0].cluster.is_some(),
        "and the row carries the seed the reference names"
    );
}

#[test]
fn a_split_of_nothing_or_of_everything_is_refused() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = four_observation_cluster();
    assert_eq!(
        split(&bench, &edited, &label, &[]).expect_err("nothing named"),
        SplitError::NoObservations
    );
    assert_eq!(
        split(&bench, &edited, &label, &[0, 1, 2, 3]).expect_err("all four named"),
        SplitError::EveryObservation(4)
    );
    assert_eq!(
        split(&bench, &edited, &label, &[0, 9]).expect_err("there is no observation 9"),
        SplitError::NoSuchObservation {
            observation: 9,
            observation_count: 4
        }
    );
    assert_eq!(bench.len(), 1, "a refused split puts nothing on");
}

// ---- The commit ------------------------------------------------------------

#[test]
fn a_commit_with_no_origin_appends_the_in_observations_keypoints() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    // A track started from a search rather than from the point has no origin.
    track.origin = None;

    let (next, report) = commit(&edited, &track).expect("two observations in, with a position");
    assert_eq!(report.replaced, None, "a track with no origin creates");
    let created = report.point;
    assert_eq!(created, edited.base_point_count() as u32);
    assert_eq!(report.map, PointMap::Created(vec![created]));
    assert_eq!(report.observation_count, 2);
    assert!(report.absorbed().is_empty());
    assert_eq!(next.point_count(), edited.point_count() + 1);
    assert!(
        next.point(0).is_some(),
        "a creation leaves the point it was read from alone"
    );

    let written = next.point(created).expect("just written");
    assert_eq!(written.point().position, WORLD);
    let stored = edited.point(0).expect("a live point");
    for k in 0..2 {
        assert_eq!(written.observations()[k].image_index, k as u32);
        assert_eq!(written.keypoint_xy(k), stored.keypoint_xy(k));
    }
    // The confidence column round-trips through the leave-one-out score.
    assert_eq!(
        written.observation_confidence().expect("the column"),
        [200, 200]
    );
    assert_eq!(
        report.label("bull"),
        "Committed track: 2 observations in bull"
    );
}

#[test]
fn a_commit_with_an_origin_replaces_the_point() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let (next, report) = commit(&edited, &track).expect("two observations in, with a position");
    assert_eq!(report.point, edited.base_point_count() as u32);
    assert_eq!(report.replaced, Some(0));
    // The map is what carries a selection on the origin onto what was written.
    assert_eq!(report.map, PointMap::Replaced(vec![(0, report.point)]));
    assert_eq!(report.map.forward(0), Some(report.point));
    assert_eq!(report.map.inverse(report.point), Some(0));
    assert!(next.point(0).is_none(), "the origin's index is spent");
    assert_eq!(next.point_count(), edited.point_count());
    assert_eq!(
        report.label("bull"),
        "Committed track: 2 observations in bull, replacing point 0"
    );

    // Re-seating the track on what was written makes a second commit a
    // replacement of the first, once the track says something else.
    let mut moved = track.with_origin(1, report.point);
    if let Stage::Track(payload) = &mut moved.stage {
        payload.position = Some(WORLD + Vector3::new(0.0, 0.0, 0.01));
    }
    let (after, second) = commit(&next, &moved).expect("still two in");
    assert!(second.changed);
    assert_eq!(second.replaced, Some(report.point));
    assert_eq!(after.point_count(), edited.point_count());
}

/// The bug a repeated press of *Commit* is: a track already seated on the point
/// it would write has nothing to write, and a commit that deleted the point and
/// re-added an identical one would mint a version and an index per press.
#[test]
fn a_commit_onto_the_point_that_already_holds_the_track_writes_nothing() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let (next, first) = commit(&edited, &track).expect("two observations in, with a position");
    assert!(first.changed);
    let settled = track.with_origin(1, first.point);

    // Ten presses of the button after the one that wrote the point leave the
    // value, the indexes and the point count exactly where the first left them.
    let mut value = next.clone();
    for _ in 0..10 {
        let (after, report) = commit(&value, &settled).expect("still two in");
        assert!(!report.changed, "a second commit wrote something");
        assert_eq!(report.point, first.point, "it named another point");
        assert_eq!(report.replaced, None, "nothing was replaced");
        assert_eq!(report.map, PointMap::Chain(Vec::new()));
        assert_eq!(report.map.forward(first.point), Some(first.point));
        assert_eq!(report.observation_count, 2);
        assert_eq!(
            report.label("bull"),
            "Committed track: no effect, point 1 of bull already holds it"
        );
        assert_eq!(after, value, "the version is not the one it was");
        value = after;
    }
    assert_eq!(value.point_count(), edited.point_count());
    assert_eq!(
        value.index_bound(),
        next.index_bound(),
        "an index was minted"
    );
}

/// What the **first** commit of an untouched point rewrites, which is why it is
/// a change and the ones after it are not: the colour, which the commit reads
/// from the consensus bitmap's centre rather than carrying the stored byte, and
/// the error, which is the mean of what an evaluation measured and so zero for
/// a track nothing has read. Every other column round-trips exactly, and after
/// the first write the two agree as well.
#[test]
fn an_untouched_point_committed_back_rewrites_its_colour_and_its_error() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let (next, report) = commit(&edited, &track).expect("two observations in, with a position");
    assert!(report.changed);
    let was = edited.point(0).expect("a live point").to_record();
    let now = next.point(report.point).expect("just written").to_record();
    // The fixture's bitmap is blank and its colour is not, so the two differ
    // here; a point whose stored colour came from its own bitmap agrees.
    assert_eq!(was.point.color, [120, 130, 140]);
    assert_eq!(now.point.color, [0, 0, 0]);
    assert_eq!(was.point.error, 0.5);
    assert_eq!(now.point.error, 0.0);
    // Everything else, column for column.
    let restated = PointRecord {
        point: Point3D {
            color: now.point.color,
            error: now.point.error,
            ..was.point.clone()
        },
        ..was
    };
    assert!(restated.agrees_with(&now));
}

/// The commit sorts its observations into image order, so a bench that holds
/// them in another order is still the track the point already carries.
#[test]
fn observations_in_another_order_are_the_same_track() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (next, first) = commit(&edited, &track).expect("two observations in, with a position");

    let mut shuffled = track.with_origin(1, first.point);
    shuffled.observations.reverse();
    let (_, report) = commit(&next, &shuffled).expect("still two in");
    assert!(
        !report.changed,
        "the rows were the same two sightings in the other order"
    );
}

/// A sighting turned out is a track the point no longer holds, so the commit
/// has something to write again.
#[test]
fn a_sighting_turned_out_after_a_commit_is_a_change() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    // Three sightings, so that turning one out still leaves two in.
    let (bench, _) = add_observation(
        &track_of(&bench, &label),
        &ObservationSeed::at_pixel(2, scene.project(2, WORLD)),
    )
    .map(|(track, report)| (install(&bench, &label, track), report))
    .expect("a finite pixel");
    let (mut track, _) = set_verdict(&track_of(&bench, &label), 2, Verdict::In)
        .expect("a live observation in an image the track does not hold");
    track.observations[2]
        .track
        .get_or_insert_with(Default::default)
        .keypoint = Some([64.0, 64.0]);

    let (next, first) = commit(&edited, &track).expect("three observations in");
    assert!(first.changed);
    let settled = track.with_origin(1, first.point);
    let (_, unchanged) = commit(&next, &settled).expect("still three in");
    assert!(!unchanged.changed, "nothing moved between the two");

    let (fewer, _) = set_verdict(&settled, 2, Verdict::Out).expect("a live observation");
    let (after, report) = commit(&next, &fewer).expect("still two in");
    assert!(report.changed, "the track lost a sighting");
    assert_eq!(report.replaced, Some(first.point));
    assert_eq!(
        after
            .point(report.point)
            .expect("just written")
            .observations()
            .len(),
        2
    );
}

/// A track with no origin creates whatever the value already holds: the same
/// landmark duplicated onto the bench commits as a second point, because the
/// question the commit asks is about *this* track's origin and not about
/// whether some row somewhere says the same thing.
#[test]
fn a_track_with_no_origin_always_writes() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut copy = track_of(&bench, &label);
    copy.origin = None;

    let (next, report) = commit(&edited, &copy).expect("two observations in, with a position");
    assert!(report.changed);
    assert_eq!(next.point_count(), edited.point_count() + 1);
    // And again: two presses of a track with no origin are two points.
    let (after, second) = commit(&next, &copy).expect("still two in");
    assert!(second.changed);
    assert_eq!(after.point_count(), edited.point_count() + 2);
}

/// An origin whose point has gone -- deleted under the track, or taken back by
/// an undo of the commit that wrote it -- names nothing, so the commit creates.
#[test]
fn a_commit_after_the_point_is_taken_back_writes_again() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (next, first) = commit(&edited, &track).expect("two observations in, with a position");
    let settled = track.with_origin(1, first.point);

    // The undo is the value before the commit, which no longer holds that
    // point at all.
    let (after, report) = commit(&edited, &settled).expect("still two in");
    assert!(report.changed, "the point the track is seated on is gone");
    assert_eq!(report.replaced, None, "there was nothing to replace");
    assert_eq!(after.point_count(), edited.point_count() + 1);
    // And a point deleted out from under a seated track is the same story.
    let mut deleted = next.clone();
    deleted.delete_point(first.point).expect("a live point");
    let (_, report) = commit(&deleted, &settled).expect("still two in");
    assert!(report.changed);
    assert_eq!(report.replaced, None);
}

/// A sighting pulled from another point is an edit even where the record is the
/// one the origin already holds: the point it was pulled from is still there to
/// absorb.
#[test]
fn a_commit_with_something_left_to_absorb_is_a_change() {
    let scene = Scene::new();
    let mut edited = edited_with_columns(&scene, WORLD);
    let other = edited
        .add_point(edited.point(0).expect("a live point").to_record())
        .expect("a well-formed record");
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (next, first) = commit(&edited, &track).expect("two observations in, with a position");

    let mut pulled = track.with_origin(1, first.point);
    pulled.observations[1].provenance = Provenance::Point { point: other };
    let (after, report) = commit(&next, &pulled).expect("still two in");
    assert!(report.changed, "the pulled-from point was still live");
    assert_eq!(report.absorbed(), [other]);
    assert!(after.point(other).is_none());

    // Once it is absorbed there is nothing left to do, and the same track
    // commits as the no-effect it now is.
    let settled = pulled.with_origin(2, report.point);
    let (_, again) = commit(&after, &settled).expect("still two in");
    assert!(!again.changed);
}

/// Which columns the comparison reads: every one the commit writes. Each of
/// these is a record the point does not hold, and each has to be written.
#[test]
fn a_column_the_commit_writes_is_a_column_it_compares() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (next, first) = commit(&edited, &track).expect("two observations in, with a position");
    let settled = track.with_origin(1, first.point);

    let moved = |column: &str, change: &dyn Fn(&mut EditableTrack)| {
        let mut track = settled.clone();
        change(&mut track);
        let (_, report) = commit(&next, &track).expect("still two in");
        assert!(
            report.changed,
            "{column} moved and the commit wrote nothing"
        );
        assert_eq!(report.replaced, Some(first.point), "{column}");
    };

    moved("the position", &|t| {
        payload_of(t).position = Some(WORLD + Vector3::new(0.0, 0.0, 1e-9));
    });
    moved("the bearing flag", &|t| payload_of(t).at_infinity = true);
    moved("the bitmap centre, which is the colour", &|t| {
        let mut bitmap = Array3::<u8>::zeros((BITMAP_R, BITMAP_R, 4));
        bitmap[[BITMAP_R / 2, BITMAP_R / 2, 1]] = 9;
        payload_of(t).bitmap = Some(bitmap);
    });
    moved("the normal confidence", &|t| {
        payload_of(t).normal_confidence = Some(7)
    });
    moved("the frame", &|t| {
        payload_of(t).frame.as_mut().expect("a surfel").half_extent[0] *= 2.0;
    });
    moved("the bitmap", &|t| {
        let mut bitmap = Array3::<u8>::zeros((BITMAP_R, BITMAP_R, 4));
        bitmap[[0, 0, 0]] = 1;
        payload_of(t).bitmap = Some(bitmap);
    });
    moved("a keypoint", &|t| {
        let measurement = t.observations[0].track.as_mut().expect("a track slot");
        let keypoint = measurement.keypoint.expect("a stored keypoint");
        measurement.keypoint = Some([keypoint[0] + 0.001, keypoint[1]]);
    });
    moved("an observation's confidence", &|t| {
        t.observations[0].track.as_mut().expect("a track slot").zncc = Some(0.5);
    });
    moved("the error", &|t| {
        t.observations[0]
            .track
            .as_mut()
            .expect("a track slot")
            .reprojection_error = Some(0.25);
    });
}

/// The track payload of a track-stage track, for a test that moves one column.
fn payload_of(track: &mut EditableTrack) -> &mut TrackPayload {
    match &mut track.stage {
        Stage::Track(payload) => payload,
        Stage::Cluster(_) => panic!("the track stage"),
    }
}

#[test]
fn an_origin_that_names_a_deleted_point_creates_instead() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let mut deleted = edited.clone();
    deleted.delete_point(0).expect("a live point");
    let (next, report) = commit(&deleted, &track).expect("the track still stands on its own");
    assert_eq!(report.replaced, None);
    assert_eq!(report.map, PointMap::Created(vec![report.point]));
    assert_eq!(next.point_count(), 1);
}

#[test]
fn a_commit_deletes_the_points_the_kept_observations_were_pulled_from() {
    let scene = Scene::new();
    let mut edited = edited_with_columns(&scene, WORLD);
    // A second point, whose sighting is then pulled into the first's track.
    let other = edited
        .add_point(edited.point(0).expect("a live point").to_record())
        .expect("a well-formed record");
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    track.observations[1].provenance = Provenance::Point { point: other };

    let (next, report) = commit(&edited, &track).expect("two observations in, with a position");
    assert_eq!(report.absorbed(), [other]);
    // A merge's map is the write and then the absorbed point's removal, so an
    // index on the absorbed point stops resolving while the origin's follows.
    assert_eq!(
        report.map,
        PointMap::Chain(vec![
            PointMap::Replaced(vec![(0, report.point)]),
            PointMap::Removed(vec![other]),
        ])
    );
    assert_eq!(report.map.forward(0), Some(report.point));
    assert_eq!(report.map.forward(other), None);
    assert!(next.point(other).is_none(), "the absorbed point is gone");
    assert_eq!(
        report.label("bull"),
        "Committed track: 2 observations in bull, replacing point 0, absorbing 1 points"
    );
}

#[test]
fn an_out_observation_pulled_from_a_point_leaves_that_point_alone() {
    let scene = Scene::new();
    let mut edited = edited_with_columns(&scene, WORLD);
    let other = edited
        .add_point(edited.point(0).expect("a live point").to_record())
        .expect("a well-formed record");
    let (bench, label) = bench_with_point(&edited, 0);
    // A third sighting, pulled from `other` and then refused.
    let (mut track, _) = add_observation(
        &track_of(&bench, &label),
        &ObservationSeed::at_pixel(2, [64.0, 64.0]),
    )
    .expect("a finite pixel");
    track.observations[2].provenance = Provenance::Point { point: other };
    track.observations[2].verdict = Verdict::Out;

    let (next, report) = commit(&edited, &track).expect("two observations in, with a position");
    assert!(report.absorbed().is_empty());
    assert!(
        next.point(other).is_some(),
        "a refused sighting absorbs nothing"
    );
}

#[test]
fn a_cluster_stage_track_refuses_to_commit() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, report) =
        create_cluster(&Bench::new(), &pixel_seed(0, [64.0, 64.0])).expect("a usable seed");
    let track = track_of(&bench, &report.label);
    let err = commit(&edited, &track).expect_err("a cluster is not a point");
    assert_eq!(err, CommitError::ClusterStage);
    assert_eq!(
        err.to_string(),
        "the track is at the cluster stage; upgrade it before committing"
    );
}

#[test]
fn a_sift_files_reconstruction_refuses_a_commit() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let sift_files = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(4)));
    let err = commit(&sift_files, &track).expect_err("a keypoint is not a feature index");
    assert_eq!(err, CommitError::NotEmbeddedPatches);
}

#[test]
fn a_track_with_one_observation_in_refuses_to_commit() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (track, _) =
        set_verdict(&track_of(&bench, &label), 1, Verdict::Out).expect("a live observation");
    let err = commit(&edited, &track).expect_err("one sighting fixes a bearing and no point");
    assert_eq!(err, CommitError::TooFewObservations(1));
    assert_eq!(
        err.to_string(),
        "1 observations are in, and a point needs two or more"
    );
}

#[test]
fn a_track_with_no_position_refuses_and_names_the_fit() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    if let Stage::Track(payload) = &mut track.stage {
        payload.position = None;
    }
    let err = commit(&edited, &track).expect_err("nothing has triangulated it");
    assert_eq!(err, CommitError::NoPosition);
    assert_eq!(
        err.to_string(),
        "the track has no position; fit it before committing"
    );
}

#[test]
fn a_track_with_no_bitmap_refuses_when_the_column_is_carried() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    if let Stage::Track(payload) = &mut track.stage {
        payload.bitmap = None;
    }
    assert_eq!(
        commit(&edited, &track).expect_err("the base stores a bitmap per point"),
        CommitError::NoBitmap
    );
}

#[test]
fn a_kept_observation_with_no_keypoint_refuses_by_name() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    track.observations[1]
        .track
        .as_mut()
        .expect("a track slot")
        .keypoint = None;
    assert_eq!(
        commit(&edited, &track).expect_err("there is no pixel to store"),
        CommitError::NoKeypoint {
            observation: 1,
            image: 1
        }
    );
}

/// The `error` column is the mean of what the last evaluation measured, not a
/// zero standing in for "a bench track has no error".
#[test]
fn the_committed_error_is_the_mean_of_the_measured_reprojections() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    for (i, expected) in track.in_observations().into_iter().zip([0.2, 0.6]) {
        track.observations[i]
            .track
            .as_mut()
            .expect("a track-stage slot")
            .reprojection_error = Some(expected);
    }
    let (next, report) = commit(&edited, &track).expect("two observations in, with a position");
    let written = next.point(report.point).expect("just written");
    assert!(
        (written.point().error - 0.4).abs() < 1e-6,
        "wrote {}",
        written.point().error
    );
}

#[test]
fn the_committed_colour_is_the_consensus_bitmap_centre() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    let mut bitmap = Array3::<u8>::zeros((BITMAP_R, BITMAP_R, 4));
    bitmap[[BITMAP_R / 2, BITMAP_R / 2, 0]] = 10;
    bitmap[[BITMAP_R / 2, BITMAP_R / 2, 1]] = 20;
    bitmap[[BITMAP_R / 2, BITMAP_R / 2, 2]] = 30;
    if let Stage::Track(payload) = &mut track.stage {
        payload.bitmap = Some(bitmap);
        payload.color = [99, 99, 99];
    }
    let (next, report) = commit(&edited, &track).expect("two observations in, with a position");
    let written = next.point(report.point).expect("just written");
    assert_eq!(written.point().color, [10, 20, 30]);
}

// ---- The evaluation --------------------------------------------------------

/// The track's own surfel, which is what an evaluation registers against.
fn frame_of(track: &EditableTrack) -> OrientedPatch {
    track
        .track()
        .expect("the track stage")
        .frame
        .clone()
        .expect("the fixture stores a patch frame")
}

/// The two kernels an evaluation chains at the track stage, called directly on
/// `scene` seeded at `seeds`: what the bench has to agree with.
fn fit_directly(
    scene: &Scene,
    frame: &OrientedPatch,
    view_set: &[u32],
    seeds: &[Option<[f64; 2]>],
) -> (KeypointLocalization, KeypointRefinement) {
    let views = scene.views();
    let options = FitOptions::default();
    let localized =
        localize_patch_keypoints(frame, &views, view_set, Some(seeds), &options.localize);
    let refined = refine_patch_keypoints(
        frame,
        &views,
        &localized.views,
        Some(
            &localized
                .keypoints
                .iter()
                .map(|&k| Some(k))
                .collect::<Vec<_>>(),
        ),
        &options.refine,
    );
    (localized, refined)
}

/// Evaluate `track` over `scene` with the default kernel parameters.
fn evaluate_over(
    scene: &Scene,
    edited: &EditedReconstruction,
    track: &EditableTrack,
) -> Result<(EditableTrack, EvaluateReport), EvaluateError> {
    evaluate(
        track,
        edited,
        &scene.views(),
        &EvaluateOptions::default(),
        &Progress::none(),
    )
}

/// Fit `track` over `scene` with the default kernel parameters.
fn fit_over(
    scene: &Scene,
    edited: &EditedReconstruction,
    track: &EditableTrack,
) -> Result<(EditableTrack, FitReport), FitError> {
    fit(
        track,
        edited,
        &scene.views(),
        &FitOptions::default(),
        &Progress::none(),
    )
}

/// Put `track` into `stage` over `scene` with the default kernel parameters.
fn stage_over(
    scene: &Scene,
    edited: &EditedReconstruction,
    track: &EditableTrack,
    stage: StageKind,
) -> Result<(EditableTrack, StageReport), StageError> {
    set_stage(
        track,
        edited,
        &scene.views(),
        stage,
        &FitOptions::default(),
        &Progress::none(),
    )
}

#[test]
fn a_track_from_a_point_fits_to_the_kernels_own_numbers() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let frame = frame_of(&track);
    let seeds: Vec<Option<[f64; 2]>> = (0..2)
        .map(|k| {
            track.observations[k]
                .track
                .as_ref()
                .and_then(|m| m.keypoint)
                .map(|p| [f64::from(p[0]), f64::from(p[1])])
        })
        .collect();

    let (measured, report) = fit_over(&scene, &edited, &track).expect("two observations in");
    assert_eq!(report.evaluate.stage, StageKind::Track);
    assert_eq!(report.placed, 2);
    assert_eq!(
        (report.evaluate.measured, report.evaluate.unmeasured),
        (2, 0)
    );

    // The same two kernels, called straight on the same frame and seeds.
    let (localized, refined) = fit_directly(&scene, &frame, &[0, 1], &seeds);
    for &image in &localized.views {
        let at = refined
            .views
            .iter()
            .position(|&v| v == image)
            .expect("the refinement kept both views");
        let expected = refined.keypoints[at];
        let m = measured.observations[image as usize]
            .track
            .as_ref()
            .expect("a track slot");
        assert_eq!(
            m.keypoint,
            Some([expected[0] as f32, expected[1] as f32]),
            "image {image}"
        );
        // The numbers beside the pixel are the reading the fit ends with, not
        // the fit's own working values: pressing Fit and then Evaluate gives
        // one account of the track and not two.
        assert!(m.zncc.expect("a score") > 0.5, "image {image}");
        assert_eq!(m.reason, None);
        assert!(m.seed_shift_px.expect("a peak") < 1.0);
        assert!(m.localizability.expect("a scored tile") > 0.0);
        assert!(m.reprojection_error.expect("a residual") < 1.0);
    }

    // The track's own point is where its sightings say it is, and the frame
    // stands there.
    let position = report.position.expect("a triangulation");
    assert!(
        (position - WORLD).norm() < 0.02,
        "the re-triangulation moved to {position}"
    );
    let payload = measured.track().expect("the track stage");
    assert_eq!(payload.position, Some(position));
    assert_eq!(
        payload.frame.as_ref().expect("a frame").center,
        position,
        "the frame follows the position"
    );
    let bitmap = payload.bitmap.as_ref().expect("a fused consensus");
    assert_eq!(bitmap.shape(), [BITMAP_R, BITMAP_R, 4]);
    assert!(
        bitmap.iter().any(|&v| v > 0),
        "the fused tile shows the plane"
    );
}

#[test]
fn an_evaluation_sets_no_verdict_and_leaves_a_pinned_one_alone() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    // A third sighting, refused by hand: an evaluation scores it like a
    // candidate and does not move it.
    let (track, added) = add_observation(
        &track,
        &ObservationSeed::at_pixel(2, scene.project(2, WORLD)),
    )
    .expect("a finite pixel");
    let (track, _) = set_verdict(&track, added.observation, Verdict::Out).expect("a live row");

    let (measured, report) = evaluate_over(&scene, &edited, &track).expect("two observations in");
    assert_eq!((report.measured, report.unmeasured), (3, 0));
    assert_eq!(measured.verdict_counts(), (2, 0, 1));
    assert!(measured.observations[2].pinned);
    assert_eq!(measured.observations[2].verdict, Verdict::Out);
    let scored = measured.observations[2]
        .track
        .as_ref()
        .expect("an out observation is scored like a candidate");
    assert!(
        scored.zncc.is_some(),
        "and the refusal stands beside its number"
    );
}

#[test]
fn a_candidate_on_the_plane_scores_and_one_nowhere_is_not_evaluated() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let (on_plane, added) = add_observation(
        &track,
        &ObservationSeed::at_pixel(2, scene.project(2, WORLD)),
    )
    .expect("a finite pixel");
    let at = added.observation;
    let (on_plane, _) = evaluate_over(&scene, &edited, &on_plane).expect("two observations in");
    let scored = on_plane.observations[at]
        .track
        .as_ref()
        .expect("a track slot");
    assert!(
        scored.zncc.expect("a score") > on_plane.thresholds.min_zncc,
        "the third camera sees the same patch: {:?}",
        scored.zncc
    );
    assert!(scored.seed_shift_px.expect("a drift") < 3.0);

    // The same gesture, pointed at nothing: the pixel is off the sensor, so no
    // round reads it -- and the row says which refusal that was rather than
    // coming back blank.
    let (nowhere, added) =
        add_observation(&track, &ObservationSeed::at_pixel(2, [10_000.0, 10_000.0]))
            .expect("a finite pixel");
    let at = added.observation;
    let (nowhere, report) = evaluate_over(&scene, &edited, &nowhere).expect("two observations in");
    assert_eq!((report.measured, report.unmeasured), (2, 1));
    let unscored = nowhere.observations[at]
        .track
        .as_ref()
        .expect("every row is written, measured or not");
    assert!(unscored.zncc.is_none(), "nothing registered there");
    assert_eq!(unscored.reason, Some(Unmeasured::OffSensor));
    assert_eq!(
        unscored.reason.expect("a reason").to_string(),
        "it sits off the photograph"
    );
}

/// An evaluation is a reading: the position, the frame and every keypoint come
/// back exactly as they went in, and what changes is what each row says about
/// itself.
#[test]
fn an_evaluation_moves_nothing_it_reads() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let (read, report) = evaluate_over(&scene, &edited, &track).expect("a framed track");
    assert_eq!((report.measured, report.unmeasured), (2, 0));
    assert_eq!(report.position, track.track().expect("the stage").position);
    assert_eq!(read.stage_kind(), StageKind::Track);
    let before = track.track().expect("the track stage");
    let after = read.track().expect("the track stage");
    assert_eq!(after.position, before.position);
    assert_eq!(after.frame, before.frame);
    assert_eq!(after.bitmap, before.bitmap);
    assert_eq!(after.condition_number, before.condition_number);
    for (was, now) in track.observations.iter().zip(&read.observations) {
        let (was, now) = (
            was.track.as_ref().expect("a track slot"),
            now.track.as_ref().expect("a track slot"),
        );
        assert_eq!(now.keypoint, was.keypoint, "a reading moves no keypoint");
        assert!(now.seed_shift_px.expect("a peak") < 1.5);
        assert!(now.projection_offset_px.expect("an offset") < 1.5);
        assert_eq!(now.reason, None);
    }
    assert_eq!(read.verdict_counts(), track.verdict_counts());
}

/// A reading has no minimum: one sighting alone is read as one sighting with
/// nothing to correlate against, which is a measurement rather than a refusal.
#[test]
fn an_evaluation_of_one_sighting_reports_it_rather_than_refusing() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (track, _) = set_verdict(&track_of(&bench, &label), 1, Verdict::Out).expect("a live row");

    assert_eq!(evaluate_preconditions(&track), Ok(()));
    let (read, report) = evaluate_over(&scene, &edited, &track).expect("a framed track");
    // The `out` row sits in an image of its own, so the two are read together
    // and both come back scored: what one sighting alone cannot do is *fit*.
    assert_eq!(report.measured + report.unmeasured, 2);
    let (alone, _) = set_verdict(&read, 1, Verdict::Candidate).expect("a live row");
    let mut alone = alone;
    alone.observations.truncate(1);
    let (alone, report) = evaluate_over(&scene, &edited, &alone).expect("a framed track");
    assert_eq!((report.measured, report.unmeasured), (0, 1));
    assert_eq!(
        alone.observations[0]
            .track
            .as_ref()
            .expect("a track slot")
            .reason,
        Some(Unmeasured::NoConsensus)
    );
}

#[test]
fn a_downgrade_then_an_upgrade_triangulates_back() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let (cluster, report) =
        stage_over(&scene, &edited, &track, StageKind::Cluster).expect("a frame to project");
    assert_eq!(cluster.stage_kind(), StageKind::Cluster);
    assert!(report.changed);
    assert_eq!(
        report.reference,
        Some(cluster.cluster().expect("a cluster").reference)
    );
    // Every observation is seeded where its keypoint was, with the shape the
    // frame projects to there.
    for (k, observation) in cluster.observations.iter().enumerate() {
        let seed = observation.cluster.as_ref().expect("a cluster slot");
        let keypoint = track.observations[k]
            .track
            .as_ref()
            .and_then(|m| m.keypoint)
            .expect("a stored keypoint");
        assert!((seed.seed_position[0] - f64::from(keypoint[0])).abs() < 1e-6);
        assert!((seed.seed_position[1] - f64::from(keypoint[1])).abs() < 1e-6);
        let det = seed.seed_shape[0][0] * seed.seed_shape[1][1]
            - seed.seed_shape[0][1] * seed.seed_shape[1][0];
        assert!(det.abs() > 0.0, "the projected shape spans an area");
        // And it is the right way round: a cluster seed is a keypoint-frame
        // shape, whose determinant is positive, where the patch-frame shape it
        // came from is negative. Sizes cannot tell the two apart, which is how
        // a mirrored seed went unnoticed.
        assert!(
            det > 0.0,
            "the seed is the surfel mirrored: {:?}",
            seed.seed_shape,
        );
        // And it spans what the surfel really covers there: the format's rule
        // states the patch's half-axes in pixels, and the seed states the same
        // footprint per keypoint-frame unit over the cluster's own radius.
        let projected = edited
            .base
            .observation_affine_shape(0, observation.image as usize, keypoint)
            .expect("the fixture's frame projects into every image");
        let expected = [
            f64::from(projected[0][0]).hypot(f64::from(projected[1][0])),
            f64::from(projected[0][1]).hypot(f64::from(projected[1][1])),
        ];
        let extents = column_extents_px(
            seed.seed_shape,
            cluster.cluster().expect("a cluster").radius,
        );
        for (got, want) in extents.iter().zip(&expected) {
            assert!(
                (got - want).abs() < 1e-6,
                "the seed spans {got} px where the frame projects to {want}",
            );
        }
        // The 3D is gone, and so is what was measured against it.
        assert!(observation.track.is_none());
    }
    assert!(cluster.track().is_none());

    let (again, report) =
        stage_over(&scene, &edited, &cluster, StageKind::Track).expect("two observations in");
    assert_eq!(again.stage_kind(), StageKind::Track);
    let position = report
        .fit
        .expect("an upgrade fits")
        .position
        .expect("a triangulation");
    assert!(
        (position - WORLD).norm() < 0.05,
        "the round trip landed at {position}"
    );
}

/// The two stages hold a shape in two chiralities, and the conversion between
/// them is one negation applied in each direction -- so a patch that goes down
/// to the cluster stage and back up is the patch it started as, the way round
/// it started.
///
/// A patch-frame shape has a **negative** determinant where it faces the
/// camera, because `v` points image-up and pixel rows count down; a
/// keypoint-frame shape, which is what a descriptor search seeds and what the
/// cluster stage rasters its template with, has a **positive** one. Seeding the
/// cluster straight from the projection handed it the surfel mirrored.
#[test]
fn a_stage_round_trip_keeps_the_patch_the_same_way_round() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let before = track
        .track()
        .expect("a track")
        .frame
        .clone()
        .expect("a frame");

    let (cluster, _) =
        stage_over(&scene, &edited, &track, StageKind::Cluster).expect("a downgrade");
    let (again, _) =
        stage_over(&scene, &edited, &cluster, StageKind::Track).expect("two observations in");
    let after = again
        .track()
        .expect("a track")
        .frame
        .clone()
        .expect("a frame");

    assert!(
        before.u_axis.dot(&after.u_axis) > 0.0,
        "the u axis came back reversed: {:?} then {:?}",
        before.u_axis,
        after.u_axis,
    );
    assert!(
        before.v_axis.dot(&after.v_axis) > 0.0,
        "the v axis came back reversed, which is the patch upside down: {:?} then {:?}",
        before.v_axis,
        after.v_axis,
    );

    // And the other direction closes too: the cluster the round trip passed
    // through is the one a second downgrade produces, sign included.
    let (down_again, _) =
        stage_over(&scene, &edited, &again, StageKind::Cluster).expect("a downgrade");
    for (k, observation) in down_again.observations.iter().enumerate() {
        let (Some(first), Some(second)) = (
            cluster.observations[k].cluster.as_ref(),
            observation.cluster.as_ref(),
        ) else {
            continue;
        };
        for r in 0..2 {
            for c in 0..2 {
                // Loose, because the upgrade re-triangulates and re-centres
                // before it reframes: what is under test is that the shape
                // comes back as itself rather than as its mirror, and a
                // percent of drift in the geometry is not that.
                let (a, b) = (first.seed_shape[r][c], second.seed_shape[r][c]);
                assert!(
                    (a - b).abs() < 0.02 * a.abs().max(1.0),
                    "observation {k} shape[{r}][{c}]: {a} then {b}",
                );
            }
        }
    }
}

/// A stage change's sentence states the stage once, whether it is read whole
/// or composed by a caller that has written the stage phrase itself.
#[test]
fn a_stage_report_states_the_stage_once() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (_, report) = stage_over(&scene, &edited, &track, StageKind::Cluster).expect("a downgrade");

    let whole = report.to_string();
    assert_eq!(whole.matches("stage").count(), 1, "{whole}");
    assert!(whole.starts_with("set to the cluster stage"), "{whole}");
    // What a caller writing its own stage phrase adds: the clause and nothing
    // that names the stage again.
    let detail = report.detail();
    assert!(!detail.contains("stage"), "{detail:?}");
    assert_eq!(format!("set to the cluster stage{detail}"), whole);
}

#[test]
fn setting_the_stage_a_track_is_already_at_changes_nothing() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (same, report) =
        stage_over(&scene, &edited, &track, StageKind::Track).expect("the stage it is at");
    assert!(!report.changed);
    assert_eq!(report.from, StageKind::Track);
    assert_eq!(report.to, StageKind::Track);
    assert_eq!(same, track);
}

/// How far each column of a cluster shape reaches, in that image's pixels: the
/// patch is `[-radius, radius]` keypoint-frame units, so a column's pixel
/// half-width is `radius` times its norm. This is the arithmetic the overlay,
/// the tile and the kernel's sampler all do.
fn column_extents_px(shape: [[f64; 2]; 2], radius: f64) -> [f64; 2] {
    [
        radius * (shape[0][0].powi(2) + shape[1][0].powi(2)).sqrt(),
        radius * (shape[0][1].powi(2) + shape[1][1].powi(2)).sqrt(),
    ]
}

/// A cluster started from a pixel is the size the gesture asked for, and stays
/// it: the seed's square spans the named half-width in pixels before anything
/// has read a photograph, the grid the kernel samples is that square, and a
/// refinement that finds the same piece of plane hands back a shape of the same
/// scale rather than one several times larger.
#[test]
fn a_pixel_cluster_spans_the_radius_it_was_asked_for() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let radius_px = PIXEL_SEED_RADIUS_PX;
    let seed = ClusterSeed::from_pixel(0, "image_0", scene.project(0, WORLD), radius_px);
    let (bench, created) = create_cluster(&Bench::new(), &seed).expect("a usable seed");
    let track = track_of(&bench, &created.label);

    let payload = track.cluster().expect("the cluster stage");
    let measurement = track.observations[0]
        .cluster
        .as_ref()
        .expect("the seed is the one observation");
    for extent in column_extents_px(measurement.seed_shape, payload.radius) {
        assert!(
            (extent - radius_px).abs() < 1e-9,
            "the seed spans {extent} px where {radius_px} was asked for",
        );
    }

    // The grid the kernel reads is that square: its samples are centred in
    // cells of `2 * radius / resolution`, so the outermost centre sits half a
    // cell inside the edge and no sample is outside it.
    let params = ClusterRefineParams {
        radius: payload.radius,
        ..ClusterRefineParams::default()
    };
    let half_cell = radius_px / f64::from(params.resolution.max(2));
    let outermost = radius_px - half_cell;
    assert!(
        sample_member_grid(
            scene.views()[0].pyramid,
            measurement.seed_position,
            measurement.seed_shape,
            &params,
        )
        .is_some(),
        "the seed's own square is inside the photograph",
    );
    assert!(
        outermost > 0.0 && outermost < radius_px,
        "the sampled grid spans up to {outermost} px, inside the {radius_px} px square",
    );

    // A second sighting of the same surface, and the round that registers them.
    let (track, added) = add_observation(
        &track,
        &ObservationSeed::at_pixel(1, scene.project(1, WORLD)),
    )
    .expect("a finite pixel");
    let (track, _) = set_verdict(&track, added.observation, Verdict::In).expect("a live row");
    let (refined, report) = evaluate_over(&scene, &edited, &track).expect("a cluster of two");
    let payload = refined.cluster().expect("the cluster stage");
    assert_eq!(payload.radius, ClusterPayload::default().radius);
    let reference = report.reference.expect("a reference was cut");
    let fitted = refined.observations[reference]
        .cluster
        .as_ref()
        .expect("a cluster slot")
        .shape
        .expect("the reference is always fitted");
    for extent in column_extents_px(fitted, payload.radius) {
        assert!(
            (extent - radius_px).abs() < 0.05 * radius_px,
            "the refined reference spans {extent} px where the seed spans {radius_px}",
        );
    }
}

#[test]
fn a_cluster_from_a_pixel_refines_upgrades_and_commits_onto_the_plane() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);

    // Two sightings of the same piece of plane, both pointed at by hand.
    let seed = ClusterSeed::from_pixel(0, "image_0", scene.project(0, WORLD), PIXEL_SEED_RADIUS_PX);
    let (bench, created) = create_cluster(&Bench::new(), &seed).expect("a usable seed");
    let track = track_of(&bench, &created.label);
    let (track, added) = add_observation(
        &track,
        &ObservationSeed::at_pixel(1, scene.project(1, WORLD)),
    )
    .expect("a finite pixel");
    let (track, _) = set_verdict(&track, added.observation, Verdict::In).expect("a live row");

    let (refined, report) = evaluate_over(&scene, &edited, &track).expect("a cluster of two");
    assert_eq!(report.stage, StageKind::Cluster);
    assert_eq!(report.measured, 2);
    let reference = report.reference.expect("a reference was cut");
    let payload = refined.cluster().expect("the cluster stage");
    assert_eq!(payload.reference, reference);
    let template = payload.template.as_ref().expect("a template was cut");
    assert_eq!(payload.radius, ClusterPayload::default().radius);
    assert!(template.samples.iter().any(|&v| v > 0.0));
    for observation in &refined.observations {
        let m = observation.cluster.as_ref().expect("a cluster slot");
        assert!(m.status.is_some());
        assert!(m.position.is_some(), "both members were fitted");
        assert!(m.localizability.expect("a scored tile") > 0.0);
    }
    assert_eq!(
        refined.observations[reference]
            .cluster
            .as_ref()
            .expect("a cluster slot")
            .status,
        Some(MemberStatus::Reference)
    );

    let (upgraded, _) =
        stage_over(&scene, &edited, &refined, StageKind::Track).expect("two observations in");
    let position = upgraded
        .track()
        .expect("the track stage")
        .position
        .expect("a triangulation");
    assert!(
        (position - WORLD).norm() < 0.1,
        "the hand-placed cluster landed at {position}"
    );
    // Only the track stage's measurements are on the observations now.
    assert!(upgraded
        .observations
        .iter()
        .all(|o| o.cluster.is_none() && o.track.is_some()));

    let (next, report) = commit(&edited, &upgraded).expect("a track with a position");
    assert_eq!(report.observation_count, 2);
    let written = next.point(report.point).expect("just written");
    assert!((written.point().position - WORLD).norm() < 0.1);
    assert_eq!(written.observations().len(), 2);
}

#[test]
fn a_split_of_a_track_stage_track_hands_back_a_cluster() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let keypoint = track_of(&bench, &label).observations[1]
        .track
        .as_ref()
        .and_then(|m| m.keypoint)
        .expect("a stored keypoint");

    let (bench, report) = split(&bench, &edited, &label, &[1]).expect("one of two");
    let first = bench.track(&label).expect("still on");
    let second = bench.track(&report.label).expect("just put on");
    assert_eq!(first.stage_kind(), StageKind::Track);
    assert_eq!(
        second.stage_kind(),
        StageKind::Cluster,
        "the half taken off is a set of patches again"
    );
    assert_eq!(second.cluster().expect("a cluster").reference, 0);
    let seed = second.observations[0]
        .cluster
        .as_ref()
        .expect("the downgrade seeded it");
    assert!((seed.seed_position[0] - f64::from(keypoint[0])).abs() < 1e-6);
    assert!(
        second.observations[0].track.is_none(),
        "the track stage's measurements went with the stage"
    );
}

// ---- The refusals ----------------------------------------------------------

#[test]
fn an_evaluation_with_fewer_views_than_images_is_refused() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let views = scene.views();
    assert_eq!(
        evaluate(
            &track,
            &edited,
            &views[..1],
            &EvaluateOptions::default(),
            &Progress::none(),
        )
        .expect_err("the reconstruction has three images"),
        EvaluateError::ViewsMissing {
            got: 1,
            expected: 3
        }
    );
}

#[test]
fn a_track_stage_fit_of_one_sighting_is_refused() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (track, _) = set_verdict(&track_of(&bench, &label), 1, Verdict::Out).expect("a live row");
    assert_eq!(
        fit_over(&scene, &edited, &track).expect_err("one sighting fixes no point"),
        FitError::TooFewObservations(1)
    );
}

/// The photograph-free half of each step's validation, asked on its own: it
/// gives the step's own answer, so a caller that asks before decoding refuses
/// in the same words the step would have.
#[test]
fn the_preconditions_are_the_steps_own_refusals() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let whole = track_of(&bench, &label);
    let (one_in, _) = set_verdict(&whole, 1, Verdict::Out).expect("a live row");

    // One sighting: neither the track-stage fit nor the upgrade to it has a
    // consensus to register against. A *reading* of the same track is not
    // refused -- it reports the sighting -- which is the whole difference
    // between the two steps.
    assert_eq!(
        fit_preconditions(&one_in).expect_err("one sighting fixes no point"),
        FitError::TooFewObservations(1)
    );
    assert_eq!(
        fit_over(&scene, &edited, &one_in).expect_err("the step agrees"),
        FitError::TooFewObservations(1)
    );
    assert_eq!(evaluate_preconditions(&one_in), Ok(()));
    let (cluster, _) =
        stage_over(&scene, &edited, &whole, StageKind::Cluster).expect("a framed track goes down");
    let (one_in_cluster, _) = set_verdict(&cluster, 1, Verdict::Out).expect("a live row");
    assert_eq!(
        set_stage_preconditions(&one_in_cluster, StageKind::Track)
            .expect_err("one sighting triangulates nothing"),
        StageError::Fit(FitError::TooFewObservations(1))
    );
    assert_eq!(
        stage_over(&scene, &edited, &one_in_cluster, StageKind::Track)
            .expect_err("the step agrees"),
        StageError::Fit(FitError::TooFewObservations(1))
    );

    // A whole track at the stage it is asked for is the change that does
    // nothing, which is not a refusal.
    assert_eq!(set_stage_preconditions(&whole, StageKind::Track), Ok(()));
    assert_eq!(evaluate_preconditions(&whole), Ok(()));
    assert_eq!(fit_preconditions(&whole), Ok(()));

    // A downgrade wants the frame it projects and the position it stands at.
    let mut frameless = whole.clone();
    if let Stage::Track(payload) = &mut frameless.stage {
        payload.frame = None;
    }
    assert_eq!(
        set_stage_preconditions(&frameless, StageKind::Cluster),
        Err(StageError::NoFrame)
    );
    let mut placeless = whole.clone();
    if let Stage::Track(payload) = &mut placeless.stage {
        payload.position = None;
    }
    assert_eq!(
        set_stage_preconditions(&placeless, StageKind::Cluster),
        Err(StageError::NoPosition)
    );
}

#[test]
fn an_observation_naming_an_image_with_no_view_is_refused() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (track, _) = add_observation(
        &track_of(&bench, &label),
        &ObservationSeed::at_pixel(9, [10.0, 10.0]),
    )
    .expect("a finite pixel");
    assert_eq!(
        evaluate_over(&scene, &edited, &track).expect_err("there is no image 9"),
        EvaluateError::NoView { image: 9 }
    );
}

/// The shape of the fatal case: an observation seeded a long way from the
/// point's projection.
///
/// The reading widens its window to reach the furthest seed and each view's
/// tile is `resolution + 4 · window` on a side, so an unbounded widening asks
/// for a tile whose cost is that distance **squared**, per view -- gigabytes
/// from one careless pixel. The bound turns that into a sentence on the row:
/// the far sighting is named and left out, and every other row is read as it
/// always was.
///
/// The fixture's images are 128 px across and its patch is about 4.8 px wide at
/// `resolution` 24, so one source-image px is about 2.5 patch-grid px: a seed 60
/// px from the projection is past the 64 grid-px bound while staying on the
/// sensor, where `OffSensor` would otherwise answer first.
#[test]
fn a_seed_far_from_the_projection_is_named_rather_than_searched_for() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    // A corner of image 2, with the point projecting near its centre.
    let projection = scene.project(2, WORLD);
    let seed = [2.0, 2.0];
    let offset = (projection[0] - seed[0]).hypot(projection[1] - seed[1]);
    assert!(
        offset > 40.0,
        "the fixture puts the projection away from the corner: {offset}"
    );
    let (track, added) =
        add_observation(&track, &ObservationSeed::at_pixel(2, seed)).expect("a finite pixel");
    let at = added.observation;

    let (read, report) = evaluate_over(&scene, &edited, &track).expect("two observations in");
    assert_eq!(
        (report.measured, report.unmeasured),
        (2, 1),
        "the far row is left out and the others are read as they were"
    );
    let row = read.observations[at]
        .track
        .as_ref()
        .expect("every row is written, measured or not");
    assert!(row.zncc.is_none(), "nothing was searched for it");
    let Some(Unmeasured::SeedTooFar {
        offset_px,
        bound_px,
    }) = row.reason
    else {
        panic!("the row names the bound it passed: {:?}", row.reason);
    };
    assert_eq!(bound_px, DEFAULT_MAX_SEED_OFFSET_PX);
    assert!(
        offset_px > bound_px,
        "the offset it names is the one that passed the bound: {offset_px}"
    );
    assert!(
        row.reason
            .expect("a reason")
            .to_string()
            .contains("beyond the 64 px bound"),
        "{}",
        row.reason.expect("a reason")
    );
    // And the row still carries what the geometry says about it, which is the
    // half of the reading that does not need a correlation.
    assert!(row.projection_offset_px.expect("a distance") > 40.0);

    // Raising the bound past the offset puts the row back in the round: the
    // bound is what decides, and nothing else about the row changed.
    let options = EvaluateOptions {
        max_seed_offset_px: offset_px * 2.0,
        ..EvaluateOptions::default()
    };
    let (wider, report) = evaluate(&track, &edited, &scene.views(), &options, &Progress::none())
        .expect("the round is small enough to run");
    assert_eq!(report.measured + report.unmeasured, 3);
    assert_ne!(
        wider.observations[at].track.as_ref().expect("a row").reason,
        row.reason,
        "past the bound is the only thing that was wrong with it"
    );
}

/// The budget is the round's, and it is checked before a byte is asked for.
#[test]
fn a_round_past_the_cache_budget_is_refused_rather_than_attempted() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let options = EvaluateOptions {
        max_cache_bytes: 1024,
        ..EvaluateOptions::default()
    };
    let refused = evaluate(&track, &edited, &scene.views(), &options, &Progress::none())
        .expect_err("two 48 px tiles do not fit in a kilobyte");
    let EvaluateError::TooLarge { bytes, budget } = refused else {
        panic!("the refusal names the budget: {refused}");
    };
    assert_eq!(budget, 1024);
    assert!(bytes > budget, "{bytes} against {budget}");
    assert!(
        refused.to_string().contains("1 KB"),
        "the sentence says both numbers in units a person holds: {refused}"
    );
    // The same track reads at the default budget, so what was refused is the
    // budget and not the track.
    assert!(evaluate_over(&scene, &edited, &track).is_ok());
}

#[test]
fn a_track_with_no_frame_has_nothing_to_register_against() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    if let Stage::Track(payload) = &mut track.stage {
        payload.frame = None;
    }
    assert_eq!(
        evaluate_over(&scene, &edited, &track).expect_err("there is no surfel"),
        EvaluateError::NoFrame
    );
    assert_eq!(
        fit_over(&scene, &edited, &track).expect_err("there is no surfel"),
        FitError::NoFrame
    );
    assert_eq!(
        stage_over(&scene, &edited, &track, StageKind::Cluster)
            .expect_err("there is no frame to project"),
        StageError::NoFrame
    );
}

// ---- Duplicating -----------------------------------------------------------

#[test]
fn a_duplicate_is_the_same_patch_with_no_origin_and_becomes_the_active_one() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    // Something measured and something ruled on, so the copy can be checked to
    // carry both: they were read against this geometry and still describe it.
    let mut track = track_of(&bench, &label);
    track.observations[0].verdict = Verdict::Out;
    track.observations[0].pinned = true;
    track.observations[1]
        .track
        .as_mut()
        .expect("a track slot")
        .seed_shift_px = Some(0.4);
    track.thresholds.min_zncc = 0.77;
    let bench = install(&bench, &label, track.clone());

    let (bench, report) = duplicate(&bench, &label).expect("the label is on the bench");
    assert_eq!(report.label, format!("{label} copy"));
    assert_eq!(report.from, label);
    assert_eq!(report.observation_count, track.observations.len());
    assert_eq!(bench.len(), 2);
    assert_eq!(
        bench.active_label(ItemKind::Track),
        Some(report.label.as_str()),
        "the copy is what the person is about to work on"
    );

    let copy = bench.track(&report.label).expect("just put on");
    assert_eq!(copy.observations, track.observations, "every sighting came");
    assert_eq!(copy.stage, track.stage, "the surfel and the bitmap came");
    assert_eq!(copy.thresholds, track.thresholds);
    assert_eq!(
        copy.origin, None,
        "a copy is a new patch, so its commit has to create rather than replace"
    );
    // And the original is exactly what it was, origin included.
    let original = bench.track(&label).expect("still on the bench");
    assert_eq!(**original, track);
    assert_eq!(original.origin.map(|o| o.point), Some(0));

    // A second duplicate of the same track takes the collision suffix.
    let (bench, again) = duplicate(&bench, &label).expect("the label is on the bench");
    assert_eq!(again.label, format!("{label} copy (2)"));
    assert_eq!(bench.len(), 3);

    assert!(matches!(
        duplicate(&bench, "nothing at all"),
        Err(DuplicateError::NoSuchTrack(_))
    ));
}

#[test]
fn a_commit_of_a_duplicate_creates_a_point_rather_than_replacing_one() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (bench, report) = duplicate(&bench, &label).expect("the label is on the bench");
    let copy = bench.track(&report.label).expect("just put on");

    let (next, committed) = commit(&edited, copy).expect("a track with a position and a frame");
    assert_eq!(committed.replaced, None, "a copy creates");
    assert_eq!(
        committed.point as usize,
        edited.point_count(),
        "it took the index after the ones that were there"
    );
    assert!(next.point(committed.point).is_some());
    // The point it was copied from is still there, which is the whole reason
    // the origin is dropped.
    assert!(next.point(0).is_some(), "the original's point was deleted");
}

// ---- Placing a sighting, and sizing and turning the patch ------------------

/// The fixture's camera swapped for one with real radial distortion, so every
/// assertion below about an edge landing under a pixel is made through a lens
/// whose forward and inverse maps are not the identity in disguise.
///
/// A straight patch edge projects to a *curve* here, which is exactly what the
/// resize has to survive: the arithmetic is in the patch's own plane, so the
/// only thing the lens is asked for is the ray under the pointer and the pixel
/// under a corner.
fn distorted(edited: &EditedReconstruction) -> EditedReconstruction {
    let mut recon = (*edited.base).clone();
    let (focal, cx, cy) = match recon.image_table.cameras[0].model {
        crate::camera::CameraModel::Pinhole {
            focal_length_x,
            principal_point_x,
            principal_point_y,
            ..
        } => (focal_length_x, principal_point_x, principal_point_y),
        _ => panic!("the fixture's camera is a pinhole"),
    };
    recon.image_table.cameras[0].model = crate::camera::CameraModel::SimpleRadial {
        focal_length: focal,
        principal_point_x: cx,
        principal_point_y: cy,
        radial_distortion_k1: 0.18,
    };
    EditedReconstruction::new(Arc::new(recon))
}

/// The camera and the pose of image `image`, as the steps read them.
fn view(edited: &EditedReconstruction, image: usize) -> (CameraIntrinsics, RigidTransform) {
    let table = &edited.base.image_table;
    let row = &table.images[image];
    let camera = table.cameras[row.camera_index as usize].clone();
    let q = row.quaternion_wxyz.quaternion();
    let pose = RigidTransform::from_wxyz_translation(
        [q.w, q.i, q.j, q.k],
        [
            row.translation_xyz.x,
            row.translation_xyz.y,
            row.translation_xyz.z,
        ],
    );
    (camera, pose)
}

/// Where a patch's `(s, t)` corner lands in a view, in that image's px.
fn corner_pixel(
    patch: &OrientedPatch,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
    s: f64,
    t: f64,
) -> [f64; 2] {
    let (xyz, w) = patch.corner_homogeneous(s, t);
    let pc = pose.transform_point_homogeneous(xyz, w);
    let (u, v) = camera
        .ray_to_pixel([pc.x, pc.y, pc.z])
        .expect("the fixture's patch is in front of every camera");
    [u, v]
}

/// Each sighting's offset from where the centre projects, in its own image's
/// pixels: the gap between where that photograph sees the patch's content and
/// where the geometry puts the patch's middle.
///
/// **What a move of the patch must not disturb.** It is what the tile is cut on
/// and the correlation is scored at, so a step that reset every keypoint to the
/// centre's projection would zero all of them and scramble the next reading.
fn projection_offsets(track: &EditableTrack, edited: &EditedReconstruction) -> Vec<[f64; 2]> {
    track
        .observations
        .iter()
        .map(|observation| {
            let (camera, pose) = view(edited, observation.image as usize);
            let frame = track
                .track()
                .and_then(|payload| payload.frame.as_ref())
                .expect("a frame");
            let centre = corner_pixel(frame, &camera, &pose, 0.0, 0.0);
            let site = observation.site().expect("a sighting");
            [site[0] - centre[0], site[1] - centre[1]]
        })
        .collect()
}

/// The outline a person sees at `observation`: the surfel re-anchored on that
/// sighting, which is the frame both the layer and the resize read.
fn outline_of(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    observation: usize,
) -> OrientedPatch {
    let sighting = &track.observations[observation];
    let (camera, pose) = view(edited, sighting.image as usize);
    track
        .track()
        .and_then(|payload| payload.frame.as_ref())
        .expect("a track from a point carries the stored patch")
        .anchored_at_keypoint(&camera, &pose, sighting.site().expect("a sighting"))
        .expect("the fixture's keypoint is the exact projection")
}

#[test]
fn moving_a_sighting_writes_its_keypoint_pins_it_and_drops_what_was_read_at_the_old_one() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    // A measurement that was read at the old keypoint, which the move must not
    // keep: it says nothing about a pixel three across.
    track.observations[1]
        .track
        .as_mut()
        .expect("a track slot")
        .seed_shift_px = Some(0.2);

    let was = track.observations[1].site().expect("a sighting");
    let pixel = [was[0] + 3.0, was[1] - 4.0];
    let (next, report) =
        set_observation_keypoint(&track, &edited, 1, pixel).expect("a pixel on the sensor");

    assert!(report.changed);
    assert_eq!(report.observation, 1);
    assert_eq!(report.pixel, pixel);
    assert_eq!(report.was, Some(was));
    assert!((report.moved_px.expect("it moved") - 5.0).abs() < 1e-9);

    let moved = &next.observations[1];
    assert_eq!(moved.site(), Some(pixel));
    assert!(
        moved.pinned,
        "a sighting a person placed is one they ruled on"
    );
    let measurement = moved.track.as_ref().expect("a track slot");
    assert_eq!(measurement.zncc, None);
    assert_eq!(measurement.seed_shift_px, None);
    // Nothing else moved: the other sighting, the surfel and the position stand.
    assert_eq!(next.observations[0], track.observations[0]);
    assert_eq!(
        next.track().map(|p| p.position),
        track.track().map(|p| p.position)
    );
    assert_eq!(
        next.track().and_then(|p| p.frame.clone()),
        track.track().and_then(|p| p.frame.clone())
    );

    // Put back exactly where it was, nothing changed -- but the pin stands,
    // which is what a hand placement is.
    let (again, report) =
        set_observation_keypoint(&next, &edited, 1, pixel).expect("the same pixel");
    assert!(!report.changed);
    assert!(again.observations[1].pinned);
}

#[test]
fn moving_a_cluster_sighting_moves_its_seed_and_keeps_the_shape_it_is_read_at() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let where_at = scene.project(0, WORLD);
    let (bench, report) =
        create_cluster(&Bench::new(), &pixel_seed(0, where_at)).expect("a usable seed");
    let mut track = track_of(&bench, &report.label);
    // A refinement to be dropped, and a refined shape to be kept.
    let refined = [[9.0, 1.0], [-1.0, 9.0]];
    let measurement = track.observations[0].cluster.as_mut().expect("a seed");
    measurement.shape = Some(refined);
    measurement.zncc = Some(0.91);
    measurement.shift_px = Some(0.3);

    let pixel = [where_at[0] + 2.5, where_at[1] + 1.5];
    let (next, report) =
        set_observation_keypoint(&track, &edited, 0, pixel).expect("a pixel on the sensor");

    assert!(report.changed);
    let cluster = next.observations[0].cluster.as_ref().expect("a seed");
    assert_eq!(cluster.seed_position, pixel);
    assert_eq!(
        cluster.seed_shape, refined,
        "the shape it is read at is kept"
    );
    assert_eq!(cluster.shape, None);
    assert_eq!(cluster.zncc, None);
    assert_eq!(cluster.shift_px, None);
    assert!(next.observations[0].pinned);
}

/// Exact under a pinhole, and within the lens's own inverse-map tolerance under
/// distortion.
///
/// The resize itself is arithmetic in the patch's plane and has no error of its
/// own; what the second bound admits is the iteration `pixel_to_ray` runs to
/// invert the radial term, which is why the two are run with different
/// tolerances rather than with one loose enough for both.
///
/// The assertions are about the **frame** the step wrote, not about the outline
/// redrawn from the keypoint it wrote beside it: a keypoint is an `f32` slot, so
/// re-anchoring on it is within a rounding of the centre and not on it. That
/// rounding is what the looser bound on the dot below allows for.
#[test]
fn a_resize_from_an_edge_lands_that_edge_under_the_pixel_and_leaves_the_far_one_alone() {
    let scene = Scene::new();
    let pinhole = edited_fixture(&scene, WORLD);
    resize_edge_case(&pinhole, 1e-9);
    resize_edge_case(&distorted(&pinhole), 1e-5);
}

fn resize_edge_case(edited: &EditedReconstruction, tolerance: f64) {
    let (bench, label) = bench_with_point(edited, 0);
    let track = track_of(&bench, &label);

    let sighting = track.observations[1].image as usize;
    let (camera, pose) = view(edited, sighting);
    let before = outline_of(&track, edited, 1);
    let unanchored = track
        .track()
        .and_then(|payload| payload.frame.clone())
        .expect("a frame");
    // The far edge's midpoint, which the resize promises not to move.
    let far_before = corner_pixel(&before, &camera, &pose, -1.0, 0.0);
    // Aim well outside the outline, along the `+u` edge's own direction.
    let out = corner_pixel(&before, &camera, &pose, 2.3, 0.0);

    let offsets = projection_offsets(&track, edited);
    let (next, report) =
        resize_from_edge(&track, edited, 1, Edge::PlusU, out).expect("a pixel the ray reaches");
    assert!(report.changed);
    assert_eq!(report.observation, Some(1));

    let after = next.track().and_then(|p| p.frame.clone()).expect("a frame");
    assert_eq!(
        after.half_extent[0], after.half_extent[1],
        "a patch frame is square, so a resize is one scale"
    );
    // The patch moved along the dragged axis alone, by the amount that holds
    // the far edge: `h' - h`.
    let moved = after.center - unanchored.center;
    assert!(
        (moved - unanchored.u_axis * (report.half - report.was)).norm() < 1e-12,
        "the surfel moved by {moved:?} rather than along +u by {}",
        report.half - report.was,
    );

    // The claim, stated on the **exact** frame the step's own numbers describe:
    // the outline is the surfel re-anchored on the dragged sighting, and that
    // sighting's plane point moved by the same displacement, so this is what is
    // drawn -- without the `f32` keypoint slot standing between the arithmetic
    // and the assertion.
    let drawn = OrientedPatch {
        center: before.center + moved,
        half_extent: after.half_extent,
        ..before.clone()
    };
    let dragged = corner_pixel(&drawn, &camera, &pose, 1.0, 0.0);
    assert!(
        (dragged[0] - out[0]).abs() < tolerance && (dragged[1] - out[1]).abs() < tolerance,
        "the dragged edge should land on {out:?}, it landed on {dragged:?}",
    );
    let far_after = corner_pixel(&drawn, &camera, &pose, -1.0, 0.0);
    assert!(
        (far_after[0] - far_before[0]).abs() < tolerance
            && (far_after[1] - far_before[1]).abs() < tolerance,
        "the far edge moved from {far_before:?} to {far_after:?}",
    );
    // And on the outline as it is really redrawn, through that slot: the same,
    // to a thousandth of a pixel.
    let redrawn = outline_of(&next, edited, 1);
    for (s, name) in [(1.0, "dragged"), (-1.0, "far")] {
        let want = if s > 0.0 { out } else { far_before };
        let got = corner_pixel(&redrawn, &camera, &pose, s, 0.0);
        assert!(
            (got[0] - want[0]).abs() < 1e-3 && (got[1] - want[1]).abs() < 1e-3,
            "the redrawn {name} edge should be at {want:?}, it is at {got:?}",
        );
    }

    // **Every sighting keeps its own offset from the centre's projection**,
    // which is what the tiles are cut on: the keypoints were carried along the
    // plane rather than reset to the centre. And nothing is pinned, because
    // where the patch is says nothing about whether a sighting belongs to it.
    for (offset, now) in offsets.iter().zip(projection_offsets(&next, edited)) {
        assert!(
            (now[0] - offset[0]).abs() < 1e-3 && (now[1] - offset[1]).abs() < 1e-3,
            "a resize scrambled a sighting's offset: {offset:?} became {now:?}",
        );
    }
    assert!(next.observations.iter().all(|o| !o.pinned));
    assert_eq!(next.track().and_then(|p| p.bitmap.clone()), None);
}

/// Sliding the patch is exact in the image it was dragged in, in-plane, and
/// felt by every sighting.
///
/// Run under a pinhole and under a distorting lens, because "the dot lands
/// under the pointer" is a claim about the projection: the arithmetic is in the
/// patch's own plane, so the only error there is is the iteration the lens
/// inverts its radial term with.
#[test]
fn sliding_the_patch_lands_the_dot_under_the_pointer_and_moves_every_sighting() {
    let scene = Scene::new();
    let pinhole = edited_fixture(&scene, WORLD);
    translate_case(&pinhole, 1e-9);
    translate_case(&distorted(&pinhole), 1e-5);
}

fn translate_case(edited: &EditedReconstruction, tolerance: f64) {
    let (bench, label) = bench_with_point(edited, 0);
    let mut track = track_of(&bench, &label);
    // A measurement to be dropped, and a pin the slide must not invent.
    track.observations[1]
        .track
        .as_mut()
        .expect("a track slot")
        .zncc = Some(0.9);

    let dragged = 1;
    let image = track.observations[dragged].image as usize;
    let (camera, pose) = view(edited, image);
    let was = track
        .track()
        .and_then(|payload| payload.frame.clone())
        .expect("a frame");
    let outline = outline_of(&track, edited, dragged);
    // Aim at a place on the outline that is not its centre, so the slide is a
    // real move: the `(0.7, 0.4)` point of the square.
    let target = corner_pixel(&outline, &camera, &pose, 0.7, 0.4);

    let offsets = projection_offsets(&track, edited);
    let (next, report) =
        translate_frame(&track, edited, dragged, target).expect("a pixel the ray reaches");
    assert!(report.changed);
    assert_eq!(report.observation, dragged);
    assert_eq!(report.image, track.observations[dragged].image);
    assert_eq!(report.placed, next.observations.len());

    let frame = next
        .track()
        .and_then(|payload| payload.frame.clone())
        .expect("a frame");
    // In-plane only: the normal, the axes and the size are untouched, and the
    // offset lies in the plane it moved across.
    assert!((frame.normal() - was.normal()).norm() < 1e-12);
    assert_eq!(frame.half_extent, was.half_extent);
    assert_eq!(frame.u_axis, was.u_axis);
    assert_eq!(frame.v_axis, was.v_axis);
    let offset = frame.center - was.center;
    assert!(
        offset.dot(&was.normal()).abs() < 1e-12,
        "the patch left its own plane: {offset:?}"
    );
    assert!((report.moved - offset.norm()).abs() < 1e-12);
    assert_eq!(offset, displacement_of(&track, &next));
    assert_eq!(next.track().and_then(|p| p.position), Some(frame.center));
    assert_eq!(next.track().and_then(|p| p.bitmap.clone()), None);

    // **The dot the drag came through lands under the pointer.** Its own plane
    // point plus the displacement is, by construction, the plane point under
    // the pixel -- so this holds without the centre going anywhere near it,
    // which is the whole point of carrying the offsets.
    let landed = next.observations[dragged].site().expect("a sighting");
    assert!(
        (landed[0] - target[0]).abs() < 1e-3 && (landed[1] - target[1]).abs() < 1e-3,
        "the dot should land on {target:?}, it landed on {landed:?}",
    );
    assert_eq!(report.pixel, landed);
    // Exactly, before the `f32` keypoint slot rounds it.
    let exact = {
        let moved = OrientedPatch {
            center: outline.center + displacement_of(&track, &next),
            ..outline.clone()
        };
        corner_pixel(&moved, &camera, &pose, 0.0, 0.0)
    };
    assert!(
        (exact[0] - target[0]).abs() < tolerance && (exact[1] - target[1]).abs() < tolerance,
        "the dragged sighting's plane point should project to {target:?}, it projects to {exact:?}",
    );

    // **Every sighting keeps its own offset from the centre's projection.**
    // That offset is where the photograph sees the patch's content against
    // where the geometry puts its middle, and it is what the tile is cut on: a
    // step that reset the keypoints to the centre would zero every one of them
    // and scramble the correlation.
    for (offset, now) in offsets.iter().zip(projection_offsets(&next, edited)) {
        assert!(
            (now[0] - offset[0]).abs() < 1e-3 && (now[1] - offset[1]).abs() < 1e-3,
            "a slide scrambled a sighting's offset: {offset:?} became {now:?}",
        );
    }
    for observation in &next.observations {
        assert!(!observation.pinned, "a translation is not a verdict");
        let measurement = observation.track.as_ref().expect("a track slot");
        assert_eq!(measurement.zncc, None);
        assert_eq!(measurement.reason, None);
    }

    // A slide to where the dragged sighting already sits changes nothing.
    let (again, report) = translate_frame(&next, edited, dragged, landed).expect("the same place");
    assert!(report.moved < 1e-3, "a slide to where it is moved it");
    let _ = again;

    // And it is the track stage's step.
    let where_at = scene_pixel(edited);
    let (bench, made) =
        create_cluster(&Bench::new(), &pixel_seed(0, where_at)).expect("a usable seed");
    assert!(matches!(
        translate_frame(&track_of(&bench, &made.label), edited, 0, where_at),
        Err(TrackEditError::WrongStage { .. })
    ));
}

/// How far the surfel moved between two versions of a track.
fn displacement_of(before: &EditableTrack, after: &EditableTrack) -> Vector3<f64> {
    let centre = |track: &EditableTrack| {
        track
            .track()
            .and_then(|payload| payload.frame.as_ref())
            .expect("a frame")
            .center
    };
    centre(after) - centre(before)
}

/// Somewhere on the sensor of image 0, for a step that has to be refused rather
/// than measured.
fn scene_pixel(edited: &EditedReconstruction) -> [f64; 2] {
    let camera = &edited.base.image_table.cameras[0];
    [camera.width as f64 / 2.0, camera.height as f64 / 2.0]
}

#[test]
fn a_resize_from_an_edge_holds_the_far_edge_of_a_direction_patch_too() {
    let scene = Scene::new();
    let edited = distorted(&edited_fixture(&scene, WORLD));
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);

    // A bearing rather than a position: the centre is a unit direction, the
    // corners are directions too, and the arithmetic has to renormalize without
    // moving any of them.
    let sighting = track.observations[0].image as usize;
    let (camera, pose) = view(&edited, sighting);
    {
        let payload = match &mut track.stage {
            Stage::Track(payload) => payload,
            Stage::Cluster(_) => unreachable!("a track from a point is at the track stage"),
        };
        let frame = payload.frame.as_mut().expect("a stored patch");
        let direction = frame.center.coords.normalize();
        *frame = OrientedPatch::from_infinity_direction(
            Point3::from(direction),
            Vector3::new(0.0, 1.0, 0.0),
            [0.02, 0.02],
        );
        payload.position = None;
    }
    // The keypoint has to be the bearing's own projection for the outline to be
    // the frame; put it there.
    let centre = corner_pixel(
        track
            .track()
            .and_then(|p| p.frame.as_ref())
            .expect("a frame"),
        &camera,
        &pose,
        0.0,
        0.0,
    );
    track.observations[0]
        .track
        .as_mut()
        .expect("a track slot")
        .keypoint = Some([centre[0] as f32, centre[1] as f32]);

    let before = outline_of(&track, &edited, 0);
    let far_before = corner_pixel(&before, &camera, &pose, 0.0, -1.0);
    let out = corner_pixel(&before, &camera, &pose, 0.0, 1.8);

    let (next, _) =
        resize_from_edge(&track, &edited, 0, Edge::PlusV, out).expect("a pixel the ray reaches");
    let after = next.track().and_then(|p| p.frame.clone()).expect("a frame");
    assert_eq!(after.w, 0.0, "a bearing stays a bearing");
    assert!(
        (after.center.coords.norm() - 1.0).abs() < 1e-12,
        "a bearing's centre is a unit direction"
    );
    assert_eq!(after.half_extent[0], after.half_extent[1]);
    let dragged = corner_pixel(&after, &camera, &pose, 0.0, 1.0);
    assert!(
        (dragged[0] - out[0]).abs() < 1e-6 && (dragged[1] - out[1]).abs() < 1e-6,
        "the dragged edge should land on {out:?}, it landed on {dragged:?}",
    );
    let far_after = corner_pixel(&after, &camera, &pose, 0.0, -1.0);
    assert!(
        (far_after[0] - far_before[0]).abs() < 1e-6 && (far_after[1] - far_before[1]).abs() < 1e-6,
        "the far edge moved from {far_before:?} to {far_after:?}",
    );
}

// ---- The world-point forms the 3D viewer names ----------------------------

/// The same track with its surfel turned into a bearing along its own
/// direction, and every sighting's keypoint put back on the bearing's own
/// projection so the outline is the frame.
///
/// What the three handles of a track at infinity are driven against: the centre
/// is a unit direction, the corners are directions too, and every step has to
/// renormalize without moving any of them.
fn as_bearing(track: &EditableTrack, edited: &EditedReconstruction) -> EditableTrack {
    let mut track = track.clone();
    {
        let payload = match &mut track.stage {
            Stage::Track(payload) => payload,
            Stage::Cluster(_) => unreachable!("a track from a point is at the track stage"),
        };
        let frame = payload.frame.as_mut().expect("a stored patch");
        let direction = frame.center.coords.normalize();
        *frame = OrientedPatch::from_infinity_direction(
            Point3::from(direction),
            Vector3::new(0.0, 1.0, 0.0),
            [0.02, 0.02],
        );
        payload.position = None;
    }
    let frame = frame_of(&track);
    for observation in &mut track.observations {
        let (camera, pose) = view(edited, observation.image as usize);
        let centre = corner_pixel(&frame, &camera, &pose, 0.0, 0.0);
        observation.track.as_mut().expect("a track slot").keypoint =
            Some([centre[0] as f32, centre[1] as f32]);
    }
    track
}

/// The place named is projected onto the plane before anything moves, so a
/// point off the plane slides the patch to the place directly under it and the
/// patch never leaves the plane it is in.
#[test]
fn a_slide_to_a_place_lands_the_centre_on_the_plane_under_it() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let was = frame_of(&track);
    let offsets = projection_offsets(&track, &edited);

    // Off the plane on purpose: the normal component is what the step drops.
    let wanted = was.u_axis * 0.05 + was.v_axis * (-0.03);
    let point = was.center + wanted + was.normal() * 0.4;

    let (next, report) = translate_frame_to(&track, &edited, point).expect("a place on the plane");
    assert!(report.changed);
    let frame = frame_of(&next);
    assert_eq!(frame.u_axis, was.u_axis);
    assert_eq!(frame.v_axis, was.v_axis);
    assert_eq!(frame.half_extent, was.half_extent);
    let moved = frame.center - was.center;
    assert!(
        (moved - wanted).norm() < 1e-12,
        "the centre should have moved by the in-plane part alone, it moved by {moved:?}",
    );
    assert!((report.moved - wanted.norm()).abs() < 1e-12);
    assert_eq!(report.center, frame.center);
    assert_eq!(report.placed, next.observations.len());
    assert_eq!(next.track().and_then(|p| p.position), Some(frame.center));
    assert_eq!(next.track().and_then(|p| p.bitmap.clone()), None);
    // Every sighting kept its own offset from where the centre projects, which
    // is what the tile is cut on, and none of them was ruled on.
    for (offset, now) in offsets.iter().zip(projection_offsets(&next, &edited)) {
        assert!(
            (now[0] - offset[0]).abs() < 1e-3 && (now[1] - offset[1]).abs() < 1e-3,
            "a slide scrambled a sighting's offset: {offset:?} became {now:?}",
        );
    }
    assert!(next.observations.iter().all(|o| !o.pinned));

    // The place it already sits at is not a move.
    let (again, report) = translate_frame_to(&next, &edited, frame.center).expect("the same place");
    assert!(!report.changed);
    assert_eq!(frame_of(&again).center, frame.center);
}

/// **One implementation of each edit.** The pixel form is the unprojection in
/// front of the world-point form, so a pointer and the place it names have to
/// leave the patch in the same position -- the pixel form acting on the outline
/// re-anchored on one sighting, and the world-point form on the surfel itself.
#[test]
fn a_pixel_gesture_is_the_place_gesture_the_pointer_named() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let frame = frame_of(&track);

    let dragged = 1;
    let (camera, pose) = view(&edited, track.observations[dragged].image as usize);
    let outline = outline_of(&track, &edited, dragged);
    let target = corner_pixel(&outline, &camera, &pose, 0.7, 0.4);
    // The place that pixel names, read from the outline's centre and carried to
    // the surfel's: the one reduction both steps make.
    let offset = outline
        .keypoint_plane_offset(&camera, &pose, target)
        .expect("the fixture's ray meets the plane");
    let point = frame.center + offset;

    let (by_pixel, _) = translate_frame(&track, &edited, dragged, target).expect("a usable drag");
    let (by_place, _) = translate_frame_to(&track, &edited, point).expect("a place on the plane");
    assert!(
        (frame_of(&by_pixel).center - frame_of(&by_place).center).norm() < 1e-12,
        "the pixel and the place it names slid the patch to two different centres",
    );

    let edge = corner_pixel(&outline, &camera, &pose, 2.3, 0.0);
    let offset = outline
        .keypoint_plane_offset(&camera, &pose, edge)
        .expect("the fixture's ray meets the plane");
    let (by_pixel, pixel_report) =
        resize_from_edge(&track, &edited, dragged, Edge::PlusU, edge).expect("a usable drag");
    let (by_place, place_report) =
        resize_from_edge_to(&track, &edited, Edge::PlusU, frame.center + offset)
            .expect("a place on the plane");
    assert!((pixel_report.half - place_report.half).abs() < 1e-12);
    assert!(
        (frame_of(&by_pixel).center - frame_of(&by_place).center).norm() < 1e-12,
        "the pixel and the place it names resized the patch about two different centres",
    );
    // The gesture's report says which sighting it was named in; the place's has
    // no photograph to name.
    assert_eq!(pixel_report.observation, Some(dragged));
    assert_eq!(place_report.observation, None);
    assert_eq!(place_report.pixel, None);
}

/// The far edge is held whoever names the near one: with the dragged edge at
/// `+h` and the far one at `-h`, a place naming the offset `p` gives `(p + h) /
/// 2` and moves the centre by `h' - h` along that axis.
#[test]
fn a_resize_to_a_place_puts_that_edge_there_and_holds_the_far_one() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let was = frame_of(&track);
    let far_before = was.to_world(-1.0, 0.0);
    // Two and a bit half-lengths out along `+u`, and off the plane again, which
    // the axis dot product drops on its own.
    let point = was.to_world(2.3, 0.0) + was.normal() * 0.4;

    let (next, report) =
        resize_from_edge_to(&track, &edited, Edge::PlusU, point).expect("a usable place");
    assert!(report.changed);
    assert_eq!(report.observation, None);
    assert_eq!(report.image, None);
    let after = frame_of(&next);
    assert_eq!(
        after.half_extent[0], after.half_extent[1],
        "a patch frame is square, so a resize is one scale"
    );
    assert!((report.half - (2.3 + 1.0) / 2.0 * was.half_extent[0]).abs() < 1e-12);
    let moved = after.center - was.center;
    assert!((moved - was.u_axis * (report.half - report.was)).norm() < 1e-12);
    assert!(
        (after.to_world(1.0, 0.0) - (was.center + was.u_axis * (2.3 * was.half_extent[0]))).norm()
            < 1e-12,
        "the dragged edge did not land on the place it was given",
    );
    assert!(
        (after.to_world(-1.0, 0.0) - far_before).norm() < 1e-12,
        "the far edge moved",
    );
    assert_eq!(next.track().and_then(|p| p.bitmap.clone()), None);
    assert!(next.observations.iter().all(|o| !o.pinned));

    // The size it already has is not a resize.
    let (_, report) = resize_from_edge_to(&next, &edited, Edge::PlusU, after.to_world(1.0, 0.0))
        .expect("the edge's own place");
    assert!(!report.changed);
}

/// A bearing's centre is a unit direction and its half-extents are stated
/// against one, so both world-point steps renormalize and the corners stay the
/// directions they were.
#[test]
fn the_place_forms_carry_a_bearing_on_the_unit_sphere() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = as_bearing(&track_of(&bench, &label), &edited);
    let was = frame_of(&track);
    let (camera, pose) = view(&edited, track.observations[0].image as usize);

    // The slide: a tangent place, which the step takes back onto the sphere.
    let point = was.center + was.u_axis * (0.7 * was.half_extent[0]);
    let (slid, report) = translate_frame_to(&track, &edited, point).expect("a tangent place");
    assert!(report.changed);
    let frame = frame_of(&slid);
    assert_eq!(frame.w, 0.0, "a bearing stays a bearing");
    assert!((frame.center.coords.norm() - 1.0).abs() < 1e-12);
    assert!(
        (frame.center.coords.normalize() - point.coords.normalize()).norm() < 1e-12,
        "the bearing did not move onto the tangent place that was named",
    );

    // The resize: the far edge is held on the sky too, which is what the
    // renormalization has to leave alone.
    let far_before = corner_pixel(&was, &camera, &pose, -1.0, 0.0);
    let (bigger, report) =
        resize_from_edge_to(&track, &edited, Edge::PlusU, was.to_world(1.8, 0.0))
            .expect("a tangent place");
    assert!(report.changed);
    let frame = frame_of(&bigger);
    assert_eq!(frame.w, 0.0);
    assert!((frame.center.coords.norm() - 1.0).abs() < 1e-12);
    assert_eq!(frame.half_extent[0], frame.half_extent[1]);
    let far_after = corner_pixel(&frame, &camera, &pose, -1.0, 0.0);
    assert!(
        (far_after[0] - far_before[0]).abs() < 1e-6 && (far_after[1] - far_before[1]).abs() < 1e-6,
        "the far edge moved from {far_before:?} to {far_after:?}",
    );
    let dragged = corner_pixel(&frame, &camera, &pose, 1.0, 0.0);
    let wanted = corner_pixel(&was, &camera, &pose, 1.8, 0.0);
    assert!(
        (dragged[0] - wanted[0]).abs() < 1e-6 && (dragged[1] - wanted[1]).abs() < 1e-6,
        "the dragged edge should land on {wanted:?}, it landed on {dragged:?}",
    );
}

/// The two refusals a place can earn: one that is not a point of the world, and
/// a track with no world geometry to name a place on.
#[test]
fn a_place_that_is_not_one_and_a_cluster_are_both_refused() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let nowhere = Point3::new(0.0, f64::NAN, 0.0);

    assert!(matches!(
        translate_frame_to(&track, &edited, nowhere),
        Err(TrackEditError::BadPlace(_))
    ));
    assert!(matches!(
        resize_from_edge_to(&track, &edited, Edge::PlusU, nowhere),
        Err(TrackEditError::BadPlace(_))
    ));

    let (bench, made) =
        create_cluster(&Bench::new(), &pixel_seed(0, scene_pixel(&edited))).expect("a usable seed");
    let cluster = track_of(&bench, &made.label);
    let somewhere = frame_of(&track).center;
    assert!(matches!(
        translate_frame_to(&cluster, &edited, somewhere),
        Err(TrackEditError::WrongStage { .. })
    ));
    assert!(matches!(
        resize_from_edge_to(&cluster, &edited, Edge::PlusU, somewhere),
        Err(TrackEditError::WrongStage { .. })
    ));
}

#[test]
fn a_resize_about_the_centre_moves_both_edges_and_drops_what_was_read_over_the_old_square() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    track.observations[0]
        .track
        .as_mut()
        .expect("a track slot")
        .zncc = Some(0.88);
    let centre = track
        .track()
        .and_then(|p| p.frame.as_ref())
        .expect("a frame")
        .center;
    let keypoints: Vec<_> = track
        .observations
        .iter()
        .map(|o| o.track.as_ref().and_then(|m| m.keypoint))
        .collect();

    let (next, report) = resize_frame(&track, 0.5).expect("a positive half-length");
    assert!(report.changed);
    let frame = next.track().and_then(|p| p.frame.clone()).expect("a frame");
    assert_eq!(frame.half_extent, [0.5, 0.5]);
    assert_eq!(frame.center, centre, "a centred resize moves nothing");
    assert_eq!(next.track().and_then(|p| p.bitmap.clone()), None);
    for (k, observation) in next.observations.iter().enumerate() {
        let measurement = observation.track.as_ref().expect("a track slot");
        assert_eq!(measurement.keypoint, keypoints[k], "the sighting stays");
        assert_eq!(
            measurement.zncc, None,
            "what was read over the old square goes"
        );
    }
    assert!(resize_frame(&track, 0.0).is_err());
    let held = track
        .track()
        .and_then(|p| p.frame.as_ref())
        .expect("a frame")
        .half_extent[0];
    assert!(
        !resize_frame(&track, held)
            .expect("the size it has")
            .1
            .changed
    );
}

#[test]
fn a_turn_keeps_the_axes_and_the_normal_and_lands_the_corner_under_the_release_point() {
    let scene = Scene::new();
    let edited = distorted(&edited_fixture(&scene, WORLD));
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (camera, pose) = view(&edited, track.observations[0].image as usize);
    let before = track
        .track()
        .and_then(|p| p.frame.clone())
        .expect("a frame");

    let angle = 0.42_f64;
    // Where the `(1, 1)` corner has to land for that turn: the corner's own
    // in-plane offset, rotated about the normal, projected through the lens.
    let rotation = nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Unit::new_normalize(before.normal()),
        angle,
    );
    let offset = before.u_axis * before.half_extent[0] + before.v_axis * before.half_extent[1];
    let expected = {
        let turned = before.center + rotation * offset;
        let pc = pose.transform_point_homogeneous(turned.coords, before.w);
        let (u, v) = camera.ray_to_pixel([pc.x, pc.y, pc.z]).expect("in front");
        [u, v]
    };

    let (next, report) = rotate_frame(&track, angle).expect("a finite angle");
    assert!(report.changed);
    assert!((report.degrees - angle.to_degrees()).abs() < 1e-12);
    let after = next.track().and_then(|p| p.frame.clone()).expect("a frame");
    assert!((after.u_axis.norm() - before.u_axis.norm()).abs() < 1e-12);
    assert!((after.v_axis.norm() - before.v_axis.norm()).abs() < 1e-12);
    assert!(
        (after.normal() - before.normal()).norm() < 1e-12,
        "a turn about the normal keeps the plane and the face"
    );
    assert_eq!(after.center, before.center, "a turn moves no sighting");
    assert_eq!(after.half_extent, before.half_extent);
    assert_eq!(
        next.observations[0].site(),
        track.observations[0].site(),
        "a turn moves no sighting"
    );

    let landed = corner_pixel(&after, &camera, &pose, 1.0, 1.0);
    assert!(
        (landed[0] - expected[0]).abs() < 1e-6 && (landed[1] - expected[1]).abs() < 1e-6,
        "the turned corner should land on {expected:?}, it landed on {landed:?}",
    );
    assert!(!rotate_frame(&track, 0.0).expect("no turn").1.changed);
    assert!(rotate_frame(&track, f64::NAN).is_err());
}

#[test]
fn a_cluster_edge_drag_scales_the_shape_and_holds_the_far_edge_of_the_parallelogram() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let where_at = scene.project(0, WORLD);
    let (bench, report) =
        create_cluster(&Bench::new(), &pixel_seed(0, where_at)).expect("a usable seed");
    let mut track = track_of(&bench, &report.label);
    // A sheared shape, so the test says something about anisotropy surviving.
    let shape = [[6.0, 1.5], [-0.5, 4.0]];
    let measurement = track.observations[0].cluster.as_mut().expect("a seed");
    measurement.seed_shape = shape;
    measurement.zncc = Some(0.8);
    let radius = track.cluster().expect("a cluster").radius;

    // The parallelogram's corners, before.
    let corner = |position: [f64; 2], shape: [[f64; 2]; 2], s: f64, t: f64| {
        [
            position[0] + shape[0][0] * s * radius + shape[0][1] * t * radius,
            position[1] + shape[1][0] * s * radius + shape[1][1] * t * radius,
        ]
    };
    let far_before = corner(where_at, shape, -1.0, 0.0);
    // A pointer inside the photograph: the step clamps one that is not, and this
    // test is about the arithmetic rather than about the clamp.
    let out = corner(where_at, shape, 1.1, 0.0);
    assert!(
        out[0] >= 0.0 && out[0] < f64::from(IMG_W) && out[1] >= 0.0 && out[1] < f64::from(IMG_H),
        "the drag target {out:?} has to be on the photograph"
    );

    let (next, report) =
        resize_from_edge(&track, &edited, 0, Edge::PlusU, out).expect("a usable drag");
    assert!(report.changed);
    let cluster = next.observations[0].cluster.as_ref().expect("a seed");
    let scaled = cluster.seed_shape;
    // One scalar: the shear is exactly the shear it had.
    let scale = scaled[0][0] / shape[0][0];
    for (row, was) in scaled.iter().zip(&shape) {
        for (now, was) in row.iter().zip(was) {
            assert!(
                (now - was * scale).abs() < 1e-12,
                "the shape was not scaled uniformly"
            );
        }
    }
    let position = cluster.seed_position;
    let dragged = corner(position, scaled, 1.0, 0.0);
    assert!(
        (dragged[0] - out[0]).abs() < 1e-9 && (dragged[1] - out[1]).abs() < 1e-9,
        "the dragged edge should land on {out:?}, it landed on {dragged:?}",
    );
    let far_after = corner(position, scaled, -1.0, 0.0);
    assert!(
        (far_after[0] - far_before[0]).abs() < 1e-9 && (far_after[1] - far_before[1]).abs() < 1e-9,
        "the far edge moved from {far_before:?} to {far_after:?}",
    );
    assert_eq!(
        cluster.zncc, None,
        "the refinement was run at another shape"
    );
    assert!((report.half - half_width_px(scaled, radius)).abs() < 1e-12);
}

#[test]
fn a_hand_set_cluster_shape_re_seeds_the_sighting_and_does_not_pin_its_verdict() {
    let scene = Scene::new();
    let where_at = scene.project(0, WORLD);
    let (bench, report) =
        create_cluster(&Bench::new(), &pixel_seed(0, where_at)).expect("a usable seed");
    let mut track = track_of(&bench, &report.label);
    let moved = [where_at[0] + 4.0, where_at[1]];
    let measurement = track.observations[0].cluster.as_mut().expect("a seed");
    measurement.position = Some(moved);
    measurement.zncc = Some(0.77);
    let radius = track.cluster().expect("a cluster").radius;

    let turned = [[0.0, -7.0], [7.0, 0.0]];
    let (next, report) = set_observation_shape(&track, 0, turned).expect("a shape with area");
    assert!(report.changed);
    assert_eq!(report.shape, turned);
    assert!((report.half_px - half_width_px(turned, radius)).abs() < 1e-12);
    let cluster = next.observations[0].cluster.as_ref().expect("a seed");
    assert_eq!(cluster.seed_position, moved, "re-seeded where it was drawn");
    assert_eq!(cluster.seed_shape, turned);
    assert_eq!(cluster.zncc, None);
    assert!(
        !next.observations[0].pinned,
        "a size is not a ruling on whether the sighting belongs"
    );
    assert!(set_observation_shape(&track, 0, [[1.0, 2.0], [2.0, 4.0]]).is_err());
}

#[test]
fn the_patch_steps_refuse_the_stage_they_do_not_belong_to() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let where_at = scene.project(0, WORLD);
    let (bench, report) =
        create_cluster(&Bench::new(), &pixel_seed(0, where_at)).expect("a usable seed");
    let cluster = track_of(&bench, &report.label);
    assert!(matches!(
        resize_frame(&cluster, 0.1),
        Err(TrackEditError::WrongStage { .. })
    ));
    assert!(matches!(
        rotate_frame(&cluster, 0.1),
        Err(TrackEditError::WrongStage { .. })
    ));

    let (bench, label) = bench_with_point(&edited, 0);
    let at_track = track_of(&bench, &label);
    assert!(matches!(
        set_observation_shape(&at_track, 0, [[5.0, 0.0], [0.0, 5.0]]),
        Err(TrackEditError::WrongStage { .. })
    ));
    assert!(matches!(
        set_observation_keypoint(&at_track, &edited, 9, [1.0, 2.0]),
        Err(TrackEditError::NoSuchObservation { .. })
    ));
}

// ---- Finite points and bearings --------------------------------------------
//
// The boundary is a property of the rays and not of the picture, so the fixture
// these tests turn is the *capture*: the same textured plane, put two hundred
// units out and looked at from cameras five centimetres apart, whose sightings
// fix a direction and no depth. One camera of it sits twenty units off to the
// side, so the very same eight sightings plus a ninth of its own do fix one --
// which is what makes promotion and demotion two settings of one dial.

/// The distant plane the bearing fixture stands on.
const FAR_Z: f64 = 200.0;

/// How far apart the bearing fixture's near cameras step along world `+x`.
const FAR_STEP: f64 = 0.05;

/// How many near cameras it has: the rays whose parallax is a third of a pixel
/// across the whole run.
const FAR_NEAR_VIEWS: usize = 8;

/// Where the one camera with real baseline sits along world `+x`.
const FAR_OFFSET_X: f64 = 20.0;

/// The image index of that camera.
const FAR_OFFSET_IMAGE: u32 = FAR_NEAR_VIEWS as u32;

/// The world half-extent the far plane's patch has: the same apparent size the
/// near scene's patch has, which is what keeps the two fixtures comparable.
const FAR_HALF_WORLD: f64 = 6.0;

/// Where the far fixture's point stands: straight ahead of the near run, on the
/// distant plane.
fn far_world() -> Point3<f64> {
    Point3::new(0.0, 0.0, FAR_Z)
}

/// The unit bearing the far point is seen along, which is what a `w = 0` row of
/// it stores.
fn far_direction() -> Point3<f64> {
    Point3::from(far_world().coords.normalize())
}

/// The capture: eight cameras stepping by [`FAR_STEP`] along world `+x`, and a
/// ninth [`FAR_OFFSET_X`] away, all looking down `+z` at the plane `z = FAR_Z`.
fn far_scene() -> Scene {
    let mut centers: Vec<[f64; 3]> = (0..FAR_NEAR_VIEWS)
        .map(|k| [k as f64 * FAR_STEP, 0.0, 0.0])
        .collect();
    centers.push([FAR_OFFSET_X, 0.0, 0.0]);
    Scene::from_centers(&centers, FAR_Z)
}

/// The angular half-extent a bearing on the far plane carries.
fn far_angular_half() -> f64 {
    FAR_HALF_WORLD / FAR_Z
}

/// A version of [`far_scene`] holding the far point as a `w = 0` bearing on its
/// tangent frame, observed by the eight near cameras.
fn far_bearing_edited(scene: &Scene) -> EditedReconstruction {
    let frame = OrientedPatch::from_infinity_direction(
        far_direction(),
        Vector3::new(0.0, 1.0, 0.0),
        [far_angular_half(), far_angular_half()],
    );
    let observing: Vec<u32> = (0..FAR_NEAR_VIEWS as u32).collect();
    EditedReconstruction::new(Arc::new(with_columns(
        fixture_of(scene, far_direction(), 0.0, &observing, &frame),
        BITMAP_R,
    )))
}

/// A version of [`far_scene`] holding the far point as a finite `w = 1` point on
/// a world-unit plane patch, observed by the same eight near cameras: the same
/// sightings, stored as the thing the geometry does not support.
fn far_finite_edited(scene: &Scene) -> EditedReconstruction {
    let frame = OrientedPatch::from_center_normal(
        far_world(),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [FAR_HALF_WORLD, FAR_HALF_WORLD],
    );
    let observing: Vec<u32> = (0..FAR_NEAR_VIEWS as u32).collect();
    EditedReconstruction::new(Arc::new(with_columns(
        fixture_of(scene, far_world(), 1.0, &observing, &frame),
        BITMAP_R,
    )))
}

/// The classification a fit reported.
fn call_of(report: &FitReport) -> TrackClassification {
    report
        .classification
        .expect("a track-stage fit classifies its rays")
}

#[test]
fn a_bearing_whose_rays_stay_parallel_stays_a_bearing_at_the_size_it_had() {
    let scene = far_scene();
    let edited = far_bearing_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let before = frame_of(&track);
    assert_eq!(before.w, 0.0, "the fixture stores a bearing");

    // What a reading of the track as it stands makes of each sighting: the fit
    // has to agree with this, because a fit that promoted the frame to a
    // provisional depth would re-warp every view and walk the sightings.
    let (read, _) = evaluate_over(&scene, &edited, &track).expect("eight sightings in");
    let seen: Vec<f64> = read
        .observations
        .iter()
        .map(|o| {
            o.track
                .as_ref()
                .and_then(|m| m.zncc)
                .expect("a bearing's tangent frame registers in every view")
        })
        .collect();
    assert!(
        seen.iter().all(|&z| z > 0.5),
        "the reading should agree in every view, it scored {seen:?}"
    );

    let (fitted, report) = fit_over(&scene, &edited, &track).expect("eight sightings in");
    let call = call_of(&report);
    assert!(
        call.at_infinity,
        "a third of a pixel of parallax over two hundred units is a bearing, \
         the fit said {call}"
    );
    assert_eq!(call.reason, ClassificationReason::DepthUnresolved);
    assert!(call.inverse_depth_z < call.inverse_depth_z_cutoff, "{call}");
    assert!(call.max_pair_angle_deg < 0.5, "{call}");
    assert_eq!(report.kept_at_seed, 0);

    let after = frame_of(&fitted);
    assert_eq!(after.w, 0.0, "a bearing stays a bearing");
    assert!(
        (after.center.coords.norm() - 1.0).abs() < 1e-12,
        "a bearing's coordinate is a unit direction, it is {}",
        after.center
    );
    assert_eq!(
        after.half_extent, before.half_extent,
        "no promotion, so no rescale"
    );
    assert_eq!(
        fitted.track().expect("the track stage").position,
        Some(after.center),
        "the coordinate and the frame's centre are one thing"
    );

    // No drift: every sighting is where the reading found it, and scores what
    // the reading scored.
    for (k, observation) in fitted.observations.iter().enumerate() {
        let m = observation.track.as_ref().expect("a track slot");
        let was = read.observations[k]
            .track
            .as_ref()
            .and_then(|m| m.keypoint)
            .expect("a seed");
        let now = m.keypoint.expect("a keypoint");
        let moved = f64::from(now[0] - was[0]).hypot(f64::from(now[1] - was[1]));
        assert!(moved < 1.0, "sighting {k} moved {moved} px");
        let zncc = m.zncc.expect("a score");
        assert!(zncc > 0.5, "sighting {k} scored {zncc}");
        assert!(
            (zncc - seen[k]).abs() < 0.2,
            "sighting {k} scored {zncc} where the reading scored {}",
            seen[k]
        );
        assert_eq!(m.walked_px, None, "sighting {k} was not walked");
    }
}

#[test]
fn a_bearings_ray_angle_is_measured_against_the_bearing() {
    // The angle to a bearing is the angle between the sighting's ray and the
    // direction itself. Measuring it against a phantom point one unit from the
    // world origin -- which is what applying the pose's translation to a unit
    // direction does -- would print tens of degrees for rays that agree to a
    // thousandth of one.
    let scene = far_scene();
    let edited = far_bearing_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let (read, report) = evaluate_over(&scene, &edited, &track).expect("eight sightings in");
    assert!(report.at_infinity, "the reading knows it read a bearing");
    for (k, observation) in read.observations.iter().enumerate() {
        let angle = observation
            .track
            .as_ref()
            .and_then(|m| m.ray_angle_deg)
            .expect("an angle");
        assert!(
            angle < 0.1,
            "sighting {k}'s ray agrees with the bearing to a thousandth of a degree, \
             the panel would show {angle}"
        );
    }
}

#[test]
fn a_finite_track_fits_finite_and_says_which_test_settled_it() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let (fitted, report) = fit_over(&scene, &edited, &track).expect("two observations in");
    let call = call_of(&report);
    assert!(
        !call.at_infinity,
        "fifteen degrees of parallax is a point, the fit said {call}"
    );
    assert_eq!(call.reason, ClassificationReason::WellConditioned);
    assert!(call.max_pair_angle_deg > 1.0, "{call}");
    assert!((call.coordinate - WORLD).norm() < 0.02, "{call}");
    let after = frame_of(&fitted);
    assert_eq!(after.w, 1.0);
    assert_eq!(after.center, call.coordinate);
}

#[test]
fn a_bearing_given_a_sighting_with_real_baseline_becomes_a_point() {
    let scene = far_scene();
    let edited = far_bearing_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    let before = frame_of(&track);

    // The ninth camera, twenty units off to the side: the sighting that gives
    // the track a depth. It sits where the plane's own point projects, which is
    // what a person pointing at the same piece of surface would do.
    let pixel = scene.project(FAR_OFFSET_IMAGE as usize, far_world());
    let (added, _) = add_observation(&track, &ObservationSeed::at_pixel(FAR_OFFSET_IMAGE, pixel))
        .expect("a finite pixel");
    let last = added.observations.len() - 1;
    let (turned, _) = set_verdict(&added, last, Verdict::In).expect("the new observation");
    track = turned;

    let (fitted, report) = fit_over(&scene, &edited, &track).expect("nine sightings in");
    let call = call_of(&report);
    assert!(
        !call.at_infinity,
        "a twenty-unit baseline resolves two hundred units of depth, the fit said {call}"
    );
    assert!(
        (call.coordinate - far_world()).norm() < 2.0,
        "the promoted point should stand on the plane, it stands at {}",
        call.coordinate
    );
    let after = frame_of(&fitted);
    assert_eq!(after.w, 1.0, "a promotion writes a place");
    assert_eq!(after.center, call.coordinate);
    // The angular extents became world ones at the placement distance, so the
    // patch is the apparent size it was.
    let grew = after.half_extent[0] / before.half_extent[0];
    assert!(
        (grew / FAR_Z - 1.0).abs() < 0.1,
        "the frame grew by {grew} where the placement distance is about {FAR_Z}"
    );
}

#[test]
fn a_finite_track_whose_rays_no_longer_resolve_a_depth_becomes_a_bearing() {
    let scene = far_scene();
    let edited = far_finite_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let before = frame_of(&track);
    assert_eq!(before.w, 1.0, "the fixture stores a place");

    let (fitted, report) = fit_over(&scene, &edited, &track).expect("eight sightings in");
    let call = call_of(&report);
    assert!(
        call.at_infinity,
        "the eight near cameras cannot resolve two hundred units, the fit said {call}"
    );
    let after = frame_of(&fitted);
    assert_eq!(after.w, 0.0, "a demotion writes a bearing");
    assert!(
        (after.center.coords.norm() - 1.0).abs() < 1e-12,
        "a bearing's coordinate is a unit direction, it is {}",
        after.center
    );
    assert!(
        (after.center.coords - far_direction().coords).norm() < 1e-3,
        "the bearing should be the direction the sightings agree on, it is {}",
        after.center
    );
    // The world extents became angular ones by the distance the frame stood at.
    let shrank = before.half_extent[0] / after.half_extent[0];
    assert!(
        (shrank / FAR_Z - 1.0).abs() < 0.1,
        "the frame shrank by {shrank} where the frame stood about {FAR_Z} out"
    );
    assert_eq!(
        fitted.track().expect("the track stage").position,
        Some(after.center)
    );
}

#[test]
fn a_sighting_the_fit_would_walk_past_the_bar_keeps_its_seed_and_says_so() {
    // Eight sightings, so the seven that agree hold the consensus still and the
    // one moved off it is the only row with anywhere to walk back to. A bar of
    // two pixels, and that one put four off the truth: the correlation will want
    // it back, and four pixels is further than the person said a sighting may be
    // moved.
    let scene = far_scene();
    let edited = far_bearing_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    track.thresholds.max_shift_px = 2.0;
    let truth = track.observations[0]
        .track
        .as_ref()
        .and_then(|m| m.keypoint)
        .expect("a stored keypoint");
    let moved = [truth[0] + 4.0, truth[1]];
    track.observations[0]
        .track
        .as_mut()
        .expect("a track slot")
        .keypoint = Some(moved);

    let (fitted, report) = fit_over(&scene, &edited, &track).expect("eight sightings in");
    assert_eq!(report.kept_at_seed, 1, "{report}");
    let held = fitted.observations[0].track.as_ref().expect("a track slot");
    assert_eq!(
        held.keypoint,
        Some(moved),
        "a sighting the fit refused to walk keeps the pixel it had"
    );
    let walked = held.walked_px.expect("the row says how far the peak sat");
    assert!(
        walked > track.thresholds.max_shift_px,
        "the peak sat {walked} px away, past the {} px bar",
        track.thresholds.max_shift_px
    );
    assert!(
        held.zncc.is_some(),
        "a sighting kept at its seed is still read there"
    );
    // The other sighting was inside the bar, so it moved and carries no flag.
    let other = fitted.observations[1].track.as_ref().expect("a track slot");
    assert_eq!(other.walked_px, None);
}

#[test]
fn an_upgrade_of_a_bearing_comes_back_a_bearing_and_commits_as_one() {
    let scene = far_scene();
    let edited = far_bearing_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let half = frame_of(&track).half_extent[0];

    // Down to the cluster stage, which throws the 3D away, and back up, which
    // builds it again from the sightings alone. A capture that only ever stated
    // a direction has to come back stating one.
    let (down, _) = stage_over(&scene, &edited, &track, StageKind::Cluster).expect("a downgrade");
    assert_eq!(down.stage_kind(), StageKind::Cluster);
    let (up, report) = stage_over(&scene, &edited, &down, StageKind::Track).expect("an upgrade");
    let fitted = report.fit.expect("an upgrade runs the track stage's fit");
    let call = call_of(&fitted);
    assert!(call.at_infinity, "the upgrade said {call}");
    let frame = frame_of(&up);
    assert_eq!(frame.w, 0.0, "an upgrade of a bearing frames a bearing");
    assert!(
        (frame.center.coords.norm() - 1.0).abs() < 1e-12,
        "a bearing's coordinate is a unit direction, it is {}",
        frame.center
    );
    assert!(
        (frame.half_extent[0] / half - 1.0).abs() < 0.3,
        "a round trip through the cluster stage should keep the size: {} against {half}",
        frame.half_extent[0]
    );

    // And the commit writes the row the format states for a bearing.
    let (next, written) = commit(&edited, &up).expect("a fitted bearing");
    let view = next.point(written.point).expect("just written");
    let point = view.point();
    assert_eq!(point.w, 0.0, "a bearing commits as a bearing");
    assert!(
        (point.position.coords.norm() - 1.0).abs() < 1e-9,
        "a w = 0 row stores a unit direction, it stores {}",
        point.position
    );
    assert_eq!(
        point.normal,
        Vector3::zeros(),
        "a w = 0 row carries a zero normal"
    );
    assert_eq!(view.normal_confidence(), Some(0));
    // And it survives materialisation, counted as a point at infinity.
    let (materialized, _) = next.materialize();
    assert_eq!(materialized.point_set.infinity_point_count, 1);
}

#[test]
fn moving_sizing_and_turning_a_bearing_keeps_its_direction_on_the_unit_sphere() {
    let scene = far_scene();
    let edited = far_bearing_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let before = frame_of(&track);
    let (camera, pose) = view(&edited, 0);
    let unit = |frame: &OrientedPatch, what: &str| {
        assert_eq!(frame.w, 0.0, "{what} kept the representation");
        assert!(
            (frame.center.coords.norm() - 1.0).abs() < 1e-12,
            "{what} left the direction at {}",
            frame.center.coords.norm()
        );
        assert!(frame.half_extent[0] > 0.0 && frame.half_extent[1] > 0.0);
    };

    // A slide: the pointer two pixels off the centre's own projection.
    let centre = corner_pixel(&before, &camera, &pose, 0.0, 0.0);
    let (slid, _) = translate_frame(&track, &edited, 0, [centre[0] + 2.0, centre[1]])
        .expect("a pixel the ray reaches");
    let frame = frame_of(&slid);
    unit(&frame, "a slide");
    assert_eq!(
        slid.track().expect("the track stage").position,
        Some(frame.center),
        "a slide carries the coordinate with the centre"
    );
    assert_ne!(frame.center, before.center, "a slide moves the bearing");

    // An edge drag.
    let out = corner_pixel(&before, &camera, &pose, 0.0, 1.8);
    let (resized, _) =
        resize_from_edge(&track, &edited, 0, Edge::PlusV, out).expect("a pixel the ray reaches");
    let frame = frame_of(&resized);
    unit(&frame, "an edge drag");
    assert_eq!(
        resized.track().expect("the track stage").position,
        Some(frame.center)
    );

    // A centred resize and a turn, neither of which moves the centre.
    let (sized, _) = resize_frame(&track, before.half_extent[0] * 1.5).expect("a positive half");
    let frame = frame_of(&sized);
    unit(&frame, "a centred resize");
    assert_eq!(frame.center, before.center);

    let (turned, _) = rotate_frame(&track, 0.3).expect("a finite angle");
    let frame = frame_of(&turned);
    unit(&frame, "a turn");
    assert_eq!(frame.center, before.center);
    assert!(
        (frame.normal() + before.center.coords).norm() < 1e-9,
        "a bearing's outward normal is minus its direction, it turned to {}",
        frame.normal()
    );
}

// ---- The criterion checked against the sightings ---------------------------
//
// The criterion answers whether a depth is *observable*. On an ill-conditioned
// solve it can clear its own bar on a depth nothing in the photographs supports:
// the least-squares midpoint of near-parallel, slightly inconsistent rays lands
// wherever the inconsistency throws it. The shape below is that case, built to
// order: the sightings agree with a bearing to a couple of pixels, the
// criterion's z-score clears the bar on a point a few units out, and the point
// reprojects three times worse than the bearing does.

/// How far the 5256-shaped fixture tilts its sightings across the run, in px
/// per camera: a weak linear trend, which is the parallax signal the midpoint
/// solve reads a depth out of.
const SKEW_TREND_PX: f64 = 0.2;

/// How far it throws them off that trend, in px, alternating: the inconsistency
/// that makes the rays skew, so the depth the solve reads is noise.
const SKEW_JITTER_PX: f64 = 1.5;

/// The 5256 shape's sightings: one per near camera, a weak linear tilt in `u`
/// about the bearing's own projection with an alternating jitter in `v`.
fn skewed_sightings(scene: &Scene) -> Vec<([f64; 2], usize)> {
    let centre = scene.project_homogeneous(0, far_direction(), 0.0);
    (0..FAR_NEAR_VIEWS)
        .map(|k| {
            let du = -SKEW_TREND_PX * (k as f64 - 3.5);
            let dv = if k % 2 == 0 {
                SKEW_JITTER_PX
            } else {
                -SKEW_JITTER_PX
            };
            ([centre[0] + du, centre[1] + dv], k)
        })
        .collect()
}

/// Classify `rays` over `views` with the defaults, and hand back the rays too.
fn classify_over(
    views: &[crate::patch::normal_refine::ProjectedImage<'_>],
    rays: &[([f64; 2], usize)],
) -> (TrackClassification, TrackRays) {
    let (_, built) = super::fit::triangulate_rays(rays, views).expect("a solve");
    let call = classify_track_rays(
        &built,
        views,
        DEFAULT_CLASSIFY_NOISE_FLOOR_PX,
        DEFAULT_CLASSIFY_Z_CUTOFF,
        RESIDUAL_MARGIN,
    );
    (call, built)
}

/// What the shared criterion alone made of `rays`, with no data check over it.
fn criterion_alone(
    views: &[crate::patch::normal_refine::ProjectedImage<'_>],
    rays: &TrackRays,
) -> crate::analysis::infinity::RayClassification {
    let centers: Vec<Point3<f64>> = views
        .iter()
        .map(|view| view.cam_from_world.inverse_translation_origin())
        .collect();
    let sigma_rad: Vec<f64> = rays
        .focal_max
        .iter()
        .map(|&f| DEFAULT_CLASSIFY_NOISE_FLOOR_PX / f)
        .collect();
    crate::analysis::infinity::classify_rays_at_infinity(
        &rays.dirs,
        &rays.centers,
        &sigma_rad,
        DEFAULT_CLASSIFY_Z_CUTOFF,
        crate::analysis::infinity::camera_extents(&centers),
    )
}

#[test]
fn a_depth_the_sightings_do_not_support_is_refused_and_the_bearing_stands() {
    let scene = far_scene();
    let views = scene.views();
    let rays = skewed_sightings(&scene);
    let (call, built) = classify_over(&views, &rays);

    // The criterion on its own calls this finite, and not on the cheap
    // pre-filter either: the solve is ill-conditioned, the baseline reaches the
    // horizon, and the z-score clears the bar. Which is the whole point of the
    // fixture -- without this the test would be checking the criterion rather
    // than the check over it.
    let alone = criterion_alone(&views, &built);
    assert!(
        matches!(
            alone.class,
            crate::analysis::infinity::Classification::Finite(_)
        ),
        "the criterion should call this finite, it called it {:?}",
        alone.class
    );
    assert!(
        alone.condition_number > crate::analysis::infinity::CONDITION_NUMBER_PREFILTER,
        "the pre-filter should not have settled it: condition {}",
        alone.condition_number
    );
    assert!(
        alone.inverse_depth_z > DEFAULT_CLASSIFY_Z_CUTOFF,
        "the z-score should clear the bar: {}",
        alone.inverse_depth_z
    );
    assert!(
        alone.resolvable_distance > call.finite_horizon,
        "the baseline should reach the horizon: {} against {}",
        alone.resolvable_distance,
        call.finite_horizon
    );

    // And the sightings say otherwise, so the bearing stands.
    assert!(
        call.at_infinity,
        "the check let a bad depth through: {call}"
    );
    assert_eq!(
        call.reason,
        ClassificationReason::FiniteDoesNotExplainTheSightings
    );
    assert!(
        call.finite_rms_px > 3.0 * call.bearing_rms_px,
        "the fixture should reproject three times worse as a point: \
         {} px against {} px",
        call.finite_rms_px,
        call.bearing_rms_px
    );
    assert!(
        call.bearing_rms_px < 3.0,
        "and the sightings should agree with the bearing to a couple of pixels: {} px",
        call.bearing_rms_px
    );
    assert!(
        (call.coordinate.coords.norm() - 1.0).abs() < 1e-12,
        "the coordinate is the unit bearing, it is {}",
        call.coordinate
    );
    // The sentence carries both residuals, which is what a person reading the
    // Action Log judges the call by.
    let said = call.to_string();
    assert!(
        said.contains("finite point would have") && said.contains("against the bearing's"),
        "the sentence should name both residuals: {said}"
    );
}

#[test]
fn a_depth_the_sightings_do_support_survives_the_check() {
    // The near scene's own track, at its exact projections: the parallax is
    // fifteen degrees and the bearing cannot bend toward it, so the point wins
    // by a wide margin and the criterion's answer stands.
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let views = scene.views();
    let rays: Vec<([f64; 2], usize)> = track
        .observations
        .iter()
        .map(|o| {
            let k = o
                .track
                .as_ref()
                .and_then(|m| m.keypoint)
                .expect("a keypoint");
            ([f64::from(k[0]), f64::from(k[1])], o.image as usize)
        })
        .collect();
    let (call, _) = classify_over(&views, &rays);

    assert!(!call.at_infinity, "{call}");
    assert_eq!(call.reason, ClassificationReason::WellConditioned);
    assert!(
        call.finite_rms_px < call.residual_margin * call.bearing_rms_px,
        "the point should explain the sightings clearly better: {} px against {} px",
        call.finite_rms_px,
        call.bearing_rms_px
    );
    assert!(
        call.finite_rms_px + DEFAULT_CLASSIFY_NOISE_FLOOR_PX < call.bearing_rms_px,
        "and by more than a pixel of noise: {} px against {} px",
        call.finite_rms_px,
        call.bearing_rms_px
    );
    assert!(call.to_string().contains("rms"), "{call}");
}

#[test]
fn a_fit_of_the_unsupported_depth_writes_the_bearing() {
    // The same shape, through the whole fit rather than through the
    // classification alone: what the step writes is the bearing, at the frame
    // size it arrived with.
    let scene = far_scene();
    let edited = far_bearing_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let mut track = track_of(&bench, &label);
    let before = frame_of(&track);
    for (k, (pixel, image)) in skewed_sightings(&scene).into_iter().enumerate() {
        assert_eq!(track.observations[k].image as usize, image);
        track.observations[k]
            .track
            .as_mut()
            .expect("a track slot")
            .keypoint = Some([pixel[0] as f32, pixel[1] as f32]);
    }
    // A bar wide enough that the fit keeps the sightings where they were put,
    // so what is under test is the classification and not the walk.
    track.thresholds.max_shift_px = 64.0;

    let (fitted, report) = fit_over(&scene, &edited, &track).expect("eight sightings in");
    let call = call_of(&report);
    assert!(call.at_infinity, "the fit wrote a point: {report}");
    let after = frame_of(&fitted);
    assert_eq!(after.w, 0.0);
    assert_eq!(after.half_extent, before.half_extent);
}

// ---- A bearing that carries no surfel --------------------------------------

/// [`far_bearing_edited`] with the patch-frame columns stripped: the bearing a
/// node with no `patch_u_halfvec` holds, which is every row of a `sift_files`
/// value.
fn far_bearing_frameless(scene: &Scene) -> EditedReconstruction {
    let frame = OrientedPatch::from_infinity_direction(
        far_direction(),
        Vector3::new(0.0, 1.0, 0.0),
        [far_angular_half(), far_angular_half()],
    );
    let observing: Vec<u32> = (0..FAR_NEAR_VIEWS as u32).collect();
    let mut recon = fixture_of(scene, far_direction(), 0.0, &observing, &frame);
    recon.point_set.patch_u_halfvec_xyz = None;
    recon.point_set.patch_v_halfvec_xyz = None;
    recon.rebuild_derived_fields();
    EditedReconstruction::new(Arc::new(recon))
}

/// A bearing is a bearing because the **point** says `w = 0`, and a node with no
/// patch frames has no frame to read a `w` off. Reading the frame put the three
/// numbers under `position` and committed them back as a place one unit from the
/// world origin.
#[test]
fn a_bearing_with_no_patch_frame_is_still_a_bearing() {
    let scene = far_scene();
    let edited = far_bearing_frameless(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let payload = track.track().expect("the track stage");

    assert_eq!(payload.frame, None, "the node stores no patch frame");
    assert!(
        payload.at_infinity,
        "the point's own w = 0 is what says it is a bearing"
    );
    let coordinate = payload.position.expect("a bearing has a coordinate");
    assert!(
        (coordinate.coords.norm() - 1.0).abs() < 1e-9,
        "a w = 0 row stores a unit direction, it stores {coordinate}"
    );

    // And it commits back as the row it came from.
    let (next, written) = commit(&edited, &track).expect("eight sightings in, with a coordinate");
    let point = next.point(written.point).expect("just written").point();
    assert_eq!(point.w, 0.0, "a frameless bearing commits as a bearing");
    assert!(
        (point.position.coords.norm() - 1.0).abs() < 1e-9,
        "it stores {}",
        point.position
    );

    // The two steps that need a surfel still refuse, in their own words.
    assert_eq!(evaluate_preconditions(&track), Err(EvaluateError::NoFrame));
    assert_eq!(fit_preconditions(&track), Err(FitError::NoFrame));
    assert_eq!(
        set_stage_preconditions(&track, StageKind::Cluster),
        Err(StageError::NoFrame)
    );
}

/// The same node's finite twin, so the flag is read and not assumed: a `w = 1`
/// row with no frame is a place.
#[test]
fn a_frameless_finite_point_is_not_a_bearing() {
    let scene = far_scene();
    let frame = OrientedPatch::from_center_normal(
        far_world(),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [FAR_HALF_WORLD, FAR_HALF_WORLD],
    );
    let observing: Vec<u32> = (0..FAR_NEAR_VIEWS as u32).collect();
    let mut recon = fixture_of(&scene, far_world(), 1.0, &observing, &frame);
    recon.point_set.patch_u_halfvec_xyz = None;
    recon.point_set.patch_v_halfvec_xyz = None;
    recon.rebuild_derived_fields();
    let edited = EditedReconstruction::new(Arc::new(recon));

    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let payload = track.track().expect("the track stage");
    assert_eq!(payload.frame, None);
    assert!(!payload.at_infinity, "a w = 1 row is a place");

    let (next, written) = commit(&edited, &track).expect("a finite commit");
    assert_eq!(
        next.point(written.point).expect("just written").point().w,
        1.0
    );
}

// ---- No effect --------------------------------------------------------------

/// A step told to do what has already been done says so and leaves the value
/// alone. The comparison cannot be exact: a pixel named on the wire or under a
/// pointer is turned into a ray, met with the patch's plane and projected back,
/// and that round trip does not return bit for bit -- which is how a drag
/// released on its own start came to report "moved by 0.000 units".
#[test]
fn a_patch_edit_that_changes_nothing_reports_no_effect() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    // The centre put back under the pixel it already projects to.
    let site = track.observations[0].site().expect("a sighting");
    let (moved, report) =
        translate_frame(&track, &edited, 0, site).expect("the sighting's own pixel");
    assert!(!report.changed, "the patch already sits there: {report:?}");
    assert_eq!(report.moved, 0.0);
    assert_eq!(moved, track, "a no-effect step gives the track back");

    // A resize repeated: the second drag names the size the first left.
    let target = [site[0] + 3.0, site[1] + 1.0];
    let (bigger, first) =
        resize_from_edge(&track, &edited, 0, Edge::PlusU, target).expect("a usable drag");
    assert!(first.changed);
    let (again, second) =
        resize_from_edge(&bigger, &edited, 0, Edge::PlusU, target).expect("the same drag");
    assert!(
        !second.changed,
        "the patch is already that size: {second:?}"
    );
    assert_eq!(again, bigger);

    // A turn under a nanoradian is no turn.
    let (still, turn) = rotate_frame(&track, 1e-12).expect("a finite angle");
    assert!(!turn.changed, "a nanoradian is not a gesture");
    assert_eq!(still, track);

    // A sighting put back within a thousandth of a pixel of where it sits.
    let nudged = [site[0] + 1e-6, site[1] - 1e-6];
    let (_, placed) =
        set_observation_keypoint(&track, &edited, 0, nudged).expect("a pixel on the sensor");
    assert!(
        !placed.changed,
        "the sighting already sits there: {placed:?}"
    );
}

/// The cluster stage's two hand edits, judged the same way.
#[test]
fn a_cluster_edit_that_changes_nothing_reports_no_effect() {
    let scene = Scene::new();
    let edited = edited_fixture(&scene, WORLD);
    let where_at = scene.project(0, WORLD);
    let (bench, report) =
        create_cluster(&Bench::new(), &pixel_seed(0, where_at)).expect("a usable seed");
    let track = track_of(&bench, &report.label);
    let shape = track.observations[0].shape().expect("a seed shape");

    let (same, report) = set_observation_shape(&track, 0, shape).expect("the shape it has");
    assert!(!report.changed, "the sighting already has that shape");
    assert_eq!(same, track);

    // An edge put back under the pixel it already reaches.
    let radius = track.cluster().expect("a cluster").radius;
    let edge = [
        where_at[0] + shape[0][0] * radius,
        where_at[1] + shape[1][0] * radius,
    ];
    let (held, report) =
        resize_from_edge(&track, &edited, 0, Edge::PlusU, edge).expect("the edge's own pixel");
    assert!(!report.changed, "the parallelogram is already that size");
    assert_eq!(held, track);
}

/// A painting that proposes the verdicts the track already carries moves
/// nothing, and the report says so rather than leaving a caller to compare two
/// tracks.
#[test]
fn a_painting_that_moves_no_verdict_reports_no_effect() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);

    let (painted, first) = apply_thresholds(&track);
    let (again, second) = apply_thresholds(&painted);
    assert!(!second.changed, "the second painting moved something");
    assert_eq!(again.observations, painted.observations);
    assert_eq!(first.changed, painted.observations != track.observations);
}

// ---- The photograph's own bounds -------------------------------------------

/// A pixel a pointer was dragged past the edge of the picture to, or a call
/// carrying two numbers of its own, names no place on the photograph: `(-500,
/// -500)` of a 128 px frame is no column and no row, and the patch whose centre
/// was slid to meet that pixel's ray was flung across the reconstruction. The
/// step takes the nearest place the photograph does name, and says that it did.
#[test]
fn a_pixel_off_the_photograph_is_brought_inside_it() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    let off = [-500.0, -500.0];

    let (_, report) = translate_frame(&track, &edited, 0, off).expect("a finite pixel");
    assert_eq!(
        report.clamped_from,
        Some(off),
        "the move did not say it clamped"
    );
    assert!(
        report.pixel[0] >= 0.0
            && report.pixel[0] < f64::from(IMG_W)
            && report.pixel[1] >= 0.0
            && report.pixel[1] < f64::from(IMG_H),
        "the centre landed at {:?}, off the photograph",
        report.pixel
    );

    let (_, report) = set_observation_keypoint(&track, &edited, 0, off).expect("a finite pixel");
    assert_eq!(report.clamped_from, Some(off));
    assert_eq!(report.pixel, [0.0, 0.0], "the nearest pixel is the corner");

    let (_, report) =
        resize_from_edge(&track, &edited, 0, Edge::PlusU, off).expect("a finite pixel");
    assert_eq!(report.clamped_from, Some(off));
    assert_eq!(report.pixel, Some([0.0, 0.0]));

    // And the far end is the largest pixel the sensor has rather than its width,
    // `[0, width)` being half open.
    let past = [f64::from(IMG_W) + 9.0, f64::from(IMG_H) + 9.0];
    let (_, report) = set_observation_keypoint(&track, &edited, 0, past).expect("a finite pixel");
    assert_eq!(report.clamped_from, Some(past));
    assert!(
        report.pixel[0] < f64::from(IMG_W) && report.pixel[1] < f64::from(IMG_H),
        "a pixel at the width names a column the sensor does not have: {:?}",
        report.pixel
    );

    // A pixel on the photograph is left exactly where it was named.
    let site = track.observations[0].site().expect("a sighting");
    let (_, report) = set_observation_keypoint(&track, &edited, 0, [site[0] + 2.0, site[1]])
        .expect("a pixel on the sensor");
    assert_eq!(report.clamped_from, None);
}

// ---- What a classification says when a candidate reprojects nowhere ---------

/// A candidate that reprojects into none of the sightings has no residual, and
/// the sentence says that rather than printing `NaN` as though it were a number
/// of pixels. The classification already treats it as no evidence.
#[test]
fn a_candidate_that_reprojects_nowhere_is_named_rather_than_printed_as_nan() {
    let call = TrackClassification {
        at_infinity: true,
        coordinate: Point3::new(0.0, 0.0, 1.0),
        reason: ClassificationReason::DepthUnresolved,
        condition_number: 1.0e6,
        inverse_depth_z: -22.30,
        inverse_depth_z_cutoff: 4.0,
        resolvable_distance: f64::NAN,
        finite_horizon: 1.0,
        max_pair_angle_deg: 0.01,
        finite_rms_px: f64::NAN,
        bearing_rms_px: 9.3,
        residual_margin: 0.8,
    };
    let said = call.to_string();
    assert!(!said.contains("NaN"), "{said}");
    assert!(
        said.contains("finite point reprojects into none of the sightings"),
        "{said}"
    );
    assert!(said.contains("9.3 px as a bearing"), "{said}");

    // The other side, and the overturned reasons, read the same way.
    let swapped = TrackClassification {
        finite_rms_px: 3.4,
        bearing_rms_px: f64::NAN,
        ..call
    };
    let said = swapped.to_string();
    assert!(!said.contains("NaN"), "{said}");
    assert!(
        said.contains("the bearing reprojects into none of the sightings"),
        "{said}"
    );

    let overturned = TrackClassification {
        reason: ClassificationReason::FiniteDoesNotExplainTheSightings,
        ..call
    };
    let said = overturned.to_string();
    assert!(!said.contains("NaN"), "{said}");
    assert!(said.contains("finite point would have"), "{said}");

    let neither = TrackClassification {
        finite_rms_px: f64::NAN,
        bearing_rms_px: f64::NAN,
        ..call
    };
    assert!(!neither.to_string().contains("NaN"), "{neither}");
}
