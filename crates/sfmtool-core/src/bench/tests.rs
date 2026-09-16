// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench and the editable track: the labels, the verdicts, the split, and
//! what a commit writes.
//!
//! The reconstruction under the commit tests is the synthetic textured-plane
//! scene the add-observation tests build -- pinhole cameras looking down world
//! `+z` at a plane, wrapped in an `embedded_patches` value whose stored
//! keypoints are the exact projections -- so what a commit should have written
//! is known to the pixel. Nothing here decodes a photograph: every step in this
//! slice is decided by what the reconstruction and the person already say.

use std::sync::Arc;

use nalgebra::Point3;
use ndarray::Array3;

use crate::patch::cloud::OrientedPatch;
use crate::patch::cluster_refine::{sample_member_grid, ClusterRefineParams, MemberStatus};
use crate::patch::keypoint_localize::{localize_patch_keypoints, KeypointLocalization};
use crate::patch::keypoint_subpixel::{refine_patch_keypoints, KeypointRefinement};
use crate::progress::Progress;
use crate::reconstruction::add_observation::tests::{
    edited as edited_fixture, fixture_with_columns, Scene, WORLD,
};
use crate::reconstruction::edited::{EditedReconstruction, PointMap};
use crate::reconstruction::SfmrReconstruction;

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
        measurement.shift_px = Some(0.5);
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
    // replacement of the first.
    let settled = track.with_origin(1, report.point);
    let (after, second) = commit(&next, &settled).expect("still two in");
    assert_eq!(second.replaced, Some(report.point));
    assert_eq!(after.point_count(), edited.point_count());
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
fn a_track_with_no_position_refuses_and_names_the_evaluation() {
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
        "the track has no position; evaluate it before committing"
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
    let options = EvaluateOptions::default();
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
        &EvaluateOptions::default(),
        &Progress::none(),
    )
}

#[test]
fn a_track_from_a_point_evaluates_to_the_kernels_own_numbers() {
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

    let (measured, report) = evaluate_over(&scene, &edited, &track).expect("two observations in");
    assert_eq!(report.stage, StageKind::Track);
    assert_eq!((report.measured, report.unmeasured), (2, 0));

    // The same two kernels, called straight on the same frame and seeds.
    let (localized, refined) = fit_directly(&scene, &frame, &[0, 1], &seeds);
    for (slot, &image) in localized.views.iter().enumerate() {
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
        assert_eq!(m.zncc, Some(localized.loo_zncc[slot]), "image {image}");
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
    assert!(scored.shift_px.expect("a drift") < 3.0);

    // The same gesture, pointed at nothing: the pixel is off the sensor, so no
    // round places it and the row stays unmeasured.
    let (nowhere, added) =
        add_observation(&track, &ObservationSeed::at_pixel(2, [10_000.0, 10_000.0]))
            .expect("a finite pixel");
    let at = added.observation;
    let (nowhere, report) = evaluate_over(&scene, &edited, &nowhere).expect("two observations in");
    assert_eq!((report.measured, report.unmeasured), (2, 1));
    let unscored = nowhere.observations[at].track.as_ref();
    assert!(
        unscored.is_none_or(|m| m.zncc.is_none()),
        "nothing registered there"
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
        .evaluate
        .expect("an upgrade evaluates")
        .position
        .expect("a triangulation");
    assert!(
        (position - WORLD).norm() < 0.05,
        "the round trip landed at {position}"
    );
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
fn a_track_stage_evaluation_of_one_sighting_is_refused() {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (track, _) = set_verdict(&track_of(&bench, &label), 1, Verdict::Out).expect("a live row");
    assert_eq!(
        evaluate_over(&scene, &edited, &track).expect_err("one sighting fixes no point"),
        EvaluateError::TooFewObservations(1)
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

    // One sighting: neither the track-stage evaluation nor the upgrade to it
    // has a consensus to register against.
    assert_eq!(
        evaluate_preconditions(&one_in).expect_err("one sighting fixes no point"),
        EvaluateError::TooFewObservations(1)
    );
    assert_eq!(
        evaluate_over(&scene, &edited, &one_in).expect_err("the step agrees"),
        EvaluateError::TooFewObservations(1)
    );
    let (cluster, _) =
        stage_over(&scene, &edited, &whole, StageKind::Cluster).expect("a framed track goes down");
    let (one_in_cluster, _) = set_verdict(&cluster, 1, Verdict::Out).expect("a live row");
    assert_eq!(
        set_stage_preconditions(&one_in_cluster, StageKind::Track)
            .expect_err("one sighting triangulates nothing"),
        StageError::Evaluate(EvaluateError::TooFewObservations(1))
    );
    assert_eq!(
        stage_over(&scene, &edited, &one_in_cluster, StageKind::Track)
            .expect_err("the step agrees"),
        StageError::Evaluate(EvaluateError::TooFewObservations(1))
    );

    // A whole track at the stage it is asked for is the change that does
    // nothing, which is not a refusal.
    assert_eq!(set_stage_preconditions(&whole, StageKind::Track), Ok(()));
    assert_eq!(evaluate_preconditions(&whole), Ok(()));

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
        stage_over(&scene, &edited, &track, StageKind::Cluster)
            .expect_err("there is no frame to project"),
        StageError::NoFrame
    );
}
