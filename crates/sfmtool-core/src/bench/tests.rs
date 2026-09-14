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

use crate::reconstruction::add_observation::tests::{
    edited as edited_fixture, fixture_with_columns, Scene, WORLD,
};
use crate::reconstruction::edited::EditedReconstruction;
use crate::reconstruction::SfmrReconstruction;

use super::*;

/// The bitmap edge the column fixture is built with.
const BITMAP_R: usize = 8;

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
    let (bench, label) = four_observation_cluster();
    let images: Vec<u32> = bench
        .track(&label)
        .expect("just built")
        .observations
        .iter()
        .map(|o| o.image)
        .collect();
    assert_eq!(images, [0, 1, 2, 3]);

    let (bench, report) = split(&bench, &label, &[1, 3]).expect("two of four");
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
    let (bench, label) = four_observation_cluster();
    // Observation 0 is the reference; move it to the second half.
    let (bench, report) = split(&bench, &label, &[0]).expect("one of four");
    let first = bench.track(&label).expect("still on");
    let second = bench.track(&report.label).expect("just put on");
    assert_eq!(first.cluster().expect("a cluster").reference, 0);
    assert_eq!(first.observations[0].image, 1);
    assert_eq!(second.cluster().expect("a cluster").reference, 0);
    assert_eq!(second.observations[0].image, 0);
}

#[test]
fn a_split_of_nothing_or_of_everything_is_refused() {
    let (bench, label) = four_observation_cluster();
    assert_eq!(
        split(&bench, &label, &[]).expect_err("nothing named"),
        SplitError::NoObservations
    );
    assert_eq!(
        split(&bench, &label, &[0, 1, 2, 3]).expect_err("all four named"),
        SplitError::EveryObservation(4)
    );
    assert_eq!(
        split(&bench, &label, &[0, 9]).expect_err("there is no observation 9"),
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
    let created = match report.outcome {
        CommitOutcome::Created(index) => index,
        other => panic!("expected a creation, got {other:?}"),
    };
    assert_eq!(created, edited.base_point_count() as u32);
    assert_eq!(report.observation_count, 2);
    assert!(report.absorbed.is_empty());
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
    assert_eq!(
        report.outcome,
        CommitOutcome::Replaced {
            point: edited.base_point_count() as u32,
            replaced: 0
        }
    );
    assert!(next.point(0).is_none(), "the origin's index is spent");
    assert_eq!(next.point_count(), edited.point_count());
    assert_eq!(
        report.label("bull"),
        "Committed track: 2 observations in bull, replacing point 0"
    );

    // Re-seating the track on what was written makes a second commit a
    // replacement of the first.
    let settled = track.with_origin(1, report.outcome.point());
    let (after, second) = commit(&next, &settled).expect("still two in");
    assert!(matches!(second.outcome, CommitOutcome::Replaced { .. }));
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
    assert!(matches!(report.outcome, CommitOutcome::Created(_)));
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
    assert_eq!(report.absorbed, vec![other]);
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
    assert!(report.absorbed.is_empty());
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
    let written = next.point(report.outcome.point()).expect("just written");
    assert_eq!(written.point().color, [10, 20, 30]);
}
