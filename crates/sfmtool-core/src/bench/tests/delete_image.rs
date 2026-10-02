// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What deleting an image from the reconstruction does to the bench: the
//! observations in it are dropped, those in later images move down by one, and
//! a track left with none is discarded.

use std::sync::Arc;

use ndarray::Array3;

use super::super::*;

const SHAPE: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

/// A cluster with one observation in each of `images`, the reference on
/// `reference` and a template cut.
fn cluster(images: &[u32], reference: usize) -> EditableTrack {
    let mut track = EditableTrack::empty_cluster();
    for (i, &image) in images.iter().enumerate() {
        let mut observation =
            Observation::seeded(image, Provenance::Pixel, [10.0 + i as f64, 20.0], SHAPE);
        observation.verdict = Verdict::In;
        track.observations.push(observation);
    }
    if let Stage::Cluster(payload) = &mut track.stage {
        payload.reference = reference;
        payload.template = Some(ClusterTemplate {
            samples: Array3::zeros((4, 4, 1)),
        });
    }
    track
}

fn put(bench: &Bench, label: &str, track: EditableTrack) -> Bench {
    bench.put(label, BenchItem::Track(Arc::new(track))).0
}

fn images(track: &EditableTrack) -> Vec<u32> {
    track.observations.iter().map(|o| o.image).collect()
}

#[test]
fn observations_in_later_images_move_down_and_the_deleted_image_is_dropped() {
    let track = cluster(&[0, 2, 3, 5], 2);
    let (next, map) = track.delete_image(2).expect("the track observes image 2");
    assert_eq!(images(&next), vec![0, 2, 4]);
    assert_eq!(map, vec![Some(0), None, Some(1), Some(2)]);
    // Each kept observation is the same sighting, at the same pixel.
    assert_eq!(next.observations[1].site(), track.observations[2].site());
    assert_eq!(next.observations[2].site(), track.observations[3].site());
    // The reference follows its observation, and its template is still its own.
    let payload = next.cluster().expect("a cluster");
    assert_eq!(payload.reference, 1);
    assert!(payload.template.is_some());
}

#[test]
fn a_track_that_observes_no_image_at_or_past_the_deleted_one_is_unchanged() {
    let track = cluster(&[0, 1], 0);
    assert!(track.delete_image(2).is_none());
}

#[test]
fn a_dropped_reference_is_reseated_and_its_template_dropped() {
    let track = cluster(&[1, 3, 4], 1);
    let (next, _) = track.delete_image(3).expect("the track observes image 3");
    let payload = next.cluster().expect("a cluster");
    assert_eq!(payload.reference, 0);
    assert!(payload.template.is_none());
}

#[test]
fn the_bench_discards_a_track_left_with_no_observations_and_shares_the_rest() {
    let bench = Bench::new();
    let bench = put(&bench, "before", cluster(&[0, 1], 0));
    let bench = put(&bench, "across", cluster(&[1, 2, 4], 0));
    let bench = put(&bench, "only", cluster(&[2, 2], 0));
    let bench = put(&bench, "empty", EditableTrack::empty_cluster());

    let (next, report) = bench.delete_image(2);

    assert_eq!(
        next.labels().collect::<Vec<_>>(),
        vec!["before", "across", "empty"]
    );
    assert_eq!(report.discarded, vec!["only".to_string()]);
    assert_eq!(report.dropped, 3);
    assert!(report.changed());
    // The untouched tracks are the same values, and every item keeps its ID.
    for label in ["before", "empty"] {
        assert!(Arc::ptr_eq(
            bench.track(label).unwrap(),
            next.track(label).unwrap()
        ));
        assert_eq!(bench.id(label), next.id(label));
    }
    let across = bench.id("across").unwrap();
    assert_eq!(next.id("across"), Some(across));
    assert_eq!(images(next.track("across").unwrap()), vec![1, 3]);
    assert_eq!(
        report.observation_map(across),
        Some(&[Some(0), None, Some(1)][..])
    );
    assert_eq!(report.observation_map(bench.id("before").unwrap()), None);
}

#[test]
fn deleting_an_image_no_track_reaches_changes_nothing() {
    let bench = put(&Bench::new(), "before", cluster(&[0, 1], 0));
    let (next, report) = bench.delete_image(5);
    assert!(!report.changed());
    assert_eq!(report.dropped, 0);
    assert_eq!(next, bench);
}
