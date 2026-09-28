// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Finding the nearby tracks, decided against the bench's synthetic capture:
//! pinhole cameras looking down world `+z` at a textured plane, with a grid of
//! points on it, one held out and queried at its pixel as the harness does; a
//! cluster on the plane; and the plane moved out to where the photographs
//! cannot tell it from infinity.

use std::sync::Arc;

use nalgebra::Point3;

use crate::bench::commit::commit;
use crate::bench::tests::scene::{edited as scene_edited, fixture_points, Scene, PLANE_Z};
use crate::bench::track::StageKind;
use crate::bench::track_at_pixel::tests::matches_file;
use crate::bench::track_at_pixel::{MatchesClusters, STATUS_KEPT, STATUS_REFERENCE};
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::*;

/// The grid's point at the middle, which the queries hold out.
const HELD_OUT: u32 = 12;

/// A five-by-five grid of points on the plane, 0.3 apart.
fn grid() -> Vec<Point3<f64>> {
    (-2..=2)
        .flat_map(|j| (-2..=2).map(move |i| Point3::new(0.3 * i as f64, 0.3 * j as f64, PLANE_Z)))
        .collect()
}

/// The grid as a version with the middle point deleted, and where that point
/// sits in image 0.
fn held_out(scene: &Scene) -> (EditedReconstruction, [f64; 2]) {
    let mut edited = EditedReconstruction::new(Arc::new(fixture_points(scene, &grid())));
    edited.delete_point(HELD_OUT).expect("the point is live");
    (edited, scene.project(0, grid()[HELD_OUT as usize]))
}

fn find(
    edited: &EditedReconstruction,
    scene: &Scene,
    sources: &NearbyTrackSources<'_>,
    pixel: [f64; 2],
    options: &NearbyTrackOptions,
) -> NearbyTracks {
    let views = scene.views();
    let grey = GreyImages::new(views.len());
    find_nearby_tracks(
        edited,
        &views,
        &grey,
        sources,
        0,
        pixel,
        options,
        &Progress::none(),
    )
    .expect("the query is valid")
}

#[test]
fn the_points_near_the_pixel_come_back_as_existing_points_on_one_layer() {
    let scene = Scene::new();
    let (edited, pixel) = held_out(&scene);
    let options = NearbyTrackOptions {
        stop: StopRule::Never,
        ..NearbyTrackOptions::default()
    };
    let found = find(
        &edited,
        &scene,
        &NearbyTrackSources::default(),
        pixel,
        &options,
    );

    // The eight grid points around the held-out one, all on the plane.
    assert_eq!(found.tracks.len(), 8);
    assert!(found
        .tracks
        .iter()
        .all(|t| t.source() == NearbySource::Points
            && t.point.is_some()
            && t.track.is_none()
            && t.usable()));
    assert_eq!(found.layers.len(), 1);
    let layer = &found.layers[0];
    assert_eq!(layer.ranking.as_ref().expect("evidence is on").rank, 1);
    assert!(layer.range[0] < PLANE_Z && PLANE_Z < layer.range[1]);
    // Each is supported by the others that are not seen in the same images:
    // every point is seen in all three, so none supports another.
    assert!(found.tracks.iter().all(|t| t.support == 0));

    // The labels run by distance from the pixel within the layer, each naming
    // its point; the nearest is `1a`.
    let order = found.bench_order();
    assert_eq!(order.len(), 8);
    assert!(order
        .windows(2)
        .all(|w| found.tracks[w[0]].distance_px() <= found.tracks[w[1]].distance_px()));
    let first = &found.tracks[order[0]];
    assert_eq!(first.order, Some(0));
    assert_eq!(
        first.label.as_deref(),
        Some(
            format!(
                "image_0@{},{} 1a pt {}",
                pixel[0].round(),
                pixel[1].round(),
                first.point.unwrap()
            )
            .as_str()
        )
    );
    let last = &found.tracks[order[7]];
    assert!(last.label.as_ref().unwrap().contains(" 1h pt "));

    // The sources with no input are skipped and named; the far-field sweep
    // runs, since no track sits at the pixel.
    let report = &found.report;
    let names: Vec<(NearbySource, Option<&str>)> = report
        .sources
        .iter()
        .map(|s| (s.source, s.skipped))
        .collect();
    assert_eq!(
        names,
        vec![
            (NearbySource::Points, None),
            (NearbySource::Clusters, Some("clusters")),
            (NearbySource::Guided, Some("descriptors")),
            (NearbySource::Constellation, Some("SIFT index")),
        ]
    );
    assert_eq!(report.sources[0].found, 8);
    assert_eq!(report.stopped_after, None);
    assert_eq!(
        report.far_field.as_ref().map(|f| f.trigger),
        Some(FarFieldTrigger::NoneAtPixel)
    );
}

#[test]
fn the_stopping_rule_stops_after_the_first_source_with_enough() {
    let scene = Scene::new();
    let (edited, pixel) = held_out(&scene);
    let found = find(
        &edited,
        &scene,
        &NearbyTrackSources::default(),
        pixel,
        &NearbyTrackOptions::default(),
    );
    // Eight points lie within 20 px, more than the two the rule asks for.
    assert_eq!(found.report.stopped_after, Some(NearbySource::Points));
    assert_eq!(found.report.sources.len(), 1);

    // Asking for more than there are runs every source.
    let options = NearbyTrackOptions {
        enough_count: 9,
        ..NearbyTrackOptions::default()
    };
    let found = find(
        &edited,
        &scene,
        &NearbyTrackSources::default(),
        pixel,
        &options,
    );
    assert_eq!(found.report.stopped_after, None);
    assert_eq!(found.report.sources.len(), 4);
}

/// Four cameras a metre or so apart, all looking down world `+z`.
const FOUR: [[f64; 3]; 4] = [
    [-0.5, -0.3, 0.0],
    [0.45, 0.25, 0.0],
    [0.1, -0.55, 0.0],
    [0.3, 0.4, 0.0],
];

#[test]
fn a_cluster_at_the_pixel_becomes_a_fitted_track_that_commits() {
    let scene = Scene::from_centers(&FOUR, PLANE_Z);
    let world = Point3::new(0.05, -0.1, PLANE_Z);
    let member = |image: u32, status: u8| {
        let p = scene.project(image as usize, world);
        (image, [p[0] as f32, p[1] as f32], status)
    };
    let names: Vec<String> = (0..4).map(|i| format!("image_{i}.jpg")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let clusters = MatchesClusters::new(
        &matches_file(
            &names,
            &[vec![
                member(1, STATUS_REFERENCE),
                member(0, STATUS_KEPT),
                member(2, STATUS_KEPT),
                member(3, STATUS_KEPT),
            ]],
            1.0,
            true,
        ),
        &names,
    )
    .expect("clusters");
    // A reconstruction of the four cameras whose one point is elsewhere.
    let edited = scene_edited(&scene, Point3::new(0.9, 0.9, PLANE_Z));
    let sources = NearbyTrackSources {
        clusters: Some(&clusters),
        ..NearbyTrackSources::default()
    };
    let pixel = scene.project(0, world);
    let options = NearbyTrackOptions {
        label: Some("here".into()),
        ..NearbyTrackOptions::default()
    };
    let found = find(&edited, &scene, &sources, pixel, &options);

    let [t] = found.tracks.as_slice() else {
        panic!("one track, got {:?}", found.tracks.len());
    };
    assert_eq!(t.source(), NearbySource::Clusters);
    assert!(t.class.bounded);
    assert_eq!(t.label.as_deref(), Some("here 1a"));
    assert_eq!(found.group_label, "here");
    // The cluster sits at the pixel in one layer, so the far field is not
    // needed.
    assert!(found.report.far_field.is_none());
    let track = t
        .track
        .as_ref()
        .expect("a usable candidate gets a track")
        .as_ref()
        .expect("the track builds");
    assert_eq!(track.stage.kind(), StageKind::Track);
    assert_eq!(track.observations.len(), 4);
    assert_eq!(track.observations[0].image, 0);
    let position = track
        .track()
        .and_then(|p| p.position)
        .expect("the upgrade triangulates");
    assert!((position - world).norm() < 1e-2, "{position}");
    assert!(found.report.tracks_seconds > 0.0);

    // It commits as a new point.
    let (next, report) = commit(&edited, track).expect("the track commits");
    assert_eq!(next.point_count(), edited.point_count() + 1);
    assert!(report.changed);

    // With building off, the track is found and labelled but not built.
    let options = NearbyTrackOptions {
        tracks: BenchTrackOptions {
            build: false,
            ..BenchTrackOptions::default()
        },
        ..NearbyTrackOptions::default()
    };
    let found = find(&edited, &scene, &sources, pixel, &options);
    assert!(found.tracks[0].track.is_none());
    assert!(found.tracks[0].label.is_some());
}

#[test]
fn a_far_pixel_gets_a_far_field_reading_when_the_sources_leave_it_open() {
    let depth = 1e7;
    let scene = Scene::from_centers(&FOUR, depth);
    let edited = scene_edited(&scene, Point3::new(0.0, 0.0, depth));
    let pixel = [64.0, 64.0];
    let options = NearbyTrackOptions {
        sources: Vec::new(),
        ..NearbyTrackOptions::default()
    };
    let found = find(
        &edited,
        &scene,
        &NearbyTrackSources::default(),
        pixel,
        &options,
    );

    let run = found.report.far_field.as_ref().expect("the sweep ran");
    assert_eq!(run.trigger, FarFieldTrigger::NoLayer);
    assert_eq!(run.report.found, 1);
    let [t] = found.tracks.as_slice() else {
        panic!("one reading, got {}", found.tracks.len());
    };
    assert_eq!(t.source(), NearbySource::FarField);
    assert_eq!(t.distance, f64::INFINITY);
    assert!(t.class.far);
    let NearbyFinding::FarField(reading) = &t.finding else {
        panic!("a far-field reading");
    };
    // It keeps the range the sweep gave it.
    assert_eq!(Some(t.range), reading.range);
    assert_eq!(t.layer, Some(0));
    assert!(t.label.as_ref().unwrap().ends_with(" 1a"));
    assert!(matches!(t.track, Some(Ok(_))));

    // Told never to run, the sweep does not, and nothing is found.
    let options = NearbyTrackOptions {
        far_field_when: FarFieldWhen::Never,
        ..options
    };
    let found = find(
        &edited,
        &scene,
        &NearbyTrackSources::default(),
        pixel,
        &options,
    );
    assert!(found.report.far_field.is_none());
    assert!(found.tracks.is_empty());
    assert!(found.layers.is_empty());
}

#[test]
fn a_query_that_names_no_place_or_a_far_field_source_is_refused() {
    let scene = Scene::new();
    let (edited, _) = held_out(&scene);
    let views = scene.views();
    let grey = GreyImages::new(views.len());
    let run = |image: u32, pixel: [f64; 2], options: &NearbyTrackOptions| {
        find_nearby_tracks(
            &edited,
            &views,
            &grey,
            &NearbyTrackSources::default(),
            image,
            pixel,
            options,
            &Progress::none(),
        )
    };
    let options = NearbyTrackOptions::default();
    assert!(matches!(
        run(7, [10.0, 10.0], &options),
        Err(NearbyTracksError::NoSuchImage { image: 7, .. })
    ));
    assert!(matches!(
        run(0, [-1.0, 10.0], &options),
        Err(NearbyTracksError::PixelOffImage { .. })
    ));
    let options = NearbyTrackOptions {
        sources: vec![NearbySource::FarField],
        ..NearbyTrackOptions::default()
    };
    assert_eq!(
        run(0, [10.0, 10.0], &options),
        Err(NearbyTracksError::NotAMatchingSource(
            NearbySource::FarField
        ))
    );
}

#[test]
fn labels_name_the_group_the_layer_the_order_and_the_point() {
    assert_eq!(
        nearby_group_label("frame_13", [412.4, 229.6]),
        "frame_13@412,230"
    );
    let g = "frame_13@412,230";
    assert_eq!(nearby_track_label(g, 1, 0, None), "frame_13@412,230 1a");
    assert_eq!(nearby_track_label(g, 1, 1, None), "frame_13@412,230 1b");
    assert_eq!(nearby_track_label(g, 2, 0, None), "frame_13@412,230 2a");
    assert_eq!(
        nearby_track_label(g, 1, 0, Some(812)),
        "frame_13@412,230 1a pt 812"
    );
    assert_eq!(nearby_track_label("g", 3, 25, None), "g 3z");
    assert_eq!(nearby_track_label("g", 3, 26, None), "g 3aa");
    assert_eq!(nearby_track_label("g", 3, 27, None), "g 3ab");
    assert_eq!(nearby_track_label("g", 3, 26 + 26 * 26, None), "g 3aaa");
}
