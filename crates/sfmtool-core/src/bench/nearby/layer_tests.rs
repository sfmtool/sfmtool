// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The depth layers, decided against the bench's synthetic capture: four
//! pinhole cameras looking down world `+z` at a textured plane four units out,
//! with candidates put along the queried pixel's ray at and off the plane.

use nalgebra::Point3;

use crate::bench::tests::scene::Scene;
use crate::bench::track_at_pixel::ViewCamera;

use super::layers::{layer_distances, rank_layers};
use super::*;

/// Four cameras a metre or so apart, all looking down world `+z`.
const CENTERS: [[f64; 3]; 4] = [
    [-0.5, -0.3, 0.0],
    [0.45, 0.25, 0.0],
    [0.1, -0.55, 0.0],
    [0.3, 0.4, 0.0],
];

/// The pixel the tests query in image 0: the middle of the photograph.
const PIXEL: [f64; 2] = [64.0, 64.0];

/// The plane's distance along world `z`.
const PLANE: f64 = 4.0;

/// A candidate's sightings and its place relative to the pixel, owned, so a
/// [`LayerCandidate`] can borrow them.
struct Owned {
    source: NearbySource,
    sightings: Vec<(u32, [f64; 2])>,
    distance_px: f64,
    range: [f64; 2],
    class: RangeClass,
}

impl Owned {
    fn candidate(&self) -> LayerCandidate<'_> {
        LayerCandidate {
            source: self.source,
            sightings: &self.sightings,
            distance_px: self.distance_px,
            max_ray_angle_deg: 10.0,
            range: self.range,
            class: self.class,
        }
    }
}

/// A candidate at the point `t` along the queried pixel's ray, sighted where
/// it lands in each of `images` (image 0 first), with its range and class.
fn at_distance(scene: &Scene, t: f64, images: &[u32], distance_px: f64) -> Owned {
    let views = scene.views();
    let camera = ViewCamera::new(&views[0]);
    let x = Point3::from(camera.center + camera.ray(PIXEL).normalize() * t);
    let sightings: Vec<(u32, [f64; 2])> = images
        .iter()
        .map(|&i| (i, scene.project(i as usize, x)))
        .collect();
    let range = distance_range(&views, 0, PIXEL, &sightings, t, 1.0).expect("valid");
    let class = classify_range(range, camera_spread(&views), &RangeOptions::default());
    Owned {
        source: NearbySource::Clusters,
        sightings,
        distance_px,
        range,
        class,
    }
}

/// A candidate with a range given outright, sighted in `images` at the pixel.
fn with_range(images: &[u32], range: [f64; 2], class: RangeClass) -> Owned {
    Owned {
        source: NearbySource::Guided,
        sightings: images.iter().map(|&i| (i, PIXEL)).collect(),
        distance_px: 0.0,
        range,
        class,
    }
}

fn run(scene: &Scene, owned: &[Owned], options: &DepthLayerOptions) -> DepthLayers {
    let views = scene.views();
    let grey = GreyImages::new(views.len());
    let candidates: Vec<LayerCandidate<'_>> = owned.iter().map(Owned::candidate).collect();
    depth_layers(&views, &grey, 0, PIXEL, &candidates, options).expect("valid")
}

const BOUNDED: RangeClass = RangeClass {
    bounded: true,
    far: false,
};

#[test]
fn two_surfaces_form_two_layers_nearest_first() {
    let scene = Scene::from_centers(&CENTERS, PLANE);
    // The plane's candidate first, the nearer one second: the layers are put
    // in order of distance, not of the list.
    let owned = [
        at_distance(&scene, PLANE, &[0, 1, 2, 3], 0.0),
        at_distance(&scene, PLANE / 2.0, &[0, 1, 2, 3], 0.0),
    ];
    assert!(owned.iter().all(|o| o.class.bounded));
    let found = run(&scene, &owned, &DepthLayerOptions::default());

    assert_eq!(found.layers.len(), 2);
    assert_eq!(found.layers[0].members, vec![1]);
    assert_eq!(found.layers[1].members, vec![0]);
    assert!(found.layers[0].range[1] < found.layers[1].range[0]);
    assert_eq!(found.layers[0].range, owned[1].range);
    // Neither supports the other: their ranges do not overlap.
    assert_eq!(found.support, vec![0, 0]);

    // The patch reads best at the plane, so its layer ranks first although it
    // is the further one, with the higher confidence.
    let near = found.layers[0].ranking.as_ref().expect("evidence is on");
    let plane = found.layers[1].ranking.as_ref().expect("evidence is on");
    assert!(plane.evidence.photo > 0.95, "{}", plane.evidence.photo);
    assert!(plane.evidence.photo > near.evidence.photo);
    assert_eq!((plane.rank, near.rank), (1, 2));
    assert!(plane.key > near.key);
    assert!(plane.confidence > near.confidence);
    // The three other images read the plane's layer well and the other badly,
    // so all three vote for it.
    assert_eq!(plane.evidence.votes, 3);
    assert_eq!(near.evidence.votes, 0);
    assert_eq!(plane.evidence.n_images, 4);
    assert!(plane.evidence.at_pixel);
    assert_eq!(plane.evidence.sources, vec![NearbySource::Clusters]);

    // Ranked by the score alone, the order is the same here.
    let by_score = run(
        &scene,
        &owned,
        &DepthLayerOptions {
            rank_by: LayerRankBy::Score,
            ..DepthLayerOptions::default()
        },
    );
    let ranks: Vec<usize> = by_score
        .layers
        .iter()
        .map(|l| l.ranking.as_ref().unwrap().rank)
        .collect();
    assert_eq!(ranks, vec![2, 1]);

    // Without the evidence, the layers are only grouped.
    let grouped = run(
        &scene,
        &owned,
        &DepthLayerOptions {
            evidence: false,
            ..DepthLayerOptions::default()
        },
    );
    assert_eq!(grouped.layers.len(), 2);
    assert!(grouped.layers.iter().all(|l| l.ranking.is_none()));
}

#[test]
fn overlapping_ranges_merge_into_one_layer() {
    let scene = Scene::from_centers(&CENTERS, PLANE);
    let owned = [
        at_distance(&scene, PLANE * 1.01, &[0, 1, 2], 3.0),
        at_distance(&scene, PLANE, &[0, 2, 3], 5.0),
    ];
    assert!(owned[0].range[0] < owned[1].range[1] && owned[1].range[0] < owned[0].range[1]);
    let found = run(&scene, &owned, &DepthLayerOptions::default());
    assert_eq!(found.layers.len(), 1);
    let layer = &found.layers[0];
    // Nearest range first.
    assert_eq!(layer.members, vec![1, 0]);
    assert_eq!(layer.range[0], owned[1].range[0]);
    assert_eq!(layer.range[1], owned[0].range[1].max(owned[1].range[1]));
    assert_eq!(layer.nearest_px, 3.0);
    assert_eq!(layer.max_views, 3);
    // Two readings on different photographs support each other.
    assert_eq!(found.support, vec![1, 1]);
    let r = layer.ranking.as_ref().unwrap();
    assert_eq!((r.rank, r.evidence.n_candidates), (1, 2));
    assert_eq!(r.evidence.n_independent, 2);
    assert_eq!(r.evidence.n_images, 4);
    assert!(!r.evidence.at_pixel);
}

#[test]
fn support_needs_different_photographs() {
    let scene = Scene::from_centers(&CENTERS, PLANE);
    let unusable = RangeClass::default();
    let owned = [
        // 0 and 1: one's images all among the other's.
        with_range(&[0, 1, 2, 3], [3.0, 5.0], BOUNDED),
        with_range(&[0, 1], [3.5, 4.5], BOUNDED),
        // 2: its own images, overlapping both.
        with_range(&[0, 2, 3, 1, 3], [3.8, 4.2], BOUNDED),
        // 3: different images, but a range the others do not reach.
        with_range(&[0, 2], [8.0, 9.0], BOUNDED),
        // 4: different images and an overlapping range, but not usable.
        with_range(&[0, 3], [3.0, 5.0], unusable),
        // 5: the same images as 1: a subset both ways.
        with_range(&[1, 0], [3.9, 4.1], BOUNDED),
    ];
    let found = run(
        &scene,
        &owned,
        &DepthLayerOptions {
            evidence: false,
            ..DepthLayerOptions::default()
        },
    );
    // None is supported: 1 and 5 rest on images 0 and 2 have, 2 has the same
    // images as 0 once its repeat is dropped, 3's range is apart and 4 is
    // not usable.
    assert_eq!(found.support, vec![0, 0, 0, 0, 0, 0]);

    let owned = [
        with_range(&[0, 1, 2], [3.0, 5.0], BOUNDED),
        with_range(&[0, 2, 3], [3.5, 4.5], BOUNDED),
        with_range(&[0, 1, 3], [4.4, 6.0], BOUNDED),
        with_range(&[0, 3], [3.0, 5.0], unusable),
    ];
    let found = run(
        &scene,
        &owned,
        &DepthLayerOptions {
            evidence: false,
            ..DepthLayerOptions::default()
        },
    );
    assert_eq!(found.support, vec![2, 2, 2, 0]);
    // The unusable one is in no layer.
    assert_eq!(found.layers.len(), 1);
    assert_eq!(found.layers[0].members, vec![0, 1, 2]);
    assert_eq!(found.layers[0].range, [3.0, 6.0]);
}

#[test]
fn equal_near_ends_keep_the_order_given() {
    let scene = Scene::from_centers(&CENTERS, PLANE);
    let owned = [
        with_range(&[0, 1], [3.0, 3.5], BOUNDED),
        with_range(&[0, 2], [3.0, 3.2], BOUNDED),
        with_range(&[0, 3], [1.0, 2.0], BOUNDED),
    ];
    let found = run(
        &scene,
        &owned,
        &DepthLayerOptions {
            evidence: false,
            ..DepthLayerOptions::default()
        },
    );
    let members: Vec<Vec<usize>> = found.layers.iter().map(|l| l.members.clone()).collect();
    assert_eq!(members, vec![vec![2], vec![0, 1]]);
}

#[test]
fn a_far_layer_is_read_from_infinity_in() {
    let d = layer_distances([10.0, f64::INFINITY], 5);
    assert_eq!(d[0], f64::INFINITY);
    assert_eq!(d[4], 10.0);
    // Even in inverse distance.
    for (k, &t) in d.iter().enumerate().skip(1) {
        assert!((1.0 / t - 0.025 * k as f64).abs() < 1e-15, "{t}");
    }
    let d = layer_distances([2.0, 4.0], 3);
    assert_eq!(d, vec![4.0, 1.0 / (0.125 + 0.25), 2.0]);
    assert_eq!(layer_distances([2.0, 4.0], 1), vec![4.0]);
    assert_eq!(layer_distances([0.0, f64::INFINITY], 3).len(), 3);
}

fn evidence(photo: f64, votes: usize, weight: f64, nearest_px: f64) -> LayerEvidence {
    LayerEvidence {
        n_candidates: 1,
        n_independent: 1,
        n_images: 2,
        max_views: 2,
        max_ray_angle_deg: 5.0,
        nearest_px,
        at_pixel: nearest_px <= 1.0,
        sources: vec![NearbySource::Points],
        weight,
        photo,
        photo_middle: photo,
        photo_both: photo,
        votes,
        votes_all: votes,
    }
}

#[test]
fn a_clear_winner_is_more_confident_than_a_close_one() {
    let clear = rank_layers(
        vec![evidence(0.95, 3, 2.0, 0.0), evidence(0.4, 0, 2.0, 0.0)],
        LayerRankBy::Key,
    );
    let close = rank_layers(
        vec![evidence(0.95, 3, 2.0, 0.0), evidence(0.93, 0, 2.0, 0.0)],
        LayerRankBy::Key,
    );
    assert_eq!((clear[0].rank, clear[1].rank), (1, 2));
    assert_eq!((close[0].rank, close[1].rank), (1, 2));
    assert!(clear[0].confidence > close[0].confidence);
    assert!(close[0].confidence > close[1].confidence);

    // The key, and the confidence as its logistic, from the constants.
    let k = &clear[0];
    assert_eq!(k.score, 0.95);
    assert!((k.key - 1.9).abs() < 1e-15);
    let margin = k.key - clear[1].key;
    let z =
        CONF_BIAS + CONF_MARGIN * margin + CONF_VOTES * 3f64.ln_1p() + CONF_SUPPORT * 2f64.ln_1p();
    assert!((k.confidence - 1.0 / (1.0 + (-z).exp())).abs() < 1e-15);

    // Nearness to the pixel counts against a layer found away from it.
    let off = rank_layers(
        vec![evidence(0.9, 0, 1.0, 30.0), evidence(0.9, 0, 1.0, 0.0)],
        LayerRankBy::Key,
    );
    assert_eq!((off[0].rank, off[1].rank), (2, 1));

    // A tie keeps the layers' order, and a lone layer's margin is 1.
    let tie = rank_layers(
        vec![evidence(0.8, 0, 1.0, 0.0), evidence(0.8, 0, 1.0, 0.0)],
        LayerRankBy::Key,
    );
    assert_eq!((tie[0].rank, tie[1].rank), (1, 2));
    let lone = rank_layers(vec![evidence(0.8, 0, 0.0, 0.0)], LayerRankBy::Key);
    let z = CONF_BIAS + CONF_MARGIN;
    assert!((lone[0].confidence - 1.0 / (1.0 + (-z).exp())).abs() < 1e-15);
}

#[test]
fn the_layers_refuse_an_image_that_is_not_there() {
    let scene = Scene::from_centers(&CENTERS, PLANE);
    let views = scene.views();
    let options = DepthLayerOptions::default();
    let grey = GreyImages::new(views.len());
    assert!(matches!(
        depth_layers(&views, &grey, 9, PIXEL, &[], &options),
        Err(DepthLayerError::NoSuchImage { image: 9, .. })
    ));
    assert!(matches!(
        depth_layers(&views, &GreyImages::new(2), 0, PIXEL, &[], &options),
        Err(DepthLayerError::GreyMismatch { grey: 2, .. })
    ));
    let none = depth_layers(&views, &grey, 0, PIXEL, &[], &options).expect("valid");
    assert!(none.layers.is_empty() && none.support.is_empty());
}
