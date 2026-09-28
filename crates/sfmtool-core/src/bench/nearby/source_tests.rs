// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The matching sources, decided against the bench's synthetic capture:
//! pinhole cameras looking down world `+z` at a textured plane, with a grid of
//! points on it that every camera sees at its exact projection. One point is
//! deleted from the version and the query is made at its pixel, the way the
//! harness holds a point out.

use std::sync::Arc;

use nalgebra::Point3;

use crate::bench::tests::scene::{fixture_points, Scene, PLANE_Z};
use crate::reconstruction::data::{ObservationSource, SfmrReconstruction};
use crate::reconstruction::edited::EditedReconstruction;

use super::*;

/// The grid's point at the middle, which the queries hold out.
const HELD_OUT: u32 = 12;

/// A five-by-five grid of points on the plane, 0.3 apart, which all three of
/// the scene's cameras see.
fn grid() -> Vec<Point3<f64>> {
    (-2..=2)
        .flat_map(|j| (-2..=2).map(move |i| Point3::new(0.3 * i as f64, 0.3 * j as f64, PLANE_Z)))
        .collect()
}

/// `recon` as a version with the middle point deleted.
fn held_out(recon: SfmrReconstruction) -> EditedReconstruction {
    let mut edited = EditedReconstruction::new(Arc::new(recon));
    edited.delete_point(HELD_OUT).expect("the point is live");
    edited
}

/// Where the held-out point sits in image 0.
fn held_out_pixel(scene: &Scene) -> [f64; 2] {
    scene.project(0, grid()[HELD_OUT as usize])
}

// ---- The reconstruction's own points -----------------------------------------

#[test]
fn the_points_observed_near_the_pixel_are_found_nearest_first() {
    let scene = Scene::new();
    let edited = held_out(fixture_points(&scene, &grid()));
    let views = scene.views();
    let pixel = held_out_pixel(&scene);
    let found = nearby_points(&edited, &views, 0, pixel, &PointsOptions::default())
        .expect("the query is valid");

    // The grid is 12 px apart in image 0: the eight around the held-out point
    // are within 40 px, and the cap keeps eight.
    assert_eq!(found.len(), 8);
    assert!(found.iter().all(|c| c.point != Some(HELD_OUT)));
    assert!(found
        .windows(2)
        .all(|w| w[0].distance_px <= w[1].distance_px));
    let first = &found[0];
    assert!((first.distance_px - 12.0).abs() < 1e-3);
    assert_eq!(first.source, NearbySource::Points);
    assert_eq!(first.id, first.point);
    let point = grid()[first.point.unwrap() as usize];
    assert!((first.position - point.coords).norm() < 1e-12);
    // Every camera sees it, the queried image's sighting first, at the pixel
    // the candidate sits at.
    assert_eq!(first.n_views(), 3);
    assert_eq!(first.image(), 0);
    assert_eq!(first.sightings[0].1, first.query_pixel);
    assert_eq!(first.errors_px.len(), 3);
    assert!(first.max_reproj_px < 1e-3);
    assert!((first.depth - PLANE_Z).abs() < 1e-9);
    assert!(first.max_ray_angle_deg > 5.0);
    // Its range holds its own distance.
    let t = first.ray_distance(&views);
    let [near, far] = first.range(&views, 1.0).expect("image 0 is a view");
    assert!(near < t && t < far);

    // A smaller radius or cap keeps fewer.
    let options = PointsOptions {
        radius_px: 13.0,
        ..PointsOptions::default()
    };
    let four = nearby_points(&edited, &views, 0, pixel, &options).expect("valid");
    assert_eq!(four.len(), 4);
    let options = PointsOptions {
        max_points: 2,
        ..PointsOptions::default()
    };
    assert_eq!(
        nearby_points(&edited, &views, 0, pixel, &options)
            .expect("valid")
            .len(),
        2
    );
}

#[test]
fn a_point_one_photograph_puts_elsewhere_is_left_out() {
    let scene = Scene::new();
    let mut recon = fixture_points(&scene, &grid());
    // The point just right of the held-out one, 3 px off in image 1.
    let moved = HELD_OUT + 1;
    let n = scene.len();
    match &mut recon.point_set.observations {
        ObservationSource::EmbeddedPatches { keypoints_xy, .. } => {
            keypoints_xy[[moved as usize * n + 1, 0]] += 3.0;
        }
        _ => unreachable!("the fixture embeds its keypoints"),
    }
    let edited = held_out(recon);
    let views = scene.views();
    let pixel = held_out_pixel(&scene);
    let options = PointsOptions {
        radius_px: 13.0,
        ..PointsOptions::default()
    };
    let found = nearby_points(&edited, &views, 0, pixel, &options).expect("valid");
    assert_eq!(found.len(), 3);
    assert!(found.iter().all(|c| c.point != Some(moved)));

    // A looser bar lets it back in, with its error.
    let options = PointsOptions {
        max_reproj_px: 4.0,
        ..options
    };
    let found = nearby_points(&edited, &views, 0, pixel, &options).expect("valid");
    let back = found
        .iter()
        .find(|c| c.point == Some(moved))
        .expect("within the looser bar");
    assert!((back.max_reproj_px - 3.0).abs() < 1e-3);
}

#[test]
fn points_with_too_few_observations_are_left_out() {
    let scene = Scene::new();
    let edited = held_out(fixture_points(&scene, &grid()));
    let views = scene.views();
    let options = PointsOptions {
        min_views: 4,
        ..PointsOptions::default()
    };
    let found = nearby_points(&edited, &views, 0, held_out_pixel(&scene), &options).expect("valid");
    assert!(found.is_empty());
}

#[test]
fn a_source_refuses_a_query_that_names_no_place() {
    let scene = Scene::new();
    let edited = held_out(fixture_points(&scene, &grid()));
    let views = scene.views();
    let options = PointsOptions::default();
    assert!(matches!(
        nearby_points(&edited, &views, 7, [10.0, 10.0], &options),
        Err(NearbySourceError::NoSuchImage { image: 7, .. })
    ));
    assert!(matches!(
        nearby_points(&edited, &views, 0, [-1.0, 10.0], &options),
        Err(NearbySourceError::PixelOffImage { .. })
    ));
    assert!(matches!(
        nearby_points(&edited, &views[..2], 0, [10.0, 10.0], &options),
        Err(NearbySourceError::InputMismatch { input: "views", .. })
    ));
}
