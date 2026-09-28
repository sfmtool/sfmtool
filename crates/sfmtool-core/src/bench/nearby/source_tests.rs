// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The matching sources, decided against the bench's synthetic capture:
//! pinhole cameras looking down world `+z` at a textured plane, with a grid of
//! points on it that every camera sees at its exact projection. One point is
//! deleted from the version and the query is made at its pixel, the way the
//! harness holds a point out. The other sources are decided on the same
//! plane seen by four cameras, with clusters, keypoints and descriptors placed
//! at the point's exact projections.

use std::sync::Arc;

use nalgebra::Point3;

use crate::bench::tests::scene::{fixture_points, Scene, PLANE_Z};
use crate::bench::track_at_pixel::tests::matches_file;
use crate::bench::track_at_pixel::{
    MatchesClusters, ViewCamera, STATUS_KEPT, STATUS_REFERENCE, STATUS_REJECTED_LOW_ZNCC,
    STATUS_REJECTED_SHIFT,
};
use crate::features::kdforest::ImageKeypoints;
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

// ---- The cluster-patches clusters --------------------------------------------

/// Four cameras a metre or so apart, all looking down world `+z`, so a
/// cluster can lose one bad member and keep three.
const FOUR: [[f64; 3]; 4] = [
    [-0.5, -0.3, 0.0],
    [0.45, 0.25, 0.0],
    [0.1, -0.55, 0.0],
    [0.3, 0.4, 0.0],
];

/// The point the clusters are built on, and where the query is made.
const ON_PLANE: Point3<f64> = Point3::new(0.05, -0.1, PLANE_Z);

/// A member at `world`'s projection in `image`, `off` px to the right, with
/// the `.matches` status `status`.
fn member(
    scene: &Scene,
    image: u32,
    world: Point3<f64>,
    off: f32,
    status: u8,
) -> (u32, [f32; 2], u8) {
    let p = scene.project(image as usize, world);
    (image, [p[0] as f32 + off, p[1] as f32], status)
}

fn four_names() -> Vec<String> {
    (0..4).map(|i| format!("image_{i}.jpg")).collect()
}

fn clusters_of(clusters: &[Vec<(u32, [f32; 2], u8)>]) -> MatchesClusters {
    let names = four_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    MatchesClusters::new(&matches_file(&names, clusters, 1.0, true), &names).expect("clusters")
}

#[test]
fn a_cluster_with_a_bad_member_keeps_the_good_ones() {
    let scene = Scene::from_centers(&FOUR, PLANE_Z);
    let views = scene.views();
    // The reference in image 1, the queried member in image 0, a kept member
    // in image 2 and one 8 px off in image 3.
    let clusters = clusters_of(&[vec![
        member(&scene, 1, ON_PLANE, 0.0, STATUS_REFERENCE),
        member(&scene, 0, ON_PLANE, 0.0, STATUS_KEPT),
        member(&scene, 2, ON_PLANE, 0.0, STATUS_KEPT),
        member(&scene, 3, ON_PLANE, 8.0, STATUS_KEPT),
    ]]);
    let pixel = scene.project(0, ON_PLANE);
    let found = nearby_cluster_tracks(
        &views,
        &clusters,
        0,
        pixel,
        &ClusterTracksOptions::default(),
    )
    .expect("valid");
    assert_eq!(found.len(), 1);
    let c = &found[0];
    assert_eq!(c.source, NearbySource::Clusters);
    assert_eq!(c.id, Some(0));
    assert_eq!(c.point, None);
    // The queried member first, then the rest in the order the file lists
    // them, without the bad one.
    let images: Vec<u32> = c.sightings.iter().map(|s| s.0).collect();
    assert_eq!(images, vec![0, 1, 2]);
    assert!(c.max_reproj_px < 1e-3);
    assert!((c.position - ON_PLANE.coords).norm() < 1e-4);
    assert!(c.distance_px < 1e-3);
    assert!((c.depth - PLANE_Z).abs() < 1e-4);

    // With the bad member within the bar, it stays.
    let options = ClusterTracksOptions {
        max_reproj_px: 20.0,
        ..ClusterTracksOptions::default()
    };
    let found = nearby_cluster_tracks(&views, &clusters, 0, pixel, &options).expect("valid");
    assert_eq!(found[0].n_views(), 4);
}

#[test]
fn a_cluster_whose_worst_member_is_the_queried_one_gives_nothing() {
    let scene = Scene::from_centers(&FOUR, PLANE_Z);
    let views = scene.views();
    let clusters = clusters_of(&[vec![
        member(&scene, 1, ON_PLANE, 0.0, STATUS_REFERENCE),
        member(&scene, 0, ON_PLANE, 8.0, STATUS_KEPT),
        member(&scene, 2, ON_PLANE, 0.0, STATUS_KEPT),
        member(&scene, 3, ON_PLANE, 0.0, STATUS_KEPT),
    ]]);
    let pixel = scene.project(0, ON_PLANE);
    let found = nearby_cluster_tracks(
        &views,
        &clusters,
        0,
        pixel,
        &ClusterTracksOptions::default(),
    )
    .expect("valid");
    assert!(found.is_empty());
}

#[test]
fn the_member_policy_decides_which_members_are_used() {
    let scene = Scene::from_centers(&FOUR, PLANE_Z);
    let views = scene.views();
    // Image 1 holds a kept member at the point and a rejected one 30 px away;
    // image 2's only member was rejected for its ZNCC.
    let clusters = clusters_of(&[vec![
        member(&scene, 0, ON_PLANE, 0.0, STATUS_REFERENCE),
        member(&scene, 1, ON_PLANE, 30.0, STATUS_REJECTED_SHIFT),
        member(&scene, 1, ON_PLANE, 0.0, STATUS_KEPT),
        member(&scene, 2, ON_PLANE, 0.0, STATUS_REJECTED_LOW_ZNCC),
    ]]);
    let pixel = scene.project(0, ON_PLANE);
    let images = |members: ClusterMembers| -> Vec<u32> {
        let options = ClusterTracksOptions {
            members,
            ..ClusterTracksOptions::default()
        };
        let found = nearby_cluster_tracks(&views, &clusters, 0, pixel, &options).expect("valid");
        found[0].sightings.iter().map(|s| s.0).collect()
    };
    // Any member: the kept one in image 1 is preferred to the rejected one,
    // and image 2's rejected member is used.
    assert_eq!(images(ClusterMembers::Any), vec![0, 1, 2]);
    // Only the reference and the kept.
    assert_eq!(images(ClusterMembers::Kept), vec![0, 1]);

    // A queried member the policy does not admit gives nothing.
    let clusters = clusters_of(&[vec![
        member(&scene, 0, ON_PLANE, 0.0, STATUS_REJECTED_LOW_ZNCC),
        member(&scene, 1, ON_PLANE, 0.0, STATUS_REFERENCE),
        member(&scene, 2, ON_PLANE, 0.0, STATUS_KEPT),
    ]]);
    let options = ClusterTracksOptions {
        members: ClusterMembers::Kept,
        ..ClusterTracksOptions::default()
    };
    assert!(nearby_cluster_tracks(&views, &clusters, 0, pixel, &options)
        .expect("valid")
        .is_empty());
}

#[test]
fn sightings_meet_where_their_rays_do_with_each_error() {
    let scene = Scene::from_centers(&FOUR, PLANE_Z);
    let views = scene.views();
    let mut sightings: Vec<(u32, [f64; 2])> = (0..4)
        .map(|i| (i, scene.project(i as usize, ON_PLANE)))
        .collect();
    let met = triangulate_sightings(&views, &sightings).expect("the rays meet");
    assert!((met.position - ON_PLANE.coords).norm() < 1e-9);
    assert!(met.max_error_px() < 1e-6);
    // One sighting off by 4 px carries most of the error.
    sightings[2].1[0] += 4.0;
    let met = triangulate_sightings(&views, &sightings).expect("the rays meet");
    let worst = (0..4)
        .max_by(|&a, &b| met.errors_px[a].total_cmp(&met.errors_px[b]))
        .unwrap();
    assert_eq!(worst, 2);
    // One ray fixes no point, and an image that is not there none either.
    assert!(triangulate_sightings(&views, &sightings[..1]).is_none());
    assert!(triangulate_sightings(&views, &[(0, [64.0, 64.0]), (9, [64.0, 64.0])]).is_none());
}

// ---- Guided matching ---------------------------------------------------------

/// A few plane points around [`ON_PLANE`], each seen by every camera, with a
/// descriptor of its own; point 0 is [`ON_PLANE`] itself.
struct Features {
    keypoints: Vec<ImageKeypoints>,
    descriptors: Vec<Vec<u8>>,
}

/// A descriptor of values in `20..=200`, from `seed`.
fn descriptor(seed: u64) -> Vec<u8> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..128)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            20 + ((s >> 33) % 181) as u8
        })
        .collect()
}

/// `d` with `by` added to its first `dims` values.
fn nudged(d: &[u8], by: u8, dims: usize) -> Vec<u8> {
    d.iter()
        .enumerate()
        .map(|(i, &v)| if i < dims { v + by } else { v })
        .collect()
}

/// Every image's keypoints at the projections of twelve plane points, the
/// first [`ON_PLANE`], with the same descriptors in image 0 and 2; image 1's
/// copy of point 0's differs by 4 in every value (a distance of 45), and image
/// 3's by 27 (a distance of 305, past the strict bar and within the loose).
fn features(scene: &Scene) -> Features {
    let mut points = vec![ON_PLANE];
    for k in 1..12 {
        let a = k as f64 * 2.1;
        let r = 0.08 + 0.02 * k as f64;
        points.push(Point3::new(
            ON_PLANE.x + r * a.cos(),
            ON_PLANE.y + r * a.sin(),
            PLANE_Z,
        ));
    }
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();
    for image in 0..scene.len() {
        let mut kp = ImageKeypoints::default();
        let mut desc = Vec::new();
        for (k, &p) in points.iter().enumerate() {
            let px = scene.project(image, p);
            kp.positions.push([px[0] as f32, px[1] as f32]);
            kp.affine_shapes.push([[1.0, 0.0], [0.0, 1.0]]);
            let d = descriptor(k as u64);
            desc.extend(match (k, image) {
                (0, 1) => nudged(&d, 4, 128),
                (0, 3) => nudged(&d, 27, 128),
                _ => d,
            });
        }
        keypoints.push(kp);
        descriptors.push(desc);
    }
    Features {
        keypoints,
        descriptors,
    }
}

/// Guided matching at [`ON_PLANE`]'s pixel in image 0: the sightings of the
/// candidate on its keypoint, row 0.
fn guided_images(scene: &Scene, f: &Features, options: &GuidedOptions) -> Vec<u32> {
    let views = scene.views();
    let descriptors: Vec<ImageDescriptors> = f
        .descriptors
        .iter()
        .map(|d| ImageDescriptors::new(d.clone()))
        .collect();
    let rays = KeypointRays::new(views.len());
    let source = GuidedSource {
        keypoints: &f.keypoints,
        descriptors: &descriptors,
        rays: &rays,
    };
    let pixel = scene.project(0, ON_PLANE);
    let found = guided_matches(&views, &source, 0, pixel, options).expect("valid");
    let c = found
        .iter()
        .find(|c| c.id == Some(0))
        .expect("the keypoint at the pixel is matched");
    assert_eq!(c.source, NearbySource::Guided);
    assert!(c.distance_px < 1e-3);
    assert!(c.max_reproj_px < 0.01);
    assert!((c.position - ON_PLANE.coords).norm() < 1e-3);
    c.sightings.iter().map(|s| s.0).collect()
}

#[test]
fn guided_matching_finds_the_match_along_the_ray_and_the_loose_one_after() {
    let scene = Scene::from_centers(&FOUR, PLANE_Z);
    let f = features(&scene);
    // Images 1 and 2 match within the bar; image 3 only within the loose one,
    // added once the others have fixed the point.
    assert_eq!(
        guided_images(&scene, &f, &GuidedOptions::default()),
        vec![0, 1, 2, 3]
    );
    let strict = GuidedOptions {
        loose_distance: 250.0,
        ..GuidedOptions::default()
    };
    assert_eq!(guided_images(&scene, &f, &strict), vec![0, 1, 2]);
}

#[test]
fn guided_matching_refuses_an_ambiguous_match() {
    let scene = Scene::from_centers(&FOUR, PLANE_Z);
    let mut f = features(&scene);
    // A keypoint in image 1 on the queried ray, further out, whose descriptor
    // is nearly as close as the true match's: 50 against 45, over the ratio.
    let views = scene.views();
    let camera = ViewCamera::new(&views[0]);
    let pixel = scene.project(0, ON_PLANE);
    let decoy = Point3::from(camera.center + camera.ray(pixel).normalize() * 6.5);
    let at = scene.project(1, decoy);
    f.keypoints[1].positions.push([at[0] as f32, at[1] as f32]);
    f.keypoints[1].affine_shapes.push([[1.0, 0.0], [0.0, 1.0]]);
    f.descriptors[1].extend(nudged(&descriptor(0), 5, 100));
    assert_eq!(
        guided_images(&scene, &f, &GuidedOptions::default()),
        vec![0, 2, 3]
    );
    // A looser ratio takes the nearer descriptor, the true match.
    let loose = GuidedOptions {
        ratio: 0.95,
        ..GuidedOptions::default()
    };
    assert_eq!(guided_images(&scene, &f, &loose), vec![0, 1, 2, 3]);
}

#[test]
fn guided_matching_refuses_descriptors_that_do_not_match_the_keypoints() {
    let scene = Scene::from_centers(&FOUR, PLANE_Z);
    let f = features(&scene);
    let views = scene.views();
    let mut descriptors: Vec<ImageDescriptors> = f
        .descriptors
        .iter()
        .map(|d| ImageDescriptors::new(d.clone()))
        .collect();
    descriptors[2] = ImageDescriptors::new(vec![0; 128]);
    let rays = KeypointRays::new(views.len());
    let source = GuidedSource {
        keypoints: &f.keypoints,
        descriptors: &descriptors,
        rays: &rays,
    };
    let pixel = scene.project(0, ON_PLANE);
    assert!(matches!(
        guided_matches(&views, &source, 0, pixel, &GuidedOptions::default()),
        Err(NearbySourceError::RowMismatch { image: 2, .. })
    ));
}
