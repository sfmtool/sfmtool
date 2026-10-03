// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::reconstruction::triangulation::isotropic_ray_weights;

fn default_params() -> InfinityParams {
    InfinityParams {
        eps_deg: 0.5,
        desc_thresh: 300.0,
        ratio: 0.8,
        min_views: 2,
    }
}

/// Build a descriptor that is `value` in every component.
fn flat_desc(value: u8) -> [u8; 128] {
    [value; 128]
}

/// One track of rays from `centers` along `dirs`, at an isotropic angular
/// noise of 1 px at a 1000 px focal length.
fn one_track(dirs: Vec<Vector3<f64>>, centers: Vec<Point3<f64>>) -> RayBatch {
    let n = dirs.len();
    RayBatch {
        weights: isotropic_ray_weights(&dirs, &vec![1e-3; n]),
        dirs,
        centers,
        offsets: vec![0, n],
    }
}

/// Unit rays from each of `centers` to `point`.
fn rays_to(point: Point3<f64>, centers: &[Point3<f64>]) -> Vec<Vector3<f64>> {
    centers
        .iter()
        .map(|c| (point.coords - c.coords).normalize())
        .collect()
}

#[test]
fn finds_single_infinite_point() {
    // 3 keypoints in 3 distinct images, identical world directions and
    // identical descriptors: one track, one member per image.
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let tracks = find_infinity_tracks(
        &[dir, dir, dir],
        &[flat_desc(50), flat_desc(50), flat_desc(50)],
        &[0, 1, 2],
        &[0, 0, 0],
        &default_params(),
    );
    assert_eq!(
        tracks,
        vec![InfinityTrack {
            members: vec![(0, 0), (1, 0), (2, 0)]
        }]
    );
}

#[test]
fn distinct_descriptors_not_merged() {
    // Same direction, different descriptors → no match.
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let tracks = find_infinity_tracks(
        &[dir, dir],
        &[flat_desc(0), flat_desc(255)],
        &[0, 1],
        &[0, 0],
        &default_params(),
    );
    assert!(tracks.is_empty(), "far descriptors must not merge");
}

#[test]
fn distinct_directions_not_merged() {
    // Identical descriptors but very different directions → no neighbour.
    let tracks = find_infinity_tracks(
        &[Vector3::new(0.0, 0.0, 1.0), Vector3::new(1.0, 0.0, 0.0)],
        &[flat_desc(50), flat_desc(50)],
        &[0, 1],
        &[0, 0],
        &default_params(),
    );
    assert!(tracks.is_empty(), "distant directions must not merge");
}

#[test]
fn one_feature_per_image_after_split() {
    // Image 1 contributes two co-directional, identical-descriptor features
    // alongside one each in images 0 and 2. The track must keep exactly one
    // feature per image.
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let tracks = find_infinity_tracks(
        &[dir, dir, dir, dir],
        &[flat_desc(50), flat_desc(50), flat_desc(50), flat_desc(50)],
        &[0, 1, 1, 2],
        &[0, 0, 1, 0],
        &default_params(),
    );

    assert_eq!(tracks.len(), 1, "one track");
    let imgs: Vec<u32> = tracks[0].members.iter().map(|(i, _)| *i).collect();
    let mut distinct = imgs.clone();
    distinct.dedup();
    assert_eq!(imgs, distinct, "no image appears twice in a track");
}

#[test]
fn min_views_filter_drops_short_track() {
    // A 2-image track is dropped when min_views = 3.
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let run = |min_views| {
        find_infinity_tracks(
            &[dir, dir],
            &[flat_desc(50), flat_desc(50)],
            &[0, 1],
            &[0, 0],
            &InfinityParams {
                min_views,
                ..default_params()
            },
        )
    };
    assert!(run(3).is_empty(), "2-image track dropped at min_views=3");
    assert_eq!(run(2).len(), 1, "kept at min_views=2");
}

#[test]
fn tracks_come_out_sorted_by_members() {
    // Two separate tracks along two directions; the order is by members, not
    // by hashing order.
    let a = Vector3::new(0.0, 0.0, 1.0);
    let b = Vector3::new(1.0, 0.0, 0.0);
    let tracks = find_infinity_tracks(
        &[b, a, b, a],
        &[flat_desc(10), flat_desc(90), flat_desc(10), flat_desc(90)],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &default_params(),
    );
    let members: Vec<_> = tracks.iter().map(|t| t.members.clone()).collect();
    assert_eq!(members, vec![vec![(0, 0), (1, 0)], vec![(0, 1), (1, 1)]]);
}

#[test]
fn parallel_rays_are_a_bearing() {
    // Cameras spread 2 units apart see one direction: the rays ask for no
    // depth, and the bearing is that direction.
    let dir = Vector3::new(0.0, 0.6, 0.8);
    let rays = one_track(
        vec![dir; 3],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ],
    );
    match decide_candidate_tracks(&rays)[0] {
        CandidateDecision::Bearing(b) => assert!((b - dir).norm() < 1e-9, "bearing {b:?}"),
        other => panic!("parallel rays → bearing, got {other:?}"),
    }
}

#[test]
fn wide_parallax_track_is_finite() {
    // Three cameras on the x-axis look at a point 2 units in front; the rays
    // differ by tens of degrees, so the rays ask for a depth and the track is
    // not a point at infinity.
    let point = Point3::new(0.0, 0.0, 2.0);
    let centers = vec![
        Point3::new(-2.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
    ];
    let rays = one_track(rays_to(point, &centers), centers);
    assert_eq!(
        decide_candidate_tracks(&rays),
        vec![CandidateDecision::Finite]
    );
}

#[test]
fn a_near_point_seen_across_a_short_baseline_is_a_bearing() {
    // A point 5 units out seen by three cameras within 1e-4 of each other: at
    // 1 px noise the rays agree with a direction, so the test calls the track
    // a bearing and discovery appends it as one (the case a capture-scale
    // gate used to drop).
    let point = Point3::new(0.3, -0.2, 5.0);
    let centers = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1e-4, 0.0, 0.0),
        Point3::new(0.0, 1e-4, 0.0),
    ];
    let rays = one_track(rays_to(point, &centers), centers);
    match decide_candidate_tracks(&rays)[0] {
        CandidateDecision::Bearing(b) => {
            let expected = point.coords.normalize();
            assert!((b - expected).norm() < 1e-4, "bearing {b:?}");
        }
        other => panic!("short-baseline rays → bearing, got {other:?}"),
    }
}

#[test]
fn a_bearing_behind_a_camera_is_dropped() {
    // Two cameras see opposite directions along one line: the bearing fits
    // both rays at no cost but lies behind one of the cameras.
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let rays = one_track(
        vec![dir, -dir],
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
    );
    assert_eq!(
        decide_candidate_tracks(&rays),
        vec![CandidateDecision::BearingBehindCamera]
    );
}

#[test]
fn a_track_with_one_usable_ray_is_unscored() {
    let rays = one_track(vec![Vector3::z()], vec![Point3::origin()]);
    assert_eq!(
        decide_candidate_tracks(&rays),
        vec![CandidateDecision::Unscored]
    );
}

#[test]
fn decisions_follow_the_tracks_in_order() {
    // A bearing track then a finite one, in one batch.
    let point = Point3::new(0.0, 0.0, 2.0);
    let wide = vec![
        Point3::new(-2.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
    ];
    let mut dirs = vec![Vector3::z(); 2];
    dirs.extend(rays_to(point, &wide));
    let mut centers = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
    centers.extend(wide);
    let rays = RayBatch {
        weights: isotropic_ray_weights(&dirs, &[1e-3; 5]),
        dirs,
        centers,
        offsets: vec![0, 2, 5],
    };
    let decisions = decide_candidate_tracks(&rays);
    assert!(matches!(decisions[0], CandidateDecision::Bearing(_)));
    assert_eq!(decisions[1], CandidateDecision::Finite);
}

/// Each candidate member's pixel, keyed by `(image, feature)`.
type Pixels = HashMap<(u32, u32), [f64; 2]>;

/// The observations of `points` in `recon` as `(image, feature)` members and
/// their pixels, the feature numbered by the observation's row. Which point an
/// observation belongs to does not matter to the ray construction.
fn members_of(recon: &SfmrReconstruction, points: &[usize]) -> (Vec<(u32, u32)>, Pixels) {
    let keypoints = recon.keypoints_xy().expect("inline keypoints");
    let mut members = Vec::new();
    let mut pixels = HashMap::new();
    for &point in points {
        let start = recon.point_set.observation_offsets[point];
        for (k, obs) in recon.observations_for_point(point).iter().enumerate() {
            let row = start + k;
            let member = (obs.image_index, row as u32);
            members.push(member);
            pixels.insert(
                member,
                [keypoints[[row, 0]] as f64, keypoints[[row, 1]] as f64],
            );
        }
    }
    (members, pixels)
}

#[test]
fn a_member_whose_pixel_gives_no_ray_is_left_out() {
    let recon = crate::analysis::reprojection_noise::tests::noisy_demo(4, [0.5, 0.5], 3);
    let (members, mut pixels) = members_of(&recon, &[0, 1]);
    assert!(members.len() >= 3);
    // The first member's pixel gives no ray.
    pixels.insert(members[0], [f64::NAN, f64::NAN]);
    let candidates = vec![InfinityTrack {
        members: members.clone(),
    }];

    let kept = candidate_rays(&recon.image_table, &candidates, &pixels, 0.5, 2);
    assert_eq!(kept.members, vec![members[1..].to_vec()]);
    assert_eq!(kept.rays.offsets, vec![0, members.len() - 1]);
    let images: Vec<usize> = members[1..].iter().map(|m| m.0 as usize).collect();
    assert_eq!(kept.ray_images, images);

    // With every member required, the candidate keeps none and is unscored.
    let short = candidate_rays(&recon.image_table, &candidates, &pixels, 0.5, members.len());
    assert_eq!(short.members, vec![Vec::new()]);
    assert_eq!(short.rays.offsets, vec![0, 0]);
    assert_eq!(
        decide_candidate_tracks(&short.rays),
        vec![CandidateDecision::Unscored]
    );
}

#[test]
fn every_candidate_is_counted_once() {
    let decisions = [
        CandidateDecision::Bearing(Vector3::z()),
        CandidateDecision::Bearing(Vector3::x()),
        CandidateDecision::Finite,
        CandidateDecision::BearingBehindCamera,
        CandidateDecision::Unscored,
        CandidateDecision::Finite,
    ];
    // The short-baseline flag counts only on bearings.
    let short_baseline = [true, false, true, true, true, false];
    let mut summary = InfinityDiscovery {
        candidates: decisions.len(),
        ..InfinityDiscovery::default()
    };
    summary.count_decisions(&decisions, &short_baseline);
    assert_eq!(
        (
            summary.bearings,
            summary.short_baseline,
            summary.finite,
            summary.bearing_behind_camera,
            summary.unscored
        ),
        (2, 1, 2, 1, 1)
    );
    assert_eq!(
        summary.bearings + summary.finite + summary.bearing_behind_camera + summary.unscored,
        summary.candidates
    );
}
