// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The point-or-bearing test over a reconstruction: the same numbers as the
//! primitives called by hand, aligned to the indexes asked for, from every
//! observation source.

use nalgebra::{Matrix2x3, Point3, Vector3};

use super::*;
use crate::analysis::reprojection_noise::tests::{as_embedded, as_sift_only, noisy_demo};
use crate::reconstruction::triangulation::point_or_bearing::{
    bearing_score, fit_point_and_bearing,
};
use crate::ObservationSource;

/// Rays, centres and weights.
type Rays = (Vec<Vector3<f64>>, Vec<Point3<f64>>, Vec<Matrix2x3<f64>>);

/// One point's rays, centres and weights, built from its inline keypoints with
/// `observed_ray` directly.
fn rays_by_hand(recon: &SfmrReconstruction, point: usize, sigma_px: f64) -> Rays {
    let keypoints = recon.keypoints_xy().unwrap();
    let start = recon.point_set.observation_offsets[point];
    let (mut dirs, mut centers, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    for (k, obs) in recon.observations_for_point(point).iter().enumerate() {
        let image = &recon.image_table.images[obs.image_index as usize];
        let camera = &recon.image_table.cameras[image.camera_index as usize];
        let pixel = [
            keypoints[[start + k, 0]] as f64,
            keypoints[[start + k, 1]] as f64,
        ];
        if let Some(ray) = observed_ray(camera, &image.quaternion_wxyz, pixel, sigma_px) {
            dirs.push(ray.dir);
            centers.push(image.camera_center());
            weights.push(ray.weight);
        }
    }
    (dirs, centers, weights)
}

#[test]
fn matches_the_primitives_called_by_hand() {
    let recon = noisy_demo(40, [0.5, 0.5], 11);
    let sigma = 0.7;
    let options = PointBearingFitOptions {
        soft_l1_scale: None,
        ..PointBearingFitOptions::default()
    };
    let result = recon
        .point_or_bearing_scores(None, Some(sigma), Some(&options))
        .unwrap();
    assert_eq!(result.sigma_px, sigma);
    assert_eq!(result.point_indexes, (0..40).collect::<Vec<_>>());
    let fits = result.fits.as_ref().unwrap();
    for (p, fit) in fits.iter().enumerate() {
        let (dirs, centers, weights) = rays_by_hand(&recon, p, sigma);
        assert_eq!(result.scores[p], bearing_score(&dirs, &centers, &weights));
        let start = Some(recon.point_set.points[p].position);
        assert_eq!(
            *fit,
            fit_point_and_bearing(&dirs, &centers, &weights, start, None, &options)
        );
    }
}

#[test]
fn defaults_to_the_measured_noise_level() {
    let recon = noisy_demo(40, [0.5, 0.5], 12);
    let measured = recon.reprojection_noise_px().unwrap().unwrap();
    let by_default = recon.point_or_bearing_scores(None, None, None).unwrap();
    let given = recon
        .point_or_bearing_scores(None, Some(measured), None)
        .unwrap();
    assert_eq!(by_default.sigma_px, measured);
    assert_eq!(by_default, given);
    assert!(by_default.fits.is_none());
}

#[test]
fn results_follow_the_indexes_asked_for() {
    let recon = noisy_demo(20, [0.5, 0.5], 13);
    let all = recon
        .point_or_bearing_scores(None, Some(0.5), Some(&PointBearingFitOptions::default()))
        .unwrap();
    let idx = [7, 2, 7, 19];
    let some = recon
        .point_or_bearing_scores(
            Some(&idx),
            Some(0.5),
            Some(&PointBearingFitOptions::default()),
        )
        .unwrap();
    assert_eq!(some.point_indexes, idx.to_vec());
    let all_fits = all.fits.unwrap();
    let some_fits = some.fits.unwrap();
    for (k, &p) in idx.iter().enumerate() {
        assert_eq!(some.scores[k], all.scores[p]);
        assert_eq!(some_fits[k], all_fits[p]);
    }
    let none = recon
        .point_or_bearing_scores(Some(&[]), Some(0.5), None)
        .unwrap();
    assert!(none.scores.is_empty());
}

#[test]
fn fewer_than_two_usable_rays_gives_none() {
    let mut recon = noisy_demo(10, [0.5, 0.5], 14);
    // Point 3's first observation loses its pixel, leaving one ray.
    let row = recon.point_set.observation_offsets[3];
    if let ObservationSource::SiftFiles {
        keypoints_xy: Some(k),
        ..
    } = &mut recon.point_set.observations
    {
        k[[row, 0]] = f32::NAN;
    }
    let result = recon
        .point_or_bearing_scores(None, Some(0.5), Some(&PointBearingFitOptions::default()))
        .unwrap();
    assert_eq!(result.scores[3], None);
    assert_eq!(result.fits.as_ref().unwrap()[3], None);
    assert!(result.scores[2].is_some());
    assert!(result.scores[4].is_some());
}

#[test]
fn points_at_infinity_are_scored() {
    let mut recon = noisy_demo(10, [0.5, 0.5], 15);
    recon.point_set.points[4].position = Point3::new(0.0, 0.0, 1.0);
    recon.point_set.points[4].w = 0.0;
    recon.rebuild_derived_fields();
    let options = PointBearingFitOptions::default();
    let result = recon
        .point_or_bearing_scores(None, Some(0.5), Some(&options))
        .unwrap();
    // Its keypoints still say where the finite point was, so it is scored, and
    // its fit has no stored position to start from.
    let (dirs, centers, weights) = rays_by_hand(&recon, 4, 0.5);
    assert_eq!(result.scores[4], bearing_score(&dirs, &centers, &weights));
    assert!(result.scores[4].is_some());
    assert_eq!(
        result.fits.unwrap()[4],
        fit_point_and_bearing(&dirs, &centers, &weights, None, None, &options)
    );
}

#[test]
fn every_observation_source_gives_the_same_scores() {
    let recon = noisy_demo(30, [0.5, 0.5], 16);
    let inline = recon.point_or_bearing_scores(None, None, None).unwrap();
    let embedded = as_embedded(&recon)
        .point_or_bearing_scores(None, None, None)
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let from_sift = as_sift_only(&recon, &dir)
        .point_or_bearing_scores(Some(&[3, 5]), None, None)
        .unwrap();
    assert_eq!(inline, embedded);
    assert_eq!(from_sift.scores, vec![inline.scores[3], inline.scores[5]]);
}

#[test]
fn refuses_what_it_cannot_run() {
    let recon = noisy_demo(5, [0.5, 0.5], 17);
    assert!(matches!(
        recon.point_or_bearing_scores(Some(&[1, 5]), None, None),
        Err(PointOrBearingError::PointIndexOutOfRange {
            index: 5,
            point_count: 5
        })
    ));
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            recon.point_or_bearing_scores(None, Some(bad), None),
            Err(PointOrBearingError::InvalidNoiseLevel(_))
        ));
    }

    let mut bearings = recon.clone();
    for point in &mut bearings.point_set.points {
        point.position = Point3::new(1.0, 0.0, 0.0);
        point.w = 0.0;
    }
    bearings.rebuild_derived_fields();
    assert!(matches!(
        bearings.point_or_bearing_scores(None, None, None),
        Err(PointOrBearingError::NoNoiseLevel)
    ));
    assert!(bearings
        .point_or_bearing_scores(None, Some(0.5), None)
        .is_ok());
}

#[test]
fn observation_ray_is_observed_ray_with_the_camera_centre() {
    let recon = noisy_demo(5, [0.5, 0.5], 18);
    let table = &recon.image_table;
    let image = &table.images[3];
    let camera = &table.cameras[image.camera_index as usize];
    let pixel = [900.0, 500.0];
    let ray = observation_ray(table, 3, pixel, 0.5).unwrap();
    let direct = observed_ray(camera, &image.quaternion_wxyz, pixel, 0.5).unwrap();
    assert_eq!(ray.dir, direct.dir);
    assert_eq!(ray.weight, direct.weight);
    assert_eq!(ray.center, image.camera_center());

    assert_eq!(observation_ray(table, 99, pixel, 0.5), None);
    assert_eq!(observation_ray(table, 3, [f64::NAN, 500.0], 0.5), None);
    assert_eq!(observation_ray(table, 3, pixel, 0.0), None);
}

#[test]
fn track_rays_keeps_one_track_per_input_track() {
    let recon = noisy_demo(5, [0.5, 0.5], 19);
    let table = &recon.image_table;
    let observations = [
        (0, [900.0, 500.0]),
        (1, [950.0, 520.0]),
        (2, [f64::NAN, 0.0]),
        (5, [960.0, 540.0]),
    ];
    // Track 0 has two rays, track 1 none, track 2 one usable of two.
    let rays = track_rays(table, &observations, &[0, 2, 2, 4], 0.5);
    assert_eq!(rays.offsets, vec![0, 2, 2, 3]);
    assert_eq!(rays.dirs.len(), 3);
    let last = observation_ray(table, 5, [960.0, 540.0], 0.5).unwrap();
    assert_eq!(
        (rays.dirs[2], rays.centers[2], rays.weights[2]),
        (last.dir, last.center, last.weight)
    );
    let scores = bearing_score_batch(&rays.dirs, &rays.centers, &rays.offsets, &rays.weights);
    assert!(scores[0].is_some());
    assert_eq!(scores[1], None);
    assert_eq!(scores[2], None);
    assert_eq!(track_rays(table, &[], &[0], 0.5).offsets, vec![0]);
}

/// Rays from three cameras on a line to `target`, weighted for an isotropic
/// angular noise of a milliradian.
fn rays_to_target(target: Point3<f64>) -> Rays {
    let centers = vec![
        Point3::new(-1.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
    ];
    let dirs: Vec<Vector3<f64>> = centers.iter().map(|c| (target - c).normalize()).collect();
    let weights = dirs
        .iter()
        .map(|d| crate::reconstruction::triangulation::isotropic_ray_weight(d, 1e-3))
        .collect();
    (dirs, centers, weights)
}

#[test]
fn a_usable_point_is_in_front_and_beyond_the_minimum_depth() {
    let (dirs, centers, _) = rays_to_target(Point3::new(0.0, 0.0, 10.0));
    let usable = |x: Point3<f64>, min_depth: f64| is_usable_point(&x, &dirs, &centers, min_depth);
    assert!(usable(Point3::new(0.0, 0.0, 10.0), 0.1));
    // Behind the cameras.
    assert!(!usable(Point3::new(0.0, 0.0, -10.0), 0.1));
    // In front, but nearer a camera centre than the minimum depth.
    assert!(!usable(Point3::new(0.0, 0.0, 10.0), 20.0));
    // Not a point at all.
    assert!(!usable(Point3::new(f64::NAN, 0.0, 10.0), 0.1));
}

#[test]
fn the_usable_fit_keeps_only_a_point_that_clears_the_rule() {
    let target = Point3::new(0.0, 0.0, 10.0);
    let (dirs, centers, weights) = rays_to_target(target);
    let placed = fit_usable_point(&dirs, &centers, &weights, None, 0.1, 25.0);
    let fit = placed.fit.expect("three rays fit");
    assert!(fit.depth_likelihood_ratio >= 25.0);
    assert!((placed.point.expect("a usable point") - target).norm() < 1e-6);

    // The same fit, refused by the minimum depth and by a threshold above its
    // likelihood ratio.
    let deep = fit_usable_point(&dirs, &centers, &weights, None, 50.0, 25.0);
    assert_eq!(deep.fit, placed.fit);
    assert_eq!(deep.point, None);
    let high = fit.depth_likelihood_ratio * 2.0;
    assert_eq!(
        fit_usable_point(&dirs, &centers, &weights, None, 0.1, high).point,
        None
    );
    // A threshold of 0 asks nothing of the likelihood ratio.
    assert!(fit_usable_point(&dirs, &centers, &weights, None, 0.1, 0.0)
        .point
        .is_some());
}
