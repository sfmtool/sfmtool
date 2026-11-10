// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::camera::{CameraModel, RollingShutter, RollingShutterScanlines};
use crate::geometry::rotation::axis_angle_to_rotation_matrix;
use crate::numeric::quantile;

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 700.0,
            principal_point_x: 480.0,
            principal_point_y: 270.0,
        },
        width: 960,
        height: 540,
        rolling_shutter: None,
    }
}

/// An equidistant fisheye whose 960-pixel image circle reaches 110° off the
/// axis, so some of its rays point behind the image plane.
fn fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 240.0,
            principal_point_x: 480.0,
            principal_point_y: 480.0,
        },
        width: 960,
        height: 960,
        rolling_shutter: None,
    }
}

/// A synthetic pair: random points seen by `camera` at the origin and after
/// the motion `X₂ = R·X₁ + t`, each position moved by up to `noise_px`, with
/// the true parallax of each match in degrees.
struct Pair {
    a: Vec<[f64; 2]>,
    b: Vec<[f64; 2]>,
    parallax_deg: Vec<f64>,
}

fn synthetic_pair(
    camera: &CameraIntrinsics,
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    n: usize,
    noise_px: f64,
    seed: u64,
) -> Pair {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pair = Pair {
        a: Vec::new(),
        b: Vec::new(),
        parallax_deg: Vec::new(),
    };
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    while pair.a.len() < n {
        let pixel = [rng.random_range(0.0..w), rng.random_range(0.0..h)];
        let ray = Vector3::from(camera.pixel_to_ray(pixel[0], pixel[1]));
        let x1 = ray * rng.random_range(2.0..20.0);
        let x2 = rotation * x1 + translation;
        let Some((u, v)) = camera.ray_to_pixel([x2.x, x2.y, x2.z]) else {
            continue;
        };
        if !(0.0..w).contains(&u) || !(0.0..h).contains(&v) {
            continue;
        }
        let mut jitter = || rng.random_range(-noise_px..=noise_px);
        pair.a.push([pixel[0] + jitter(), pixel[1] + jitter()]);
        pair.b.push([u + jitter(), v + jitter()]);
        let turned = rotation * ray;
        let seen = x2.normalize();
        pair.parallax_deg.push(
            turned
                .cross(&seen)
                .norm()
                .atan2(turned.dot(&seen))
                .to_degrees(),
        );
    }
    pair
}

fn measured(outcome: EssentialSplitOutcome) -> EssentialSplit {
    match outcome {
        EssentialSplitOutcome::Measured(split) => split,
        other => panic!("not measured: {other:?}"),
    }
}

fn turn(axis: [f64; 3], degrees: f64) -> Matrix3<f64> {
    axis_angle_to_rotation_matrix(&Vector3::from(axis).normalize(), degrees.to_radians())
}

/// The recovered rotation and parallax quantiles match the truth.
fn assert_recovers(camera: CameraIntrinsics, rotation_deg: f64, translation: Vector3<f64>) {
    let rotation = turn([0.2, 1.0, 0.1], rotation_deg);
    let pair = synthetic_pair(&camera, &rotation, &translation, 400, 0.3, 3);
    let split = measured(
        measure_essential_split(&pair.a, &pair.b, &EssentialSplitOptions::new(camera)).unwrap(),
    );
    assert_eq!(split.matches, 400);
    assert!(split.inliers >= 390, "{} inliers of 400", split.inliers);
    assert!(
        (split.rotation_deg - rotation_deg).abs() < 0.05,
        "rotation {} against {rotation_deg}",
        split.rotation_deg
    );
    for (q, got) in [(0.5, split.parallax_p50_deg), (0.9, split.parallax_p90_deg)] {
        let truth = quantile(&pair.parallax_deg, q);
        assert!(
            (got - truth).abs() < 0.02 * truth,
            "parallax at {q}: {got} against {truth}"
        );
    }
}

#[test]
fn a_pinhole_pair_gives_its_rotation_and_parallax() {
    assert_recovers(pinhole(), 6.0, Vector3::new(0.6, 0.05, -0.2));
}

#[test]
fn a_fisheye_pair_past_the_hemisphere_gives_its_rotation_and_parallax() {
    let camera = fisheye();
    // Some of the rays are more than 90° off the axis.
    let pair = synthetic_pair(
        &camera,
        &Matrix3::identity(),
        &Vector3::zeros(),
        400,
        0.0,
        3,
    );
    let behind = pair
        .a
        .iter()
        .filter(|p| camera.pixel_to_ray(p[0], p[1])[2] > 0.0)
        .count();
    assert!(behind > 20, "{behind} rays behind the image plane");
    assert_recovers(camera, 15.0, Vector3::new(0.8, -0.1, 0.3));
}

#[test]
fn a_pure_rotation_has_no_parallax() {
    let camera = pinhole();
    let rotation = turn([0.1, 1.0, -0.3], 8.0);
    let pair = synthetic_pair(&camera, &rotation, &Vector3::zeros(), 400, 0.3, 5);
    let split = measured(
        measure_essential_split(&pair.a, &pair.b, &EssentialSplitOptions::new(camera)).unwrap(),
    );
    assert!(split.inliers >= 390, "{} inliers of 400", split.inliers);
    assert!(
        (split.rotation_deg - 8.0).abs() < 0.05,
        "rotation {}",
        split.rotation_deg
    );
    // 0.3 pixels at a 700-pixel focal is 0.025°, so the parallax is noise.
    assert!(split.parallax_p50_deg < 0.03, "{}", split.parallax_p50_deg);
    assert!(split.parallax_p90_deg < 0.05, "{}", split.parallax_p90_deg);
}

#[test]
fn too_few_matches_are_not_measured() {
    let camera = pinhole();
    let pair = synthetic_pair(
        &camera,
        &turn([0.0, 1.0, 0.0], 3.0),
        &Vector3::new(0.5, 0.0, 0.0),
        12,
        0.0,
        7,
    );
    let options = EssentialSplitOptions::new(camera);
    assert!(matches!(
        measure_essential_split(&pair.a, &pair.b, &options).unwrap(),
        EssentialSplitOutcome::Measured(_)
    ));
    // One fewer than min_inliers, and a position with no ray is not counted.
    let mut a = pair.a[..11].to_vec();
    let mut b = pair.b[..11].to_vec();
    a.push([f64::NAN, 10.0]);
    b.push(pair.b[11]);
    assert_eq!(
        measure_essential_split(&a, &b, &options).unwrap(),
        EssentialSplitOutcome::NotMeasured {
            matches: 11,
            reason: SplitNotMeasured::TooFewMatches
        }
    );
    assert_eq!(
        measure_essential_split(&[], &[], &options).unwrap(),
        EssentialSplitOutcome::NotMeasured {
            matches: 0,
            reason: SplitNotMeasured::TooFewMatches
        }
    );
}

#[test]
fn unrelated_positions_have_no_consensus() {
    let mut rng = StdRng::seed_from_u64(9);
    let mut random = || -> Vec<[f64; 2]> {
        (0..40)
            .map(|_| [rng.random_range(0.0..960.0), rng.random_range(0.0..540.0)])
            .collect()
    };
    let (a, b) = (random(), random());
    assert_eq!(
        measure_essential_split(&a, &b, &EssentialSplitOptions::new(pinhole())).unwrap(),
        EssentialSplitOutcome::NotMeasured {
            matches: 40,
            reason: SplitNotMeasured::NoConsensus
        }
    );
}

#[test]
fn the_split_repeats_exactly_for_a_seed() {
    // A third of the matches are wrong and the consensus draws few samples,
    // so which samples it draws changes the answer.
    let camera = pinhole();
    let mut pair = synthetic_pair(
        &camera,
        &turn([1.0, 0.3, 0.0], 4.0),
        &Vector3::new(0.3, 0.2, 0.1),
        200,
        1.0,
        11,
    );
    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..100 {
        pair.a
            .push([rng.random_range(0.0..960.0), rng.random_range(0.0..540.0)]);
        pair.b
            .push([rng.random_range(0.0..960.0), rng.random_range(0.0..540.0)]);
    }
    let options = |seed| EssentialSplitOptions {
        seed,
        samples: 16,
        ..EssentialSplitOptions::new(camera.clone())
    };
    let first = measure_essential_split(&pair.a, &pair.b, &options(42)).unwrap();
    for _ in 0..3 {
        assert_eq!(
            measure_essential_split(&pair.a, &pair.b, &options(42)).unwrap(),
            first
        );
    }
    // The seed reaches the sampler: another seed gives another answer.
    assert!((0..8)
        .any(|seed| measure_essential_split(&pair.a, &pair.b, &options(seed)).unwrap() != first));
}

#[test]
fn options_out_of_range_are_refused() {
    let base = EssentialSplitOptions::new(pinhole());
    let mut wide_focal = pinhole();
    wide_focal.model = CameraModel::SimplePinhole {
        focal_length: 0.0,
        principal_point_x: 480.0,
        principal_point_y: 270.0,
    };
    let mut shutter = pinhole();
    shutter.rolling_shutter = Some(RollingShutter {
        scanlines: RollingShutterScanlines::Rows,
        rates: [0.0, 0.01, 0.0],
    });
    let cases = [
        (
            "essential_split.camera",
            EssentialSplitOptions {
                camera: wide_focal,
                ..base.clone()
            },
        ),
        (
            "essential_split.camera",
            EssentialSplitOptions {
                camera: shutter,
                ..base.clone()
            },
        ),
        (
            "essential_split.tolerance_px",
            EssentialSplitOptions {
                tolerance_px: 0.0,
                ..base.clone()
            },
        ),
        (
            "essential_split.tolerance_px",
            EssentialSplitOptions {
                tolerance_px: f64::INFINITY,
                ..base.clone()
            },
        ),
        (
            "essential_split.min_inliers",
            EssentialSplitOptions {
                min_inliers: 7,
                ..base.clone()
            },
        ),
        (
            "essential_split.samples",
            EssentialSplitOptions {
                samples: 0,
                ..base.clone()
            },
        ),
    ];
    for (name, options) in cases {
        let err = measure_essential_split(&[], &[], &options).unwrap_err();
        assert!(
            matches!(err, FramePairError::Options { name: n, .. } if n == name),
            "{name}: {err}"
        );
    }
    // The bounds themselves are accepted.
    let edge = EssentialSplitOptions {
        min_inliers: 8,
        samples: 1,
        ..base.clone()
    };
    assert!(measure_essential_split(&[], &[], &edge).is_ok());
    let err = measure_essential_split(&[[1.0, 1.0]], &[], &base).unwrap_err();
    assert!(matches!(err, FramePairError::FeatureArrays { .. }), "{err}");
}
