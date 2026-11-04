// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Matrix3, Rotation3, Vector3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use sfmtool_sift_format::{SiftContentHash, SiftFeatures, SiftMetadata, SIFT_FORMAT_VERSION};

use super::*;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;

fn random_descriptors(rng: &mut StdRng, n: usize) -> Vec<u8> {
    (0..n * DESCRIPTOR_DIM)
        .map(|_| rng.random::<u8>())
        .collect()
}

fn frame(positions: Vec<[f32; 2]>, descriptors: Vec<u8>) -> FrameFeatures {
    FrameFeatures::new(
        positions,
        descriptors,
        WIDTH,
        HEIGHT,
        KdForestParams::balanced(),
    )
    .expect("valid features")
}

fn random_pixels(rng: &mut StdRng, n: usize, margin: f32) -> Vec<[f32; 2]> {
    (0..n)
        .map(|_| {
            [
                rng.random_range(margin..WIDTH as f32 - margin),
                rng.random_range(margin..HEIGHT as f32 - margin),
            ]
        })
        .collect()
}

fn camera_matrix(f: f64) -> Matrix3<f64> {
    Matrix3::new(
        f,
        0.0,
        f64::from(WIDTH) / 2.0,
        0.0,
        f,
        f64::from(HEIGHT) / 2.0,
        0.0,
        0.0,
        1.0,
    )
}

fn inside(p: [f64; 2]) -> bool {
    p[0] > 0.0 && p[0] < f64::from(WIDTH) && p[1] > 0.0 && p[1] < f64::from(HEIGHT)
}

/// Two frames of the same features, the second's positions from `map`, with
/// the features `map` sends outside the image dropped from both.
fn mapped_pair(
    rng: &mut StdRng,
    a_positions: Vec<[f32; 2]>,
    map: impl Fn([f64; 2]) -> Option<[f64; 2]>,
) -> (FrameFeatures, FrameFeatures) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for p in a_positions {
        if let Some(q) = map([f64::from(p[0]), f64::from(p[1])]).filter(|&q| inside(q)) {
            a.push(p);
            b.push([q[0] as f32, q[1] as f32]);
        }
    }
    let descriptors = random_descriptors(rng, a.len());
    (frame(a, descriptors.clone()), frame(b, descriptors))
}

/// Points of a scene with depth from 4 to 12 seen by camera `a` at the origin
/// and camera `b` moved sideways by 1 and turned by 3 degrees.
fn scene_pair(rng: &mut StdRng, n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let k = camera_matrix(800.0);
    let rotation = Rotation3::from_euler_angles(0.0, 3f64.to_radians(), 0.0);
    let centre = Vector3::new(1.0, 0.0, 0.0);
    let project = |x: Vector3<f64>| {
        let p = k * x;
        [p[0] / p[2], p[1] / p[2]]
    };
    let mut xa = Vec::new();
    let mut xb = Vec::new();
    while xa.len() < n {
        let world = Vector3::new(
            rng.random_range(-3.0..3.0),
            rng.random_range(-2.0..2.0),
            rng.random_range(4.0..12.0),
        );
        let in_b = rotation * (world - centre);
        if in_b[2] <= 0.0 {
            continue;
        }
        let (pa, pb) = (project(world), project(in_b));
        if inside(pa) && inside(pb) {
            xa.push(pa);
            xb.push(pb);
        }
    }
    (xa, xb)
}

fn narrow(points: &[[f64; 2]]) -> Vec<[f32; 2]> {
    points.iter().map(|p| [p[0] as f32, p[1] as f32]).collect()
}

#[test]
fn a_known_translation_gives_its_step() {
    let mut rng = StdRng::seed_from_u64(1);
    let positions = random_pixels(&mut rng, 500, 1.0);
    let (a, b) = mapped_pair(&mut rng, positions, |p| Some([p[0] + 48.0, p[1] - 24.0]));
    let stats = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert_eq!(stats.matches, a.len());
    assert!(a.len() > 400, "{} features kept", a.len());
    // |(48, -24)| = 53.666 pixels of a 960-pixel long side.
    let expected = 100.0 * 48f64.hypot(24.0) / 960.0;
    assert!(
        (stats.step_percent - expected).abs() < 1e-4,
        "step {} against {expected}",
        stats.step_percent
    );
}

#[test]
fn the_step_is_the_median_displacement() {
    // 40 features moved by 10 pixels and 31 by 100: the median is 10 pixels,
    // where a mean would be 49.
    let mut rng = StdRng::seed_from_u64(2);
    let a_positions = random_pixels(&mut rng, 71, 120.0);
    let b_positions: Vec<[f32; 2]> = a_positions
        .iter()
        .enumerate()
        .map(|(i, p)| [p[0] + if i < 40 { 10.0 } else { 100.0 }, p[1]])
        .collect();
    let descriptors = random_descriptors(&mut rng, 71);
    let a = frame(a_positions, descriptors.clone());
    let b = frame(b_positions, descriptors);
    let stats = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert_eq!(stats.matches, 71);
    assert!((stats.step_percent - 100.0 * 10.0 / 960.0).abs() < 1e-4);
}

#[test]
fn a_pure_rotation_fits_one_homography_that_is_a_rotation() {
    let mut rng = StdRng::seed_from_u64(3);
    let k = camera_matrix(960.0);
    let rotation =
        Rotation3::from_euler_angles(2f64.to_radians(), 5f64.to_radians(), 1.0f64.to_radians());
    let h = k * rotation.matrix() * k.try_inverse().unwrap();
    let positions = random_pixels(&mut rng, 600, 1.0);
    let (a, b) = mapped_pair(&mut rng, positions, |p| {
        let q = h * Vector3::new(p[0], p[1], 1.0);
        Some([q[0] / q[2], q[1] / q[2]])
    });
    let stats = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert!(stats.matches > 300, "{} matches", stats.matches);
    assert!(stats.f_inlier_share.unwrap() > 0.99, "{stats:?}");
    assert!(stats.h_over_f.unwrap() > 0.99, "{stats:?}");
    assert!(stats.rotation_error.unwrap() < 0.01, "{stats:?}");
}

#[test]
fn the_rotation_error_is_zero_for_a_rotation_at_a_tried_focal_length() {
    // f = 0.3 · 10^(40/79) of the long side is the 41st focal length tried.
    let f = 960.0 * 0.3 * 10f64.powf(40.0 / 79.0);
    let k = camera_matrix(f);
    let rotation = Rotation3::from_euler_angles(0.1, -0.2, 0.05);
    // Any scale of H is the same homography.
    let h = 3.7 * k * rotation.matrix() * k.try_inverse().unwrap();
    let error = rotation_error(&h, WIDTH, HEIGHT).unwrap();
    assert!(error < 1e-9, "error {error}");
    // A homography that scales the image is no rotation at any focal length.
    let zoom = Matrix3::new(1.2, 0.0, -96.0, 0.0, 1.2, -54.0, 0.0, 0.0, 1.0);
    assert!(rotation_error(&zoom, WIDTH, HEIGHT).unwrap() > 0.05);
    // A singular H has no M of determinant 1.
    assert_eq!(rotation_error(&Matrix3::zeros(), WIDTH, HEIGHT), None);
}

#[test]
fn a_scene_with_depth_seen_from_two_places_fits_no_homography() {
    let mut rng = StdRng::seed_from_u64(4);
    let (xa, xb) = scene_pair(&mut rng, 600);
    let descriptors = random_descriptors(&mut rng, xa.len());
    let a = frame(narrow(&xa), descriptors.clone());
    let b = frame(narrow(&xb), descriptors);
    let stats = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert_eq!(stats.matches, 600);
    assert!(stats.f_inlier_share.unwrap() > 0.99, "{stats:?}");
    assert!(stats.h_over_f.unwrap() < 0.5, "{stats:?}");
    assert!(stats.rotation_error.unwrap() > 0.05, "{stats:?}");
}

#[test]
fn fewer_than_min_matches_give_no_shares() {
    let mut rng = StdRng::seed_from_u64(5);
    let positions = random_pixels(&mut rng, 20, 50.0);
    let (a, b) = mapped_pair(&mut rng, positions, |p| Some([p[0] + 9.6, p[1]]));
    let options = FramePairOptions::default();
    let stats = match_frame_pair(&a, &b, &options).unwrap();
    assert_eq!(stats.matches, 20);
    assert!((stats.step_percent - 1.0).abs() < 1e-4);
    assert_eq!(stats.f_inlier_share, None);
    assert_eq!(stats.h_over_f, None);
    assert_eq!(stats.rotation_error, None);
    // The same pair at a lower floor has shares.
    let stats = match_frame_pair(
        &a,
        &b,
        &FramePairOptions {
            min_matches: 20,
            ..options
        },
    )
    .unwrap();
    assert!(stats.f_inlier_share.is_some());
    assert!(stats.h_over_f.is_some());
}

#[test]
fn no_matches_give_a_nan_step() {
    let mut rng = StdRng::seed_from_u64(6);
    let a = frame(
        random_pixels(&mut rng, 50, 1.0),
        random_descriptors(&mut rng, 50),
    );
    let empty = frame(Vec::new(), Vec::new());
    for (x, y) in [(&a, &empty), (&empty, &a), (&empty, &empty)] {
        let stats = match_frame_pair(x, y, &FramePairOptions::default()).unwrap();
        assert_eq!(stats.matches, 0);
        assert!(stats.step_percent.is_nan());
        assert_eq!(stats.f_inlier_share, None);
    }
}

#[test]
fn the_ratio_test_rejects_a_feature_with_two_equal_neighbours() {
    // Frame b holds each descriptor twice, at two places, so no feature of a
    // has a nearest neighbour closer than its second.
    let mut rng = StdRng::seed_from_u64(7);
    let descriptors = random_descriptors(&mut rng, 40);
    let a = frame(random_pixels(&mut rng, 40, 1.0), descriptors.clone());
    let doubled = [descriptors.clone(), descriptors].concat();
    let b = frame(random_pixels(&mut rng, 80, 1.0), doubled);
    let stats = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert_eq!(stats.matches, 0);
}

#[test]
fn only_mutual_best_matches_count() {
    // a's features 40..80 are near copies of its features 0..40. Each finds
    // the same feature of b as its original does, and passes the ratio test
    // there, but that feature's own best match is the original.
    let mut rng = StdRng::seed_from_u64(8);
    let originals = random_descriptors(&mut rng, 40);
    let near_copies: Vec<u8> = originals
        .as_chunks::<DESCRIPTOR_DIM>()
        .0
        .iter()
        .flat_map(|d| {
            let mut d = d.to_vec();
            d[0] = d[0].wrapping_add(if d[0] < 128 { 3 } else { 253 });
            d
        })
        .collect();
    let a = frame(
        random_pixels(&mut rng, 80, 1.0),
        [originals.clone(), near_copies].concat(),
    );
    let b = frame(random_pixels(&mut rng, 40, 1.0), originals);
    let stats = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert_eq!(stats.matches, 40);
}

#[test]
fn the_statistics_repeat_with_the_seed() {
    let mut rng = StdRng::seed_from_u64(9);
    let (xa, mut xb) = scene_pair(&mut rng, 800);
    // Half a pixel of noise everywhere and a third of the matches wrong, so
    // the RANSAC samples decide the inliers.
    for (i, q) in xb.iter_mut().enumerate() {
        if i % 3 == 0 {
            *q = [
                rng.random_range(0.0..f64::from(WIDTH)),
                rng.random_range(0.0..f64::from(HEIGHT)),
            ];
        } else {
            q[0] += 0.5 * rng.sample::<f64, _>(StandardNormal);
            q[1] += 0.5 * rng.sample::<f64, _>(StandardNormal);
        }
    }
    let descriptors = random_descriptors(&mut rng, xa.len());
    let a = frame(narrow(&xa), descriptors.clone());
    let b = frame(narrow(&xb), descriptors);
    let options = FramePairOptions {
        seed: 17,
        ..FramePairOptions::default()
    };
    let first = match_frame_pair(&a, &b, &options).unwrap();
    let share = first.f_inlier_share.unwrap();
    assert!(share > 0.5 && share < 0.75, "{first:?}");
    for _ in 0..3 {
        assert_eq!(match_frame_pair(&a, &b, &options).unwrap(), first);
    }
}

#[test]
fn identical_or_collinear_matches_give_a_result() {
    let mut rng = StdRng::seed_from_u64(10);
    let n = 100;
    // Every feature at one place in both frames.
    let same = vec![[300.0f32, 200.0]; n];
    let descriptors = random_descriptors(&mut rng, n);
    let a = frame(same.clone(), descriptors.clone());
    let b = frame(same, descriptors.clone());
    let stats = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert_eq!(stats.matches, n);
    assert_eq!(stats.step_percent, 0.0);
    assert!(stats.f_inlier_share.is_some());
    // Every feature on one line, moved along it.
    let line: Vec<[f32; 2]> = (0..n).map(|i| [5.0 + 9.0 * i as f32, 100.0]).collect();
    let moved: Vec<[f32; 2]> = line.iter().map(|p| [p[0] + 2.0, p[1]]).collect();
    let a = frame(line, descriptors.clone());
    let b = frame(moved, descriptors);
    let stats = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert_eq!(stats.matches, n);
    assert!((stats.step_percent - 100.0 * 2.0 / 960.0).abs() < 1e-4);
    assert!(stats.f_inlier_share.is_some());
}

#[test]
fn feature_arrays_that_disagree_are_refused() {
    let forest = KdForestParams::balanced();
    let err = FrameFeatures::new(vec![[1.0, 2.0]; 3], vec![0; 128 * 2], WIDTH, HEIGHT, forest)
        .unwrap_err();
    assert!(matches!(err, FramePairError::FeatureArrays { .. }), "{err}");
    assert!(err.to_string().contains("256 bytes"), "{err}");
    let err = FrameFeatures::new(
        vec![[1.0, 2.0], [f32::NAN, 0.0]],
        vec![0; 256],
        WIDTH,
        HEIGHT,
        forest,
    )
    .unwrap_err();
    assert!(matches!(err, FramePairError::FeatureArrays { .. }), "{err}");
    assert!(err.to_string().contains("position 1"), "{err}");
    let err = FrameFeatures::new(
        vec![[f32::INFINITY, 0.0]],
        vec![0; 128],
        WIDTH,
        HEIGHT,
        forest,
    )
    .unwrap_err();
    assert!(matches!(err, FramePairError::FeatureArrays { .. }), "{err}");
}

#[test]
fn an_image_side_of_zero_is_refused() {
    for (width, height) in [(0, 540), (960, 0)] {
        let err = FrameFeatures::new(
            Vec::new(),
            Vec::new(),
            width,
            height,
            KdForestParams::balanced(),
        )
        .unwrap_err();
        assert_eq!(err, FramePairError::ImageSize { width, height });
    }
}

#[test]
fn a_forest_parameter_of_zero_is_refused() {
    let balanced = KdForestParams::balanced();
    for (name, forest) in [
        (
            "forest.num_trees",
            KdForestParams {
                num_trees: 0,
                ..balanced
            },
        ),
        (
            "forest.split_dim_candidates",
            KdForestParams {
                split_dim_candidates: 0,
                ..balanced
            },
        ),
        (
            "forest.leaf_size",
            KdForestParams {
                leaf_size: 0,
                ..balanced
            },
        ),
    ] {
        let err =
            FrameFeatures::new(vec![[1.0, 1.0]], vec![0; 128], WIDTH, HEIGHT, forest).unwrap_err();
        assert!(
            matches!(err, FramePairError::Options { name: n, .. } if n == name),
            "{err}"
        );
    }
}

#[test]
fn options_out_of_range_are_refused() {
    let mut rng = StdRng::seed_from_u64(11);
    let a = frame(
        random_pixels(&mut rng, 10, 1.0),
        random_descriptors(&mut rng, 10),
    );
    let defaults = FramePairOptions::default();
    let cases = [
        (
            "ratio",
            FramePairOptions {
                ratio: 0.0,
                ..defaults.clone()
            },
        ),
        (
            "ratio",
            FramePairOptions {
                ratio: 1.01,
                ..defaults.clone()
            },
        ),
        (
            "ratio",
            FramePairOptions {
                ratio: f32::NAN,
                ..defaults.clone()
            },
        ),
        (
            "min_matches",
            FramePairOptions {
                min_matches: 6,
                ..defaults.clone()
            },
        ),
        (
            "fundamental_error_per_960px",
            FramePairOptions {
                fundamental_error_per_960px: 0.0,
                ..defaults.clone()
            },
        ),
        (
            "fundamental_error_per_960px",
            FramePairOptions {
                fundamental_error_per_960px: f64::INFINITY,
                ..defaults.clone()
            },
        ),
        (
            "homography_error_per_960px",
            FramePairOptions {
                homography_error_per_960px: -1.0,
                ..defaults.clone()
            },
        ),
        (
            "homography_error_per_960px",
            FramePairOptions {
                homography_error_per_960px: f64::NAN,
                ..defaults.clone()
            },
        ),
    ];
    for (name, options) in cases {
        let err = match_frame_pair(&a, &a, &options).unwrap_err();
        assert!(
            matches!(err, FramePairError::Options { name: n, .. } if n == name),
            "{name}: {err}"
        );
    }
    // The bounds themselves are accepted.
    let edge = FramePairOptions {
        ratio: 1.0,
        min_matches: 7,
        ..defaults
    };
    assert!(match_frame_pair(&a, &a, &edge).is_ok());
}

#[test]
fn frames_of_different_sizes_are_refused() {
    let a = frame(Vec::new(), Vec::new());
    let b = FrameFeatures::new(
        Vec::new(),
        Vec::new(),
        HEIGHT,
        WIDTH,
        KdForestParams::balanced(),
    )
    .unwrap();
    let err = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap_err();
    assert_eq!(
        err,
        FramePairError::SizeMismatch {
            a: (WIDTH, HEIGHT),
            b: (HEIGHT, WIDTH)
        }
    );
}

fn sift_metadata(feature_count: u32) -> SiftMetadata {
    SiftMetadata {
        version: SIFT_FORMAT_VERSION,
        image_name: "frame_0001.jpg".to_string(),
        image_file_xxh128: "0".repeat(32),
        image_file_size: 1,
        image_width: WIDTH,
        image_height: HEIGHT,
        feature_count,
        sharpness: Some(1.0),
        gradient_structure_tensor_xx_yy_xy: Some([1.0, 1.0, 0.0]),
    }
}

fn sift_features(n: usize) -> SiftFeatures {
    SiftFeatures {
        content_hash: SiftContentHash::default(),
        positions_xy: vec![[10.0, 20.0]; n],
        affine_shapes: vec![[[1.0, 0.0], [0.0, 1.0]]; n],
        descriptors: vec![7; n * DESCRIPTOR_DIM],
    }
}

#[test]
fn from_sift_takes_the_size_from_the_metadata() {
    let features = FrameFeatures::from_sift(
        sift_features(3),
        &sift_metadata(3),
        KdForestParams::balanced(),
    )
    .unwrap();
    assert_eq!(
        (features.len(), features.width(), features.height()),
        (3, WIDTH, HEIGHT)
    );
}

#[test]
fn from_sift_refuses_features_and_metadata_of_different_counts() {
    let err = FrameFeatures::from_sift(
        sift_features(3),
        &sift_metadata(4),
        KdForestParams::balanced(),
    )
    .unwrap_err();
    assert!(matches!(err, FramePairError::FeatureArrays { .. }), "{err}");
    assert!(err.to_string().contains("records 4"), "{err}");
}
