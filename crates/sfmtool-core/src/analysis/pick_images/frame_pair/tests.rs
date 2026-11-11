// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use sfmtool_sift_format::{SiftMetadata, SiftPositionsDescriptors, SIFT_FORMAT_VERSION};

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

fn sift_features(n: usize) -> SiftPositionsDescriptors {
    (vec![[10.0, 20.0]; n], vec![7; n * DESCRIPTOR_DIM])
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

fn camera(width: u32, height: u32) -> crate::camera::CameraIntrinsics {
    crate::camera::CameraIntrinsics {
        model: crate::camera::CameraModel::SimplePinhole {
            focal_length: 700.0,
            principal_point_x: f64::from(width) / 2.0,
            principal_point_y: f64::from(height) / 2.0,
        },
        width,
        height,
        rolling_shutter: None,
    }
}

#[test]
fn a_camera_adds_the_essential_split_of_the_matches() {
    // The second frame is the first turned 5° about the vertical axis.
    let camera = camera(WIDTH, HEIGHT);
    let turn = crate::geometry::rotation::axis_angle_to_rotation_matrix(
        &nalgebra::Vector3::y(),
        5f64.to_radians(),
    );
    let mut rng = StdRng::seed_from_u64(12);
    let positions = random_pixels(&mut rng, 400, 1.0);
    let (a, b) = mapped_pair(&mut rng, positions, |p| {
        let ray = turn * nalgebra::Vector3::from(camera.pixel_to_ray(p[0], p[1]));
        camera
            .ray_to_pixel([ray.x, ray.y, ray.z])
            .map(|(u, v)| [u, v])
    });
    let without = match_frame_pair(&a, &b, &FramePairOptions::default()).unwrap();
    assert_eq!(without.essential_split, None);
    let options = FramePairOptions {
        essential_split: Some(EssentialSplitOptions::new(camera.clone())),
        ..FramePairOptions::default()
    };
    let with = match_frame_pair(&a, &b, &options).unwrap();
    assert_eq!(
        (with.matches, with.step_percent),
        (without.matches, without.step_percent)
    );
    let Some(EssentialSplitOutcome::Measured(split)) = with.essential_split else {
        panic!("not measured: {:?}", with.essential_split);
    };
    assert_eq!(split.matches, with.matches);
    assert!((split.rotation_deg - 5.0).abs() < 0.01, "{split:?}");
    assert!(split.parallax_p90_deg < 0.01, "{split:?}");

    // A camera of another size than the frames is refused.
    let options = FramePairOptions {
        essential_split: Some(EssentialSplitOptions::new(
            crate::camera::CameraIntrinsics {
                width: WIDTH / 2,
                height: HEIGHT / 2,
                ..camera
            },
        )),
        ..FramePairOptions::default()
    };
    assert_eq!(
        match_frame_pair(&a, &b, &options).unwrap_err(),
        FramePairError::CameraSize {
            camera: (WIDTH / 2, HEIGHT / 2),
            frames: (WIDTH, HEIGHT)
        }
    );
}
