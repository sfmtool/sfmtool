// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nalgebra::{Matrix3, Vector3};

fn test_intrinsics() -> Matrix3<f64> {
    Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0)
}

/// Smoke test: match_image_pair works with per-image dimensions and
/// forward motion. The actual both-epipoles safety logic is tested in
/// rectification::tests::test_rectification_safe_requires_both_epipoles_outside.
#[test]
fn test_match_image_pair_forward_motion_asymmetric_dimensions() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.0, 0.0, 1.0);

    // 4 features at the image corners, each with a unique descriptor.
    // Avoid the epipole (320, 240) where polar matching is degenerate.
    let n = 4;
    let positions = [
        100.0, 80.0, // top-left
        500.0, 80.0, // top-right
        100.0, 400.0, // bottom-left
        500.0, 400.0, // bottom-right
    ];
    let mut descriptors = vec![0u8; n * 128];
    for i in 0..n {
        descriptors[i * 128] = (i * 50) as u8;
    }

    let matches = match_image_pair(
        &k,
        &k,
        &r,
        &t1,
        &r,
        &t2,
        640,
        480, // image 1: normal size
        10,
        10, // image 2: different size
        &positions,
        &descriptors,
        n,
        &positions,
        &descriptors,
        n,
        128,
        5,
        None,
        50,
        None,
        None,
        None,
    )
    .unwrap();

    assert_eq!(
        matches.len(),
        n,
        "All {n} features should match (got {})",
        matches.len()
    );
    for (idx1, idx2, dist) in &matches {
        assert_eq!(idx1, idx2, "Feature {idx1} should match itself");
        assert_eq!(*dist, 0.0);
    }
}

/// A rotation of `angle_deg` degrees about the unit `axis`.
fn axis_rotation(axis: Vector3<f64>, angle_deg: f64) -> Matrix3<f64> {
    *nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Unit::new_normalize(axis),
        angle_deg.to_radians(),
    )
    .matrix()
}

/// On the rectified path the sweep sorts by rectified positions, but the
/// geometric filter must read the raw positions: its `StereoPairGeometry`
/// holds the original cameras. Each feature pair here is an exact projection of
/// one 3D point, with affine shapes that agree with the poses, so every pair
/// passes the filter at the raw positions and `match_image_pair` must find
/// all of them under a tight size gate.
#[test]
fn test_match_image_pair_rectified_geometric_filter_reads_raw_positions() {
    let k = test_intrinsics();
    // Camera 1 at the origin; camera 2 one unit to the right, turned towards
    // the scene and rolled, so rectification moves every point well away from
    // its raw position.
    let r1 = Matrix3::identity();
    let t1 = Vector3::zeros();
    let r2 = axis_rotation(Vector3::new(0.0, 0.0, 1.0), 12.0)
        * axis_rotation(Vector3::new(0.0, 1.0, 0.0), -10.0);
    let c2 = Vector3::new(1.0, 0.0, 0.0);
    let t2 = -(r2 * c2);
    let geom = StereoPairGeometry::new(&k, &k, &r1, &r2, &t1, &t2);

    let mut points = Vec::new();
    for iy in 0..4 {
        for ix in 0..5 {
            let depth = 3.0 + 0.7 * ((ix + 2 * iy) % 5) as f64;
            let x = (ix as f64 - 2.0) * 0.25 * depth;
            let y = (iy as f64 - 1.5) * 0.25 * depth;
            points.push(Vector3::new(x / 2.0, y / 2.0, depth));
        }
    }
    let n = points.len();

    let project = |r: &Matrix3<f64>, t: &Vector3<f64>, p: &Vector3<f64>| {
        let x = k * (r * p + t);
        [x[0] / x[2], x[1] / x[2]]
    };
    let base_shape = [3.0, -1.0, 1.0, 3.0];
    let rotated = [
        geom.r_2d[0] * base_shape[0] + geom.r_2d[1] * base_shape[2],
        geom.r_2d[0] * base_shape[1] + geom.r_2d[1] * base_shape[3],
        geom.r_2d[2] * base_shape[0] + geom.r_2d[3] * base_shape[2],
        geom.r_2d[2] * base_shape[1] + geom.r_2d[3] * base_shape[3],
    ];
    let mut positions1 = Vec::new();
    let mut positions2 = Vec::new();
    let mut affines1 = Vec::new();
    let mut affines2 = Vec::new();
    let mut descriptors = vec![0u8; n * 128];
    for (i, p) in points.iter().enumerate() {
        positions1.extend_from_slice(&project(&r1, &t1, p));
        positions2.extend_from_slice(&project(&r2, &t2, p));
        let d1 = (r1 * p + t1)[2];
        let d2 = (r2 * p + t2)[2];
        assert!(d2 > 0.0);
        affines1.extend_from_slice(&base_shape);
        affines2.extend(rotated.iter().map(|a| a * d1 / d2));
        descriptors[i * 128 + i] = 200;
    }

    let config = GeometricFilterConfig {
        max_angle_difference: 5.0,
        min_triangulation_angle: 1.0,
        geometric_size_ratio_min: 0.98,
        geometric_size_ratio_max: 1.02,
    };

    // The filter's mask for each true pair: all pass at the raw positions, and
    // at least one fails at the rectified positions, so the end-to-end check
    // below can tell which positions the rectified path hands the filter.
    let pair_passes = |p1: &[f64], p2: &[f64], i: usize| {
        two_stage_geometric_filter(
            [p1[i * 2], p1[i * 2 + 1]],
            &affines1[i * 4..i * 4 + 4].try_into().unwrap(),
            &p2[i * 2..i * 2 + 2],
            &affines2[i * 4..i * 4 + 4],
            1,
            &geom,
            &config,
        )[0]
    };
    assert!((0..n).all(|i| pair_passes(&positions1, &positions2, i)));

    let f =
        crate::camera::epipolar::compute_fundamental_matrix(&k, &r1, &t1, &k, &r2, &t2).unwrap();
    assert!(
        crate::camera::rectification::check_rectification_safe_from_f(&f, 640, 480, 50)
            && crate::camera::rectification::check_rectification_safe_from_f(
                &f.transpose(),
                640,
                480,
                50
            ),
        "the pair must take the rectified path"
    );
    let r_rel = r2 * r1.transpose();
    let t_rel = t2 - r_rel * t1;
    let rect = crate::camera::rectification::compute_stereo_rectification(
        &k, &k, &r_rel, &t_rel, 640, 480,
    );
    let k_inv = k.try_inverse().unwrap();
    let rectified1 = crate::camera::rectification::rectify_points(
        &positions1,
        n,
        &k_inv,
        &rect.r1_rect,
        &rect.p1,
    );
    let rectified2 = crate::camera::rectification::rectify_points(
        &positions2,
        n,
        &k_inv,
        &rect.r2_rect,
        &rect.p2,
    );
    assert!((0..n).any(|i| !pair_passes(&rectified1, &rectified2, i)));

    let mut matches = match_image_pair(
        &k,
        &k,
        &r1,
        &t1,
        &r2,
        &t2,
        640,
        480,
        640,
        480,
        &positions1,
        &descriptors,
        n,
        &positions2,
        &descriptors,
        n,
        128,
        n,
        None,
        50,
        Some(&affines1),
        Some(&affines2),
        Some(&config),
    )
    .unwrap();
    matches.sort_by_key(|m| m.0);
    let pairs: Vec<(usize, usize)> = matches.iter().map(|m| (m.0, m.1)).collect();
    let expected: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    assert_eq!(pairs, expected);
}

/// Four features whose descriptors differ, for the input-checking tests.
fn four_features() -> (Vec<f64>, Vec<u8>) {
    let positions = vec![100.0, 80.0, 500.0, 80.0, 100.0, 400.0, 500.0, 400.0];
    let mut descriptors = vec![0u8; 4 * 128];
    for i in 0..4 {
        descriptors[i * 128] = (i * 50) as u8;
    }
    (positions, descriptors)
}

/// `match_image_pair` between two images of [`four_features`] with sideways
/// motion, taking image 1's positions, image 2's descriptors and, when
/// `affines` is `Some`, both images' affine shapes and the default filter.
fn pair_with(
    positions1: &[f64],
    descriptors2: &[u8],
    affines: Option<(&[f64], &[f64])>,
) -> Result<PairMatches, MatchImagePairError> {
    let (positions, descriptors) = four_features();
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let config = GeometricFilterConfig::default();
    match_image_pair(
        &k,
        &k,
        &r,
        &Vector3::zeros(),
        &r,
        &Vector3::new(1.0, 0.0, 0.0),
        640,
        480,
        640,
        480,
        positions1,
        &descriptors,
        4,
        &positions,
        descriptors2,
        4,
        128,
        5,
        None,
        50,
        affines.map(|a| a.0),
        affines.map(|a| a.1),
        affines.map(|_| &config),
    )
}

#[test]
fn match_image_pair_refuses_short_feature_arrays() {
    let (positions, descriptors) = four_features();
    assert_eq!(pair_with(&positions, &descriptors, None).unwrap().len(), 4);

    assert_eq!(
        pair_with(&positions[..6], &descriptors, None),
        Err(MatchImagePairError::FeatureArrayLength {
            array: "positions1",
            image: None,
            rows: 4,
            width: 2,
            len: 6,
        })
    );
    assert_eq!(
        pair_with(&positions, &descriptors[..3 * 128], None),
        Err(MatchImagePairError::FeatureArrayLength {
            array: "descriptors2",
            image: None,
            rows: 4,
            width: 128,
            len: 3 * 128,
        })
    );
    let affines = [1.0, 0.0, 0.0, 1.0].repeat(4);
    assert!(pair_with(&positions, &descriptors, Some((&affines, &affines))).is_ok());
    assert_eq!(
        pair_with(&positions, &descriptors, Some((&affines, &affines[..12]))),
        Err(MatchImagePairError::FeatureArrayLength {
            array: "affines2",
            image: None,
            rows: 4,
            width: 4,
            len: 12,
        })
    );
}

/// The arguments of a two-image, one-camera `match_image_pairs_batch` call
/// over [`four_features`], which a test changes before [`BatchArgs::run`].
struct BatchArgs {
    pairs: Vec<(usize, usize)>,
    intrinsics: Vec<Matrix3<f64>>,
    rotations: Vec<Matrix3<f64>>,
    translations: Vec<Vector3<f64>>,
    camera_indices: Vec<usize>,
    positions: Vec<Vec<f64>>,
    descriptors: Vec<Vec<u8>>,
    num_features: Vec<usize>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    affines: Option<Vec<Vec<f64>>>,
}

impl BatchArgs {
    fn new() -> Self {
        let (positions, descriptors) = four_features();
        Self {
            pairs: vec![(0, 1)],
            intrinsics: vec![test_intrinsics()],
            rotations: vec![Matrix3::identity(); 2],
            translations: vec![Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0)],
            camera_indices: vec![0, 0],
            positions: vec![positions.clone(), positions],
            descriptors: vec![descriptors.clone(), descriptors],
            num_features: vec![4, 4],
            widths: vec![640],
            heights: vec![480],
            affines: None,
        }
    }

    fn run(&self) -> Result<Vec<PairMatches>, MatchImagePairError> {
        let positions: Vec<&[f64]> = self.positions.iter().map(Vec::as_slice).collect();
        let descriptors: Vec<&[u8]> = self.descriptors.iter().map(Vec::as_slice).collect();
        let affines: Option<Vec<&[f64]>> = self
            .affines
            .as_ref()
            .map(|a| a.iter().map(Vec::as_slice).collect());
        let config = GeometricFilterConfig::default();
        match_image_pairs_batch(
            &self.pairs,
            &self.intrinsics,
            &self.rotations,
            &self.translations,
            &self.camera_indices,
            &positions,
            &descriptors,
            &self.num_features,
            &self.widths,
            &self.heights,
            128,
            5,
            None,
            50,
            affines.as_deref(),
            affines.as_ref().map(|_| &config),
        )
    }
}

#[test]
fn match_image_pairs_batch_matches_valid_input() {
    let result = BatchArgs::new().run().unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 4);

    let mut args = BatchArgs::new();
    args.affines = Some(vec![[1.0, 0.0, 0.0, 1.0].repeat(4); 2]);
    assert_eq!(args.run().unwrap().len(), 1);
}

#[test]
fn match_image_pairs_batch_refuses_mismatched_lists() {
    let mut args = BatchArgs::new();
    args.rotations.pop();
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::ListLength {
            list: "rotations",
            len: 1,
            reference: "camera_indices",
            expected: 2,
        })
    );

    let mut args = BatchArgs::new();
    args.num_features.push(0);
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::ListLength {
            list: "num_features",
            len: 3,
            reference: "camera_indices",
            expected: 2,
        })
    );

    let mut args = BatchArgs::new();
    args.widths.push(640);
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::ListLength {
            list: "widths",
            len: 2,
            reference: "intrinsics",
            expected: 1,
        })
    );

    let mut args = BatchArgs::new();
    args.affines = Some(vec![[1.0, 0.0, 0.0, 1.0].repeat(4)]);
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::ListLength {
            list: "affines",
            len: 1,
            reference: "camera_indices",
            expected: 2,
        })
    );
}

#[test]
fn match_image_pairs_batch_refuses_out_of_range_indexes() {
    let mut args = BatchArgs::new();
    args.camera_indices[1] = 1;
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::CameraIndexOutOfRange {
            image: 1,
            camera: 1,
            num_cameras: 1,
        })
    );

    let mut args = BatchArgs::new();
    args.pairs.push((1, 2));
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::ImageIndexOutOfRange {
            pair: (1, 2),
            num_images: 2,
        })
    );
}

#[test]
fn match_image_pairs_batch_refuses_short_feature_arrays() {
    let mut args = BatchArgs::new();
    args.descriptors[1].truncate(3 * 128);
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::FeatureArrayLength {
            array: "descriptors",
            image: Some(1),
            rows: 4,
            width: 128,
            len: 3 * 128,
        })
    );

    let mut args = BatchArgs::new();
    args.positions[0].push(0.0);
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::FeatureArrayLength {
            array: "positions",
            image: Some(0),
            rows: 4,
            width: 2,
            len: 9,
        })
    );

    // An image no pair names is checked too.
    let mut args = BatchArgs::new();
    args.pairs.clear();
    args.affines = Some(vec![[1.0, 0.0, 0.0, 1.0].repeat(4), vec![1.0; 3]]);
    assert_eq!(
        args.run(),
        Err(MatchImagePairError::FeatureArrayLength {
            array: "affines",
            image: Some(1),
            rows: 4,
            width: 4,
            len: 3,
        })
    );
}

#[test]
fn match_image_pair_error_messages_name_the_input() {
    let err = MatchImagePairError::FeatureArrayLength {
        array: "descriptors",
        image: Some(3),
        rows: 4,
        width: 128,
        len: 384,
    };
    assert_eq!(
        err.to_string(),
        "descriptors[3] has 384 elements but 4 features of 128 elements each need 512"
    );
    let err = MatchImagePairError::CameraIndexOutOfRange {
        image: 1,
        camera: 2,
        num_cameras: 1,
    };
    assert_eq!(
        err.to_string(),
        "image 1 has camera index 2, out of range for 1 cameras"
    );
}
