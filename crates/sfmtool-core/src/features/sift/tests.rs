// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_default_params() {
    let p = SiftParams::default();
    assert_eq!(p.octave_layers, 3);
    assert_eq!(p.sigma, 1.6);
    assert_eq!(p.blur_radius_factor, 2.25);
    assert_eq!(p.input_sigma, 0.5);
    assert!(p.double_image);
    assert_eq!(p.contrast_threshold, 0.0067);
    assert_eq!(p.edge_threshold, 10.0);
    assert_eq!(p.max_num_features, Some(8192));
    assert_eq!(p.orientation_bins, 36);
    assert_eq!(p.descriptor_width, 4);
    assert_eq!(p.descriptor_bins, 8);
}

#[test]
fn test_keypoint_similarity_roundtrip() {
    let scale = 4.0f32;
    let orientation = 0.7f32;
    let kp = SiftKeypoint::from_similarity(10.0, 20.0, scale, orientation, 1, 1.5, 0.1);
    assert!((kp.scale() - scale).abs() < 1e-5, "scale {}", kp.scale());
    assert!(
        (kp.orientation() - orientation).abs() < 1e-5,
        "orientation {}",
        kp.orientation()
    );
    // The affine shape is the expected scaled rotation.
    let (sin, cos) = orientation.sin_cos();
    assert!((kp.affine_shape[0][0] - scale * cos).abs() < 1e-5);
    assert!((kp.affine_shape[0][1] - (-scale * sin)).abs() < 1e-5);
    assert!((kp.affine_shape[1][0] - scale * sin).abs() < 1e-5);
    assert!((kp.affine_shape[1][1] - scale * cos).abs() < 1e-5);
}

#[test]
fn test_descriptors_container() {
    let mut d = Descriptors::default();
    assert!(d.is_empty());
    d = Descriptors::from_rows(vec![[1u8; 128], [2u8; 128]]);
    assert_eq!(d.len(), 2);
    assert_eq!(d.rows()[1][0], 2);
}

/// Load a checked-in real image, run the full detector, and sanity-check the
/// output: a non-trivial keypoint count, finite in-bounds coordinates,
/// positive scales, and descending-size ordering.
#[test]
fn test_detect_keypoints_real_image() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_01.jpg"
    );
    let dynimg = image::open(path).expect("load test image").to_rgb8();
    let (w, h) = (dynimg.width(), dynimg.height());
    // Convert to gray via the default image-to-gray formula on the raw RGB.
    let params = SiftParams::default();
    let img = gray_from_rgb(w, h, dynimg.as_raw(), &params.image_to_gray);

    let detection = detect_keypoints(&img, &params);
    let kps = &detection.keypoints;

    // Plausible non-trivial count for a 270x480 textured image.
    assert!(
        kps.len() > 50,
        "expected a non-trivial keypoint count, got {}",
        kps.len()
    );

    for kp in kps {
        assert!(kp.x.is_finite() && kp.y.is_finite(), "non-finite coord");
        assert!(
            kp.x >= 0.0 && kp.x < w as f32 && kp.y >= 0.0 && kp.y < h as f32,
            "coord ({}, {}) out of bounds {}x{}",
            kp.x,
            kp.y,
            w,
            h
        );
        assert!(kp.scale() > 0.0, "non-positive scale {}", kp.scale());
    }

    // Sorted by descending feature size.
    for pair in kps.windows(2) {
        assert!(
            pair[0].scale() >= pair[1].scale(),
            "not sorted by descending size: {} then {}",
            pair[0].scale(),
            pair[1].scale()
        );
    }
}

/// `max_num_features` caps the output to the largest-scale keypoints.
#[test]
fn test_max_num_features_cap() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_01.jpg"
    );
    let dynimg = image::open(path).expect("load test image").to_rgb8();
    let (w, h) = (dynimg.width(), dynimg.height());
    let mut params = SiftParams::default();
    let img = gray_from_rgb(w, h, dynimg.as_raw(), &params.image_to_gray);

    // Uncapped pool.
    params.max_num_features = None;
    let uncapped = detect_keypoints(&img, &params).keypoints;
    assert!(
        uncapped.len() > 100,
        "need a non-trivial pool to test the cap, got {}",
        uncapped.len()
    );

    // Cap to half the pool. Enough candidates exist to fill it exactly.
    let cap = uncapped.len() / 2;
    params.max_num_features = Some(cap);
    let capped = detect_keypoints(&img, &params).keypoints;
    assert_eq!(capped.len(), cap, "hard cap not honored");

    // The cap keeps the largest-scale keypoints: the largest is retained and
    // the smallest retained scale is no smaller than the first dropped one.
    assert_eq!(capped[0].scale(), uncapped[0].scale());
    assert!(capped.last().unwrap().scale() >= uncapped[cap].scale());
}

/// End-to-end `extract_sift` on a real image: keypoints and descriptors are
/// parallel, every descriptor has at least one non-zero byte, and all bytes
/// are in range (trivially true for `u8`, but assert the count alignment and
/// non-degeneracy).
#[test]
fn test_extract_sift_real_image() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_01.jpg"
    );
    let dynimg = image::open(path).expect("load test image").to_rgb8();
    let (w, h) = (dynimg.width(), dynimg.height());
    let params = SiftParams::default();
    let img = gray_from_rgb(w, h, dynimg.as_raw(), &params.image_to_gray);

    let features = extract_sift(&img, &params);
    assert_eq!(
        features.keypoints.len(),
        features.descriptors.len(),
        "keypoint/descriptor count mismatch"
    );
    assert!(!features.keypoints.is_empty(), "no features extracted");

    for (i, row) in features.descriptors.rows().iter().enumerate() {
        assert!(row.iter().any(|&b| b != 0), "descriptor {i} is all zero");
        // Bytes are u8 so within [0, 255] by construction; assert the length.
        assert_eq!(row.len(), 128);
    }
}

#[test]
fn test_extract_sift_partial_describes_prefix() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_01.jpg"
    );
    let dynimg = image::open(path).expect("load test image").to_rgb8();
    let (w, h) = (dynimg.width(), dynimg.height());
    let params = SiftParams::default();
    let img = gray_from_rgb(w, h, dynimg.as_raw(), &params.image_to_gray);

    let full = extract_sift(&img, &params);
    let n = full.keypoints.len();
    assert!(n > 16, "need enough keypoints for the test (got {n})");

    // A cap describes only the prefix: every keypoint is still returned, but
    // only `k` descriptors, and they equal the first `k` of the full extract
    // (same keypoints, same order).
    let partial = extract_sift_partial(&img, &params, Some(16));
    assert_eq!(
        partial.keypoints.len(),
        n,
        "detection must find every keypoint"
    );
    assert_eq!(
        partial.descriptors.len(),
        16,
        "only the prefix is described"
    );
    for i in 0..16 {
        assert_eq!(
            partial.descriptors.rows()[i],
            full.descriptors.rows()[i],
            "described prefix must match the full extraction at {i}"
        );
    }

    // A cap >= the keypoint count (or None) describes everything.
    assert_eq!(
        extract_sift_partial(&img, &params, Some(n + 100))
            .descriptors
            .len(),
        n
    );
    assert_eq!(
        extract_sift_partial(&img, &params, None).descriptors.len(),
        n
    );
}

/// Load the checked-in test image as the detector sees it.
fn seoul_bull_gray(params: &SiftParams) -> GrayImage {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_01.jpg"
    );
    let dynimg = image::open(path).expect("load test image").to_rgb8();
    let (w, h) = (dynimg.width(), dynimg.height());
    gray_from_rgb(w, h, dynimg.as_raw(), &params.image_to_gray)
}

/// Describing a detected keypoint at its own position, shape and size
/// reproduces the extractor's descriptor byte for byte: the query path derives
/// the same `(octave, layer)` detection recorded and runs the same kernel.
///
/// Both query forms are covered -- the stored 2x2 shape, and the
/// `(scale, orientation)` pair recovered from it -- because they are the two ways
/// a caller has a keypoint in hand.
#[test]
fn test_describe_keypoints_matches_the_extractor() {
    let params = SiftParams::default();
    let img = seoul_bull_gray(&params);

    let features = extract_sift(&img, &params);
    let n = features.keypoints.len();
    assert!(n > 50, "need a non-trivial keypoint count, got {n}");

    let from_shape: Vec<QueryKeypoint> = features
        .keypoints
        .iter()
        .map(|kp| QueryKeypoint {
            x: kp.x,
            y: kp.y,
            affine_shape: kp.affine_shape,
        })
        .collect();
    let from_similarity: Vec<QueryKeypoint> = features
        .keypoints
        .iter()
        .map(|kp| QueryKeypoint::from_similarity(kp.x, kp.y, kp.scale(), kp.orientation()))
        .collect();

    for (form, queries) in [("shape", &from_shape), ("similarity", &from_similarity)] {
        let described =
            describe_keypoints(&img, &params, queries).expect("every keypoint is valid");
        assert_eq!(described.len(), n);
        for (i, (a, b)) in features
            .descriptors
            .rows()
            .iter()
            .zip(described.rows())
            .enumerate()
        {
            assert_eq!(
                a, b,
                "{form} query {i} at ({}, {}) must reproduce the extractor's descriptor",
                queries[i].x, queries[i].y
            );
        }
    }
}

/// The `(octave, layer)` a query's size implies is the one detection recorded
/// for that keypoint -- the invariant the byte-identity above rests on.
#[test]
fn test_query_keypoint_recovers_the_detected_octave_and_layer() {
    let params = SiftParams::default();
    let img = seoul_bull_gray(&params);
    let detection = detect_keypoints(&img, &params);

    for kp in &detection.keypoints {
        let query = QueryKeypoint {
            x: kp.x,
            y: kp.y,
            affine_shape: kp.affine_shape,
        };
        let derived = query.to_sift_keypoint(&detection.scale_space);
        assert_eq!(
            derived.octave,
            kp.octave,
            "octave for a keypoint of size {}",
            kp.scale()
        );
        assert_eq!(
            derived.layer.round(),
            kp.layer.round(),
            "pyramid level for a keypoint of size {}",
            kp.scale()
        );
        // The size round-trips through the pair.
        let scale = detection
            .scale_space
            .abs_sigma_full(derived.octave, derived.layer as f64);
        assert!(
            ((scale as f32) - kp.scale()).abs() <= 1e-4 * kp.scale(),
            "size {} came back as {scale}",
            kp.scale()
        );
    }
}

/// A keypoint the image does not contain, or one with no size, is refused by
/// name rather than panicking or silently describing noise.
#[test]
fn test_describe_keypoints_refuses_bad_queries() {
    let params = SiftParams::default();
    let img = seoul_bull_gray(&params);
    let (w, h) = (img.width(), img.height());
    let good = QueryKeypoint::from_similarity(10.0, 20.0, 4.0, 0.3);

    for bad in [
        QueryKeypoint::from_similarity(-1.0, 20.0, 4.0, 0.0),
        QueryKeypoint::from_similarity(10.0, h as f32, 4.0, 0.0),
        QueryKeypoint::from_similarity(w as f32 + 5.0, 20.0, 4.0, 0.0),
        QueryKeypoint::from_similarity(f32::NAN, 20.0, 4.0, 0.0),
    ] {
        let err = describe_keypoints(&img, &params, &[good, bad]).expect_err("outside the image");
        assert!(
            matches!(err, DescribeKeypointsError::OutsideImage { index: 1, .. }),
            "expected an OutsideImage refusal naming keypoint 1, got {err:?}"
        );
        assert!(err.to_string().contains("outside"), "{err}");
    }

    // A *negative* scale is not degenerate: the shape it builds is the same
    // keypoint turned 180 degrees, and its column norms are positive. What has
    // no size is a zero or non-finite shape.
    for shape in [
        QueryKeypoint::from_similarity(10.0, 20.0, 0.0, 0.0).affine_shape,
        QueryKeypoint::from_similarity(10.0, 20.0, f32::NAN, 0.0).affine_shape,
        [[f32::INFINITY, 0.0], [0.0, 4.0]],
    ] {
        let bad = QueryKeypoint {
            x: 10.0,
            y: 20.0,
            affine_shape: shape,
        };
        let err = describe_keypoints(&img, &params, &[good, bad]).expect_err("no size");
        assert!(
            matches!(err, DescribeKeypointsError::BadScale { index: 1, .. }),
            "expected a BadScale refusal naming keypoint 1, got {err:?}"
        );
        assert!(err.to_string().contains("positive size"), "{err}");
    }

    // An empty query is not an error; it describes nothing.
    assert!(describe_keypoints(&img, &params, &[])
        .expect("empty is fine")
        .is_empty());
}

/// A keypoint at a size no octave holds is described at the nearest octave
/// rather than refused, and a size finer than octave 0 is not silently turned
/// into a negative octave index.
#[test]
fn test_query_keypoint_clamps_a_size_outside_the_pyramid() {
    let params = SiftParams::default();
    let img = seoul_bull_gray(&params);
    let scale_space = ScaleSpace::build(&img, &params);
    let octaves = scale_space.num_octaves() as i32;

    let tiny = QueryKeypoint::from_similarity(100.0, 100.0, 1e-4, 0.0);
    assert_eq!(tiny.to_sift_keypoint(&scale_space).octave, 0);
    let huge = QueryKeypoint::from_similarity(100.0, 100.0, 1e6, 0.0);
    assert_eq!(huge.to_sift_keypoint(&scale_space).octave, octaves - 1);

    // Both still describe, without panicking.
    let described = describe_keypoints(&img, &params, &[tiny, huge]).expect("valid queries");
    assert_eq!(described.len(), 2);
}
