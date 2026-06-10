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
