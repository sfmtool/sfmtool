use super::super::detect_keypoints;
use super::*;
use crate::features::optical_flow::GrayImage;

/// Render a Gaussian blob (bright on a flat mid-gray background) centered at
/// `(cx, cy)` with standard deviation `blob_sigma`.
fn gaussian_blob(w: u32, h: u32, cx: f32, cy: f32, blob_sigma: f32, amp: f32) -> GrayImage {
    let mut data = vec![0.5f32; (w * h) as usize];
    let inv = 1.0 / (2.0 * blob_sigma * blob_sigma);
    for row in 0..h {
        for col in 0..w {
            let dx = col as f32 + 0.5 - cx;
            let dy = row as f32 + 0.5 - cy;
            let v = 0.5 + amp * (-(dx * dx + dy * dy) * inv).exp();
            data[(row * w + col) as usize] = v;
        }
    }
    GrayImage::new(w, h, data)
}

#[test]
fn test_constant_image_no_keypoints() {
    let img = GrayImage::new_constant(64, 64, 0.5);
    let ss = ScaleSpace::build(&img, &SiftParams::default());
    let kps = detect_and_localize(&ss, &SiftParams::default());
    assert!(
        kps.is_empty(),
        "constant image yielded {} keypoints",
        kps.len()
    );
}

#[test]
fn test_gaussian_blob_detected() {
    let img = gaussian_blob(64, 64, 32.0, 32.0, 4.0, 0.4);
    let detection = detect_keypoints(&img, &SiftParams::default());
    let kps = &detection.keypoints;
    assert!(!kps.is_empty(), "blob produced no keypoints");
    // A keypoint near the blob center should exist.
    let near = kps
        .iter()
        .any(|k| (k.x - 32.0).abs() < 4.0 && (k.y - 32.0).abs() < 4.0);
    assert!(near, "no keypoint near blob center: {:?}", kps);
    // Its scale should be on the order of the blob sigma (a few px), not tiny
    // or enormous.
    let center_kp = kps
        .iter()
        .filter(|k| (k.x - 32.0).abs() < 4.0 && (k.y - 32.0).abs() < 4.0)
        .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
        .unwrap();
    let size = center_kp.scale();
    assert!(size > 1.0 && size < 30.0, "blob keypoint size {size}");
}

#[test]
fn test_edge_ridge_rejected() {
    // A long vertical bright ridge: strong response along an edge, so the
    // edge test should reject the (few) candidates it produces.
    let w = 80u32;
    let h = 80u32;
    let mut data = vec![0.5f32; (w * h) as usize];
    for row in 0..h {
        for col in 38..42 {
            data[(row * w + col) as usize] = 0.9;
        }
    }
    // Soften it a little so it has interior structure.
    let img = GrayImage::new(w, h, data);
    let detection = detect_keypoints(&img, &SiftParams::default());
    // Any surviving keypoints should not cluster along the ridge interior as
    // a well-localized blob; assert none sit squarely in the long uniform
    // middle of the ridge.
    let on_ridge_middle = detection
        .keypoints
        .iter()
        .filter(|k| k.x > 37.0 && k.x < 43.0 && k.y > 20.0 && k.y < 60.0);
    assert_eq!(
        on_ridge_middle.count(),
        0,
        "edge ridge produced keypoints along its length"
    );
}

#[test]
fn test_low_contrast_rejected() {
    // A very faint blob: above the structural noise but below the contrast
    // threshold, so it should be rejected.
    let img = gaussian_blob(64, 64, 32.0, 32.0, 4.0, 0.005);
    let detection = detect_keypoints(&img, &SiftParams::default());
    assert!(
        detection.keypoints.is_empty(),
        "low-contrast blob produced {} keypoints",
        detection.keypoints.len()
    );
}
