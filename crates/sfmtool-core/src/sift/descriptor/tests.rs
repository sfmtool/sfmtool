use super::*;
use crate::optical_flow::GrayImage;
use crate::sift::SiftParams;

// Default magnification / clamp (mirroring `SiftParams::default`) for the
// tests that call the descriptor functions directly.
const DEFAULT_MAGNIFICATION: f32 = 3.0;
const DEFAULT_CLAMP: f32 = 0.2;

/// A smooth image with a single dominant gradient direction (linear ramp
/// along `dir`), scaled by `amplitude`.
fn directional_ramp(w: u32, h: u32, dir: f32, amplitude: f32) -> GrayImage {
    let (s, c) = dir.sin_cos();
    let mut data = vec![0.0f32; (w * h) as usize];
    for row in 0..h {
        for col in 0..w {
            let x = col as f32;
            let y = row as f32;
            data[(row * w + col) as usize] = 0.5 + amplitude * (x * c + y * s);
        }
    }
    GrayImage::new(w, h, data)
}

fn center_keypoint(orientation: f32) -> SiftKeypoint {
    // Octave 0 of a doubled-by-default 80x80 image is 160x160; its center in
    // full-resolution coords is ~ (40, 40). Use abs_sigma at layer 1 as scale.
    SiftKeypoint::from_similarity(40.0, 40.0, 4.0, orientation, 0, 1.0, 0.1)
}

#[test]
fn test_single_direction_concentrates_orientation_bins() {
    // Gradient everywhere points along +x (dir = 0). With the descriptor
    // oriented to 0, every sample's relative orientation is ~0, so mass should
    // pile into orientation bin 0 of the histograms.
    let img = directional_ramp(80, 80, 0.0, 0.002);
    let ss = ScaleSpace::build(&img, &SiftParams::default());
    let kp = center_keypoint(0.0);
    let hist = accumulate_histogram(&ss, &kp, DEFAULT_MAGNIFICATION);

    // Sum the mass per orientation bin across all spatial cells.
    let mut per_ori = [0.0f32; B];
    for cell in 0..D * D {
        for o in 0..B {
            per_ori[o] += hist[cell * B + o];
        }
    }
    let total: f32 = per_ori.iter().sum();
    assert!(total > 0.0, "no descriptor mass accumulated");
    // Bin 0 (relative orientation 0) should dominate.
    let max_bin = (0..B)
        .max_by(|&a, &b| per_ori[a].partial_cmp(&per_ori[b]).unwrap())
        .unwrap();
    assert_eq!(max_bin, 0, "orientation mass not in bin 0: {per_ori:?}");
    assert!(
        per_ori[0] > 0.5 * total,
        "bin 0 should hold most mass: {per_ori:?}"
    );
}

#[test]
fn test_normalization_invariants() {
    let img = directional_ramp(80, 80, 0.7, 0.003);
    let ss = ScaleSpace::build(&img, &SiftParams::default());
    let kp = center_keypoint(0.3);
    let raw = accumulate_histogram(&ss, &kp, DEFAULT_MAGNIFICATION);
    let norm = normalize_clamp_renorm(&raw, DEFAULT_CLAMP);

    // Unit length.
    let len = l2_norm(&norm);
    assert!((len - 1.0).abs() < 1e-4, "not unit length: {len}");
    // The spec's normalization is a single clamp-then-renormalize pass
    // (matching OpenCV/COLMAP). Clamping to the clamp value and renormalizing
    // scales the whole vector up, so components that were pinned at the clamp
    // can creep modestly above it. Bound that creep generously rather than
    // expecting the strict `x <= clamp` invariant a multi-pass scheme would give.
    for &x in norm.iter() {
        assert!(
            x <= DEFAULT_CLAMP * 1.1,
            "component {x} exceeds clamp by too much"
        );
    }
}

#[test]
fn test_contrast_invariance() {
    // Multiplying the image gradients by a positive constant must not change
    // the quantized descriptor (normalization removes the scale).
    let img_a = directional_ramp(80, 80, 0.4, 0.002);
    let img_b = directional_ramp(80, 80, 0.4, 0.006); // 3x the gradient.
    let ss_a = ScaleSpace::build(&img_a, &SiftParams::default());
    let ss_b = ScaleSpace::build(&img_b, &SiftParams::default());
    let kp = center_keypoint(0.4);
    let da = compute_descriptor(&ss_a, &kp, DEFAULT_MAGNIFICATION, DEFAULT_CLAMP);
    let db = compute_descriptor(&ss_b, &kp, DEFAULT_MAGNIFICATION, DEFAULT_CLAMP);
    assert_eq!(da, db, "descriptor changed under contrast scaling");
}

#[test]
fn test_rotation_permutes_layout() {
    // Rotating the keypoint orientation by +90 degrees cyclically permutes the
    // orientation bins. With the `ori - theta` binning, increasing the keypoint
    // orientation by +90deg (2 of 8 bins) shifts relative-orientation bins *up*
    // by 2, so ori90[o] matches ori0[(o - 2) mod 8].
    let img = directional_ramp(80, 80, 0.0, 0.002);
    let ss = ScaleSpace::build(&img, &SiftParams::default());

    let hist0 = accumulate_histogram(&ss, &center_keypoint(0.0), DEFAULT_MAGNIFICATION);
    let hist90 = accumulate_histogram(&ss, &center_keypoint(PI / 2.0), DEFAULT_MAGNIFICATION);

    let mut ori0 = [0.0f32; B];
    let mut ori90 = [0.0f32; B];
    for cell in 0..D * D {
        for o in 0..B {
            ori0[o] += hist0[cell * B + o];
            ori90[o] += hist90[cell * B + o];
        }
    }
    // With `ori - theta` binning, +90deg (2 of 8 bins) shifts relative-orientation
    // bins up by 2. Compare ori90[o] to ori0[(o - 2) mod B].
    let total0: f32 = ori0.iter().sum();
    let total90: f32 = ori90.iter().sum();
    assert!(total0 > 0.0 && total90 > 0.0);
    for o in 0..B {
        let expected = ori0[(o + B - 2) % B] / total0;
        let got = ori90[o] / total90;
        assert!(
            (expected - got).abs() < 0.15,
            "rotation mismatch at bin {o}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn test_zero_image_zero_descriptor() {
    let img = GrayImage::new_constant(80, 80, 0.5);
    let ss = ScaleSpace::build(&img, &SiftParams::default());
    let kp = center_keypoint(0.0);
    let d = compute_descriptor(&ss, &kp, DEFAULT_MAGNIFICATION, DEFAULT_CLAMP);
    assert!(d.iter().all(|&b| b == 0), "flat image should give zeros");
}
