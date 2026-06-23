use super::*;

#[test]
fn test_pyramid_levels() {
    let img = GrayImage::checkerboard(64, 64);
    let pyr = ImagePyramid::build(&img, 4);
    assert_eq!(pyr.num_levels(), 4);
    assert_eq!(pyr.level(0).width(), 64);
    assert_eq!(pyr.level(0).height(), 64);
    assert_eq!(pyr.level(1).width(), 32);
    assert_eq!(pyr.level(1).height(), 32);
    assert_eq!(pyr.level(2).width(), 16);
    assert_eq!(pyr.level(2).height(), 16);
    assert_eq!(pyr.level(3).width(), 8);
    assert_eq!(pyr.level(3).height(), 8);
}

#[test]
fn test_pyramid_preserves_original() {
    let img = GrayImage::checkerboard(32, 32);
    let pyr = ImagePyramid::build(&img, 3);
    // Level 0 should be identical to the original
    assert_eq!(pyr.level(0).data(), img.data());
}

#[test]
fn test_blur_downsample_constant_image() {
    let img = GrayImage::new_constant(16, 16, 0.5);
    let down = blur_downsample_2x(&img);
    assert_eq!(down.width(), 8);
    assert_eq!(down.height(), 8);
    // Blurring+downsampling a constant image should preserve the constant
    for &val in down.data() {
        assert!((val - 0.5).abs() < 1e-5, "Expected 0.5, got {}", val);
    }
}

#[test]
fn test_pyramid_single_level() {
    let img = GrayImage::new_constant(8, 8, 0.3);
    let pyr = ImagePyramid::build(&img, 1);
    assert_eq!(pyr.num_levels(), 1);
    assert_eq!(pyr.level(0).width(), 8);
}

#[test]
fn test_blur_downsample_centering() {
    // Create image with a linear gradient: pixel value = column index
    let w = 16u32;
    let h = 4u32;
    let data: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|c| c as f32)).collect();
    let img = GrayImage::new(w, h, data);
    let down = blur_downsample_2x(&img);

    assert_eq!(down.width(), 8);
    assert_eq!(down.height(), 2);

    // For a linear gradient, each output pixel should be centered between
    // input pairs: output[oc] ≈ 2*oc + 0.5, since the 6-tap kernel is
    // centered at 2*oc + 0.5 and a linear function evaluates exactly at
    // the center regardless of kernel shape.
    // Interior pixels (away from edges) should be exact.
    for oc in 2..6 {
        let expected = 2.0 * oc as f32 + 0.5;
        let actual = down.data()[oc]; // row 0
        assert!(
            (actual - expected).abs() < 1e-4,
            "col {}: expected {}, got {}",
            oc,
            expected,
            actual
        );
    }
}
