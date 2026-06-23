use super::*;

#[test]
fn test_sample_bilinear_at_pixel_center() {
    let img = GrayImage::new(3, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    // Pixel center of (0,0) is at (0.5, 0.5)
    let val = sample_bilinear(&img, 0.5, 0.5);
    assert!((val - 0.1).abs() < 1e-6, "got {}", val);

    // Pixel center of (1,1) is at (1.5, 1.5)
    let val = sample_bilinear(&img, 1.5, 1.5);
    assert!((val - 0.5).abs() < 1e-6, "got {}", val);
}

#[test]
fn test_sample_bilinear_interpolated() {
    let img = GrayImage::new(2, 2, vec![0.0, 1.0, 0.0, 1.0]);
    // At (1.0, 0.5): midpoint between pixel (0,0) and (1,0)
    let val = sample_bilinear(&img, 1.0, 0.5);
    assert!((val - 0.5).abs() < 1e-6, "got {}", val);
}

#[test]
fn test_sample_bilinear_four_pixel_average() {
    let img = GrayImage::new(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    // At (1.0, 1.0): center of 4 pixels -> average = (0+1+2+3)/4 = 1.5
    let val = sample_bilinear(&img, 1.0, 1.0);
    assert!((val - 1.5).abs() < 1e-6, "got {}", val);
}

#[test]
fn test_warp_image_zero_flow() {
    let img = GrayImage::new(4, 4, (0..16).map(|i| i as f32 / 15.0).collect());
    let flow = FlowField::new(4, 4);
    let warped = warp_image(&img, &flow);
    for i in 0..16 {
        assert!(
            (warped.data()[i] - img.data()[i]).abs() < 1e-5,
            "Pixel {} differs: {} vs {}",
            i,
            warped.data()[i],
            img.data()[i]
        );
    }
}
