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

/// Two patches cover the same pixel and disagree; only one of them predicts the
/// target's actual content there. The weighted average must land nearer the flow
/// that fits, not halfway between the two.
///
/// This is a regression test for the weight being *live*. Eq. 3's clamp is written
/// on 0–255 intensities, so before `INTENSITY_SCALE` was applied to the [0, 1]
/// values `GrayImage` stores, the clamp always bound, every weight was exactly 1.0,
/// and densification silently degraded to a box average. That defect is invisible to
/// any test that only checks a single patch, or patches that agree.
#[test]
fn test_densify_weights_by_photometric_error() {
    // A horizontal ramp: intensity encodes position, so a wrong flow at a pixel
    // shows up as a proportional photometric error.
    let w = 8u32;
    let h = 4u32;
    let ramp: Vec<f32> = (0..w * h)
        .map(|i| (i % w) as f32 / (w - 1) as f32)
        .collect();
    let ref_image = GrayImage::new(w, h, ramp.clone());

    // `densify_flow` samples the target at (col + 0.5 + dx), so a true flow of
    // dx = +2 means tgt(col) == ref(col - 2): the ramp shifted right by 2px.
    let tgt_image = GrayImage::new(
        w,
        h,
        (0..w * h)
            .map(|i| {
                let col = ((i % w) as i32 - 2).clamp(0, w as i32 - 1) as u32;
                ramp[((i / w) * w + col) as usize]
            })
            .collect(),
    );

    let patch_size = 4;
    let truth = 2.0;
    let wrong = -2.0;
    let patches = vec![
        PatchResult {
            grid_x: 0,
            grid_y: 0,
            final_flow: (truth, 0.0),
        },
        PatchResult {
            grid_x: 0,
            grid_y: 0,
            final_flow: (wrong, 0.0),
        },
    ];

    let dense = densify_flow(&patches, &ref_image, &tgt_image, w, h, patch_size);

    // Interior pixel of the overlap, away from the clamped edges where both
    // candidates sample the same saturated value and genuinely tie.
    let (dx, _) = dense.get(1, 1);
    let midpoint = 0.5 * (truth + wrong);
    assert!(
        (dx - truth).abs() < (dx - midpoint).abs(),
        "densified dx {dx} is no closer to the correct flow {truth} than to the \
         unweighted midpoint {midpoint}: the photometric weight is not discriminating"
    );
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
