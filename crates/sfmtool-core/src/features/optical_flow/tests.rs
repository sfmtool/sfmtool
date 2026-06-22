use super::*;

#[test]
fn test_flow_field_new() {
    let flow = FlowField::new(10, 20);
    assert_eq!(flow.width(), 10);
    assert_eq!(flow.height(), 20);
    assert_eq!(flow.u_slice().len(), 10 * 20);
    assert_eq!(flow.v_slice().len(), 10 * 20);
}

#[test]
fn test_flow_field_get_set() {
    let mut flow = FlowField::new(10, 10);
    flow.set(3, 5, 1.5, -2.5);
    let (dx, dy) = flow.get(3, 5);
    assert!((dx - 1.5).abs() < 1e-6);
    assert!((dy - (-2.5)).abs() < 1e-6);
}

#[test]
fn test_flow_field_sample_at_pixel_center() {
    let mut flow = FlowField::new(4, 4);
    flow.set(0, 0, 3.0, 4.0);
    let (dx, dy) = flow.sample(0.5, 0.5);
    assert!((dx - 3.0).abs() < 1e-6);
    assert!((dy - 4.0).abs() < 1e-6);
}

#[test]
fn test_advect_points_constant_flow() {
    let mut flow = FlowField::new(10, 10);
    for row in 0..10 {
        for col in 0..10 {
            flow.set(col, row, 2.0, 3.0);
        }
    }
    let points = vec![(1.5, 1.5), (5.5, 5.5)];
    let advected = flow.advect_points(&points);
    assert!((advected[0].0 - 3.5).abs() < 1e-5);
    assert!((advected[0].1 - 4.5).abs() < 1e-5);
    assert!((advected[1].0 - 7.5).abs() < 1e-5);
    assert!((advected[1].1 - 8.5).abs() < 1e-5);
}

#[test]
fn test_upsample_2x() {
    let mut flow = FlowField::new(4, 4);
    for row in 0..4 {
        for col in 0..4 {
            flow.set(col, row, 1.0, 2.0);
        }
    }
    let up = flow.upsample_2x();
    assert_eq!(up.width(), 8);
    assert_eq!(up.height(), 8);
    // Magnitude should be doubled
    let (dx, dy) = up.get(4, 4);
    assert!((dx - 2.0).abs() < 0.1);
    assert!((dy - 4.0).abs() < 0.1);
}

#[test]
fn test_gray_image_from_u8() {
    let data = vec![0u8, 128, 255];
    let img = GrayImage::from_u8(3, 1, &data);
    assert!((img.get_pixel(0, 0) - 0.0).abs() < 1e-6);
    assert!((img.get_pixel(1, 0) - 128.0 / 255.0).abs() < 1e-6);
    assert!((img.get_pixel(2, 0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_identical_images_near_zero_flow() {
    let img = GrayImage::checkerboard(64, 64);
    let params = DisFlowParams::fast();
    let flow = compute_optical_flow(&img, &img, &params, None);

    // Flow should be near zero for identical images
    let mut max_flow = 0.0f32;
    for row in 0..flow.height() {
        for col in 0..flow.width() {
            let (dx, dy) = flow.get(col, row);
            max_flow = max_flow.max(dx.abs()).max(dy.abs());
        }
    }
    assert!(
        max_flow < 1.0,
        "Identical images should produce near-zero flow, got max {}",
        max_flow
    );
}

#[test]
fn test_horizontal_shift() {
    let img_a = GrayImage::checkerboard(128, 128);
    let img_b = GrayImage::shifted(&img_a, 3.0, 0.0);
    let params = DisFlowParams {
        finest_scale: Some(0),
        coarsest_scale: Some(3),
        variational_refinement: false,
        ..DisFlowParams::default_quality()
    };
    let flow = compute_optical_flow(&img_a, &img_b, &params, None);

    // Check flow in the center region (away from borders)
    let mut sum_dx = 0.0;
    let mut sum_dy = 0.0;
    let mut count = 0;
    let margin = 20;
    for row in margin..flow.height() - margin {
        for col in margin..flow.width() - margin {
            let (dx, dy) = flow.get(col, row);
            sum_dx += dx;
            sum_dy += dy;
            count += 1;
        }
    }
    let avg_dx = sum_dx / count as f32;
    let avg_dy = sum_dy / count as f32;

    assert!(
        (avg_dx - 3.0).abs() < 2.0,
        "Expected avg dx ~3.0, got {}",
        avg_dx
    );
    assert!(avg_dy.abs() < 2.0, "Expected avg dy ~0.0, got {}", avg_dy);
}

#[test]
fn test_presets() {
    // Just verify presets don't panic and produce valid params
    let fast = DisFlowParams::fast();
    assert_eq!(fast.patch_size, 8);
    assert!(!fast.variational_refinement);

    let default = DisFlowParams::default_quality();
    assert_eq!(default.patch_size, 8);
    assert!(default.variational_refinement);

    let hq = DisFlowParams::high_quality();
    assert_eq!(hq.patch_size, 12);
    assert!(hq.variational_refinement);
}

#[test]
fn test_coarsest_scale_computation() {
    let params = DisFlowParams::default_quality();
    // For 2880 width: floor(log2(2*2880/(5*8))) = floor(log2(144)) = 7
    assert_eq!(params.compute_coarsest_scale(2880), 7);
    // For 640 width: floor(log2(2*640/(5*8))) = floor(log2(32)) = 5
    assert_eq!(params.compute_coarsest_scale(640), 5);
}

#[test]
fn test_downsample_2x() {
    let mut flow = FlowField::new(8, 8);
    for row in 0..8 {
        for col in 0..8 {
            flow.set(col, row, 4.0, 6.0);
        }
    }
    let down = flow.downsample_2x();
    assert_eq!(down.width(), 4);
    assert_eq!(down.height(), 4);
    // Magnitude should be halved
    let (dx, dy) = down.get(2, 2);
    assert!((dx - 2.0).abs() < 0.1, "Expected dx ~2.0, got {}", dx);
    assert!((dy - 3.0).abs() < 0.1, "Expected dy ~3.0, got {}", dy);
}

#[test]
fn test_downsample_upsample_roundtrip() {
    let mut flow = FlowField::new(8, 8);
    for row in 0..8 {
        for col in 0..8 {
            flow.set(col, row, 4.0, -2.0);
        }
    }
    let roundtrip = flow.downsample_2x().upsample_2x();
    assert_eq!(roundtrip.width(), 8);
    assert_eq!(roundtrip.height(), 8);
    // Center pixel should recover roughly the original values
    let (dx, dy) = roundtrip.get(4, 4);
    assert!((dx - 4.0).abs() < 0.5, "Expected dx ~4.0, got {}", dx);
    assert!((dy - (-2.0)).abs() < 0.5, "Expected dy ~-2.0, got {}", dy);
}

#[test]
fn test_compose_flow_with_zero() {
    let mut flow_ab = FlowField::new(10, 10);
    for row in 0..10 {
        for col in 0..10 {
            flow_ab.set(col, row, 2.0, 3.0);
        }
    }
    let flow_bc = FlowField::new(10, 10); // zero flow
    let composed = compose_flow(&flow_ab, &flow_bc);
    // Composing with zero should give the original
    let (dx, dy) = composed.get(5, 5);
    assert!((dx - 2.0).abs() < 1e-5);
    assert!((dy - 3.0).abs() < 1e-5);
}

#[test]
fn test_compose_flow_additive() {
    // Two constant flows should add
    let mut flow_ab = FlowField::new(10, 10);
    let mut flow_bc = FlowField::new(10, 10);
    for row in 0..10 {
        for col in 0..10 {
            flow_ab.set(col, row, 1.0, 0.0);
            flow_bc.set(col, row, 0.0, 2.0);
        }
    }
    let composed = compose_flow(&flow_ab, &flow_bc);
    let (dx, dy) = composed.get(5, 5);
    assert!((dx - 1.0).abs() < 1e-5, "Expected dx ~1.0, got {}", dx);
    assert!((dy - 2.0).abs() < 1e-5, "Expected dy ~2.0, got {}", dy);
}

#[test]
fn test_compute_optical_flow_with_init_identical() {
    // With identical images, init flow should be refined toward zero
    let img = GrayImage::checkerboard(64, 64);
    let params = DisFlowParams::fast();

    let mut init = FlowField::new(64, 64);
    for row in 0..64 {
        for col in 0..64 {
            init.set(col, row, 1.0, 1.0);
        }
    }

    let flow = compute_optical_flow_with_init(&img, &img, &params, &init, None);

    // Should be near zero (solver corrects the bad init)
    let mut max_flow = 0.0f32;
    let margin = 10;
    for row in margin..flow.height() - margin {
        for col in margin..flow.width() - margin {
            let (dx, dy) = flow.get(col, row);
            max_flow = max_flow.max(dx.abs()).max(dy.abs());
        }
    }
    assert!(
        max_flow < 2.0,
        "Expected near-zero flow for identical images with init, got max {}",
        max_flow
    );
}

#[test]
fn test_compute_optical_flow_with_init_shift() {
    let img_a = GrayImage::checkerboard(128, 128);
    let img_b = GrayImage::shifted(&img_a, 3.0, 0.0);
    let params = DisFlowParams {
        finest_scale: Some(0),
        coarsest_scale: Some(3),
        variational_refinement: false,
        ..DisFlowParams::default_quality()
    };

    // Provide a good initial flow
    let mut init = FlowField::new(128, 128);
    for row in 0..128 {
        for col in 0..128 {
            init.set(col, row, 3.0, 0.0);
        }
    }

    let flow = compute_optical_flow_with_init(&img_a, &img_b, &params, &init, None);

    // Check center region
    let mut sum_dx = 0.0;
    let mut count = 0;
    let margin = 20;
    for row in margin..flow.height() - margin {
        for col in margin..flow.width() - margin {
            sum_dx += flow.get(col, row).0;
            count += 1;
        }
    }
    let avg_dx = sum_dx / count as f32;
    assert!(
        (avg_dx - 3.0).abs() < 1.5,
        "Expected avg dx ~3.0, got {}",
        avg_dx
    );
}
