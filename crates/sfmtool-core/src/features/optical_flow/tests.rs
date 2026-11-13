// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

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
    let flow = compute_optical_flow(&img, &img, &params, None).unwrap();

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
    let flow = compute_optical_flow(&img_a, &img_b, &params, None).unwrap();

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

/// A smooth texture with structure on both axes, sampled at `(x - dx, y)`, so
/// the flow from the `dx = 0` image to this one is `(dx, 0)` everywhere.
fn smooth_texture(width: u32, height: u32, dx: f32) -> GrayImage {
    let mut img = GrayImage::new_constant(width, height, 0.0);
    for row in 0..height {
        for col in 0..width {
            let x = col as f32 - dx;
            let y = row as f32;
            let v = 0.5 + 0.2 * (0.4 * x).sin() + 0.2 * (0.33 * y + 0.1 * x).cos();
            img.set_pixel(col, row, v);
        }
    }
    img
}

#[test]
fn test_odd_size_edge_gets_flow_with_finest_scale_above_zero() {
    // At finest scale 1 the flow is solved at floor(65 / 2) x floor(49 / 2)
    // and upsampled to 64 x 48, one pixel short of the image on each axis.
    // The last column and row repeat their neighbours' flow rather than
    // being left at zero.
    let (w, h) = (65, 49);
    let img_a = smooth_texture(w, h, 0.0);
    let img_b = smooth_texture(w, h, 1.0);
    let params = DisFlowParams {
        finest_scale: Some(1),
        coarsest_scale: Some(1),
        ..DisFlowParams::default_quality()
    };
    let flow = compute_optical_flow(&img_a, &img_b, &params, None).unwrap();
    assert_eq!((flow.width(), flow.height()), (w, h));
    for row in 0..h {
        assert_eq!(flow.get(w - 1, row), flow.get(w - 2, row), "row {row}");
    }
    for col in 0..w {
        assert_eq!(flow.get(col, h - 1), flow.get(col, h - 2), "col {col}");
    }
    let (dx, _) = flow.get(w - 1, h / 2);
    assert!((dx - 1.0).abs() < 0.5, "last column dx {dx}");
}

#[test]
fn test_resize_flow_to_repeats_edges_and_crops() {
    let mut flow = FlowField::new(2, 2);
    flow.set(0, 0, 1.0, -1.0);
    flow.set(1, 0, 2.0, -2.0);
    flow.set(0, 1, 3.0, -3.0);
    flow.set(1, 1, 4.0, -4.0);
    let grown = resize_flow_to(flow.clone(), 3, 3);
    assert_eq!(grown.get(2, 0), (2.0, -2.0));
    assert_eq!(grown.get(0, 2), (3.0, -3.0));
    assert_eq!(grown.get(2, 2), (4.0, -4.0));
    assert_eq!(grown.get(1, 1), (4.0, -4.0));
    let cropped = resize_flow_to(flow, 1, 2);
    assert_eq!(cropped.u_slice(), &[1.0, 3.0]);
    assert_eq!(cropped.v_slice(), &[-1.0, -3.0]);
}

#[test]
fn test_full_resolution_dense_patches_beat_half_resolution() {
    // The setting the high_quality preset cannot reach: patches on every
    // pixel at full resolution. On a sub-pixel shift of a small image it is
    // closer to the true flow than the same patches solved at half
    // resolution and upsampled.
    let (w, h) = (64, 64);
    let img_a = smooth_texture(w, h, 0.0);
    let img_b = smooth_texture(w, h, 0.6);
    let error = |finest: u32| {
        let params = DisFlowParams {
            patch_size: 12,
            patch_stride: 1,
            finest_scale: Some(finest),
            coarsest_scale: Some(1),
            ..DisFlowParams::high_quality()
        };
        let flow = compute_optical_flow(&img_a, &img_b, &params, None).unwrap();
        let errors: Vec<f64> = (8..h - 8)
            .flat_map(|row| (8..w - 8).map(move |col| (col, row)))
            .map(|(col, row)| {
                let (dx, dy) = flow.get(col, row);
                f64::from((dx - 0.6).hypot(dy))
            })
            .collect();
        crate::numeric::median(&errors)
    };
    let full = error(0);
    let half = error(1);
    assert!(full < 0.05, "full-resolution median error {full}");
    assert!(full < 0.7 * half, "full {full} vs half {half}");
}

#[test]
fn test_invalid_params_return_errors() {
    let img = GrayImage::checkerboard(40, 24);
    let run = |params: DisFlowParams| {
        compute_optical_flow(&img, &img, &params, None)
            .err()
            .unwrap()
    };
    let base = DisFlowParams::default_quality;
    assert_eq!(
        run(DisFlowParams {
            patch_size: 0,
            ..base()
        }),
        OpticalFlowError::PatchSizeZero
    );
    assert_eq!(
        run(DisFlowParams {
            patch_stride: 0,
            ..base()
        }),
        OpticalFlowError::PatchStrideZero
    );
    assert_eq!(
        run(DisFlowParams {
            patch_size: 25,
            ..base()
        }),
        OpticalFlowError::PatchLargerThanImage {
            patch_size: 25,
            width: 40,
            height: 24
        }
    );
    // A patch the size of the image's shorter side is accepted.
    let params = DisFlowParams {
        patch_size: 24,
        ..base()
    };
    assert!(compute_optical_flow(&img, &img, &params, None).is_ok());
    for (field, params) in [
        (
            "grad_descent_iterations",
            DisFlowParams {
                grad_descent_iterations: 0,
                ..base()
            },
        ),
        (
            "variational_jacobi_iterations",
            DisFlowParams {
                variational_jacobi_iterations: 0,
                ..base()
            },
        ),
        (
            "variational_outer_iterations_base",
            DisFlowParams {
                variational_outer_iterations_base: 0,
                ..base()
            },
        ),
    ] {
        assert_eq!(run(params), OpticalFlowError::ZeroIterations { field });
    }
    for value in [-1.0, f32::INFINITY] {
        assert_eq!(
            run(DisFlowParams {
                variational_gamma: value,
                ..base()
            }),
            OpticalFlowError::InvalidWeight {
                field: "variational_gamma",
                value
            }
        );
    }
    assert!(matches!(
        run(DisFlowParams {
            variational_alpha: f32::NAN,
            ..base()
        }),
        OpticalFlowError::InvalidWeight {
            field: "variational_alpha",
            ..
        }
    ));
    // 24 rows halve to 1 row at level 4 and to none at level 5.
    assert_eq!(
        run(DisFlowParams {
            finest_scale: Some(5),
            ..base()
        }),
        OpticalFlowError::ScaleTooCoarse {
            field: "finest_scale",
            scale: 5,
            max_scale: 4
        }
    );
    assert_eq!(
        run(DisFlowParams {
            coarsest_scale: Some(u32::MAX),
            ..base()
        }),
        OpticalFlowError::ScaleTooCoarse {
            field: "coarsest_scale",
            scale: u32::MAX,
            max_scale: 4
        }
    );
    assert_eq!(
        run(DisFlowParams {
            finest_scale: Some(2),
            coarsest_scale: Some(1),
            ..base()
        }),
        OpticalFlowError::FinestAboveCoarsest {
            finest: 2,
            coarsest: 1
        }
    );
    let other = GrayImage::checkerboard(40, 23);
    assert_eq!(
        compute_optical_flow(&img, &other, &base(), None)
            .err()
            .unwrap(),
        OpticalFlowError::ImageSizeMismatch {
            a: (40, 24),
            b: (40, 23)
        }
    );
    let init = FlowField::new(39, 24);
    assert_eq!(
        compute_optical_flow_with_init(&img, &img, &base(), &init, None)
            .err()
            .unwrap(),
        OpticalFlowError::InitialFlowSizeMismatch {
            image: (40, 24),
            flow: (39, 24)
        }
    );
}

#[test]
fn test_stride_beyond_the_image_places_one_patch_per_axis() {
    // Any stride larger than the image leaves one patch per axis, so a stride
    // of u32::MAX, whose next patch position overflows, gives the same flow
    // as a stride of the image width.
    let img_a = smooth_texture(40, 24, 0.0);
    let img_b = smooth_texture(40, 24, 0.6);
    let flow = |patch_stride: u32| {
        let params = DisFlowParams {
            patch_stride,
            finest_scale: Some(0),
            coarsest_scale: Some(0),
            ..DisFlowParams::default_quality()
        };
        compute_optical_flow(&img_a, &img_b, &params, None).unwrap()
    };
    let max = flow(u32::MAX);
    let one = flow(40);
    assert_eq!(max.u_slice(), one.u_slice());
    assert_eq!(max.v_slice(), one.v_slice());
    assert!(max.u_slice().iter().any(|&u| u != 0.0));
}

#[test]
fn test_resolve_scales() {
    let base = DisFlowParams::default_quality;
    // Automatic: floor(log2(2 * 640 / 40)) = 5, finest 5 - 2.
    assert_eq!(base().resolve_scales(640, 480), Ok((3, 5)));
    // A short image lowers the automatic coarsest scale to its deepest
    // level: 9 rows reach level 3, one row high.
    assert_eq!(base().resolve_scales(640, 9), Ok((1, 3)));
    // An explicit finest scale raises an automatic coarsest scale to it.
    let params = DisFlowParams {
        finest_scale: Some(4),
        ..base()
    };
    assert_eq!(params.resolve_scales(100, 100), Ok((4, 4)));
}

#[test]
fn test_presets() {
    // Just verify presets don't panic and produce valid params
    // The strides are those the paper's overlaps (0.3, 0.4, 0.75) give,
    // floor(patch_size * (1 - overlap)).
    let fast = DisFlowParams::fast();
    assert_eq!(fast.patch_size, 8);
    assert_eq!(fast.patch_stride, 5);
    assert!(!fast.variational_refinement);

    let default = DisFlowParams::default_quality();
    assert_eq!(default.patch_size, 8);
    assert_eq!(default.patch_stride, 4);
    assert!(default.variational_refinement);

    let hq = DisFlowParams::high_quality();
    assert_eq!(hq.patch_size, 12);
    assert_eq!(hq.patch_stride, 3);
    assert_eq!(hq.finest_scale, Some(1));
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

    let flow = compute_optical_flow_with_init(&img, &img, &params, &init, None).unwrap();

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

    let flow = compute_optical_flow_with_init(&img_a, &img_b, &params, &init, None).unwrap();

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
