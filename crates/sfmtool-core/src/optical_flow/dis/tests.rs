use super::*;

#[test]
fn test_compute_gradients_constant() {
    let img = GrayImage::new_constant(8, 8, 0.5);
    let (gx, gy) = compute_gradients(&img);
    for &v in gx.data() {
        assert!(v.abs() < 1e-6);
    }
    for &v in gy.data() {
        assert!(v.abs() < 1e-6);
    }
}

#[test]
fn test_compute_gradients_horizontal_ramp() {
    // Linear ramp in x: pixel values = col / (w-1)
    let w = 8u32;
    let h = 4u32;
    let data: Vec<f32> = (0..h)
        .flat_map(|_| (0..w).map(|c| c as f32 / (w - 1) as f32))
        .collect();
    let img = GrayImage::new(w, h, data);
    let (gx, gy) = compute_gradients(&img);

    // Interior pixels should have gx ≈ 1/(w-1) * 0.5 * 2 = 1/(w-1)
    // which is the central difference of a linear ramp with step 1/(w-1)
    let expected_gx = 1.0 / (w - 1) as f32;
    for row in 0..h {
        for col in 1..w - 1 {
            let val = gx.get_pixel(col, row);
            assert!(
                (val - expected_gx).abs() < 1e-5,
                "gx at ({},{}) = {}, expected {}",
                col,
                row,
                val,
                expected_gx
            );
        }
    }

    // gy should be ~0 for horizontal ramp
    for row in 1..h - 1 {
        for col in 0..w {
            let val = gy.get_pixel(col, row);
            assert!(val.abs() < 1e-5, "gy at ({},{}) = {}", col, row, val);
        }
    }
}

/// Generate a deterministic pseudo-random image for SIMD vs scalar testing.
/// Uses a simple LCG to avoid pulling in a rand crate dependency.
fn pseudo_random_image(width: u32, height: u32, seed: u32) -> GrayImage {
    let mut state = seed;
    let data: Vec<f32> = (0..(width as usize * height as usize))
        .map(|_| {
            // LCG: state = state * 1103515245 + 12345
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            // Map to [0, 1]
            (state >> 16) as f32 / 65535.0
        })
        .collect();
    GrayImage::new(width, height, data)
}

/// Helper: call both scalar and SSE2 paths and assert they match.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn assert_scalar_simd_equivalent(
    tgt_image: &GrayImage,
    template: &[f32],
    template_mean: f32,
    patch_gx: &[f32],
    patch_gy: &[f32],
    patch_x: u32,
    patch_y: u32,
    ps: usize,
    u: (f32, f32),
    normalize: bool,
) {
    let (b0_scalar, b1_scalar) = compute_iteration_scalar(
        tgt_image,
        template,
        template_mean,
        patch_gx,
        patch_gy,
        patch_x,
        patch_y,
        ps,
        u,
        normalize,
    );

    let ux_floor = u.0.floor() as i32;
    let uy_floor = u.1.floor() as i32;
    let x0_base = patch_x as i32 + ux_floor;
    let y0_base = patch_y as i32 + uy_floor;
    assert!(x0_base >= 0);
    assert!((x0_base + ps as i32) < tgt_image.width() as i32);
    assert!(y0_base >= 0);
    assert!((y0_base + ps as i32) < tgt_image.height() as i32);

    let (b0_simd, b1_simd) = unsafe {
        compute_iteration_sse2(
            tgt_image.data().as_ptr(),
            tgt_image.width() as usize,
            template,
            template_mean,
            patch_gx,
            patch_gy,
            ps,
            x0_base as usize,
            y0_base as usize,
            u.0 - u.0.floor(),
            u.1 - u.1.floor(),
            normalize,
        )
    };

    assert!(
        (b0_scalar - b0_simd).abs() < 1e-4,
        "b0 mismatch: scalar={b0_scalar}, simd={b0_simd}, diff={}",
        (b0_scalar - b0_simd).abs()
    );
    assert!(
        (b1_scalar - b1_simd).abs() < 1e-4,
        "b1 mismatch: scalar={b1_scalar}, simd={b1_simd}, diff={}",
        (b1_scalar - b1_simd).abs()
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_simd_vs_scalar_ps4_zero_flow() {
    let tgt = pseudo_random_image(32, 32, 42);
    let ref_img = pseudo_random_image(32, 32, 99);
    let ps = 4usize;
    let patch_x = 10u32;
    let patch_y = 10u32;

    // Build template and gradients from ref image
    let (grad_x, grad_y) = compute_gradients(&ref_img);
    let mut template = vec![0.0f32; ps * ps];
    let mut patch_gx = vec![0.0f32; ps * ps];
    let mut patch_gy = vec![0.0f32; ps * ps];
    let mut template_mean = 0.0f32;
    for py in 0..ps {
        for px in 0..ps {
            let idx = py * ps + px;
            let col = patch_x as usize + px;
            let row = patch_y as usize + py;
            template[idx] = ref_img.get_pixel(col as u32, row as u32);
            template_mean += template[idx];
            patch_gx[idx] = grad_x.get_pixel(col as u32, row as u32);
            patch_gy[idx] = grad_y.get_pixel(col as u32, row as u32);
        }
    }
    template_mean /= (ps * ps) as f32;

    assert_scalar_simd_equivalent(
        &tgt,
        &template,
        template_mean,
        &patch_gx,
        &patch_gy,
        patch_x,
        patch_y,
        ps,
        (0.0, 0.0),
        true,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_simd_vs_scalar_ps8_fractional_flow() {
    let tgt = pseudo_random_image(64, 64, 123);
    let ref_img = pseudo_random_image(64, 64, 456);
    let ps = 8usize;
    let patch_x = 16u32;
    let patch_y = 20u32;

    let (grad_x, grad_y) = compute_gradients(&ref_img);
    let mut template = vec![0.0f32; ps * ps];
    let mut patch_gx = vec![0.0f32; ps * ps];
    let mut patch_gy = vec![0.0f32; ps * ps];
    let mut template_mean = 0.0f32;
    for py in 0..ps {
        for px in 0..ps {
            let idx = py * ps + px;
            let col = patch_x as usize + px;
            let row = patch_y as usize + py;
            template[idx] = ref_img.get_pixel(col as u32, row as u32);
            template_mean += template[idx];
            patch_gx[idx] = grad_x.get_pixel(col as u32, row as u32);
            patch_gy[idx] = grad_y.get_pixel(col as u32, row as u32);
        }
    }
    template_mean /= (ps * ps) as f32;

    // Test several fractional flow values
    for &u in &[
        (0.0, 0.0),
        (1.3, -0.7),
        (-2.5, 3.2),
        (0.0, 4.9),
        (-3.0, -1.0),
    ] {
        for normalize in [true, false] {
            assert_scalar_simd_equivalent(
                &tgt,
                &template,
                template_mean,
                &patch_gx,
                &patch_gy,
                patch_x,
                patch_y,
                ps,
                u,
                normalize,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_simd_vs_scalar_ps8_negative_fractional_flow() {
    // Specifically test negative fractional parts, where floor(-0.3) = -1
    let tgt = pseudo_random_image(64, 64, 789);
    let ref_img = pseudo_random_image(64, 64, 321);
    let ps = 8usize;
    let patch_x = 20u32;
    let patch_y = 20u32;

    let (grad_x, grad_y) = compute_gradients(&ref_img);
    let mut template = vec![0.0f32; ps * ps];
    let mut patch_gx = vec![0.0f32; ps * ps];
    let mut patch_gy = vec![0.0f32; ps * ps];
    let mut template_mean = 0.0f32;
    for py in 0..ps {
        for px in 0..ps {
            let idx = py * ps + px;
            let col = patch_x as usize + px;
            let row = patch_y as usize + py;
            template[idx] = ref_img.get_pixel(col as u32, row as u32);
            template_mean += template[idx];
            patch_gx[idx] = grad_x.get_pixel(col as u32, row as u32);
            patch_gy[idx] = grad_y.get_pixel(col as u32, row as u32);
        }
    }
    template_mean /= (ps * ps) as f32;

    for &u in &[(-0.3, -0.7), (-0.9, -0.1), (-5.5, 2.3)] {
        assert_scalar_simd_equivalent(
            &tgt,
            &template,
            template_mean,
            &patch_gx,
            &patch_gy,
            patch_x,
            patch_y,
            ps,
            u,
            true,
        );
    }
}

#[test]
fn test_zero_flow_for_identical_images() {
    let img = GrayImage::checkerboard(32, 32);
    let mut flow = FlowField::new(32, 32);
    let params = DisFlowParams {
        variational_refinement: false,
        ..DisFlowParams::fast()
    };
    refine_flow_at_level(&img, &img, &mut flow, &params, 0, None);

    let mut max_flow = 0.0f32;
    for row in 0..32 {
        for col in 0..32 {
            let (dx, dy) = flow.get(col, row);
            max_flow = max_flow.max(dx.abs()).max(dy.abs());
        }
    }
    assert!(
        max_flow < 0.5,
        "Identical images should give near-zero flow, got max {}",
        max_flow
    );
}
