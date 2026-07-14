use super::*;

/// Helper: create a simple identity warp map (each pixel maps to itself).
fn identity_warp_map(width: u32, height: u32) -> WarpMap {
    let mut data = vec![0.0f32; 2 * (width as usize) * (height as usize)];
    for row in 0..height {
        for col in 0..width {
            let idx = (row as usize * width as usize + col as usize) * 2;
            data[idx] = col as f32 + 0.5;
            data[idx + 1] = row as f32 + 0.5;
        }
    }
    WarpMap::new(width, height, data)
}

/// Helper: create a warp map that applies a translation (dx, dy).
fn translation_warp_map(width: u32, height: u32, dx: f32, dy: f32) -> WarpMap {
    let mut data = vec![0.0f32; 2 * (width as usize) * (height as usize)];
    for row in 0..height {
        for col in 0..width {
            let idx = (row as usize * width as usize + col as usize) * 2;
            data[idx] = col as f32 + 0.5 + dx;
            data[idx + 1] = row as f32 + 0.5 + dy;
        }
    }
    WarpMap::new(width, height, data)
}

// -----------------------------------------------------------------------
// ImageU8 basic tests
// -----------------------------------------------------------------------

#[test]
fn test_image_u8_construction_and_pixel_access() {
    let data = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let img = ImageU8::new(2, 2, 3, data);

    assert_eq!(img.width(), 2);
    assert_eq!(img.height(), 2);
    assert_eq!(img.channels(), 3);

    // Top-left pixel: (10, 20, 30)
    assert_eq!(img.get_pixel(0, 0, 0), 10);
    assert_eq!(img.get_pixel(0, 0, 1), 20);
    assert_eq!(img.get_pixel(0, 0, 2), 30);

    // Top-right pixel: (40, 50, 60)
    assert_eq!(img.get_pixel(1, 0, 0), 40);
    assert_eq!(img.get_pixel(1, 0, 1), 50);
    assert_eq!(img.get_pixel(1, 0, 2), 60);

    // Bottom-left pixel: (70, 80, 90)
    assert_eq!(img.get_pixel(0, 1, 0), 70);
    assert_eq!(img.get_pixel(0, 1, 1), 80);
    assert_eq!(img.get_pixel(0, 1, 2), 90);

    // Bottom-right pixel: (100, 110, 120)
    assert_eq!(img.get_pixel(1, 1, 0), 100);
    assert_eq!(img.get_pixel(1, 1, 1), 110);
    assert_eq!(img.get_pixel(1, 1, 2), 120);
}

#[test]
fn test_image_u8_single_channel() {
    let data = vec![0, 64, 128, 255];
    let img = ImageU8::new(2, 2, 1, data);
    assert_eq!(img.get_pixel(0, 0, 0), 0);
    assert_eq!(img.get_pixel(1, 0, 0), 64);
    assert_eq!(img.get_pixel(0, 1, 0), 128);
    assert_eq!(img.get_pixel(1, 1, 0), 255);
}

#[test]
fn test_image_u8_from_channels() {
    let img = ImageU8::from_channels(4, 3, 3);
    assert_eq!(img.width(), 4);
    assert_eq!(img.height(), 3);
    assert_eq!(img.channels(), 3);
    assert_eq!(img.data().len(), 4 * 3 * 3);
    assert!(img.data().iter().all(|&v| v == 0));
}

#[test]
#[should_panic(expected = "data length")]
fn test_image_u8_wrong_data_length_panics() {
    ImageU8::new(2, 2, 3, vec![0; 10]);
}

#[test]
fn test_image_u8_data_mut() {
    let mut img = ImageU8::from_channels(2, 2, 1);
    img.data_mut()[0] = 42;
    assert_eq!(img.get_pixel(0, 0, 0), 42);
}

// -----------------------------------------------------------------------
// Downsample tests
// -----------------------------------------------------------------------

#[test]
fn test_downsample_2x_single_channel() {
    // 4x4 image with known values.
    #[rustfmt::skip]
        let data = vec![
            10, 20, 30, 40,
            50, 60, 70, 80,
            90, 100, 110, 120,
            130, 140, 150, 160,
        ];
    let img = ImageU8::new(4, 4, 1, data);
    let down = img.downsample_2x();
    assert_eq!(down.width(), 2);
    assert_eq!(down.height(), 2);

    // Top-left block average: (10+20+50+60)/4 = 35
    assert_eq!(down.get_pixel(0, 0, 0), 35);
    // Top-right block: (30+40+70+80)/4 = 55
    assert_eq!(down.get_pixel(1, 0, 0), 55);
    // Bottom-left block: (90+100+130+140)/4 = 115
    assert_eq!(down.get_pixel(0, 1, 0), 115);
    // Bottom-right block: (110+120+150+160)/4 = 135
    assert_eq!(down.get_pixel(1, 1, 0), 135);
}

#[test]
fn test_downsample_2x_rgb() {
    // 2x2 RGB image.
    #[rustfmt::skip]
        let data = vec![
            10, 20, 30,   40, 50, 60,
            70, 80, 90,  100, 110, 120,
        ];
    let img = ImageU8::new(2, 2, 3, data);
    let down = img.downsample_2x();
    assert_eq!(down.width(), 1);
    assert_eq!(down.height(), 1);

    // R: (10+40+70+100)/4 = 55
    assert_eq!(down.get_pixel(0, 0, 0), 55);
    // G: (20+50+80+110)/4 = 65
    assert_eq!(down.get_pixel(0, 0, 1), 65);
    // B: (30+60+90+120)/4 = 75
    assert_eq!(down.get_pixel(0, 0, 2), 75);
}

// -----------------------------------------------------------------------
// Pyramid tests
// -----------------------------------------------------------------------

#[test]
fn test_pyramid_builds_correct_levels() {
    let img = ImageU8::from_channels(64, 32, 1);
    let pyr = ImageU8Pyramid::build(&img, 4);
    assert_eq!(pyr.num_levels(), 4);
    assert_eq!(pyr.level(0).width(), 64);
    assert_eq!(pyr.level(0).height(), 32);
    assert_eq!(pyr.level(1).width(), 32);
    assert_eq!(pyr.level(1).height(), 16);
    assert_eq!(pyr.level(2).width(), 16);
    assert_eq!(pyr.level(2).height(), 8);
    assert_eq!(pyr.level(3).width(), 8);
    assert_eq!(pyr.level(3).height(), 4);
}

#[test]
fn test_pyramid_halves_dimensions() {
    let img = ImageU8::from_channels(128, 128, 3);
    let pyr = ImageU8Pyramid::build(&img, 6);
    for i in 1..pyr.num_levels() {
        assert_eq!(pyr.level(i).width(), pyr.level(i - 1).width() / 2);
        assert_eq!(pyr.level(i).height(), pyr.level(i - 1).height() / 2);
        assert_eq!(pyr.level(i).channels(), 3);
    }
}

#[test]
fn test_pyramid_single_level() {
    let img = ImageU8::from_channels(8, 8, 1);
    let pyr = ImageU8Pyramid::build(&img, 1);
    assert_eq!(pyr.num_levels(), 1);
    assert_eq!(pyr.level(0).width(), 8);
}

#[test]
fn test_pyramid_stops_at_small_dimension() {
    // 4x4 → 2x2 → 1x1. Requesting 10 levels should stop after 3
    // because 1x1 has width < 2 so no further downsample.
    let img = ImageU8::from_channels(4, 4, 1);
    let pyr = ImageU8Pyramid::build(&img, 10);
    assert_eq!(pyr.num_levels(), 3);
    assert_eq!(pyr.level(0).width(), 4);
    assert_eq!(pyr.level(1).width(), 2);
    assert_eq!(pyr.level(2).width(), 1);
}

// -----------------------------------------------------------------------
// sample_bilinear_u8 tests
// -----------------------------------------------------------------------

#[test]
fn test_sample_bilinear_at_pixel_center() {
    let data = vec![100, 200, 50, 150];
    let img = ImageU8::new(2, 2, 1, data);

    // Pixel center of (0,0) is at (0.5, 0.5).
    let val = sample_bilinear_u8(&img, 0.5, 0.5, 0);
    assert!((val - 100.0).abs() < 1e-3, "got {}", val);

    // Pixel center of (1,0) is at (1.5, 0.5).
    let val = sample_bilinear_u8(&img, 1.5, 0.5, 0);
    assert!((val - 200.0).abs() < 1e-3, "got {}", val);

    // Pixel center of (1,1) is at (1.5, 1.5).
    let val = sample_bilinear_u8(&img, 1.5, 1.5, 0);
    assert!((val - 150.0).abs() < 1e-3, "got {}", val);
}

#[test]
fn test_sample_bilinear_interpolated() {
    let data = vec![0, 100, 0, 100];
    let img = ImageU8::new(2, 2, 1, data);

    // Midpoint between pixel (0,0) and (1,0): x=1.0, y=0.5
    let val = sample_bilinear_u8(&img, 1.0, 0.5, 0);
    assert!((val - 50.0).abs() < 1e-3, "got {}", val);
}

#[test]
fn test_sample_bilinear_four_pixel_average() {
    let data = vec![0, 100, 200, 60];
    let img = ImageU8::new(2, 2, 1, data);

    // Center of four pixels: (1.0, 1.0) -> average = (0+100+200+60)/4 = 90
    let val = sample_bilinear_u8(&img, 1.0, 1.0, 0);
    assert!((val - 90.0).abs() < 1e-3, "got {}", val);
}

#[test]
fn sample_bilinear_u8_all_matches_per_channel() {
    // The channel-batched gather must be bit-identical to per-channel
    // `sample_bilinear_u8` + round/clamp across every channel count and a range
    // of in-, edge-, and out-of-bounds coordinates (clamping paths included).
    for channels in [1u32, 3, 4] {
        let w = 5u32;
        let h = 4u32;
        let n = (w * h * channels) as usize;
        // Deterministic pseudo-random-ish byte pattern.
        let data: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let img = ImageU8::new(w, h, channels, data);

        for &y in &[-1.3f32, 0.0, 0.5, 1.7, 2.5, 3.9, 5.2] {
            for &x in &[-0.7f32, 0.0, 0.5, 1.25, 2.5, 4.5, 6.1] {
                let mut got = vec![0u8; channels as usize];
                sample_bilinear_u8_all(&img, x, y, &mut got);
                for ch in 0..channels {
                    let val = sample_bilinear_u8(&img, x, y, ch);
                    let want = (val + 0.5).clamp(0.0, 255.0) as u8;
                    assert_eq!(
                        got[ch as usize], want,
                        "channels={channels} x={x} y={y} ch={ch}"
                    );
                }
            }
        }
    }
}

#[test]
fn sample_bilinear_with_grad_u8_all_matches_per_channel() {
    // The channel-batched value+gradient gather must be bit-identical to
    // per-channel `sample_bilinear_with_grad_u8` for value AND both gradients,
    // across channel counts and in-/edge-/out-of-bounds coordinates.
    for channels in [1u32, 3, 4] {
        let w = 5u32;
        let h = 4u32;
        let n = (w * h * channels) as usize;
        let data: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let img = ImageU8::new(w, h, channels, data);

        for &y in &[-1.3f32, 0.0, 0.5, 1.7, 2.5, 3.9, 5.2] {
            for &x in &[-0.7f32, 0.0, 0.5, 1.25, 2.5, 4.5, 6.1] {
                let c = channels as usize;
                let mut val = vec![0f32; c];
                let mut gx = vec![0f32; c];
                let mut gy = vec![0f32; c];
                sample_bilinear_with_grad_u8_all(&img, x, y, &mut val, &mut gx, &mut gy);
                for ch in 0..channels {
                    let (wv, wgx, wgy) = sample_bilinear_with_grad_u8(&img, x, y, ch);
                    let k = ch as usize;
                    assert_eq!(
                        val[k].to_bits(),
                        wv.to_bits(),
                        "value channels={channels} x={x} y={y} ch={ch}"
                    );
                    assert_eq!(
                        gx[k].to_bits(),
                        wgx.to_bits(),
                        "grad_x channels={channels} x={x} y={y} ch={ch}"
                    );
                    assert_eq!(
                        gy[k].to_bits(),
                        wgy.to_bits(),
                        "grad_y channels={channels} x={x} y={y} ch={ch}"
                    );
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// remap_bilinear tests
// -----------------------------------------------------------------------

#[test]
fn test_remap_bilinear_identity_preserves_image() {
    // 4x4 single-channel image with gradient values.
    let data: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
    let src = ImageU8::new(4, 4, 1, data.clone());
    let map = identity_warp_map(4, 4);

    let result = remap_bilinear(&src, &map);

    assert_eq!(result.width(), 4);
    assert_eq!(result.height(), 4);
    for (i, (got, want)) in result.data().iter().zip(data.iter()).enumerate() {
        assert_eq!(got, want, "pixel {i} differs: {got} vs {want}");
    }
}

#[test]
fn test_remap_bilinear_identity_rgb() {
    // 3x3 RGB image.
    let mut data = Vec::new();
    for i in 0..9 {
        data.push((i * 10) as u8); // R
        data.push((i * 20) as u8); // G
        data.push((i * 30) as u8); // B
    }
    let src = ImageU8::new(3, 3, 3, data.clone());
    let map = identity_warp_map(3, 3);

    let result = remap_bilinear(&src, &map);

    assert_eq!(result.width(), 3);
    assert_eq!(result.height(), 3);
    assert_eq!(result.channels(), 3);
    for (i, (got, want)) in result.data().iter().zip(data.iter()).enumerate() {
        assert_eq!(got, want, "byte {i} differs: {got} vs {want}");
    }
}

#[test]
fn test_remap_bilinear_with_translation() {
    // 4x4 single-channel image: value = col * 50.
    #[rustfmt::skip]
        let data = vec![
            0, 50, 100, 150,
            0, 50, 100, 150,
            0, 50, 100, 150,
            0, 50, 100, 150,
        ];
    let src = ImageU8::new(4, 4, 1, data);

    // Shift by +1 pixel in x: each output pixel samples from (col+1.5)
    // in source, i.e. one pixel to the right.
    let map = translation_warp_map(4, 4, 1.0, 0.0);
    let result = remap_bilinear(&src, &map);

    // Output col 0 should get source col 1 value (50).
    assert_eq!(result.get_pixel(0, 0, 0), 50);
    // Output col 1 should get source col 2 value (100).
    assert_eq!(result.get_pixel(1, 0, 0), 100);
    // Output col 2 should get source col 3 value (150).
    assert_eq!(result.get_pixel(2, 0, 0), 150);
    // Output col 3 is clamped to source col 3 (150).
    assert_eq!(result.get_pixel(3, 0, 0), 150);
}

#[test]
fn test_remap_bilinear_nan_gives_black() {
    let src = ImageU8::new(2, 2, 1, vec![100, 100, 100, 100]);

    // Warp map with NaN entries.
    let data = vec![f32::NAN, f32::NAN, 1.5, 0.5, 0.5, 1.5, f32::NAN, f32::NAN];
    let map = WarpMap::new(2, 2, data);

    let result = remap_bilinear(&src, &map);

    // NaN pixels should be black (0).
    assert_eq!(result.get_pixel(0, 0, 0), 0);
    assert_eq!(result.get_pixel(1, 1, 0), 0);
    // Valid pixels should be 100.
    assert_eq!(result.get_pixel(1, 0, 0), 100);
    assert_eq!(result.get_pixel(0, 1, 0), 100);
}

#[test]
fn test_remap_bilinear_multichannel_rgb() {
    // 2x2 RGB image.
    #[rustfmt::skip]
        let data = vec![
            255, 0, 0,      0, 255, 0,
            0, 0, 255,    128, 128, 128,
        ];
    let src = ImageU8::new(2, 2, 3, data.clone());
    let map = identity_warp_map(2, 2);

    let result = remap_bilinear(&src, &map);

    for (i, (got, want)) in result.data().iter().zip(data.iter()).enumerate() {
        assert_eq!(got, want, "byte {i} differs");
    }
}

// -----------------------------------------------------------------------
// remap_aniso tests
// -----------------------------------------------------------------------

#[test]
fn test_remap_aniso_isotropic_matches_bilinear() {
    // With an identity warp map and SVD indicating no compression
    // (sigma_major=1, sigma_minor=1), remap_aniso should produce the
    // same result as remap_bilinear.
    let data: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
    let src = ImageU8::new(4, 4, 1, data);
    let mut map = identity_warp_map(4, 4);
    map.compute_svd();

    let bilinear_result = remap_bilinear(&src, &map);
    let aniso_result = remap_aniso(&src, &map, 16);

    assert_eq!(bilinear_result.width(), aniso_result.width());
    assert_eq!(bilinear_result.height(), aniso_result.height());
    for i in 0..bilinear_result.data().len() {
        let diff = (bilinear_result.data()[i] as i32 - aniso_result.data()[i] as i32).abs();
        assert!(
            diff <= 1,
            "pixel {} differs too much: {} vs {} (diff {})",
            i,
            bilinear_result.data()[i],
            aniso_result.data()[i],
            diff
        );
    }
}

#[test]
#[should_panic(expected = "requires WarpMap SVD")]
fn test_remap_aniso_panics_without_svd() {
    let src = ImageU8::from_channels(4, 4, 1);
    let map = identity_warp_map(4, 4);
    remap_aniso(&src, &map, 16);
}

// -----------------------------------------------------------------------
// Value+gradient sampler tests (Phase 3B analytic Jacobian)
// -----------------------------------------------------------------------

/// Build a 32x32 single-channel test image with a smoothly varying sinusoidal
/// intensity. Use a low spatial frequency so a small h central difference is a
/// good approximation to the analytic bilinear gradient inside one cell.
fn smooth_test_image() -> ImageU8 {
    let w: u32 = 32;
    let h: u32 = 32;
    let mut data = Vec::with_capacity((w * h) as usize);
    for row in 0..h {
        for col in 0..w {
            let x = col as f64 / w as f64;
            let y = row as f64 / h as f64;
            let v = 128.0
                + 60.0 * (2.0 * std::f64::consts::PI * x).sin()
                + 40.0 * (2.0 * std::f64::consts::PI * y).cos();
            data.push(v.clamp(0.0, 255.0).round() as u8);
        }
    }
    ImageU8::new(w, h, 1, data)
}

#[test]
fn sample_bilinear_with_grad_matches_value_only() {
    let img = smooth_test_image();
    // Pick a handful of interior sample points (avoid integer pixel centers
    // where the cell-boundary changes hit identically in both paths).
    let samples = [
        (10.3, 12.7),
        (5.0, 7.0),
        (15.5, 4.25),
        (20.9, 20.1),
        (1.5, 1.5),
    ];
    for &(x, y) in &samples {
        let v_only = sample_bilinear_u8(&img, x, y, 0);
        let (v_grad, _, _) = sample_bilinear_with_grad_u8(&img, x, y, 0);
        assert!(
            (v_only - v_grad).abs() < 1e-5,
            "value mismatch at ({x}, {y}): value-only={v_only}, value+grad={v_grad}"
        );
    }
}

#[test]
fn sample_bilinear_with_grad_matches_finite_difference() {
    let img = smooth_test_image();
    // Interior sample points well away from cell boundaries. Bilinear's
    // analytic gradient is piecewise-constant within one cell (pixel-center
    // grid; cells are split at integer + 0.5 coordinates). A central
    // difference must stay inside one cell to recover the analytic value
    // exactly — so pick coordinates well clear of `.5` fractional parts.
    let samples: &[(f32, f32)] = &[
        (10.3, 12.7),
        (5.4, 7.2),
        (15.2, 4.1),
        (20.9, 20.1),
        (8.6, 18.4),
    ];
    let h = 1e-3_f32;
    for &(x, y) in samples {
        let (_, gx_analytic, gy_analytic) = sample_bilinear_with_grad_u8(&img, x, y, 0);
        let vxp = sample_bilinear_u8(&img, x + h, y, 0);
        let vxm = sample_bilinear_u8(&img, x - h, y, 0);
        let vyp = sample_bilinear_u8(&img, x, y + h, 0);
        let vym = sample_bilinear_u8(&img, x, y - h, 0);
        let gx_fd = (vxp - vxm) / (2.0 * h);
        let gy_fd = (vyp - vym) / (2.0 * h);
        // Bilinear is piecewise-bilinear in (x, y); within a cell the gradient
        // is constant, so FD recovers the analytic value to within the f32
        // numerical noise of the `(v(x+h) - v(x-h)) / (2h)` evaluation on
        // u8-valued taps. 0.05 grayscale units / pixel is well below the
        // textured-image gradient magnitudes (~tens of units/pixel here).
        let tol = 0.05_f32;
        assert!(
            (gx_analytic - gx_fd).abs() < tol,
            "dI/dx mismatch at ({x}, {y}): analytic={gx_analytic}, fd={gx_fd}"
        );
        assert!(
            (gy_analytic - gy_fd).abs() < tol,
            "dI/dy mismatch at ({x}, {y}): analytic={gy_analytic}, fd={gy_fd}"
        );
    }
}

#[test]
fn remap_bilinear_with_grad_value_matches_remap_bilinear() {
    // remap_bilinear quantizes to u8 with round; remap_bilinear_with_grad
    // returns the raw f32. The two should agree to within rounding.
    let img = smooth_test_image();
    let map = translation_warp_map(20, 20, 0.37, 0.21);
    let v_only = remap_bilinear(&img, &map);
    let vg = remap_bilinear_with_grad(&img, &map);
    for row in 0..20u32 {
        for col in 0..20u32 {
            let want = v_only.get_pixel(col, row, 0) as f32;
            let (got, _, _) = vg.get_pixel_with_grad(col, row, 0);
            assert!(
                (want - got.round()).abs() < 1.0,
                "value mismatch at ({col}, {row}): u8={want}, f32={got}"
            );
        }
    }
}

#[test]
fn remap_aniso_with_grad_value_matches_remap_aniso_with_pyramid() {
    // Same identity warp + isotropic SVD as in
    // test_remap_aniso_isotropic_matches_bilinear: the value path should
    // agree, and the gradient path should be non-zero on the smooth image.
    let img = smooth_test_image();
    let pyr = ImageU8Pyramid::build(&img, 4);
    let mut map = identity_warp_map(20, 20);
    map.compute_svd();
    let v_only = remap_aniso_with_pyramid(&pyr, &map, 16);
    let vg = remap_aniso_with_grad(&pyr, &map, 16);
    for row in 0..20u32 {
        for col in 0..20u32 {
            let want = v_only.get_pixel(col, row, 0) as f32;
            let (got, _, _) = vg.get_pixel_with_grad(col, row, 0);
            assert!(
                (want - got.round()).abs() < 1.0,
                "value mismatch at ({col}, {row}): u8={want}, f32={got}"
            );
        }
    }
}

#[test]
fn remap_aniso_with_grad_matches_finite_difference_within_lod_cell() {
    // Identity warp + isotropic SVD: sigma_major == sigma_minor == 1, which
    // triggers the non-compressive single-bilinear-sample path. So the aniso
    // gradient should match the bilinear gradient pixel-for-pixel.
    let img = smooth_test_image();
    let pyr = ImageU8Pyramid::build(&img, 4);
    let mut map = identity_warp_map(20, 20);
    map.compute_svd();

    let vg_aniso = remap_aniso_with_grad(&pyr, &map, 16);
    let vg_bi = remap_bilinear_with_grad(&img, &map);
    let (ax, ay) = (vg_aniso.grad_x(), vg_aniso.grad_y());
    let (bx, by) = (vg_bi.grad_x(), vg_bi.grad_y());
    for i in 0..(20 * 20) {
        let dx = (ax[i] - bx[i]).abs();
        let dy = (ay[i] - by[i]).abs();
        assert!(
            dx < 1e-4,
            "aniso vs bilinear dI/dx mismatch at idx {i}: aniso={}, bi={}",
            ax[i],
            bx[i]
        );
        assert!(
            dy < 1e-4,
            "aniso vs bilinear dI/dy mismatch at idx {i}: aniso={}, bi={}",
            ay[i],
            by[i]
        );
    }
}

#[test]
#[should_panic(expected = "requires WarpMap SVD")]
fn remap_aniso_with_grad_panics_without_svd() {
    let img = ImageU8::from_channels(4, 4, 1);
    let pyr = ImageU8Pyramid::build(&img, 2);
    let map = identity_warp_map(4, 4);
    remap_aniso_with_grad(&pyr, &map, 16);
}

/// Build a uniformly-compressive warp: each dest pixel `(col, row)` maps to
/// source `(scale_x * (col + 0.5), scale_y * (row + 0.5))`. The local Jacobian
/// is diagonal `[[scale_x, 0], [0, scale_y]]`, so the SVD gives
/// `sigma_major = max(scale_x, scale_y)`, `sigma_minor = min(...)`. With both
/// scales > 1 (compressive on both axes) the aniso path takes its full
/// `n`-footprint walk and active mip blend — the case the within-LOD-cell test
/// can't reach.
fn scaled_warp_map(width: u32, height: u32, scale_x: f32, scale_y: f32) -> WarpMap {
    let mut data = vec![0.0f32; 2 * (width as usize) * (height as usize)];
    for row in 0..height {
        for col in 0..width {
            let idx = (row as usize * width as usize + col as usize) * 2;
            data[idx] = scale_x * (col as f32 + 0.5);
            data[idx + 1] = scale_y * (row as f32 + 0.5);
        }
    }
    WarpMap::new(width, height, data)
}

/// Build a 256×256 single-channel source image with a smoothly varying intensity
/// (low spatial frequency so an `h ≈ 0.5` FD is a good reference for the analytic
/// gradient at sub-pixel offsets), and the matching 5-level pyramid.
fn smooth_compressive_test_pyramid() -> ImageU8Pyramid {
    let w_src: u32 = 256;
    let mut data = Vec::with_capacity((w_src * w_src) as usize);
    for row in 0..w_src {
        for col in 0..w_src {
            let x = col as f64 / w_src as f64;
            let y = row as f64 / w_src as f64;
            let v = 128.0
                + 50.0 * (2.0 * std::f64::consts::PI * x).sin()
                + 40.0 * (2.0 * std::f64::consts::PI * y).cos();
            data.push(v.clamp(0.0, 255.0).round() as u8);
        }
    }
    ImageU8Pyramid::build(&ImageU8::new(w_src, w_src, 1, data), 5)
}

/// Drive [`remap_aniso_with_grad`] over a per-axis-scaled compressive warp and
/// compare against an FD of the value-only `remap_aniso_with_pyramid` at
/// sub-pixel source-coord offsets `±h`. Diagonal `J = diag(scale_x, scale_y)`,
/// so a dest-translation `(dx, dy)` is exactly a source-translation
/// `(scale_x·dx, scale_y·dy)` — pin `dx = h / scale_x` and `dy = h / scale_y`
/// per axis to get a uniform source-pixel step. Asserts `(min_smaj, min_smin)`
/// hold at the centre pixel so each caller picks a `(scale_x, scale_y)` that
/// genuinely engages the path being tested.
fn assert_compressive_aniso_gradient_matches_fd(
    scale_x: f32,
    scale_y: f32,
    min_smaj: f32,
    min_smin: f32,
    min_n: u32,
    why: &str,
) {
    let pyr = smooth_compressive_test_pyramid();

    // Dest grid that maps fully into the source: `dst_size * scale ≤ src_size`.
    let w_dst: u32 = 32;
    let mut map = scaled_warp_map(w_dst, w_dst, scale_x, scale_y);
    map.compute_svd();
    // Sanity: the warp is genuinely compressive in the way the caller intended.
    let (smaj, smin, _, _) = map.get_svd(w_dst / 2, w_dst / 2);
    assert!(
        smaj >= min_smaj && smin >= min_smin,
        "fixture not compressive enough ({why}): \
         sigma_major={smaj}, sigma_minor={smin}, wanted ≥({min_smaj}, {min_smin})"
    );
    let ratio = smaj / smin.max(1.0);
    let n = (ratio.ceil() as u32).clamp(1, 16);
    assert!(
        n >= min_n,
        "fixture does not drive enough footprint samples ({why}): \
         n={n}, wanted ≥ {min_n} (sigma_major/sigma_minor={ratio})"
    );

    let vg_analytic = remap_aniso_with_grad(&pyr, &map, 16);

    let h: f32 = 0.5; // source-pixel step; small enough vs. the smooth texture
    let h_dest_x = h / scale_x; // dest step that = h in source x
    let h_dest_y = h / scale_y;

    fn shifted_scaled_warp_map(w: u32, h: u32, sx: f32, sy: f32, dx: f32, dy: f32) -> WarpMap {
        let mut data = vec![0.0f32; 2 * (w as usize) * (h as usize)];
        for row in 0..h {
            for col in 0..w {
                let idx = (row as usize * w as usize + col as usize) * 2;
                data[idx] = sx * (col as f32 + 0.5 + dx);
                data[idx + 1] = sy * (row as f32 + 0.5 + dy);
            }
        }
        WarpMap::new(w, h, data)
    }
    let mut map_xp = shifted_scaled_warp_map(w_dst, w_dst, scale_x, scale_y, h_dest_x, 0.0);
    let mut map_xm = shifted_scaled_warp_map(w_dst, w_dst, scale_x, scale_y, -h_dest_x, 0.0);
    let mut map_yp = shifted_scaled_warp_map(w_dst, w_dst, scale_x, scale_y, 0.0, h_dest_y);
    let mut map_ym = shifted_scaled_warp_map(w_dst, w_dst, scale_x, scale_y, 0.0, -h_dest_y);
    for m in [&mut map_xp, &mut map_xm, &mut map_yp, &mut map_ym] {
        m.compute_svd();
    }
    let v_xp = remap_aniso_with_pyramid(&pyr, &map_xp, 16);
    let v_xm = remap_aniso_with_pyramid(&pyr, &map_xm, 16);
    let v_yp = remap_aniso_with_pyramid(&pyr, &map_yp, 16);
    let v_ym = remap_aniso_with_pyramid(&pyr, &map_ym, 16);

    // Compare at the interior — skip the 2-pixel boundary band where the warp's
    // own Jacobian uses one-sided differences (and where FD on the value path
    // can pick up boundary artifacts of the source image).
    let mut max_dx_err = 0.0f32;
    let mut max_dy_err = 0.0f32;
    let mut max_grad_mag = 0.0f32;
    for row in 4..w_dst - 4 {
        for col in 4..w_dst - 4 {
            let (_, gx, gy) = vg_analytic.get_pixel_with_grad(col, row, 0);
            let vxp = v_xp.get_pixel(col, row, 0) as f32;
            let vxm = v_xm.get_pixel(col, row, 0) as f32;
            let vyp = v_yp.get_pixel(col, row, 0) as f32;
            let vym = v_ym.get_pixel(col, row, 0) as f32;
            let gx_fd = (vxp - vxm) / (2.0 * h);
            let gy_fd = (vyp - vym) / (2.0 * h);
            max_dx_err = max_dx_err.max((gx - gx_fd).abs());
            max_dy_err = max_dy_err.max((gy - gy_fd).abs());
            max_grad_mag = max_grad_mag.max(gx.abs().max(gy.abs()));
        }
    }
    // Tolerance: the value-only path quantizes to u8 (`round`), so its FD has
    // ±1-LSB-per-sample noise on a difference, plus footprint-walk + mip-blend
    // discretization. Empirically the analytic gradient agrees to within a few
    // grayscale units / source-px on this texture; absolute tolerance is the
    // right thing to set since grads pass through zero. Sanity-check that the
    // gradient is non-trivial — a slack test on flat content would prove
    // nothing.
    assert!(
        max_grad_mag > 1.0,
        "fixture's gradient is too small to validate ({why}): max |g| = {max_grad_mag}"
    );
    assert!(
        max_dx_err < 5.0,
        "compressive aniso dI/dx vs FD ({why}): max err = {max_dx_err} (max |g| = {max_grad_mag})"
    );
    assert!(
        max_dy_err < 5.0,
        "compressive aniso dI/dy vs FD ({why}): max err = {max_dy_err} (max |g| = {max_grad_mag})"
    );
}

#[test]
fn remap_aniso_with_grad_compressive_isotropic_matches_finite_difference() {
    // Isotropic compression: `scale_x == scale_y == 3` ⇒ `sigma_major =
    // sigma_minor = 3` ⇒ `ratio = 1` ⇒ `n = 1`. Drives the mip blend
    // (`level_f = log2(3) ≈ 1.585`, `level_lo = 1`, `level_hi = 2`,
    // `frac ≈ 0.585`) — exercises that branch and the per-level `1/scale`
    // gradient scaling. **Does NOT** exercise the multi-sample footprint walk,
    // which needs `sigma_major > sigma_minor`; the anisotropic test below does.
    assert_compressive_aniso_gradient_matches_fd(3.0, 3.0, 2.5, 2.5, 1, "isotropic 3×3");
}

// -----------------------------------------------------------------------
// remap_bilinear_mip tests
// -----------------------------------------------------------------------

/// Deterministic pseudo-random single-channel texture with per-pixel detail, so
/// pyramid levels genuinely differ (a smooth texture box-averages to nearly the
/// same values, which would let a level-selection bug slip through the u8
/// rounding).
fn noisy_test_image(w: u32, h: u32) -> ImageU8 {
    let data: Vec<u8> = (0..(w as usize * h as usize))
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    ImageU8::new(w, h, 1, data)
}

#[test]
fn remap_bilinear_mip_matches_bilinear_when_uncompressed() {
    // Identity warp (sigma_major = 1) and a mild 1.3x compression
    // (round(log2(1.3)) = 0): every pixel selects level 0, so the output is
    // bit-exact with remap_bilinear from the full-resolution image.
    let img = noisy_test_image(32, 32);
    let pyr = ImageU8Pyramid::build(&img, 4);

    let mut ident = identity_warp_map(20, 20);
    ident.compute_svd();
    assert_eq!(
        remap_bilinear_mip(&pyr, &ident).data(),
        remap_bilinear(&img, &ident).data(),
        "identity warp must be bit-exact with remap_bilinear"
    );

    let mut mild = scaled_warp_map(20, 20, 1.3, 1.3);
    mild.compute_svd();
    assert_eq!(
        remap_bilinear_mip(&pyr, &mild).data(),
        remap_bilinear(&img, &mild).data(),
        "sub-sqrt(2) compression must still select level 0 (bit-exact)"
    );
}

#[test]
fn remap_bilinear_mip_selects_level_from_compression() {
    // A uniformly 2x- (4x-) compressing affine map has sigma_major = 2 (4), so
    // every pixel selects level 1 (2): the output must equal directly sampling
    // that pyramid level at (sx / 2^l, sy / 2^l) with remap_bilinear's rounding.
    let img = noisy_test_image(64, 64);
    let pyr = ImageU8Pyramid::build(&img, 4);

    for (scale, level) in [(2.0f32, 1usize), (4.0, 2)] {
        let w_dst = (64.0 / scale) as u32 - 1;
        // The sub-pixel source offset keeps the sample points off the exact
        // 2x2 box-average positions, where a level-0 bilinear tap coincides
        // with the level-1 box downsample and the two levels are
        // indistinguishable (a pure `scaled_warp_map` at scale 2 lands
        // exactly there).
        let mut map = offset_scaled_warp_map(w_dst, w_dst, scale, 0.7, 0.7);
        map.compute_svd();
        let out = remap_bilinear_mip(&pyr, &map);
        let lvl_scale = (1u32 << level) as f32;
        let mut differs_from_level0 = false;
        for row in 0..w_dst {
            for col in 0..w_dst {
                let (sx, sy) = map.get(col, row);
                let val = sample_bilinear_u8(pyr.level(level), sx / lvl_scale, sy / lvl_scale, 0);
                let want = (val + 0.5).clamp(0.0, 255.0) as u8;
                assert_eq!(
                    out.get_pixel(col, row, 0),
                    want,
                    "scale {scale}: ({col}, {row}) should sample level {level}"
                );
                let val0 = sample_bilinear_u8(pyr.level(0), sx, sy, 0);
                if (val0 + 0.5).clamp(0.0, 255.0) as u8 != want {
                    differs_from_level0 = true;
                }
            }
        }
        // Sanity: on this texture the selected level is distinguishable from a
        // plain level-0 render, so the assertions above genuinely pin the level.
        assert!(
            differs_from_level0,
            "scale {scale}: fixture cannot distinguish level {level} from level 0"
        );
    }
}

#[test]
fn remap_bilinear_mip_clamps_to_last_level() {
    // 16x compression wants level 4, but the pyramid only has levels 0..=2:
    // the selection must clamp to the last built level.
    let img = noisy_test_image(64, 64);
    let pyr = ImageU8Pyramid::build(&img, 3);
    let mut map = scaled_warp_map(4, 4, 16.0, 16.0);
    map.compute_svd();
    let out = remap_bilinear_mip(&pyr, &map);
    let last = pyr.num_levels() - 1;
    let lvl_scale = (1u32 << last) as f32;
    for row in 0..4u32 {
        for col in 0..4u32 {
            let (sx, sy) = map.get(col, row);
            let val = sample_bilinear_u8(pyr.level(last), sx / lvl_scale, sy / lvl_scale, 0);
            let want = (val + 0.5).clamp(0.0, 255.0) as u8;
            assert_eq!(out.get_pixel(col, row, 0), want, "({col}, {row})");
        }
    }
}

#[test]
fn remap_bilinear_mip_nan_gives_black() {
    let img = noisy_test_image(8, 8);
    let pyr = ImageU8Pyramid::build(&img, 3);
    // Warp map with NaN entries (same layout as the remap_bilinear NaN test).
    let data = vec![f32::NAN, f32::NAN, 1.5, 0.5, 0.5, 1.5, f32::NAN, f32::NAN];
    let mut map = WarpMap::new(2, 2, data);
    map.compute_svd();
    let result = remap_bilinear_mip(&pyr, &map);
    assert_eq!(result.get_pixel(0, 0, 0), 0);
    assert_eq!(result.get_pixel(1, 1, 0), 0);
    // Valid pixels sample normally (identity-scale here, so level 0).
    assert_eq!(
        result.get_pixel(1, 0, 0),
        (sample_bilinear_u8(pyr.level(0), 1.5, 0.5, 0) + 0.5).clamp(0.0, 255.0) as u8
    );
    assert_eq!(
        result.get_pixel(0, 1, 0),
        (sample_bilinear_u8(pyr.level(0), 0.5, 1.5, 0) + 0.5).clamp(0.0, 255.0) as u8
    );
}

#[test]
#[should_panic(expected = "requires WarpMap SVD")]
fn remap_bilinear_mip_panics_without_svd() {
    let img = ImageU8::from_channels(4, 4, 1);
    let pyr = ImageU8Pyramid::build(&img, 2);
    let map = identity_warp_map(4, 4);
    remap_bilinear_mip(&pyr, &map);
}

#[test]
#[should_panic(expected = "requires WarpMap SVD")]
fn remap_bilinear_mip_with_grad_panics_without_svd() {
    let img = ImageU8::from_channels(4, 4, 1);
    let pyr = ImageU8Pyramid::build(&img, 2);
    let map = identity_warp_map(4, 4);
    remap_bilinear_mip_with_grad(&pyr, &map);
}

/// A uniformly `scale`-compressing warp with a constant sub-pixel source offset
/// `(ox, oy)` (full-res source px). The offset keeps the level-l sample points
/// off the bilinear cell boundaries (a pure `scaled_warp_map` at scale 2 lands
/// exactly on level-1 pixel centers, where the piecewise gradient switches
/// cells), and shifting it by ±h realizes an exact source-coordinate
/// translation for finite differencing. The linear part is unchanged, so the
/// SVD — and hence the selected mip level — is identical across shifts.
fn offset_scaled_warp_map(width: u32, height: u32, scale: f32, ox: f32, oy: f32) -> WarpMap {
    let mut data = vec![0.0f32; 2 * (width as usize) * (height as usize)];
    for row in 0..height {
        for col in 0..width {
            let idx = (row as usize * width as usize + col as usize) * 2;
            data[idx] = scale * (col as f32 + 0.5) + ox;
            data[idx + 1] = scale * (row as f32 + 0.5) + oy;
        }
    }
    WarpMap::new(width, height, data)
}

#[test]
fn remap_bilinear_mip_with_grad_value_matches_value_only() {
    // The grad variant returns the raw f32 of the same per-level bilinear tap
    // the value-only path rounds to u8 — they must agree within rounding, on
    // both a level-0 (identity) and a level-1 (2x-compressive) map.
    let pyr = smooth_compressive_test_pyramid();
    for scale in [1.0f32, 2.0] {
        let w_dst = 20u32;
        let mut map = offset_scaled_warp_map(w_dst, w_dst, scale, 0.3, 0.37);
        map.compute_svd();
        let v_only = remap_bilinear_mip(&pyr, &map);
        let vg = remap_bilinear_mip_with_grad(&pyr, &map);
        for row in 0..w_dst {
            for col in 0..w_dst {
                let want = v_only.get_pixel(col, row, 0) as f32;
                let (got, _, _) = vg.get_pixel_with_grad(col, row, 0);
                assert!(
                    (want - got.round()).abs() < 1.0,
                    "scale {scale}: value mismatch at ({col}, {row}): u8={want}, f32={got}"
                );
            }
        }
    }
}

#[test]
fn remap_bilinear_mip_with_grad_matches_finite_difference_in_full_res_coords() {
    // Gradients must come back in FULL-RES source-pixel coords: a bilinear
    // gradient at level l is per level-pixel, so the sampler rescales it by
    // 1/2^l. Verify against central finite differences of the (unquantized)
    // value path under an exact ±h source-coordinate translation — at level 1
    // an unscaled gradient would be 2x too large, which this catches.
    let pyr = smooth_compressive_test_pyramid();
    let w_dst = 20u32;
    let h: f32 = 0.05; // full-res source px; well inside one level-1 cell
    for scale in [2.0f32, 4.0] {
        // Offset 0.3 full-res px keeps the level sample points off the
        // pixel-center cell boundaries, so the ±h probes stay inside one
        // bilinear cell where the analytic gradient is exact.
        let mk = |ox: f32, oy: f32| {
            let mut m = offset_scaled_warp_map(w_dst, w_dst, scale, ox, oy);
            m.compute_svd();
            m
        };
        let map = mk(0.3, 0.37);
        // Sanity: sigma_major >= scale means a level > 0 is engaged, so the
        // 1/2^l gradient rescale is actually exercised.
        let (smaj, _, _, _) = map.get_svd(w_dst / 2, w_dst / 2);
        assert!(
            smaj >= scale - 0.1,
            "fixture not compressive enough: sigma_major={smaj}"
        );

        let vg = remap_bilinear_mip_with_grad(&pyr, &map);
        let v_xp = remap_bilinear_mip_with_grad(&pyr, &mk(0.3 + h, 0.37));
        let v_xm = remap_bilinear_mip_with_grad(&pyr, &mk(0.3 - h, 0.37));
        let v_yp = remap_bilinear_mip_with_grad(&pyr, &mk(0.3, 0.37 + h));
        let v_ym = remap_bilinear_mip_with_grad(&pyr, &mk(0.3, 0.37 - h));

        let mut max_grad_mag = 0.0f32;
        for row in 2..w_dst - 2 {
            for col in 2..w_dst - 2 {
                let (_, gx, gy) = vg.get_pixel_with_grad(col, row, 0);
                let gx_fd = (v_xp.get_pixel_with_grad(col, row, 0).0
                    - v_xm.get_pixel_with_grad(col, row, 0).0)
                    / (2.0 * h);
                let gy_fd = (v_yp.get_pixel_with_grad(col, row, 0).0
                    - v_ym.get_pixel_with_grad(col, row, 0).0)
                    / (2.0 * h);
                max_grad_mag = max_grad_mag.max(gx.abs().max(gy.abs()));
                assert!(
                    (gx - gx_fd).abs() < 0.05,
                    "scale {scale}: dI/dx at ({col}, {row}): analytic={gx}, fd={gx_fd}"
                );
                assert!(
                    (gy - gy_fd).abs() < 0.05,
                    "scale {scale}: dI/dy at ({col}, {row}): analytic={gy}, fd={gy_fd}"
                );
            }
        }
        // Sanity: the texture has real gradient signal at this level.
        assert!(
            max_grad_mag > 0.1,
            "scale {scale}: fixture gradient too small to validate ({max_grad_mag})"
        );
    }
}

#[test]
fn remap_aniso_with_grad_compressive_anisotropic_matches_finite_difference() {
    // Anisotropic compression: `scale_x = 4`, `scale_y = 1.2` ⇒ `sigma_major =
    // 4`, `sigma_minor = 1.2` ⇒ `ratio = 3.33` ⇒ `n = 4`. The for-loop in
    // `sample_aniso_with_grad` runs four times, walking four samples along the
    // major axis — this is the case the reviewer flagged as missing from the
    // isotropic-only test, where the `t * sigma_major * major_d{x,y}` step,
    // per-sample summation, and `sum / n` normalization are all exercised.
    // Mip blend is active here too (`level_f = log2(1.2) ≈ 0.263`,
    // `level_lo = 0`, `level_hi = 1`, `frac ≈ 0.263`).
    assert_compressive_aniso_gradient_matches_fd(4.0, 1.2, 3.5, 1.0, 4, "anisotropic 4×1.2");
}
