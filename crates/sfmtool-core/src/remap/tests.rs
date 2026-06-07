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
