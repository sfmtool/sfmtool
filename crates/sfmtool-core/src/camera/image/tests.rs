// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
fn test_pyramid_from_image_matches_build() {
    // Non-uniform content, so that a level which came out of a different loop
    // would differ rather than agree by being flat everywhere.
    let (w, h) = (48u32, 32u32);
    let data: Vec<u8> = (0..(w * h * 3))
        .map(|i| ((i * 37 + (i / 7) * 11) % 251) as u8)
        .collect();
    let img = ImageU8::new(w, h, 3, data.clone());
    let built = ImageU8Pyramid::build(&img, 5);
    let owned = ImageU8Pyramid::from_image(ImageU8::new(w, h, 3, data), 5);
    assert_eq!(built.num_levels(), owned.num_levels());
    for i in 0..built.num_levels() {
        assert_eq!(built.level(i).width(), owned.level(i).width());
        assert_eq!(built.level(i).height(), owned.level(i).height());
        assert_eq!(built.level(i).channels(), owned.level(i).channels());
        assert_eq!(built.level(i).data(), owned.level(i).data(), "level {i}");
    }
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
// Reading image files
// -----------------------------------------------------------------------

#[test]
fn read_rgba_keeps_a_png_alpha_and_matches_read_rgb_colour() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("alpha.png");
    let pixels = vec![
        10, 20, 30, 0, //
        40, 50, 60, 128, //
        70, 80, 90, 255,
    ];
    ::image::RgbaImage::from_raw(3, 1, pixels.clone())
        .unwrap()
        .save(&path)
        .unwrap();

    let rgba = ImageU8::read_rgba(&path).unwrap();
    assert_eq!((rgba.width(), rgba.height(), rgba.channels()), (3, 1, 4));
    assert_eq!(rgba.data(), &pixels[..]);

    let rgb = ImageU8::read_rgb(&path).unwrap();
    assert_eq!(rgb.data(), &[10, 20, 30, 40, 50, 60, 70, 80, 90]);
}

#[test]
fn read_rgba_writes_opaque_alpha_for_an_image_without_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("opaque.png");
    ::image::RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6])
        .unwrap()
        .save(&path)
        .unwrap();

    let rgba = ImageU8::read_rgba(&path).unwrap();
    assert_eq!(rgba.data(), &[1, 2, 3, 255, 4, 5, 6, 255]);
}

#[test]
fn the_contents_not_the_extension_choose_the_decoder() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("real.png");
    ::image::RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6])
        .unwrap()
        .save(&png)
        .unwrap();
    let misnamed = dir.path().join("actually_png.jpg");
    std::fs::copy(&png, &misnamed).unwrap();

    assert_eq!(
        ImageU8::read_rgb(&misnamed).unwrap().data(),
        &[1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        ImageU8::read_rgba(&misnamed).unwrap().data(),
        &[1, 2, 3, 255, 4, 5, 6, 255]
    );
}

#[test]
fn image_has_alpha_reads_the_header() {
    let dir = tempfile::tempdir().unwrap();
    let rgba = dir.path().join("rgba.png");
    let rgb = dir.path().join("rgb.png");
    ::image::RgbaImage::from_raw(1, 1, vec![1, 2, 3, 4])
        .unwrap()
        .save(&rgba)
        .unwrap();
    ::image::RgbImage::from_raw(1, 1, vec![1, 2, 3])
        .unwrap()
        .save(&rgb)
        .unwrap();
    let misnamed = dir.path().join("rgba.jpg");
    std::fs::copy(&rgba, &misnamed).unwrap();

    assert!(image_has_alpha(&rgba).unwrap());
    assert!(image_has_alpha(&misnamed).unwrap());
    assert!(!image_has_alpha(&rgb).unwrap());
    assert!(image_has_alpha(&dir.path().join("missing.png")).is_err());
}
