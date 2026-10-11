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

// -----------------------------------------------------------------------
// Writing image files
// -----------------------------------------------------------------------

/// A `width x height` image whose bytes vary in every channel, so a swapped or
/// dropped channel changes the data.
fn gradient(width: u32, height: u32, channels: u32) -> ImageU8 {
    let data = (0..width * height * channels)
        .map(|i| ((i * 37 + i / channels * 11) % 256) as u8)
        .collect();
    ImageU8::new(width, height, channels, data)
}

#[test]
fn write_png_round_trips_grey_rgb_and_rgba_exactly() {
    let dir = tempfile::tempdir().unwrap();
    for channels in [1, 3, 4] {
        let image = gradient(17, 9, channels);
        let path = dir.path().join(format!("c{channels}.png"));
        image.write(&path, DEFAULT_JPEG_QUALITY).unwrap();
        let back = match channels {
            4 => ImageU8::read_rgba(&path).unwrap(),
            _ => ImageU8::read_rgb(&path).unwrap(),
        };
        assert_eq!((back.width(), back.height()), (17, 9));
        if channels == 1 {
            let grey: Vec<u8> = back.data().chunks(3).map(|p| p[0]).collect();
            assert_eq!(grey, image.data());
        } else {
            assert_eq!(back.data(), image.data(), "{channels} channels");
        }
    }
}

#[test]
fn write_jpeg_is_close_and_its_quality_sets_its_size() {
    let dir = tempfile::tempdir().unwrap();
    // A smooth image, as a photograph is, so the JPEG error stays small.
    let (w, h) = (64u32, 48u32);
    let data = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [(x * 4) as u8, (y * 5) as u8, 128]))
        .collect();
    let image = ImageU8::new(w, h, 3, data);

    let q95 = dir.path().join("q95.jpg");
    let q30 = dir.path().join("q30.JPEG");
    image.write(&q95, DEFAULT_JPEG_QUALITY).unwrap();
    image.write(&q30, 30).unwrap();

    let back = ImageU8::read_rgb(&q95).unwrap();
    let max_error = back
        .data()
        .iter()
        .zip(image.data())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(max_error <= 6, "max error {max_error} at quality 95");
    let size = |p: &std::path::Path| std::fs::metadata(p).unwrap().len();
    assert!(size(&q30) < size(&q95));
}

#[test]
fn write_refuses_what_it_cannot_encode_and_leaves_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let rgba = gradient(4, 4, 4);
    let rgb = gradient(4, 4, 3);

    let jpeg = dir.path().join("alpha.jpg");
    assert!(matches!(
        rgba.write(&jpeg, DEFAULT_JPEG_QUALITY),
        Err(::image::ImageError::Unsupported(_))
    ));
    assert!(!jpeg.exists());

    let unknown = dir.path().join("image.xyz");
    assert!(matches!(
        rgb.write(&unknown, DEFAULT_JPEG_QUALITY),
        Err(::image::ImageError::Unsupported(_))
    ));
    assert!(!unknown.exists());

    let bad_quality = dir.path().join("q0.jpg");
    assert!(matches!(
        rgb.write(&bad_quality, 0),
        Err(::image::ImageError::Parameter(_))
    ));
    assert!(!bad_quality.exists());

    let missing_dir = dir.path().join("missing").join("image.png");
    assert!(matches!(
        rgb.write(&missing_dir, DEFAULT_JPEG_QUALITY),
        Err(::image::ImageError::IoError(_))
    ));

    // A quality outside 1 to 100 is refused for every format.
    let png = dir.path().join("q0.png");
    assert!(matches!(
        rgb.write(&png, 0),
        Err(::image::ImageError::Parameter(_))
    ));

    let empty = ImageU8::new(0, 4, 3, Vec::new());
    let two_channels = ImageU8::new(2, 2, 2, vec![0; 8]);
    let too_wide = ImageU8::new(65536, 1, 1, vec![0; 65536]);
    for (image, name) in [
        (&empty, "empty.png"),
        (&two_channels, "two.png"),
        (&too_wide, "wide.jpg"),
    ] {
        let path = dir.path().join(name);
        assert!(
            matches!(
                image.write(&path, DEFAULT_JPEG_QUALITY),
                Err(::image::ImageError::Parameter(_))
            ),
            "{name}"
        );
        assert!(!path.exists());
    }
}

#[test]
fn a_failed_write_leaves_no_temporary_file_and_a_good_one_replaces_the_old() {
    let dir = tempfile::tempdir().unwrap();
    // A directory where the file should go: the rename into place fails.
    std::fs::create_dir(dir.path().join("taken.png")).unwrap();
    assert!(matches!(
        gradient(4, 4, 3).write(&dir.path().join("taken.png"), DEFAULT_JPEG_QUALITY),
        Err(::image::ImageError::IoError(_))
    ));
    let names = || -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };
    assert_eq!(names(), ["taken.png"]);

    let path = dir.path().join("image.png");
    gradient(4, 4, 3)
        .write(&path, DEFAULT_JPEG_QUALITY)
        .unwrap();
    let second = gradient(6, 2, 3);
    second.write(&path, DEFAULT_JPEG_QUALITY).unwrap();
    assert_eq!(ImageU8::read_rgb(&path).unwrap().data(), second.data());
    assert_eq!(names(), ["image.png", "taken.png"]);
}

/// The `(horizontal, vertical)` sampling factors of each component in the
/// baseline frame header (SOF0) of `jpeg`.
fn jpeg_sampling_factors(jpeg: &[u8]) -> Vec<(u8, u8)> {
    let sof = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("a baseline JPEG has an SOF0 marker");
    let components = jpeg[sof + 9] as usize;
    (0..components)
        .map(|i| {
            let factors = jpeg[sof + 10 + 3 * i + 1];
            (factors >> 4, factors & 0x0F)
        })
        .collect()
}

#[test]
fn write_jpeg_subsamples_chroma_420_and_decodes_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    // Odd sides, so the last chroma block covers a partial 2 x 2 block.
    let (w, h) = (37u32, 23u32);
    let data = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [200, (x * 6) as u8, (y * 10) as u8]))
        .collect();
    let image = ImageU8::new(w, h, 3, data);
    let path = dir.path().join("rgb.jpg");
    image.write(&path, DEFAULT_JPEG_QUALITY).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(jpeg_sampling_factors(&bytes), [(2, 2), (1, 1), (1, 1)]);
    let back = ImageU8::read_rgb(&path).unwrap();
    assert_eq!((back.width(), back.height()), (w, h));
    let mean_error = back
        .data()
        .iter()
        .zip(image.data())
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / image.data().len() as f64;
    assert!(mean_error < 2.0, "mean error {mean_error}");

    // A grey image is written as a one-component JPEG and reads back close.
    let grey = gradient(19, 11, 1);
    let grey_path = dir.path().join("grey.jpg");
    grey.write(&grey_path, DEFAULT_JPEG_QUALITY).unwrap();
    assert_eq!(
        jpeg_sampling_factors(&std::fs::read(&grey_path).unwrap()).len(),
        1
    );
    let grey_back = ImageU8::read_rgb(&grey_path).unwrap();
    assert_eq!((grey_back.width(), grey_back.height()), (19, 11));
}

/// A file another process holds open, without letting it be deleted, as
/// Python's `open` does on Windows, cannot be replaced by a rename; the write
/// falls back to writing it in place.
#[cfg(windows)]
#[test]
fn write_replaces_a_file_held_open_without_delete_sharing() {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_READ: u32 = 0x1;
    const FILE_SHARE_WRITE: u32 = 0x2;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("held.png");
    gradient(4, 4, 3)
        .write(&path, DEFAULT_JPEG_QUALITY)
        .unwrap();
    let held = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(&path)
        .unwrap();

    let second = gradient(5, 3, 3);
    second.write(&path, DEFAULT_JPEG_QUALITY).unwrap();
    drop(held);
    assert_eq!(ImageU8::read_rgb(&path).unwrap().data(), second.data());
    let names: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
    assert_eq!(names.len(), 1, "the temporary file is removed");
}

/// Encode `data` (`components` channels of `color`) with `jpeg-encoder` at
/// `quality`, with its optimized Huffman tables on or off.
fn encode_jpeg(
    data: &[u8],
    (w, h): (u16, u16),
    color: jpeg_encoder::ColorType,
    sampling: jpeg_encoder::SamplingFactor,
    quality: u8,
    optimized_huffman: bool,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut bytes, quality);
    encoder.set_sampling_factor(sampling);
    encoder.set_optimized_huffman_tables(optimized_huffman);
    encoder.encode(data, w, h, color).unwrap();
    bytes
}

/// The number of start-of-scan (SOS) markers in `jpeg`.
fn jpeg_scan_count(jpeg: &[u8]) -> usize {
    jpeg.windows(2).filter(|w| *w == [0xFF, 0xDA]).count()
}

/// With optimized Huffman tables, `jpeg-encoder` writes a baseline JPEG with
/// each component in a scan of its own. zune-jpeg 0.5.15, the `image` crate's
/// JPEG decoder, returns wrong pixels for such a file without an error: whole
/// images at 4:2:0, and the last row at some odd sizes at 4:4:4. The reader
/// decodes JPEG with `jpeg-decoder` instead, and reads them as OpenCV does.
#[test]
fn read_decodes_a_jpeg_with_one_scan_per_component() {
    use jpeg_encoder::SamplingFactor;
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        (64u16, 64u16, SamplingFactor::R_4_2_0, DEFAULT_JPEG_QUALITY),
        (37, 23, SamplingFactor::R_4_2_0, DEFAULT_JPEG_QUALITY),
        (1, 50, SamplingFactor::R_4_2_0, DEFAULT_JPEG_QUALITY),
        (1, 50, SamplingFactor::R_4_4_4, DEFAULT_JPEG_QUALITY),
        (17, 9, SamplingFactor::R_4_4_4, 50),
    ];
    for (w, h, sampling, quality) in cases {
        let data: Vec<u8> = (0..u32::from(h))
            .flat_map(|y| {
                (0..u32::from(w)).flat_map(move |x| {
                    [
                        200,
                        (x * 255 / u32::from(w)) as u8,
                        (y * 255 / u32::from(h)) as u8,
                    ]
                })
            })
            .collect();
        let bytes = encode_jpeg(
            &data,
            (w, h),
            jpeg_encoder::ColorType::Rgb,
            sampling,
            quality,
            true,
        );
        assert_eq!(jpeg_scan_count(&bytes), 3, "one scan per component");
        let path = dir.path().join("scans.jpg");
        std::fs::write(&path, &bytes).unwrap();

        let back = ImageU8::read_rgb(&path).unwrap();
        assert_eq!((back.width(), back.height()), (u32::from(w), u32::from(h)));
        let row_len = 3 * usize::from(w);
        let row_errors: Vec<f64> = back
            .data()
            .chunks(row_len)
            .zip(data.chunks(row_len))
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| f64::from(a.abs_diff(*b)))
                    .sum::<f64>()
                    / row_len as f64
            })
            .collect();
        let mean_error = row_errors.iter().sum::<f64>() / row_errors.len() as f64;
        let worst_row = row_errors.iter().cloned().fold(0.0, f64::max);
        let case = format!("{w}x{h} {sampling:?} q{quality}");
        assert!(mean_error < 3.0, "{case}: mean error {mean_error}");
        assert!(
            worst_row < 8.0,
            "{case}: worst row's mean error {worst_row}"
        );
        assert_eq!(
            ImageU8::read_rgba(&path)
                .unwrap()
                .data()
                .chunks(4)
                .map(|p| p[3])
                .max(),
            Some(255)
        );
        assert!(!image_has_alpha(&path).unwrap());
    }
}

/// A CMYK JPEG (Adobe-inverted, as Photoshop writes it) reads as RGB with
/// `r = (255 - c) * (255 - k) / 255`, as OpenCV converts it, whether its
/// components are stored as CMYK or as YCCK.
#[test]
fn read_converts_a_cmyk_jpeg_to_rgb() {
    let dir = tempfile::tempdir().unwrap();
    let (w, h) = (24u16, 16u16);
    let cmyk: Vec<u8> = (0..u32::from(h))
        .flat_map(|y| {
            (0..u32::from(w))
                .flat_map(move |x| [(x * 10) as u8, (y * 15) as u8, 60, ((x + y) * 2) as u8])
        })
        .collect();
    let expected: Vec<u8> = cmyk
        .chunks(4)
        .flat_map(|p| {
            let k = 255.0 - f64::from(p[3]);
            [0, 1, 2].map(|i| ((255.0 - f64::from(p[i])) * k / 255.0).round() as u8)
        })
        .collect();
    for color in [
        jpeg_encoder::ColorType::Cmyk,
        jpeg_encoder::ColorType::CmykAsYcck,
    ] {
        let bytes = encode_jpeg(
            &cmyk,
            (w, h),
            color,
            jpeg_encoder::SamplingFactor::R_4_4_4,
            100,
            false,
        );
        let path = dir.path().join("cmyk.jpg");
        std::fs::write(&path, &bytes).unwrap();
        let back = ImageU8::read_rgb(&path).unwrap();
        let mean_error = back
            .data()
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / expected.len() as f64;
        assert!(mean_error < 2.0, "{color:?}: mean error {mean_error}");
    }
}

#[test]
fn read_refuses_a_jpeg_cut_before_its_first_scan_and_a_non_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short.jpg");
    gradient(40, 30, 3)
        .write(&path, DEFAULT_JPEG_QUALITY)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    // Only the markers before the scan data.
    let sos = bytes.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
    std::fs::write(&path, &bytes[..sos]).unwrap();
    assert!(ImageU8::read_rgb(&path).is_err());

    let text = dir.path().join("text.jpg");
    std::fs::write(&text, b"not an image").unwrap();
    assert!(ImageU8::read_rgb(&text).is_err());
    assert!(image_has_alpha(&text).is_err());
    assert!(image_dimensions(&text).is_err());
}

/// A smooth RGB test image, `w x h`, as a photograph is.
fn smooth_rgb(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [(x * 255 / w) as u8, (y * 255 / h) as u8, 128]))
        .collect()
}

/// The mean and largest `|a - b|` over two equal-length byte slices.
fn mean_and_max_difference(a: &[u8], b: &[u8]) -> (f64, u8) {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum();
    let max = a
        .iter()
        .zip(b)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    (sum / a.len() as f64, max)
}

/// Encode a noisy `w x h` RGB image as a 4:2:0 JPEG: noisy, so the scan data,
/// not the headers, makes up most of the file.
fn noisy_jpeg(w: u16, h: u16, progressive: bool, restart_interval: Option<u16>) -> Vec<u8> {
    let image = gradient(u32::from(w), u32::from(h), 3);
    let mut bytes = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut bytes, DEFAULT_JPEG_QUALITY);
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_2_0);
    encoder.set_progressive(progressive);
    if let Some(interval) = restart_interval {
        encoder.set_restart_interval(interval);
    }
    encoder
        .encode(image.data(), w, h, jpeg_encoder::ColorType::Rgb)
        .unwrap();
    assert_eq!(bytes[bytes.len() - 2..], [0xFF, 0xD9]);
    bytes
}

/// Check that `cut`, a decode of a baseline file cut part way through its
/// scan, matches `complete` above the cut and is flat grey (128) below it,
/// as libjpeg and zune-jpeg fill it, not the texture of zero-bit Huffman
/// codes. Returns the number of grey rows at the bottom.
fn assert_cut_decode_is_grey_below_the_cut(cut: &ImageU8, complete: &ImageU8) -> usize {
    let row = 3 * cut.width() as usize;
    let rows: Vec<&[u8]> = cut.data().chunks(row).collect();
    let grey_rows = rows
        .iter()
        .rev()
        .take_while(|r| r.iter().all(|&v| v == 128))
        .count();
    let height = rows.len();
    // The fill starts at most one row (the 4:2:0 chroma upsampling of the
    // row at the cut) below the decoded part, and covers at least an MCU row.
    assert!(grey_rows >= 16, "only {grey_rows} grey rows at the bottom");
    let decoded = height - grey_rows;
    assert!(decoded >= 32, "only {decoded} rows decoded");
    // The rows above the cut's MCU row are the complete file's, within the
    // few grey levels zune-jpeg and jpeg-decoder differ by.
    let top = (decoded / 16 - 1) * 16 * row;
    let (mean, max) = mean_and_max_difference(&cut.data()[..top], &complete.data()[..top]);
    assert!(mean < 1.0 && max <= 10, "mean {mean}, max {max}");
    grey_rows
}

/// A JPEG that ends before its end-of-image marker is decoded by the `image`
/// crate's decoder (zune-jpeg), as libjpeg decodes it: one that lacks only
/// the marker reads as the complete file does (within the few grey levels the
/// two decoders differ by), and in one cut part way through its scan the rows
/// the scan did not reach are flat grey.
#[test]
fn read_decodes_a_jpeg_without_its_end_marker_or_cut_mid_scan() {
    let dir = tempfile::tempdir().unwrap();
    let (w, h) = (160u16, 128u16);
    for progressive in [false, true] {
        let bytes = noisy_jpeg(w, h, progressive, None);
        let complete_path = dir.path().join("complete.jpg");
        std::fs::write(&complete_path, &bytes).unwrap();
        let complete = ImageU8::read_rgb(&complete_path).unwrap();

        let no_eoi = dir.path().join("no_eoi.jpg");
        std::fs::write(&no_eoi, &bytes[..bytes.len() - 2]).unwrap();
        let back = ImageU8::read_rgb(&no_eoi).unwrap();
        let (mean, max) = mean_and_max_difference(back.data(), complete.data());
        assert!(
            mean < 1.0 && max <= 10,
            "progressive {progressive}: mean {mean}, max {max}"
        );

        let cut = dir.path().join("cut.jpg");
        std::fs::write(&cut, &bytes[..bytes.len() * 6 / 10]).unwrap();
        let back = ImageU8::read_rgb(&cut).unwrap();
        assert_eq!((back.width(), back.height()), (u32::from(w), u32::from(h)));
        if !progressive {
            assert_cut_decode_is_grey_below_the_cut(&back, &complete);
        }
    }
}

/// A baseline JPEG with restart markers (DRI), cut part way through its scan,
/// reads as one without them does: `jpeg-decoder` refuses it, finding the end
/// of the file where a restart marker was due, and zune-jpeg reads it.
#[test]
fn read_decodes_a_jpeg_with_restart_markers_cut_mid_scan() {
    let dir = tempfile::tempdir().unwrap();
    let (w, h) = (160u16, 128u16);
    let bytes = noisy_jpeg(w, h, false, Some(4));
    assert!(bytes.windows(2).any(|m| m == [0xFF, 0xDD]), "a DRI segment");
    let complete_path = dir.path().join("complete.jpg");
    std::fs::write(&complete_path, &bytes).unwrap();
    let complete = ImageU8::read_rgb(&complete_path).unwrap();
    // Cuts that leave at least one whole MCU row (16 rows) undecoded.
    for percent in [30, 60] {
        let cut = dir.path().join("cut.jpg");
        std::fs::write(&cut, &bytes[..bytes.len() * percent / 100]).unwrap();
        let back = ImageU8::read_rgb(&cut).unwrap();
        assert_eq!((back.width(), back.height()), (u32::from(w), u32::from(h)));
        assert_cut_decode_is_grey_below_the_cut(&back, &complete);
    }
}

/// A JPEG whose header claims 65500 x 65500 pixels is refused from its
/// header, before any scan is decoded, for both a baseline and a progressive
/// frame, so it takes neither the time nor the memory of decoding it.
#[test]
fn read_refuses_a_jpeg_too_large_to_decode_before_decoding_it() {
    let dir = tempfile::tempdir().unwrap();
    let data = smooth_rgb(64, 64);
    for (progressive, sof) in [(false, 0xC0u8), (true, 0xC2)] {
        let mut bytes = Vec::new();
        let mut encoder = jpeg_encoder::Encoder::new(&mut bytes, DEFAULT_JPEG_QUALITY);
        encoder.set_progressive(progressive);
        encoder
            .encode(&data, 64, 64, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        let at = bytes
            .windows(2)
            .position(|m| m == [0xFF, sof])
            .expect("a frame header");
        // Height and width follow the marker, length and precision.
        bytes[at + 5..at + 9].copy_from_slice(&[0xFF, 0xDC, 0xFF, 0xDC]);
        let path = dir.path().join("huge.jpg");
        std::fs::write(&path, &bytes).unwrap();

        assert_eq!(image_dimensions(&path).unwrap(), (65500, 65500));
        let result = ImageU8::read_rgb(&path);
        assert!(
            matches!(result, Err(::image::ImageError::Limits(_))),
            "progressive {progressive}: {:?}",
            result.err()
        );
        let (result, reached_end) = jpeg_decoder_reads(&bytes);
        assert!(matches!(result, Err(::image::ImageError::Limits(_))));
        assert!(!reached_end, "refused before the scan data was read");
    }
}

/// Decode `bytes` with `jpeg-decoder` as the reader first does, returning the
/// result and whether the decoder read to the end of the bytes. A refusal
/// with the end not reached was made from the markers before the scan data:
/// decoding a scan of these test files reads to their end.
fn jpeg_decoder_reads(bytes: &[u8]) -> (Result<::image::DynamicImage, ::image::ImageError>, bool) {
    let mut source = EndTrackingReader::new(bytes);
    let result = decode_jpeg_with_jpeg_decoder(&mut source, bytes);
    (result, source.reached_end)
}

/// A lossless (SOF3) JPEG, `w x h` with `components` components at
/// `precision` bits, every sample `2^(precision - 1)`: each difference from
/// the predictor is zero, coded with a one-symbol Huffman table as one 0 bit.
fn lossless_jpeg(precision: u8, components: u8, w: u16, h: u16) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xD8];
    // Frame header.
    bytes.extend_from_slice(&[0xFF, 0xC3, 0, 8 + 3 * components, precision]);
    bytes.extend_from_slice(&h.to_be_bytes());
    bytes.extend_from_slice(&w.to_be_bytes());
    bytes.push(components);
    for id in 1..=components {
        bytes.extend_from_slice(&[id, 0x11, 0]);
    }
    // DC Huffman table 0: one code of length 1, for difference category 0.
    bytes.extend_from_slice(&[0xFF, 0xC4, 0, 20, 0x00, 1]);
    bytes.extend_from_slice(&[0; 15]);
    bytes.push(0);
    // Scan header: every component, predictor 1, no point transform.
    bytes.extend_from_slice(&[0xFF, 0xDA, 0, 6 + 2 * components, components]);
    for id in 1..=components {
        bytes.extend_from_slice(&[id, 0x00]);
    }
    bytes.extend_from_slice(&[1, 0, 0]);
    let bits = usize::from(w) * usize::from(h) * usize::from(components);
    bytes.extend(std::iter::repeat_n(0u8, bits.div_ceil(8)));
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    bytes
}

/// A lossless JPEG is read at 8 bits and refused, with an error rather than a
/// panic or wrong values, at any other precision: `jpeg-decoder` returns 2
/// bytes per sample for those while naming an 8-bit pixel format.
#[test]
fn read_accepts_a_lossless_jpeg_only_at_8_bits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lossless.jpg");
    for components in [1, 3] {
        std::fs::write(&path, lossless_jpeg(8, components, 8, 8)).unwrap();
        let back = ImageU8::read_rgb(&path).unwrap();
        assert_eq!((back.width(), back.height()), (8, 8));
        assert!(back.data().iter().all(|&v| v == 128), "{components}");
        assert_eq!(image_dimensions(&path).unwrap(), (8, 8));

        for precision in [6, 12, 16] {
            std::fs::write(&path, lossless_jpeg(precision, components, 8, 8)).unwrap();
            let result = ImageU8::read_rgb(&path);
            assert!(
                matches!(result, Err(::image::ImageError::Unsupported(_))),
                "{components} components at {precision} bits: {:?}",
                result.err()
            );
            assert!(ImageU8::read_rgba(&path).is_err());
            assert_eq!(image_dimensions(&path).unwrap(), (8, 8));
            assert!(!image_has_alpha(&path).unwrap());
        }
    }

    // The precision is read from the frame header, so a large 16-bit file is
    // refused before its scan is decoded: this header claims 65500 x 1300
    // RGB samples, which would take seconds and gigabytes to decode.
    let mut bytes = lossless_jpeg(16, 3, 8, 8);
    let sof = bytes.windows(2).position(|m| m == [0xFF, 0xC3]).unwrap();
    bytes[sof + 5..sof + 9].copy_from_slice(&[0x05, 0x14, 0xFF, 0xDC]);
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(image_dimensions(&path).unwrap(), (65500, 1300));
    let result = ImageU8::read_rgb(&path);
    assert!(
        matches!(result, Err(::image::ImageError::Unsupported(_))),
        "{:?}",
        result.err()
    );
    let (result, reached_end) = jpeg_decoder_reads(&bytes);
    assert!(matches!(result, Err(::image::ImageError::Unsupported(_))));
    assert!(!reached_end, "refused before the scan data was read");
}

#[test]
fn jpeg_layout_reads_the_frame_and_scan_headers() {
    for precision in [6, 8, 12, 16] {
        let layout = jpeg_layout(&lossless_jpeg(precision, 3, 4, 4));
        assert_eq!(layout.precision, Some(precision));
        assert_eq!(layout.frame_components, 3);
        assert!(!layout.sequential);
        assert_eq!(layout.scan_components, [3]);
    }
    let rgb = smooth_rgb(37, 23);
    let jpeg = |optimized_huffman| {
        encode_jpeg(
            &rgb,
            (37, 23),
            jpeg_encoder::ColorType::Rgb,
            jpeg_encoder::SamplingFactor::R_4_2_0,
            DEFAULT_JPEG_QUALITY,
            optimized_huffman,
        )
    };
    let interleaved = jpeg(false);
    let expected = JpegLayout {
        precision: Some(8),
        sequential: true,
        frame_components: 3,
        scan_components: vec![3],
    };
    assert_eq!(jpeg_layout(&interleaved), expected);
    let scans = jpeg(true);
    assert_eq!(jpeg_layout(&scans).scan_components, [1, 1, 1]);

    // Stray bytes between segments and fill bytes before a marker are skipped
    // as `jpeg-decoder` skips them, and it reads such a file as the plain one.
    let sof = interleaved
        .windows(2)
        .position(|m| m == [0xFF, 0xC0])
        .unwrap();
    let insert = |extra: &[u8]| {
        let mut file = interleaved[..sof].to_vec();
        file.extend_from_slice(extra);
        file.extend_from_slice(&interleaved[sof..]);
        file
    };
    let padded = insert(&[0x12, 0x34, 0xFF, 0xFF]);
    assert_eq!(jpeg_layout(&padded), expected);
    let (plain, _) = jpeg_decoder_reads(&interleaved);
    let (decoded, reached_end) = jpeg_decoder_reads(&padded);
    assert!(!reached_end);
    assert_eq!(decoded.unwrap().into_rgb8(), plain.unwrap().into_rgb8());
    // A standalone marker has no length to skip (`jpeg-decoder` refuses a
    // restart marker outside a scan, but its scan data holds them).
    assert_eq!(jpeg_layout(&insert(&[0xFF, 0xD3, 0xFF, 0x01])), expected);

    // No frame header, or a segment that runs past the end.
    assert_eq!(
        jpeg_layout(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 3, 1]).precision,
        None
    );
    assert_eq!(
        jpeg_layout(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 200]),
        JpegLayout::default()
    );
    assert_eq!(jpeg_layout(&[0xFF, 0xD8]), JpegLayout::default());
    assert!(jpeg_layout(&scans[..scans.len() / 2]).scan_components.len() < 3);
}

/// A file with one scan per component that lacks only its end-of-image
/// marker, or is cut inside a later scan, stays with `jpeg-decoder`, which
/// reads it within a few grey levels (zune-jpeg would misread it). A whole
/// one followed by trailing data, another JPEG or stray bytes never reaches
/// the end of the bytes, so it decodes as the plain file does.
#[test]
fn read_keeps_a_jpeg_with_one_scan_per_component_on_jpeg_decoder() {
    let dir = tempfile::tempdir().unwrap();
    let (w, h) = (64u16, 48u16);
    let rgb = smooth_rgb(u32::from(w), u32::from(h));
    let bytes = encode_jpeg(
        &rgb,
        (w, h),
        jpeg_encoder::ColorType::Rgb,
        jpeg_encoder::SamplingFactor::R_4_2_0,
        DEFAULT_JPEG_QUALITY,
        true,
    );
    assert_eq!(jpeg_scan_count(&bytes), 3);
    let read = |name: &str, contents: &[u8]| {
        let path = dir.path().join(name);
        std::fs::write(&path, contents).unwrap();
        ImageU8::read_rgb(&path).unwrap()
    };
    let complete = read("complete.jpg", &bytes);
    let (mean, _) = mean_and_max_difference(complete.data(), &rgb);
    assert!(mean < 3.0, "complete: mean error {mean}");

    let no_eoi = read("no_eoi.jpg", &bytes[..bytes.len() - 2]);
    let (mean, max) = mean_and_max_difference(no_eoi.data(), complete.data());
    assert!(
        mean < 2.0 && max <= 10,
        "no end marker: mean {mean}, max {max}"
    );

    // Cut inside the last (Cr) scan: luma and Cb are whole, so the error
    // stays far below a misread's.
    let last_scan = bytes.windows(2).rposition(|m| m == [0xFF, 0xDA]).unwrap();
    let cut = read("cut.jpg", &bytes[..(last_scan + bytes.len()) / 2]);
    let (mean, _) = mean_and_max_difference(cut.data(), &rgb);
    assert!(mean < 30.0, "cut in the last scan: mean error {mean}");
    // Cut inside the first (luma) scan, so the chroma scans never begin:
    // refused, since jpeg-decoder needs data for every component, where
    // zune-jpeg would return wrong pixels.
    let first_scan = bytes.windows(2).position(|m| m == [0xFF, 0xDA]).unwrap();
    let path = dir.path().join("cut_first.jpg");
    std::fs::write(&path, &bytes[..first_scan + 40]).unwrap();
    assert!(ImageU8::read_rgb(&path).is_err());

    let other = encode_jpeg(
        &smooth_rgb(8, 8),
        (8, 8),
        jpeg_encoder::ColorType::Rgb,
        jpeg_encoder::SamplingFactor::R_4_2_0,
        DEFAULT_JPEG_QUALITY,
        false,
    );
    let trailers: [(&str, &[u8]); 3] = [
        ("garbage", b"\x00\x01trailing garbage \xFF\xD8 \xFF"),
        ("concatenated", &other),
        ("fill", &[0xFF; 64]),
    ];
    for (name, trailer) in trailers {
        let mut file = bytes.clone();
        file.extend_from_slice(trailer);
        let (_, reached_end) = jpeg_decoder_reads(&file);
        assert!(!reached_end, "{name}");
        assert_eq!(read(name, &file).data(), complete.data(), "{name}");
    }
}

/// The header reads, `image_dimensions` and `image_has_alpha`, agree with the
/// reader on JPEGs the `image` crate's decoder cannot read: an 8-bit lossless
/// file, which zune-jpeg 0.5.15 refuses even at its header, and a file with
/// one scan per component, which it misdecodes.
#[test]
fn header_reads_agree_with_the_reader_on_jpegs_zune_cannot_read() {
    let dir = tempfile::tempdir().unwrap();
    let lossless = dir.path().join("lossless.jpg");
    std::fs::write(&lossless, lossless_jpeg(8, 3, 12, 5)).unwrap();
    let (w, h) = (37u16, 23u16);
    let scans = dir.path().join("scans.jpg");
    std::fs::write(
        &scans,
        encode_jpeg(
            &smooth_rgb(u32::from(w), u32::from(h)),
            (w, h),
            jpeg_encoder::ColorType::Rgb,
            jpeg_encoder::SamplingFactor::R_4_2_0,
            DEFAULT_JPEG_QUALITY,
            true,
        ),
    )
    .unwrap();
    for path in [&lossless, &scans] {
        let image = ImageU8::read_rgb(path).unwrap();
        assert_eq!(
            image_dimensions(path).unwrap(),
            (image.width(), image.height())
        );
        assert!(!image_has_alpha(path).unwrap());
    }
    assert_eq!(image_dimensions(&lossless).unwrap(), (12, 5));
    assert_eq!(image_dimensions(&scans).unwrap(), (37, 23));
}
