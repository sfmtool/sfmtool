// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tile decodes of synthetic mip level files and region decodes of synthetic
//! originals, each compared with the same pixels of the whole decode.

use std::sync::Arc;

use jpeg_encoder::SamplingFactor;

use super::*;
use crate::camera::image::ImageU8;
use crate::camera::mipmap::file::encode_mip_level;
use crate::camera::mipmap::record::{MipLevelRecord, ENCODER, MIP_LEVEL_RECORD_VERSION};
use crate::camera::mipmap::region::index_intervals;
use crate::camera::mipmap::test_support::{encode_jpeg, synthetic_rgb, JpegOptions};
use crate::camera::mipmap::{ExpectedMipLevel, MipLevelFile, MipmapError};

/// What a mip level file of `image` must say to be read.
fn expected_for(record: &MipLevelRecord) -> ExpectedMipLevel {
    ExpectedMipLevel {
        source: record.source.clone(),
        source_width: record.source_width,
        source_height: record.source_height,
        level: record.level as usize,
        width: record.width,
        height: record.height,
        jpeg_quality: record.jpeg_quality,
    }
}

/// `image` as the bytes of a level-1 mip level file with a restart interval
/// of `restart_interval` MCUs, grey when `grey`, and what it must say.
fn mip_level_bytes(
    image: &ImageU8,
    grey: bool,
    restart_interval: u16,
) -> (Vec<u8>, ExpectedMipLevel) {
    let record = MipLevelRecord {
        version: MIP_LEVEL_RECORD_VERSION,
        source: "images/a.png".to_string(),
        source_size: 1,
        source_modified_ns: None,
        source_xxh128: "0".repeat(32),
        source_width: image.width() * 2,
        source_height: image.height() * 2,
        level: 1,
        out_of_core_long_side: 64,
        region_readable_original: true,
        width: image.width(),
        height: image.height(),
        stored_levels: 1,
        jpeg_quality: 90,
        components: if grey { 1 } else { 3 },
        restart_interval_mcus: restart_interval,
        encoder: ENCODER.to_string(),
        sfmtool_version: "test".to_string(),
    };
    let bytes = encode_mip_level(image, grey, 90, &record).unwrap();
    (bytes, expected_for(&record))
}

fn mip_level(image: &ImageU8, grey: bool, restart_interval: u16) -> MipLevelFile {
    let (bytes, expected) = mip_level_bytes(image, grey, restart_interval);
    MipLevelFile::from_bytes(bytes.into(), &expected).unwrap()
}

/// `image` with its three channels set to its green channel.
fn grey_of(image: &ImageU8) -> ImageU8 {
    let data = image
        .data()
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[1]; 3])
        .collect();
    ImageU8::new(image.width(), image.height(), 3, data)
}

/// Pixels `x0..x1` by `y0..y1` of an RGB image.
fn crop(image: &ImageU8, (x0, y0, x1, y1): (u32, u32, u32, u32)) -> Vec<u8> {
    let w = image.width() as usize;
    (y0 as usize..y1 as usize)
        .flat_map(|y| &image.data()[(y * w + x0 as usize) * 3..(y * w + x1 as usize) * 3])
        .copied()
        .collect()
}

/// The tiles of `tile` px with their one-pixel border that cover a `width` x
/// `height` image, its corner pixels, the whole image and some rectangles
/// that start at every alignment to the 8 px blocks.
fn rectangles(width: u32, height: u32, tile: u32) -> Vec<(u32, u32, u32, u32)> {
    let mut all = Vec::new();
    for ty in 0..height.div_ceil(tile) {
        for tx in 0..width.div_ceil(tile) {
            all.push((
                (tx * tile).saturating_sub(1),
                (ty * tile).saturating_sub(1),
                (tx * tile + tile + 1).min(width),
                (ty * tile + tile + 1).min(height),
            ));
        }
    }
    for (x, y) in [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ] {
        all.push((x, y, x + 1, y + 1));
    }
    all.push((0, 0, width, height));
    // Rectangles that start on an MCU of 16 px, where the frame of a region
    // decode would start without its margin.
    for y0 in (0..height).step_by(16) {
        for x0 in (0..width).step_by(16) {
            all.push((x0, y0, (x0 + 5).min(width), (y0 + 5).min(height)));
        }
    }
    for y0 in (0..height).step_by(17) {
        for x0 in (0..width).step_by(13) {
            all.push((x0, y0, (x0 + 7).min(width), (y0 + 9).min(height)));
        }
    }
    all
}

#[test]
fn a_whole_decode_is_the_decoders_rgb_but_for_its_rounding() {
    // The decoder's own conversion rounds some pixels differently, by one
    // grey level, and reads a grey file the same.
    let image = synthetic_rgb(301, 203);
    for (grey, source) in [(false, synthetic_rgb(301, 203)), (true, grey_of(&image))] {
        let file = mip_level(&source, grey, 16);
        let decoded = file.decode().unwrap();
        let reference = ImageU8::decode_rgb(file.bytes()).unwrap();
        let worst = decoded
            .data()
            .iter()
            .zip(reference.data())
            .map(|(&a, &b)| a.abs_diff(b))
            .max();
        assert!(worst <= Some(1), "grey {grey}: {worst:?}");
    }
}

#[test]
fn every_tile_of_a_mip_level_equals_the_same_pixels_of_its_whole_decode() {
    // (width, height): MCUs per row and in all.
    let sizes: [(u32, u32); 4] = [(301, 203), (97, 45), (33, 17), (1, 1)];
    for restart_interval in [2u16, 16] {
        let n = usize::from(restart_interval);
        let mcus = |w: u32, h: u32| (w.div_ceil(8) * h.div_ceil(8)) as usize;
        // The sizes give intervals that cross MCU rows and a short last one.
        assert!(sizes
            .iter()
            .any(|&(w, _)| !(w.div_ceil(8) as usize).is_multiple_of(n)));
        assert!(sizes.iter().any(|&(w, h)| !mcus(w, h).is_multiple_of(n)));
        for (width, height) in sizes {
            let image = synthetic_rgb(width, height);
            for (grey, source) in [
                (false, synthetic_rgb(width, height)),
                (true, grey_of(&image)),
            ] {
                let file = mip_level(&source, grey, restart_interval);
                let whole = file.decode().unwrap();
                for rect in rectangles(width, height, 32) {
                    let (x0, y0, x1, y1) = rect;
                    let tile = file.decode_region(x0, y0, x1, y1).unwrap();
                    assert_eq!((tile.width(), tile.height()), (x1 - x0, y1 - y0));
                    assert!(
                        tile.data() == crop(&whole, rect),
                        "{width} x {height}, interval {n}, grey {grey}, {rect:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_rectangle_outside_the_level_is_refused() {
    let file = mip_level(&synthetic_rgb(97, 45), false, 16);
    for (x0, y0, x1, y1) in [
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (5, 5, 4, 6),
        (0, 0, 98, 1),
        (0, 0, 1, 46),
        (97, 0, 98, 1),
        (0, 0, u32::MAX, u32::MAX),
    ] {
        assert!(
            matches!(
                file.decode_region(x0, y0, x1, y1),
                Err(MipmapError::OutOfRange { .. })
            ),
            "{x0} {y0} {x1} {y1}"
        );
    }
}

/// The byte ranges of the restart intervals of a mip level file.
fn intervals_of(bytes: &[u8], width: u32, height: u32, n: usize) -> Vec<std::ops::Range<usize>> {
    let header = crate::camera::mipmap::region::read_header(bytes).unwrap();
    let mcus = (width.div_ceil(8) * height.div_ceil(8)) as usize;
    index_intervals(bytes, header.scan_data, mcus.div_ceil(n)).unwrap()
}

#[test]
fn a_damaged_interval_is_stale_only_for_the_tiles_that_need_it() {
    let image = synthetic_rgb(301, 203);
    let (bytes, expected) = mip_level_bytes(&image, false, 16);
    let whole = MipLevelFile::from_bytes(bytes.clone().into(), &expected)
        .unwrap()
        .decode()
        .unwrap();
    // Interval 3 holds MCUs 48 to 63 of the 38 a row: pixels 80 to 207 of
    // row 8 to 15. Extra bytes in it leave no restart marker where the
    // decoder expects one.
    let damaged_at = intervals_of(&bytes, 301, 203, 16)[3].end;
    let mut damaged = bytes[..damaged_at].to_vec();
    damaged.extend_from_slice(&[0x55; 32]);
    damaged.extend_from_slice(&bytes[damaged_at..]);
    let file = MipLevelFile::from_bytes(damaged.into(), &expected).unwrap();
    assert!(matches!(
        file.decode_region(64, 0, 128, 32),
        Err(MipmapError::Stale { why, .. }) if why.contains("do not decode")
    ));
    assert!(matches!(file.decode(), Err(MipmapError::Stale { .. })));
    // A tile that needs none of it still reads.
    let far = (0, 160, 64, 203);
    assert_eq!(
        file.decode_region(far.0, far.1, far.2, far.3)
            .unwrap()
            .data(),
        crop(&whole, far)
    );
}

#[test]
fn a_cut_or_damaged_mip_level_file_is_an_error_and_never_a_panic() {
    let image = synthetic_rgb(33, 17);
    let (bytes, expected) = mip_level_bytes(&image, false, 2);
    for end in 0..bytes.len() {
        assert!(matches!(
            MipLevelFile::from_bytes(bytes[..end].to_vec().into(), &expected),
            Err(MipmapError::Stale { .. })
        ));
    }
    for at in 0..bytes.len() {
        for change in [0xFF, 0x01, 0x80] {
            let mut damaged = bytes.clone();
            damaged[at] ^= change;
            if let Ok(file) = MipLevelFile::from_bytes(damaged.into(), &expected) {
                let _ = file.decode();
                for (x0, y0, x1, y1) in [(0, 0, 33, 17), (9, 3, 30, 17), (32, 16, 33, 17)] {
                    let _ = file.decode_region(x0, y0, x1, y1);
                }
            }
        }
    }
}

/// A JPEG of `image` with the restart interval `r` at `sampling`, or grey.
fn original(image: &ImageU8, sampling: SamplingFactor, r: u16, grey: bool) -> Vec<u8> {
    encode_jpeg(
        image,
        JpegOptions {
            sampling,
            restart_interval: r,
            grey,
            ..JpegOptions::default()
        },
    )
}

#[test]
fn every_region_of_an_original_equals_the_same_pixels_of_read_rgb() {
    // 333 px is 21 MCUs of 16 px and 42 of 8 px; 37 px is 3 and 5.
    let cases = [
        ("4:2:0", SamplingFactor::F_2_2, 333, 211, 3),
        ("4:2:0", SamplingFactor::F_2_2, 37, 29, 1),
        ("4:2:2", SamplingFactor::F_2_1, 333, 211, 7),
        ("4:4:0", SamplingFactor::F_1_2, 333, 211, 6),
        ("4:4:0", SamplingFactor::F_1_2, 37, 29, 5),
        ("4:4:4", SamplingFactor::F_1_1, 333, 211, 3),
        ("grey", SamplingFactor::F_1_1, 333, 211, 2),
    ];
    for (name, sampling, width, height, r) in cases {
        let bytes = original(&synthetic_rgb(width, height), sampling, r, name == "grey");
        let whole = ImageU8::decode_rgb(&bytes).unwrap();
        let regions = OriginalRegions::from_bytes(bytes.into()).unwrap();
        assert_eq!((regions.width(), regions.height()), (width, height));
        for rect in rectangles(width, height, 48) {
            let (x0, y0, x1, y1) = rect;
            let region = regions.decode_region(x0, y0, x1, y1).unwrap();
            assert_eq!((region.width(), region.height()), (x1 - x0, y1 - y0));
            assert!(
                region.data() == crop(&whole, rect),
                "{name} {width} x {height}, interval {r}: {rect:?}"
            );
        }
    }
}

#[test]
fn an_original_region_outside_the_image_is_refused() {
    let bytes = original(&synthetic_rgb(37, 29), SamplingFactor::F_2_2, 1, false);
    let regions = OriginalRegions::from_bytes(bytes.into()).unwrap();
    for (x0, y0, x1, y1) in [(0, 0, 0, 1), (3, 0, 2, 1), (0, 0, 38, 1), (0, 0, 1, 30)] {
        assert!(matches!(
            regions.decode_region(x0, y0, x1, y1),
            Err(::image::ImageError::Parameter(_))
        ));
    }
}

#[test]
fn a_cut_or_damaged_original_is_an_error_and_never_a_panic() {
    let bytes = original(&synthetic_rgb(37, 29), SamplingFactor::F_2_2, 1, false);
    for end in 0..bytes.len() {
        assert!(OriginalRegions::from_bytes(bytes[..end].to_vec().into()).is_err());
    }
    for at in 0..bytes.len() {
        for change in [0xFF, 0x01, 0x80] {
            let mut damaged = bytes.clone();
            damaged[at] ^= change;
            if let Ok(regions) = OriginalRegions::from_bytes(Arc::from(damaged)) {
                let (w, h) = (regions.width(), regions.height());
                for (x0, y0, x1, y1) in [(0, 0, w, h), (w / 2, h / 2, w, h), (w - 1, 0, w, 1)] {
                    let _ = regions.decode_region(x0, y0, x1, y1);
                }
            }
        }
    }
}
