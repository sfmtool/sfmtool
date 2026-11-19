// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use jpeg_encoder::SamplingFactor;

use super::*;
use crate::camera::mipmap::test_support::{encode_jpeg, encode_png, synthetic_rgb, JpegOptions};

fn jpeg(width: u32, height: u32, options: JpegOptions) -> Vec<u8> {
    encode_jpeg(&synthetic_rgb(width, height), options)
}

fn with_interval(sampling: SamplingFactor, restart_interval: u16) -> JpegOptions {
    JpegOptions {
        sampling,
        restart_interval,
        ..JpegOptions::default()
    }
}

#[test]
fn every_sampling_with_a_dividing_interval_is_region_readable() {
    // 300 x 200: 38 MCUs of 8 px or 19 of 16 px per row.
    for (sampling, r, sampling_hv, per_row, rows) in [
        (SamplingFactor::F_1_1, 19, (1, 1), 38, 25),
        (SamplingFactor::F_2_2, 19, (2, 2), 19, 13),
        (SamplingFactor::F_2_1, 1, (2, 1), 19, 25),
        (SamplingFactor::F_1_2, 2, (1, 2), 38, 13),
    ] {
        let layout = region_layout(&jpeg(300, 200, with_interval(sampling, r))).unwrap();
        assert_eq!(layout.luma_sampling, sampling_hv);
        assert_eq!((layout.mcus_per_row, layout.mcu_rows), (per_row, rows));
        assert_eq!(
            layout.intervals.len(),
            (rows * per_row / u32::from(r)) as usize
        );
        assert_eq!(
            (layout.width, layout.height, layout.components),
            (300, 200, 3)
        );
    }
    let grey = JpegOptions {
        grey: true,
        ..with_interval(SamplingFactor::F_1_1, 38)
    };
    let layout = region_layout(&jpeg(300, 200, grey)).unwrap();
    assert_eq!(layout.components, 1);
    assert_eq!(layout.intervals.len(), 25);
}

#[test]
fn the_intervals_are_the_bytes_between_the_restart_markers() {
    let bytes = jpeg(64, 16, with_interval(SamplingFactor::F_1_1, 4));
    let layout = region_layout(&bytes).unwrap();
    assert_eq!(layout.intervals.len(), 4);
    for (k, pair) in layout.intervals.windows(2).enumerate() {
        assert_eq!(pair[0].end + 2, pair[1].start);
        assert_eq!(bytes[pair[0].end..pair[0].end + 2], [0xFF, 0xD0 + k as u8]);
    }
    let last = layout.intervals.last().unwrap();
    assert_eq!(bytes[last.end..last.end + 2], [0xFF, 0xD9]);
}

#[test]
fn each_rule_refuses_its_case() {
    let image = synthetic_rgb(300, 200);
    assert_eq!(
        region_layout(&encode_png(&image)),
        Err(NotRegionReadable::NotJpeg)
    );
    let progressive = JpegOptions {
        progressive: true,
        ..with_interval(SamplingFactor::F_2_2, 19)
    };
    assert!(matches!(
        region_layout(&encode_jpeg(&image, progressive)),
        Err(NotRegionReadable::Frame(_))
    ));
    assert!(matches!(
        region_layout(&encode_jpeg(&image, with_interval(SamplingFactor::F_2_2, 0))),
        Err(NotRegionReadable::RestartInterval(why)) if why.contains("no DRI")
    ));
    // 7 does not divide the 19 MCUs of a row, so an interval would run into
    // the next row.
    assert!(matches!(
        region_layout(&encode_jpeg(&image, with_interval(SamplingFactor::F_2_2, 7))),
        Err(NotRegionReadable::RestartInterval(why)) if why.contains("does not divide")
    ));
    // 1040 px at 4:4:4 is 130 MCUs: an interval of 65 MCUs is 520 px and
    // passes, one of 130 is 1040 px and is wider than 1024.
    let wide = synthetic_rgb(1040, 16);
    assert!(region_layout(&encode_jpeg(
        &wide,
        with_interval(SamplingFactor::F_1_1, 65)
    ))
    .is_ok());
    assert!(matches!(
        region_layout(&encode_jpeg(&wide, with_interval(SamplingFactor::F_1_1, 130))),
        Err(NotRegionReadable::RestartInterval(why)) if why.contains("wider than 1024")
    ));
}

/// The offset of the first `FF marker` in `bytes`.
fn find_marker(bytes: &[u8], marker: u8) -> usize {
    bytes.windows(2).position(|w| w == [0xFF, marker]).unwrap()
}

#[test]
fn components_the_decoder_reads_as_rgb_are_refused() {
    let mut bytes = jpeg(64, 16, with_interval(SamplingFactor::F_1_1, 4));
    let sof = find_marker(&bytes, 0xC0);
    // The component identifiers follow the 2-byte marker, the 2-byte length
    // and 6 bytes of the frame header, 3 bytes apart.
    for (k, id) in b"RGB".iter().copied().enumerate() {
        bytes[sof + 10 + 3 * k] = id;
    }
    // The scan header names the components by the same identifiers.
    let sos = find_marker(&bytes, 0xDA);
    for (k, id) in b"RGB".iter().copied().enumerate() {
        bytes[sos + 5 + 2 * k] = id;
    }
    assert!(matches!(
        region_layout(&bytes),
        Err(NotRegionReadable::Colour(why)) if why.contains("RGB")
    ));
}

#[test]
fn a_damaged_scan_is_refused() {
    let bytes = jpeg(64, 16, with_interval(SamplingFactor::F_1_1, 4));
    // Cut short before end-of-image.
    assert!(matches!(
        region_layout(&bytes[..bytes.len() - 2]),
        Err(NotRegionReadable::Intervals(_))
    ));
    // RST0 renumbered RST1.
    let mut swapped = bytes.clone();
    let rst0 = find_marker(&swapped, 0xD0);
    swapped[rst0 + 1] = 0xD1;
    assert!(matches!(
        region_layout(&swapped),
        Err(NotRegionReadable::Intervals(why)) if why.contains("RST1")
    ));
    // Bytes after end-of-image are ignored.
    let mut trailing = bytes.clone();
    trailing.extend_from_slice(b"trailing bytes");
    assert!(region_layout(&trailing).is_ok());
}

#[test]
fn each_rule_refuses_a_file_that_breaks_only_it() {
    // 4:4:4, 8 MCUs a row and 2 rows, an interval of 4 MCUs: readable.
    let base = jpeg(64, 16, with_interval(SamplingFactor::F_1_1, 4));
    assert!(region_layout(&base).is_ok());
    let sof = find_marker(&base, 0xC0);
    let sos = find_marker(&base, 0xDA);
    let dri = find_marker(&base, 0xDD);
    let patched = |at: usize, value: u8| {
        let mut bytes = base.clone();
        bytes[at] = value;
        region_layout(&bytes)
    };
    // Rule 2. The frame header's body starts 4 bytes after its marker: the
    // precision, the height, the width, then 3 bytes per component.
    assert!(matches!(
        patched(sof + 4, 12),
        Err(NotRegionReadable::Frame(why)) if why.contains("12-bit")
    ));
    let mut no_width = base.clone();
    no_width[sof + 7..sof + 9].copy_from_slice(&[0, 0]);
    assert!(matches!(
        region_layout(&no_width),
        Err(NotRegionReadable::Frame(why)) if why.contains("0 x 16")
    ));
    // Rule 4: a component's sampling byte is its second.
    assert!(matches!(
        patched(sof + 11 + 3, 0x21),
        Err(NotRegionReadable::Sampling(why)) if why.contains("chroma sampled 2 x 1")
    ));
    assert!(matches!(
        patched(sof + 11, 0x31),
        Err(NotRegionReadable::Sampling(why)) if why.contains("luma sampled 3 x 1")
    ));
    // Rule 5. The scan header's body is the count, two bytes per component,
    // then the spectral selection and the approximation.
    let mut reordered = base.clone();
    reordered.swap(sos + 7, sos + 9);
    assert!(matches!(
        region_layout(&reordered),
        Err(NotRegionReadable::Scan(why)) if why.contains("holds components")
    ));
    for (at, value) in [(sos + 11, 1), (sos + 12, 62), (sos + 13, 0x10)] {
        assert!(matches!(
            patched(at, value),
            Err(NotRegionReadable::Scan(why)) if why.contains("spectral selection")
        ));
    }
    let mut second_scan = base[..base.len() - 2].to_vec();
    second_scan.extend_from_slice(&[0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 63, 0, 0x12, 0xFF, 0xD9]);
    assert!(matches!(
        region_layout(&second_scan),
        Err(NotRegionReadable::Scan(why)) if why.contains("a second scan")
    ));
    // Rule 6: the interval follows the DRI segment's length.
    let mut zero = base.clone();
    zero[dri + 4..dri + 6].copy_from_slice(&[0, 0]);
    assert!(matches!(
        region_layout(&zero),
        Err(NotRegionReadable::RestartInterval(why)) if why.contains("an interval of 0")
    ));
}
