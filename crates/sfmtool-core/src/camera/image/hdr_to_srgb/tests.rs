// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

const ZERO_CHROMA: u16 = 32768;

fn options(transfer: HdrTransfer, exposure_stops: f32) -> HdrToSrgbOptions {
    HdrToSrgbOptions {
        transfer,
        exposure_stops,
    }
}

/// Pixels given one by one as the three planes the function takes.
fn planar(pixels: &[[u16; 3]]) -> Vec<u16> {
    (0..3)
        .flat_map(|c| pixels.iter().map(move |p| p[c]))
        .collect()
}

/// One pixel through the whole function.
fn convert_one(ycbcr: [u16; 3], transfer: HdrTransfer, stops: f32) -> [u8; 3] {
    let image = hdr_ycbcr_to_srgb(&ycbcr, 1, 1, &options(transfer, stops)).unwrap();
    let d = image.data();
    [d[0], d[1], d[2]]
}

/// A grey at a 10-bit narrow-range signal fraction, as a 16-bit code.
fn grey(fraction: f64) -> [u16; 3] {
    let code10 = 64.0 + fraction * 876.0;
    [(code10 * 64.0).round() as u16, ZERO_CHROMA, ZERO_CHROMA]
}

/// The 16-bit narrow-range Y'CbCr codes of a BT.2020 R'G'B' signal.
fn ycbcr_of(rgb: [f64; 3]) -> [u16; 3] {
    let y = 0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2];
    let cb = (rgb[2] - y) / 1.8814;
    let cr = (rgb[0] - y) / 1.4746;
    [
        ((16.0 + 219.0 * y) * 256.0).round() as u16,
        ((128.0 + 224.0 * cb) * 256.0).round() as u16,
        ((128.0 + 224.0 * cr) * 256.0).round() as u16,
    ]
}

// The expected codes below were computed by hand from the recipe in
// specs/core/camera/hdr-to-srgb.md, in f64, independently of this code; the
// value before rounding is beside each, and none is within 0.1 of a rounding
// boundary.

#[test]
fn hlg_signal_levels_give_the_hand_computed_codes() {
    // HLG 0%, 50% (136.89), 75% = reference white, 203.15 cd/m² before the
    // roll-off (229.02), and 100% = 1000 cd/m², the roll-off's source peak,
    // which reaches 203 cd/m², the sRGB white.
    for (fraction, code) in [(0.0, 0), (0.5, 137), (0.75, 229), (1.0, 255)] {
        assert_eq!(
            convert_one(grey(fraction), HdrTransfer::Hlg, 0.0),
            [code; 3],
            "HLG {fraction}"
        );
    }
    // Above narrow-range white, the signal is clamped to 1.
    assert_eq!(
        convert_one([65535, ZERO_CHROMA, ZERO_CHROMA], HdrTransfer::Hlg, 0.0),
        [255; 3]
    );
}

#[test]
fn pq_signal_levels_give_the_hand_computed_codes() {
    // PQ 203 cd/m², reference white: Y' = PQ⁻¹(203) = 0.58069, code 36652
    // (228.99). PQ 100 cd/m²: code 32581 (185.74). Black, and 10,000 cd/m²,
    // beyond the roll-off's source peak, which clips to white.
    for (y, code) in [(36652, 229), (32581, 186), (4096, 0), (60160, 255)] {
        assert_eq!(
            convert_one([y, ZERO_CHROMA, ZERO_CHROMA], HdrTransfer::Pq, 0.0),
            [code; 3],
            "PQ code {y}"
        );
    }
}

#[test]
fn exposure_scales_the_light_before_the_roll_off() {
    // HLG 75% one stop down: 101.6 cd/m², above the 87.8 cd/m² knee
    // (186.91); one stop up: 406.3 cd/m² (250.00).
    assert_eq!(convert_one(grey(0.75), HdrTransfer::Hlg, -1.0), [187; 3]);
    assert_eq!(convert_one(grey(0.75), HdrTransfer::Hlg, 1.0), [250; 3]);
}

#[test]
fn saturated_bt2020_primaries_have_negative_bt709_components_set_to_zero() {
    // The BT.2020 green primary at HLG 75% is outside BT.709: its red and blue
    // come out negative and are set to 0, and its green is rolled off on its
    // own value (231.12). The red primary likewise (238.62).
    let green = ycbcr_of([0.0, 0.75, 0.0]);
    assert_eq!(green, [32605, 17269, 12994]);
    assert_eq!(convert_one(green, HdrTransfer::Hlg, 0.0), [0, 231, 0]);
    let red = ycbcr_of([0.75, 0.0, 0.0]);
    assert_eq!(red, [15142, 26763, 54272]);
    assert_eq!(convert_one(red, HdrTransfer::Hlg, 0.0), [239, 0, 0]);
}

#[test]
fn every_channel_rises_with_its_signal() {
    for transfer in [HdrTransfer::Hlg, HdrTransfer::Pq] {
        // Every grey code from black to white.
        let greys: Vec<[u16; 3]> = (4096..=60160)
            .map(|y| [y, ZERO_CHROMA, ZERO_CHROMA])
            .collect();
        let width = greys.len() as u32;
        let greys = planar(&greys);
        let image = hdr_ycbcr_to_srgb(&greys, width, 1, &options(transfer, 0.0)).unwrap();
        let out = image.data();
        assert!(out
            .as_chunks::<3>()
            .0
            .iter()
            .all(|p| p[0] == p[1] && p[1] == p[2]));
        assert!(out
            .as_chunks::<3>()
            .0
            .iter()
            .zip(out.as_chunks::<3>().0.iter().skip(1))
            .all(|(a, b)| a[0] <= b[0]));
        assert_eq!((out[0], out[out.len() - 1]), (0, 255), "{transfer:?}");

        // Each primary alone, at every 10-bit signal level: its own output
        // channel never falls.
        for channel in 0..3 {
            let ramp: Vec<[u16; 3]> = (0..=876)
                .map(|level| {
                    let mut rgb = [0.0; 3];
                    rgb[channel] = level as f64 / 876.0;
                    ycbcr_of(rgb)
                })
                .collect();
            let image = hdr_ycbcr_to_srgb(&planar(&ramp), 877, 1, &options(transfer, 0.0)).unwrap();
            let own: Vec<u8> = image
                .data()
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| p[channel])
                .collect();
            assert!(
                own.windows(2).all(|w| w[0] <= w[1]),
                "{transfer:?} channel {channel}: {own:?}"
            );
            assert!(own[876] > 200, "{transfer:?} channel {channel}");
        }
    }
}

/// A frame of `width` x `height` pixels that repeats a set of distinct
/// pixels, each at many places.
fn patterned_frame(width: u32, height: u32) -> (Vec<u16>, Vec<[u16; 3]>) {
    let palette: Vec<[u16; 3]> = (0..37u32)
        .map(|i| {
            [
                (4096 + i * 1531 % 56065) as u16,
                (20000 + i * 977 % 25000) as u16,
                (45000 - i * 613 % 25000) as u16,
            ]
        })
        .collect();
    let frame: Vec<[u16; 3]> = (0..width * height)
        .map(|i| palette[(i as usize * 7) % palette.len()])
        .collect();
    (planar(&frame), palette)
}

#[test]
fn a_pixel_converts_the_same_wherever_it_is_in_the_frame() {
    let (width, height) = (61, 23);
    let (frame, palette) = patterned_frame(width, height);
    for transfer in [HdrTransfer::Hlg, HdrTransfer::Pq] {
        let image = hdr_ycbcr_to_srgb(&frame, width, height, &options(transfer, 0.5)).unwrap();
        let alone: Vec<[u8; 3]> = palette
            .iter()
            .map(|&p| convert_one(p, transfer, 0.5))
            .collect();
        for (i, out) in image.data().as_chunks::<3>().0.iter().enumerate() {
            assert_eq!(*out, alone[(i * 7) % palette.len()], "pixel {i}");
        }
        // The palette spans more than one output value, so the comparison
        // above can tell pixels apart.
        assert!(alone.iter().any(|p| *p != alone[0]));
    }
}

#[test]
fn the_output_does_not_depend_on_the_thread_count() {
    let (width, height) = (97, 41);
    let (frame, _) = patterned_frame(width, height);
    let opts = options(HdrTransfer::Hlg, -0.25);
    let first = hdr_ycbcr_to_srgb(&frame, width, height, &opts).unwrap();
    let again = hdr_ycbcr_to_srgb(&frame, width, height, &opts).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let one_thread = pool
        .install(|| hdr_ycbcr_to_srgb(&frame, width, height, &opts))
        .unwrap();
    assert_eq!(first.data(), again.data());
    assert_eq!(first.data(), one_thread.data());
    assert_eq!(
        (first.width(), first.height(), first.channels()),
        (97, 41, 3)
    );
}

#[test]
fn bad_input_is_an_error() {
    let opts = options(HdrTransfer::Pq, 0.0);
    assert_eq!(
        hdr_ycbcr_to_srgb(&[0u16; 5], 1, 2, &opts).err(),
        Some(HdrToSrgbError::SampleCount {
            width: 1,
            height: 2,
            samples: 5
        })
    );
    assert!(hdr_ycbcr_to_srgb(&[], u32::MAX, u32::MAX, &opts).is_err());
    for stops in [f32::NAN, f32::INFINITY, 16.5, -17.0] {
        assert!(matches!(
            hdr_ycbcr_to_srgb(
                &[4096, ZERO_CHROMA, ZERO_CHROMA],
                1,
                1,
                &options(HdrTransfer::Hlg, stops)
            ),
            Err(HdrToSrgbError::Exposure(_))
        ));
    }
    // The limit itself is taken.
    assert!(hdr_ycbcr_to_srgb(
        &[4096, ZERO_CHROMA, ZERO_CHROMA],
        1,
        1,
        &options(HdrTransfer::Hlg, 16.0)
    )
    .is_ok());
}

#[test]
fn the_roll_off_constants_are_bt2390s() {
    let r = roll_off();
    assert!((r.max_lum - 0.772_370).abs() < 1e-5, "{}", r.max_lum);
    assert!((r.knee - 0.658_555).abs() < 1e-5, "{}", r.knee);
    assert!((r.knee_nits - 87.836).abs() < 1e-2, "{}", r.knee_nits);
    // The source peak reaches the target peak; reference white is pulled
    // down to 159.04 cd/m².
    assert!((eetf(1000.0, r) - 203.0).abs() < 1e-2);
    assert!((eetf(203.0, r) - 159.042).abs() < 1e-2);
    assert_eq!(eetf(50.0, r), 50.0);
}
