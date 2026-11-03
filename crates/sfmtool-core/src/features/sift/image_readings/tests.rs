// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// An RGB image whose three channels all hold `value(x, y)`.
fn grey_rgb(w: usize, h: usize, value: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let v = value(x, y);
            [v, v, v]
        })
        .collect()
}

/// The conversion that reads the green channel, so a sample `v` is `v / 255`
/// with no weights to round.
fn green() -> ImageToGray {
    ImageToGray::from_gray_formula("G").unwrap()
}

fn readings(w: usize, h: usize, rgb: &[u8], to_gray: &ImageToGray) -> ImageReadings {
    image_readings_from_rgb(w as u32, h as u32, rgb, to_gray).unwrap()
}

#[test]
fn a_constant_image_reads_zero() {
    // At its own size every sample is the same double, so the readings are
    // exactly zero.
    let rgb = grey_rgb(300, 200, |_, _| 77);
    assert_eq!(
        readings(300, 200, &rgb, &ImageToGray::default()),
        ImageReadings::ZERO
    );
    // Reduced by 4/3, the coverage weights are rounded, so the samples agree
    // only to a few ulps.
    let rgb = grey_rgb(640, 480, |_, _| 200);
    let r = readings(640, 480, &rgb, &ImageToGray::default());
    assert!(r.sharpness < 1e-28, "{r:?}");
    for v in r.gradient_structure_tensor_xx_yy_xy {
        assert!(v.abs() < 1e-28, "{r:?}");
    }
}

#[test]
fn a_vertical_step_edge_reads_by_hand() {
    // Columns 0, 0, 0, 1, 1, 1 over five rows. Interior: x in 1..=4, y in
    // 1..=3, n = 12. Gx is 0, 1/2, 1/2, 0 across a row, so Exx = 4 · 3 / 4 /
    // 12 = 1/8; Gy = 0. The Laplacian is 0, 1, −1, 0 across a row, so its mean
    // is 0 and its variance 1/2.
    let rgb = grey_rgb(6, 5, |x, _| if x >= 3 { 255 } else { 0 });
    let r = readings(6, 5, &rgb, &green());
    assert_eq!(r.sharpness, 0.5);
    assert_eq!(r.gradient_structure_tensor_xx_yy_xy, [0.125, 0.0, 0.0]);

    // The same edge turned a quarter: the energy moves to Eyy.
    let rgb = grey_rgb(5, 6, |_, y| if y >= 3 { 255 } else { 0 });
    let r = readings(5, 6, &rgb, &green());
    assert_eq!(r.sharpness, 0.5);
    assert_eq!(r.gradient_structure_tensor_xx_yy_xy, [0.0, 0.125, 0.0]);
}

#[test]
fn a_linear_ramp_reads_its_slopes() {
    // I = (10 x + 20 y) / 255: the Sobel derivatives divided by 8 are the
    // slopes, and the Laplacian of a plane is 0.
    let rgb = grey_rgb(6, 6, |x, y| (10 * x + 20 * y) as u8);
    let r = readings(6, 6, &rgb, &green());
    let (a, b) = (10.0 / 255.0, 20.0 / 255.0);
    let [xx, yy, xy] = r.gradient_structure_tensor_xx_yy_xy;
    assert!((xx - a * a).abs() < 1e-15, "{xx}");
    assert!((yy - b * b).abs() < 1e-15, "{yy}");
    assert!((xy - a * b).abs() < 1e-15, "{xy}");
    assert!(r.sharpness < 1e-30, "{}", r.sharpness);
}

#[test]
fn a_checkerboard_has_the_largest_laplacian_and_no_gradient() {
    // Each interior Laplacian is ±4, half of each, so the variance is 16, the
    // bound for samples in [0, 1]. The columns either side of a pixel are
    // equal, so both Sobel derivatives are 0.
    let rgb = grey_rgb(6, 6, |x, y| if (x + y) % 2 == 0 { 255 } else { 0 });
    let r = readings(6, 6, &rgb, &green());
    assert_eq!(r.sharpness, 16.0);
    assert_eq!(r.gradient_structure_tensor_xx_yy_xy, [0.0; 3]);
}

#[test]
fn the_reading_image_size_rule() {
    // At or below 480 on the long side: unchanged, never enlarged.
    for (w, h) in [
        (480, 270),
        (270, 480),
        (100, 50),
        (480, 480),
        (3, 3),
        (1, 1),
    ] {
        assert_eq!(reading_image_size(w, h), (w, h));
    }
    // Above 480: the long side becomes 480, the short side S · 480 / L
    // rounded half up, portrait kept portrait.
    let cases = [
        ((481, 481), (480, 480)),
        ((1920, 1080), (480, 270)),
        ((1080, 1920), (270, 480)),
        ((3840, 2160), (480, 270)),
        ((4000, 3000), (480, 360)),
        // 129 · 480 / 641 = 96.6: rounded to 97, both odd.
        ((641, 129), (480, 97)),
        ((129, 641), (97, 480)),
        // Exact halves round up: 2.5 -> 3, 1.5 -> 2, 0.5 -> 1.
        ((960, 5), (480, 3)),
        ((5, 960), (3, 480)),
        ((960, 3), (480, 2)),
        ((960, 1), (480, 1)),
        // 0.24 rounds to 0, which becomes 1.
        ((2000, 1), (480, 1)),
    ];
    for ((w, h), expected) in cases {
        assert_eq!(reading_image_size(w, h), expected, "{w} x {h}");
    }
}

#[test]
fn area_taps_are_the_exact_overlaps() {
    // 3 to 2: the middle pixel is split half and half.
    let taps = area_taps(3, 2);
    assert_eq!(
        taps,
        vec![
            vec![(0, 2.0 / 3.0), (1, 1.0 / 3.0)],
            vec![(1, 1.0 / 3.0), (2, 2.0 / 3.0)],
        ]
    );
    // An integer factor is equal weights; the same size is the identity.
    assert_eq!(
        area_taps(4, 2),
        vec![vec![(0, 0.5), (1, 0.5)], vec![(2, 0.5), (3, 0.5)]]
    );
    assert_eq!(
        area_taps(3, 3),
        vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]]
    );
    // Every output's weights sum to one, and every source pixel is used.
    for (n, m) in [(1920, 480), (2040, 480), (641, 97), (1000, 7), (5, 3)] {
        let taps = area_taps(n, m);
        let mut used = vec![0f64; n];
        for row in &taps {
            let total: f64 = row.iter().map(|t| t.1).sum();
            assert!((total - 1.0).abs() < 1e-14, "{n} -> {m}: {total}");
            for &(s, w) in row {
                used[s] += w * n as f64 / m as f64;
            }
        }
        assert!(used.iter().all(|&u| (u - 1.0).abs() < 1e-12), "{n} -> {m}");
    }
}

#[test]
fn the_reading_image_is_the_area_integral() {
    // Each output pixel against a direct two-dimensional integral of the
    // piecewise-constant grey image over its footprint.
    let (w, h, ow, oh) = (11usize, 7usize, 4usize, 3usize);
    let rgb = grey_rgb(w, h, |x, y| ((x * 37 + y * 101) % 256) as u8);
    let image = reading_image(w, h, &rgb, &green(), ow, oh);
    let grey = |x: usize, y: usize| f64::from(rgb[(y * w + x) * 3 + 1]) / 255.0;
    for j in 0..oh {
        for i in 0..ow {
            let (x0, x1) = (
                i as f64 * w as f64 / ow as f64,
                (i + 1) as f64 * w as f64 / ow as f64,
            );
            let (y0, y1) = (
                j as f64 * h as f64 / oh as f64,
                (j + 1) as f64 * h as f64 / oh as f64,
            );
            let mut integral = 0.0;
            for y in 0..h {
                for x in 0..w {
                    let ox = (x1.min(x as f64 + 1.0) - x0.max(x as f64)).max(0.0);
                    let oy = (y1.min(y as f64 + 1.0) - y0.max(y as f64)).max(0.0);
                    integral += ox * oy * grey(x, y);
                }
            }
            let expected = integral / ((x1 - x0) * (y1 - y0));
            let got = image[j * ow + i];
            assert!(
                (got - expected).abs() < 1e-14,
                "({i}, {j}): {got} vs {expected}"
            );
        }
    }
}

#[test]
fn a_large_image_is_reduced_before_the_readings() {
    // A 1-pixel checkerboard at 960 x 960 reduces by 2 to a flat 0.5 grey.
    let rgb = grey_rgb(960, 960, |x, y| if (x + y) % 2 == 0 { 255 } else { 0 });
    assert_eq!(readings(960, 960, &rgb, &green()), ImageReadings::ZERO);
    // At 480 x 480 it is read as it is.
    let rgb = grey_rgb(480, 480, |x, y| if (x + y) % 2 == 0 { 255 } else { 0 });
    assert_eq!(readings(480, 480, &rgb, &green()).sharpness, 16.0);
}

#[test]
fn quantize_bits_rounds_the_grey_samples() {
    // The step edge again, whose sharpness is half the step squared. "0.4 * G"
    // makes the bright side 0.4, which 1 bit rounds to 0 and 2 bits to 1/3.
    let rgb = grey_rgb(6, 5, |x, _| if x >= 3 { 255 } else { 0 });
    let scaled = |bits| ImageToGray::new("0.4 * G", bits).unwrap();
    assert_eq!(readings(6, 5, &rgb, &scaled(Some(1))), ImageReadings::ZERO);
    let r = readings(6, 5, &rgb, &scaled(Some(2)));
    let third = 1.0f64 / 3.0;
    assert!((r.sharpness - 0.5 * third * third).abs() < 1e-15, "{r:?}");
    let r = readings(6, 5, &rgb, &scaled(None));
    assert!((r.sharpness - 0.5 * 0.4 * 0.4).abs() < 1e-15, "{r:?}");
}

#[test]
fn a_non_finite_reading_is_an_error() {
    // R ** -1 at R = 0 is an infinity.
    let rgb = grey_rgb(8, 8, |x, _| if x == 4 { 0 } else { 128 });
    let inverse = ImageToGray::from_gray_formula("R ** -1").unwrap();
    let err = image_readings_from_rgb(8, 8, &rgb, &inverse).unwrap_err();
    match &err {
        ImageReadingsError::NotFinite { image_to_gray, .. } => {
            assert_eq!(image_to_gray, r#"{"formula":"R ** -1"}"#);
        }
        other => panic!("expected NotFinite, got {other:?}"),
    }
    assert!(err.to_string().contains("R ** -1"), "{err}");
    // Rounded to a bit depth, the infinity clamps to 1 and the readings are
    // finite.
    let clamped = ImageToGray::new("R ** -1", Some(8)).unwrap();
    assert!(image_readings_from_rgb(8, 8, &rgb, &clamped).is_ok());
    // A NaN stays NaN through the rounding.
    let nan = ImageToGray::new("-1 ** 0.5", Some(8)).unwrap();
    assert!(matches!(
        image_readings_from_rgb(8, 8, &rgb, &nan),
        Err(ImageReadingsError::NotFinite { .. })
    ));
    // A non-finite sample on the reading image's corner still reaches the
    // Sobel sums.
    let corner = grey_rgb(8, 8, |x, y| if (x, y) == (0, 0) { 0 } else { 128 });
    assert!(image_readings_from_rgb(8, 8, &corner, &inverse).is_err());
}

#[test]
fn degenerate_sizes_read_zero() {
    let nan = ImageToGray::from_gray_formula("-1 ** 0.5").unwrap();
    for (w, h) in [
        (0, 0),
        (1, 1),
        (2, 2),
        (2, 100),
        (100, 2),
        (1000, 2),
        (2000, 1),
    ] {
        let rgb = grey_rgb(w, h, |x, y| ((x * 7 + y * 3) % 256) as u8);
        // The conversion is not evaluated, so even a NaN formula reads 0.
        assert_eq!(readings(w, h, &rgb, &nan), ImageReadings::ZERO, "{w} x {h}");
    }
    // 3 x 3 has one interior pixel: a variance of one value is 0, but the
    // tensor is that pixel's gradient.
    let rgb = grey_rgb(3, 3, |x, _| if x == 2 { 255 } else { 0 });
    let r = readings(3, 3, &rgb, &green());
    assert_eq!(r.sharpness, 0.0);
    assert_eq!(r.gradient_structure_tensor_xx_yy_xy, [0.25, 0.0, 0.0]);
}

#[test]
fn a_wrong_pixel_count_is_an_error() {
    let err = image_readings_from_rgb(4, 4, &[0u8; 47], &green()).unwrap_err();
    assert_eq!(
        err,
        ImageReadingsError::PixelCount {
            width: 4,
            height: 4,
            len: 47
        }
    );
    // A size whose byte count overflows is refused, not wrapped.
    assert!(image_readings_from_rgb(u32::MAX, u32::MAX, &[], &green()).is_err());
}

#[test]
fn the_result_does_not_depend_on_the_thread_count() {
    let mut state = 0x2545_F491u32;
    let rgb: Vec<u8> = (0..1283 * 721 * 3)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect();
    let to_gray = ImageToGray::default();
    let many = readings(1283, 721, &rgb, &to_gray);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| readings(1283, 721, &rgb, &to_gray));
    assert_eq!(many, one);
    assert!(many.sharpness > 0.0);
}
