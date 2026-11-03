// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_precedence_pow_over_mul_over_add() {
    // 2 + 3 * 4 ** 2 = 2 + 3 * 16 = 2 + 48 = 50
    let f = parse_gray_formula("2 + 3 * 4 ** 2").unwrap();
    assert_eq!(f.eval(0.0, 0.0, 0.0), 50.0);
}

#[test]
fn test_pow_right_associative() {
    // 2 ** 3 ** 2 = 2 ** 9 = 512 (right-assoc), not (2**3)**2 = 64
    let f = parse_gray_formula("2 ** 3 ** 2").unwrap();
    assert_eq!(f.eval(0.0, 0.0, 0.0), 512.0);
}

#[test]
fn test_parentheses_override_precedence() {
    // (2 + 3) * 4 = 20
    let f = parse_gray_formula("(2 + 3) * 4").unwrap();
    assert_eq!(f.eval(0.0, 0.0, 0.0), 20.0);
}

#[test]
fn test_negative_literal() {
    // -2 * 3 + 10 = 4
    let f = parse_gray_formula("-2 * 3 + 10").unwrap();
    assert_eq!(f.eval(0.0, 0.0, 0.0), 4.0);
}

#[test]
fn test_default_colmap_formula() {
    let f = parse_gray_formula(DEFAULT_GRAY_FORMULA).unwrap();
    // Pure white -> 0.2126 + 0.7152 + 0.0722 = 1.0, to an f32 rounding.
    assert!((f.eval(1.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
    // Pure red -> 0.2126, the f32 nearest the literal.
    assert_eq!(f.eval(1.0, 0.0, 0.0), 0.2126f32);
    // A sample value: the same operations in the same order, in f32.
    let v = f.eval(0.5, 0.25, 0.75);
    let expected = 0.2126f32 * 0.5 + 0.7152f32 * 0.25 + 0.0722f32 * 0.75;
    assert_eq!(v.to_bits(), expected.to_bits());
    // ... which is not the f64 evaluation rounded to f32 for this colour.
    let in_f64 = (0.2126f64 * 0.5 + 0.7152 * 0.25 + 0.0722 * 0.75) as f32;
    assert_ne!(v.to_bits(), in_f64.to_bits());
}

#[test]
fn test_channel_selection() {
    let f = parse_gray_formula("G").unwrap();
    assert_eq!(f.eval(0.1, 0.7, 0.3), 0.7);
}

#[test]
fn test_gray_from_rgb_small_image() {
    // 2x1 image: white pixel then black pixel.
    let rgb = vec![255u8, 255, 255, 0, 0, 0];
    let img = gray_from_rgb(2, 1, &rgb, &ImageToGray::default());
    assert_eq!(img.width(), 2);
    assert_eq!(img.height(), 1);
    assert!((img.get_pixel(0, 0) - 1.0).abs() < 1e-6);
    assert!((img.get_pixel(1, 0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_gray_from_rgb_no_upper_clamp() {
    // A formula that exceeds 1.0 must not be clamped.
    let f = ImageToGray::from_gray_formula("2 * R").unwrap();
    let rgb = vec![255u8, 0, 0];
    let img = gray_from_rgb(1, 1, &rgb, &f);
    assert!((img.get_pixel(0, 0) - 2.0).abs() < 1e-6);
}

#[test]
fn test_gray_from_rgb_rounds_to_quantize_bits() {
    // 0.2126 * 100/255 + 0.7152 * 150/255 + 0.0722 * 200/255 = 0.5606..., which
    // rounds to 143/255 at 8 bits; at 1 bit it rounds to 1.
    let rgb = vec![100u8, 150, 200];
    let exact = gray_from_rgb(1, 1, &rgb, &ImageToGray::default()).get_pixel(0, 0);
    let eight = ImageToGray::new(DEFAULT_GRAY_FORMULA, Some(8)).unwrap();
    let rounded = gray_from_rgb(1, 1, &rgb, &eight).get_pixel(0, 0);
    assert_eq!(rounded, 143.0f32 / 255.0);
    assert_ne!(rounded, exact);
    // (0, 14, 76) is an exact tie, 0.7152 · 14 + 0.0722 · 76 = 15.5 levels.
    // In f32 the formula's value times 255 rounds to 15.5, which rounds up to
    // 16; the same formula in f64 gives 15.499999999999998 and level 15.
    let tie = gray_from_rgb(1, 1, &[0, 14, 76], &eight).get_pixel(0, 0);
    assert_eq!(tie, 16.0f32 / 255.0);
    let one = ImageToGray::new(DEFAULT_GRAY_FORMULA, Some(1)).unwrap();
    assert_eq!(gray_from_rgb(1, 1, &rgb, &one).get_pixel(0, 0), 1.0);
    // Rounding clamps, so a value above 1 reads 1.
    let doubled = ImageToGray::new("2 * R", Some(8)).unwrap();
    assert_eq!(
        gray_from_rgb(1, 1, &[255, 0, 0], &doubled).get_pixel(0, 0),
        1.0
    );
}

#[test]
fn test_parse_errors() {
    assert!(parse_gray_formula("R +").is_err());
    assert!(parse_gray_formula("R + + G").is_err());
    assert!(parse_gray_formula("(R + G").is_err());
    assert!(parse_gray_formula("R G").is_err());
    assert!(parse_gray_formula("X").is_err());
    assert!(parse_gray_formula("R - G").is_err()); // no subtraction operator
}

#[test]
fn test_image_to_gray_from_json() {
    let value = serde_json::json!({"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8});
    let to_gray = ImageToGray::from_json(&value).unwrap();
    assert_eq!(to_gray.formula(), "0.299*R + 0.587*G + 0.114*B");
    assert_eq!(to_gray.quantize_bits(), Some(8));
    assert_eq!(to_gray.to_json(), value);
    assert_eq!(
        to_gray.gray_formula(),
        &parse_gray_formula("0.299*R + 0.587*G + 0.114*B").unwrap()
    );

    let plain = ImageToGray::from_json(&serde_json::json!({"formula": "G"})).unwrap();
    assert_eq!(plain.quantize_bits(), None);
    assert_eq!(plain.to_json(), serde_json::json!({"formula": "G"}));
    assert_eq!(plain, ImageToGray::from_gray_formula("G").unwrap());
    // The bound values of quantize_bits are accepted.
    for b in [1, 16] {
        let v = serde_json::json!({"formula": "G", "quantize_bits": b});
        assert_eq!(
            ImageToGray::from_json(&v).unwrap().quantize_bits(),
            Some(b as u8)
        );
    }
}

#[test]
fn test_image_to_gray_refusals() {
    use serde_json::json;
    let refused = |v: serde_json::Value| ImageToGray::from_json(&v).unwrap_err();
    assert_eq!(refused(json!("G")), ImageToGrayError::NotAnObject);
    assert_eq!(refused(json!(null)), ImageToGrayError::NotAnObject);
    assert_eq!(refused(json!({})), ImageToGrayError::MissingFormula);
    assert_eq!(
        refused(json!({"quantize_bits": 8})),
        ImageToGrayError::MissingFormula
    );
    assert_eq!(
        refused(json!({"formula": 3})),
        ImageToGrayError::FormulaNotAString
    );
    assert!(matches!(
        refused(json!({"formula": "R - G"})),
        ImageToGrayError::Formula { .. }
    ));
    for bad in [
        json!(0),
        json!(17),
        json!(-1),
        json!(8.5),
        json!(8.0),
        json!("8"),
        json!(null),
        json!(true),
        json!(300),
    ] {
        let err = refused(json!({"formula": "G", "quantize_bits": bad}));
        assert_eq!(
            err,
            ImageToGrayError::QuantizeBits(bad.to_string()),
            "{bad}"
        );
    }
    assert_eq!(
        refused(json!({"formula": "G", "gamma": 2.2})),
        ImageToGrayError::UnknownKey("gamma".into())
    );
    assert_eq!(
        ImageToGray::new("G", Some(0)).unwrap_err(),
        ImageToGrayError::QuantizeBits("0".into())
    );
    assert_eq!(
        ImageToGray::new("G", Some(17)).unwrap_err(),
        ImageToGrayError::QuantizeBits("17".into())
    );
    // Each message names what is wrong.
    let message = refused(json!({"formula": "R +"})).to_string();
    assert!(
        message.contains("\"R +\"") && message.contains("does not parse"),
        "{message}"
    );
    let message = refused(json!({"formula": "G", "quantize_bits": 17})).to_string();
    assert!(
        message.contains("1 to 16") && message.contains("17"),
        "{message}"
    );
}

#[test]
fn test_overlong_formulas_are_refused_not_overflowed() {
    // A sum of n terms nests n deep, and so does a run of parentheses; both
    // are refused by the token bound before anything recurses.
    let sum = vec!["R"; 5000].join("+");
    assert!(parse_gray_formula(&sum).is_err());
    let nested = format!("{}R{}", "(".repeat(100_000), ")".repeat(100_000));
    let err = parse_gray_formula(&nested).unwrap_err();
    assert!(err.message.contains("tokens"), "{err}");
    // A formula at the bound still parses.
    let at_bound = vec!["R"; MAX_FORMULA_TOKENS.div_ceil(2)].join("+");
    assert!(parse_gray_formula(&at_bound).is_ok());
}

#[test]
fn test_quantize_rounds_half_up_and_clamps() {
    // With 1 bit the only levels are 0 and 1: an exact half rounds up (half
    // to even would give 0), and the f32 just below it rounds down, which
    // `floor(v + 0.5)` would get wrong.
    assert_eq!(quantize(0.5, 1), 1.0);
    assert_eq!(quantize(0.499_999_97, 1), 0.0);
    // 2 bits: levels of 1/3; 0.5 · 3 = 1.5 rounds up to 2/3.
    assert_eq!(quantize(0.5, 2), 2.0 / 3.0);
    assert_eq!(quantize(0.166_666_6, 2), 0.0);
    // 8 bits: 127.5 rounds up to 128.
    assert_eq!(quantize(0.5, 8), 128.0 / 255.0);
    assert_eq!(quantize(100.0 / 255.0, 8), 100.0 / 255.0);
    // 16 bits.
    assert_eq!(quantize(0.5, 16), 32768.0 / 65535.0);
    // Clamping to [0, 1], infinities included; a NaN stays NaN.
    for bits in [1, 8, 16] {
        assert_eq!(quantize(-0.3, bits), 0.0);
        assert_eq!(quantize(1.7, bits), 1.0);
        assert_eq!(quantize(f32::INFINITY, bits), 1.0);
        assert_eq!(quantize(f32::NEG_INFINITY, bits), 0.0);
        assert_eq!(quantize(1.0, bits), 1.0);
        assert!(quantize(f32::NAN, bits).is_nan());
    }
    // ImageToGray applies it after the formula, and not without quantize_bits.
    let to_gray = ImageToGray::new("2 * G", Some(8)).unwrap();
    assert_eq!(to_gray.gray(0.0, 0.75, 0.0), 1.0);
    assert_eq!(to_gray.gray(0.0, 0.1, 0.0), 51.0 / 255.0);
    let plain = ImageToGray::from_gray_formula("2 * G").unwrap();
    assert_eq!(plain.gray(0.0, 0.75, 0.0), 1.5);
}

#[test]
fn test_gray_row_equals_gray_per_pixel() {
    // Every channel value an 8-bit image can hold, in three different orders,
    // as one row of 256 pixels.
    let rgb: Vec<u8> = (0..256usize)
        .flat_map(|v| [v as u8, ((v * 7) % 256) as u8, ((v * 13 + 5) % 256) as u8])
        .collect();
    let mut scratch = GrayRowScratch::default();
    for (formula, bits) in [
        (DEFAULT_GRAY_FORMULA, None),
        (DEFAULT_GRAY_FORMULA, Some(8)),
        ("0.2126*R**2.2 + 0.7152*G**2.2 + 0.0722*B**2.2", None),
        ("(R + G) * 0.5 ** (B + 1)", Some(3)),
        ("R ** -1 + G", None),
        ("-1 ** 0.5", Some(8)),
    ] {
        let to_gray = ImageToGray::new(formula, bits).unwrap();
        let mut out = vec![0f32; 256];
        to_gray.gray_row(&rgb, &mut out, &mut scratch);
        for (i, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
            let unit = |v: u8| f32::from(v) / 255.0;
            let one = to_gray.gray(unit(p[0]), unit(p[1]), unit(p[2]));
            assert!(
                out[i].to_bits() == one.to_bits() || (out[i].is_nan() && one.is_nan()),
                "{formula} at {i}: {} vs {one}",
                out[i]
            );
        }
    }
}

#[test]
fn test_gray_from_rgb_is_gray_row_whatever_the_thread_count() {
    // An image of more than one band of rows, its rows made one at a time by
    // gray_row, and by gray_from_rgb on the pool and on one thread.
    let (w, h) = (37usize, 3 * GRAY_BAND_ROWS + 5);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7919 % 251) as u8).collect();
    let to_gray = ImageToGray::default();
    let mut scratch = GrayRowScratch::default();
    let mut rows = vec![0f32; w * h];
    for (y, row) in rows.chunks_exact_mut(w).enumerate() {
        to_gray.gray_row(&rgb[y * w * 3..(y + 1) * w * 3], row, &mut scratch);
    }
    let pool = gray_from_rgb(w as u32, h as u32, &rgb, &to_gray);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| gray_from_rgb(w as u32, h as u32, &rgb, &to_gray));
    assert_eq!(pool.data(), &rows[..]);
    assert_eq!(one.data(), &rows[..]);
    // An empty image is empty, not a panic.
    assert!(gray_from_rgb(0, 4, &[], &to_gray).data().is_empty());
}
