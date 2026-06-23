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
    // Pure white -> 0.2126 + 0.7152 + 0.0722 = 1.0
    assert!((f.eval(1.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
    // Pure red -> 0.2126
    assert!((f.eval(1.0, 0.0, 0.0) - 0.2126).abs() < 1e-12);
    // A sample value
    let v = f.eval(0.5, 0.25, 0.75);
    let expected = 0.2126 * 0.5 + 0.7152 * 0.25 + 0.0722 * 0.75;
    assert!((v - expected).abs() < 1e-12);
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
    let f = parse_gray_formula(DEFAULT_GRAY_FORMULA).unwrap();
    let img = gray_from_rgb(2, 1, &rgb, &f);
    assert_eq!(img.width(), 2);
    assert_eq!(img.height(), 1);
    assert!((img.get_pixel(0, 0) - 1.0).abs() < 1e-6);
    assert!((img.get_pixel(1, 0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_gray_from_rgb_no_upper_clamp() {
    // A formula that exceeds 1.0 must not be clamped.
    let f = parse_gray_formula("2 * R").unwrap();
    let rgb = vec![255u8, 0, 0];
    let img = gray_from_rgb(1, 1, &rgb, &f);
    assert!((img.get_pixel(0, 0) - 2.0).abs() < 1e-6);
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
