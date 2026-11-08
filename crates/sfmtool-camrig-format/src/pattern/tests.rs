// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn count_frame_fields_distinguishes_escape_and_field() {
    assert_eq!(count_frame_fields("cam_%04d.jpg"), 1);
    assert_eq!(count_frame_fields("cam_%d.jpg"), 1);
    assert_eq!(count_frame_fields("cam_%d_%04d.jpg"), 2);
    assert_eq!(count_frame_fields("f%%.jpg"), 0);
    assert_eq!(count_frame_fields("%%d.jpg"), 0); // escaped `%` then literal `d`
    assert_eq!(count_frame_fields("*.jpg"), 0);
}

#[test]
fn pattern_to_glob_widens_frame_fields_only() {
    assert_eq!(pattern_to_glob("cam_%04d.jpg"), "cam_*.jpg");
    assert_eq!(pattern_to_glob("imgs/**/*.jpg"), "imgs/**/*.jpg");
    assert_eq!(pattern_to_glob("f%%.jpg"), "f%.jpg");
}

#[test]
fn frame_field_matches_digits_only() {
    let m = |p: &str| pattern_matches("cam_%04d.jpg", p, false);
    assert!(m("cam_0007.jpg"));
    assert!(!m("cam_x.jpg"));
    assert!(!m("cam_.jpg"));
    assert!(!m("cam_007a.jpg"));
}

#[test]
fn zero_padded_field_matches_the_number_as_printf_writes_it() {
    let m = |p: &str| pattern_matches("shot1_%04d.jpg", p, false);
    assert!(m("shot1_0007.jpg"));
    assert!(m("shot1_0000.jpg")); // 0 padded to the width
    assert!(m("shot1_9999.jpg"));
    // A number wider than the field is written whole, as ffmpeg writes 12345.
    assert!(m("shot1_12345.jpg"));
    // Under-padded and over-padded numbers are not what %04d writes.
    assert!(!m("shot1_007.jpg"));
    assert!(!m("shot1_7.jpg"));
    assert!(!m("shot1_00007.jpg"));
    assert!(!m("shot1_012345.jpg"));
}

#[test]
fn unpadded_field_matches_a_number_without_leading_zeros() {
    let m = |p: &str| pattern_matches("cam_%d.jpg", p, false);
    assert!(m("cam_7.jpg"));
    assert!(m("cam_0.jpg"));
    assert!(m("cam_120.jpg"));
    assert!(!m("cam_07.jpg"));
    assert!(!m("cam_00.jpg"));
    // `%0d` and `%01d` write what `%d` writes.
    for pattern in ["cam_%0d.jpg", "cam_%01d.jpg"] {
        assert!(pattern_matches(pattern, "cam_7.jpg", false));
        assert!(pattern_matches(pattern, "cam_0.jpg", false));
        assert!(pattern_matches(pattern, "cam_120.jpg", false));
        assert!(!pattern_matches(pattern, "cam_07.jpg", false));
    }
}

#[test]
fn width_without_the_zero_flag_pads_with_spaces() {
    let m = |p: &str| pattern_matches("cam_%3d.jpg", p, false);
    assert!(m("cam_  7.jpg"));
    assert!(m("cam_ 42.jpg"));
    assert!(m("cam_123.jpg"));
    assert!(m("cam_1234.jpg"));
    assert!(!m("cam_7.jpg"));
    assert!(!m("cam_007.jpg"));
    assert!(!m("cam_   7.jpg"));
    assert!(!m("cam_ 07.jpg"));
    assert_eq!(
        pattern_frame_index("cam_%3d.jpg", "cam_  7.jpg", false),
        Some(7)
    );
    // A wildcard before the field leaves the field all the spaces it needs.
    assert!(pattern_matches("*%3d.jpg", "x  7.jpg", false));
}

#[test]
fn any_padding_reads_the_frame_field_as_any_run_of_digits() {
    let m = |p: &str| pattern_matches_any_padding("shot1_%04d.jpg", p, false);
    assert!(m("shot1_0007.jpg"));
    assert!(m("shot1_007.jpg"));
    assert!(m("shot1_00007.jpg"));
    assert!(!m("shot1_x.jpg"));
    assert!(!m("shot2_0007.jpg"));
    assert!(pattern_matches_any_padding(
        "cam_%d.jpg",
        "cam_007.jpg",
        false
    ));
    assert_eq!(
        pattern_frame_index_any_padding("shot1_%04d.jpg", "shot1_007.jpg", false),
        Some(7)
    );
    assert_eq!(
        pattern_frame_index("shot1_%04d.jpg", "shot1_007.jpg", false),
        None
    );
}

#[test]
fn star_stays_within_a_segment() {
    assert!(pattern_matches("imgs/*.jpg", "imgs/a.jpg", false));
    assert!(!pattern_matches("imgs/*.jpg", "imgs/sub/a.jpg", false));
}

#[test]
fn globstar_spans_whole_segments() {
    let m = |p: &str| pattern_matches("imgs/**/*.jpg", p, false);
    assert!(m("imgs/a.jpg")); // `**` matches zero segments
    assert!(m("imgs/x/a.jpg"));
    assert!(m("imgs/x/y/a.jpg"));
    assert!(!m("other/a.jpg"));
}

#[test]
fn trailing_globstar_needs_at_least_one_segment() {
    assert!(pattern_matches("a/**", "a/b/c.jpg", false));
    assert!(pattern_matches("a/**", "a/b.jpg", false));
    assert!(!pattern_matches("a/**", "a/", false));
}

#[test]
fn escaped_percent_is_literal() {
    assert!(pattern_matches("f%%.jpg", "f%.jpg", false));
    assert!(!pattern_matches("f%%.jpg", "f%%.jpg", false));
}

#[test]
fn case_insensitive_matching_is_opt_in() {
    assert!(!pattern_matches("cam_%d.JPG", "cam_7.jpg", false));
    assert!(pattern_matches("cam_%d.JPG", "cam_7.jpg", true));
}

#[test]
fn pattern_frame_index_captures_the_frame_field_integer() {
    let idx = |p: &str, path: &str| pattern_frame_index(p, path, false);
    assert_eq!(idx("cam_%04d.jpg", "cam_0007.jpg"), Some(7));
    assert_eq!(idx("cam_%d.jpg", "cam_42.jpg"), Some(42));
    assert_eq!(idx("cam_%04d.jpg", "cam_10000.jpg"), Some(10000));
    assert_eq!(idx("cam_%04d.jpg", "cam_0000.jpg"), Some(0));
    assert_eq!(idx("cam_%d.jpg", "cam_0.jpg"), Some(0));
    // A literal digit after the field: the field takes the longest number
    // printf could have written before it.
    assert_eq!(idx("cam_%d1.jpg", "cam_1231.jpg"), Some(123));
    assert_eq!(idx("cam_%04d1.jpg", "cam_000071.jpg"), None);
    assert_eq!(idx("cam_%04d1.jpg", "cam_00071.jpg"), Some(7));
    // `%%` before the field is a literal percent.
    assert_eq!(idx("f%%_%03d.jpg", "f%_042.jpg"), Some(42));
    assert_eq!(idx("f%%_%03d.jpg", "f%_42.jpg"), None);
    // Frame field behind glob wildcards.
    assert_eq!(
        idx("left/**/frame_%06d.jpg", "left/a/b/frame_000123.jpg"),
        Some(123)
    );
    assert_eq!(idx("*/frame_%d.png", "sensor/frame_9.png"), Some(9));
}

#[test]
fn pattern_frame_index_is_none_without_a_field_or_a_match() {
    // No frame field at all.
    assert_eq!(pattern_frame_index("*.jpg", "a.jpg", false), None);
    // Frame field present, but the path does not match the pattern.
    assert_eq!(pattern_frame_index("cam_%d.jpg", "cam_x.jpg", false), None);
    assert_eq!(
        pattern_frame_index("cam_%d.jpg", "other_7.jpg", false),
        None
    );
}

#[test]
fn validate_pattern_accepts_globs_and_one_frame_field() {
    assert!(validate_pattern("*.jpg").is_ok());
    assert!(validate_pattern("imgs/**/cam_%04d.png").is_ok());
    assert!(validate_pattern("a/**").is_ok());
}

#[test]
fn validate_pattern_rejects_malformed_patterns() {
    assert!(validate_pattern("").unwrap_err().contains("empty"));
    assert!(validate_pattern("/abs/x.jpg")
        .unwrap_err()
        .contains("must be relative"));
    assert!(validate_pattern("C:/x.jpg")
        .unwrap_err()
        .contains("must be relative"));
    assert!(validate_pattern("../x.jpg").unwrap_err().contains(".."));
    assert!(validate_pattern("cam_%d_%04d.jpg")
        .unwrap_err()
        .contains("at most one"));
    assert!(validate_pattern("a**b/x.jpg")
        .unwrap_err()
        .contains("whole path segment"));
}

#[test]
fn pattern_literal_is_the_text_of_a_pattern_without_wildcards_or_frame_fields() {
    assert_eq!(
        pattern_literal("photos/DSC_0042.JPG").as_deref(),
        Some("photos/DSC_0042.JPG")
    );
    assert_eq!(pattern_literal("f%%.jpg").as_deref(), Some("f%.jpg"));
    assert_eq!(pattern_literal("%%d.jpg").as_deref(), Some("%d.jpg"));
    assert_eq!(pattern_literal("a%b.jpg").as_deref(), Some("a%b.jpg"));
    assert_eq!(pattern_literal("cam_%04d.jpg"), None);
    assert_eq!(pattern_literal("cam_%d.jpg"), None);
    assert_eq!(pattern_literal("imgs/*.jpg"), None);
    assert_eq!(pattern_literal("imgs/**/a.jpg"), None);
    for literal in ["photos/DSC_0042.JPG", "f%%.jpg", "%%d.jpg", "a%b.jpg"] {
        let text = pattern_literal(literal).unwrap();
        assert!(pattern_matches(literal, &text, false));
    }
}
