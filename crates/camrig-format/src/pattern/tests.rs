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
    assert!(m("cam_10000.jpg")); // frame index wider than the pad
    assert!(!m("cam_x.jpg"));
    assert!(!m("cam_.jpg"));
    assert!(!m("cam_007a.jpg"));
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
