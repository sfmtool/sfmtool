// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.camrig` image-pattern grammar.
//!
//! A sensor image pattern is a relative forward-slash path built from literal
//! text plus three kinds of special token (see the format spec, *How
//! `.camrig` files fit into workspaces*):
//!
//! * a **frame field** — `%d` or `%0Nd`, at most one per pattern — matches a
//!   run of digits and assigns the frame index;
//! * `*` — matches zero or more characters within a single path segment;
//! * `**` — occupying a whole path segment, matches zero or more whole
//!   segments, separators included.
//!
//! `%%` is an escaped literal `%`. This module is the single authority on
//! that grammar: the format's own [`CamRigData::validate`] and — through the
//! PyO3 bindings — the Python workspace tooling both go through it, so the
//! rule is implemented exactly once.
//!
//! [`CamRigData::validate`]: crate::CamRigData::validate

/// If a frame field (`%`, an optional `0`, optional digits, then `d`) starts
/// at byte `i`, return its byte length; otherwise `None`.
///
/// `%%` is the escaped literal percent and is deliberately not a frame field.
fn frame_field_at(bytes: &[u8], i: usize) -> Option<usize> {
    if bytes.get(i) != Some(&b'%') || bytes.get(i + 1) == Some(&b'%') {
        return None;
    }
    let mut j = i + 1;
    if bytes.get(j) == Some(&b'0') {
        j += 1;
    }
    while bytes.get(j).is_some_and(u8::is_ascii_digit) {
        j += 1;
    }
    (bytes.get(j) == Some(&b'd')).then_some(j + 1 - i)
}

/// Count the frame fields (`%d` / `%0Nd`) in an image pattern.
///
/// `%%` is an escaped literal `%`, not a frame field; glob wildcards never
/// form one.
pub fn count_frame_fields(pattern: &str) -> usize {
    let bytes = pattern.as_bytes();
    let mut i = 0;
    let mut count = 0;
    while i < bytes.len() {
        // `%%` is an escaped percent; skip both bytes so the second `%` is
        // not re-examined as the start of a field.
        if bytes[i] == b'%' && bytes.get(i + 1) == Some(&b'%') {
            i += 2;
        } else if let Some(len) = frame_field_at(bytes, i) {
            count += 1;
            i += len;
        } else {
            i += 1;
        }
    }
    count
}

/// Convert an image pattern to a loose glob for filesystem enumeration.
///
/// Each frame field becomes `*` and `%%` becomes a literal `%`; the glob
/// wildcards `*` / `**` are already glob syntax and pass through untouched.
/// The result deliberately over-matches (a frame field is digits-only, a `*`
/// is not) — [`pattern_matches`] is the strict filter for the glob hits.
pub fn pattern_to_glob(pattern: &str) -> String {
    let bytes = pattern.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && bytes.get(i + 1) == Some(&b'%') {
            out.push(b'%');
            i += 2;
        } else if let Some(len) = frame_field_at(bytes, i) {
            out.push(b'*');
            i += len;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    // Every byte written is either copied verbatim from `pattern` or an ASCII
    // `%` / `*`, so the result is still valid UTF-8.
    String::from_utf8(out).expect("pattern_to_glob produced invalid UTF-8")
}

/// Check that `pattern` is a structurally valid `.camrig` image pattern.
///
/// Enforces the grammar from the format spec independent of sensor count: a
/// non-empty relative forward-slash path, no `..` component, every `**` a
/// whole path segment, and at most one frame field. The multi-sensor rule
/// (every pattern needs a frame field) depends on the rig and stays in
/// [`CamRigData::validate`](crate::CamRigData::validate).
///
/// Returns the human-readable reason on failure; callers wrap it in the
/// error type they need.
pub fn validate_pattern(pattern: &str) -> Result<(), String> {
    if pattern.is_empty() {
        return Err("image pattern is empty".to_string());
    }
    let bytes = pattern.as_bytes();
    if bytes[0] == b'/' || (bytes.len() >= 2 && bytes[1] == b':') {
        return Err(format!(
            "image pattern '{pattern}' must be relative, not absolute"
        ));
    }
    for segment in pattern.split('/') {
        if segment == ".." {
            return Err(format!(
                "image pattern '{pattern}' must not contain a '..' component"
            ));
        }
        if segment.contains("**") && segment != "**" {
            return Err(format!(
                "image pattern '{pattern}' has a '**' that is not a whole path \
                 segment; '**' must stand alone between '/' separators"
            ));
        }
    }
    let fields = count_frame_fields(pattern);
    if fields > 1 {
        return Err(format!(
            "image pattern '{pattern}' has {fields} frame fields (%d / %0Nd); \
             a pattern may contain at most one"
        ));
    }
    Ok(())
}

/// One token of a compiled image pattern.
enum Tok {
    /// A literal byte that must appear verbatim.
    Lit(u8),
    /// A frame field: one or more digits.
    Frame,
    /// `*`: zero or more non-`/` characters.
    Star,
    /// `**/`: zero or more whole `segment/` runs.
    SegStar,
    /// A trailing `**`: one or more characters, separators included.
    TrailStar,
}

/// Tokenize an image pattern. A `**` that is not a whole path segment (which
/// [`validate_pattern`] rejects) degrades to a plain `*` so matching a
/// spec-invalid pattern never panics.
fn tokenize(pattern: &str) -> Vec<Tok> {
    let bytes = pattern.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && bytes.get(i + 1) == Some(&b'%') {
            toks.push(Tok::Lit(b'%'));
            i += 2;
            continue;
        }
        if let Some(len) = frame_field_at(bytes, i) {
            toks.push(Tok::Frame);
            i += len;
            continue;
        }
        if bytes[i] == b'*' {
            if bytes.get(i + 1) == Some(&b'*') {
                let seg_start = i == 0 || bytes[i - 1] == b'/';
                if seg_start && bytes.get(i + 2) == Some(&b'/') {
                    toks.push(Tok::SegStar);
                    i += 3;
                    continue;
                }
                if seg_start && i + 2 == bytes.len() {
                    toks.push(Tok::TrailStar);
                    i += 2;
                    continue;
                }
                // A `**` that is not a whole segment is spec-invalid; fall
                // through and emit one `*`, leaving the other for the next
                // iteration.
            }
            toks.push(Tok::Star);
            i += 1;
            continue;
        }
        toks.push(Tok::Lit(bytes[i]));
        i += 1;
    }
    toks
}

fn byte_eq(a: u8, b: u8, case_insensitive: bool) -> bool {
    if case_insensitive {
        a.eq_ignore_ascii_case(&b)
    } else {
        a == b
    }
}

fn matches_tokens(toks: &[Tok], path: &[u8], ci: bool) -> bool {
    let Some((tok, rest)) = toks.split_first() else {
        return path.is_empty();
    };
    match tok {
        Tok::Lit(c) => {
            !path.is_empty() && byte_eq(path[0], *c, ci) && matches_tokens(rest, &path[1..], ci)
        }
        Tok::Frame => {
            let digits = path.iter().take_while(|b| b.is_ascii_digit()).count();
            (1..=digits).any(|n| matches_tokens(rest, &path[n..], ci))
        }
        Tok::Star => {
            let seg = path.iter().take_while(|&&b| b != b'/').count();
            (0..=seg).any(|n| matches_tokens(rest, &path[n..], ci))
        }
        Tok::SegStar => {
            // Zero segments, or one whole `segment/` then `**/` again.
            if matches_tokens(rest, path, ci) {
                return true;
            }
            match path.iter().position(|&b| b == b'/') {
                Some(slash) if slash > 0 => matches_tokens(toks, &path[slash + 1..], ci),
                _ => false,
            }
        }
        Tok::TrailStar => !path.is_empty(),
    }
}

/// Whether `relative_path` (a forward-slash relative path) matches `pattern`.
///
/// `case_insensitive` should mirror the filesystem the loose glob ran on:
/// case-insensitive filesystems (Windows) glob case-insensitively, so the
/// strict check must too, or it would reject a hit the glob accepted. Case
/// folding is ASCII-only.
pub fn pattern_matches(pattern: &str, relative_path: &str, case_insensitive: bool) -> bool {
    matches_tokens(
        &tokenize(pattern),
        relative_path.as_bytes(),
        case_insensitive,
    )
}

/// Match `toks` against `path`; on success return the digit slice consumed by
/// the (at most one) frame field — `None` when the pattern has no frame field.
/// A mismatch returns the outer `None`.
///
/// The frame field is greedy: it consumes the longest digit run for which the
/// remainder of the pattern still matches, the printf `%d` convention. A
/// well-formed pattern has at most one frame field, so once a `Frame` token is
/// consumed the remaining tokens carry no further capture.
fn capture_tokens<'p>(toks: &[Tok], path: &'p [u8], ci: bool) -> Option<Option<&'p [u8]>> {
    let Some((tok, rest)) = toks.split_first() else {
        return path.is_empty().then_some(None);
    };
    match tok {
        Tok::Lit(c) => (!path.is_empty() && byte_eq(path[0], *c, ci))
            .then(|| capture_tokens(rest, &path[1..], ci))
            .flatten(),
        Tok::Frame => {
            let digits = path.iter().take_while(|b| b.is_ascii_digit()).count();
            (1..=digits)
                .rev()
                .find(|&n| capture_tokens(rest, &path[n..], ci).is_some())
                .map(|n| Some(&path[..n]))
        }
        Tok::Star => {
            let seg = path.iter().take_while(|&&b| b != b'/').count();
            (0..=seg).find_map(|n| capture_tokens(rest, &path[n..], ci))
        }
        Tok::SegStar => {
            if let Some(cap) = capture_tokens(rest, path, ci) {
                return Some(cap);
            }
            match path.iter().position(|&b| b == b'/') {
                Some(slash) if slash > 0 => capture_tokens(toks, &path[slash + 1..], ci),
                _ => None,
            }
        }
        Tok::TrailStar => (!path.is_empty()).then_some(None),
    }
}

/// The frame index `pattern`'s frame field captures from `relative_path`.
///
/// Returns `None` when the pattern has no frame field, when `relative_path`
/// does not match `pattern`, or when the captured digit run does not fit a
/// `u64`. `case_insensitive` mirrors [`pattern_matches`]. The grammar — what
/// counts as a frame field — is the one this module owns, so frame-index
/// extraction stays consistent with pattern matching and validation.
pub fn pattern_frame_index(
    pattern: &str,
    relative_path: &str,
    case_insensitive: bool,
) -> Option<u64> {
    let toks = tokenize(pattern);
    let digits = capture_tokens(&toks, relative_path.as_bytes(), case_insensitive)??;
    // `digits` is an all-ASCII-digit slice produced by the `Frame` token.
    std::str::from_utf8(digits).ok()?.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
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
}
