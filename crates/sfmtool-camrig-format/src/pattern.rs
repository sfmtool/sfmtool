// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.camrig` image-pattern grammar.
//!
//! A sensor image pattern is a relative forward-slash path built from literal
//! text plus three kinds of special token (see the format spec, *How
//! `.camrig` files fit into workspaces*):
//!
//! * a **frame field** — `%d` or `%0Nd`, at most one per pattern — matches a
//!   number written exactly as printf writes it with that field, and assigns
//!   the frame index;
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

/// A frame field's printf conversion: whether it pads with zeros (the `0`
/// flag) and its width (`0` when it has none).
#[derive(Clone, Copy)]
struct FrameField {
    zero_pad: bool,
    width: usize,
}

impl FrameField {
    /// Whether printf, formatting a number with this field, writes `digits`,
    /// a non-empty run of ASCII digits.
    ///
    /// printf writes a number without leading zeros (`0` itself as one
    /// digit), then, under the `0` flag, pads it on the left with zeros to the
    /// field's width. So the text is exactly the width when padding was
    /// added, and a number wider than the field is written whole and
    /// unpadded. A field without the `0` flag is read as `%d`: a width of 1
    /// never pads, and [`validate_pattern`] refuses a wider one, which printf
    /// would pad with spaces.
    fn writes(self, digits: &[u8]) -> bool {
        let len = digits.len();
        let unpadded = len == 1 || digits[0] != b'0';
        let width = if self.zero_pad { self.width.max(1) } else { 1 };
        len == width || (len > width && unpadded)
    }
}

/// If a frame field (`%`, an optional `0`, optional digits, then `d`) starts
/// at byte `i`, return its byte length and its conversion; otherwise `None`.
///
/// `%%` is the escaped literal percent and is deliberately not a frame field.
fn frame_field_at(bytes: &[u8], i: usize) -> Option<(usize, FrameField)> {
    if bytes.get(i) != Some(&b'%') || bytes.get(i + 1) == Some(&b'%') {
        return None;
    }
    let mut j = i + 1;
    let zero_pad = bytes.get(j) == Some(&b'0');
    if zero_pad {
        j += 1;
    }
    let mut width = 0usize;
    while let Some(digit) = bytes.get(j).filter(|b| b.is_ascii_digit()) {
        width = width
            .saturating_mul(10)
            .saturating_add(usize::from(digit - b'0'));
        j += 1;
    }
    (bytes.get(j) == Some(&b'd')).then_some((j + 1 - i, FrameField { zero_pad, width }))
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
        } else if let Some((len, _)) = frame_field_at(bytes, i) {
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
/// The result deliberately over-matches (a frame field matches only a number
/// as printf writes it, a `*` matches any text) — [`pattern_matches`] is the
/// strict filter for the glob hits.
pub fn pattern_to_glob(pattern: &str) -> String {
    let bytes = pattern.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && bytes.get(i + 1) == Some(&b'%') {
            out.push(b'%');
            i += 2;
        } else if let Some((len, _)) = frame_field_at(bytes, i) {
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

/// The one path `pattern` matches, when it has no wildcard (`*`, `**`) and no
/// frame field: its text with each `%%` read as a literal `%`. `None` when the
/// pattern has a wildcard or a frame field.
///
/// A caller holding a set of candidate names can look such a pattern up in
/// the set rather than test every candidate against it.
pub fn pattern_literal(pattern: &str) -> Option<String> {
    let bytes = pattern.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && bytes.get(i + 1) == Some(&b'%') {
            out.push(b'%');
            i += 2;
        } else if bytes[i] == b'*' || frame_field_at(bytes, i).is_some() {
            return None;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    // Every byte written is copied from `pattern` or is an ASCII `%`.
    Some(String::from_utf8(out).expect("pattern_literal produced invalid UTF-8"))
}

/// Check that `pattern` is a structurally valid `.camrig` image pattern.
///
/// Enforces the grammar from the format spec independent of sensor count: a
/// non-empty relative forward-slash path, no `..` component, every `**` a
/// whole path segment, at most one frame field, and no frame field with a
/// width but no `0` flag (`%4d`), which printf pads with spaces; `%1d` never
/// pads and is accepted as `%d`. The multi-sensor rule
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
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && bytes.get(i + 1) == Some(&b'%') {
            i += 2;
        } else if let Some((len, field)) = frame_field_at(bytes, i) {
            if !field.zero_pad && field.width > 1 {
                let text = &pattern[i..i + len];
                let digits = &text[1..len - 1];
                return Err(format!(
                    "image pattern '{pattern}' has the frame field '{text}', which pads \
                     a number with spaces; use '%0{digits}d' for a number zero-padded to \
                     {digits} digits or '%d' for a number without padding"
                ));
            }
            i += len;
        } else {
            i += 1;
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
    /// A frame field: a number as printf writes it with the field, or, under
    /// [`Padding::Any`], any run of one or more digits.
    Frame(FrameField),
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
        if let Some((len, field)) = frame_field_at(bytes, i) {
            toks.push(Tok::Frame(field));
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

/// Which numbers a frame field matches.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Padding {
    /// The number exactly as printf writes it with the field: the grammar.
    Exact,
    /// Any run of one or more digits, however it is padded.
    Any,
}

/// Match `toks` against `path`; on success return the digit slice consumed by
/// the (at most one) frame field — `None` when the pattern has no frame field.
/// A mismatch returns the outer `None`.
///
/// The frame field is greedy: of the texts it can match at this point, it
/// takes the longest for which the remainder of the pattern still matches,
/// the printf `%d` convention. A well-formed pattern has at most one frame
/// field, so once a `Frame` token is consumed the remaining tokens carry no
/// further capture.
fn capture_tokens<'p>(
    toks: &[Tok],
    path: &'p [u8],
    ci: bool,
    padding: Padding,
) -> Option<Option<&'p [u8]>> {
    let Some((tok, rest)) = toks.split_first() else {
        return path.is_empty().then_some(None);
    };
    match tok {
        Tok::Lit(c) => (!path.is_empty() && byte_eq(path[0], *c, ci))
            .then(|| capture_tokens(rest, &path[1..], ci, padding))
            .flatten(),
        Tok::Frame(field) => {
            let digits = path.iter().take_while(|b| b.is_ascii_digit()).count();
            (1..=digits)
                .rev()
                .filter(|&n| padding == Padding::Any || field.writes(&path[..n]))
                .find(|&n| capture_tokens(rest, &path[n..], ci, padding).is_some())
                .map(|n| Some(&path[..n]))
        }
        Tok::Star => {
            let seg = path.iter().take_while(|&&b| b != b'/').count();
            (0..=seg).find_map(|n| capture_tokens(rest, &path[n..], ci, padding))
        }
        Tok::SegStar => {
            // Zero segments, or one whole `segment/` then `**/` again.
            if let Some(cap) = capture_tokens(rest, path, ci, padding) {
                return Some(cap);
            }
            match path.iter().position(|&b| b == b'/') {
                Some(slash) if slash > 0 => capture_tokens(toks, &path[slash + 1..], ci, padding),
                _ => None,
            }
        }
        Tok::TrailStar => (!path.is_empty()).then_some(None),
    }
}

/// Whether `relative_path` (a forward-slash relative path) matches `pattern`.
///
/// A frame field matches a number only as printf writes it with that field:
/// `%04d` matches `0007` and `12345`, not `007` or `00007`, and `%d` matches
/// `7` and `0`, not `07`.
///
/// `case_insensitive` should mirror the filesystem the loose glob ran on:
/// case-insensitive filesystems (Windows) glob case-insensitively, so the
/// strict check must too, or it would reject a hit the glob accepted. Case
/// folding is ASCII-only.
pub fn pattern_matches(pattern: &str, relative_path: &str, case_insensitive: bool) -> bool {
    capture_tokens(
        &tokenize(pattern),
        relative_path.as_bytes(),
        case_insensitive,
        Padding::Exact,
    )
    .is_some()
}

/// Whether `relative_path` matches `pattern` with its frame field read as any
/// run of one or more digits, however padded: `cam_%04d.jpg` then matches
/// `cam_007.jpg` too.
///
/// This is not the grammar, which [`pattern_matches`] implements. It finds
/// the files a pattern misses only because their number is padded otherwise
/// than its frame field writes it, so a caller can name them rather than
/// leave them out without a word. For a pattern with no frame field it is
/// [`pattern_matches`].
pub fn pattern_matches_any_padding(
    pattern: &str,
    relative_path: &str,
    case_insensitive: bool,
) -> bool {
    capture_tokens(
        &tokenize(pattern),
        relative_path.as_bytes(),
        case_insensitive,
        Padding::Any,
    )
    .is_some()
}

/// The frame index `pattern`'s frame field captures from `relative_path`.
///
/// Returns `None` when the pattern has no frame field, when `relative_path`
/// does not match `pattern`, or when the captured number does not fit a
/// `u64`. `case_insensitive` mirrors [`pattern_matches`]. The grammar — what
/// counts as a frame field and which numbers it matches — is the one this
/// module owns, so frame-index extraction stays consistent with pattern
/// matching and validation.
pub fn pattern_frame_index(
    pattern: &str,
    relative_path: &str,
    case_insensitive: bool,
) -> Option<u64> {
    frame_index(pattern, relative_path, case_insensitive, Padding::Exact)
}

/// The frame index `pattern`'s frame field captures from `relative_path` when
/// the field is read as any run of digits, as [`pattern_matches_any_padding`]
/// reads it; otherwise as [`pattern_frame_index`].
pub fn pattern_frame_index_any_padding(
    pattern: &str,
    relative_path: &str,
    case_insensitive: bool,
) -> Option<u64> {
    frame_index(pattern, relative_path, case_insensitive, Padding::Any)
}

fn frame_index(
    pattern: &str,
    relative_path: &str,
    case_insensitive: bool,
    padding: Padding,
) -> Option<u64> {
    let toks = tokenize(pattern);
    let digits = capture_tokens(&toks, relative_path.as_bytes(), case_insensitive, padding)??;
    // `digits` is an all-ASCII-digit slice produced by the `Frame` token.
    std::str::from_utf8(digits).ok()?.parse::<u64>().ok()
}

#[cfg(test)]
mod tests;
