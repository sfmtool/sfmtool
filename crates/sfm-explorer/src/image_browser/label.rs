// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The text drawn under each thumbnail: the image's name with the start every
//! name in the reconstruction shares replaced by `…`.
//!
//! The names of one capture differ only near their ends, in the frame number
//! and, for a rig, the sensor's directory, so the shared start says nothing
//! that tells two thumbnails apart. The end is kept as it is, file extension
//! included, so the label still reads as a file name: `…04.jpg` for
//! `seoul_bull_sculpture_04.jpg`, `…left/frame_01.jpg` for
//! `fisheye_left/frame_01.jpg` beside `fisheye_right/…`.
//!
//! A label still too wide for its thumbnail is shortened in the middle by
//! [`crate::elide::middle`], which keeps its start and its end.

use crate::elide::{self, ELLIPSIS};

/// How many leading characters of every name a label leaves out.
///
/// The longest start every name shares, backed up to the last place a reader
/// would split a name: just after a separator (`/`, `\`, `_`, `-`, `.` or a
/// space), or where a run of digits begins. So `frame_1.jpg` and
/// `frame_12.jpg`, which share `frame_1`, both keep their whole number. The
/// cut always leaves at least one character of every name, and is `0`, so a
/// label is the whole name, for fewer than two names, for names that share no
/// such start, or where the shared start is a single character, which `…`
/// would replace without saving anything. Counted in characters, not bytes.
pub(super) fn shared_prefix_chars(names: &[&str]) -> usize {
    let [first, rest @ ..] = names else {
        return 0;
    };
    if rest.is_empty() {
        return 0;
    }
    let first: Vec<char> = first.chars().collect();
    let mut shared = first.len();
    let mut shortest = first.len();
    for name in rest {
        let mut length = 0;
        let mut common = 0;
        for (i, c) in name.chars().enumerate() {
            if common == i && i < shared && first[i] == c {
                common += 1;
            }
            length += 1;
        }
        shared = shared.min(common);
        shortest = shortest.min(length);
    }
    // At least one character of the shortest name stays, so every cut has a
    // character after it in every name.
    let limit = shared.min(shortest.saturating_sub(1));
    let is_separator = |c: char| matches!(c, '/' | '\\' | '_' | '-' | '.' | ' ');
    // Whether every name has a digit at `at`: below `shared` they all have
    // the first name's character, at `shared` they differ.
    let digit_in_every_name = |at: usize| {
        if at < shared {
            first[at].is_ascii_digit()
        } else {
            names
                .iter()
                .all(|name| name.chars().nth(at).is_some_and(|c| c.is_ascii_digit()))
        }
    };
    (2..=limit)
        .rev()
        .find(|&cut| {
            let before = first[cut - 1];
            is_separator(before) || (!before.is_ascii_digit() && digit_in_every_name(cut))
        })
        .unwrap_or(0)
}

/// The label for `name`: its first `prefix_chars` characters replaced by `…`
/// (none replaced when `prefix_chars` is `0`), shortened in the middle to fit
/// `width`.
///
/// `measure` is how wide a string is when drawn, as in
/// [`crate::elide::middle`]. A shortened label keeps its leading `…` and cuts
/// the rest, so it reads `…left/fr…_01.jpg` rather than losing the mark that
/// says the start was left out. A width too narrow for any of it gives an
/// empty label.
pub(super) fn label(
    name: &str,
    prefix_chars: usize,
    width: f32,
    measure: impl Fn(&str) -> f32,
) -> String {
    if prefix_chars == 0 {
        return elide::middle(name, width, measure);
    }
    let rest: String = name.chars().skip(prefix_chars).collect();
    let marked = |s: &str| format!("{ELLIPSIS}{s}");
    let cut = elide::middle(&rest, width, |s| measure(&marked(s)));
    if cut.is_empty() {
        return cut;
    }
    marked(&cut)
}

#[cfg(test)]
mod tests;
