// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Measured in characters, which is what makes these tests about the cut
//! rather than about a font.

use super::{middle, ELLIPSIS};

/// One unit per character, the measurement a fixed-width column would give.
fn chars(s: &str) -> f32 {
    s.chars().count() as f32
}

/// A text that already fits comes back untouched.
#[test]
fn a_text_that_fits_is_not_cut() {
    let text = "images/seattle_backyard_01.jpg";
    assert_eq!(middle(text, chars(text), chars), text);
    assert_eq!(middle(text, chars(text) + 20.0, chars), text);
}

/// A text that does not fit keeps both ends, and what comes back fits.
#[test]
fn a_long_text_keeps_both_ends_and_fits() {
    let text = "images/seattle_backyard_01.jpg";
    for room in [8.0, 12.0, 20.0, 25.0] {
        let cut = middle(text, room, chars);
        assert!(cut.contains(ELLIPSIS), "{room}: {cut:?} has no mark");
        assert!(
            chars(&cut) <= room,
            "{room}: {cut:?} is {} wide",
            chars(&cut),
        );
        assert!(cut.starts_with('i'), "{room}: the head went: {cut:?}");
        assert!(cut.ends_with('g'), "{room}: the tail went: {cut:?}");
    }
}

/// The directory and the file name both survive at a width a column of this
/// panel actually has, which is the whole point of cutting the middle.
#[test]
fn both_ends_of_a_path_survive() {
    let cut = middle("images/seattle_backyard_01.jpg", 20.0, chars);
    assert_eq!(cut, "images/sea\u{2026}rd_01.jpg", "{cut:?}");
}

/// The two ends stay within a character of each other.
#[test]
fn the_two_ends_stay_even() {
    let text = "20260628-01-solve-seattle_backyard_1-26-embedded";
    for room in [6.0, 10.0, 17.0, 24.0, 40.0] {
        let cut = middle(text, room, chars);
        let (head, tail) = cut.split_once(ELLIPSIS).expect("a mark");
        let diff = head.chars().count().abs_diff(tail.chars().count());
        assert!(diff <= 1, "{room}: {cut:?} splits {head:?} / {tail:?}");
    }
}

/// A column with no room for the mark draws nothing, rather than a lone dot
/// that reads as content.
#[test]
fn a_column_with_no_room_draws_nothing() {
    assert_eq!(middle("images/foo.jpg", 0.5, chars), "");
}

/// The result always says something was cut: it is never the whole text with
/// a mark stuck in the middle of it.
#[test]
fn a_cut_text_is_always_shorter_than_the_whole() {
    let text = "abcdef";
    for room in 1..=5 {
        let cut = middle(text, room as f32, chars);
        assert!(
            cut.chars().filter(|c| *c != ELLIPSIS).count() < text.chars().count(),
            "{room}: {cut:?} kept everything",
        );
    }
}

/// Multi-byte characters are cut on character boundaries, not byte ones.
#[test]
fn a_cut_lands_on_a_character_boundary() {
    let text = "café_naïve_résumé_αβγδε.jpg";
    let cut = middle(text, 12.0, chars);
    assert!(cut.contains(ELLIPSIS), "{cut:?}");
    assert!(chars(&cut) <= 12.0, "{cut:?}");
}
