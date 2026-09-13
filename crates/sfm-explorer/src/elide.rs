// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shortening a text to a width it has to fit in.
//!
//! Taken out of the **middle**, because both ends of the strings this window
//! shows carry meaning and the middle is the part a reader can infer. A
//! reconstruction is called `20260628-01-solve-seattle_backyard_1-26-embedded`
//! and an observation names `images/seattle_backyard_01.jpg`: cut from the
//! right, each becomes the date or the directory it shares with everything
//! else on screen; cut from the left, each loses the family it belongs to.
//! Keeping both ends costs one glyph and spends it on the part that was
//! predictable.
//!
//! The measurement is the caller's, rather than this module reaching for a
//! font. Text is shaped before it is drawn, so the width of a string is not
//! the sum of its glyphs' widths and only the thing that will lay it out can
//! say how wide it is. A caller hands in whatever it draws with, and a test
//! hands in one unit per character.

/// The mark that stands for what was cut.
pub(crate) const ELLIPSIS: char = '\u{2026}';

/// `text` shortened to fit `width`, with the cut taken out of the middle.
///
/// `measure` is how wide a string will be when drawn. Returns `text` unchanged
/// when it already fits, so the common case costs one measurement.
pub(crate) fn middle(text: &str, width: f32, measure: impl Fn(&str) -> f32) -> String {
    if measure(text) <= width {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mark = ELLIPSIS.to_string();
    if measure(&mark) > width {
        // Narrower than the mark itself. A caller drawing into a column this
        // narrow has a layout problem rather than a text problem, and an empty
        // cell says so where a lone dot would look like content.
        return String::new();
    }
    // `kept` characters survive, split evenly with the odd one going to the
    // head, so the two ends stay within a character of each other and the
    // result does not lurch as the column changes width by a pixel.
    let candidate = |kept: usize| -> String {
        let head = kept.div_ceil(2);
        let tail = kept - head;
        let mut out: String = chars[..head].iter().collect();
        out.push(ELLIPSIS);
        out.extend(&chars[chars.len() - tail..]);
        out
    };
    // The most that could survive still leaves one character for the mark to
    // stand for, so a result is always shorter than what it replaced.
    let most = chars.len().saturating_sub(1);
    let (mut low, mut high) = (0usize, most);
    while low < high {
        let mid = (low + high).div_ceil(2);
        if measure(&candidate(mid)) <= width {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    // Shaping means width is not strictly monotonic in the count, so the
    // search lands near the answer rather than on it. Step down to one that
    // actually fits.
    let mut kept = low;
    loop {
        let out = candidate(kept);
        if kept == 0 || measure(&out) <= width {
            return out;
        }
        kept -= 1;
    }
}

#[cfg(test)]
mod tests;
