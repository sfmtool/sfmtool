// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Measured in characters where a width matters, so these tests are about the
//! names and the cut rather than about a font.

use super::{label, shared_prefix_chars};
use crate::elide::ELLIPSIS;

/// One unit per character.
fn chars(s: &str) -> f32 {
    s.chars().count() as f32
}

/// Every name's label at a width that fits any of them.
fn labels(names: &[&str]) -> Vec<String> {
    let prefix = shared_prefix_chars(names);
    names
        .iter()
        .map(|name| label(name, prefix, f32::INFINITY, chars))
        .collect()
}

fn numbered(pattern: &str, count: usize) -> Vec<String> {
    (1..=count)
        .map(|i| pattern.replace("{}", &format!("{i:02}")))
        .collect()
}

fn refs(names: &[String]) -> Vec<&str> {
    names.iter().map(String::as_str).collect()
}

#[test]
fn seoul_bull_keeps_the_number_and_the_extension() {
    let names = numbered("seoul_bull_sculpture_{}.jpg", 17);
    let labels = labels(&refs(&names));
    assert_eq!(labels[3], "\u{2026}04.jpg");
    assert_eq!(labels[16], "\u{2026}17.jpg");
}

#[test]
fn dino_drops_its_directory_with_the_shared_start() {
    let names = numbered("images/dino_dog_toy_{}.jpg", 85);
    let labels = labels(&refs(&names));
    assert_eq!(labels[52], "\u{2026}53.jpg");
}

#[test]
fn a_rig_keeps_the_sensor_that_tells_its_images_apart() {
    let mut names = numbered("fisheye_left/frame_{}.jpg", 24);
    names.extend(numbered("fisheye_right/frame_{}.jpg", 24));
    let labels = labels(&refs(&names));
    assert_eq!(labels[0], "\u{2026}left/frame_01.jpg");
    assert_eq!(labels[24], "\u{2026}right/frame_01.jpg");
}

/// Nine images share their tens digit as well, and the label keeps it.
#[test]
fn a_shared_digit_stays_with_its_number() {
    let names = numbered("seoul_bull_sculpture_{}.jpg", 9);
    assert_eq!(labels(&refs(&names))[0], "\u{2026}01.jpg");
    let names = ["frame_1.jpg", "frame_12.jpg"];
    assert_eq!(labels(&names), ["\u{2026}1.jpg", "\u{2026}12.jpg"]);
    // With no separator, the run of digits is where the cut goes.
    let names = ["frame1.jpg", "frame12.jpg"];
    assert_eq!(labels(&names), ["\u{2026}1.jpg", "\u{2026}12.jpg"]);
}

/// A shared start that ends inside a word is backed up to the word's start.
#[test]
fn a_word_is_not_split() {
    let names = ["cam_left/0001.jpg", "cam_lower/0001.jpg"];
    assert_eq!(
        labels(&names),
        ["\u{2026}left/0001.jpg", "\u{2026}lower/0001.jpg"]
    );
}

#[test]
fn a_single_image_is_labelled_with_its_whole_name() {
    assert_eq!(shared_prefix_chars(&["seoul_bull_sculpture_04.jpg"]), 0);
    assert_eq!(
        labels(&["seoul_bull_sculpture_04.jpg"]),
        ["seoul_bull_sculpture_04.jpg"]
    );
    assert_eq!(shared_prefix_chars(&[]), 0);
}

#[test]
fn names_with_no_shared_start_are_labelled_in_full() {
    let names = ["IMG_0001.jpg", "DSC_0002.jpg"];
    assert_eq!(labels(&names), names);
    // A single shared character is not worth the mark that would replace it.
    let names = ["a_1.jpg", "ab_2.jpg"];
    assert_eq!(labels(&names), names);
}

/// A name that is all shared start keeps a character, so no label is a lone
/// mark.
#[test]
fn every_label_keeps_a_character_of_its_name() {
    let names = ["shot_", "shot_", "shot_2"];
    assert_eq!(shared_prefix_chars(&names), 0);
    let names = ["shot_a", "shot_a"];
    assert_eq!(labels(&names), ["\u{2026}a", "\u{2026}a"]);
}

/// The prefix is counted in characters, so a multi-byte start is cut whole.
#[test]
fn non_ascii_names_are_cut_on_characters() {
    let names = ["서울_황소_01.jpg", "서울_황소_02.jpg"];
    assert_eq!(shared_prefix_chars(&names), 6);
    assert_eq!(labels(&names), ["\u{2026}01.jpg", "\u{2026}02.jpg"]);
    let names = ["été_a1.jpg", "été_b2.jpg"];
    assert_eq!(shared_prefix_chars(&names), 4);
    assert_eq!(labels(&names), ["\u{2026}a1.jpg", "\u{2026}b2.jpg"]);
}

/// Too wide for its thumbnail, a label keeps its mark, its start and its end,
/// and fits.
#[test]
fn a_label_too_wide_is_shortened_in_the_middle() {
    let name = "fisheye_left/frame_01.jpg";
    // `…left/frame_01.jpg` is 18 characters.
    assert_eq!(label(name, 8, 18.0, chars), "\u{2026}left/frame_01.jpg");
    let cut = label(name, 8, 16.0, chars);
    assert_eq!(cut, "\u{2026}left/fr\u{2026}_01.jpg");
    for width in [4.0, 6.0, 9.0, 12.0, 15.0] {
        let cut = label(name, 8, width, chars);
        assert!(chars(&cut) <= width, "{width}: {cut:?}");
        assert!(cut.starts_with(ELLIPSIS), "{width}: {cut:?}");
        assert!(cut.ends_with('g'), "{width}: {cut:?}");
        assert_eq!(cut.matches(ELLIPSIS).count(), 2, "{width}: {cut:?}");
    }
    // A whole name with nothing left out is shortened the same way.
    let cut = label("IMG_0001.jpg", 0, 8.0, chars);
    assert_eq!(cut, "IMG_\u{2026}jpg");
}

/// A thumbnail too narrow for the marks gets no label rather than a lone mark.
#[test]
fn a_thumbnail_too_narrow_gets_no_label() {
    assert_eq!(label("fisheye_left/frame_01.jpg", 8, 1.5, chars), "");
    assert_eq!(label("IMG_0001.jpg", 0, 0.5, chars), "");
}
