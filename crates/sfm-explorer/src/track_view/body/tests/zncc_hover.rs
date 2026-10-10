// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the *ZNCC* cell's hover: what it draws for a row whose
//! bitmap was blurred, one read unblurred, the reference's own row and a row
//! with no score, run through `Context::run_ui`.

use sfmtool_core::bench::{TrackMeasurement, Unmeasured};
use sfmtool_core::patch::self_similarity::{SelfSimilarityEllipse, SelfSimilarityEllipseUnits};

use super::super::zncc_hover::{
    blurred_bitmap_image, show_zncc_hover, zncc_hover_text, ZnccHoverText, ZnccHoverTiles,
    ZNCC_HOVER_TILE_SIDE,
};

const BITMAP: egui::TextureId = egui::TextureId::User(1);
const BLURRED: egui::TextureId = egui::TextureId::User(2);
const TILE: egui::TextureId = egui::TextureId::User(3);

fn tiles() -> ZnccHoverTiles {
    ZnccHoverTiles {
        bitmap: Some(BITMAP),
        blurred: Some(BLURRED),
        tile: Some(TILE),
    }
}

/// A scored row: plain 0.50 whole and 0.40 middle, blur-matched 0.60 and
/// 0.55, the grids at 0.1·k plain and 0.1·k + 0.05 blur-matched.
fn scored(sigma: f64, sharper: bool) -> TrackMeasurement {
    let grid = |offset: f64| {
        std::array::from_fn(|i| std::array::from_fn(|j| 0.1 * (3 * i + j) as f64 + offset))
    };
    let blurred = sigma > 0.0;
    TrackMeasurement {
        plain_zncc: Some(0.5),
        plain_zncc_middle: Some(0.4),
        plain_zncc_grid: Some(grid(0.0)),
        blur_matched_zncc: Some(if blurred { 0.6 } else { 0.5 }),
        blur_matched_zncc_middle: Some(if blurred { 0.55 } else { 0.4 }),
        blur_matched_zncc_grid: Some(grid(if blurred { 0.05 } else { 0.0 })),
        bitmap_blur_sigma: Some(sigma),
        sharper_than_bitmap: Some(sharper),
        seed_shift_px: Some(0.2),
        ..TrackMeasurement::default()
    }
}

/// A round self-similarity ellipse of 0.6 grid px, the row's own sharpness
/// reading.
fn ellipse() -> SelfSimilarityEllipseUnits {
    SelfSimilarityEllipseUnits {
        grid_px: SelfSimilarityEllipse {
            axes: [0.6, 0.6],
            axes_is_at_least: [false; 2],
            major_angle: f64::NAN,
            matrix: [[0.36, 0.0], [0.0, 0.36]],
        },
        image_px: None,
        patch: None,
    }
}

/// Draw the hover for `m` in one frame: the textures its pictures draw, in
/// order, and the text it lays out, joined.
fn drawn(m: &TrackMeasurement, is_reference: bool) -> (Vec<egui::TextureId>, String) {
    fn walk(shape: &egui::Shape, textures: &mut Vec<egui::TextureId>, text: &mut String) {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, textures, text)),
            egui::Shape::Mesh(mesh) if mesh.texture_id != egui::TextureId::default() => {
                textures.push(mesh.texture_id);
            }
            egui::Shape::Rect(rect) if rect.fill_texture_id() != egui::TextureId::default() => {
                textures.push(rect.fill_texture_id());
            }
            egui::Shape::Text(t) => {
                text.push_str(t.galley.text());
                text.push('\n');
            }
            _ => {}
        }
    }
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
        show_zncc_hover(ui, m, is_reference, tiles());
    });
    output.textures_delta.clear();
    let (mut textures, mut text) = (Vec::new(), String::new());
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut textures, &mut text);
    }
    (textures, text)
}

/// A row whose bitmap was blurred draws the three tiles side by side, the
/// bitmap, the blurred bitmap and the row's tile, with the width under the
/// middle one, and under them both sets of scores, whole, middle and ninths.
#[test]
fn a_blurred_row_draws_three_tiles_and_both_scores() {
    let m = scored(0.83, false);
    let (textures, text) = drawn(&m, false);
    assert_eq!(textures, vec![BITMAP, BLURRED, TILE], "{text}");
    assert!(text.contains("Blurred by \u{3c3} 0.83 grid px"), "{text}");
    for number in ["50.0%", "60.0%", "40.0%", "55.0%"] {
        assert!(text.contains(number), "{number}: {text}");
    }
    assert!(text.contains("ninths, plain"), "{text}");
    assert!(text.contains("ninths, blur-matched"), "{text}");
    // The last ninth, 80% plain and 85% blur-matched, on one line.
    let last = text
        .lines()
        .find(|line| line.contains("80") && line.contains("85"))
        .unwrap_or_else(|| panic!("{text}"));
    assert!(last.find("80") < last.find("85"), "{last}");
    assert!(!text.contains("could not align"), "{text}");
}

/// A row read unblurred draws the bitmap and its own tile with the note
/// between them, and says why: sharper than the bitmap, or the ratio did not
/// select it.
#[test]
fn an_unblurred_row_says_why_in_place_of_the_blurred_bitmap() {
    let (textures, text) = drawn(&scored(0.0, true), false);
    assert_eq!(textures, vec![BITMAP, TILE], "{text}");
    assert!(text.contains("Read unblurred"), "{text}");
    assert!(text.contains("could replace the reference"), "{text}");
    // A row with its own sharpness reading names both reasons the bitmap may
    // not have been blurred; one without names the missing reading.
    let mut read = scored(0.0, false);
    read.zncc_self_similarity_ellipse = Some(ellipse());
    let (textures, text) = drawn(&read, false);
    assert_eq!(textures, vec![BITMAP, TILE], "{text}");
    assert!(text.contains("ratio of 1.25"), "{text}");
    assert!(
        text.contains("the bitmap's own sharpness could not be read"),
        "{text}"
    );
    let (_, text) = drawn(&scored(0.0, false), false);
    assert!(text.contains("this row has no sharpness reading"), "{text}");
    assert!(!text.contains("ratio of 1.25"), "{text}");
}

/// The reference's own row draws the bitmap alone and says its score is 1
/// and is not computed.
#[test]
fn the_reference_row_draws_the_bitmap_alone() {
    let mut m = scored(0.0, false);
    m.plain_zncc = Some(1.0);
    let (textures, text) = drawn(&m, true);
    assert_eq!(textures, vec![BITMAP], "{text}");
    assert!(text.contains("is 1 (100%) and is not computed"), "{text}");
    assert!(matches!(
        zncc_hover_text(&m, true),
        ZnccHoverText::Reference(_)
    ));
}

/// A row with no score draws no picture and gives its reason; a row the
/// localizer could not read says so under whatever else is drawn.
#[test]
fn a_row_with_no_score_gives_its_reason() {
    let m = TrackMeasurement {
        reason: Some(Unmeasured::NoBitmap),
        seed_shift_px: Some(0.0),
        ..TrackMeasurement::default()
    };
    let (textures, text) = drawn(&m, false);
    assert!(textures.is_empty(), "{text}");
    assert!(text.contains(&Unmeasured::NoBitmap.to_string()), "{text}");

    let mut unread = scored(0.83, false);
    unread.seed_shift_px = None;
    let (_, text) = drawn(&unread, false);
    assert!(
        text.contains("could not align this row to the reference's render"),
        "{text}"
    );
}

/// The hover's tiles are drawn at a third of the tile's own hover picture.
#[test]
fn the_tiles_are_drawn_at_the_tile_hover_size() {
    assert_eq!(ZNCC_HOVER_TILE_SIDE, 96.0);
}

/// The blurred bitmap is the stored bitmap through core's kernel: no blur
/// leaves a textured bitmap as it was, a blur flattens it, and a sample
/// without data is drawn black.
#[test]
fn the_blurred_bitmap_is_the_bitmap_through_the_kernel() {
    let side = 12;
    let mut rgba = ndarray::Array3::<u8>::zeros((side, side, 4));
    for ((r, c, ch), v) in rgba.indexed_iter_mut() {
        *v = if ch == 3 {
            255
        } else {
            (((r + c) % 2) * 200 + 20) as u8
        };
    }
    rgba[[0, 0, 3]] = 0;
    let spread = |image: &egui::ColorImage| {
        let levels: Vec<u8> = image.pixels[side + 1..].iter().map(|p| p.r()).collect();
        levels.iter().max().unwrap() - levels.iter().min().unwrap()
    };
    let sharp = blurred_bitmap_image(rgba.view(), 0.0).expect("an image");
    assert_eq!(sharp.pixels[side + 1].r(), 20);
    assert_eq!(sharp.pixels[side + 2].r(), 220);
    let soft = blurred_bitmap_image(rgba.view(), 1.5).expect("an image");
    assert!(spread(&soft) < spread(&sharp) / 4, "{}", spread(&soft));
    assert_eq!(soft.pixels[0], egui::Color32::BLACK);
}
