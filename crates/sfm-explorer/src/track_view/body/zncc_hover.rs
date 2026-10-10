// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The *ZNCC* cell's hover at the track stage: the comparison behind the
//! row's score against the stored patch bitmap.
//!
//! Three tiles side by side: the stored bitmap, the bitmap as blurred for this
//! row (or the note that the pair was read unblurred, and why), and the row's
//! own tile. Under them, the plain and the blur-matched scores side by side,
//! for the whole tile, the middle and each ninth. The reference's own row
//! shows the bitmap alone, and a row with no score says why it has none.
//!
//! The blurred bitmap is the stored bitmap blurred by the row's
//! `bitmap_blur_sigma` with core's kernel (`TilePlanes::blurred`), the one the
//! blur-matched score was read on ([`blurred_bitmap_image`]).

use sfmtool_core::bench::TrackMeasurement;
use sfmtool_core::patch::blur_matched::BlurScratch;
use sfmtool_core::patch::stored_bitmap::stored_bitmap_planes;

use super::reference::grid_lines;
use super::tile::{CONTEXT_FACTOR, CONTEXT_HOVER_SIDE};

/// Side of each tile in the hover, in points: the size the tile's own hover
/// view draws the tile at, a third of its picture.
pub(super) const ZNCC_HOVER_TILE_SIDE: f32 = CONTEXT_HOVER_SIDE / CONTEXT_FACTOR as f32;

/// The pictures the hover draws, uploaded by the table; `None` where one is
/// not drawn (not rendered yet, or nothing to draw).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ZnccHoverTiles {
    /// The stored patch bitmap.
    pub(super) bitmap: Option<egui::TextureId>,
    /// The bitmap as blurred for this row; `None` where the pair was read
    /// plain.
    pub(super) blurred: Option<egui::TextureId>,
    /// The row's own tile.
    pub(super) tile: Option<egui::TextureId>,
}

/// What the hover says about a row, apart from its pictures.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ZnccHoverText {
    /// The reference's own row: the bitmap is its render, and its score is 1
    /// and is not computed.
    Reference(String),
    /// A row with a score.
    Scored {
        /// The caption of the middle slot: the blur's width, or why the pair
        /// was read unblurred.
        blurred: String,
        /// Whether the middle slot holds a picture.
        is_blurred: bool,
        /// The plain and blur-matched scores side by side, whole, middle and
        /// each ninth, in monospace.
        table: String,
    },
    /// A row with no score, and why.
    Unscored(String),
}

/// The sentence under the pictures where the localizer could not read the
/// row.
const UNALIGNED: &str = "The localizer could not align this row to the reference's render, so \
    the bars do not judge it.";

/// What the hover says about `m`. `is_reference` says the bitmap is this
/// row's own render.
pub(super) fn zncc_hover_text(m: &TrackMeasurement, is_reference: bool) -> ZnccHoverText {
    if m.blur_matched_zncc.is_some() && is_reference {
        return ZnccHoverText::Reference(
            "This row is the track's reference: the patch bitmap is its render at its \
             keypoint, so its score against the bitmap is 1 (100%) and is not computed."
                .to_string(),
        );
    }
    let Some(plain) = m.plain_zncc else {
        // A row read back from a committed point carries only the
        // blur-matched score its `observation_confidence` stored.
        if let Some(stored) = m.blur_matched_zncc.filter(|v| v.is_finite()) {
            return ZnccHoverText::Unscored(format!(
                "Read back from the committed point: its blur-matched score against the \
                 stored patch bitmap was {:.0}%. The next evaluation scores it again.",
                100.0 * stored
            ));
        }
        return ZnccHoverText::Unscored(match m.reason {
            Some(reason) => format!("No score against the stored patch bitmap: {reason}."),
            None => "No score against the stored patch bitmap.".to_string(),
        });
    };
    let sigma = m.bitmap_blur_sigma.filter(|&s| s > 0.0);
    let blurred = match sigma {
        Some(sigma) => format!("Blurred by \u{3c3} {sigma:.2} grid px"),
        None if m.sharper_than_bitmap == Some(true) => {
            "Read unblurred: this row's tile is sharper than the bitmap along every \
             direction, so it could replace the reference."
                .to_string()
        }
        // The row carries its own sharpness reading but not the bitmap's, so
        // the one reason the row can name for sure is a reading of its own
        // that is missing.
        None if !m
            .zncc_self_similarity_ellipse
            .is_some_and(|e| e.grid_px.axes.iter().all(|a| a.is_finite())) =>
        {
            "Read unblurred: this row has no sharpness reading, so the bitmap was not \
             blurred for it."
                .to_string()
        }
        None => "Read unblurred: the bitmap was not blurred for this row. It is not sharper \
                 than this row's tile along every direction by the ratio of 1.25, or the \
                 bitmap's own sharpness could not be read."
            .to_string(),
    };
    ZnccHoverText::Scored {
        blurred,
        is_blurred: sigma.is_some(),
        table: score_table(m, plain),
    }
}

/// The plain and the blur-matched scores side by side: the whole tile and
/// the middle in percent, then each ninth as two 3×3 grids.
fn score_table(m: &TrackMeasurement, plain: f64) -> String {
    let percent = |v: Option<f64>| match v {
        Some(v) if v.is_finite() => format!("{:.1}%", 100.0 * v),
        Some(_) => "NaN".to_string(),
        None => "-".to_string(),
    };
    let mut lines = vec![
        format!("{:<8}{:>8}{:>15}", "", "plain", "blur-matched"),
        format!(
            "{:<8}{:>8}{:>15}",
            "whole",
            percent(Some(plain)),
            percent(m.blur_matched_zncc)
        ),
        format!(
            "{:<8}{:>8}{:>15}",
            "middle",
            percent(m.plain_zncc_middle),
            percent(m.blur_matched_zncc_middle)
        ),
    ];
    if let (Some(p), Some(b)) = (m.plain_zncc_grid, m.blur_matched_zncc_grid) {
        lines.push(String::new());
        lines.push(format!("{:<18}{}", "ninths, plain", "ninths, blur-matched"));
        for (p, b) in grid_lines(p).into_iter().zip(grid_lines(b)) {
            lines.push(format!("{p:<18}{b}"));
        }
    }
    lines.join("\n")
}

/// Draw the hover for `m` into `ui`.
pub(super) fn show_zncc_hover(
    ui: &mut egui::Ui,
    m: &TrackMeasurement,
    is_reference: bool,
    tiles: ZnccHoverTiles,
) {
    let side = egui::vec2(ZNCC_HOVER_TILE_SIDE, ZNCC_HOVER_TILE_SIDE);
    match zncc_hover_text(m, is_reference) {
        ZnccHoverText::Reference(sentence) => {
            ui.set_max_width(3.0 * ZNCC_HOVER_TILE_SIDE);
            captioned(ui, tiles.bitmap, side, "Stored bitmap");
            ui.label(sentence);
        }
        ZnccHoverText::Scored {
            blurred,
            is_blurred,
            table,
        } => {
            let width = 3.0 * ZNCC_HOVER_TILE_SIDE + 2.0 * ui.spacing().item_spacing.x;
            ui.set_max_width(width);
            ui.label(
                "Scores against the stored patch bitmap. The min ZNCC bars judge the \
                 blur-matched ones.",
            );
            ui.horizontal_top(|ui| {
                captioned(ui, tiles.bitmap, side, "Stored bitmap");
                if is_blurred {
                    captioned(ui, tiles.blurred, side, &blurred);
                } else {
                    ui.vertical(|ui| {
                        ui.set_width(ZNCC_HOVER_TILE_SIDE);
                        ui.add(egui::Label::new(&blurred).wrap());
                    });
                }
                captioned(ui, tiles.tile, side, "This row");
            });
            ui.label(egui::RichText::new(table).monospace());
        }
        ZnccHoverText::Unscored(sentence) => {
            ui.label(sentence);
        }
    }
    if m.seed_shift_px.is_none() {
        ui.label(UNALIGNED);
    }
}

/// One picture at `side`, or a faint square where there is none, with
/// `caption` under it.
fn captioned(ui: &mut egui::Ui, texture: Option<egui::TextureId>, side: egui::Vec2, caption: &str) {
    ui.vertical(|ui| {
        ui.set_width(side.x);
        match texture {
            Some(texture) => {
                ui.add(egui::Image::new((texture, side)));
            }
            None => {
                let (rect, _) = ui.allocate_exact_size(side, egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, 2.0, ui.visuals().faint_bg_color);
            }
        }
        ui.add(egui::Label::new(caption).wrap());
    });
}

/// The stored bitmap, `(R, R, channels)`, blurred by `sigma` grid px with the
/// kernel its blur-matched score is read on, as an opaque RGBA image. A
/// sample the bitmap has no data at (alpha `0` with two or four channels)
/// carries none into the blur and is drawn black. `None` for an empty bitmap.
pub(super) fn blurred_bitmap_image(
    bitmap: ndarray::ArrayView3<'_, u8>,
    sigma: f64,
) -> Option<egui::ColorImage> {
    let planes = stored_bitmap_planes(bitmap)?;
    let (h, w) = (planes.side, planes.side);
    let blurred = planes.blurred(sigma, &mut BlurScratch::default());
    let n = h * w;
    let level = |c: usize, k: usize| -> u8 {
        if blurred.data[k] {
            blurred.values[c.min(blurred.channels - 1) * n + k]
                .round()
                .clamp(0.0, 255.0) as u8
        } else {
            0
        }
    };
    let rgba: Vec<u8> = (0..n)
        .flat_map(|k| [level(0, k), level(1, k), level(2, k), 255])
        .collect();
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}
