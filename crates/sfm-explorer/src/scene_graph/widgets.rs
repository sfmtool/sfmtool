// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scene Graph's leaf widgets and formatters: the two toggles every row is
//! built from, and the three functions that turn counts into row text.
//!
//! They are the panel's vocabulary rather than any one row's business — the
//! node row, the three group rows and the patches row all draw the same eye,
//! and a count is spelled the same way wherever it appears. Keeping them
//! together is what makes "did the group eye and the node eye drift apart?" a
//! question with no way to be true.

use eframe::egui;

use crate::scene::SceneNode;

use super::{EYE_GLYPH, ROW_HEIGHT, TOGGLE_WIDTH};

#[cfg(test)]
mod tests;

/// The master/group eye. Dimmed when off; the tooltip says what it governs.
pub(super) fn eye_toggle(
    ui: &mut egui::Ui,
    id: egui::Id,
    on: &mut bool,
    lit: Option<bool>,
    tooltip: &str,
) -> egui::Response {
    glyph_toggle(ui, id, on, lit, EYE_GLYPH, tooltip)
}

/// A one-glyph toggle button: full-strength when lit, weak otherwise.
///
/// `lit` is normally `None` — paint the glyph from the flag it toggles. The
/// master eye passes `Some(effective visibility)` instead, so a node hidden by
/// another node's solo reads as dark without its own flag being touched: the
/// eye still says what it will do the moment the solo ends.
///
/// Under an explicit id rather than the auto id a bare `ui.add` would take. An
/// auto id is a count of what was allocated before the widget in this `Ui`, so
/// adding or removing anything earlier in the row moves every later widget's
/// identity — and with it the hover, click and tooltip state egui keys on it.
pub(super) fn glyph_toggle(
    ui: &mut egui::Ui,
    id: egui::Id,
    on: &mut bool,
    lit: Option<bool>,
    glyph: &str,
    tooltip: &str,
) -> egui::Response {
    let color = if lit.unwrap_or(*on) {
        ui.visuals().strong_text_color()
    } else {
        ui.visuals().weak_text_color()
    };
    let button = egui::Button::new(egui::RichText::new(glyph).color(color))
        .frame(false)
        .min_size(egui::vec2(TOGGLE_WIDTH, ROW_HEIGHT));
    let response = ui
        .push_id(id, |ui| ui.add(button))
        .inner
        .on_hover_text(tooltip);
    if response.clicked() {
        *on = !*on;
    }
    response
}

/// `1.2M pts · 243 imgs · 2 cams`, elided to what is left of the row.
///
/// Three counts make this the longest row in a panel that defaults to 18% of
/// the window, so rather than truncate or wrap it drops a count at a time: the
/// camera count first (it is also on its own group row, one line down), then
/// the image count, leaving the point count — which has no other home in the
/// tree — last to go.
///
/// Measured against the width actually left after the label rather than
/// against a character budget, so widening the panel brings the counts back.
pub(super) fn counts_text(ui: &egui::Ui, node: &SceneNode) -> String {
    let points = format!("{} pts", compact_count(node.recon.points.len()));
    let images = format!("{} imgs", compact_count(node.recon.images.len()));
    let cameras = format!("{} cams", compact_count(node.recon.cameras.len()));
    let available = ui.available_width();
    let font = egui::TextStyle::Small.resolve(ui.style());
    let fits = |text: &str| {
        let galley =
            ui.painter()
                .layout_no_wrap(text.to_owned(), font.clone(), egui::Color32::PLACEHOLDER);
        galley.size().x <= available
    };
    let all_three = format!("{points} · {images} · {cameras}");
    if fits(&all_three) {
        return all_three;
    }
    let without_cameras = format!("{points} · {images}");
    if fits(&without_cameras) {
        return without_cameras;
    }
    points
}

/// `1234567` → `"1.2M"`, `12345` → `"12.3K"`, `999` → `"999"`.
///
/// The reconstruction row has to stay one line at an 18%-wide panel, and an
/// exact count is available one row down on the group rows.
pub(super) fn compact_count(n: usize) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}K", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

/// `1204551` → `"1,204,551"`.
pub(super) fn with_thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
