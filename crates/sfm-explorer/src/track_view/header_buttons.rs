// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two icon buttons both of Track View's headers draw beside a point ID:
//! copy it, and go to a point by ID. Shared so the view-only header and the
//! bench header offer the same controls in the same place.

/// A small "go to point" button drawn as a right-pointing arrow into a bar.
/// Returns true if clicked.
///
/// An icon rather than a labelled button because it sits inside a header that
/// is already a dense run of `|`-separated numbers, and the copy button beside
/// it set the size a control there is allowed to be.
pub(crate) fn goto_button(ui: &mut egui::Ui) -> bool {
    let icon_size = ui.text_style_height(&egui::TextStyle::Body);
    let padding = 2.0;
    let total = icon_size + padding * 2.0;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(total, total), egui::Sense::click());

    if ui.is_rect_visible(rect) {
        let color = if response.hovered() {
            ui.visuals().strong_text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let stroke = egui::Stroke::new(1.0_f32, color);
        let c = rect.center();
        let arm = icon_size * 0.3;
        let head = icon_size * 0.22;
        // Shaft, then the two barbs of the arrowhead, then the bar it points
        // into — the standard "jump to" glyph.
        ui.painter().line_segment(
            [c + egui::vec2(-arm, 0.0), c + egui::vec2(arm * 0.5, 0.0)],
            stroke,
        );
        for dy in [-head, head] {
            ui.painter().line_segment(
                [
                    c + egui::vec2(arm * 0.5 - head, dy),
                    c + egui::vec2(arm * 0.5, 0.0),
                ],
                stroke,
            );
        }
        ui.painter().line_segment(
            [
                c + egui::vec2(arm, -icon_size * 0.32),
                c + egui::vec2(arm, icon_size * 0.32),
            ],
            stroke,
        );
    }

    let clicked = response.clicked();
    response.on_hover_text("Go to a point by index or ID");
    clicked
}

/// A small "copy to clipboard" button drawn as two overlapping rectangles.
/// Returns true if clicked.
pub(crate) fn copy_button(ui: &mut egui::Ui, tooltip: &str) -> bool {
    let icon_size = ui.text_style_height(&egui::TextStyle::Body);
    let padding = 2.0;
    let total = icon_size + padding * 2.0;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(total, total), egui::Sense::click());

    if ui.is_rect_visible(rect) {
        let color = if response.hovered() {
            ui.visuals().strong_text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let stroke = egui::Stroke::new(1.0_f32, color);

        // Two overlapping rounded rectangles (the standard "copy" icon).
        let inset = padding + 1.0;
        let offset = icon_size * 0.22;
        // Back rectangle (offset down-right)
        let back = egui::Rect::from_min_size(
            rect.min + egui::vec2(inset + offset, inset),
            egui::vec2(icon_size * 0.55, icon_size * 0.65),
        );
        ui.painter()
            .rect_stroke(back, 1.0, stroke, egui::StrokeKind::Outside);
        // Front rectangle (offset up-left, filled with panel background)
        let front = back.translate(egui::vec2(-offset, offset));
        ui.painter()
            .rect_filled(front, 1.0, ui.visuals().panel_fill);
        ui.painter()
            .rect_stroke(front, 1.0, stroke, egui::StrokeKind::Outside);
    }

    let clicked = response.clicked();
    response.on_hover_text(tooltip);
    clicked
}
