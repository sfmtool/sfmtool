// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The panel's header: a one-line point summary (ID, coordinates, error, track
//! length, triangulation diagnostics) and, for embedded-patches
//! reconstructions, the stored-patch preview tile drawn beneath it.

use sfmtool_core::SfmrReconstruction;

use super::{PointTrackDetail, STORED_PATCH_SIZE};

impl PointTrackDetail {
    /// Draw the point summary header bar, returning whether its Go to Point
    /// button was clicked.
    pub(super) fn show_header(
        &self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        point_idx: usize,
        point: &sfmtool_core::Point3D,
    ) -> bool {
        let point_id = format!("pt3d_{}_{}", self.hash_prefix, point_idx);
        let coords = format!(
            "{:.3}, {:.3}, {:.3}",
            point.position.x, point.position.y, point.position.z
        );
        let obs_count = recon.observation_counts[point_idx];

        let mut goto_clicked = false;
        ui.horizontal_wrapped(|ui| {
            // Color swatch
            let [r, g, b] = point.color;
            let color = egui::Color32::from_rgb(r, g, b);
            let (rect, swatch_response) =
                ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, color);
            ui.painter().rect_stroke(
                rect,
                2.0,
                egui::Stroke::new(1.0_f32, ui.visuals().weak_text_color()),
                egui::StrokeKind::Outside,
            );
            swatch_response.on_hover_text(format!("rgb({r}, {g}, {b})"));

            // Point ID — monospace, with copy button
            ui.label(egui::RichText::new(&point_id).monospace().strong());
            if copy_button(ui, "Copy Point ID") {
                ui.ctx().copy_text(point_id.clone());
            }
            // Beside Copy, because these are the two halves of one round trip:
            // copy an ID out of this header, paste it back into the dialog this
            // button opens — here, or in another session entirely.
            goto_clicked = goto_button(ui);

            ui.label("|");

            // XYZ coordinates — with copy button
            ui.label(format!("xyz: ({coords})"));
            if copy_button(ui, "Copy coordinates") {
                ui.ctx().copy_text(coords.clone());
            }

            ui.label("|");

            // Error
            ui.label(format!("error: {:.2}px", point.error));

            ui.label("|");

            // Track length
            ui.label(format!("track: {} obs", obs_count));

            // Max triangulation angle
            if self.max_angle_deg > 0.0 {
                ui.label("|");
                ui.label(format!("max pair angle: {:.1}°", self.max_angle_deg));
            }

            // Triangulation observability diagnostics (complementary to the
            // max angle — scale-free and correct in the near-infinity regime).
            if self.inverse_depth_z.is_finite() {
                ui.label("|");
                ui.label(format!("depth z: {:.1}", self.inverse_depth_z));
            }
            if self.condition_number.is_finite() {
                ui.label("|");
                ui.label(format!("cond: {:.0}", self.condition_number));
            }
        });
        goto_clicked
    }

    /// Draw the stored-patch preview tile, when the reconstruction carries one
    /// for the selected point. Draws nothing otherwise.
    pub(super) fn show_stored_patch_tile(&self, ui: &mut egui::Ui) {
        if let Some(texture) = &self.stored_patch_texture {
            ui.horizontal(|ui| {
                ui.label("Stored patch:");
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(STORED_PATCH_SIZE, STORED_PATCH_SIZE),
                    egui::Sense::hover(),
                );
                ui.painter().image(
                    texture.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            });
        }
    }
}

/// A small "go to point" button drawn as a right-pointing arrow into a bar.
/// Returns true if clicked.
///
/// An icon rather than a labelled button because it sits inside a header that
/// is already a dense run of `|`-separated numbers, and the copy button beside
/// it set the size a control here is allowed to be.
fn goto_button(ui: &mut egui::Ui) -> bool {
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
fn copy_button(ui: &mut egui::Ui, tooltip: &str) -> bool {
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
