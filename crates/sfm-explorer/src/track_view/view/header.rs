// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The panel's header: a one-line point summary (ID, coordinates, error, track
//! length, triangulation diagnostics) and, for embedded-patches
//! reconstructions, the stored-patch preview tile drawn beneath it.

use super::{PointTrackView, STORED_PATCH_SIZE};
use crate::track_view::header_buttons::{copy_button, goto_button};

impl PointTrackView {
    /// Draw the point summary header bar, returning whether its Go to Point
    /// button was clicked.
    pub(super) fn show_header(
        &self,
        ui: &mut egui::Ui,
        // Read through the overlay by the caller, so a modified point's track
        // length is this version's rather than the base's.
        obs_count: u32,
        point: &sfmtool_core::Point3D,
    ) -> bool {
        let point_id = self.point_id.clone();
        // Homogeneous, because `w` is the whole difference between a position
        // and a direction: it is `1` for a finite point and `0` for one at
        // infinity, whose `position` is then a unit direction rather than a
        // place. Printing three numbers under the label `xyz` reads as a point
        // a metre from the origin, which is the one thing it is not.
        let coords = format!(
            "{:.3}, {:.3}, {:.3}, {:.0}",
            point.position.x, point.position.y, point.position.z, point.w
        );

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

            // Homogeneous coordinates, with a copy button
            ui.label(format!("xyzw: ({coords})"));
            if copy_button(ui, "Copy coordinates") {
                ui.ctx().copy_text(coords.clone());
            }

            ui.label("|");

            // Error
            ui.label(format!("error: {:.2}px", point.error));

            ui.label("|");

            // Track length
            ui.label(format!("track: {} obs", obs_count));

            // Said in words as well as in `w`, because the rest of this row is
            // about a point that has a place and this one does not: the lines
            // that would say where it is are absent rather than zero, and a
            // reader is owed the reason.
            if point.w == 0.0 {
                ui.label("|");
                ui.label(egui::RichText::new("at infinity").color(ui.visuals().warn_fg_color));
            }

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
