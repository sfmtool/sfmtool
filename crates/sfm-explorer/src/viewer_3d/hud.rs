// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The viewport heads-up display.
//!
//! A floating panel of 3D-viewport display controls, anchored to the top-right
//! of the viewport and collapsed by default to a single gear glyph. See
//! `specs/gui/gui-viewport-hud.md`.
//!
//! The HUD is built *before* [`Viewer3D::show`] allocates the viewport painter,
//! because every input path in the viewport has to consult the rect it occupies
//! ([`Viewer3D::hud_rect`]). Building first is safe: an [`egui::Area`] is a
//! separate layer, so it still paints above the scene texture regardless of
//! call order.

use eframe::egui;

use crate::state::AppState;

use super::Viewer3D;

/// Width of the expanded panel. Fixed, so slider tracks do not jitter as the
/// labels beside them change width.
const HUD_WIDTH: f32 = 220.0;

/// Inset of the HUD from the top-right corner of the viewport.
const HUD_INSET: f32 = 8.0;

/// Track width for the HUD's sliders. Narrow enough that track + value box +
/// label fit inside [`HUD_WIDTH`] without wrapping.
const HUD_SLIDER_WIDTH: f32 = 78.0;

/// Smallest viewport that will show the expanded panel. Below this the HUD
/// stays collapsed and the gear is the only affordance — the spec's "not more
/// than about a third of the viewport" rule, resolved against a panel of fixed
/// width and bounded height: at the minimum width the panel spans half the
/// viewport horizontally but only a fraction of it vertically, so the area it
/// covers stays under a third.
const HUD_MIN_VIEWPORT: egui::Vec2 = egui::vec2(2.0 * (HUD_WIDTH + 2.0 * HUD_INSET), 300.0);

/// Vertical space the HUD leaves for the sections, as a fraction of the
/// viewport height. Beyond it the section list scrolls.
const HUD_MAX_BODY_FRACTION: f32 = 0.6;

/// Stable id of one collapsible section, independent of the `Ui` that hosts it.
///
/// Explicit ids (rather than `CollapsingHeader`'s id-salt-plus-parent scheme)
/// keep the open/closed state addressable from outside the HUD — which is also
/// how the tests tell a section that was drawn from one that was omitted.
pub(crate) fn section_id(key: &str) -> egui::Id {
    egui::Id::new(("viewport_hud_section", key))
}

/// Whether `state`'s reconstruction carries everything the patch surfel pass
/// needs: patch frames *and* the bitmaps to texture them with.
pub(crate) fn has_patch_data(state: &AppState) -> bool {
    state.reconstruction.as_ref().is_some_and(|r| {
        r.patch_u_halfvec_xyz.is_some()
            && r.patch_v_halfvec_xyz.is_some()
            && r.patch_bitmaps_y_x_rgba.is_some()
    })
}

/// Draw one collapsible section, remembering its open/closed state for the
/// session under [`section_id`].
fn section(
    ui: &mut egui::Ui,
    key: &str,
    title: &str,
    default_open: bool,
    body: impl FnOnce(&mut egui::Ui),
) {
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        section_id(key),
        default_open,
    );
    let header = state.show_header(ui, |ui| {
        ui.label(egui::RichText::new(title).strong());
    });
    let _ = header.body(body);
}

impl Viewer3D {
    /// Builds the HUD over the viewport and records the rect it occupies in
    /// [`Viewer3D::hud_rect`].
    ///
    /// Call this immediately before [`Viewer3D::show`], from the same `Ui`:
    /// `show` allocates `ui.available_size()`, so the rect it ends up painting
    /// into is exactly the space still available here. The HUD itself consumes
    /// no layout space — it lives on its own `Area` layer.
    pub fn show_hud(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        let viewport = ui.available_rect_before_wrap();
        if viewport.width() <= 0.0 || viewport.height() <= 0.0 {
            self.hud_rect = None;
            return;
        }

        let expanded = self.hud_open
            && viewport.width() >= HUD_MIN_VIEWPORT.x
            && viewport.height() >= HUD_MIN_VIEWPORT.y;
        let max_body_height = viewport.height() * HUD_MAX_BODY_FRACTION;

        // Anchored to the viewport rect and recomputed every frame: the 3D
        // viewer lives in a dock tab the user can resize or re-dock, and a
        // fixed screen position would detach from the panel. `RIGHT_TOP` as
        // the pivot means the collapsed gear and the wider expanded panel
        // share a right edge without either having to know its own width.
        let anchor = egui::pos2(viewport.right() - HUD_INSET, viewport.top() + HUD_INSET);
        let ctx = ui.ctx().clone();
        let area = egui::Area::new(egui::Id::new("viewport_hud"))
            .order(egui::Order::Middle)
            .pivot(egui::Align2::RIGHT_TOP)
            .fixed_pos(anchor)
            .constrain_to(viewport)
            .show(&ctx, |ui| {
                if !expanded {
                    if ui.button("⚙").on_hover_text("Display controls").clicked() {
                        self.hud_open = true;
                    }
                    return;
                }

                ui.set_width(HUD_WIDTH);
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_width(HUD_WIDTH);
                    ui.spacing_mut().slider_width = HUD_SLIDER_WIDTH;
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Display").strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .button("✕")
                                .on_hover_text("Collapse display controls")
                                .clicked()
                            {
                                self.hud_open = false;
                            }
                        });
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .max_height(max_body_height)
                        .auto_shrink([false, true])
                        .show(ui, |ui| self.hud_sections(ui, state));
                });
            });

        // The union of everything the HUD drew — gear when collapsed, gear plus
        // panel when expanded. Every viewport input path is gated on it.
        self.hud_rect = Some(area.response.rect);
    }

    /// The section list of the expanded HUD.
    fn hud_sections(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        let has_patches = has_patch_data(state);

        section(ui, "layers", "Layers", true, |ui| {
            ui.checkbox(&mut state.show_points, "Points");
            ui.checkbox(&mut state.show_camera_images, "Cameras");
            ui.checkbox(&mut state.show_grid, "Grid");
            // Greyed rather than hidden: unlike the Patches *section*, the
            // toggle stays visible so the capability remains discoverable on a
            // reconstruction that happens not to carry patches.
            ui.add_enabled(
                has_patches,
                egui::Checkbox::new(&mut state.show_patches, "Patches"),
            )
            .on_disabled_hover_text("This reconstruction carries no patch bitmaps");
        });

        section(ui, "size", "Size", true, |ui| {
            ui.add(
                egui::Slider::new(&mut state.point_size_log2, -3.0..=3.0)
                    .text("Points")
                    .fixed_decimals(1),
            );
            if ui.button("Reset point size").clicked() {
                state.point_size_log2 = 0.0;
            }
            ui.add(
                egui::Slider::new(&mut state.infinity_point_px, 1.0..=16.0)
                    .text("∞ (px)")
                    .fixed_decimals(1),
            );
            ui.add(
                egui::Slider::new(&mut state.length_scale, 0.001..=100.0)
                    .logarithmic(true)
                    .text("Scene")
                    .fixed_decimals(3),
            );
        });

        // Hidden, not greyed: the four patch sliders are dead weight on the
        // common reconstruction that carries no patch bitmaps.
        if has_patches {
            section(ui, "patches", "Patches", false, |ui| {
                ui.add(
                    egui::Slider::new(&mut state.patch_opacity, 0.0..=1.0)
                        .text("Opacity")
                        .fixed_decimals(2),
                );
                ui.add(
                    egui::Slider::new(&mut state.patch_size_log2, -3.0..=3.0)
                        .text("Size")
                        .fixed_decimals(1),
                );
                ui.add(
                    egui::Slider::new(&mut state.patch_alpha_cutoff, 0.0..=1.0)
                        .text("Edge cutoff")
                        .fixed_decimals(2),
                );
            });
        }

        section(ui, "camera", "Camera", true, |ui| {
            let mut fov_degrees = self.camera.fov.to_degrees();
            let response = ui.add(
                egui::Slider::new(&mut fov_degrees, 10.0..=120.0)
                    .text("FOV °")
                    .fixed_decimals(0),
            );
            if response.changed() {
                self.camera.fov = fov_degrees.to_radians();
            }
            if ui.button("Reset FOV").clicked() {
                self.camera.fov = std::f64::consts::FRAC_PI_4;
            }
        });
    }
}

#[cfg(test)]
mod tests;
