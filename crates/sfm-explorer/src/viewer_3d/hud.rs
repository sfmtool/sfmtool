// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The viewport heads-up display.
//!
//! A floating panel of 3D-viewport display controls, anchored to the top-right
//! of the viewport and collapsed by default to a single gear glyph. See
//! `specs/gui/viewport-hud.md`.
//!
//! The HUD is built *before* [`Viewer3D::show`] allocates the viewport painter,
//! because every input path in the viewport has to consult the rect it occupies
//! ([`Viewer3D::hud_rect`]). Building first is safe: an [`egui::Area`] is a
//! separate layer, so it still paints above the scene texture regardless of
//! call order.

use eframe::egui;

use crate::action_log::Kind;
use crate::state::AppState;

use super::display::{Field, FieldValue};
use super::Viewer3D;

/// Width of the expanded panel. Fixed, so slider tracks do not jitter as the
/// labels beside them change width.
const HUD_WIDTH: f32 = 220.0;

/// Inset of the HUD from the top-right corner of the viewport.
const HUD_INSET: f32 = 8.0;

/// Track width for the HUD's sliders. Narrow enough that track + value box +
/// label fit inside [`HUD_WIDTH`] without wrapping.
const HUD_SLIDER_WIDTH: f32 = 78.0;

/// Smallest viewport the expanded panel opens itself in. Below this the HUD
/// stays collapsed until the gear is clicked, which opens it at any size — the
/// spec's "not more
/// than about a third of the viewport" rule, resolved against a panel of fixed
/// width and bounded height: at the minimum width the panel spans half the
/// viewport horizontally but only a fraction of it vertically, so the area it
/// covers stays under a third.
const HUD_MIN_VIEWPORT: egui::Vec2 = egui::vec2(2.0 * (HUD_WIDTH + 2.0 * HUD_INSET), 300.0);

/// Opacity of the expanded panel's fill, so a little of the scene shows
/// through it. Low enough to read as floating over the viewport rather than
/// bolted to it, high enough that slider tracks and checkmarks stay legible
/// against a bright point cloud.
const HUD_FILL_OPACITY: f32 = 0.88;

/// Glyph on the collapsed gear, and the one that closes the expanded panel.
///
/// Both have to exist in egui's bundled fonts or they render as a replacement
/// box, and which characters those are is not stable across egui releases: the
/// close button was U+2716 HEAVY MULTIPLICATION X until egui 0.36 stopped
/// covering the whole U+2715..U+2718 run, so it is now U+1F5D9 CANCELLATION X.
/// That one is also the better fit — it lays out to the same 12.6px advance as
/// the gear, so collapsed and expanded buttons are the same width.
/// `the_hud_glyphs_are_available_in_the_bundled_fonts` in `hud/tests.rs`
/// guards this.
const HUD_EXPAND_GLYPH: &str = "⚙";
const HUD_COLLAPSE_GLYPH: &str = "🗙";

/// Stable id of one collapsible section, independent of the `Ui` that hosts it.
///
/// Explicit ids (rather than `CollapsingHeader`'s id-salt-plus-parent scheme)
/// keep the open/closed state addressable from outside the HUD — which is also
/// how the tests tell a section that was drawn from one that was omitted.
pub(crate) fn section_id(key: &str) -> egui::Id {
    egui::Id::new(("viewport_hud_section", key))
}

/// Whether *any* loaded reconstruction carries everything the patch surfel pass
/// needs: patch frames *and* the bitmaps to texture them with.
///
/// Any rather than the selected one: this HUD toggle is a master switch across
/// every node (the per-node eyes live in the Scene panel), so greying it out
/// because the *selected* node happens to have no patches would disable a
/// control that still governs something on screen.
pub(crate) fn has_patch_data(state: &AppState) -> bool {
    state.scene.iter().any(|node| node.has_patch_data())
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

/// One HUD checkbox, recorded as `Grid off` when the click changed it.
///
/// The label is the field's own ([`Field::label`]), which is also the run the
/// entry folds under and the word it opens with: two checkboxes ticked inside
/// a second are two acts and keep two lines. The entry is written by
/// [`Field::record_widget`], so it reads as the one `set_viewer_3d_display`
/// leaves for the same field.
fn checkbox(
    ui: &mut egui::Ui,
    log: &mut crate::action_log::ActionLog,
    on: &mut bool,
    field: Field,
) -> egui::Response {
    let response = ui.checkbox(on, field.label());
    field.record_widget(log, &response, FieldValue::Flag(*on));
    response
}

/// One HUD slider over `field`'s range, labelled `text`, recorded as
/// `Point size 3.0` when the drag changed it.
///
/// A drag records once a frame while it is moving; those entries coalesce into
/// a single line carrying the value it was let go at, which is the granularity
/// wanted and needs no `drag_stopped` plumbing. The run is the field's, so this
/// slider's run and the next slider's stay two lines however close together
/// they were moved.
fn slider(
    ui: &mut egui::Ui,
    log: &mut crate::action_log::ActionLog,
    value: &mut f32,
    field: Field,
    text: &'static str,
) {
    let response = ui.add(field.range().slider(&mut *value).text(text));
    field.record_widget(log, &response, FieldValue::Number(*value));
}

impl Viewer3D {
    /// Builds the HUD over the viewport and records the rect it occupies in
    /// [`Viewer3D::hud_rect`].
    ///
    /// Call this immediately before [`Viewer3D::show`], from the same `Ui`:
    /// `show` allocates `ui.available_size()`, so the rect it ends up painting
    /// into is exactly the space still available here. The HUD itself consumes
    /// no layout space — it lives on its own `Area` layer.
    pub fn show_hud(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut AppState,
        diagnostics: Option<(u32, u32, u32, u32)>,
        handler_ok: bool,
    ) {
        let viewport = ui.available_rect_before_wrap();
        if viewport.width() <= 0.0 || viewport.height() <= 0.0 {
            self.hud_rect = None;
            return;
        }

        // The size rule keeps the HUD from opening itself over a small
        // viewport. A click on the gear is a request to see the controls, so
        // it opens the panel whatever the size.
        let fits =
            viewport.width() >= HUD_MIN_VIEWPORT.x && viewport.height() >= HUD_MIN_VIEWPORT.y;
        let expanded = self.hud_open && (fits || self.hud_opened_by_click);

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
                    if ui
                        .button(HUD_EXPAND_GLYPH)
                        .on_hover_text("Display controls")
                        .clicked()
                    {
                        self.hud_open = true;
                        self.hud_opened_by_click = !fits;
                    }
                    return;
                }

                ui.set_width(HUD_WIDTH);
                let style = ui.style();
                let frame = egui::Frame::popup(style)
                    .fill(style.visuals.window_fill.gamma_multiply(HUD_FILL_OPACITY));
                frame.show(ui, |ui| {
                    ui.set_width(HUD_WIDTH);
                    ui.spacing_mut().slider_width = HUD_SLIDER_WIDTH;
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Display").strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .button(HUD_COLLAPSE_GLYPH)
                                .on_hover_text("Collapse display controls")
                                .clicked()
                            {
                                self.hud_open = false;
                                self.hud_opened_by_click = false;
                            }
                        });
                    });
                    ui.separator();
                    // No scroll container: every section is visible at once.
                    // The panel takes whatever height its content needs, and
                    // `constrain_to(viewport)` keeps it on screen.
                    self.hud_sections(ui, state, diagnostics, handler_ok);
                });
            });

        // The union of everything the HUD drew — gear when collapsed, gear plus
        // panel when expanded. Every viewport input path is gated on it.
        self.hud_rect = Some(area.response.rect);
    }

    /// The section list of the expanded HUD.
    fn hud_sections(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut AppState,
        diagnostics: Option<(u32, u32, u32, u32)>,
        handler_ok: bool,
    ) {
        let has_patches = has_patch_data(state);

        section(ui, "layers", "Layers", true, |ui| {
            let log = &mut state.action_log;
            checkbox(ui, log, &mut state.show_points, Field::ShowPoints);
            checkbox(
                ui,
                log,
                &mut state.show_camera_images,
                Field::ShowCameraImages,
            );
            checkbox(ui, log, &mut state.show_grid, Field::ShowGrid);
            // Greyed rather than hidden: unlike the Patches *section*, the
            // toggle stays visible so the capability remains discoverable on a
            // reconstruction that happens not to carry patches.
            let patches = ui
                .add_enabled(
                    has_patches,
                    egui::Checkbox::new(&mut state.show_patches, Field::ShowPatches.label()),
                )
                .on_disabled_hover_text("This reconstruction carries no patch bitmaps");
            Field::ShowPatches.record_widget(log, &patches, FieldValue::Flag(state.show_patches));
            checkbox(
                ui,
                log,
                &mut state.show_points_at_infinity,
                Field::ShowPointsAtInfinity,
            )
            .on_hover_text("Draw w = 0 points — directions with no parallax");
            checkbox(
                ui,
                log,
                &mut state.show_target_indicator,
                Field::ShowTargetIndicator,
            )
            .on_hover_text(
                "Draw the orbit target all the time. Holding Alt shows it while Alt is held.",
            );
        });

        section(ui, "size", "Size", true, |ui| {
            let log = &mut state.action_log;
            slider(
                ui,
                log,
                &mut state.point_size_log2,
                Field::PointSizeLog2,
                "Points",
            );
            if ui.button("Reset point size").clicked() {
                state.point_size_log2 = 0.0;
                // Another value of the same control, so a drag this button
                // interrupts folds into it rather than leaving the abandoned
                // value on the line above.
                Field::PointSizeLog2.record(log, FieldValue::Number(0.0));
            }
            slider(
                ui,
                log,
                &mut state.infinity_point_px,
                Field::InfinityPointPx,
                "∞ (px)",
            );
            slider(
                ui,
                log,
                &mut state.length_scale,
                Field::LengthScale,
                "Scene",
            );
        });

        // Hidden, not greyed: the four patch sliders are dead weight on the
        // common reconstruction that carries no patch bitmaps.
        if has_patches {
            section(ui, "patches", "Patches", false, |ui| {
                let log = &mut state.action_log;
                slider(
                    ui,
                    log,
                    &mut state.patch_opacity,
                    Field::PatchOpacity,
                    "Opacity",
                );
                slider(
                    ui,
                    log,
                    &mut state.patch_size_log2,
                    Field::PatchSizeLog2,
                    "Size",
                );
                slider(
                    ui,
                    log,
                    &mut state.patch_alpha_cutoff,
                    Field::PatchAlphaCutoff,
                    "Edge cutoff",
                );
            });
        }

        section(ui, "camera", "Camera", true, |ui| {
            checkbox(
                ui,
                &mut state.action_log,
                &mut self.maintain_z_up,
                Field::MaintainZUp,
            )
            .on_hover_text(
                "Turn the view back to +Z up whenever it is not looking through a camera. \
                 Q and E turn this off.",
            );
            let mut fov_degrees = self.camera.fov.to_degrees();
            let response = ui.add(
                egui::Slider::new(&mut fov_degrees, 10.0..=120.0)
                    .text("FOV °")
                    .fixed_decimals(0),
            );
            if response.changed() {
                self.camera.fov = fov_degrees.to_radians();
            }
            // The same text the MCP `set_view` fov form records, so the two
            // ways to change one number read as one action.
            state
                .action_log
                .changed(&response, Kind::Display, "Field of view", || {
                    format!("Field of view {fov_degrees:.1}°")
                });
            if ui.button("Reset FOV").clicked() {
                self.camera.fov = std::f64::consts::FRAC_PI_4;
                state.action_log.record_run(
                    Kind::Display,
                    "Field of view",
                    format!("Field of view {:.1}°", self.camera.fov.to_degrees()),
                );
            }
        });

        // Four parameters that were plumbed to the GPU but had no widget at
        // all — they could only be changed by editing the defaults in state.rs.
        section(ui, "advanced", "Advanced", false, |ui| {
            let log = &mut state.action_log;
            slider(
                ui,
                log,
                &mut state.edl_line_thickness,
                Field::EdlLineThickness,
                "EDL width",
            );
            slider(
                ui,
                log,
                &mut state.frustum_size_multiplier,
                Field::FrustumSizeMultiplier,
                "Frustum",
            );
            slider(
                ui,
                log,
                &mut state.target_size_multiplier,
                Field::TargetSizeMultiplier,
                "Target",
            );
            slider(
                ui,
                log,
                &mut state.target_fog_multiplier,
                Field::TargetFogMultiplier,
                "Target fog",
            );
        });

        section(ui, "debug", "Debug", false, |ui| {
            let log = &mut state.action_log;
            checkbox(
                ui,
                log,
                &mut state.show_controls_help,
                Field::ShowControlsHelp,
            );
            checkbox(ui, log, &mut state.show_fps, Field::ShowFps);
            // The touchpad counters used to be burned into the top-right corner
            // of every frame. They are developer instrumentation, so they live
            // here now and are off unless this section is open.
            ui.separator();
            ui.label(format!(
                "Touchpad: {}",
                if handler_ok { "OK" } else { "FAIL" }
            ));
            if let Some((hits, contacts, updates, global)) = diagnostics {
                ui.label(format!("H={hits} C={contacts}"));
                ui.label(format!("U={updates} G={global}"));
            }
        });
    }
}

#[cfg(test)]
mod tests;
