// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! View manipulation for the image detail panel: drag-pan, right-drag zoom,
//! mouse-wheel / trackpad scroll, pinch-to-zoom, the Z / double-click view
//! reset, and Windows DirectManipulation gesture events.

use super::ImageDetail;
use crate::platform::{self, GestureEvent, ScrollInput};

/// Drag zoom speed: maps pixel delta to zoom factor.
const DRAG_ZOOM_SPEED: f32 = 0.005;
/// Mouse wheel zoom speed: maps line delta to zoom factor.
const MOUSE_WHEEL_ZOOM_SPEED: f32 = 0.15;
/// Trackpad scroll zoom speed (for Ctrl+scroll zoom).
const TRACKPAD_ZOOM_SPEED: f32 = 0.01;

impl ImageDetail {
    /// Process all view-manipulation input for the current frame, mutating
    /// `pan`/`zoom`. Returns `true` if a double-click reset the view, signalling
    /// the caller to skip feature interaction and return early.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn handle_input(
        &mut self,
        ui: &egui::Ui,
        interact_response: &egui::Response,
        panel_rect: egui::Rect,
        panel_center: egui::Pos2,
        panel_size: egui::Vec2,
        display_size: egui::Vec2,
        scroll_input: &ScrollInput,
        gesture_events: &[GestureEvent],
    ) -> bool {
        let pointer_over = platform::pointer_in_rect(ui.ctx(), panel_rect);

        // Double-click to reset view
        if interact_response.double_clicked() {
            self.reset_view();
            // Don't process this as a feature click
            return true;
        }

        // Z key to reset view (when panel is hovered)
        if pointer_over {
            let z_pressed = ui.input(|i| i.key_pressed(egui::Key::Z));
            if z_pressed {
                self.reset_view();
            }
        }

        // Drag to pan
        // On Windows, EnableMouseInPointer makes all buttons report as Primary.
        // Use platform-specific button state to distinguish middle/right.
        #[cfg(target_os = "windows")]
        let (_middle_down, secondary_down) = {
            let state = crate::platform::windows::mouse_button_state();
            (
                state & crate::platform::windows::BUTTON_MIDDLE != 0,
                state & crate::platform::windows::BUTTON_RIGHT != 0,
            )
        };
        #[cfg(not(target_os = "windows"))]
        let (_middle_down, secondary_down) = (false, false);

        let any_button_dragging = ui.input(|i| {
            let pointer = &i.pointer;
            pointer.is_moving() && pointer.any_down() && interact_response.hovered()
        });
        if any_button_dragging || interact_response.dragged() {
            let delta = if interact_response.dragged() {
                interact_response.drag_delta()
            } else {
                ui.input(|i| i.pointer.delta())
            };
            if secondary_down {
                // Right-drag vertical = zoom
                let cursor_rel = ui
                    .input(|i| i.pointer.hover_pos())
                    .map(|p| p - panel_center)
                    .unwrap_or(egui::Vec2::ZERO);
                let zoom_factor = 1.0 + delta.y * DRAG_ZOOM_SPEED;
                self.zoom_at(zoom_factor, cursor_rel);
            } else {
                // Left-drag, middle-drag, shift+drag = pan
                self.pan += delta;
            }
            self.clamp_pan(display_size, panel_size);
        }

        // Scroll wheel / trackpad scroll
        if pointer_over {
            if scroll_input.has_trackpad_scroll() {
                let delta = scroll_input.delta;
                let mods = scroll_input.modifiers;
                if mods.ctrl || mods.command {
                    // Ctrl+scroll = zoom toward cursor
                    let cursor_rel = ui
                        .input(|i| i.pointer.hover_pos())
                        .map(|p| p - panel_center)
                        .unwrap_or(egui::Vec2::ZERO);
                    let zoom_factor = 1.0 + delta.y * TRACKPAD_ZOOM_SPEED;
                    self.zoom_at(zoom_factor, cursor_rel);
                } else {
                    // Trackpad two-finger scroll = pan.
                    // Sign convention matches 3D viewer shift+scroll → camera.pan(delta.x, -delta.y):
                    // camera moves by (delta.x, -delta.y), content moves opposite.
                    self.pan.x -= delta.x;
                    self.pan.y += delta.y;
                }
                self.clamp_pan(display_size, panel_size);
            } else if scroll_input.has_mouse_wheel() {
                // Mouse wheel = zoom toward cursor
                let cursor_rel = ui
                    .input(|i| i.pointer.hover_pos())
                    .map(|p| p - panel_center)
                    .unwrap_or(egui::Vec2::ZERO);
                let zoom_factor = 1.0 + scroll_input.delta.y * MOUSE_WHEEL_ZOOM_SPEED;
                self.zoom_at(zoom_factor, cursor_rel);
                self.clamp_pan(display_size, panel_size);
            }
        }

        // Pinch-to-zoom (trackpad pinch) — only when pointer over panel
        let zoom_delta = if pointer_over {
            ui.input(|i| i.zoom_delta())
        } else {
            1.0
        };
        if zoom_delta != 1.0 {
            let cursor_rel = ui
                .input(|i| i.pointer.hover_pos())
                .map(|p| p - panel_center)
                .unwrap_or(egui::Vec2::ZERO);
            self.zoom_at(zoom_delta, cursor_rel);
            self.clamp_pan(display_size, panel_size);
        }

        // DirectManipulation gesture events (Windows precision touchpad)
        let gesture_events = if pointer_over { gesture_events } else { &[] };
        for event in gesture_events {
            ui.ctx().request_repaint();
            match event {
                GestureEvent::Pan { dx, dy } => {
                    let modifiers = ui.input(|i| i.modifiers);
                    if modifiers.ctrl || modifiers.command {
                        // Ctrl+pan = zoom
                        let cursor_rel = ui
                            .input(|i| i.pointer.hover_pos())
                            .map(|p| p - panel_center)
                            .unwrap_or(egui::Vec2::ZERO);
                        let zoom_factor = 1.0 + (*dy as f32) * DRAG_ZOOM_SPEED;
                        self.zoom_at(zoom_factor, cursor_rel);
                    } else {
                        // DM pan = scroll the viewport (push convention, opposite of grab).
                        // Sign convention matches 3D viewer shift+DM → camera.pan(dx, dy):
                        // camera moves by (dx, dy), content moves opposite.
                        self.pan.x -= *dx as f32;
                        self.pan.y += *dy as f32;
                    }
                    self.clamp_pan(display_size, panel_size);
                }
                GestureEvent::Zoom { scale } => {
                    let cursor_rel = ui
                        .input(|i| i.pointer.hover_pos())
                        .map(|p| p - panel_center)
                        .unwrap_or(egui::Vec2::ZERO);
                    self.zoom_at(*scale as f32, cursor_rel);
                    self.clamp_pan(display_size, panel_size);
                }
            }
        }

        false
    }
}
