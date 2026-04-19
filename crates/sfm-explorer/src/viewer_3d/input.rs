// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Input handling for the 3D viewer.
//!
//! Mouse drag, scroll/trackpad, platform gestures, keyboard shortcuts,
//! and click handling — all extracted from [`Viewer3D::show`].

use eframe::egui::{self, Rect};
use nalgebra::{Point3, Vector3};
use sfmtool_core::SfmrReconstruction;

use super::{
    Viewer3D, ViewportCamera, DRAG_ZOOM_SPEED, MOUSE_WHEEL_ZOOM_SPEED, TRACKPAD_ZOOM_SPEED,
};
use crate::platform::GestureEvent;

impl Viewer3D {
    /// Handles mouse drag interactions (orbit, pan, zoom, nodal pan).
    pub(super) fn handle_drag(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: Rect,
        fly_keys_held: bool,
    ) {
        let any_button_dragging = ui.input(|i| {
            let pointer = &i.pointer;
            pointer.is_moving() && pointer.any_down() && response.hovered()
        });

        // Read button state from the platform layer (Windows subclass proc) since
        // EnableMouseInPointer(true) makes egui report all buttons as Primary.
        // Middle-drag = pan, right-drag = zoom, left-drag = modifier-dependent.
        #[cfg(target_os = "windows")]
        let (middle_down, secondary_down) = {
            let state = crate::platform::windows::mouse_button_state();
            (
                state & crate::platform::windows::BUTTON_MIDDLE != 0,
                state & crate::platform::windows::BUTTON_RIGHT != 0,
            )
        };
        #[cfg(not(target_os = "windows"))]
        let (middle_down, secondary_down) = (false, false);

        if any_button_dragging || response.dragged() {
            // Cancel any in-progress target transition when user starts navigating
            self.cancel_transition();
            let delta = if response.dragged() {
                response.drag_delta()
            } else {
                ui.input(|i| i.pointer.delta())
            };
            // Lock fly-mode nodal pan when drag starts with fly keys held
            if !self.fly_drag_locked && fly_keys_held {
                self.fly_drag_locked = true;
            }
            let modifiers = ui.input(|i| i.modifiers);
            if self.fly_drag_locked {
                // Fly mode: all drags become nodal pan (mouse-look) — keeps camera view
                self.camera.nodal_pan(delta.x as f64, delta.y as f64);
            } else if middle_down {
                // Middle-drag = pan — exits camera view
                self.camera_view = None;
                self.camera.pan(
                    -delta.x as f64,
                    delta.y as f64,
                    rect.width() as f64,
                    rect.height() as f64,
                );
            } else if secondary_down {
                // Right-drag = zoom (Alt+right = target push/pull)
                if modifiers.alt {
                    // Target push/pull — keeps camera view
                    self.camera
                        .target_push_pull(delta.y as f64 * DRAG_ZOOM_SPEED);
                } else if self.camera_view.is_some() {
                    // Zoom FOV — keeps camera view
                    self.camera.zoom_fov(delta.y as f64 * DRAG_ZOOM_SPEED);
                } else {
                    // Zoom — exits camera view
                    self.camera_view = None;
                    self.camera.zoom(delta.y as f64 * DRAG_ZOOM_SPEED);
                }
            } else if modifiers.alt && modifiers.shift {
                // Alt+Shift+drag = pan — exits camera view
                self.camera_view = None;
                self.camera.pan(
                    -delta.x as f64,
                    delta.y as f64,
                    rect.width() as f64,
                    rect.height() as f64,
                );
            } else if modifiers.alt {
                if self.camera_view.is_some() {
                    // Alt+drag in camera view = orbit — exits camera view
                    self.camera_view = None;
                    self.camera.orbit(delta.x as f64, delta.y as f64);
                } else {
                    // Alt+drag = nodal pan (free-look) — keeps camera view
                    self.camera.nodal_pan(delta.x as f64, delta.y as f64);
                }
            } else if modifiers.ctrl || modifiers.command {
                if self.camera_view.is_some() {
                    // Ctrl+drag in camera view = zoom FOV — keeps camera view
                    self.camera.zoom_fov(delta.y as f64 * DRAG_ZOOM_SPEED);
                } else {
                    // Ctrl+drag = zoom (vertical motion) — exits camera view
                    self.camera_view = None;
                    self.camera.zoom(delta.y as f64 * DRAG_ZOOM_SPEED);
                }
            } else if modifiers.shift {
                // Shift+drag = pan — exits camera view
                self.camera_view = None;
                self.camera.pan(
                    -delta.x as f64,
                    delta.y as f64,
                    rect.width() as f64,
                    rect.height() as f64,
                );
            } else if self.camera_view.is_some() {
                // Unmodified drag in camera view = nodal pan (free-look) — keeps camera view
                self.camera.nodal_pan(delta.x as f64, delta.y as f64);
            } else {
                // Left-drag = orbit — exits camera view
                self.camera_view = None;
                self.camera.orbit(delta.x as f64, delta.y as f64);
            }
        } else {
            // No drag in progress — release fly drag lock
            self.fly_drag_locked = false;
        }
    }

    /// Handles scroll events (trackpad scroll and mouse wheel).
    pub(super) fn handle_scroll(
        &mut self,
        rect: Rect,
        scroll_input: &crate::platform::ScrollInput,
        fly_keys_held: bool,
    ) {
        // Cancel target transition on any scroll/gesture input
        if scroll_input.has_trackpad_scroll() || scroll_input.has_mouse_wheel() {
            self.cancel_transition();
        }
        if scroll_input.has_trackpad_scroll() {
            let delta = scroll_input.delta;
            let mods = scroll_input.modifiers;
            if fly_keys_held {
                // Fly mode: two-finger drag becomes nodal pan — keeps camera view
                self.camera.nodal_pan(-delta.x as f64, delta.y as f64);
            } else if mods.alt && (mods.ctrl || mods.command) {
                // Target push/pull — keeps camera view
                self.camera
                    .target_push_pull(delta.y as f64 * TRACKPAD_ZOOM_SPEED);
            } else if mods.alt && mods.shift {
                // Pan — exits camera view
                self.camera_view = None;
                self.camera.pan(
                    delta.x as f64,
                    -delta.y as f64,
                    rect.width() as f64,
                    rect.height() as f64,
                );
            } else if mods.alt {
                if self.camera_view.is_some() {
                    // Alt+scroll in camera view = orbit — exits camera view
                    self.camera_view = None;
                    self.camera.orbit(-delta.x as f64, delta.y as f64);
                } else {
                    // Nodal pan (free-look) — keeps camera view
                    self.camera.nodal_pan(-delta.x as f64, delta.y as f64);
                }
            } else if mods.ctrl || mods.command {
                if self.camera_view.is_some() {
                    // Zoom FOV — keeps camera view
                    self.camera.zoom_fov(delta.y as f64 * TRACKPAD_ZOOM_SPEED);
                } else {
                    // Zoom — exits camera view
                    self.camera_view = None;
                    self.camera.zoom(delta.y as f64 * TRACKPAD_ZOOM_SPEED);
                }
            } else if mods.shift {
                // Pan — exits camera view
                self.camera_view = None;
                self.camera.pan(
                    delta.x as f64,
                    -delta.y as f64,
                    rect.width() as f64,
                    rect.height() as f64,
                );
            } else if self.camera_view.is_some() {
                // Unmodified scroll in camera view = nodal pan (free-look) — keeps camera view
                self.camera.nodal_pan(-delta.x as f64, delta.y as f64);
            } else {
                // Orbit — exits camera view
                self.camera_view = None;
                self.camera.orbit(-delta.x as f64, delta.y as f64);
            }
        } else if scroll_input.has_mouse_wheel() {
            let delta = scroll_input.delta;
            let mods = scroll_input.modifiers;
            if mods.alt {
                // Target push/pull — keeps camera view
                self.camera
                    .target_push_pull(delta.y as f64 * MOUSE_WHEEL_ZOOM_SPEED);
            } else if self.camera_view.is_some() {
                // Zoom FOV — keeps camera view
                self.camera
                    .zoom_fov(delta.y as f64 * MOUSE_WHEEL_ZOOM_SPEED);
            } else {
                // Zoom — exits camera view
                self.camera_view = None;
                self.camera.zoom(delta.y as f64 * MOUSE_WHEEL_ZOOM_SPEED);
            }
        }
    }

    /// Handles pinch-to-zoom gesture.
    pub(super) fn handle_pinch(&mut self, ui: &egui::Ui, pointer_over: bool) {
        let zoom_delta = if pointer_over {
            ui.input(|i| i.zoom_delta())
        } else {
            1.0
        };
        if zoom_delta != 1.0 {
            if self.alt_held {
                // Alt+pinch = target push/pull — keeps camera view
                self.camera
                    .target_push_pull((zoom_delta - 1.0) as f64 * 1.0);
            } else if self.camera_view.is_some() {
                // Pinch zoom FOV — keeps camera view
                self.camera.zoom_fov((zoom_delta - 1.0) as f64 * 1.0);
            } else {
                // Pinch zoom — exits camera view
                self.camera_view = None;
                log::debug!("Pinch zoom_delta: {:.4}", zoom_delta);
                // zoom_delta > 1 means zoom in (pinch spread), < 1 means zoom out (pinch together)
                self.camera.zoom((zoom_delta - 1.0) as f64 * 1.0);
            }
        }
    }

    /// Handles platform-specific precision touchpad gestures.
    pub(super) fn handle_gestures(
        &mut self,
        ui: &egui::Ui,
        gesture_events: &[GestureEvent],
        rect: Rect,
        fly_keys_held: bool,
    ) {
        for event in gesture_events {
            // Force repaint while gestures are pouring in
            ui.ctx().request_repaint();
            // Cancel target transition on any gesture input
            self.cancel_transition();

            match event {
                GestureEvent::Pan { dx, dy } => {
                    let modifiers = ui.input(|i| i.modifiers);
                    if fly_keys_held {
                        // Fly mode: touchpad pan becomes nodal pan — keeps camera view
                        self.camera.nodal_pan(-*dx, *dy);
                    } else if modifiers.alt && (modifiers.ctrl || modifiers.command) {
                        // Target push/pull — keeps camera view
                        self.camera.target_push_pull(*dy * DRAG_ZOOM_SPEED);
                    } else if modifiers.alt && modifiers.shift {
                        // Pan — exits camera view
                        self.camera_view = None;
                        self.camera
                            .pan(*dx, *dy, rect.width() as f64, rect.height() as f64);
                    } else if modifiers.alt {
                        if self.camera_view.is_some() {
                            // Alt+gesture in camera view = orbit — exits camera view
                            self.camera_view = None;
                            self.camera.orbit(-*dx, *dy);
                        } else {
                            // Nodal pan (free-look) — keeps camera view
                            self.camera.nodal_pan(-*dx, *dy);
                        }
                    } else if modifiers.shift {
                        // Pan — exits camera view
                        self.camera_view = None;
                        self.camera
                            .pan(*dx, *dy, rect.width() as f64, rect.height() as f64);
                    } else if modifiers.ctrl || modifiers.command {
                        if self.camera_view.is_some() {
                            // Zoom FOV — keeps camera view
                            self.camera.zoom_fov(*dy * DRAG_ZOOM_SPEED);
                        } else {
                            // Zoom — exits camera view
                            self.camera_view = None;
                            self.camera.zoom(*dy * DRAG_ZOOM_SPEED);
                        }
                    } else if self.camera_view.is_some() {
                        // Unmodified gesture in camera view = nodal pan (free-look) — keeps camera view
                        self.camera.nodal_pan(-*dx, *dy);
                    } else {
                        // Orbit — exits camera view
                        self.camera_view = None;
                        self.camera.orbit(-*dx, *dy);
                    }
                }
                GestureEvent::Zoom { scale } => {
                    let modifiers = ui.input(|i| i.modifiers);
                    if modifiers.alt {
                        // Target push/pull — keeps camera view
                        self.camera.target_push_pull((*scale - 1.0) * 35.0);
                    } else if self.camera_view.is_some() {
                        // Zoom FOV — keeps camera view
                        self.camera.zoom_fov((*scale - 1.0) * 35.0);
                    } else {
                        // Zoom — exits camera view
                        self.camera_view = None;
                        self.camera.zoom((*scale - 1.0) * 35.0);
                    }
                }
            }
        }
    }

    /// Handles WASD fly navigation (continuous movement while keys held).
    pub(super) fn handle_fly_keys(&mut self, ui: &egui::Ui, fly_keys_held: bool) {
        if !fly_keys_held {
            return;
        }

        let dt = ui.input(|i| i.stable_dt) as f64;
        let speed = self.camera.camera.target_distance * dt;
        let sprint = if ui.input(|i| i.modifiers.shift) {
            3.0
        } else {
            1.0
        };

        // WASD/RF movement — moves camera center, exits camera view
        self.cancel_transition();
        let mut fwd = 0.0;
        let mut right = 0.0;
        let mut up = 0.0;
        ui.input(|i| {
            if i.key_down(egui::Key::W) {
                fwd += 1.0;
            }
            if i.key_down(egui::Key::S) {
                fwd -= 1.0;
            }
            if i.key_down(egui::Key::D) {
                right += 1.0;
            }
            if i.key_down(egui::Key::A) {
                right -= 1.0;
            }
            if i.key_down(egui::Key::R) {
                up += 1.0;
            }
            if i.key_down(egui::Key::F) {
                up -= 1.0;
            }
        });
        if fwd != 0.0 || right != 0.0 || up != 0.0 {
            self.camera_view = None;
            self.camera.fly_move(
                fwd * speed * sprint,
                right * speed * sprint,
                up * speed * sprint,
            );
        }

        // QE tilt — orientation only, keeps camera view
        let tilt_speed = std::f64::consts::FRAC_PI_2 * dt * sprint;
        ui.input(|i| {
            if i.key_down(egui::Key::Q) {
                self.camera.tilt(-tilt_speed);
            }
            if i.key_down(egui::Key::E) {
                self.camera.tilt(tilt_speed);
            }
        });

        ui.ctx().request_repaint(); // continuous animation while flying
    }

    /// Handles keyboard shortcuts (Z zoom-to-fit/camera view, comma/period navigate, Home reset).
    pub(super) fn handle_keyboard(
        &mut self,
        ui: &egui::Ui,
        rect: Rect,
        reconstruction: &SfmrReconstruction,
        selected_image: &mut Option<usize>,
    ) {
        ui.input(|i| {
            let current_time = i.time;
            if i.key_pressed(egui::Key::Z) {
                if let Some(img_idx) = *selected_image {
                    // Z with frustum selected = view through camera
                    self.enter_camera_view(img_idx, reconstruction, current_time);
                } else {
                    // Z with no selection = zoom to fit
                    if !reconstruction.points.is_empty() {
                        let aspect = rect.width() as f64 / rect.height() as f64;
                        let points: Vec<Point3<f64>> =
                            reconstruction.points.iter().map(|p| p.position).collect();
                        if let Some((end_pos, end_dist)) =
                            self.camera.compute_zoom_to_fit(&points, aspect)
                        {
                            self.start_transition(
                                end_pos,
                                self.camera.camera.orientation,
                                end_dist,
                                self.camera.fov,
                                self.camera.world_up,
                                None,
                                false,
                                current_time,
                            );
                        }
                    }
                }
            }
            // ,/. navigate to previous/next image. In camera view mode this
            // also switches which camera we're viewing through; otherwise the
            // viewport stays put and only the selection changes.
            if !reconstruction.images.is_empty() {
                let n = reconstruction.images.len();
                let in_camera_view = self.camera_view.is_some();
                let cur = self
                    .camera_view
                    .as_ref()
                    .map(|cv| cv.image_index)
                    .or(*selected_image);
                if i.key_pressed(egui::Key::Comma) {
                    let prev = match cur {
                        None => 0,
                        Some(0) => n - 1,
                        Some(c) => c - 1,
                    };
                    if in_camera_view {
                        self.switch_camera_view(prev, reconstruction);
                    }
                    *selected_image = Some(prev);
                }
                if i.key_pressed(egui::Key::Period) {
                    let next = match cur {
                        None => 0,
                        Some(c) if c + 1 >= n => 0,
                        Some(c) => c + 1,
                    };
                    if in_camera_view {
                        self.switch_camera_view(next, reconstruction);
                    }
                    *selected_image = Some(next);
                }
            }
            if i.key_pressed(egui::Key::Home) {
                self.camera_view = None;
                if i.modifiers.shift {
                    // Shift+Home = full view reset
                    self.camera = ViewportCamera::default();
                    self.view_initialized = false;
                } else {
                    // Home = level horizon: reset world_up to Z-up,
                    // re-orient camera to align with new up without moving
                    self.camera.world_up = Vector3::z();
                    let forward = self.camera.camera.forward();
                    self.camera.set_orientation_from_forward(forward);
                }
            }
        });
    }

    /// Handles click events (depth/entity pick requests).
    pub(super) fn handle_click(&mut self, ui: &egui::Ui, response: &egui::Response, rect: Rect) {
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let ppp = ui.ctx().pixels_per_point();
                let px = ((pos.x - rect.left()) * ppp) as u32;
                let py = ((pos.y - rect.top()) * ppp) as u32;
                self.pending_click = Some([px, py]);
                self.pending_click_is_alt = self.alt_held;
                self.pending_click_is_double = response.double_clicked();
                self.pick_ppp = ppp;
                self.pick_rect = rect;
            }
        }
    }
}
