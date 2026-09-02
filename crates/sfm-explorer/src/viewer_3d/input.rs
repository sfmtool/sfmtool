// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Input handling for the 3D viewer.
//!
//! Mouse drag, scroll/trackpad, platform gestures, keyboard shortcuts,
//! and click handling — all extracted from [`Viewer3D::show`].

use eframe::egui::{self, Rect};
use nalgebra::Vector3;

use super::{
    Viewer3D, ViewportCamera, DRAG_ZOOM_SPEED, MOUSE_WHEEL_ZOOM_SPEED, TRACKPAD_ZOOM_SPEED,
};
use crate::platform::GestureEvent;
use crate::scene::{ImageRef, SceneNode};
use crate::state::AppState;

impl Viewer3D {
    /// Handles `[` / `]` — stepping the **selected reconstruction** back and
    /// forward in tree order, the reconstruction analogue of `,` / `.` for
    /// images (which is why it lives here beside them).
    ///
    /// When an image is selected, stepping carries the selection to the
    /// **same-named image** in the new reconstruction if one exists — and, in
    /// camera view, the view follows it. Flipping `[` / `]` while looking
    /// through a camera is the core comparison move: same photo, two solves.
    /// With no same-named image the finer selection clears, per the invariant
    /// that all of it lives inside the selected reconstruction.
    ///
    /// **A solo travels with the step.** Solo and selection are otherwise
    /// independent — soloing does not select, selecting does not solo — but a
    /// solo left behind on the node you just stepped away from would leave the
    /// viewport showing a reconstruction no panel is talking about any more,
    /// and pressing `]` again would appear to do nothing at all. Soloed
    /// stepping is A/B comparison in its sharpest form: one reconstruction on
    /// screen at a time, the same photo, one keystroke apart.
    ///
    /// Called from `dock.rs` rather than from [`Viewer3D::show`]: stepping is
    /// the one viewport binding that needs the whole scene, not just the node
    /// being drawn.
    pub fn handle_recon_step(&mut self, ui: &egui::Ui, state: &mut AppState) {
        if state.scene.len() < 2 {
            return; // nothing to step to
        }
        let forward = ui.input(|i| {
            let back = i.key_pressed(egui::Key::OpenBracket);
            let fwd = i.key_pressed(egui::Key::CloseBracket);
            // Both in one frame cancel out rather than picking a winner.
            (fwd != back).then_some(fwd)
        });
        let Some(forward) = forward else {
            return;
        };

        let Some(current) = state.selected_recon else {
            return;
        };
        let Some(from) = state.scene.iter().position(|n| n.id == current) else {
            return;
        };
        let n = state.scene.len();
        let to = if forward {
            (from + 1) % n
        } else {
            (from + n - 1) % n
        };
        let new_id = state.scene[to].id;

        // Match by image *name*: the same photo solved twice is the same name
        // in two reconstructions, whatever index each solve gave it.
        let name = state
            .selected_image
            .filter(|i| i.recon == current)
            .and_then(|i| state.scene[from].recon.images.get(i.index()))
            .map(|image| image.name.clone());
        let carried = name.and_then(|name| {
            state.scene[to]
                .recon
                .images
                .iter()
                .position(|image| image.name == name)
        });

        // Through the setter: it is what scopes every finer selection —
        // image, camera and point — to the node being stepped onto.
        state.select_recon(new_id);
        // An active solo follows the step (see the doc comment); no solo stays
        // no solo — stepping never starts one.
        if state.solo.is_some() {
            state.solo = Some(new_id);
        }
        // A 3D point has no cross-reconstruction identity, so it never carries.
        state.selected_point = None;
        let was_in_camera_view = self
            .camera_view
            .as_ref()
            .is_some_and(|cv| cv.image.recon == current);
        match carried {
            Some(index) => {
                let image = ImageRef::new(new_id, index);
                state.select_image(Some(image));
                if was_in_camera_view {
                    self.switch_camera_view(image, &state.scene[to]);
                }
            }
            None => {
                state.select_image(None);
                if was_in_camera_view {
                    self.camera_view = None;
                }
            }
        }
    }

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
        node: &SceneNode,
        selected_image: &mut Option<ImageRef>,
        log: &mut crate::action_log::ActionLog,
    ) {
        use crate::action_log::Kind;

        let reconstruction = &node.recon;
        let recon_id = node.id;
        ui.input(|i| {
            let current_time = i.time;
            if i.key_pressed(egui::Key::Z) {
                if let Some(image) = selected_image.filter(|s| s.recon == recon_id) {
                    // Z with frustum selected = view through camera
                    self.enter_camera_view(image, node, current_time, log);
                } else {
                    // Z with no selection = zoom to fit, on where the node is
                    // drawn rather than on its native coordinates.
                    let aspect = rect.width() as f64 / rect.height() as f64;
                    let points = crate::scene::world_points(node);
                    let framed = !points.is_empty() && aspect > 0.0 && !aspect.is_nan();
                    self.zoom_to_fit_points(&points, aspect, current_time);
                    if framed {
                        log.record(Kind::View, format!("Framed {}", node.label));
                    }
                }
            }
            // ,/. navigate to previous/next image. In camera view mode this
            // also switches which camera we're viewing through; otherwise the
            // viewport stays put and only the selection changes.
            //
            // Stepping stays inside `recon_id`: an image or camera view
            // belonging to another node is not a position in this sequence, so
            // it reads as "nothing selected" and stepping starts from the top.
            if !reconstruction.images.is_empty() {
                let n = reconstruction.images.len();
                let in_camera_view = self.camera_view.is_some();
                let cur = self
                    .camera_view
                    .as_ref()
                    .map(|cv| cv.image)
                    .or(*selected_image)
                    .and_then(|image| image.index_in(recon_id));
                if i.key_pressed(egui::Key::Comma) {
                    let prev = match cur {
                        None => 0,
                        Some(0) => n - 1,
                        Some(c) => c - 1,
                    };
                    let prev = ImageRef::new(recon_id, prev);
                    if in_camera_view {
                        self.switch_camera_view(prev, node);
                    }
                    *selected_image = Some(prev);
                }
                if i.key_pressed(egui::Key::Period) {
                    let next = match cur {
                        None => 0,
                        Some(c) if c + 1 >= n => 0,
                        Some(c) => c + 1,
                    };
                    let next = ImageRef::new(recon_id, next);
                    if in_camera_view {
                        self.switch_camera_view(next, node);
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
                    log.record(Kind::View, "Reset the view");
                } else {
                    log.record(Kind::View, "Levelled the horizon");
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
