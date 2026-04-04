// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! 3D reconstruction viewer.
//!
//! Renders point clouds and camera frustums with orbit/pan/zoom camera
//! navigation, keyboard fly mode, and animated transitions.

#![allow(dead_code)]

mod camera;
mod input;
mod overlay;

pub use camera::{best_fit_fov, ViewportCamera};

use eframe::egui::{self, Color32, Pos2, Rect, Sense};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use sfmtool_core::{Camera, SfmrReconstruction};

use crate::platform::GestureEvent;

/// Drag/gesture zoom speed: maps pixel deltas to zoom amount.
const DRAG_ZOOM_SPEED: f64 = 0.13125;

/// Trackpad Ctrl+scroll zoom speed: maps trackpad point deltas to zoom amount.
const TRACKPAD_ZOOM_SPEED: f64 = 0.00375;

/// Mouse wheel zoom speed: maps line deltas to zoom amount.
const MOUSE_WHEEL_ZOOM_SPEED: f64 = 0.75;

/// Rotation speed of the target indicator in radians per second.
const TARGET_ROTATION_SPEED: f64 = std::f64::consts::PI / 6.0; // 30 deg/sec

/// Duration of animated camera transitions in seconds.
const CAMERA_TRANSITION_DURATION: f64 = 0.2;

/// Animated camera transition for smooth navigation.
///
/// Interpolates camera state over ~200ms using slerp (orientation) + lerp
/// (position, distance, FOV) with smoothstep easing. Used for orbit target
/// changes, zoom-to-fit, and camera view transitions.
struct CameraTransition {
    start_position: Point3<f64>,
    end_position: Point3<f64>,
    start_orientation: UnitQuaternion<f64>,
    end_orientation: UnitQuaternion<f64>,
    start_distance: f64,
    end_distance: f64,
    start_fov: f64,
    end_fov: f64,
    start_world_up: Vector3<f64>,
    end_world_up: Vector3<f64>,
    start_time: f64,
    /// Camera view mode to activate when the transition completes.
    pending_camera_view: Option<CameraViewMode>,
    /// Whether to trigger a target flash on completion.
    flash_on_complete: bool,
}

impl CameraTransition {
    /// Returns the interpolation progress with smoothstep easing, or None if complete.
    fn progress(&self, current_time: f64) -> Option<f64> {
        let elapsed = current_time - self.start_time;
        if elapsed >= CAMERA_TRANSITION_DURATION {
            return None; // transition complete
        }
        let t = elapsed / CAMERA_TRANSITION_DURATION;
        // Smoothstep: 3t² - 2t³ (ease-in/ease-out)
        Some(t * t * (3.0 - 2.0 * t))
    }
}

/// Computed end state for a camera view switch.
struct SwitchCameraViewState {
    position: Point3<f64>,
    orientation: UnitQuaternion<f64>,
    distance: f64,
    world_up: Vector3<f64>,
    camera_view: CameraViewMode,
}

/// Camera view mode state: active when viewing through a selected camera.
///
/// Stores the SfM camera's world-from-camera rotation so the background mesh
/// can be rendered with the correct relative rotation during free-look navigation.
pub struct CameraViewMode {
    /// Index of the image being viewed through (stable across selection changes).
    pub image_index: usize,
    /// World-from-camera rotation of the SfM camera being viewed.
    /// Used to compute the relative view rotation for the BG mesh.
    pub r_world_from_cam: UnitQuaternion<f64>,
}

/// 3D viewer state and rendering.
pub struct Viewer3D {
    /// Viewport camera.
    pub camera: ViewportCamera,
    /// Whether the view has been initialized to frame the data.
    pub view_initialized: bool,
    /// Camera view mode — active when viewing through a selected camera.
    pub camera_view: Option<CameraViewMode>,
    /// Accumulated scroll deltas for combining X/Y from separate events.
    /// Windows precision touchpads send X and Y scrolls as separate events,
    /// so we accumulate them to enable diagonal scrolling.
    scroll_accum: egui::Vec2,
    /// Last known panel size in physical pixels, used by SceneRenderer
    /// to create offscreen textures at the correct resolution.
    pub panel_size: [u32; 2],
    /// Mouse position in texture pixels (set each frame from hover pos).
    pub hover_pixel: Option<[u32; 2]>,
    /// Whether the Alt key is currently held.
    pub alt_held: bool,
    /// Whether the Alt key was held on the previous frame (for edge detection).
    alt_was_held: bool,
    /// Timestamp of the last Alt key press (for double-tap detection).
    last_alt_press_time: f64,
    /// Whether the target indicator is locked on via Alt double-tap.
    pub target_keep_visible: bool,
    /// Supernova effect activation level (0.0 = off, 1.0 = fully on).
    pub supernova_active: f32,
    /// Target's view-space position [x, y, z] for the supernova effect (z is positive = in front).
    pub supernova_view_pos: [f32; 3],
    /// Elapsed time for supernova wave animation (seconds).
    pub supernova_time: f32,
    /// Current rotation angle of the target indicator (radians).
    pub target_indicator_rotation: f64,
    /// Timestamp of the last target change for the flash animation.
    pub target_flash_start: Option<f64>,
    /// Whether the target indicator should be visible this frame.
    pub target_indicator_visible: bool,
    /// Flash animation radius scale (1.0 = normal, >1.0 during flash).
    pub target_indicator_radius_scale: f32,
    /// Flash animation alpha scale (1.0 = normal, >1.0 during flash).
    pub target_indicator_alpha_scale: f32,
    /// Pending click request: screen pixel position [x, y] in texture pixels.
    /// Both depth and entity pick are read back from this single click.
    pub pending_click: Option<[u32; 2]>,
    /// Whether the pending click was Alt+Click (sets orbit target from depth).
    pub pending_click_is_alt: bool,
    /// Whether the pending click was a double-click (enters camera view mode).
    pub pending_click_is_double: bool,
    /// Pixels per point at the time of the click request.
    pick_ppp: f32,
    /// Screen rect at the time of the click request.
    pick_rect: Rect,
    /// Whether any WASD/RF/QE fly key is currently held.
    fly_keys_held: bool,
    /// Whether a drag was initiated while fly keys were held (locks nodal pan for the drag).
    fly_drag_locked: bool,
    /// Active animated camera transition (orbit target, zoom-to-fit, camera view).
    target_transition: Option<CameraTransition>,
}

impl Default for Viewer3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Viewer3D {
    /// Creates a new 3D viewer.
    pub fn new() -> Self {
        Self {
            camera: ViewportCamera::default(),
            view_initialized: false,
            camera_view: None,
            scroll_accum: egui::Vec2::ZERO,
            panel_size: [0, 0],
            hover_pixel: None,
            alt_held: false,
            alt_was_held: false,
            last_alt_press_time: -1.0,
            target_keep_visible: false,
            supernova_active: 0.0,
            supernova_view_pos: [0.0, 0.0, 5.0],
            supernova_time: 0.0,
            target_indicator_rotation: 0.0,
            target_flash_start: None,
            target_indicator_visible: false,
            target_indicator_radius_scale: 1.0,
            target_indicator_alpha_scale: 1.0,
            pending_click: None,
            pending_click_is_alt: false,
            pending_click_is_double: false,
            pick_ppp: 1.0,
            pick_rect: Rect::NOTHING,
            fly_keys_held: false,
            fly_drag_locked: false,
            target_transition: None,
        }
    }

    /// Shows the 3D viewer UI and renders the reconstruction.
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        reconstruction: &SfmrReconstruction,
        selected_image: &mut Option<usize>,
        show_grid: bool,
        length_scale: f32,
        gesture_events: &[GestureEvent],
        scroll_input: &crate::platform::ScrollInput,
        diagnostics: Option<(u32, u32, u32, u32)>,
        handler_ok: bool,
        scene_texture_id: Option<egui::TextureId>,
        hover_depth: Option<f32>,
        hover_pick_id: u32,
    ) {
        // Allocate the entire available space for the 3D view.
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        self.alt_held = ui.input(|i| i.modifiers.alt);

        // Detect Alt double-tap to toggle target lock
        let current_time = ui.input(|i| i.time);
        if self.alt_held && !self.alt_was_held {
            // Alt just pressed — check for double-tap (within 300ms)
            if current_time - self.last_alt_press_time < 0.3 {
                self.target_keep_visible = !self.target_keep_visible;
                self.last_alt_press_time = -1.0; // reset to prevent triple-tap toggle
            } else {
                self.last_alt_press_time = current_time;
            }
        }
        self.alt_was_held = self.alt_held;

        // Check fly key state early — used by drag handling and supernova suppression
        self.fly_keys_held = ui.input(|i| {
            i.key_down(egui::Key::W)
                || i.key_down(egui::Key::A)
                || i.key_down(egui::Key::S)
                || i.key_down(egui::Key::D)
                || i.key_down(egui::Key::R)
                || i.key_down(egui::Key::F)
                || i.key_down(egui::Key::Q)
                || i.key_down(egui::Key::E)
        });
        let fly_keys_held = self.fly_keys_held;

        // Animate supernova activation (200ms fade)
        let dt = ui.input(|i| i.stable_dt);
        self.supernova_time = current_time as f32;
        let fade_speed = 5.0; // 1.0 / 0.2 seconds
        if self.alt_held || self.target_keep_visible {
            self.supernova_active = (self.supernova_active + dt * fade_speed).min(1.0);
        } else {
            self.supernova_active = (self.supernova_active - dt * fade_speed).max(0.0);
        }
        if self.supernova_active > 0.0 && self.supernova_active < 1.0 {
            ui.ctx().request_repaint(); // keep animating the fade
        }

        // Animate camera transition (smooth movement for orbit target, zoom, camera view)
        self.animate_transition(ui, current_time);

        // Initialize view to frame all points on first show
        if !self.view_initialized && !reconstruction.points.is_empty() {
            let aspect = rect.width() as f64 / rect.height() as f64;
            let points: Vec<Point3<f64>> =
                reconstruction.points.iter().map(|p| p.position).collect();
            self.camera.zoom_to_fit(&points, aspect);
            self.view_initialized = true;
        }

        // Handle all input
        self.handle_drag(ui, &response, rect, fly_keys_held);

        let pointer_over = crate::platform::pointer_in_rect(ui.ctx(), rect);
        if pointer_over {
            self.handle_scroll(rect, scroll_input, fly_keys_held);
        }
        self.handle_pinch(ui, pointer_over);

        let gesture_events = if pointer_over { gesture_events } else { &[] };
        self.handle_gestures(ui, gesture_events, rect, fly_keys_held);
        self.handle_fly_keys(ui, fly_keys_held);
        self.handle_keyboard(ui, rect, reconstruction, selected_image);
        self.handle_click(ui, &response, rect);

        // Record mouse position in texture pixels for GPU depth readback
        let ppp = ui.ctx().pixels_per_point();
        self.hover_pixel = response.hover_pos().map(|pos| {
            let px = ((pos.x - rect.left()) * ppp) as u32;
            let py = ((pos.y - rect.top()) * ppp) as u32;
            [px, py]
        });

        // Compute target view-space position for supernova effect
        let target = self.camera.target();
        let view_pos = self.camera.world_to_view(&target);
        self.supernova_view_pos = [
            view_pos.x as f32,
            view_pos.y as f32,
            -view_pos.z as f32, // positive = in front of camera
        ];

        // Track panel size in physical pixels for the scene renderer
        let ppp = ui.ctx().pixels_per_point();
        self.panel_size = [(rect.width() * ppp) as u32, (rect.height() * ppp) as u32];

        // Background: GPU-rendered scene texture or solid color fallback
        if let Some(tex_id) = scene_texture_id {
            let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
            let mut mesh = egui::Mesh::with_texture(tex_id);
            mesh.add_rect_with_uv(rect, uv, Color32::WHITE);
            painter.add(egui::Shape::mesh(mesh));
        } else {
            painter.rect_filled(rect, 0.0, Color32::from_rgb(30, 30, 35));
        }

        // Draw grid if enabled
        if show_grid {
            self.draw_grid(&painter, rect, length_scale);
        }

        // Draw axis indicator in corner
        self.draw_axis_indicator(&painter, rect);

        // Update target indicator state for GPU rendering
        self.update_target_indicator_state(ui);

        // Draw info overlay
        let fps = 1.0 / ui.input(|i| i.predicted_dt as f64);
        self.draw_info_overlay(
            &painter,
            rect,
            reconstruction,
            diagnostics,
            handler_ok,
            hover_depth,
            hover_pick_id,
            fps,
        );
    }

    /// Animates the camera transition (smooth movement for orbit target, zoom, camera view).
    fn animate_transition(&mut self, ui: &egui::Ui, current_time: f64) {
        if let Some(ref transition) = self.target_transition {
            if let Some(t) = transition.progress(current_time) {
                self.camera.camera.position = transition.start_position
                    + (transition.end_position - transition.start_position) * t;
                self.camera.camera.orientation = transition
                    .start_orientation
                    .slerp(&transition.end_orientation, t);
                self.camera.camera.target_distance = transition.start_distance
                    + (transition.end_distance - transition.start_distance) * t;
                self.camera.fov =
                    transition.start_fov + (transition.end_fov - transition.start_fov) * t;
                self.camera.world_up = transition
                    .start_world_up
                    .lerp(&transition.end_world_up, t)
                    .normalize();
                ui.ctx().request_repaint(); // keep animating
            }
        }
        // Handle transition completion separately to take ownership
        if self
            .target_transition
            .as_ref()
            .is_some_and(|t| t.progress(current_time).is_none())
        {
            let transition = self.target_transition.take().unwrap();
            self.camera.camera.position = transition.end_position;
            self.camera.camera.orientation = transition.end_orientation;
            self.camera.camera.target_distance = transition.end_distance;
            self.camera.fov = transition.end_fov;
            self.camera.world_up = transition.end_world_up;
            self.camera_view = transition.pending_camera_view;
            if transition.flash_on_complete {
                self.target_flash_start = Some(current_time);
            }
        }
    }

    /// Applies a depth pick result to set the orbit target with smooth animation.
    ///
    /// Called from `main.rs` after the depth readback completes. Instead of
    /// snapping instantly, starts an animated transition (slerp + lerp with
    /// smoothstep easing over ~200ms).
    pub fn apply_pick_result(&mut self, depth: f32, click_pixel: [u32; 2], current_time: f64) {
        if depth <= 0.0 {
            return;
        }

        let [px, py] = click_pixel;

        // Convert texture pixels back to screen coordinates for unprojection
        let screen_x = self.pick_rect.left() + px as f32 / self.pick_ppp;
        let screen_y = self.pick_rect.top() + py as f32 / self.pick_ppp;

        let world_point = self
            .camera
            .unproject(screen_x, screen_y, depth as f64, self.pick_rect);

        // Compute the end state: orientation and distance for the new target
        let direction = world_point - self.camera.camera.position;
        let new_distance = direction.norm();
        if new_distance < 1e-10 {
            return; // target too close to camera, would produce NaN
        }
        let new_forward = direction / new_distance;
        let new_distance = new_distance.max(0.1);

        let end_orientation = Camera::orientation_from_forward(new_forward, self.camera.world_up);

        self.start_transition(
            self.camera.camera.position,
            end_orientation,
            new_distance,
            self.camera.fov,
            self.camera.world_up,
            None,
            true,
            current_time,
        );
    }

    /// Starts a smooth animated transition to the given camera end state.
    ///
    /// Captures the current camera state as the start and interpolates over
    /// ~200ms. If a transition is already in progress, it is replaced (the
    /// current interpolated state becomes the new start).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_transition(
        &mut self,
        end_position: Point3<f64>,
        end_orientation: UnitQuaternion<f64>,
        end_distance: f64,
        end_fov: f64,
        end_world_up: Vector3<f64>,
        pending_camera_view: Option<CameraViewMode>,
        flash_on_complete: bool,
        current_time: f64,
    ) {
        self.target_transition = Some(CameraTransition {
            start_position: self.camera.camera.position,
            end_position,
            start_orientation: self.camera.camera.orientation,
            end_orientation,
            start_distance: self.camera.camera.target_distance,
            end_distance,
            start_fov: self.camera.fov,
            end_fov,
            start_world_up: self.camera.world_up,
            end_world_up,
            start_time: current_time,
            pending_camera_view,
            flash_on_complete,
        });
    }

    /// Cancels any in-progress camera transition, snapping to the current interpolated state.
    pub(crate) fn cancel_transition(&mut self) {
        self.target_transition = None;
    }

    /// Enter camera view mode with a smooth animated transition.
    ///
    /// Computes the target camera pose from the image's extrinsics, deactivates
    /// the current camera view (if any), and starts an animated transition.
    /// Camera view mode is activated when the transition completes.
    pub fn enter_camera_view(
        &mut self,
        img_idx: usize,
        reconstruction: &SfmrReconstruction,
        current_time: f64,
    ) {
        let image = &reconstruction.images[img_idx];
        let camera = &reconstruction.cameras[image.camera_index as usize];

        let r_world_from_cam = image.quaternion_wxyz.inverse();

        // Compute end state
        let end_position = image.camera_center();
        let flip = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI);
        let end_orientation = flip * image.quaternion_wxyz;

        let end_distance = reconstruction
            .depth_statistics
            .images
            .get(img_idx)
            .and_then(|stats| stats.observed.median_z)
            .unwrap_or(self.camera.camera.target_distance);

        // world_up = up direction of the end orientation
        let end_world_up = end_orientation.inverse() * Vector3::new(0.0, 1.0, 0.0);

        let end_fov = if !camera.model.is_fisheye() {
            let (fx, fy) = camera.focal_lengths();
            let vfov_cam = (camera.height as f64 / (2.0 * fy)).atan() * 2.0;
            let hfov_cam = (camera.width as f64 / (2.0 * fx)).atan() * 2.0;
            let aspect = if self.panel_size[0] > 0 && self.panel_size[1] > 0 {
                self.panel_size[0] as f64 / self.panel_size[1] as f64
            } else {
                16.0 / 9.0
            };
            best_fit_fov(vfov_cam, hfov_cam, aspect)
        } else {
            self.camera.fov
        };

        let pending = CameraViewMode {
            image_index: img_idx,
            r_world_from_cam,
        };

        // Deactivate current camera view during transition
        self.camera_view = None;

        self.start_transition(
            end_position,
            end_orientation,
            end_distance,
            end_fov,
            end_world_up,
            Some(pending),
            false,
            current_time,
        );
    }

    /// Computes the end state for switching camera view, preserving relative orientation.
    ///
    /// Returns `None` if not currently in camera view.
    fn compute_switch_camera_view(
        &self,
        new_img_idx: usize,
        reconstruction: &SfmrReconstruction,
    ) -> Option<SwitchCameraViewState> {
        let old_r_world_from_cam = self.camera_view.as_ref()?.r_world_from_cam;

        let new_image = &reconstruction.images[new_img_idx];
        let new_qwxyz = new_image.quaternion_wxyz;

        // Compute new orientation preserving relative viewing direction.
        //   new_orientation = orientation * old_r_world_from_cam * new_qwxyz
        let orientation = self.camera.camera.orientation * old_r_world_from_cam * new_qwxyz;
        let position = new_image.camera_center();
        let distance = reconstruction
            .depth_statistics
            .images
            .get(new_img_idx)
            .and_then(|stats| stats.observed.median_z)
            .unwrap_or(self.camera.camera.target_distance);

        // Transform world_up through the relative rotation between cameras
        let r_old_cam_from_world = old_r_world_from_cam.inverse();
        let r_world_from_new_cam = new_qwxyz.inverse();
        let up_in_old_cam = r_old_cam_from_world * self.camera.world_up;
        let world_up = (r_world_from_new_cam * up_in_old_cam).normalize();

        Some(SwitchCameraViewState {
            position,
            orientation,
            distance,
            world_up,
            camera_view: CameraViewMode {
                image_index: new_img_idx,
                r_world_from_cam: r_world_from_new_cam,
            },
        })
    }

    /// Switch from one camera view to another instantly, preserving relative orientation.
    ///
    /// Used by `,`/`.` keys for rapid camera switching.
    pub fn switch_camera_view(&mut self, new_img_idx: usize, reconstruction: &SfmrReconstruction) {
        let Some(state) = self.compute_switch_camera_view(new_img_idx, reconstruction) else {
            return;
        };
        self.camera.camera.position = state.position;
        self.camera.camera.orientation = state.orientation;
        self.camera.camera.target_distance = state.distance;
        self.camera.world_up = state.world_up;
        self.camera_view = Some(state.camera_view);
    }

    /// Switch from one camera view to another with a smooth animated transition.
    ///
    /// Used by double-click on frustum/image strip when already in camera view.
    pub fn animated_switch_camera_view(
        &mut self,
        new_img_idx: usize,
        reconstruction: &SfmrReconstruction,
        current_time: f64,
    ) {
        let Some(state) = self.compute_switch_camera_view(new_img_idx, reconstruction) else {
            self.enter_camera_view(new_img_idx, reconstruction, current_time);
            return;
        };
        self.camera_view = None;
        self.start_transition(
            state.position,
            state.orientation,
            state.distance,
            self.camera.fov,
            state.world_up,
            Some(state.camera_view),
            false,
            current_time,
        );
    }

    /// Updates target indicator state for GPU rendering.
    ///
    /// Advances the rotation animation and computes flash animation state.
    /// The actual rendering is done by `SceneRenderer::render_target_indicator`.
    fn update_target_indicator_state(&mut self, ui: &egui::Ui) {
        self.target_indicator_visible =
            self.alt_held || self.target_keep_visible || self.target_flash_start.is_some();

        if !self.target_indicator_visible {
            return;
        }

        // Advance rotation animation
        let dt = ui.input(|i| i.stable_dt) as f64;
        self.target_indicator_rotation += TARGET_ROTATION_SPEED * dt;
        if self.target_indicator_rotation > std::f64::consts::TAU {
            self.target_indicator_rotation -= std::f64::consts::TAU;
        }

        // Flash animation: expand radius and brighten on target change
        let current_time = ui.input(|i| i.time);
        let (radius_scale, alpha_scale) = if let Some(flash_start) = self.target_flash_start {
            let elapsed = current_time - flash_start;
            if elapsed > 0.3 {
                self.target_flash_start = None;
                (1.0_f32, 1.0_f32)
            } else {
                let t = (elapsed / 0.3) as f32;
                let ease_out = 1.0 - (1.0 - t) * (1.0 - t); // quadratic ease-out
                (1.0 + 0.5 * (1.0 - ease_out), 1.0 + 1.0 * (1.0 - ease_out))
            }
        } else {
            (1.0, 1.0)
        };

        self.target_indicator_radius_scale = radius_scale;
        self.target_indicator_alpha_scale = alpha_scale;

        // Request continuous repaint for rotation animation
        ui.ctx().request_repaint();
    }
}