// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! App method implementations.
//!
//! Contains the core rendering loop (`run_ui_and_paint`), GPU state
//! synchronization, readback processing, and platform-specific helpers.
//!
//! `run_ui_and_paint` is a thin orchestrator that wires four per-frame phases:
//! - [`App::prepare_uploads`] — sync GPU buffers/uniforms from app state.
//! - [`App::render_scene`] — encode the 3D scene render passes.
//! - [`App::run_egui_pass`] — run the egui/dock UI and tessellate.
//! - [`App::process_pick_readback`] — apply hover/selection from GPU pick.

#[cfg(target_os = "windows")]
use std::time::Instant;

use egui_dock::DockArea;
use egui_winit::State as EguiWinitState;
use sfmtool_core::SfmrReconstruction;
use winit::window::Window;

use crate::dock::{self, TabContext};
use crate::platform;
use crate::scene_renderer;
use crate::App;

#[cfg(target_os = "windows")]
use crate::platform::windows::WinGestureHandler;
#[cfg(target_os = "windows")]
use crate::DM_UPDATE_INTERVAL;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::HWND;

impl App {
    #[cfg(target_os = "windows")]
    pub(crate) fn window_hwnd(&self) -> Option<HWND> {
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};

        let window = self.window.as_ref()?;
        let window_handle = window.window_handle().ok()?;
        if let RawWindowHandle::Win32(win32) = window_handle.as_raw() {
            Some(HWND(win32.hwnd.get() as *mut std::ffi::c_void))
        } else {
            None
        }
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn try_init_gesture_handler(&mut self) {
        if self.gesture_handler.is_some() {
            return;
        }
        let Some(early_dm) = self.early_dm.as_ref() else {
            return;
        };
        let Some(hwnd) = self.window_hwnd() else {
            return;
        };

        match WinGestureHandler::new(hwnd, early_dm) {
            Ok(handler) => {
                log::info!("Windows precision touchpad gesture handler initialized");
                self.gesture_handler = Some(handler);
                self.early_dm = None;
                self.next_dm_update = Some(Instant::now() + DM_UPDATE_INTERVAL);
            }
            Err(e) => {
                log::warn!("Failed to initialize gesture handler: {:?}", e);
                self.early_dm = None;
            }
        }
    }

    /// Per-frame render + UI loop. Orchestrates the four phase methods below.
    pub(crate) fn run_ui_and_paint(&mut self) {
        // Bail unless every GPU resource is initialized. The phase helpers take
        // `&mut self`, so we clone the cheap Arc-backed device/queue handles up
        // front to avoid holding conflicting borrows of `self` across the calls.
        if self.window.is_none()
            || self.egui_winit_state.is_none()
            || self.wgpu_device.is_none()
            || self.wgpu_queue.is_none()
            || self.wgpu_surface.is_none()
            || self.wgpu_surface_config.is_none()
            || self.egui_renderer.is_none()
        {
            return;
        }
        let device = self.wgpu_device.clone().unwrap();
        let queue = self.wgpu_queue.clone().unwrap();
        // `Arc<Window>` is cheap to clone; owning it here (rather than borrowing
        // `self.window`) frees `self` for the `&mut self` phase methods below and
        // lets `run_egui_pass` take a non-`Option` `&Window`.
        let window = self.window.clone().unwrap();

        // Ensure scene texture and pipeline match the 3D panel size
        let [pw, ph] = self.viewer_3d.panel_size;
        if pw > 0 && ph > 0 {
            let renderer = self.egui_renderer.as_mut().unwrap();
            self.scene_renderer.ensure_size(&device, renderer, pw, ph);
        }

        // Phase 1: sync all GPU buffers/uniforms from the current app state.
        self.prepare_uploads(&device, &queue);

        // Phase 2: render the 3D scene into the offscreen texture. The encoder is
        // created here (not inside `render_scene`) because it is shared with the
        // egui pass below — both run in the same submission.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });
        self.render_scene(&queue, &mut encoder);

        // Phase 3: run the egui/dock UI, publish accessibility, and tessellate.
        // `egui_winit::State` is not `Clone` and `run_egui_pass` needs it `&mut`,
        // so move it out of `self` for the call and restore it afterward — this
        // hands the phase method a non-`Option` `&mut State` rather than relying
        // on it to re-unwrap a field the top-of-function guard already checked.
        let mut egui_winit_state = self.egui_winit_state.take().unwrap();
        let (clipped_primitives, textures_delta, pixels_per_point) =
            self.run_egui_pass(&window, &mut egui_winit_state);
        self.egui_winit_state = Some(egui_winit_state);

        // --- Acquire the surface, encode the egui pass, submit, and present. ---
        let renderer = self.egui_renderer.as_mut().unwrap();

        // Apply egui's texture set-deltas now, before acquiring the surface. Doing
        // this unconditionally keeps the renderer's texture set in sync with the
        // egui context even on frames we cannot present (see below); otherwise a
        // skipped `set` makes a later partial update panic with "texture has not
        // been allocated yet".
        for (id, image_delta) in &textures_delta.set {
            renderer.update_texture(&device, &queue, *id, image_delta);
        }

        let surface = self.wgpu_surface.as_ref().unwrap();
        let surface_config = self.wgpu_surface_config.as_ref().unwrap();

        // Acquire the surface texture only now, after the egui pass above has
        // already published the AccessKit tree via handle_platform_output. A
        // window that can't present its surface — e.g. occluded / off-screen on
        // a headless CI runner — still updates its accessibility tree each
        // frame; we just skip the GPU submit and present. Free released egui
        // textures before returning so the renderer stays in sync.
        let output = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(output)
            | wgpu::CurrentSurfaceTexture::Suboptimal(output) => output,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(&device, surface_config);
                for id in &textures_delta.free {
                    renderer.free_texture(id);
                }
                return;
            }
            other => {
                log::error!("wgpu surface error: {:?}", other);
                for id in &textures_delta.free {
                    renderer.free_texture(id);
                }
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());

        let screen_descriptor = eframe::egui_wgpu::ScreenDescriptor {
            size_in_pixels: [surface_config.width, surface_config.height],
            pixels_per_point,
        };

        // Update buffers and render (encoder was created earlier for the scene pass)
        let user_cmd_bufs = renderer.update_buffers(
            &device,
            &queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.12,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        renderer.render(
            &mut render_pass.forget_lifetime(),
            &clipped_primitives,
            &screen_descriptor,
        );

        // Submit
        let mut cmd_bufs: Vec<wgpu::CommandBuffer> = user_cmd_bufs;
        cmd_bufs.push(encoder.finish());
        queue.submit(cmd_bufs);
        output.present();

        // Phase 4: apply hover/selection from the 5x5 depth + pick readback.
        self.process_pick_readback(&device);

        // Free textures released by egui this frame.
        let renderer = self.egui_renderer.as_mut().unwrap();
        for id in &textures_delta.free {
            renderer.free_texture(id);
        }
    }

    /// Phase 1: upload/refresh all GPU buffers and uniforms from app state —
    /// point cloud, frustum geometry + colors, track rays, camera-view
    /// background image, adaptive clip planes, and per-frame camera uniforms.
    fn prepare_uploads(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Upload point cloud data to GPU if the reconstruction changed
        let hidden_image = self.viewer_3d.camera_view.as_ref().map(|cv| cv.image_index);
        if self.state.points_need_upload {
            if let Some(ref recon) = self.state.reconstruction {
                self.scene_renderer.upload_points(device, recon);
                let point_scale = scene_renderer::DEFAULT_LENGTH_SCALE_MULTIPLIER
                    * self.scene_renderer.auto_point_size();
                self.state.length_scale = match self.scene_renderer.camera_nn_scale() {
                    Some(cam_scale) => point_scale.min(cam_scale),
                    None => point_scale,
                };
                self.scene_renderer.upload_thumbnails(device, queue, recon);
                self.scene_renderer.upload_patches(device, queue, recon);
                self.scene_renderer.upload_frustums(
                    device,
                    recon,
                    self.state.length_scale,
                    self.state.frustum_size_multiplier,
                );
                let track_images = dock::compute_track_images(&self.state, recon);
                self.scene_renderer.update_frustum_colors(
                    queue,
                    recon.images.len(),
                    self.state.selected_image,
                    hidden_image,
                    &track_images,
                );
                self.prev_frustum_length_scale = self.state.length_scale;
                self.prev_frustum_size_multiplier = self.state.frustum_size_multiplier;
                self.prev_selected_image = self.state.selected_image;
                self.prev_selected_point = self.state.selected_point;
                self.prev_hidden_image = hidden_image;
            }
            self.state.points_need_upload = false;
        }

        // Re-upload frustum geometry only if length_scale or frustum_size_multiplier changed
        let geometry_changed = self.state.length_scale != self.prev_frustum_length_scale
            || self.state.frustum_size_multiplier != self.prev_frustum_size_multiplier;
        let point_selection_changed = self.state.selected_point != self.prev_selected_point;
        let colors_changed = self.state.selected_image != self.prev_selected_image
            || point_selection_changed
            || hidden_image != self.prev_hidden_image;
        if geometry_changed {
            if let Some(ref recon) = self.state.reconstruction {
                self.scene_renderer.upload_frustums(
                    device,
                    recon,
                    self.state.length_scale,
                    self.state.frustum_size_multiplier,
                );
                let track_images = dock::compute_track_images(&self.state, recon);
                self.scene_renderer.update_frustum_colors(
                    queue,
                    recon.images.len(),
                    self.state.selected_image,
                    hidden_image,
                    &track_images,
                );
                self.prev_frustum_length_scale = self.state.length_scale;
                self.prev_frustum_size_multiplier = self.state.frustum_size_multiplier;
                self.prev_selected_image = self.state.selected_image;
                self.prev_selected_point = self.state.selected_point;
                self.prev_hidden_image = hidden_image;
            }
        } else if colors_changed {
            // Only update colors — no geometry recomputation needed
            if let Some(ref recon) = self.state.reconstruction {
                let track_images = dock::compute_track_images(&self.state, recon);
                self.scene_renderer.update_frustum_colors(
                    queue,
                    recon.images.len(),
                    self.state.selected_image,
                    hidden_image,
                    &track_images,
                );
                self.prev_selected_image = self.state.selected_image;
                self.prev_selected_point = self.state.selected_point;
                self.prev_hidden_image = hidden_image;
            }
        }

        // Upload/clear track ray geometry when selected point changes
        if point_selection_changed {
            if let Some(ref recon) = self.state.reconstruction {
                if let Some(point_idx) = self.state.selected_point {
                    if point_idx < recon.points.len() {
                        // Pre-populate SIFT cache for all images in the track
                        // (sift_files only; embedded_patches has no `.sift`
                        // files and reads its keypoints inline).
                        if recon.feature_indexes().is_some() {
                            for obs in recon.observations_for_point(point_idx) {
                                let img_idx = obs.image_index as usize;
                                let read_count =
                                    recon.max_track_feature_index[img_idx] as usize + 1;
                                crate::state::ensure_sift_cached(
                                    &mut self.state.sift_cache,
                                    recon,
                                    img_idx,
                                    read_count,
                                );
                            }
                        }
                        self.scene_renderer.upload_track_rays(
                            device,
                            recon,
                            point_idx,
                            &self.state.sift_cache,
                        );
                    } else {
                        self.scene_renderer.clear_track_rays();
                    }
                } else {
                    self.scene_renderer.clear_track_rays();
                }
            }
        }

        // Upload/clear background image for camera view mode
        if let (Some(ref cv), Some(ref recon)) =
            (&self.viewer_3d.camera_view, &self.state.reconstruction)
        {
            self.scene_renderer
                .upload_bg_image(device, queue, recon, cv.image_index);
        } else {
            self.scene_renderer.clear_bg_image();
        }

        // Update adaptive clip planes based on scene bounds and camera distance.
        // Uses time-based smoothing so transitions are frame-rate independent.
        if self.state.reconstruction.is_some() {
            let dt = self.egui_ctx.input(|i| i.stable_dt as f64);
            self.viewer_3d.camera.update_clip_planes(
                self.scene_renderer.scene_center(),
                self.scene_renderer.scene_radius(),
                dt,
            );
        }

        // Update camera uniforms for the current frame
        let target_radius = self.state.target_size_multiplier
            * self.viewer_3d.target_indicator_radius_scale
            * self.state.length_scale;
        self.scene_renderer.update_uniforms(
            queue,
            &self.viewer_3d.camera,
            self.state.point_size_log2,
            self.state.infinity_point_px,
            self.state.edl_line_thickness,
            self.viewer_3d.supernova_view_pos,
            self.viewer_3d.supernova_active,
            target_radius,
            self.viewer_3d.supernova_time,
            self.state.selected_point,
            // Suppress hover highlight when equal to selection (spec requirement).
            self.state
                .hovered_point
                .filter(|h| self.state.selected_point != Some(*h)),
            self.state
                .hovered_image
                .filter(|h| self.state.selected_image != Some(*h)),
            self.state.patch_size_log2,
            self.state.patch_opacity,
            self.state.patch_alpha_cutoff,
        );

        // Update background image uniforms every frame in camera view (viewport
        // resize or free-look rotation changes the view_proj).
        if self.viewer_3d.camera_view.is_some() {
            self.scene_renderer
                .update_bg_image_uniforms(queue, &self.viewer_3d.camera);
        }
    }

    /// Phase 2: encode the 3D scene render passes (scene, target indicator,
    /// track rays) and the depth/pick readback copy into `encoder`.
    fn render_scene(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let in_camera_view = self.viewer_3d.camera_view.is_some();

        // Render the 3D scene to the offscreen texture
        self.scene_renderer.render(
            encoder,
            self.state.show_points,
            self.state.show_camera_images,
            // At zero opacity patches are invisible; skip the draw so they don't
            // still write depth/pick and swallow track rays and point clicks.
            self.state.show_patches && self.state.patch_opacity > 0.0,
            in_camera_view,
        );

        // Render target indicator (after EDL pass, blended onto EDL output)
        if self.viewer_3d.target_indicator_visible {
            let target = self.viewer_3d.camera.target();
            let target_pos = [target.x as f32, target.y as f32, target.z as f32];
            let world_up = self.viewer_3d.camera.world_up;
            let world_up_f32 = [world_up.x as f32, world_up.y as f32, world_up.z as f32];
            self.scene_renderer.update_target_uniforms(
                queue,
                &self.viewer_3d.camera,
                target_pos,
                self.viewer_3d.target_indicator_rotation as f32,
                world_up_f32,
                self.viewer_3d.target_indicator_alpha_scale,
                self.state.target_size_multiplier * self.viewer_3d.target_indicator_radius_scale,
                self.state.target_fog_multiplier,
                self.state.length_scale,
            );
            self.scene_renderer.render_target_indicator(encoder);
        }

        // Render track rays (after target indicator, also post-EDL)
        self.scene_renderer
            .update_track_ray_uniforms(queue, &self.viewer_3d.camera);
        self.scene_renderer.render_track_rays(encoder);

        // Copy 5x5 depth + pick region under the mouse (shared by hover + click)
        if let Some([px, py]) = self.viewer_3d.hover_pixel {
            self.scene_renderer.copy_readback_region(encoder, px, py);
        }
    }

    /// Phase 3: run the egui/dock UI for this frame, publish the AccessKit tree
    /// via `handle_platform_output`, and tessellate. Returns the tessellated
    /// primitives, the frame's texture deltas, and the pixels-per-point scale.
    fn run_egui_pass(
        &mut self,
        window: &Window,
        egui_winit_state: &mut EguiWinitState,
    ) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
        let scene_texture_id = self.scene_renderer.texture_id();
        let hover_depth = self.scene_renderer.hover_depth();
        let hover_pick_id = self.scene_renderer.hover_pick_id();

        let raw_input = egui_winit_state.take_egui_input(window);

        // Gather gesture events
        #[cfg(target_os = "windows")]
        let (gesture_events, diagnostics) = self
            .gesture_handler
            .as_ref()
            .map(|h| {
                let events = h.poll_events();
                if !events.is_empty() {
                    self.egui_ctx.request_repaint();
                }
                (events, Some(h.get_diagnostics()))
            })
            .unwrap_or_default();
        #[cfg(not(target_os = "windows"))]
        let (gesture_events, diagnostics) = (Vec::new(), None);

        #[cfg(target_os = "windows")]
        let handler_ok = self.gesture_handler.is_some();
        #[cfg(not(target_os = "windows"))]
        let handler_ok = false;

        let app_state = &mut self.state;
        let viewer_3d = &mut self.viewer_3d;
        let image_browser = &mut self.image_browser;
        let image_detail = &mut self.image_detail;
        let point_track_detail = &mut self.point_track_detail;
        let dock_state = &mut self.dock_state;

        let full_output = self.egui_ctx.run_ui(raw_input, |root_ui| {
            // Accumulate scroll events once per frame, with DM-aware suppression.
            let scroll_input = platform::ScrollInput::from_ctx(
                root_ui.ctx(),
                handler_ok && !gesture_events.is_empty(),
            );

            egui::Panel::top("menu_bar").show_inside(root_ui, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("Open...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("SfM Reconstruction", &["sfmr"])
                                .pick_file()
                            {
                                app_state.load_file(&path);
                            }
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("Load Demo Data...").clicked() {
                            app_state.show_demo_dialog = true;
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("Quit").clicked() {
                            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.menu_button("View", |ui| {
                        ui.checkbox(&mut app_state.show_points, "Show Points");
                        ui.checkbox(&mut app_state.show_camera_images, "Show Camera Images");
                        ui.checkbox(&mut app_state.show_grid, "Show Grid");
                        ui.separator();
                        ui.label("Point Size");
                        ui.add(
                            egui::Slider::new(&mut app_state.point_size_log2, -3.0..=3.0)
                                .text("log₂")
                                .fixed_decimals(1),
                        );
                        if ui.button("Reset Size").clicked() {
                            app_state.point_size_log2 = 0.0;
                        }
                        ui.separator();
                        ui.label("Infinity Point Size");
                        ui.add(
                            egui::Slider::new(&mut app_state.infinity_point_px, 1.0..=16.0)
                                .text("px")
                                .fixed_decimals(1),
                        );
                        ui.separator();
                        // Patch surfel controls — disabled unless the loaded
                        // reconstruction carries patch frames with bitmaps.
                        let has_patches = app_state.reconstruction.as_ref().is_some_and(|r| {
                            r.patch_u_halfvec_xyz.is_some()
                                && r.patch_v_halfvec_xyz.is_some()
                                && r.patch_bitmaps_y_x_rgba.is_some()
                        });
                        ui.add_enabled_ui(has_patches, |ui| {
                            ui.checkbox(&mut app_state.show_patches, "Show Patches");
                            ui.label("Patch Opacity");
                            ui.add(
                                egui::Slider::new(&mut app_state.patch_opacity, 0.0..=1.0)
                                    .fixed_decimals(2),
                            );
                            ui.label("Patch Size");
                            ui.add(
                                egui::Slider::new(&mut app_state.patch_size_log2, -3.0..=3.0)
                                    .text("log₂")
                                    .fixed_decimals(1),
                            );
                            ui.label("Patch Edge Cutoff");
                            ui.add(
                                egui::Slider::new(&mut app_state.patch_alpha_cutoff, 0.0..=1.0)
                                    .fixed_decimals(2),
                            );
                        });
                        ui.separator();
                        ui.label("Length Scale");
                        ui.add(
                            egui::Slider::new(&mut app_state.length_scale, 0.001..=100.0)
                                .logarithmic(true)
                                .fixed_decimals(3),
                        );
                        ui.separator();
                        ui.label("Field of View");
                        let mut fov_degrees = viewer_3d.camera.fov.to_degrees();
                        let response = ui.add(
                            egui::Slider::new(&mut fov_degrees, 10.0..=120.0)
                                .text("°")
                                .fixed_decimals(0),
                        );
                        if response.changed() {
                            viewer_3d.camera.fov = fov_degrees.to_radians();
                        }
                        if ui.button("Reset FOV").clicked() {
                            viewer_3d.camera.fov = std::f64::consts::FRAC_PI_4;
                        }
                    });
                });
            });

            if app_state.show_demo_dialog {
                let mut open = true;
                let mut load_clicked = false;
                egui::Window::new("Load Demo Data")
                    .open(&mut open)
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(root_ui.ctx(), |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Number of points:");
                            ui.add(
                                egui::DragValue::new(&mut app_state.demo_num_points)
                                    .range(1..=100_000)
                                    .speed(10.0),
                            );
                        });
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() {
                                load_clicked = true;
                            }
                            if ui.button("Cancel").clicked() {
                                app_state.show_demo_dialog = false;
                            }
                        });
                    });
                if !open {
                    app_state.show_demo_dialog = false;
                }
                if load_clicked {
                    app_state.reconstruction =
                        Some(SfmrReconstruction::demo(app_state.demo_num_points));
                    app_state.status_message = None;
                    app_state.points_need_upload = true;
                    app_state.show_demo_dialog = false;
                }
            }

            egui::CentralPanel::default().show_inside(root_ui, |ui| {
                let mut tab_context = TabContext {
                    state: app_state,
                    viewer_3d,
                    image_browser,
                    image_detail,
                    point_track_detail,
                    scene_texture_id,
                    hover_depth,
                    hover_pick_id,
                    gesture_events: &gesture_events,
                    scroll_input: &scroll_input,
                    diagnostics,
                    handler_ok,
                };
                DockArea::new(dock_state).show_inside(ui, &mut tab_context);
            });
        });

        egui_winit_state.handle_platform_output(window, full_output.platform_output);

        // Tessellate now so the caller only has to update textures + present.
        let pixels_per_point = full_output.pixels_per_point;
        let clipped_primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, pixels_per_point);
        (
            clipped_primitives,
            full_output.textures_delta,
            pixels_per_point,
        )
    }

    /// Phase 4: read back the 5x5 depth + pick region (shared by hover + click)
    /// and update transient hover state plus pending-click selection.
    fn process_pick_readback(&mut self, device: &wgpu::Device) {
        let Some(readback) = self.scene_renderer.read_readback_result(device) else {
            return;
        };

        // Update transient hover state from GPU pick buffer.
        // Only when the 3D viewer has pointer focus (hover_pixel is set for
        // the current frame). This avoids stale one-frame-delayed readback
        // results from overwriting hover state after the pointer left.
        if self.viewer_3d.hover_pixel.is_some() {
            if let Some((tag, index)) = readback.pick {
                if tag == scene_renderer::PICK_TAG_FRUSTUM {
                    self.state.hovered_image = Some(index as usize);
                    self.state.hovered_point = None;
                } else if tag == scene_renderer::PICK_TAG_POINT {
                    self.state.hovered_point = Some(index as usize);
                    self.state.hovered_image = None;
                } else {
                    self.state.hovered_image = None;
                    self.state.hovered_point = None;
                }
            } else {
                self.state.hovered_image = None;
                self.state.hovered_point = None;
            }
        }

        // Handle click using the same readback result
        if let Some(click_pixel) = self.viewer_3d.pending_click.take() {
            // Alt+Click: set orbit target from depth
            if self.viewer_3d.pending_click_is_alt {
                if let Some(depth) = readback.depth {
                    let current_time = self.egui_ctx.input(|i| i.time);
                    self.viewer_3d
                        .apply_pick_result(depth, click_pixel, current_time);
                }
            }

            // Entity pick: select frustum or point
            if let Some((tag, index)) = readback.pick {
                if tag == scene_renderer::PICK_TAG_FRUSTUM {
                    let idx = index as usize;
                    if self.viewer_3d.pending_click_is_double {
                        // Double-click on frustum → enter/switch camera view mode
                        self.state.selected_image = Some(idx);
                        if let Some(ref recon) = self.state.reconstruction {
                            let current_time = self.egui_ctx.input(|i| i.time);
                            if self.viewer_3d.camera_view.is_some() {
                                self.viewer_3d.animated_switch_camera_view(
                                    idx,
                                    recon,
                                    current_time,
                                );
                            } else {
                                self.viewer_3d.enter_camera_view(idx, recon, current_time);
                            }
                        }
                    } else {
                        self.state.selected_image = Some(idx);
                    }
                } else if tag == scene_renderer::PICK_TAG_POINT {
                    let idx = index as usize;
                    self.state.selected_point = Some(idx);
                }
            } else if !self.viewer_3d.pending_click_is_alt {
                // Clicked on background (non-Alt) — deselect
                self.state.selected_image = None;
                self.state.selected_point = None;
            }
        }
    }
}
