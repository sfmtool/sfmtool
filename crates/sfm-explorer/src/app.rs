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
use winit::window::Window;

use crate::dock::{self, Tab, TabContext};
use crate::goto_point;
use crate::platform;
use crate::scene::ImageRef;
use crate::scene_renderer::{NodeDisplay, PickTarget};
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

        // Keep the window title in step with the loaded file. Compared against
        // the last applied title rather than set unconditionally: `set_title`
        // is a window-manager round-trip, and this runs every frame.
        let title = self.state.window_title();
        if title != self.applied_title {
            window.set_title(&title);
            self.applied_title = title;
        }

        // Camera view is keyed by `ImageRef`, so replacing the loaded file
        // leaves it pointing at a reconstruction that is no longer in the
        // scene. Drop it rather than let it address nothing (the spec's
        // "clears any camera-view state pointing into it" on node removal).
        if self
            .viewer_3d
            .camera_view
            .as_ref()
            .is_some_and(|cv| self.state.node(cv.image.recon).is_none())
        {
            self.viewer_3d.camera_view = None;
        }

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
        // `mut` only so the delta can be `clear`ed once handled: since epaint
        // 0.36 a `TexturesDelta` debug-asserts on drop that nothing was left
        // unapplied, and every path out of this function below has to say so.
        let (clipped_primitives, mut textures_delta, pixels_per_point) =
            self.run_egui_pass(&window, &mut egui_winit_state);
        self.egui_winit_state = Some(egui_winit_state);

        // --- Acquire the surface, encode the egui pass, submit, and present. ---
        let renderer = self.egui_renderer.as_mut().unwrap();

        // Apply egui's texture set-deltas now, before acquiring the surface. Doing
        // this unconditionally keeps the renderer's texture set in sync with the
        // egui context even on frames we cannot present (see below); otherwise a
        // skipped `set` makes a later partial update panic with "texture has not
        // been allocated yet".
        for (id, image_deltas) in &textures_delta.set {
            // egui 0.36 batches a frame's deltas per texture; they are ordered
            // and must be applied in sequence (a full `set` followed by partial
            // patches on top of it).
            for image_delta in image_deltas {
                renderer.update_texture(&device, &queue, *id, image_delta);
            }
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
                textures_delta.clear();
                return;
            }
            other => {
                log::error!("wgpu surface error: {:?}", other);
                for id in &textures_delta.free {
                    renderer.free_texture(id);
                }
                textures_delta.clear();
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
        queue.present(output);

        // Phase 4: apply hover/selection from the 5x5 depth + pick readback.
        self.process_pick_readback(&device);

        // Free textures released by egui this frame.
        let renderer = self.egui_renderer.as_mut().unwrap();
        for id in &textures_delta.free {
            renderer.free_texture(id);
        }
        textures_delta.clear();
    }

    /// Phase 1: upload/refresh all GPU buffers and uniforms from app state —
    /// point cloud, frustum geometry + colors, track rays, camera-view
    /// background image, adaptive clip planes, and per-frame camera uniforms.
    ///
    /// Everything node-shaped here is a loop over `state.scene`, keyed by
    /// `ReconId`. The scene still holds at most one node (phase 3 lifts that),
    /// so the loops run once — but the renderer no longer assumes it.
    fn prepare_uploads(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // A node that has left the scene takes its GPU bundle with it.
        let scene = &self.state.scene;
        self.scene_renderer
            .retain_nodes(|id| scene.iter().any(|n| n.id == id));

        let hidden_image = self.viewer_3d.camera_view.as_ref().map(|cv| cv.image);

        // Per-node data upload, consuming each node's `needs_upload`.
        let mut uploaded_any = false;
        for i in 0..self.state.scene.len() {
            if !self.state.scene[i].needs_upload {
                continue;
            }
            let id = self.state.scene[i].id;
            let recon = &self.state.scene[i].recon;
            self.scene_renderer.upload_points(device, id, recon);
            self.scene_renderer
                .upload_thumbnails(device, queue, id, recon);
            self.scene_renderer.upload_patches(device, queue, id, recon);
            self.state.scene[i].needs_upload = false;
            uploaded_any = true;
        }

        // Mirror each node's Scene-panel display state onto its bundle. After
        // the uploads above, so a node loaded this frame already has a bundle
        // to carry it. This is how effective visibility, the interaction cursor
        // and the tint reach the draw loop and the per-recon uniform write: the
        // renderer ANDs these with the global HUD toggles and never looks at
        // the scene itself.
        //
        // `visible` is the one composed value: the node's own eye AND the
        // scene's solo override, resolved here by `scene::is_visible` so the
        // draw filter, the bounds union and the stats overlay are all reading
        // the same rule rather than three copies of it.
        //
        // The node transform rides along in the same sync: from the bundle it
        // becomes the per-recon `model` matrix, scales the node's splat size,
        // and moves its bounding sphere into the union.
        {
            let renderer = &mut self.scene_renderer;
            let solo = self.state.solo;
            for node in &self.state.scene {
                renderer.set_node_display(
                    node.id,
                    NodeDisplay {
                        visible: crate::scene::is_visible(node, solo),
                        show_points: node.show_points,
                        show_camera_images: node.show_camera_images,
                        show_patches: node.show_patches,
                        show_points_at_infinity: node.show_points_at_infinity,
                        interactive: node.interactive,
                        tint: node.tint,
                    },
                );
                renderer.set_node_transform(node.id, node.transform.clone());
            }
        }

        // Setting or resetting a transform is a world-space change, so it
        // re-derives `length_scale` and re-sizes frustum geometry exactly as a
        // fresh upload does.
        let transform_changed = self.state.transform_epoch != self.prev_transform_epoch;
        self.prev_transform_epoch = self.state.transform_epoch;

        // `length_scale` is global and re-derived from the union of the loaded
        // nodes, exactly as it is re-derived on load today. Frustum geometry
        // below depends on it, so it has to settle before that upload.
        if uploaded_any || transform_changed {
            if let Some(seed) = self.scene_renderer.length_scale_seed() {
                self.state.length_scale = seed;
            }
        }

        // Re-upload frustum geometry if length_scale or frustum_size_multiplier
        // changed — including the change a fresh upload just made.
        let geometry_changed = uploaded_any
            || transform_changed
            || self.state.length_scale != self.prev_frustum_length_scale
            || self.state.frustum_size_multiplier != self.prev_frustum_size_multiplier;
        let point_selection_changed = self.state.selected_point != self.prev_selected_point;
        let colors_changed = self.state.selected_image != self.prev_selected_image
            || self.state.selected_camera != self.prev_selected_camera
            || point_selection_changed
            || hidden_image != self.prev_hidden_image;
        if geometry_changed || colors_changed {
            for node in &self.state.scene {
                let id = node.id;
                if geometry_changed {
                    // Frustum stubs are built in the node's own coordinates and
                    // then scaled by its `model` matrix, so the length passed in
                    // is divided by the node's scale — what reaches the screen
                    // is `length_scale` in *world* units, whatever frame the
                    // node was solved in.
                    let scale = node.transform.scale as f32;
                    let node_length_scale = if scale > 0.0 && scale.is_finite() {
                        self.state.length_scale / scale
                    } else {
                        self.state.length_scale
                    };
                    self.scene_renderer.upload_frustums(
                        device,
                        id,
                        &node.recon,
                        node_length_scale,
                        self.state.frustum_size_multiplier,
                    );
                }
                // Colors index the owning node's own buffer, so these stay
                // local indices.
                let track_images = dock::compute_track_images(&self.state, node);
                let sibling_images =
                    crate::scene::camera_sibling_images(node, self.state.selected_camera);
                self.scene_renderer.update_frustum_colors(
                    queue,
                    id,
                    node.recon.images.len(),
                    self.state.selected_image_in(id),
                    hidden_image.and_then(|h| h.index_in(id)),
                    &track_images,
                    &sibling_images,
                );
            }
            self.prev_frustum_length_scale = self.state.length_scale;
            self.prev_frustum_size_multiplier = self.state.frustum_size_multiplier;
            self.prev_selected_image = self.state.selected_image;
            self.prev_selected_camera = self.state.selected_camera;
            self.prev_selected_point = self.state.selected_point;
            self.prev_hidden_image = hidden_image;
        }

        // Upload/clear track ray geometry when the selected point changes, or
        // when a transform moved the node the selection lives in. Track rays are
        // a singleton serving the single selection, built from the node that
        // owns the selected point — and, having no per-recon `model` matrix of
        // their own, they are built through that node's transform on the CPU.
        if point_selection_changed || transform_changed {
            let selected = self
                .state
                .selected_point
                .and_then(|p| Some((p, crate::scene::node_by_id(&self.state.scene, p.recon)?)));
            match selected {
                Some((point, node)) if point.index() < node.recon.points.len() => {
                    let (id, recon, transform) = (node.id, &node.recon, node.transform.clone());
                    let point_idx = point.index();
                    // Pre-populate SIFT cache for all images in the track
                    // (sift_files only; embedded_patches has no `.sift`
                    // files and reads its keypoints inline).
                    if recon.feature_indexes().is_some() {
                        for obs in recon.observations_for_point(point_idx) {
                            let img_idx = obs.image_index as usize;
                            let read_count = recon.max_track_feature_index[img_idx] as usize + 1;
                            crate::state::ensure_sift_cached(
                                &mut self.state.sift_cache,
                                recon,
                                ImageRef::new(id, img_idx),
                                read_count,
                            );
                        }
                    }
                    self.scene_renderer.upload_track_rays(
                        device,
                        recon,
                        point,
                        &self.state.sift_cache,
                        &transform,
                    );
                }
                _ => self.scene_renderer.clear_track_rays(),
            }
        }

        // Upload/clear background image for camera view mode. The background is
        // a singleton too, serving the camera view's own node — whose transform
        // the uniform update below needs, the mesh being built in that node's
        // own coordinates.
        let bg_transform =
            match hidden_image.and_then(|image| Some((image, self.state.node(image.recon)?))) {
                Some((image, node)) => {
                    let transform = node.transform.clone();
                    self.scene_renderer
                        .upload_bg_image(device, queue, &node.recon, image);
                    Some(transform)
                }
                None => {
                    self.scene_renderer.clear_bg_image();
                    None
                }
            };

        // Update adaptive clip planes from the union of the loaded nodes'
        // bounds. Uses time-based smoothing so transitions are frame-rate
        // independent.
        if !self.state.scene.is_empty() {
            let dt = self.egui_ctx.input(|i| i.stable_dt as f64);
            let (center, radius) = self.scene_renderer.scene_bounds();
            self.viewer_3d.camera.update_clip_planes(center, radius, dt);
        }

        // Update camera uniforms for the current frame. Selection and hover go
        // in as refs: the renderer knows each node's pick base, so it is the
        // one place that can turn a ref into the global index the shaders
        // compare against.
        let target_radius = self.state.target_size_multiplier
            * self.viewer_3d.target_indicator_radius_scale
            * self.state.length_scale;
        self.scene_renderer.update_uniforms(
            queue,
            &self.viewer_3d.camera,
            self.state.point_size_log2,
            self.state.infinity_point_px,
            self.state.show_points_at_infinity,
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
        // resize or free-look rotation changes the view_proj). Writing the node
        // transform here rather than baking it into the mesh is what keeps an
        // `Align to…` while camera view is open from leaving a stale background.
        if let Some(transform) = bg_transform.filter(|_| self.viewer_3d.camera_view.is_some()) {
            self.scene_renderer
                .update_bg_image_uniforms(queue, &self.viewer_3d.camera, &transform);
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
        let hover_pick = self.scene_renderer.hover_pick();

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
        let scene_graph = &mut self.scene_graph;
        let image_browser = &mut self.image_browser;
        let image_detail = &mut self.image_detail;
        let point_track_detail = &mut self.point_track_detail;
        let intrinsics_detail = &mut self.intrinsics_detail;
        let dock_state = &mut self.dock_state;

        let mut quit_requested = false;

        let full_output = self.egui_ctx.run_ui(raw_input, |root_ui| {
            // Accumulate scroll events once per frame, with DM-aware suppression.
            let scroll_input = platform::ScrollInput::from_ctx(
                root_ui.ctx(),
                handler_ok && !gesture_events.is_empty(),
            );

            egui::Panel::top("menu_bar").show(root_ui, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("Open...").clicked() {
                            // Multi-select, and every chosen file *appends* a
                            // node. Re-opening a loaded path reloads it in
                            // place instead (see `AppState::load_file`).
                            if let Some(paths) = rfd::FileDialog::new()
                                .add_filter("SfM Reconstruction", &["sfmr"])
                                .pick_files()
                            {
                                for path in paths {
                                    app_state.load_file(&path);
                                }
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                !app_state.scene.is_empty(),
                                egui::Button::new("Close All"),
                            )
                            .clicked()
                        {
                            for node in &app_state.scene {
                                let id = node.id;
                                image_browser.forget_recon(id);
                                image_detail.forget_recon(id);
                                point_track_detail.forget_recon(id);
                                intrinsics_detail.forget_recon(id);
                            }
                            app_state.close_all();
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("Load Demo Data...").clicked() {
                            app_state.show_demo_dialog = true;
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("Quit").clicked() {
                            // Not `send_viewport_cmd(ViewportCommand::Close)`:
                            // this app drives its own winit loop and never
                            // reads `full_output.viewport_output`, so the
                            // command was silently dropped and Quit did
                            // nothing. The flag is read straight after the
                            // egui pass, where the event loop can act on it.
                            quit_requested = true;
                            ui.close();
                        }
                    });
                    ui.menu_button("Go", |ui| {
                        if ui
                            .add(
                                egui::Button::new("Go to Point...")
                                    .shortcut_text(ui.ctx().format_shortcut(&goto_point::SHORTCUT)),
                            )
                            .clicked()
                        {
                            app_state.open_goto_point();
                            ui.close();
                        }
                    });
                    // No View menu: the display controls it used to hold belong
                    // to the 3D viewport's own HUD (`viewer_3d/hud.rs`), on the
                    // principle that a panel owns its controls, and the dock
                    // panels are all permanent (`TabViewer::closeable` is
                    // false), so there is nothing app-global left for it.
                });
            });

            // Ctrl/Cmd+G opens the same dialog from anywhere, gated on egui's
            // own keyboard arbitration so a HUD `DragValue` — or the dialog's
            // own text field — keeps the key while it is being typed into.
            // `open` is idempotent, so racing the menu item is harmless.
            if !root_ui.ctx().egui_wants_keyboard_input()
                && root_ui.input_mut(|i| i.consume_shortcut(&goto_point::SHORTCUT))
            {
                app_state.open_goto_point();
            }

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
                    // Same node-creation path as File > Open, so the demo load
                    // resets the caches and selection too.
                    app_state.load_demo(app_state.demo_num_points);
                    app_state.show_demo_dialog = false;
                }
            }

            // Go to Point. `select_point` also selects the owning
            // reconstruction, so a pasted ID naming a *different* loaded file
            // moves the whole session there — which is what makes an ID copied
            // out of one session usable in the next.
            if let Some(point) =
                app_state
                    .goto_point
                    .show(root_ui.ctx(), &app_state.scene, app_state.selected_recon)
            {
                app_state.select_point(point);
                // Raise the panel that answers "what is this point?", so the
                // jump has something to show for itself even when Point Track
                // is tabbed behind Image Detail (which is the default layout).
                if let Some(path) = dock_state.find_tab(&Tab::PointTrackDetail) {
                    let _ = dock_state.set_active_tab(path);
                }
            }

            egui::CentralPanel::default().show(root_ui, |ui| {
                let mut tab_context = TabContext {
                    state: app_state,
                    viewer_3d,
                    scene_graph,
                    image_browser,
                    image_detail,
                    point_track_detail,
                    intrinsics_detail,
                    scene_texture_id,
                    hover_depth,
                    hover_pick,
                    gesture_events: &gesture_events,
                    scroll_input: &scroll_input,
                    diagnostics,
                    handler_ok,
                };
                DockArea::new(dock_state).show_inside(ui, &mut tab_context);
            });
        });

        self.quit_requested |= quit_requested;

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

        // `readback.pick` arrives already decoded: the pick id carries a 2-bit
        // tag and a global index, and the renderer's sorted base tables turn
        // that back into a ref naming its own reconstruction.

        // Update transient hover state from GPU pick buffer.
        // Only when the 3D viewer has pointer focus (hover_pixel is set for
        // the current frame). This avoids stale one-frame-delayed readback
        // results from overwriting hover state after the pointer left.
        if self.viewer_3d.hover_pixel.is_some() {
            match readback.pick {
                Some(PickTarget::Image(image)) => {
                    self.state.hovered_image = Some(image);
                    self.state.hovered_point = None;
                }
                Some(PickTarget::Point(point)) => {
                    self.state.hovered_point = Some(point);
                    self.state.hovered_image = None;
                }
                None => {
                    self.state.hovered_image = None;
                    self.state.hovered_point = None;
                }
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
            match readback.pick {
                Some(PickTarget::Image(image)) => {
                    // Selecting an entity selects its reconstruction too — the
                    // viewport can pick into any visible node, so this is where
                    // a click can move the selection between files.
                    self.state.select_image(Some(image));
                    if self.viewer_3d.pending_click_is_double {
                        // Double-click on frustum → enter/switch camera view mode
                        if let Some(node) = crate::scene::node_by_id(&self.state.scene, image.recon)
                        {
                            let current_time = self.egui_ctx.input(|i| i.time);
                            if self.viewer_3d.camera_view.is_some() {
                                self.viewer_3d.animated_switch_camera_view(
                                    image,
                                    node,
                                    current_time,
                                );
                            } else {
                                self.viewer_3d.enter_camera_view(image, node, current_time);
                            }
                        }
                    }
                }
                Some(PickTarget::Point(point)) => {
                    self.state.select_point(point);
                }
                None if !self.viewer_3d.pending_click_is_alt => {
                    // Clicked on background (non-Alt) — deselect. The image
                    // goes; the camera it named stays, per the coupling rule
                    // in `AppState::select_image`.
                    self.state.select_image(None);
                    self.state.selected_point = None;
                }
                None => {}
            }
        }
    }
}
