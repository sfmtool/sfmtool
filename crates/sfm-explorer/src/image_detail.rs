// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image detail panel — full-resolution image display for the selected camera,
//! with SIFT feature overlays and heatmap visualization modes.

use crate::colormap;
use crate::platform::{self, GestureEvent, ScrollInput};
use crate::state::{CachedSiftFeatures, FeatureDisplaySettings, OverlayMode};
use kiddo::SquaredEuclidean;
use sfmtool_core::SfmrReconstruction;

/// Drag zoom speed: maps pixel delta to zoom factor.
const DRAG_ZOOM_SPEED: f32 = 0.005;
/// Mouse wheel zoom speed: maps line delta to zoom factor.
const MOUSE_WHEEL_ZOOM_SPEED: f32 = 0.15;
/// Trackpad scroll zoom speed (for Ctrl+scroll zoom).
const TRACKPAD_ZOOM_SPEED: f32 = 0.01;
/// Maximum zoom level (32× = pixel-level inspection).
const MAX_ZOOM: f32 = 32.0;
/// Minimum overlap in pixels between image and panel when panning.
const PAN_MARGIN: f32 = 50.0;

/// Prepared feature overlay state for the current image in the detail panel.
struct FeatureOverlayState {
    image_idx: usize,
    overlay_mode: OverlayMode,
    tracked_only: bool,
    max_features: Option<usize>,
    min_feature_size: Option<f32>,
    max_feature_size: Option<f32>,
    features: Vec<DisplayFeature>,
    tree: kiddo::KdTree<f32, 2>,
}

/// Image detail panel state.
pub struct ImageDetail {
    /// Currently loaded full-res image: (image_index, texture_handle).
    loaded_image: Option<(usize, egui::TextureHandle)>,
    /// Prepared feature overlay for the current image.
    feature_overlay: Option<FeatureOverlayState>,
    /// Offset of image center from panel center, in panel pixels.
    pan: egui::Vec2,
    /// Zoom level. 1.0 = fit image to panel. >1.0 = zoomed in.
    zoom: f32,
    /// Previously displayed image index, for detecting changes and resetting view.
    prev_selected_image: Option<usize>,
}

/// A feature to draw on the image detail panel.
struct DisplayFeature {
    /// Feature position in image pixel coordinates (x, y).
    position: [f32; 2],
    /// 2x2 affine shape matrix [[a11, a12], [a21, a22]].
    affine_shape: [[f32; 2]; 2],
    /// The 3D point index this feature maps to, or `u32::MAX` if untracked.
    point_index: u32,
}

impl DisplayFeature {
    fn is_tracked(&self) -> bool {
        self.point_index != u32::MAX
    }
}

/// Sentinel value for untracked features.
const UNTRACKED: u32 = u32::MAX;

/// Response from the image detail panel.
pub struct ImageDetailResponse {
    /// If Some, the user clicked a feature — select this 3D point.
    pub select_point: Option<usize>,
    /// Point index currently under the pointer (for cross-panel hover).
    pub hovered_point: Option<usize>,
    /// Whether the pointer is currently inside the detail panel.
    pub has_pointer: bool,
}

impl ImageDetail {
    pub fn new() -> Self {
        Self {
            loaded_image: None,
            feature_overlay: None,
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
            prev_selected_image: None,
        }
    }

    /// Reset pan and zoom to fit the image in the panel.
    fn reset_view(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.zoom = 1.0;
    }

    /// Apply zoom centered at a cursor position (in panel coordinates relative to panel center).
    fn zoom_at(&mut self, zoom_factor: f32, cursor_rel: egui::Vec2) {
        let old_zoom = self.zoom;
        self.zoom = (self.zoom * zoom_factor).clamp(1.0, MAX_ZOOM);
        let ratio = self.zoom / old_zoom;
        // Adjust pan so the point under the cursor stays fixed.
        self.pan = self.pan * ratio + cursor_rel * (1.0 - ratio);
    }

    /// Clamp pan so the image overlaps the panel by at least PAN_MARGIN pixels.
    fn clamp_pan(&mut self, display_size: egui::Vec2, panel_size: egui::Vec2) {
        let max_pan_x = (display_size.x + panel_size.x) / 2.0 - PAN_MARGIN;
        let max_pan_y = (display_size.y + panel_size.y) / 2.0 - PAN_MARGIN;
        self.pan.x = self.pan.x.clamp(-max_pan_x, max_pan_x);
        self.pan.y = self.pan.y.clamp(-max_pan_y, max_pan_y);
    }

    /// Show the image detail panel.
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        selected_image: Option<usize>,
        selected_point: Option<usize>,
        hovered_point: Option<usize>,
        preserve_view: bool,
        gesture_events: &[GestureEvent],
        scroll_input: &ScrollInput,
        sift_features: Option<&CachedSiftFeatures>,
        feature_display: &FeatureDisplaySettings,
    ) -> ImageDetailResponse {
        let mut response = ImageDetailResponse {
            select_point: None,
            hovered_point: None,
            has_pointer: false,
        };

        // If no image selected, show placeholder
        let Some(img_idx) = selected_image else {
            ui.centered_and_justified(|ui| {
                ui.label("No image selected");
            });
            if self.loaded_image.is_some() {
                self.loaded_image = None;
                self.feature_overlay = None;
            }
            self.prev_selected_image = None;
            return response;
        };

        // Reset view when selected image changes, unless the caller requests
        // preserving pan/zoom (e.g. during animation playback).
        if self.prev_selected_image != Some(img_idx) {
            if !preserve_view {
                self.reset_view();
            }
            self.prev_selected_image = Some(img_idx);
        }

        // Load the full-resolution image if it changed
        if self.loaded_image.as_ref().map(|(i, _)| *i) != Some(img_idx) {
            self.load_image(ui.ctx(), recon, img_idx);
            self.feature_overlay = None; // reset overlay on image change
        }

        // Determine whether to show features based on overlay mode
        let show_features = feature_display.overlay_mode != OverlayMode::None;

        // Rebuild overlay if settings changed (mode, filters, etc.)
        let cache_valid = self.feature_overlay.as_ref().is_some_and(|c| {
            c.image_idx == img_idx
                && c.overlay_mode == feature_display.overlay_mode
                && c.tracked_only == feature_display.tracked_only
                && c.max_features == feature_display.max_features
                && c.min_feature_size == feature_display.min_feature_size
                && c.max_feature_size == feature_display.max_feature_size
        });
        if show_features && !cache_valid {
            self.load_display_features(recon, img_idx, sift_features, feature_display);
        } else if !show_features {
            // In None mode, still load tracked features for selected point display
            let tracked_overlay_valid = self
                .feature_overlay
                .as_ref()
                .is_some_and(|c| c.image_idx == img_idx && c.tracked_only);
            if !tracked_overlay_valid {
                self.load_tracked_features(recon, img_idx, sift_features);
            }
        }

        // Display the image fitted to the panel
        let Some((_, ref texture)) = self.loaded_image else {
            ui.centered_and_justified(|ui| {
                ui.label("Failed to load image");
            });
            return response;
        };

        let tex_size = texture.size_vec2();
        let panel_rect = ui.available_rect_before_wrap();
        let panel_size = panel_rect.size();
        let panel_center = panel_rect.center();

        // Base scale: fits the image to the panel at zoom=1.0
        let base_scale = (panel_size.x / tex_size.x).min(panel_size.y / tex_size.y);
        let effective_scale = base_scale * self.zoom;
        let display_size = egui::vec2(tex_size.x * effective_scale, tex_size.y * effective_scale);

        // Image rect with pan offset
        let image_center = panel_center + self.pan;
        let image_rect = egui::Rect::from_center_size(image_center, display_size);

        // Allocate the full panel rect for interaction (not just the image rect),
        // so we can handle scroll/drag even when the image is smaller than the panel.
        let interact_rect = panel_rect;
        let interact_id = ui.id().with("image_detail_interact");
        let interact_response =
            ui.interact(interact_rect, interact_id, egui::Sense::click_and_drag());
        response.has_pointer = interact_response.hovered();

        // Draw the image (clipped to panel)
        let painter = ui.painter_at(panel_rect);
        painter.image(
            texture.id(),
            image_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // --- Input handling ---

        let pointer_over = platform::pointer_in_rect(ui.ctx(), panel_rect);

        // Double-click to reset view
        if interact_response.double_clicked() {
            self.reset_view();
            // Don't process this as a feature click
            return response;
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

        // Recompute image rect after pan/zoom changes from input
        let effective_scale = base_scale * self.zoom;
        let display_size = egui::vec2(tex_size.x * effective_scale, tex_size.y * effective_scale);
        let image_center = panel_center + self.pan;
        let image_rect = egui::Rect::from_center_size(image_center, display_size);

        // --- Feature overlays ---

        if let Some(ref overlay) = self.feature_overlay {
            let features = &overlay.features;
            let feature_tree = &overlay.tree;
            let image_to_panel = |px: f32, py: f32| -> egui::Pos2 {
                egui::pos2(
                    image_rect.min.x + px * effective_scale,
                    image_rect.min.y + py * effective_scale,
                )
            };
            let panel_to_image = |pos: egui::Pos2| -> [f32; 2] {
                [
                    (pos.x - image_rect.min.x) / effective_scale,
                    (pos.y - image_rect.min.y) / effective_scale,
                ]
            };

            match feature_display.overlay_mode {
                OverlayMode::None => {
                    // In None mode, only draw the selected point's feature (if any)
                    if let Some(sel_point) = selected_point {
                        for feature in features {
                            if feature.is_tracked() && feature.point_index as usize == sel_point {
                                let center =
                                    image_to_panel(feature.position[0], feature.position[1]);
                                draw_feature_ellipse(
                                    &painter,
                                    center,
                                    &feature.affine_shape,
                                    effective_scale,
                                    egui::Color32::YELLOW,
                                    2.0,
                                );
                                painter.circle_filled(center, 4.0, egui::Color32::YELLOW);
                            }
                        }
                    }
                }
                OverlayMode::Features => {
                    // Draw all features: green (tracked) or gray (untracked)
                    for feature in features {
                        let center = image_to_panel(feature.position[0], feature.position[1]);
                        if !panel_rect.expand(20.0).contains(center) {
                            continue;
                        }
                        let is_selected = feature.is_tracked()
                            && selected_point == Some(feature.point_index as usize);

                        if is_selected {
                            draw_feature_ellipse(
                                &painter,
                                center,
                                &feature.affine_shape,
                                effective_scale,
                                egui::Color32::YELLOW,
                                2.0,
                            );
                            painter.circle_filled(center, 4.0, egui::Color32::YELLOW);
                        } else if feature.is_tracked() {
                            draw_feature_ellipse(
                                &painter,
                                center,
                                &feature.affine_shape,
                                effective_scale,
                                egui::Color32::from_rgb(0, 200, 0),
                                1.0,
                            );
                            painter.circle_filled(center, 2.0, egui::Color32::from_rgb(220, 0, 0));
                        } else {
                            // Untracked: gray ellipse, no center dot
                            draw_feature_ellipse(
                                &painter,
                                center,
                                &feature.affine_shape,
                                effective_scale,
                                egui::Color32::from_rgb(128, 128, 128),
                                0.5,
                            );
                        }
                    }
                }
                OverlayMode::ReprojError => {
                    // Compute value range from tracked features
                    let (vmin, vmax) = compute_error_range(features, recon);
                    // Draw colored circles for tracked features
                    for feature in features {
                        if !feature.is_tracked() {
                            continue;
                        }
                        let center = image_to_panel(feature.position[0], feature.position[1]);
                        if !panel_rect.expand(10.0).contains(center) {
                            continue;
                        }
                        let error = recon
                            .points
                            .get(feature.point_index as usize)
                            .map(|p| if p.error.is_finite() { p.error } else { vmax })
                            .unwrap_or(0.0);
                        let is_selected = selected_point == Some(feature.point_index as usize);
                        let color = colormap::error_color(error, vmin, vmax);
                        let radius = 5.0;
                        painter.circle_filled(center, radius, color);
                        if is_selected {
                            painter.circle_stroke(
                                center,
                                radius + 2.0,
                                egui::Stroke::new(2.0, egui::Color32::YELLOW),
                            );
                        }
                    }
                    // Draw colorbar legend
                    colormap::draw_colorbar(
                        &painter,
                        panel_rect,
                        "Reproj Error (px)",
                        vmin,
                        vmax,
                        colormap::error_color,
                    );
                }
                OverlayMode::TrackLength => {
                    // Compute value range from tracked features
                    let (vmin, vmax) = compute_track_length_range(features, recon);
                    // Draw colored circles for tracked features
                    for feature in features {
                        if !feature.is_tracked() {
                            continue;
                        }
                        let center = image_to_panel(feature.position[0], feature.position[1]);
                        if !panel_rect.expand(10.0).contains(center) {
                            continue;
                        }
                        let track_len = recon
                            .observation_counts
                            .get(feature.point_index as usize)
                            .copied()
                            .unwrap_or(1) as f32;
                        let is_selected = selected_point == Some(feature.point_index as usize);
                        let color = colormap::track_length_color(track_len, vmin, vmax);
                        let radius = 5.0;
                        painter.circle_filled(center, radius, color);
                        if is_selected {
                            painter.circle_stroke(
                                center,
                                radius + 2.0,
                                egui::Stroke::new(2.0, egui::Color32::YELLOW),
                            );
                        }
                    }
                    // Draw colorbar legend
                    colormap::draw_colorbar(
                        &painter,
                        panel_rect,
                        "Track Length",
                        vmin,
                        vmax,
                        colormap::track_length_color,
                    );
                }
            }

            // Hit testing for feature clicks (only tracked features)
            if interact_response.clicked() {
                if let Some(pointer_pos) = ui.input(|i| i.pointer.interact_pos()) {
                    let hit_radius_px = 8.0 / effective_scale;
                    response.select_point = find_nearest_tracked_feature(
                        features,
                        feature_tree,
                        &panel_to_image(pointer_pos),
                        hit_radius_px,
                    );
                }
            }

            // Draw cyan highlight for externally hovered point (from 3D viewer),
            // matching the 3D viewport's bright cyan hover color.
            if let Some(hp) = hovered_point {
                if selected_point != Some(hp) {
                    let cyan = egui::Color32::from_rgb(0, 255, 255);
                    for f in features.iter() {
                        if f.point_index as usize == hp {
                            let center = image_to_panel(f.position[0], f.position[1]);
                            draw_feature_ellipse(
                                &painter,
                                center,
                                &f.affine_shape,
                                effective_scale,
                                cyan,
                                2.0,
                            );
                            painter.circle_filled(center, 4.0, cyan);
                            break;
                        }
                    }
                }
            }

            // Tooltip on hover
            if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
                if panel_rect.contains(pointer_pos) {
                    let hit_radius_px = 8.0 / effective_scale;
                    if let Some(point_idx) = find_nearest_tracked_feature(
                        features,
                        feature_tree,
                        &panel_to_image(pointer_pos),
                        hit_radius_px,
                    ) {
                        // Report hover for cross-panel feedback.
                        response.hovered_point = Some(point_idx);

                        let tooltip_text = if let Some(pt) = recon.points.get(point_idx) {
                            let obs_count = recon
                                .observation_counts
                                .get(point_idx)
                                .copied()
                                .unwrap_or(0);
                            format!(
                                "Point3D #{point_idx} | err: {:.3}px | tracklen: {obs_count}",
                                pt.error
                            )
                        } else {
                            format!("Point3D #{point_idx}")
                        };
                        let font = egui::FontId::proportional(12.0);
                        let galley =
                            painter.layout_no_wrap(tooltip_text, font, egui::Color32::WHITE);
                        let padding = 3.0;
                        let tooltip_size = galley.size() + egui::vec2(padding * 2.0, padding * 2.0);
                        let mut tooltip_pos = pointer_pos + egui::vec2(12.0, -20.0);
                        // Clamp to keep tooltip within the panel
                        if tooltip_pos.x + tooltip_size.x > panel_rect.right() {
                            tooltip_pos.x = panel_rect.right() - tooltip_size.x;
                        }
                        let text_rect =
                            egui::Rect::from_min_size(tooltip_pos, galley.size()).expand(padding);
                        painter.rect_filled(text_rect, 2.0, egui::Color32::from_black_alpha(200));
                        painter.galley(tooltip_pos, galley, egui::Color32::WHITE);
                    }
                }
            }
        }

        response
    }

    fn load_image(&mut self, ctx: &egui::Context, recon: &SfmrReconstruction, img_idx: usize) {
        let Some(img) = recon.images.get(img_idx) else {
            self.loaded_image = None;
            return;
        };
        let image_path = recon.workspace_dir.join(&img.name);
        match image::open(&image_path) {
            Ok(dyn_image) => {
                let rgba = dyn_image.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let pixels = rgba.into_raw();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
                let texture = ctx.load_texture(
                    format!("detail_{img_idx}"),
                    color_image,
                    egui::TextureOptions::LINEAR,
                );
                self.loaded_image = Some((img_idx, texture));
            }
            Err(e) => {
                log::warn!(
                    "Failed to load detail image {}: {}",
                    image_path.display(),
                    e
                );
                self.loaded_image = None;
            }
        }
    }

    /// Build tracked-only feature list from the shared SIFT cache (for None overlay mode).
    fn load_tracked_features(
        &mut self,
        recon: &SfmrReconstruction,
        img_idx: usize,
        cached_sift: Option<&CachedSiftFeatures>,
    ) {
        let feature_to_point = &recon.image_feature_to_point[img_idx];
        if feature_to_point.is_empty() || cached_sift.is_none() {
            self.feature_overlay = Some(FeatureOverlayState {
                image_idx: img_idx,
                overlay_mode: OverlayMode::None,
                tracked_only: true,
                max_features: None,
                min_feature_size: None,
                max_feature_size: None,
                features: Vec::new(),
                tree: kiddo::KdTree::<f32, 2>::new(),
            });
            return;
        }
        let cached = cached_sift.unwrap();
        let num_features = cached.positions_xy.len();
        let mut features = Vec::with_capacity(feature_to_point.len());
        for (&feat_idx, &point_idx) in feature_to_point {
            let fi = feat_idx as usize;
            if fi < num_features {
                features.push(DisplayFeature {
                    position: cached.positions_xy[fi],
                    affine_shape: cached.affine_shapes[fi],
                    point_index: point_idx,
                });
            }
        }
        let mut tree = kiddo::KdTree::<f32, 2>::new();
        for (i, feature) in features.iter().enumerate() {
            tree.add(&feature.position, i as u64);
        }
        log::info!(
            "Loaded {} tracked features for image {}",
            features.len(),
            img_idx,
        );
        self.feature_overlay = Some(FeatureOverlayState {
            image_idx: img_idx,
            overlay_mode: OverlayMode::None,
            tracked_only: true,
            max_features: None,
            min_feature_size: None,
            max_feature_size: None,
            features,
            tree,
        });
    }

    /// Build display feature list for overlay modes (Features/ReprojError/TrackLength).
    fn load_display_features(
        &mut self,
        recon: &SfmrReconstruction,
        img_idx: usize,
        cached_sift: Option<&CachedSiftFeatures>,
        settings: &FeatureDisplaySettings,
    ) {
        let Some(cached) = cached_sift else {
            self.feature_overlay = Some(FeatureOverlayState {
                image_idx: img_idx,
                overlay_mode: settings.overlay_mode,
                tracked_only: settings.tracked_only,
                max_features: settings.max_features,
                min_feature_size: settings.min_feature_size,
                max_feature_size: settings.max_feature_size,
                features: Vec::new(),
                tree: kiddo::KdTree::<f32, 2>::new(),
            });
            return;
        };

        let feature_to_point = &recon.image_feature_to_point[img_idx];
        let num_features = cached.positions_xy.len();

        // Apply max_features limit
        let limit = settings
            .max_features
            .map_or(num_features, |m| m.min(num_features));

        // Apply min_feature_size filter: features are sorted by decreasing size,
        // so scan from the end of the prefix to find the cutoff.
        let effective_count = if let Some(min_size) = settings.min_feature_size {
            let mut cutoff = limit;
            for i in (0..limit).rev() {
                if feature_size(&cached.affine_shapes[i]) >= min_size {
                    cutoff = i + 1;
                    break;
                }
                if i == 0 {
                    cutoff = 0;
                }
            }
            cutoff
        } else {
            limit
        };

        let mut features = Vec::with_capacity(effective_count);
        for i in 0..effective_count {
            // Skip features larger than max_feature_size
            if let Some(max_size) = settings.max_feature_size {
                if feature_size(&cached.affine_shapes[i]) > max_size {
                    continue;
                }
            }

            let point_index = feature_to_point
                .get(&(i as u32))
                .copied()
                .unwrap_or(UNTRACKED);

            // Skip untracked features if tracked_only is set
            if settings.tracked_only && point_index == UNTRACKED {
                continue;
            }

            features.push(DisplayFeature {
                position: cached.positions_xy[i],
                affine_shape: cached.affine_shapes[i],
                point_index,
            });
        }

        let mut tree = kiddo::KdTree::<f32, 2>::new();
        for (i, feature) in features.iter().enumerate() {
            tree.add(&feature.position, i as u64);
        }

        let tracked_count = features.iter().filter(|f| f.is_tracked()).count();
        log::info!(
            "Loaded {} features ({} tracked) for image {} (mode: {:?})",
            features.len(),
            tracked_count,
            img_idx,
            settings.overlay_mode,
        );
        self.feature_overlay = Some(FeatureOverlayState {
            image_idx: img_idx,
            overlay_mode: settings.overlay_mode,
            tracked_only: settings.tracked_only,
            max_features: settings.max_features,
            min_feature_size: settings.min_feature_size,
            max_feature_size: settings.max_feature_size,
            features,
            tree,
        });
    }

    /// Clear the cached image (e.g., when reconstruction changes).
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.loaded_image = None;
        self.feature_overlay = None;
        self.reset_view();
        self.prev_selected_image = None;
    }
}

/// Draw an oriented ellipse from a 2×2 affine shape matrix.
///
/// The affine matrix A maps the unit circle to the ellipse: p = A @ [cos(t), sin(t)]^T.
/// We decompose via SVD to get semi-axis lengths and rotation angle, following the
/// same approach as `sift_file.py:draw_sift_features()`.
fn draw_feature_ellipse(
    painter: &egui::Painter,
    center: egui::Pos2,
    affine: &[[f32; 2]; 2],
    scale: f32,
    color: egui::Color32,
    thickness: f32,
) {
    // SVD of 2x2 matrix: A = U * diag(s) * V^T
    // Semi-axis lengths are the singular values.
    // Rotation angle is atan2(a21, a11) (COLMAP convention).
    let a11 = affine[0][0];
    let a12 = affine[0][1];
    let a21 = affine[1][0];
    let a22 = affine[1][1];

    // Compute singular values via the characteristic equation of A^T * A
    let ata00 = a11 * a11 + a21 * a21;
    let ata01 = a11 * a12 + a21 * a22;
    let ata11 = a12 * a12 + a22 * a22;

    let trace = ata00 + ata11;
    let det = ata00 * ata11 - ata01 * ata01;
    let disc = ((trace * trace / 4.0 - det).max(0.0)).sqrt();
    let s1 = ((trace / 2.0 + disc).max(0.0)).sqrt();
    let s2 = ((trace / 2.0 - disc).max(0.0)).sqrt();

    // Rotation angle from the first column of the affine matrix
    let angle = a21.atan2(a11);

    // Skip degenerate ellipses
    if s1 < 0.1 || s2 < 0.1 {
        return;
    }

    // Sample points around the ellipse
    let n = 32;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let points: Vec<egui::Pos2> = (0..=n)
        .map(|i| {
            let t = (i as f32) * std::f32::consts::TAU / (n as f32);
            let ex = s1 * t.cos();
            let ey = s2 * t.sin();
            // Rotate and scale to panel coordinates
            let rx = cos_a * ex - sin_a * ey;
            let ry = sin_a * ex + cos_a * ey;
            egui::pos2(center.x + rx * scale, center.y + ry * scale)
        })
        .collect();

    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(thickness, color),
    ));
}

/// Find the nearest tracked feature to a position in image pixel coordinates.
/// Uses a KD-tree for O(log n) lookup instead of linear scan.
/// `hit_radius_px` is the maximum distance in image pixels.
/// Returns the point_index of the nearest tracked feature, or None if none is close enough.
fn find_nearest_tracked_feature(
    features: &[DisplayFeature],
    tree: &kiddo::KdTree<f32, 2>,
    query_px: &[f32; 2],
    hit_radius_px: f32,
) -> Option<usize> {
    if features.is_empty() {
        return None;
    }
    let hit_radius_sq = hit_radius_px * hit_radius_px;
    // Check a few nearest neighbors in case the closest is untracked
    let neighbors = tree.nearest_n::<SquaredEuclidean>(query_px, 5);
    for neighbor in neighbors {
        if neighbor.distance > hit_radius_sq {
            break;
        }
        let feature = &features[neighbor.item as usize];
        if feature.is_tracked() {
            return Some(feature.point_index as usize);
        }
    }
    None
}

/// Compute the size of a feature from its 2x2 affine shape matrix.
/// Size = average of column norms.
fn feature_size(affine: &[[f32; 2]; 2]) -> f32 {
    let col0_norm = (affine[0][0] * affine[0][0] + affine[1][0] * affine[1][0]).sqrt();
    let col1_norm = (affine[0][1] * affine[0][1] + affine[1][1] * affine[1][1]).sqrt();
    0.5 * (col0_norm + col1_norm)
}

/// Compute the reprojection error range for tracked features in the display list.
fn compute_error_range(features: &[DisplayFeature], recon: &SfmrReconstruction) -> (f32, f32) {
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for feature in features {
        if !feature.is_tracked() {
            continue;
        }
        if let Some(pt) = recon.points.get(feature.point_index as usize) {
            if pt.error.is_finite() {
                vmin = vmin.min(pt.error);
                vmax = vmax.max(pt.error);
            }
        }
    }
    if vmin > vmax {
        (0.0, 1.0)
    } else if (vmax - vmin).abs() < 1e-6 {
        (vmin - 0.5, vmax + 0.5)
    } else {
        (vmin, vmax)
    }
}

/// Compute the track length (observation count) range for tracked features.
fn compute_track_length_range(
    features: &[DisplayFeature],
    recon: &SfmrReconstruction,
) -> (f32, f32) {
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for feature in features {
        if !feature.is_tracked() {
            continue;
        }
        let count = recon
            .observation_counts
            .get(feature.point_index as usize)
            .copied()
            .unwrap_or(1) as f32;
        vmin = vmin.min(count);
        vmax = vmax.max(count);
    }
    if vmin > vmax {
        (1.0, 10.0)
    } else if (vmax - vmin).abs() < 1e-6 {
        (vmin - 0.5, vmax + 0.5)
    } else {
        (vmin, vmax)
    }
}
