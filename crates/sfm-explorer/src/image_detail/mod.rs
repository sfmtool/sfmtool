// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image detail panel — full-resolution image display for the selected camera,
//! with SIFT feature overlays and heatmap visualization modes.
//!
//! [`ImageDetail::show`] orchestrates the panel each frame; the heavier pieces
//! live in sibling modules:
//! - [`input`] — drag/scroll/pinch/keyboard/gesture view manipulation.
//! - [`overlay`] — the feature-overlay draw modes, hit-testing, and tooltip.

mod input;
mod overlay;

use crate::platform::{GestureEvent, ScrollInput};
use crate::state::{CachedSiftFeatures, FeatureDisplaySettings, OverlayMode};
use sfmtool_core::SfmrReconstruction;

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
    /// Max pairwise angle (degrees) between observing rays for this feature's
    /// 3D point — the track's widest triangulation baseline. NaN for untracked
    /// features or when not populated.
    max_track_angle_deg: f32,
    /// Inverse-depth z-score (`depth / σ_depth`) of this feature's 3D point.
    /// NaN for untracked / infinity points or when not populated.
    inverse_depth_z: f32,
    /// Condition number of this feature's 3D point's triangulation normal
    /// matrix. NaN for untracked / infinity points or when not populated.
    condition_number: f32,
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

        // --- Input handling --- (returns true on double-click view reset)
        if self.handle_input(
            ui,
            &interact_response,
            panel_rect,
            panel_center,
            panel_size,
            display_size,
            scroll_input,
            gesture_events,
        ) {
            return response;
        }

        // Recompute image rect after pan/zoom changes from input
        let effective_scale = base_scale * self.zoom;
        let display_size = egui::vec2(tex_size.x * effective_scale, tex_size.y * effective_scale);
        let image_center = panel_center + self.pan;
        let image_rect = egui::Rect::from_center_size(image_center, display_size);

        // --- Feature overlays ---
        self.draw_overlays(
            ui,
            &painter,
            &interact_response,
            recon,
            feature_display,
            selected_point,
            hovered_point,
            image_rect,
            panel_rect,
            effective_scale,
            &mut response,
        );

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
                    max_track_angle_deg: f32::NAN,
                    inverse_depth_z: f32::NAN,
                    condition_number: f32::NAN,
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
                max_track_angle_deg: f32::NAN,
                inverse_depth_z: f32::NAN,
                condition_number: f32::NAN,
            });
        }

        // Populate per-point diagnostics only when the active overlay consumes
        // them. Each iterates a point's observations, so we pay only on demand.
        match settings.overlay_mode {
            OverlayMode::MaxTrackAngle => {
                for feature in features.iter_mut() {
                    if feature.is_tracked() {
                        feature.max_track_angle_deg =
                            compute_max_track_angle_deg(recon, feature.point_index as usize);
                    }
                }
            }
            OverlayMode::DepthReliability | OverlayMode::ConditionNumber => {
                for feature in features.iter_mut() {
                    if feature.is_tracked() {
                        let (cond, z) = crate::point_track_detail::compute_point_diagnostics(
                            recon,
                            feature.point_index as usize,
                        );
                        feature.condition_number = cond;
                        feature.inverse_depth_z = z;
                    }
                }
            }
            _ => {}
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

/// Compute the size of a feature from its 2x2 affine shape matrix.
/// Size = average of column norms.
fn feature_size(affine: &[[f32; 2]; 2]) -> f32 {
    let col0_norm = (affine[0][0] * affine[0][0] + affine[1][0] * affine[1][0]).sqrt();
    let col1_norm = (affine[0][1] * affine[0][1] + affine[1][1] * affine[1][1]).sqrt();
    0.5 * (col0_norm + col1_norm)
}

/// Compute the max pairwise angle (degrees) between world-space rays from
/// observing cameras to a 3D point. Single-observation points return 0.0.
fn compute_max_track_angle_deg(recon: &SfmrReconstruction, point_idx: usize) -> f32 {
    let Some(pt) = recon.points.get(point_idx) else {
        return f32::NAN;
    };
    let point_pos = pt.position;
    let observations = recon.observations_for_point(point_idx);
    let mut world_rays: Vec<[f64; 3]> = Vec::with_capacity(observations.len());
    for obs in observations {
        let img_idx = obs.image_index as usize;
        let Some(image) = recon.images.get(img_idx) else {
            continue;
        };
        let cam_center = image.camera_center();
        let dir = point_pos - cam_center;
        let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if len > 1e-12 {
            world_rays.push([dir.x / len, dir.y / len, dir.z / len]);
        }
    }
    if world_rays.len() < 2 {
        return 0.0;
    }
    crate::point_track_detail::compute_max_pairwise_angle(&world_rays)
}
