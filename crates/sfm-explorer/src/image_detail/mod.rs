// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image detail panel — full-resolution image display for the selected camera,
//! with SIFT feature overlays and heatmap visualization modes.
//!
//! [`ImageDetail::show`] orchestrates the panel each frame; the heavier pieces
//! live in sibling modules:
//! - [`input`] — drag/scroll/pinch/keyboard/gesture view manipulation.
//! - [`overlay`] — the feature-overlay draw modes, hit-testing, and tooltip.
//! - [`mod@intrinsics`] — the intrinsics overlay layer, drawn independently of
//!   the feature mode and composing with whichever one is active.

mod input;
mod intrinsics;
mod overlay;
#[cfg(test)]
mod tests;

pub(crate) use intrinsics::{show_intrinsics_controls, CameraLayer};

use crate::platform::{GestureEvent, ScrollInput};
use crate::scene::{CameraRef, ImageRef, ReconId};
use crate::state::{
    CachedSiftFeatures, FeatureDisplaySettings, IntrinsicsDisplaySettings, OverlayMode,
};
use crate::texture::rgb_to_color_image;
use sfmtool_core::camera::remap::ImageU8;
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::SfmrReconstruction;
use std::collections::HashMap;

use intrinsics::View;

/// Maximum zoom level (32× = pixel-level inspection).
const MAX_ZOOM: f32 = 32.0;
/// Minimum overlap in pixels between image and panel when panning.
const PAN_MARGIN: f32 = 50.0;

/// Prepared feature overlay state for the current image in the detail panel.
struct FeatureOverlayState {
    /// The image this overlay was built for. A ref, so a file replacement that
    /// leaves the same index selected still invalidates it.
    image: ImageRef,
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
    /// Currently loaded full-res image and the texture built from it.
    loaded_image: Option<(ImageRef, egui::TextureHandle)>,
    /// Prepared feature overlay for the current image.
    feature_overlay: Option<FeatureOverlayState>,
    /// The intrinsics layer's per-camera products, keyed by [`CameraRef`].
    ///
    /// Bounded by the number of *distinct* intrinsics across the loaded nodes,
    /// which is small even for a per-image-intrinsics solve, so there is no
    /// eviction beyond [`ImageDetail::forget_recon`].
    intrinsics: HashMap<CameraRef, CameraLayer>,
    /// Offset of image center from panel center, in panel pixels.
    pan: egui::Vec2,
    /// Zoom level. 1.0 = fit image to panel. >1.0 = zoomed in.
    zoom: f32,
    /// Displayed image extent that the current [`ImageDetail::pan`] was
    /// measured against, from the last frame that drew one. `None` until a
    /// frame has drawn, and again after a view reset. See
    /// [`ImageDetail::rescale_view`].
    last_display_size: Option<egui::Vec2>,
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
///
/// Point indices are local to the reconstruction the panel was shown with;
/// `dock.rs` pairs them back into [`crate::scene::PointRef`]s.
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
            intrinsics: HashMap::new(),
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
            last_display_size: None,
        }
    }

    /// Drop everything cached for a reconstruction that has left the scene.
    ///
    /// A [`ReconId`] is never reused, so these entries could only ever go stale
    /// rather than alias — this is about the GPU texture they hold, not about
    /// correctness.
    pub fn forget_recon(&mut self, id: ReconId) {
        if self
            .loaded_image
            .as_ref()
            .is_some_and(|(image, _)| image.recon == id)
        {
            self.loaded_image = None;
        }
        if self
            .feature_overlay
            .as_ref()
            .is_some_and(|overlay| overlay.image.recon == id)
        {
            self.feature_overlay = None;
        }
        self.intrinsics.retain(|camera, _| camera.recon != id);
    }

    /// The cached intrinsics-layer report for `camera`, rebuilt when the camera
    /// or the grid density changes.
    ///
    /// Called from the toolbar as well as from the draw pass, so the popup's
    /// footer and the on-image legend are reading one computation rather than
    /// two that could disagree.
    pub(crate) fn intrinsics_layer(
        &mut self,
        camera_ref: CameraRef,
        camera: &CameraIntrinsics,
        grid_cols: usize,
    ) -> &mut CameraLayer {
        let stale = self
            .intrinsics
            .get(&camera_ref)
            .is_none_or(|layer| layer.grid.0 != grid_cols.max(1));
        if stale {
            self.intrinsics
                .insert(camera_ref, CameraLayer::compute(camera, grid_cols));
        }
        self.intrinsics
            .get_mut(&camera_ref)
            .expect("just inserted when stale")
    }

    /// Reset pan and zoom to fit the image in the panel.
    fn reset_view(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.zoom = 1.0;
        // Nothing left to carry, so skip the next frame's rescale rather than
        // measure a zero pan against a stale extent.
        self.last_display_size = None;
    }

    /// Hold the framed region of the image fixed when the displayed extent
    /// changes.
    ///
    /// The view deliberately outlives the image it was set on: switching
    /// images, switching reconstructions and resizing the panel all keep
    /// whatever region was being inspected, so two images can be compared by
    /// flipping between them while zoomed in. `pan` alone cannot do that — it
    /// is in panel pixels, so the same value frames a different part of an
    /// image of another resolution. What does survive is `pan / display_size`:
    /// the offset of the image centre from the panel centre as a fraction of
    /// the displayed image, which is exactly the normalized image coordinate
    /// `0.5 - pan / display_size` sitting at the panel centre. Rescaling `pan`
    /// by the extent ratio holds that coordinate fixed, per axis so a change of
    /// aspect ratio is handled too. In the common case — two images of equal
    /// size in an unchanged panel — the ratio is 1 and the view carries over
    /// untouched.
    fn rescale_view(&mut self, display_size: egui::Vec2) {
        let Some(previous) = self.last_display_size else {
            return;
        };
        if previous.x <= 0.0 || previous.y <= 0.0 {
            return;
        }
        self.pan.x *= display_size.x / previous.x;
        self.pan.y *= display_size.y / previous.y;
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
        recon_id: ReconId,
        selected_image: Option<usize>,
        selected_point: Option<usize>,
        hovered_point: Option<usize>,
        gesture_events: &[GestureEvent],
        scroll_input: &ScrollInput,
        sift_features: Option<&CachedSiftFeatures>,
        full_res: Option<&ImageU8>,
        feature_display: &FeatureDisplaySettings,
        intrinsics_display: &mut IntrinsicsDisplaySettings,
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
            return response;
        };

        let image_ref = ImageRef::new(recon_id, img_idx);

        // Load the full-resolution image if it changed. The CPU pixels come
        // from the shared `full_res_cache` (decoded once, in dock.rs); this
        // panel only uploads them to a GPU texture.
        if self.loaded_image.as_ref().map(|(i, _)| *i) != Some(image_ref) {
            self.load_image(ui.ctx(), full_res, image_ref);
            self.feature_overlay = None; // reset overlay on image change
        }

        // Determine whether to show features based on overlay mode
        let show_features = feature_display.overlay_mode != OverlayMode::None;

        // Rebuild overlay if settings changed (mode, filters, etc.)
        let cache_valid = self.feature_overlay.as_ref().is_some_and(|c| {
            c.image == image_ref
                && c.overlay_mode == feature_display.overlay_mode
                && c.tracked_only == feature_display.tracked_only
                && c.max_features == feature_display.max_features
                && c.min_feature_size == feature_display.min_feature_size
                && c.max_feature_size == feature_display.max_feature_size
        });
        if show_features && !cache_valid {
            self.load_display_features(recon, image_ref, sift_features, feature_display);
        } else if !show_features {
            // In None mode, still load tracked features for selected point display
            let tracked_overlay_valid = self
                .feature_overlay
                .as_ref()
                .is_some_and(|c| c.image == image_ref && c.tracked_only);
            if !tracked_overlay_valid {
                self.load_tracked_features(recon, image_ref, sift_features);
            }
        }

        // Display the image fitted to the panel. Read the texture's size and id
        // out here rather than holding the handle: the view bookkeeping below
        // needs `&mut self`.
        let Some((tex_size, texture_id)) = self
            .loaded_image
            .as_ref()
            .map(|(_, texture)| (texture.size_vec2(), texture.id()))
        else {
            ui.centered_and_justified(|ui| {
                ui.label("Failed to load image");
            });
            return response;
        };

        let panel_rect = ui.available_rect_before_wrap();
        let panel_size = panel_rect.size();
        let panel_center = panel_rect.center();

        // Base scale: fits the image to the panel at zoom=1.0
        let base_scale = (panel_size.x / tex_size.x).min(panel_size.y / tex_size.y);
        let effective_scale = base_scale * self.zoom;
        let display_size = egui::vec2(tex_size.x * effective_scale, tex_size.y * effective_scale);

        // Carry the framed region across an image or reconstruction switch and
        // across a panel resize — all of which reach here as a change of extent.
        self.rescale_view(display_size);
        self.clamp_pan(display_size, panel_size);

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
            texture_id,
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
        // The extent `pan` is now measured against, for the next frame's rescale.
        self.last_display_size = Some(display_size);
        let image_center = panel_center + self.pan;
        let image_rect = egui::Rect::from_center_size(image_center, display_size);

        // --- Intrinsics layer, beneath the features ---
        //
        // `I` toggles it while the pointer is over the panel, alongside the
        // panel's existing `Z`: it is a control users flip constantly once it
        // composes, which is the whole reason it is a layer.
        if response.has_pointer && ui.input(|i| i.key_pressed(egui::Key::I)) {
            intrinsics_display.enabled = !intrinsics_display.enabled;
        }
        let camera_ref = recon
            .images
            .get(img_idx)
            .map(|image| CameraRef::new(recon_id, image.camera_index as usize));
        let camera = camera_ref
            .and_then(|camera_ref| recon.cameras.get(camera_ref.index()))
            .cloned();
        let view = View {
            origin: image_rect.min,
            scale: base_scale * self.zoom,
            fit: base_scale,
        };
        let intrinsics_readout = match (intrinsics_display.enabled, camera_ref, &camera) {
            (true, Some(camera_ref), Some(camera)) => self.draw_intrinsics(
                &painter,
                camera_ref,
                camera,
                &view,
                panel_rect,
                intrinsics_display,
                ui.input(|i| i.pointer.hover_pos()),
            ),
            _ => None,
        };

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
            intrinsics_readout.as_deref(),
            &mut response,
        );

        // The one mark that draws over the features rather than under them.
        if let Some(camera) = &camera {
            if intrinsics_display.enabled {
                intrinsics::draw_principal_point(&painter, camera, &view, panel_rect);
            }
        }

        response
    }

    /// Build the display texture from the shared full-res CPU image (decoded
    /// once into `AppState::full_res_cache`). `None` means the decode failed,
    /// in which case the "Failed to load image" placeholder path applies.
    fn load_image(&mut self, ctx: &egui::Context, full_res: Option<&ImageU8>, image: ImageRef) {
        let Some(img) = full_res else {
            self.loaded_image = None;
            return;
        };
        let img_idx = image.index();
        // Expand 3-channel RGB to RGBA for the GPU upload.
        let (w, h) = (img.width() as usize, img.height() as usize);
        let color_image = rgb_to_color_image(img.data(), [w, h]);
        let texture = ctx.load_texture(
            format!("detail_{img_idx}"),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.loaded_image = Some((image, texture));
    }

    /// Build tracked-only feature list from the shared SIFT cache (for None overlay mode).
    fn load_tracked_features(
        &mut self,
        recon: &SfmrReconstruction,
        image: ImageRef,
        cached_sift: Option<&CachedSiftFeatures>,
    ) {
        let img_idx = image.index();
        // Embedded-patches reconstructions keep keypoints inline (no `.sift`
        // cache, empty `image_feature_to_point`); build the tracked-feature list
        // from the per-observation keypoints. Every embedded observation belongs
        // to a point, so all are tracked.
        if recon.feature_indexes().is_none() {
            let features = embedded_image_features(recon, img_idx);
            let tree = build_feature_tree(&features);
            log::info!(
                "Loaded {} embedded tracked features for image {}",
                features.len(),
                img_idx,
            );
            self.feature_overlay = Some(FeatureOverlayState {
                image,
                overlay_mode: OverlayMode::None,
                tracked_only: true,
                max_features: None,
                min_feature_size: None,
                max_feature_size: None,
                features,
                tree,
            });
            return;
        }

        let feature_to_point = &recon.image_feature_to_point[img_idx];
        if feature_to_point.is_empty() || cached_sift.is_none() {
            self.feature_overlay = Some(FeatureOverlayState {
                image,
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
            image,
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
        image: ImageRef,
        cached_sift: Option<&CachedSiftFeatures>,
        settings: &FeatureDisplaySettings,
    ) {
        let img_idx = image.index();
        // Embedded-patches: build features from the inline per-observation
        // keypoints, with affine shapes derived by projecting each point's patch
        // frame. Every embedded observation is tracked (no untracked keypoints),
        // so `tracked_only` is a no-op; size filters and the max-features cap
        // apply just like the SIFT path.
        if recon.feature_indexes().is_none() {
            let mut features = embedded_image_features(recon, img_idx);
            features.retain(|f| {
                let size = feature_size(&f.affine_shape);
                settings.min_feature_size.is_none_or(|mn| size >= mn)
                    && settings.max_feature_size.is_none_or(|mx| size <= mx)
            });
            // Keep the largest features when capping, mirroring the SIFT path
            // (whose cache is pre-sorted by decreasing size).
            if let Some(max) = settings.max_features {
                if features.len() > max {
                    features.sort_by(|a, b| {
                        feature_size(&b.affine_shape).total_cmp(&feature_size(&a.affine_shape))
                    });
                    features.truncate(max);
                }
            }
            populate_feature_diagnostics(&mut features, recon, settings.overlay_mode);
            let tree = build_feature_tree(&features);
            log::info!(
                "Loaded {} embedded features for image {} (mode: {:?})",
                features.len(),
                img_idx,
                settings.overlay_mode,
            );
            self.feature_overlay = Some(FeatureOverlayState {
                image,
                overlay_mode: settings.overlay_mode,
                tracked_only: settings.tracked_only,
                max_features: settings.max_features,
                min_feature_size: settings.min_feature_size,
                max_feature_size: settings.max_feature_size,
                features,
                tree,
            });
            return;
        }

        let Some(cached) = cached_sift else {
            self.feature_overlay = Some(FeatureOverlayState {
                image,
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
        populate_feature_diagnostics(&mut features, recon, settings.overlay_mode);

        let tree = build_feature_tree(&features);

        let tracked_count = features.iter().filter(|f| f.is_tracked()).count();
        log::info!(
            "Loaded {} features ({} tracked) for image {} (mode: {:?})",
            features.len(),
            tracked_count,
            img_idx,
            settings.overlay_mode,
        );
        self.feature_overlay = Some(FeatureOverlayState {
            image,
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
        self.intrinsics.clear();
        self.reset_view();
    }
}

/// Compute the size of a feature from its 2x2 affine shape matrix.
/// Size = average of column norms.
fn feature_size(affine: &[[f32; 2]; 2]) -> f32 {
    let col0_norm = (affine[0][0] * affine[0][0] + affine[1][0] * affine[1][0]).sqrt();
    let col1_norm = (affine[0][1] * affine[0][1] + affine[1][1] * affine[1][1]).sqrt();
    0.5 * (col0_norm + col1_norm)
}

/// Build a 2-D kd-tree over feature positions for hit-testing / hover.
fn build_feature_tree(features: &[DisplayFeature]) -> kiddo::KdTree<f32, 2> {
    let mut tree = kiddo::KdTree::<f32, 2>::new();
    for (i, feature) in features.iter().enumerate() {
        tree.add(&feature.position, i as u64);
    }
    tree
}

/// Feature list for an `embedded_patches` reconstruction: every observation
/// landing in `img_idx`, as a tracked feature at its inline keypoint. The affine
/// shape is derived by projecting the point's patch frame into this image
/// (`observation_affine_shape`); it falls back to a degenerate (zero) shape when
/// the point has no usable patch frame, in which case `draw_feature_ellipse`
/// skips the ellipse and only the centre dot draws. O(total observations):
/// embedded recons have no per-image keypoint index (`image_feature_to_point`
/// is empty).
fn embedded_image_features(recon: &SfmrReconstruction, img_idx: usize) -> Vec<DisplayFeature> {
    let Some(kxy) = recon.keypoints_xy() else {
        return Vec::new();
    };
    let mut features = Vec::new();
    for point_idx in 0..recon.points.len() {
        let obs_start = recon.observation_offsets[point_idx];
        for (k, obs) in recon.observations_for_point(point_idx).iter().enumerate() {
            if obs.image_index as usize == img_idx {
                let row = obs_start + k;
                let position = [kxy[[row, 0]], kxy[[row, 1]]];
                let affine_shape = recon
                    .observation_affine_shape(point_idx, img_idx, position)
                    .unwrap_or([[0.0; 2]; 2]);
                features.push(DisplayFeature {
                    position,
                    affine_shape,
                    point_index: point_idx as u32,
                    max_track_angle_deg: f32::NAN,
                    inverse_depth_z: f32::NAN,
                    condition_number: f32::NAN,
                });
            }
        }
    }
    features
}

/// Populate the per-point diagnostics an overlay mode consumes (only for the
/// modes that need them; each iterates a point's observations, so we pay only
/// on demand).
fn populate_feature_diagnostics(
    features: &mut [DisplayFeature],
    recon: &SfmrReconstruction,
    mode: OverlayMode,
) {
    match mode {
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
