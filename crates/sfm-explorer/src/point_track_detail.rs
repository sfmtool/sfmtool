// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point Track Detail panel — shows all observations of a selected 3D point.
//!
//! When a 3D point is selected (via click in the 3D viewer or feature click in
//! the Image Detail panel), this panel displays a header with point summary
//! statistics and a scrollable table of per-image observations.

use std::collections::HashMap;
use std::path::Path;

use nalgebra::Vector3;
use ndarray::Axis;
use sfmtool_core::camera::remap::{remap_bilinear, ImageU8};
use sfmtool_core::camera::WarpMap;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::SfmrReconstruction;

use crate::platform::{self, GestureEvent};
use crate::state::CachedSiftFeatures;

/// Precomputed data for one observation in the track.
struct TrackObservationData {
    /// Index into `recon.images`.
    image_index: usize,
    /// Feature index within the image's SIFT file.
    feature_index: usize,
    /// Feature position in image pixel coordinates.
    feature_xy: [f32; 2],
    /// Per-observation reprojection error in pixels.
    reproj_error: f32,
    /// Angular discrepancy between observation ray and point direction, in degrees.
    ray_angle_deg: f32,
    /// SIFT feature size (average radius in pixels from affine shape).
    feature_size: f32,
    /// Truncated display name (e.g. "…/fisheye_left/image_0345.jpg").
    image_name: String,
    /// Full image path from the reconstruction.
    image_full_name: String,
}

/// Point Track Detail panel state.
pub struct PointTrackDetail {
    /// The point index we've prepared data for, or None.
    prepared_point: Option<usize>,
    /// Precomputed observation data for the current point.
    observations: Vec<TrackObservationData>,
    /// Maximum angle (degrees) between any pair of observation rays in the track.
    max_angle_deg: f32,
    /// Inverse-depth z-score (`depth / σ_depth`) of the triangulation; NaN when
    /// undefined (point at infinity or fewer than two usable rays).
    inverse_depth_z: f32,
    /// Condition number of the triangulation's normal matrix; NaN when undefined.
    condition_number: f32,
    /// Cached thumbnail textures keyed by image index.
    thumbnail_textures: HashMap<usize, egui::TextureHandle>,
    /// The selected point's oriented patch frame (from the stored patch
    /// half-vectors), or None when the reconstruction carries no frame or the
    /// point has no patch. Gates the per-observation "Patch" column.
    patch_frame: Option<OrientedPatch>,
    /// Stored patch bitmap texture for the selected point (header tile), if any.
    stored_patch_texture: Option<egui::TextureHandle>,
    /// Per-observation patch tiles rendered from full-res images, keyed by
    /// image index. Rebuilt on point-selection change. Tiles where the patch is
    /// not visible in the view warp to all-black and are drawn as such (a future
    /// N/A flag may distinguish "not visible" from a genuinely dark surface).
    rendered_patch_textures: HashMap<usize, egui::TextureHandle>,
    /// The content_xxh128 hash prefix (first 8 hex chars) for Point IDs.
    hash_prefix: String,
    /// Tracked vertical scroll offset for DM gesture scrolling.
    scroll_offset_y: Option<f32>,
}

/// Response from the Point Track Detail panel.
pub struct PointTrackDetailResponse {
    /// If Some, the user clicked a row — select this image.
    pub select_image: Option<usize>,
    /// If Some, the user double-clicked a row — enter camera view for this image.
    pub request_camera_view: Option<usize>,
    /// Image index currently under the pointer (for cross-panel hover).
    pub hovered_image: Option<usize>,
    /// Whether the pointer is currently inside the panel.
    pub has_pointer: bool,
}

/// Height of each thumbnail in the observation table.
const THUMB_SIZE: f32 = 48.0;
/// Size of the feature dot overlay on thumbnails.
const DOT_RADIUS: f32 = 3.0;
/// Display size of the per-observation rendered patch tile (matches the
/// thumbnail so the tile sits flush beside it).
const PATCH_TILE: f32 = THUMB_SIZE;
/// Render resolution of per-observation patch tiles (rendered crisp at this
/// resolution, displayed scaled to `PATCH_TILE`).
const PATCH_RES: u32 = 64;
/// Display size of the stored-patch header tile.
const STORED_PATCH_SIZE: f32 = 64.0;

impl PointTrackDetail {
    pub fn new() -> Self {
        Self {
            prepared_point: None,
            observations: Vec::new(),
            max_angle_deg: 0.0,
            inverse_depth_z: f32::NAN,
            condition_number: f32::NAN,
            thumbnail_textures: HashMap::new(),
            patch_frame: None,
            stored_patch_texture: None,
            rendered_patch_textures: HashMap::new(),
            hash_prefix: String::new(),
            scroll_offset_y: None,
        }
    }

    /// Show the point track detail panel.
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        selected_point: Option<usize>,
        hovered_image: Option<usize>,
        sift_cache: &HashMap<usize, CachedSiftFeatures>,
        full_res_cache: &HashMap<usize, Option<ImageU8>>,
        gesture_events: &[GestureEvent],
        scroll_input: &platform::ScrollInput,
    ) -> PointTrackDetailResponse {
        let mut response = PointTrackDetailResponse {
            select_image: None,
            request_camera_view: None,
            hovered_image: None,
            has_pointer: false,
        };

        // Check if pointer is in this panel
        let panel_rect = ui.available_rect_before_wrap();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            if panel_rect.contains(pos) {
                response.has_pointer = true;
            }
        }

        // No point selected — show placeholder
        let Some(point_idx) = selected_point else {
            ui.centered_and_justified(|ui| {
                ui.label("No point selected");
            });
            self.prepared_point = None;
            self.observations.clear();
            return response;
        };

        if point_idx >= recon.points.len() {
            ui.centered_and_justified(|ui| {
                ui.label("No point selected");
            });
            self.prepared_point = None;
            self.observations.clear();
            return response;
        }

        // Prepare observation data if selected point changed
        if self.prepared_point != Some(point_idx) {
            self.prepare_observations(ui.ctx(), recon, point_idx, sift_cache);
            self.prepared_point = Some(point_idx);
            self.scroll_offset_y = None;
            // Update hash prefix from reconstruction
            let hash = &recon.content_hash.content_xxh128;
            self.hash_prefix = if hash.len() >= 8 {
                hash[..8].to_string()
            } else {
                "00000000".to_string()
            };
        }

        let point = &recon.points[point_idx];

        // --- Header: Point Summary ---
        self.show_header(ui, recon, point_idx, point);

        // --- Stored-patch header tile (embedded-patches reconstructions) ---
        if let Some(texture) = &self.stored_patch_texture {
            ui.horizontal(|ui| {
                ui.label("Stored patch:");
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(STORED_PATCH_SIZE, STORED_PATCH_SIZE),
                    egui::Sense::hover(),
                );
                ui.painter().image(
                    texture.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            });
        }

        ui.separator();

        // --- Observation Table ---
        self.show_observation_table(
            ui,
            recon,
            hovered_image,
            full_res_cache,
            gesture_events,
            scroll_input,
            &mut response,
        );

        response
    }

    /// Draw the point summary header bar.
    fn show_header(
        &self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        point_idx: usize,
        point: &sfmtool_core::Point3D,
    ) {
        let point_id = format!("pt3d_{}_{}", self.hash_prefix, point_idx);
        let coords = format!(
            "{:.3}, {:.3}, {:.3}",
            point.position.x, point.position.y, point.position.z
        );
        let obs_count = recon.observation_counts[point_idx];

        ui.horizontal_wrapped(|ui| {
            // Color swatch
            let [r, g, b] = point.color;
            let color = egui::Color32::from_rgb(r, g, b);
            let (rect, swatch_response) =
                ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, color);
            ui.painter().rect_stroke(
                rect,
                2.0,
                egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
                egui::StrokeKind::Outside,
            );
            swatch_response.on_hover_text(format!("rgb({r}, {g}, {b})"));

            // Point ID — monospace, with copy button
            ui.label(egui::RichText::new(&point_id).monospace().strong());
            if copy_button(ui, "Copy Point ID") {
                ui.ctx().copy_text(point_id.clone());
            }

            ui.label("|");

            // XYZ coordinates — with copy button
            ui.label(format!("xyz: ({coords})"));
            if copy_button(ui, "Copy coordinates") {
                ui.ctx().copy_text(coords.clone());
            }

            ui.label("|");

            // Error
            ui.label(format!("error: {:.2}px", point.error));

            ui.label("|");

            // Track length
            ui.label(format!("track: {} obs", obs_count));

            // Max triangulation angle
            if self.max_angle_deg > 0.0 {
                ui.label("|");
                ui.label(format!("max pair angle: {:.1}°", self.max_angle_deg));
            }

            // Triangulation observability diagnostics (complementary to the
            // max angle — scale-free and correct in the near-infinity regime).
            if self.inverse_depth_z.is_finite() {
                ui.label("|");
                ui.label(format!("depth z: {:.1}", self.inverse_depth_z));
            }
            if self.condition_number.is_finite() {
                ui.label("|");
                ui.label(format!("cond: {:.0}", self.condition_number));
            }
        });
    }

    /// Draw the scrollable observation table.
    #[allow(clippy::too_many_arguments)]
    fn show_observation_table(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        hovered_image: Option<usize>,
        full_res_cache: &HashMap<usize, Option<ImageU8>>,
        gesture_events: &[GestureEvent],
        scroll_input: &platform::ScrollInput,
        response: &mut PointTrackDetailResponse,
    ) {
        let row_height = THUMB_SIZE + 8.0; // thumbnail + padding

        // Compute vertical scroll delta from DM gestures and scroll input.
        let mut extra_scroll_y = 0.0f32;
        let panel_rect = ui.available_rect_before_wrap();
        if platform::pointer_in_rect(ui.ctx(), panel_rect) {
            for event in gesture_events {
                if let GestureEvent::Pan { dy, .. } = event {
                    extra_scroll_y += *dy as f32;
                }
            }
            if scroll_input.has_trackpad_scroll() {
                extra_scroll_y += scroll_input.delta.y;
            } else if scroll_input.has_mouse_wheel() {
                let multiplier = if matches!(scroll_input.unit, egui::MouseWheelUnit::Line) {
                    row_height
                } else {
                    200.0
                };
                extra_scroll_y += scroll_input.delta.y * multiplier;
            }
        }

        let mut scroll_area = egui::ScrollArea::vertical().auto_shrink([false, false]);

        if extra_scroll_y != 0.0 {
            // Negative because scroll offset increases when content moves up,
            // but pan dy is positive when panning up (content moves down).
            let current = self.scroll_offset_y.unwrap_or(0.0);
            let new_offset = (current - extra_scroll_y).max(0.0);
            scroll_area = scroll_area.vertical_scroll_offset(new_offset);
        }

        // Fixed column x-offsets so headers and row content always align.
        // When the selected point has a patch frame, a rendered-patch tile is
        // drawn immediately right of the thumbnail and every text column
        // shifts right by its width; otherwise the layout is unchanged.
        let has_patch_column = self.patch_frame.is_some();
        let col_patch = THUMB_SIZE + 8.0;
        let patch_shift = if has_patch_column {
            PATCH_TILE + 8.0
        } else {
            0.0
        };
        let col_image = THUMB_SIZE + 8.0 + patch_shift;
        let col_name = col_image + 50.0;
        let col_feat = col_name + 170.0;
        let col_size = col_feat + 55.0;
        let col_error = col_size + 50.0;
        let col_angle = col_error + 60.0;
        let col_xy = col_angle + 55.0;

        let scroll_output = scroll_area.show(ui, |ui| {
            // Table header — paint labels at fixed x-offsets.
            let header_rect = ui.available_rect_before_wrap();
            let header_y = header_rect.min.y;
            let x0 = header_rect.min.x;
            let painter = ui.painter();
            let header_font = egui::TextStyle::Body.resolve(ui.style());
            let strong_color = ui.visuals().strong_text_color();
            let mut header_labels: Vec<(f32, &str)> = Vec::with_capacity(8);
            if has_patch_column {
                header_labels.push((col_patch, "Patch"));
            }
            header_labels.extend_from_slice(&[
                (col_image, "Image"),
                (col_name, "Name"),
                (col_feat, "Feat #"),
                (col_size, "Size"),
                (col_error, "Error"),
                (col_angle, "Angle"),
                (col_xy, "Feature (x, y)"),
            ]);
            for (x_off, text) in header_labels {
                painter.text(
                    egui::pos2(x0 + x_off, header_y),
                    egui::Align2::LEFT_TOP,
                    text,
                    header_font.clone(),
                    strong_color,
                );
            }
            let header_height = ui.text_style_height(&egui::TextStyle::Body);
            ui.allocate_space(egui::vec2(ui.available_width(), header_height));
            ui.separator();

            for obs_i in 0..self.observations.len() {
                let obs = &self.observations[obs_i];
                let obs_image_index = obs.image_index;
                let obs_feature_index = obs.feature_index;
                let obs_feature_xy = obs.feature_xy;
                let obs_reproj_error = obs.reproj_error;
                let obs_ray_angle_deg = obs.ray_angle_deg;
                let obs_feature_size = obs.feature_size;
                let obs_image_name = obs.image_name.clone();
                let obs_image_full_name = obs.image_full_name.clone();
                let is_hovered_image = hovered_image == Some(obs_image_index);

                // Row background color for hover highlight
                let row_rect = ui.available_rect_before_wrap();
                let row_rect = egui::Rect::from_min_size(
                    row_rect.min,
                    egui::vec2(row_rect.width(), row_height),
                );

                // Interact with the row
                let row_response = ui.allocate_rect(row_rect, egui::Sense::click());

                // Draw hover/highlight background
                let is_pointer_on_row = row_response.hovered();
                if is_hovered_image || is_pointer_on_row {
                    let bg_color = if is_pointer_on_row {
                        ui.visuals().widgets.hovered.bg_fill
                    } else {
                        ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.5)
                    };
                    ui.painter().rect_filled(row_rect, 0.0, bg_color);
                }

                // Set hovered_image when pointer is on this row
                if is_pointer_on_row {
                    response.hovered_image = Some(obs_image_index);
                }

                // Handle click/double-click
                if row_response.double_clicked() {
                    response.request_camera_view = Some(obs_image_index);
                    response.select_image = Some(obs_image_index);
                } else if row_response.clicked() {
                    response.select_image = Some(obs_image_index);
                }

                // Draw row content at fixed column offsets.
                let x0 = row_rect.min.x;
                let cy = row_rect.center().y;

                // Thumbnail with feature dot overlay (vertically centered in the row)
                let thumb_y = cy - THUMB_SIZE / 2.0;
                let mut thumb_ui =
                    ui.new_child(egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(
                        egui::pos2(row_rect.min.x, thumb_y),
                        egui::vec2(THUMB_SIZE, THUMB_SIZE),
                    )));
                self.draw_thumbnail(
                    &mut thumb_ui,
                    recon,
                    obs_image_index,
                    obs_feature_xy,
                    obs_reproj_error,
                );

                // Rendered patch tile beside the thumbnail (embedded-patches
                // reconstructions only; rendered lazily from the shared
                // full-res image cache and cached per image index).
                if has_patch_column {
                    self.ensure_rendered_patch(ui.ctx(), recon, obs_image_index, full_res_cache);
                    if let Some(texture) = self.rendered_patch_textures.get(&obs_image_index) {
                        let patch_rect = egui::Rect::from_min_size(
                            egui::pos2(x0 + col_patch, cy - PATCH_TILE / 2.0),
                            egui::vec2(PATCH_TILE, PATCH_TILE),
                        );
                        ui.painter().image(
                            texture.id(),
                            patch_rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    }
                }

                let painter = ui.painter();
                let font = egui::TextStyle::Body.resolve(ui.style());
                let text_color = ui.visuals().text_color();
                let weak_color = ui.visuals().weak_text_color();

                // Image index
                painter.text(
                    egui::pos2(x0 + col_image, cy),
                    egui::Align2::LEFT_CENTER,
                    format!("{}", obs_image_index),
                    font.clone(),
                    text_color,
                );

                // Image name — right-aligned and clipped to column bounds so
                // long paths show the distinguishing suffix without overflowing.
                // Tooltip shows the full path on hover.
                let name_col_gap = 8.0;
                let name_clip = egui::Rect::from_x_y_ranges(
                    (x0 + col_name)..=(x0 + col_feat - name_col_gap),
                    row_rect.y_range(),
                );
                painter.with_clip_rect(name_clip).text(
                    egui::pos2(x0 + col_feat - name_col_gap, cy),
                    egui::Align2::RIGHT_CENTER,
                    &obs_image_name,
                    font.clone(),
                    weak_color,
                );
                let name_response = ui.interact(
                    name_clip,
                    ui.id().with(("name", obs_i)),
                    egui::Sense::hover(),
                );
                name_response.on_hover_text(&obs_image_full_name);

                // Feature index
                painter.text(
                    egui::pos2(x0 + col_feat, cy),
                    egui::Align2::LEFT_CENTER,
                    format!("{}", obs_feature_index),
                    font.clone(),
                    text_color,
                );

                // Feature size
                let size_text = if obs_feature_size > 0.0 {
                    format!("{:.1}", obs_feature_size)
                } else {
                    "N/A".to_string()
                };
                painter.text(
                    egui::pos2(x0 + col_size, cy),
                    egui::Align2::LEFT_CENTER,
                    size_text,
                    font.clone(),
                    text_color,
                );

                // Reprojection error
                let error_text = if obs_reproj_error.is_nan() {
                    "N/A".to_string()
                } else {
                    format!("{:.2}px", obs_reproj_error)
                };
                painter.text(
                    egui::pos2(x0 + col_error, cy),
                    egui::Align2::LEFT_CENTER,
                    error_text,
                    font.clone(),
                    text_color,
                );

                // Ray angle
                let angle_text = if obs_ray_angle_deg.is_nan() {
                    "N/A".to_string()
                } else {
                    format!("{:.2}°", obs_ray_angle_deg)
                };
                painter.text(
                    egui::pos2(x0 + col_angle, cy),
                    egui::Align2::LEFT_CENTER,
                    angle_text,
                    font.clone(),
                    text_color,
                );

                // Feature position
                painter.text(
                    egui::pos2(x0 + col_xy, cy),
                    egui::Align2::LEFT_CENTER,
                    format!("({:.1}, {:.1})", obs_feature_xy[0], obs_feature_xy[1]),
                    font.clone(),
                    text_color,
                );
            }
        });

        // Track scroll offset for next frame's DM gesture application.
        self.scroll_offset_y = Some(scroll_output.state.offset.y);
    }

    /// Draw a thumbnail with a feature dot overlay colored by reprojection error.
    fn draw_thumbnail(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        img_idx: usize,
        feature_xy: [f32; 2],
        reproj_error: f32,
    ) {
        // Load thumbnail texture if not cached
        if !self.thumbnail_textures.contains_key(&img_idx) {
            self.load_thumbnail(ui.ctx(), recon, img_idx);
        }

        let (thumb_rect, _) =
            ui.allocate_exact_size(egui::vec2(THUMB_SIZE, THUMB_SIZE), egui::Sense::hover());

        if let Some(texture) = self.thumbnail_textures.get(&img_idx) {
            // Draw thumbnail
            ui.painter().image(
                texture.id(),
                thumb_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Draw feature dot overlay: map feature pixel coords to screen coords.
            let camera_idx = recon.images[img_idx].camera_index as usize;
            let intrinsics = &recon.cameras[camera_idx];
            let img_w = intrinsics.width as f32;
            let img_h = intrinsics.height as f32;

            let sx = thumb_rect.min.x + (feature_xy[0] / img_w) * thumb_rect.width();
            let sy = thumb_rect.min.y + (feature_xy[1] / img_h) * thumb_rect.height();

            let dot_center = egui::pos2(sx, sy);
            let dot_color = error_color(reproj_error);
            ui.painter()
                .circle_filled(dot_center, DOT_RADIUS, dot_color);
            // Dark outline for visibility
            ui.painter().circle_stroke(
                dot_center,
                DOT_RADIUS,
                egui::Stroke::new(1.0, egui::Color32::BLACK),
            );
        }
    }

    /// Load a single thumbnail texture into the cache.
    fn load_thumbnail(&mut self, ctx: &egui::Context, recon: &SfmrReconstruction, idx: usize) {
        let rgb_slice = recon.thumbnails_y_x_rgb.index_axis(Axis(0), idx);
        let rgb_data: Vec<u8> = if let Some(slice) = rgb_slice.as_slice() {
            slice.to_vec()
        } else {
            rgb_slice.iter().copied().collect()
        };

        let thumb_h = rgb_slice.shape()[0];
        let thumb_w = rgb_slice.shape()[1];
        let mut rgba = Vec::with_capacity(thumb_h * thumb_w * 4);
        for pixel in rgb_data.chunks_exact(3) {
            rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
        }
        let image = egui::ColorImage::from_rgba_unmultiplied([thumb_w, thumb_h], &rgba);
        let texture = ctx.load_texture(
            format!("track_thumb_{idx}"),
            image,
            egui::TextureOptions::LINEAR,
        );
        self.thumbnail_textures.insert(idx, texture);
    }

    /// Render the patch tile for one observation if not already cached: warp
    /// the observation's full-res image through the selected point's patch
    /// frame (`WarpMap::from_patch` + `remap_bilinear`). A patch not visible in
    /// this view warps to an all-black tile and is drawn as such. A missing
    /// source image is not cached (the dock pre-caches full-res images, so this
    /// only happens transiently).
    fn ensure_rendered_patch(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        img_idx: usize,
        full_res_cache: &HashMap<usize, Option<ImageU8>>,
    ) {
        if self.rendered_patch_textures.contains_key(&img_idx) {
            return;
        }
        let Some(frame) = self.patch_frame.as_ref() else {
            return;
        };
        let Some(src) = full_res_cache.get(&img_idx).and_then(|o| o.as_ref()) else {
            return;
        };
        let image = &recon.images[img_idx];
        let camera = &recon.cameras[image.camera_index as usize];
        let q = image.quaternion_wxyz.quaternion();
        let cam_from_world = RigidTransform::from_wxyz_translation(
            [q.w, q.i, q.j, q.k],
            [
                image.translation_xyz.x,
                image.translation_xyz.y,
                image.translation_xyz.z,
            ],
        );
        let map = WarpMap::from_patch(frame, camera, &cam_from_world, PATCH_RES);
        let tile = remap_bilinear(src, &map);
        // Expand 3-channel RGB (same channel count as the cached source) to RGBA.
        let (w, h) = (tile.width() as usize, tile.height() as usize);
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in tile.data().chunks_exact(3) {
            rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
        let point_idx = self.prepared_point.unwrap_or(0);
        let texture = ctx.load_texture(
            format!("track_patch_{point_idx}_{img_idx}"),
            color_image,
            egui::TextureOptions::NEAREST,
        );
        self.rendered_patch_textures.insert(img_idx, texture);
    }

    /// Prepare observation data for a newly selected point.
    fn prepare_observations(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        point_idx: usize,
        sift_cache: &HashMap<usize, CachedSiftFeatures>,
    ) {
        self.observations.clear();
        self.thumbnail_textures.clear();

        // Per-point patch state (embedded-patches reconstructions): the
        // oriented patch frame gates the per-observation "Patch" column, the
        // stored bitmap feeds the header tile. Rendered tiles rebuild lazily.
        self.patch_frame = build_patch_frame(recon, point_idx);
        self.stored_patch_texture = build_stored_patch_texture(ctx, recon, point_idx);
        self.rendered_patch_textures.clear();

        let point_pos = recon.points[point_idx].position;
        // Keypoints come from one of two sources: SIFT feature positions read
        // into the cache (`sift_files`, via `feature_indexes`) or keypoints
        // stored inline on the reconstruction (`embedded_patches`, via
        // `keypoints_xy`, indexed per observation). For embedded keypoints the
        // affine shape (and hence size) is derived by projecting the point's
        // patch frame into the view (`observation_affine_shape`).
        let feature_indexes = recon.feature_indexes();
        let keypoints_xy = recon.keypoints_xy();
        let obs_start = recon.observation_offsets[point_idx];
        let observations = recon.observations_for_point(point_idx);

        // Collect world-space rays from each camera center to the point
        // for max-angle computation.
        let mut world_rays: Vec<[f64; 3]> = Vec::with_capacity(observations.len());

        for (k, obs) in observations.iter().enumerate() {
            let img_idx = obs.image_index as usize;
            let obs_global = obs_start + k;
            let image = &recon.images[img_idx];
            let camera = &recon.cameras[image.camera_index as usize];

            // Feature index (SIFT), position, and size for this observation.
            let (feature_index, feature_xy, feature_size) = if let Some(fis) = feature_indexes {
                let feat_idx = fis[obs_global] as usize;
                let cached_sift = sift_cache.get(&img_idx);
                let xy = cached_sift
                    .and_then(|sift| sift.positions_xy.get(feat_idx))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                let size = cached_sift
                    .and_then(|sift| sift.affine_shapes.get(feat_idx))
                    .map(|a| {
                        let col0 = (a[0][0] * a[0][0] + a[1][0] * a[1][0]).sqrt();
                        let col1 = (a[0][1] * a[0][1] + a[1][1] * a[1][1]).sqrt();
                        0.5 * (col0 + col1)
                    })
                    .unwrap_or(0.0);
                (feat_idx, xy, size)
            } else if let Some(kxy) = keypoints_xy {
                // Embedded keypoint: no SIFT feature index, so report the
                // observation index. The affine shape (and hence size) is derived
                // by projecting the point's patch frame into this image.
                let xy = [kxy[[obs_global, 0]], kxy[[obs_global, 1]]];
                let size = recon
                    .observation_affine_shape(point_idx, img_idx, xy)
                    .map(|a| {
                        let col0 = (a[0][0] * a[0][0] + a[1][0] * a[1][0]).sqrt();
                        let col1 = (a[0][1] * a[0][1] + a[1][1] * a[1][1]).sqrt();
                        0.5 * (col0 + col1)
                    })
                    .unwrap_or(0.0);
                (obs_global, xy, size)
            } else {
                (0, [0.0, 0.0], 0.0)
            };

            // --- Compute per-observation reprojection error and ray angle ---
            let (reproj_error, ray_angle_deg) =
                compute_observation_metrics(&point_pos, image, camera, feature_xy);

            // Collect world-space ray for max-angle computation
            let cam_center = image.camera_center();
            let dir = point_pos - cam_center;
            let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
            if len > 1e-12 {
                world_rays.push([dir.x / len, dir.y / len, dir.z / len]);
            }

            let image_full_name = image.name.clone();
            let image_name = truncated_path_suffix(&image_full_name);

            self.observations.push(TrackObservationData {
                image_index: img_idx,
                feature_index,
                feature_xy,
                reproj_error,
                ray_angle_deg,
                feature_size,
                image_name,
                image_full_name,
            });
        }

        // Sort by image index (should already be sorted, but ensure it)
        self.observations.sort_by_key(|o| o.image_index);

        // Compute max angle between any pair of observation rays.
        self.max_angle_deg = compute_max_pairwise_angle(&world_rays);

        // Triangulation observability diagnostics for this point.
        let (condition_number, inverse_depth_z) = compute_point_diagnostics(recon, point_idx);
        self.condition_number = condition_number;
        self.inverse_depth_z = inverse_depth_z;
    }

    /// Clear all cached state (e.g. when reconstruction changes).
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.prepared_point = None;
        self.observations.clear();
        self.max_angle_deg = 0.0;
        self.inverse_depth_z = f32::NAN;
        self.condition_number = f32::NAN;
        self.thumbnail_textures.clear();
        self.patch_frame = None;
        self.stored_patch_texture = None;
        self.rendered_patch_textures.clear();
        self.hash_prefix.clear();
        self.scroll_offset_y = None;
    }
}

/// Build the selected point's oriented patch frame from the stored patch
/// half-vectors, or `None` when the reconstruction carries no frame or the
/// point's `u` half-vector is zero (no patch for this point). The stored
/// arrays are half-*vectors* (`axis * half_extent`); split them into unit
/// axis and half-extent like `PatchCloud::from_halfvec_arrays`. For a point
/// at infinity the stored u/v are already the tangent frame — the same frame
/// applies, just re-marked with `w = 0`.
fn build_patch_frame(recon: &SfmrReconstruction, point_idx: usize) -> Option<OrientedPatch> {
    let u_arr = recon.patch_u_halfvec_xyz.as_ref()?;
    let v_arr = recon.patch_v_halfvec_xyz.as_ref()?;
    if point_idx >= u_arr.nrows() || point_idx >= v_arr.nrows() {
        return None;
    }
    let u = Vector3::new(
        u_arr[[point_idx, 0]] as f64,
        u_arr[[point_idx, 1]] as f64,
        u_arr[[point_idx, 2]] as f64,
    );
    let hu = u.norm();
    if hu <= 1e-12 {
        return None;
    }
    let v = Vector3::new(
        v_arr[[point_idx, 0]] as f64,
        v_arr[[point_idx, 1]] as f64,
        v_arr[[point_idx, 2]] as f64,
    );
    let hv = v.norm();
    let u_axis = u / hu;
    let v_axis = if hv > 1e-12 { v / hv } else { v };
    let point = &recon.points[point_idx];
    let mut patch = OrientedPatch::new(point.position, u_axis, v_axis, [hu, hv]);
    if point.w == 0.0 {
        patch.w = 0.0;
    }
    Some(patch)
}

/// Build the stored-patch header texture for the selected point from
/// `patch_bitmaps_y_x_rgba`, or `None` when the array is absent or the
/// point's bitmap is all-zero (no stored patch). Displays RGB only: the
/// alpha channel (per-texel cross-view confidence) is forced opaque.
fn build_stored_patch_texture(
    ctx: &egui::Context,
    recon: &SfmrReconstruction,
    point_idx: usize,
) -> Option<egui::TextureHandle> {
    let bitmaps = recon.patch_bitmaps_y_x_rgba.as_ref()?;
    if point_idx >= bitmaps.shape()[0] {
        return None;
    }
    let bitmap = bitmaps.index_axis(Axis(0), point_idx);
    let h = bitmap.shape()[0];
    let w = bitmap.shape()[1];
    let mut rgba: Vec<u8> = if let Some(slice) = bitmap.as_slice() {
        slice.to_vec()
    } else {
        bitmap.iter().copied().collect()
    };
    if rgba.iter().all(|&b| b == 0) {
        return None;
    }
    for px in rgba.chunks_exact_mut(4) {
        px[3] = 255;
    }
    let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
    Some(ctx.load_texture(
        format!("stored_patch_{point_idx}"),
        image,
        egui::TextureOptions::NEAREST,
    ))
}

/// Map reprojection error (pixels) to a green→yellow→red color.
///
/// - 0.0 px → green (0, 200, 0)
/// - 1.0 px → yellow (255, 255, 0)
/// - 2.0+ px → red (255, 0, 0)
fn error_color(error: f32) -> egui::Color32 {
    if error.is_nan() {
        return egui::Color32::from_rgb(128, 128, 128); // gray for N/A
    }
    let t = error.clamp(0.0, 2.0) / 2.0; // 0..1 over range 0..2 px
    if t < 0.5 {
        // green → yellow (t: 0..0.5 → s: 0..1)
        let s = t * 2.0;
        egui::Color32::from_rgb((s * 255.0) as u8, (200.0 + s * 55.0) as u8, 0)
    } else {
        // yellow → red (t: 0.5..1 → s: 0..1)
        let s = (t - 0.5) * 2.0;
        egui::Color32::from_rgb(255, ((1.0 - s) * 255.0) as u8, 0)
    }
}

/// Compute per-observation reprojection error and ray angle for one observation.
///
/// Returns `(reproj_error_px, ray_angle_deg)`. If the point is behind the
/// camera, returns `(NaN, NaN)`.
fn compute_observation_metrics(
    point_pos: &nalgebra::Point3<f64>,
    image: &sfmtool_core::SfmrImage,
    camera: &sfmtool_core::CameraIntrinsics,
    feature_xy: [f32; 2],
) -> (f32, f32) {
    // Transform point from world to camera space: p_cam = R * p_world + t
    let r = image.quaternion_wxyz.to_rotation_matrix();
    let p_cam = r * point_pos.coords + image.translation_xyz;

    // Canonical cameras look down -Z, so in-front points have z < 0 and depth
    // is -z. Point behind camera — return NaN to signal invalid.
    let depth = -p_cam.z;
    if depth <= 0.0 {
        return (f32::NAN, f32::NAN);
    }

    // Project to image plane (undistorted normalized canonical coords, p/(-z))
    let x = p_cam.x / depth;
    let y = p_cam.y / depth;

    // Apply distortion + intrinsics to get pixel coordinates
    let (u_proj, v_proj) = camera.project(x, y);

    // Reprojection error in pixels
    let du = u_proj - feature_xy[0] as f64;
    let dv = v_proj - feature_xy[1] as f64;
    let reproj_error = (du * du + dv * dv).sqrt() as f32;

    // Ray angle: angle between the observation ray and the actual point direction
    // Both computed in camera space.
    let obs_ray = camera.pixel_to_ray(feature_xy[0] as f64, feature_xy[1] as f64);
    let obs_ray = Vector3::new(obs_ray[0], obs_ray[1], obs_ray[2]);

    let point_dir = p_cam.normalize();

    let dot = obs_ray.dot(&point_dir).clamp(-1.0, 1.0);
    let ray_angle_deg = dot.acos().to_degrees() as f32;

    (reproj_error, ray_angle_deg)
}

/// Triangulation observability diagnostics for a 3D point, computed from the
/// rays from each observing camera to the *stored* point (no `.sift` reads):
/// `(condition_number, inverse_depth_z)`. Returns `(NaN, NaN)` for points at
/// infinity, missing points, or fewer than two usable rays. The per-ray angular
/// noise is `max(reproj_error, 1px) / f`, matching the classifier's policy.
pub(crate) fn compute_point_diagnostics(
    recon: &SfmrReconstruction,
    point_idx: usize,
) -> (f32, f32) {
    use sfmtool_core::reconstruction::triangulation::{depth_uncertainty_batch, triangulate_batch};

    let Some(pt) = recon.points.get(point_idx) else {
        return (f32::NAN, f32::NAN);
    };
    if pt.is_at_infinity() {
        return (f32::NAN, f32::NAN);
    }
    let observations = recon.observations_for_point(point_idx);
    let noise = (pt.error as f64).max(1.0);
    let mut dirs = Vec::with_capacity(observations.len());
    let mut centers = Vec::with_capacity(observations.len());
    let mut sigma = Vec::with_capacity(observations.len());
    for obs in observations {
        let img_idx = obs.image_index as usize;
        let Some(image) = recon.images.get(img_idx) else {
            continue;
        };
        let center = image.camera_center();
        let dir = pt.position - center;
        let len = dir.norm();
        if len > 1e-12 {
            dirs.push(dir / len);
            centers.push(center);
            let (fx, fy) = recon.cameras[image.camera_index as usize].focal_lengths();
            sigma.push(noise / fx.max(fy));
        }
    }
    if dirs.len() < 2 {
        return (f32::NAN, f32::NAN);
    }
    let offsets = [0usize, dirs.len()];
    let tris = triangulate_batch(&dirs, &centers, &offsets);
    let dus = depth_uncertainty_batch(&tris, &dirs, &centers, &offsets, &sigma);
    (
        tris[0].condition_number as f32,
        dus[0].inverse_depth_z as f32,
    )
}

/// Compute the maximum angle (in degrees) between any pair of world-space rays.
pub(crate) fn compute_max_pairwise_angle(rays: &[[f64; 3]]) -> f32 {
    let mut min_dot = 1.0f64;
    for i in 0..rays.len() {
        for j in (i + 1)..rays.len() {
            let dot = rays[i][0] * rays[j][0] + rays[i][1] * rays[j][1] + rays[i][2] * rays[j][2];
            if dot < min_dot {
                min_dot = dot;
            }
        }
    }
    min_dot.clamp(-1.0, 1.0).acos().to_degrees() as f32
}

/// A small "copy to clipboard" button drawn as two overlapping rectangles.
/// Returns true if clicked.
fn copy_button(ui: &mut egui::Ui, tooltip: &str) -> bool {
    let icon_size = ui.text_style_height(&egui::TextStyle::Body);
    let padding = 2.0;
    let total = icon_size + padding * 2.0;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(total, total), egui::Sense::click());

    if ui.is_rect_visible(rect) {
        let color = if response.hovered() {
            ui.visuals().strong_text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let stroke = egui::Stroke::new(1.0, color);

        // Two overlapping rounded rectangles (the standard "copy" icon).
        let inset = padding + 1.0;
        let offset = icon_size * 0.22;
        // Back rectangle (offset down-right)
        let back = egui::Rect::from_min_size(
            rect.min + egui::vec2(inset + offset, inset),
            egui::vec2(icon_size * 0.55, icon_size * 0.65),
        );
        ui.painter()
            .rect_stroke(back, 1.0, stroke, egui::StrokeKind::Outside);
        // Front rectangle (offset up-left, filled with panel background)
        let front = back.translate(egui::vec2(-offset, offset));
        ui.painter()
            .rect_filled(front, 1.0, ui.visuals().panel_fill);
        ui.painter()
            .rect_stroke(front, 1.0, stroke, egui::StrokeKind::Outside);
    }

    let clicked = response.clicked();
    response.on_hover_text(tooltip);
    clicked
}

/// Return a short display name from an image path, keeping the filename plus
/// its parent directory so that rig images sharing the same filename are
/// distinguishable. For example `images/fisheye_left/image_0345.jpg` becomes
/// `…/fisheye_left/image_0345.jpg`. Plain filenames without a parent are
/// returned as-is.
fn truncated_path_suffix(path_str: &str) -> String {
    let p = Path::new(path_str);
    let file_name = match p.file_name() {
        Some(f) => f.to_string_lossy(),
        None => return path_str.to_string(),
    };
    match p.parent().and_then(|par| par.file_name()) {
        Some(parent_dir) => format!("\u{2026}/{}/{}", parent_dir.to_string_lossy(), file_name),
        None => file_name.into_owned(),
    }
}
