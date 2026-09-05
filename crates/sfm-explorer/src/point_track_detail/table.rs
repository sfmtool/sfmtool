// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The scrollable per-observation table: column layout, header row, one row per
//! observation, and the thumbnails those rows draw.
//!
//! Rows are painted at fixed x-offsets rather than laid out by egui, so the
//! header and every row stay aligned; [`ColumnLayout`] is the single place
//! those offsets are computed.

use std::collections::HashMap;

use ndarray::Axis;
use sfmtool_core::camera::remap::ImageU8;
use sfmtool_core::SfmrReconstruction;

use super::{PointTrackDetail, PointTrackDetailResponse, PATCH_TILE, THUMB_SIZE};
use crate::colormap;
use crate::platform::{self, GestureEvent};
use crate::scene::{ImageRef, ReconId};
use crate::texture::thumbnail_color_image;

/// Height of one observation row: the thumbnail plus vertical padding.
const ROW_HEIGHT: f32 = THUMB_SIZE + 8.0;
/// Size of the feature dot overlay on thumbnails.
const DOT_RADIUS: f32 = 3.0;

/// Where this panel puts the top of the reprojection-error ramp, in pixels.
///
/// Fixed rather than fitted to the track, because a track's dots are read
/// against each other *and* against the absolute number in the Error column: a
/// range that shrank to the best and worst of seven observations would paint a
/// sub-pixel track in full red. The Image Detail overlay fits its range to the
/// image instead, for the opposite reason — there the question is which
/// features in *this* frame are the bad ones.
pub(super) const ERROR_RAMP_MAX_PX: f32 = 2.0;

/// The grey a dot gets when the observation has no error to colour — the point
/// is behind the camera, so `compute_observation_metrics` returned NaN.
///
/// Kept out of the ramp deliberately: "no measurement" is not a position on a
/// green-to-red scale, and a NaN normalized into one would come out at an end
/// of it and read as an extreme.
pub(super) const NO_ERROR_COLOR: egui::Color32 = egui::Color32::from_rgb(128, 128, 128);

/// The colour of one thumbnail's feature dot.
///
/// [`colormap::error_color`] is the same ramp the Image Detail panel's
/// reprojection-error overlay draws, so the two panels agree about what a
/// given error looks like at a given range.
pub(super) fn error_dot_color(error: f32) -> egui::Color32 {
    if error.is_nan() {
        return NO_ERROR_COLOR;
    }
    colormap::error_color(error, 0.0, ERROR_RAMP_MAX_PX)
}

/// Fixed column x-offsets, relative to the left edge of the table.
///
/// When the selected point has a patch frame, a rendered-patch tile is drawn
/// immediately right of the thumbnail and every text column shifts right by its
/// width; otherwise the layout is unchanged.
struct ColumnLayout {
    /// Whether the patch column is present; when false, `patch` is unused.
    has_patch: bool,
    patch: f32,
    image: f32,
    name: f32,
    feat: f32,
    size: f32,
    error: f32,
    angle: f32,
    xy: f32,
}

impl ColumnLayout {
    fn new(has_patch: bool) -> Self {
        let patch_shift = if has_patch { PATCH_TILE + 8.0 } else { 0.0 };
        let image = THUMB_SIZE + 8.0 + patch_shift;
        let name = image + 50.0;
        let feat = name + 170.0;
        let size = feat + 55.0;
        // Wide enough for the two-extent "123.4x56.7" form, not just one number.
        let error = size + 95.0;
        let angle = error + 60.0;
        Self {
            has_patch,
            patch: THUMB_SIZE + 8.0,
            image,
            name,
            feat,
            size,
            error,
            angle,
            xy: angle + 55.0,
        }
    }
}

impl PointTrackDetail {
    /// Draw the scrollable observation table.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn show_observation_table(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        recon_id: ReconId,
        hovered_image: Option<usize>,
        full_res_cache: &HashMap<ImageRef, Option<ImageU8>>,
        gesture_events: &[GestureEvent],
        scroll_input: &platform::ScrollInput,
        response: &mut PointTrackDetailResponse,
    ) {
        let extra_scroll_y = gesture_scroll_delta(ui, gesture_events, scroll_input);

        let mut scroll_area = egui::ScrollArea::vertical().auto_shrink([false, false]);

        if extra_scroll_y != 0.0 {
            // Negative because scroll offset increases when content moves up,
            // but pan dy is positive when panning up (content moves down).
            let current = self.scroll_offset_y.unwrap_or(0.0);
            let new_offset = (current - extra_scroll_y).max(0.0);
            scroll_area = scroll_area.vertical_scroll_offset(new_offset);
        }

        let cols = ColumnLayout::new(self.patch_frame.is_some());

        let scroll_output = scroll_area.show(ui, |ui| {
            draw_table_header(ui, &cols);

            for obs_i in 0..self.observations.len() {
                self.draw_observation_row(
                    ui,
                    recon,
                    recon_id,
                    obs_i,
                    hovered_image,
                    full_res_cache,
                    &cols,
                    response,
                );
            }
        });

        // Track scroll offset for next frame's DM gesture application.
        self.scroll_offset_y = Some(scroll_output.state.offset.y);
    }

    /// Draw one observation row: hover/selection background, thumbnail, patch
    /// tile, and the text columns. Updates `response` with hover and click
    /// outcomes for this row.
    #[allow(clippy::too_many_arguments)]
    fn draw_observation_row(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        recon_id: ReconId,
        obs_i: usize,
        hovered_image: Option<usize>,
        full_res_cache: &HashMap<ImageRef, Option<ImageU8>>,
        cols: &ColumnLayout,
        response: &mut PointTrackDetailResponse,
    ) {
        // Copy the row's fields out up front: drawing the thumbnail and patch
        // tile takes `&mut self`, which would conflict with holding a borrow
        // into `self.observations`.
        let obs = &self.observations[obs_i];
        let obs_image_index = obs.image_index;
        let obs_image = ImageRef::new(recon_id, obs_image_index);
        let obs_feature_index = obs.feature_index;
        let obs_feature_xy = obs.feature_xy;
        let obs_reproj_error = obs.reproj_error;
        let obs_ray_angle_deg = obs.ray_angle_deg;
        let obs_feature_extents = obs.feature_extents;
        let obs_image_name = obs.image_name.clone();
        let obs_image_full_name = obs.image_full_name.clone();
        let is_hovered_image = hovered_image == Some(obs_image_index);

        // Row background color for hover highlight
        let row_rect = ui.available_rect_before_wrap();
        let row_rect =
            egui::Rect::from_min_size(row_rect.min, egui::vec2(row_rect.width(), ROW_HEIGHT));

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
            obs_image,
            obs_feature_xy,
            obs_reproj_error,
        );

        // Rendered patch tile beside the thumbnail (embedded-patches
        // reconstructions only; rendered lazily from the shared
        // full-res image cache and cached per image index).
        if cols.has_patch {
            self.ensure_rendered_patch(ui.ctx(), recon, obs_image, full_res_cache);
            if let Some(texture) = self.rendered_patch_textures.get(&obs_image) {
                let patch_rect = egui::Rect::from_min_size(
                    egui::pos2(x0 + cols.patch, cy - PATCH_TILE / 2.0),
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
            egui::pos2(x0 + cols.image, cy),
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
            (x0 + cols.name)..=(x0 + cols.feat - name_col_gap),
            row_rect.y_range(),
        );
        painter.with_clip_rect(name_clip).text(
            egui::pos2(x0 + cols.feat - name_col_gap, cy),
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
            egui::pos2(x0 + cols.feat, cy),
            egui::Align2::LEFT_CENTER,
            format!("{}", obs_feature_index),
            font.clone(),
            text_color,
        );

        // Feature size
        let size_text = format_feature_size(obs_feature_extents);
        painter.text(
            egui::pos2(x0 + cols.size, cy),
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
            egui::pos2(x0 + cols.error, cy),
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
            egui::pos2(x0 + cols.angle, cy),
            egui::Align2::LEFT_CENTER,
            angle_text,
            font.clone(),
            text_color,
        );

        // Feature position
        painter.text(
            egui::pos2(x0 + cols.xy, cy),
            egui::Align2::LEFT_CENTER,
            format!("({:.1}, {:.1})", obs_feature_xy[0], obs_feature_xy[1]),
            font.clone(),
            text_color,
        );
    }

    /// Draw a thumbnail with a feature dot overlay colored by reprojection error.
    fn draw_thumbnail(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        image: ImageRef,
        feature_xy: [f32; 2],
        reproj_error: f32,
    ) {
        let img_idx = image.index();

        // Load thumbnail texture if not cached
        if !self.thumbnail_textures.contains_key(&image) {
            self.load_thumbnail(ui.ctx(), recon, image);
        }

        let (thumb_rect, _) =
            ui.allocate_exact_size(egui::vec2(THUMB_SIZE, THUMB_SIZE), egui::Sense::hover());

        if let Some(texture) = self.thumbnail_textures.get(&image) {
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
            let dot_color = error_dot_color(reproj_error);
            ui.painter()
                .circle_filled(dot_center, DOT_RADIUS, dot_color);
            // Dark outline for visibility
            ui.painter().circle_stroke(
                dot_center,
                DOT_RADIUS,
                egui::Stroke::new(1.0_f32, egui::Color32::BLACK),
            );
        }
    }

    /// Load a single thumbnail texture into the cache.
    fn load_thumbnail(&mut self, ctx: &egui::Context, recon: &SfmrReconstruction, image: ImageRef) {
        let idx = image.index();
        let color_image = thumbnail_color_image(recon.thumbnails_y_x_rgb.index_axis(Axis(0), idx));
        let texture = ctx.load_texture(
            format!("track_thumb_{idx}"),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.thumbnail_textures.insert(image, texture);
    }
}

/// Render the Size column's text for one observation.
///
/// `extents` are the observation's two *full* feature widths in pixels, larger
/// first — the span the drawn patch quad covers, not the half-axis radius the
/// affine shape's column norms give directly.
///
/// Both extents are always printed, larger first (`20.3x7.7`, or `14.0x14.0`
/// for a circular feature), so an obliquely-viewed patch reads as
/// foreshortened rather than as a merely smaller feature and the reader never
/// has to guess which form they are looking at. One decimal throughout; a
/// degenerate (zero) shape prints `N/A`; a fully collapsed (edge-on) shape
/// shows the collapse explicitly (`9.0x0.0`).
pub(super) fn format_feature_size(extents: [f32; 2]) -> String {
    let [major, minor] = extents;
    if !major.is_finite() || major <= 0.0 {
        return "N/A".to_string();
    }
    format!("{major:.1}x{minor:.1}")
}

/// Vertical scroll delta (in points) contributed by DirectManipulation pan
/// gestures and the scroll wheel/trackpad this frame. Zero unless the pointer
/// is inside the table.
fn gesture_scroll_delta(
    ui: &egui::Ui,
    gesture_events: &[GestureEvent],
    scroll_input: &platform::ScrollInput,
) -> f32 {
    let panel_rect = ui.available_rect_before_wrap();
    if !platform::pointer_in_rect(ui.ctx(), panel_rect) {
        return 0.0;
    }
    let mut delta = 0.0f32;
    for event in gesture_events {
        if let GestureEvent::Pan { dy, .. } = event {
            delta += *dy as f32;
        }
    }
    if scroll_input.has_trackpad_scroll() {
        delta += scroll_input.delta.y;
    } else if scroll_input.has_mouse_wheel() {
        let multiplier = if matches!(scroll_input.unit, egui::MouseWheelUnit::Line) {
            ROW_HEIGHT
        } else {
            200.0
        };
        delta += scroll_input.delta.y * multiplier;
    }
    delta
}

/// Paint the table's column headers at the same fixed x-offsets the rows use,
/// then allocate their height and a separator.
fn draw_table_header(ui: &mut egui::Ui, cols: &ColumnLayout) {
    let header_rect = ui.available_rect_before_wrap();
    let header_y = header_rect.min.y;
    let x0 = header_rect.min.x;
    let painter = ui.painter();
    let header_font = egui::TextStyle::Body.resolve(ui.style());
    let strong_color = ui.visuals().strong_text_color();
    let mut header_labels: Vec<(f32, &str)> = Vec::with_capacity(8);
    if cols.has_patch {
        header_labels.push((cols.patch, "Patch"));
    }
    header_labels.extend_from_slice(&[
        (cols.image, "Image"),
        (cols.name, "Name"),
        (cols.feat, "Feat #"),
        (cols.size, "Size"),
        (cols.error, "Error"),
        (cols.angle, "Angle"),
        (cols.xy, "Feature (x, y)"),
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
}
