// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image browser panel — horizontally-scrollable strip of 128x128 thumbnails.
//!
//! Uses manual offset-based panning instead of `ScrollArea` so that Windows
//! DirectManipulation gesture events can drive the horizontal scroll.

use std::collections::HashMap;

use eframe::egui::{self, Sense};
use ndarray::Axis;
use sfmtool_core::SfmrReconstruction;

use crate::platform::GestureEvent;

/// Response from the image browser (selection changes and hover state).
pub struct ImageBrowserResponse {
    /// If Some, the user clicked a thumbnail to select it.
    pub selection_changed: Option<Option<usize>>,
    /// If Some, user double-clicked a thumbnail — request camera view mode.
    pub request_camera_view: Option<usize>,
    /// If Some, animation advanced to this image — request instant camera switch
    /// (only meaningful when camera view mode is already active).
    pub request_camera_switch: Option<usize>,
    /// Image index currently under the pointer (for cross-panel hover).
    pub hovered_image: Option<usize>,
    /// Whether the pointer is currently inside the browser panel.
    pub has_pointer: bool,
}

/// Playback direction for image animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PlayDirection {
    Forward,
    Backward,
}

/// Animation playback state for the image browser.
struct AnimationState {
    /// Whether playback is currently active.
    playing: bool,
    /// Playback direction.
    direction: PlayDirection,
    /// Playback speed in frames per second.
    fps: f32,
    /// Wall-clock time of the last frame advance (from `ui.input(|i| i.time)`).
    last_time: f64,
    /// Whether to loop at sequence boundaries.
    looping: bool,
}

impl AnimationState {
    fn new() -> Self {
        Self {
            playing: false,
            direction: PlayDirection::Forward,
            fps: 10.0,
            last_time: 0.0,
            looping: true,
        }
    }

    fn reset(&mut self) {
        self.playing = false;
        self.last_time = 0.0;
    }
}

/// Navigation minibar state — thin bar below the thumbnail strip.
struct NavigationMinibar {
    /// Texture: width = num_images, height = 8, RGBA.
    /// Each column has 8 pixels representing the average color of each vertical eighth of the thumbnail.
    color_barcode: Option<egui::TextureHandle>,
    /// Number of images when the barcode was last built (for invalidation).
    cached_image_count: usize,
    /// Whether the minibar is currently being dragged.
    dragging: bool,
}

impl NavigationMinibar {
    fn new() -> Self {
        Self {
            color_barcode: None,
            cached_image_count: 0,
            dragging: false,
        }
    }

    fn invalidate(&mut self) {
        self.color_barcode = None;
        self.cached_image_count = 0;
    }
}

/// Image browser panel state.
pub struct ImageBrowser {
    /// Cached egui textures for thumbnails, keyed by image index.
    thumbnail_cache: HashMap<usize, egui::TextureHandle>,
    /// Number of images when the cache was last built (for invalidation).
    cached_image_count: usize,
    /// The selected image index from the previous frame (for auto-scroll).
    prev_selected: Option<usize>,
    /// Index of the next thumbnail to lazily load.
    next_lazy_load: usize,
    /// Horizontal scroll offset in logical pixels.
    offset_x: f32,
    /// Previous frame's thumbnail height, for rescaling offset on resize.
    prev_img_height: f32,
    /// Navigation minibar below the thumbnail strip.
    minibar: NavigationMinibar,
    /// Animation playback state.
    animation: AnimationState,
}

impl ImageBrowser {
    /// Creates a new image browser with empty cache.
    pub fn new() -> Self {
        Self {
            thumbnail_cache: HashMap::new(),
            cached_image_count: 0,
            prev_selected: None,
            next_lazy_load: 0,
            offset_x: 0.0,
            prev_img_height: 0.0,
            minibar: NavigationMinibar::new(),
            animation: AnimationState::new(),
        }
    }

    /// Whether animation playback is currently active.
    pub fn is_playing(&self) -> bool {
        self.animation.playing
    }

    /// Show the image browser strip in the given UI region.
    ///
    /// Returns an [`ImageBrowserResponse`] indicating any selection changes.
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        selected_image: Option<usize>,
        track_images: &[usize],
        hover_track_images: &[usize],
        hovered_image: Option<usize>,
        camera_view_image: Option<usize>,
        gesture_events: &[GestureEvent],
        scroll_input: &crate::platform::ScrollInput,
    ) -> ImageBrowserResponse {
        let mut response = ImageBrowserResponse {
            selection_changed: None,
            request_camera_view: None,
            request_camera_switch: None,
            hovered_image: None,
            has_pointer: false,
        };

        let num_images = recon.images.len();

        // Cache invalidation: if the reconstruction changed, clear everything.
        if num_images != self.cached_image_count {
            self.thumbnail_cache.clear();
            self.cached_image_count = num_images;
            self.next_lazy_load = 0;
            self.offset_x = 0.0;
            self.minibar.invalidate();
            self.animation.reset();
        }

        // Lazy-load up to 8 thumbnails per frame (background loading).
        let mut loaded_this_frame = 0;
        while loaded_this_frame < 8 && self.next_lazy_load < num_images {
            let idx = self.next_lazy_load;
            if !self.thumbnail_cache.contains_key(&idx) {
                self.load_thumbnail(ui.ctx(), recon, idx);
                loaded_this_frame += 1;
            }
            self.next_lazy_load += 1;
        }
        // Request repaint if there are more thumbnails to load.
        if self.next_lazy_load < num_images {
            ui.ctx().request_repaint();
        }

        let spacing = 4.0_f32;
        let label_height = 18.0_f32;
        let minibar_height = 20.0_f32;
        let img_height = (ui.available_height() - label_height - minibar_height).max(16.0);

        // When the panel resizes, rescale offset_x so the viewport center stays anchored.
        if self.prev_img_height > 0.0 && img_height != self.prev_img_height {
            let panel_width = ui.available_width();
            let center = self.offset_x + panel_width / 2.0;
            let scale = img_height / self.prev_img_height;
            self.offset_x = (center * scale - panel_width / 2.0).max(0.0);
        }
        self.prev_img_height = img_height;

        // Compute layout: (x_start, width) for each thumbnail.
        let mut thumb_positions: Vec<(f32, f32)> = Vec::with_capacity(num_images);
        let mut x = 0.0_f32;
        for i in 0..num_images {
            let cam = &recon.cameras[recon.images[i].camera_index as usize];
            let aspect = cam.width as f32 / cam.height as f32;
            let w = img_height * aspect;
            thumb_positions.push((x, w));
            x += w + spacing;
        }
        let total_width = if num_images > 0 { x - spacing } else { 0.0 };

        // Allocate the full panel area for interaction.
        let (panel_response, painter) =
            ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let full_rect = panel_response.rect;

        // Split: thumbnail area on top, minibar at bottom.
        let minibar_rect = egui::Rect::from_min_max(
            egui::pos2(full_rect.left(), full_rect.bottom() - minibar_height),
            full_rect.max,
        );
        let panel_rect = egui::Rect::from_min_max(
            full_rect.min,
            egui::pos2(full_rect.right(), minibar_rect.top()),
        );

        // ── Input handling ─────────────────────────────────────────────

        // Check if the pointer is in the minibar for minibar-specific interaction.
        let pointer_pos = ui.input(|i| i.pointer.interact_pos());
        let pointer_in_minibar = pointer_pos
            .map(|p| minibar_rect.contains(p))
            .unwrap_or(false);

        // Play button rect on the left edge of the minibar (computed early for exclusion).
        let play_btn_size = minibar_height - 4.0;
        let play_btn_rect = egui::Rect::from_min_size(
            egui::pos2(minibar_rect.left() + 2.0, minibar_rect.top() + 2.0),
            egui::vec2(play_btn_size, play_btn_size),
        );
        let pointer_in_play_btn = pointer_pos
            .map(|p| play_btn_rect.contains(p))
            .unwrap_or(false);

        // Minibar: click or drag to select the image at that position.
        // Exclude the play button area from minibar interaction.
        if pointer_in_minibar && !pointer_in_play_btn && panel_response.drag_started() {
            self.minibar.dragging = true;
        }
        if !panel_response.dragged() {
            self.minibar.dragging = false;
        }

        let minibar_select = (self.minibar.dragging
            || (pointer_in_minibar && !pointer_in_play_btn && panel_response.clicked()))
            && num_images > 0;
        if minibar_select {
            if let Some(pos) = pointer_pos {
                let frac = ((pos.x - minibar_rect.left()) / minibar_rect.width()).clamp(0.0, 1.0);
                let idx = ((frac * num_images as f32) as usize).min(num_images - 1);
                response.selection_changed = Some(Some(idx));
            }
        }

        // ── Animation keyboard controls ──────────────────────────────
        if num_images >= 2 {
            let space = ui.input(|i| i.key_pressed(egui::Key::Space));
            let left = ui.input(|i| i.key_pressed(egui::Key::ArrowLeft));
            let right = ui.input(|i| i.key_pressed(egui::Key::ArrowRight));
            let bracket_left = ui.input(|i| i.key_pressed(egui::Key::OpenBracket));
            let bracket_right = ui.input(|i| i.key_pressed(egui::Key::CloseBracket));

            if space {
                self.animation.playing = !self.animation.playing;
                if self.animation.playing {
                    self.animation.last_time = ui.input(|i| i.time);
                    // Auto-select first/last image if none selected.
                    if selected_image.is_none() {
                        let start = match self.animation.direction {
                            PlayDirection::Forward => 0,
                            PlayDirection::Backward => num_images - 1,
                        };
                        response.selection_changed = Some(Some(start));
                    }
                }
            }
            if left {
                self.animation.playing = false;
                let current = selected_image.unwrap_or(0);
                let prev = if current == 0 {
                    num_images - 1
                } else {
                    current - 1
                };
                response.selection_changed = Some(Some(prev));
                response.request_camera_switch = Some(prev);
            }
            if right {
                self.animation.playing = false;
                let current = selected_image.unwrap_or(0);
                let next = if current + 1 >= num_images {
                    0
                } else {
                    current + 1
                };
                response.selection_changed = Some(Some(next));
                response.request_camera_switch = Some(next);
            }
            if bracket_left {
                self.animation.fps = (self.animation.fps / 2.0).max(1.0);
            }
            if bracket_right {
                self.animation.fps = (self.animation.fps * 2.0).min(60.0);
            }
        }

        // ── Animation frame advance ──────────────────────────────────
        if self.animation.playing && num_images >= 2 {
            let now = ui.input(|i| i.time);
            let dt = now - self.animation.last_time;
            let frame_interval = 1.0 / self.animation.fps as f64;

            if dt >= frame_interval {
                self.animation.last_time = now;
                let current = selected_image.unwrap_or(0);
                let next = match self.animation.direction {
                    PlayDirection::Forward => {
                        if current + 1 >= num_images {
                            if self.animation.looping {
                                0
                            } else {
                                self.animation.playing = false;
                                current
                            }
                        } else {
                            current + 1
                        }
                    }
                    PlayDirection::Backward => {
                        if current == 0 {
                            if self.animation.looping {
                                num_images - 1
                            } else {
                                self.animation.playing = false;
                                current
                            }
                        } else {
                            current - 1
                        }
                    }
                };
                if next != current {
                    response.selection_changed = Some(Some(next));
                    response.request_camera_switch = Some(next);
                }
            }
            ui.ctx().request_repaint();
        }

        // Pause animation on manual interaction (minibar click/drag).
        if minibar_select {
            self.animation.playing = false;
        }

        if !self.minibar.dragging {
            if crate::platform::pointer_in_rect(ui.ctx(), panel_rect) {
                // DirectManipulation gesture events → horizontal pan.
                for event in gesture_events {
                    if let GestureEvent::Pan { dx, .. } = event {
                        self.offset_x += *dx as f32;
                    }
                }

                // Scroll wheel / trackpad scroll → horizontal pan.
                // Uses pre-accumulated ScrollInput with DM-aware suppression.
                if scroll_input.has_trackpad_scroll() {
                    let delta = scroll_input.delta;
                    self.offset_x -= delta.x + delta.y;
                } else if scroll_input.has_mouse_wheel() {
                    let delta = scroll_input.delta;
                    let multiplier = if matches!(scroll_input.unit, egui::MouseWheelUnit::Line) {
                        50.0
                    } else {
                        200.0 // Page
                    };
                    self.offset_x -= (delta.x + delta.y) * multiplier;
                }
            }

            // Left-button drag → horizontal pan (grab the content).
            if panel_response.dragged() {
                self.offset_x -= panel_response.drag_delta().x;
            }
        }

        // Auto-scroll when the selection changes externally.
        let selection_changed_this_frame = selected_image != self.prev_selected;
        self.prev_selected = selected_image;

        if selection_changed_this_frame {
            if let Some(idx) = selected_image {
                if let Some(&(thumb_x, thumb_w)) = thumb_positions.get(idx) {
                    let visible_left = self.offset_x;
                    let visible_right = self.offset_x + panel_rect.width();
                    let thumb_center = thumb_x + thumb_w / 2.0;
                    if thumb_x < visible_left || thumb_x + thumb_w > visible_right {
                        self.offset_x = (thumb_center - panel_rect.width() / 2.0).max(0.0);
                    }
                }
            }
        }

        // Clamp offset to valid range.
        let max_offset = (total_width - panel_rect.width()).max(0.0);
        self.offset_x = self.offset_x.clamp(0.0, max_offset);

        // ── Rendering ──────────────────────────────────────────────────

        response.has_pointer = panel_response.hovered();
        let clicked = panel_response.clicked() && !pointer_in_minibar;
        let double_clicked = panel_response.double_clicked() && !pointer_in_minibar;
        let font = egui::FontId::proportional(11.0);

        for (i, &(pos_x, thumb_w)) in thumb_positions.iter().enumerate() {
            let screen_x = panel_rect.left() + pos_x - self.offset_x;

            // Cull thumbnails that are completely off-screen.
            if screen_x + thumb_w < panel_rect.left() || screen_x > panel_rect.right() {
                continue;
            }

            // Ensure thumbnail is loaded (may not have been lazy-loaded yet).
            if !self.thumbnail_cache.contains_key(&i) {
                self.load_thumbnail(ui.ctx(), recon, i);
            }

            let thumb_rect = egui::Rect::from_min_size(
                egui::pos2(screen_x, panel_rect.top()),
                egui::vec2(thumb_w, img_height),
            );

            // Draw thumbnail image or placeholder.
            if let Some(texture) = self.thumbnail_cache.get(&i) {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                let mut mesh = egui::Mesh::with_texture(texture.id());
                mesh.add_rect_with_uv(thumb_rect, uv, egui::Color32::WHITE);
                painter.add(egui::Shape::mesh(mesh));
            } else {
                painter.rect_filled(thumb_rect, 0.0, egui::Color32::from_gray(40));
            }

            // Hover-track border: soft semi-transparent orange when hovered
            // point observes this image. Larger and softer than the solid
            // selected-track border to feel transient rather than committed.
            let is_in_hover_track =
                !hover_track_images.is_empty() && hover_track_images.contains(&i);
            if is_in_hover_track && !track_images.contains(&i) {
                painter.rect_stroke(
                    thumb_rect.expand(5.0),
                    2.0,
                    egui::Stroke::new(3.0, egui::Color32::from_rgba_unmultiplied(255, 165, 0, 120)),
                    egui::StrokeKind::Outside,
                );
            }

            // Hovered-image highlight from 3D viewer (bright white border,
            // matching the full-opacity white frustum hover in the 3D viewport).
            let is_hovered_from_3d = hovered_image == Some(i) && selected_image != Some(i);
            if is_hovered_from_3d {
                painter.rect_stroke(
                    thumb_rect.expand(1.0),
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                    egui::StrokeKind::Outside,
                );
            }

            // Concentric highlight borders (outermost to innermost):
            //   White  (6px) = camera view mode
            //   Orange (4px) = track membership
            //   Cyan   (2px) = image selection
            let is_camera_view = camera_view_image == Some(i);
            let is_in_track = track_images.contains(&i);
            let is_selected = selected_image == Some(i);
            if is_camera_view {
                painter.rect_stroke(
                    thumb_rect.expand(6.0),
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                    egui::StrokeKind::Outside,
                );
            }
            if is_in_track {
                painter.rect_stroke(
                    thumb_rect.expand(4.0),
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 165, 0)),
                    egui::StrokeKind::Outside,
                );
            }
            if is_selected {
                painter.rect_stroke(
                    thumb_rect.expand(2.0),
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::CYAN),
                    egui::StrokeKind::Outside,
                );
            }

            // Label below the thumbnail.
            painter.text(
                egui::pos2(
                    screen_x + thumb_w / 2.0,
                    panel_rect.top() + img_height + 2.0,
                ),
                egui::Align2::CENTER_TOP,
                format!("{i}"),
                font.clone(),
                egui::Color32::from_gray(180),
            );

            // Hit-test for click / double-click / hover on this thumbnail.
            if let Some(pos) = pointer_pos {
                if thumb_rect.contains(pos) {
                    // Report hover for cross-panel feedback.
                    response.hovered_image = Some(i);

                    if double_clicked {
                        self.animation.playing = false;
                        response.request_camera_view = Some(i);
                        response.selection_changed = Some(Some(i));
                    } else if clicked {
                        self.animation.playing = false;
                        response.selection_changed = Some(Some(i));
                    }
                }
            }
        }

        // ── Navigation minibar ────────────────────────────────────────

        // Build barcode texture once all thumbnails are loaded.
        if self.minibar.color_barcode.is_none()
            && self.thumbnail_cache.len() == num_images
            && num_images > 0
        {
            self.build_barcode(ui.ctx(), recon);
        }

        // Background: dark fill.
        painter.rect_filled(minibar_rect, 0.0, egui::Color32::from_gray(20));

        // Color barcode.
        if let Some(tex) = &self.minibar.color_barcode {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            let mut mesh = egui::Mesh::with_texture(tex.id());
            mesh.add_rect_with_uv(minibar_rect, uv, egui::Color32::WHITE);
            painter.add(egui::Shape::mesh(mesh));
        }

        // Play controls overlay on left edge of minibar (uses pre-computed play_btn_rect).
        {
            let btn_center = play_btn_rect.center();
            let half = play_btn_size / 2.0 - 2.0;

            // Draw play or pause icon.
            if self.animation.playing {
                // Pause: two vertical bars.
                let bar_w = 2.0;
                let gap = 2.0;
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(btn_center.x - gap - bar_w, btn_center.y - half),
                        egui::vec2(bar_w, half * 2.0),
                    ),
                    0.0,
                    egui::Color32::WHITE,
                );
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(btn_center.x + gap, btn_center.y - half),
                        egui::vec2(bar_w, half * 2.0),
                    ),
                    0.0,
                    egui::Color32::WHITE,
                );
            } else {
                // Play: right-pointing triangle.
                let points = vec![
                    egui::pos2(btn_center.x - half * 0.6, btn_center.y - half),
                    egui::pos2(btn_center.x + half * 0.8, btn_center.y),
                    egui::pos2(btn_center.x - half * 0.6, btn_center.y + half),
                ];
                painter.add(egui::Shape::convex_polygon(
                    points,
                    egui::Color32::WHITE,
                    egui::Stroke::NONE,
                ));
            }

            // Click on play button toggles playback.
            if pointer_in_play_btn && panel_response.clicked() {
                self.animation.playing = !self.animation.playing;
                if self.animation.playing {
                    self.animation.last_time = ui.input(|i| i.time);
                    if selected_image.is_none() && num_images >= 2 {
                        response.selection_changed = Some(Some(0));
                    }
                }
            }

            // FPS label to the right of the button.
            let fps_text = if self.animation.fps == self.animation.fps.floor() {
                format!("{} fps", self.animation.fps as u32)
            } else {
                format!("{:.1} fps", self.animation.fps)
            };
            painter.text(
                egui::pos2(play_btn_rect.right() + 4.0, btn_center.y),
                egui::Align2::LEFT_CENTER,
                fps_text,
                egui::FontId::proportional(10.0),
                egui::Color32::from_gray(180),
            );
        }

        // Viewport indicator: semi-transparent overlay showing the visible portion.
        if total_width > 0.0 {
            let vis_frac_start = self.offset_x / total_width;
            let vis_frac_end = (self.offset_x + panel_rect.width()) / total_width;
            let ind_left = minibar_rect.left() + vis_frac_start * minibar_rect.width();
            let ind_right = minibar_rect.left() + vis_frac_end.min(1.0) * minibar_rect.width();
            let ind_rect = egui::Rect::from_min_max(
                egui::pos2(ind_left, minibar_rect.top()),
                egui::pos2(ind_right, minibar_rect.bottom()),
            );
            painter.rect_stroke(
                ind_rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::WHITE),
                egui::StrokeKind::Outside,
            );
        }

        // Selection ticks.
        if num_images > 0 {
            let tick_width = 2.0_f32;
            // Cyan tick for selected image.
            if let Some(idx) = selected_image {
                let frac = (idx as f32 + 0.5) / num_images as f32;
                let x = minibar_rect.left() + frac * minibar_rect.width();
                let tick_rect = egui::Rect::from_min_max(
                    egui::pos2(x - tick_width / 2.0, minibar_rect.top()),
                    egui::pos2(x + tick_width / 2.0, minibar_rect.bottom()),
                );
                painter.rect_filled(tick_rect, 0.0, egui::Color32::CYAN);
            }
            // Orange ticks for track images.
            let track_color = egui::Color32::from_rgb(255, 165, 0);
            for &idx in track_images {
                let frac = (idx as f32 + 0.5) / num_images as f32;
                let x = minibar_rect.left() + frac * minibar_rect.width();
                let tick_rect = egui::Rect::from_min_max(
                    egui::pos2(x - tick_width / 2.0, minibar_rect.top()),
                    egui::pos2(x + tick_width / 2.0, minibar_rect.bottom()),
                );
                painter.rect_filled(tick_rect, 0.0, track_color);
            }
        }

        response
    }

    /// Build the color barcode texture from thumbnail vertical band averages.
    ///
    /// Each image contributes a column of 8 pixels, where each pixel is the
    /// average color of the corresponding vertical eighth of the thumbnail.
    fn build_barcode(&mut self, ctx: &egui::Context, recon: &SfmrReconstruction) {
        const BANDS: usize = 8;
        const THUMB_H: usize = 128;
        let rows_per_band = THUMB_H / BANDS; // 16

        let num_images = recon.images.len();
        // Texture layout: width = num_images, height = BANDS, row-major (top to bottom).
        let mut pixels = Vec::with_capacity(num_images * BANDS * 4);
        // egui textures are stored row-major, so we iterate band (row) first.
        for band in 0..BANDS {
            let y_start = band * rows_per_band;
            let y_end = y_start + rows_per_band;
            for i in 0..num_images {
                let rgb_slice = recon.thumbnails_y_x_rgb.index_axis(Axis(0), i);
                // Shape is (128, 128, 3). Average the horizontal band [y_start..y_end].
                let (mut r_sum, mut g_sum, mut b_sum) = (0u64, 0u64, 0u64);
                let mut count = 0u64;
                for y in y_start..y_end {
                    for x in 0..THUMB_H {
                        r_sum += rgb_slice[[y, x, 0]] as u64;
                        g_sum += rgb_slice[[y, x, 1]] as u64;
                        b_sum += rgb_slice[[y, x, 2]] as u64;
                        count += 1;
                    }
                }
                let r = (r_sum / count) as u8;
                let g = (g_sum / count) as u8;
                let b = (b_sum / count) as u8;
                pixels.extend_from_slice(&[r, g, b, 255]);
            }
        }
        let image = egui::ColorImage::from_rgba_unmultiplied([num_images, BANDS], &pixels);
        let texture = ctx.load_texture("minibar_barcode", image, egui::TextureOptions::LINEAR);
        self.minibar.color_barcode = Some(texture);
        self.minibar.cached_image_count = num_images;
    }

    /// Load a single thumbnail into the texture cache.
    fn load_thumbnail(&mut self, ctx: &egui::Context, recon: &SfmrReconstruction, idx: usize) {
        let rgb_slice = recon.thumbnails_y_x_rgb.index_axis(Axis(0), idx);
        // The slice is (128, 128, 3) in C-order. Get contiguous data.
        let rgb_data: Vec<u8> = if let Some(slice) = rgb_slice.as_slice() {
            slice.to_vec()
        } else {
            rgb_slice.iter().copied().collect()
        };

        let mut rgba = Vec::with_capacity(128 * 128 * 4);
        for pixel in rgb_data.chunks_exact(3) {
            rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
        }
        let image = egui::ColorImage::from_rgba_unmultiplied([128, 128], &rgba);
        let texture = ctx.load_texture(format!("thumb_{idx}"), image, egui::TextureOptions::LINEAR);
        self.thumbnail_cache.insert(idx, texture);
    }
}