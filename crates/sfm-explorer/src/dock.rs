// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Dock layout types and tab rendering.
//!
//! Defines the four-panel dock layout (3D Viewer, Image Browser, Image Detail,
//! Point Track Detail) and the `TabViewer` implementation that renders each
//! panel's content.

use egui_dock::TabViewer;

use crate::image_browser::ImageBrowser;
use crate::image_detail::ImageDetail;
use crate::platform;
use crate::point_track_detail::PointTrackDetail;
use crate::scene::{selected_node, ImageRef, PointRef, SceneNode};
use crate::state::{AppState, FeatureDisplaySettings, OverlayMode};
use crate::viewer_3d::Viewer3D;

/// Tabs that can appear in the dock area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    Viewer3D,
    ImageBrowser,
    ImageDetail,
    PointTrackDetail,
}

impl Tab {
    pub(crate) fn title(self) -> &'static str {
        match self {
            Tab::Viewer3D => "3D Viewer",
            Tab::ImageBrowser => "Image Browser",
            Tab::ImageDetail => "Image Detail",
            Tab::PointTrackDetail => "Point Track",
        }
    }
}

/// Holds mutable references to all state needed to render any tab.
pub(crate) struct TabContext<'a> {
    pub state: &'a mut AppState,
    pub viewer_3d: &'a mut Viewer3D,
    pub image_browser: &'a mut ImageBrowser,
    pub image_detail: &'a mut ImageDetail,
    pub point_track_detail: &'a mut PointTrackDetail,
    // Per-frame values needed by viewer_3d.show():
    pub scene_texture_id: Option<egui::TextureId>,
    pub hover_depth: Option<f32>,
    /// The entity under the cursor in the 3D viewport, decoded from the pick
    /// buffer — a ref, so the overlay knows which reconstruction it names.
    pub hover_pick: Option<crate::scene_renderer::PickTarget>,
    pub gesture_events: &'a [platform::GestureEvent],
    pub scroll_input: &'a platform::ScrollInput,
    pub diagnostics: Option<(u32, u32, u32, u32)>,
    pub handler_ok: bool,
}

impl TabViewer for TabContext<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Viewer3D => {
                if self.state.selected_recon.is_some() {
                    // The HUD goes up before the viewport claims the rect: it
                    // lives on its own `Area` layer (so it still paints on top),
                    // and `show` below consults the rect it occupies to arbitrate
                    // every pointer input path.
                    self.viewer_3d
                        .show_hud(ui, self.state, self.diagnostics, self.handler_ok);
                }
                // Fetched only after `show_hud` has handed back its `&mut
                // AppState`: the node borrows `state.scene`, and the two cannot
                // overlap.
                let node = selected_node(&self.state.scene, self.state.selected_recon);
                if let Some(node) = node {
                    self.viewer_3d.show(
                        ui,
                        &node.recon,
                        node.id,
                        &mut self.state.selected_image,
                        self.state.show_grid,
                        self.state.length_scale,
                        self.gesture_events,
                        self.scroll_input,
                        self.state.show_controls_help,
                        self.state.show_fps,
                        self.scene_texture_id,
                        self.hover_depth,
                        self.hover_pick,
                    );
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(100.0);
                            if let Some(ref msg) = self.state.status_message {
                                ui.colored_label(egui::Color32::RED, msg);
                                ui.add_space(20.0);
                            }
                            ui.heading("SfM Explorer");
                            ui.add_space(20.0);
                            ui.label("No reconstruction loaded.");
                            ui.add_space(10.0);
                            ui.label("Use File > Open to load a .sfmr file,");
                            ui.label("or File > Load Demo Data to see sample data.");
                        });
                    });
                }
            }
            Tab::ImageBrowser => {
                let node = selected_node(&self.state.scene, self.state.selected_recon);
                if let Some(node) = node {
                    let recon = &node.recon;
                    let id = node.id;
                    let track_images = compute_track_images(self.state, node);
                    let hover_track_images = compute_hover_track_images(self.state, node);
                    let camera_view_image = self
                        .viewer_3d
                        .camera_view
                        .as_ref()
                        .and_then(|cv| cv.image.index_in(id));
                    let response = self.image_browser.show(
                        ui,
                        recon,
                        id,
                        self.state.selected_image_in(id),
                        &track_images,
                        &hover_track_images,
                        self.state.hovered_image_in(id),
                        camera_view_image,
                        self.gesture_events,
                        self.scroll_input,
                    );
                    if let Some(new_sel) = response.selection_changed {
                        self.state.selected_image = new_sel.map(|i| ImageRef::new(id, i));
                    }
                    if response.has_pointer {
                        // Browser owns hover state when it has the pointer.
                        self.state.hovered_image =
                            response.hovered_image.map(|i| ImageRef::new(id, i));
                        // Clear point hover from other panels since browser
                        // doesn't produce hovered_point.
                        self.state.hovered_point = None;
                    }
                    if let Some(img_idx) = response.request_camera_view {
                        let current_time = ui.input(|i| i.time);
                        let image = ImageRef::new(id, img_idx);
                        if self.viewer_3d.camera_view.is_some() {
                            self.viewer_3d
                                .animated_switch_camera_view(image, recon, current_time);
                        } else {
                            self.viewer_3d.enter_camera_view(image, recon, current_time);
                        }
                    }
                    // Instant camera switch during animation playback.
                    if let Some(img_idx) = response.request_camera_switch {
                        if self.viewer_3d.camera_view.is_some() {
                            self.viewer_3d
                                .switch_camera_view(ImageRef::new(id, img_idx), recon);
                        }
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("No reconstruction loaded");
                    });
                }
            }
            Tab::ImageDetail => {
                let node = selected_node(&self.state.scene, self.state.selected_recon);
                if let Some(node) = node {
                    let recon = &node.recon;
                    let id = node.id;
                    // Read out before the cache borrows below: they hold `&mut`
                    // into `state`, which rules out an `&self.state` method call
                    // for as long as their results are alive.
                    let selected_image = self.state.selected_image_in(id);
                    let selected_point = self.state.selected_point_in(id);
                    let hovered_point = self.state.hovered_point_in(id);
                    // Overlay toolbar at the top of the detail panel
                    show_overlay_toolbar(ui, &mut self.state.feature_display);

                    // Determine how many SIFT features to load based on overlay mode
                    let read_count_for_image = |idx: usize| -> usize {
                        if self.state.feature_display.overlay_mode
                            == crate::state::OverlayMode::None
                        {
                            // Only need tracked features
                            recon.max_track_feature_index[idx] as usize + 1
                        } else {
                            // Need up to max_features (or all tracked features, whichever is more)
                            let tracked = recon.max_track_feature_index[idx] as usize + 1;
                            let display = self
                                .state
                                .feature_display
                                .max_features
                                .unwrap_or(usize::MAX);
                            tracked.max(display)
                        }
                    };

                    // Only `sift_files` reconstructions have `.sift` companions;
                    // an embedded_patches recon reads its keypoints inline, so
                    // skip the (always-failing, per-frame) cache probe.
                    let sift = selected_image.and_then(|idx| {
                        recon.feature_indexes()?;
                        let read_count = read_count_for_image(idx);
                        crate::state::ensure_sift_cached(
                            &mut self.state.sift_cache,
                            recon,
                            ImageRef::new(id, idx),
                            read_count,
                        )
                    });
                    // Full-res CPU pixels come from the shared cache (also
                    // used by the Point Track Detail patch tiles), so each
                    // image is decoded from disk at most once.
                    let full_res = selected_image.and_then(|idx| {
                        crate::state::ensure_full_res_cached(
                            &mut self.state.full_res_cache,
                            recon,
                            ImageRef::new(id, idx),
                        )
                    });
                    let detail_response = self.image_detail.show(
                        ui,
                        recon,
                        id,
                        selected_image,
                        selected_point,
                        hovered_point,
                        self.image_browser.is_playing(),
                        self.gesture_events,
                        self.scroll_input,
                        sift,
                        full_res,
                        &self.state.feature_display,
                    );
                    if let Some(point_idx) = detail_response.select_point {
                        self.state.selected_point = Some(PointRef::new(id, point_idx));
                    }
                    if detail_response.has_pointer {
                        // Detail owns hover state when it has the pointer.
                        self.state.hovered_point =
                            detail_response.hovered_point.map(|p| PointRef::new(id, p));
                        // Clear image hover from other panels since detail
                        // doesn't produce hovered_image.
                        self.state.hovered_image = None;
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("No reconstruction loaded");
                    });
                }
            }
            Tab::PointTrackDetail => {
                let node = selected_node(&self.state.scene, self.state.selected_recon);
                if let Some(node) = node {
                    let recon = &node.recon;
                    let id = node.id;
                    let selected_point = self.state.selected_point_in(id);
                    // Ensure SIFT positions are cached for all images in the
                    // track (sift_files only; embedded_patches reads keypoints
                    // inline, so the `.sift` probe would fail every time).
                    if recon.feature_indexes().is_some() {
                        if let Some(pt_idx) = selected_point {
                            if pt_idx < recon.points.len() {
                                for img_idx in recon.track_image_indices(pt_idx) {
                                    let need = recon.max_track_feature_index[img_idx] as usize + 1;
                                    crate::state::ensure_sift_cached(
                                        &mut self.state.sift_cache,
                                        recon,
                                        ImageRef::new(id, img_idx),
                                        need,
                                    );
                                }
                            }
                        }
                    }
                    // Pre-cache full-res images for every observing image of
                    // the selected point so the panel can render per-observation
                    // patch tiles from an immutable cache reference. Only
                    // needed when the recon carries patch frames (the tiles
                    // are gated on them).
                    if recon.patch_u_halfvec_xyz.is_some() {
                        if let Some(pt_idx) = selected_point {
                            if pt_idx < recon.points.len() {
                                for img_idx in recon.track_image_indices(pt_idx) {
                                    crate::state::ensure_full_res_cached(
                                        &mut self.state.full_res_cache,
                                        recon,
                                        ImageRef::new(id, img_idx),
                                    );
                                }
                            }
                        }
                    }
                    let track_response = self.point_track_detail.show(
                        ui,
                        recon,
                        id,
                        selected_point,
                        self.state.hovered_image_in(id),
                        &self.state.sift_cache,
                        &self.state.full_res_cache,
                        self.gesture_events,
                        self.scroll_input,
                    );
                    if let Some(img_idx) = track_response.select_image {
                        self.state.selected_image = Some(ImageRef::new(id, img_idx));
                    }
                    if let Some(img_idx) = track_response.request_camera_view {
                        let current_time = ui.input(|i| i.time);
                        let image = ImageRef::new(id, img_idx);
                        if self.viewer_3d.camera_view.is_some() {
                            self.viewer_3d
                                .animated_switch_camera_view(image, recon, current_time);
                        } else {
                            self.viewer_3d.enter_camera_view(image, recon, current_time);
                        }
                    }
                    if track_response.has_pointer {
                        // Track detail owns hover state when it has the pointer.
                        self.state.hovered_image =
                            track_response.hovered_image.map(|i| ImageRef::new(id, i));
                        // Clear point hover from other panels since track detail
                        // doesn't produce hovered_point.
                        self.state.hovered_point = None;
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("No reconstruction loaded");
                    });
                }
            }
        }
    }

    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        false
    }
}

/// Draw the overlay mode toolbar at the top of the image detail panel.
fn show_overlay_toolbar(ui: &mut egui::Ui, settings: &mut FeatureDisplaySettings) {
    ui.horizontal(|ui| {
        ui.label("Overlay:");
        egui::ComboBox::from_id_salt("overlay_mode")
            .selected_text(settings.overlay_mode.label())
            .width(100.0)
            .show_ui(ui, |ui| {
                for mode in OverlayMode::ALL {
                    ui.selectable_value(&mut settings.overlay_mode, mode, mode.label());
                }
            });

        if settings.overlay_mode != OverlayMode::None {
            ui.separator();
            ui.label("Max:");
            egui::ComboBox::from_id_salt("max_features")
                .selected_text(match settings.max_features {
                    Some(n) => format!("{n}"),
                    None => "All".to_string(),
                })
                .width(60.0)
                .show_ui(ui, |ui| {
                    for &preset in &[100usize, 500, 1000, 5000] {
                        ui.selectable_value(
                            &mut settings.max_features,
                            Some(preset),
                            format!("{preset}"),
                        );
                    }
                    ui.selectable_value(&mut settings.max_features, None, "All");
                });

            ui.separator();
            let mut has_size_filter =
                settings.min_feature_size.is_some() || settings.max_feature_size.is_some();
            ui.checkbox(&mut has_size_filter, "Min/max size:");
            ui.add(
                egui::DragValue::new(&mut settings.min_feature_size_value)
                    .range(0.0..=1000.0)
                    .speed(0.2)
                    .suffix("px"),
            );
            ui.add(
                egui::DragValue::new(&mut settings.max_feature_size_value)
                    .range(0.5..=1000.0)
                    .speed(0.5)
                    .suffix("px"),
            );
            if has_size_filter {
                settings.min_feature_size = Some(settings.min_feature_size_value);
                settings.max_feature_size = Some(settings.max_feature_size_value);
            } else {
                settings.min_feature_size = None;
                settings.max_feature_size = None;
            };

            if settings.overlay_mode == OverlayMode::Features {
                ui.separator();
                ui.checkbox(&mut settings.tracked_only, "Tracked only");
            }
        }
    });
    ui.separator();
}

/// Return the image indices in the selected point's track, or empty if none.
///
/// Both this and [`compute_hover_track_images`] return indices **local to
/// `node`**: a track never spans reconstructions, so the ids the caller already
/// holds are enough context and every consumer (frustum colors, browser
/// borders) works in one recon's index space anyway.
pub(crate) fn compute_track_images(state: &AppState, node: &SceneNode) -> Vec<usize> {
    let Some(point_idx) = state.selected_point_in(node.id) else {
        return Vec::new();
    };
    if point_idx >= node.recon.points.len() {
        return Vec::new();
    }
    node.recon.track_image_indices(point_idx)
}

/// Return the image indices in the hovered point's track, or empty if none.
pub(crate) fn compute_hover_track_images(state: &AppState, node: &SceneNode) -> Vec<usize> {
    let Some(point) = state.hovered_point else {
        return Vec::new();
    };
    // Suppress if same as selected point (selected track is already shown).
    if state.selected_point == Some(point) {
        return Vec::new();
    }
    let Some(point_idx) = point.index_in(node.id) else {
        return Vec::new();
    };
    if point_idx >= node.recon.points.len() {
        return Vec::new();
    }
    node.recon.track_image_indices(point_idx)
}
