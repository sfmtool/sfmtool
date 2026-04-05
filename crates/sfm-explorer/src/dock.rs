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
use crate::state::{AppState, FeatureDisplaySettings, OverlayMode};
use crate::viewer_3d::Viewer3D;
use sfmtool_core::SfmrReconstruction;

/// Tabs that can appear in the dock area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    Viewer3D,
    ImageBrowser,
    ImageDetail,
    PointTrackDetail,
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
    pub hover_pick_id: u32,
    pub gesture_events: &'a [platform::GestureEvent],
    pub scroll_input: &'a platform::ScrollInput,
    pub diagnostics: Option<(u32, u32, u32, u32)>,
    pub handler_ok: bool,
}

impl TabViewer for TabContext<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewer3D => "3D Viewer".into(),
            Tab::ImageBrowser => "Image Browser".into(),
            Tab::ImageDetail => "Image Detail".into(),
            Tab::PointTrackDetail => "Point Track".into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Viewer3D => {
                if let Some(ref recon) = self.state.reconstruction {
                    self.viewer_3d.show(
                        ui,
                        recon,
                        &mut self.state.selected_image,
                        self.state.show_grid,
                        self.state.length_scale,
                        self.gesture_events,
                        self.scroll_input,
                        self.diagnostics,
                        self.handler_ok,
                        self.scene_texture_id,
                        self.hover_depth,
                        self.hover_pick_id,
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
                if let Some(ref recon) = self.state.reconstruction {
                    let track_images = compute_track_images(self.state, recon);
                    let hover_track_images = compute_hover_track_images(self.state, recon);
                    let camera_view_image =
                        self.viewer_3d.camera_view.as_ref().map(|cv| cv.image_index);
                    let response = self.image_browser.show(
                        ui,
                        recon,
                        self.state.selected_image,
                        &track_images,
                        &hover_track_images,
                        self.state.hovered_image,
                        camera_view_image,
                        self.gesture_events,
                        self.scroll_input,
                    );
                    if let Some(new_sel) = response.selection_changed {
                        self.state.selected_image = new_sel;
                    }
                    if response.has_pointer {
                        // Browser owns hover state when it has the pointer.
                        self.state.hovered_image = response.hovered_image;
                        // Clear point hover from other panels since browser
                        // doesn't produce hovered_point.
                        self.state.hovered_point = None;
                    }
                    if let Some(img_idx) = response.request_camera_view {
                        let current_time = ui.input(|i| i.time);
                        if self.viewer_3d.camera_view.is_some() {
                            self.viewer_3d.animated_switch_camera_view(
                                img_idx,
                                recon,
                                current_time,
                            );
                        } else {
                            self.viewer_3d
                                .enter_camera_view(img_idx, recon, current_time);
                        }
                    }
                    // Instant camera switch during animation playback.
                    if let Some(img_idx) = response.request_camera_switch {
                        if self.viewer_3d.camera_view.is_some() {
                            self.viewer_3d.switch_camera_view(img_idx, recon);
                        }
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("No reconstruction loaded");
                    });
                }
            }
            Tab::ImageDetail => {
                if let Some(ref recon) = self.state.reconstruction {
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

                    let sift = self.state.selected_image.and_then(|idx| {
                        let read_count = read_count_for_image(idx);
                        crate::state::ensure_sift_cached(
                            &mut self.state.sift_cache,
                            recon,
                            idx,
                            read_count,
                        )
                    });
                    let detail_response = self.image_detail.show(
                        ui,
                        recon,
                        self.state.selected_image,
                        self.state.selected_point,
                        self.state.hovered_point,
                        self.image_browser.is_playing(),
                        self.gesture_events,
                        self.scroll_input,
                        sift,
                        &self.state.feature_display,
                    );
                    if let Some(point_idx) = detail_response.select_point {
                        self.state.selected_point = Some(point_idx);
                    }
                    if detail_response.has_pointer {
                        // Detail owns hover state when it has the pointer.
                        self.state.hovered_point = detail_response.hovered_point;
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
                if let Some(ref recon) = self.state.reconstruction {
                    // Ensure SIFT positions are cached for all images in the track.
                    if let Some(pt_idx) = self.state.selected_point {
                        if pt_idx < recon.points.len() {
                            for img_idx in recon.track_image_indices(pt_idx) {
                                let need = recon.max_track_feature_index[img_idx] as usize + 1;
                                crate::state::ensure_sift_cached(
                                    &mut self.state.sift_cache,
                                    recon,
                                    img_idx,
                                    need,
                                );
                            }
                        }
                    }
                    let track_response = self.point_track_detail.show(
                        ui,
                        recon,
                        self.state.selected_point,
                        self.state.hovered_image,
                        &self.state.sift_cache,
                        self.gesture_events,
                        self.scroll_input,
                    );
                    if let Some(img_idx) = track_response.select_image {
                        self.state.selected_image = Some(img_idx);
                    }
                    if let Some(img_idx) = track_response.request_camera_view {
                        let current_time = ui.input(|i| i.time);
                        if self.viewer_3d.camera_view.is_some() {
                            self.viewer_3d.animated_switch_camera_view(
                                img_idx,
                                recon,
                                current_time,
                            );
                        } else {
                            self.viewer_3d
                                .enter_camera_view(img_idx, recon, current_time);
                        }
                    }
                    if track_response.has_pointer {
                        // Track detail owns hover state when it has the pointer.
                        self.state.hovered_image = track_response.hovered_image;
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
pub(crate) fn compute_track_images(state: &AppState, recon: &SfmrReconstruction) -> Vec<usize> {
    let Some(point_idx) = state.selected_point else {
        return Vec::new();
    };
    if point_idx >= recon.points.len() {
        return Vec::new();
    }
    recon.track_image_indices(point_idx)
}

/// Return the image indices in the hovered point's track, or empty if none.
pub(crate) fn compute_hover_track_images(
    state: &AppState,
    recon: &SfmrReconstruction,
) -> Vec<usize> {
    let Some(point_idx) = state.hovered_point else {
        return Vec::new();
    };
    // Suppress if same as selected point (selected track is already shown).
    if state.selected_point == Some(point_idx) {
        return Vec::new();
    }
    if point_idx >= recon.points.len() {
        return Vec::new();
    }
    recon.track_image_indices(point_idx)
}
