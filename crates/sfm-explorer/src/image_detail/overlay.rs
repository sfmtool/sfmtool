// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Feature-overlay rendering for the image detail panel: the per-`OverlayMode`
//! draw branches, feature click hit-testing, cross-panel hover highlight, and
//! the point tooltip, plus the geometry/color helpers they use.

use super::{DisplayFeature, ImageDetail, ImageDetailResponse};
use crate::colormap;
use crate::state::{FeatureDisplaySettings, OverlayMode};
use sfmtool_core::bench::EditableTrack;
use sfmtool_core::spatial::PointCloud2;
use sfmtool_core::EditedReconstruction;

/// What the pointer found where a gesture was made: the feature nearest it
/// inside the hit radius, and whether that feature observes a point.
///
/// The distinction is the whole reason this is not an `Option<usize>`. An
/// `embedded_patches` node keeps its keypoints inline, one per observation, so
/// every feature it draws belongs to a point; a `sift_files` node draws the
/// `.sift` keypoints, and a keypoint the solve matched to nothing is a feature
/// with no point behind it. The gestures that need a point refuse on
/// [`FeatureHit::Unmatched`] in words that say which of the two it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FeatureHit {
    /// A feature observing this point, by its live index in the version.
    Point(usize),
    /// A keypoint this version has no live point for.
    Unmatched,
}

/// The start-a-cluster entry's label, and what Track View's empty
/// state quotes so the two cannot drift.
///
/// "On the bench" rather than the bare verb: nothing this entry does reaches
/// the reconstruction, and the commit in Track View is what crosses
/// back.
pub(crate) const START_CLUSTER_LABEL: &str = "Start cluster on the bench here";

/// The add-a-bench-observation entry's label. "Bench track" rather than
/// "track": the row joins the bench's **active** track, which is not the
/// selected point's.
pub(crate) const ADD_BENCH_OBSERVATION_LABEL: &str = "Add observation to bench track here";

/// What the panel is told about the node's bench.
///
/// The panel is handed a reconstruction value and a selection and holds nothing
/// else, so the two things the bench costs it are passed in: the dock reads
/// them off `AppState` beside the selection. Both the menu's two bench entries
/// and the layer that draws the active track
/// ([`mod@super::bench_track`]) read this one value, so what is offered and
/// what is drawn cannot disagree about which track is the active one.
///
/// It also carries the one tool setting the layer's handles are read by,
/// Track View's *Lock*, which is not bench state but belongs beside it: the
/// drag it decides the meaning of is a drag of the active track.
#[derive(Debug, Clone, Copy)]
pub struct BenchMenu<'a> {
    /// Why no step on this node can run -- a background task is holding it --
    /// or `None` when one can.
    pub busy: Option<&'a str>,
    /// The bench's active track, or `None` when no track is on the bench. A
    /// gesture that names no item means the active one.
    pub active_track: Option<&'a EditableTrack>,
    /// Track View's *Lock*: at the track stage, whether the dot slides the
    /// patch with every sighting following (`true`) or moves that one
    /// sighting's keypoint alone (`false`), in which case the outline's edges
    /// and corners take no drag. Read at the cluster stage not at all, where
    /// every handle is already one sighting's.
    pub lock: bool,
}

impl Default for BenchMenu<'_> {
    /// Nothing busy, nothing on the bench, and the lock on, which is what the
    /// box starts ticked at.
    fn default() -> Self {
        Self {
            busy: None,
            active_track: None,
            lock: true,
        }
    }
}

/// Which point the menu's Edit-on-Bench entry would stage, and why it is greyed
/// when it would stage none.
///
/// What it needs beyond a node no background task is holding is a point, which
/// the feature under the place the menu was opened at is what names. Both ways
/// of having none are refusals a reader can act on, so they are told apart:
/// there was no feature there at all, or there was one the solve matched to no
/// point.
pub(crate) fn edit_on_bench_entry(
    bench: BenchMenu<'_>,
    feature: Option<FeatureHit>,
) -> Result<usize, String> {
    if let Some(why) = bench.busy {
        return Err(why.to_string());
    }
    match feature {
        Some(FeatureHit::Point(point)) => Ok(point),
        Some(FeatureHit::Unmatched) => Err(UNMATCHED_FEATURE.to_string()),
        None => Err(NO_FEATURE.to_string()),
    }
}

/// Why Edit on Bench is greyed over a keypoint the solve matched to no point.
pub(crate) const UNMATCHED_FEATURE: &str =
    "That feature belongs to no 3D point, so there is no track to put on the bench.";

/// Why Edit on Bench is greyed where there is no feature at all.
pub(crate) const NO_FEATURE: &str = "There is no feature here, so there is no point to put on \
                                     the bench.";

/// Whether the menu's start-a-cluster entry can run, and why not when it is
/// greyed.
///
/// A pixel on a photograph of the node is the whole input, and the menu is only
/// ever drawn over one, so the only thing that stops it is the node being busy.
/// There is no `None` arm and no `sift_files` arm: a cluster is a seed in one
/// image's pixels, which is defined whatever the node's observations are backed
/// by.
pub(crate) fn start_cluster_entry(bench: BenchMenu<'_>) -> Result<(), String> {
    match bench.busy {
        Some(why) => Err(why.to_string()),
        None => Ok(()),
    }
}

/// Whether the menu's add-to-the-bench-track entry can run, and why not when it
/// is greyed.
///
/// What it needs beyond the pixel is a track to add to, which is the bench's
/// active one. An image the track already holds an observation in is **not** a
/// refusal: a second sighting in one image joins as a candidate and is scored
/// like any other, and what a track cannot do is hold two `in` observations of
/// one image, which is a verdict rather than this gesture
/// (`sfmtool_core::bench::add_observation`).
pub(crate) fn add_bench_observation_entry(bench: BenchMenu<'_>) -> Result<(), String> {
    if let Some(why) = bench.busy {
        return Err(why.to_string());
    }
    if bench.active_track.is_none() {
        return Err(
            "No track is being edited: tick Edit in Track View, or double-click a \
                    Bench item in the Scene tree."
                .to_string(),
        );
    }
    Ok(())
}

impl ImageDetail {
    /// The point the feature under the pointer observes, at the panel geometry
    /// the frame starts from, or `None` where there is no such feature.
    ///
    /// The one question a double-click asks before the view input runs: on a
    /// feature with a point behind it the gesture is Edit on Bench, and
    /// anywhere else it is the zoom. The hit radius is the click's own, so a
    /// double-click stages the point a single click there would have selected.
    pub(super) fn point_under_pointer(
        &self,
        ui: &egui::Ui,
        image_rect: egui::Rect,
        effective_scale: f32,
    ) -> Option<usize> {
        let overlay = self.feature_overlay.as_ref()?;
        let pos = ui.input(|i| i.pointer.interact_pos())?;
        let pixel = [
            (pos.x - image_rect.min.x) / effective_scale,
            (pos.y - image_rect.min.y) / effective_scale,
        ];
        find_nearest_tracked_feature(
            &overlay.features,
            &overlay.tree,
            &pixel,
            8.0 / effective_scale,
        )
    }

    /// Draw feature overlays for the current image, run click hit-testing and
    /// hover reporting, and render the hover tooltip. Populates
    /// `response.select_point` / `response.hovered_point`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_overlays(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        interact_response: &egui::Response,
        edited: &EditedReconstruction,
        feature_display: &FeatureDisplaySettings,
        selected_point: Option<usize>,
        hovered_point: Option<usize>,
        bench: BenchMenu<'_>,
        image_rect: egui::Rect,
        panel_rect: egui::Rect,
        effective_scale: f32,
        intrinsics_readout: Option<&str>,
        response: &mut ImageDetailResponse,
    ) {
        let Some(ref overlay) = self.feature_overlay else {
            return;
        };
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
                            let center = image_to_panel(feature.position[0], feature.position[1]);
                            draw_feature_ellipse(
                                painter,
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
                            painter,
                            center,
                            &feature.affine_shape,
                            effective_scale,
                            egui::Color32::YELLOW,
                            2.0,
                        );
                        painter.circle_filled(center, 4.0, egui::Color32::YELLOW);
                    } else if feature.is_tracked() {
                        draw_feature_ellipse(
                            painter,
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
                            painter,
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
                let (vmin, vmax) = compute_error_range(features, edited);
                // The one arm with no `None`: a point missing from the cloud
                // colours as zero and a non-finite error saturates the ramp,
                // rather than either dropping out of the picture.
                draw_value_overlay(
                    painter,
                    panel_rect,
                    features,
                    selected_point,
                    image_to_panel,
                    |feature| {
                        Some(
                            edited
                                .point(feature.point_index)
                                .map(|v| {
                                    let e = v.point().error;
                                    if e.is_finite() {
                                        e
                                    } else {
                                        vmax
                                    }
                                })
                                .unwrap_or(0.0),
                        )
                    },
                    (vmin, vmax),
                    &colormap::ERROR_COLORMAP,
                    "Reproj Error (px)",
                );
            }
            OverlayMode::TrackLength => {
                let (vmin, vmax) = compute_track_length_range(features, edited);
                draw_value_overlay(
                    painter,
                    panel_rect,
                    features,
                    selected_point,
                    image_to_panel,
                    |feature| {
                        Some(
                            edited
                                .point(feature.point_index)
                                .map_or(1, |v| v.observations().len())
                                as f32,
                        )
                    },
                    (vmin, vmax),
                    &colormap::QUALITY_COLORMAP,
                    "Track Length",
                );
            }
            OverlayMode::MaxTrackAngle => {
                let range = compute_finite_value_range(features, |f| f.max_track_angle_deg);
                draw_value_overlay(
                    painter,
                    panel_rect,
                    features,
                    selected_point,
                    image_to_panel,
                    |feature| finite(feature.max_track_angle_deg),
                    range,
                    &colormap::QUALITY_COLORMAP,
                    "Max Track Angle (°)",
                );
            }
            OverlayMode::DepthReliability => {
                let range = compute_finite_value_range(features, |f| f.inverse_depth_z);
                draw_value_overlay(
                    painter,
                    panel_rect,
                    features,
                    selected_point,
                    image_to_panel,
                    |feature| finite(feature.inverse_depth_z),
                    range,
                    &colormap::QUALITY_COLORMAP,
                    "Inverse-depth z",
                );
            }
            OverlayMode::ConditionNumber => {
                // Condition numbers span orders of magnitude — color in log10.
                let range =
                    compute_finite_value_range(features, |f| log10_condition(f.condition_number));
                draw_value_overlay(
                    painter,
                    panel_rect,
                    features,
                    selected_point,
                    image_to_panel,
                    |feature| finite(log10_condition(feature.condition_number)),
                    range,
                    &colormap::ERROR_COLORMAP,
                    "log10(Condition #)",
                );
            }
        }

        // ── The context menu ──
        //
        // Opened by a secondary *click* rather than by the raw button state the
        // pan/zoom handler reads: a right **drag** is this panel's zoom
        // (`input.rs`), and egui's own drag threshold is what tells the two
        // apart, so a zoom gesture never puts a menu up.
        //
        // The pixel is recorded on the frame the menu opens, because the
        // entries below are laid out on later frames, by which time the pointer
        // has moved off the place the user named.
        if interact_response.clicked_by(egui::PointerButton::Secondary) {
            if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                response.context_menu_pixel = Some(panel_to_image(pos));
            }
        }
        crate::context_menu::on_secondary_click(interact_response).show(|ui| {
            // ── Edit on Bench, first ──
            //
            // The point is the one the feature at the place the menu was opened
            // at observes, hit-tested through the same rule a left click
            // selects by, so the entry and the click cannot disagree about
            // which point is under the pointer. Refused rather than hidden
            // where there is none, so a reader can see the gesture exists and
            // read why it cannot run here.
            let pixel = response.context_menu_pixel.or(self.menu_pixel);
            let hit = pixel.and_then(|pixel| {
                feature_at(features, feature_tree, &pixel, 8.0 / effective_scale)
            });
            let point = edit_on_bench_entry(bench, hit);
            let button = egui::Button::new(crate::viewer_3d::EDIT_ON_BENCH_LABEL);
            let clicked = match &point {
                Ok(_) => ui.add(button).clicked(),
                Err(why) => {
                    ui.add_enabled(false, button).on_disabled_hover_text(why);
                    false
                }
            };
            if clicked {
                response.edit_on_bench = point.ok();
                ui.close();
            }

            // ── The bench's two entries ──
            //
            // Nothing either of them does reaches the reconstruction: each is a
            // step on the node's bench (`crate::bench`), and the commit in
            // Track View is what crosses back. They are offered whatever
            // backs an observation, because a bench track is seeds in one
            // image's pixels until it is committed.
            //
            // The pixel is the one the menu was opened at, carried out in the
            // response rather than read back off the app state, so a bench
            // gesture is the click that made it.
            let button = egui::Button::new(START_CLUSTER_LABEL);
            let clicked = match start_cluster_entry(bench) {
                Ok(()) => ui.add(button).clicked(),
                Err(why) => {
                    ui.add_enabled(false, button).on_disabled_hover_text(why);
                    false
                }
            };
            if clicked {
                response.start_bench_cluster = pixel;
                ui.close();
            }
            let button = egui::Button::new(ADD_BENCH_OBSERVATION_LABEL);
            let clicked = match add_bench_observation_entry(bench) {
                Ok(()) => ui.add(button).clicked(),
                Err(why) => {
                    ui.add_enabled(false, button).on_disabled_hover_text(why);
                    false
                }
            };
            if clicked {
                response.add_bench_observation = pixel;
                ui.close();
            }
        });

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
                            painter,
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

        // Tooltip on hover. One tooltip, composed: the feature layer's text if
        // the pointer is on a feature, the intrinsics layer's readout below a
        // separator if the layer is on, and either alone otherwise. Two
        // tooltips fighting for the cursor would be worse than either.
        if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
            if panel_rect.contains(pointer_pos) {
                let hit_radius_px = 8.0 / effective_scale;
                let hit = find_nearest_tracked_feature(
                    features,
                    feature_tree,
                    &panel_to_image(pointer_pos),
                    hit_radius_px,
                );
                let feature_text = hit.map(|point_idx| {
                    // Report hover for cross-panel feedback.
                    response.hovered_point = Some(point_idx);

                    if let Some(view) = edited.point(point_idx as u32) {
                        let pt = view.point();
                        let obs_count = view.observations().len() as u32;
                        let feat = features
                            .iter()
                            .find(|f| f.point_index as usize == point_idx);
                        let max_track_angle =
                            feat.map(|f| f.max_track_angle_deg).unwrap_or(f32::NAN);
                        let inverse_depth_z = feat.map(|f| f.inverse_depth_z).unwrap_or(f32::NAN);
                        let condition_number = feat.map(|f| f.condition_number).unwrap_or(f32::NAN);
                        let mut text = format!(
                            "Point3D #{point_idx} | err: {:.3}px | tracklen: {obs_count}",
                            pt.error
                        );
                        if max_track_angle.is_finite() {
                            text.push_str(&format!(" | max angle: {max_track_angle:.2}°"));
                        }
                        if inverse_depth_z.is_finite() {
                            text.push_str(&format!(" | depth z: {inverse_depth_z:.1}"));
                        }
                        if condition_number.is_finite() {
                            text.push_str(&format!(" | cond: {condition_number:.0}"));
                        }
                        text
                    } else {
                        format!("Point3D #{point_idx}")
                    }
                });
                draw_tooltip(
                    painter,
                    pointer_pos,
                    panel_rect,
                    feature_text.as_deref(),
                    intrinsics_readout,
                );
            }
        }
    }
}

/// Paint the composed hover tooltip: the feature layer's line, the intrinsics
/// layer's readout, or both with a rule between them.
///
/// With `readout` `None` this is byte for byte the tooltip the panel has always
/// drawn — same padding, same offset, same clamping — which is the regression a
/// composed tooltip most plausibly breaks.
pub(super) fn draw_tooltip(
    painter: &egui::Painter,
    pointer_pos: egui::Pos2,
    panel_rect: egui::Rect,
    feature_text: Option<&str>,
    readout: Option<&str>,
) {
    /// Vertical space the separating rule takes between the two blocks.
    const RULE_HEIGHT: f32 = 7.0;

    let font = egui::FontId::proportional(12.0);
    let layout =
        |text: &str| painter.layout_no_wrap(text.to_owned(), font.clone(), egui::Color32::WHITE);
    let (top, bottom) = match (feature_text, readout) {
        (Some(feature), readout) => (Some(layout(feature)), readout.map(layout)),
        (None, Some(readout)) => (Some(layout(readout)), None),
        (None, None) => return,
    };
    let Some(top) = top else {
        return;
    };

    let gap = bottom.as_ref().map_or(0.0, |_| RULE_HEIGHT);
    let size = egui::vec2(
        top.size()
            .x
            .max(bottom.as_ref().map_or(0.0, |g| g.size().x)),
        top.size().y + gap + bottom.as_ref().map_or(0.0, |g| g.size().y),
    );

    let padding = 3.0;
    let tooltip_size = size + egui::vec2(padding * 2.0, padding * 2.0);
    let mut tooltip_pos = pointer_pos + egui::vec2(12.0, -20.0);
    // Clamp to keep tooltip within the panel
    if tooltip_pos.x + tooltip_size.x > panel_rect.right() {
        tooltip_pos.x = panel_rect.right() - tooltip_size.x;
    }
    let text_rect = egui::Rect::from_min_size(tooltip_pos, size).expand(padding);
    painter.rect_filled(text_rect, 2.0, egui::Color32::from_black_alpha(200));

    let top_height = top.size().y;
    painter.galley(tooltip_pos, top, egui::Color32::WHITE);
    if let Some(bottom) = bottom {
        // A painted rule rather than a row of box-drawing characters: egui's
        // bundled proportional font has none, and a separator that renders as
        // replacement boxes would be worse than no separator at all.
        let y = tooltip_pos.y + top_height + RULE_HEIGHT / 2.0;
        painter.line_segment(
            [
                egui::pos2(text_rect.left() + 2.0, y),
                egui::pos2(text_rect.right() - 2.0, y),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_white_alpha(90)),
        );
        painter.galley(
            egui::pos2(tooltip_pos.x, tooltip_pos.y + top_height + RULE_HEIGHT),
            bottom,
            egui::Color32::WHITE,
        );
    }
}

/// Draw an oriented ellipse from a 2×2 affine shape matrix.
///
/// The affine matrix A maps the unit circle to the ellipse: p = A @ [cos(t), sin(t)]^T,
/// and that is exactly what is drawn. A decomposition into axis lengths and one
/// rotation angle is not: the singular values give the axes' lengths, but the
/// major axis lies along the left singular vector, which for a sheared or
/// anisotropic shape (a patch frame projected into an oblique view) is not the
/// first column's direction. Only a similarity, a detected SIFT shape, has the
/// two agree.
fn draw_feature_ellipse(
    painter: &egui::Painter,
    center: egui::Pos2,
    affine: &[[f32; 2]; 2],
    scale: f32,
    color: egui::Color32,
    thickness: f32,
) {
    let Some(points) = ellipse_points(center, affine, scale) else {
        return;
    };
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(thickness, color),
    ));
}

/// The closed polyline of `affine` applied to the unit circle, at `center` and
/// scaled to panel pixels, or `None` for a shape too thin to draw: the smaller
/// singular value under a tenth of a source pixel.
pub(crate) fn ellipse_points(
    center: egui::Pos2,
    affine: &[[f32; 2]; 2],
    scale: f32,
) -> Option<Vec<egui::Pos2>> {
    let [[a11, a12], [a21, a22]] = *affine;
    // The singular values, for the degeneracy test only.
    let ata00 = a11 * a11 + a21 * a21;
    let ata01 = a11 * a12 + a21 * a22;
    let ata11 = a12 * a12 + a22 * a22;
    let trace = ata00 + ata11;
    let det = ata00 * ata11 - ata01 * ata01;
    let disc = ((trace * trace / 4.0 - det).max(0.0)).sqrt();
    let s1 = ((trace / 2.0 + disc).max(0.0)).sqrt();
    let s2 = ((trace / 2.0 - disc).max(0.0)).sqrt();
    if s1 < 0.1 || s2 < 0.1 {
        return None;
    }

    let n = 32;
    Some(
        (0..=n)
            .map(|i| {
                let t = (i as f32) * std::f32::consts::TAU / (n as f32);
                let (c, s) = (t.cos(), t.sin());
                let ex = a11 * c + a12 * s;
                let ey = a21 * c + a22 * s;
                egui::pos2(center.x + ex * scale, center.y + ey * scale)
            })
            .collect(),
    )
}

/// Find the feature nearest a position in image pixel coordinates, and say
/// whether it observes a point.
///
/// Uses a KD-tree for O(log n) lookup instead of a linear scan. `hit_radius_px`
/// is the maximum distance in image pixels. A tracked feature inside the radius
/// wins over an untracked one nearer the query, which is what makes the answer
/// a superset of [`find_nearest_tracked_feature`]'s: the two cannot disagree
/// about which point is under the pointer, so the context menu offers what a
/// click there would select.
pub(super) fn feature_at(
    features: &[DisplayFeature],
    tree: &PointCloud2<f32>,
    query_px: &[f32; 2],
    hit_radius_px: f32,
) -> Option<FeatureHit> {
    if features.is_empty() {
        return None;
    }
    // Check a few nearest neighbours in case the closest is untracked. The
    // result is ordered by distance and padded with `u32::MAX` once the
    // radius runs out of candidates.
    const CANDIDATES: usize = 5;
    let neighbors = tree.nearest_k_within_radius(query_px, 1, CANDIDATES, hit_radius_px);
    let mut any = false;
    for &index in &neighbors {
        if index == u32::MAX {
            break;
        }
        any = true;
        let feature = &features[index as usize];
        if feature.is_tracked() {
            return Some(FeatureHit::Point(feature.point_index as usize));
        }
    }
    any.then_some(FeatureHit::Unmatched)
}

/// [`feature_at`] narrowed to the point a feature observes: what a click
/// selects and what the hover tooltip describes.
fn find_nearest_tracked_feature(
    features: &[DisplayFeature],
    tree: &PointCloud2<f32>,
    query_px: &[f32; 2],
    hit_radius_px: f32,
) -> Option<usize> {
    match feature_at(features, tree, query_px, hit_radius_px) {
        Some(FeatureHit::Point(point)) => Some(point),
        Some(FeatureHit::Unmatched) | None => None,
    }
}

/// Radius of a value-coloured feature dot, and the gap the selection ring
/// leaves outside it.
const VALUE_DOT_RADIUS: f32 = 5.0;

/// `Some(value)` for a number worth colouring, `None` for one the ramp has
/// nothing to say about.
fn finite(value: f32) -> Option<f32> {
    value.is_finite().then_some(value)
}

/// Draw one value-driven overlay: a coloured dot per tracked feature the
/// extractor has a number for, a ring around the selected one, and the
/// colorbar that says what the colours mean.
///
/// This is the body all five heatmap modes share. What a mode actually is —
/// which number it reads off a feature, over what range, against which of the
/// two ramps, under what title — is exactly the four arguments after
/// `to_panel`; everything else about drawing a heatmap is here, once. The
/// alternative is five copies of a loop whose only interesting line is the
/// one that reads the value, which is what this replaced.
///
/// `value` returning `None` drops the feature entirely rather than colouring
/// it: a NaN would clamp to one end of the ramp and read as a real
/// measurement at that extreme.
#[allow(clippy::too_many_arguments)]
fn draw_value_overlay(
    painter: &egui::Painter,
    panel_rect: egui::Rect,
    features: &[DisplayFeature],
    selected_point: Option<usize>,
    to_panel: impl Fn(f32, f32) -> egui::Pos2,
    value: impl Fn(&DisplayFeature) -> Option<f32>,
    (vmin, vmax): (f32, f32),
    map: &colormap::Colormap,
    label: &str,
) {
    for feature in features {
        if !feature.is_tracked() {
            continue;
        }
        let Some(value) = value(feature) else {
            continue;
        };
        let center = to_panel(feature.position[0], feature.position[1]);
        if !panel_rect.expand(10.0).contains(center) {
            continue;
        }
        painter.circle_filled(
            center,
            VALUE_DOT_RADIUS,
            colormap::ramp(value, vmin, vmax, map),
        );
        if selected_point == Some(feature.point_index as usize) {
            painter.circle_stroke(
                center,
                VALUE_DOT_RADIUS + 2.0,
                egui::Stroke::new(2.0_f32, egui::Color32::YELLOW),
            );
        }
    }
    colormap::draw_colorbar(painter, panel_rect, label, vmin, vmax, |v, lo, hi| {
        colormap::ramp(v, lo, hi, map)
    });
}

/// Compute the reprojection error range for tracked features in the display list.
fn compute_error_range(features: &[DisplayFeature], edited: &EditedReconstruction) -> (f32, f32) {
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for feature in features {
        if !feature.is_tracked() {
            continue;
        }
        if let Some(view) = edited.point(feature.point_index) {
            let error = view.point().error;
            if error.is_finite() {
                vmin = vmin.min(error);
                vmax = vmax.max(error);
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

/// `log10` of a condition number, guarding the degenerate `∞` (and clamping at
/// 1 so the result is non-negative). Non-finite input maps to NaN (skipped).
fn log10_condition(condition_number: f32) -> f32 {
    if condition_number.is_finite() {
        condition_number.max(1.0).log10()
    } else {
        f32::NAN
    }
}

/// Compute the value range across tracked features for an arbitrary per-feature
/// accessor, ignoring non-finite values. Falls back to a unit range when there
/// is no finite data, and pads a degenerate (zero-width) range.
fn compute_finite_value_range(
    features: &[DisplayFeature],
    value: impl Fn(&DisplayFeature) -> f32,
) -> (f32, f32) {
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for feature in features {
        if !feature.is_tracked() {
            continue;
        }
        let v = value(feature);
        if v.is_finite() {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
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
    edited: &EditedReconstruction,
) -> (f32, f32) {
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for feature in features {
        if !feature.is_tracked() {
            continue;
        }
        let count = edited
            .point(feature.point_index)
            .map_or(1, |v| v.observations().len()) as f32;
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
