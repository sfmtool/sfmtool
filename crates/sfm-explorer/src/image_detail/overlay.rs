// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Feature-overlay rendering for the image detail panel: the per-`OverlayMode`
//! draw branches, feature click hit-testing, cross-panel hover highlight, and
//! the point tooltip, plus the geometry/color helpers they use.

use super::{DisplayFeature, ImageDetail, ImageDetailResponse};
use crate::colormap;
use crate::state::{FeatureDisplaySettings, OverlayMode};
use kiddo::SquaredEuclidean;
use sfmtool_core::SfmrReconstruction;

impl ImageDetail {
    /// Draw feature overlays for the current image, run click hit-testing and
    /// hover reporting, and render the hover tooltip. Populates
    /// `response.select_point` / `response.hovered_point`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_overlays(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        interact_response: &egui::Response,
        recon: &SfmrReconstruction,
        feature_display: &FeatureDisplaySettings,
        selected_point: Option<usize>,
        hovered_point: Option<usize>,
        image_rect: egui::Rect,
        panel_rect: egui::Rect,
        effective_scale: f32,
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
                    painter,
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
                    painter,
                    panel_rect,
                    "Track Length",
                    vmin,
                    vmax,
                    colormap::track_length_color,
                );
            }
            OverlayMode::MaxTrackAngle => {
                let (vmin, vmax) = compute_finite_value_range(features, |f| f.max_track_angle_deg);
                for feature in features {
                    if !feature.is_tracked() || !feature.max_track_angle_deg.is_finite() {
                        continue;
                    }
                    let center = image_to_panel(feature.position[0], feature.position[1]);
                    if !panel_rect.expand(10.0).contains(center) {
                        continue;
                    }
                    let is_selected = selected_point == Some(feature.point_index as usize);
                    let color =
                        colormap::max_track_angle_color(feature.max_track_angle_deg, vmin, vmax);
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
                colormap::draw_colorbar(
                    painter,
                    panel_rect,
                    "Max Track Angle (°)",
                    vmin,
                    vmax,
                    colormap::max_track_angle_color,
                );
            }
            OverlayMode::DepthReliability => {
                let (vmin, vmax) = compute_finite_value_range(features, |f| f.inverse_depth_z);
                for feature in features {
                    if !feature.is_tracked() || !feature.inverse_depth_z.is_finite() {
                        continue;
                    }
                    let center = image_to_panel(feature.position[0], feature.position[1]);
                    if !panel_rect.expand(10.0).contains(center) {
                        continue;
                    }
                    let is_selected = selected_point == Some(feature.point_index as usize);
                    let color =
                        colormap::depth_reliability_color(feature.inverse_depth_z, vmin, vmax);
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
                colormap::draw_colorbar(
                    painter,
                    panel_rect,
                    "Inverse-depth z",
                    vmin,
                    vmax,
                    colormap::depth_reliability_color,
                );
            }
            OverlayMode::ConditionNumber => {
                // Condition numbers span orders of magnitude — color in log10.
                let (vmin, vmax) =
                    compute_finite_value_range(features, |f| log10_condition(f.condition_number));
                for feature in features {
                    let log_cond = log10_condition(feature.condition_number);
                    if !feature.is_tracked() || !log_cond.is_finite() {
                        continue;
                    }
                    let center = image_to_panel(feature.position[0], feature.position[1]);
                    if !panel_rect.expand(10.0).contains(center) {
                        continue;
                    }
                    let is_selected = selected_point == Some(feature.point_index as usize);
                    let color = colormap::condition_number_color(log_cond, vmin, vmax);
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
                colormap::draw_colorbar(
                    painter,
                    panel_rect,
                    "log10(Condition #)",
                    vmin,
                    vmax,
                    colormap::condition_number_color,
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
                    };
                    let font = egui::FontId::proportional(12.0);
                    let galley = painter.layout_no_wrap(tooltip_text, font, egui::Color32::WHITE);
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
