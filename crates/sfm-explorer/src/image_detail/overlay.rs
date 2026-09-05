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
                let (vmin, vmax) = compute_error_range(features, recon);
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
                            recon
                                .points
                                .get(feature.point_index as usize)
                                .map(|p| if p.error.is_finite() { p.error } else { vmax })
                                .unwrap_or(0.0),
                        )
                    },
                    (vmin, vmax),
                    &colormap::ERROR_COLORMAP,
                    "Reproj Error (px)",
                );
            }
            OverlayMode::TrackLength => {
                let (vmin, vmax) = compute_track_length_range(features, recon);
                draw_value_overlay(
                    painter,
                    panel_rect,
                    features,
                    selected_point,
                    image_to_panel,
                    |feature| {
                        Some(
                            recon
                                .observation_counts
                                .get(feature.point_index as usize)
                                .copied()
                                .unwrap_or(1) as f32,
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

                    if let Some(pt) = recon.points.get(point_idx) {
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
