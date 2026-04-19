// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Colormap utilities for heatmap visualization.
//!
//! Maps scalar values to RGB colors using piecewise-linear colormaps.
//! The `error` colormap matches Python's `visualization/_colormap.py`;
//! the `quality` colormap is a GUI-only variant so track length and
//! max track angle share the same red→yellow→green encoding.

/// A colormap defined by control points (position, r, g, b).
/// Positions are in [0, 1], colors are in [0, 255].
struct Colormap {
    stops: &'static [(f32, u8, u8, u8)],
}

impl Colormap {
    /// Map a normalized value in [0, 1] to an egui Color32.
    fn sample(&self, t: f32) -> egui::Color32 {
        let t = t.clamp(0.0, 1.0);
        let stops = self.stops;

        // Find surrounding control points
        for i in 0..stops.len() - 1 {
            let (t0, r0, g0, b0) = stops[i];
            let (t1, r1, g1, b1) = stops[i + 1];
            if t0 <= t && t <= t1 {
                let frac = if (t1 - t0).abs() < 1e-9 {
                    0.0
                } else {
                    (t - t0) / (t1 - t0)
                };
                let r = r0 as f32 + frac * (r1 as f32 - r0 as f32);
                let g = g0 as f32 + frac * (g1 as f32 - g0 as f32);
                let b = b0 as f32 + frac * (b1 as f32 - b0 as f32);
                return egui::Color32::from_rgb(r as u8, g as u8, b as u8);
            }
        }
        // Fallback: last color
        let (_, r, g, b) = stops[stops.len() - 1];
        egui::Color32::from_rgb(r, g, b)
    }
}

/// Error colormap: green (good, low error) -> yellow -> red (bad, high error).
const ERROR_COLORMAP: Colormap = Colormap {
    stops: &[(0.0, 0, 200, 0), (0.5, 255, 255, 0), (1.0, 255, 0, 0)],
};

/// Quality colormap: red (low/bad) -> yellow -> green (high/good).
/// Inverse of ERROR_COLORMAP — used for metrics where higher is better
/// (track length, max track angle). Shared so visually comparable metrics
/// use the same encoding.
const QUALITY_COLORMAP: Colormap = Colormap {
    stops: &[(0.0, 255, 0, 0), (0.5, 255, 255, 0), (1.0, 0, 200, 0)],
};

/// Map a reprojection error value to a color.
/// `vmin` and `vmax` define the range (values outside are clamped).
pub fn error_color(value: f32, vmin: f32, vmax: f32) -> egui::Color32 {
    let t = if (vmax - vmin).abs() < 1e-9 {
        0.5
    } else {
        (value - vmin) / (vmax - vmin)
    };
    ERROR_COLORMAP.sample(t)
}

/// Map a track length (observation count) to a color.
/// `vmin` and `vmax` define the range (values outside are clamped).
pub fn track_length_color(value: f32, vmin: f32, vmax: f32) -> egui::Color32 {
    let t = if (vmax - vmin).abs() < 1e-9 {
        0.5
    } else {
        (value - vmin) / (vmax - vmin)
    };
    QUALITY_COLORMAP.sample(t)
}

/// Map a max track angle (degrees) to a color.
/// `vmin` and `vmax` define the range (values outside are clamped).
pub fn max_track_angle_color(value: f32, vmin: f32, vmax: f32) -> egui::Color32 {
    let t = if (vmax - vmin).abs() < 1e-9 {
        0.5
    } else {
        (value - vmin) / (vmax - vmin)
    };
    QUALITY_COLORMAP.sample(t)
}

/// Draw a vertical colorbar legend on the painter.
///
/// Renders a gradient bar with min/max labels in the bottom-right corner
/// of the given rect.
pub fn draw_colorbar(
    painter: &egui::Painter,
    panel_rect: egui::Rect,
    label: &str,
    vmin: f32,
    vmax: f32,
    color_fn: impl Fn(f32, f32, f32) -> egui::Color32,
) {
    let bar_width = 16.0;
    let bar_height = 120.0;
    let margin = 12.0;
    let text_margin = 4.0;

    let bar_right = panel_rect.right() - margin;
    let bar_left = bar_right - bar_width;
    let bar_bottom = panel_rect.bottom() - margin - 16.0; // room for bottom label
    let bar_top = bar_bottom - bar_height;

    // Background for readability
    let bg_rect = egui::Rect::from_min_max(
        egui::pos2(bar_left - 50.0, bar_top - 20.0),
        egui::pos2(bar_right + 4.0, bar_bottom + 20.0),
    );
    painter.rect_filled(bg_rect, 4.0, egui::Color32::from_black_alpha(160));

    // Draw gradient bar as horizontal stripes
    let n_stripes = 32;
    for i in 0..n_stripes {
        let frac = (i as f32 + 0.5) / n_stripes as f32;
        // Top of bar = vmax, bottom = vmin (high values at top)
        let value = vmax - frac * (vmax - vmin);
        let color = color_fn(value, vmin, vmax);
        let stripe_rect = egui::Rect::from_min_max(
            egui::pos2(
                bar_left,
                bar_top + (i as f32 / n_stripes as f32) * bar_height,
            ),
            egui::pos2(
                bar_right,
                bar_top + ((i + 1) as f32 / n_stripes as f32) * bar_height,
            ),
        );
        painter.rect_filled(stripe_rect, 0.0, color);
    }

    // Border
    painter.rect_stroke(
        egui::Rect::from_min_max(
            egui::pos2(bar_left, bar_top),
            egui::pos2(bar_right, bar_bottom),
        ),
        0.0,
        egui::Stroke::new(1.0, egui::Color32::WHITE),
        egui::StrokeKind::Outside,
    );

    let font = egui::FontId::proportional(11.0);

    // Max label (top)
    let max_text = format!("{vmax:.2}");
    painter.text(
        egui::pos2(bar_left - text_margin, bar_top),
        egui::Align2::RIGHT_TOP,
        &max_text,
        font.clone(),
        egui::Color32::WHITE,
    );

    // Min label (bottom)
    let min_text = format!("{vmin:.2}");
    painter.text(
        egui::pos2(bar_left - text_margin, bar_bottom),
        egui::Align2::RIGHT_BOTTOM,
        &min_text,
        font.clone(),
        egui::Color32::WHITE,
    );

    // Title label (above bar)
    painter.text(
        egui::pos2((bar_left + bar_right) / 2.0, bar_top - 4.0),
        egui::Align2::CENTER_BOTTOM,
        label,
        font,
        egui::Color32::WHITE,
    );
}
