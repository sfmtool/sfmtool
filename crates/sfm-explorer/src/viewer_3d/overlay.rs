// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Overlay drawing for the 3D viewer.
//!
//! Ground plane grid, axis indicator, and info text overlays
//! drawn via egui's painter.

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use nalgebra::{Point3, Vector3};

use crate::scene::{node_by_id, visible_stats, SceneNode};
use crate::scene_renderer::PickTarget;

use super::Viewer3D;

impl Viewer3D {
    /// Draws a ground plane grid on the XY plane (Z=0).
    ///
    /// Grid step and extent adapt to `length_scale` so the grid is meaningful
    /// at any scene scale. Step snaps to the nearest power of 10 of
    /// `length_scale * 5`, giving clean round numbers (0.01, 0.1, 1, 10, ...).
    pub(super) fn draw_grid(&self, painter: &egui::Painter, rect: Rect, length_scale: f32) {
        let grid_color = Color32::from_rgba_unmultiplied(100, 100, 100, 100);

        // Snap grid step to nearest power of 10
        let raw_step = (length_scale * 5.0) as f64;
        let grid_step = if raw_step > 0.0 {
            10.0_f64.powf(raw_step.log10().round())
        } else {
            1.0
        };
        let grid_extent = grid_step * 10.0;
        let axis_length = grid_step * 2.0;

        // Draw grid lines parallel to X axis
        let mut y = -grid_extent;
        while y <= grid_extent {
            let p1 = Point3::new(-grid_extent, y, 0.0);
            let p2 = Point3::new(grid_extent, y, 0.0);
            if let Some((s1, s2)) = self.camera.project_line_clipped(&p1, &p2, rect) {
                painter.line_segment([s1, s2], Stroke::new(1.0_f32, grid_color));
            }
            y += grid_step;
        }

        // Draw grid lines parallel to Y axis
        let mut x = -grid_extent;
        while x <= grid_extent {
            let p1 = Point3::new(x, -grid_extent, 0.0);
            let p2 = Point3::new(x, grid_extent, 0.0);
            if let Some((s1, s2)) = self.camera.project_line_clipped(&p1, &p2, rect) {
                painter.line_segment([s1, s2], Stroke::new(1.0_f32, grid_color));
            }
            x += grid_step;
        }

        // Draw origin axes (X=red, Y=green, Z=blue), scaled to grid
        let origin = Point3::origin();
        let x_end = Point3::new(axis_length, 0.0, 0.0);
        let y_end = Point3::new(0.0, axis_length, 0.0);
        let z_end = Point3::new(0.0, 0.0, axis_length);

        if let Some((s1, s2)) = self.camera.project_line_clipped(&origin, &x_end, rect) {
            painter.line_segment([s1, s2], Stroke::new(2.0_f32, Color32::RED));
        }
        if let Some((s1, s2)) = self.camera.project_line_clipped(&origin, &y_end, rect) {
            painter.line_segment([s1, s2], Stroke::new(2.0_f32, Color32::GREEN));
        }
        if let Some((s1, s2)) = self.camera.project_line_clipped(&origin, &z_end, rect) {
            painter.line_segment(
                [s1, s2],
                Stroke::new(2.0_f32, Color32::from_rgb(80, 80, 255)),
            );
        }
    }

    /// Draws a small axis indicator in the corner showing current orientation.
    pub(super) fn draw_axis_indicator(&self, painter: &egui::Painter, rect: Rect) {
        let indicator_size = 50.0;
        let center = Pos2::new(rect.left() + 40.0, rect.bottom() - 40.0);

        // Get camera's view direction to rotate the indicator
        let view = self.camera.view_matrix();

        // Transform unit axes by the view rotation (just the rotation part)
        let transform_axis = |axis: Vector3<f64>| -> Pos2 {
            let transformed = Vector3::new(
                view[(0, 0)] * axis.x + view[(0, 1)] * axis.y + view[(0, 2)] * axis.z,
                view[(1, 0)] * axis.x + view[(1, 1)] * axis.y + view[(1, 2)] * axis.z,
                view[(2, 0)] * axis.x + view[(2, 1)] * axis.y + view[(2, 2)] * axis.z,
            );
            Pos2::new(
                center.x + (transformed.x * indicator_size * 0.5) as f32,
                center.y - (transformed.y * indicator_size * 0.5) as f32,
            )
        };

        let x_end = transform_axis(Vector3::x());
        let y_end = transform_axis(Vector3::y());
        let z_end = transform_axis(Vector3::z());

        // Draw axes
        painter.line_segment([center, x_end], Stroke::new(2.0_f32, Color32::RED));
        painter.line_segment([center, y_end], Stroke::new(2.0_f32, Color32::GREEN));
        painter.line_segment(
            [center, z_end],
            Stroke::new(2.0_f32, Color32::from_rgb(80, 80, 255)),
        );

        // Draw labels
        let font = egui::FontId::proportional(10.0);
        painter.text(
            x_end,
            egui::Align2::CENTER_CENTER,
            "X",
            font.clone(),
            Color32::RED,
        );
        painter.text(
            y_end,
            egui::Align2::CENTER_CENTER,
            "Y",
            font.clone(),
            Color32::GREEN,
        );
        painter.text(
            z_end,
            egui::Align2::CENTER_CENTER,
            "Z",
            font,
            Color32::from_rgb(80, 80, 255),
        );
    }

    /// Draws an info overlay with controls and stats.
    ///
    /// The top-right corner is deliberately left empty: it belongs to the
    /// viewport HUD (`hud.rs`), and the touchpad diagnostics that used to be
    /// burned in here now live in its Debug section.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_info_overlay(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        scene: &[SceneNode],
        solo: Option<crate::scene::ReconId>,
        show_controls_help: bool,
        show_fps: bool,
        status_message: Option<&str>,
        hover_depth: Option<f32>,
        hover_pick: Option<PickTarget>,
        fps: f64,
    ) {
        let font = egui::FontId::proportional(12.0);
        let text_color = Color32::from_rgba_unmultiplied(200, 200, 200, 180);

        // Top-left: stats summed over the *visible* nodes, and the frame rate
        // if it is wanted. Points at infinity are called out separately — they
        // are directions rather than locations, so a lone total hides how much
        // of the cloud has no position at all.
        let stats = scene_stats_text(scene, solo, show_fps, fps);
        painter.text(
            Pos2::new(rect.left() + 10.0, rect.top() + 10.0),
            egui::Align2::LEFT_TOP,
            stats,
            font.clone(),
            text_color,
        );

        // Directly under the stats: the status message. This is where an
        // `Align to…` outcome lands — the empty-state text in `dock.rs` only
        // shows it when *nothing* is loaded, and an alignment by definition
        // happens with two files open.
        if let Some(status) = status_message {
            painter.text(
                Pos2::new(rect.left() + 10.0, rect.top() + 28.0),
                egui::Align2::LEFT_TOP,
                status,
                font.clone(),
                Color32::from_rgb(235, 215, 150),
            );
        }

        // Top-middle: camera info
        let cam_info = format!(
            "Pos: [{:.1}, {:.1}, {:.1}]",
            self.camera.position().x,
            self.camera.position().y,
            self.camera.position().z
        );
        painter.text(
            Pos2::new(rect.center().x, rect.top() + 10.0),
            egui::Align2::CENTER_TOP,
            cam_info,
            font.clone(),
            text_color,
        );

        // Bottom-left: entity + depth info under cursor. The pick is a ref, so
        // it names its own reconstruction; with more than one file loaded the
        // text says which — `Camera: run_a / IMG_0001.jpg`, `Point3D run_a
        // #88231` — and with one it stays exactly as it was.
        let depth_val = hover_depth.filter(|&d| d > 0.0);
        let hover_text = hover_overlay_text(scene, hover_pick, depth_val);

        if !hover_text.is_empty() {
            painter.text(
                Pos2::new(rect.left() + 10.0, rect.bottom() - 30.0),
                egui::Align2::LEFT_BOTTOM,
                hover_text,
                font.clone(),
                text_color,
            );
        }

        // Bottom-right: controls help
        if show_controls_help {
            let controls = "Drag: orbit | Shift: pan | Scroll: zoom | Alt+drag: free-look | WASD: fly | Alt: target";
            painter.text(
                Pos2::new(rect.right() - 10.0, rect.bottom() - 10.0),
                egui::Align2::RIGHT_BOTTOM,
                controls,
                font,
                text_color,
            );
        }
    }
}

/// The top-left stats line: entity totals over the nodes actually drawn, led by
/// the reconstruction count once more than one is contributing.
///
/// "Drawn" is the same rule the draw loop and the scene bounds use
/// (`scene::is_visible`): the node's eye AND the solo override. Soloing one of
/// two loaded files therefore drops the line back to that file's own totals,
/// with no leading count — which is exactly what is on screen.
pub(crate) fn scene_stats_text(
    scene: &[SceneNode],
    solo: Option<crate::scene::ReconId>,
    show_fps: bool,
    fps: f64,
) -> String {
    let totals = visible_stats(scene, solo);
    let points = if totals.points_at_infinity > 0 {
        format!(
            "{} points ({} at infinity)",
            totals.points, totals.points_at_infinity
        )
    } else {
        format!("{} points", totals.points)
    };
    let mut text = String::new();
    if totals.recons > 1 {
        text.push_str(&format!("{} reconstructions | ", totals.recons));
    }
    text.push_str(&format!("{} | {} images", points, totals.images));
    if show_fps {
        text.push_str(&format!(" | {fps:.0} fps"));
    }
    text
}

/// The bottom-left hover line for what the pick buffer resolved to.
///
/// The reconstruction label is included only when more than one file is
/// loaded — with a single one it would be noise, and it is exactly the text
/// `ui_basic`-adjacent expectations were written against.
pub(crate) fn hover_overlay_text(
    scene: &[SceneNode],
    hover_pick: Option<PickTarget>,
    depth: Option<f32>,
) -> String {
    let qualify = scene.len() > 1;
    match hover_pick {
        Some(PickTarget::Point(point)) => {
            let label = qualify
                .then(|| node_by_id(scene, point.recon))
                .flatten()
                .map(|node| format!("{} ", node.label))
                .unwrap_or_default();
            match depth {
                Some(depth) => format!("Point3D {}#{} | depth: {:.4}", label, point.index(), depth),
                None => format!("Point3D {}#{}", label, point.index()),
            }
        }
        Some(PickTarget::Image(image)) => {
            let node = node_by_id(scene, image.recon);
            let name = node
                .and_then(|node| node.recon.images.get(image.index()))
                .map(|img| img.name.as_str())
                .unwrap_or("?");
            match node.filter(|_| qualify) {
                Some(node) => format!("Camera: {} / {}", node.label, name),
                None => format!("Camera: {}", name),
            }
        }
        None => match depth {
            Some(depth) => format!("depth: {:.4}", depth),
            None => String::new(),
        },
    }
}
