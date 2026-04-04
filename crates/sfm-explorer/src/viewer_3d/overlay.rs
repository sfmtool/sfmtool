// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Overlay drawing for the 3D viewer.
//!
//! Ground plane grid, axis indicator, and info text overlays
//! drawn via egui's painter.

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use nalgebra::{Point3, Vector3};
use sfmtool_core::SfmrReconstruction;

use crate::scene_renderer::{PICK_INDEX_MASK, PICK_TAG_FRUSTUM, PICK_TAG_MASK, PICK_TAG_POINT};

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
                painter.line_segment([s1, s2], Stroke::new(1.0, grid_color));
            }
            y += grid_step;
        }

        // Draw grid lines parallel to Y axis
        let mut x = -grid_extent;
        while x <= grid_extent {
            let p1 = Point3::new(x, -grid_extent, 0.0);
            let p2 = Point3::new(x, grid_extent, 0.0);
            if let Some((s1, s2)) = self.camera.project_line_clipped(&p1, &p2, rect) {
                painter.line_segment([s1, s2], Stroke::new(1.0, grid_color));
            }
            x += grid_step;
        }

        // Draw origin axes (X=red, Y=green, Z=blue), scaled to grid
        let origin = Point3::origin();
        let x_end = Point3::new(axis_length, 0.0, 0.0);
        let y_end = Point3::new(0.0, axis_length, 0.0);
        let z_end = Point3::new(0.0, 0.0, axis_length);

        if let Some((s1, s2)) = self.camera.project_line_clipped(&origin, &x_end, rect) {
            painter.line_segment([s1, s2], Stroke::new(2.0, Color32::RED));
        }
        if let Some((s1, s2)) = self.camera.project_line_clipped(&origin, &y_end, rect) {
            painter.line_segment([s1, s2], Stroke::new(2.0, Color32::GREEN));
        }
        if let Some((s1, s2)) = self.camera.project_line_clipped(&origin, &z_end, rect) {
            painter.line_segment([s1, s2], Stroke::new(2.0, Color32::from_rgb(80, 80, 255)));
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
        painter.line_segment([center, x_end], Stroke::new(2.0, Color32::RED));
        painter.line_segment([center, y_end], Stroke::new(2.0, Color32::GREEN));
        painter.line_segment(
            [center, z_end],
            Stroke::new(2.0, Color32::from_rgb(80, 80, 255)),
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_info_overlay(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        reconstruction: &SfmrReconstruction,
        diagnostics: Option<(u32, u32, u32, u32)>,
        handler_ok: bool,
        hover_depth: Option<f32>,
        hover_pick_id: u32,
        fps: f64,
    ) {
        let font = egui::FontId::proportional(12.0);
        let text_color = Color32::from_rgba_unmultiplied(200, 200, 200, 180);

        // Top-left: stats + FPS
        let stats = format!(
            "{} points | {} images | {:.0} fps",
            reconstruction.points.len(),
            reconstruction.images.len(),
            fps,
        );
        painter.text(
            Pos2::new(rect.left() + 10.0, rect.top() + 10.0),
            egui::Align2::LEFT_TOP,
            stats,
            font.clone(),
            text_color,
        );

        // Top-right: diagnostics
        let diag_text = if let Some((hits, contacts, updates, global)) = diagnostics {
            format!(
                "H={}|C={}|U={}|G={} [{}]",
                hits,
                contacts,
                updates,
                global,
                if handler_ok { "OK" } else { "FAIL" }
            )
        } else {
            format!("[{}]", if handler_ok { "OK" } else { "FAIL" })
        };

        painter.text(
            Pos2::new(rect.right() - 10.0, rect.top() + 10.0),
            egui::Align2::RIGHT_TOP,
            diag_text,
            font.clone(),
            text_color,
        );

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

        // Bottom-left: entity + depth info under cursor
        let tag = hover_pick_id & PICK_TAG_MASK;
        let index = (hover_pick_id & PICK_INDEX_MASK) as usize;
        let depth_val = hover_depth.filter(|&d| d > 0.0);

        let hover_text = match tag {
            t if t == PICK_TAG_POINT => {
                if let Some(depth) = depth_val {
                    format!("Point3D #{} | depth: {:.4}", index, depth)
                } else {
                    format!("Point3D #{}", index)
                }
            }
            t if t == PICK_TAG_FRUSTUM => {
                let name = reconstruction
                    .images
                    .get(index)
                    .map(|img| img.name.as_str())
                    .unwrap_or("?");
                format!("Camera: {}", name)
            }
            _ => {
                if let Some(depth) = depth_val {
                    format!("depth: {:.4}", depth)
                } else {
                    String::new()
                }
            }
        };

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
        let controls =
            "Drag: orbit | Shift: pan | Scroll: zoom | Alt+drag: free-look | WASD: fly | Alt: target";
        painter.text(
            Pos2::new(rect.right() - 10.0, rect.bottom() - 10.0),
            egui::Align2::RIGHT_BOTTOM,
            controls,
            font,
            text_color,
        );
    }
}