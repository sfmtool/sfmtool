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
        let stats = painter.layout_no_wrap(stats, font.clone(), text_color);
        let stats_pos = Pos2::new(rect.left() + 10.0, rect.top() + STATS_TOP);
        let stats_right = stats_pos.x + stats.size().x;
        painter.galley(stats_pos, stats, text_color);

        // Directly under the stats: the status message. This is where an
        // `Align to…` outcome lands — the empty-state text in `dock.rs` only
        // shows it when *nothing* is loaded, and an alignment by definition
        // happens with two files open.
        let mut status_right = None;
        if let Some(status) = status_message {
            let status = painter.layout_no_wrap(status.to_owned(), font.clone(), BANNER_COLOUR);
            let status_pos = Pos2::new(rect.left() + 10.0, rect.top() + STATUS_TOP);
            status_right = Some(status_pos.x + status.size().x);
            painter.galley(status_pos, status, BANNER_COLOUR);
        }

        // Top-middle: camera info, on the first row where it touches neither
        // the stats nor the status message (`camera_position_top`).
        let cam_info = format!(
            "Pos: [{:.1}, {:.1}, {:.1}]",
            self.camera.position().x,
            self.camera.position().y,
            self.camera.position().z
        );
        let cam_info = painter.layout_no_wrap(cam_info, font.clone(), text_color);
        let cam_left = rect.center().x - cam_info.size().x / 2.0;
        let cam_top = camera_position_top(stats_right, status_right, cam_left);
        painter.galley(
            Pos2::new(cam_left, rect.top() + cam_top),
            cam_info,
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

        // Bottom-right: controls help, for the tool that is on, since the two
        // read the left drag differently.
        if show_controls_help {
            let controls = match self.tool {
                super::brush::Tool::Navigate => {
                    "Drag: orbit | Shift: pan | Scroll: zoom | Alt+drag: free-look | WASD: fly | Alt: target"
                }
                super::brush::Tool::EraserBrush => {
                    "Drag: delete points | Esc: cancel stroke | Alt+drag: orbit | Shift: pan | Scroll: zoom | WASD: fly"
                }
            };
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

impl Viewer3D {
    /// Draw the lock banner: what camera is in hand, what it costs, and how to
    /// end it.
    ///
    /// In the HUD's own vocabulary -- a small panel of 12 pt text on a dark
    /// ground -- and in the viewport rather than in the HUD panel, because it
    /// describes the gesture the viewport is in the middle of and disappears
    /// with it. Drawn top-centre at [`BANNER_TOP`], under the camera-position
    /// line and the status line.
    pub(super) fn draw_lock_banner(&self, painter: &egui::Painter, rect: Rect, node: &SceneNode) {
        let Some(lock) = self.camera_lock.as_ref() else {
            return;
        };
        if lock.image.recon != node.id {
            return;
        }
        let pending = crate::camera_lock::pending_pose(self, node);
        let (now, stored) = crate::camera_lock::residuals(lock, &pending);
        let basename = node
            .recon()
            .image_table
            .images
            .get(lock.image.index())
            .map(|image| crate::resect::basename(&image.name))
            .unwrap_or("?");
        let lines = crate::camera_lock::banner_lines(basename, &node.label, now, stored);

        let font = egui::FontId::proportional(12.0);
        let line_height = 16.0;
        let width = 320.0_f32;
        let height = line_height * lines.len() as f32 + 10.0;
        let top_left = Pos2::new(rect.center().x - width / 2.0, rect.top() + BANNER_TOP);
        let panel = Rect::from_min_size(top_left, egui::Vec2::new(width, height));
        painter.rect_filled(panel, 4.0, Color32::from_rgba_unmultiplied(20, 20, 24, 210));
        painter.rect_stroke(
            panel,
            4.0,
            Stroke::new(1.0, Color32::from_rgb(235, 215, 150)),
            egui::StrokeKind::Inside,
        );
        for (row, line) in lines.iter().enumerate() {
            painter.text(
                Pos2::new(
                    panel.center().x,
                    panel.top() + 5.0 + row as f32 * line_height,
                ),
                egui::Align2::CENTER_TOP,
                line,
                font.clone(),
                Color32::from_rgb(235, 215, 150),
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
/// loaded: with a single one it names the only reconstruction there is.
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
                .and_then(|node| node.recon().image_table.images.get(image.index()))
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

/// The colour the banners draw their words and frame in.
const BANNER_COLOUR: Color32 = Color32::from_rgb(235, 215, 150);

/// How far below the 3D view's top edge its stats line is drawn.
const STATS_TOP: f32 = 10.0;

/// How far below the 3D view's top edge its status line is drawn, under the
/// stats line.
pub(crate) const STATUS_TOP: f32 = 28.0;

/// How far below the 3D view's top edge the camera position is drawn when it
/// would touch the text on both rows above it: a row of its own, under the
/// status line.
pub(crate) const POSITION_THIRD_ROW_TOP: f32 = 46.0;

/// How far below the 3D view's top edge the banners start: under the stats
/// line at [`STATS_TOP`], the status line at [`STATUS_TOP`] and the camera
/// position's own row at [`POSITION_THIRD_ROW_TOP`], each a 12 pt line, so a
/// banner never covers the status message or the camera position.
pub(crate) const BANNER_TOP: f32 = 64.0;

/// The least horizontal space kept between the centred camera position and
/// the text to its left on the same row.
const POSITION_GAP: f32 = 12.0;

/// Which row of the 3D view's top text the camera position goes on, as its
/// distance below the view's top edge.
///
/// The position is centred, with its left edge at `position_left`. It stays on
/// the stats line's row while [`POSITION_GAP`] still separates it from the
/// stats, whose right edge is at `stats_right`. In a narrower view it drops to
/// the status line's row when no status message is shown or the message, whose
/// right edge is at `status_right`, also leaves that gap; otherwise it takes a
/// row of its own under the status line. All three are x coordinates on the
/// screen.
fn camera_position_top(stats_right: f32, status_right: Option<f32>, position_left: f32) -> f32 {
    let clears = |right: f32| right + POSITION_GAP <= position_left;
    if clears(stats_right) {
        STATS_TOP
    } else if status_right.is_none_or(clears) {
        STATUS_TOP
    } else {
        POSITION_THIRD_ROW_TOP
    }
}

/// `text` laid out for a banner in 12 pt words of `colour`, wrapping at
/// `wrap`: in the proportional font, with each run of characters it has no
/// glyph for (the `→` of a model change) in the monospace font, which has the
/// arrow, so it is not drawn as an empty box. A character neither font has is
/// still drawn as one, and the banner's glyph test fails on one its sample
/// text holds.
pub(crate) fn banner_job(
    ctx: &egui::Context,
    text: &str,
    colour: Color32,
    wrap: f32,
) -> egui::text::LayoutJob {
    let proportional = egui::FontId::proportional(12.0);
    let monospace = egui::FontId::monospace(12.0);
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = wrap;
    ctx.fonts_mut(|fonts| {
        let mut run = String::new();
        let mut run_font: Option<egui::FontId> = None;
        for c in text.chars() {
            let font = if c.is_whitespace() || fonts.has_glyph(&proportional, c) {
                proportional.clone()
            } else {
                monospace.clone()
            };
            if run_font.as_ref().is_some_and(|f| *f != font) {
                let f = run_font.take().expect("just checked");
                job.append(&std::mem::take(&mut run), 0.0, text_format(f, colour));
            }
            run_font = Some(font);
            run.push(c);
        }
        if let Some(f) = run_font {
            job.append(&run, 0.0, text_format(f, colour));
        }
    });
    job
}

fn text_format(font_id: egui::FontId, color: Color32) -> egui::TextFormat {
    egui::TextFormat {
        font_id,
        color,
        ..Default::default()
    }
}

/// Draw the camera model proposal's banner, top-centre in the 3D view, in the
/// Move Camera lock banner's style: 12 pt words on a dark ground in a framed
/// panel.
///
/// Each line wraps at the narrower of the view less a margin and 640 px, so
/// at any width the words stay inside the view and no line runs into the
/// next. A line about points that lose their position is drawn in the
/// failure colour. Returns the panel's rectangle, `None` with no lines.
pub(crate) fn draw_proposal_banner(
    painter: &egui::Painter,
    rect: Rect,
    lines: &[crate::camera_model_proposal::banner::BannerLine],
) -> Option<Rect> {
    if lines.is_empty() {
        return None;
    }
    const PAD: f32 = 6.0;
    const GAP: f32 = 2.0;
    let wrap = ((rect.width() - 16.0).min(640.0) - 2.0 * PAD).max(40.0);
    let failure = Color32::from_rgb(
        crate::scene::FAILURE_RGB[0],
        crate::scene::FAILURE_RGB[1],
        crate::scene::FAILURE_RGB[2],
    );
    let galleys: Vec<std::sync::Arc<egui::Galley>> = lines
        .iter()
        .map(|line| {
            let colour = if line.failure { failure } else { BANNER_COLOUR };
            painter.layout_job(banner_job(painter.ctx(), &line.text, colour, wrap))
        })
        .collect();
    let width = galleys.iter().map(|g| g.size().x).fold(0.0_f32, f32::max) + 2.0 * PAD;
    let height = galleys.iter().map(|g| g.size().y).sum::<f32>()
        + GAP * (galleys.len() - 1) as f32
        + 2.0 * PAD;
    let top_left = Pos2::new(rect.center().x - width / 2.0, rect.top() + BANNER_TOP);
    let panel = Rect::from_min_size(top_left, egui::Vec2::new(width, height));
    painter.rect_filled(panel, 4.0, Color32::from_rgba_unmultiplied(20, 20, 24, 210));
    painter.rect_stroke(
        panel,
        4.0,
        Stroke::new(1.0, BANNER_COLOUR),
        egui::StrokeKind::Inside,
    );
    let mut y = panel.top() + PAD;
    for galley in galleys {
        let height = galley.size().y;
        painter.galley(Pos2::new(panel.left() + PAD, y), galley, BANNER_COLOUR);
        y += height + GAP;
    }
    Some(panel)
}

#[cfg(test)]
mod tests;
