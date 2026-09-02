// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The intrinsics overlay layer: the camera's reference frame, drawn on the
//! photograph.
//!
//! Where the feature overlays are the *data* — keypoints, reprojection error,
//! track length — this layer is the *frame that data sits in*: the principal
//! point, how far it is from the image centre, the angular axes and iso-angle
//! rings the lens actually projects, and the displacement field measuring the
//! lens against its family's ideal map. It is an **independent layer**, not an
//! eighth [`crate::state::OverlayMode`], so every joint question ("do the
//! keypoints crowd the distorted rim?", "is the error heatmap hot where the
//! distortion is?") is one glance rather than two and a memory. See
//! `specs/gui/camera-intrinsics.md` § "Image Detail: the Intrinsics overlay
//! layer".
//!
//! # Compositing
//!
//! Two layers drawing on one image need three rules, and the panel's draw order
//! in [`super::ImageDetail::show`] is where they land:
//!
//! - **Z-order.** Everything here draws *beneath* the features, so a keypoint is
//!   never hidden by an axis or an arrow — except the principal-point marker,
//!   which draws last, on top of everything: it is a dozen pixels of ink whose
//!   whole job is to be locatable, and a dense heatmap would otherwise bury it.
//! - **Colour.** The heatmap modes sweep the full colormap, so no hue is safe to
//!   reserve. Every stroke, arrow and label here is near-white with a dark halo
//!   ([`haloed_segment`], [`haloed_text`]), legible over both a bright sky and a
//!   black colormap floor.
//! - **Weight.** Reference geometry draws at [`REFERENCE_ALPHA`]; the principal
//!   point, the centre marker and all text draw fully opaque. The layer is a
//!   reference grid, not a subject.
//!
//! # Caching
//!
//! Nothing here is per-frame work. [`CameraLayer`] is computed once per
//! [`crate::scene::CameraRef`] and held by [`super::ImageDetail`], because the
//! displacement field is a few hundred ray round trips and `undistort` is
//! iterative for the OpenCV family. It is rebuilt only when the camera, the
//! grid density or the node changes.

mod axes;
mod controls;
mod field;
mod hover;

#[cfg(test)]
mod tests;

pub(crate) use controls::show_intrinsics_controls;

use axes::AxisGeometry;
use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke};
use sfmtool_core::camera::report;
use sfmtool_core::camera::CameraIntrinsics;

use crate::state::IntrinsicsDisplaySettings;

/// Near-white, the one colour that survives an arbitrary colormap underneath.
const INK_RGB: [u8; 3] = [246, 247, 250];

/// The dark halo laid under every stroke and label, so near-white stays
/// readable over a bright sky as well as over a dark one.
const HALO: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 190);

/// Opacity of reference geometry — axes, rings, arrows — so the photograph and
/// the features stay readable through it. Text and the point markers are opaque.
const REFERENCE_ALPHA: u8 = 179;

/// Font of every label the layer paints on the image itself.
const LABEL_SIZE: f32 = 10.0;

/// Below this the principal point and the image centre are the same place, and
/// a connector between them would be a smudge rather than a measurement.
const OFFSET_EPSILON_PX: f64 = 0.5;

/// Near-white at `alpha`.
fn ink(alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(INK_RGB[0], INK_RGB[1], INK_RGB[2], alpha)
}

/// The image-to-panel transform the panel is currently drawing at.
///
/// Everything the layer computes is in image pixel coordinates — the same
/// continuous convention the rest of the viewer uses, `(0, 0)` the top-left
/// *corner* — so it pans and zooms with the photograph rather than with the
/// panel.
#[derive(Clone, Copy)]
pub(super) struct View {
    /// Image pixel `(0, 0)` in panel coordinates.
    pub origin: Pos2,
    /// Panel pixels per image pixel.
    pub scale: f32,
    /// Panel pixels per image pixel with the image *fitted* to the panel — the
    /// scale at zoom 1. The arrow exaggeration is fitted to this rather than to
    /// `scale`, so that zooming does not step it up and down the ladder; see
    /// [`field::auto_scale`].
    pub fit: f32,
}

impl View {
    /// An image-pixel position in panel coordinates.
    pub(super) fn at(&self, p: [f64; 2]) -> Pos2 {
        Pos2::new(
            self.origin.x + p[0] as f32 * self.scale,
            self.origin.y + p[1] as f32 * self.scale,
        )
    }

    /// A panel position back in image pixels — what the hover readout works
    /// from.
    pub(super) fn image_pixel(&self, pos: Pos2) -> [f64; 2] {
        [
            f64::from((pos.x - self.origin.x) / self.scale),
            f64::from((pos.y - self.origin.y) / self.scale),
        ]
    }
}

/// One camera's overlay products, computed once and cached by
/// [`crate::scene::CameraRef`].
pub(crate) struct CameraLayer {
    /// The grid the displacement field was sampled on, `(cols, rows)`. Part of
    /// the cache's validity: changing the density rebuilds the field.
    pub grid: (usize, usize),
    /// Where this model stops describing a lens, in degrees off-axis, or `None`
    /// when it describes one at every angle
    /// ([`report::trustworthy_max_theta_deg`]).
    pub limit_deg: Option<f64>,
    /// The largest displacement over the grid nodes **inside**
    /// [`Self::limit_deg`], in image pixels.
    ///
    /// The bound is not a nicety. On a circular fisheye the image rectangle's
    /// corners are outside the lens's image circle, where the distortion
    /// polynomial is unconstrained and folds: on `kerry_park` those nodes report
    /// displacements of hundreds of pixels against the 13 px the lens actually
    /// displaces anything. An unfiltered maximum is a true statement about two
    /// forward maps and a false one about a camera, so it is what neither the
    /// legend quotes nor the arrow scale is fitted to.
    pub max_px: f64,
    /// Grid nodes outside [`Self::limit_deg`], and the total the grid produced.
    pub extrapolated: (usize, usize),
    /// The displacement field itself, one entry per surviving grid node. Empty
    /// for a model that is its own ideal map.
    pub arrows: Vec<Arrow>,
    /// The exaggeration the auto scale last resolved to.
    ///
    /// Written by the draw pass and read by the settings popup, which is laid
    /// out above the image and so quotes the previous frame's value. The
    /// alternative is laying the panel out twice to tell the user a number that
    /// the on-image legend restates every frame anyway.
    pub auto_scale: f32,
    /// The angular reference grid, resampled when the zoom crosses a
    /// half-octave bucket or the rings are switched on.
    geometry: Option<AxisGeometry>,
}

impl CameraLayer {
    /// Compute the layer's products for `camera` at `cols` arrows across.
    pub(super) fn compute(camera: &CameraIntrinsics, cols: usize) -> Self {
        let cols = cols.max(1);
        let rows = grid_rows(camera, cols);
        let limit_deg = report::trustworthy_max_theta_deg(camera);
        let inside = |theta_deg: f64| limit_deg.is_none_or(|limit| theta_deg <= limit);

        // A model that *is* its own ideal map has an identically-zero field, so
        // there is nothing to measure and nothing to draw.
        let field = if camera.has_distortion() {
            report::distortion_field(camera, cols, rows)
        } else {
            Vec::new()
        };
        let total = field.len();
        let mut max_px = 0.0_f64;
        let mut outside = 0;
        let arrows: Vec<Arrow> = field
            .into_iter()
            .map(|sample| {
                let trusted = inside(sample.theta_deg);
                if trusted {
                    max_px = max_px.max(
                        (sample.pixel[0] - sample.reference[0])
                            .hypot(sample.pixel[1] - sample.reference[1]),
                    );
                } else {
                    outside += 1;
                }
                Arrow {
                    reference: sample.reference,
                    pixel: sample.pixel,
                    trusted,
                }
            })
            .collect();

        Self {
            grid: (cols, rows),
            limit_deg,
            max_px,
            extrapolated: (outside, total),
            arrows,
            auto_scale: 1.0,
            geometry: None,
        }
    }

    /// The width of one grid cell in image pixels — what caps the arrow scale,
    /// so the field stays a field rather than becoming a tangle.
    fn cell_px(&self, camera: &CameraIntrinsics) -> f64 {
        f64::from(camera.width) / self.grid.0 as f64
    }
}

/// One node of the displacement field: where the model actually puts a ray, and
/// where the family's ideal map would have put it.
pub(crate) struct Arrow {
    /// The ideal map's pixel — the arrow's head at `×1`, i.e. where the content
    /// under [`Self::pixel`] belongs.
    pub reference: [f64; 2],
    /// The model's pixel, and the grid node itself — the arrow's tail. The tail
    /// is the real pixel because the field is drawn on the real image; see
    /// [`mod@field`].
    pub pixel: [f64; 2],
    /// Whether the sampled ray is inside [`CameraLayer::limit_deg`], and so
    /// whether this node is a measurement at all. See [`mod@field`].
    pub trusted: bool,
}

/// Grid rows that keep the sampled cells square at `cols` across — the same
/// rule the Camera Intrinsics panel's own field uses, so the panel's number and
/// this layer's legend describe one field at the default density.
fn grid_rows(camera: &CameraIntrinsics, cols: usize) -> usize {
    if camera.width == 0 {
        return cols;
    }
    let ratio = f64::from(camera.height) / f64::from(camera.width);
    ((cols as f64 * ratio).round() as usize).max(1)
}

/// How much wider than its stroke a halo is drawn.
///
/// One and a half pixels rather than two: at two, a mark made of several short
/// strokes a few pixels apart — the principal-point reticle — has its halos
/// merge into a dark blob with the mark lost inside it.
const HALO_WIDTH: f32 = 1.5;

/// Draw a polyline in near-white over a dark halo.
///
/// The halo is the same polyline drawn wider underneath, which is cheaper and
/// steadier than an outline pass and is what keeps a 1 px stroke legible over a
/// bright sky as well as over a dark one.
fn haloed_line(painter: &egui::Painter, points: Vec<Pos2>, width: f32, color: Color32) {
    if points.len() < 2 {
        return;
    }
    painter.add(egui::Shape::line(
        points.clone(),
        Stroke::new(width + HALO_WIDTH, HALO),
    ));
    painter.add(egui::Shape::line(points, Stroke::new(width, color)));
}

/// Draw a single segment in near-white over a dark halo.
fn haloed_segment(painter: &egui::Painter, from: Pos2, to: Pos2, width: f32, color: Color32) {
    painter.line_segment([from, to], Stroke::new(width + HALO_WIDTH, HALO));
    painter.line_segment([from, to], Stroke::new(width, color));
}

/// Draw text in near-white over a dark halo, and return the rect it occupied.
///
/// Four offset copies rather than a backing plate: the labels sit on top of the
/// photograph at arbitrary places, and a plate at each of them would read as a
/// second layer of its own.
fn haloed_text(
    painter: &egui::Painter,
    pos: Pos2,
    anchor: Align2,
    text: impl Into<String>,
    color: Color32,
) -> Rect {
    let text = text.into();
    let font = FontId::proportional(LABEL_SIZE);
    for offset in [
        egui::vec2(-1.0, 0.0),
        egui::vec2(1.0, 0.0),
        egui::vec2(0.0, -1.0),
        egui::vec2(0.0, 1.0),
    ] {
        painter.text(pos + offset, anchor, text.clone(), font.clone(), HALO);
    }
    painter.text(pos, anchor, text, font, color)
}

impl super::ImageDetail {
    /// Draw everything in the layer that sits *beneath* the feature overlays,
    /// and return the text the layer contributes to the panel's one tooltip.
    ///
    /// The principal-point marker is deliberately not here: it draws last, over
    /// the features, from [`draw_principal_point`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_intrinsics(
        &mut self,
        painter: &egui::Painter,
        camera_ref: crate::scene::CameraRef,
        camera: &CameraIntrinsics,
        view: &View,
        panel: Rect,
        settings: &IntrinsicsDisplaySettings,
        pointer: Option<Pos2>,
    ) -> Option<String> {
        let layer = self.intrinsics_layer(camera_ref, camera, settings.grid_cols);

        // Resample the grid only when the zoom has crossed a bucket or the
        // rings have been switched on: the ladder is a function of the scale,
        // and a pinch gesture would otherwise rebuild two polylines and a
        // handful of contours sixty times a second.
        let bucket = axes::scale_bucket(view.scale);
        let stale = layer
            .geometry
            .as_ref()
            .is_none_or(|grid| grid.bucket != bucket || grid.has_rings != settings.rings);
        if stale {
            let limit_deg = layer.limit_deg;
            layer.geometry = Some(AxisGeometry::compute(
                camera,
                view.scale,
                settings.rings,
                limit_deg,
            ));
        }
        if let Some(grid) = &layer.geometry {
            grid.draw(painter, view, panel, settings.axes);
        }

        // The field on top of the grid: the grid is the frame, the field is the
        // measurement sitting in it. Both are still beneath the features.
        let mut lines = legend_lines(settings);
        if settings.distortion && !layer.arrows.is_empty() {
            layer.auto_scale = field::auto_scale(layer.max_px, layer.cell_px(camera), view.fit);
            let scale = settings.distortion_scale.unwrap_or(layer.auto_scale);
            lines.extend(field::draw(painter, layer, view, panel, scale));
        }

        draw_centre_offset(painter, camera, view, panel);
        draw_legend(painter, panel, &lines);

        // Not painted here: the panel has one tooltip, and this is the text the
        // feature layer appends its own to. See [`mod@hover`].
        let limit_deg = layer.limit_deg;
        pointer
            .filter(|pos| panel.contains(*pos))
            .and_then(|pos| hover::readout(camera, limit_deg, view.image_pixel(pos)))
    }
}

/// What the layer says about itself, in the panel's top-left corner.
///
/// Top-left because the heatmap modes' colorbar occupies the bottom-right, and
/// two things explaining themselves in one corner is worse than either.
fn draw_legend(painter: &egui::Painter, panel: Rect, lines: &[String]) {
    if lines.is_empty() {
        return;
    }
    let font = FontId::proportional(LABEL_SIZE + 1.0);
    let galleys: Vec<_> = lines
        .iter()
        .map(|line| painter.layout_no_wrap(line.clone(), font.clone(), ink(255)))
        .collect();
    let width = galleys.iter().map(|g| g.size().x).fold(0.0_f32, f32::max);
    let height: f32 = galleys.iter().map(|g| g.size().y).sum();
    let origin = panel.min + egui::vec2(8.0, 8.0);
    painter.rect_filled(
        Rect::from_min_size(origin, egui::vec2(width, height)).expand(4.0),
        3.0,
        Color32::from_black_alpha(150),
    );
    let mut y = origin.y;
    for galley in galleys {
        let step = galley.size().y;
        painter.galley(Pos2::new(origin.x, y), galley, ink(255));
        y += step;
    }
}

/// The legend's lines for the current settings.
fn legend_lines(settings: &IntrinsicsDisplaySettings) -> Vec<String> {
    let mut lines = Vec::new();
    if settings.axes || settings.rings {
        // Which way the signed tick labels run, so nobody has to infer it from
        // the picture. See `axes::Axis::ray` for where the convention is fixed.
        lines.push("angles: off-axis, + right / + up".to_owned());
    }
    lines
}

/// Draw the principal point, the image centre and the offset between them.
///
/// This is the part of the layer that is always drawn when it is enabled: the
/// sub-toggles gate the parts that put real ink across the photograph, and
/// "where is the optical axis, and is it where it should be?" is the question
/// the layer exists to answer at a glance.
///
/// The principal-point marker itself is **not** here — it is drawn last, over
/// the features, by [`draw_principal_point`].
pub(super) fn draw_centre_offset(
    painter: &egui::Painter,
    camera: &CameraIntrinsics,
    view: &View,
    panel: Rect,
) {
    let (cx, cy) = camera.principal_point();
    let centre = [
        f64::from(camera.width) / 2.0,
        f64::from(camera.height) / 2.0,
    ];

    // Under half a pixel the two marks would be drawn on top of each other and
    // the connector between them would be a smudge, so the whole clause goes:
    // the principal-point marker already marks the spot, and a second reticle
    // inside its halo would read as one blob rather than as two coincident
    // facts.
    let Some(label) = offset_label(camera) else {
        return;
    };

    let centre_pos = view.at(centre);
    if !panel.expand(60.0).contains(centre_pos) {
        return;
    }

    // The image centre: a faint `+`, deliberately fainter than the principal
    // point, because it is a fact about the frame rather than about the lens.
    let faint = ink(REFERENCE_ALPHA);
    let arm = 4.0;
    haloed_segment(
        painter,
        centre_pos - egui::vec2(arm, 0.0),
        centre_pos + egui::vec2(arm, 0.0),
        1.0,
        faint,
    );
    haloed_segment(
        painter,
        centre_pos - egui::vec2(0.0, arm),
        centre_pos + egui::vec2(0.0, arm),
        1.0,
        faint,
    );

    let principal_pos = view.at([cx, cy]);
    haloed_segment(painter, centre_pos, principal_pos, 1.0, faint);

    let midpoint = Pos2::new(
        0.5 * (centre_pos.x + principal_pos.x),
        0.5 * (centre_pos.y + principal_pos.y),
    );
    haloed_text(
        painter,
        midpoint + egui::vec2(0.0, -6.0),
        Align2::CENTER_BOTTOM,
        label,
        ink(255),
    );
}

/// How far the principal point sits from the image centre, as a label — or
/// `None` when the two are the same place to within
/// [`OFFSET_EPSILON_PX`].
///
/// Quoted both in pixels and as a fraction of the half-diagonal, because the
/// pixel figure alone does not answer "is that a lot?": the same 12 px is a
/// rounding error on a full frame and a real decentring on a thumbnail.
fn offset_label(camera: &CameraIntrinsics) -> Option<String> {
    let (cx, cy) = camera.principal_point();
    let offset = [
        cx - f64::from(camera.width) / 2.0,
        cy - f64::from(camera.height) / 2.0,
    ];
    let distance = offset[0].hypot(offset[1]);
    if distance < OFFSET_EPSILON_PX {
        return None;
    }
    let half_diagonal = f64::from(camera.width).hypot(f64::from(camera.height)) / 2.0;
    let percent = if half_diagonal > 0.0 {
        100.0 * distance / half_diagonal
    } else {
        0.0
    };
    Some(format!(
        "{DELTA} ({}, {}) px {MIDDOT} {percent:.1}%",
        signed(offset[0]),
        signed(offset[1])
    ))
}

/// The `Δ` of the offset label. Spelled once so the glyph test can pin it.
pub(super) const DELTA: &str = "Δ";

/// A signed number with one decimal, using the typographic minus the rest of
/// the layer's labels use.
fn signed(value: f64) -> String {
    if value < 0.0 {
        format!("{}{:.1}", MINUS, -value)
    } else {
        format!("+{value:.1}")
    }
}

/// U+2212 MINUS SIGN, which lines up with `+` where an ASCII hyphen does not.
/// Spelled once so the glyph test can pin it.
pub(super) const MINUS: &str = "−";

/// U+00B7 MIDDLE DOT, the separator between clauses of a label. Spelled once so
/// the glyph test can pin it.
pub(super) const MIDDOT: &str = "·";

/// U+00B0 DEGREE SIGN, on every angle the layer writes. Spelled once so the
/// glyph test can pin it.
pub(super) const DEGREE: &str = "°";

/// Draw the principal-point marker: a reticle — an open ring with four arms
/// reaching out of it.
///
/// Called **last**, after the feature overlays, and fully opaque. Everything
/// else in the layer yields to the features; this one mark does not, because a
/// dense heatmap would otherwise bury the one position on the image that the
/// whole panel is oriented around.
///
/// The arms start *outside* the ring rather than crossing at the centre. A
/// cross drawn through a 3-pixel circle, both haloed, comes out as a filled
/// dark disc a few pixels across with no readable shape in it; leaving the
/// middle open is what makes it a mark you can put on a pixel.
pub(super) fn draw_principal_point(
    painter: &egui::Painter,
    camera: &CameraIntrinsics,
    view: &View,
    panel: Rect,
) {
    /// Radius of the ring.
    const RING: f32 = 3.5;
    /// Where each arm starts and ends, measured from the centre.
    const ARM: (f32, f32) = (RING + 1.5, RING + 6.0);

    let (cx, cy) = camera.principal_point();
    let at = view.at([cx, cy]);
    if !panel.expand(20.0).contains(at) {
        return;
    }
    let opaque = ink(255);
    for direction in [
        egui::vec2(1.0, 0.0),
        egui::vec2(-1.0, 0.0),
        egui::vec2(0.0, 1.0),
        egui::vec2(0.0, -1.0),
    ] {
        haloed_segment(
            painter,
            at + direction * ARM.0,
            at + direction * ARM.1,
            1.5,
            opaque,
        );
    }
    painter.circle_stroke(at, RING, Stroke::new(1.5 + HALO_WIDTH, HALO));
    painter.circle_stroke(at, RING, Stroke::new(1.5, opaque));
}
