// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The **Change to proposed** field: what an open camera model proposal does
//! to the pixels of the image shown.
//!
//! Three parts, drawn in place of the distortion field while a proposal with
//! a fitted camera is open on the image's camera and the field mode is
//! [`crate::state::FieldMode::ChangeToProposed`]:
//!
//! - **Arrows over the image grid.** Each arrow's tail is a grid node, the
//!   pixel the current camera gives a ray, and its head is the pixel the
//!   proposed camera gives that same ray, exaggerated and scaled as the
//!   distortion field's arrows are ([`super::distortion_field::auto_scale`]).
//!   Only nodes inside the current camera's trusted bound get one: past it the
//!   current camera has no reliable pixel to start from.
//! - **Past the trusted bound**, the proposed camera's iso-angle rings every
//!   5° from the fit's largest angle to the edge of the frame, in the
//!   proposed camera's colour, so the band the current camera cannot describe
//!   is shown as the proposed one divides it.
//! - **The observation layer**: for each observation of the image, a tick at
//!   the keypoint and a line to its reprojection under the current camera and
//!   one under the proposed camera, at true scale.
//!
//! See `specs/gui/camera-intrinsics.md` § "The Change to proposed field".

use egui::{Align2, Color32, Pos2, Rect, Vec2};
use sfmtool_core::camera::report;
use sfmtool_core::camera::CameraIntrinsics;

use super::axes::{contour, panel_points, scale_bucket, Contour};
use super::{
    haloed_line, haloed_segment, haloed_text, ink, Arrow, View, DEGREE, MIDDOT, REFERENCE_ALPHA,
};
use crate::camera_model_proposal::ObservationReprojection;

/// The proposed camera's colour on the photograph: the accent the Camera
/// Intrinsics panel draws the proposed camera in, light enough to read over a
/// dark halo.
const PROPOSED_RGB: [u8; 3] = [108, 182, 255];

/// Spacing of the proposed camera's rings past the trusted bound, in degrees.
const RING_STEP_DEG: f64 = 5.0;

/// How far a ring label keeps from the panel's edge and from any other label,
/// in panel pixels: the halo reaches a pixel past the glyphs.
const LABEL_MARGIN: f32 = 2.0;

/// Half the length of the tick at an observation's keypoint, in panel pixels.
const TICK: f32 = 2.5;

/// The proposed camera's colour at `alpha`.
pub(super) fn proposed_ink(alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(PROPOSED_RGB[0], PROPOSED_RGB[1], PROPOSED_RGB[2], alpha)
}

/// One proposal generation's change field over one camera's image grid.
pub(crate) struct ChangeLayer {
    /// The grid the field was sampled on, `(cols, rows)`, rows chosen to keep
    /// the cells square as the distortion field's are.
    pub(crate) grid: (usize, usize),
    /// The current camera's trusted bound, or `None` when it has none.
    pub(crate) limit_deg: Option<f64>,
    /// One arrow per grid node inside the bound: the tail the node, the head
    /// the proposed camera's pixel for the ray the current camera gives it.
    pub(crate) arrows: Vec<Arrow>,
    /// The longest arrow, in image pixels.
    pub(crate) max_px: f64,
    /// Grid nodes past the bound, and the total the grid produced.
    pub(crate) past_bound: (usize, usize),
    /// The angles of the proposed camera's rings, ascending.
    pub(crate) ring_angles_deg: Vec<f64>,
    /// The exaggeration the auto scale last resolved to, for the popup.
    pub(crate) auto_scale: f32,
    /// The rings, sampled at a zoom bucket.
    rings: Option<(i32, Vec<Contour>)>,
}

impl ChangeLayer {
    /// The change from `current` to `proposed` at `cols` nodes across, for a
    /// fit whose largest angle was `theta_fit_deg`.
    pub(crate) fn compute(
        current: &CameraIntrinsics,
        proposed: &CameraIntrinsics,
        cols: usize,
        theta_fit_deg: f64,
    ) -> Self {
        let cols = cols.max(1);
        let (w, h) = (f64::from(current.width), f64::from(current.height));
        let rows = if current.width == 0 {
            cols
        } else {
            ((cols as f64 * h / w).round() as usize).max(1)
        };
        let limit_deg = report::trustworthy_max_theta_deg(current);
        let mut arrows = Vec::new();
        let mut max_px = 0.0_f64;
        let mut past = 0;
        let mut total = 0;
        for j in 0..rows {
            for i in 0..cols {
                let u = (i as f64 + 0.5) * w / cols as f64;
                let v = (j as f64 + 0.5) * h / rows as f64;
                total += 1;
                let theta = report::off_axis_angle_deg(current, u, v);
                if limit_deg.is_some_and(|limit| theta > limit) {
                    past += 1;
                    continue;
                }
                let ray = current.pixel_to_ray(u, v);
                let Some((pu, pv)) = proposed.ray_to_pixel(ray) else {
                    continue;
                };
                max_px = max_px.max((pu - u).hypot(pv - v));
                arrows.push(Arrow {
                    reference: [pu, pv],
                    pixel: [u, v],
                    trusted: true,
                });
            }
        }
        Self {
            grid: (cols, rows),
            limit_deg,
            arrows,
            max_px,
            past_bound: (past, total),
            ring_angles_deg: ring_angles(proposed, theta_fit_deg),
            auto_scale: 1.0,
            rings: None,
        }
    }

    /// The width of one grid cell in image pixels, which caps the scale.
    fn cell_px(&self, camera: &CameraIntrinsics) -> f64 {
        f64::from(camera.width) / self.grid.0 as f64
    }

    /// Draw the arrows and the rings at `scale`, and return the legend lines
    /// and the rings' labels, which [`place_ring_labels`] paints once the
    /// legend's box is known.
    pub(super) fn draw(
        &mut self,
        painter: &egui::Painter,
        current: &CameraIntrinsics,
        proposed: &CameraIntrinsics,
        view: &View,
        panel: Rect,
        scale: Option<f32>,
    ) -> (Vec<String>, Vec<RingLabel>) {
        self.auto_scale =
            super::distortion_field::auto_scale(self.max_px, self.cell_px(current), view.fit);
        let scale = scale.unwrap_or(self.auto_scale);
        let color = ink(REFERENCE_ALPHA);
        for arrow in &self.arrows {
            super::distortion_field::draw_arrow(painter, arrow, view, panel, scale, color);
        }

        let bucket = scale_bucket(view.scale);
        if self.rings.as_ref().is_none_or(|(b, _)| *b != bucket) {
            let rings = self
                .ring_angles_deg
                .iter()
                .filter_map(|&angle| {
                    contour(proposed, angle, view.scale, format!("{angle:.0}{DEGREE}"))
                })
                .collect();
            self.rings = Some((bucket, rings));
        }
        let ring_labels = match &self.rings {
            Some((_, rings)) => draw_rings(painter, rings, view, panel),
            None => Vec::new(),
        };

        let mut lines = vec![format!(
            "change to proposed max {:.2} px {MIDDOT} shown {}{}",
            self.max_px,
            super::controls::TIMES,
            super::controls::scale_text(scale)
        )];
        if let (Some(limit), (past, total)) = (self.limit_deg, self.past_bound) {
            if past > 0 {
                lines.push(format!(
                    "{past} of {total} nodes past {limit:.1}{DEGREE} {MIDDOT} proposed rings \
                     every {RING_STEP_DEG:.0}{DEGREE}"
                ));
            }
        }
        (lines, ring_labels)
    }
}

/// The angles of the proposed camera's rings: every multiple of
/// [`RING_STEP_DEG`] past the fit's largest angle, out to the frame's corner
/// under the proposed camera, and short of its own trusted bound where it has
/// one.
pub(crate) fn ring_angles(proposed: &CameraIntrinsics, theta_fit_deg: f64) -> Vec<f64> {
    let Some(fov) = report::field_of_view(proposed) else {
        return Vec::new();
    };
    let edge = report::trustworthy_max_theta_deg(proposed)
        .map_or(fov.max_off_axis, |limit| limit.min(fov.max_off_axis));
    let mut angles = Vec::new();
    let mut angle = (theta_fit_deg / RING_STEP_DEG).floor() * RING_STEP_DEG + RING_STEP_DEG;
    while angle <= edge {
        angles.push(angle);
        angle += RING_STEP_DEG;
    }
    angles
}

/// A ring label waiting to be placed: its text and the point it names, in
/// panel coordinates.
pub(super) struct RingLabel {
    pub(super) text: String,
    pub(super) at: Pos2,
}

/// Paint the rings in the proposed camera's colour, and return their labels
/// for [`place_ring_labels`] to put down once every other label of the layer,
/// the legend's box included, is known.
fn draw_rings(
    painter: &egui::Painter,
    rings: &[Contour],
    view: &View,
    panel: Rect,
) -> Vec<RingLabel> {
    let color = proposed_ink(REFERENCE_ALPHA);
    let mut labels = Vec::new();
    for ring in rings {
        for run in &ring.runs {
            haloed_line(painter, panel_points(run, view), 1.0, color);
        }
        let at = view.at(ring.label_at);
        if panel.contains(at) {
            labels.push(RingLabel {
                text: ring.label.clone(),
                at,
            });
        }
    }
    labels
}

/// Paint each ring label where [`label_rect`] puts it, keeping clear of
/// `avoid` (the layer's other labels and the legend's box) and of the ring
/// labels already painted; a label with no such place is left off.
pub(super) fn place_ring_labels(
    painter: &egui::Painter,
    labels: &[RingLabel],
    panel: Rect,
    avoid: &mut Vec<Rect>,
) {
    let font = egui::FontId::proportional(super::LABEL_SIZE);
    for label in labels {
        let size = painter
            .layout_no_wrap(label.text.clone(), font.clone(), proposed_ink(255))
            .size();
        if let Some(rect) = label_rect(size, label.at, panel, avoid) {
            haloed_text(
                painter,
                rect.min,
                Align2::LEFT_TOP,
                label.text.clone(),
                proposed_ink(255),
            );
            avoid.push(rect);
        }
    }
}

/// Where a label of `size` naming the point `at` goes: centred above it, moved
/// inward so the whole label is inside `panel`, or `None` when it cannot fit
/// there or would touch one of `avoid`.
pub(super) fn label_rect(size: Vec2, at: Pos2, panel: Rect, avoid: &[Rect]) -> Option<Rect> {
    // The halo reaches a pixel past the glyphs; keep it inside as well.
    let inner = panel.shrink(LABEL_MARGIN);
    if size.x > inner.width() || size.y > inner.height() {
        return None;
    }
    let rect = Rect::from_min_size(Pos2::new(at.x - size.x / 2.0, at.y - 3.0 - size.y), size);
    let dx = (inner.left() - rect.left()).max(0.0) + (inner.right() - rect.right()).min(0.0);
    let dy = (inner.top() - rect.top()).max(0.0) + (inner.bottom() - rect.bottom()).min(0.0);
    let rect = rect.translate(egui::vec2(dx, dy));
    let clear = avoid
        .iter()
        .all(|other| !other.intersects(rect.expand(LABEL_MARGIN)));
    clear.then_some(rect)
}

/// Draw the observation layer, and return its legend line, or nothing when
/// the image has no observation to draw.
pub(super) fn draw_observations(
    painter: &egui::Painter,
    observations: &[ObservationReprojection],
    view: &View,
    panel: Rect,
) -> Option<String> {
    if observations.is_empty() {
        return None;
    }
    let current = ink(REFERENCE_ALPHA);
    let proposed = proposed_ink(255);
    let grown = panel.expand(20.0);
    for observation in observations {
        let at = view.at(observation.keypoint);
        let ends = [
            (observation.current, current),
            (observation.proposed, proposed),
        ];
        let visible = grown.contains(at)
            || ends
                .iter()
                .any(|(end, _)| end.is_some_and(|end| grown.contains(view.at(end))));
        if !visible {
            continue;
        }
        for (end, color) in ends {
            if let Some(end) = end {
                haloed_segment(painter, at, view.at(end), 1.0, color);
            }
        }
        let tick = ink(255);
        haloed_segment(
            painter,
            at - egui::vec2(TICK, 0.0),
            at + egui::vec2(TICK, 0.0),
            1.0,
            tick,
        );
        haloed_segment(
            painter,
            at - egui::vec2(0.0, TICK),
            at + egui::vec2(0.0, TICK),
            1.0,
            tick,
        );
    }
    Some(format!(
        "{} observations {MIDDOT} reprojection: white current, blue proposed",
        observations.len()
    ))
}
