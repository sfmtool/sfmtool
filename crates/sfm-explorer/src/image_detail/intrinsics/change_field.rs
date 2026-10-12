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

use egui::{Align2, Color32, Pos2, Rect};
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

/// Below this distance from the last labelled ring's label, in panel pixels,
/// a ring is drawn without its label, so close rings do not stack theirs.
const RING_LABEL_SPACING: f32 = 16.0;

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

    /// Draw the arrows and the rings at `scale`, and return the legend lines.
    pub(super) fn draw(
        &mut self,
        painter: &egui::Painter,
        current: &CameraIntrinsics,
        proposed: &CameraIntrinsics,
        view: &View,
        panel: Rect,
        scale: Option<f32>,
    ) -> Vec<String> {
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
        if let Some((_, rings)) = &self.rings {
            draw_rings(painter, rings, view, panel);
        }

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
        lines
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

/// Paint the rings in the proposed camera's colour, labelling one only where
/// its label clears the last one drawn.
fn draw_rings(painter: &egui::Painter, rings: &[Contour], view: &View, panel: Rect) {
    let color = proposed_ink(REFERENCE_ALPHA);
    let mut last_label: Option<Pos2> = None;
    for ring in rings {
        for run in &ring.runs {
            haloed_line(painter, panel_points(run, view), 1.0, color);
        }
        let at = view.at(ring.label_at);
        if !panel.contains(at) {
            continue;
        }
        if last_label.is_some_and(|last| last.distance(at) < RING_LABEL_SPACING) {
            continue;
        }
        haloed_text(
            painter,
            at - egui::vec2(0.0, 3.0),
            Align2::CENTER_BOTTOM,
            ring.label.clone(),
            proposed_ink(255),
        );
        last_label = Some(at);
    }
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
