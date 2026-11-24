// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurements the 3D view draws: a mark at each endpoint that resolves
//! to a finite position and a line between two such, drawn in the scene, and
//! the off-screen indicator for the selected one.
//!
//! What is drawn, the hit test and the press are
//! [`crate::measurement_marks`]'s, which Image Detail shares; what is here is
//! where an endpoint lands in the viewport, the marks and lines as the scene
//! pass draws them ([`instances`]), and what a click on the indicator does. The
//! pass draws a piece in front of the scene at full strength and a piece
//! behind it at `BEHIND_OPACITY`; the hit test reads the projection alone, so
//! a measurement behind the scene is hovered and clicked as one in front is.
//! See `specs/gui/measurement-view.md` § "Drawing measurements".

use eframe::egui::{self, Color32, Rect, Vec2};
use nalgebra::Point3;

use crate::measurement_marks::{
    self, Indicator, MeasurementLayer, Place, Screen, ScreenMark, ScreenMeasurement, Target,
};
use crate::scene_renderer::OverlayInstance;

use super::{Viewer3D, ViewportCamera};

#[cfg(test)]
mod tests;

/// How far toward the eye a measurement's pieces are drawn, in panel points
/// at each piece's own depth: the selected ring's radius and its halo.
///
/// An endpoint lies on the surface it was read from, so a mark or a line
/// drawn exactly there would be in front of that surface on some pixels and
/// behind it on others. Pulled toward the eye by as far as the selected
/// ring's halo reaches from its centre, measured in the same world length at
/// its depth, a mark and its halo on a surface turned up to 45 degrees from
/// the view stay in front of it whatever the field of view and the panel's
/// size, while an occluder more than that length in front of it still hides
/// it.
pub(crate) const PULL_POINTS: f64 = (measurement_marks::MARK_RADIUS
    * measurement_marks::SELECTED_MARK_SCALE
    + (measurement_marks::EMPHASISED_LINE_WIDTH + measurement_marks::HALO_EXTRA_WIDTH) / 2.0)
    as f64;

/// The most of its distance toward the eye a piece is drawn, however wide the
/// field of view and however short the panel.
const MAX_TOWARD_EYE: f64 = 0.25;

/// What fraction of its distance toward the eye every piece is drawn through
/// `camera` in `rect`: [`PULL_POINTS`] as a world length at the piece's depth.
///
/// One panel point at view depth `d` is `d · 2·tan(vfov / 2) / height` long,
/// so a pull of a fixed number of points is the same fraction of every
/// piece's depth, and the pull is a uniform scale toward the eye. A point
/// moved along its own line of sight projects where it did, so the marks land
/// where the hit test reads them. The 3D view hands it to the pass once a
/// frame ([`crate::scene_renderer::OverlayView`]), and the shader moves each
/// piece flagged [`OverlayInstance::PULL_TOWARD_EYE`] by it.
pub(crate) fn toward_eye_fraction(camera: &ViewportCamera, rect: Rect) -> f64 {
    let aspect = f64::from(rect.width()) / f64::from(rect.height());
    let per_point = 2.0 * (camera.vertical_fov(aspect) / 2.0).tan() / f64::from(rect.height());
    let fraction = PULL_POINTS * per_point;
    if fraction.is_finite() {
        fraction.clamp(0.0, MAX_TOWARD_EYE)
    } else {
        0.0
    }
}

/// How every piece of a measurement is drawn: held at `BEHIND_OPACITY` behind
/// the scene, and pulled toward the eye.
const FLAGS: u32 = OverlayInstance::HOLD | OverlayInstance::PULL_TOWARD_EYE;

/// The rings, dots and lines of `layer` as the scene pass draws them, in draw
/// order, built from nothing that depends on the camera.
///
/// Every halo comes first, then every stroke, then every dot, each in the
/// layer's order: so no measurement's halo crosses another's stroke, and a
/// dot sits on its own ring's line. A measurement's halos are its line's and
/// then each mark's ring's; its strokes are its line and then each mark's
/// ring. A ring and a dot are one instance each, sized in points about the
/// endpoint, which the pass turns to face the eye there; the widths are the
/// strokes' in points. Every piece is flagged to hold at `BEHIND_OPACITY`
/// behind the scene and to be drawn toward the eye ([`toward_eye_fraction`]).
///
/// An endpoint at infinity or refused draws nothing and the other endpoint
/// still draws its mark. A finite endpoint behind the eye draws no mark and a
/// line with one end behind it is clipped at the near plane: the pass does
/// both, as it clips the bench figure's segments.
pub(crate) fn instances(layer: &MeasurementLayer) -> Vec<OverlayInstance> {
    let halo = encoded_rgba(measurement_marks::HALO);
    let mut halos = Vec::new();
    let mut strokes = Vec::new();
    let mut dots = Vec::new();
    for drawn in &layer.drawn {
        let color = encoded_rgba(drawn.color);
        let half = drawn.width / 2.0;
        let halo_half = (drawn.width + measurement_marks::HALO_EXTRA_WIDTH) / 2.0;
        let finite: Vec<_> = drawn
            .endpoints
            .iter()
            .filter_map(|e| match e.place {
                Place::Finite(p) => Some((e.side, homogeneous(&p))),
                Place::AtInfinity | Place::Refused => None,
            })
            .collect();
        if let [(_, a), (_, b)] = finite.as_slice() {
            halos.push(OverlayInstance::segment(*a, *b, halo, halo_half, FLAGS));
            strokes.push(OverlayInstance::segment(*a, *b, color, half, FLAGS));
        }
        for (side, p) in &finite {
            let radius = measurement_marks::mark_radius(drawn, *side);
            halos.push(OverlayInstance::ring(*p, halo, radius, halo_half, FLAGS));
            strokes.push(OverlayInstance::ring(*p, color, radius, half, FLAGS));
            dots.push(OverlayInstance::disc(
                *p,
                color,
                measurement_marks::MARK_DOT_RADIUS,
                FLAGS,
            ));
        }
    }
    halos.append(&mut strokes);
    halos.append(&mut dots);
    halos
}

/// A finite world point as the pass takes it.
fn homogeneous(p: &Point3<f64>) -> [f32; 4] {
    [p.x as f32, p.y as f32, p.z as f32, 1.0]
}

/// `color` as sRGB-encoded values, not premultiplied, as the measurements'
/// pipeline takes it.
///
/// That pipeline draws into the measurements' layer
/// (`MEASUREMENT_TARGET_FORMAT`), cleared to transparent, with straight alpha,
/// so one piece leaves `c·a` there with alpha `a`. Dividing egui's
/// premultiplied bytes `p` by `a` here, without rounding, leaves `p` itself,
/// and egui composites the layer over the scene as it composited `p` when it
/// painted the measurement: a translucent colour shows as egui painted it, not
/// only an opaque one.
pub(crate) fn encoded_rgba(color: Color32) -> [f32; 4] {
    let [r, g, b, a] = color.to_array().map(f32::from);
    if a == 0.0 {
        return [0.0; 4];
    }
    [r / a, g / a, b / a, a / 255.0]
}

impl Viewer3D {
    /// Build the measurements' instances from [`Self::measurements`] for
    /// `app.rs` to write, when [`Self::measurement_key`] differs from the key
    /// they were last built on; otherwise do nothing.
    pub(super) fn build_measurement_instances(&mut self) {
        if self.measurement_built.as_ref() == Some(&self.measurement_key) {
            return;
        }
        self.measurement_upload = Some(instances(&self.measurements));
        self.measurement_built = Some(self.measurement_key.clone());
        #[cfg(test)]
        {
            self.measurement_builds += 1;
        }
    }

    /// Draw no measurements in the scene: the 3D view has no node to show.
    /// Empties the store's region once, and the next frame that shows a node
    /// builds the instances again.
    pub(crate) fn clear_measurement_instances(&mut self) {
        if self.measurement_built.take().is_some() {
            self.measurement_upload = Some(Vec::new());
        }
    }

    /// The layer's measurements as the viewport camera sees them now in
    /// `rect`, with the indicator where the selected measurement is drawn
    /// and no part of it is in view.
    ///
    /// An endpoint at infinity or refused draws nothing; a finite one behind
    /// the eye draws no mark, and a line with one end behind it is clipped at
    /// the near plane.
    pub(crate) fn measurement_screen(&self, painter: &egui::Painter, rect: Rect) -> Screen {
        let layer = &self.measurements;
        let mut screen = Screen::default();
        for (index, drawn) in layer.drawn.iter().enumerate() {
            let finite: Vec<_> = drawn
                .endpoints
                .iter()
                .filter_map(|e| match e.place {
                    Place::Finite(p) => Some((e.side, p)),
                    Place::AtInfinity | Place::Refused => None,
                })
                .collect();
            let marks = finite
                .iter()
                .filter_map(|(side, p)| {
                    let pos = self.camera.project_homogeneous(p.coords, 1.0, rect)?;
                    Some(ScreenMark {
                        side: *side,
                        pos,
                        warning: false,
                    })
                })
                .collect();
            let line = match finite.as_slice() {
                [(_, a), (_, b)] => self
                    .camera
                    .project_line_clipped(a, b, rect)
                    .map(|(a, b)| [a, b]),
                _ => None,
            };
            screen
                .measurements
                .push(ScreenMeasurement { index, marks, line });
        }
        screen.indicator = self.measurement_indicator(&screen, painter, rect);
        screen
    }

    /// The off-screen indicator for the selected measurement, or `None` while
    /// any part of it is in view or it has no finite endpoint.
    ///
    /// It points toward the projection of the midpoint of the two endpoints,
    /// or of the one finite endpoint. The direction on screen is the point's
    /// view-space `(x, -y)`: in front of the eye that is the direction of its
    /// projection from the view's centre, and behind it, where a projection
    /// lands on the opposite side, it is that projection reflected through
    /// the centre, so the arrow points the way to turn.
    fn measurement_indicator(
        &self,
        screen: &Screen,
        painter: &egui::Painter,
        rect: Rect,
    ) -> Option<Indicator> {
        let layer = &self.measurements;
        let selected = layer.selected()?;
        if screen
            .of(layer, &selected.id)
            .is_some_and(|m| m.in_view(rect))
        {
            return None;
        }
        let points = selected.world_points();
        if points.is_empty() {
            return None;
        }
        let centroid = points
            .iter()
            .fold(nalgebra::Vector3::zeros(), |sum, p| sum + p.coords)
            / points.len() as f64;
        let view = self.camera.world_to_view(&nalgebra::Point3::from(centroid));
        let toward = Vec2::new(view.x as f32, -view.y as f32);
        let size = measurement_marks::label_size(painter, &selected.title);
        Indicator::new(rect, toward, &selected.title, size)
    }

    /// Follow a press over the measurements for this frame, against what the
    /// last camera showed, and say whether the pointer is theirs (G7).
    ///
    /// `taken` is a press something that takes it first already holds: the
    /// bench figure's handle or the brush's stroke. A click on a mark or a line
    /// becomes the gesture the dock applies ([`Viewer3D::measurement_gesture`]);
    /// a click on the indicator frames the selected measurement here, with
    /// [`Viewer3D::frame_keeping_orientation`].
    pub(super) fn update_measurement_press(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        painter: &egui::Painter,
        rect: Rect,
        taken: bool,
        current_time: f64,
    ) -> bool {
        if taken && self.measurement_press.is_none() {
            self.measurement_hover = None;
            return false;
        }
        let screen = self.measurement_screen(painter, rect);
        let followed =
            measurement_marks::follow(&mut self.measurement_press, ui, response, |pos| {
                screen.hit(&self.measurements, pos)
            });
        if let Some((press, double)) = &followed.released {
            if !press.cancelled && !press.moved {
                match &press.target {
                    Target::Indicator => {
                        let points = self
                            .measurements
                            .selected()
                            .map(|d| d.world_points())
                            .unwrap_or_default();
                        self.frame_keeping_orientation(&points, current_time);
                    }
                    target => {
                        self.measurement_gesture = measurement_marks::click_gesture(target, *double)
                    }
                }
            }
        }
        if let Some(target) = &followed.hovered {
            ui.ctx().set_cursor_icon(measurement_marks::cursor(
                false,
                self.measurement_press.is_some(),
            ));
            self.measurement_hover = Some(target.clone());
        } else {
            self.measurement_hover = None;
        }
        followed.owns
    }

    /// Draw the off-screen indicator and the hovered measurement's text over
    /// the scene. The marks and lines are the scene pass's ([`instances`]).
    pub(super) fn draw_measurements(
        &self,
        painter: &egui::Painter,
        response: &egui::Response,
        rect: Rect,
    ) {
        let screen = self.measurement_screen(painter, rect);
        screen.draw_indicator(painter, &self.measurements);
        let text = self
            .measurement_hover
            .as_ref()
            .and_then(|target| measurement_marks::hover_text(&self.measurements, target));
        if let (Some(text), Some(pos)) = (text, response.hover_pos()) {
            measurement_marks::draw_hover_text(painter, pos, text, rect);
        }
    }
}
