// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurements the 3D view draws: a mark at each endpoint that resolves
//! to a finite position and a line between two such, projected through the
//! viewport camera, and the off-screen indicator for the selected one.
//!
//! What is drawn, the hit test and the press are
//! [`crate::measurement_marks`]'s, which Image Detail shares; what is here is
//! where an endpoint lands in the viewport, and what a click on the indicator
//! does. See `specs/gui/measurement-view.md` § "Drawing measurements".

use eframe::egui::{self, Rect, Vec2};

use crate::measurement_marks::{
    self, Indicator, Place, Screen, ScreenMark, ScreenMeasurement, Target,
};

use super::Viewer3D;

#[cfg(test)]
mod tests;

impl Viewer3D {
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
    /// `taken` is a press something drawn above them already holds: the bench
    /// figure's handle or the brush's stroke. A click on a mark or a line
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

    /// Draw the measurements over the scene, and the hovered one's text
    /// beside the pointer.
    pub(super) fn draw_measurements(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        response: &egui::Response,
        rect: Rect,
    ) {
        let screen = self.measurement_screen(painter, rect);
        screen.draw(painter, &self.measurements, ui.visuals().warn_fg_color);
        let text = self
            .measurement_hover
            .as_ref()
            .and_then(|target| measurement_marks::hover_text(&self.measurements, target));
        if let (Some(text), Some(pos)) = (text, response.hover_pos()) {
            measurement_marks::draw_hover_text(painter, pos, text, rect);
        }
    }
}
