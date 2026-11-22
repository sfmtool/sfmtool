// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurements Image Detail draws: a mark at each endpoint pixel in the
//! image shown and a line where both endpoints are in it, the focused item's
//! marks as handles, and the off-screen indicator for the selected one.
//!
//! What is drawn, the hit test and the press are
//! [`crate::measurement_marks`]'s, which the 3D view shares; what is here is
//! where a pixel lands in the panel, the handle drag, and what a click on the
//! indicator does. See `specs/gui/measurement-view.md` § "Drawing
//! measurements".

use eframe::egui::{self, Pos2, Rect, Vec2};

use crate::measurement_marks::{
    self, DrawnId, Indicator, MeasurementGesture, Place, Press, Screen, ScreenMark,
    ScreenMeasurement, Target,
};
use crate::scene::{ImageRef, ReconId};

use super::{ImageDetail, ImageDetailResponse, Look};

#[cfg(test)]
mod tests;

/// Where the image shown sits in the panel this frame.
#[derive(Debug, Clone, Copy)]
pub(super) struct Placement {
    pub(super) image: ImageRef,
    /// The image's size, in its own pixels.
    pub(super) image_size: Vec2,
    /// The image's rectangle in the panel.
    pub(super) image_rect: Rect,
    /// Panel points per image pixel.
    pub(super) scale: f32,
    /// The panel body.
    pub(super) panel_rect: Rect,
}

impl Placement {
    /// Where image pixel `pixel` lands in the panel.
    fn to_panel(self, pixel: [f64; 2]) -> Pos2 {
        self.image_rect.min + Vec2::new(pixel[0] as f32, pixel[1] as f32) * self.scale
    }

    /// The image pixel under `pos`, held inside the image: a drag past an
    /// edge lands on the edge.
    fn to_image(self, pos: Pos2) -> [f64; 2] {
        let local = (pos - self.image_rect.min) / self.scale;
        let inside = |v: f32, extent: f32| f64::from(v.clamp(0.0, (extent - 1e-3).max(0.0)));
        [
            inside(local.x, self.image_size.x),
            inside(local.y, self.image_size.y),
        ]
    }

    fn node(self) -> ReconId {
        self.image.recon
    }
}

impl ImageDetail {
    /// The layer's measurements with an endpoint in the image shown, at
    /// their pixels in the panel, with the indicator where the selected one
    /// has an endpoint in the image and no part of it is in the panel.
    ///
    /// A mark is drawn whether or not its endpoint resolves, since the pixel
    /// is what the measurement stores; a refused endpoint's is in the
    /// warning colour.
    pub(super) fn measurement_screen(&self, painter: &egui::Painter, at: &Placement) -> Screen {
        let layer = &self.measurements;
        let mut screen = Screen::default();
        for (index, drawn) in layer.drawn.iter().enumerate() {
            if drawn.node != at.node() {
                continue;
            }
            let marks: Vec<ScreenMark> = drawn
                .endpoints
                .iter()
                .filter(|e| e.image as usize == at.image.index())
                .map(|e| ScreenMark {
                    side: e.side,
                    pos: at.to_panel(e.pixel),
                    warning: e.place == Place::Refused,
                })
                .collect();
            if marks.is_empty() {
                continue;
            }
            let line = match marks.as_slice() {
                [a, b] => Some([a.pos, b.pos]),
                _ => None,
            };
            screen
                .measurements
                .push(ScreenMeasurement { index, marks, line });
        }
        screen.indicator = (|| {
            let selected = layer.selected().filter(|d| d.node == at.node())?;
            let shown = screen.of(layer, &selected.id)?;
            if shown.in_view(at.panel_rect) {
                return None;
            }
            let target = indicator_target(shown);
            let size = measurement_marks::label_size(painter, &selected.title);
            Indicator::new(
                at.panel_rect,
                target - at.panel_rect.center(),
                &selected.title,
                size,
            )
        })();
        screen
    }

    /// Follow a press over the measurements for this frame and say whether
    /// the pointer is theirs (G7).
    ///
    /// `taken` is a press something else already holds, the bench layer's
    /// handle or Create Track Here's chord. A click on a mark or a line is the
    /// gesture the dock applies; a released drag of the focused item's handle
    /// is the step that sets that endpoint to the pixel under the release, in
    /// the image shown; a click on the indicator centres the panel on the
    /// pixel it points to, at the current zoom.
    pub(super) fn update_measurement_press(
        &mut self,
        ui: &egui::Ui,
        interact: &egui::Response,
        painter: &egui::Painter,
        at: &Placement,
        taken: bool,
        response: &mut ImageDetailResponse,
    ) -> bool {
        if taken && self.measurement_press.is_none() {
            self.measurement_hover = None;
            return false;
        }
        let screen = self.measurement_screen(painter, at);
        let followed =
            measurement_marks::follow(&mut self.measurement_press, ui, interact, |pos| {
                screen.hit(&self.measurements, pos)
            });
        if let Some((press, double)) = &followed.released {
            self.release_measurement_press(press, *double, &screen, at, response);
        }
        let held = self.measurement_press.is_some();
        self.measurement_hover = followed.hovered.clone();
        if let Some(target) = &followed.hovered {
            let handle = measurement_marks::is_handle(&self.measurements, target);
            ui.ctx()
                .set_cursor_icon(measurement_marks::cursor(handle, held));
        }
        followed.owns
    }

    /// What a press that came up asks for.
    fn release_measurement_press(
        &mut self,
        press: &Press,
        double: bool,
        screen: &Screen,
        at: &Placement,
        response: &mut ImageDetailResponse,
    ) {
        if press.cancelled {
            return;
        }
        if press.moved {
            if let Target::Mark(DrawnId::Item { node, label }, side) = &press.target {
                if measurement_marks::is_handle(&self.measurements, &press.target) {
                    response.measurement_gesture = Some(MeasurementGesture::SetEndpoint(
                        crate::bench::measurement::EndpointSet {
                            node: *node,
                            item: label.clone(),
                            side: *side,
                            image: at.image.index() as u32,
                            pixel: at.to_image(press.to),
                        },
                    ));
                }
            }
            return;
        }
        match &press.target {
            Target::Indicator => {
                let Some(target) = self
                    .measurements
                    .selected()
                    .and_then(|d| screen.of(&self.measurements, &d.id))
                    .map(indicator_target)
                else {
                    return;
                };
                let pixel = at.to_image(target);
                let geometry = self.geometry(at.image, at.image_size, at.panel_rect.size());
                self.look(
                    &Look::Pixel {
                        pixel: [pixel[0] as f32, pixel[1] as f32],
                        zoom: None,
                    },
                    geometry,
                );
            }
            target => {
                response.measurement_gesture = measurement_marks::click_gesture(target, double)
            }
        }
    }

    /// Draw the measurements over the photograph, a held handle at the pixel
    /// it would set, and the hovered one's text beside the pointer.
    pub(super) fn draw_measurements(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        interact: &egui::Response,
        at: &Placement,
    ) {
        let mut screen = self.measurement_screen(painter, at);
        if let Some(press) = self.measurement_press.as_ref().filter(|p| p.moved) {
            if let Target::Mark(id, side) = &press.target {
                if measurement_marks::is_handle(&self.measurements, &press.target) {
                    let pos = at.to_panel(at.to_image(press.to));
                    screen.move_mark(&self.measurements, id, *side, pos);
                }
            }
        }
        screen.draw(painter, &self.measurements, ui.visuals().warn_fg_color);
        let text = self
            .measurement_hover
            .as_ref()
            .and_then(|target| measurement_marks::hover_text(&self.measurements, target));
        if let (Some(text), Some(pos)) = (text, interact.hover_pos()) {
            measurement_marks::draw_hover_text(painter, pos, text, at.panel_rect);
        }
    }
}

/// Where the indicator points in the panel: the midpoint of the two marks
/// where both endpoints are in the image, or the one mark.
fn indicator_target(shown: &ScreenMeasurement) -> Pos2 {
    let sum = shown
        .marks
        .iter()
        .fold(Vec2::ZERO, |sum, mark| sum + mark.pos.to_vec2());
    (sum / shown.marks.len().max(1) as f32).to_pos2()
}
