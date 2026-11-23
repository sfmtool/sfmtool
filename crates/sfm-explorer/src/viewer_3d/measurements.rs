// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurements the 3D view draws: a mark at each endpoint that resolves
//! to a finite position and a line between two such, drawn in the scene, and
//! the off-screen indicator for the selected one.
//!
//! What is drawn, the hit test and the press are
//! [`crate::measurement_marks`]'s, which Image Detail shares; what is here is
//! where an endpoint lands in the viewport, the marks and lines as the scene
//! pass draws them ([`ribbons`]), and what a click on the indicator does. The
//! pass draws a piece in front of the scene at full strength and a piece
//! behind it at `BEHIND_OPACITY`; the hit test reads the projection alone, so
//! a measurement behind the scene is hovered and clicked as one in front is.
//! See `specs/gui/measurement-view.md` § "Drawing measurements".

use eframe::egui::{self, Color32, Rect, Vec2};
use nalgebra::Point3;

use crate::measurement_marks::{
    self, Indicator, MeasurementLayer, Place, Screen, ScreenMark, ScreenMeasurement, Target,
};

use super::{Viewer3D, ViewportCamera};

#[cfg(test)]
mod tests;

/// Chords in an endpoint mark's ring.
const RING_SEGMENTS: usize = 24;

/// How far toward the eye a measurement's pieces are drawn, in panel points
/// at each piece's own depth: the selected ring's radius and its halo.
///
/// An endpoint lies on the surface it was read from, so a mark or a line
/// drawn exactly there would be in front of that surface on some pixels and
/// behind it on others. Pulled toward the eye by a little more than its own
/// ring's radius, measured in the same world length at its depth, a mark on a
/// surface turned up to about 50 degrees from the view stays in front of it
/// whatever the field of view and the panel's size, while an occluder more
/// than that length in front of it still hides it.
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
/// where the hit test reads them.
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

/// One straight piece of a drawn measurement as the 3D view's scene pass
/// draws it: the line, a chord of a mark's ring, the halo under either, or
/// the dot at a mark's centre.
///
/// The pass draws it as one camera-facing ribbon of the bench figure's kind,
/// at full strength where it is in front of the scene and at
/// `BEHIND_OPACITY` where it is behind, with no fade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ribbon {
    /// One endpoint in the shared world space, `(xyz, w)` with `w = 1`.
    pub(crate) a: [f32; 4],
    /// The other.
    pub(crate) b: [f32; 4],
    /// Its colour in linear light, not premultiplied, as the pass blends it
    /// onto the sRGB scene texture.
    pub(crate) color: [f32; 4],
    /// Half its width, in physical pixels.
    pub(crate) half_width: f32,
}

/// The marks and lines of `layer` as the scene pass draws them through
/// `camera` in `rect`, in draw order, with `pixels_per_point` physical pixels
/// to a panel point.
///
/// Each measurement draws its line, the halo under the stroke, and then each
/// mark: a ring of [`RING_SEGMENTS`] chords with its halo under it, and the
/// dot at its centre, the order Image Detail paints them in. A mark's ring is
/// the size it is on screen whatever its distance: it is built in the plane
/// facing the eye at the endpoint's depth, its radius in panel points
/// unprojected there. Every piece is drawn [`toward_eye_fraction`] of the way
/// toward the eye.
///
/// An endpoint at infinity or refused draws nothing and the other endpoint
/// still draws its mark. A finite endpoint behind the eye draws no mark, and
/// a line with one end behind it keeps that end: the pass clips the segment
/// at the near plane, as it clips the bench figure's.
pub(crate) fn ribbons(
    layer: &MeasurementLayer,
    camera: &ViewportCamera,
    rect: Rect,
    pixels_per_point: f32,
) -> Vec<Ribbon> {
    let eye = camera.position();
    let keep = 1.0 - toward_eye_fraction(camera, rect);
    let toward_eye = |p: &Point3<f64>| eye + (p - eye) * keep;
    let half = |width: f32| width * 0.5 * pixels_per_point;
    let halo = linear_rgba(measurement_marks::HALO);
    let mut out = Vec::new();
    for drawn in &layer.drawn {
        let color = linear_rgba(drawn.color);
        let halo_width = drawn.width + measurement_marks::HALO_EXTRA_WIDTH;
        let finite: Vec<_> = drawn
            .endpoints
            .iter()
            .filter_map(|e| match e.place {
                Place::Finite(p) => Some((e.side, p)),
                Place::AtInfinity | Place::Refused => None,
            })
            .collect();
        if let [(_, a), (_, b)] = finite.as_slice() {
            let (a, b) = (homogeneous(&toward_eye(a)), homogeneous(&toward_eye(b)));
            for (color, width) in [(halo, halo_width), (color, drawn.width)] {
                out.push(Ribbon {
                    a,
                    b,
                    color,
                    half_width: half(width),
                });
            }
        }
        for (side, p) in &finite {
            let Some(centre) = camera.project_homogeneous(p.coords, 1.0, rect) else {
                continue;
            };
            let depth = -camera.world_to_view(p).z * keep;
            let at = |offset: Vec2| {
                let c = centre + offset;
                homogeneous(&camera.unproject(c.x, c.y, depth, rect))
            };
            let radius = measurement_marks::mark_radius(drawn, *side);
            let ring: Vec<[f32; 4]> = (0..=RING_SEGMENTS)
                .map(|k| {
                    let angle = k as f32 * std::f32::consts::TAU / RING_SEGMENTS as f32;
                    at(Vec2::angled(angle) * radius)
                })
                .collect();
            for (color, width) in [(halo, halo_width), (color, drawn.width)] {
                out.extend(ring.windows(2).map(|chord| Ribbon {
                    a: chord[0],
                    b: chord[1],
                    color,
                    half_width: half(width),
                }));
            }
            // The dot: a ribbon as wide as it is long.
            let dot = measurement_marks::MARK_DOT_RADIUS;
            out.push(Ribbon {
                a: at(Vec2::new(-dot, 0.0)),
                b: at(Vec2::new(dot, 0.0)),
                color,
                half_width: half(2.0 * dot),
            });
        }
    }
    out
}

/// A finite world point as the pass takes it.
fn homogeneous(p: &Point3<f64>) -> [f32; 4] {
    [p.x as f32, p.y as f32, p.z as f32, 1.0]
}

/// `color` in linear light and not premultiplied, as the pass draws it onto
/// the sRGB scene texture, so it shows as the colour egui paints.
pub(crate) fn linear_rgba(color: Color32) -> [f32; 4] {
    let [r, g, b, a] = color.to_srgba_unmultiplied();
    egui::Rgba::from_srgba_unmultiplied(r, g, b, a).to_rgba_unmultiplied()
}

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

    /// Draw the off-screen indicator and the hovered measurement's text over
    /// the scene. The marks and lines are the scene pass's ([`ribbons`]).
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
