// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The distortion displacement field: where the content under each pixel
//! belongs, drawn as an arrow from the pixel the model actually projects a ray
//! to — a real pixel of the photograph on screen — toward where the family's
//! ideal map would have put that ray. Each arrow is what rectifying this image
//! would do to the pixel at its tail.
//!
//! # Why the tail is the real pixel
//!
//! The other direction is just as true arithmetically, and was drawn first: an
//! arrow from the ideal position to the actual one is a faithful picture of how
//! a pixel moves from undistorted to distorted, and its sign is right (a
//! positive `k1` puts the actual pixel further out, so those arrows pointed
//! outward). It is still the wrong direction to draw *here*, because the field
//! is painted **on the distorted photograph**. Every pixel on screen is an
//! actual pixel, so an arrow tailed at an ideal position starts at a point that
//! does not exist in the image being looked at — and a reader seeing an arrow
//! on a photograph reads it as "*this* content moves *that* way", which is only
//! true when the tail is on the real pixel.
//!
//! Tailing every arrow at its own grid node has a second benefit: the tails sit
//! on an exact regular lattice instead of the slightly warped one the ideal
//! positions form, so the field reads as a field.
//!
//! # The exaggeration, and why it is honest
//!
//! A real lens displaces a handful of pixels over a frame hundreds of pixels
//! across, so an unexaggerated field is invisible. [`auto_scale`] picks the
//! smallest multiplier from [`crate::state::IntrinsicsDisplaySettings::SCALE_LADDER`]
//! that brings the largest displacement up to [`MIN_ARROW_PX`] on screen, capped
//! so that no arrow outgrows its own grid cell. The legend states the true
//! maximum and the multiplier on every frame: an exaggerated field that does not
//! admit it is a lie, and this is a diagnostic tool.
//!
//! # Only the trustworthy half of the grid is drawn as arrows
//!
//! The auto scale fits the largest displacement in the grid, and on a circular
//! fisheye the largest displacement in the grid is not a lens at all:
//! `kerry_park`'s image rectangle has corners 150° off-axis, outside the lens's
//! image circle, where the `k1..k4` polynomial folds and reports 273 px of
//! "distortion" against the 13 px the lens actually applies. Fitting the scale
//! to that picks ×1 and makes every real arrow invisible, and the legend's
//! `max N px` would be quoting the artefact.
//!
//! So the field is split by [`super::Arrow::trusted`], the flag
//! [`sfmtool_core::camera::report::trustworthy_max_theta_deg`] and
//! `DistortionSample::theta_deg` exist to produce, and the two halves are drawn
//! differently in kind rather than in degree:
//!
//! - **inside the bound** — an arrow, scaled and counted, and the maximum the
//!   legend quotes;
//! - **outside it** — a small open dot at the grid node and nothing else. The
//!   node was sampled and there is no measurement there, which is a different
//!   statement from "the lens displaces this ray by 240 pixels". Drawing those
//!   arrows at the trustworthy scale would also throw a dozen frame-crossing
//!   strokes across the picture.
//!
//! The plot solved the same problem for a curve by shading, dotting and
//! excluding from the range. A field is not a curve: there is no continuous
//! path to dot, and the region is a ring around the outside of the frame rather
//! than a tail. What carries over is the principle — the extrapolated part is
//! *visible*, *distinguished in kind*, and *out of every number* — and the
//! boundary itself is drawn, by [`super::axes`], as a labelled dashed contour.

use egui::{Color32, Pos2, Rect};

use super::{ink, Arrow, CameraLayer, View, DEGREE, MIDDOT, REFERENCE_ALPHA};
use crate::state::IntrinsicsDisplaySettings;

/// How long the largest arrow should be on screen, in panel pixels, before the
/// auto scale is satisfied.
const MIN_ARROW_PX: f64 = 8.0;

/// Arrowhead length as a fraction of the arrow's own drawn length, and the
/// longest it may get in panel pixels.
const HEAD: (f32, f32) = (0.35, 5.0);

/// Half-angle of the arrowhead's two barbs.
const HEAD_ANGLE: f32 = 0.42;

/// Below this drawn length, in panel pixels, an arrow gets no head: two barbs
/// on a three-pixel shaft render as a blob rather than as an arrow, which is
/// what the first draft put across the middle of `kerry_park` where the lens
/// displaces almost nothing.
const HEAD_FLOOR: f32 = 3.0;

/// And below this, no arrow at all. A sub-pixel stroke at the centre of the
/// frame is not a small measurement, it is the absence of one.
const ARROW_FLOOR: f32 = 0.75;

/// Radius of the marker left at a grid node outside the trustworthy bound.
const EXTRAPOLATED_DOT: f32 = 1.6;

/// The exaggeration to draw the field at, given the largest **trustworthy**
/// displacement and the geometry it sits in.
///
/// Two rules, in order:
///
/// 1. the smallest ladder value that brings `max_px` to [`MIN_ARROW_PX`] on
///    screen, so the field is legible at all;
/// 2. capped so no arrow exceeds one grid cell, so the field still reads as a
///    field rather than as a tangle.
///
/// The screen length is measured at the panel's **fit** scale, not at the
/// current zoom. The spec asks for both "at least 8 panel pixels" and "computed
/// per camera, not per frame", and those two are in tension: panel pixels depend
/// on the zoom, so a scale honestly recomputed against the live view would step
/// down the ladder as the user zoomed in — the flicker the spec is ruling out.
/// Fixing it at the fit scale resolves it in the direction the spec asks for:
/// the multiplier changes only when the camera or the panel size changes, never
/// while panning or zooming, and zooming in makes the arrows bigger along with
/// everything else in the image.
pub(super) fn auto_scale(max_px: f64, cell_px: f64, fit_scale: f32) -> f32 {
    let ladder = IntrinsicsDisplaySettings::SCALE_LADDER;
    if !max_px.is_finite() || max_px <= 0.0 {
        return ladder[0];
    }
    let on_screen = |scale: f32| f64::from(scale) * max_px * f64::from(fit_scale.max(1e-3));
    let legible = ladder
        .iter()
        .copied()
        .find(|scale| on_screen(*scale) >= MIN_ARROW_PX)
        .unwrap_or(ladder[ladder.len() - 1]);
    // The cap is in image pixels on both sides, so unlike the floor it does not
    // depend on the view at all.
    let capped = ladder
        .iter()
        .copied()
        .rfind(|scale| f64::from(*scale) * max_px <= cell_px)
        .unwrap_or(ladder[0]);
    legible.min(capped).max(ladder[0])
}

/// Draw the field, and return the legend lines describing it.
pub(super) fn draw(
    painter: &egui::Painter,
    layer: &CameraLayer,
    view: &View,
    panel: Rect,
    scale: f32,
) -> Vec<String> {
    let reference = ink(REFERENCE_ALPHA);
    for arrow in &layer.arrows {
        if arrow.trusted {
            draw_arrow(painter, arrow, view, panel, scale, reference);
        } else {
            draw_extrapolated_node(painter, arrow, view, panel, reference);
        }
    }
    legend(layer, scale)
}

/// One displacement arrow, from the model's pixel toward the ideal map's — the
/// correction, drawn on the pixel it corrects. See this module's own docs for
/// why the tail is the real pixel.
fn draw_arrow(
    painter: &egui::Painter,
    arrow: &Arrow,
    view: &View,
    panel: Rect,
    scale: f32,
    color: Color32,
) {
    let tail = view.at(arrow.pixel);
    let head = Pos2::new(
        tail.x + (arrow.reference[0] - arrow.pixel[0]) as f32 * scale * view.scale,
        tail.y + (arrow.reference[1] - arrow.pixel[1]) as f32 * scale * view.scale,
    );
    let grown = panel.expand(20.0);
    if !grown.contains(tail) && !grown.contains(head) {
        return;
    }
    let length = (head - tail).length();
    if length < ARROW_FLOOR {
        return;
    }
    super::haloed_segment(painter, tail, head, 1.0, color);
    if length < HEAD_FLOOR {
        return;
    }
    let barb = (length * HEAD.0).min(HEAD.1);
    let direction = (head - tail) / length;
    let (sin, cos) = HEAD_ANGLE.sin_cos();
    for sign in [1.0_f32, -1.0] {
        let back = egui::vec2(
            -(direction.x * cos - sign * direction.y * sin),
            -(sign * direction.x * sin + direction.y * cos),
        );
        super::haloed_segment(painter, head, head + back * barb, 1.0, color);
    }
}

/// The marker left where the model is extrapolating: the node was sampled, and
/// there is no measurement to draw there.
fn draw_extrapolated_node(
    painter: &egui::Painter,
    arrow: &Arrow,
    view: &View,
    panel: Rect,
    color: Color32,
) {
    let at = view.at(arrow.pixel);
    if !panel.contains(at) {
        return;
    }
    painter.circle_filled(at, EXTRAPOLATED_DOT + super::HALO_WIDTH / 2.0, super::HALO);
    painter.circle_filled(at, EXTRAPOLATED_DOT, color);
}

/// What the field says about itself, for the layer's legend.
fn legend(layer: &CameraLayer, scale: f32) -> Vec<String> {
    let mut lines = vec![format!(
        "distortion max {:.1} px {MIDDOT} shown {}{}",
        layer.max_px,
        super::controls::TIMES,
        super::controls::scale_text(scale)
    )];
    if let (Some(limit), (outside, total)) = (layer.limit_deg, layer.extrapolated) {
        if outside > 0 {
            lines.push(format!(
                "{outside} of {total} nodes past {limit:.1}{DEGREE} {MIDDOT} marked, not measured"
            ));
        }
    }
    lines
}
