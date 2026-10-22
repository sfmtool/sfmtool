// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a hover view of a patch's picture draws over it and says under it:
//! the dot where the observation sits, the ring where the point projects, the
//! dashed line between them, and the caption's bullet about the ring.
//!
//! Track View's tile and crop hover views and the Track Table's *Crop* hover
//! view all draw through these, so the three mark one place the same way and
//! say one thing about it.

/// How many times the patch's width a hover view of a tile or a crop shows.
/// For display only: no number is read from the hover view.
///
/// Odd, so that the patch sits in the middle with whole patch widths of the
/// photograph on every side and its box falls on texel boundaries whatever the
/// tile's resolution: at `3` the patch is the middle third of the picture, with
/// one patch width of the photograph on every side.
pub(crate) const CONTEXT_FACTOR: u32 = 3;

/// Side of a hover view's picture on screen, in points: large enough to judge
/// a patch width of photograph on each side of the patch, small enough for a
/// tooltip beside a row.
pub(crate) const CONTEXT_HOVER_SIDE: f32 = 288.0;

/// What the ring of a hover view marks the projection of, which is what Track
/// View's *Proj. err* column measures to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProjectionOf {
    /// The track's triangulated point.
    Point,
    /// The patch's centre, before the track is triangulated.
    PatchCentre,
}

/// Draw the marks a hover view puts over its picture, in screen points: a dot
/// at `keypoint`, where the observation sits, a ring at `projection`, where
/// the track's point or its patch's centre projects, and a dashed line from
/// the one to the other.
///
/// Each mark is a light stroke over a wider dark one, so it reads over a
/// bright photograph and a dark one alike, the ring in amber.
pub(crate) fn paint_marks(
    painter: &egui::Painter,
    keypoint: Option<egui::Pos2>,
    projection: Option<egui::Pos2>,
) {
    let halo = egui::Stroke::new(3.0, egui::Color32::from_black_alpha(170));
    let light = egui::Stroke::new(1.5, egui::Color32::from_gray(240));
    let projection_color = egui::Color32::from_rgb(255, 196, 64);
    if let (Some(keypoint), Some(projection)) = (keypoint, projection) {
        let line = [keypoint, projection];
        for stroke in [halo, light] {
            painter.extend(egui::Shape::dashed_line(&line, stroke, 5.0, 4.0));
        }
    }
    if let Some(at) = projection {
        painter.circle_stroke(at, 5.0, egui::Stroke::new(4.0, halo.color));
        painter.circle_stroke(at, 5.0, egui::Stroke::new(2.0, projection_color));
    }
    if let Some(at) = keypoint {
        painter.circle(at, 3.0, light.color, egui::Stroke::new(1.5, halo.color));
    }
}

/// The bullet a hover view's caption gives its ring: what was projected, and
/// how far from the observation it landed.
///
/// `outside` is whether the ring lies outside the picture, `None` where there
/// is no ring to draw; `projection_px` the distance in the photograph's
/// pixels, `None` where nothing projected; `projection_of` what was projected,
/// `None` at the cluster stage, which has no point.
pub(crate) fn projection_bullet(
    projection_of: Option<ProjectionOf>,
    outside: Option<bool>,
    projection_px: Option<f64>,
) -> String {
    let what = match projection_of {
        None => return "\u{2022} No ring: a cluster has no point to project".to_string(),
        Some(ProjectionOf::Point) => "the track's point",
        Some(ProjectionOf::PatchCentre) => "the patch's centre",
    };
    match (outside, projection_px) {
        (Some(outside), Some(px)) => format!(
            "\u{2022} Ring: where {what} projects, {px:.2} px away along the dashed line{}",
            if outside { ", outside this view" } else { "" }
        ),
        (None, Some(px)) => format!(
            "\u{2022} No ring: {what} projects {px:.2} px away, off the patch's plane as this \
             view sees it",
        ),
        _ => format!("\u{2022} No ring: {what} does not project into this photograph"),
    }
}
