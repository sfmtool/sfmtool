// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench layer: the active track of the node's bench, drawn over the image
//! in the detail panel.
//!
//! It is a layer rather than an overlay mode, for the reason the intrinsics
//! layer is: it says something about a different thing from the feature
//! overlays (the bench, not the reconstruction), so it composes with whichever
//! mode is active and draws after all of them. It is also the only thing in
//! this panel drawn in the bench's own colours, so a mark on the bench is never
//! read as committed structure.
//!
//! What it draws is the track's own geometry rather than a symbol for it:
//!
//! - At the **track stage** the surfel's square boundary is sampled and each
//!   sample projected through the camera, so the outline is the curve a
//!   distorting lens really maps that square to. Beside it, each observation's
//!   keypoint, and for a candidate the segment from the keypoint to the
//!   surfel's own projection, which is the shift the *Shift* column reports.
//! - At the **cluster stage** there is no geometry, so what is drawn is the
//!   parallelogram the template's square maps to under the observation's
//!   refined affine shape, with the seed's own parallelogram dashed behind it.
//!   The two apart are how far the refinement moved and how much it turned.

use egui::{Color32, Pos2, Rect, Shape, Stroke};
use nalgebra::Vector3;
use sfmtool_core::bench::{EditableTrack, Observation, Stage, Verdict};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::ImageTable;

use super::ImageDetailResponse;

/// The bench's colours, one per verdict, in a hue no feature overlay uses.
///
/// The overlays draw the reconstruction in greens, greys and the colormaps; the
/// bench is violet throughout, so what is committed and what is being
/// considered are never confused at a glance.
pub(super) const IN_COLOR: Color32 = Color32::from_rgb(255, 92, 246);
pub(super) const CANDIDATE_COLOR: Color32 = Color32::from_rgb(170, 140, 255);
pub(super) const OUT_COLOR: Color32 = Color32::from_rgb(130, 104, 150);

/// Stroke width of the bench's outlines. Thicker than a feature ellipse's, so
/// the layer reads as being on top of the overlay rather than part of it.
const STROKE_WIDTH: f32 = 2.5;

/// Radius of the mark drawn at an observation's own position, in panel px.
const KEYPOINT_RADIUS: f32 = 4.0;

/// Samples per edge of the projected surfel boundary, before the size of the
/// projection is known.
const BASE_SAMPLES: usize = 8;

/// The most samples per edge, for a surfel that fills the panel.
const MAX_SAMPLES: usize = 64;

/// What one observation of the track offers a click in this image.
struct Mark {
    /// The row it selects in the Track Edit panel.
    observation: usize,
    /// The outline's bounding box, in panel coordinates.
    rect: Rect,
    /// Where the observation itself sits, which decides between two marks whose
    /// boxes both hold the pointer.
    anchor: Pos2,
}

/// Draw the active bench track in `img_idx`, and report a click on one of its
/// marks.
///
/// Nothing is drawn for an image the track does not observe: the track is a set
/// of sightings, and a photograph it has none in has nothing of it to show.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw(
    painter: &egui::Painter,
    interact_response: &egui::Response,
    image_table: &ImageTable,
    img_idx: usize,
    track: &EditableTrack,
    image_rect: Rect,
    effective_scale: f32,
    response: &mut ImageDetailResponse,
) {
    let here: Vec<(usize, &Observation)> = track
        .observations
        .iter()
        .enumerate()
        .filter(|(_, o)| o.image as usize == img_idx)
        .collect();
    if here.is_empty() {
        return;
    }
    let to_panel = |p: [f64; 2]| -> Pos2 {
        Pos2::new(
            image_rect.min.x + p[0] as f32 * effective_scale,
            image_rect.min.y + p[1] as f32 * effective_scale,
        )
    };

    let mut marks = Vec::new();
    match &track.stage {
        Stage::Track(payload) => {
            let view = view_of(image_table, img_idx);
            draw_track_stage(
                painter,
                &here,
                payload.frame.as_ref(),
                view.as_ref(),
                &to_panel,
                &mut marks,
            );
        }
        Stage::Cluster(payload) => {
            let radius = payload
                .template
                .as_ref()
                .map_or(1.0, |template| template.radius);
            draw_cluster_stage(painter, &here, radius, &to_panel, &mut marks);
        }
    }

    // The layer is on top, so a click one of its marks catches does not also
    // reach the features under it: two selections from one click would be two
    // answers to one gesture.
    if interact_response.clicked() {
        if let Some(pos) = interact_response.interact_pointer_pos() {
            let hit = marks
                .iter()
                .filter(|mark| mark.rect.contains(pos))
                .min_by(|a, b| {
                    (a.anchor - pos)
                        .length_sq()
                        .total_cmp(&(b.anchor - pos).length_sq())
                });
            if let Some(mark) = hit {
                response.select_bench_row = Some(mark.observation);
                response.select_point = None;
            }
        }
    }
}

/// The track stage: the surfel's outline once, and each observation's own mark.
///
/// One outline for the image rather than one per observation, because there is
/// one surfel: two candidates in a photograph are two readings of where it
/// lands, not two squares. Its colour is the strongest verdict among them, so
/// an image the track is `in` reads as `in`.
fn draw_track_stage(
    painter: &egui::Painter,
    here: &[(usize, &Observation)],
    frame: Option<&OrientedPatch>,
    view: Option<&(CameraIntrinsics, RigidTransform)>,
    to_panel: &impl Fn([f64; 2]) -> Pos2,
    marks: &mut Vec<Mark>,
) {
    let strongest = here
        .iter()
        .map(|(_, o)| o.verdict)
        .min_by_key(|v| match v {
            Verdict::In => 0,
            Verdict::Candidate => 1,
            Verdict::Out => 2,
        })
        .unwrap_or(Verdict::Candidate);

    // The surfel's outline, and where its centre lands: both are `None` for a
    // track nothing has triangulated yet, which still has observations to mark.
    let outline = match (frame, view) {
        (Some(frame), Some((camera, pose))) => {
            project_outline(frame, camera, pose).map(|(runs, closed)| {
                let panel: Vec<Vec<Pos2>> = runs
                    .iter()
                    .map(|run| run.iter().map(|p| to_panel(*p)).collect())
                    .collect();
                (panel, closed)
            })
        }
        _ => None,
    };
    let center = match (frame, view) {
        (Some(frame), Some((camera, pose))) => {
            project(camera, pose, frame.center.coords, frame.w).map(to_panel)
        }
        _ => None,
    };

    let mut bounds: Option<Rect> = None;
    if let Some((runs, closed)) = &outline {
        let stroke = Stroke::new(STROKE_WIDTH, color_of(strongest));
        for run in runs {
            if run.len() < 2 {
                continue;
            }
            for point in run {
                bounds = Some(
                    bounds.map_or(Rect::from_pos(*point), |r| r.union(Rect::from_pos(*point))),
                );
            }
            if *closed {
                painter.add(Shape::closed_line(run.clone(), stroke));
            } else {
                painter.add(Shape::line(run.clone(), stroke));
            }
        }
    }

    for (index, observation) in here {
        let color = color_of(observation.verdict);
        let at = observation
            .track
            .as_ref()
            .and_then(|m| m.keypoint)
            .map(|k| to_panel([k[0] as f64, k[1] as f64]))
            .or_else(|| {
                observation
                    .cluster
                    .as_ref()
                    .map(|m| to_panel(m.best_position()))
            });
        let Some(at) = at else {
            continue;
        };
        painter.circle_filled(at, KEYPOINT_RADIUS, color);
        // The shift, drawn rather than tabulated: a candidate is exactly the
        // question of whether the sighting and the surfel are the same thing,
        // and the gap between the two dots is that question.
        if observation.verdict == Verdict::Candidate {
            if let Some(center) = center {
                painter.line_segment([at, center], Stroke::new(1.5, color));
                painter.circle_stroke(center, KEYPOINT_RADIUS * 0.75, Stroke::new(1.5, color));
            }
        }
        marks.push(Mark {
            observation: *index,
            rect: bounds
                .unwrap_or_else(|| Rect::from_center_size(at, egui::Vec2::splat(0.0)))
                .expand(KEYPOINT_RADIUS),
            anchor: at,
        });
    }
}

/// The cluster stage: per observation, the template's square under the refined
/// shape, with the seed's own square dashed behind it.
fn draw_cluster_stage(
    painter: &egui::Painter,
    here: &[(usize, &Observation)],
    radius: f64,
    to_panel: &impl Fn([f64; 2]) -> Pos2,
    marks: &mut Vec<Mark>,
) {
    for (index, observation) in here {
        let Some(measurement) = observation.cluster.as_ref() else {
            continue;
        };
        let color = color_of(observation.verdict);
        // Behind, and dashed, because it is where the observation started
        // rather than what it is: a refinement that moved far is a refinement
        // that should be looked at.
        if measurement.position.is_some() || measurement.shape.is_some() {
            let seed = parallelogram(
                measurement.seed_position,
                measurement.seed_shape,
                radius,
                to_panel,
            );
            let mut closed = seed.clone();
            closed.push(seed[0]);
            painter.extend(Shape::dashed_line(
                &closed,
                Stroke::new(1.5, color.gamma_multiply(0.7)),
                4.0,
                4.0,
            ));
        }
        let corners = parallelogram(
            measurement.best_position(),
            measurement.shape.unwrap_or(measurement.seed_shape),
            radius,
            to_panel,
        );
        painter.add(Shape::closed_line(
            corners.clone(),
            Stroke::new(STROKE_WIDTH, color),
        ));
        let at = to_panel(measurement.best_position());
        painter.circle_filled(at, KEYPOINT_RADIUS, color);
        let mut rect = Rect::from_pos(corners[0]);
        for corner in &corners[1..] {
            rect = rect.union(Rect::from_pos(*corner));
        }
        marks.push(Mark {
            observation: *index,
            rect: rect.expand(KEYPOINT_RADIUS),
            anchor: at,
        });
    }
}

/// The four panel-space corners of the square `[-radius, radius]^2` mapped
/// through `shape` at `position`, in the same `(s, t)` order the surfel's
/// boundary walks.
fn parallelogram(
    position: [f64; 2],
    shape: [[f64; 2]; 2],
    radius: f64,
    to_panel: &impl Fn([f64; 2]) -> Pos2,
) -> Vec<Pos2> {
    [(-1.0f64, -1.0f64), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .into_iter()
        .map(|(s, t)| {
            let (s, t) = (s * radius, t * radius);
            to_panel([
                position[0] + shape[0][0] * s + shape[0][1] * t,
                position[1] + shape[1][0] * s + shape[1][1] * t,
            ])
        })
        .collect()
}

/// The surfel's boundary, projected into the view, as the runs of consecutive
/// samples that landed.
///
/// A sample behind the camera or outside the lens model's domain ends a run and
/// starts the next, so an outline that leaves the model's field is drawn as the
/// arcs that are defined rather than closed across the gap with a chord that
/// means nothing. `None` when fewer than two samples landed at all.
///
/// The density follows the size of the projection: the corners are projected
/// first to measure it, and the edges then sampled finely enough that the
/// distortion shows as a curve rather than as a polygon.
fn project_outline(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
) -> Option<(Vec<Vec<[f64; 2]>>, bool)> {
    let corners: Vec<[f64; 2]> = frame
        .boundary(1)
        .into_iter()
        .filter_map(|p| project(camera, pose, p.coords, frame.w))
        .collect();
    let span = corners.iter().fold(0.0f64, |span, a| {
        corners.iter().fold(span, |span, b| {
            span.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs())
        })
    });
    // One sample per dozen source pixels of the widest side, which is finer
    // than the eye can tell a chord from an arc at any zoom the panel offers.
    let samples = ((span / 12.0).ceil() as usize).clamp(BASE_SAMPLES, MAX_SAMPLES);

    let projected: Vec<Option<[f64; 2]>> = frame
        .boundary(samples)
        .iter()
        .map(|point| project(camera, pose, point.coords, frame.w))
        .collect();
    let n = projected.len();
    // Nothing broke: one closed loop, and the caller joins its ends.
    let Some(gap) = (0..n).find(|&i| projected[i].is_none()) else {
        return Some((vec![projected.into_iter().flatten().collect()], true));
    };
    // Walk from the sample after the first gap, so a run that spans the point
    // the boundary happens to start at is one run rather than two.
    let mut runs: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut run: Vec<[f64; 2]> = Vec::new();
    for step in 1..=n {
        match projected[(gap + step) % n] {
            Some(pixel) => run.push(pixel),
            None if run.len() > 1 => runs.push(std::mem::take(&mut run)),
            None => run.clear(),
        }
    }
    (!runs.is_empty()).then_some((runs, false))
}

/// Project a homogeneous world point into the view, or `None` when it falls
/// behind the camera or outside the lens model's domain.
///
/// The frame test is deliberately absent: an outline whose corner projects a
/// little outside the photograph is an outline with a corner off-screen, which
/// the panel clips, and not a sample that failed to project.
fn project(
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
    xyz: Vector3<f64>,
    w: f64,
) -> Option<[f64; 2]> {
    let pc = pose.transform_point_homogeneous(xyz, w);
    // A point in front of a perspective camera has `z < 0`. A ray-path model
    // images past 90 degrees off axis, where `z >= 0` is a legitimate sighting
    // and the model's own domain is the only oracle.
    if !camera.model.needs_ray_path() && pc.z >= 0.0 {
        return None;
    }
    camera.ray_to_pixel([pc.x, pc.y, pc.z]).map(|(u, v)| [u, v])
}

/// The camera and the pose of one image, or `None` when the image or its camera
/// is not in the table.
fn view_of(image_table: &ImageTable, img_idx: usize) -> Option<(CameraIntrinsics, RigidTransform)> {
    let image = image_table.images.get(img_idx)?;
    let camera = image_table.cameras.get(image.camera_index as usize)?;
    Some((camera.clone(), crate::scene::cam_from_world(image)))
}

/// The colour one verdict is drawn in.
fn color_of(verdict: Verdict) -> Color32 {
    match verdict {
        Verdict::In => IN_COLOR,
        Verdict::Candidate => CANDIDATE_COLOR,
        Verdict::Out => OUT_COLOR,
    }
}
