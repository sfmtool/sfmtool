// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench layer: the active track of the node's bench, drawn over the image
//! in the detail panel, and edited on it.
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
//! - At the **track stage** the patch's square boundary is sampled and each
//!   sample projected through the camera, so the outline is the curve a
//!   distorting lens really maps that square to. Beside it, each observation's
//!   keypoint and, for **every** observation and in that observation's own
//!   verdict colour, the segment from the keypoint to the patch's own
//!   projection, which is the projection offset the *Proj. off* column
//!   reports. Where a sighting sits on the projection the segment has no
//!   length and is not seen, which is the answer as much as a long one is.
//! - At the **cluster stage** there is no geometry, so what is drawn is the
//!   parallelogram the template's square maps to under the observation's
//!   refined affine shape, with the seed's own parallelogram dashed behind it.
//!   The two apart are how far the refinement moved and how much it turned.
//!
//! **What it draws, it edits.** At the track stage every handle edits the one
//! patch and each photograph shows where it lands: the dot slides it across
//! its own plane, an edge resizes it, a corner turns it. At the cluster stage
//! there is no shared geometry, so each handle is that sighting's own -- the
//! dot moves its seed and the outline is its own affine shape. Each drag
//! previews by drawing the track the release would produce, and the release is
//! one version. The geometry a pointer is read against is
//! [`crate::bench::geometry`], which the wire's patch tools read it against
//! too.

use egui::{Color32, CursorIcon, Pos2, Rect, Shape, Stroke, Vec2};
use sfmtool_core::bench::{EditableTrack, Observation, Stage, Verdict};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::ImageTable;

use crate::bench::geometry::{self, PatchEdit};
use crate::bench::{distance_to_segment, resize_cursor, verdict_color};
use sfmtool_core::bench::Edge;
use sfmtool_core::EditedReconstruction;

use super::ImageDetailResponse;

/// Stroke width of the bench's outlines. Thicker than a feature ellipse's, so
/// the layer reads as being on top of the overlay rather than part of it.
const STROKE_WIDTH: f32 = 2.5;

/// Radius of the mark drawn at an observation's own position, in panel px.
const KEYPOINT_RADIUS: f32 = 4.0;

/// How far from a dot or a corner the pointer still grabs it, in panel px.
///
/// Generous against the 4 px dot it draws, because the cost of the two misses
/// is not symmetric: a handle missed by two pixels pans the photograph instead,
/// which is a gesture the person then has to undo by eye, while a handle caught
/// a little early is released with no motion and does nothing.
const HANDLE_HIT_RADIUS: f32 = 9.0;

/// How far from an edge's polyline the pointer still grabs it, in panel px.
///
/// Narrower than a corner's reach, and tested after the corners, so the corner
/// where two edges meet turns rather than resizing whichever edge won the
/// distance.
const EDGE_HIT_WIDTH: f32 = 8.0;

/// Radius of the arc glyph drawn beside a hovered corner, in panel px.
const TURN_GLYPH_RADIUS: f32 = 9.0;

/// Samples per edge of the projected patch boundary, before the size of the
/// projection is known.
const BASE_SAMPLES: usize = 8;

/// The most samples per edge, for a patch that fills the panel.
const MAX_SAMPLES: usize = 64;

/// The four `(s, t)` corners of a patch's square, in the order
/// [`OrientedPatch::boundary`] walks them, so consecutive pairs are its edges.
const CORNERS: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// What the pointer has hold of.
///
/// Every handle names an observation: the sighting a dot moves, and, for the
/// outline, the sighting whose place the outline is drawn at -- the frame is
/// re-anchored on it at the track stage and is its own shape at the cluster
/// stage, so the pointer is read against the square that observation is looking
/// at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Handle {
    /// The observation's own mark: dragging it places the sighting.
    Keypoint {
        /// The observation, by its position in the track's list.
        observation: usize,
    },
    /// One edge of the outline: dragging it resizes the patch, holding the
    /// opposite edge still.
    Edge {
        /// The observation the outline is drawn at.
        observation: usize,
        /// Which edge of the patch's square it is.
        edge: Edge,
    },
    /// One corner of the outline: dragging it turns the patch.
    Corner {
        /// The observation the outline is drawn at.
        observation: usize,
        /// Which corner, as an index into [`CORNERS`].
        corner: usize,
    },
}

/// A handle being dragged, and where the pointer has taken it.
///
/// Both pixels are in the **source image's** own coordinates, so a pan or a
/// zoom during the drag moves the outline with the photograph rather than
/// dragging the patch out from under the pointer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Drag {
    /// The image the drag started in. A drag that outlives a switch of image is
    /// dropped: the handle names a place in a photograph that is no longer up.
    pub(super) image: usize,
    /// What is being dragged.
    pub(super) handle: Handle,
    /// Where the pointer pressed, in source-image px.
    pub(super) from: [f64; 2],
    /// Where the pointer is now, in source-image px.
    pub(super) to: [f64; 2],
    /// Whether the pointer has left the press at all.
    ///
    /// A press is taken as a handle before egui would call it a drag, so this
    /// is what tells the two apart at the release: a press that never moved is
    /// a click, which selects the row under it and edits nothing.
    pub(super) moved: bool,
    /// Set by Escape: the gesture is abandoned, so nothing is previewed and
    /// nothing is pushed, but the view still does not pan until the button
    /// comes up.
    pub(super) cancelled: bool,
}

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

/// The layer's geometry for one image, in panel coordinates: what is drawn, and
/// what the pointer can take hold of.
///
/// Built once per frame, before the view consumes the drag, and read twice: the
/// hit test runs off it in front of the pan, and the paint runs off the copy
/// built from the previewed track.
pub(super) struct Layer {
    /// One per observation of the track in this image.
    sightings: Vec<Sighting>,
    /// The outlines: one at the track stage, one per observation at the cluster
    /// stage.
    outlines: Vec<Outline>,
    /// The patch's own projection, which every observation's
    /// projection-offset segment runs to. `None` at the cluster stage and for
    /// a track nothing has triangulated.
    center: Option<Pos2>,
    /// The seed parallelograms drawn dashed behind the cluster stage's
    /// outlines. Never a handle: where an observation *started* is not a thing
    /// to drag.
    seeds: Vec<(Vec<Pos2>, Color32)>,
}

/// One observation's own mark.
struct Sighting {
    observation: usize,
    verdict: Verdict,
    at: Pos2,
}

/// One patch outline, as the boundary samples that landed.
struct Outline {
    /// The observation it is drawn at.
    observation: usize,
    /// The verdict its colour comes from.
    verdict: Verdict,
    /// The `4 * per_edge` samples in boundary order, `None` for one that did
    /// not project.
    samples: Vec<Option<Pos2>>,
    /// Samples per edge, so corner `k` is sample `k * per_edge`.
    per_edge: usize,
}

impl Outline {
    /// Corner `k`, when it projected.
    fn corner(&self, k: usize) -> Option<Pos2> {
        self.samples.get(k * self.per_edge).copied().flatten()
    }

    /// Edge `k`, from corner `k` to corner `k + 1`, with the samples that did
    /// not project dropped.
    fn edge(&self, k: usize) -> Vec<Pos2> {
        let n = self.samples.len();
        (0..=self.per_edge)
            .filter_map(|i| self.samples[(k * self.per_edge + i) % n])
            .collect()
    }

    /// The runs of consecutive samples that landed, and whether the boundary
    /// closed.
    ///
    /// A sample behind the camera or outside the lens model's domain ends a run
    /// and starts the next, so an outline that leaves the model's field is
    /// drawn as the arcs that are defined rather than closed across the gap
    /// with a chord that means nothing.
    fn runs(&self) -> (Vec<Vec<Pos2>>, bool) {
        let n = self.samples.len();
        let Some(gap) = (0..n).find(|&i| self.samples[i].is_none()) else {
            return (vec![self.samples.iter().flatten().copied().collect()], true);
        };
        // Walk from the sample after the first gap, so a run that spans the
        // point the boundary happens to start at is one run rather than two.
        let mut runs: Vec<Vec<Pos2>> = Vec::new();
        let mut run: Vec<Pos2> = Vec::new();
        for step in 1..=n {
            match self.samples[(gap + step) % n] {
                Some(point) => run.push(point),
                None if run.len() > 1 => runs.push(std::mem::take(&mut run)),
                None => run.clear(),
            }
        }
        if run.len() > 1 {
            runs.push(run);
        }
        (runs, false)
    }

    /// The box every sample that landed sits in.
    fn bounds(&self) -> Option<Rect> {
        self.samples.iter().flatten().fold(None, |bounds, point| {
            Some(bounds.map_or(Rect::from_pos(*point), |r: Rect| {
                r.union(Rect::from_pos(*point))
            }))
        })
    }
}

impl Layer {
    /// The layer's geometry for `track` in `img_idx`, or `None` for an image
    /// the track does not observe.
    ///
    /// Nothing is built for such an image: the track is a set of sightings, and
    /// a photograph it has none in has nothing of it to show.
    pub(super) fn build(
        image_table: &ImageTable,
        img_idx: usize,
        track: &EditableTrack,
        image_rect: Rect,
        effective_scale: f32,
    ) -> Option<Self> {
        let here: Vec<(usize, &Observation)> = track
            .observations
            .iter()
            .enumerate()
            .filter(|(_, o)| o.image as usize == img_idx)
            .collect();
        if here.is_empty() {
            return None;
        }
        let to_panel = |p: [f64; 2]| -> Pos2 {
            Pos2::new(
                image_rect.min.x + p[0] as f32 * effective_scale,
                image_rect.min.y + p[1] as f32 * effective_scale,
            )
        };
        let mut layer = Layer {
            sightings: Vec::new(),
            outlines: Vec::new(),
            center: None,
            seeds: Vec::new(),
        };
        for (index, observation) in &here {
            if let Some(site) = observation.site() {
                layer.sightings.push(Sighting {
                    observation: *index,
                    verdict: observation.verdict,
                    at: to_panel(site),
                });
            }
        }
        match &track.stage {
            Stage::Track(payload) => {
                let view = geometry::view_of(image_table, img_idx);
                layer.build_track_stage(
                    &here,
                    payload.placement.as_ref(),
                    view.as_ref(),
                    &to_panel,
                );
            }
            // The cluster's own radius, which it carries from the moment it is
            // started: a seed draws at the size it was named before anything
            // has evaluated it, and an evaluation that finds the same scale
            // leaves the outline where it is.
            Stage::Cluster(payload) => layer.build_cluster_stage(&here, payload.radius, &to_panel),
        }
        Some(layer)
    }

    /// The track stage: the patch's outline once, anchored on the sighting the
    /// strongest verdict names.
    ///
    /// One outline for the image rather than one per observation, because there
    /// is one patch: two candidates in a photograph are two readings of where
    /// it lands, not two squares. Its colour is the strongest verdict among
    /// them, so an image the track is `in` reads as `in`. It is drawn through
    /// the frame re-anchored on that sighting, as the tile is rendered: the
    /// sighting is where the patch sits in this photograph, and the geometric
    /// projection is where the 3D says it should, which the hollow centre and
    /// the projection-offset segment show separately.
    fn build_track_stage(
        &mut self,
        here: &[(usize, &Observation)],
        frame: Option<&OrientedPatch>,
        view: Option<&(CameraIntrinsics, RigidTransform)>,
        to_panel: &impl Fn([f64; 2]) -> Pos2,
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
        let anchor = here
            .iter()
            .find(|(_, o)| o.verdict == strongest)
            .or_else(|| here.first());
        let (Some((index, observation)), Some((camera, pose))) = (anchor, view) else {
            return;
        };
        self.center = frame
            .and_then(|frame| geometry::project(camera, pose, frame.center.coords, frame.w))
            .map(to_panel);
        let Some(frame) = frame else {
            return;
        };
        let anchored = geometry::anchored_frame(frame, camera, pose, observation);
        let (samples, per_edge) = project_outline(&anchored, camera, pose);
        self.outlines.push(Outline {
            observation: *index,
            verdict: strongest,
            samples: samples.into_iter().map(|p| p.map(to_panel)).collect(),
            per_edge,
        });
    }

    /// The cluster stage: per observation, the template's square under the
    /// refined shape, with the seed's own square dashed behind it.
    ///
    /// `radius` is the cluster's template half-width in keypoint-frame units,
    /// which is the only thing that says how large these shapes are: a shape
    /// maps one such unit to pixels and the patch is `[-radius, radius]` of
    /// them.
    fn build_cluster_stage(
        &mut self,
        here: &[(usize, &Observation)],
        radius: f64,
        to_panel: &impl Fn([f64; 2]) -> Pos2,
    ) {
        for (index, observation) in here {
            let Some(measurement) = observation.cluster.as_ref() else {
                continue;
            };
            // Behind, and dashed, because it is where the observation started
            // rather than what it is: a refinement that moved far is a
            // refinement that should be looked at.
            if measurement.position.is_some() || measurement.shape.is_some() {
                self.seeds.push((
                    parallelogram(
                        measurement.seed_position,
                        measurement.seed_shape,
                        radius,
                        to_panel,
                    ),
                    verdict_color(observation.verdict).gamma_multiply(0.7),
                ));
            }
            let corners = parallelogram(
                measurement.best_position(),
                measurement.shape.unwrap_or(measurement.seed_shape),
                radius,
                to_panel,
            );
            self.outlines.push(Outline {
                observation: *index,
                verdict: observation.verdict,
                samples: corners.into_iter().map(Some).collect(),
                per_edge: 1,
            });
        }
    }

    /// The handle nearest `pos`, or `None` when the pointer is on none of them.
    ///
    /// Dots first, then corners, then edges: the dot sits inside the outline it
    /// anchors, and the corner is where two edges meet, so a nearest-thing
    /// search over all three at once would make the two smaller handles
    /// unreachable.
    pub(super) fn hit(&self, pos: Pos2) -> Option<Handle> {
        let nearest = |best: Option<(f32, Handle)>, distance: f32, handle: Handle| match best {
            Some((d, _)) if d <= distance => best,
            _ => Some((distance, handle)),
        };
        let mut found = None;
        for sighting in &self.sightings {
            let distance = (sighting.at - pos).length();
            if distance <= HANDLE_HIT_RADIUS {
                found = nearest(
                    found,
                    distance,
                    Handle::Keypoint {
                        observation: sighting.observation,
                    },
                );
            }
        }
        if let Some((_, handle)) = found {
            return Some(handle);
        }
        for outline in &self.outlines {
            for corner in 0..4 {
                let Some(at) = outline.corner(corner) else {
                    continue;
                };
                let distance = (at - pos).length();
                if distance <= HANDLE_HIT_RADIUS {
                    found = nearest(
                        found,
                        distance,
                        Handle::Corner {
                            observation: outline.observation,
                            corner,
                        },
                    );
                }
            }
        }
        if let Some((_, handle)) = found {
            return Some(handle);
        }
        for outline in &self.outlines {
            for edge in 0..4 {
                let Some(distance) = distance_to_polyline(&outline.edge(edge), pos) else {
                    continue;
                };
                if distance <= EDGE_HIT_WIDTH {
                    found = nearest(
                        found,
                        distance,
                        Handle::Edge {
                            observation: outline.observation,
                            edge: geometry::edge_of(edge),
                        },
                    );
                }
            }
        }
        found.map(|(_, handle)| handle)
    }

    /// The cursor `handle` asks for, with the outline's own orientation on
    /// screen deciding which resize cursor an edge takes.
    ///
    /// An edge that looks horizontal is one you move up and down, so it takes
    /// the vertical resize cursor, and an oblique edge takes the diagonal whose
    /// slope it has. A corner turns, and egui has no cursor for that, so it
    /// takes [`CursorIcon::Alias`] and the arc glyph says the rest.
    fn cursor(&self, handle: Handle) -> CursorIcon {
        match handle {
            Handle::Keypoint { .. } => CursorIcon::Move,
            Handle::Corner { .. } => CursorIcon::Alias,
            Handle::Edge { observation, edge } => self
                .outlines
                .iter()
                .find(|outline| outline.observation == observation)
                .and_then(|outline| {
                    let k = (0..4).find(|k| geometry::edge_of(*k) == edge)?;
                    let points = outline.edge(k);
                    let (first, last) = (points.first()?, points.last()?);
                    Some(resize_cursor(*last - *first))
                })
                .unwrap_or(CursorIcon::Move),
        }
    }

    /// Paint the layer.
    fn paint(&self, painter: &egui::Painter, hovered: Option<Handle>) {
        for (seed, color) in &self.seeds {
            let mut closed = seed.clone();
            if let Some(first) = seed.first() {
                closed.push(*first);
            }
            painter.extend(Shape::dashed_line(
                &closed,
                Stroke::new(1.5, *color),
                4.0,
                4.0,
            ));
        }
        for outline in &self.outlines {
            let stroke = Stroke::new(STROKE_WIDTH, verdict_color(outline.verdict));
            let (runs, closed) = outline.runs();
            for run in runs {
                if run.len() < 2 {
                    continue;
                }
                if closed {
                    painter.add(Shape::closed_line(run, stroke));
                } else {
                    painter.add(Shape::line(run, stroke));
                }
            }
        }
        for sighting in &self.sightings {
            let color = verdict_color(sighting.verdict);
            painter.circle_filled(sighting.at, KEYPOINT_RADIUS, color);
            // The projection offset, drawn rather than tabulated, and drawn
            // for every observation whatever its verdict: where this image's
            // feature sits relative to where the patch says it should is the
            // question the layer exists to answer, and an observation already
            // judged in is exactly the one whose answer is worth seeing. A
            // sighting that sits on the projection draws a segment of no
            // length, so nothing needs special casing for the agreeing case.
            if let Some(center) = self.center {
                painter.line_segment([sighting.at, center], Stroke::new(1.5, color));
                painter.circle_stroke(center, KEYPOINT_RADIUS * 0.75, Stroke::new(1.5, color));
            }
        }
        // The one mark that is about the pointer rather than about the track:
        // a corner says nothing about turning by looking like a corner, so
        // while one is under the pointer it wears an arc.
        if let Some(Handle::Corner {
            observation,
            corner,
        }) = hovered
        {
            if let Some(outline) = self
                .outlines
                .iter()
                .find(|outline| outline.observation == observation)
            {
                if let Some(at) = outline.corner(corner) {
                    let away = outline
                        .bounds()
                        .map(|bounds| (at - bounds.center()).normalized())
                        .unwrap_or(Vec2::new(1.0, 0.0));
                    draw_turn_glyph(
                        painter,
                        at + away * TURN_GLYPH_RADIUS,
                        verdict_color(outline.verdict),
                    );
                }
            }
        }
    }

    /// The marks a click selects a Track Edit row by: each sighting, with the
    /// outline it belongs to as its reach.
    fn marks(&self) -> Vec<Mark> {
        let bounds =
            self.outlines
                .iter()
                .fold(None, |bounds, outline| match (bounds, outline.bounds()) {
                    (Some(a), Some(b)) => Some(Rect::union(a, b)),
                    (a, b) => a.or(b),
                });
        self.sightings
            .iter()
            .map(|sighting| Mark {
                observation: sighting.observation,
                rect: bounds
                    .unwrap_or_else(|| Rect::from_center_size(sighting.at, Vec2::splat(0.0)))
                    .expand(KEYPOINT_RADIUS),
                anchor: sighting.at,
            })
            .collect()
    }

    /// What this drag would do to the track, in the form the core steps take.
    ///
    /// Two of the three gestures are already in those terms: a dot drag and an
    /// edge drag both name a pixel, and the arithmetic behind the edge -- the
    /// opposite edge held still -- is `sfmtool_core::bench::resize_patch_to_pixel`'s
    /// rather than this panel's, so a tool call and a drag cannot resize
    /// differently. What is left here is the turn, which is a reading of two
    /// pointer positions against the patch and has no other home.
    ///
    /// `None` when the pointer names nothing the patch can be given: a ray that
    /// misses its plane, or a turn about the centre itself.
    pub(super) fn edit(
        image_table: &ImageTable,
        track: &EditableTrack,
        drag: &Drag,
    ) -> Option<PatchEdit> {
        if drag.cancelled {
            return None;
        }
        let observation = match drag.handle {
            Handle::Keypoint { observation }
            | Handle::Edge { observation, .. }
            | Handle::Corner { observation, .. } => observation,
        };
        match drag.handle {
            // The dot means different things at the two stages, because the two
            // stages have different things to move: a track-stage track has one
            // patch and every sighting is a view of it, so dragging the mark
            // slides the **patch** and every sighting follows; a cluster has no
            // shared geometry at all, so the mark is that sighting's own seed
            // and nothing else moves.
            Handle::Keypoint { .. } => Some(match track.stage {
                Stage::Track(_) => PatchEdit::TranslateToPixel {
                    observation,
                    pixel: drag.to,
                },
                Stage::Cluster(_) => PatchEdit::Sight {
                    observation,
                    pixel: drag.to,
                },
            }),
            Handle::Edge { edge, .. } => Some(PatchEdit::ResizeToPixel {
                observation,
                edge,
                pixel: drag.to,
            }),
            Handle::Corner { .. } => {
                let sighting = track.observations.get(observation)?;
                match &track.stage {
                    Stage::Track(payload) => {
                        let frame = payload.placement.as_ref()?;
                        let (camera, pose) = geometry::view_of(image_table, drag.image)?;
                        let anchored = geometry::anchored_frame(frame, &camera, &pose, sighting);
                        let angle_rad =
                            geometry::turn_between(&anchored, &camera, &pose, drag.from, drag.to)?;
                        Some(PatchEdit::Spin { angle_rad })
                    }
                    Stage::Cluster(_) => {
                        let angle_rad =
                            geometry::pixel_turn_between(sighting.site()?, drag.from, drag.to)?;
                        Some(PatchEdit::SpinShape {
                            observation,
                            angle_rad,
                        })
                    }
                }
            }
        }
    }
}

/// Draw the active bench track in `img_idx`, and report a click on one of its
/// marks.
///
/// `drag` is the handle the pointer has hold of, if any: the layer is drawn
/// from the track that drag would produce, so what a person sees mid-gesture is
/// what releasing would leave behind.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw(
    painter: &egui::Painter,
    ui: &egui::Ui,
    interact_response: &egui::Response,
    edited: &EditedReconstruction,
    img_idx: usize,
    track: &EditableTrack,
    image_rect: Rect,
    effective_scale: f32,
    drag: Option<&Drag>,
    hovered: Option<Handle>,
    response: &mut ImageDetailResponse,
) {
    // The preview: the track the release would push, drawn instead of the one
    // on the bench. Through the same function the release goes through, so
    // there is one answer rather than a drawn guess and a pushed result.
    let image_table = &edited.base.image_table;
    let previewed = drag
        .filter(|drag| drag.image == img_idx && !drag.cancelled)
        .and_then(|drag| Layer::edit(image_table, track, drag))
        .and_then(|edit| geometry::apply(track, edited, &edit).ok())
        .map(|(next, _)| next);
    let shown = previewed.as_ref().unwrap_or(track);

    let Some(layer) = Layer::build(image_table, img_idx, shown, image_rect, effective_scale) else {
        return;
    };
    layer.paint(painter, hovered);
    if let Some(handle) = hovered {
        ui.ctx().set_cursor_icon(match drag {
            Some(drag) if !drag.cancelled => match drag.handle {
                Handle::Keypoint { .. } => CursorIcon::Grabbing,
                other => layer.cursor(other),
            },
            _ => layer.cursor(handle),
        });
    }

    // The layer is on top, so a click one of its marks catches does not also
    // reach the features under it: two selections from one click would be two
    // answers to one gesture.
    if interact_response.clicked() {
        if let Some(pos) = interact_response.interact_pointer_pos() {
            let marks = layer.marks();
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

/// The four panel-space corners of the square `[-radius, radius]^2` mapped
/// through `shape` at `position`, in the same `(s, t)` order the patch's
/// boundary walks.
fn parallelogram(
    position: [f64; 2],
    shape: [[f64; 2]; 2],
    radius: f64,
    to_panel: &impl Fn([f64; 2]) -> Pos2,
) -> Vec<Pos2> {
    CORNERS
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

/// The patch's boundary projected into the view, as one sample per boundary
/// point with `None` where it did not land, and how many samples each edge got.
///
/// The density follows the size of the projection: the corners are projected
/// first to measure it, and the edges then sampled finely enough that the
/// distortion shows as a curve rather than as a polygon.
fn project_outline(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
) -> (Vec<Option<[f64; 2]>>, usize) {
    let corners: Vec<[f64; 2]> = frame
        .boundary(1)
        .into_iter()
        .filter_map(|p| geometry::project(camera, pose, p.coords, frame.w))
        .collect();
    let span = corners.iter().fold(0.0f64, |span, a| {
        corners.iter().fold(span, |span, b| {
            span.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs())
        })
    });
    // One sample per dozen source pixels of the widest side, which is finer
    // than the eye can tell a chord from an arc at any zoom the panel offers.
    let samples = ((span / 12.0).ceil() as usize).clamp(BASE_SAMPLES, MAX_SAMPLES);
    let projected = frame
        .boundary(samples)
        .iter()
        .map(|point| geometry::project(camera, pose, point.coords, frame.w))
        .collect();
    (projected, samples)
}

/// The distance from `pos` to the polyline `points`, or `None` for a polyline
/// of fewer than two points.
fn distance_to_polyline(points: &[Pos2], pos: Pos2) -> Option<f32> {
    if points.len() < 2 {
        return None;
    }
    points
        .windows(2)
        .map(|segment| distance_to_segment(segment[0], segment[1], pos))
        .fold(None, |best: Option<f32>, d| {
            Some(best.map_or(d, |best| best.min(d)))
        })
}

/// A small arc with an arrowhead, drawn beside a hovered corner to say that
/// dragging it turns the patch.
fn draw_turn_glyph(painter: &egui::Painter, at: Pos2, color: Color32) {
    const FROM: f32 = -0.6 * std::f32::consts::PI;
    const TO: f32 = 0.85 * std::f32::consts::PI;
    const STEPS: usize = 12;
    let point = |angle: f32| {
        Pos2::new(
            at.x + TURN_GLYPH_RADIUS * 0.62 * angle.cos(),
            at.y + TURN_GLYPH_RADIUS * 0.62 * angle.sin(),
        )
    };
    let arc: Vec<Pos2> = (0..=STEPS)
        .map(|i| point(FROM + (TO - FROM) * i as f32 / STEPS as f32))
        .collect();
    let stroke = Stroke::new(1.5, color);
    painter.add(Shape::line(arc, stroke));
    // The head, two short strokes off the arc's end along its tangent.
    let tip = point(TO);
    let tangent = Vec2::new(-TO.sin(), TO.cos());
    let outward = (tip - at).normalized();
    let head = TURN_GLYPH_RADIUS * 0.34;
    painter.line_segment([tip, tip - tangent * head + outward * head * 0.6], stroke);
    painter.line_segment([tip, tip - tangent * head - outward * head * 0.6], stroke);
}
