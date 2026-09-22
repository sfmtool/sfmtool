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
//!   Out of the outline's centre stands the patch's **normal**, the 3D
//!   viewport's segment and arrowhead projected into the photograph, so a
//!   photograph taken from the side shows which way the square faces against
//!   the surface it is meant to lie on.
//! - At the **cluster stage** there is no geometry, so what is drawn is the
//!   parallelogram the template's square maps to under the observation's
//!   refined affine shape, with the seed's own parallelogram dashed behind it.
//!   The two apart are how far the refinement moved and how much it turned.
//!
//! In a photograph the track has **no** observation in, the track stage still
//! has something to say: where its patch would be seen. There the patch's own
//! square is projected through that image's camera and drawn as a **ghost
//! outline**, the member outline's stroke at [`GHOST_OPACITY`], so the one
//! patch can be followed across the whole capture without reading as a
//! sighting. With Track View's *Lock* ticked the ghost is a handle as well: a
//! view from another side can make plain where the patch belongs, so it offers
//! the patch-wide handles a member outline does, read against the patch as it
//! stands rather than against a keypoint. See [`Layer::ghost`].
//!
//! **What it draws, it edits.** At the track stage every handle edits the one
//! patch and each photograph shows where it lands: the dot slides it across
//! its own plane, an edge resizes it, a corner turns it, the normal's segment
//! moves it along the normal and the arrowhead turns the normal itself. At the
//! cluster stage there is no shared geometry, so each handle is that
//! sighting's own: the dot moves its seed and the outline is its own affine
//! shape.
//!
//! **Track View's *Lock* decides what the track stage's dot is.** Ticked, which
//! is how it starts, the dot is the patch's, as above. Cleared, the dot is that
//! one sighting's keypoint, and the patch and every other sighting stay where
//! they are: the gesture that fixes a keypoint that settled on the wrong
//! detail. A track-stage sighting has a place of its own and no shape, depth or
//! facing of its own, its size, turn and normal being the patch's, so while the
//! lock is off the outline's edges and corners and the normal's segment and
//! arrowhead are drawn and take no drag, and the ghost is display only. Any of
//! them would move every sighting at once, which is the one thing a cleared
//! lock promises a drag here will not do. Each drag previews by drawing the
//! track the release would produce, and the release is one version. The
//! geometry a pointer is read against is [`crate::bench::geometry`], which the
//! wire's patch tools read it against too.

use egui::{Color32, CursorIcon, Pos2, Rect, Shape, Stroke, Vec2};
use sfmtool_core::bench::{Edge, EditableTrack, Observation, Stage, Verdict, Viewpoint};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::EditedReconstruction;
use sfmtool_core::ImageTable;

use crate::bench::geometry::{self, PatchEdit, Tilt};
use crate::bench::{distance_to_segment, resize_cursor, verdict_color};

use super::ImageDetailResponse;

/// Stroke width of the bench's outlines. Thicker than a feature ellipse's, so
/// the layer reads as being on top of the overlay rather than part of it.
const STROKE_WIDTH: f32 = 2.5;

/// Stroke width of the normal's segment and arrowhead: the offset segments'
/// width, so the arrow reads as standing off the square rather than as a
/// fifth edge of it.
const NORMAL_STROKE_WIDTH: f32 = 1.5;

/// How opaque the ghost outline is, as a fraction of the member outline's
/// colour: 80% opaque, so it is plainly the same patch and still plainly not a
/// sighting.
pub(super) const GHOST_OPACITY: f32 = 0.8;

/// Radius of the mark drawn at an observation's own position, in panel px.
const KEYPOINT_RADIUS: f32 = 4.0;

/// How far from a dot, a corner or the arrowhead the pointer still grabs it,
/// in panel px.
///
/// Generous against the 4 px dot it draws, because the cost of the two misses
/// is not symmetric: a handle missed by two pixels pans the photograph instead,
/// which is a gesture the person then has to undo by eye, while a handle caught
/// a little early is released with no motion and does nothing.
const HANDLE_HIT_RADIUS: f32 = 9.0;

/// How far from an edge's polyline or the normal's segment the pointer still
/// grabs it, in panel px.
///
/// Narrower than a corner's reach, and tested after the corners, so the corner
/// where two edges meet turns rather than resizing whichever edge won the
/// distance.
const EDGE_HIT_WIDTH: f32 = 8.0;

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
/// Every outline handle names the [`Viewpoint`] its outline was drawn from,
/// which is the square the pointer is read against: an observation's, the
/// patch re-anchored on that sighting at the track stage and its own shape at
/// the cluster stage, or this image's own, the patch as it stands, for the
/// ghost.
///
/// Not `Eq`, the arrowhead carrying the axis its swing turns about: the gesture
/// is decided at the press and travels with the drag, as the 3D viewport's
/// does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Handle {
    /// The observation's own mark: dragging it places the sighting.
    Keypoint {
        /// The observation, by its position in the track's list.
        observation: usize,
    },
    /// The ghost outline's centre mark: dragging it slides the patch across
    /// its own plane until its centre sits under the pointer in this image.
    Centre,
    /// One edge of the outline: dragging it resizes the patch, holding the
    /// opposite edge still.
    Edge {
        /// The view the outline is drawn from.
        outline: Viewpoint,
        /// Which edge of the patch's square it is.
        edge: Edge,
    },
    /// One corner of the outline: dragging it turns the patch.
    Corner {
        /// The view the outline is drawn from.
        outline: Viewpoint,
        /// Which corner, as an index into [`CORNERS`].
        corner: usize,
    },
    /// The normal's segment: dragging it moves the patch along its normal.
    Normal {
        /// The view the outline the normal stands out of is drawn from.
        outline: Viewpoint,
    },
    /// The arrowhead at the far end of the segment: dragging it turns the
    /// normal, by the gesture [`geometry::tilt_gesture`] chose at the press.
    Arrowhead {
        /// The view the outline the normal stands out of is drawn from.
        outline: Viewpoint,
        /// Which of the arrowhead's two gestures the drag makes.
        tilt: Tilt,
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
    /// The row it selects in Track View.
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
    /// One per observation of the track in this image. None for the ghost.
    sightings: Vec<Sighting>,
    /// The outlines: one at the track stage and for the ghost, one per
    /// observation at the cluster stage.
    outlines: Vec<Outline>,
    /// The patch's own projection, which every observation's
    /// projection-offset segment runs to. `None` at the cluster stage, for a
    /// track nothing has triangulated, and for the ghost, which has no segment
    /// to end.
    center: Option<Pos2>,
    /// The seed parallelograms drawn dashed behind the cluster stage's
    /// outlines. Never a handle: where an observation *started* is not a thing
    /// to drag.
    seeds: Vec<(Vec<Pos2>, Color32)>,
    /// Whether the outlines' edges and corners take a drag. False only at the
    /// track stage with the lock off; see [`outline_takes_drag`].
    outline_handles: bool,
    /// The ghost outline's centre mark, the patch's own projection, when the
    /// ghost offers handles.
    ghost_centre: Option<Pos2>,
    /// The normal standing out of the outline, when there is one to draw.
    normal: Option<Arrow>,
    /// Whether the normal's segment and arrowhead take a drag: the lock, since
    /// a move along the normal or a turn of it moves every sighting.
    normal_handles: bool,
}

/// One observation's own mark.
struct Sighting {
    observation: usize,
    verdict: Verdict,
    at: Pos2,
}

/// One patch outline, as the boundary samples that landed.
struct Outline {
    /// The view it is drawn from.
    viewpoint: Viewpoint,
    /// The colour it is stroked in.
    color: Color32,
    /// The `4 * per_edge` samples in boundary order, `None` for one that did
    /// not project.
    samples: Vec<Option<Pos2>>,
    /// Samples per edge, so corner `k` is sample `k * per_edge`.
    per_edge: usize,
    /// Where the square it outlines is centred on screen, which is what a
    /// corner spins about: the sighting for an anchored outline and at the
    /// cluster stage, the patch's own projection for the ghost.
    pivot: Option<Pos2>,
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

    /// The box every sample that landed sits in.
    fn bounds(&self) -> Option<Rect> {
        self.samples.iter().flatten().fold(None, |bounds, point| {
            Some(bounds.map_or(Rect::from_pos(*point), |r: Rect| {
                r.union(Rect::from_pos(*point))
            }))
        })
    }
}

/// The patch's normal projected into the photograph: the segment from the
/// outline's centre to the tip, and the arrowhead's two barbs.
///
/// The 3D viewport's arrow ([`geometry::arrow`]), built in the world at the
/// same length, [`geometry::NORMAL_LENGTH`] half-lengths, with its barbs
/// squared to this photograph's camera, and then projected through the lens as
/// the outline is. Its size is therefore the patch's: a small patch has a small
/// arrow, and the arrow foreshortens with the view as the square does, which is
/// what makes it something to align against.
struct Arrow {
    /// The view of the outline it stands out of.
    outline: Viewpoint,
    /// Where it leaves the square, the outline's own centre.
    tail: Pos2,
    /// The far end, where the arrowhead is grabbed.
    tip: Pos2,
    /// The barbs' far ends.
    barbs: [Pos2; 2],
    /// The colour of the outline it stands out of.
    color: Color32,
    /// Which gesture a press on the arrowhead makes from this camera.
    tilt: Option<Tilt>,
}

impl Layer {
    /// The layer's geometry for `track` in `img_idx`, or `None` when there is
    /// nothing to draw there.
    ///
    /// An image the track observes gets the member drawing. An image it does
    /// not gets the ghost, when there is one ([`Layer::ghost`]).
    ///
    /// `lock` is Track View's *Lock*, which decides whether the patch-wide
    /// handles take a drag at all.
    pub(super) fn build(
        image_table: &ImageTable,
        img_idx: usize,
        track: &EditableTrack,
        image_rect: Rect,
        effective_scale: f32,
        lock: bool,
    ) -> Option<Self> {
        let here: Vec<(usize, &Observation)> = track
            .observations
            .iter()
            .enumerate()
            .filter(|(_, o)| o.image as usize == img_idx)
            .collect();
        let to_panel = panel_mapping(image_rect, effective_scale);
        if here.is_empty() {
            return Self::ghost(image_table, img_idx, track, &to_panel, lock);
        }
        let mut layer = Layer {
            sightings: Vec::new(),
            outlines: Vec::new(),
            center: None,
            seeds: Vec::new(),
            outline_handles: outline_takes_drag(track, lock),
            ghost_centre: None,
            normal: None,
            normal_handles: lock,
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
                    track,
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
    /// strongest verdict names, with the normal standing out of its centre.
    ///
    /// One outline for the image rather than one per observation, because there
    /// is one patch: two candidates in a photograph are two readings of where
    /// it lands, not two squares. Its colour is the strongest verdict among
    /// them, so an image the track is `in` reads as `in`. It is drawn through
    /// the frame re-anchored on that sighting, as the tile is rendered: the
    /// sighting is where the patch sits in this photograph, and the geometric
    /// projection is where the 3D says it should, which the hollow centre and
    /// the projection-offset segment show separately.
    ///
    /// The normal leaves the **anchored** centre, which is the sighting's dot,
    /// for the reason the outline is anchored: the arrow belongs to the square
    /// drawn, and one standing out of the hollow centre instead would part from
    /// its own outline by the projection offset. The anchored frame is the
    /// patch moved across its own plane, so its normal line is parallel to the
    /// patch's and the two gestures read the same depth and the same facing off
    /// either; the square a person looks at is the one the pointer is read
    /// against.
    fn build_track_stage(
        &mut self,
        track: &EditableTrack,
        here: &[(usize, &Observation)],
        patch: Option<&OrientedPatch>,
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
        let (Some((index, _)), Some((camera, pose))) = (anchor, view) else {
            return;
        };
        self.center = patch
            .and_then(|patch| geometry::project(camera, pose, patch.center.coords, patch.w))
            .map(to_panel);
        let viewpoint = Viewpoint::Observation(*index);
        let Some(square) = square_of(track, viewpoint, camera, pose) else {
            return;
        };
        let color = verdict_color(strongest);
        let (samples, per_edge) = project_outline(&square, camera, pose);
        let pivot = self
            .sightings
            .iter()
            .find(|sighting| sighting.observation == *index)
            .map(|sighting| sighting.at)
            .or(self.center);
        self.outlines.push(Outline {
            viewpoint,
            color,
            samples: samples.into_iter().map(|p| p.map(to_panel)).collect(),
            per_edge,
            pivot,
        });
        self.normal = normal_arrow(&square, viewpoint, camera, pose, color, to_panel);
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
            let pivot = self
                .sightings
                .iter()
                .find(|sighting| sighting.observation == *index)
                .map(|sighting| sighting.at);
            self.outlines.push(Outline {
                viewpoint: Viewpoint::Observation(*index),
                color: verdict_color(observation.verdict),
                samples: corners.into_iter().map(Some).collect(),
                per_edge: 1,
                pivot,
            });
        }
    }

    /// The ghost outline: the track-stage patch's own square projected into a
    /// photograph the track has no observation in.
    ///
    /// An image counts as the track's when it holds an observation of any
    /// verdict, so a `candidate` or an `out` sighting keeps the member drawing
    /// and only an image with none gets the ghost. What is projected is the
    /// patch **itself**, not a frame re-anchored on a keypoint, because there is
    /// no keypoint here to anchor on: the ghost is where the 3D places the
    /// square, which is also where the hollow centre of a member image sits.
    ///
    /// There is nothing to draw, and so no ghost, at the cluster stage (no
    /// shared geometry exists to project), at a track stage with no placement
    /// yet, and in a view that cannot see the square: the patch's back face
    /// turned toward the camera ([`OrientedPatch::is_front_facing`]), its plane
    /// seen within [`geometry::MIN_PLANE_ANGLE_DEG`] of edge-on
    /// ([`geometry::plane_is_edge_on`]), or its centre behind the camera or
    /// outside the lens model's domain. Past those tests the boundary is
    /// sampled and drawn as a member outline is, a sample that fails to project
    /// breaking the curve rather than being bridged.
    ///
    /// **With `lock` ticked the ghost is a handle**, read through
    /// [`Viewpoint::Image`]: the centre mark slides the patch until its own
    /// centre is under the pointer, an edge resizes it with the far edge held,
    /// a corner spins it about its normal, and the normal it now draws moves
    /// and tilts it. They are the member outline's patch-wide edits read
    /// against the square this camera sees. With the lock cleared it is display
    /// only, having no keypoint of its own for the dot to move: no mark, no
    /// arrow, no handle, and a press on it pans.
    fn ghost(
        image_table: &ImageTable,
        img_idx: usize,
        track: &EditableTrack,
        to_panel: &impl Fn([f64; 2]) -> Pos2,
        lock: bool,
    ) -> Option<Self> {
        let Stage::Track(payload) = &track.stage else {
            return None;
        };
        let patch = payload.placement.as_ref()?;
        let (camera, pose) = geometry::view_of(image_table, img_idx)?;
        if !patch.is_front_facing(&pose)
            || geometry::plane_is_edge_on(patch, pose.inverse_translation_origin())
        {
            return None;
        }
        let centre = geometry::project(&camera, &pose, patch.center.coords, patch.w)?;
        let viewpoint = Viewpoint::Image(img_idx as u32);
        let (samples, per_edge) = project_outline(patch, &camera, &pose);
        let color = ghost_color();
        let centre = to_panel(centre);
        Some(Layer {
            sightings: Vec::new(),
            outlines: vec![Outline {
                viewpoint,
                color,
                samples: samples.into_iter().map(|p| p.map(to_panel)).collect(),
                per_edge,
                pivot: Some(centre),
            }],
            center: None,
            seeds: Vec::new(),
            outline_handles: lock,
            ghost_centre: lock.then_some(centre),
            normal: if lock {
                normal_arrow(patch, viewpoint, &camera, &pose, color, to_panel)
            } else {
                None
            },
            normal_handles: lock,
        })
    }

    /// The handle nearest `pos`, or `None` when the pointer is on none of them.
    ///
    /// The arrowhead, then the dots, then corners, then edges, then the
    /// normal's segment. The dot sits inside the outline it anchors and the
    /// corner is where two edges meet, so a nearest-thing search over all of
    /// them at once would make the smaller handles unreachable. The segment is
    /// last and the arrowhead first for the 3D viewport's reason: the segment
    /// leaves the centre, where the dot already is, and has the whole of its
    /// length to be reached along, while the arrowhead is a point at its far
    /// end that anything tested before it would take the presses of.
    pub(super) fn hit(&self, pos: Pos2) -> Option<Handle> {
        let nearest = |best: Option<(f32, Handle)>, distance: f32, handle: Handle| match best {
            Some((d, _)) if d <= distance => best,
            _ => Some((distance, handle)),
        };
        if let Some(arrow) = self.normal.as_ref().filter(|_| self.normal_handles) {
            if let Some(tilt) = arrow.tilt {
                if (arrow.tip - pos).length() <= HANDLE_HIT_RADIUS {
                    return Some(Handle::Arrowhead {
                        outline: arrow.outline,
                        tilt,
                    });
                }
            }
        }
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
        if let Some(centre) = self.ghost_centre {
            let distance = (centre - pos).length();
            if distance <= HANDLE_HIT_RADIUS {
                found = nearest(found, distance, Handle::Centre);
            }
        }
        if let Some((_, handle)) = found {
            return Some(handle);
        }
        // An outline that takes no drag is not a handle, so a press on it falls
        // through to the view and pans, as a press anywhere off the layer does.
        if self.outline_handles {
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
                                outline: outline.viewpoint,
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
                                outline: outline.viewpoint,
                                edge: geometry::edge_of(edge),
                            },
                        );
                    }
                }
            }
            if let Some((_, handle)) = found {
                return Some(handle);
            }
        }
        self.normal
            .as_ref()
            .filter(|arrow| {
                self.normal_handles
                    && distance_to_segment(arrow.tail, arrow.tip, pos) <= EDGE_HIT_WIDTH
            })
            .map(|arrow| Handle::Normal {
                outline: arrow.outline,
            })
    }

    /// The cursor `handle` asks for, with the outline's own orientation on
    /// screen deciding which resize cursor an edge takes.
    ///
    /// An edge that looks horizontal is one you move up and down, so it takes
    /// the vertical resize cursor, and an oblique edge takes the diagonal whose
    /// slope it has.
    ///
    /// A corner spins the patch, and egui's cursors are the CSS set, which has
    /// never had one for a rotation. So a corner takes the resize cursor lying
    /// along the way it **travels**: the tangent of the circle it spins on,
    /// which is the perpendicular of its radius from the pivot. Because
    /// [`resize_cursor`] answers with the perpendicular of whatever it is
    /// handed, the radius is what it gets. Running the pointer along an edge
    /// and onto the corner then turns the cursor from across the edge to along
    /// the arc, and that turn is exactly the difference between the two
    /// gestures.
    ///
    /// The normal's two handles take the 3D viewport's cursors: the segment is
    /// dragged along itself, so it hands [`resize_cursor`] its own
    /// perpendicular; an aiming arrowhead is free in two directions and takes
    /// `AllScroll`; a swinging one travels along its arc and takes the
    /// corner's reading, the radius from the tail.
    fn cursor(&self, handle: Handle) -> CursorIcon {
        let outline = |viewpoint: Viewpoint| {
            self.outlines
                .iter()
                .find(move |outline| outline.viewpoint == viewpoint)
        };
        match handle {
            Handle::Keypoint { .. } | Handle::Centre => CursorIcon::Move,
            Handle::Corner {
                outline: at,
                corner,
            } => outline(at)
                .and_then(|outline| Some(resize_cursor(outline.corner(corner)? - outline.pivot?)))
                .unwrap_or(CursorIcon::Move),
            Handle::Edge { outline: at, edge } => outline(at)
                .and_then(|outline| {
                    let k = (0..4).find(|k| geometry::edge_of(*k) == edge)?;
                    let points = outline.edge(k);
                    let (first, last) = (points.first()?, points.last()?);
                    Some(resize_cursor(*last - *first))
                })
                .unwrap_or(CursorIcon::Move),
            Handle::Normal { .. } => self
                .normal
                .as_ref()
                .map(|arrow| {
                    let along = arrow.tip - arrow.tail;
                    resize_cursor(egui::vec2(-along.y, along.x))
                })
                .unwrap_or(CursorIcon::Move),
            Handle::Arrowhead {
                tilt: Tilt::Aim, ..
            } => CursorIcon::AllScroll,
            Handle::Arrowhead {
                tilt: Tilt::Swing(_),
                ..
            } => self
                .normal
                .as_ref()
                .map(|arrow| resize_cursor(arrow.tip - arrow.tail))
                .unwrap_or(CursorIcon::Move),
        }
    }

    /// Paint the layer.
    fn paint(&self, painter: &egui::Painter) {
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
            paint_boundary(
                painter,
                &outline.samples,
                Stroke::new(STROKE_WIDTH, outline.color),
            );
        }
        // Polylines rather than segments, so the arrow is never read as one of
        // the projection-offset segments, which are the layer's only segments.
        if let Some(arrow) = &self.normal {
            let stroke = Stroke::new(NORMAL_STROKE_WIDTH, arrow.color);
            painter.add(Shape::line(vec![arrow.tail, arrow.tip], stroke));
            painter.add(Shape::line(
                vec![arrow.barbs[0], arrow.tip, arrow.barbs[1]],
                stroke,
            ));
        }
        // Hollow, as a member image's mark for the same place is: the patch's
        // own projection, which a filled dot would misread as a keypoint.
        if let Some(centre) = self.ghost_centre {
            painter.circle_stroke(centre, KEYPOINT_RADIUS, Stroke::new(1.5, ghost_color()));
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
    }

    /// The marks a click selects a Track View row by: each sighting, with the
    /// outline it belongs to as its reach. The ghost has no sighting, so a
    /// click on it selects nothing.
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
    /// Two of the plane gestures are already in those terms: a dot drag and an
    /// edge drag both name a pixel and the view it is in, and the arithmetic
    /// behind the edge (the opposite edge held still) is
    /// `sfmtool_core::bench::resize_patch_to_pixel`'s rather than this panel's,
    /// so a tool call and a drag cannot resize differently. What is left here
    /// is the three readings of two pointer positions against the patch that
    /// have no other home: the turn a corner swept, and the normal's two
    /// gestures, each pixel read as a ray of this photograph's camera
    /// ([`geometry::pixel_ray`]) against the geometry the 3D viewport reads its
    /// own rays against.
    ///
    /// `lock` is Track View's *Lock*: at the track stage it is what makes the
    /// dot the patch's ([`PatchEdit::TranslateToPixel`]) or the sighting's
    /// alone ([`PatchEdit::Sight`]), and with it off every patch-wide handle
    /// (an edge, a corner, the normal, the arrowhead and the ghost's centre)
    /// edits nothing, the sighting having no size, turn, depth or facing of its
    /// own.
    ///
    /// `None` when the pointer names nothing the patch can be given: a ray that
    /// misses its plane, or a turn about the centre itself.
    pub(super) fn edit(
        image_table: &ImageTable,
        track: &EditableTrack,
        drag: &Drag,
        lock: bool,
    ) -> Option<PatchEdit> {
        if drag.cancelled {
            return None;
        }
        // The same rules the hit test was built by, asked again rather than
        // assumed: a drag is judged by the setting in force when it is read.
        let takes_drag = match drag.handle {
            Handle::Keypoint { .. } => true,
            Handle::Edge { .. } | Handle::Corner { .. } => outline_takes_drag(track, lock),
            Handle::Centre | Handle::Normal { .. } | Handle::Arrowhead { .. } => {
                lock && matches!(track.stage, Stage::Track(_))
            }
        };
        if !takes_drag {
            return None;
        }
        let read_square = |outline: Viewpoint| {
            let (camera, pose) = geometry::view_of(image_table, drag.image)?;
            let square = square_of(track, outline, &camera, &pose)?;
            let from = geometry::pixel_ray(&camera, &pose, drag.from)?;
            let to = geometry::pixel_ray(&camera, &pose, drag.to)?;
            Some((square, camera, pose, from, to))
        };
        match drag.handle {
            // The dot means different things at the two stages, because the two
            // stages have different things to move: a track-stage track has one
            // patch and every sighting is a view of it, so dragging the mark
            // slides the **patch** and every sighting follows; a cluster has no
            // shared geometry at all, so the mark is that sighting's own seed
            // and nothing else moves. The lock off makes the track stage's mark
            // the sighting's own too, which is the one step both stages share.
            Handle::Keypoint { observation } => Some(match track.stage {
                Stage::Track(_) if lock => PatchEdit::TranslateToPixel {
                    viewpoint: Viewpoint::Observation(observation),
                    pixel: drag.to,
                },
                Stage::Track(_) | Stage::Cluster(_) => PatchEdit::Sight {
                    observation,
                    pixel: drag.to,
                },
            }),
            Handle::Centre => Some(PatchEdit::TranslateToPixel {
                viewpoint: Viewpoint::Image(drag.image as u32),
                pixel: drag.to,
            }),
            Handle::Edge { outline, edge } => Some(PatchEdit::ResizeToPixel {
                viewpoint: outline,
                edge,
                pixel: drag.to,
            }),
            Handle::Corner { outline, .. } => match (&track.stage, outline) {
                (Stage::Track(_), _) => {
                    let (square, camera, pose, _, _) = read_square(outline)?;
                    let angle_rad =
                        geometry::turn_between(&square, &camera, &pose, drag.from, drag.to)?;
                    Some(PatchEdit::Spin { angle_rad })
                }
                (Stage::Cluster(_), Viewpoint::Observation(observation)) => {
                    let sighting = track.observations.get(observation)?;
                    let angle_rad =
                        geometry::pixel_turn_between(sighting.site()?, drag.from, drag.to)?;
                    Some(PatchEdit::SpinShape {
                        observation,
                        angle_rad,
                    })
                }
                (Stage::Cluster(_), Viewpoint::Image(_)) => None,
            },
            // Both ends are points of the drawn normal's own line, so their
            // difference along it is the whole gesture, and a segment grabbed
            // at its tip does not jump the patch out to where the tip was.
            Handle::Normal { outline } => {
                let (square, _, _, from, to) = read_square(outline)?;
                let start = geometry::normal_line_point(&square, from.0, from.1)?;
                let now = geometry::normal_line_point(&square, to.0, to.1)?;
                Some(PatchEdit::Translate {
                    by: [0.0, 0.0, (now - start).dot(&square.normal())],
                })
            }
            Handle::Arrowhead { outline, tilt } => {
                let (square, _, _, from, to) = read_square(outline)?;
                let start = geometry::tilt_point(&square, tilt, from.0, from.1)?;
                let now = geometry::tilt_point(&square, tilt, to.0, to.1)?;
                let normal = geometry::tilt_normal(&square, tilt, start, now)?;
                Some(PatchEdit::Tilt {
                    normal: [normal.x, normal.y, normal.z],
                })
            }
        }
    }
}

/// The ghost outline's colour: the `in` violet at [`GHOST_OPACITY`].
pub(super) fn ghost_color() -> Color32 {
    verdict_color(Verdict::In).gamma_multiply(GHOST_OPACITY)
}

/// The square an outline drawn from `viewpoint` shows in a view, at the track
/// stage: the patch re-anchored on the observation's keypoint, or the patch
/// itself for an image. `None` at the cluster stage, for a track with no
/// placement, and for an observation index past the end.
///
/// The one place both the drawing and the drag decide it, so the square a
/// pointer is read against is the square drawn.
fn square_of(
    track: &EditableTrack,
    viewpoint: Viewpoint,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
) -> Option<OrientedPatch> {
    let Stage::Track(payload) = &track.stage else {
        return None;
    };
    let patch = payload.placement.as_ref()?;
    Some(match viewpoint {
        Viewpoint::Observation(observation) => {
            geometry::anchored_frame(patch, camera, pose, track.observations.get(observation)?)
        }
        Viewpoint::Image(_) => patch.clone(),
    })
}

/// The normal standing out of `square` as this photograph sees it, or `None`
/// when there is none to draw.
///
/// None for a normal seen end-on from this camera
/// ([`geometry::normal_is_end_on`]), the bar the 3D viewport refuses its
/// segment at: seen down its own length the arrow collapses onto the centre and
/// a pixel of pointer motion along it is an unbounded distance. That also
/// covers a direction patch, whose normal is its bearing. None as well where
/// the tail, the tip or a barb falls behind the camera or outside the lens
/// model's domain, an arrow drawn in part saying nothing about where it points.
fn normal_arrow(
    square: &OrientedPatch,
    viewpoint: Viewpoint,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
    color: Color32,
    to_panel: &impl Fn([f64; 2]) -> Pos2,
) -> Option<Arrow> {
    let eye = pose.inverse_translation_origin();
    if geometry::normal_is_end_on(square, eye) {
        return None;
    }
    let arrow = geometry::arrow(
        square.center,
        square.normal(),
        geometry::NORMAL_LENGTH * square.half_extent[0],
        eye,
        square.u_axis,
    );
    let at = |point: nalgebra::Point3<f64>| {
        geometry::project(camera, pose, point.coords, 1.0).map(to_panel)
    };
    Some(Arrow {
        outline: viewpoint,
        tail: at(arrow.tail)?,
        tip: at(arrow.tip)?,
        barbs: [at(arrow.barbs[0])?, at(arrow.barbs[1])?],
        color,
        tilt: geometry::tilt_gesture(square, eye),
    })
}

/// Stroke a projected patch boundary, one polyline per run of samples that
/// landed, closed when every sample did.
fn paint_boundary(painter: &egui::Painter, samples: &[Option<Pos2>], stroke: Stroke) {
    let (runs, closed) = runs(samples);
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

/// The runs of consecutive boundary samples that landed, and whether the
/// boundary closed.
///
/// A sample behind the camera or outside the lens model's domain ends a run and
/// starts the next, so an outline that leaves the model's field is drawn as the
/// arcs that are defined rather than closed across the gap with a chord that
/// means nothing.
fn runs(samples: &[Option<Pos2>]) -> (Vec<Vec<Pos2>>, bool) {
    let n = samples.len();
    let Some(gap) = (0..n).find(|&i| samples[i].is_none()) else {
        return (vec![samples.iter().flatten().copied().collect()], true);
    };
    // Walk from the sample after the first gap, so a run that spans the point
    // the boundary happens to start at is one run rather than two.
    let mut runs: Vec<Vec<Pos2>> = Vec::new();
    let mut run: Vec<Pos2> = Vec::new();
    for step in 1..=n {
        match samples[(gap + step) % n] {
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

/// Draw the active bench track in `img_idx`, and report a click on one of its
/// marks.
///
/// `drag` is the handle the pointer has hold of, if any: the layer is drawn
/// from the track that drag would produce, so what a person sees mid-gesture is
/// what releasing would leave behind. `lock` is Track View's *Lock*, read by
/// the preview exactly as the release reads it.
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
    lock: bool,
    response: &mut ImageDetailResponse,
) {
    // The preview: the track the release would push, drawn instead of the one
    // on the bench. Through the same function the release goes through, so
    // there is one answer rather than a drawn guess and a pushed result.
    let image_table = &edited.base.image_table;
    let previewed = drag
        .filter(|drag| drag.image == img_idx && !drag.cancelled)
        .and_then(|drag| Layer::edit(image_table, track, drag, lock))
        .and_then(|edit| geometry::apply(track, edited, &edit).ok())
        .map(|(next, _)| next);
    let shown = previewed.as_ref().unwrap_or(track);

    let Some(layer) = Layer::build(
        image_table,
        img_idx,
        shown,
        image_rect,
        effective_scale,
        lock,
    ) else {
        return;
    };
    layer.paint(painter);
    if let Some(handle) = hovered {
        ui.ctx().set_cursor_icon(match drag {
            Some(drag) if !drag.cancelled => match drag.handle {
                Handle::Keypoint { .. } | Handle::Centre => CursorIcon::Grabbing,
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

/// Where a source pixel lands in the panel, for the image drawn in
/// `image_rect` at `effective_scale` panel px per source px.
fn panel_mapping(image_rect: Rect, effective_scale: f32) -> impl Fn([f64; 2]) -> Pos2 {
    move |p: [f64; 2]| {
        Pos2::new(
            image_rect.min.x + p[0] as f32 * effective_scale,
            image_rect.min.y + p[1] as f32 * effective_scale,
        )
    }
}

/// Whether an outline's edges and corners take a drag: always at the cluster
/// stage, whose outline is each sighting's own shape, and at the track stage
/// only with the lock on, the outline there being the patch's.
///
/// One rule, asked by the hit test and again by [`Layer::edit`], so a handle
/// the layer does not offer is also one it does not act on.
fn outline_takes_drag(track: &EditableTrack, lock: bool) -> bool {
    lock || matches!(track.stage, Stage::Cluster(_))
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
