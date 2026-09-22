// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench's active track as world geometry, for the 3D viewer to draw.
//!
//! Where the Image Detail panel's bench layer shows the track in each
//! photograph that observes it, this shows it where it stands: the patch frame
//! as a square in the world, its outward normal standing off it, and one mark
//! per observation saying where that photograph sees the patch's content
//! against where the frame's middle is. One difference from Image Detail is
//! deliberate -- there the outline is the patch **re-anchored on that image's
//! keypoint**, because the panel shows where the sighting is in that
//! photograph; here there is no photograph and the frame drawn is the patch
//! itself, at its centre.
//!
//! What this module produces is a
//! [`Figure`](crate::viewer_3d::bench_track::Figure): world points, world
//! directions and colours, with no window and no GPU in it, so the geometry is
//! assertable without a frame and without a device. [`crate::scene_renderer`]
//! turns it into instances and draws it after the EDL pass, depth-aware, so
//! where the frame cuts through the point cloud the part in front reads at full
//! strength and the part behind is drawn through, dimmed.
//!
//! **What it draws, it edits.** The handles
//! ([`Handles`](crate::viewer_3d::bench_track::Handles)) project the very figure
//! the pass drew back through the viewport's camera, so the square a person
//! takes hold of is the square they can see: the dot slides it across its plane,
//! an edge resizes it with the far edge held, a corner turns it about its
//! normal, the normal's own segment moves it along that normal, the arrowhead
//! at the far end of that segment turns the normal itself, and an
//! observation's circle selects that row in Track View. The
//! pointer is read as a **ray of the viewport's camera** met with the patch's
//! own geometry ([`crate::bench::geometry`]), which is the same idea the Image
//! Detail panel reads a pixel by, and the edit it names is handed to the same
//! core step. The segment and the arrowhead are the two handles with no
//! counterpart in a photograph, which is why they are here: a sighting names
//! the ray the patch lies along, and says nothing about how far down it the
//! surface is or which way it faces.

use egui::{Color32, CursorIcon, Pos2, Rect};
use nalgebra::{Point3, Vector3};
use sfmtool_core::bench::{Axis, Edge, EditableTrack, Stage};
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::{EditedReconstruction, Se3Transform};

use crate::bench::geometry::{self, PatchEdit, Tilt};
use crate::bench::{distance_to_segment, resize_cursor, verdict_color};
use crate::scene::ReconId;

use super::ViewportCamera;

/// Segments in the centre disc and in each mark's hollow circle.
const CIRCLE_SEGMENTS: usize = 24;

/// The disc's and the circles' radius, as a fraction of the frame's
/// half-length: small enough to sit inside the square it marks the middle of.
const CIRCLE_RADIUS: f64 = 1.0 / 8.0;

/// How far everything drawn in the frame's plane is lifted along the outward
/// normal, as a fraction of the frame's half-length. The patch's own bitmap
/// lies in that plane, and the pass reads in-front-or-behind off the depth
/// buffer, so geometry exactly coplanar with it flickers between the two. A
/// fraction of the half-length keeps the lift independent of the scene's size.
const PLANE_LIFT: f64 = 1e-3;

/// How far the normal stands off the frame, in half-lengths -- one side length,
/// which is long enough to be grabbed and short enough not to cross the scene.
///
/// Visible to [`crate::bench::geometry`], which states the arrowhead's aiming
/// lever in terms of it: the arrow a person sees and the travel their pointer
/// turns it by are two facts about one handle, and a second literal could drift
/// from this one silently.
pub(crate) const NORMAL_LENGTH: f64 = 2.0;

/// How far back along the normal the arrowhead's barbs reach, as a fraction of
/// the normal's own length.
const BARB_BACK: f64 = 0.25;

/// How far to either side they reach, in the same units.
const BARB_SIDE: f64 = 0.1;

/// How far behind the scene the figure fades to the floor, in half-lengths.
///
/// Two side lengths, so the falloff is in the patch's own units: a frame sunk a
/// fraction of its size into a surface reads as slightly veiled, and one well
/// behind it sits at the floor.
const FOG_DISTANCE: f64 = 4.0;

/// How much larger the circle of the observation selected in Track View is
/// drawn.
///
/// The one thing the figure says about the selection, and the other half of the
/// click that sets it: a row picked in the panel can then be found in the world,
/// and a mark picked in the world can be seen to be that row.
const SELECTED_CIRCLE_SCALE: f64 = 1.6;

/// How far from the dot, a corner or an observation's circle the pointer still
/// grabs it, in panel px.
///
/// Generous against the marks they draw, for the reason the Image Detail
/// layer's reach is: a handle missed by two pixels orbits the scene instead,
/// which the person then has to undo by eye, while one caught a little early is
/// released without motion and does nothing.
///
/// Visible to the viewport's own tests, which aim presses clear of it.
pub(super) const HANDLE_HIT_RADIUS: f32 = 9.0;

/// How far from an edge of the square the pointer still grabs it, in panel px.
///
/// Narrower than a corner's reach, and tested after the corners, so the corner
/// where two edges meet turns rather than resizing whichever edge won the
/// distance.
const EDGE_HIT_WIDTH: f32 = 8.0;

/// The four `(s, t)` corners of the frame's square, in the order
/// [`OrientedPatch::boundary`] walks them, so consecutive pairs are its edges.
const CORNERS: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// What the dock tells the 3D viewer about the node's bench, once a frame.
///
/// The viewer does not read the bench, for the reason the Image Detail panel
/// does not ([`crate::image_detail::BenchMenu`]): the panel is handed `&mut`
/// into the state further down the same call, so what it needs is read out
/// beside the selection and passed in.
#[derive(Clone, Copy)]
pub(crate) struct BenchTrack<'a> {
    /// The node whose bench it is. A gesture that outlives a change of selected
    /// node is dropped: its handle names a frame that is no longer up.
    pub(crate) node: ReconId,
    /// The active track, which is the one item of the bench this layer draws.
    pub(crate) track: &'a EditableTrack,
    /// The node's value at its cursor, for the cameras and poses the marks
    /// unproject their keypoints through.
    pub(crate) edited: &'a EditedReconstruction,
    /// The node's similarity. The figure is built in the reconstruction's own
    /// coordinates and put through this, because the pass draws from one shared
    /// buffer with no per-recon `model` matrix of its own.
    pub(crate) transform: &'a Se3Transform,
    /// The observation row selected in Track View, whose circle is drawn
    /// [`SELECTED_CIRCLE_SCALE`] larger.
    pub(crate) selected: Option<usize>,
    /// Whether a background task holds the node. The layer still draws -- what
    /// is being worked on does not stop being worth seeing -- and no handle
    /// takes a press.
    pub(crate) busy: bool,
}

/// One straight edge of the figure: two homogeneous world endpoints and the
/// colour it is drawn in.
///
/// `w` is `1.0` for a place and `0.0` for a direction, and it is the projection
/// the pass gives the endpoint: a direction is projected rotation-only, so the
/// figure of a track at infinity has no parallax and stays on the sky as the
/// viewer moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Stroke {
    /// One endpoint, `(xyz, w)`.
    pub(crate) a: [f32; 4],
    /// The other.
    pub(crate) b: [f32; 4],
    /// The bench violet this stroke carries, its channels over 255 -- the
    /// convention every pass in this renderer states its colours in.
    pub(crate) color: [f32; 4],
}

/// One vertex of the filled centre disc, which is a triangle list rather than a
/// ribbon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DiscVertex {
    /// Where, `(xyz, w)`, on [`Stroke`]'s convention.
    pub(crate) at: [f32; 4],
    /// The colour, on [`Stroke::color`]'s convention.
    pub(crate) color: [f32; 4],
}

/// One observation's mark: where that photograph sees the patch's content.
///
/// The segment runs from the frame's centre to `q_i`, the point of the frame's
/// plane the observation's keypoint ray meets, and the circle is drawn there.
/// Both are in that observation's own verdict colour, whatever the verdict is:
/// an observation already judged in is exactly the one whose answer is worth
/// seeing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mark {
    /// Which observation it is, by its position in the track's list -- which is
    /// not this mark's own position, an observation whose ray misses the frame
    /// drawing none. It is the row a click on the circle selects.
    pub(crate) observation: usize,
    /// Centre to `q_i`.
    pub(crate) segment: Stroke,
    /// The hollow circle at `q_i`, in the frame's plane.
    pub(crate) circle: Vec<Stroke>,
}

/// The whole figure, in world coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Figure {
    /// The square, four edges in `(s, t)` boundary order, in the `in` colour
    /// whatever any observation's verdict is. Edge `k` runs from corner `k` to
    /// corner `k + 1`, so `frame[k].a` is corner `k`.
    pub(crate) frame: [Stroke; 4],
    /// Where the frame's centre is, on [`Stroke`]'s `(xyz, w)` convention.
    ///
    /// The disc is drawn around it and every mark's segment runs from it, so it
    /// is in the figure twice over already; it is named once here because the
    /// hit test wants the dot itself rather than a vertex of the fan.
    pub(crate) centre: [f32; 4],
    /// The filled centre disc, as `CIRCLE_SEGMENTS` triangles fanned from the
    /// centre. In the frame's plane, so it foreshortens with the frame and goes
    /// to a line when the frame is edge-on -- which is what says it is part of
    /// the patch and not a billboard like a 3D point.
    pub(crate) disc: Vec<DiscVertex>,
    /// The normal's segment and the arrowhead's two barbs, or `None` for a
    /// track at infinity: a direction patch's normal is fixed by its bearing,
    /// so there is nothing to say and nothing to grab.
    pub(crate) normal: Option<[Stroke; 3]>,
    /// One per observation whose keypoint ray meets the frame, in the track's
    /// own observation order.
    pub(crate) marks: Vec<Mark>,
    /// How far behind the scene a fragment fades to the floor opacity, in world
    /// units: [`FOG_DISTANCE`] half-lengths, so the fade is stated in the
    /// patch's own size rather than in the scene's.
    pub(crate) fog_distance: f32,
}

impl Figure {
    /// Every stroke, in the order the pass uploads them.
    pub(crate) fn strokes(&self) -> impl Iterator<Item = &Stroke> {
        self.frame.iter().chain(self.normal.iter().flatten()).chain(
            self.marks
                .iter()
                .flat_map(|mark| std::iter::once(&mark.segment).chain(mark.circle.iter())),
        )
    }
}

/// The figure `bench`'s active track draws, or `None` when it draws nothing.
///
/// `None` for a cluster-stage item, which has no geometry behind it, and for a
/// track nothing has given a frame. `eye` is the viewport camera's position in
/// world coordinates, which the arrowhead's barbs turn to face.
pub(crate) fn figure(bench: &BenchTrack<'_>, eye: Point3<f64>) -> Option<Figure> {
    let Stage::Track(payload) = &bench.track.stage else {
        return None;
    };
    let frame = payload.placement.as_ref()?;
    // The patch's own `w` rather than the payload's `at_infinity`: what is
    // drawn is the patch, and core keeps the two equal wherever both exist.
    let at_infinity = frame.w == 0.0;
    // A patch is square, so one half-length says the whole of the size.
    let half = frame.half_extent[0];
    if !half.is_finite() || half <= 0.0 {
        return None;
    }

    let rotation = bench.transform.rotation.to_rotation_matrix();
    // A place travels through the whole similarity; a direction keeps only its
    // rotation, the translation dropping out and the uniform scale cancelling
    // in the projection's divide.
    // A direction has no depth to fight over, so only a place is lifted.
    let lift = frame.normal() * (PLANE_LIFT * half);
    let place = |xyz: Vector3<f64>| -> [f32; 4] {
        if at_infinity {
            let d = rotation * xyz;
            [d.x as f32, d.y as f32, d.z as f32, 0.0]
        } else {
            let p = bench.transform.apply_to_point(&Point3::from(xyz + lift));
            [p.x as f32, p.y as f32, p.z as f32, 1.0]
        }
    };

    let in_color = rgba(crate::bench::IN_COLOR);
    let centre = place(frame.center.coords);

    let corners: Vec<[f32; 4]> = CORNERS
        .into_iter()
        .map(|(s, t)| place(frame.corner_homogeneous(s, t).0))
        .collect();
    let frame_edges = std::array::from_fn(|k| Stroke {
        a: corners[k],
        b: corners[(k + 1) % 4],
        color: in_color,
    });

    let radius = half * CIRCLE_RADIUS;
    let disc = disc(frame, radius, &place, in_color);
    let normal = (!at_infinity).then(|| arrow(frame, half, bench.transform, &rotation, eye));

    let image_table = &bench.edited.base.image_table;
    let marks = bench
        .track
        .observations
        .iter()
        .enumerate()
        .filter_map(|(observation, sighting)| {
            let site = sighting.site()?;
            let (camera, pose) = geometry::view_of(image_table, sighting.image as usize)?;
            // The one unprojection, core's own: the keypoint's ray meets the
            // frame's plane and what it named is read off that meeting. A ray
            // that cannot meet the plane in front of its camera -- including a
            // bearing pointing away from a direction patch -- says so here.
            let offset = frame.keypoint_plane_offset(&camera, &pose, site)?;
            let q = frame.center.coords + offset;
            let color = rgba(verdict_color(sighting.verdict));
            let radius = if bench.selected == Some(observation) {
                radius * SELECTED_CIRCLE_SCALE
            } else {
                radius
            };
            Some(Mark {
                observation,
                segment: Stroke {
                    a: centre,
                    b: place(q),
                    color,
                },
                circle: circle(frame, q, radius, &place, color),
            })
        })
        .collect();

    Some(Figure {
        frame: frame_edges,
        centre,
        disc,
        normal,
        marks,
        fog_distance: (FOG_DISTANCE * half * bench.transform.scale) as f32,
    })
}

/// The track's patch, or `None` when there is nothing to take hold of: a
/// cluster-stage item, or a track nothing has given a frame.
///
/// The frame is in the **reconstruction's own** coordinates, which is where the
/// core steps act and where a pointer has to be brought back to.
pub(crate) fn placement_of(track: &EditableTrack) -> Option<&OrientedPatch> {
    match &track.stage {
        Stage::Track(payload) => payload.placement.as_ref(),
        Stage::Cluster(_) => None,
    }
}

/// The centre disc, as a triangle fan about the frame's centre laid out as a
/// triangle list.
fn disc(
    frame: &OrientedPatch,
    radius: f64,
    place: &impl Fn(Vector3<f64>) -> [f32; 4],
    color: [f32; 4],
) -> Vec<DiscVertex> {
    let centre = DiscVertex {
        at: place(frame.center.coords),
        color,
    };
    let rim: Vec<DiscVertex> = ring(frame, frame.center.coords, radius)
        .into_iter()
        .map(|at| DiscVertex {
            at: place(at),
            color,
        })
        .collect();
    (0..CIRCLE_SEGMENTS)
        .flat_map(|k| [centre, rim[k], rim[(k + 1) % CIRCLE_SEGMENTS]])
        .collect()
}

/// A hollow circle in the frame's plane, as its `CIRCLE_SEGMENTS` chords.
fn circle(
    frame: &OrientedPatch,
    at: Vector3<f64>,
    radius: f64,
    place: &impl Fn(Vector3<f64>) -> [f32; 4],
    color: [f32; 4],
) -> Vec<Stroke> {
    let rim: Vec<[f32; 4]> = ring(frame, at, radius).into_iter().map(place).collect();
    (0..CIRCLE_SEGMENTS)
        .map(|k| Stroke {
            a: rim[k],
            b: rim[(k + 1) % CIRCLE_SEGMENTS],
            color,
        })
        .collect()
}

/// `CIRCLE_SEGMENTS` points of the circle of radius `radius` about `at`, in the
/// frame's own plane.
fn ring(frame: &OrientedPatch, at: Vector3<f64>, radius: f64) -> Vec<Vector3<f64>> {
    (0..CIRCLE_SEGMENTS)
        .map(|k| {
            let angle = std::f64::consts::TAU * k as f64 / CIRCLE_SEGMENTS as f64;
            at + frame.u_axis * (radius * angle.cos()) + frame.v_axis * (radius * angle.sin())
        })
        .collect()
}

/// The normal's segment and the arrowhead's two barbs, in world coordinates.
///
/// The barbs are built after the transform rather than before it, because what
/// squares them to the viewer is the eye, which is in world coordinates. A
/// similarity preserves angles, so the head is the same shape either way.
fn arrow(
    frame: &OrientedPatch,
    half: f64,
    transform: &Se3Transform,
    rotation: &nalgebra::Matrix3<f64>,
    eye: Point3<f64>,
) -> [Stroke; 3] {
    let color = rgba(crate::bench::IN_COLOR);
    let centre = transform.apply_to_point(&frame.center);
    let normal = rotation * frame.normal();
    let length = NORMAL_LENGTH * half * transform.scale;
    let tip = centre + normal * length;
    // Perpendicular to the normal and as square to the viewer as it can be, so
    // the head never goes edge-on. When the normal points at the eye the cross
    // product vanishes and the frame's own `u` stands in.
    let side = normal.cross(&(eye - tip));
    let side = if side.norm() > 1e-12 {
        side.normalize()
    } else {
        (rotation * frame.u_axis).normalize()
    };
    let back = tip - normal * (BARB_BACK * length);
    let point = |p: Point3<f64>| [p.x as f32, p.y as f32, p.z as f32, 1.0];
    let tip = point(tip);
    [
        Stroke {
            a: point(centre),
            b: tip,
            color,
        },
        Stroke {
            a: tip,
            b: point(back + side * (BARB_SIDE * length)),
            color,
        },
        Stroke {
            a: tip,
            b: point(back - side * (BARB_SIDE * length)),
            color,
        },
    ]
}

// ---- The handles -----------------------------------------------------------

/// What the pointer has hold of in the viewport.
///
/// Not `Eq`, the arrowhead's swing carrying the axis it turns about: the gesture
/// is decided at the press and the handle is what a drag holds on to, so the
/// axis travels with it rather than being read again each frame off an eye that
/// has since moved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Handle {
    /// The centre dot: dragging it slides the patch across its own plane.
    Dot,
    /// One edge of the square: dragging it resizes the patch, holding the
    /// opposite edge still.
    Edge(Edge),
    /// One corner: dragging it turns the patch about its outward normal.
    Corner(usize),
    /// The normal's segment: dragging it moves the patch along its own normal,
    /// which is one of the two things no photograph can say -- a sighting names
    /// the ray the patch lies along and not how far down it the surface is.
    Normal,
    /// The arrowhead at the far end of that segment: dragging it turns the
    /// normal, which is the other. Which of its two gestures the drag makes was
    /// decided at the press ([`geometry::tilt_gesture`]) and is carried here,
    /// so it does not change character halfway through.
    Arrowhead(Tilt),
    /// One observation's circle. It edits nothing -- where a photograph sees the
    /// patch's content is that photograph's answer and not a thing to drag --
    /// but a click on it selects that row in Track View, as clicking a mark in
    /// the Image Detail panel does.
    Circle {
        /// The observation, by its position in the track's list.
        observation: usize,
    },
}

/// What a gesture over the figure asks of the app, once the button comes up.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BenchGesture {
    /// A drag finished: apply this edit, through the call the wire's patch
    /// tools make, as one version and one Action Log row.
    Edit(PatchEdit),
    /// A circle was clicked: select that observation's row in Track View.
    SelectRow(usize),
}

/// A handle being dragged, and where the pointer has taken it.
///
/// Both ends are **places on the handle's own geometry**, in the
/// reconstruction's coordinates, rather than pixels of the window: what the
/// gesture means is a statement about the patch, so it does not depend on where
/// the figure happened to be drawn. Which geometry depends on the handle -- the
/// frame's plane for the three that name a place on it, the normal's line for
/// the segment -- and the press decided the handle, so both ends of one drag are
/// read the same way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Drag {
    /// The node whose bench it edits. A drag that outlives a change of selected
    /// node is dropped.
    pub(crate) node: ReconId,
    /// What is being dragged.
    pub(crate) handle: Handle,
    /// Where the press met the handle's geometry.
    from: Point3<f64>,
    /// Where the pointer meets it now.
    to: Point3<f64>,
    /// Where the press landed in the panel.
    ///
    /// A press is taken as a handle before egui would call it a drag, so this is
    /// what tells the two apart at the release: a press that never moved is a
    /// click, which selects the row under it and edits nothing.
    press: Pos2,
    /// Whether the pointer has left the press at all.
    pub(crate) moved: bool,
    /// Set by Escape: the gesture is abandoned, so nothing is previewed and
    /// nothing is pushed, but the viewport still does not orbit until the button
    /// comes up.
    pub(crate) cancelled: bool,
}

impl Drag {
    /// A gesture just taken at `press`, which met the handle's geometry at
    /// `at`.
    pub(crate) fn new(node: ReconId, handle: Handle, at: Point3<f64>, press: Pos2) -> Self {
        Self {
            node,
            handle,
            from: at,
            to: at,
            press,
            moved: false,
            cancelled: false,
        }
    }

    /// Follow the pointer: it is at `pos` in the panel and, when the ray still
    /// reads against the handle's geometry, at `at` on it.
    pub(crate) fn follow(&mut self, pos: Pos2, at: Option<Point3<f64>>) {
        self.moved |= pos != self.press;
        if let Some(at) = at {
            self.to = at;
        }
    }

    /// What this drag would do to the track, in the form the core steps take.
    ///
    /// The two places it carries are already in the patch's own terms, so the
    /// readings left here are the two a place alone does not state: the turn a
    /// corner swept, and the half-length an edge's place names. `None` when the
    /// gesture names nothing the patch can be given: a cancelled drag, a circle,
    /// or a turn read about the centre itself.
    pub(crate) fn edit(&self, frame: &OrientedPatch) -> Option<PatchEdit> {
        if self.cancelled {
            return None;
        }
        match self.handle {
            // The press's own offset from the centre is kept, so a dot grabbed
            // a little off centre does not jump under the pointer.
            //
            // **The `n` component is set to zero here**, which is where that
            // constraint belongs: the two places are meetings of a ray with the
            // patch's plane and so lie in it only to their last bits, and the
            // dot is the handle that means "across the plane and nowhere else".
            // Reading the travel on `u` and `v` alone says so outright, rather
            // than leaving the step to project a displacement it was never told
            // was meant to be tangential.
            Handle::Dot => {
                let travel = self.to - self.from;
                Some(PatchEdit::Translate {
                    by: [travel.dot(&frame.u_axis), travel.dot(&frame.v_axis), 0.0],
                })
            }
            // With the dragged edge at `+h` from the centre and the far one at
            // `-h`, the half-length that puts the edge under the pointer and
            // holds the far one is `(p + h) / 2`, `p` being the place's offset
            // along that edge's own axis. Reading it on the axis drops whatever
            // component off the plane the meeting had.
            Handle::Edge(edge) => {
                let direction = match edge.axis() {
                    Axis::U => frame.u_axis,
                    Axis::V => frame.v_axis,
                } * edge.sign();
                let along = (self.to - frame.center).dot(&direction);
                Some(PatchEdit::Resize {
                    half_length: (along + frame.half_extent[0]) / 2.0,
                    moved_edge: Some(edge),
                })
            }
            Handle::Corner(_) => geometry::turn_on_plane(frame, self.from, self.to)
                .map(|angle_rad| PatchEdit::Spin { angle_rad }),
            // Both ends are points of the normal's own line, so their
            // difference along it is the whole gesture -- and it is a
            // difference, so a segment grabbed at its tip does not jump the
            // patch out to where the tip was.
            Handle::Normal => Some(PatchEdit::Translate {
                by: [0.0, 0.0, (self.to - self.from).dot(&frame.normal())],
            }),
            // Both gestures state their answer as a normal, so the core step
            // knows nothing of which one named it.
            Handle::Arrowhead(tilt) => {
                geometry::tilt_normal(frame, tilt, self.from, self.to).map(|normal| {
                    PatchEdit::Tilt {
                        normal: [normal.x, normal.y, normal.z],
                    }
                })
            }
            Handle::Circle { .. } => None,
        }
    }
}

/// The figure's handles in panel coordinates: what the pointer can take hold of
/// on the frame it is looking at.
///
/// Built from the [`Figure`] the pass drew and the camera it was drawn through,
/// so the hit test is against the picture on screen rather than against a second
/// projection written to be aimed at. It needs no pick-buffer entry, and
/// occlusion does not enter into it: a handle drawn through the point cloud is
/// grabbed like any other, the figure's depth-aware blend being about what is
/// seen and this about where the pointer is.
pub(crate) struct Handles {
    /// The centre dot, when it projected.
    dot: Option<Pos2>,
    /// The four corners in [`CORNERS`] order, `None` for one behind the eye.
    corners: [Option<Pos2>; 4],
    /// One per mark that projected: which observation it is, and where.
    circles: Vec<(usize, Pos2)>,
    /// The normal's segment, centre first and the arrow's tip second, when both
    /// ends projected. `None` for a track at infinity, which draws no normal.
    ///
    /// The tip is the **arrowhead's** own place as well: the head is drawn at
    /// the far end of this very segment, so there is one projection rather than
    /// two that could disagree.
    normal: Option<(Pos2, Pos2)>,
    /// Which gesture a press on the arrowhead would make, or `None` when there
    /// is no arrowhead to press.
    ///
    /// Carried here rather than worked out in [`Handles::hit`] because it is a
    /// reading of the **eye** against the frame, in the reconstruction's own
    /// coordinates, and neither of those is in the picture this projection is
    /// of.
    tilt: Option<Tilt>,
}

impl Handles {
    /// The figure as `camera` drew it in `rect`, with `tilt` the gesture its
    /// arrowhead would make.
    pub(crate) fn project(
        figure: &Figure,
        camera: &ViewportCamera,
        rect: Rect,
        tilt: Option<Tilt>,
    ) -> Self {
        let at = |p: [f32; 4]| {
            camera.project_homogeneous(
                Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])),
                f64::from(p[3]),
                rect,
            )
        };
        Handles {
            dot: at(figure.centre),
            corners: std::array::from_fn(|k| at(figure.frame[k].a)),
            circles: figure
                .marks
                .iter()
                .filter_map(|mark| Some((mark.observation, at(mark.segment.b)?)))
                .collect(),
            // The drawn segment's own endpoints, so what is grabbed is the line
            // on screen: the first stroke of the arrow runs centre to tip.
            normal: figure
                .normal
                .as_ref()
                .and_then(|arrow| Some((at(arrow[0].a)?, at(arrow[0].b)?))),
            tilt,
        }
    }

    /// The handle under `pos`, or `None` when the pointer is on none of them.
    ///
    /// The arrowhead, then the corners, then the dot, then the circles, then
    /// the edges, then the normal's segment. The dot and the circles sit inside
    /// the square they mark and a corner is where two edges meet, so a
    /// nearest-thing search over all of them at once would make the smaller
    /// handles unreachable; the normal's segment is last because it leaves the
    /// centre, where every other handle already is, and a person reaching for
    /// it has the whole of its length to reach for. The arrowhead is **first**
    /// for the other half of that reason: it is a point at the far end of that
    /// same segment, so anything tested before it would take every press meant
    /// for it.
    pub(crate) fn hit(&self, pos: Pos2) -> Option<Handle> {
        let nearest = |best: Option<(f32, Handle)>, distance: f32, handle: Handle| match best {
            Some((d, _)) if d <= distance => best,
            _ => Some((distance, handle)),
        };
        if let Some((tilt, (_, tip))) = self.tilt.zip(self.normal) {
            if (tip - pos).length() <= HANDLE_HIT_RADIUS {
                return Some(Handle::Arrowhead(tilt));
            }
        }
        let mut found = None;
        for (k, corner) in self.corners.iter().enumerate() {
            let Some(at) = corner else { continue };
            let distance = (*at - pos).length();
            if distance <= HANDLE_HIT_RADIUS {
                found = nearest(found, distance, Handle::Corner(k));
            }
        }
        if let Some((_, handle)) = found {
            return Some(handle);
        }
        if self
            .dot
            .is_some_and(|at| (at - pos).length() <= HANDLE_HIT_RADIUS)
        {
            return Some(Handle::Dot);
        }
        for (observation, at) in &self.circles {
            let distance = (*at - pos).length();
            if distance <= HANDLE_HIT_RADIUS {
                found = nearest(
                    found,
                    distance,
                    Handle::Circle {
                        observation: *observation,
                    },
                );
            }
        }
        if let Some((_, handle)) = found {
            return Some(handle);
        }
        for k in 0..4 {
            let (Some(a), Some(b)) = (self.corners[k], self.corners[(k + 1) % 4]) else {
                continue;
            };
            let distance = distance_to_segment(a, b, pos);
            if distance <= EDGE_HIT_WIDTH {
                found = nearest(found, distance, Handle::Edge(geometry::edge_of(k)));
            }
        }
        if let Some((_, handle)) = found {
            return Some(handle);
        }
        self.normal
            .filter(|(a, b)| distance_to_segment(*a, *b, pos) <= EDGE_HIT_WIDTH)
            .map(|_| Handle::Normal)
    }

    /// The cursor `handle` asks for, with the square's own orientation **on
    /// screen** deciding which resize cursor an edge or a corner takes.
    ///
    /// There is no rotation cursor to give a corner: egui's set is the CSS one
    /// and neither has ever had such a thing, so what a corner takes is the
    /// resize cursor lying along the way it **travels** -- the tangent of the
    /// circle it turns on, which is the perpendicular of its own radius from the
    /// centre. Since [`resize_cursor`] answers with the perpendicular of what it
    /// is handed, handing it the radius is what asks for the tangent. Running
    /// the pointer along an edge and onto the corner then turns the cursor from
    /// across the edge to along the arc, which is the difference between the two
    /// gestures. A circle selects rather than moves, so it takes the pointing
    /// hand every other selectable mark in this window takes.
    ///
    /// The normal's segment is dragged **along** itself, which is an edge's
    /// case turned around, so what it hands [`resize_cursor`] is the segment's
    /// perpendicular: the cursor wanted lies on the line, and that function
    /// answers with the perpendicular of what it is given.
    ///
    /// The arrowhead takes the cursor of the gesture it is about to make. An
    /// **aim** is free in two directions at once, which is `AllScroll`; winit
    /// draws that with the same glyph as the dot's `Move` on Windows, both
    /// landing on `IDC_SIZEALL`, and that is expected rather than a mistake --
    /// the distinction is for this code's own clarity and for the platforms
    /// that render the two apart. A **swing** travels along the arc the
    /// arrowhead turns on, which is the corner's own case: its cursor is the
    /// perpendicular of its radius from the centre, so what [`resize_cursor`]
    /// is handed is that radius.
    pub(crate) fn cursor(&self, handle: Handle) -> CursorIcon {
        match handle {
            Handle::Dot => CursorIcon::Move,
            Handle::Circle { .. } => CursorIcon::PointingHand,
            Handle::Arrowhead(Tilt::Aim) => CursorIcon::AllScroll,
            Handle::Arrowhead(Tilt::Swing(_)) => self
                .normal
                .map(|(_, tip)| tip)
                .zip(self.centre())
                .map(|(tip, centre)| resize_cursor(tip - centre))
                .unwrap_or(CursorIcon::Move),
            Handle::Normal => self
                .normal
                .map(|(centre, tip)| {
                    let along = tip - centre;
                    resize_cursor(egui::vec2(-along.y, along.x))
                })
                .unwrap_or(CursorIcon::Move),
            Handle::Corner(k) => self
                .corners
                .get(k)
                .copied()
                .flatten()
                .zip(self.centre())
                .map(|(corner, centre)| resize_cursor(corner - centre))
                .unwrap_or(CursorIcon::Move),
            Handle::Edge(edge) => (0..4)
                .find(|k| geometry::edge_of(*k) == edge)
                .and_then(|k| Some(resize_cursor(self.corners[(k + 1) % 4]? - self.corners[k]?)))
                .unwrap_or(CursorIcon::Move),
        }
    }

    /// The figure's centre on screen: the dot, or the mean of the corners that
    /// projected when it did not.
    ///
    /// A corner's radius is measured from here. The fallback matters because the
    /// dot is one point and can be the one behind the eye, while a corner the
    /// pointer is on has to have projected to be under it.
    fn centre(&self) -> Option<Pos2> {
        if let Some(dot) = self.dot {
            return Some(dot);
        }
        let placed: Vec<Pos2> = self.corners.iter().flatten().copied().collect();
        let sum = placed
            .iter()
            .fold(egui::Vec2::ZERO, |sum, at| sum + at.to_vec2());
        (!placed.is_empty()).then(|| (sum / placed.len() as f32).to_pos2())
    }
}

/// One of the bench's violets as the pass takes it.
fn rgba(color: Color32) -> [f32; 4] {
    let [r, g, b, a] = color.to_array();
    [
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
        f32::from(a) / 255.0,
    ]
}

/// Crate-visible: the pass's own tests upload the figures these build.
#[cfg(test)]
pub(crate) mod tests;
