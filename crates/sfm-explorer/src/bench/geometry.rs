// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a pointer means against a patch, and the one edit each answer is.
//!
//! The bench handles of both panels and the wire's patch tools ask the same
//! questions -- where is this sighting, how large is this patch, which way up is
//! it, how far off is it -- and they have to agree to the pixel, because a panel
//! draws the answer while the tool states it. So all of them build the same
//! [`PatchEdit`] and hand it to the same [`apply`], which is one core step
//! each. The last of the questions is the 3D viewport's alone: a photograph
//! names the ray the patch lies along and not how far down it the surface is.
//!
//! The arithmetic that turns a pointer into a patch is core's
//! (`sfmtool_core::bench::resize_from_edge`,
//! `OrientedPatch::keypoint_plane_offset`): a pixel is a ray, the ray meets the
//! patch's own plane, and what the pointer named is read off that meeting.
//! Nothing here approximates the projection -- an edge put under the pointer
//! reprojects onto the pointer, through whatever distortion the lens has,
//! because the point it was built from is the one the lens maps there. What is
//! left for this module is the two readings the core steps do not take as
//! inputs: the turn a corner drag swept, and the size a sentence reports.

use nalgebra::{Point3, Vector3};
use sfmtool_core::bench::{
    self, Edge, EditableTrack, MoveObservationReport, Observation, OffsetFrameReport, ResizeReport,
    RotateFrameReport, ShapeReport, TrackEditError, TranslateFrameReport, TranslateToReport,
};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::{EditedReconstruction, ImageTable};

#[cfg(test)]
mod tests;

/// An in-plane offset this small names no direction, so no turn can be read off
/// it.
const MIN_OFFSET: f64 = 1e-12;

/// One hand edit of a track's geometry.
///
/// What a drag of a bench handle in either panel produces and what each of the
/// wire's patch tools names, in the form the core steps take: a pixel of one
/// image or a place in the world for the gestures that name one, an angle for
/// the two that turn something, and a signed length for the one that settles a
/// depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PatchEdit {
    /// Slide the track-stage surfel across its own plane until its centre sits
    /// under this pixel of that observation's image. Every sighting follows.
    Translate {
        /// The observation whose image the pixel is in.
        observation: usize,
        /// Where, in that image's own px.
        pixel: [f64; 2],
    },
    /// Put one observation's own sighting at this pixel of its own image, and
    /// leave every other where it is. The cluster stage's dot, where there is
    /// no shared geometry to move.
    Move {
        /// The observation, by its position in the track's list.
        observation: usize,
        /// Where, in that image's own px.
        pixel: [f64; 2],
    },
    /// Put one edge of the outline drawn at this observation under this pixel,
    /// with the opposite edge left where it is.
    ResizeFromEdge {
        /// The observation whose sighting the outline is drawn at.
        observation: usize,
        /// Which edge was grabbed.
        edge: Edge,
        /// Where its midpoint should land, in that image's own px.
        pixel: [f64; 2],
    },
    /// Slide the track-stage surfel across its own plane until its centre sits
    /// at this place. Every sighting follows.
    ///
    /// The 3D viewport's centre-dot drag, where there is no photograph to name
    /// a pixel of and the square a person takes hold of is the surfel itself.
    SlideTo {
        /// Where, in the reconstruction's own coordinates.
        point: [f64; 3],
    },
    /// Put one edge of the surfel's square at this place, with the opposite
    /// edge left where it is. The 3D viewport's edge drag.
    ResizeFromEdgeTo {
        /// Which edge was grabbed.
        edge: Edge,
        /// Where it should lie, in the reconstruction's own coordinates.
        point: [f64; 3],
    },
    /// Move the track-stage surfel this many world units along its outward
    /// normal, positive toward the face it shows.
    ///
    /// The 3D viewport's normal-segment drag, and the one edit no photograph
    /// can name: a sighting says which ray the patch lies along and nothing
    /// about how far down it the surface is.
    Offset {
        /// How far, in the reconstruction's own units.
        distance: f64,
    },
    /// Turn the track-stage surfel by this many radians about its normal.
    Rotate {
        /// The turn, positive about the outward normal.
        angle_rad: f64,
    },
    /// Turn one cluster-stage sighting's shape by this many radians in its own
    /// image's pixels.
    RotateShape {
        /// The observation, by its position in the track's list.
        observation: usize,
        /// The turn, positive from `+x` toward `+y` of the image raster.
        angle_rad: f64,
    },
}

/// What one [`PatchEdit`] did, as the core step's own report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EditReport {
    /// The surfel slid across its plane, every sighting following.
    Translated(TranslateFrameReport),
    /// The same slide, named as a place in the world rather than as a pixel.
    SlidTo(TranslateToReport),
    /// One sighting placed by hand.
    Moved(MoveObservationReport),
    /// The surfel moved along its own normal, every sighting following.
    Offset(OffsetFrameReport),
    /// The patch resized by one of its edges.
    Resized(ResizeReport),
    /// The surfel turned.
    Rotated(RotateFrameReport),
    /// One sighting's shape turned.
    Turned {
        /// The step's own report.
        report: ShapeReport,
        /// How far it turned, in degrees.
        degrees: f64,
    },
}

/// Below this many degrees a cluster-stage turn is no turn: the same tolerance
/// core judges a surfel's turn by ([`bench::rotate_frame`]), read in the unit
/// this report carries.
const NO_EFFECT_DEG: f64 = 1e-9_f64.to_degrees();

impl EditReport {
    /// Whether anything changed. A drag that ends where it started pushes no
    /// version, the way a verdict an observation already holds does.
    ///
    /// Every answer is the core step's own, which is where the tolerance lives:
    /// the pixel a pointer names is unprojected onto the patch's plane and
    /// projected back, and that round trip does not return bit for bit, so an
    /// exact comparison reads a re-statement of where the patch already is as a
    /// move (`specs/gui/bench.md` section "The wire").
    pub(crate) fn changed(&self) -> bool {
        match self {
            EditReport::Translated(report) => report.changed,
            EditReport::SlidTo(report) => report.changed,
            EditReport::Offset(report) => report.changed,
            EditReport::Moved(report) => report.changed,
            EditReport::Resized(report) => report.changed,
            EditReport::Rotated(report) => report.changed,
            EditReport::Turned { report, degrees } => {
                report.changed && degrees.abs() > NO_EFFECT_DEG
            }
        }
    }

    /// The pixel the gesture landed on, for the three edits that name one.
    ///
    /// The pixel **used**, which is the clamped one where the caller named a
    /// place off the photograph: the reply and the log row both say where the
    /// patch went rather than where it was aimed.
    pub(crate) fn pixel(&self) -> Option<[f64; 2]> {
        match self {
            EditReport::Translated(report) => Some(report.pixel),
            EditReport::Moved(report) => Some(report.pixel),
            EditReport::Resized(report) => report.pixel,
            EditReport::SlidTo(_)
            | EditReport::Offset(_)
            | EditReport::Rotated(_)
            | EditReport::Turned { .. } => None,
        }
    }

    /// The pixel that was asked for, when it sat off the photograph and the step
    /// brought it inside.
    pub(crate) fn clamped_from(&self) -> Option<[f64; 2]> {
        match self {
            EditReport::Translated(report) => report.clamped_from,
            EditReport::Moved(report) => report.clamped_from,
            EditReport::Resized(report) => report.clamped_from,
            EditReport::SlidTo(_)
            | EditReport::Offset(_)
            | EditReport::Rotated(_)
            | EditReport::Turned { .. } => None,
        }
    }

    /// The sentence a step that had no effect records, naming what was asked for
    /// and why nothing came of it.
    ///
    /// The step's **own** sentence, one per gesture, because the Action Log row
    /// and the reply both carry it and a reader of either should be able to tell
    /// which gesture it was without a version to read it off.
    pub(crate) fn no_effect_sentence(&self, label: &str) -> String {
        match self {
            EditReport::Translated(_) | EditReport::SlidTo(_) => {
                format!("Moved {label}: no effect, the patch already sits there")
            }
            // Its own sentence, because a reader of the Action Log should be
            // able to tell an offset from a slide without a version to read it
            // off: the two gestures move the patch in different directions and
            // one of them is not in the plane.
            EditReport::Offset(_) => {
                format!("Moved {label} along its normal: no effect, the patch already stands there")
            }
            EditReport::Moved(report) => format!(
                "Moved observation {} of {label}: no effect, the sighting already sits there",
                report.observation
            ),
            EditReport::Resized(_) => {
                format!("Resized {label}: no effect, the patch is already that size")
            }
            EditReport::Rotated(_) => format!("Rotated {label}: no effect, no turn was asked for"),
            EditReport::Turned { report, .. } => format!(
                "Rotated observation {} of {label}: no effect, no turn was asked for",
                report.observation
            ),
        }
    }
}

/// Apply one edit to a track, giving the next track and what the step
/// reported.
///
/// The one door both callers go through, and the one the panel's live preview
/// goes through too: what the outline looks like mid-drag is the track this
/// function would produce from the pointer where it is, so what a person
/// releases on is what lands.
pub(crate) fn apply(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    edit: &PatchEdit,
) -> Result<(EditableTrack, EditReport), TrackEditError> {
    match *edit {
        PatchEdit::Translate { observation, pixel } => {
            let (next, report) = bench::translate_frame(track, edited, observation, pixel)?;
            Ok((next, EditReport::Translated(report)))
        }
        PatchEdit::Move { observation, pixel } => {
            let (next, report) =
                bench::set_observation_keypoint(track, edited, observation, pixel)?;
            Ok((next, EditReport::Moved(report)))
        }
        PatchEdit::ResizeFromEdge {
            observation,
            edge,
            pixel,
        } => {
            let (next, report) = bench::resize_from_edge(track, edited, observation, edge, pixel)?;
            Ok((next, EditReport::Resized(report)))
        }
        PatchEdit::SlideTo { point } => {
            let (next, report) = bench::translate_frame_to(track, edited, Point3::from(point))?;
            Ok((next, EditReport::SlidTo(report)))
        }
        PatchEdit::ResizeFromEdgeTo { edge, point } => {
            let (next, report) =
                bench::resize_from_edge_to(track, edited, edge, Point3::from(point))?;
            Ok((next, EditReport::Resized(report)))
        }
        PatchEdit::Offset { distance } => {
            let (next, report) = bench::offset_frame(track, edited, distance)?;
            Ok((next, EditReport::Offset(report)))
        }
        PatchEdit::Rotate { angle_rad } => {
            let (next, report) = bench::rotate_frame(track, angle_rad)?;
            Ok((next, EditReport::Rotated(report)))
        }
        PatchEdit::RotateShape {
            observation,
            angle_rad,
        } => {
            if !angle_rad.is_finite() {
                return Err(TrackEditError::BadAngle(angle_rad));
            }
            let shape = turned_shape(track, observation, angle_rad)?;
            let (next, report) = bench::set_observation_shape(track, observation, shape)?;
            Ok((
                next,
                EditReport::Turned {
                    report,
                    degrees: angle_rad.to_degrees(),
                },
            ))
        }
    }
}

/// One observation's shape turned by `angle_rad` in its own image's pixels.
///
/// `R(angle) * shape`, which turns every offset the shape maps by that angle
/// about the sighting: the parallelogram spins in place and keeps its size and
/// its shear.
fn turned_shape(
    track: &EditableTrack,
    observation: usize,
    angle_rad: f64,
) -> Result<[[f64; 2]; 2], TrackEditError> {
    let shape = track
        .observations
        .get(observation)
        .ok_or(TrackEditError::NoSuchObservation {
            observation,
            observation_count: track.observations.len(),
        })?
        .shape()
        .ok_or(TrackEditError::NoPlace { observation })?;
    let (sin, cos) = angle_rad.sin_cos();
    Ok([
        [
            cos * shape[0][0] - sin * shape[1][0],
            cos * shape[0][1] - sin * shape[1][1],
        ],
        [
            sin * shape[0][0] + cos * shape[1][0],
            sin * shape[0][1] + cos * shape[1][1],
        ],
    ])
}

// ---- Reading a pointer against a projected patch ---------------------------

/// The camera and the pose of one image, or `None` when the image or its camera
/// is not in the table.
pub(crate) fn view_of(
    image_table: &ImageTable,
    image: usize,
) -> Option<(CameraIntrinsics, RigidTransform)> {
    let row = image_table.images.get(image)?;
    let camera = image_table.cameras.get(row.camera_index as usize)?;
    Some((camera.clone(), crate::scene::cam_from_world(row)))
}

/// Project a homogeneous world point into the view, or `None` when it falls
/// behind the camera or outside the lens model's domain.
///
/// The frame test is deliberately absent: a corner that projects a little
/// outside the photograph is a corner off-screen, which the panel clips, and
/// not a sample that failed to project.
pub(crate) fn project(
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

/// The track's surfel re-anchored on where `observation` sits in its
/// photograph, or the surfel itself when the ray cannot meet it.
///
/// The frame the Image Detail layer outlines, so it is the frame a drag of that
/// outline is read against: a pointer is judged against the square a person can
/// see, not against the one the 3D position alone would draw. The same frame
/// `resize_from_edge` reads and writes.
pub(crate) fn anchored_frame(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
    observation: &Observation,
) -> OrientedPatch {
    observation
        .site()
        .and_then(|site| frame.anchored_at_keypoint(camera, pose, site))
        .unwrap_or_else(|| frame.clone())
}

/// The turn, in radians about the patch's outward normal, that carries the
/// in-plane direction of `from` onto that of `to`.
///
/// Both pixels are unprojected onto the patch's plane and read on its axes, so
/// the angle is the one in the **patch**, not the one on screen: a patch seen
/// at a slant turns by what the person aimed at on its own surface rather than
/// by the foreshortened angle the pointer swept.
pub(crate) fn turn_between(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
    from: [f64; 2],
    to: [f64; 2],
) -> Option<f64> {
    let start = in_plane(frame, camera, pose, from)?;
    let now = in_plane(frame, camera, pose, to)?;
    Some(wrapped(now.1.atan2(now.0) - start.1.atan2(start.0)))
}

/// An angle in `(-pi, pi]`, so a drag across the frame's own `-u` axis reads as
/// the small turn it looks like rather than as its explement.
fn wrapped(angle: f64) -> f64 {
    angle - std::f64::consts::TAU * (angle / std::f64::consts::TAU).round()
}

/// One pixel's in-plane offset from the patch's centre, on the patch's own
/// axes, or `None` when the ray cannot meet the patch or meets it at its
/// centre, where no direction is named.
fn in_plane(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
    pixel: [f64; 2],
) -> Option<(f64, f64)> {
    let offset = frame.keypoint_plane_offset(camera, pose, pixel)?;
    let pair = (offset.dot(&frame.u_axis), offset.dot(&frame.v_axis));
    (pair.0.hypot(pair.1) > MIN_OFFSET).then_some(pair)
}

/// The turn, in radians, from `from` to `to` about `position`, in one image's
/// own pixels: the cluster stage's rotation, which has no plane to read
/// anything on and so is the angle the pointer swept.
pub(crate) fn pixel_turn_between(position: [f64; 2], from: [f64; 2], to: [f64; 2]) -> Option<f64> {
    let start = (from[0] - position[0], from[1] - position[1]);
    let now = (to[0] - position[0], to[1] - position[1]);
    if start.0.hypot(start.1) <= MIN_OFFSET || now.0.hypot(now.1) <= MIN_OFFSET {
        return None;
    }
    Some(wrapped(now.1.atan2(now.0) - start.1.atan2(start.0)))
}

/// Which edge of the patch's square the `k`th edge of its boundary is.
///
/// The boundary walks the corners `(-1, -1)`, `(1, -1)`, `(1, 1)`, `(-1, 1)`,
/// so its first edge runs along `t = -1` and its third along `t = +1`; the other
/// two are `s = ±1`. Here rather than in either layer, because both draw the
/// square in that order and a pointer on the `k`th edge has to name the same
/// edge to the same core step whichever panel it is in.
pub(crate) fn edge_of(k: usize) -> Edge {
    match k {
        0 => Edge::MinusV,
        1 => Edge::PlusU,
        2 => Edge::PlusV,
        _ => Edge::MinusU,
    }
}

// ---- Reading a pointer of the 3D viewport against a patch ------------------

/// How square-on the frame's plane has to be seen before a pointer can be read
/// against it, in degrees.
///
/// Three of the viewport's handles -- the dot, an edge and a corner -- name a
/// point of that plane, and a plane seen edge-on turns a pixel of pointer motion
/// into an unbounded distance along it. Under this angle they take no press at
/// all, which is a refusal the person can see (the figure is a line) rather than
/// a patch flung across the reconstruction.
pub(crate) const MIN_PLANE_ANGLE_DEG: f64 = 5.0;

/// How far from a direction patch's bearing a pointer may point and still name a
/// place on its tangent plane, in degrees.
///
/// The tangent plane is never edge-on, the viewer being in effect at the
/// sphere's centre, so this is the one refusal a track at infinity has: at a
/// right angle to the bearing `r . d` goes to zero and the tangent point runs
/// off to infinity.
pub(crate) const MAX_BEARING_ANGLE_DEG: f64 = 85.0;

/// A ray direction this short names no direction.
const MIN_DIRECTION: f64 = 1e-12;

/// Where the ray from `origin` along `direction` meets the frame, in the
/// frame's own coordinates.
///
/// The counterpart of `OrientedPatch::keypoint_plane_offset` for a pointer that
/// is already a ray rather than a pixel of a photograph, and the one place the
/// 3D viewport's three plane handles read the pointer.
///
/// For a finite frame this is the ray's meeting with the plane the square lies
/// in, in front of the eye. For a **direction patch** (`w == 0`) the origin
/// drops out -- a direction has no parallax, and the viewer is in effect at the
/// sphere's centre -- so what meets the tangent plane is the ray's direction
/// alone, at `r / (r . d)`, which is the same reading an observation's keypoint
/// gets. `None` when the ray runs along the plane, meets it behind the eye, or
/// points more than [`MAX_BEARING_ANGLE_DEG`] from the bearing.
pub(crate) fn plane_point(
    frame: &OrientedPatch,
    origin: Point3<f64>,
    direction: Vector3<f64>,
) -> Option<Point3<f64>> {
    let length = direction.norm();
    if !length.is_finite() || length < MIN_DIRECTION {
        return None;
    }
    let ray = direction / length;
    if frame.w == 0.0 {
        let bearing = frame.center.coords.normalize();
        let along = ray.dot(&bearing);
        if along <= MAX_BEARING_ANGLE_DEG.to_radians().cos() {
            return None;
        }
        return Some(Point3::from(ray / ray.dot(&frame.center.coords)));
    }
    let normal = frame.normal();
    let denominator = ray.dot(&normal);
    if denominator.abs() < MIN_DIRECTION {
        return None;
    }
    let distance = (frame.center - origin).dot(&normal) / denominator;
    (distance > 0.0 && distance.is_finite()).then(|| origin + ray * distance)
}

/// Whether the frame's plane is too near edge-on from `eye` for a pointer to be
/// read against it: the angle between the view ray through the centre and the
/// plane is under [`MIN_PLANE_ANGLE_DEG`].
///
/// Never true for a **direction patch**, whose tangent plane faces the eye by
/// construction: the bearing is read rotation-only, so the viewer stands at the
/// centre of the sphere it is tangent to.
pub(crate) fn plane_is_edge_on(frame: &OrientedPatch, eye: Point3<f64>) -> bool {
    if frame.w == 0.0 {
        return false;
    }
    let Some(cosine) = view_cosine(frame, eye) else {
        return true;
    };
    // The sine of the angle between the ray and the plane is its cosine against
    // the normal, so this is that angle without an `asin`.
    cosine < MIN_PLANE_ANGLE_DEG.to_radians().sin()
}

/// Whether the frame's normal is too near end-on from `eye` for a pointer to be
/// read along it: the angle between the normal and the view ray through the
/// centre is under [`MIN_PLANE_ANGLE_DEG`].
///
/// The mirror of [`plane_is_edge_on`], on the same bar and on the same cosine:
/// one refuses the view where that cosine goes to zero and this refuses the
/// view where it goes to one, so the view in which the plane handles die is the
/// view in which the normal is at its best and the other way about. Between
/// them some handle always takes a press.
///
/// The **line** is undirected, so a view straight down the normal and a view
/// straight up it are equally bad and the test is on the magnitude. What goes
/// wrong there is [`normal_line_point`]'s divide: its denominator is the squared
/// sine of this angle, which is what makes a pixel of pointer motion an
/// unbounded distance along the line.
///
/// Always true for a **direction patch**, whose normal is its own bearing:
/// there is no line standing off the frame to take hold of.
pub(crate) fn normal_is_end_on(frame: &OrientedPatch, eye: Point3<f64>) -> bool {
    if frame.w == 0.0 {
        return true;
    }
    let Some(cosine) = view_cosine(frame, eye) else {
        return true;
    };
    cosine > MIN_PLANE_ANGLE_DEG.to_radians().cos()
}

/// The magnitude of the cosine between the frame's normal and the view ray
/// through its centre, or `None` when `eye` names no view of it.
///
/// The one number both degenerate-view tests are decided by, which is why they
/// are complementary rather than merely similar.
fn view_cosine(frame: &OrientedPatch, eye: Point3<f64>) -> Option<f64> {
    let view = frame.center - eye;
    let length = view.norm();
    (length.is_finite() && length >= MIN_DIRECTION)
        .then(|| (view.dot(&frame.normal()) / length).abs())
}

/// The point of the frame's normal line `c + t n` nearest the ray from `origin`
/// along `direction`.
///
/// Where the 3D viewport's normal-segment drag reads the pointer, and the
/// counterpart of [`plane_point`] for the one handle that does not name a place
/// on the plane. A point rather than the parameter `t`, so the press and the
/// pointer are the same kind of thing as the plane handles' two places and the
/// drag carries them in one pair; the signed distance the gesture means is their
/// difference along `n`, which is the one reading left.
///
/// The two lines rarely meet, so what is taken is the nearest point of the
/// **line** -- the closest-approach solve, whose denominator `1 - (d . n)^2` is
/// the squared sine of the angle between the ray and the normal. That is the
/// quantity [`normal_is_end_on`] refuses at the press, so the guard here is for
/// the rays a live view can still throw rather than a second policy.
///
/// `None` for a **direction patch**, which has no normal to move along.
pub(crate) fn normal_line_point(
    frame: &OrientedPatch,
    origin: Point3<f64>,
    direction: Vector3<f64>,
) -> Option<Point3<f64>> {
    if frame.w == 0.0 {
        return None;
    }
    let length = direction.norm();
    if !length.is_finite() || length < MIN_DIRECTION {
        return None;
    }
    let ray = direction / length;
    let normal = frame.normal();
    let along = ray.dot(&normal);
    let denominator = 1.0 - along * along;
    if denominator < MIN_DIRECTION {
        return None;
    }
    let to_centre = frame.center - origin;
    let t = (to_centre.dot(&ray) * along - to_centre.dot(&normal)) / denominator;
    t.is_finite().then(|| frame.center + normal * t)
}

/// The turn, in radians about the patch's outward normal, that carries the
/// in-plane direction of `from` onto that of `to`, both read about the frame's
/// centre.
///
/// The 3D viewport's corner drag, and the counterpart of [`turn_between`] for a
/// pointer that is already a point of the patch's plane rather than a pixel.
/// `None` when either place is the centre itself, where no direction is named.
pub(crate) fn turn_on_plane(
    frame: &OrientedPatch,
    from: Point3<f64>,
    to: Point3<f64>,
) -> Option<f64> {
    let start = on_axes(frame, from)?;
    let now = on_axes(frame, to)?;
    Some(wrapped(now.1.atan2(now.0) - start.1.atan2(start.0)))
}

/// One place's offset from the frame's centre, on the frame's own axes, or
/// `None` when it names no direction in the plane.
fn on_axes(frame: &OrientedPatch, point: Point3<f64>) -> Option<(f64, f64)> {
    let offset = point - frame.center;
    let pair = (offset.dot(&frame.u_axis), offset.dot(&frame.v_axis));
    (pair.0.hypot(pair.1) > MIN_OFFSET).then_some(pair)
}

/// How wide the patch is in this image's own pixels: how far the projection of
/// its `+u` edge's midpoint sits from the projection of its centre.
///
/// The number a size is reported in, because a world half-length says nothing
/// to a person looking at a photograph. `None` when either projection fails.
pub(crate) fn half_width_px(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
) -> Option<f64> {
    let (center, w) = frame.corner_homogeneous(0.0, 0.0);
    let center = project(camera, pose, center, w)?;
    let (edge, w) = frame.corner_homogeneous(1.0, 0.0);
    let edge = project(camera, pose, edge, w)?;
    Some((edge[0] - center[0]).hypot(edge[1] - center[1]))
}
