// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a pointer means against a patch, and the one edit each answer is.
//!
//! The bench handles of both panels and the wire's patch tools ask the same
//! questions -- where is this sighting, how large is this patch, which way up is
//! it, how far off is it -- and they have to agree to the pixel, because a panel
//! draws the answer while the tool states it. So all of them build the same
//! [`PatchEdit`] and hand it to the same [`apply`], which is one core step
//! each. The last two of the questions are the 3D viewport's alone: a
//! photograph names the ray the patch lies along, and says nothing about how
//! far down it the surface is or which way it faces.
//!
//! The arithmetic that turns a pointer into a patch is core's
//! (`sfmtool_core::bench::resize_patch_to_pixel`,
//! `OrientedPatch::keypoint_plane_offset`): a pixel is a ray, the ray meets the
//! patch's own plane, and what the pointer named is read off that meeting.
//! Nothing here approximates the projection -- an edge put under the pointer
//! reprojects onto the pointer, through whatever distortion the lens has,
//! because the point it was built from is the one the lens maps there. What is
//! left for this module is the two readings the core steps do not take as
//! inputs: the turn a corner drag swept, and the size a sentence reports.

use nalgebra::{Point3, Rotation3, Unit, Vector3};
use sfmtool_core::bench::{
    self, Edge, EditableTrack, Observation, ResizeReport, ShapeReport, SightReport, SpinReport,
    TiltReport, TrackEditError, TranslateReport, TranslateToPixelReport,
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
/// wire's patch tools names, in the form the core steps take: a displacement on
/// the patch's own axes or a pixel of one image for the gestures that move it, a
/// world half-length or a pixel for the ones that size it, an angle for the two
/// that turn something in a plane, and a direction for the one that settles
/// which way the surface faces.
///
/// **Each word is said once along the path.** The enclosing type already says
/// this is an edit of a patch, so the variants are the verbs alone; only the
/// wire, whose namespace is flat, spells `translate_bench_patch` out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PatchEdit {
    /// Slide the track-stage patch across its own plane until its centre sits
    /// under this pixel of that observation's image. Every sighting follows.
    /// The track stage's dot while Track View's *Lock* is ticked.
    TranslateToPixel {
        /// The observation whose image the pixel is in.
        observation: usize,
        /// Where, in that image's own px.
        pixel: [f64; 2],
    },
    /// Move the track-stage patch by this displacement on its own orthonormal
    /// axes `[u, v, n]`, in world units. Every sighting follows.
    ///
    /// The 3D viewport's centre-dot drag names a tangential displacement and its
    /// normal-segment drag a normal one; the normal part is the one thing no
    /// photograph can name, a sighting saying which ray the patch lies along and
    /// nothing about how far down it the surface is.
    Translate {
        /// `[u, v, n]`, in the reconstruction's own units.
        by: [f64; 3],
    },
    /// Resize the track-stage patch to this world half-length, holding the far
    /// edge when `moved_edge` names one and the centre when it does not.
    Resize {
        /// The new half-length, in the reconstruction's own units.
        half_length: f64,
        /// Which edge moves, or `None` for both about a held centre.
        moved_edge: Option<Edge>,
    },
    /// Put one edge of the outline drawn at this observation under this pixel,
    /// with the opposite edge left where it is.
    ///
    /// The one size gesture that spans both stages, a pixel being meaningful at
    /// either where a world half-length is not.
    ResizeToPixel {
        /// The observation whose sighting the outline is drawn at.
        observation: usize,
        /// Which edge was grabbed.
        edge: Edge,
        /// Where its midpoint should land, in that image's own px.
        pixel: [f64; 2],
    },
    /// Turn the track-stage patch by this many radians about its own normal.
    Spin {
        /// The turn, positive about the outward normal.
        angle_rad: f64,
    },
    /// Turn one cluster-stage sighting's shape by this many radians in its own
    /// image's pixels.
    SpinShape {
        /// The observation, by its position in the track's list.
        observation: usize,
        /// The turn, positive from `+x` toward `+y` of the image raster.
        angle_rad: f64,
    },
    /// Turn the track-stage patch about its centre until it faces this outward
    /// normal, by the least rotation and no further than the observations can
    /// still see it.
    ///
    /// The 3D viewport's arrowhead drag, and the other edit no photograph can
    /// name: a sighting says which ray the patch lies along and nothing about
    /// which way the surface under it faces.
    Tilt {
        /// The outward normal wanted, in the reconstruction's own coordinates.
        /// Any non-zero length: the step reads the direction.
        normal: [f64; 3],
    },
    /// Put one observation's own sighting at this pixel of its own image, and
    /// leave every other where it is. The cluster stage's dot, where there is
    /// no shared geometry to move, and the track stage's with Track View's
    /// *Lock* cleared, where the patch stays put and one keypoint moves.
    Sight {
        /// The observation, by its position in the track's list.
        observation: usize,
        /// Where, in that image's own px.
        pixel: [f64; 2],
    },
    /// Give one cluster-stage sighting this affine shape outright.
    Shape {
        /// The observation, by its position in the track's list.
        observation: usize,
        /// Keypoint-frame units to that image's pixels.
        shape: [[f64; 2]; 2],
    },
}

/// What one [`PatchEdit`] did, as the core step's own report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EditReport {
    /// The patch slid across its plane to a pixel, every sighting following.
    TranslatedToPixel(TranslateToPixelReport),
    /// The patch moved on its own axes, every sighting following.
    Translated(TranslateReport),
    /// The patch resized, by one of its edges or about its centre.
    Resized(ResizeReport),
    /// The patch turned about its own normal.
    Spun(SpinReport),
    /// One cluster sighting's shape turned.
    SpunShape {
        /// The step's own report.
        report: ShapeReport,
        /// How far it turned, in degrees.
        degrees: f64,
    },
    /// The patch turned to face a new normal, every sighting rebuilt on the
    /// turned axes.
    Tilted(TiltReport),
    /// One sighting placed by hand.
    Sighted(SightReport),
    /// One cluster sighting's shape set by hand.
    Shaped(ShapeReport),
}

/// Below this many degrees a cluster-stage turn is no turn: the same tolerance
/// core judges a patch's turn by ([`bench::spin_patch`]), read in the unit this
/// report carries.
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
            EditReport::TranslatedToPixel(report) => report.changed,
            EditReport::Translated(report) => report.changed,
            EditReport::Tilted(report) => report.changed,
            EditReport::Sighted(report) => report.changed,
            EditReport::Resized(report) => report.changed,
            EditReport::Spun(report) => report.changed,
            EditReport::Shaped(report) => report.changed,
            EditReport::SpunShape { report, degrees } => {
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
            EditReport::TranslatedToPixel(report) => Some(report.pixel),
            EditReport::Sighted(report) => Some(report.pixel),
            EditReport::Resized(report) => report.pixel,
            EditReport::Translated(_)
            | EditReport::Tilted(_)
            | EditReport::Spun(_)
            | EditReport::Shaped(_)
            | EditReport::SpunShape { .. } => None,
        }
    }

    /// The pixel that was asked for, when it sat off the photograph and the step
    /// brought it inside.
    pub(crate) fn clamped_from(&self) -> Option<[f64; 2]> {
        match self {
            EditReport::TranslatedToPixel(report) => report.clamped_from,
            EditReport::Sighted(report) => report.clamped_from,
            EditReport::Resized(report) => report.clamped_from,
            EditReport::Translated(_)
            | EditReport::Tilted(_)
            | EditReport::Spun(_)
            | EditReport::Shaped(_)
            | EditReport::SpunShape { .. } => None,
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
            EditReport::TranslatedToPixel(_) => {
                format!("Moved {label}: no effect, the patch already sits there")
            }
            // One step moves the patch in two directions, and a reader of the
            // Action Log should be able to tell them apart without a version to
            // read it off: a displacement along the normal is a statement about
            // depth, which is not a thing any photograph could have said.
            EditReport::Translated(report) => {
                if report.by.x == 0.0 && report.by.y == 0.0 && report.by.z != 0.0 {
                    format!(
                        "Moved {label} along its normal: no effect, the patch already stands there"
                    )
                } else {
                    format!("Moved {label}: no effect, the patch already sits there")
                }
            }
            // Two different nothings, and a reader should be able to tell them
            // apart: a tilt held against the cap of an observation that is
            // already looking along the surface turns by nothing, and that is a
            // fact about the sightings rather than about what was asked for.
            EditReport::Tilted(report) => match report.stopped {
                Some(_) => format!(
                    "Tilted {label}: no effect, the patch already faces as far over as its \
                     observations allow"
                ),
                None => format!("Tilted {label}: no effect, no turn was asked for"),
            },
            EditReport::Sighted(report) => format!(
                "Moved observation {} of {label}: no effect, the sighting already sits there",
                report.observation
            ),
            EditReport::Resized(_) => {
                format!("Resized {label}: no effect, the patch is already that size")
            }
            EditReport::Spun(_) => format!("Spun {label}: no effect, no turn was asked for"),
            EditReport::SpunShape { report, .. } => format!(
                "Spun observation {} of {label}: no effect, no turn was asked for",
                report.observation
            ),
            EditReport::Shaped(report) => format!(
                "Shaped observation {} of {label}: no effect, the sighting already has that shape",
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
        PatchEdit::TranslateToPixel { observation, pixel } => {
            let (next, report) =
                bench::translate_patch_to_pixel(track, edited, observation, pixel)?;
            Ok((next, EditReport::TranslatedToPixel(report)))
        }
        PatchEdit::Translate { by } => {
            let (next, report) = bench::translate_patch(track, edited, Vector3::from(by))?;
            Ok((next, EditReport::Translated(report)))
        }
        PatchEdit::Resize {
            half_length,
            moved_edge,
        } => {
            let (next, report) = bench::resize_patch(track, edited, half_length, moved_edge)?;
            Ok((next, EditReport::Resized(report)))
        }
        PatchEdit::ResizeToPixel {
            observation,
            edge,
            pixel,
        } => {
            let (next, report) =
                bench::resize_patch_to_pixel(track, edited, observation, edge, pixel)?;
            Ok((next, EditReport::Resized(report)))
        }
        PatchEdit::Spin { angle_rad } => {
            let (next, report) = bench::spin_patch(track, angle_rad)?;
            Ok((next, EditReport::Spun(report)))
        }
        PatchEdit::Tilt { normal } => {
            let (next, report) = bench::tilt_patch(track, edited, Vector3::from(normal))?;
            Ok((next, EditReport::Tilted(report)))
        }
        PatchEdit::Sight { observation, pixel } => {
            let (next, report) = bench::sight_observation(track, edited, observation, pixel)?;
            Ok((next, EditReport::Sighted(report)))
        }
        PatchEdit::Shape { observation, shape } => {
            let (next, report) = bench::shape_observation(track, observation, shape)?;
            Ok((next, EditReport::Shaped(report)))
        }
        PatchEdit::SpinShape {
            observation,
            angle_rad,
        } => {
            if !angle_rad.is_finite() {
                return Err(TrackEditError::BadAngle(angle_rad));
            }
            let shape = turned_shape(track, observation, angle_rad)?;
            let (next, report) = bench::shape_observation(track, observation, shape)?;
            Ok((
                next,
                EditReport::SpunShape {
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

/// The track's patch re-anchored on where `observation` sits in its
/// photograph, or the patch itself when the ray cannot meet it.
///
/// The frame the Image Detail layer outlines, so it is the frame a drag of that
/// outline is read against: a pointer is judged against the square a person can
/// see, not against the one the 3D position alone would draw. The same frame
/// `resize_patch_to_pixel` reads and writes.
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

/// `direction` as a unit vector, or `None` when it names no direction.
fn unit(direction: Vector3<f64>) -> Option<Vector3<f64>> {
    let length = direction.norm();
    (length.is_finite() && length >= MIN_DIRECTION).then(|| direction / length)
}

/// Where the ray from `origin` along the unit `ray` meets the plane through
/// `at` square to `normal`, in front of the eye.
///
/// The one ray-plane meeting the viewport's readings are built out of. Which
/// plane is the whole of what tells them apart: the patch's own for the three
/// handles that name a place on it, the one square to the normal for the
/// arrowhead's aim, and the one the arrowhead travels in for its swing. Every
/// one of them runs through the patch's centre, the aim's `AIM_LEVER` being a
/// lever the old normal is carried on rather than a distance a plane stands
/// off at. `None` when the ray runs along the plane or meets it behind the
/// eye, which is not the place a pointer named.
fn ray_plane(
    origin: Point3<f64>,
    ray: Vector3<f64>,
    at: Point3<f64>,
    normal: Vector3<f64>,
) -> Option<Point3<f64>> {
    let denominator = ray.dot(&normal);
    if denominator.abs() < MIN_DIRECTION {
        return None;
    }
    let distance = (at - origin).dot(&normal) / denominator;
    (distance > 0.0 && distance.is_finite()).then(|| origin + ray * distance)
}

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
    let ray = unit(direction)?;
    if frame.w == 0.0 {
        let bearing = frame.center.coords.normalize();
        let along = ray.dot(&bearing);
        if along <= MAX_BEARING_ANGLE_DEG.to_radians().cos() {
            return None;
        }
        return Some(Point3::from(ray / ray.dot(&frame.center.coords)));
    }
    ray_plane(origin, ray, frame.center, frame.normal())
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
/// The one number all three of this module's view tests are decided by, which
/// is why the two refusals are complementary rather than merely similar and why
/// the arrowhead ([`tilt_gesture`]) needs no refusal of its own: it reads the
/// same cosine at a third bar, and its two gestures fail where the two refusals
/// do, one each.
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
    let ray = unit(direction)?;
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
    turn_about(frame.center, frame.normal(), from, to)
}

/// The turn, in radians about the unit `axis`, that carries the direction from
/// `centre` to `from` onto the direction from `centre` to `to`, both read in
/// the plane through `centre` square to `axis`.
///
/// One reading serving the corner and the arrowhead's swing, which are the same
/// gesture about two different axes: the corner turns the square about its own
/// normal, and the swing turns the normal about an axis lying in the square
/// ([`tilt_gesture`]). Each place's component along `axis` is dropped first, so
/// a meeting that sits a rounding off the plane names the direction directly
/// under it. `None` when either place is `centre` itself, where no direction is
/// named.
pub(crate) fn turn_about(
    centre: Point3<f64>,
    axis: Vector3<f64>,
    from: Point3<f64>,
    to: Point3<f64>,
) -> Option<f64> {
    let flat = |point: Point3<f64>| {
        let offset = point - centre;
        let offset = offset - axis * offset.dot(&axis);
        (offset.norm() > MIN_OFFSET).then_some(offset)
    };
    let start = flat(from)?;
    let now = flat(to)?;
    // The signed angle about `axis` straight from the pair, rather than from a
    // difference of two `atan2`s that would then have to be brought back into
    // range: `atan2` already answers in `(-pi, pi]`.
    Some(start.cross(&now).dot(&axis).atan2(start.dot(&now)))
}

// ---- The arrowhead's two gestures ------------------------------------------

/// How near the line of sight the frame's normal has to lie before the
/// arrowhead **aims** rather than swings, in degrees.
///
/// The same cosine the two degenerate-view refusals are decided by
/// ([`plane_is_edge_on`], [`normal_is_end_on`]), read at a third bar, and that
/// is why the arrowhead needs no refusal of its own: the aim's plane is square
/// to the view exactly where the swing's axis is ill determined, and the swing's
/// axis is well determined exactly where the aim's plane is edge-on. The two
/// gestures are each other's cure, so between them the handle is always live.
pub(crate) const AIM_ANGLE_DEG: f64 = 45.0;

/// The aim's lever arm, in the frame's own half-lengths: the normal is answered
/// as `4h` along the old one plus the pointer's travel, so `4h` of travel is 45
/// degrees.
///
/// Twice the arrow's own [`NORMAL_LENGTH`](crate::viewer_3d::bench_track::NORMAL_LENGTH),
/// stated in terms of it rather than as a second literal, because the two are
/// facts about one handle: the travel is read on the plane through the centre,
/// which is also the plane the arrow is drawn out of, so `4h` of travel is
/// **twice the arrow's own drawn length whatever the zoom** and the aim is half
/// as sensitive as the figure looks. A small correction is a small motion,
/// which is what the handle is for, a normal being read off a surface a few
/// degrees at a time.
///
/// It is a lever and **not** a distance the plane stands at. Standing the plane
/// off by it would put the plane behind the eye whenever the camera came within
/// `4h` of a patch facing it -- which is exactly the view the aim is for -- and
/// the press would fall through to the viewport's navigation. It would also tie
/// the handle's sensitivity to the eye's distance, since what a pixel of pointer
/// is worth on a plane depends on how far that plane is; read on the centre's
/// own plane, the gesture is the same gesture at every zoom.
pub(crate) const AIM_LEVER: f64 = 2.0 * crate::viewer_3d::bench_track::NORMAL_LENGTH;

/// Which of the arrowhead's two gestures a press makes, and what it is fixed
/// to.
///
/// Decided at the press and carried for the whole drag, as the handle itself
/// is, so a gesture does not change character halfway through.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Tilt {
    /// The normal lies near the line of sight, so the pointer's ray is met with
    /// the plane through the centre square to that normal, and the travel
    /// across it since the press swings the normal off by [`AIM_LEVER`].
    Aim,
    /// The normal lies across the line of sight, so the pointer turns it about
    /// this one axis: the unit vector of the frame's **own plane** nearest the
    /// eye. The normal therefore keeps to the one plane through it square to
    /// this axis and never rolls toward or away from the viewer, which is the
    /// motion that is hardest to aim when the arrowhead is nearly side-on.
    Swing(Vector3<f64>),
}

/// Which gesture the arrowhead makes from `eye`, or `None` when there is no
/// arrowhead to press: a **direction patch**, whose normal is fixed by its
/// bearing, or an eye that names no view of the frame.
pub(crate) fn tilt_gesture(frame: &OrientedPatch, eye: Point3<f64>) -> Option<Tilt> {
    if frame.w == 0.0 {
        return None;
    }
    let cosine = view_cosine(frame, eye)?;
    if cosine > AIM_ANGLE_DEG.to_radians().cos() {
        return Some(Tilt::Aim);
    }
    // The part of the view direction square to the normal, which is the unit
    // vector of the frame's plane nearest the eye. It is well determined
    // exactly where the aim is not, `view_cosine` being its own complement.
    let to_eye = unit(eye - frame.center)?;
    let normal = frame.normal();
    unit(to_eye - normal * to_eye.dot(&normal)).map(Tilt::Swing)
}

/// Where the pointer's ray meets the geometry `tilt` reads it against.
///
/// The counterpart of [`plane_point`] and [`normal_line_point`] for the
/// arrowhead. Both gestures read a plane **through the centre**, and which one
/// is the whole of what tells them apart: square to the normal for an aim, and
/// square to the swing's axis -- the plane the arrowhead travels in, and, the
/// axis pointing at the eye as nearly as the frame allows, the plane most
/// nearly facing the window -- for a swing.
///
/// Neither plane can fall behind the eye, both passing through a centre that is
/// in front of it whenever the figure is on screen at all, and neither can be
/// caught edge-on: an aim's plane is square to the normal and the aim is chosen
/// only where the normal lies within [`AIM_ANGLE_DEG`] of the view, and a
/// swing's axis is the one most nearly pointing at the eye. So the arrowhead
/// answers from every view it is drawn in, which is what it means for it to
/// take no degenerate-view refusal.
///
/// The frame is the one the press was taken against, so the plane is fixed for
/// the whole drag and the gesture is a single map from the window onto the
/// sphere of normals rather than a thing that moves as it is used.
pub(crate) fn tilt_point(
    frame: &OrientedPatch,
    tilt: Tilt,
    origin: Point3<f64>,
    direction: Vector3<f64>,
) -> Option<Point3<f64>> {
    let ray = unit(direction)?;
    let square_to = match tilt {
        Tilt::Aim => frame.normal(),
        Tilt::Swing(axis) => axis,
    };
    ray_plane(origin, ray, frame.center, square_to)
}

/// The outward normal a drag of the arrowhead names, from the two places
/// [`tilt_point`] read for it.
///
/// **Aiming** answers with the old normal on a lever of [`AIM_LEVER`]
/// half-lengths, swung by however far the pointer has travelled across the
/// plane since the press: `unit(4h n + travel)`. It is the **travel** and not
/// the meeting itself that is read, for the reason the centre dot's press is
/// kept: a press that took the arrowhead is not standing where the centre is,
/// and reading the meeting outright would jump the normal over before the
/// pointer had moved at all. The travel lies in the plane and so is square to
/// `n`, which leaves the whole `4h` standing along the old normal: the sum of a
/// fixed vector and one square to it can never turn through a right angle, so
/// one gesture turns the normal by less than 90 degrees and cannot push it
/// through the frame at all.
///
/// **Swinging** answers with the normal turned about the axis by the angle the
/// pointer swept about the centre, which is the corner's reading with the swing
/// axis in place of the normal -- and is why the two take the same cursor.
///
/// `None` when the places name no answer: a swing read about the centre itself.
pub(crate) fn tilt_normal(
    frame: &OrientedPatch,
    tilt: Tilt,
    from: Point3<f64>,
    to: Point3<f64>,
) -> Option<Vector3<f64>> {
    match tilt {
        Tilt::Aim => unit(frame.normal() * (AIM_LEVER * frame.half_extent[0]) + (to - from)),
        Tilt::Swing(axis) => {
            let angle = turn_about(frame.center, axis, from, to)?;
            Some(Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle) * frame.normal())
        }
    }
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
