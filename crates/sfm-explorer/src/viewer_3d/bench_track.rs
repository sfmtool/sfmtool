// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench's active track as world geometry, for the 3D viewer to draw.
//!
//! Where the Image Detail panel's bench layer shows the track in each
//! photograph that observes it, this shows it where it stands: the patch frame
//! as a square in the world, its outward normal standing off it, and one mark
//! per observation saying where that photograph sees the patch's content
//! against where the frame's middle is. One difference from Image Detail is
//! deliberate -- there the outline is the surfel **re-anchored on that image's
//! keypoint**, because the panel shows where the sighting is in that
//! photograph; here there is no photograph and the frame drawn is the surfel
//! itself, at its centre.
//!
//! What this module produces is a
//! [`Figure`](crate::viewer_3d::bench_track::Figure): world points, world
//! directions and colours, with no window and no GPU in it, so the geometry is
//! assertable without a frame and without a device. [`crate::scene_renderer`]
//! turns it into instances and draws it after the EDL pass, depth-aware, so
//! where the frame cuts through the point cloud the part in front reads at full
//! strength and the part behind is drawn through, dimmed.

use egui::Color32;
use nalgebra::{Point3, Vector3};
use sfmtool_core::bench::{EditableTrack, Stage};
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::{EditedReconstruction, Se3Transform};

use crate::bench::{geometry, verdict_color};

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
const NORMAL_LENGTH: f64 = 2.0;

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

/// The four `(s, t)` corners of the frame's square, in the order
/// [`OrientedPatch::boundary`] walks them, so consecutive pairs are its edges.
const CORNERS: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// What the dock tells the 3D viewer about the node's bench, once a frame.
///
/// The viewer does not read the bench, for the reason the Image Detail panel
/// does not ([`crate::image_detail::BenchMenu`]): the panel is handed `&mut`
/// into the state further down the same call, so what it needs is read out
/// beside the selection and passed in.
pub(crate) struct BenchTrack<'a> {
    /// The active track, which is the one item of the bench this layer draws.
    pub(crate) track: &'a EditableTrack,
    /// The node's value at its cursor, for the cameras and poses the marks
    /// unproject their keypoints through.
    pub(crate) edited: &'a EditedReconstruction,
    /// The node's similarity. The figure is built in the reconstruction's own
    /// coordinates and put through this, because the pass draws from one shared
    /// buffer with no per-recon `model` matrix of its own.
    pub(crate) transform: &'a Se3Transform,
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
    /// Centre to `q_i`.
    pub(crate) segment: Stroke,
    /// The hollow circle at `q_i`, in the frame's plane.
    pub(crate) circle: Vec<Stroke>,
}

/// The whole figure, in world coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Figure {
    /// The square, four edges in `(s, t)` boundary order, in the `in` colour
    /// whatever any observation's verdict is.
    pub(crate) frame: [Stroke; 4],
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
    let frame = payload.frame.as_ref()?;
    // The frame's own `w` rather than the payload's `at_infinity`: what is
    // drawn is the surfel, and core keeps the two equal wherever both exist.
    let at_infinity = frame.w == 0.0;
    // Patch frames are square, so one half-length says the whole of the size.
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
        .filter_map(|observation| {
            let site = observation.site()?;
            let (camera, pose) = geometry::view_of(image_table, observation.image as usize)?;
            // The one unprojection, core's own: the keypoint's ray meets the
            // frame's plane and what it named is read off that meeting. A ray
            // that cannot meet the plane in front of its camera -- including a
            // bearing pointing away from a direction patch -- says so here.
            let offset = frame.keypoint_plane_offset(&camera, &pose, site)?;
            let q = frame.center.coords + offset;
            let color = rgba(verdict_color(observation.verdict));
            Some(Mark {
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
        disc,
        normal,
        marks,
        fog_distance: (FOG_DISTANCE * half * bench.transform.scale) as f32,
    })
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
