// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a ray of the 3D viewport means against a patch, asserted on patches
//! written by hand.
//!
//! No reconstruction and no window: these are the two meetings the viewport's
//! plane handles are built on -- a ray with a finite frame's plane and a
//! direction with a bearing's tangent plane -- plus the two refusals that keep
//! either from being read where it says nothing.

use nalgebra::{Point3, Vector3};
use sfmtool_core::bench::Edge;
use sfmtool_core::patch::cloud::OrientedPatch;

use super::*;

/// A unit square in the `z = 0` plane, facing `+z`.
fn flat() -> OrientedPatch {
    OrientedPatch::new(Point3::origin(), Vector3::x(), Vector3::y(), [1.0, 1.0])
}

/// The same square taken to the sky along `+z`, tangent to the unit sphere.
fn bearing() -> OrientedPatch {
    OrientedPatch::from_infinity_direction(Point3::new(0.0, 0.0, 1.0), Vector3::y(), [0.2, 0.2])
}

#[test]
fn a_ray_meets_a_finite_frame_where_the_plane_is_and_nowhere_else() {
    let frame = flat();
    let eye = Point3::new(0.4, -0.3, 5.0);

    // Straight down: the meeting is the eye's own footprint on the plane.
    let at = plane_point(&frame, eye, -Vector3::z()).expect("the ray meets the plane");
    assert!((at - Point3::new(0.4, -0.3, 0.0)).norm() < 1e-12);

    // Along the plane: nothing to meet.
    assert!(plane_point(&frame, eye, Vector3::x()).is_none());
    // Away from it: the meeting is behind the eye, which is not what the
    // pointer named.
    assert!(plane_point(&frame, eye, Vector3::z()).is_none());
    // And a direction that is not one.
    assert!(plane_point(&frame, eye, Vector3::zeros()).is_none());
}

#[test]
fn a_ray_meets_a_bearings_tangent_plane_at_its_own_direction_over_the_bearing() {
    let frame = bearing();
    // The origin is dropped: a direction has no parallax, and the viewer is in
    // effect at the centre of the sphere the plane is tangent to.
    let ray = Vector3::new(0.1, -0.05, 1.0);
    let near = plane_point(&frame, Point3::origin(), ray).expect("the ray meets the plane");
    let far = plane_point(&frame, Point3::new(9.0, -4.0, 3.0), ray).expect("the same ray");
    assert!(
        (near - far).norm() < 1e-12,
        "a bearing read the eye's place"
    );
    // `r / (r . d)`, which is `d + a u + b v`.
    assert!((near - Point3::new(0.1, -0.05, 1.0)).norm() < 1e-12);

    // A pointer past the right angle: `r . d` goes to zero and the tangent
    // point runs off, so it names nothing.
    let across = Vector3::new(1.0, 0.0, 0.05);
    assert!(
        f64::acos(across.normalize().dot(&Vector3::z())).to_degrees() > MAX_BEARING_ANGLE_DEG,
        "the fixture's ray should be outside the bearing's cap"
    );
    assert!(plane_point(&frame, Point3::origin(), across).is_none());
    // And one just inside it is read.
    assert!(plane_point(&frame, Point3::origin(), Vector3::new(1.0, 0.0, 1.0)).is_some());
}

#[test]
fn a_plane_seen_edge_on_is_refused_and_a_bearings_never_is() {
    let frame = flat();
    // Square on.
    assert!(!plane_is_edge_on(&frame, Point3::new(0.0, 0.0, 4.0)));
    // In the plane itself, which is where a pixel of pointer motion becomes an
    // unbounded distance along it.
    assert!(plane_is_edge_on(&frame, Point3::new(4.0, 0.0, 0.0)));

    // The bar is the angle between the view ray and the plane, so a view at
    // just under it is refused and one at just over it is not.
    let at = |degrees: f64| {
        let (sin, cos) = degrees.to_radians().sin_cos();
        Point3::new(4.0 * cos, 0.0, 4.0 * sin)
    };
    assert!(plane_is_edge_on(&frame, at(MIN_PLANE_ANGLE_DEG - 0.5)));
    assert!(!plane_is_edge_on(&frame, at(MIN_PLANE_ANGLE_DEG + 0.5)));

    // A bearing's tangent plane faces the eye by construction.
    assert!(!plane_is_edge_on(&bearing(), Point3::new(400.0, 0.0, 0.0)));
}

/// The two refusals are one bar read from the two ends: an edge-on plane is a
/// normal at its best and the other way about, so the view that kills one set of
/// handles is the view the other set wants.
#[test]
fn the_normal_is_refused_end_on_exactly_where_the_plane_is_at_its_best() {
    let frame = flat();
    // Straight down the normal: the plane is square on and the line runs into
    // the eye, where the closest-approach solve has nothing to divide by.
    let down = Point3::new(0.0, 0.0, 4.0);
    assert!(normal_is_end_on(&frame, down));
    assert!(!plane_is_edge_on(&frame, down));

    // Straight up it -- the far side -- is exactly as bad, the line being
    // undirected.
    assert!(normal_is_end_on(&frame, Point3::new(0.0, 0.0, -4.0)));

    // In the plane, where the plane handles die and the normal stands across
    // the view at its longest.
    let across = Point3::new(4.0, 0.0, 0.0);
    assert!(!normal_is_end_on(&frame, across));
    assert!(plane_is_edge_on(&frame, across));

    // The bar is the same angle for both, measured from the two ends: at
    // `MIN_PLANE_ANGLE_DEG` off the normal neither is refused, and inside it
    // only the normal is.
    let at = |degrees: f64| {
        let (sin, cos) = degrees.to_radians().sin_cos();
        Point3::new(4.0 * sin, 0.0, 4.0 * cos)
    };
    assert!(normal_is_end_on(&frame, at(MIN_PLANE_ANGLE_DEG - 0.5)));
    assert!(!normal_is_end_on(&frame, at(MIN_PLANE_ANGLE_DEG + 0.5)));
    assert!(!plane_is_edge_on(&frame, at(MIN_PLANE_ANGLE_DEG - 0.5)));

    // A bearing has no normal to take hold of at all.
    assert!(normal_is_end_on(&bearing(), Point3::new(0.0, 0.0, 4.0)));
}

#[test]
fn a_ray_names_the_point_of_the_normal_line_it_comes_nearest() {
    let frame = flat();
    let eye = Point3::new(5.0, 0.0, 2.0);

    // Straight at the line: the answer is the point the ray runs through.
    let at = normal_line_point(&frame, eye, -Vector3::x()).expect("the ray crosses the line");
    assert!((at - Point3::new(0.0, 0.0, 2.0)).norm() < 1e-12);

    // The two lines need not meet: a ray passing a unit to one side names the
    // point of the line it comes nearest, which is the same one.
    let past = Point3::new(5.0, 1.0, 2.0);
    let at = normal_line_point(&frame, past, -Vector3::x()).expect("a closest approach");
    assert!((at - Point3::new(0.0, 0.0, 2.0)).norm() < 1e-12);

    // The press's own place on the line, which is what a drag subtracts: a ray
    // aimed higher names a point further along the normal.
    let higher = normal_line_point(&frame, Point3::new(5.0, 0.0, 3.5), -Vector3::x())
        .expect("a closest approach");
    assert!((higher - at).dot(&frame.normal()) > 1.0);

    // A ray along the line itself divides by the squared sine of nothing.
    assert!(normal_line_point(&frame, eye, Vector3::z()).is_none());
    assert!(normal_line_point(&frame, eye, Vector3::zeros()).is_none());
    // And a bearing has no normal line.
    assert!(normal_line_point(&bearing(), eye, -Vector3::x()).is_none());
}

#[test]
fn a_turn_on_the_plane_is_the_angle_swept_about_the_centre() {
    let frame = flat();
    let quarter = turn_on_plane(
        &frame,
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(-1.0, 1.0, 0.0),
    )
    .expect("two places off the centre");
    assert!((quarter.to_degrees() - 90.0).abs() < 1e-9);

    // The short way round, not its explement.
    let back = turn_on_plane(
        &frame,
        Point3::new(-1.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    )
    .expect("two places off the centre");
    assert!((back.to_degrees() + 90.0).abs() < 1e-9);

    // The centre itself names no direction.
    assert!(turn_on_plane(&frame, frame.center, Point3::new(1.0, 0.0, 0.0)).is_none());
}

// ---- The arrowhead's two gestures ------------------------------------------

/// A ray from `origin` that lands exactly on `at`.
fn ray_to(origin: Point3<f64>, at: Point3<f64>) -> (Point3<f64>, Vector3<f64>) {
    (origin, at - origin)
}

/// An eye `degrees` off the frame's normal, on the `sign` side of it, at a
/// distance of four half-lengths.
fn eye_off_normal(frame: &OrientedPatch, degrees: f64, sign: f64) -> Point3<f64> {
    let (sin, cos) = degrees.to_radians().sin_cos();
    frame.center + (frame.normal() * (cos * sign) + frame.u_axis * sin) * 4.0
}

/// Which gesture the arrowhead makes is decided by where the normal points, on
/// the **magnitude** of its cosine against the view: a patch showing its back is
/// as square to the view as one showing its face.
#[test]
fn the_arrowhead_aims_within_forty_five_degrees_of_the_view_and_swings_outside_it() {
    let frame = flat();
    for sign in [1.0, -1.0] {
        assert_eq!(
            tilt_gesture(&frame, eye_off_normal(&frame, 0.0, sign)),
            Some(Tilt::Aim),
            "straight down the normal is the aim's own view",
        );
        assert_eq!(
            tilt_gesture(&frame, eye_off_normal(&frame, AIM_ANGLE_DEG - 0.5, sign)),
            Some(Tilt::Aim),
        );
        assert!(
            matches!(
                tilt_gesture(&frame, eye_off_normal(&frame, AIM_ANGLE_DEG + 0.5, sign)),
                Some(Tilt::Swing(_)),
            ),
            "past the bar the arrowhead swings",
        );
        assert!(
            matches!(
                tilt_gesture(&frame, eye_off_normal(&frame, 90.0, sign)),
                Some(Tilt::Swing(_)),
            ),
            "a view along the plane is the swing's own",
        );
    }
    // A direction patch draws no arrowhead, so there is no gesture to make.
    assert_eq!(tilt_gesture(&bearing(), Point3::new(0.0, 0.0, 4.0)), None);
    // And an eye at the centre names no view of the frame at all.
    assert_eq!(tilt_gesture(&frame, frame.center), None);
}

/// The aim is a map of the window onto the sphere of normals, read off the
/// plane through the **centre** square to the normal and levered by
/// [`AIM_LEVER`] half-lengths. Two things follow from that lever, and they are
/// the whole of why it is the length it is.
#[test]
fn an_aim_turns_forty_five_degrees_in_four_half_lengths_and_never_reaches_ninety() {
    let frame = flat();
    let half = frame.half_extent[0];
    let lever = AIM_LEVER * half;
    let eye = frame.center + frame.normal() * 20.0;
    let on_plane = |across: f64| frame.center + frame.u_axis * across;

    // The meeting is the plane's own point, and that plane runs through the
    // centre rather than standing off it.
    let (origin, direction) = ray_to(eye, on_plane(1.5));
    let at = tilt_point(&frame, Tilt::Aim, origin, direction).expect("the ray meets the plane");
    assert!((at - on_plane(1.5)).norm() < 1e-9);

    // The press's own place on that plane, which every answer is measured from:
    // a press on the arrowhead is not standing where the centre is, the head
    // being drawn out along the normal.
    let (origin, direction) = ray_to(eye, on_plane(0.7));
    let press = tilt_point(&frame, Tilt::Aim, origin, direction).expect("the plane");
    let aimed = |across: f64| {
        let (origin, direction) = ray_to(eye, on_plane(across));
        let at = tilt_point(&frame, Tilt::Aim, origin, direction).expect("the plane");
        tilt_normal(&frame, Tilt::Aim, press, at).expect("a direction")
    };
    let degrees = |n: Vector3<f64>| n.dot(&frame.normal()).clamp(-1.0, 1.0).acos().to_degrees();

    // A drag that ends where it started names the normal the patch already has.
    assert!((aimed(0.7) - frame.normal()).norm() < 1e-9);

    // `AIM_LEVER` half-lengths of travel is 45 degrees, which is the arrow's own
    // drawn length twice over: the handle is half as sensitive as the figure
    // looks.
    let tilted = aimed(0.7 + lever);
    assert!((degrees(tilted) - 45.0).abs() < 1e-9, "{}", degrees(tilted));

    // And no aim reaches 90 however far the pointer goes: the travel is square
    // to the lever, so the sum can never turn through a right angle, and one
    // gesture cannot push the normal through the frame.
    for across in [10.0, 1e3, 1e9] {
        let far = aimed(0.7 + across * half);
        assert!(
            degrees(far) < 90.0 && far.dot(&frame.normal()) > 0.0,
            "an aim {across} half-lengths out reached {} degrees",
            degrees(far),
        );
    }
}

/// The aim answers from **inside** the lever's own length, which a plane
/// standing `AIM_LEVER` half-lengths out could not.
///
/// The patch facing the eye is the aim's own view, and it is also the view a
/// person zooms into to work on a normal, so an eye a half-length or two off
/// the surface is the ordinary case rather than a corner of one. A plane stood
/// off by the lever would sit *behind* such an eye -- the standoff being
/// measured toward it -- and the press would read nothing and fall through to
/// the viewport's navigation, with the arrowhead's own cursor still showing.
/// Reading the centre's own plane, there is no such distance.
#[test]
fn an_aim_answers_from_closer_in_than_its_own_lever() {
    let frame = flat();
    let half = frame.half_extent[0];
    let degrees = |n: Vector3<f64>| n.dot(&frame.normal()).clamp(-1.0, 1.0).acos().to_degrees();

    // Nearer than the lever, at it, and comfortably past it: the same gesture
    // and the same answer, the reading having no distance in it at all.
    for standoff in [0.5, 1.0, AIM_LEVER - 0.5, AIM_LEVER, AIM_LEVER + 0.5, 50.0] {
        let eye = frame.center + frame.normal() * (standoff * half);
        assert_eq!(
            tilt_gesture(&frame, eye),
            Some(Tilt::Aim),
            "a patch facing the eye at {standoff} half-lengths is the aim's view",
        );
        let at = |across: f64| {
            let (origin, direction) = ray_to(eye, frame.center + frame.u_axis * across);
            tilt_point(&frame, Tilt::Aim, origin, direction)
                .unwrap_or_else(|| panic!("an eye {standoff} half-lengths out read no point"))
        };
        // The press is off the centre, as a press on the drawn arrowhead is.
        let turned = tilt_normal(&frame, Tilt::Aim, at(0.3), at(0.3 + AIM_LEVER * half))
            .expect("a direction");
        assert!(
            (degrees(turned) - 45.0).abs() < 1e-9,
            "at {standoff} half-lengths the same travel turned {} degrees",
            degrees(turned),
        );
    }
}

/// The swing turns the normal about one axis lying in the frame's own plane --
/// the one nearest the eye -- so the normal keeps to the single plane through
/// it square to that axis and never rolls toward or away from the viewer.
#[test]
fn a_swing_turns_about_the_axis_of_the_frames_plane_nearest_the_eye() {
    let frame = flat();
    // Well past the bar, and leaning toward `+u` and a little toward `+v`, so
    // the axis is not one of the frame's own axes by accident.
    let eye = frame.center + frame.u_axis * 6.0 + frame.v_axis * 2.0 + frame.normal() * 1.0;
    let Some(Tilt::Swing(axis)) = tilt_gesture(&frame, eye) else {
        panic!("this view is the swing's")
    };

    // In the frame's plane, and the nearest direction there to the eye: no
    // other in-plane unit vector has a larger cosine against the view.
    assert!(axis.dot(&frame.normal()).abs() < 1e-12, "{axis:?}");
    assert!((axis.norm() - 1.0).abs() < 1e-12);
    let to_eye = (eye - frame.center).normalize();
    for k in 0..72 {
        let angle = std::f64::consts::TAU * f64::from(k) / 72.0;
        let other = frame.u_axis * angle.cos() + frame.v_axis * angle.sin();
        assert!(
            other.dot(&to_eye) <= axis.dot(&to_eye) + 1e-12,
            "{other:?} lies nearer the eye than the axis does",
        );
    }

    // The pointer is read against the plane through the centre square to that
    // axis, which is the plane the arrowhead travels in.
    let tip = frame.center + frame.normal() * 2.0;
    let from = tilt_point(&frame, Tilt::Swing(axis), eye, tip - eye).expect("the ray meets it");
    assert!((from - frame.center).dot(&axis).abs() < 1e-9);

    // And the normal it names stays square to the axis: the turn is about that
    // axis and nothing else.
    let swung = frame.center + (tip - frame.center) * 0.6 + axis.cross(&frame.normal()) * 1.4;
    let to = tilt_point(&frame, Tilt::Swing(axis), eye, swung - eye).expect("the ray meets it");
    let normal = tilt_normal(&frame, Tilt::Swing(axis), from, to).expect("a direction");
    assert!((normal.norm() - 1.0).abs() < 1e-12);
    assert!(
        normal.dot(&axis).abs() < 1e-9,
        "the swing rolled the normal"
    );
    assert!(
        (normal - frame.normal()).norm() > 1e-3,
        "the swing turned the normal by nothing",
    );

    // A swing read about the centre itself names no answer.
    assert!(tilt_normal(&frame, Tilt::Swing(axis), frame.center, to).is_none());
}

#[test]
fn the_boundarys_edges_are_named_in_the_order_the_square_is_walked() {
    assert_eq!(
        [edge_of(0), edge_of(1), edge_of(2), edge_of(3)],
        [Edge::MinusV, Edge::PlusU, Edge::PlusV, Edge::MinusU],
    );
}
