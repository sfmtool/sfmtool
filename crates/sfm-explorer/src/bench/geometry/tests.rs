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

#[test]
fn the_boundarys_edges_are_named_in_the_order_the_square_is_walked() {
    assert_eq!(
        [edge_of(0), edge_of(1), edge_of(2), edge_of(3)],
        [Edge::MinusV, Edge::PlusU, Edge::PlusV, Edge::MinusU],
    );
}
