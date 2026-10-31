// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How a view sees a patch: the direction from the patch to the camera
//! ([`surface_to_camera`]) and the [`viewing_angle`] and tilt direction read
//! from it.

use nalgebra::Vector3;

use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;

/// The unit direction from `patch`'s centre towards the camera of
/// `cam_from_world`, or `None` where the camera sits on the centre.
///
/// For a finite patch (`w == 1`) it is the camera centre minus the patch
/// centre, normalized. For a patch at infinity (`w == 0`) the centre is the
/// bearing from every camera to the patch, so the direction is minus that
/// bearing. [`viewing_angle`] reads it to measure a view's angle to the patch.
pub fn surface_to_camera(
    patch: &OrientedPatch,
    cam_from_world: &RigidTransform,
) -> Option<Vector3<f64>> {
    let d = if patch.w == 0.0 {
        -patch.center.coords
    } else {
        cam_from_world.inverse_translation_origin() - patch.center
    };
    let norm = d.norm();
    (norm > 1e-12 && norm.is_finite()).then(|| d / norm)
}

/// The angle below which [`viewing_angle`] gives no tilt direction, in
/// degrees. At this angle the ray's component in the patch plane is
/// `sin 0.1° ≈ 0.0017` of its length, and its direction there is decided by
/// rounding in the pose and the normal rather than by the view.
pub const MIN_TILT_ANGLE_DEG: f64 = 0.1;

/// How one view sees a patch: the angle between its ray and the patch's
/// normal, and which way in the patch's plane the ray leans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewingAngle {
    /// The viewing angle `θ`, in degrees: the angle between the patch's
    /// outward normal and the direction from the patch's centre to the
    /// camera, so `0` for a view facing the patch and `90` for one that
    /// grazes it. Over `90` for a view of the patch's back.
    pub angle_deg: f64,
    /// The tilt direction, in degrees: the angle of the ray from the camera,
    /// projected into the patch's plane, measured from the patch's `u` axis
    /// towards its `v` axis, in `[-180, 180]`. Along this direction the view
    /// foreshortens the patch by `cos θ`, and across it not at all. `None`
    /// when `θ` is under [`MIN_TILT_ANGLE_DEG`], where the view faces the
    /// patch and has no tilt direction.
    pub tilt_direction_deg: Option<f64>,
}

/// The viewing angle and tilt direction of the camera of `cam_from_world` on
/// `patch`, read at the patch's centre ([`surface_to_camera`]), or `None`
/// where the camera sits on the centre.
///
/// Measured at the patch's centre, so a caller that wants the angle at an
/// observation's keypoint passes the patch re-anchored on that keypoint
/// ([`OrientedPatch::anchored_at_keypoint`]), whose centre lies on the
/// keypoint's ray.
///
/// ```
/// use nalgebra::{Point3, Vector3};
/// use sfmtool_core::geometry::RigidTransform;
/// use sfmtool_core::patch::cloud::OrientedPatch;
/// use sfmtool_core::patch::normal_refine::viewing_angle;
///
/// // A patch at the origin facing +z, seen from a camera at (0, 0, 2) and
/// // from one at (2, 0, 2), 45 degrees off the normal towards +x.
/// let patch = OrientedPatch::new(Point3::origin(), Vector3::x(), Vector3::y(), [0.1, 0.1]);
/// let above = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -2.0]);
/// let facing = viewing_angle(&patch, &above).unwrap();
/// assert!(facing.angle_deg.abs() < 1e-9);
/// assert_eq!(facing.tilt_direction_deg, None);
/// let aside = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-2.0, 0.0, -2.0]);
/// let oblique = viewing_angle(&patch, &aside).unwrap();
/// assert!((oblique.angle_deg - 45.0).abs() < 1e-9);
/// // The ray from that camera runs towards -x, so it leans along -u.
/// assert!((oblique.tilt_direction_deg.unwrap().abs() - 180.0).abs() < 1e-9);
/// ```
pub fn viewing_angle(
    patch: &OrientedPatch,
    cam_from_world: &RigidTransform,
) -> Option<ViewingAngle> {
    let to_camera = surface_to_camera(patch, cam_from_world)?;
    let n = patch.normal();
    let cosine = to_camera.dot(&n).clamp(-1.0, 1.0);
    let angle_deg = cosine.acos().to_degrees();
    // The ray from the camera is `-to_camera`; its part in the plane is the
    // direction the view foreshortens the patch along.
    let ray = -to_camera;
    let in_plane = ray - n * ray.dot(&n);
    let tilt_direction_deg =
        (angle_deg >= MIN_TILT_ANGLE_DEG && in_plane.norm() > 0.0).then(|| {
            in_plane
                .dot(&patch.v_axis)
                .atan2(in_plane.dot(&patch.u_axis))
                .to_degrees()
        });
    Some(ViewingAngle {
        angle_deg,
        tilt_direction_deg,
    })
}
