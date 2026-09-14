// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `OrientedPatch` (surfel) value type.

use nalgebra::{Point3, Vector3};
use pyo3::prelude::*;

use sfmtool_core::patch::cloud::OrientedPatch;

use crate::geometry::rigid_transform::PyRigidTransform;
use crate::PyCameraIntrinsics;

/// An oriented planar patch (surfel) in world space.
///
/// The plane is spanned by orthonormal in-plane axes ``u_axis`` and ``v_axis``;
/// the frame is right-handed with outward normal ``u_axis × v_axis``. A
/// ``(col, row)`` render steps ``col`` along ``+u_axis`` and ``row`` along
/// ``−v_axis`` (rows count downward), so the front face renders un-mirrored.
/// The patch covers the world points
/// ``center + s·half_extent[0]·u_axis + t·half_extent[1]·v_axis`` for
/// ``(s, t)`` in ``[-1, 1]²``. Pair with :meth:`WarpMap.from_patch` to render its
/// appearance in a camera. See ``specs/core/patch/patch-cloud.md``.
#[pyclass(name = "OrientedPatch", module = "sfmtool.patches")]
pub struct PyOrientedPatch {
    pub(crate) inner: OrientedPatch,
}

#[pymethods]
impl PyOrientedPatch {
    /// Construct from an explicit center, in-plane axes, and per-axis half-extent.
    #[new]
    #[pyo3(signature = (center, u_axis, v_axis, half_extent))]
    fn new(center: [f64; 3], u_axis: [f64; 3], v_axis: [f64; 3], half_extent: [f64; 2]) -> Self {
        Self {
            inner: OrientedPatch::new(
                Point3::new(center[0], center[1], center[2]),
                Vector3::new(u_axis[0], u_axis[1], u_axis[2]),
                Vector3::new(v_axis[0], v_axis[1], v_axis[2]),
                half_extent,
            ),
        }
    }

    /// Build from a center, a normal, and an ``up_hint`` that pins the in-plane
    /// rotation about the normal.
    ///
    /// Args:
    ///     center: Patch center in world coordinates.
    ///     normal: Surface normal (need not be unit length).
    ///     up_hint: The "up" reference direction; projected onto the plane to set
    ///         ``v_axis`` (``u_axis = v_axis × normal`` is the in-plane "right"
    ///         axis). If parallel to the normal an arbitrary axis is used.
    ///     half_extent: World-space half-size along ``(u, v)``.
    #[staticmethod]
    #[pyo3(signature = (center, normal, up_hint, half_extent))]
    fn from_center_normal(
        center: [f64; 3],
        normal: [f64; 3],
        up_hint: [f64; 3],
        half_extent: [f64; 2],
    ) -> Self {
        Self {
            inner: OrientedPatch::from_center_normal(
                Point3::new(center[0], center[1], center[2]),
                Vector3::new(normal[0], normal[1], normal[2]),
                Vector3::new(up_hint[0], up_hint[1], up_hint[2]),
                half_extent,
            ),
        }
    }

    /// Build the tangent-sphere frame for a **point at infinity** with direction
    /// ``direction`` (``w == 0``): outward normal ``normalize(-direction)``,
    /// ``u, v`` perpendicular to the direction, the in-plane rotation pinned by
    /// ``up_hint``. ``center`` stores the direction itself. Pair with
    /// :meth:`WarpMap.from_patch`, which projects the (direction-valued) corners
    /// without applying the camera translation.
    #[staticmethod]
    #[pyo3(signature = (direction, up_hint, half_extent))]
    fn from_infinity_direction(
        direction: [f64; 3],
        up_hint: [f64; 3],
        half_extent: [f64; 2],
    ) -> Self {
        Self {
            inner: OrientedPatch::from_infinity_direction(
                Point3::new(direction[0], direction[1], direction[2]),
                Vector3::new(up_hint[0], up_hint[1], up_hint[2]),
                half_extent,
            ),
        }
    }

    /// Build a **finite** patch from one observation: the keypoint's ray at
    /// ``depth``, framed fronto-parallel, carrying the in-plane axes and
    /// half-extents that the keypoint's 2x2 ``affine_shape`` unprojects to.
    ///
    /// The inverse of the rule the ``.sfmr`` format states for reading a
    /// keypoint's shape off a patch frame (``SfmrReconstruction
    /// .observation_affine_shape``): that rule projects the anchor and the two
    /// half-axis tips and takes the pixel differences as the shape's columns, so
    /// this one back-projects the two tip pixels and takes the world differences
    /// as the half-axes. Round-tripping a shape through the two returns it,
    /// distortion included.
    ///
    /// Args:
    ///     camera: The observing view's intrinsics.
    ///     cam_from_world: The observing view's pose (``p_cam = R·p_world + t``).
    ///     keypoint: The observation's pixel, ``(x, y)``.
    ///     affine_shape: ``[[a11, a12], [a21, a22]]``, columns first: column 0
    ///         becomes ``u``, column 1 becomes ``v``. The ``.sift`` layout.
    ///     depth: Distance from the camera centre along the pixel's **unit
    ///         bearing** -- a range, not a camera-frame ``z``, so it reads the
    ///         same for a fisheye.
    ///
    /// Returns:
    ///     The patch, or ``None`` when the depth is not a positive finite
    ///     distance, the camera model has no ray for the keypoint or a tip pixel,
    ///     a tip's ray runs parallel to (or away from) the patch plane, or the
    ///     shape unprojects to a degenerate frame.
    ///
    /// Note:
    ///     A patch frame's ``v`` points up in the image while pixel rows count
    ///     down, so a front-facing patch projects to a shape of negative
    ///     determinant. A ``.sift``-style scaled rotation has positive
    ///     determinant, so its second column is negated here to keep the patch
    ///     facing the camera.
    #[staticmethod]
    #[pyo3(signature = (camera, cam_from_world, keypoint, affine_shape, depth))]
    fn from_affine_shape_at_depth(
        camera: &PyCameraIntrinsics,
        cam_from_world: &PyRigidTransform,
        keypoint: [f64; 2],
        affine_shape: [[f64; 2]; 2],
        depth: f64,
    ) -> Option<Self> {
        OrientedPatch::from_affine_shape_at_depth(
            &camera.inner,
            &cam_from_world.inner,
            keypoint,
            affine_shape,
            depth,
        )
        .map(|inner| Self { inner })
    }

    #[getter]
    fn center(&self) -> [f64; 3] {
        let c = self.inner.center;
        [c.x, c.y, c.z]
    }

    #[getter]
    fn u_axis(&self) -> [f64; 3] {
        let v = self.inner.u_axis;
        [v.x, v.y, v.z]
    }

    #[getter]
    fn v_axis(&self) -> [f64; 3] {
        let v = self.inner.v_axis;
        [v.x, v.y, v.z]
    }

    #[getter]
    fn half_extent(&self) -> [f64; 2] {
        self.inner.half_extent
    }

    /// Homogeneous weight of the anchor: ``1.0`` for a finite patch (``center``
    /// is a position), ``0.0`` for a point at infinity (``center`` is a direction
    /// and the patch is tangent to the unit sphere).
    #[getter]
    fn w(&self) -> f64 {
        self.inner.w
    }

    /// Outward normal (``u_axis × v_axis``).
    #[getter]
    fn normal(&self) -> [f64; 3] {
        let n = self.inner.normal();
        [n.x, n.y, n.z]
    }

    /// Whether the ``cam_from_world`` camera looks at the patch's front face.
    fn is_front_facing(&self, cam_from_world: &PyRigidTransform) -> bool {
        self.inner.is_front_facing(&cam_from_world.inner)
    }
}
