// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for sfmtool-core oriented patches.

use nalgebra::{Point3, Vector3};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use sfmtool_core::patch_cloud::{OrientedPatch, PatchCloud, PatchExtent, PatchNormal, ViewReduce};

use crate::py_rigid_transform::PyRigidTransform;
use crate::py_sfmr_reconstruction::PySfmrReconstruction;

/// An oriented planar patch (surfel) in world space.
///
/// The plane is spanned by orthonormal in-plane axes ``u_axis`` and ``v_axis``;
/// the outward normal is ``u_axis × v_axis``. The patch covers the world points
/// ``center + s·half_extent[0]·u_axis + t·half_extent[1]·v_axis`` for
/// ``(s, t)`` in ``[-1, 1]²``. Pair with :meth:`WarpMap.from_patch` to render its
/// appearance in a camera. See ``specs/core/patch-cloud.md``.
#[pyclass(name = "OrientedPatch", module = "sfmtool")]
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
    ///     up_hint: In-plane reference direction; projected onto the plane to set
    ///         ``u_axis``. If parallel to the normal an arbitrary axis is used.
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

/// A collection of oriented patches built from a reconstruction's 3D points.
///
/// See :meth:`from_reconstruction` and ``specs/core/patch-cloud.md``.
#[pyclass(name = "PatchCloud", module = "sfmtool")]
pub struct PyPatchCloud {
    inner: PatchCloud,
}

#[pymethods]
impl PyPatchCloud {
    /// Build one oriented patch per finite 3D point of a reconstruction.
    ///
    /// Args:
    ///     recon: The reconstruction.
    ///     normal: Normal policy — ``"stored"`` (the reconstruction's stored
    ///         normal, i.e. the mean viewing direction), ``"mean_viewing"``, or
    ///         ``"geometric"`` (local PCA plane fit over ``k_neighbors`` points).
    ///     k_neighbors: Neighbor count for the ``"geometric"`` policy.
    ///     extent: Half-size policy — ``"fixed"`` (world units = ``extent_value``),
    ///         ``"relative_spacing"`` (``extent_value`` × median point spacing),
    ///         or ``"pixel_radius"`` (back-project ``extent_value`` px in each
    ///         observing view, reduced across views by ``pixel_reduce``).
    ///     extent_value: The scalar for the chosen extent policy.
    ///     pixel_reduce: For ``"pixel_radius"``, how to reduce the per-view
    ///         world sizes across a point's views: ``"min"`` (default; keeps the
    ///         patch within the pixel budget in every view), ``"max"``,
    ///         ``"median"``, or ``"mean"``.
    #[staticmethod]
    #[pyo3(signature = (
        recon, normal="mean_viewing", k_neighbors=12,
        extent="relative_spacing", extent_value=2.0, pixel_reduce="min"
    ))]
    fn from_reconstruction(
        recon: &PySfmrReconstruction,
        normal: &str,
        k_neighbors: usize,
        extent: &str,
        extent_value: f64,
        pixel_reduce: &str,
    ) -> PyResult<Self> {
        let normal = match normal {
            "stored" => PatchNormal::Stored,
            "mean_viewing" | "mean" => PatchNormal::MeanViewing,
            "geometric" => PatchNormal::Geometric { k_neighbors },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown normal policy: {other:?} (expected stored|mean_viewing|geometric)"
                )))
            }
        };
        let extent = match extent {
            "fixed" => PatchExtent::Fixed(extent_value),
            "relative_spacing" => PatchExtent::RelativeToSpacing(extent_value),
            "pixel_radius" => {
                let across = match pixel_reduce {
                    "min" => ViewReduce::Min,
                    "max" => ViewReduce::Max,
                    "median" => ViewReduce::Median,
                    "mean" => ViewReduce::Mean,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown pixel_reduce: {other:?} (expected min|max|median|mean)"
                        )))
                    }
                };
                PatchExtent::PixelRadius {
                    radius_px: extent_value,
                    across,
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                "unknown extent policy: {other:?} (expected fixed|relative_spacing|pixel_radius)"
            )))
            }
        };
        Ok(Self {
            inner: PatchCloud::from_reconstruction(&recon.inner, normal, extent),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The oriented patch at index `i`.
    fn __getitem__(&self, i: usize) -> PyResult<PyOrientedPatch> {
        if i >= self.inner.len() {
            return Err(PyIndexError::new_err("patch index out of range"));
        }
        Ok(PyOrientedPatch {
            inner: self.inner.patch(i).clone(),
        })
    }

    /// Source 3D-point index for each patch (parallel to the cloud).
    #[getter]
    fn point_ids(&self) -> Vec<u32> {
        self.inner.point_ids.clone()
    }
}
