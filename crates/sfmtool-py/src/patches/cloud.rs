// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `PatchCloud` type: construction, indexing, and accessors. The heavy
//! per-point kernels live in sibling modules as additional `#[pymethods]`
//! blocks (refine_normals, select_views, localize_keypoints, refine_keypoints,
//! localizability).

use nalgebra::{Point3, Vector3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use sfmtool_core::patch::cloud::{PatchCloud, PatchExtent, PatchNormal};

use super::args::{parse_extent, parse_normal};
use crate::PySfmrReconstruction;

use super::oriented_patch::PyOrientedPatch;
use super::views::PyCameraViews;

/// A collection of oriented patches built from a reconstruction's 3D points.
///
/// See :meth:`from_reconstruction` and ``specs/core/patch-cloud.md``.
#[pyclass(name = "PatchCloud", module = "sfmtool.patches")]
pub struct PyPatchCloud {
    pub(crate) inner: PatchCloud,
}

#[pymethods]
impl PyPatchCloud {
    /// Build one oriented patch per finite 3D point of a reconstruction.
    ///
    /// Args:
    ///     recon: The reconstruction.
    ///     normal: Normal policy — ``"stored"`` (default; the reconstruction's
    ///         stored estimated normal, whatever is in the ``.sfmr``, falling back
    ///         to the mean viewing direction where that is zero/degenerate),
    ///         ``"mean_viewing"`` (mean direction to the observing cameras), or
    ///         ``"geometric"`` (local PCA plane fit over ``k_neighbors`` points).
    ///     k_neighbors: Neighbor count for the ``"geometric"`` policy.
    ///     extent: Half-size policy — ``"feature_size"`` (default; ``extent_value``
    ///         × each observation's keypoint scale back-projected to world, reduced
    ///         by ``feature_reduce``; reads the ``.sift`` files and raises
    ///         ``ValueError`` if a point has no readable scale in any view),
    ///         ``"fixed"`` (world
    ///         units = ``extent_value``), ``"relative_spacing"`` (``extent_value`` ×
    ///         median point spacing), or ``"pixel_radius"`` (back-project
    ///         ``extent_value`` px in each observing view, reduced by ``pixel_reduce``).
    ///     extent_value: The scalar for the chosen extent policy (default 2.5; for
    ///         ``"feature_size"`` the keypoint-scale half-extent multiplier, so the
    ///         full patch edge is ``5 ×`` the projected feature size).
    ///     pixel_reduce: For ``"pixel_radius"``, the view reduce — ``"min"``
    ///         (default), ``"max"``, ``"median"``, or ``"mean"``.
    ///     feature_reduce: For ``"feature_size"``, the view reduce (default
    ///         ``"median"``).
    ///     exclude_points_at_infinity: When ``False`` (default), each point at
    ///         infinity also gets a tangent-sphere frame (``w = 0``) around its
    ///         direction; every patch operation handles these. Pass ``True`` to
    ///         emit patches for finite points only (e.g. an operation that scatters
    ///         per-point results back and must leave infinity points untouched).
    #[staticmethod]
    #[pyo3(signature = (
        recon, normal="stored", k_neighbors=12,
        extent="feature_size", extent_value=2.5,
        pixel_reduce="min", feature_reduce="median",
        exclude_points_at_infinity=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_reconstruction(
        recon: &PySfmrReconstruction,
        normal: &str,
        k_neighbors: usize,
        extent: &str,
        extent_value: f64,
        pixel_reduce: &str,
        feature_reduce: &str,
        exclude_points_at_infinity: bool,
    ) -> PyResult<Self> {
        let normal = parse_normal(normal, k_neighbors)?;
        let extent = parse_extent(extent, extent_value, pixel_reduce, feature_reduce)?;
        let inner = PatchCloud::from_reconstruction(
            &recon.inner,
            normal,
            extent,
            exclude_points_at_infinity,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a patch cloud from in-memory arrays instead of a reconstruction — the
    /// array counterpart of :meth:`from_reconstruction` (same normal / extent
    /// policies, up-hint, infinity frames, and errors). See
    /// ``specs/core/patch-cloud.md``, "Patch operations without a reconstruction".
    ///
    /// Args:
    ///     views: A :class:`CameraViews` — the posed views the patches are sized
    ///         and oriented against.
    ///     positions_xyzw: ``(P, 4)`` float64 homogeneous 3D points; a ``w = 0`` row
    ///         is a point at infinity (its ``xyz`` is a direction).
    ///     track_point_indexes: ``(M,)`` uint32 point id per observation, grouped by
    ///         point (**nondecreasing**; a violation is a ``ValueError``). Every
    ///         point must have at least one observation.
    ///     track_image_indexes: ``(M,)`` uint32 image (view) index per observation,
    ///         parallel to ``track_point_indexes``.
    ///     keypoint_scales: Optional ``(M,)`` float64 keypoint scale ``σ`` per
    ///         observation; **required** for ``extent="feature_size"``. A ``NaN``
    ///         entry counts as an unreadable scale (same
    ///         ``MissingFeatureScale`` handling as a missing ``.sift`` scale).
    ///     normals: Optional ``(P, 3)`` float64 per-point normals; **required** for
    ///         ``normal="stored"`` (a zero/degenerate row falls back to the mean
    ///         viewing direction).
    ///     normal: Normal policy — ``"mean_viewing"`` (default), ``"stored"``, or
    ///         ``"geometric"``.
    ///     k_neighbors, extent, extent_value, pixel_reduce, feature_reduce,
    ///     exclude_points_at_infinity: As in :meth:`from_reconstruction`
    ///         (``extent`` defaults to ``"feature_size"``, ``extent_value`` to
    ///         ``2.5``).
    ///
    /// The resulting patches' ``point_indexes`` are row indexes into
    /// ``positions_xyzw``.
    #[staticmethod]
    #[pyo3(signature = (
        views, positions_xyzw, track_point_indexes, track_image_indexes,
        keypoint_scales=None, normals=None, normal="mean_viewing", k_neighbors=12,
        extent="feature_size", extent_value=2.5, pixel_reduce="min",
        feature_reduce="median", exclude_points_at_infinity=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_tracks(
        views: &PyCameraViews,
        positions_xyzw: PyReadonlyArray2<'_, f64>,
        track_point_indexes: PyReadonlyArray1<'_, u32>,
        track_image_indexes: PyReadonlyArray1<'_, u32>,
        keypoint_scales: Option<PyReadonlyArray1<'_, f64>>,
        normals: Option<PyReadonlyArray2<'_, f64>>,
        normal: &str,
        k_neighbors: usize,
        extent: &str,
        extent_value: f64,
        pixel_reduce: &str,
        feature_reduce: &str,
        exclude_points_at_infinity: bool,
    ) -> PyResult<Self> {
        let normal_policy = parse_normal(normal, k_neighbors)?;
        let extent_policy = parse_extent(extent, extent_value, pixel_reduce, feature_reduce)?;

        let pos = positions_xyzw.as_array();
        if pos.shape()[1] != 4 {
            return Err(PyValueError::new_err(format!(
                "positions_xyzw must have shape (P, 4), got (_, {})",
                pos.shape()[1]
            )));
        }
        let p = pos.shape()[0];
        let positions: Vec<Point3<f64>> = (0..p)
            .map(|i| Point3::new(pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]))
            .collect();
        let weights: Vec<f64> = (0..p).map(|i| pos[[i, 3]]).collect();

        let tpi = track_point_indexes.as_array();
        let tii = track_image_indexes.as_array();
        let m = tpi.shape()[0];
        if tii.shape()[0] != m {
            return Err(PyValueError::new_err(format!(
                "track_point_indexes and track_image_indexes must be the same length \
                 ({m} vs {})",
                tii.shape()[0]
            )));
        }
        let n_images = views.cameras.len() as u32;

        // Group the observations into per-point offsets, validating that the point
        // ids are nondecreasing (the grouping the shared routine assumes), in range,
        // that image indexes are in range, and that every point has ≥1 observation.
        let mut obs_offsets = vec![0usize; p + 1];
        let mut counts = vec![0usize; p];
        let mut prev: Option<u32> = None;
        for j in 0..m {
            let pid = tpi[j];
            if let Some(pv) = prev {
                if pid < pv {
                    return Err(PyValueError::new_err(
                        "track_point_indexes must be nondecreasing (grouped by point)",
                    ));
                }
            }
            prev = Some(pid);
            if pid as usize >= p {
                return Err(PyValueError::new_err(format!(
                    "track_point_indexes[{j}] = {pid} is out of range for {p} points"
                )));
            }
            let img = tii[j];
            if img >= n_images {
                return Err(PyValueError::new_err(format!(
                    "track_image_indexes[{j}] = {img} is out of range for {n_images} views"
                )));
            }
            counts[pid as usize] += 1;
        }
        for (pid, &c) in counts.iter().enumerate() {
            if c == 0 {
                return Err(PyValueError::new_err(format!(
                    "point {pid} has no observation; every point needs at least one"
                )));
            }
            obs_offsets[pid + 1] = obs_offsets[pid] + c;
        }
        let obs_images: Vec<u32> = tii.to_vec();

        // FeatureSize needs the caller's per-observation scales.
        let scales_vec: Option<Vec<f64>> = match &keypoint_scales {
            Some(arr) => {
                let a = arr.as_array();
                if a.shape()[0] != m {
                    return Err(PyValueError::new_err(format!(
                        "keypoint_scales must have length {m} (parallel to the tracks), got {}",
                        a.shape()[0]
                    )));
                }
                Some(a.to_vec())
            }
            None => {
                if matches!(extent_policy, PatchExtent::FeatureSize { .. }) {
                    return Err(PyValueError::new_err(
                        "keypoint_scales is required for extent=\"feature_size\"",
                    ));
                }
                None
            }
        };

        // Stored normals (required for normal="stored").
        let stored_vec: Option<Vec<Vector3<f64>>> = match &normals {
            Some(arr) => {
                let a = arr.as_array();
                if a.shape()[0] != p || a.shape()[1] != 3 {
                    return Err(PyValueError::new_err(format!(
                        "normals must have shape ({p}, 3), got ({}, {})",
                        a.shape()[0],
                        a.shape()[1]
                    )));
                }
                Some(
                    (0..p)
                        .map(|i| Vector3::new(a[[i, 0]], a[[i, 1]], a[[i, 2]]))
                        .collect(),
                )
            }
            None => {
                if matches!(normal_policy, PatchNormal::Stored) {
                    return Err(PyValueError::new_err(
                        "normals is required for normal=\"stored\"",
                    ));
                }
                None
            }
        };

        // Per-view poses + focal lengths from the CameraViews.
        let cam_focals: Vec<f64> = views.cameras.iter().map(|c| c.focal_lengths().0).collect();

        let inner = PatchCloud::from_tracks(
            &positions,
            &weights,
            stored_vec.as_deref(),
            &obs_offsets,
            &obs_images,
            scales_vec.as_deref(),
            &views.quaternions,
            &views.translations,
            &cam_focals,
            normal_policy,
            extent_policy,
            exclude_points_at_infinity,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a patch cloud from per-point half-extent vector arrays and centers.
    ///
    /// The inverse of the per-point ``patch_u_halfvec_xyz`` / ``patch_v_halfvec_xyz``
    /// layout: each present row (non-zero ``u``) becomes one patch whose
    /// ``point_indexes`` entry is that row index, with the half-extent vectors split
    /// back into a unit axis and a half-size. Use this to assemble a renumbered or
    /// culled cloud (e.g. after dropping points) to hand to
    /// ``SfmrReconstruction.clone_with_changes(patches=...)``.
    ///
    /// Args:
    ///     half_u_xyz: ``(N, 3)`` float32 in-plane half-extent vectors ``u``.
    ///     half_v_xyz: ``(N, 3)`` float32 in-plane half-extent vectors ``v``.
    ///     centers: ``(N, 3)`` float64 patch centers (each point's position).
    #[staticmethod]
    #[pyo3(signature = (half_u_xyz, half_v_xyz, centers))]
    fn from_halfvec_arrays(
        half_u_xyz: PyReadonlyArray2<'_, f32>,
        half_v_xyz: PyReadonlyArray2<'_, f32>,
        centers: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let u = half_u_xyz.as_array();
        let v = half_v_xyz.as_array();
        let c = centers.as_array();
        let n = u.shape()[0];
        if u.shape()[1] != 3 || v.shape()[1] != 3 || c.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "from_halfvec_arrays: half_u_xyz, half_v_xyz, centers must each have shape (N, 3)",
            ));
        }
        if v.shape()[0] != n || c.shape()[0] != n {
            return Err(PyValueError::new_err(format!(
                "from_halfvec_arrays: row counts must match (u={}, v={}, centers={})",
                n,
                v.shape()[0],
                c.shape()[0]
            )));
        }
        let centers_vec: Vec<Point3<f64>> = (0..n)
            .map(|i| Point3::new(c[[i, 0]], c[[i, 1]], c[[i, 2]]))
            .collect();
        let inner = PatchCloud::from_halfvec_arrays(&u.to_owned(), &v.to_owned(), &centers_vec);
        Ok(Self { inner })
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
    fn point_indexes(&self) -> Vec<u32> {
        self.inner.point_indexes.clone()
    }
}
