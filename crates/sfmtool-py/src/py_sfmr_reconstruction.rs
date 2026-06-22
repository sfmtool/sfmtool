// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core SfmrReconstruction type.

use nalgebra::Point3;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use sfmtool_core::analysis::infinity::Classification;
use sfmtool_core::geometry::viewing_angle::viewing_rays;
use sfmtool_core::reconstruction::triangulation::{depth_uncertainty_batch, triangulate_batch};
use sfmtool_core::SfmrReconstruction;

use crate::helpers::{serde_to_py, u128_bytes_to_py};
use crate::py_sfmr_io::parse_sfmr_data_from_dict;
use crate::PyCameraIntrinsics;

/// A loaded SfM reconstruction from a `.sfmr` file.
///
/// Wraps the Rust `SfmrReconstruction` and exposes scalar properties,
/// metadata, cameras, and image names to Python.
#[pyclass(name = "SfmrReconstruction", module = "sfmtool")]
pub struct PySfmrReconstruction {
    pub(crate) inner: SfmrReconstruction,
}

#[pymethods]
impl PySfmrReconstruction {
    /// Load a reconstruction from a `.sfmr` file path.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = SfmrReconstruction::load(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a reconstruction from a data dict (same format as `write_sfmr`).
    ///
    /// The dict must contain metadata, cameras, image data, point data,
    /// and track data. Depth statistics and reprojection errors are
    /// recomputed from scratch (using camera models and `.sift` files),
    /// so the values passed in for those fields are ignored.
    ///
    /// Args:
    ///     workspace_dir: Resolved workspace directory path.
    ///     data: Dict with the same keys as `write_sfmr` expects.
    #[staticmethod]
    fn from_data(
        py: Python<'_>,
        workspace_dir: PathBuf,
        data: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let mut sfmr_data = parse_sfmr_data_from_dict(py, data, false)?;
        sfmr_data.workspace_dir = Some(workspace_dir.clone());
        let mut inner = SfmrReconstruction::from_sfmr_data(sfmr_data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        inner.workspace_dir = workspace_dir;
        inner
            .recompute_depth_statistics()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        inner
            .recompute_point_errors()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Save this reconstruction to a `.sfmr` file path.
    ///
    /// Optionally updates the metadata with operation details before writing.
    /// If ``operation`` is provided, the metadata fields ``operation``, ``tool``,
    /// ``tool_version``, ``timestamp``, and counts are updated automatically.
    ///
    /// Args:
    ///     path: Output file path.
    ///     operation: Name of the operation (e.g. ``"xform"``, ``"sfm_solve"``).
    ///     tool_name: Tool that performed the operation (default ``"sfmtool"``).
    ///     tool_options: Optional dict of operation-specific metadata to merge
    ///         into ``metadata.tool_options``.
    ///
    /// The write preserves the in-memory ``normals`` of every point that has one
    /// (recomputing only the missing/zero rows from geometry), so normals set via
    /// ``clone_with_changes(normals=...)`` — e.g. by ``sfm xform
    /// --refine-normals`` — survive the round trip. A reconstruction with
    /// ``has_normals`` ``False`` writes no normals at all. Any attached patch
    /// cloud is written as the per-point patch frame in ``points3d/`` (format
    /// version 3+).
    #[pyo3(signature = (path, operation=None, tool_name=None, tool_options=None))]
    fn save(
        &mut self,
        py: Python<'_>,
        path: PathBuf,
        operation: Option<&str>,
        tool_name: Option<&str>,
        tool_options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        // Update metadata if operation is provided
        if let Some(op) = operation {
            let meta = &mut self.inner.metadata;
            meta.operation = op.to_string();
            meta.tool = tool_name.unwrap_or("sfmtool").to_string();
            meta.image_count = self.inner.images.len() as u32;
            meta.point_count = self.inner.points.len() as u32;
            meta.observation_count = self.inner.tracks.len() as u32;
            meta.camera_count = self.inner.cameras.len() as u32;
            meta.timestamp = chrono::Local::now().to_rfc3339();

            // Update workspace paths relative to output file
            let output_path = std::path::absolute(&path).unwrap_or_else(|_| path.clone());
            if let Some(parent) = output_path.parent() {
                if let Some(rel) = pathdiff::diff_paths(&self.inner.workspace_dir, parent) {
                    meta.workspace.relative_path = rel.to_string_lossy().replace('\\', "/");
                }
            }
            meta.workspace.absolute_path = self.inner.workspace_dir.to_string_lossy().to_string();
        }

        // Merge tool_options if provided
        if let Some(opts) = tool_options {
            for (key, value) in opts.iter() {
                let k: String = key.extract()?;
                let v: serde_json::Value = crate::helpers::py_to_serde(py, &value)?;
                self.inner.metadata.tool_options.insert(k, v);
            }
        }

        self.inner
            .save(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Number of registered images.
    #[getter]
    fn image_count(&self) -> usize {
        self.inner.image_count()
    }

    /// Number of 3D points.
    #[getter]
    fn point_count(&self) -> usize {
        self.inner.point_count()
    }

    /// Number of 3D points at infinity (`w == 0`).
    #[getter]
    fn infinity_point_count(&self) -> usize {
        self.inner.infinity_point_count
    }

    /// Number of track observations.
    #[getter]
    fn observation_count(&self) -> usize {
        self.inner.observation_count()
    }

    /// Number of camera models.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.camera_count()
    }

    /// Resolved workspace directory path as a string.
    #[getter]
    fn workspace_dir(&self) -> String {
        self.inner.workspace_dir.to_string_lossy().to_string()
    }

    /// The content_xxh128 hash string (32 lowercase hex chars).
    #[getter]
    fn content_xxh128(&self) -> &str {
        &self.inner.content_hash.content_xxh128
    }

    /// Reconstruction metadata as a Python dict (serialized via JSON).
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.inner.metadata)
    }

    /// Camera intrinsic parameters as a Python list of CameraIntrinsics objects.
    #[getter]
    fn cameras(&self) -> Vec<PyCameraIntrinsics> {
        self.inner
            .cameras
            .iter()
            .map(|c| PyCameraIntrinsics { inner: c.clone() })
            .collect()
    }

    /// List of image names (paths relative to the workspace).
    #[getter]
    fn image_names(&self) -> Vec<String> {
        self.inner.images.iter().map(|im| im.name.clone()).collect()
    }

    // ── Image data array getters ─────────────────────────────────────

    /// Camera index for each image, shape `(N,)`.
    #[getter]
    fn camera_indexes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let vec: Vec<u32> = self.inner.images.iter().map(|im| im.camera_index).collect();
        PyArray1::from_vec(py, vec)
    }

    /// WXYZ quaternions for each image pose, shape `(N, 4)`.
    #[getter]
    fn quaternions_wxyz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let vec2: Vec<Vec<f64>> = self
            .inner
            .images
            .iter()
            .map(|im| {
                let q = im.quaternion_wxyz.quaternion();
                vec![q.w, q.i, q.j, q.k]
            })
            .collect();
        PyArray2::from_vec2(py, &vec2).unwrap()
    }

    /// Translation vectors for each image pose, shape `(N, 3)`.
    #[getter]
    fn translations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let vec2: Vec<Vec<f64>> = self
            .inner
            .images
            .iter()
            .map(|im| {
                vec![
                    im.translation_xyz.x,
                    im.translation_xyz.y,
                    im.translation_xyz.z,
                ]
            })
            .collect();
        PyArray2::from_vec2(py, &vec2).unwrap()
    }

    /// Per-image feature tool hashes as ``list[bytes]`` (16 bytes each), or
    /// ``None`` unless :attr:`feature_source` is ``"sift_files"`` (an
    /// ``embedded_patches`` reconstruction has no `.sift` link).
    #[getter]
    fn feature_tool_hashes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        self.inner
            .feature_tool_hashes()
            .map(|h| u128_bytes_to_py(py, h))
            .transpose()
    }

    /// Per-image SIFT content hashes as ``list[bytes]`` (16 bytes each), or
    /// ``None`` unless :attr:`feature_source` is ``"sift_files"``.
    #[getter]
    fn sift_content_hashes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        self.inner
            .sift_content_hashes()
            .map(|h| u128_bytes_to_py(py, h))
            .transpose()
    }

    /// Observation source: ``"sift_files"`` (observations reference ``.sift``
    /// features) or ``"embedded_patches"`` (per-observation keypoints stored
    /// inline). See ``specs/formats/sfmr-v4-patch-keypoints.md``.
    #[getter]
    fn feature_source(&self) -> &str {
        self.inner.feature_source()
    }

    /// Per-observation sub-pixel keypoints ``(K, 2)`` (image-space ``(u, v)``),
    /// parallel to the track arrays. ``None`` unless :attr:`feature_source` is
    /// ``"embedded_patches"``.
    #[getter]
    fn keypoints_xy<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        self.inner
            .keypoints_xy()
            .map(|kp| kp.clone().into_pyarray(py))
    }

    /// Per-image source-image hashes as ``list[bytes]`` (16-byte XXH128 each), or
    /// ``None`` unless :attr:`feature_source` is ``"embedded_patches"``. The same
    /// value the image's ``.sift`` records as ``image_file_xxh128``.
    #[getter]
    fn image_file_hashes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        self.inner
            .image_file_hashes()
            .map(|h| u128_bytes_to_py(py, h))
            .transpose()
    }

    // ── Point data array getters ─────────────────────────────────────

    /// 3D point positions, shape `(M, 3)`.
    ///
    /// For a finite point this is its Euclidean position; for a point at
    /// infinity (`w == 0`) it is a unit-length direction. Use
    /// `point_is_at_infinity` to disambiguate the two, or `positions_xyzw` for
    /// the unambiguous homogeneous form.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let vec2: Vec<Vec<f64>> = self
            .inner
            .points
            .iter()
            .map(|pt| vec![pt.position.x, pt.position.y, pt.position.z])
            .collect();
        PyArray2::from_vec2(py, &vec2).unwrap()
    }

    /// Homogeneous 3D point positions, shape `(M, 4)`.
    ///
    /// Each row is `(x, y, z, w)`: a finite point has `w == 1` with Euclidean
    /// coordinates `(x, y, z)`; a point at infinity has `w == 0` with a
    /// unit-length direction `(x, y, z)`. This is the canonical form: it is
    /// what `clone_with_changes(positions=...)` accepts and what `.sfmr`
    /// version 2 files store.
    #[getter]
    fn positions_xyzw<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let vec2: Vec<Vec<f64>> = self
            .inner
            .points
            .iter()
            .map(|pt| vec![pt.position.x, pt.position.y, pt.position.z, pt.w])
            .collect();
        PyArray2::from_vec2(py, &vec2).unwrap()
    }

    /// Boolean mask of points at infinity (`w == 0`), shape `(M,)`.
    ///
    /// Pairs with `positions` for finite-only handling, e.g.
    /// ``recon.positions[~recon.point_is_at_infinity]`` selects the Euclidean
    /// positions of just the finite points.
    #[getter]
    fn point_is_at_infinity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let vec: Vec<bool> = self
            .inner
            .points
            .iter()
            .map(|pt| pt.is_at_infinity())
            .collect();
        PyArray1::from_vec(py, vec)
    }

    /// RGB colors for 3D points, shape `(M, 3)`.
    #[getter]
    fn colors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let vec2: Vec<Vec<u8>> = self
            .inner
            .points
            .iter()
            .map(|pt| vec![pt.color[0], pt.color[1], pt.color[2]])
            .collect();
        PyArray2::from_vec2(py, &vec2).unwrap()
    }

    /// Reprojection errors for 3D points, shape `(M,)`.
    #[getter]
    fn errors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let vec: Vec<f32> = self.inner.points.iter().map(|pt| pt.error).collect();
        PyArray1::from_vec(py, vec)
    }

    /// Surface normals for 3D points, shape `(M, 3)`.
    ///
    /// Always returns an array; when :attr:`has_normals` is ``False`` every row
    /// is ``(0, 0, 0)`` (the reconstruction carries no normals).
    #[getter]
    fn normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let vec2: Vec<Vec<f32>> = self
            .inner
            .points
            .iter()
            .map(|pt| vec![pt.normal.x, pt.normal.y, pt.normal.z])
            .collect();
        PyArray2::from_vec2(py, &vec2).unwrap()
    }

    /// Whether this reconstruction carries per-point normals. When ``False``,
    /// :attr:`normals` is all-zero and no `normals_xyz` array is written.
    #[getter]
    fn has_normals(&self) -> bool {
        self.inner.has_normals
    }

    /// The attached oriented-patch cloud, or ``None`` when this reconstruction
    /// carries no patch data. The returned cloud is a copy (geometry only;
    /// bitmaps, if stored, are not loaded into the cloud).
    #[getter]
    fn patches(&self) -> Option<crate::py_patch_cloud::PyPatchCloud> {
        let u = self.inner.patch_u_halfvec_xyz.as_ref()?;
        let v = self.inner.patch_v_halfvec_xyz.as_ref()?;
        // The patch center for each point is the point's own position.
        let centers: Vec<Point3<f64>> = self.inner.points.iter().map(|p| p.position).collect();
        Some(crate::py_patch_cloud::PyPatchCloud {
            inner: sfmtool_core::patch::cloud::PatchCloud::from_halfvec_arrays(u, v, &centers),
        })
    }

    /// The per-3D-point RGBA patch bitmaps as a ``(N, R, R, 4)`` uint8 array, or
    /// ``None`` when none are stored. Rows for points with no patch are zero; the
    /// alpha channel holds a per-pixel cross-view agreement confidence. Attach via
    /// ``clone_with_changes(patch_bitmaps=…)`` (the patch frame must be present).
    #[getter]
    fn patch_bitmaps<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray4<u8>>> {
        let b = self.inner.patch_bitmaps_y_x_rgba.as_ref()?;
        Some(b.clone().into_pyarray(py))
    }

    // ── Track data array getters ─────────────────────────────────────

    /// Image indexes for track observations, shape `(K,)`.
    #[getter]
    fn track_image_indexes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let vec: Vec<u32> = self.inner.tracks.iter().map(|t| t.image_index).collect();
        PyArray1::from_vec(py, vec)
    }

    /// Feature indexes for track observations, shape `(K,)`, or ``None`` unless
    /// :attr:`feature_source` is ``"sift_files"`` (an ``embedded_patches``
    /// reconstruction has none; use :attr:`keypoints_xy` instead).
    #[getter]
    fn track_feature_indexes<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u32>>> {
        self.inner
            .feature_indexes()
            .map(|f| PyArray1::from_vec(py, f.to_vec()))
    }

    /// 3D point indexes for track observations, shape `(K,)`.
    #[getter]
    fn track_point_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let vec: Vec<u32> = self.inner.tracks.iter().map(|t| t.point_index).collect();
        PyArray1::from_vec(py, vec)
    }

    /// Observation counts per 3D point, shape `(M,)`.
    #[getter]
    fn observation_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_vec(py, self.inner.observation_counts.clone())
    }

    /// RGB thumbnails for each image, shape `(N, 128, 128, 3)`.
    ///
    /// Returns a read-only numpy view into the Rust-owned data (zero-copy).
    #[getter]
    fn thumbnails_y_x_rgb<'py>(self_: &Bound<'py, Self>) -> Bound<'py, PyArray4<u8>> {
        // Get a raw pointer to the thumbnail array. This is safe because:
        // 1. #[pyclass] objects are heap-allocated and pinned — the data won't move.
        // 2. The returned numpy array holds a reference to `self_` (via the base/container
        //    object), preventing garbage collection while the view is alive.
        let ptr = {
            let borrow = self_.borrow();
            &borrow.inner.thumbnails_y_x_rgb as *const ndarray::Array4<u8>
        };
        unsafe { PyArray4::borrow_from_array(&*ptr, self_.clone().into_any()) }
    }

    // ── Depth data getters ───────────────────────────────────────────

    /// Depth statistics as a Python dict (serialized via JSON).
    #[getter]
    fn depth_statistics(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.inner.depth_statistics)
    }

    /// Depth histogram counts, shape `(N, num_buckets)`.
    #[getter]
    fn depth_histogram_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        PyArray2::from_vec2(py, &self.inner.depth_histogram_counts).unwrap()
    }

    /// Return a new reconstruction containing only the images at
    /// `image_indices` (0-based, in the given order).
    ///
    /// Observations referencing removed images are dropped. Frames with no
    /// kept image are dropped; rig and sensor definitions are preserved.
    ///
    /// If ``drop_orphaned_points`` is true, 3D points with zero remaining
    /// observations are removed and point IDs are remapped. Otherwise, all
    /// 3D points are kept (some may have zero observations).
    #[pyo3(signature = (image_indices, drop_orphaned_points=false))]
    fn subset_by_image_indices(
        &self,
        image_indices: PyReadonlyArray1<u32>,
        drop_orphaned_points: bool,
    ) -> PyResult<Self> {
        let slice = image_indices.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "image_indices must be a contiguous array: {e}"
            ))
        })?;
        let filtered = self
            .inner
            .subset_by_image_indices(slice, drop_orphaned_points)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: filtered })
    }

    /// Filter 3D points by a boolean mask, returning a new reconstruction.
    ///
    /// Points where `mask[i]` is `true` are kept. Tracks are filtered and
    /// remapped to contiguous point IDs. Image data is copied unchanged.
    fn filter_points_by_mask(&self, mask: PyReadonlyArray1<bool>) -> PyResult<Self> {
        let mask_slice = mask.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("mask must be a contiguous array: {e}"))
        })?;
        if mask_slice.len() != self.inner.points.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask length ({}) must match point count ({})",
                mask_slice.len(),
                self.inner.points.len()
            )));
        }
        let filtered = self.inner.filter_points_by_mask(mask_slice);
        Ok(Self { inner: filtered })
    }

    /// Compute per-observation reprojection errors for a single image.
    ///
    /// Loads feature positions from the image's `.sift` file, projects each
    /// observed 3D point through the camera, and returns the pixel distance
    /// between projected and observed positions.
    ///
    /// Returns a `(K, 2)` array where column 0 is the feature index and
    /// column 1 is the reprojection error in pixels. Points behind the
    /// camera produce NaN.
    fn compute_observation_reprojection_errors<'py>(
        &self,
        py: Python<'py>,
        image_index: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if image_index >= self.inner.image_count() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "image_index {} out of range (image_count={})",
                image_index,
                self.inner.image_count()
            )));
        }
        let results = self
            .inner
            .compute_observation_reprojection_errors(image_index)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let n = results.len();
        let mut flat = Vec::with_capacity(n * 2);
        for (feat_idx, error) in &results {
            flat.push(*feat_idx as f32);
            flat.push(*error);
        }
        let arr = ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Recompute per-point mean reprojection errors from scratch.
    ///
    /// For each image, loads feature positions from the `.sift` file and
    /// reprojects all observed 3D points through the camera model. Each
    /// point's error is set to the mean pixel-space reprojection error
    /// across all its observations.
    ///
    /// This replaces any errors read from COLMAP/GLOMAP binary files,
    /// ensuring errors are always in pixel coordinates.
    fn recompute_point_errors(&mut self) -> PyResult<()> {
        self.inner
            .recompute_point_errors()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Apply an SE(3) similarity transform to this reconstruction.
    ///
    /// Transforms all 3D point positions and camera poses. Returns a new
    /// reconstruction; the original is unchanged.
    fn apply_transform(&self, transform: &crate::PySe3Transform) -> Self {
        Self {
            inner: self.inner.apply_se3_transform(&transform.inner),
        }
    }

    /// Reclassify finite points whose depth is unconstrained as points at
    /// infinity, returning a new reconstruction.
    ///
    /// A finite point becomes a point at infinity (``w = 0``) when the
    /// triangulation of its observation rays is statistically indistinguishable
    /// from infinity: a degenerate or behind-camera solve, or — in the
    /// ill-conditioned regime — an inverse-depth z-score below the cutoff. The
    /// per-ray angular noise is ``max(reprojection_error, noise_floor_px) / fᵢ``.
    /// Its coordinate is replaced with the bearing-mean direction of its
    /// observation rays. Points already at infinity, and points with fewer than
    /// two observations, are left unchanged.
    ///
    /// Args:
    ///     noise_floor_px: SIFT keypoint localisation noise floor in pixels;
    ///         the per-point noise estimate is never taken below this. Defaults
    ///         to 1.0.
    #[pyo3(signature = (noise_floor_px=None))]
    fn classify_points_at_infinity(&self, noise_floor_px: Option<f64>) -> Self {
        let floor =
            noise_floor_px.unwrap_or(sfmtool_core::analysis::infinity::DEFAULT_NOISE_FLOOR_PX);
        Self {
            inner: self.inner.classify_points_at_infinity(floor),
        }
    }

    /// Triangulation observability diagnostics for the stored finite points.
    ///
    /// Re-triangulates each finite point from its observing cameras' rays toward
    /// the *stored* point (no ``.sift`` reads) and returns the per-point
    /// conditioning and the noise-calibrated depth uncertainty. Points at
    /// infinity and points with fewer than two observations get ``NaN`` (their
    /// depth is undefined). Arrays are length ``M = point_count`` and align with
    /// ``positions`` / ``errors`` for masking.
    ///
    /// Args:
    ///     noise_px: Per-ray pixel localisation noise; the per-point noise is
    ///         ``max(reprojection_error, noise_px)``. Defaults to 1.0.
    ///
    /// Returns:
    ///     A dict of arrays, each shape ``(M,)`` float64: ``condition_number``
    ///     (of the normal matrix ``A``, scales with track length),
    ///     ``depth_sigma`` (1σ depth uncertainty along the mean viewing
    ///     direction), and ``inverse_depth_z`` (scale-free ``depth / sigma``;
    ///     small ⇒ near-infinity).
    #[pyo3(signature = (noise_px=1.0))]
    fn triangulation_diagnostics(&self, py: Python<'_>, noise_px: f64) -> PyResult<Py<PyAny>> {
        let recon = &self.inner;
        let centers: Vec<Point3<f64>> = recon.images.iter().map(|im| im.camera_center()).collect();
        let focal_max: Vec<f64> = recon
            .images
            .iter()
            .map(|im| {
                let (fx, fy) = recon.cameras[im.camera_index as usize].focal_lengths();
                fx.max(fy)
            })
            .collect();

        let m = recon.points.len();
        let mut condition_number = vec![f64::NAN; m];
        let mut depth_sigma = vec![f64::NAN; m];
        let mut inverse_depth_z = vec![f64::NAN; m];

        // Build one CSR batch over every diagnosable finite point.
        let mut dirs = Vec::new();
        let mut ray_centers = Vec::new();
        let mut sigma_rad = Vec::new();
        let mut offsets = vec![0usize];
        let mut point_of_track = Vec::new();
        for (pidx, pt) in recon.points.iter().enumerate() {
            if pt.is_at_infinity() {
                continue;
            }
            let obs = recon.observations_for_point(pidx);
            if obs.len() < 2 {
                continue;
            }
            let rays = viewing_rays(
                pt.position,
                &centers,
                obs.iter().map(|o| o.image_index as usize),
            );
            if rays.len() < 2 {
                continue;
            }
            let noise = (pt.error as f64).max(noise_px);
            for (img, r) in &rays {
                dirs.push(*r);
                ray_centers.push(centers[*img]);
                sigma_rad.push(noise / focal_max[*img]);
            }
            offsets.push(dirs.len());
            point_of_track.push(pidx);
        }

        let tris = triangulate_batch(&dirs, &ray_centers, &offsets);
        let dus = depth_uncertainty_batch(&tris, &dirs, &ray_centers, &offsets, &sigma_rad);
        for (track, &pidx) in point_of_track.iter().enumerate() {
            condition_number[pidx] = tris[track].condition_number;
            depth_sigma[pidx] = dus[track].sigma;
            inverse_depth_z[pidx] = dus[track].inverse_depth_z;
        }

        let dict = PyDict::new(py);
        dict.set_item("condition_number", PyArray1::from_vec(py, condition_number))?;
        dict.set_item("depth_sigma", PyArray1::from_vec(py, depth_sigma))?;
        dict.set_item("inverse_depth_z", PyArray1::from_vec(py, inverse_depth_z))?;
        Ok(dict.into_any().unbind())
    }

    /// Full triangulation analysis of a single 3D point, re-deriving its rays
    /// from the workspace ``.sift`` files.
    ///
    /// Reads the observing images' ``.sift`` files, so they must be present.
    /// Returns a dict: ``w``, ``position`` (3,), ``error``, ``color`` (3,),
    /// ``classification`` ("finite"/"at_infinity"/"indeterminate"),
    /// ``triangulated_point`` (3,), ``eigenvalues`` (3,), ``condition_number``,
    /// ``in_front``, ``depth``, ``sigma``, ``inverse_depth_z``,
    /// ``resolvable_distance``, ``finite_horizon``, ``baseline_span``,
    /// ``max_ray_angle_deg``, and ``observations`` (list of dicts with
    /// ``image_index``, ``image_name``, ``feature_index``, ``incidence_deg``).
    ///
    /// Args:
    ///     point_index: Index into the points array.
    ///     noise_px: Per-ray pixel noise; per-point noise is
    ///         ``max(reprojection_error, noise_px)``. Defaults to 1.0.
    #[pyo3(signature = (point_index, noise_px=1.0))]
    fn inspect_point(
        &self,
        py: Python<'_>,
        point_index: usize,
        noise_px: f64,
    ) -> PyResult<Py<PyAny>> {
        if point_index >= self.inner.points.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "point index {point_index} out of range (0..{})",
                self.inner.points.len()
            )));
        }
        let rep = self
            .inner
            .inspect_point(point_index, noise_px)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("w", rep.w)?;
        dict.set_item("position", [rep.position.x, rep.position.y, rep.position.z])?;
        dict.set_item("error", rep.error)?;
        dict.set_item("color", [rep.color[0], rep.color[1], rep.color[2]])?;
        dict.set_item(
            "classification",
            match rep.classification {
                Classification::Finite(_) => "finite",
                Classification::Infinity(_) => "at_infinity",
                Classification::Indeterminate => "indeterminate",
            },
        )?;
        dict.set_item(
            "triangulated_point",
            [
                rep.triangulated_point.x,
                rep.triangulated_point.y,
                rep.triangulated_point.z,
            ],
        )?;
        dict.set_item("eigenvalues", rep.eigenvalues.to_vec())?;
        dict.set_item("condition_number", rep.condition_number)?;
        dict.set_item("in_front", rep.in_front)?;
        dict.set_item("depth", rep.depth)?;
        dict.set_item("sigma", rep.sigma)?;
        dict.set_item("inverse_depth_z", rep.inverse_depth_z)?;
        dict.set_item("resolvable_distance", rep.resolvable_distance)?;
        dict.set_item("finite_horizon", rep.finite_horizon)?;
        dict.set_item("baseline_span", rep.baseline_span)?;
        dict.set_item("max_ray_angle_deg", rep.max_ray_angle_deg)?;

        let obs_list = PyList::empty(py);
        for o in &rep.observations {
            let od = PyDict::new(py);
            od.set_item("image_index", o.image_index)?;
            od.set_item("image_name", &o.image_name)?;
            od.set_item("feature_index", o.feature_index)?;
            od.set_item("incidence_deg", o.incidence_deg)?;
            obs_list.append(od)?;
        }
        dict.set_item("observations", obs_list)?;
        Ok(dict.into_any().unbind())
    }

    /// Materialise every point at infinity as a finite point, returning a new
    /// reconstruction.
    ///
    /// A ``w = 0`` point is placed along its stored direction, at the
    /// camera-cloud centroid plus the largest per-camera distance beyond which
    /// its parallax falls below one pixel. This does not triangulate — a point
    /// at infinity has no depth to recover. Finite points are unchanged.
    fn materialize_points_at_infinity(&self) -> Self {
        Self {
            inner: self.inner.materialize_points_at_infinity(),
        }
    }

    /// Discover points at infinity (and near-infinite distant points) and
    /// append them as new points/tracks, returning a new reconstruction.
    ///
    /// Un-projects every keypoint in every image to a world-space direction,
    /// clusters co-directional keypoints within ``eps_deg`` on the unit sphere,
    /// confirms each cluster with mutual SIFT descriptor matching (Lowe ratio +
    /// one feature per image), and emits each surviving track that spans at
    /// least ``min_views`` images. A track whose parallax signal falls below
    /// ``noise_floor_px`` becomes a ``w = 0`` point at infinity; otherwise it is
    /// triangulated to a finite distant point. Candidate tracks that duplicate
    /// an existing point are skipped.
    ///
    /// Args:
    ///     eps_deg: Angular clustering radius in degrees. Tighter values demand
    ///         more nearly parallel rays (more "infinite").
    ///     desc_thresh: Maximum L2 SIFT descriptor distance for a match.
    ///     ratio: Lowe ratio test against the second-best in-image match.
    ///     min_views: Minimum distinct images a track must span.
    ///     max_features: Per-image cap on the largest keypoints to read; reads
    ///         all when None.
    ///     noise_floor_px: SIFT keypoint localisation noise floor in pixels.
    #[pyo3(signature = (eps_deg, desc_thresh=200.0, ratio=0.8, min_views=2, max_features=None, noise_floor_px=1.0))]
    fn find_points_at_infinity(
        &self,
        eps_deg: f64,
        desc_thresh: f64,
        ratio: f64,
        min_views: usize,
        max_features: Option<usize>,
        noise_floor_px: f64,
    ) -> PyResult<Self> {
        let inner = self
            .inner
            .find_points_at_infinity(
                eps_deg,
                desc_thresh,
                ratio,
                min_views,
                max_features,
                noise_floor_px,
            )
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Source metadata as a Python dict.
    #[getter]
    fn source_metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.inner.metadata)
    }

    /// Rig/frame data as a Python dict, or None if no rig data.
    #[getter]
    fn rig_frame_data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use crate::helpers::rig_frame_data_to_py;
        match &self.inner.rig_frame_data {
            Some(rf) => rig_frame_data_to_py(py, rf),
            None => Ok(py.None()),
        }
    }

    /// World space unit string, or None.
    #[getter]
    fn world_space_unit(&self) -> Option<String> {
        self.inner.metadata.world_space_unit.clone()
    }

    /// Create a copy of this reconstruction with some fields replaced.
    ///
    /// Accepts optional keyword arguments for the fields to replace.
    /// Unspecified fields are copied from the original.
    ///
    /// ``positions`` accepts either ``(N, 3)`` Euclidean coordinates (``w = 1``)
    /// or ``(N, 4)`` homogeneous coordinates (``w = 0`` marks a point at
    /// infinity).
    ///
    /// Supported fields: ``positions``, ``colors``, ``errors``,
    /// ``quaternions_wxyz``, ``translations``, ``track_image_indexes``,
    /// ``track_feature_indexes``, ``track_point_ids``, ``observation_counts``,
    /// ``normals``, ``patches`` (a ``PatchCloud`` or ``None``),
    /// ``patch_bitmaps`` (an ``(N, R, R, 4)`` uint8 array or ``None``; requires
    /// the patch frame, so pass ``patches`` too unless one is already attached),
    /// ``image_names``, ``camera_indexes``, ``cameras``,
    /// ``feature_tool_hashes``, ``sift_content_hashes``, ``thumbnails_y_x_rgb``,
    /// ``rig_frame_data``, ``world_space_unit``, ``feature_source``,
    /// ``keypoints_xy`` (a ``(K, 2)`` float32 array or ``None``),
    /// ``image_file_hashes`` (a ``list[bytes]`` or ``None``). To produce an
    /// ``embedded_patches`` reconstruction, set ``feature_source``,
    /// ``keypoints_xy``, and ``image_file_hashes`` together.
    ///
    /// When the track arrays are replaced, ``observation_counts`` is recomputed
    /// from the new tracks (which must stay grouped by point), so an
    /// ``observation_counts`` value passed in the same call is ignored.
    #[pyo3(signature = (**kwargs))]
    fn clone_with_changes(
        &self,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: crate::recon_clone::clone_with_changes(&self.inner, py, kwargs)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SfmrReconstruction(images={}, points={}, observations={})",
            self.inner.image_count(),
            self.inner.point_count(),
            self.inner.observation_count(),
        )
    }
}
