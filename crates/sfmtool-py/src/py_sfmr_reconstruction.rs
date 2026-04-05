// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core SfmrReconstruction type.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::{Path, PathBuf};

use sfmtool_core::SfmrReconstruction;

use crate::helpers::{serde_to_py, u128_bytes_to_py};
use crate::py_sfmr_io::parse_sfmr_data_from_dict;
use crate::PyCameraIntrinsics;

/// A loaded SfM reconstruction from a `.sfmr` file.
///
/// Wraps the Rust `SfmrReconstruction` and exposes scalar properties,
/// metadata, cameras, and image names to Python.
#[pyclass(name = "SfmrReconstruction")]
pub struct PySfmrReconstruction {
    pub(crate) inner: SfmrReconstruction,
}

#[pymethods]
impl PySfmrReconstruction {
    /// Load a reconstruction from a `.sfmr` file path.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = SfmrReconstruction::load(Path::new(path))
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
    fn from_data(py: Python<'_>, workspace_dir: &str, data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut sfmr_data = parse_sfmr_data_from_dict(py, data, false)?;
        sfmr_data.workspace_dir = Some(PathBuf::from(workspace_dir));
        let mut inner = SfmrReconstruction::from_sfmr_data(sfmr_data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        inner.workspace_dir = PathBuf::from(workspace_dir);
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
    #[pyo3(signature = (path, operation=None, tool_name=None, tool_options=None))]
    fn save(
        &mut self,
        py: Python<'_>,
        path: &str,
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
            meta.points3d_count = self.inner.points.len() as u32;
            meta.observation_count = self.inner.tracks.len() as u32;
            meta.camera_count = self.inner.cameras.len() as u32;
            meta.timestamp = chrono::Local::now().to_rfc3339();

            // Update workspace paths relative to output file
            let output_path =
                std::path::absolute(Path::new(path)).unwrap_or_else(|_| PathBuf::from(path));
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
            .save(Path::new(path))
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

    /// Feature tool hashes as `list[bytes]` (16 bytes each).
    #[getter]
    fn feature_tool_hashes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let hashes: Vec<[u8; 16]> = self
            .inner
            .images
            .iter()
            .map(|im| im.feature_tool_hash)
            .collect();
        u128_bytes_to_py(py, &hashes)
    }

    /// SIFT content hashes as `list[bytes]` (16 bytes each).
    #[getter]
    fn sift_content_hashes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let hashes: Vec<[u8; 16]> = self
            .inner
            .images
            .iter()
            .map(|im| im.sift_content_hash)
            .collect();
        u128_bytes_to_py(py, &hashes)
    }

    // ── Point data array getters ─────────────────────────────────────

    /// 3D point positions, shape `(M, 3)`.
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

    /// Estimated surface normals for 3D points, shape `(M, 3)`.
    #[getter]
    fn estimated_normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let vec2: Vec<Vec<f32>> = self
            .inner
            .points
            .iter()
            .map(|pt| {
                vec![
                    pt.estimated_normal.x,
                    pt.estimated_normal.y,
                    pt.estimated_normal.z,
                ]
            })
            .collect();
        PyArray2::from_vec2(py, &vec2).unwrap()
    }

    // ── Track data array getters ─────────────────────────────────────

    /// Image indexes for track observations, shape `(K,)`.
    #[getter]
    fn track_image_indexes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let vec: Vec<u32> = self.inner.tracks.iter().map(|t| t.image_index).collect();
        PyArray1::from_vec(py, vec)
    }

    /// Feature indexes for track observations, shape `(K,)`.
    #[getter]
    fn track_feature_indexes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let vec: Vec<u32> = self.inner.tracks.iter().map(|t| t.feature_index).collect();
        PyArray1::from_vec(py, vec)
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
    #[getter]
    fn thumbnails_y_x_rgb<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.inner
            .thumbnails_y_x_rgb
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
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
        self.inner
            .metadata
            .tool_options
            .get("world_space_unit")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Create a copy of this reconstruction with some fields replaced.
    ///
    /// Accepts optional keyword arguments for the fields to replace.
    /// Unspecified fields are copied from the original.
    ///
    /// Supported fields: ``positions``, ``colors``, ``errors``,
    /// ``quaternions_wxyz``, ``translations``, ``track_image_indexes``,
    /// ``track_feature_indexes``, ``track_point_ids``, ``observation_counts``,
    /// ``estimated_normals``, ``image_names``, ``camera_indexes``, ``cameras``,
    /// ``feature_tool_hashes``, ``sift_content_hashes``, ``thumbnails_y_x_rgb``,
    /// ``rig_frame_data``, ``world_space_unit``.
    #[pyo3(signature = (**kwargs))]
    fn clone_with_changes(
        &self,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        use nalgebra::{UnitQuaternion, Vector3};
        use numpy::{PyReadonlyArray1, PyReadonlyArray2};

        use crate::helpers::{extract_cameras_as_sfmr, extract_rig_frame_data, py_to_u128_bytes};

        /// Helper to extract a typed numpy array with a clear error message.
        macro_rules! extract_array1 {
            ($value:expr, $param:expr, $ty:ty) => {
                $value.extract::<PyReadonlyArray1<$ty>>().map_err(|_| {
                    let actual_type = $value
                        .get_type()
                        .qualname()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());
                    let actual_dtype = $value
                        .getattr("dtype")
                        .ok()
                        .and_then(|d| d.str().ok())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "clone_with_changes(): '{}' must be a contiguous ndarray with dtype {}, \
                         got {} with dtype {}",
                        $param,
                        stringify!($ty),
                        actual_type,
                        actual_dtype
                    ))
                })
            };
        }
        macro_rules! extract_array2 {
            ($value:expr, $param:expr, $ty:ty) => {
                $value.extract::<PyReadonlyArray2<$ty>>().map_err(|_| {
                    let actual_type = $value.get_type().qualname()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());
                    let actual_dtype = $value
                        .getattr("dtype")
                        .ok()
                        .and_then(|d| d.str().ok())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "clone_with_changes(): '{}' must be a 2D contiguous ndarray with dtype {}, \
                         got {} with dtype {}",
                        $param,
                        stringify!($ty),
                        actual_type,
                        actual_dtype
                    ))
                })
            };
        }

        let mut recon = self.inner.clone();

        let Some(kw) = kwargs else {
            return Ok(Self { inner: recon });
        };

        // Track whether we need to rebuild images from scratch
        let mut new_image_names: Option<Vec<String>> = None;
        let mut new_camera_indexes: Option<Vec<u32>> = None;
        let mut new_feature_tool_hashes: Option<Vec<[u8; 16]>> = None;
        let mut new_sift_content_hashes: Option<Vec<[u8; 16]>> = None;

        for (key, value) in kw.iter() {
            let key_str: String = key.extract()?;
            match key_str.as_str() {
                "positions" => {
                    let arr = extract_array2!(value, "positions", f64)?;
                    let s = arr.as_slice().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'positions' must be C-contiguous: {e}"
                        ))
                    })?;
                    if arr.shape()[1] != 3 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'positions' must have shape (N, 3), \
                             got shape ({}, {})",
                            arr.shape()[0],
                            arr.shape()[1]
                        )));
                    }
                    // Allow changing number of points
                    let n = arr.shape()[0];
                    recon.points.resize(
                        n,
                        sfmtool_core::Point3D {
                            position: nalgebra::Point3::origin(),
                            color: [0, 0, 0],
                            error: 0.0,
                            estimated_normal: Vector3::zeros(),
                        },
                    );
                    for (i, pt) in recon.points.iter_mut().enumerate() {
                        let off = i * 3;
                        pt.position = nalgebra::Point3::new(s[off], s[off + 1], s[off + 2]);
                    }
                }
                "colors" => {
                    let arr = extract_array2!(value, "colors", u8)?;
                    let s = arr.as_slice().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'colors' must be C-contiguous: {e}"
                        ))
                    })?;
                    if arr.shape()[1] != 3 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'colors' must have shape (N, 3), \
                             got shape ({}, {})",
                            arr.shape()[0],
                            arr.shape()[1]
                        )));
                    }
                    if arr.shape()[0] != recon.points.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'colors' length ({}) must match point count ({}). \
                             Hint: pass 'positions' first if changing point count.",
                            arr.shape()[0],
                            recon.points.len()
                        )));
                    }
                    for (i, pt) in recon.points.iter_mut().enumerate() {
                        let off = i * 3;
                        pt.color = [s[off], s[off + 1], s[off + 2]];
                    }
                }
                "errors" => {
                    let arr = extract_array1!(value, "errors", f32)?;
                    let s = arr.as_slice().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'errors' must be C-contiguous: {e}"
                        ))
                    })?;
                    if s.len() != recon.points.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'errors' length ({}) must match point count ({}). \
                             Hint: pass 'positions' first if changing point count.",
                            s.len(),
                            recon.points.len()
                        )));
                    }
                    for (i, pt) in recon.points.iter_mut().enumerate() {
                        pt.error = s[i];
                    }
                }
                "estimated_normals" => {
                    let arr = extract_array2!(value, "estimated_normals", f32)?;
                    let s = arr.as_slice().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'estimated_normals' must be C-contiguous: {e}"
                        ))
                    })?;
                    if arr.shape()[1] != 3 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'estimated_normals' must have shape (N, 3), \
                             got shape ({}, {})",
                            arr.shape()[0],
                            arr.shape()[1]
                        )));
                    }
                    if arr.shape()[0] != recon.points.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'estimated_normals' length ({}) must match point count ({})",
                            arr.shape()[0],
                            recon.points.len()
                        )));
                    }
                    for (i, pt) in recon.points.iter_mut().enumerate() {
                        let off = i * 3;
                        pt.estimated_normal = Vector3::new(s[off], s[off + 1], s[off + 2]);
                    }
                }
                "quaternions_wxyz" => {
                    let arr = extract_array2!(value, "quaternions_wxyz", f64)?;
                    let s = arr.as_slice().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'quaternions_wxyz' must be C-contiguous: {e}"
                        ))
                    })?;
                    if arr.shape()[1] != 4 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'quaternions_wxyz' must have shape (N, 4), \
                             got shape ({}, {})",
                            arr.shape()[0],
                            arr.shape()[1]
                        )));
                    }
                    let n = arr.shape()[0];
                    // Resize images if needed (when combined with image_names)
                    while recon.images.len() < n {
                        recon.images.push(sfmtool_core::SfmrImage {
                            name: String::new(),
                            camera_index: 0,
                            quaternion_wxyz: UnitQuaternion::identity(),
                            translation_xyz: Vector3::zeros(),
                            feature_tool_hash: [0u8; 16],
                            sift_content_hash: [0u8; 16],
                        });
                    }
                    recon.images.truncate(n);
                    for (i, im) in recon.images.iter_mut().enumerate() {
                        let off = i * 4;
                        im.quaternion_wxyz = UnitQuaternion::new_normalize(
                            nalgebra::Quaternion::new(s[off], s[off + 1], s[off + 2], s[off + 3]),
                        );
                    }
                }
                "translations" => {
                    let arr = extract_array2!(value, "translations", f64)?;
                    let s = arr.as_slice().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'translations' must be C-contiguous: {e}"
                        ))
                    })?;
                    if arr.shape()[1] != 3 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'translations' must have shape (N, 3), \
                             got shape ({}, {})",
                            arr.shape()[0],
                            arr.shape()[1]
                        )));
                    }
                    if arr.shape()[0] != recon.images.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'translations' length ({}) must match image count ({}). \
                             Hint: pass 'quaternions_wxyz' or 'image_names' first to resize.",
                            arr.shape()[0],
                            recon.images.len()
                        )));
                    }
                    for (i, im) in recon.images.iter_mut().enumerate() {
                        let off = i * 3;
                        im.translation_xyz = Vector3::new(s[off], s[off + 1], s[off + 2]);
                    }
                }
                "track_image_indexes" | "track_feature_indexes" | "track_point_ids" => {
                    // These must all be set together to rebuild tracks
                    // Defer to after the loop
                }
                "observation_counts" => {
                    let arr = extract_array1!(value, "observation_counts", u32)?;
                    recon.observation_counts = arr
                        .as_slice()
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "clone_with_changes(): 'observation_counts' must be C-contiguous: {e}"
                            ))
                        })?
                        .to_vec();
                }
                "image_names" => {
                    new_image_names = Some(value.extract()?);
                }
                "camera_indexes" => {
                    let arr = extract_array1!(value, "camera_indexes", u32)?;
                    new_camera_indexes = Some(
                        arr.as_slice()
                            .map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "clone_with_changes(): 'camera_indexes' must be C-contiguous: {e}"
                                ))
                            })?
                            .to_vec(),
                    );
                }
                "cameras" => {
                    use sfmtool_core::CameraIntrinsics;
                    let sfmr_cameras = extract_cameras_as_sfmr(&value)?;
                    recon.cameras = sfmr_cameras
                        .iter()
                        .map(|sc| {
                            CameraIntrinsics::try_from(sc).map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "clone_with_changes(): failed to convert camera: {e}"
                                ))
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                }
                "feature_tool_hashes" => {
                    new_feature_tool_hashes = Some(py_to_u128_bytes(&value)?);
                }
                "sift_content_hashes" => {
                    new_sift_content_hashes = Some(py_to_u128_bytes(&value)?);
                }
                "thumbnails_y_x_rgb" => {
                    let arr: numpy::PyReadonlyArray4<u8> = value.extract().map_err(|_| {
                        let actual_type = value
                            .get_type()
                            .qualname()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|_| "unknown".to_string());
                        let actual_dtype = value
                            .getattr("dtype")
                            .ok()
                            .and_then(|d| d.str().ok())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "clone_with_changes(): 'thumbnails_y_x_rgb' must be a 4D contiguous \
                             ndarray with dtype uint8 and shape (N, 128, 128, 3), \
                             got {} with dtype {}",
                            actual_type, actual_dtype
                        ))
                    })?;
                    recon.thumbnails_y_x_rgb = arr.as_array().to_owned();
                }
                "rig_frame_data" => {
                    if value.is_none() {
                        recon.rig_frame_data = None;
                    } else {
                        // Wrap in a temporary dict for extract_rig_frame_data
                        let tmp = PyDict::new(py);
                        tmp.set_item("rig_frame_data", &value)?;
                        recon.rig_frame_data = extract_rig_frame_data(py, &tmp)?;
                    }
                }
                "world_space_unit" => {
                    if value.is_none() {
                        recon.metadata.tool_options.remove("world_space_unit");
                    } else {
                        let unit: String = value.extract()?;
                        recon.metadata.tool_options.insert(
                            "world_space_unit".to_string(),
                            serde_json::Value::String(unit),
                        );
                    }
                }
                other => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "clone_with_changes() got unexpected keyword argument: '{other}'"
                    )));
                }
            }
        }

        // Apply image-level field updates that may change the image count
        if let Some(names) = new_image_names {
            let n = names.len();
            // Resize images vec to match
            while recon.images.len() < n {
                recon.images.push(sfmtool_core::SfmrImage {
                    name: String::new(),
                    camera_index: 0,
                    quaternion_wxyz: UnitQuaternion::identity(),
                    translation_xyz: Vector3::zeros(),
                    feature_tool_hash: [0u8; 16],
                    sift_content_hash: [0u8; 16],
                });
            }
            recon.images.truncate(n);
            for (i, im) in recon.images.iter_mut().enumerate() {
                im.name.clone_from(&names[i]);
            }
        }
        if let Some(ref indexes) = new_camera_indexes {
            if indexes.len() != recon.images.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'camera_indexes' length ({}) must match image count ({})",
                    indexes.len(),
                    recon.images.len()
                )));
            }
            for (i, im) in recon.images.iter_mut().enumerate() {
                im.camera_index = indexes[i];
            }
        }
        if let Some(ref hashes) = new_feature_tool_hashes {
            if hashes.len() != recon.images.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'feature_tool_hashes' length ({}) must match image count ({})",
                    hashes.len(),
                    recon.images.len()
                )));
            }
            for (i, im) in recon.images.iter_mut().enumerate() {
                im.feature_tool_hash = hashes[i];
            }
        }
        if let Some(ref hashes) = new_sift_content_hashes {
            if hashes.len() != recon.images.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'sift_content_hashes' length ({}) must match image count ({})",
                    hashes.len(),
                    recon.images.len()
                )));
            }
            for (i, im) in recon.images.iter_mut().enumerate() {
                im.sift_content_hash = hashes[i];
            }
        }

        // Resize depth_histogram_counts to match the (possibly new) image count.
        // When the image count changes, histogram data becomes stale so we reset it.
        if recon.depth_histogram_counts.len() != recon.images.len() {
            let num_buckets = recon.depth_statistics.num_histogram_buckets as usize;
            recon.depth_histogram_counts = vec![vec![0u32; num_buckets]; recon.images.len()];
        }

        // Rebuild tracks if any track arrays were provided
        let has_tracks = kw.contains("track_image_indexes")?
            || kw.contains("track_feature_indexes")?
            || kw.contains("track_point_ids")?;

        if has_tracks {
            let img_idx = kw.get_item("track_image_indexes")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                     track_point_ids must all be provided together",
                )
            })?;
            let img_idx: PyReadonlyArray1<u32> =
                extract_array1!(img_idx, "track_image_indexes", u32)?;

            let feat_idx = kw.get_item("track_feature_indexes")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                     track_point_ids must all be provided together",
                )
            })?;
            let feat_idx: PyReadonlyArray1<u32> =
                extract_array1!(feat_idx, "track_feature_indexes", u32)?;

            let pt_idx = kw.get_item("track_point_ids")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                     track_point_ids must all be provided together",
                )
            })?;
            let pt_idx: PyReadonlyArray1<u32> = extract_array1!(pt_idx, "track_point_ids", u32)?;

            let img_s = img_idx.as_slice().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'track_image_indexes' must be C-contiguous: {e}"
                ))
            })?;
            let feat_s = feat_idx.as_slice().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'track_feature_indexes' must be C-contiguous: {e}"
                ))
            })?;
            let pt_s = pt_idx.as_slice().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'track_point_ids' must be C-contiguous: {e}"
                ))
            })?;

            if img_s.len() != feat_s.len() || img_s.len() != pt_s.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): track arrays must all have the same length, \
                     got track_image_indexes={}, track_feature_indexes={}, track_point_ids={}",
                    img_s.len(),
                    feat_s.len(),
                    pt_s.len()
                )));
            }

            recon.tracks = (0..img_s.len())
                .map(|i| sfmtool_core::TrackObservation {
                    image_index: img_s[i],
                    feature_index: feat_s[i],
                    point_index: pt_s[i],
                })
                .collect();
        }

        // Recompute derived fields
        recon.rebuild_derived_indexes();

        Ok(Self { inner: recon })
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
