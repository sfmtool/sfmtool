// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core SfmrReconstruction type.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
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
    fn save(&self, path: &str) -> PyResult<()> {
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

    fn __repr__(&self) -> String {
        format!(
            "SfmrReconstruction(images={}, points={}, observations={})",
            self.inner.image_count(),
            self.inner.point_count(),
            self.inner.observation_count(),
        )
    }
}