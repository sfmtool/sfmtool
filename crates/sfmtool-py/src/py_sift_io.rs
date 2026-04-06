// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.sift` file I/O.

use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use sift_format::{self, FeatureToolMetadata, SiftContentHash, SiftData, SiftMetadata};

use crate::helpers::{get_item, py_to_serde, serde_to_py};

/// Convert SiftData to a Python dict.
fn sift_data_to_py(py: Python<'_>, data: SiftData) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    dict.set_item(
        "feature_tool_metadata",
        serde_to_py(py, &data.feature_tool_metadata)?,
    )?;
    dict.set_item("metadata", serde_to_py(py, &data.metadata)?)?;
    dict.set_item("content_hash", serde_to_py(py, &data.content_hash)?)?;
    dict.set_item("positions_xy", data.positions_xy.into_pyarray(py))?;
    dict.set_item("affine_shapes", data.affine_shapes.into_pyarray(py))?;
    dict.set_item("descriptors", data.descriptors.into_pyarray(py))?;
    dict.set_item("thumbnail_y_x_rgb", data.thumbnail_y_x_rgb.into_pyarray(py))?;

    Ok(dict.into())
}

/// Read a complete .sift file, returning a dict with numpy arrays and metadata.
///
/// Returns a dict with keys:
///   feature_tool_metadata, metadata, content_hash (dicts),
///   positions_xy (N,2 float32), affine_shapes (N,2,2 float32),
///   descriptors (N,128 uint8), thumbnail_y_x_rgb (128,128,3 uint8).
#[pyfunction]
pub fn read_sift(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let data = sift_format::read_sift(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    sift_data_to_py(py, data)
}

/// Read only metadata from a .sift file (fast, no binary data).
///
/// Returns a dict with keys: feature_tool_metadata, metadata, content_hash.
#[pyfunction]
pub fn read_sift_metadata(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let (tool_meta, meta, hash) = sift_format::read_sift_metadata(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("feature_tool_metadata", serde_to_py(py, &tool_meta)?)?;
    dict.set_item("metadata", serde_to_py(py, &meta)?)?;
    dict.set_item("content_hash", serde_to_py(py, &hash)?)?;
    Ok(dict.into())
}

/// Read the first `count` features from a .sift file.
///
/// If `count` exceeds the feature count, returns all features.
#[pyfunction]
pub fn read_sift_partial(py: Python<'_>, path: PathBuf, count: usize) -> PyResult<Py<PyAny>> {
    let data = sift_format::read_sift_partial(&path, count)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    sift_data_to_py(py, data)
}

/// Write a .sift file from a dict of numpy arrays and metadata.
///
/// The dict should have the same keys as returned by `read_sift`.
/// The `content_hash` key is ignored (recomputed on write).
#[pyfunction]
#[pyo3(signature = (path, data, zstd_level=3))]
pub fn write_sift(
    py: Python<'_>,
    path: PathBuf,
    data: &Bound<'_, PyDict>,
    zstd_level: i32,
) -> PyResult<()> {
    let feature_tool_metadata: FeatureToolMetadata =
        py_to_serde(py, &get_item(data, "feature_tool_metadata")?)?;
    let metadata: SiftMetadata = py_to_serde(py, &get_item(data, "metadata")?)?;

    let positions_xy: PyReadonlyArray2<f32> = get_item(data, "positions_xy")?.extract()?;
    let affine_shapes: PyReadonlyArray3<f32> = get_item(data, "affine_shapes")?.extract()?;
    let descriptors: PyReadonlyArray2<u8> = get_item(data, "descriptors")?.extract()?;
    let thumbnail_y_x_rgb: PyReadonlyArray3<u8> = get_item(data, "thumbnail_y_x_rgb")?.extract()?;

    let sift_data = SiftData {
        feature_tool_metadata,
        metadata,
        content_hash: SiftContentHash::default(),
        positions_xy: positions_xy.as_array().to_owned(),
        affine_shapes: affine_shapes.as_array().to_owned(),
        descriptors: descriptors.as_array().to_owned(),
        thumbnail_y_x_rgb: thumbnail_y_x_rgb.as_array().to_owned(),
    };

    sift_format::write_sift(&path, &sift_data, zstd_level)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Verify integrity of a .sift file.
///
/// Returns a tuple (is_valid, error_messages).
#[pyfunction]
pub fn verify_sift(path: PathBuf) -> PyResult<(bool, Vec<String>)> {
    sift_format::verify_sift(&path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
