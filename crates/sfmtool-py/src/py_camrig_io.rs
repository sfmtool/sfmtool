// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.camrig` file inspection.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::serde_to_py;

/// Read only the metadata from a `.camrig` file (fast, no binary data).
///
/// Returns a dict with keys `metadata` and `content_hash`.
#[pyfunction]
pub fn read_camrig_metadata(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let (metadata, content_hash) = camrig_format::read_camrig_metadata(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("metadata", serde_to_py(py, &metadata)?)?;
    dict.set_item("content_hash", serde_to_py(py, &content_hash)?)?;
    Ok(dict.into())
}

/// Verify integrity of a `.camrig` file.
///
/// Returns a tuple `(is_valid, error_messages)`.
#[pyfunction]
pub fn verify_camrig(path: PathBuf) -> PyResult<(bool, Vec<String>)> {
    camrig_format::verify_camrig(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
