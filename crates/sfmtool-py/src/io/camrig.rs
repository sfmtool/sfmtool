// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.camrig` file I/O and image-pattern matching.

use std::path::PathBuf;

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{py_to_serde, serde_to_py};

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

/// Read a complete `.camrig` file into a dict.
///
/// Returns a dict with keys `metadata`, `content_hash`, `cameras`,
/// `sensor_image_patterns`, `camera_indexes`, `quaternions_wxyz`, and
/// `translations_xyz`. Structural constraints are enforced; an invalid file
/// raises `ValueError`.
#[pyfunction]
pub fn read_camrig(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let data = camrig_format::read_camrig(&path).map_err(camrig_err_to_py)?;
    let dict = PyDict::new(py);
    dict.set_item("metadata", serde_to_py(py, &data.metadata)?)?;
    dict.set_item("content_hash", serde_to_py(py, &data.content_hash)?)?;
    dict.set_item("cameras", serde_to_py(py, &data.cameras)?)?;
    dict.set_item("sensor_image_patterns", data.sensor_image_patterns)?;
    dict.set_item("camera_indexes", data.camera_indexes)?;
    dict.set_item("quaternions_wxyz", data.quaternions_wxyz.into_pyarray(py))?;
    dict.set_item("translations_xyz", data.translations_xyz.into_pyarray(py))?;
    Ok(dict.into())
}

/// Map a `CamRigError` to an appropriate Python exception: I/O failures
/// become `IOError`, structural problems become `ValueError`.
fn camrig_err_to_py(e: camrig_format::CamRigError) -> PyErr {
    use camrig_format::CamRigError;
    match e {
        CamRigError::Io(_) | CamRigError::IoPath { .. } => {
            pyo3::exceptions::PyIOError::new_err(e.to_string())
        }
        _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
    }
}

/// Write a `.camrig` file from columnar rig data.
///
/// The arguments mirror the `.camrig` file layout (see
/// `specs/formats/camrig-file-format.md`). Structural constraints are
/// validated before writing; an invalid rig raises `ValueError`.
///
/// Args:
///     path: Output `.camrig` file path.
///     name: Human-readable rig name stored in the file.
///     rig_type: Rig-type hint, e.g. `"generic"`, `"stereo_pair"`.
///     cameras: List of camera dicts, each `{model, width, height, parameters}`.
///     sensor_image_patterns: Per-sensor image pattern strings. Either empty
///         (geometry-only rig) or one per sensor.
///     camera_indexes: Per-sensor index into `cameras`; its length is the
///         sensor count.
///     quaternions_wxyz: `(S, 4)` float64 array of `sensor_from_rig` WXYZ
///         quaternions.
///     translations_xyz: `(S, 3)` float64 array of `sensor_from_rig`
///         translations.
///     rig_attributes: Optional free-form `rig_type`-specific dict.
///     zstd_level: zstd compression level (default 3).
#[pyfunction]
#[pyo3(signature = (
    path,
    name,
    rig_type,
    cameras,
    sensor_image_patterns,
    camera_indexes,
    quaternions_wxyz,
    translations_xyz,
    rig_attributes=None,
    zstd_level=3,
))]
#[allow(clippy::too_many_arguments)]
pub fn write_camrig<'py>(
    py: Python<'py>,
    path: PathBuf,
    name: String,
    rig_type: String,
    cameras: &Bound<'py, PyAny>,
    sensor_image_patterns: Vec<String>,
    camera_indexes: Vec<u32>,
    quaternions_wxyz: numpy::PyReadonlyArray2<'py, f64>,
    translations_xyz: numpy::PyReadonlyArray2<'py, f64>,
    rig_attributes: Option<&Bound<'py, PyAny>>,
    zstd_level: i32,
) -> PyResult<()> {
    let cameras: Vec<camrig_format::CamRigCamera> = py_to_serde(py, cameras)?;
    let rig_attributes: serde_json::Value = match rig_attributes {
        Some(obj) => py_to_serde(py, obj)?,
        None => serde_json::Value::Object(serde_json::Map::new()),
    };
    let quaternions_wxyz = quaternions_wxyz.as_array().to_owned();
    let translations_xyz = translations_xyz.as_array().to_owned();

    let data = camrig_format::CamRigData {
        metadata: camrig_format::CamRigMetadata {
            version: camrig_format::CAMRIG_FORMAT_VERSION,
            name,
            sensor_count: camera_indexes.len() as u32,
            camera_count: cameras.len() as u32,
            rig_type,
            rig_attributes,
        },
        content_hash: camrig_format::CamRigContentHash::default(),
        cameras,
        sensor_image_patterns,
        camera_indexes,
        quaternions_wxyz,
        translations_xyz,
    };

    py.detach(|| camrig_format::write_camrig(&path, &data, zstd_level))
        .map_err(camrig_err_to_py)
}

/// Validate a `.camrig` image pattern's structure.
///
/// Raises `ValueError` (with the reason) when the pattern is not a valid
/// `.camrig` image pattern: empty, absolute, containing a `..` component, a
/// `**` that is not a whole path segment, or more than one frame field. The
/// `camrig-format` crate owns the rule, so workspace tooling and the format's
/// own `validate()` agree by construction.
#[pyfunction]
pub fn validate_camrig_pattern(pattern: &str) -> PyResult<()> {
    camrig_format::validate_pattern(pattern).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Convert a `.camrig` image pattern to a loose glob for filesystem
/// enumeration: each frame field (`%d` / `%0Nd`) becomes `*`, `%%` becomes a
/// literal `%`, and the glob wildcards `*` / `**` pass through unchanged.
///
/// The glob deliberately over-matches; pair it with `camrig_pattern_matches`
/// to filter the hits against the exact pattern grammar.
#[pyfunction]
pub fn camrig_pattern_to_glob(pattern: &str) -> String {
    camrig_format::pattern_to_glob(pattern)
}

/// Whether `relative_path` (a forward-slash relative path) matches a
/// `.camrig` image `pattern` under the exact pattern grammar.
///
/// Set `case_insensitive` to mirror the filesystem that produced the glob
/// hits — case-insensitive filesystems (Windows) glob case-insensitively, so
/// the strict check must too. The `camrig-format` crate owns the grammar.
#[pyfunction]
pub fn camrig_pattern_matches(pattern: &str, relative_path: &str, case_insensitive: bool) -> bool {
    camrig_format::pattern_matches(pattern, relative_path, case_insensitive)
}

/// The frame index a `.camrig` image `pattern`'s frame field captures from
/// `relative_path`.
///
/// Returns `None` when the pattern has no frame field, when `relative_path`
/// does not match the pattern, or when the captured digits overflow a 64-bit
/// integer. `case_insensitive` mirrors `camrig_pattern_matches`.
#[pyfunction]
pub fn camrig_pattern_frame_index(
    pattern: &str,
    relative_path: &str,
    case_insensitive: bool,
) -> Option<u64> {
    camrig_format::pattern_frame_index(pattern, relative_path, case_insensitive)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_camrig_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(read_camrig, m)?)?;
    m.add_function(wrap_pyfunction!(verify_camrig, m)?)?;
    m.add_function(wrap_pyfunction!(write_camrig, m)?)?;
    m.add_function(wrap_pyfunction!(validate_camrig_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(camrig_pattern_to_glob, m)?)?;
    m.add_function(wrap_pyfunction!(camrig_pattern_matches, m)?)?;
    m.add_function(wrap_pyfunction!(camrig_pattern_frame_index, m)?)?;
    Ok(())
}
