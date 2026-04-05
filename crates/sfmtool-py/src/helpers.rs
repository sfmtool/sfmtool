// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared helper functions for Python ↔ Rust conversions.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use sfmr_format::{FramesMetadata, RigFrameData, RigsMetadata, SfmrCamera};

use crate::PyCameraIntrinsics;

/// Serialize a serde-compatible value to a Python object via JSON round-trip.
pub(crate) fn serde_to_py<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let json_str = serde_json::to_string(value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let json_mod = py.import("json")?;
    Ok(json_mod.call_method1("loads", (json_str,))?.into())
}

/// Deserialize a Python dict/list to a serde-compatible type via JSON round-trip.
pub(crate) fn py_to_serde<T: serde::de::DeserializeOwned>(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<T> {
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Get a required key from a Python dict, raising KeyError if missing.
pub(crate) fn get_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))
}

/// Convert `Vec<[u8; 16]>` to a Python `list[bytes]` of 16-byte elements.
pub(crate) fn u128_bytes_to_py<'py>(
    py: Python<'py>,
    values: &[[u8; 16]],
) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, values.iter().map(|v| PyBytes::new(py, v)))
}

/// Convert a Python `list[bytes]` of 16-byte elements to `Vec<[u8; 16]>`.
pub(crate) fn py_to_u128_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<[u8; 16]>> {
    let list: Vec<Vec<u8>> = obj.extract()?;
    list.iter()
        .enumerate()
        .map(|(i, v)| {
            if v.len() != 16 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Element at index {i} has length {}, expected 16",
                    v.len()
                )));
            }
            let mut arr = [0u8; 16];
            arr.copy_from_slice(v);
            Ok(arr)
        })
        .collect()
}

/// Extract a cameras list from Python as `Vec<SfmrCamera>`.
///
/// Accepts `list[CameraIntrinsics]` (typed Rust objects).
pub(crate) fn extract_cameras_as_sfmr(obj: &Bound<'_, PyAny>) -> PyResult<Vec<SfmrCamera>> {
    let list = obj
        .downcast::<PyList>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("cameras must be a list"))?;

    let mut cameras = Vec::with_capacity(list.len());
    for item in list.iter() {
        let py_cam: PyCameraIntrinsics = item.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "cameras list must contain CameraIntrinsics objects",
            )
        })?;
        cameras.push(SfmrCamera::from(&py_cam.inner));
    }
    Ok(cameras)
}

/// Convert optional `RigFrameData` to a Python dict, or `None`.
pub(crate) fn rig_frame_data_to_py(py: Python<'_>, rf: &RigFrameData) -> PyResult<Py<PyAny>> {
    use numpy::IntoPyArray;

    let dict = PyDict::new(py);
    dict.set_item("rigs_metadata", serde_to_py(py, &rf.rigs_metadata)?)?;
    dict.set_item(
        "sensor_camera_indexes",
        rf.sensor_camera_indexes.clone().into_pyarray(py),
    )?;
    dict.set_item(
        "sensor_quaternions_wxyz",
        rf.sensor_quaternions_wxyz.clone().into_pyarray(py),
    )?;
    dict.set_item(
        "sensor_translations_xyz",
        rf.sensor_translations_xyz.clone().into_pyarray(py),
    )?;
    dict.set_item("frames_metadata", serde_to_py(py, &rf.frames_metadata)?)?;
    dict.set_item("rig_indexes", rf.rig_indexes.clone().into_pyarray(py))?;
    dict.set_item(
        "image_sensor_indexes",
        rf.image_sensor_indexes.clone().into_pyarray(py),
    )?;
    dict.set_item(
        "image_frame_indexes",
        rf.image_frame_indexes.clone().into_pyarray(py),
    )?;
    Ok(dict.into())
}

/// Extract optional rig/frame data from a Python dict.
///
/// Returns `None` if the dict does not contain `"rig_frame_data"` or it is `None`.
pub(crate) fn extract_rig_frame_data(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
) -> PyResult<Option<RigFrameData>> {
    let rf_obj = match data.get_item("rig_frame_data")? {
        Some(obj) if !obj.is_none() => obj,
        _ => return Ok(None),
    };
    let rf_dict = rf_obj
        .downcast::<PyDict>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("rig_frame_data must be a dict"))?;

    let rigs_metadata: RigsMetadata = py_to_serde(py, &get_item(rf_dict, "rigs_metadata")?)?;

    let sensor_camera_indexes: PyReadonlyArray1<u32> =
        get_item(rf_dict, "sensor_camera_indexes")?.extract()?;
    let sensor_quaternions_wxyz: numpy::PyReadonlyArray2<f64> =
        get_item(rf_dict, "sensor_quaternions_wxyz")?.extract()?;
    let sensor_translations_xyz: numpy::PyReadonlyArray2<f64> =
        get_item(rf_dict, "sensor_translations_xyz")?.extract()?;

    let frames_metadata: FramesMetadata = py_to_serde(py, &get_item(rf_dict, "frames_metadata")?)?;

    let rig_indexes: PyReadonlyArray1<u32> = get_item(rf_dict, "rig_indexes")?.extract()?;
    let image_sensor_indexes: PyReadonlyArray1<u32> =
        get_item(rf_dict, "image_sensor_indexes")?.extract()?;
    let image_frame_indexes: PyReadonlyArray1<u32> =
        get_item(rf_dict, "image_frame_indexes")?.extract()?;

    Ok(Some(RigFrameData {
        rigs_metadata,
        sensor_camera_indexes: sensor_camera_indexes.as_array().to_owned(),
        sensor_quaternions_wxyz: sensor_quaternions_wxyz.as_array().to_owned(),
        sensor_translations_xyz: sensor_translations_xyz.as_array().to_owned(),
        frames_metadata,
        rig_indexes: rig_indexes.as_array().to_owned(),
        image_sensor_indexes: image_sensor_indexes.as_array().to_owned(),
        image_frame_indexes: image_frame_indexes.as_array().to_owned(),
    }))
}

/// Extract an optional (3,3) float64 matrix from a dict, returning row-major [f64; 9].
pub(crate) fn extract_optional_3x3_matrix(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<[f64; 9]>> {
    if let Some(obj) = dict.get_item(key)? {
        let arr: numpy::PyReadonlyArray2<f64> = obj.extract()?;
        let view = arr.as_array();
        let mut out = [0.0f64; 9];
        for r in 0..3 {
            for c in 0..3 {
                out[r * 3 + c] = view[[r, c]];
            }
        }
        Ok(Some(out))
    } else {
        Ok(None)
    }
}
