// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Python form of a reconstruction's physical distance measurements: a
//! dict of the five `measurements/` columns, named as the `.sfmr` file names
//! them. `read_sfmr`, `write_sfmr`, `SfmrReconstruction.from_data`,
//! `SfmrReconstruction.measurements` and `clone_with_changes(measurements=...)`
//! all take this one form.

use numpy::IntoPyArray;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_sfmr_format::MeasurementsData;

use crate::helpers::{dtype_name, get_item, get_optional_item};

/// The keys of the five columns, as the `.sfmr` file names them.
pub(crate) const COLUMN_KEYS: [&str; 5] = [
    "distances",
    "units",
    "image_indexes",
    "endpoints_xy",
    "labels",
];

/// Extract one required numpy array from a measurements dict, copied into
/// standard (C) layout so Fortran-ordered and strided inputs work unchanged.
fn extract_array<'py, T, D>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    expected: &str,
) -> PyResult<ndarray::Array<T, D>>
where
    T: numpy::Element + Clone,
    D: ndarray::Dimension,
{
    let value = get_item(dict, key)?;
    let array = value
        .extract::<numpy::PyReadonlyArray<'py, T, D>>()
        .map_err(|_| {
            let actual_type = value
                .get_type()
                .qualname()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            let actual_dtype = dtype_name(&value).unwrap_or_else(|_| "unknown".to_string());
            PyTypeError::new_err(format!(
                "measurements['{key}'] must be {expected}, got {actual_type} with dtype {actual_dtype}"
            ))
        })?;
    Ok(array.as_array().as_standard_layout().into_owned())
}

/// Convert `MeasurementsData` into a Python dict of fresh numpy arrays and
/// `labels` (`list[str]`).
pub(crate) fn measurements_to_py<'py>(
    py: Python<'py>,
    measurements: &MeasurementsData,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("distances", measurements.distances.clone().into_pyarray(py))?;
    d.set_item("units", measurements.units.clone().into_pyarray(py))?;
    d.set_item(
        "image_indexes",
        measurements.image_indexes.clone().into_pyarray(py),
    )?;
    d.set_item(
        "endpoints_xy",
        measurements.endpoints_xy.clone().into_pyarray(py),
    )?;
    d.set_item("labels", &measurements.labels)?;
    Ok(d)
}

/// Convert a Python measurements dict (or `None`) into `Option<MeasurementsData>`.
///
/// Raises `TypeError` for non-dict input, wrong array dtype/rank, or non-list /
/// non-str `labels`; `KeyError` for a missing required column; and `ValueError`
/// for an unexpected extra key. Value and shape constraints (`N >= 1`,
/// consistent column lengths, positive finite distances, valid unit codes,
/// in-bounds image indexes and endpoint coordinates) are enforced by
/// `sfmtool_sfmr_format::validate_measurements`.
pub(crate) fn measurements_from_py(value: &Bound<'_, PyAny>) -> PyResult<Option<MeasurementsData>> {
    if value.is_none() {
        return Ok(None);
    }
    let d = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("measurements must be a dict or None"))?;

    let distances = extract_array::<f32, ndarray::Ix1>(d, "distances", "a 1D float32 array")?;
    let units = extract_array::<u8, ndarray::Ix1>(d, "units", "a 1D uint8 array")?;
    let image_indexes =
        extract_array::<u32, ndarray::Ix2>(d, "image_indexes", "a 2D uint32 array")?;
    let endpoints_xy = extract_array::<f32, ndarray::Ix3>(d, "endpoints_xy", "a 3D float32 array")?;

    let labels_obj = get_item(d, "labels")?;
    let labels_list = labels_obj.cast::<PyList>().map_err(|_| {
        let actual_type = labels_obj
            .get_type()
            .qualname()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        PyTypeError::new_err(format!(
            "measurements['labels'] must be a list of str, got {actual_type}"
        ))
    })?;
    let mut labels = Vec::with_capacity(labels_list.len());
    for (i, item) in labels_list.iter().enumerate() {
        let s: String = item.extract().map_err(|_| {
            let actual_type = item
                .get_type()
                .qualname()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            PyTypeError::new_err(format!(
                "measurements['labels'][{i}] must be str, got {actual_type}"
            ))
        })?;
        labels.push(s);
    }

    for key in d.keys() {
        let valid = key
            .extract::<String>()
            .ok()
            .is_some_and(|k| COLUMN_KEYS.contains(&k.as_str()));
        if !valid {
            return Err(PyValueError::new_err(format!(
                "measurements has unexpected key {key:?}; expected keys are {COLUMN_KEYS:?}"
            )));
        }
    }

    Ok(Some(MeasurementsData {
        distances,
        units,
        image_indexes,
        endpoints_xy,
        labels,
    }))
}

/// Extract optional `"measurements"` from an `.sfmr` data dict.
///
/// Returns `Ok(None)` when `"measurements"` is absent or `None`.
pub(crate) fn extract_measurements(data: &Bound<'_, PyDict>) -> PyResult<Option<MeasurementsData>> {
    match get_optional_item(data, "measurements")? {
        Some(value) => measurements_from_py(&value),
        None => Ok(None),
    }
}
