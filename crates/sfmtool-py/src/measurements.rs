// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Python form of a reconstruction's measurements: a dict with one key per
//! kind present, each holding that kind's columns named as the `.sfmr` file
//! names them, or `None` when there are none. The one kind is `"distance"`, a
//! dict of the five `measurements/distance/` columns; a kind that is absent
//! has no key. `read_sfmr`, `write_sfmr`, `SfmrReconstruction.from_data`,
//! `SfmrReconstruction.measurements` and `clone_with_changes(measurements=...)`
//! all take this one form.

use numpy::IntoPyArray;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_sfmr_format::{DistanceMeasurementsData, MeasurementsData};

use crate::helpers::{dtype_name, get_item, get_optional_item};

/// The keys of the kinds of measurement, as the `.sfmr` file names their
/// subdirectories of `measurements/`.
pub(crate) const KIND_KEYS: [&str; 1] = ["distance"];

/// The keys of the distance kind's five columns, as the `.sfmr` file names
/// them.
pub(crate) const DISTANCE_COLUMN_KEYS: [&str; 5] = [
    "distances",
    "units",
    "image_indexes",
    "endpoints_xy",
    "labels",
];

/// The qualified name of `value`'s type, for an error message.
fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .qualname()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Extract one required numpy array from a distance measurements dict, copied
/// into standard (C) layout so Fortran-ordered and strided inputs work
/// unchanged.
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
            let actual_type = type_name(&value);
            let actual_dtype = dtype_name(&value).unwrap_or_else(|_| "unknown".to_string());
            PyTypeError::new_err(format!(
                "measurements['distance']['{key}'] must be {expected}, got {actual_type} with \
                 dtype {actual_dtype}"
            ))
        })?;
    Ok(array.as_array().as_standard_layout().into_owned())
}

/// Convert `DistanceMeasurementsData` into a Python dict of fresh numpy arrays
/// and `labels` (`list[str]`).
fn distance_measurements_to_py<'py>(
    py: Python<'py>,
    measurements: &DistanceMeasurementsData,
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

/// Convert `MeasurementsData` into its Python form: `None` when no kind is
/// present, else a dict with a key for each kind that is.
pub(crate) fn measurements_to_py<'py>(
    py: Python<'py>,
    measurements: &MeasurementsData,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if measurements.is_empty() {
        return Ok(None);
    }
    let d = PyDict::new(py);
    if let Some(distance) = &measurements.distance {
        d.set_item("distance", distance_measurements_to_py(py, distance)?)?;
    }
    Ok(Some(d))
}

/// Convert the Python form of the distance kind into `DistanceMeasurementsData`.
fn distance_measurements_from_py(value: &Bound<'_, PyAny>) -> PyResult<DistanceMeasurementsData> {
    let d = value.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(format!(
            "measurements['distance'] must be a dict, got {}",
            type_name(value)
        ))
    })?;

    let distances = extract_array::<f32, ndarray::Ix1>(d, "distances", "a 1D float32 array")?;
    let units = extract_array::<u8, ndarray::Ix1>(d, "units", "a 1D uint8 array")?;
    let image_indexes =
        extract_array::<u32, ndarray::Ix2>(d, "image_indexes", "a 2D uint32 array")?;
    let endpoints_xy = extract_array::<f32, ndarray::Ix3>(d, "endpoints_xy", "a 3D float32 array")?;

    let labels_obj = get_item(d, "labels")?;
    let labels_list = labels_obj.cast::<PyList>().map_err(|_| {
        PyTypeError::new_err(format!(
            "measurements['distance']['labels'] must be a list of str, got {}",
            type_name(&labels_obj)
        ))
    })?;
    let mut labels = Vec::with_capacity(labels_list.len());
    for (i, item) in labels_list.iter().enumerate() {
        let s: String = item.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "measurements['distance']['labels'][{i}] must be str, got {}",
                type_name(&item)
            ))
        })?;
        labels.push(s);
    }

    for key in d.keys() {
        let valid = key
            .extract::<String>()
            .ok()
            .is_some_and(|k| DISTANCE_COLUMN_KEYS.contains(&k.as_str()));
        if !valid {
            return Err(PyValueError::new_err(format!(
                "measurements['distance'] has unexpected key {key:?}; expected keys are \
                 {DISTANCE_COLUMN_KEYS:?}"
            )));
        }
    }

    Ok(DistanceMeasurementsData {
        distances,
        units,
        image_indexes,
        endpoints_xy,
        labels,
    })
}

/// Convert the Python form of a reconstruction's measurements (a dict of
/// kinds, or `None`) into `MeasurementsData`.
///
/// Raises `TypeError` for a non-dict value or kind, a wrong array dtype/rank,
/// or non-list / non-str `labels`; `KeyError` for a missing required column;
/// and `ValueError` for an unknown kind or column key, or for a dict naming no
/// kind (`None` is the form for no measurements). Value and shape constraints
/// (`N >= 1`, consistent column lengths, positive finite distances or NaN for
/// none, valid unit codes, in-bounds image indexes and endpoint coordinates or
/// an absent endpoint's `0xFFFFFFFF` and NaN coordinates, labels that are not
/// empty) are enforced by `sfmtool_sfmr_format::validate_measurements`, so the
/// arrays carry the file's sentinels as they are.
pub(crate) fn measurements_from_py(value: &Bound<'_, PyAny>) -> PyResult<MeasurementsData> {
    if value.is_none() {
        return Ok(MeasurementsData::default());
    }
    let d = value.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(format!(
            "measurements must be a dict of kinds or None, got {}",
            type_name(value)
        ))
    })?;
    for key in d.keys() {
        let valid = key
            .extract::<String>()
            .ok()
            .is_some_and(|k| KIND_KEYS.contains(&k.as_str()));
        if !valid {
            return Err(PyValueError::new_err(format!(
                "measurements has unexpected kind {key:?}; the kinds are {KIND_KEYS:?}"
            )));
        }
    }
    let distance = d
        .get_item("distance")?
        .map(|v| distance_measurements_from_py(&v))
        .transpose()?;
    let measurements = MeasurementsData { distance };
    if measurements.is_empty() {
        return Err(PyValueError::new_err(
            "measurements names no kind; pass None when there are no measurements",
        ));
    }
    Ok(measurements)
}

/// Extract optional `"measurements"` from an `.sfmr` data dict.
///
/// Returns no measurements when `"measurements"` is absent or `None`.
pub(crate) fn extract_measurements(data: &Bound<'_, PyDict>) -> PyResult<MeasurementsData> {
    match get_optional_item(data, "measurements")? {
        Some(value) => measurements_from_py(&value),
        None => Ok(MeasurementsData::default()),
    }
}
