// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Python form of each observation's stored readings: a dict of the eight
//! `tracks/` columns, named as the `.sfmr` file names them, and an `"options"`
//! dict of what they were read with. `read_sfmr`, `write_sfmr`,
//! `SfmrReconstruction.observation_readings` and
//! `clone_with_changes(observation_readings=...)` all take this one form.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::reconstruction::{ObservationReadingOptions, ObservationReadings};
use sfmtool_sfmr_format::ObservationReadingColumns;

use crate::helpers::{py_to_serde, serde_to_py};

/// The keys of the eight columns, as the `.sfmr` file names them.
pub(crate) const COLUMN_KEYS: [&str; 8] = [
    "zncc_self_similarity_ellipse_axes",
    "zncc_self_similarity_ellipse_axes_is_at_least",
    "zncc_self_similarity_ellipse_major_angle",
    "zncc_self_similarity_cos_view_angle",
    "zncc_self_similarity_tilt_angle",
    "zncc_self_similarity_zoom",
    "plain_bitmap_zncc",
    "blur_matched_bitmap_zncc",
];

/// The options as a dict, under the keys `tracks/metadata.json` stores them
/// by: `resolution`, `sampler` (`"per_view"`, `"bilinear"`, `"bilinear_mip"`
/// or `"anisotropic"`), `score_window` (`"uniform"`, `"gaussian"` or
/// `"gaussian_disk"`), `score_window_sigma`, `max_radius`, `flat_floor`,
/// `noise`, `relative_tolerance` and `anisotropic_threshold` (`None` for one
/// sampler for every view).
pub(crate) fn options_to_py<'py>(
    py: Python<'py>,
    options: &ObservationReadingOptions,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(serde_to_py(py, options)?.into_bound(py))
}

/// The options from such a dict. Every key is required: readings are only
/// comparable under the same options, so none is assumed. A numpy scalar
/// value is taken as the Python number it holds.
pub(crate) fn options_from_py(value: &Bound<'_, PyAny>) -> PyResult<ObservationReadingOptions> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("observation_readings['options'] must be a dict"))?;
    let plain = PyDict::new(value.py());
    for (k, v) in dict.iter() {
        let v = if v.hasattr("item")? && !v.is_instance_of::<pyo3::types::PyString>() {
            v.call_method0("item")?
        } else {
            v
        };
        plain.set_item(k, v)?;
    }
    py_to_serde(value.py(), plain.as_any())
        .map_err(|e| PyValueError::new_err(format!("observation_readings['options']: {e}")))
}

/// The columns as a dict of numpy arrays, with `"options"`.
pub(crate) fn columns_to_py<'py>(
    py: Python<'py>,
    columns: ObservationReadingColumns,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("options", options_to_py(py, &columns.options)?)?;
    d.set_item(
        COLUMN_KEYS[0],
        columns.zncc_self_similarity_ellipse_axes.into_pyarray(py),
    )?;
    d.set_item(
        COLUMN_KEYS[1],
        columns
            .zncc_self_similarity_ellipse_axes_is_at_least
            .into_pyarray(py),
    )?;
    d.set_item(
        COLUMN_KEYS[2],
        columns
            .zncc_self_similarity_ellipse_major_angle
            .into_pyarray(py),
    )?;
    d.set_item(
        COLUMN_KEYS[3],
        columns.zncc_self_similarity_cos_view_angle.into_pyarray(py),
    )?;
    d.set_item(
        COLUMN_KEYS[4],
        columns.zncc_self_similarity_tilt_angle.into_pyarray(py),
    )?;
    d.set_item(
        COLUMN_KEYS[5],
        columns.zncc_self_similarity_zoom.into_pyarray(py),
    )?;
    d.set_item(COLUMN_KEYS[6], columns.plain_bitmap_zncc.into_pyarray(py))?;
    d.set_item(
        COLUMN_KEYS[7],
        columns.blur_matched_bitmap_zncc.into_pyarray(py),
    )?;
    Ok(d)
}

/// The in-memory readings as a dict of numpy arrays, with `"options"`.
pub(crate) fn readings_to_py<'py>(
    py: Python<'py>,
    readings: &ObservationReadings,
) -> PyResult<Bound<'py, PyDict>> {
    columns_to_py(
        py,
        ObservationReadingColumns::from_rows(&readings.rows, readings.options),
    )
}

/// The columns from a dict of numpy arrays, `None` for `None`. Every column
/// key is required, with the dtype and row shape the file stores; `"options"`
/// may be left out, or any key of it, for the defaults every writer in
/// `sfmtool` reads under. Row counts are checked against `observation_count`
/// where it is given.
pub(crate) fn columns_from_py(
    value: &Bound<'_, PyAny>,
    observation_count: Option<usize>,
) -> PyResult<Option<ObservationReadingColumns>> {
    if value.is_none() {
        return Ok(None);
    }
    let d = value.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("observation_readings must be a dict of the eight columns, or None")
    })?;
    let item = |key: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(key)?.ok_or_else(|| {
            PyValueError::new_err(format!("observation_readings is missing '{key}'"))
        })
    };
    let f32_1 = |key: &str| -> PyResult<ndarray::Array1<f32>> {
        let a: PyReadonlyArray1<f32> = item(key)?.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "observation_readings['{key}'] must be a 1D float32 array"
            ))
        })?;
        Ok(a.as_array().as_standard_layout().into_owned())
    };
    let f32_2 = |key: &str| -> PyResult<ndarray::Array2<f32>> {
        let a: PyReadonlyArray2<f32> = item(key)?.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "observation_readings['{key}'] must be a 2D float32 array"
            ))
        })?;
        Ok(a.as_array().as_standard_layout().into_owned())
    };
    let flags: PyReadonlyArray2<u8> = item(COLUMN_KEYS[1])?.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "observation_readings['{}'] must be a 2D uint8 array",
            COLUMN_KEYS[1]
        ))
    })?;
    let columns = ObservationReadingColumns {
        zncc_self_similarity_ellipse_axes: f32_2(COLUMN_KEYS[0])?,
        zncc_self_similarity_ellipse_axes_is_at_least: flags
            .as_array()
            .as_standard_layout()
            .into_owned(),
        zncc_self_similarity_ellipse_major_angle: f32_1(COLUMN_KEYS[2])?,
        zncc_self_similarity_cos_view_angle: f32_1(COLUMN_KEYS[3])?,
        zncc_self_similarity_tilt_angle: f32_1(COLUMN_KEYS[4])?,
        zncc_self_similarity_zoom: f32_2(COLUMN_KEYS[5])?,
        plain_bitmap_zncc: f32_1(COLUMN_KEYS[6])?,
        blur_matched_bitmap_zncc: f32_1(COLUMN_KEYS[7])?,
        options: options_from_py(&item("options")?)?,
    };
    // Without a count to hold them to, the first column sets it.
    let rows = observation_count.unwrap_or(columns.zncc_self_similarity_ellipse_axes.nrows());
    columns
        .validate(rows)
        .map_err(|e| PyValueError::new_err(format!("observation_readings: {e}")))?;
    Ok(Some(columns))
}

/// The in-memory readings from a dict of numpy arrays, `None` for `None`.
pub(crate) fn readings_from_py(
    value: &Bound<'_, PyAny>,
    observation_count: Option<usize>,
) -> PyResult<Option<ObservationReadings>> {
    Ok(
        columns_from_py(value, observation_count)?.map(|c| ObservationReadings {
            rows: c.rows(),
            options: c.options,
        }),
    )
}
