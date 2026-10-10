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

use sfmtool_core::camera::sampler::SamplerChoice;
use sfmtool_core::reconstruction::{
    observation_reading_options, ObservationReadingOptions, ObservationReadings,
};
use sfmtool_sfmr_format::ObservationReadingColumns;

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

/// The options as a dict: `max_radius`, `flat_floor`, `noise`,
/// `relative_tolerance` and `anisotropic_threshold` (`None` where every render
/// used one sampler).
fn options_to_py<'py>(
    py: Python<'py>,
    options: &ObservationReadingOptions,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("max_radius", options.max_radius)?;
    d.set_item("flat_floor", options.flat_floor)?;
    d.set_item("noise", options.noise)?;
    d.set_item("relative_tolerance", options.relative_tolerance)?;
    d.set_item("anisotropic_threshold", options.anisotropic_threshold)?;
    Ok(d)
}

fn options_from_py(value: Option<Bound<'_, PyAny>>) -> PyResult<ObservationReadingOptions> {
    let defaults = observation_reading_options(SamplerChoice::per_view());
    let Some(value) = value.filter(|v| !v.is_none()) else {
        return Ok(defaults);
    };
    let d = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("observation_readings['options'] must be a dict"))?;
    let get = |key: &str| d.get_item(key);
    Ok(ObservationReadingOptions {
        max_radius: match get("max_radius")? {
            Some(v) => v.extract()?,
            None => defaults.max_radius,
        },
        flat_floor: match get("flat_floor")? {
            Some(v) => v.extract()?,
            None => defaults.flat_floor,
        },
        noise: match get("noise")? {
            Some(v) => v.extract()?,
            None => defaults.noise,
        },
        relative_tolerance: match get("relative_tolerance")? {
            Some(v) => v.extract()?,
            None => defaults.relative_tolerance,
        },
        anisotropic_threshold: match get("anisotropic_threshold")? {
            Some(v) => v.extract()?,
            None => defaults.anisotropic_threshold,
        },
    })
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
        Ok(a.as_array().to_owned())
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
        options: options_from_py(d.get_item("options")?)?,
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
