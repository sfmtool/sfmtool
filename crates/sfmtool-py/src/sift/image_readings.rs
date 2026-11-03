// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for an image's grey conversion and its sharpness readings:
//! `ImageToGray`, `ImageReadings` and `image_readings_from_rgb`.

use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use sfmtool_core::features::sift::{
    image_readings_from_rgb as core_image_readings_from_rgb, ImageReadings, ImageToGray,
};

use crate::helpers::{py_to_serde, serde_to_py};

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The conversion from an image's colour samples to its grey samples.
///
/// A formula over ``R``, ``G`` and ``B`` (each the 8-bit sample divided by
/// 255), evaluated in double precision, and an optional ``quantize_bits``
/// from 1 to 16 that rounds its value to
/// ``round(clamp(v, 0, 1) * (2**b - 1)) / (2**b - 1)``, an exact half rounded
/// up. It is the ``image_to_gray`` object
/// ``{"formula": ..., "quantize_bits": ...}``.
///
/// Args:
///     formula: the formula, in the grammar of the ``gray_formula`` feature
///         option (``+``, ``*``, ``**``, parentheses, ``R``, ``G``, ``B`` and
///         decimal literals).
///     quantize_bits: the bit depth to round to, or None for no rounding.
///
/// Raises:
///     ValueError: if the formula does not parse or ``quantize_bits`` is not
///         from 1 to 16.
#[pyclass(name = "ImageToGray", module = "sfmtool.sift", frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct PyImageToGray {
    pub(crate) inner: ImageToGray,
}

#[pymethods]
impl PyImageToGray {
    #[new]
    #[pyo3(signature = (formula, quantize_bits=None))]
    fn new(formula: &str, quantize_bits: Option<i64>) -> PyResult<Self> {
        let bits = match quantize_bits {
            None => None,
            Some(b) => Some(u8::try_from(b).map_err(|_| {
                PyValueError::new_err(format!(
                    "image_to_gray 'quantize_bits' must be an integer from 1 to 16, got {b}"
                ))
            })?),
        };
        let inner = ImageToGray::new(formula, bits).map_err(value_error)?;
        Ok(Self { inner })
    }

    /// The conversion a version 1 ``gray_formula`` string records: the formula
    /// with no rounding.
    ///
    /// Raises:
    ///     ValueError: if the formula does not parse.
    #[staticmethod]
    fn from_gray_formula(formula: &str) -> PyResult<Self> {
        let inner = ImageToGray::from_gray_formula(formula).map_err(value_error)?;
        Ok(Self { inner })
    }

    /// Read the ``image_to_gray`` object, given as a dict or as JSON text.
    ///
    /// It has a string ``formula`` and, optionally, an integer
    /// ``quantize_bits`` from 1 to 16, and no other key.
    ///
    /// Raises:
    ///     ValueError: for any other value, naming what is wrong.
    #[staticmethod]
    fn from_json(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let json: serde_json::Value = if let Ok(text) = value.cast::<PyString>() {
            serde_json::from_str(text.to_str()?).map_err(|e| {
                PyValueError::new_err(format!("image_to_gray is not valid JSON: {e}"))
            })?
        } else {
            py_to_serde(py, value)?
        };
        let inner = ImageToGray::from_json(&json).map_err(value_error)?;
        Ok(Self { inner })
    }

    /// The ``image_to_gray`` object as a dict: ``formula``, and
    /// ``quantize_bits`` only when present.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.inner.to_json())
    }

    /// The formula's text, as given.
    #[getter]
    fn formula(&self) -> &str {
        self.inner.formula()
    }

    /// The bit depth the formula's value is rounded to, or None.
    #[getter]
    fn quantize_bits(&self) -> Option<u8> {
        self.inner.quantize_bits()
    }

    /// The grey sample of channel values ``r``, ``g``, ``b`` on ``[0, 1]``.
    fn gray(&self, r: f64, g: f64, b: f64) -> f64 {
        self.inner.gray(r, g, b)
    }

    fn __repr__(&self) -> String {
        match self.inner.quantize_bits() {
            Some(b) => format!("ImageToGray({:?}, quantize_bits={b})", self.inner.formula()),
            None => format!("ImageToGray({:?})", self.inner.formula()),
        }
    }
}

/// An image's sharpness and gradient structure tensor, from
/// ``image_readings_from_rgb``.
///
/// Attributes:
///     sharpness: the population variance of the 3 x 3 Laplacian over the
///         reading image's interior, in grey units squared.
///     gradient_structure_tensor_xx_yy_xy: ``(Exx, Eyy, Exy)``, the means of
///         ``Gx**2``, ``Gy**2`` and ``Gx*Gy`` over the interior, with ``Gx``
///         and ``Gy`` the Sobel derivatives divided by 8, x right and y down.
#[pyclass(name = "ImageReadings", module = "sfmtool.sift", frozen)]
pub struct PyImageReadings {
    inner: ImageReadings,
}

#[pymethods]
impl PyImageReadings {
    #[getter]
    fn sharpness(&self) -> f64 {
        self.inner.sharpness
    }

    #[getter]
    fn gradient_structure_tensor_xx_yy_xy(&self) -> (f64, f64, f64) {
        let [xx, yy, xy] = self.inner.gradient_structure_tensor_xx_yy_xy;
        (xx, yy, xy)
    }

    fn __repr__(&self) -> String {
        let [xx, yy, xy] = self.inner.gradient_structure_tensor_xx_yy_xy;
        format!(
            "ImageReadings(sharpness={}, gradient_structure_tensor_xx_yy_xy=({xx}, {yy}, {xy}))",
            self.inner.sharpness
        )
    }
}

/// The sharpness and gradient structure tensor of an 8-bit RGB image.
///
/// The image is made grey by ``image_to_gray`` in double precision, reduced
/// by an exact area average to a long side of 480 pixels (an image no larger
/// is used at its own size; the short side is rounded half up), and read over
/// its interior pixels. An image whose reading image is under 3 pixels on a
/// side reads 0.
///
/// Args:
///     image: (H, W, 3) uint8 RGB array, in stored pixel order.
///     image_to_gray: the ``ImageToGray`` conversion.
///
/// Returns:
///     ImageReadings.
///
/// Raises:
///     ValueError: if the array is not (H, W, 3), or a reading is not finite
///         under the conversion (the message names the conversion).
#[pyfunction]
#[pyo3(signature = (image, image_to_gray))]
pub fn image_readings_from_rgb(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, u8>,
    image_to_gray: &PyImageToGray,
) -> PyResult<PyImageReadings> {
    let shape = image.shape();
    let (h, w) = match shape {
        [h, w, 3] => (*h, *w),
        other => {
            return Err(PyValueError::new_err(format!(
                "image must be (H, W, 3) uint8 RGB, got shape {other:?}"
            )))
        }
    };
    let (width, height) = match (u32::try_from(w), u32::try_from(h)) {
        (Ok(width), Ok(height)) => (width, height),
        _ => {
            return Err(PyValueError::new_err(format!(
                "image is {w} x {h}, larger than a u32 side"
            )))
        }
    };
    let data = to_contiguous!(image);
    let data = &*data;
    let to_gray = &image_to_gray.inner;
    let inner = py
        .detach(|| core_image_readings_from_rgb(width, height, data, to_gray))
        .map_err(value_error)?;
    Ok(PyImageReadings { inner })
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImageToGray>()?;
    m.add_class::<PyImageReadings>()?;
    m.add_function(wrap_pyfunction!(image_readings_from_rgb, m)?)?;
    Ok(())
}
