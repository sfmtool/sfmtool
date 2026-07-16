// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for batched reprojection residuals: compose per-image
//! world-to-camera poses with the shared camera model's canonical projection.

use std::borrow::Cow;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::geometry::{
    inlier_fraction as core_inlier_fraction, reprojection_residuals as core_reprojection_residuals,
};

use crate::geometry::PyCameraIntrinsics;

/// Per-observation reprojection residual `(proj − observed)` in pixels.
///
/// Poses are canonical world-to-camera (`x_cam = R·X + t`; the camera looks
/// along −Z). A point behind the camera or outside the model's valid domain
/// gets `invalid_residual` on its x component (y = 0), so it survives as a
/// large-residual outlier for downstream trims/inlier counts.
///
/// Args:
///     camera: ``CameraIntrinsics`` shared by all images.
///     quaternions_wxyz: (n_img, 4) world-to-camera rotations (WXYZ).
///     translations: (n_img, 3) world-to-camera translations.
///     points: (n_pt, 3) world points (canonical frame; NaN → invalid).
///     uv: (n_obs, 2) observed pixels.
///     obs_image: (n_obs,) uint32 image index per observation.
///     obs_point: (n_obs,) uint32 point index per observation.
///     invalid_residual: Residual magnitude for invalid observations
///         (default 1e6; pass float('inf') to exclude them by norm).
///
/// Returns:
///     (n_obs, 2) float64 array of (dx, dy) residuals.
#[pyfunction]
#[pyo3(signature = (
    camera,
    quaternions_wxyz,
    translations,
    points,
    uv,
    obs_image,
    obs_point,
    invalid_residual=1e6,
))]
#[allow(clippy::too_many_arguments)]
pub fn reprojection_residuals<'py>(
    py: Python<'py>,
    camera: PyRef<'_, PyCameraIntrinsics>,
    quaternions_wxyz: PyReadonlyArray2<'py, f64>,
    translations: PyReadonlyArray2<'py, f64>,
    points: PyReadonlyArray2<'py, f64>,
    uv: PyReadonlyArray2<'py, f64>,
    obs_image: PyReadonlyArray1<'py, u32>,
    obs_point: PyReadonlyArray1<'py, u32>,
    invalid_residual: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if quaternions_wxyz.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "quaternions_wxyz must have shape (n_img, 4)",
        ));
    }
    if translations.shape()[1] != 3 || points.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "translations and points must have shape (n, 3)",
        ));
    }
    if uv.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "uv must have shape (n_obs, 2)",
        ));
    }
    if obs_image.shape()[0] != obs_point.shape()[0] || obs_image.shape()[0] != uv.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obs_image, obs_point, and uv must share the same length",
        ));
    }

    let q = to_contiguous!(quaternions_wxyz);
    let t = to_contiguous!(translations);
    let p = to_contiguous!(points);
    let uvd = to_contiguous!(uv);
    let oi = to_contiguous!(obs_image);
    let op = to_contiguous!(obs_point);

    let res =
        core_reprojection_residuals(&camera.inner, &q, &t, &p, &uvd, &oi, &op, invalid_residual);
    let rows: Vec<Vec<f64>> = res.chunks(2).map(|c| vec![c[0], c[1]]).collect();
    PyArray2::from_vec2(py, &rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Fraction of observations whose reprojection residual norm is below
/// ``threshold_px``.
///
/// Args:
///     residuals: (n_obs, 2) float64 (dx, dy) from ``reprojection_residuals``.
///     threshold_px: Inlier pixel threshold.
///
/// Returns:
///     Float in [0, 1].
#[pyfunction]
pub fn inlier_fraction(residuals: PyReadonlyArray2<'_, f64>, threshold_px: f64) -> PyResult<f64> {
    if residuals.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "residuals must have shape (n_obs, 2)",
        ));
    }
    let r = to_contiguous!(residuals);
    Ok(core_inlier_fraction(&r, threshold_px))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(reprojection_residuals, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(inlier_fraction, m)?)?;
    Ok(())
}
