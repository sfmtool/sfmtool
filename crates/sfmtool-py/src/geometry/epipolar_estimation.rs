// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for fundamental-matrix estimation from 2D-2D
//! correspondences (7-point LO-RANSAC ``estimate_fundamental``) and the
//! Bougnoux focal-length recovery ``focal_from_fundamental``
//! (see ``specs/core/epipolar-estimation.md``).

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use nalgebra::Matrix3;
use sfmtool_core::geometry::epipolar_estimation::{
    estimate_fundamental as core_estimate, focal_from_fundamental as core_focal,
    FundamentalEstimate, FundamentalOptions,
};

/// Read an (N, 2) float64 array into pixel correspondences.
fn read_points2(arr: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<[f64; 2]>> {
    if arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (N, 2), got (N, {})",
            arr.shape()[1]
        )));
    }
    Ok(arr
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect())
}

/// Build the estimate dict returned to Python.
fn estimate_to_dict<'py>(
    py: Python<'py>,
    est: &FundamentalEstimate,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let f = &est.f_matrix;
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|i| vec![f[(i, 0)], f[(i, 1)], f[(i, 2)]])
        .collect();
    let f_arr = PyArray2::from_vec2(py, &rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    d.set_item("f_matrix", f_arr)?;
    d.set_item("inliers", PyArray1::from_slice(py, &est.inliers))?;
    d.set_item("iterations", est.iterations)?;
    Ok(d)
}

/// Robustly estimate the fundamental matrix from pixel correspondences
/// (7-point LO-RANSAC; see ``specs/core/epipolar-estimation.md``).
///
/// Args:
///     points1: (N, 2) float64 pixels in image 1.
///     points2: (N, 2) float64 pixels in image 2.
///     max_error_px: Inlier bound on the Sampson distance, pixels (default 3.0).
///     confidence: Adaptive-termination confidence (default 0.999).
///     max_iterations: Hard trial cap (default 10000).
///     min_inliers: Reject a consensus below this (default 12).
///     seed: SplitMix64 sampler seed; same inputs + seed => bit-identical
///         output (default 0).
///     local_optimization: Refit each new best consensus on its inliers with
///         the normalized 8-point solver (default True).
///
/// Returns:
///     A dict ``{"f_matrix" (3, 3) float64 rank-2 unit-Frobenius, "inliers"
///     (N,) bool, "iterations" int}``, or ``None`` when no consensus reaches
///     ``min_inliers``.
#[pyfunction]
#[pyo3(signature = (
    points1,
    points2,
    *,
    max_error_px=3.0,
    confidence=0.999,
    max_iterations=10_000,
    min_inliers=12,
    seed=0,
    local_optimization=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_fundamental(
    py: Python<'_>,
    points1: PyReadonlyArray2<'_, f64>,
    points2: PyReadonlyArray2<'_, f64>,
    max_error_px: f64,
    confidence: f64,
    max_iterations: u32,
    min_inliers: usize,
    seed: u64,
    local_optimization: bool,
) -> PyResult<Option<Py<PyAny>>> {
    let x1 = read_points2(&points1, "points1")?;
    let x2 = read_points2(&points2, "points2")?;
    if x1.len() != x2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "correspondence count mismatch: {} vs {}",
            x1.len(),
            x2.len()
        )));
    }

    let options = FundamentalOptions {
        max_error_px,
        confidence,
        max_iterations,
        min_inliers,
        seed,
        local_optimization,
    };
    match core_estimate(&x1, &x2, &options) {
        Some(est) => Ok(Some(estimate_to_dict(py, &est)?.into_any().unbind())),
        None => Ok(None),
    }
}

/// Focal length of camera 1 in pixels from the fundamental matrix and the two
/// principal points (Bougnoux; see ``specs/core/epipolar-estimation.md``).
///
/// Assumes square pixels, zero skew, and known principal points. Returns
/// ``None`` for the classical focal-recovery degeneracies (intersecting
/// optical axes, pure forward translation, rotation-dominant motion).
///
/// Args:
///     f_matrix: (3, 3) float64 fundamental matrix.
///     principal_point1: (2,) pixels, camera 1.
///     principal_point2: (2,) pixels, camera 2.
///
/// Returns:
///     The camera-1 focal length in pixels, or ``None`` when degenerate.
#[pyfunction]
pub fn focal_from_fundamental(
    f_matrix: PyReadonlyArray2<'_, f64>,
    principal_point1: [f64; 2],
    principal_point2: [f64; 2],
) -> PyResult<Option<f64>> {
    if f_matrix.shape() != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "f_matrix must have shape (3, 3)",
        ));
    }
    let view = f_matrix.as_array();
    let f = Matrix3::new(
        view[(0, 0)],
        view[(0, 1)],
        view[(0, 2)],
        view[(1, 0)],
        view[(1, 1)],
        view[(1, 2)],
        view[(2, 0)],
        view[(2, 1)],
        view[(2, 2)],
    );
    Ok(core_focal(&f, principal_point1, principal_point2))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(estimate_fundamental, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(focal_from_fundamental, m)?)?;
    Ok(())
}
