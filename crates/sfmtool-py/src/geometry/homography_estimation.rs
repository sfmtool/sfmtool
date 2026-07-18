// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for homography estimation from 2D-2D correspondences
//! (4-point LO-RANSAC ``estimate_homography``; see
//! ``specs/core/focal-vote.md``).

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::homography_estimation::{
    estimate_homography as core_estimate, HomographyEstimate, HomographyOptions,
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
    est: &HomographyEstimate,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let h = &est.h_matrix;
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|i| vec![h[(i, 0)], h[(i, 1)], h[(i, 2)]])
        .collect();
    let h_arr = PyArray2::from_vec2(py, &rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    d.set_item("h_matrix", h_arr)?;
    d.set_item("inliers", PyArray1::from_slice(py, &est.inliers))?;
    d.set_item("iterations", est.iterations)?;
    Ok(d)
}

/// Robustly estimate the homography from pixel correspondences (4-point
/// LO-RANSAC; see ``specs/core/focal-vote.md``).
///
/// Args:
///     points1: (N, 2) float64 pixels in image 1.
///     points2: (N, 2) float64 pixels in image 2.
///     max_error_px: Inlier bound on the symmetric transfer error, pixels
///         (default 3.0).
///     confidence: Adaptive-termination confidence (default 0.999).
///     max_iterations: Hard trial cap (default 10000).
///     min_inliers: Reject a consensus below this (default 4).
///     seed: SplitMix64 sampler seed; same inputs + seed => bit-identical
///         output (default 0).
///     local_optimization: Refit each new best consensus on its inliers with
///         the normalized DLT (default True).
///
/// Returns:
///     A dict ``{"h_matrix" (3, 3) float64 unit-Frobenius, "inliers" (N,)
///     bool, "iterations" int}``, or ``None`` when no consensus reaches
///     ``min_inliers``.
#[pyfunction]
#[pyo3(signature = (
    points1,
    points2,
    *,
    max_error_px=3.0,
    confidence=0.999,
    max_iterations=10_000,
    min_inliers=4,
    seed=0,
    local_optimization=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_homography(
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

    let options = HomographyOptions {
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(estimate_homography, m)?)?;
    Ok(())
}
