// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the ray-space two-view estimators
//! (``sfmtool_core::geometry::relative_pose``): a robust epipolar matrix and a
//! robust rotation, both estimated on **unit rays** at a camera the caller
//! already has.
//!
//! These are the focal-vote column scan's estimators evaluated once at a known
//! camera model instead of scanned over candidate focals. Inputs are unit rays
//! (e.g. from ``CameraIntrinsics.pixel_to_ray_batch``), so a field of view past
//! 180° is ordinary input: rays with a non-positive ``z`` participate like any
//! others, and residual bounds are angles.

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use nalgebra::{Matrix3, Vector3};
use sfmtool_core::geometry::relative_pose::{
    estimate_essential_rays as core_essential, fit_ray_rotation as core_rotation, EpipolarSide,
    RayEssentialOptions, RayRotationOptions,
};

/// Read an (N, 3) float64 array of rays, normalizing each to unit length.
fn read_rays(arr: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<Vector3<f64>>> {
    if arr.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (N, 3), got (N, {})",
            arr.shape()[1]
        )));
    }
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| {
            let v = Vector3::new(r[0], r[1], r[2]);
            let n = v.norm();
            if n <= 0.0 || !n.is_finite() {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} holds a zero or non-finite ray"
                )))
            } else {
                Ok(v / n)
            }
        })
        .collect()
}

/// A (3, 3) numpy array from a nalgebra matrix.
fn matrix_to_py<'py>(py: Python<'py>, m: &Matrix3<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|i| vec![m[(i, 0)], m[(i, 1)], m[(i, 2)]])
        .collect();
    PyArray2::from_vec2(py, &rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

fn parse_side(name: &str) -> PyResult<EpipolarSide> {
    match name {
        "both" => Ok(EpipolarSide::Both),
        "one" => Ok(EpipolarSide::One),
        "two" => Ok(EpipolarSide::Two),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "side must be one of 'both', 'one', 'two'; got {other:?}"
        ))),
    }
}

fn check_same_length(a: usize, b: usize) -> PyResult<()> {
    if a != b {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "correspondence count mismatch: {a} vs {b}"
        )));
    }
    Ok(())
}

/// Robustly estimate the ray-space epipolar matrix from unit-ray
/// correspondences.
///
/// Args:
///     rays1: (N, 3) float64 rays in camera 1 (normalized on read).
///     rays2: (N, 3) float64 rays in camera 2.
///     max_angle_rad: Consensus bound on the ray-to-epipolar-plane angle.
///         Derive it from a pixel tolerance through the camera map's local
///         ``dr/dtheta`` (``tol_px / f`` for an equidistant map).
///     min_inliers: Reject a consensus below this (default 12).
///     samples: Minimal samples drawn, 8 indexes each (default 512).
///     seed: SplitMix64 sampler seed; same inputs + seed => bit-identical
///         output (default 0).
///     side: ``"both"`` (default, the larger of the two one-sided angles),
///         ``"two"`` (image-2 rays against ``E x1``) or ``"one"``
///         (image-1 rays against ``E^T x2``).
///
/// Returns:
///     A dict ``{"e_matrix" (3, 3) float64 unit-Frobenius, "inliers" (N,)
///     bool, "essentialness" float, "residuals_rad" (N,) float64, "rms_rad"
///     float}``, or ``None`` when no consensus reaches ``min_inliers``.
#[pyfunction]
#[pyo3(signature = (
    rays1,
    rays2,
    *,
    max_angle_rad,
    min_inliers=12,
    samples=512,
    seed=0,
    side="both",
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_essential_rays(
    py: Python<'_>,
    rays1: PyReadonlyArray2<'_, f64>,
    rays2: PyReadonlyArray2<'_, f64>,
    max_angle_rad: f64,
    min_inliers: usize,
    samples: usize,
    seed: u64,
    side: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let r1 = read_rays(&rays1, "rays1")?;
    let r2 = read_rays(&rays2, "rays2")?;
    check_same_length(r1.len(), r2.len())?;
    let options = RayEssentialOptions {
        max_angle_rad,
        min_inliers,
        samples,
        seed,
        side: parse_side(side)?,
    };
    let Some(out) = core_essential(&r1, &r2, &options) else {
        return Ok(None);
    };
    let d = PyDict::new(py);
    d.set_item("e_matrix", matrix_to_py(py, &out.e_matrix)?)?;
    d.set_item("inliers", PyArray1::from_slice(py, &out.inliers))?;
    d.set_item("essentialness", out.essentialness)?;
    d.set_item(
        "residuals_rad",
        PyArray1::from_slice(py, &out.residuals_rad),
    )?;
    d.set_item("rms_rad", out.rms_rad)?;
    Ok(Some(d.into_any().unbind()))
}

/// Robustly fit a rotation of unit rays — the far-field model of a pair.
///
/// Args:
///     rays1: (N, 3) float64 rays in camera 1 (normalized on read).
///     rays2: (N, 3) float64 rays in camera 2.
///     max_angle_rad: Consensus bound on the angle between a rotated ray and
///         its partner.
///     min_inliers: Reject a consensus below this (default 20).
///     samples: Minimal samples drawn, 3 indexes each (default 512).
///     seed: SplitMix64 sampler seed (default 0).
///
/// Returns:
///     A dict ``{"rotation" (3, 3) float64 with ``R x1 ~ x2``, "inliers" (N,)
///     bool, "residuals_rad" (N,) float64, "rms_rad" float}``, or ``None``
///     when no consensus reaches ``min_inliers``.
#[pyfunction]
#[pyo3(signature = (rays1, rays2, *, max_angle_rad, min_inliers=20, samples=512, seed=0))]
pub fn fit_ray_rotation(
    py: Python<'_>,
    rays1: PyReadonlyArray2<'_, f64>,
    rays2: PyReadonlyArray2<'_, f64>,
    max_angle_rad: f64,
    min_inliers: usize,
    samples: usize,
    seed: u64,
) -> PyResult<Option<Py<PyAny>>> {
    let r1 = read_rays(&rays1, "rays1")?;
    let r2 = read_rays(&rays2, "rays2")?;
    check_same_length(r1.len(), r2.len())?;
    let options = RayRotationOptions {
        max_angle_rad,
        min_inliers,
        samples,
        seed,
    };
    let Some(out) = core_rotation(&r1, &r2, &options) else {
        return Ok(None);
    };
    let d = PyDict::new(py);
    d.set_item("rotation", matrix_to_py(py, &out.rotation)?)?;
    d.set_item("inliers", PyArray1::from_slice(py, &out.inliers))?;
    d.set_item(
        "residuals_rad",
        PyArray1::from_slice(py, &out.residuals_rad),
    )?;
    d.set_item("rms_rad", out.rms_rad)?;
    Ok(Some(d.into_any().unbind()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(estimate_essential_rays, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(fit_ray_rotation, m)?)?;
    Ok(())
}
