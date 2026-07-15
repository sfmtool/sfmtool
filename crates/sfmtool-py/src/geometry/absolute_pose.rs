// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the absolute-pose solver: the Lambda Twist minimal
//! solver ``p3p_solve`` and the robust RANSAC estimator
//! ``estimate_absolute_pose`` (see ``specs/core/absolute-pose.md``).

use nalgebra::{Point3, Vector3};
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::absolute_pose::{
    estimate_absolute_pose as core_estimate, p3p_solve as core_p3p_solve, AbsolutePoseEstimate,
    AbsolutePoseOptions,
};

use crate::geometry::PyCameraIntrinsics;

/// Read an (N, 3) float64 array into world points.
fn read_points3(arr: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<Point3<f64>>> {
    if arr.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (N, 3), got (N, {})",
            arr.shape()[1]
        )));
    }
    let view = arr.as_array();
    Ok(view
        .rows()
        .into_iter()
        .map(|r| Point3::new(r[0], r[1], r[2]))
        .collect())
}

/// Read an (N, 3) float64 array into bearing vectors.
fn read_bearings(arr: &PyReadonlyArray2<'_, f64>) -> Vec<Vector3<f64>> {
    arr.as_array()
        .rows()
        .into_iter()
        .map(|r| Vector3::new(r[0], r[1], r[2]))
        .collect()
}

/// Solve for up to four world-to-camera poses from three bearing/point
/// correspondences (Lambda Twist P3P; see ``specs/core/absolute-pose.md``).
///
/// Args:
///     bearings: (3, 3) float64 unit ray directions in the canonical camera
///         frame (a camera looks along −Z; a point in front has z < 0).
///     points3d: (3, 3) float64 world points.
///
/// Returns:
///     A list of ``(quaternion_wxyz, translation)`` pairs — each a (4,) and a
///     (3,) float64 numpy array for the world-to-camera pose
///     ``x_cam = R·X + t``. Empty for degenerate inputs (collinear points,
///     coincident/antipodal bearings, non-finite values).
#[pyfunction]
pub fn p3p_solve(
    py: Python<'_>,
    bearings: PyReadonlyArray2<'_, f64>,
    points3d: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
    if bearings.shape() != [3, 3] || points3d.shape() != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "p3p_solve requires bearings (3, 3) and points3d (3, 3)",
        ));
    }
    let b = read_bearings(&bearings);
    let x = read_points3(&points3d, "points3d")?;
    let barr = [b[0], b[1], b[2]];
    let xarr = [x[0], x[1], x[2]];
    let mut out = Vec::new();
    for (rot, t) in core_p3p_solve(&barr, &xarr) {
        let q = rot.into_inner();
        let quat = PyArray1::from_vec(py, vec![q.w, q.i, q.j, q.k])
            .into_any()
            .unbind();
        let tr = PyArray1::from_vec(py, vec![t.x, t.y, t.z])
            .into_any()
            .unbind();
        out.push((quat, tr));
    }
    Ok(out)
}

/// Build the estimate dict returned to Python.
fn estimate_to_dict<'py>(
    py: Python<'py>,
    est: &AbsolutePoseEstimate,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let q = est.rotation.into_inner();
    d.set_item(
        "quaternion_wxyz",
        PyArray1::from_vec(py, vec![q.w, q.i, q.j, q.k]),
    )?;
    d.set_item(
        "translation",
        PyArray1::from_vec(
            py,
            vec![est.translation.x, est.translation.y, est.translation.z],
        ),
    )?;
    d.set_item("inliers", PyArray1::from_slice(py, &est.inliers))?;
    d.set_item("iterations", est.iterations)?;
    Ok(d)
}

/// Robustly estimate a camera's world-to-camera pose from 2D-3D
/// correspondences (P3P + RANSAC; see ``specs/core/absolute-pose.md``).
///
/// The first argument is either (N, 2) pixel observations — requiring
/// ``camera`` — or (N, 3) unit bearing vectors in the canonical camera frame
/// (a camera looks along −Z; a point in front has z < 0).
///
/// Args:
///     points2d_or_bearings: (N, 2) float64 pixels or (N, 3) float64 bearings.
///     points3d: (N, 3) float64 world points.
///     camera: ``CameraIntrinsics``; required for (N, 2) pixel input. With
///         pixels the threshold defaults to ``atan(max_error_px / f_mean)``
///         where ``f_mean`` is the mean of the camera's focal lengths.
///     max_error_px: Pixel inlier bound, converted to an angular threshold via
///         the camera's mean focal length (default 4.0).
///     max_angular_error: Angular inlier bound in radians; overrides
///         ``max_error_px`` when given. Required for (N, 3) input without a
///         camera.
///     confidence: Adaptive-termination confidence (default 0.999).
///     max_iterations: Hard trial cap (default 50000).
///     min_inliers: Reject a consensus below this (default 6).
///     seed: SplitMix64 sampler seed; same inputs + seed => bit-identical
///         output (default 0).
///     local_optimization: Refit each new best consensus on its inliers
///         (default True).
///
/// Returns:
///     A dict ``{"quaternion_wxyz" (4,), "translation" (3,), "inliers" (N,),
///     "iterations"}`` for the canonical world-to-camera pose, or ``None``
///     when no consensus reaches ``min_inliers``.
#[pyfunction]
#[pyo3(signature = (
    points2d_or_bearings,
    points3d,
    *,
    camera=None,
    max_error_px=4.0,
    max_angular_error=None,
    confidence=0.999,
    max_iterations=50_000,
    min_inliers=6,
    seed=0,
    local_optimization=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_absolute_pose(
    py: Python<'_>,
    points2d_or_bearings: PyReadonlyArray2<'_, f64>,
    points3d: PyReadonlyArray2<'_, f64>,
    camera: Option<PyRef<'_, PyCameraIntrinsics>>,
    max_error_px: f64,
    max_angular_error: Option<f64>,
    confidence: f64,
    max_iterations: u32,
    min_inliers: usize,
    seed: u64,
    local_optimization: bool,
) -> PyResult<Option<Py<PyAny>>> {
    let points = read_points3(&points3d, "points3d")?;
    let cols = points2d_or_bearings.shape()[1];

    let (bearings, angular) = match cols {
        3 => {
            let bearings = read_bearings(&points2d_or_bearings);
            let angular = match max_angular_error {
                Some(a) => a,
                None => match &camera {
                    Some(cam) => {
                        let (fx, fy) = cam.inner.focal_lengths();
                        (max_error_px / (0.5 * (fx + fy))).atan()
                    }
                    None => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "(N, 3) bearing input needs max_angular_error or a camera to \
                             derive the angular threshold",
                        ));
                    }
                },
            };
            (bearings, angular)
        }
        2 => {
            let cam = camera.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "(N, 2) pixel input requires a camera to convert pixels to bearings",
                )
            })?;
            let view = points2d_or_bearings.as_array();
            let pixels: Vec<[f64; 2]> = view.rows().into_iter().map(|r| [r[0], r[1]]).collect();
            let rays = cam.inner.pixel_to_ray_batch(&pixels);
            let bearings: Vec<Vector3<f64>> = rays
                .iter()
                .map(|r| Vector3::new(r[0], r[1], r[2]))
                .collect();
            let angular = match max_angular_error {
                Some(a) => a,
                None => {
                    let (fx, fy) = cam.inner.focal_lengths();
                    (max_error_px / (0.5 * (fx + fy))).atan()
                }
            };
            (bearings, angular)
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "points2d_or_bearings must be (N, 2) pixels or (N, 3) bearings, got (N, {other})"
            )));
        }
    };

    if bearings.len() != points.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "correspondence count mismatch: {} observations vs {} points",
            bearings.len(),
            points.len()
        )));
    }

    let options = AbsolutePoseOptions {
        max_angular_error: angular,
        confidence,
        max_iterations,
        min_inliers,
        seed,
        local_optimization,
    };
    match core_estimate(&bearings, &points, &options) {
        Some(est) => Ok(Some(estimate_to_dict(py, &est)?.into_any().unbind())),
        None => Ok(None),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(p3p_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(estimate_absolute_pose, m)?)?;
    Ok(())
}
