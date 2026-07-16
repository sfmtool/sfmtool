// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for trimmed pose-only resection refinement
//! ([`sfmtool_core::geometry::refine_absolute_pose`]).

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::refine_absolute_pose as core_refine;

use crate::geometry::PyCameraIntrinsics;

/// Refine a world-to-camera pose against 2D-3D correspondences by trimmed
/// pixel-reprojection least squares (canonical frame; the camera looks along
/// −Z).
///
/// Args:
///     camera: ``CameraIntrinsics`` for the observations.
///     uv: (N, 2) observed pixels.
///     points: (N, 3) world points (canonical frame).
///     init_quaternion_wxyz: (4,) initial world-to-camera rotation (WXYZ).
///     init_translation: (3,) initial world-to-camera translation.
///     trim_rounds: L2 refits on the best-fitting ``keep_fraction`` (default 5).
///     keep_fraction: Fraction retained each trim round (default 0.6).
///     inlier_px: Final-inlier pixel threshold (default 3.0).
///
/// Returns:
///     A dict ``{"quaternion_wxyz" (4,), "translation" (3,),
///     "inlier_fraction"}`` for the refined world-to-camera pose.
#[pyfunction]
#[pyo3(signature = (
    camera,
    uv,
    points,
    init_quaternion_wxyz,
    init_translation,
    trim_rounds=5,
    keep_fraction=0.6,
    inlier_px=3.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn refine_absolute_pose<'py>(
    py: Python<'py>,
    camera: PyRef<'_, PyCameraIntrinsics>,
    uv: PyReadonlyArray2<'py, f64>,
    points: PyReadonlyArray2<'py, f64>,
    init_quaternion_wxyz: [f64; 4],
    init_translation: [f64; 3],
    trim_rounds: usize,
    keep_fraction: f64,
    inlier_px: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if uv.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "uv must have shape (N, 2)",
        ));
    }
    if points.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points must have shape (N, 3)",
        ));
    }
    if uv.shape()[0] != points.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "uv and points must have the same length",
        ));
    }

    let uv_pts: Vec<[f64; 2]> = uv
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect();
    let world: Vec<[f64; 3]> = points
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();

    let init_r = UnitQuaternion::from_quaternion(Quaternion::new(
        init_quaternion_wxyz[0],
        init_quaternion_wxyz[1],
        init_quaternion_wxyz[2],
        init_quaternion_wxyz[3],
    ));
    let init_t = Vector3::new(
        init_translation[0],
        init_translation[1],
        init_translation[2],
    );

    let out = core_refine(
        &camera.inner,
        &uv_pts,
        &world,
        &init_r,
        &init_t,
        trim_rounds,
        keep_fraction,
        inlier_px,
    );

    let q = out.rotation.into_inner();
    let d = PyDict::new(py);
    d.set_item(
        "quaternion_wxyz",
        PyArray1::from_vec(py, vec![q.w, q.i, q.j, q.k]),
    )?;
    d.set_item(
        "translation",
        PyArray1::from_vec(
            py,
            vec![out.translation.x, out.translation.y, out.translation.z],
        ),
    )?;
    d.set_item("inlier_fraction", out.inlier_fraction)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(refine_absolute_pose, m)?)?;
    Ok(())
}
