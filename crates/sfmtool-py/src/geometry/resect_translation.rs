// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for rotation-locked resection
//! ([`sfmtool_core::geometry::resect_translation`]).

use nalgebra::{Quaternion, UnitQuaternion};
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::resect_translation as core_resect;

use crate::geometry::PyCameraIntrinsics;

/// Resect a camera's translation against known world points with the
/// world-to-camera rotation locked (canonical frame; the camera looks along
/// −Z). See ``specs/core/rotation-locked-resection.md``.
///
/// Each observation's unit ray must be parallel to ``R·X + t``; with ``R``
/// fixed the solve is linear in ``t`` and trimmed over three rounds against a
/// pixel residual gate. Ray-space rows make the mechanism
/// camera-model-agnostic (fisheye resects through the same equations).
///
/// Args:
///     camera: ``CameraIntrinsics`` for the observations.
///     rotation_wxyz: (4,) fixed world-to-camera rotation (WXYZ).
///     points: (N, 3) world points (canonical frame).
///     uv: (N, 2) observed pixels.
///     max_error_px: Trim gate on the pixel residual norm (default 8.0).
///     min_inliers: Fewer survivors than this at any round fails (default 10).
///
/// Returns:
///     A dict ``{"translation" (3,), "inliers" (N,) bool,
///     "residual_norms" (N,)}`` for the world-to-camera translation, or
///     ``None`` when the survivor set falls below ``min_inliers``.
#[pyfunction]
#[pyo3(signature = (
    camera,
    rotation_wxyz,
    points,
    uv,
    max_error_px=8.0,
    min_inliers=10,
))]
#[allow(clippy::too_many_arguments)]
pub fn resect_translation<'py>(
    py: Python<'py>,
    camera: PyRef<'_, PyCameraIntrinsics>,
    rotation_wxyz: [f64; 4],
    points: PyReadonlyArray2<'py, f64>,
    uv: PyReadonlyArray2<'py, f64>,
    max_error_px: f64,
    min_inliers: usize,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if points.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points must have shape (N, 3)",
        ));
    }
    if uv.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "uv must have shape (N, 2)",
        ));
    }
    if points.shape()[0] != uv.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points and uv must have the same length",
        ));
    }

    let world: Vec<[f64; 3]> = points
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let uv_pts: Vec<[f64; 2]> = uv
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect();

    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(
        rotation_wxyz[0],
        rotation_wxyz[1],
        rotation_wxyz[2],
        rotation_wxyz[3],
    ));

    let Some(out) = core_resect(
        &camera.inner,
        &rotation,
        &world,
        &uv_pts,
        max_error_px,
        min_inliers,
    ) else {
        return Ok(None);
    };

    let d = PyDict::new(py);
    d.set_item(
        "translation",
        PyArray1::from_vec(
            py,
            vec![out.translation.x, out.translation.y, out.translation.z],
        ),
    )?;
    d.set_item("inliers", PyArray1::from_slice(py, &out.inliers))?;
    d.set_item("residual_norms", PyArray1::from_vec(py, out.residual_norms))?;
    Ok(Some(d))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(resect_translation, m)?)?;
    Ok(())
}
