// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for epipolar-curve plotting (distortion-aware epipolar
//! "lines", which curve for fisheye / wide-FOV cameras).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::epipolar::{plot_epipolar_curves_batch, EpipolarCurveOptions};
use sfmtool_core::RigidTransform;

use crate::PyCameraIntrinsics;

/// Plot epipolar curves in image 2 for a batch of pixels in image 1.
///
/// `points1` is an `(N, 2)` array of pixel coordinates. `anchor_depths` is an
/// `(N,)` array of per-feature seed depths in camera 1 — typically the
/// reconstructed depth of the observed track when triangulated, otherwise the
/// baseline length `‖C2 − C1‖`. `cam1`/`cam2` are `CameraIntrinsics`;
/// `q1_wxyz`/`q2_wxyz` are WXYZ quaternions and `t1`/`t2` the translations of
/// the `cam_from_world` poses.
///
/// Returns a list of `(M, 2)` arrays — one polyline per input point. `M` varies
/// per curve (the adaptive sampler emits sparse vertices where curvature is
/// low and dense vertices where it's high) and may be 0 for degenerate
/// baselines or features with no in-image projection in `cam2`. All returned
/// vertices are inside `[0, cam2.width) × [0, cam2.height)` — no further
/// image-rectangle clipping is needed at draw time.
#[pyfunction]
#[pyo3(name = "epipolar_curves")]
#[pyo3(signature = (points1, anchor_depths, cam1, q1_wxyz, t1, cam2, q2_wxyz, t2,
                     *, curvature_tolerance=0.5, max_vertices=256))]
#[allow(clippy::too_many_arguments)]
pub fn epipolar_curves_py(
    py: Python<'_>,
    points1: PyReadonlyArray2<'_, f64>,
    anchor_depths: PyReadonlyArray1<'_, f64>,
    cam1: &PyCameraIntrinsics,
    q1_wxyz: [f64; 4],
    t1: [f64; 3],
    cam2: &PyCameraIntrinsics,
    q2_wxyz: [f64; 4],
    t2: [f64; 3],
    curvature_tolerance: f64,
    max_vertices: usize,
) -> PyResult<Vec<Py<PyArray2<f64>>>> {
    if points1.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points1 must have shape (N, 2)",
        ));
    }
    let n = points1.shape()[0];
    if anchor_depths.shape()[0] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "anchor_depths length {} does not match points1 length {}",
            anchor_depths.shape()[0],
            n
        )));
    }
    let pts: Vec<[f64; 2]> = points1
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect();
    let anchors: Vec<f64> = anchor_depths.as_array().iter().copied().collect();

    let pose1 = RigidTransform::from_wxyz_translation(q1_wxyz, t1);
    let pose2 = RigidTransform::from_wxyz_translation(q2_wxyz, t2);
    let opts = EpipolarCurveOptions {
        curvature_tolerance,
        max_vertices,
    };

    let curves = plot_epipolar_curves_batch(
        &pts,
        &anchors,
        &cam1.inner,
        &pose1,
        &cam2.inner,
        &pose2,
        &opts,
    );

    Ok(curves
        .into_iter()
        .map(|curve| {
            let m = curve.len();
            let flat: Vec<f64> = curve.into_iter().flatten().collect();
            Array2::from_shape_vec((m, 2), flat)
                .expect("flat length is 2*m by construction")
                .into_pyarray(py)
                .into()
        })
        .collect())
}
