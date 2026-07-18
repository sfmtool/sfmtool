// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the staged bundle adjustment
//! ([`sfmtool_core::geometry::bundle_adjust`]).

use std::borrow::Cow;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::camera::CameraModel;
use sfmtool_core::geometry::{bundle_adjust as core_bundle_adjust, BaSchedule};

use crate::geometry::PyCameraIntrinsics;

/// Staged bundle adjustment for images sharing one camera model.
///
/// Jointly refines world-to-camera poses, world points, and optionally the
/// shared focal length by minimizing soft-L1 pixel reprojection error over a
/// trim schedule with inter-round retriangulation (canonical frame; the
/// camera looks along −Z). See ``specs/core/bundle-adjustment.md``.
///
/// Args:
///     camera: ``CameraIntrinsics`` shared by all images (carries the
///         initial focal).
///     quaternions_wxyz: (n_img, 4) world-to-camera rotations (WXYZ).
///     translations: (n_img, 3) world-to-camera translations.
///     points: (n_pt, 3) world points; NaN rows are re-admitted by the
///         retriangulation rounds when observed twice.
///     uv: (n_obs, 2) observed pixels.
///     obs_image: (n_obs,) uint32 image index per observation.
///     obs_point: (n_obs,) uint32 point index per observation.
///     point_at_infinity: Optional (n_pt,) bool mask marking points at
///         infinity. A marked row of ``points`` is a world-frame direction
///         (normalized on input and returned as a unit direction) whose
///         observations depend on rotation and camera model only; an image
///         whose surviving observations are all directions keeps its
///         translation frozen. Absent or all-``False`` reproduces the
///         finite-only kernel bit for bit.
///     opt_f: Release the shared focal (SIMPLE_PINHOLE only).
///     schedule: [(trim_px, loss_scale), ...] staged rounds
///         (default [(50, 5), (12, 2), (4, 1)]).
///     max_iters: LM iteration budget per round (default 60).
///     min_track: Trim survivors a point needs to stay in a solve (default 2).
///     min_obs: Below this many trim survivors the round exits degenerate:
///         state passes through, all residual norms +inf (default 12).
///
/// Returns:
///     A dict ``{"focal", "quaternions_wxyz" (n_img, 4), "translations"
///     (n_img, 3), "points" (n_pt, 3), "residual_norms" (n_obs,)}``.
///     ``residual_norms`` are unweighted reprojection norms at the final
///     state, ``+inf`` where the point is non-finite / behind the camera /
///     outside the model domain.
#[pyfunction]
#[pyo3(signature = (
    camera,
    quaternions_wxyz,
    translations,
    points,
    uv,
    obs_image,
    obs_point,
    point_at_infinity=None,
    opt_f=false,
    schedule=vec![(50.0, 5.0), (12.0, 2.0), (4.0, 1.0)],
    max_iters=60,
    min_track=2,
    min_obs=12,
))]
#[allow(clippy::too_many_arguments)]
pub fn bundle_adjust<'py>(
    py: Python<'py>,
    camera: PyRef<'_, PyCameraIntrinsics>,
    quaternions_wxyz: PyReadonlyArray2<'py, f64>,
    translations: PyReadonlyArray2<'py, f64>,
    points: PyReadonlyArray2<'py, f64>,
    uv: PyReadonlyArray2<'py, f64>,
    obs_image: PyReadonlyArray1<'py, u32>,
    obs_point: PyReadonlyArray1<'py, u32>,
    point_at_infinity: Option<PyReadonlyArray1<'py, bool>>,
    opt_f: bool,
    schedule: Vec<(f64, f64)>,
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
) -> PyResult<Bound<'py, PyDict>> {
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
    if translations.shape()[0] != quaternions_wxyz.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "quaternions_wxyz and translations must share n_img",
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
    if schedule.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "schedule must have at least one (trim_px, loss_scale) round",
        ));
    }
    if opt_f && !matches!(camera.inner.model, CameraModel::SimplePinhole { .. }) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opt_f requires a SIMPLE_PINHOLE camera",
        ));
    }

    let n_img = quaternions_wxyz.shape()[0];
    let n_pt = points.shape()[0];
    if let Some(ref mask) = point_at_infinity {
        if mask.shape()[0] != n_pt {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "point_at_infinity must have shape (n_pt,)",
            ));
        }
    }
    let q_in = to_contiguous!(quaternions_wxyz);
    let t_in = to_contiguous!(translations);
    let p_in = to_contiguous!(points);
    let uv_in = to_contiguous!(uv);
    let oi = to_contiguous!(obs_image);
    let op = to_contiguous!(obs_point);
    if let Some(&bad) = oi.iter().find(|&&i| i as usize >= n_img) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "obs_image index {bad} out of range ({n_img} images)"
        )));
    }
    if let Some(&bad) = op.iter().find(|&&p| p as usize >= n_pt) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "obs_point index {bad} out of range ({n_pt} points)"
        )));
    }

    let mut quats: Vec<UnitQuaternion<f64>> = (0..n_img)
        .map(|i| {
            UnitQuaternion::from_quaternion(Quaternion::new(
                q_in[i * 4],
                q_in[i * 4 + 1],
                q_in[i * 4 + 2],
                q_in[i * 4 + 3],
            ))
        })
        .collect();
    let mut trans: Vec<Vector3<f64>> = (0..n_img)
        .map(|i| Vector3::new(t_in[i * 3], t_in[i * 3 + 1], t_in[i * 3 + 2]))
        .collect();
    let mut pts: Vec<[f64; 3]> = (0..n_pt)
        .map(|p| [p_in[p * 3], p_in[p * 3 + 1], p_in[p * 3 + 2]])
        .collect();
    let uv_rows: Vec<[f64; 2]> = uv_in.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let stages: Vec<BaSchedule> = schedule
        .iter()
        .map(|&(trim_px, loss_scale)| BaSchedule {
            trim_px,
            loss_scale,
        })
        .collect();

    let inf_mask: Option<Vec<bool>> =
        point_at_infinity.map(|mask| to_contiguous!(mask).into_owned());

    let cam = camera.inner.clone();
    let (out, quats, trans, pts) = py.detach(move || {
        let out = core_bundle_adjust(
            &cam,
            &mut quats,
            &mut trans,
            &mut pts,
            &uv_rows,
            &oi,
            &op,
            inf_mask.as_deref(),
            opt_f,
            &stages,
            max_iters,
            min_track,
            min_obs,
        );
        (out, quats, trans, pts)
    });

    let q_rows: Vec<Vec<f64>> = quats
        .iter()
        .map(|q| {
            let q = q.into_inner();
            vec![q.w, q.i, q.j, q.k]
        })
        .collect();
    let t_rows: Vec<Vec<f64>> = trans.iter().map(|t| vec![t.x, t.y, t.z]).collect();
    let p_rows: Vec<Vec<f64>> = pts.iter().map(|p| p.to_vec()).collect();

    let d = PyDict::new(py);
    d.set_item("focal", out.focal)?;
    d.set_item(
        "quaternions_wxyz",
        PyArray2::from_vec2(py, &q_rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    )?;
    d.set_item(
        "translations",
        PyArray2::from_vec2(py, &t_rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    )?;
    d.set_item(
        "points",
        PyArray2::from_vec2(py, &p_rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    )?;
    d.set_item("residual_norms", PyArray1::from_vec(py, out.residual_norms))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(bundle_adjust, m)?)?;
    Ok(())
}
