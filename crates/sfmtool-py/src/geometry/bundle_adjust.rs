// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the staged bundle adjustment
//! ([`sfmtool_core::geometry::bundle_adjust()`]).

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
///     protected: Optional (n_obs,) bool mask marking protected
///         observations. A protected observation is never removed by the
///         inter-round trim gates — it stays in the solve set every round
///         regardless of its residual and always counts toward ``min_track``
///         survival — and passes through the robust loss at the wider scale
///         ``protected_loss_scale * loss_scale``. Absent or all-``False``
///         reproduces the unprotected behavior bit for bit. Composable with
///         ``point_at_infinity``.
///     protected_loss_scale: Multiplier on each stage's loss scale for
///         protected observations (default 3.0; must be positive and
///         finite).
///     opt_f: Release the shared focal (SIMPLE_PINHOLE,
///         EQUIDISTANT_FISHEYE, SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE or
///         SFMTOOL_PINHOLE — the models whose projection multiplies the focal
///         onto a distorted coordinate that does not itself read it, where the
///         kernel's analytic focal column is exact; any other model raises).
///     opt_k1: Release the shared radial coefficient (SIMPLE_RADIAL_FISHEYE
///         only — the one model carrying it; any other model raises). The
///         staged use is fixed -> opt_f -> opt_f + opt_k1, so the curvature
///         rung opens on a focal that has already settled.
///     opt_bspline: Release the shared radial spline coefficients
///         (SFMTOOL_FISHEYE or SFMTOOL_PINHOLE — the two models carrying
///         them, and the spline must be defined: at least two coefficients on
///         a positive ``bspline_theta_max`` / ``bspline_rho_max``; anything
///         else raises). Mutually exclusive
///         with ``opt_k1`` (no model carries both parameters). The staged
///         use mirrors the curvature rung's: fixed -> opt_f -> opt_f +
///         opt_bspline.
///     schedule: [(trim_px, loss_scale), ...] staged rounds
///         (default [(50, 5), (12, 2), (4, 1)]).
///     max_iters: LM iteration budget per round (default 60).
///     min_track: Trim survivors a point needs to stay in a solve (default 2).
///     min_obs: Below this many trim survivors the round exits degenerate:
///         state passes through, all residual norms +inf (default 12).
///
/// Returns:
///     A dict ``{"focal", "k1", "bspline_coefficients" (n_coeffs,),
///     "quaternions_wxyz" (n_img, 4), "translations" (n_img, 3), "points"
///     (n_pt, 3), "residual_norms" (n_obs,)}``.
///     ``k1`` is the shared radial coefficient after the solve — the input
///     one unless ``opt_k1``, and 0.0 for models that have none.
///     ``bspline_coefficients`` mirrors it for the radial spline: the
///     coefficients after the solve — the camera's input ones unless
///     ``opt_bspline``, and an empty array for models that carry no spline.
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
    protected=None,
    protected_loss_scale=3.0,
    opt_f=false,
    opt_k1=false,
    opt_bspline=false,
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
    protected: Option<PyReadonlyArray1<'py, bool>>,
    protected_loss_scale: f64,
    opt_f: bool,
    opt_k1: bool,
    opt_bspline: bool,
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
    // The focal column `∂(u, v)/∂f = (u − cx)/f` is exact only where the focal
    // multiplies an `f`-independent distorted coordinate: the two single-focal
    // distortion-free models, the one-coefficient fisheye whose `k1` rides on
    // the ray's own `θ`, and the two spline models whose dimensionless radial
    // spline rides on the ray's own radial coordinate the same way. Everything
    // else is rejected loudly rather than degraded to a fixed-parameter solve
    // behind the caller's back.
    let releasable = matches!(
        camera.inner.model,
        CameraModel::SimplePinhole { .. }
            | CameraModel::EquidistantFisheye { .. }
            | CameraModel::SimpleRadialFisheye { .. }
            | CameraModel::SfmtoolFisheye { .. }
            | CameraModel::SfmtoolPinhole { .. }
    );
    if opt_f && !releasable {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opt_f requires a SIMPLE_PINHOLE, EQUIDISTANT_FISHEYE, \
             SIMPLE_RADIAL_FISHEYE, SFMTOOL_FISHEYE or SFMTOOL_PINHOLE camera",
        ));
    }
    // The two distortion rungs live on different models, so no camera could
    // ever satisfy both releases — reject the combination up front with the
    // real reason rather than whichever model gate happens to fire first.
    if opt_k1 && opt_bspline {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opt_k1 and opt_bspline are mutually exclusive (no camera model \
             carries both a radial coefficient and a spline)",
        ));
    }
    // The curvature rung exists on exactly one model — no other camera has a
    // single radial coefficient acting on `θ` for `f·θ³·û` to be its exact
    // derivative.
    if opt_k1 && !matches!(camera.inner.model, CameraModel::SimpleRadialFisheye { .. }) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opt_k1 requires a SIMPLE_RADIAL_FISHEYE camera",
        ));
    }
    // The spline rung exists on the two models that carry a spline, whose
    // dimensionless coefficients act on the ray's own radial coordinate — and
    // the spline must be defined for there to be anything to release.
    if opt_bspline {
        match camera.inner.model.radial_spline() {
            Some((bspline, d_max, _)) => {
                if bspline.len() < 2 || !(d_max.is_finite() && d_max > 0.0) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "opt_bspline requires a defined spline (at least two \
                         coefficients on a positive bspline_theta_max / \
                         bspline_rho_max)",
                    ));
                }
            }
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "opt_bspline requires a SFMTOOL_FISHEYE or SFMTOOL_PINHOLE camera",
                ));
            }
        }
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
    if let Some(ref mask) = protected {
        if mask.shape()[0] != uv.shape()[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "protected must have shape (n_obs,)",
            ));
        }
    }
    if !(protected_loss_scale.is_finite() && protected_loss_scale > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "protected_loss_scale must be positive and finite",
        ));
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
    let prot_mask: Option<Vec<bool>> = protected.map(|mask| to_contiguous!(mask).into_owned());

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
            prot_mask.as_deref(),
            protected_loss_scale,
            opt_f,
            opt_k1,
            opt_bspline,
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
    d.set_item("k1", out.k1)?;
    d.set_item("bspline_coefficients", PyArray1::from_vec(py, out.bspline))?;
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
