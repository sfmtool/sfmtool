// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the staged bundle adjustment
//! ([`sfmtool_core::geometry::bundle_adjust()`]).

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::camera::CameraModel;
use sfmtool_core::geometry::{
    bundle_adjust as core_bundle_adjust, BaSchedule, DistanceReference, FreePointPolicy,
    PointConstraints,
};
use sfmtool_core::progress::Progress;

use crate::geometry::PyCameraIntrinsics;

/// Read the `distance_from` argument into one optional reference per point.
///
/// The argument is a sequence of `n_pt` entries, each an image index or a
/// sequence of image indices whose camera centres are averaged; a negative index
/// (the documented `-1`) is "no origin". One iteration covers every accepted
/// form, because a numpy integer array yields scalars that convert to an index
/// and a list of lists yields sequences -- so the caller can pass the flat array
/// that covers the common single-image case without the binding growing a second
/// path for it.
fn parse_distance_from(
    obj: &Bound<'_, PyAny>,
    n_pt: usize,
) -> PyResult<Vec<Option<DistanceReference>>> {
    let mut out: Vec<Option<DistanceReference>> = Vec::with_capacity(n_pt);
    for (p, item) in obj.try_iter()?.enumerate() {
        let item = item?;
        if let Ok(k) = item.extract::<i64>() {
            out.push((k >= 0).then_some(DistanceReference::Image(k as u32)));
            continue;
        }
        let ks: Vec<i64> = item.extract().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "distance_from[{p}] must be an image index or a sequence of image \
                 indices, got {}",
                item.get_type()
                    .qualname()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|_| "an unknown type".to_string())
            ))
        })?;
        let kept: Vec<u32> = ks.iter().filter(|&&k| k >= 0).map(|&k| k as u32).collect();
        if kept.len() != ks.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "distance_from[{p}] mixes a negative index into a set of images; \
                 a set is averaged, so every member has to be a real image"
            )));
        }
        out.push((!kept.is_empty()).then_some(DistanceReference::ImageMean(kept)));
    }
    if out.len() != n_pt {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "distance_from must have one entry per point: {} given for {n_pt} points",
            out.len()
        )));
    }
    Ok(out)
}

/// Build the kernel's point constraints from the three per-point arguments,
/// or `None` when the caller owns no point -- which is the off position the
/// parity requirement is stated against.
///
/// The rules are [`PointConstraints::from_arrays`]'s, so this binding and the
/// reconstruction-level adjustment that reads the same statements out of a
/// file's constraint columns cannot disagree about what they mean; what is left
/// here is the exception type.
fn build_constraints(
    held: Option<&[bool]>,
    distance: Option<&[f64]>,
    origins: &[Option<DistanceReference>],
    n_pt: usize,
    n_img: usize,
) -> PyResult<Option<PointConstraints>> {
    PointConstraints::from_arrays(held, distance, origins, n_pt, n_img)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Staged bundle adjustment for images sharing one camera model.
///
/// Jointly refines world-to-camera poses, world points, and optionally the
/// shared focal length by minimizing soft-L1 pixel reprojection error over a
/// trim schedule with inter-round retriangulation (canonical frame; the
/// camera looks along −Z). See ``specs/core/geometry/bundle-adjustment.md``.
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
///     held: Optional (n_pt,) bool mask marking held points. A held point's
///         coordinate is the caller's for the whole solve: its observations
///         still form residuals and still drive the cameras and the lens, it
///         owns no parameters, the re-estimation skips it, and it comes back
///         exactly as it went in. Absent or all-``False`` is every point
///         unheld.
///     distance: Optional (n_pt,) float64 array of ranged points' distances --
///         a strictly positive world-unit distance, ``+inf`` for a direction,
///         and ``NaN`` for a point this rule says nothing about. A ranged
///         point is ``X = O + r * d``: the caller owns ``r``, the solve owns
///         the unit direction ``d``, and ``O`` comes from ``distance_from``.
///         Held and ranged are exclusive on one point.
///     distance_from: Optional (n_pt,) sequence naming where each finite
///         distance is measured from: an image index, or a sequence of image
///         indices whose camera centres are averaged, with ``-1`` (or an empty
///         sequence) where there is none. A finite ``distance`` requires one --
///         the adjustment's gauge is free, so a distance from a fixed world
///         coordinate would constrain nothing -- while ``+inf`` and ``NaN``
///         rows ignore it.
///     free_points_cross: Re-decide every free point's representation at each
///         inter-round re-estimation, from its own rays at the current
///         geometry: a track whose widest ray pair opens past the noise floor
///         is finite, one that closes below it is a direction, and so is one
///         that solves behind a camera observing it. ``False`` honours the
///         caller's ``point_at_infinity`` mask for the whole solve, which
///         reproduces the standing kernel bit for bit.
///     noise_floor_scale: The constant ``c`` in the noise-floor angle
///         ``theta_floor = c * s / f``, with ``s`` the round's loss scale in
///         pixels and ``f`` the camera's current focal (default 2.0; must be
///         positive and finite). Read only under ``free_points_cross``.
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
///     (n_pt, 3), "residual_norms" (n_obs,), "point_at_infinity" (n_pt,)}``.
///     ``point_at_infinity`` is the representation each point ended with:
///     ``True`` where its returned row is a world-frame direction and
///     ``False`` where it is a position. A free point's entry is the input
///     mask unless ``free_points_cross`` let the re-estimation re-decide it, a
///     held point's is its input value, and a ranged point's is whether its
///     distance is infinite.
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
    held=None,
    distance=None,
    distance_from=None,
    free_points_cross=false,
    noise_floor_scale=sfmtool_core::geometry::DEFAULT_NOISE_FLOOR_SCALE,
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
    held: Option<PyReadonlyArray1<'py, bool>>,
    distance: Option<PyReadonlyArray1<'py, f64>>,
    distance_from: Option<Bound<'py, PyAny>>,
    free_points_cross: bool,
    noise_floor_scale: f64,
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
    if let Some(ref mask) = held {
        if mask.shape()[0] != n_pt {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "held must have shape (n_pt,)",
            ));
        }
    }
    if let Some(ref r) = distance {
        if r.shape()[0] != n_pt {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "distance must have shape (n_pt,)",
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
    if !(noise_floor_scale.is_finite() && noise_floor_scale > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "noise_floor_scale must be positive and finite",
        ));
    }
    let held_mask: Option<Vec<bool>> = held.map(|m| to_contiguous!(m).into_owned());
    let distance_values: Option<Vec<f64>> = distance.map(|r| to_contiguous!(r).into_owned());
    let origins = match &distance_from {
        Some(obj) => parse_distance_from(obj, n_pt)?,
        None => vec![None; n_pt],
    };
    let constraints = build_constraints(
        held_mask.as_deref(),
        distance_values.as_deref(),
        &origins,
        n_pt,
        n_img,
    )?;
    let free_points = FreePointPolicy {
        cross: free_points_cross,
        noise_floor_scale,
    };
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
    let uv_rows: Vec<[f64; 2]> = uv_in
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| [c[0], c[1]])
        .collect();
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
            constraints.as_ref(),
            free_points,
            prot_mask.as_deref(),
            protected_loss_scale,
            opt_f,
            opt_k1,
            opt_bspline,
            &stages,
            max_iters,
            min_track,
            min_obs,
            &Progress::none(),
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
    d.set_item(
        "point_at_infinity",
        PyArray1::from_vec(py, out.point_at_infinity),
    )?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(bundle_adjust, m)?)?;
    Ok(())
}
