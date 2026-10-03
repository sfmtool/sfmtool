// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the point-or-bearing test: the batch score and fit over
//! the CSR ray layout, the construction of a ray and its noise weight from an
//! observation, and the dicts the reconstruction-level convenience returns.
//!
//! See `specs/core/reconstruction/batch-triangulation-api.md` § "Point or
//! bearing".

use nalgebra::{Matrix2x3, Point3, Quaternion, UnitQuaternion};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::reconstruction::triangulation::{
    bearing_score_batch as core_bearing_score_batch,
    fit_point_and_bearing_batch as core_fit_point_and_bearing_batch, is_finite,
    observed_ray as core_observed_ray, BearingScore, PointBearingFit, PointBearingFitOptions,
    DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD, DEFAULT_POINT_FIT_MAX_ITERATIONS,
    DEFAULT_SOFT_L1_SCALE,
};

use super::triangulation::{csr_offsets, point3_rows, vec3_rows};
use crate::PyCameraIntrinsics;

/// `(N, 3)` float64 from rows.
fn rows3<'py>(py: Python<'py>, rows: Vec<[f64; 3]>) -> Bound<'py, PyArray2<f64>> {
    let n = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    ndarray::Array2::from_shape_vec((n, 3), flat)
        .expect("n rows of 3")
        .into_pyarray(py)
}

/// `(N, 2, 3)` float64 from weights.
fn weights3<'py>(py: Python<'py>, weights: &[Matrix2x3<f64>]) -> Bound<'py, PyArray3<f64>> {
    let mut flat = Vec::with_capacity(weights.len() * 6);
    for w in weights {
        for r in 0..2 {
            for c in 0..3 {
                flat.push(w[(r, c)]);
            }
        }
    }
    ndarray::Array3::from_shape_vec((weights.len(), 2, 3), flat)
        .expect("n weights of 2x3")
        .into_pyarray(py)
}

const NAN3: [f64; 3] = [f64::NAN; 3];

/// The dict of a batch of [`BearingScore`]s, with `is_finite` at `threshold`.
/// Float fields are NaN, flags false and counts 0 where a track has no score.
pub(crate) fn scores_to_dict<'py>(
    py: Python<'py>,
    scores: &[Option<BearingScore>],
    threshold: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let get = |f: fn(&BearingScore) -> f64| -> Vec<f64> {
        scores
            .iter()
            .map(|s| s.as_ref().map_or(f64::NAN, f))
            .collect()
    };
    let dict = PyDict::new(py);
    dict.set_item(
        "scored",
        PyArray1::from_vec(py, scores.iter().map(Option::is_some).collect()),
    )?;
    dict.set_item(
        "bearing",
        rows3(
            py,
            scores
                .iter()
                .map(|s| s.as_ref().map_or(NAN3, |s| s.bearing.into()))
                .collect(),
        ),
    )?;
    dict.set_item(
        "bearing_cost",
        PyArray1::from_vec(py, get(|s| s.bearing_cost)),
    )?;
    dict.set_item(
        "depth_score",
        PyArray1::from_vec(py, get(|s| s.depth_score)),
    )?;
    dict.set_item(
        "midpoint_bound",
        PyArray1::from_vec(py, get(|s| s.midpoint_bound)),
    )?;
    dict.set_item(
        "bearing_in_front_of_all_cameras",
        PyArray1::from_vec(
            py,
            scores
                .iter()
                .map(|s| {
                    s.as_ref()
                        .is_some_and(|s| s.bearing_in_front_of_all_cameras)
                })
                .collect(),
        ),
    )?;
    dict.set_item(
        "num_views",
        PyArray1::from_vec(
            py,
            scores
                .iter()
                .map(|s| s.as_ref().map_or(0, |s| s.num_views as i64))
                .collect(),
        ),
    )?;
    dict.set_item(
        "is_finite",
        PyArray1::from_vec(
            py,
            scores
                .iter()
                .map(|s| s.as_ref().is_some_and(|s| is_finite(s, threshold)))
                .collect(),
        ),
    )?;
    Ok(dict)
}

/// The dict of a batch of [`PointBearingFit`]s. Float fields are NaN, flags
/// false and counts 0 where a track has no fit; `point` is NaN also where the
/// fit's inverse depth is 0.
pub(crate) fn fits_to_dict<'py>(
    py: Python<'py>,
    fits: &[Option<PointBearingFit>],
) -> PyResult<Bound<'py, PyDict>> {
    let get = |f: fn(&PointBearingFit) -> f64| -> Vec<f64> {
        fits.iter()
            .map(|s| s.as_ref().map_or(f64::NAN, f))
            .collect()
    };
    let get3 = |f: fn(&PointBearingFit) -> [f64; 3]| -> Vec<[f64; 3]> {
        fits.iter().map(|s| s.as_ref().map_or(NAN3, f)).collect()
    };
    let dict = PyDict::new(py);
    dict.set_item(
        "fitted",
        PyArray1::from_vec(py, fits.iter().map(Option::is_some).collect()),
    )?;
    dict.set_item("bearing", rows3(py, get3(|f| f.bearing.into())))?;
    dict.set_item(
        "bearing_cost",
        PyArray1::from_vec(py, get(|f| f.bearing_cost)),
    )?;
    dict.set_item("anchor", rows3(py, get3(|f| f.anchor.coords.into())))?;
    dict.set_item("direction", rows3(py, get3(|f| f.direction.into())))?;
    dict.set_item(
        "inverse_depth",
        PyArray1::from_vec(py, get(|f| f.inverse_depth)),
    )?;
    dict.set_item(
        "point",
        rows3(py, get3(|f| f.point.map_or(NAN3, |p| p.coords.into()))),
    )?;
    dict.set_item("point_cost", PyArray1::from_vec(py, get(|f| f.point_cost)))?;
    dict.set_item(
        "depth_likelihood_ratio",
        PyArray1::from_vec(py, get(|f| f.depth_likelihood_ratio)),
    )?;
    dict.set_item(
        "in_front_of_all_cameras",
        PyArray1::from_vec(
            py,
            fits.iter()
                .map(|f| f.as_ref().is_some_and(|f| f.in_front_of_all_cameras))
                .collect(),
        ),
    )?;
    dict.set_item(
        "num_views",
        PyArray1::from_vec(
            py,
            fits.iter()
                .map(|f| f.as_ref().map_or(0, |f| f.num_views as i64))
                .collect(),
        ),
    )?;
    Ok(dict)
}

/// Refuse a threshold `is_finite` cannot read: it must be finite and not
/// negative.
pub(crate) fn check_threshold(threshold: f64) -> PyResult<()> {
    if threshold.is_finite() && threshold >= 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "threshold must be finite and not negative, got {threshold}"
        )))
    }
}

/// Point indexes from a 1-D numpy array of any integer dtype, or from a list
/// or tuple of integers. A negative index is an `IndexError`.
pub(crate) fn point_index_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    fn from_array<T: numpy::Element + Copy + TryInto<usize>>(
        obj: &Bound<'_, PyAny>,
    ) -> Option<PyResult<Vec<usize>>> {
        let a = obj.extract::<PyReadonlyArray1<'_, T>>().ok()?;
        Some(
            a.as_array()
                .iter()
                .map(|&i| {
                    i.try_into().map_err(|_| {
                        pyo3::exceptions::PyIndexError::new_err(
                            "point indexes must not be negative",
                        )
                    })
                })
                .collect(),
        )
    }
    let typed = from_array::<i64>(obj)
        .or_else(|| from_array::<i32>(obj))
        .or_else(|| from_array::<i16>(obj))
        .or_else(|| from_array::<i8>(obj))
        .or_else(|| from_array::<u64>(obj))
        .or_else(|| from_array::<u32>(obj))
        .or_else(|| from_array::<u16>(obj))
        .or_else(|| from_array::<u8>(obj));
    if let Some(result) = typed {
        return result;
    }
    let values: Vec<i64> = obj.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "point_indexes must be a 1-D integer array or a sequence of integers",
        )
    })?;
    values
        .into_iter()
        .map(|i| {
            usize::try_from(i).map_err(|_| {
                pyo3::exceptions::PyIndexError::new_err(format!("point index {i} is negative"))
            })
        })
        .collect()
}

/// Fit options from the Python keywords, refusing a scale or an iteration
/// count the fit cannot use.
pub(crate) fn fit_options(
    soft_l1_scale: Option<f64>,
    max_iterations: usize,
) -> PyResult<PointBearingFitOptions> {
    if let Some(s) = soft_l1_scale {
        if !(s.is_finite() && s > 0.0) {
            return Err(PyValueError::new_err(format!(
                "soft_l1_scale must be finite and positive, or None, got {s}"
            )));
        }
    }
    Ok(PointBearingFitOptions {
        soft_l1_scale,
        max_iterations,
    })
}

/// The rays, centres, offsets and weights of a batch call, validated.
struct BatchRays {
    dirs: Vec<nalgebra::Vector3<f64>>,
    centers: Vec<Point3<f64>>,
    offsets: Vec<usize>,
    weights: Vec<Matrix2x3<f64>>,
}

fn batch_rays(
    dirs: &PyReadonlyArray2<f64>,
    centers: &PyReadonlyArray2<f64>,
    offsets: &PyReadonlyArray1<i64>,
    weights: &PyReadonlyArray3<f64>,
) -> PyResult<BatchRays> {
    let t = dirs.shape()[0];
    if centers.shape()[0] != t {
        return Err(PyValueError::new_err(
            "dirs and centers must have the same length",
        ));
    }
    let ws = weights.shape();
    if ws[0] != t || ws[1] != 2 || ws[2] != 3 {
        return Err(PyValueError::new_err(format!(
            "weights must have shape ({t}, 2, 3), one 2x3 weight per ray, got ({}, {}, {})",
            ws[0], ws[1], ws[2]
        )));
    }
    let w = to_contiguous!(weights);
    Ok(BatchRays {
        dirs: vec3_rows("dirs", dirs)?,
        centers: point3_rows("centers", centers)?,
        offsets: csr_offsets(offsets, t)?,
        weights: w
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| Matrix2x3::new(c[0], c[1], c[2], c[3], c[4], c[5]))
            .collect(),
    })
}

/// Optional per-track points (starts or anchors), one row per track; NaN rows
/// mean none for that track.
fn per_track_points(
    name: &str,
    a: Option<&PyReadonlyArray2<f64>>,
    m: usize,
) -> PyResult<Option<Vec<Point3<f64>>>> {
    let Some(a) = a else { return Ok(None) };
    if a.shape()[0] != m {
        return Err(PyValueError::new_err(format!(
            "{name} must have one row per track ({m}), got {}",
            a.shape()[0]
        )));
    }
    point3_rows(name, a).map(Some)
}

/// Score a batch of tracks: the bearing that best explains each track's rays,
/// and how strongly the rays ask for a depth.
///
/// Tracks are flattened CSR-style as for ``triangulate_batch``: track ``t``
/// owns ``dirs[offsets[t]:offsets[t+1]]`` and the matching ``centers`` and
/// ``weights``. Each ray's residual is ``W (I - d dᵀ) m``, in units of the
/// noise; build the weights with ``observed_rays``.
///
/// Args:
///     dirs: Unit world-space rays, shape ``(T, 3)`` float64.
///     centers: Matching camera centres, shape ``(T, 3)`` float64.
///     offsets: CSR track boundaries, shape ``(M + 1,)`` int64.
///     weights: Per-ray 2x3 world-frame noise weights, shape ``(T, 2, 3)``
///         float64.
///     threshold: The value of the score (and of the likelihood ratio) above
///         which ``is_finite`` calls a track a finite point, finite and not
///         negative. Defaults to
///         ``DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`` (25).
///
/// Returns:
///     A dict of arrays, one row per track: ``scored`` ``(M,)`` bool (false
///     where fewer than two rays are usable), ``bearing`` ``(M, 3)``,
///     ``bearing_cost``, ``depth_score``, ``midpoint_bound`` ``(M,)`` float64,
///     ``bearing_in_front_of_all_cameras`` ``(M,)`` bool, ``num_views``
///     ``(M,)`` int64 and ``is_finite`` ``(M,)`` bool. Unscored rows are NaN,
///     false or 0.
#[pyfunction]
#[pyo3(signature = (dirs, centers, offsets, weights, threshold=DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD))]
pub fn bearing_score_batch<'py>(
    py: Python<'py>,
    dirs: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray3<'py, f64>,
    threshold: f64,
) -> PyResult<Bound<'py, PyDict>> {
    check_threshold(threshold)?;
    let rays = batch_rays(&dirs, &centers, &offsets, &weights)?;
    let scores = py.detach(|| {
        core_bearing_score_batch(&rays.dirs, &rays.centers, &rays.offsets, &rays.weights)
    });
    scores_to_dict(py, &scores, threshold)
}

/// Fit both models (a bearing, and a point by inverse depth about an anchor)
/// to each track, and the exact likelihood ratio between them.
///
/// Inputs as for ``bearing_score_batch``.
///
/// Args:
///     dirs, centers, offsets, weights: As for ``bearing_score_batch``.
///     starts: Optional ``(M, 3)`` float64 points to start each point fit
///         from (a stored position); a NaN row means none for that track.
///     anchors: Optional ``(M, 3)`` float64 anchors of the point model; a NaN
///         row means the centroid of the track's camera centres.
///     soft_l1_scale: Soft-L1 loss scale in noise units, or ``None`` for plain
///         least squares, the loss whose ratio the score approximates. Defaults
///         to ``DEFAULT_SOFT_L1_SCALE`` (3), as the core does.
///     max_iterations: Levenberg-Marquardt iterations per fit. Defaults to
///         ``DEFAULT_POINT_FIT_MAX_ITERATIONS`` (20).
///
/// Returns:
///     A dict of arrays, one row per track: ``fitted`` ``(M,)`` bool,
///     ``bearing`` ``(M, 3)``, ``bearing_cost``, ``anchor`` ``(M, 3)``,
///     ``direction`` ``(M, 3)``, ``inverse_depth``, ``point`` ``(M, 3)`` (NaN
///     where the inverse depth is 0), ``point_cost``,
///     ``depth_likelihood_ratio``, ``in_front_of_all_cameras`` ``(M,)`` bool
///     and ``num_views`` ``(M,)`` int64. Unfitted rows are NaN, false or 0.
#[pyfunction]
#[pyo3(signature = (
    dirs,
    centers,
    offsets,
    weights,
    starts=None,
    anchors=None,
    soft_l1_scale=Some(DEFAULT_SOFT_L1_SCALE),
    max_iterations=DEFAULT_POINT_FIT_MAX_ITERATIONS,
))]
#[allow(clippy::too_many_arguments)]
pub fn fit_point_and_bearing_batch<'py>(
    py: Python<'py>,
    dirs: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray3<'py, f64>,
    starts: Option<PyReadonlyArray2<'py, f64>>,
    anchors: Option<PyReadonlyArray2<'py, f64>>,
    soft_l1_scale: Option<f64>,
    max_iterations: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let rays = batch_rays(&dirs, &centers, &offsets, &weights)?;
    let m = rays.offsets.len().saturating_sub(1);
    let starts = per_track_points("starts", starts.as_ref(), m)?;
    let anchors = per_track_points("anchors", anchors.as_ref(), m)?;
    let options = fit_options(soft_l1_scale, max_iterations)?;
    let fits = py.detach(|| {
        core_fit_point_and_bearing_batch(
            &rays.dirs,
            &rays.centers,
            &rays.offsets,
            &rays.weights,
            starts.as_deref(),
            anchors.as_deref(),
            &options,
        )
    });
    fits_to_dict(py, &fits)
}

/// The world-space ray through each pixel and its noise weight, for one
/// camera.
///
/// The weight is ``(1/sigma_px) J R``: ``J`` the 2x3 derivative of the
/// camera's projection at the camera-frame ray, ``R`` the world-to-camera
/// rotation. It maps a world-frame change of the ray's direction to pixels over
/// the noise, so ``bearing_score_batch``'s residuals are, to first order, the
/// pixel residuals over ``sigma_px``.
///
/// Args:
///     camera: The ``CameraIntrinsics`` every pixel was taken with.
///     cam_from_world_wxyz: World-to-camera rotations as WXYZ quaternions,
///         shape ``(N, 4)`` float64, one per pixel (normalised here).
///     pixels: Observed pixels, shape ``(N, 2)`` float64.
///     sigma_px: Per-axis pixel noise, finite and positive.
///
/// Returns:
///     A dict: ``valid`` ``(N,)`` bool, ``dirs`` ``(N, 3)`` float64 and
///     ``weights`` ``(N, 2, 3)`` float64. A pixel is not valid (NaN rows) when
///     its un-projected ray does not project back to it, which is outside the
///     camera model's domain.
#[pyfunction]
pub fn observed_rays<'py>(
    py: Python<'py>,
    camera: PyRef<'py, PyCameraIntrinsics>,
    cam_from_world_wxyz: PyReadonlyArray2<'py, f64>,
    pixels: PyReadonlyArray2<'py, f64>,
    sigma_px: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if !(sigma_px.is_finite() && sigma_px > 0.0) {
        return Err(PyValueError::new_err(format!(
            "sigma_px must be finite and positive, got {}",
            sfmtool_core::readable::Readable(sigma_px)
        )));
    }
    let n = pixels.shape()[0];
    if pixels.shape()[1] != 2 {
        return Err(PyValueError::new_err("pixels must have shape (N, 2)"));
    }
    let qs = cam_from_world_wxyz.shape();
    if qs[0] != n || qs[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "cam_from_world_wxyz must have shape ({n}, 4), got ({}, {})",
            qs[0], qs[1]
        )));
    }
    let q = to_contiguous!(cam_from_world_wxyz);
    let px = to_contiguous!(pixels);
    let camera = camera.inner.clone();
    let mut valid = Vec::with_capacity(n);
    let mut dirs = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for (q, p) in q.as_chunks::<4>().0.iter().zip(px.as_chunks::<2>().0) {
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
        match core_observed_ray(&camera, &rotation, [p[0], p[1]], sigma_px) {
            Some(ray) => {
                valid.push(true);
                dirs.push(ray.dir.into());
                weights.push(ray.weight);
            }
            None => {
                valid.push(false);
                dirs.push(NAN3);
                weights.push(Matrix2x3::from_element(f64::NAN));
            }
        }
    }
    let dict = PyDict::new(py);
    dict.set_item("valid", PyArray1::from_vec(py, valid))?;
    dict.set_item("dirs", rows3(py, dirs))?;
    dict.set_item("weights", weights3(py, &weights))?;
    Ok(dict)
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bearing_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fit_point_and_bearing_batch, m)?)?;
    m.add_function(wrap_pyfunction!(observed_rays, m)?)?;
    m.add(
        "DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD",
        DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
    )?;
    m.add("DEFAULT_SOFT_L1_SCALE", DEFAULT_SOFT_L1_SCALE)?;
    m.add(
        "DEFAULT_POINT_FIT_MAX_ITERATIONS",
        DEFAULT_POINT_FIT_MAX_ITERATIONS,
    )?;
    m.add(
        "REPROJECTION_NOISE_OUTLIER_GATE",
        sfmtool_core::analysis::reprojection_noise::OUTLIER_GATE,
    )?;
    m.add(
        "DEFAULT_MIN_DEPTH_FRACTION",
        sfmtool_core::analysis::infinity::DEFAULT_MIN_DEPTH_FRACTION,
    )?;
    Ok(())
}
