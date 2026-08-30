// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for point estimation: re-reading every track from its own
//! observations at one geometry, with the per-track rules held as options.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::reconstruction::point_estimation::{
    estimate_points_from_observations, estimate_points_from_rays, FewObservations, ObservationSet,
    PointRules, PointVerdict, RaySet,
};

use crate::geometry::PyCameraIntrinsics;

/// The verdict code table, as a dict from name to code.
fn verdict_codes(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("finite", PointVerdict::Finite.code())?;
    d.set_item("marked", PointVerdict::Marked.code())?;
    d.set_item("thin", PointVerdict::Thin.code())?;
    d.set_item("behind", PointVerdict::Behind.code())?;
    d.set_item("over_bar", PointVerdict::OverBar.code())?;
    d.set_item("few", PointVerdict::Few.code())?;
    d.set_item("finite_pruned", PointVerdict::FinitePruned.code())?;
    Ok(d)
}

/// Re-estimate every track from its own observations at one geometry.
///
/// Two input forms. Pass ``dirs``, ``centres`` and ``offsets`` for the ray form
/// (world rays and camera centres, CSR over tracks), or ``uv``, ``obs_image``,
/// ``obs_point``, ``camera``, ``quaternions_wxyz``, ``translations`` and
/// ``n_points`` for the observation form, which builds the rays through the
/// camera and each image's pose. The reprojection bar needs the observation
/// form.
///
/// Every rule has an off position and every off position is the default; with
/// all of them off the operation is the batch triangulation solve.
///
/// Args:
///     dirs: (n_obs, 3) float64 world rays, ray form.
///     centres: (n_obs, 3) float64 camera centres, ray form.
///     offsets: (n_track + 1,) int64 CSR boundaries, ray form.
///     uv: (n_obs, 2) float64 observed pixels, observation form.
///     obs_image: (n_obs,) uint32 image index, observation form.
///     obs_point: (n_obs,) uint32 track index, observation form.
///     camera: ``CameraIntrinsics`` shared by every image, observation form.
///     quaternions_wxyz: (n_img, 4) world-to-camera rotations, observation form.
///     translations: (n_img, 3) world-to-camera translations, observation form.
///     n_points: How many tracks the result indexes, observation form.
///     marks: (n_track,) bool incoming direction flags, or None for the rule
///         off. A marked track is not solved.
///     floor_rad: Angular floor in radians, or None for the rule off.
///     cheirality: Demote a point behind any observing camera (default False).
///     prune_behind: Read that demotion per observation (default False). Where
///         the observations seeing the point behind them are a strict minority,
///         they are dropped, the survivors are solved again and the rules are
///         re-read over the reduced track; the verdict is then
///         ``finite_pruned`` and the dropped rows come back in ``pruned``.
///         Needs ``cheirality``.
///     bar_px: Reprojection bound in pixels, or None for the rule off.
///     few: ``"absent"`` (default) or ``"bearing"``.
///
/// Returns:
///     A dict with ``xyzw`` (n_track, 4) float64 (w = 1 position, w = 0
///     bearing, all NaN for absent), ``verdicts`` (n_track,) uint8 (see
///     ``VERDICT_CODES``), ``in_front`` (n_track,) bool, ``pruned`` (n_obs,)
///     bool over the observations given, and ``census``, a dict of the counts
///     per verdict plus ``seen``, ``pruned_obs`` and
///     ``triangulation_angle_median_deg``.
#[pyfunction]
#[pyo3(signature = (
    *,
    dirs=None,
    centres=None,
    offsets=None,
    uv=None,
    obs_image=None,
    obs_point=None,
    camera=None,
    quaternions_wxyz=None,
    translations=None,
    n_points=None,
    marks=None,
    floor_rad=None,
    cheirality=false,
    prune_behind=false,
    bar_px=None,
    few="absent",
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_points<'py>(
    py: Python<'py>,
    dirs: Option<PyReadonlyArray2<'py, f64>>,
    centres: Option<PyReadonlyArray2<'py, f64>>,
    offsets: Option<PyReadonlyArray1<'py, i64>>,
    uv: Option<PyReadonlyArray2<'py, f64>>,
    obs_image: Option<PyReadonlyArray1<'py, u32>>,
    obs_point: Option<PyReadonlyArray1<'py, u32>>,
    camera: Option<PyRef<'_, PyCameraIntrinsics>>,
    quaternions_wxyz: Option<PyReadonlyArray2<'py, f64>>,
    translations: Option<PyReadonlyArray2<'py, f64>>,
    n_points: Option<usize>,
    marks: Option<PyReadonlyArray1<'py, bool>>,
    floor_rad: Option<f64>,
    cheirality: bool,
    prune_behind: bool,
    bar_px: Option<f64>,
    few: &str,
) -> PyResult<Py<PyAny>> {
    let rules = PointRules {
        floor_rad,
        cheirality,
        prune_behind,
        bar_px,
        few: match few {
            "absent" => FewObservations::Absent,
            "bearing" => FewObservations::Bearing,
            other => {
                return Err(PyValueError::new_err(format!(
                    "few must be 'absent' or 'bearing', got {other:?}"
                )))
            }
        },
    };
    // The prune is a reading of the cheirality rule, so asking for it with that
    // rule off is a request nothing would carry out.
    if prune_behind && !cheirality {
        return Err(PyValueError::new_err(
            "prune_behind reads the cheirality rule and needs cheirality=True",
        ));
    }
    let mask = marks.as_ref().map(|m| to_contiguous!(m));

    let ray_form = dirs.is_some() || centres.is_some() || offsets.is_some();
    let obs_form = uv.is_some() || obs_image.is_some() || obs_point.is_some();
    if ray_form && obs_form {
        return Err(PyValueError::new_err(
            "pass either the ray form (dirs, centres, offsets) or the observation form, not both",
        ));
    }

    let out = if ray_form {
        let (d, c, o) = match (dirs, centres, offsets) {
            (Some(d), Some(c), Some(o)) => (d, c, o),
            _ => {
                return Err(PyValueError::new_err(
                    "the ray form needs dirs, centres and offsets",
                ))
            }
        };
        if d.shape()[1] != 3 || c.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "dirs and centres must have shape (n_obs, 3)",
            ));
        }
        if d.shape()[0] != c.shape()[0] {
            return Err(PyValueError::new_err(
                "dirs and centres must share the same length",
            ));
        }
        if bar_px.is_some() {
            return Err(PyValueError::new_err(
                "bar_px needs the observation form: the ray form carries no pixels",
            ));
        }
        let dd = to_contiguous!(d);
        let cc = to_contiguous!(c);
        let oo = to_contiguous!(o);
        let offs = csr_offsets(&oo, d.shape()[0])?;
        if let Some(m) = &mask {
            check_marks(m.len(), offs.len().saturating_sub(1))?;
        }
        py.detach(|| {
            estimate_points_from_rays(
                RaySet {
                    dirs: &dd,
                    centres: &cc,
                    offsets: &offs,
                },
                mask.as_deref(),
                rules,
            )
        })
    } else {
        let (u, oi, op, cam, q, t) = match (
            uv,
            obs_image,
            obs_point,
            camera,
            quaternions_wxyz,
            translations,
        ) {
            (Some(u), Some(oi), Some(op), Some(cam), Some(q), Some(t)) => (u, oi, op, cam, q, t),
            _ => {
                return Err(PyValueError::new_err(
                    "the observation form needs uv, obs_image, obs_point, camera, \
                     quaternions_wxyz and translations",
                ))
            }
        };
        let n_tracks = n_points.ok_or_else(|| {
            PyValueError::new_err("the observation form needs n_points, the track count")
        })?;
        if u.shape()[1] != 2 {
            return Err(PyValueError::new_err("uv must have shape (n_obs, 2)"));
        }
        if q.shape()[1] != 4 {
            return Err(PyValueError::new_err(
                "quaternions_wxyz must have shape (n_img, 4)",
            ));
        }
        if t.shape()[1] != 3 || t.shape()[0] != q.shape()[0] {
            return Err(PyValueError::new_err(
                "translations must have shape (n_img, 3) matching quaternions_wxyz",
            ));
        }
        let n_obs = u.shape()[0];
        if oi.shape()[0] != n_obs || op.shape()[0] != n_obs {
            return Err(PyValueError::new_err(
                "uv, obs_image and obs_point must share the same length",
            ));
        }
        if let Some(m) = &mask {
            check_marks(m.len(), n_tracks)?;
        }
        let uu = to_contiguous!(u);
        let ii = to_contiguous!(oi);
        let pp = to_contiguous!(op);
        let qq = to_contiguous!(q);
        let tt = to_contiguous!(t);
        let n_img = q.shape()[0];
        // The core takes the rotations as given rather than renormalizing, so a
        // caller holding a unit quaternion gets its own rotation back; one that
        // is not unit would silently scale every ray.
        for (i, c) in qq.chunks_exact(4).enumerate() {
            let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + c[3] * c[3]).sqrt();
            if !(n - 1.0).abs().le(&1e-9) {
                return Err(PyValueError::new_err(format!(
                    "quaternions_wxyz[{i}] has norm {n}, which is not a rotation"
                )));
            }
        }
        for (k, &i) in ii.iter().enumerate() {
            if i as usize >= n_img {
                return Err(PyValueError::new_err(format!(
                    "obs_image[{k}] = {i} is past the {n_img} images given"
                )));
            }
        }
        for (k, &p) in pp.iter().enumerate() {
            if p as usize >= n_tracks {
                return Err(PyValueError::new_err(format!(
                    "obs_point[{k}] = {p} is past the {n_tracks} points given"
                )));
            }
        }
        let inner = cam.inner.clone();
        py.detach(move || {
            estimate_points_from_observations(
                &inner,
                ObservationSet {
                    uv: &uu,
                    obs_image: &ii,
                    obs_point: &pp,
                    quats_wxyz: &qq,
                    translations: &tt,
                    n_tracks,
                },
                mask.as_deref(),
                rules,
            )
        })
    };

    let n = out.xyzw.len();
    let flat: Vec<f64> = out.xyzw.into_iter().flatten().collect();
    let xyzw = PyArray1::from_vec(py, flat)
        .reshape([n, 4])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("xyzw", xyzw)?;
    dict.set_item(
        "verdicts",
        PyArray1::from_vec(
            py,
            out.verdicts.iter().map(|v| v.code()).collect::<Vec<u8>>(),
        ),
    )?;
    dict.set_item("in_front", PyArray1::from_vec(py, out.in_front))?;
    dict.set_item("pruned", PyArray1::from_vec(py, out.pruned))?;
    let c = out.census;
    let census = PyDict::new(py);
    census.set_item("seen", c.seen)?;
    census.set_item("finite", c.finite)?;
    census.set_item("marked", c.marked)?;
    census.set_item("thin", c.thin)?;
    census.set_item("behind", c.behind)?;
    census.set_item("over_bar", c.over_bar)?;
    census.set_item("few", c.few)?;
    census.set_item("finite_pruned", c.finite_pruned)?;
    census.set_item("pruned_obs", c.pruned_obs)?;
    census.set_item(
        "triangulation_angle_median_deg",
        c.triangulation_angle_median_deg,
    )?;
    dict.set_item("census", census)?;
    Ok(dict.into_any().unbind())
}

/// Validate CSR offsets against the ray count.
fn csr_offsets(raw: &[i64], n_obs: usize) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(raw.len());
    let mut prev = 0i64;
    for (k, &o) in raw.iter().enumerate() {
        if o < 0 {
            return Err(PyValueError::new_err(format!(
                "offsets must be non-negative, got {o} at index {k}"
            )));
        }
        if o < prev {
            return Err(PyValueError::new_err(format!(
                "offsets must be non-decreasing, got {o} after {prev} at index {k}"
            )));
        }
        prev = o;
        out.push(o as usize);
    }
    if prev as usize > n_obs {
        return Err(PyValueError::new_err(format!(
            "offsets[-1] = {prev} exceeds the number of rays {n_obs}"
        )));
    }
    Ok(out)
}

/// One mark per track, or none at all.
fn check_marks(given: usize, want: usize) -> PyResult<()> {
    if given != want {
        return Err(PyValueError::new_err(format!(
            "marks must have one entry per track: {given} given for {want} tracks"
        )));
    }
    Ok(())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_points, m)?)?;
    m.add("VERDICT_CODES", verdict_codes(m.py())?)?;
    Ok(())
}
