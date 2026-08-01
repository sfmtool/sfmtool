// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the observation adjacency graph.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::analysis::observation_adjacency::{
    build_observation_adjacency as core_build_observation_adjacency, ObservationAdjacencyParams,
};

/// Build the point-level adjacency graph implied by how close the points'
/// observations sit in the images that see both.
///
/// Two points are adjacent when their keypoint separation lands in the annulus
/// ``[a_lo, b_max]`` — in units of the pair radius ``min(radius_p, radius_q)``
/// — in a ``majority`` of the images observing both, the pair shares at least
/// ``min_shared_images`` images, and the median relative difference of their
/// ranges from those cameras is at most ``range_tol``. Set ``a_lo = 0`` to
/// admit coincident observations (duplicate collapse) or ``range_tol = inf`` to
/// disable the range vet (aliased-pair inspection).
///
/// Args:
///     keypoints_xy: Per observation, shape ``(N, 2)`` float64.
///     track_point_indexes: Per observation, shape ``(N,)`` uint32.
///     track_image_indexes: Per observation, shape ``(N,)`` uint32.
///     radii_px: Per point, shape ``(P,)`` float32; ``<= 0`` excludes the point.
///     point_is_at_infinity: Per point, shape ``(P,)`` bool.
///     positions: Per point, shape ``(P, 3)`` float64.
///     quaternions_wxyz: Per image, shape ``(I, 4)`` float64 (world-to-camera).
///     translations: Per image, shape ``(I, 3)`` float64. Camera centers are
///         derived as ``-Rᵀ·t``.
///     b_max: Annulus outer edge in pair-radius units (keyword-only, required).
///     a_lo: Annulus inner edge in pair-radius units.
///     min_shared_images: Support floor.
///     majority: Required fraction of shared images landing in the annulus.
///     range_tol: Median relative range gate; ``inf`` disables it.
///
/// Returns:
///     A dict of symmetric-CSR arrays: ``offsets`` ``(P + 1,)`` uint32,
///     and, per directed edge, ``neighbours`` uint32, ``separation_med`` /
///     ``separation_min`` / ``separation_max`` float32, ``shared_images`` /
///     ``annulus_hits`` uint32, ``range_mismatch`` float32. Each point's
///     neighbours are sorted by ``(separation_med, neighbour index)``.
#[pyfunction]
#[pyo3(signature = (
    keypoints_xy,
    track_point_indexes,
    track_image_indexes,
    radii_px,
    point_is_at_infinity,
    positions,
    quaternions_wxyz,
    translations,
    *,
    b_max,
    a_lo=1.0,
    min_shared_images=2,
    majority=0.5,
    range_tol=0.05,
))]
#[allow(clippy::too_many_arguments)]
pub fn build_observation_adjacency(
    py: Python<'_>,
    keypoints_xy: PyReadonlyArray2<f64>,
    track_point_indexes: PyReadonlyArray1<u32>,
    track_image_indexes: PyReadonlyArray1<u32>,
    radii_px: PyReadonlyArray1<f32>,
    point_is_at_infinity: PyReadonlyArray1<bool>,
    positions: PyReadonlyArray2<f64>,
    quaternions_wxyz: PyReadonlyArray2<f64>,
    translations: PyReadonlyArray2<f64>,
    b_max: f64,
    a_lo: f64,
    min_shared_images: u32,
    majority: f64,
    range_tol: f64,
) -> PyResult<Py<PyAny>> {
    let n_obs = keypoints_xy.shape()[0];
    let n_points = radii_px.shape()[0];
    let n_images = quaternions_wxyz.shape()[0];

    if keypoints_xy.shape()[1] != 2 {
        return Err(PyValueError::new_err("keypoints_xy must have shape (N, 2)"));
    }
    if positions.shape()[1] != 3 {
        return Err(PyValueError::new_err("positions must have shape (P, 3)"));
    }
    if quaternions_wxyz.shape()[1] != 4 {
        return Err(PyValueError::new_err(
            "quaternions_wxyz must have shape (I, 4)",
        ));
    }
    if translations.shape() != [n_images, 3] {
        return Err(PyValueError::new_err(
            "translations must have shape (I, 3) matching quaternions_wxyz",
        ));
    }
    for (name, len) in [
        ("track_point_indexes", track_point_indexes.shape()[0]),
        ("track_image_indexes", track_image_indexes.shape()[0]),
    ] {
        if len != n_obs {
            return Err(PyValueError::new_err(format!(
                "{name} has {len} entries but keypoints_xy has {n_obs}"
            )));
        }
    }
    for (name, len) in [
        ("point_is_at_infinity", point_is_at_infinity.shape()[0]),
        ("positions", positions.shape()[0]),
    ] {
        if len != n_points {
            return Err(PyValueError::new_err(format!(
                "{name} has {len} entries but radii_px has {n_points}"
            )));
        }
    }
    if a_lo.is_nan() || b_max.is_nan() || a_lo < 0.0 || b_max < a_lo {
        return Err(PyValueError::new_err(format!(
            "expected 0 <= a_lo <= b_max, got a_lo={a_lo}, b_max={b_max}"
        )));
    }

    let keypoints_data = to_contiguous!(keypoints_xy);
    let point_idx_data = to_contiguous!(track_point_indexes);
    let image_idx_data = to_contiguous!(track_image_indexes);
    let radii_data = to_contiguous!(radii_px);
    let infinity_data = to_contiguous!(point_is_at_infinity);
    let positions_data = to_contiguous!(positions);
    let quat_data = to_contiguous!(quaternions_wxyz);
    let trans_data = to_contiguous!(translations);

    for (name, idx, bound) in [
        ("track_point_indexes", &point_idx_data, n_points),
        ("track_image_indexes", &image_idx_data, n_images),
    ] {
        if let Some(&bad) = idx.iter().find(|&&i| i as usize >= bound) {
            return Err(PyValueError::new_err(format!(
                "{name} contains {bad}, out of range for {bound} entries"
            )));
        }
    }

    let keypoints: Vec<[f64; 2]> = keypoints_data
        .chunks_exact(2)
        .map(|c| [c[0], c[1]])
        .collect();
    let point_positions: Vec<[f64; 3]> = positions_data
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    // Camera center: C = -R_world_from_cam · t.
    let camera_centers: Vec<[f64; 3]> = (0..n_images)
        .map(|i| {
            let qo = i * 4;
            let quat = UnitQuaternion::new_normalize(Quaternion::new(
                quat_data[qo],
                quat_data[qo + 1],
                quat_data[qo + 2],
                quat_data[qo + 3],
            ));
            let r_world_from_cam = quat.to_rotation_matrix().transpose();
            let to = i * 3;
            let t = Vector3::new(trans_data[to], trans_data[to + 1], trans_data[to + 2]);
            let c = -(r_world_from_cam.matrix() * t);
            [c[0], c[1], c[2]]
        })
        .collect();

    let params = ObservationAdjacencyParams {
        a_lo,
        b_max,
        min_shared_images,
        majority,
        range_tol,
    };

    let adjacency = py.detach(|| {
        core_build_observation_adjacency(
            &keypoints,
            &point_idx_data,
            &image_idx_data,
            &radii_data,
            &infinity_data,
            &point_positions,
            &camera_centers,
            &params,
        )
    });

    let dict = PyDict::new(py);
    dict.set_item("offsets", PyArray1::from_vec(py, adjacency.offsets))?;
    dict.set_item("neighbours", PyArray1::from_vec(py, adjacency.neighbours))?;
    dict.set_item(
        "separation_med",
        PyArray1::from_vec(py, adjacency.separation_med),
    )?;
    dict.set_item(
        "separation_min",
        PyArray1::from_vec(py, adjacency.separation_min),
    )?;
    dict.set_item(
        "separation_max",
        PyArray1::from_vec(py, adjacency.separation_max),
    )?;
    dict.set_item(
        "shared_images",
        PyArray1::from_vec(py, adjacency.shared_images),
    )?;
    dict.set_item(
        "annulus_hits",
        PyArray1::from_vec(py, adjacency.annulus_hits),
    )?;
    dict.set_item(
        "range_mismatch",
        PyArray1::from_vec(py, adjacency.range_mismatch),
    )?;
    Ok(dict.into_any().unbind())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_observation_adjacency, m)?)?;
    Ok(())
}
