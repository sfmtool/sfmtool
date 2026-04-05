// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for spatial analysis, SE3 transforms, viewing angles,
//! alignment, point correspondence, and track filtering.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;

use crate::py_se3_transform::PySe3Transform;

// ── SE3 transform acceleration ────────────────────────────────────────────

/// Apply SE3 similarity transform to batched camera poses.
///
/// Takes transform parameters and camera pose arrays, returns transformed poses.
#[pyfunction]
pub fn apply_se3_to_camera_poses_py(
    py: Python<'_>,
    rotation_wxyz: PyReadonlyArray1<f64>,
    translation: PyReadonlyArray1<f64>,
    scale: f64,
    quaternions_wxyz: PyReadonlyArray2<f64>,
    translations_xyz: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let rot_data = to_contiguous!(rotation_wxyz);
    let trans_data = to_contiguous!(translation);
    let q_data = to_contiguous!(quaternions_wxyz);
    let t_data = to_contiguous!(translations_xyz);

    let n = quaternions_wxyz.shape()[0];
    let rot_wxyz: [f64; 4] = rot_data
        .as_ref()
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("rotation must have 4 elements"))?;
    let trans_xyz: [f64; 3] = trans_data
        .as_ref()
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("translation must have 3 elements"))?;

    let mut out_quats = vec![0.0f64; n * 4];
    let mut out_trans = vec![0.0f64; n * 3];

    sfmtool_core::transform::apply_se3_to_camera_poses(
        rot_wxyz,
        trans_xyz,
        scale,
        &q_data,
        &t_data,
        &mut out_quats,
        &mut out_trans,
    );

    // Convert to numpy arrays
    let out_q = numpy::PyArray2::from_vec2(
        py,
        &out_quats.chunks(4).map(|c| c.to_vec()).collect::<Vec<_>>(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let out_t = numpy::PyArray2::from_vec2(
        py,
        &out_trans.chunks(3).map(|c| c.to_vec()).collect::<Vec<_>>(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok((out_q.into_any().unbind(), out_t.into_any().unbind()))
}

// ── Viewing angle analysis ────────────────────────────────────────────────

/// Compute which points to keep based on minimum viewing angle threshold.
///
/// Returns a 1D boolean numpy array of length num_points.
#[pyfunction]
pub fn compute_narrow_track_mask(
    py: Python<'_>,
    quaternions_wxyz: PyReadonlyArray2<f64>,
    translations: PyReadonlyArray2<f64>,
    positions: PyReadonlyArray2<f64>,
    track_point_ids: PyReadonlyArray1<u32>,
    track_image_indexes: PyReadonlyArray1<u32>,
    min_angle_rad: f64,
) -> PyResult<Py<PyAny>> {
    let num_images = quaternions_wxyz.shape()[0];
    let num_points = positions.shape()[0];

    let q_data = to_contiguous!(quaternions_wxyz);
    let t_data = to_contiguous!(translations);
    let pos_data = to_contiguous!(positions);
    let point_ids_data = to_contiguous!(track_point_ids);
    let img_idx_data = to_contiguous!(track_image_indexes);

    let keep = sfmtool_core::viewing_angle::compute_narrow_track_mask(
        &q_data,
        &t_data,
        num_images,
        &pos_data,
        num_points,
        &point_ids_data,
        &img_idx_data,
        min_angle_rad,
    );

    // Convert Vec<bool> to numpy bool array
    let arr = numpy::PyArray1::from_vec(py, keep);
    Ok(arr.into_any().unbind())
}

// ── Alignment (Kabsch + RANSAC) ───────────────────────────────────────────

/// Kabsch algorithm for optimal rotation, translation, and scale.
///
/// Returns an SE3Transform representing the optimal similarity transform.
#[pyfunction]
pub fn kabsch_algorithm_rs(
    source_points: PyReadonlyArray2<f64>,
    target_points: PyReadonlyArray2<f64>,
) -> PyResult<PySe3Transform> {
    let n_points = source_points.shape()[0];
    let src_data = to_contiguous!(source_points);
    let tgt_data = to_contiguous!(target_points);

    let transform = sfmtool_core::alignment::kabsch_algorithm(&src_data, &tgt_data, n_points)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(PySe3Transform { inner: transform })
}

/// RANSAC outlier rejection for point alignment.
///
/// Returns boolean mask (N,).
#[pyfunction]
#[pyo3(signature = (source_points, target_points, max_iterations=1000, threshold=0.1, min_sample_size=3, seed=42))]
pub fn ransac_alignment_rs(
    py: Python<'_>,
    source_points: PyReadonlyArray2<f64>,
    target_points: PyReadonlyArray2<f64>,
    max_iterations: usize,
    threshold: f64,
    min_sample_size: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let n_points = source_points.shape()[0];
    let src_data = to_contiguous!(source_points);
    let tgt_data = to_contiguous!(target_points);

    let mask = sfmtool_core::alignment::ransac_alignment(
        &src_data,
        &tgt_data,
        n_points,
        max_iterations,
        threshold,
        min_sample_size,
        seed,
    );

    let arr = numpy::PyArray1::from_vec(py, mask);
    Ok(arr.into_any().unbind())
}

// ── Point correspondence ──────────────────────────────────────────────────

/// Find corresponding 3D points between two reconstructions via shared features.
///
/// Returns (source_ids, target_ids) as two 1D uint32 numpy arrays.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn find_point_correspondences_py(
    py: Python<'_>,
    source_track_image_indexes: PyReadonlyArray1<u32>,
    source_track_feature_indexes: PyReadonlyArray1<u32>,
    source_track_point_ids: PyReadonlyArray1<u32>,
    target_track_image_indexes: PyReadonlyArray1<u32>,
    target_track_feature_indexes: PyReadonlyArray1<u32>,
    target_track_point_ids: PyReadonlyArray1<u32>,
    shared_images_source: PyReadonlyArray1<u32>,
    shared_images_target: PyReadonlyArray1<u32>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let src_img = to_contiguous!(source_track_image_indexes);
    let src_feat = to_contiguous!(source_track_feature_indexes);
    let src_pts = to_contiguous!(source_track_point_ids);
    let tgt_img = to_contiguous!(target_track_image_indexes);
    let tgt_feat = to_contiguous!(target_track_feature_indexes);
    let tgt_pts = to_contiguous!(target_track_point_ids);
    let shared_src = to_contiguous!(shared_images_source);
    let shared_tgt = to_contiguous!(shared_images_target);

    let result = sfmtool_core::point_correspondence::find_point_correspondences(
        &src_img,
        &src_feat,
        &src_pts,
        &tgt_img,
        &tgt_feat,
        &tgt_pts,
        &shared_src,
        &shared_tgt,
    );

    let source_ids = numpy::PyArray1::from_vec(py, result.source_ids);
    let target_ids = numpy::PyArray1::from_vec(py, result.target_ids);

    Ok((
        source_ids.into_any().unbind(),
        target_ids.into_any().unbind(),
    ))
}

// ── Track Filtering ─────────────────────────────────────────────────────

/// Filter tracks by point mask and remap point IDs.
///
/// Returns (track_image_indexes, track_feature_indexes, track_point_ids) as 1D uint32 arrays.
#[pyfunction]
pub fn filter_tracks_by_point_mask_py(
    py: Python<'_>,
    points_to_keep_mask: PyReadonlyArray1<bool>,
    track_image_indexes: PyReadonlyArray1<u32>,
    track_feature_indexes: PyReadonlyArray1<u32>,
    track_point_ids: PyReadonlyArray1<u32>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let mask_data = to_contiguous!(points_to_keep_mask);
    let img_data = to_contiguous!(track_image_indexes);
    let feat_data = to_contiguous!(track_feature_indexes);
    let point_data = to_contiguous!(track_point_ids);

    let result = sfmtool_core::filter::filter_tracks_by_point_mask(
        &mask_data,
        &img_data,
        &feat_data,
        &point_data,
    );

    Ok((
        numpy::PyArray1::from_vec(py, result.track_image_indexes)
            .into_any()
            .unbind(),
        numpy::PyArray1::from_vec(py, result.track_feature_indexes)
            .into_any()
            .unbind(),
        numpy::PyArray1::from_vec(py, result.track_point_ids)
            .into_any()
            .unbind(),
    ))
}