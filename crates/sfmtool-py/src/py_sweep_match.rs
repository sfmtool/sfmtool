// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for sweep-based feature matching algorithms.

use nalgebra::{Matrix3, Vector3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;

use sfmtool_core::feature_match;

/// One-way sweep match from sorted keypoints 1 to sorted keypoints 2.
///
/// Both keypoint arrays must already be sorted by Y coordinate.
/// Returns list of (sorted_idx1, sorted_idx2, distance) tuples.
#[pyfunction]
#[pyo3(signature = (sorted_kpts1, sorted_descs1, sorted_kpts2, sorted_descs2, window_size, threshold=None))]
pub fn match_one_way_sweep_py(
    sorted_kpts1: PyReadonlyArray2<f64>,
    sorted_descs1: PyReadonlyArray2<u8>,
    sorted_kpts2: PyReadonlyArray2<f64>,
    sorted_descs2: PyReadonlyArray2<u8>,
    window_size: usize,
    threshold: Option<f64>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let k1_data = to_contiguous!(sorted_kpts1);
    let d1_data = to_contiguous!(sorted_descs1);
    let k2_data = to_contiguous!(sorted_kpts2);
    let d2_data = to_contiguous!(sorted_descs2);
    let n1 = sorted_kpts1.shape()[0];
    let n2 = sorted_kpts2.shape()[0];

    let matches = feature_match::match_one_way_sweep(
        &k1_data,
        &d1_data,
        n1,
        &k2_data,
        &d2_data,
        n2,
        window_size,
        threshold,
    );

    Ok(matches
        .into_iter()
        .map(|(idx1, (idx2, dist))| (idx1, idx2, dist))
        .collect())
}

/// One-way sweep match with geometric filtering on pre-sorted features.
///
/// Both keypoint arrays must already be sorted by Y coordinate.
/// Returns list of (sorted_idx1, sorted_idx2, distance) tuples.
#[pyfunction]
#[pyo3(signature = (sorted_kpts1, sorted_descs1, sorted_kpts2, sorted_descs2,
                     sorted_affines1, sorted_affines2,
                     k1, k2, r1, r2, t1, t2, window_size, threshold=None,
                     max_angle_diff=15.0, min_tri_angle=5.0, size_ratio_min=0.8, size_ratio_max=1.25))]
#[allow(clippy::too_many_arguments)]
pub fn match_one_way_sweep_geometric_py(
    sorted_kpts1: PyReadonlyArray2<f64>,
    sorted_descs1: PyReadonlyArray2<u8>,
    sorted_kpts2: PyReadonlyArray2<f64>,
    sorted_descs2: PyReadonlyArray2<u8>,
    sorted_affines1: PyReadonlyArray2<f64>,
    sorted_affines2: PyReadonlyArray2<f64>,
    k1: PyReadonlyArray2<f64>,
    k2: PyReadonlyArray2<f64>,
    r1: PyReadonlyArray2<f64>,
    r2: PyReadonlyArray2<f64>,
    t1: PyReadonlyArray1<f64>,
    t2: PyReadonlyArray1<f64>,
    window_size: usize,
    threshold: Option<f64>,
    max_angle_diff: f64,
    min_tri_angle: f64,
    size_ratio_min: f64,
    size_ratio_max: f64,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let k1_data = to_contiguous!(k1);
    let k2_data = to_contiguous!(k2);
    let r1_data = to_contiguous!(r1);
    let r2_data = to_contiguous!(r2);
    let t1_data = to_contiguous!(t1);
    let t2_data = to_contiguous!(t2);

    let k1_mat = Matrix3::from_row_slice(&k1_data);
    let k2_mat = Matrix3::from_row_slice(&k2_data);
    let r1_mat = Matrix3::from_row_slice(&r1_data);
    let r2_mat = Matrix3::from_row_slice(&r2_data);
    let t1_vec = Vector3::from_row_slice(&t1_data);
    let t2_vec = Vector3::from_row_slice(&t2_data);

    let geom = feature_match::StereoPairGeometry::new(
        &k1_mat, &k2_mat, &r1_mat, &r2_mat, &t1_vec, &t2_vec,
    );
    let config = feature_match::GeometricFilterConfig {
        max_angle_difference: max_angle_diff,
        min_triangulation_angle: min_tri_angle,
        geometric_size_ratio_min: size_ratio_min,
        geometric_size_ratio_max: size_ratio_max,
    };

    let kp1_data = to_contiguous!(sorted_kpts1);
    let d1_data = to_contiguous!(sorted_descs1);
    let kp2_data = to_contiguous!(sorted_kpts2);
    let d2_data = to_contiguous!(sorted_descs2);
    let a1_data = to_contiguous!(sorted_affines1);
    let a2_data = to_contiguous!(sorted_affines2);

    let n1 = sorted_kpts1.shape()[0];
    let n2 = sorted_kpts2.shape()[0];

    let matches = feature_match::match_one_way_sweep_geometric(
        &kp1_data,
        &d1_data,
        n1,
        &kp2_data,
        &d2_data,
        n2,
        &a1_data,
        &a2_data,
        window_size,
        threshold,
        &geom,
        &config,
    );

    Ok(matches
        .into_iter()
        .map(|(idx1, (idx2, dist))| (idx1, idx2, dist))
        .collect())
}

/// Bidirectional sort-and-sweep matching with mutual best match.
///
/// Takes unsorted keypoints, sorts internally, and returns mutual matches.
/// Returns list of (orig_idx1, orig_idx2, distance) tuples.
#[pyfunction]
#[pyo3(signature = (keypoints1, descriptors1, keypoints2, descriptors2, window_size, threshold=None))]
pub fn mutual_best_match_sweep_py(
    keypoints1: PyReadonlyArray2<f64>,
    descriptors1: PyReadonlyArray2<u8>,
    keypoints2: PyReadonlyArray2<f64>,
    descriptors2: PyReadonlyArray2<u8>,
    window_size: usize,
    threshold: Option<f64>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let k1_data = to_contiguous!(keypoints1);
    let d1_data = to_contiguous!(descriptors1);
    let k2_data = to_contiguous!(keypoints2);
    let d2_data = to_contiguous!(descriptors2);
    let n1 = keypoints1.shape()[0];
    let n2 = keypoints2.shape()[0];
    let desc_len = descriptors1.shape()[1];

    Ok(feature_match::mutual_best_match_sweep(
        &k1_data,
        &d1_data,
        n1,
        &k2_data,
        &d2_data,
        n2,
        desc_len,
        window_size,
        threshold,
    ))
}

/// Bidirectional polar sweep matching for in-frame epipole cases.
///
/// Returns list of (orig_idx1, orig_idx2, distance) tuples, or None if
/// epipoles are at infinity.
#[pyfunction]
#[pyo3(signature = (positions1, descriptors1, positions2, descriptors2, f_matrix, window_size, threshold=None, min_radius=10.0))]
#[allow(clippy::too_many_arguments)]
pub fn polar_mutual_best_match_py(
    positions1: PyReadonlyArray2<f64>,
    descriptors1: PyReadonlyArray2<u8>,
    positions2: PyReadonlyArray2<f64>,
    descriptors2: PyReadonlyArray2<u8>,
    f_matrix: PyReadonlyArray2<f64>,
    window_size: usize,
    threshold: Option<f64>,
    min_radius: f64,
) -> PyResult<Option<Vec<(usize, usize, f64)>>> {
    let p1_data = to_contiguous!(positions1);
    let d1_data = to_contiguous!(descriptors1);
    let p2_data = to_contiguous!(positions2);
    let d2_data = to_contiguous!(descriptors2);
    let f_data = to_contiguous!(f_matrix);

    let n1 = positions1.shape()[0];
    let n2 = positions2.shape()[0];
    let desc_len = descriptors1.shape()[1];

    let f_arr: [f64; 9] = f_data
        .as_ref()
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("f_matrix must be 3x3"))?;

    Ok(feature_match::polar_mutual_best_match(
        &p1_data,
        &d1_data,
        n1,
        &p2_data,
        &d2_data,
        n2,
        desc_len,
        &f_arr,
        window_size,
        threshold,
        min_radius,
    ))
}

/// Bidirectional sort-and-sweep matching with geometric filtering.
///
/// Takes unsorted keypoints with affine shapes and camera geometry,
/// applies two-stage geometric filtering during matching.
/// Returns list of (orig_idx1, orig_idx2, distance) tuples.
#[pyfunction]
#[pyo3(signature = (keypoints1, descriptors1, keypoints2, descriptors2, affines1, affines2,
                     k1, k2, r1, r2, t1, t2, window_size, threshold=None,
                     max_angle_diff=15.0, min_tri_angle=5.0, size_ratio_min=0.8, size_ratio_max=1.25))]
#[allow(clippy::too_many_arguments)]
pub fn mutual_best_match_sweep_geometric_py(
    keypoints1: PyReadonlyArray2<f64>,
    descriptors1: PyReadonlyArray2<u8>,
    keypoints2: PyReadonlyArray2<f64>,
    descriptors2: PyReadonlyArray2<u8>,
    affines1: PyReadonlyArray2<f64>,
    affines2: PyReadonlyArray2<f64>,
    k1: PyReadonlyArray2<f64>,
    k2: PyReadonlyArray2<f64>,
    r1: PyReadonlyArray2<f64>,
    r2: PyReadonlyArray2<f64>,
    t1: PyReadonlyArray1<f64>,
    t2: PyReadonlyArray1<f64>,
    window_size: usize,
    threshold: Option<f64>,
    max_angle_diff: f64,
    min_tri_angle: f64,
    size_ratio_min: f64,
    size_ratio_max: f64,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let k1_data = to_contiguous!(k1);
    let k2_data = to_contiguous!(k2);
    let r1_data = to_contiguous!(r1);
    let r2_data = to_contiguous!(r2);
    let t1_data = to_contiguous!(t1);
    let t2_data = to_contiguous!(t2);

    let k1_mat = Matrix3::from_row_slice(&k1_data);
    let k2_mat = Matrix3::from_row_slice(&k2_data);
    let r1_mat = Matrix3::from_row_slice(&r1_data);
    let r2_mat = Matrix3::from_row_slice(&r2_data);
    let t1_vec = Vector3::from_row_slice(&t1_data);
    let t2_vec = Vector3::from_row_slice(&t2_data);

    let geom = feature_match::StereoPairGeometry::new(
        &k1_mat, &k2_mat, &r1_mat, &r2_mat, &t1_vec, &t2_vec,
    );
    let config = feature_match::GeometricFilterConfig {
        max_angle_difference: max_angle_diff,
        min_triangulation_angle: min_tri_angle,
        geometric_size_ratio_min: size_ratio_min,
        geometric_size_ratio_max: size_ratio_max,
    };

    let kp1_data = to_contiguous!(keypoints1);
    let d1_data = to_contiguous!(descriptors1);
    let kp2_data = to_contiguous!(keypoints2);
    let d2_data = to_contiguous!(descriptors2);
    let a1_data = to_contiguous!(affines1);
    let a2_data = to_contiguous!(affines2);

    let n1 = keypoints1.shape()[0];
    let n2 = keypoints2.shape()[0];
    let desc_len = descriptors1.shape()[1];

    Ok(feature_match::mutual_best_match_sweep_geometric(
        &kp1_data,
        &d1_data,
        n1,
        &kp2_data,
        &d2_data,
        n2,
        &a1_data,
        &a2_data,
        desc_len,
        window_size,
        threshold,
        &geom,
        &config,
    ))
}

/// Bidirectional polar sweep matching with geometric filtering.
///
/// Returns list of (orig_idx1, orig_idx2, distance) tuples, or None if
/// epipoles are at infinity.
#[pyfunction]
#[pyo3(signature = (positions1, descriptors1, positions2, descriptors2, affines1, affines2,
                     f_matrix, k1, k2, r1, r2, t1, t2, window_size, threshold=None, min_radius=10.0,
                     max_angle_diff=15.0, min_tri_angle=5.0, size_ratio_min=0.8, size_ratio_max=1.25))]
#[allow(clippy::too_many_arguments)]
pub fn polar_mutual_best_match_geometric_py(
    positions1: PyReadonlyArray2<f64>,
    descriptors1: PyReadonlyArray2<u8>,
    positions2: PyReadonlyArray2<f64>,
    descriptors2: PyReadonlyArray2<u8>,
    affines1: PyReadonlyArray2<f64>,
    affines2: PyReadonlyArray2<f64>,
    f_matrix: PyReadonlyArray2<f64>,
    k1: PyReadonlyArray2<f64>,
    k2: PyReadonlyArray2<f64>,
    r1: PyReadonlyArray2<f64>,
    r2: PyReadonlyArray2<f64>,
    t1: PyReadonlyArray1<f64>,
    t2: PyReadonlyArray1<f64>,
    window_size: usize,
    threshold: Option<f64>,
    min_radius: f64,
    max_angle_diff: f64,
    min_tri_angle: f64,
    size_ratio_min: f64,
    size_ratio_max: f64,
) -> PyResult<Option<Vec<(usize, usize, f64)>>> {
    let k1_data = to_contiguous!(k1);
    let k2_data = to_contiguous!(k2);
    let r1_data = to_contiguous!(r1);
    let r2_data = to_contiguous!(r2);
    let t1_data = to_contiguous!(t1);
    let t2_data = to_contiguous!(t2);
    let f_data = to_contiguous!(f_matrix);

    let k1_mat = Matrix3::from_row_slice(&k1_data);
    let k2_mat = Matrix3::from_row_slice(&k2_data);
    let r1_mat = Matrix3::from_row_slice(&r1_data);
    let r2_mat = Matrix3::from_row_slice(&r2_data);
    let t1_vec = Vector3::from_row_slice(&t1_data);
    let t2_vec = Vector3::from_row_slice(&t2_data);

    let f_arr: [f64; 9] = f_data
        .as_ref()
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("f_matrix must be 3x3"))?;

    let geom = feature_match::StereoPairGeometry::new(
        &k1_mat, &k2_mat, &r1_mat, &r2_mat, &t1_vec, &t2_vec,
    );
    let config = feature_match::GeometricFilterConfig {
        max_angle_difference: max_angle_diff,
        min_triangulation_angle: min_tri_angle,
        geometric_size_ratio_min: size_ratio_min,
        geometric_size_ratio_max: size_ratio_max,
    };

    let p1_data = to_contiguous!(positions1);
    let d1_data = to_contiguous!(descriptors1);
    let p2_data = to_contiguous!(positions2);
    let d2_data = to_contiguous!(descriptors2);
    let a1_data = to_contiguous!(affines1);
    let a2_data = to_contiguous!(affines2);

    let n1 = positions1.shape()[0];
    let n2 = positions2.shape()[0];
    let desc_len = descriptors1.shape()[1];

    Ok(feature_match::polar_mutual_best_match_geometric(
        &p1_data,
        &d1_data,
        n1,
        &p2_data,
        &d2_data,
        n2,
        &a1_data,
        &a2_data,
        desc_len,
        &f_arr,
        window_size,
        threshold,
        min_radius,
        &geom,
        &config,
    ))
}