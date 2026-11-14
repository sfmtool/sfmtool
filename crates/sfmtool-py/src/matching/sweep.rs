// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for sweep-based feature matching algorithms.

use nalgebra::{Matrix3, Vector3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::features::feature_match;

use super::image::check_shape;
use crate::helpers::value_err;

/// Checks the keypoint and descriptor arrays of one sweep call against each
/// other, with the binding's argument names, and returns `(n1, n2,
/// desc_len)`: the row counts of the two keypoint arrays and the width of the
/// first descriptor array.
fn feature_counts(
    names: [&str; 4],
    kpts1: &PyReadonlyArray2<f64>,
    descs1: &PyReadonlyArray2<u8>,
    kpts2: &PyReadonlyArray2<f64>,
    descs2: &PyReadonlyArray2<u8>,
) -> PyResult<(usize, usize, usize)> {
    let n1 = kpts1.shape()[0];
    let n2 = kpts2.shape()[0];
    let desc_len = descs1.shape()[1];
    check_shape(names[0], kpts1.shape(), &[n1, 2])?;
    check_shape(names[1], descs1.shape(), &[n1, desc_len])?;
    check_shape(names[2], kpts2.shape(), &[n2, 2])?;
    check_shape(names[3], descs2.shape(), &[n2, desc_len])?;
    Ok((n1, n2, desc_len))
}

/// Checks the two affine-shape arrays of a geometric sweep, which must be
/// `(n1, 4)` and `(n2, 4)`.
fn check_affines(
    names: [&str; 2],
    affines1: &PyReadonlyArray2<f64>,
    affines2: &PyReadonlyArray2<f64>,
    n1: usize,
    n2: usize,
) -> PyResult<()> {
    check_shape(names[0], affines1.shape(), &[n1, 4])?;
    check_shape(names[1], affines2.shape(), &[n2, 4])
}

/// The camera geometry of a geometric sweep, from `k1`, `k2`, `r1` and `r2`,
/// which must be of shape (3, 3), and `t1` and `t2`, which must be of
/// shape (3,).
fn stereo_geometry(
    k1: &PyReadonlyArray2<f64>,
    k2: &PyReadonlyArray2<f64>,
    r1: &PyReadonlyArray2<f64>,
    r2: &PyReadonlyArray2<f64>,
    t1: &PyReadonlyArray1<f64>,
    t2: &PyReadonlyArray1<f64>,
) -> PyResult<feature_match::StereoPairGeometry> {
    for (name, m) in [("k1", k1), ("k2", k2), ("r1", r1), ("r2", r2)] {
        check_shape(name, m.shape(), &[3, 3])?;
    }
    check_shape("t1", t1.shape(), &[3])?;
    check_shape("t2", t2.shape(), &[3])?;
    let matrix = |m: &PyReadonlyArray2<f64>| Matrix3::from_row_slice(&to_contiguous!(m));
    let vector = |v: &PyReadonlyArray1<f64>| Vector3::from_row_slice(&to_contiguous!(v));
    Ok(feature_match::StereoPairGeometry::new(
        &matrix(k1),
        &matrix(k2),
        &matrix(r1),
        &matrix(r2),
        &vector(t1),
        &vector(t2),
    ))
}

/// The fundamental matrix of a polar sweep, which must be of shape (3, 3), as
/// its nine entries in row-major order.
fn fundamental_matrix(f_matrix: &PyReadonlyArray2<f64>) -> PyResult<[f64; 9]> {
    check_shape("f_matrix", f_matrix.shape(), &[3, 3])?;
    let f_data = to_contiguous!(f_matrix);
    let mut f_arr = [0.0; 9];
    f_arr.copy_from_slice(&f_data);
    Ok(f_arr)
}

/// One-way sweep match from sorted keypoints 1 to sorted keypoints 2.
///
/// Both keypoint arrays must already be sorted by Y coordinate.
/// Returns list of (sorted_idx1, sorted_idx2, distance) tuples.
///
/// Raises:
///     ValueError: A keypoint array is not of shape (N, 2), or a descriptor
///         array does not have one row per keypoint or is not as wide as
///         ``sorted_descs1``.
#[pyfunction]
#[pyo3(name = "match_one_way_sweep")]
#[pyo3(signature = (sorted_kpts1, sorted_descs1, sorted_kpts2, sorted_descs2, window_size, threshold=None))]
pub fn match_one_way_sweep_py(
    sorted_kpts1: PyReadonlyArray2<f64>,
    sorted_descs1: PyReadonlyArray2<u8>,
    sorted_kpts2: PyReadonlyArray2<f64>,
    sorted_descs2: PyReadonlyArray2<u8>,
    window_size: usize,
    threshold: Option<f64>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let (n1, n2, desc_len) = feature_counts(
        [
            "sorted_kpts1",
            "sorted_descs1",
            "sorted_kpts2",
            "sorted_descs2",
        ],
        &sorted_kpts1,
        &sorted_descs1,
        &sorted_kpts2,
        &sorted_descs2,
    )?;
    let k1_data = to_contiguous!(sorted_kpts1);
    let d1_data = to_contiguous!(sorted_descs1);
    let k2_data = to_contiguous!(sorted_kpts2);
    let d2_data = to_contiguous!(sorted_descs2);

    let matches = feature_match::match_one_way_sweep(
        &k1_data,
        &d1_data,
        n1,
        &k2_data,
        &d2_data,
        n2,
        desc_len,
        window_size,
        threshold,
    )
    .map_err(value_err)?;

    Ok(matches
        .into_iter()
        .map(|(idx1, (idx2, dist))| (idx1, idx2, dist))
        .collect())
}

/// One-way sweep match with geometric filtering on pre-sorted features.
///
/// Both keypoint arrays must already be sorted by Y coordinate.
/// Returns list of (sorted_idx1, sorted_idx2, distance) tuples.
///
/// Raises:
///     ValueError: ``k1``, ``k2``, ``r1`` or ``r2`` is not of shape (3, 3),
///         ``t1`` or ``t2`` is not of shape (3,), a keypoint array is not of
///         shape (N, 2), a descriptor array does not have one row per keypoint
///         or is not as wide as ``sorted_descs1``, or an affine-shape array is
///         not of shape (N, 4) for its image's N.
#[pyfunction]
#[pyo3(name = "match_one_way_sweep_geometric")]
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
    let geom = stereo_geometry(&k1, &k2, &r1, &r2, &t1, &t2)?;
    let (n1, n2, desc_len) = feature_counts(
        [
            "sorted_kpts1",
            "sorted_descs1",
            "sorted_kpts2",
            "sorted_descs2",
        ],
        &sorted_kpts1,
        &sorted_descs1,
        &sorted_kpts2,
        &sorted_descs2,
    )?;
    check_affines(
        ["sorted_affines1", "sorted_affines2"],
        &sorted_affines1,
        &sorted_affines2,
        n1,
        n2,
    )?;
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

    let matches = feature_match::match_one_way_sweep_geometric(
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
    )
    .map_err(value_err)?;

    Ok(matches
        .into_iter()
        .map(|(idx1, (idx2, dist))| (idx1, idx2, dist))
        .collect())
}

/// Bidirectional sort-and-sweep matching with mutual best match.
///
/// Takes unsorted keypoints, sorts internally, and returns mutual matches.
/// Returns list of (orig_idx1, orig_idx2, distance) tuples.
///
/// Raises:
///     ValueError: A keypoint array is not of shape (N, 2), or a descriptor
///         array does not have one row per keypoint or is not as wide as
///         ``descriptors1``.
#[pyfunction]
#[pyo3(name = "mutual_best_match_sweep")]
#[pyo3(signature = (keypoints1, descriptors1, keypoints2, descriptors2, window_size, threshold=None))]
pub fn mutual_best_match_sweep_py(
    keypoints1: PyReadonlyArray2<f64>,
    descriptors1: PyReadonlyArray2<u8>,
    keypoints2: PyReadonlyArray2<f64>,
    descriptors2: PyReadonlyArray2<u8>,
    window_size: usize,
    threshold: Option<f64>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let (n1, n2, desc_len) = feature_counts(
        ["keypoints1", "descriptors1", "keypoints2", "descriptors2"],
        &keypoints1,
        &descriptors1,
        &keypoints2,
        &descriptors2,
    )?;
    let k1_data = to_contiguous!(keypoints1);
    let d1_data = to_contiguous!(descriptors1);
    let k2_data = to_contiguous!(keypoints2);
    let d2_data = to_contiguous!(descriptors2);

    feature_match::mutual_best_match_sweep(
        &k1_data,
        &d1_data,
        n1,
        &k2_data,
        &d2_data,
        n2,
        desc_len,
        window_size,
        threshold,
    )
    .map_err(value_err)
}

/// Bidirectional polar sweep matching for in-frame epipole cases.
///
/// Returns list of (orig_idx1, orig_idx2, distance) tuples, or None if
/// epipoles are at infinity.
///
/// Raises:
///     ValueError: ``f_matrix`` is not of shape (3, 3), a positions array is
///         not of shape (N, 2), or a descriptor array does not have one row
///         per position or is not as wide as ``descriptors1``.
#[pyfunction]
#[pyo3(name = "polar_mutual_best_match")]
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
    let f_arr = fundamental_matrix(&f_matrix)?;
    let (n1, n2, desc_len) = feature_counts(
        ["positions1", "descriptors1", "positions2", "descriptors2"],
        &positions1,
        &descriptors1,
        &positions2,
        &descriptors2,
    )?;
    let p1_data = to_contiguous!(positions1);
    let d1_data = to_contiguous!(descriptors1);
    let p2_data = to_contiguous!(positions2);
    let d2_data = to_contiguous!(descriptors2);

    feature_match::polar_mutual_best_match(
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
    )
    .map_err(value_err)
}

/// Bidirectional sort-and-sweep matching with geometric filtering.
///
/// Takes unsorted keypoints with affine shapes and camera geometry,
/// applies two-stage geometric filtering during matching.
/// Returns list of (orig_idx1, orig_idx2, distance) tuples.
///
/// Raises:
///     ValueError: ``k1``, ``k2``, ``r1`` or ``r2`` is not of shape (3, 3),
///         ``t1`` or ``t2`` is not of shape (3,), a keypoint array is not of
///         shape (N, 2), a descriptor array does not have one row per keypoint
///         or is not as wide as ``descriptors1``, or an affine-shape array is
///         not of shape (N, 4) for its image's N.
#[pyfunction]
#[pyo3(name = "mutual_best_match_sweep_geometric")]
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
    let geom = stereo_geometry(&k1, &k2, &r1, &r2, &t1, &t2)?;
    let (n1, n2, desc_len) = feature_counts(
        ["keypoints1", "descriptors1", "keypoints2", "descriptors2"],
        &keypoints1,
        &descriptors1,
        &keypoints2,
        &descriptors2,
    )?;
    check_affines(["affines1", "affines2"], &affines1, &affines2, n1, n2)?;
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

    feature_match::mutual_best_match_sweep_geometric(
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
    )
    .map_err(value_err)
}

/// Bidirectional polar sweep matching with geometric filtering.
///
/// Returns list of (orig_idx1, orig_idx2, distance) tuples, or None if
/// epipoles are at infinity.
///
/// Raises:
///     ValueError: ``f_matrix``, ``k1``, ``k2``, ``r1`` or ``r2`` is not of
///         shape (3, 3), ``t1`` or ``t2`` is not of shape (3,), a positions
///         array is not of shape (N, 2), a descriptor array does not have one
///         row per position or is not as wide as ``descriptors1``, or an
///         affine-shape array is not of shape (N, 4) for its image's N.
#[pyfunction]
#[pyo3(name = "polar_mutual_best_match_geometric")]
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
    let f_arr = fundamental_matrix(&f_matrix)?;
    let geom = stereo_geometry(&k1, &k2, &r1, &r2, &t1, &t2)?;
    let (n1, n2, desc_len) = feature_counts(
        ["positions1", "descriptors1", "positions2", "descriptors2"],
        &positions1,
        &descriptors1,
        &positions2,
        &descriptors2,
    )?;
    check_affines(["affines1", "affines2"], &affines1, &affines2, n1, n2)?;
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

    feature_match::polar_mutual_best_match_geometric(
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
    )
    .map_err(value_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(match_one_way_sweep_py, m)?)?;
    m.add_function(wrap_pyfunction!(match_one_way_sweep_geometric_py, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_best_match_sweep_py, m)?)?;
    m.add_function(wrap_pyfunction!(polar_mutual_best_match_py, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_best_match_sweep_geometric_py, m)?)?;
    m.add_function(wrap_pyfunction!(polar_mutual_best_match_geometric_py, m)?)?;
    Ok(())
}
