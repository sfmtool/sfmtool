// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for viewing angles, alignment, point correspondence, and
//! point and track merging.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::geometry::se3_transform::PySe3Transform;
use crate::helpers::value_err;
use crate::PySfmrReconstruction;

// ── Viewing angle analysis ────────────────────────────────────────────────

/// Compute which points to keep based on minimum viewing angle threshold.
///
/// ``positions_xyzw`` is ``(N, 4)``, homogeneous, as
/// ``SfmrReconstruction.positions_xyzw`` gives it. A finite point is kept when
/// the widest angle between two of its observation rays is at least
/// ``min_angle_rad``; a point at infinity (``w = 0``) is always kept, since its
/// rays are parallel by definition.
///
/// Returns a 1D boolean numpy array of length num_points.
#[pyfunction]
pub fn compute_narrow_track_mask(
    py: Python<'_>,
    quaternions_wxyz: PyReadonlyArray2<f64>,
    translations: PyReadonlyArray2<f64>,
    positions_xyzw: PyReadonlyArray2<f64>,
    track_point_indexes: PyReadonlyArray1<u32>,
    track_image_indexes: PyReadonlyArray1<u32>,
    min_angle_rad: f64,
) -> PyResult<Py<PyAny>> {
    let num_images = quaternions_wxyz.shape()[0];
    let num_points = positions_xyzw.shape()[0];

    let q_data = to_contiguous!(quaternions_wxyz);
    let t_data = to_contiguous!(translations);
    let pos_data = to_contiguous!(positions_xyzw);
    let point_ids_data = to_contiguous!(track_point_indexes);
    let img_idx_data = to_contiguous!(track_image_indexes);

    let keep = sfmtool_core::geometry::viewing_angle::compute_narrow_track_mask(
        &q_data,
        &t_data,
        num_images,
        &pos_data,
        num_points,
        &point_ids_data,
        &img_idx_data,
        min_angle_rad,
    )
    .map_err(value_err)?;

    // Convert Vec<bool> to numpy bool array
    let arr = numpy::PyArray1::from_vec(py, keep);
    Ok(arr.into_any().unbind())
}

// ── Alignment (least-squares fit + RANSAC) ─────────────────────────────────

/// Estimate the similarity-or-rigid transform aligning source points to target
/// points by least squares over corresponded points.
///
/// With the defaults this is the classical single-shot similarity fit; raising
/// ``rounds`` trims gross mismatches by iteratively refitting on the best-fitting
/// ``keep_fraction`` of correspondences.
///
/// Args:
///     source_points: (N, 3) source points.
///     target_points: (N, 3) target points.
///     rounds: Refit iterations; each after the first re-selects the inlier
///         subset. 1 (default) disables trimming.
///     keep_fraction: Fraction of correspondences kept each round after the
///         first. Ignored when ``rounds == 1``.
///     estimate_scale: Fit a similarity (True) or a rigid transform (False).
///
/// Returns:
///     An ``Se3Transform`` (scale 1.0 when ``estimate_scale`` is False).
#[pyfunction]
#[pyo3(name = "estimate_alignment")]
#[pyo3(signature = (source_points, target_points, rounds=1, keep_fraction=1.0, estimate_scale=true))]
pub fn estimate_alignment_py(
    source_points: PyReadonlyArray2<f64>,
    target_points: PyReadonlyArray2<f64>,
    rounds: usize,
    keep_fraction: f64,
    estimate_scale: bool,
) -> PyResult<PySe3Transform> {
    let n_points = source_points.shape()[0];
    let src_data = to_contiguous!(source_points);
    let tgt_data = to_contiguous!(target_points);

    let transform = sfmtool_core::analysis::alignment::estimate_alignment(
        &src_data,
        &tgt_data,
        n_points,
        sfmtool_core::analysis::alignment::AlignmentParams {
            rounds,
            keep_fraction,
            estimate_scale,
        },
    )
    .map_err(value_err)?;

    Ok(PySe3Transform { inner: transform })
}

/// RANSAC outlier rejection for point alignment.
///
/// Returns boolean mask (N,).
#[pyfunction]
#[pyo3(name = "ransac_alignment")]
#[pyo3(signature = (source_points, target_points, max_iterations=1000, threshold=0.1, min_sample_size=3, seed=42))]
pub fn ransac_alignment_py(
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

    let mask = sfmtool_core::analysis::alignment::ransac_alignment(
        &src_data,
        &tgt_data,
        n_points,
        max_iterations,
        threshold,
        min_sample_size,
        seed,
    )
    .map_err(value_err)?;

    let arr = numpy::PyArray1::from_vec(py, mask);
    Ok(arr.into_any().unbind())
}

// ── Point correspondence ──────────────────────────────────────────────────

/// Find corresponding 3D points between two reconstructions via shared features.
///
/// Returns (source_ids, target_ids) as two 1D uint32 numpy arrays.
#[pyfunction]
#[pyo3(name = "find_point_correspondences")]
#[allow(clippy::too_many_arguments)]
pub fn find_point_correspondences_py(
    py: Python<'_>,
    source_track_image_indexes: PyReadonlyArray1<u32>,
    source_track_feature_indexes: PyReadonlyArray1<u32>,
    source_track_point_indexes: PyReadonlyArray1<u32>,
    target_track_image_indexes: PyReadonlyArray1<u32>,
    target_track_feature_indexes: PyReadonlyArray1<u32>,
    target_track_point_indexes: PyReadonlyArray1<u32>,
    shared_images_source: PyReadonlyArray1<u32>,
    shared_images_target: PyReadonlyArray1<u32>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let src_img = to_contiguous!(source_track_image_indexes);
    let src_feat = to_contiguous!(source_track_feature_indexes);
    let src_pts = to_contiguous!(source_track_point_indexes);
    let tgt_img = to_contiguous!(target_track_image_indexes);
    let tgt_feat = to_contiguous!(target_track_feature_indexes);
    let tgt_pts = to_contiguous!(target_track_point_indexes);
    let shared_src = to_contiguous!(shared_images_source);
    let shared_tgt = to_contiguous!(shared_images_target);

    let result = sfmtool_core::reconstruction::point_correspondence::find_point_correspondences(
        &src_img,
        &src_feat,
        &src_pts,
        &tgt_img,
        &tgt_feat,
        &tgt_pts,
        &shared_src,
        &shared_tgt,
    )
    .map_err(value_err)?;

    let source_ids = numpy::PyArray1::from_vec(py, result.source_ids);
    let target_ids = numpy::PyArray1::from_vec(py, result.target_ids);

    Ok((
        source_ids.into_any().unbind(),
        target_ids.into_any().unbind(),
    ))
}

/// Find corresponding 3D points between two reconstructions by keypoint
/// position in shared images.
///
/// Observations of each shared image pair are matched by mutual nearest
/// keypoint within `pixel_threshold` pixels; each match votes for its
/// `(source point, target point)` pair, and the votes are resolved greedily,
/// most votes first, into a one-to-one mapping of pairs with at least
/// `min_votes` votes. `source_keypoints_xy` and `target_keypoints_xy` are
/// `(M, 2)` arrays parallel to the track columns.
///
/// Returns (source_ids, target_ids) as two 1D uint32 numpy arrays, in the
/// order the pairs were accepted. Pairs with a point at infinity are kept;
/// the caller drops them.
#[pyfunction]
#[pyo3(
    name = "find_point_correspondences_by_coordinate",
    signature = (
        source_track_image_indexes,
        source_keypoints_xy,
        source_track_point_indexes,
        target_track_image_indexes,
        target_keypoints_xy,
        target_track_point_indexes,
        shared_images_source,
        shared_images_target,
        pixel_threshold = 2.0,
        min_votes = 2,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn find_point_correspondences_by_coordinate_py(
    py: Python<'_>,
    source_track_image_indexes: PyReadonlyArray1<u32>,
    source_keypoints_xy: PyReadonlyArray2<f64>,
    source_track_point_indexes: PyReadonlyArray1<u32>,
    target_track_image_indexes: PyReadonlyArray1<u32>,
    target_keypoints_xy: PyReadonlyArray2<f64>,
    target_track_point_indexes: PyReadonlyArray1<u32>,
    shared_images_source: PyReadonlyArray1<u32>,
    shared_images_target: PyReadonlyArray1<u32>,
    pixel_threshold: f64,
    min_votes: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    use sfmtool_core::reconstruction::point_correspondence::{
        find_point_correspondences_by_coordinate, CoordinateCorrespondenceOptions,
    };

    let src_img = to_contiguous!(source_track_image_indexes);
    let src_pts = to_contiguous!(source_track_point_indexes);
    let tgt_img = to_contiguous!(target_track_image_indexes);
    let tgt_pts = to_contiguous!(target_track_point_indexes);
    let shared_src = to_contiguous!(shared_images_source);
    let shared_tgt = to_contiguous!(shared_images_target);
    let rows = |name: &str, xy: &PyReadonlyArray2<f64>, n: usize| -> PyResult<Vec<[f64; 2]>> {
        let a = xy.as_array();
        if a.ncols() != 2 || a.nrows() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} must have shape ({n}, 2), one row per observation; got {:?}",
                a.shape()
            )));
        }
        Ok(a.rows().into_iter().map(|r| [r[0], r[1]]).collect())
    };
    let src_xy = rows("source_keypoints_xy", &source_keypoints_xy, src_img.len())?;
    let tgt_xy = rows("target_keypoints_xy", &target_keypoints_xy, tgt_img.len())?;
    if src_pts.len() != src_img.len()
        || tgt_pts.len() != tgt_img.len()
        || shared_src.len() != shared_tgt.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the track columns of each side, and the two shared-image arrays, must have equal lengths",
        ));
    }

    let result = find_point_correspondences_by_coordinate(
        &src_img,
        &src_xy,
        &src_pts,
        &tgt_img,
        &tgt_xy,
        &tgt_pts,
        &shared_src,
        &shared_tgt,
        &CoordinateCorrespondenceOptions {
            pixel_threshold,
            min_votes,
        },
    );

    Ok((
        numpy::PyArray1::from_vec(py, result.source_ids)
            .into_any()
            .unbind(),
        numpy::PyArray1::from_vec(py, result.target_ids)
            .into_any()
            .unbind(),
    ))
}

// ── Point and track merging ─────────────────────────────────────────────

/// Merge 3D points and tracks from multiple aligned reconstructions.
///
/// Args:
///     reconstructions: List of SfmrReconstruction objects.
///     correspondence_groups: List of groups, each group a list of (recon_idx, point_id).
///     image_mapping: Dict mapping image name -> list of (recon_idx, old_img_idx).
///
/// Points at infinity are kept: a merged point is finite when any point
/// merged into it is finite, at the mean of the finite points alone, and at
/// infinity otherwise, along the normalised sum of the directions.
///
/// Returns:
///     Dict with keys: positions_xyzw (N, 4), with w = 1 for a finite point and
///     w = 0 for a point at infinity, whose x, y, z are a unit direction;
///     colors, errors, image_indexes, feature_indexes, point_indexes.
#[pyfunction]
#[pyo3(name = "merge_points_and_tracks")]
pub fn merge_points_and_tracks_py(
    py: Python<'_>,
    reconstructions: &Bound<'_, PyList>,
    correspondence_groups: &Bound<'_, PyList>,
    image_mapping: &Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    // Borrow all reconstructions.
    let recon_refs: Vec<PyRef<PySfmrReconstruction>> = reconstructions
        .iter()
        .map(|item| {
            item.extract::<PyRef<PySfmrReconstruction>>()
                .map_err(PyErr::from)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let inner_refs: Vec<&sfmtool_core::SfmrReconstruction> =
        recon_refs.iter().map(|r| r.inner()).collect();

    // Parse correspondence groups: list[list[tuple[int, int]]].
    let groups: Vec<Vec<(usize, u32)>> = correspondence_groups
        .iter()
        .map(|group| {
            let group_list = group.cast::<PyList>()?;
            group_list
                .iter()
                .map(|item| {
                    let tuple = item.cast::<PyTuple>()?;
                    let recon_idx: usize = tuple.get_item(0)?.extract()?;
                    let point_id: u32 = tuple.get_item(1)?.extract()?;
                    Ok((recon_idx, point_id))
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Parse the image mapping, in the dict's order, which is the merged image
    // order: dict[str, list[tuple[int, int]]].
    let mapping: Vec<Vec<(usize, u32)>> = image_mapping
        .values()
        .iter()
        .map(|occurrences| {
            occurrences
                .cast::<PyList>()?
                .iter()
                .map(|occ| {
                    let tuple = occ.cast::<PyTuple>()?;
                    let recon_idx: usize = tuple.get_item(0)?.extract()?;
                    let old_img_idx: u32 = tuple.get_item(1)?.extract()?;
                    Ok((recon_idx, old_img_idx))
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Call the Rust core function.
    let result = sfmtool_core::reconstruction::point_correspondence::merge_points_and_tracks(
        &inner_refs,
        &groups,
        &mapping,
    )
    .map_err(value_err)?;

    // Build output dict with numpy arrays.
    let dict = PyDict::new(py);
    let positions_xyzw = numpy::PyArray2::from_vec2(
        py,
        &result
            .positions_xyzw
            .iter()
            .map(|p| p.to_vec())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let colors = numpy::PyArray2::from_vec2(
        py,
        &result.colors.iter().map(|c| c.to_vec()).collect::<Vec<_>>(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    dict.set_item("positions_xyzw", positions_xyzw)?;
    dict.set_item("colors", colors)?;
    dict.set_item("errors", numpy::PyArray1::from_vec(py, result.errors))?;
    dict.set_item(
        "image_indexes",
        numpy::PyArray1::from_vec(py, result.track_image_indexes),
    )?;
    dict.set_item(
        "feature_indexes",
        numpy::PyArray1::from_vec(py, result.track_feature_indexes),
    )?;
    dict.set_item(
        "point_indexes",
        numpy::PyArray1::from_vec(py, result.track_point_indexes),
    )?;

    Ok(dict.unbind())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_narrow_track_mask, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_alignment_py, m)?)?;
    m.add_function(wrap_pyfunction!(ransac_alignment_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_point_correspondences_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        find_point_correspondences_by_coordinate_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(merge_points_and_tracks_py, m)?)?;
    Ok(())
}
