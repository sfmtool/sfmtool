// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for image pair graph operations.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;

use sfmtool_core::image_pair_graph;

/// Build covisibility pairs from track observations.
/// Returns list of (i, j, count) tuples sorted by count descending.
#[pyfunction]
#[pyo3(signature = (quaternions_wxyz, track_point_ids, track_image_indexes, angle_threshold_deg=90.0))]
pub fn build_covisibility_pairs_py(
    quaternions_wxyz: PyReadonlyArray2<f64>,
    track_point_ids: PyReadonlyArray1<u32>,
    track_image_indexes: PyReadonlyArray1<u32>,
    angle_threshold_deg: f64,
) -> PyResult<Vec<(u32, u32, u32)>> {
    let num_images = quaternions_wxyz.shape()[0];

    let q_data = to_contiguous!(quaternions_wxyz);
    let point_ids_data = to_contiguous!(track_point_ids);
    let img_idx_data = to_contiguous!(track_image_indexes);

    Ok(image_pair_graph::build_covisibility_pairs(
        &q_data,
        num_images,
        &point_ids_data,
        &img_idx_data,
        angle_threshold_deg,
    ))
}

/// Build frustum intersection pairs using Monte Carlo volume estimation.
/// Returns list of (i, j, volume) tuples sorted by volume descending.
#[pyfunction]
#[pyo3(signature = (
    quaternions_wxyz, translations,
    fx, fy, cx, cy, widths, heights,
    histogram_counts, histogram_min_z, histogram_max_z,
    near_percentile=5.0, far_percentile=95.0,
    num_samples=100, angle_threshold_deg=90.0, seed=42
))]
#[allow(clippy::too_many_arguments)]
pub fn build_frustum_intersection_pairs_py(
    quaternions_wxyz: PyReadonlyArray2<f64>,
    translations: PyReadonlyArray2<f64>,
    fx: PyReadonlyArray1<f64>,
    fy: PyReadonlyArray1<f64>,
    cx: PyReadonlyArray1<f64>,
    cy: PyReadonlyArray1<f64>,
    widths: PyReadonlyArray1<u32>,
    heights: PyReadonlyArray1<u32>,
    histogram_counts: PyReadonlyArray2<u32>,
    histogram_min_z: PyReadonlyArray1<f64>,
    histogram_max_z: PyReadonlyArray1<f64>,
    near_percentile: f64,
    far_percentile: f64,
    num_samples: usize,
    angle_threshold_deg: f64,
    seed: u64,
) -> PyResult<Vec<(u32, u32, f64)>> {
    let num_images = quaternions_wxyz.shape()[0];
    let num_bins = histogram_counts.shape()[1];

    let q_data = to_contiguous!(quaternions_wxyz);
    let t_data = to_contiguous!(translations);
    let fx_data = to_contiguous!(fx);
    let fy_data = to_contiguous!(fy);
    let cx_data = to_contiguous!(cx);
    let cy_data = to_contiguous!(cy);
    let widths_data = to_contiguous!(widths);
    let heights_data = to_contiguous!(heights);
    let hist_counts_data = to_contiguous!(histogram_counts);
    let hist_min_data = to_contiguous!(histogram_min_z);
    let hist_max_data = to_contiguous!(histogram_max_z);

    Ok(image_pair_graph::build_frustum_intersection_pairs(
        &q_data,
        &t_data,
        num_images,
        &fx_data,
        &fy_data,
        &cx_data,
        &cy_data,
        &widths_data,
        &heights_data,
        &hist_counts_data,
        &hist_min_data,
        &hist_max_data,
        num_bins,
        near_percentile,
        far_percentile,
        num_samples,
        angle_threshold_deg,
        seed,
    ))
}
