// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for low-level descriptor matching operations.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;

use sfmtool_core::feature_match;

/// Compute L2 distance between two SIFT descriptors (1D uint8 arrays).
#[pyfunction]
pub fn descriptor_distance(a: PyReadonlyArray1<u8>, b: PyReadonlyArray1<u8>) -> PyResult<f64> {
    let a_data = to_contiguous!(a);
    let b_data = to_contiguous!(b);
    if a_data.len() != b_data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "descriptor length mismatch: {} vs {}",
            a_data.len(),
            b_data.len()
        )));
    }
    Ok(feature_match::descriptor_distance_l2(&a_data, &b_data))
}

/// Find the best matching descriptor in a 2D candidates array (M×D uint8).
///
/// Returns (best_index, distance) or (None, inf) if no match found.
#[pyfunction]
#[pyo3(signature = (query, candidates, threshold=None))]
pub fn find_best_descriptor_match(
    query: PyReadonlyArray1<u8>,
    candidates: PyReadonlyArray2<u8>,
    threshold: Option<f64>,
) -> PyResult<(Option<usize>, f64)> {
    let q_data = to_contiguous!(query);
    let c_data = to_contiguous!(candidates);
    let desc_len = candidates.shape()[1];

    match feature_match::find_best_match_contiguous(&q_data, &c_data, desc_len, threshold) {
        Some((idx, dist)) => Ok((Some(idx), dist)),
        None => Ok((None, f64::INFINITY)),
    }
}
