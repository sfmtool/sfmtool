// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for low-level descriptor matching operations.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;

use sfmtool_core::features::feature_match;
use sfmtool_core::features::feature_match::descriptor;

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

/// Match source features to target features using candidate indices and
/// descriptor distance, with deduplication.
///
/// For each query point, examines its candidate target indices, computes
/// descriptor L2 distances, and picks the best match under the threshold.
/// If multiple source features match the same target, keeps the one with
/// the lowest descriptor distance.
///
/// Args:
///     candidates: (N, K) uint32 array of candidate target indices per query.
///     in_bounds_idx: (N,) uint32 array of source feature indices for each
///         row of `candidates`.
///     descriptors1: (N, D) uint8 source descriptors.
///     descriptors2: (M, D) uint8 target descriptors.
///     descriptor_threshold: maximum L2 distance to accept a match.
///
/// Returns:
///     (M, 2) uint32 array of deduplicated (src_idx, dst_idx) matched pairs.
#[pyfunction]
pub fn match_candidates_by_descriptor(
    py: Python<'_>,
    candidates: PyReadonlyArray2<'_, u32>,
    in_bounds_idx: numpy::PyReadonlyArray1<'_, u32>,
    descriptors1: PyReadonlyArray2<'_, u8>,
    descriptors2: PyReadonlyArray2<'_, u8>,
    descriptor_threshold: f64,
) -> PyResult<Py<numpy::PyArray2<u32>>> {
    let cand_shape = candidates.shape();
    let n_queries = cand_shape[0];
    let k = cand_shape[1];

    let ibi_shape = in_bounds_idx.shape();
    if ibi_shape[0] != n_queries {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "in_bounds_idx length must match candidates row count",
        ));
    }

    let desc1_shape = descriptors1.shape();
    let desc2_shape = descriptors2.shape();
    let desc_len = desc1_shape[1];
    if desc2_shape[1] != desc_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "descriptors1 and descriptors2 must have the same descriptor length",
        ));
    }

    // Borrow all arrays zero-copy
    let cand_data = candidates
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("candidates must be C-contiguous"))?;
    let ibi_data = in_bounds_idx.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("in_bounds_idx must be C-contiguous")
    })?;
    let desc1_data = descriptors1.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("descriptors1 must be C-contiguous")
    })?;
    let desc2_data = descriptors2.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("descriptors2 must be C-contiguous")
    })?;

    let matches = py.detach(|| {
        descriptor::match_candidates_and_deduplicate(
            cand_data,
            ibi_data,
            desc1_data,
            desc2_data,
            n_queries,
            k,
            desc_len,
            descriptor_threshold,
        )
    });

    let n_matches = matches.len();
    let flat: Vec<u32> = matches.into_iter().flatten().collect();
    let array = ndarray::Array2::from_shape_vec((n_matches, 2), flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(array.into_pyarray(py).into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(descriptor_distance, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_descriptor_match, m)?)?;
    m.add_function(wrap_pyfunction!(match_candidates_by_descriptor, m)?)?;
    Ok(())
}
