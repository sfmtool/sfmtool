// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the background-floor track-cluster matcher (see
//! `specs/core/track-cluster-matching.md`).

use std::borrow::Cow;

use ndarray::ArrayView2;
use numpy::{PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::features::cluster_match::{self, BackgroundFloorParams, Clusters};

use crate::py_kdforest::{extract_u8_2d, resolve_forest_params};

/// Extract a 1-D `uint32` array, with a clear error if the dtype is wrong.
fn extract_u32_1d<'py>(
    arr: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<PyReadonlyArray1<'py, u32>> {
    arr.extract::<PyReadonlyArray1<u32>>().map_err(|_| {
        let dtype = arr
            .getattr("dtype")
            .and_then(|d| d.getattr("name"))
            .and_then(|n| n.extract::<String>())
            .unwrap_or_else(|_| "?".to_string());
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{what} must be a 1-D uint32 array, got {dtype}"
        ))
    })
}

/// Extract the `(N, 128)` descriptor corpus, validating its width.
fn extract_corpus<'py>(
    descriptors: &Bound<'py, PyAny>,
) -> PyResult<(numpy::PyReadonlyArray2<'py, u8>, usize, usize)> {
    let descriptors = extract_u8_2d(descriptors, "descriptors")?;
    let shape = descriptors.shape();
    let (n, dim) = (shape[0], shape[1]);
    if dim != 128 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "descriptors must be (N, 128); got width {dim}"
        )));
    }
    Ok((descriptors, n, dim))
}

/// Background-floor track-cluster matcher: materialize the clusters.
///
/// Args:
///     descriptors: (N, 128) uint8 corpus, every image's SIFT descriptors
///         concatenated image by image.
///     image_starts: (n_images + 1,) uint32 CSR offsets; image i owns rows
///         image_starts[i]:image_starts[i+1].
///     d: Background rank; the d-th-nearest distance is the floor (default 10).
///         The k-NN query width is derived as d + 1.
///     alpha: Keep cross-image neighbours within alpha * floor (default 0.8).
///     min_size: Record a cluster only if it spans >= this many images
///         (default 2).
///     preset / num_trees / leaf_size / max_leaf_checks / seed: forest config,
///         same meaning as KdForest. The default preset is "accurate".
///
/// Returns (CSR clusters — the primary artefact):
///     Tuple (cluster_starts, member_images, member_features):
///     - cluster_starts: (C+1,) uint32 CSR offsets into the member arrays.
///     - member_images: (M,) uint32 member image index.
///     - member_features: (M,) uint32 member feature index (.sift row).
#[pyfunction]
#[pyo3(signature = (descriptors, image_starts, d=10, alpha=0.8, min_size=2,
                    preset=None, num_trees=None, leaf_size=None,
                    max_leaf_checks=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn background_floor_clusters(
    py: Python<'_>,
    descriptors: &Bound<'_, PyAny>,
    image_starts: &Bound<'_, PyAny>,
    d: usize,
    alpha: f32,
    min_size: usize,
    preset: Option<&str>,
    num_trees: Option<usize>,
    leaf_size: Option<usize>,
    max_leaf_checks: Option<usize>,
    seed: Option<u64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    if d == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "d (background rank) must be at least 1",
        ));
    }
    let (descriptors, n, dim) = extract_corpus(descriptors)?;
    let data: Cow<'_, [u8]> = to_contiguous!(descriptors);
    let image_starts = extract_u32_1d(image_starts, "image_starts")?;
    let starts: Cow<'_, [u32]> = to_contiguous!(image_starts);

    let forest = resolve_forest_params(
        preset,
        "accurate",
        num_trees,
        leaf_size,
        max_leaf_checks,
        seed,
    )?;
    let params = BackgroundFloorParams {
        d,
        alpha,
        min_size,
        forest,
    };

    let clusters = py
        .detach(|| {
            let view = ArrayView2::from_shape((n, dim), data.as_ref()).expect("contiguous corpus");
            cluster_match::background_floor_clusters(view, &starts, &params)
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let cluster_starts =
        numpy::PyArray1::from_vec(py, clusters.cluster_starts.into_raw_vec_and_offset().0);
    let member_images =
        numpy::PyArray1::from_vec(py, clusters.member_images.into_raw_vec_and_offset().0);
    let member_features =
        numpy::PyArray1::from_vec(py, clusters.member_features.into_raw_vec_and_offset().0);
    Ok((
        cluster_starts.into_any().unbind(),
        member_images.into_any().unbind(),
        member_features.into_any().unbind(),
    ))
}

/// Derived view: expand clusters into per-image-pair matches.
///
/// Args:
///     cluster_starts / member_images / member_features: the arrays returned
///         by background_floor_clusters.
///     descriptors: the same (N, 128) uint8 corpus the clusters were built
///         from (supplies the L2 match distances).
///     image_starts: the same (n_images + 1,) uint32 CSR offsets.
///
/// Returns:
///     Tuple (image_index_pairs, match_counts, match_feature_indexes,
///     match_descriptor_distances):
///     - image_index_pairs: (P, 2) uint32 sorted pairs with i < j.
///     - match_counts: (P,) uint32 matches per pair.
///     - match_feature_indexes: (M, 2) uint32 feature pairs grouped by pair.
///     - match_descriptor_distances: (M,) float32 Euclidean L2 distances.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn clusters_to_pair_matches(
    py: Python<'_>,
    cluster_starts: &Bound<'_, PyAny>,
    member_images: &Bound<'_, PyAny>,
    member_features: &Bound<'_, PyAny>,
    descriptors: &Bound<'_, PyAny>,
    image_starts: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let cluster_starts = extract_u32_1d(cluster_starts, "cluster_starts")?;
    let member_images = extract_u32_1d(member_images, "member_images")?;
    let member_features = extract_u32_1d(member_features, "member_features")?;
    let (descriptors, n, dim) = extract_corpus(descriptors)?;
    let data: Cow<'_, [u8]> = to_contiguous!(descriptors);
    let image_starts = extract_u32_1d(image_starts, "image_starts")?;

    let starts: Cow<'_, [u32]> = to_contiguous!(cluster_starts);
    let images: Cow<'_, [u32]> = to_contiguous!(member_images);
    let features: Cow<'_, [u32]> = to_contiguous!(member_features);
    let img_starts: Cow<'_, [u32]> = to_contiguous!(image_starts);

    // Validate the CSR arrays up front so bad indices surface as ValueError
    // rather than a panic in the core expansion.
    let m = images.len();
    if features.len() != m {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "member_images ({m}) and member_features ({}) must have the same length",
            features.len()
        )));
    }
    let csr_valid = !starts.is_empty()
        && starts[0] == 0
        && starts.windows(2).all(|w| w[0] <= w[1])
        && *starts.last().unwrap() as usize == m;
    if !csr_valid {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cluster_starts must be non-decreasing, start at 0, and end at M ({m})"
        )));
    }
    let img_starts_valid = img_starts.len() >= 2
        && img_starts[0] == 0
        && img_starts.windows(2).all(|w| w[0] <= w[1])
        && *img_starts.last().unwrap() as usize == n;
    if !img_starts_valid {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "image_starts must be non-decreasing, start at 0, and end at N ({n})"
        )));
    }
    let n_images = img_starts.len() - 1;
    for i in 0..m {
        let img = images[i] as usize;
        if img >= n_images
            || (img_starts[img] + features[i]) as usize >= img_starts[img + 1] as usize
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cluster member {i} (image {img}, feature {}) is out of range",
                features[i]
            )));
        }
    }

    let pairs = py.detach(|| {
        let clusters = Clusters {
            cluster_starts: ndarray::Array1::from_vec(starts.into_owned()),
            member_images: ndarray::Array1::from_vec(images.into_owned()),
            member_features: ndarray::Array1::from_vec(features.into_owned()),
        };
        let view = ArrayView2::from_shape((n, dim), data.as_ref()).expect("contiguous corpus");
        cluster_match::clusters_to_pair_matches(&clusters, view, &img_starts)
    });

    let pair_count = pairs.image_index_pairs.nrows();
    let match_count = pairs.match_feature_indexes.nrows();
    let image_index_pairs =
        numpy::PyArray1::from_vec(py, pairs.image_index_pairs.into_raw_vec_and_offset().0)
            .reshape([pair_count, 2])?;
    let match_counts =
        numpy::PyArray1::from_vec(py, pairs.match_counts.into_raw_vec_and_offset().0);
    let match_feature_indexes =
        numpy::PyArray1::from_vec(py, pairs.match_feature_indexes.into_raw_vec_and_offset().0)
            .reshape([match_count, 2])?;
    let match_descriptor_distances = numpy::PyArray1::from_vec(
        py,
        pairs.match_descriptor_distances.into_raw_vec_and_offset().0,
    );
    Ok((
        image_index_pairs.into_any().unbind(),
        match_counts.into_any().unbind(),
        match_feature_indexes.into_any().unbind(),
        match_descriptor_distances.into_any().unbind(),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(background_floor_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(clusters_to_pair_matches, m)?)?;
    Ok(())
}
