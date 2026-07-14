// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the background-floor track-cluster matcher (see
//! `specs/core/track-cluster-matching.md`) and cluster-patch refinement (see
//! `specs/core/cluster-patch-refinement.md`).

use std::borrow::Cow;

use ndarray::{ArrayView2, ArrayView3};
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::features::cluster_match::{self, BackgroundFloorParams, Clusters};
use sfmtool_core::patch::cluster_refine::{
    refine_cluster_patches as core_refine_cluster_patches, warp_consistency_residuals,
    ClusterRefineParams, FeatureGeometry,
};

use crate::py_patch_cloud::{build_pyramids_from_image_list, parse_patch_window};
use crate::py_progress::ProgressCounter;
use crate::spatial::kdforest::{extract_u8_2d, resolve_forest_params};

/// Extract a 1-D `uint32` array, with a clear error if the dtype is wrong.
pub(crate) fn extract_u32_1d<'py>(
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

/// Refine SIFT clusters into patch clusters (see
/// `specs/core/cluster-patch-refinement.md`).
///
/// Per cluster: exclude members whose own patch fails the localizability
/// gate (see `specs/core/patch-localizability.md`), pick a reference member
/// (largest SIFT scale), build a Gaussian-windowed z-normalized template
/// around its detection, refine an affine warp to every other member by a
/// shift → similarity → affine Nelder-Mead cascade on the windowed ZNCC
/// (seeded from the SIFT affine shapes), vet by achieved ZNCC and
/// translation drift, and keep at most one member per image.
///
/// Args:
///     images: One HxW / HxWxC uint8 numpy array per image, in the
///         images-section order the cluster arrays index.
///     positions: Per image, the (N, 2) float32 SIFT keypoint positions
///         (COLMAP pixel convention), parallel to ``images``.
///     affine_shapes: Per image, the (N, 2, 2) float32 SIFT affine shapes,
///         parallel to ``images``.
///     cluster_starts: (C+1,) uint32 CSR offsets; cluster c owns members
///         cluster_starts[c]:cluster_starts[c+1].
///     member_images: (M,) uint32 member image index.
///     member_features: (M,) uint32 member feature index. An out-of-range
///         feature index (or a degenerate affine shape) marks the member
///         not_evaluated rather than raising.
///     radius: Template half-width, keypoint-frame units (default 4.0).
///     resolution: Template samples per axis (default 25).
///     window: "gaussian_disk" (default), "gaussian", or "uniform".
///     window_sigma: Window sigma in normalized patch coordinates (the grid
///         spans [-1, 1]); default 0.5 = the prototype's radius/2
///         keypoint-frame units.
///     min_zncc: Member acceptance threshold on the achieved windowed ZNCC
///         (default 0.85).
///     max_shift_px: Max translation drift from the SIFT seed, px
///         (default 3.0).
///     max_keypoint_uncertainty: Exclude a member before reference selection
///         and refinement when its own patch's predicted keypoint position
///         uncertainty (noise-normalized structure-tensor sigma_pos,
///         template-grid px) exceeds this; the member is marked
///         rejected_unlocalizable. 0 disables the gate. Default 0.35 — the
///         same default value as embed-patches' cull, though scored here on
///         the member's template-grid patch with the refinement window
///         rather than on the consensus.
///     max_iters: Nelder-Mead iterations per cascade stage (default 120).
///     progress: Optional ProgressCounter, bumped once per finished cluster.
///
/// Returns:
///     A dict mapping 1:1 onto the ``cluster_patches/`` section:
///     ``reference_members`` (C,) uint32 (0xFFFFFFFF = unrefinable),
///     ``member_status`` (M,) uint8, ``member_affines`` (M, 2, 3) float64
///     (leading 2x2 ``A`` plus the member's refined absolute keypoint
///     position ``p = A·x_ref + t`` in the last column; reference rows are
///     identity | x_ref), ``member_zncc`` (M,) float32,
///     ``member_shift_px`` (M,) float32, ``member_consistency_residual``
///     (M,) float32 — the member's relative misfit against a joint
///     weak-perspective factorization of all cluster warps (lower = more
///     consistent; NaN where not fitted; see
///     specs/core/cluster-warp-consistency.md). A stored signal, not a
///     gate.
#[pyfunction]
#[pyo3(signature = (images, positions, affine_shapes,
                    cluster_starts, member_images, member_features, *,
                    radius = 4.0, resolution = 25,
                    window = "gaussian_disk", window_sigma = None,
                    min_zncc = 0.85, max_shift_px = 3.0,
                    max_keypoint_uncertainty = 0.35,
                    max_iters = 120, progress = None))]
#[allow(clippy::too_many_arguments)]
pub fn refine_cluster_patches<'py>(
    py: Python<'py>,
    images: Vec<Bound<'py, PyAny>>,
    positions: Vec<PyReadonlyArray2<'py, f32>>,
    affine_shapes: Vec<PyReadonlyArray3<'py, f32>>,
    cluster_starts: &Bound<'py, PyAny>,
    member_images: &Bound<'py, PyAny>,
    member_features: &Bound<'py, PyAny>,
    radius: f64,
    resolution: u32,
    window: &str,
    window_sigma: Option<f64>,
    min_zncc: f64,
    max_shift_px: f64,
    max_keypoint_uncertainty: f64,
    max_iters: u32,
    progress: Option<ProgressCounter>,
) -> PyResult<Bound<'py, PyDict>> {
    let n_images = images.len();
    if positions.len() != n_images || affine_shapes.len() != n_images {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "images ({n_images}), positions ({}), and affine_shapes ({}) must be parallel",
            positions.len(),
            affine_shapes.len()
        )));
    }
    // Per-image feature-array consistency.
    let mut pos_data: Vec<(Cow<'_, [f32]>, usize)> = Vec::with_capacity(n_images);
    let mut aff_data: Vec<(Cow<'_, [f32]>, usize)> = Vec::with_capacity(n_images);
    for (i, (p, a)) in positions.iter().zip(&affine_shapes).enumerate() {
        let (pn, pc) = (p.shape()[0], p.shape()[1]);
        let ash = a.shape();
        if pc != 2 || ash[1] != 2 || ash[2] != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "image {i}: positions must be (N, 2) and affine_shapes (N, 2, 2)"
            )));
        }
        if ash[0] != pn {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "image {i}: positions ({pn}) and affine_shapes ({}) row counts differ",
                ash[0]
            )));
        }
        pos_data.push((to_contiguous!(p), pn));
        aff_data.push((to_contiguous!(a), ash[0]));
    }

    // CSR consistency (mirrors clusters_to_pair_matches's up-front gate).
    let cluster_starts = extract_u32_1d(cluster_starts, "cluster_starts")?;
    let member_images = extract_u32_1d(member_images, "member_images")?;
    let member_features = extract_u32_1d(member_features, "member_features")?;
    let starts: Cow<'_, [u32]> = to_contiguous!(cluster_starts);
    let m_images: Cow<'_, [u32]> = to_contiguous!(member_images);
    let m_features: Cow<'_, [u32]> = to_contiguous!(member_features);
    let m = m_images.len();
    if m_features.len() != m {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "member_images ({m}) and member_features ({}) must have the same length",
            m_features.len()
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
    if let Some(&bad) = m_images.iter().find(|&&i| i as usize >= n_images) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "member_images contains image index {bad} out of range for {n_images} images"
        )));
    }

    let params = ClusterRefineParams {
        radius,
        resolution,
        window: parse_patch_window(window, window_sigma.unwrap_or(0.5))?,
        min_zncc,
        max_shift_px,
        max_keypoint_uncertainty,
        max_iters,
        ..ClusterRefineParams::default()
    };

    // Decode images and build the pyramids (rayon, GIL-free); the cluster
    // path has no reconstruction, so there is no camera-dimension check.
    let pyramids = build_pyramids_from_image_list(py, &images, |_, _| Ok(()))?;

    let progress_handle = progress.as_ref().map(|p| p.handle());
    let (result, consistency) = py.detach(|| {
        let features: Vec<FeatureGeometry<'_>> = pos_data
            .iter()
            .zip(&aff_data)
            .map(|((p, pn), (a, an))| FeatureGeometry {
                positions_xy: ArrayView2::from_shape((*pn, 2), p.as_ref())
                    .expect("contiguous positions"),
                affine_shapes: ArrayView3::from_shape((*an, 2, 2), a.as_ref())
                    .expect("contiguous affine shapes"),
            })
            .collect();
        let result = core_refine_cluster_patches(
            &pyramids,
            &features,
            &starts,
            &m_images,
            &m_features,
            &params,
            progress_handle.as_deref(),
        );
        let consistency = warp_consistency_residuals(
            &starts,
            &m_images,
            &result.member_status,
            &result.reference_members,
            result.member_affines.view(),
            n_images,
        );
        (result, consistency)
    });

    let dict = PyDict::new(py);
    dict.set_item(
        "reference_members",
        result.reference_members.into_pyarray(py),
    )?;
    let status_u8: Vec<u8> = result.member_status.iter().map(|&s| s as u8).collect();
    dict.set_item("member_status", status_u8.into_pyarray(py))?;
    dict.set_item("member_affines", result.member_affines.into_pyarray(py))?;
    dict.set_item("member_zncc", result.member_zncc.into_pyarray(py))?;
    dict.set_item("member_shift_px", result.member_shift_px.into_pyarray(py))?;
    dict.set_item("member_consistency_residual", consistency.into_pyarray(py))?;
    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(background_floor_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(clusters_to_pair_matches, m)?)?;
    m.add_function(wrap_pyfunction!(refine_cluster_patches, m)?)?;
    Ok(())
}
