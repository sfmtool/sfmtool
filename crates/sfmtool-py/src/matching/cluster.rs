// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the background-floor track-cluster matcher (see
//! `specs/core/features/track-cluster-matching.md`) and cluster-patch refinement (see
//! `specs/core/patch/cluster-patch-refinement.md`).

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
    member_cell_data, refine_cluster_patches as core_refine_cluster_patches,
    warp_consistency_residuals, ClusterRefineParams, FeatureGeometry, LoopStop, PiecewiseParams,
};

use crate::patches::args::parse_patch_window;
use crate::patches::views::build_pyramids_from_image_list;
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

/// Background-floor track-cluster matcher over a persistent `.kdf` forest.
///
/// The out-of-core twin of `background_floor_clusters`: it opens a `.kdf`, runs
/// the same self-join against it, and clusters the result. Neither
/// the corpus nor the forest is held in memory — each query descriptor is read
/// from the file, used and dropped, and the index stays on disk behind a bounded
/// cache. What remains resident is that cache plus the `N x (d + 1)` neighbour
/// table the clustering stage consumes.
///
/// Results are identical to the in-memory matcher on the same forest, because
/// the file stores that forest's exact topology and leaf order.
///
/// Args:
///     path: A `.kdf` written from the forest to match against.
///     image_starts: (n_images + 1,) uint32 CSR offsets over the corpus, in the
///         same feature-ID order the `.kdf` was written from.
///     d: Background rank; the k-NN query width is d + 1 (default 10).
///     alpha: Keep cross-image neighbours within alpha * floor (default 0.8).
///     min_size: Record a cluster only if it spans >= this many images.
///     max_leaf_checks: Per-query budget. A `.kdf` stores no build-time default,
///         so this is always the caller's choice (default 128).
///     cache_bytes / max_chunk_bytes / query_workers: reader limits, as for
///         `LazyKdForest`.
///
/// Returns:
///     Tuple (cluster_starts, member_images, member_features), as for
///     `background_floor_clusters`.
///
/// Raises:
///     ValueError: The inputs disagree with each other.
///     OSError: The file is malformed or damaged.
#[pyfunction]
#[pyo3(signature = (path, image_starts, d=10, alpha=0.8, min_size=2,
                    max_leaf_checks=128, cache_bytes=None, max_chunk_bytes=None,
                    query_workers=None))]
#[allow(clippy::too_many_arguments)]
pub fn background_floor_clusters_kdf(
    py: Python<'_>,
    path: std::path::PathBuf,
    image_starts: &Bound<'_, PyAny>,
    d: usize,
    alpha: f32,
    min_size: usize,
    max_leaf_checks: usize,
    cache_bytes: Option<usize>,
    max_chunk_bytes: Option<usize>,
    query_workers: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    use sfmtool_core::features::cluster_match::LazyClusterError;
    use sfmtool_core::features::kdforest::{KdfOpenOptions, LazyKdForestU8};

    if d == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "d (background rank) must be at least 1",
        ));
    }
    let image_starts = extract_u32_1d(image_starts, "image_starts")?;
    let starts: Cow<'_, [u32]> = to_contiguous!(image_starts);

    let mut options = KdfOpenOptions::default();
    if let Some(v) = cache_bytes {
        options.cache_bytes = v;
        options.max_in_flight_bytes = v;
    }
    if let Some(v) = max_chunk_bytes {
        options.max_chunk_bytes = v;
        options.max_compressed_bytes = v;
    }
    if let Some(v) = query_workers {
        options.query_workers = v;
    }

    let params = BackgroundFloorParams {
        d,
        alpha,
        min_size,
        forest: resolve_forest_params(None, "accurate", None, None, Some(max_leaf_checks), None)?,
    };

    let clusters = py.detach(|| -> PyResult<_> {
        let lazy = LazyKdForestU8::open(&path, options).map_err(crate::spatial::kdf::to_py_err)?;
        // The corpus and the index stay on disk through the join; only the
        // neighbour table and the clustering scratch are held.
        cluster_match::background_floor_clusters_lazy(
            &lazy,
            &starts,
            &params,
            &sfmtool_core::progress::Progress::none(),
        )
        .map_err(|e| match e {
            LazyClusterError::Kdf(e) => crate::spatial::kdf::to_py_err(e),
            LazyClusterError::Cluster(e) => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        })
    })?;

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

/// Background-floor track-cluster matcher: materialize the clusters.
///
/// Args:
///     descriptors: (N, 128) uint8 corpus, every image's SIFT descriptors
///         concatenated image by image.
///     image_starts: (n_images + 1,) uint32 CSR offsets; image i owns rows
///         ``image_starts[i]:image_starts[i+1]``.
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
/// `specs/core/patch/cluster-patch-refinement.md`).
///
/// Per cluster: exclude members whose own patch does not pin a position (the
/// member gate on the ZNCC self-similarity radius, see
/// `specs/core/patch/zncc-self-similarity-radius.md`), pick a reference member
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
///         ``cluster_starts[c]:cluster_starts[c+1]``.
///     member_images: (M,) uint32 member image index.
///     member_features: (M,) uint32 member feature index. An out-of-range
///         feature index (or a degenerate affine shape) marks the member
///         not_evaluated rather than raising.
///     radius: Template half-width, keypoint-frame units (default 6.0 — a
///         full edge of 12 keypoint-frame units, SIFT's ~12x descriptor
///         window).
///     resolution: Template samples per axis (default 25).
///     window: "gaussian_disk" (default), "gaussian", or "uniform".
///     window_sigma: Window sigma in normalized patch coordinates (the grid
///         spans [-1, 1]); default 0.5 = the prototype's radius/2
///         keypoint-frame units.
///     min_zncc: Member acceptance threshold on the achieved windowed ZNCC
///         (default 0.85).
///     max_shift_px: Max translation drift from the SIFT seed, px
///         (default 3.0).
///     max_member_zncc_self_similarity_radius: Exclude a member before
///         reference selection and refinement when its own patch does not
///         pin a position: its ZNCC self-similarity radius, how far the
///         template-grid patch at its SIFT seed can slide over itself and
///         still match itself as well as a true match between two views
///         would, is above this bar, in template-grid px. The member is
///         marked rejected_unlocalizable. A NaN radius fails; the radius
///         reads at most 3, so a bar of 3 or more turns nothing out, and 0
///         disables the gate exactly. Default 2.5, the same bar as the
///         keypoint localizer's member gate
///         (specs/core/patch/zncc-self-similarity-radius.md).
///     max_iters: Nelder-Mead iterations per cascade stage (default 120).
///     piecewise: Run the piecewise refinement after the cascade for every
///         kept member: the nine cells of the reference's patch are
///         registered separately against the member's photograph, and a
///         robust affine map is fitted to their shifts (default False, which
///         carries no cells). See
///         specs/core/patch/cluster-patch-refinement.md.
///     move_shape: Piecewise setting: let the fitted map move the member's
///         shape and position, by a loop that applies it as an update while
///         the whole-member ZNCC does not fall (default True). False measures
///         the cells once at the cascade's shape and leaves every member
///         output exactly the cascade's.
///     cell_shift_bound_px: Piecewise setting: the search bound for a cell's
///         shift from its affine placement, template grid px (default 2.0).
///     min_cell_zncc: Piecewise setting: a cell whose ZNCC at its optimum is
///         below this is refused as refused_zncc (default 0.8).
///     min_cell_curvature: Piecewise setting: a cell whose ZNCC peak is
///         flatter than this along its flattest direction, ZNCC per grid
///         px squared, is refused as refused_curvature (default 0.02).
///     update_tolerance_px: Piecewise setting, read only with
///         ``move_shape``: the loop stops when the affine update moves every
///         cell centre by less than this, grid px (default 0.05).
///     max_iterations: Piecewise setting, read only with ``move_shape``: the
///         most renders the loop makes for one member (default 5); without
///         ``move_shape`` the stage renders once. Each piecewise setting
///         left as None takes the Rust default of ``PiecewiseParams``, the
///         value given above; the settings are ignored without
///         ``piecewise``.
///     progress: Optional ProgressCounter, bumped once per finished cluster.
///
/// Returns:
///     A dict mapping 1:1 onto the ``cluster_patches/`` section:
///     ``reference_members`` (C,) uint32 (0xFFFFFFFF = unrefinable),
///     ``member_status`` (M,) uint8, ``member_positions`` (M, 2) float64
///     (the member's refined absolute keypoint position ``p``),
///     ``member_affine_shapes`` (M, 2, 2) float64 (its absolute affine shape
///     ``S = W·S_ref``; the reference member's own row is ``S_ref``, and
///     ``W = S·S_ref**-1`` recovers the reference->member warp). Both are
///     all-zero for a member the cascade never fitted -- ``member_status``
///     says which. ``member_zncc`` (M,) float32, ``member_zncc_middle``
///     (M,) float32 (the same samples read over only the middle square of the
///     grid, half its width; not stored in the ``.matches`` section),
///     ``member_zncc_grid`` (M, 3, 3) float32 (the same samples read over each
///     cell of a three-by-three split of the grid with every pixel weighted
///     equally, ``[m, row, col]`` from the top-left cell; not stored either),
///     ``member_shift_px`` (M,) float32, ``member_consistency_residual``
///     (M,) float32 — the member's relative misfit against a joint
///     weak-perspective factorization of all cluster warps (lower = more
///     consistent; NaN where not fitted; see
///     specs/core/patch/cluster-warp-consistency.md). A stored signal, not a
///     gate. With ``piecewise`` the dict also carries the per-cell columns of
///     the ``cluster_patches/`` section, cells ``[m, row, col]`` from the
///     top-left, with readings only for kept members:
///     ``member_cell_shift_px`` (M, 3, 3, 2) float32 (each cell's displacement
///     from where the member's returned affine shape places it, template grid
///     px, with no fitted affine map removed; NaN where not measured),
///     ``member_cell_zncc`` (M, 3, 3) float32,
///     ``member_cell_status`` (M, 3, 3) uint8 (0 fitted, 1 refused_curvature,
///     2 refused_zncc, 3 not_attempted, 4 refused_bound, 5 refused_outlier)
///     and ``member_cell_iterations`` (M,) uint8, and ``piecewise_options``, a
///     dict of the six piecewise settings the run used, keyed by their
///     argument names. It also carries two per-member readings of the loop
///     that the ``.matches`` file does not store:
///     ``member_cell_loop_stop`` (M,) uint8, why the loop stopped (0 not run,
///     which every member that is not kept also reads, 1 converged, 2
///     reached the cap, 3 an update that would lower the whole-member ZNCC
///     was rejected, 4 the update stopped shrinking, 5 measured without
///     ``move_shape``), and
///     ``member_cell_update_accepted`` (M,) bool, whether the last fitted
///     update was applied to the returned shape, always False without
///     ``move_shape``. Without ``piecewise`` those seven keys are None.
#[pyfunction]
#[pyo3(signature = (images, positions, affine_shapes,
                    cluster_starts, member_images, member_features, *,
                    radius = 6.0, resolution = 25,
                    window = "gaussian_disk", window_sigma = None,
                    min_zncc = 0.85, max_shift_px = 3.0,
                    max_member_zncc_self_similarity_radius = 2.5,
                    max_iters = 120, piecewise = false, move_shape = None,
                    cell_shift_bound_px = None, min_cell_zncc = None,
                    min_cell_curvature = None, update_tolerance_px = None,
                    max_iterations = None, progress = None))]
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
    max_member_zncc_self_similarity_radius: f64,
    max_iters: u32,
    piecewise: bool,
    move_shape: Option<bool>,
    cell_shift_bound_px: Option<f32>,
    min_cell_zncc: Option<f32>,
    min_cell_curvature: Option<f32>,
    update_tolerance_px: Option<f32>,
    max_iterations: Option<u8>,
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
        max_member_zncc_self_similarity_radius,
        max_iters,
        piecewise: piecewise.then(|| {
            let default = PiecewiseParams::default();
            PiecewiseParams {
                move_shape: move_shape.unwrap_or(default.move_shape),
                cell_shift_bound_px: cell_shift_bound_px.unwrap_or(default.cell_shift_bound_px),
                min_cell_zncc: min_cell_zncc.unwrap_or(default.min_cell_zncc),
                min_cell_curvature: min_cell_curvature.unwrap_or(default.min_cell_curvature),
                update_tolerance_px: update_tolerance_px.unwrap_or(default.update_tolerance_px),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            }
        }),
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
            result.member_affine_shapes.view(),
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
    dict.set_item("member_positions", result.member_positions.into_pyarray(py))?;
    dict.set_item(
        "member_affine_shapes",
        result.member_affine_shapes.into_pyarray(py),
    )?;
    dict.set_item("member_zncc", result.member_zncc.into_pyarray(py))?;
    dict.set_item(
        "member_zncc_middle",
        result.member_zncc_middle.into_pyarray(py),
    )?;
    let m = result.member_zncc_grid.len();
    let grid = ndarray::Array3::from_shape_vec(
        (m, 3, 3),
        result
            .member_zncc_grid
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect(),
    )
    .expect("nine values per member");
    dict.set_item("member_zncc_grid", grid.into_pyarray(py))?;
    dict.set_item("member_shift_px", result.member_shift_px.into_pyarray(py))?;
    dict.set_item("member_consistency_residual", consistency.into_pyarray(py))?;
    let cell_keys = [
        "member_cell_shift_px",
        "member_cell_zncc",
        "member_cell_status",
        "member_cell_iterations",
        "member_cell_loop_stop",
        "member_cell_update_accepted",
    ];
    if let Some(pp) = params.piecewise.as_ref() {
        // The settings as the decimal values they were written as (0.8, not
        // the f32's 0.800000011920929), so a recorded value reads as given.
        let decimal = |v: f32| -> f64 { v.to_string().parse().expect("an f32 prints as a float") };
        let options = PyDict::new(py);
        options.set_item("move_shape", pp.move_shape)?;
        options.set_item("cell_shift_bound_px", decimal(pp.cell_shift_bound_px))?;
        options.set_item("min_cell_zncc", decimal(pp.min_cell_zncc))?;
        options.set_item("min_cell_curvature", decimal(pp.min_cell_curvature))?;
        options.set_item("update_tolerance_px", decimal(pp.update_tolerance_px))?;
        options.set_item("max_iterations", pp.max_iterations)?;
        dict.set_item("piecewise_options", options)?;
        let cells = member_cell_data(&result.cells);
        dict.set_item(cell_keys[0], cells.shift_px.into_pyarray(py))?;
        dict.set_item(cell_keys[1], cells.zncc.into_pyarray(py))?;
        dict.set_item(cell_keys[2], cells.status.into_pyarray(py))?;
        dict.set_item(cell_keys[3], cells.iterations.into_pyarray(py))?;
        let stop: Vec<u8> = result
            .cells
            .iter()
            .map(|c| c.map_or(LoopStop::NotRun, |c| c.stop) as u8)
            .collect();
        let accepted: Vec<bool> = result
            .cells
            .iter()
            .map(|c| c.is_some_and(|c| c.final_update_accepted))
            .collect();
        dict.set_item(cell_keys[4], stop.into_pyarray(py))?;
        dict.set_item(cell_keys[5], accepted.into_pyarray(py))?;
    } else {
        for key in cell_keys.into_iter().chain(["piecewise_options"]) {
            dict.set_item(key, py.None())?;
        }
    }
    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(background_floor_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(background_floor_clusters_kdf, m)?)?;
    m.add_function(wrap_pyfunction!(clusters_to_pair_matches, m)?)?;
    m.add_function(wrap_pyfunction!(refine_cluster_patches, m)?)?;
    Ok(())
}
