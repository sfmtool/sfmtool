// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the randomized kd-tree forest ANN index.
//!
//! [`PyKdForest`] wraps [`sfmtool_core::features::kdforest::KdForestU8`]: it builds once
//! from an `(N, D)` `uint8` descriptor array and answers batched approximate
//! k-NN queries. The `(indices, distances)` output is exactly the layout the
//! `sfmtool.feature_match` ratio test consumes, so an approximate matcher
//! backend slots in alongside the exact scanner.

use std::borrow::Cow;

use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::features::kdforest::{KdForestParams, KdForestU8};

use super::constellation::DEFAULTS;

/// Extract an `(N, D)` `uint8` array, with a clear error if the dtype is wrong.
///
/// PyO3's own extraction failure for a mismatched dtype is opaque; this mirrors
/// the explicit dtype errors the sibling `kdtree` bindings give.
pub(crate) fn extract_u8_2d<'py>(
    arr: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<PyReadonlyArray2<'py, u8>> {
    arr.extract::<PyReadonlyArray2<u8>>().map_err(|_| {
        let dtype = crate::helpers::dtype_name(arr).unwrap_or_else(|_| "?".to_string());
        pyo3::exceptions::PyTypeError::new_err(format!("{what} must be a uint8 array, got {dtype}"))
    })
}

/// Resolve a preset name to base parameters.
fn parse_preset(preset: Option<&str>, default_preset: &str) -> PyResult<KdForestParams> {
    match preset.unwrap_or(default_preset) {
        "balanced" => Ok(KdForestParams::balanced()),
        "fast" => Ok(KdForestParams::fast()),
        "accurate" => Ok(KdForestParams::accurate()),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown preset {other:?}; expected 'balanced', 'fast', or 'accurate'"
        ))),
    }
}

/// Resolve a preset name plus per-field overrides into [`KdForestParams`],
/// validating the overridden values. Shared by [`PyKdForest::new`] and the
/// cluster-match bindings so forest configuration means the same thing
/// everywhere.
pub(crate) fn resolve_forest_params(
    preset: Option<&str>,
    default_preset: &str,
    num_trees: Option<usize>,
    leaf_size: Option<usize>,
    max_leaf_checks: Option<usize>,
    seed: Option<u64>,
) -> PyResult<KdForestParams> {
    let mut params = parse_preset(preset, default_preset)?;
    if let Some(t) = num_trees {
        params.num_trees = t;
    }
    if let Some(l) = leaf_size {
        params.leaf_size = l;
    }
    if let Some(m) = max_leaf_checks {
        params.max_leaf_checks = m;
    }
    if let Some(s) = seed {
        params.seed = s;
    }
    if params.num_trees == 0 || params.leaf_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_trees and leaf_size must be positive",
        ));
    }
    Ok(params)
}

/// A randomized kd-tree forest over `uint8` descriptors (e.g. SIFT).
///
/// Build once from an `(N, D)` uint8 array, then issue batched approximate
/// nearest-neighbor queries. Squared-L2 distance is used internally (integer
/// domain); reported distances are Euclidean.
#[pyclass(name = "KdForest", module = "sfmtool.spatial")]
pub struct PyKdForest {
    inner: KdForestU8,
}

impl PyKdForest {
    /// Wrap a forest built elsewhere, for the sibling `.kdf` load binding.
    pub(crate) fn from_inner(inner: KdForestU8) -> Self {
        Self { inner }
    }

    /// The wrapped core forest, for the sibling `.kdf` export binding.
    pub(crate) fn inner(&self) -> &KdForestU8 {
        &self.inner
    }
}

#[pymethods]
impl PyKdForest {
    /// Build a forest from an `(N, D)` uint8 descriptor array.
    ///
    /// Args:
    ///     descriptors: (N, D) uint8 array; D is inferred from the array width.
    ///     preset: "balanced" (default), "fast", or "accurate".
    ///     num_trees: Override the number of trees (T).
    ///     leaf_size: Override the max points per leaf bucket.
    ///     max_leaf_checks: Override the default per-query budget (L_max).
    ///     seed: Override the base RNG seed.
    #[new]
    #[pyo3(signature = (descriptors, preset=None, num_trees=None, leaf_size=None, max_leaf_checks=None, seed=None))]
    fn new(
        py: Python<'_>,
        descriptors: &Bound<'_, PyAny>,
        preset: Option<&str>,
        num_trees: Option<usize>,
        leaf_size: Option<usize>,
        max_leaf_checks: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let descriptors = extract_u8_2d(descriptors, "descriptors")?;
        let shape = descriptors.shape();
        let n = shape[0];
        let dim = shape[1];
        if dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "descriptors must have a positive width",
            ));
        }

        let params = resolve_forest_params(
            preset,
            "balanced",
            num_trees,
            leaf_size,
            max_leaf_checks,
            seed,
        )?;

        let data: Cow<[u8]> = to_contiguous!(descriptors);

        let inner = py.detach(|| KdForestU8::build(&data, n, dim, params));
        Ok(Self { inner })
    }

    /// Number of indexed descriptors.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the forest holds no descriptors.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Descriptor dimensionality.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The numpy dtype of the indexed descriptors (always "uint8"), mirroring
    /// the sibling `KdTree2d`/`KdTree3d` introspection surface.
    #[getter]
    fn dtype(&self) -> &'static str {
        "uint8"
    }

    /// The default per-query budget (L_max) this forest was built with.
    #[getter]
    fn max_leaf_checks(&self) -> usize {
        self.inner.params().max_leaf_checks
    }

    /// One tree's leaf-ordered point IDs and each leaf's start within them.
    ///
    /// Args:
    ///     tree: Zero-based tree index.
    ///
    /// Returns:
    ///     Tuple (point_ids, leaf_starts) as uint32 arrays. Leaf i owns
    ///     `point_ids[leaf_starts[i] : leaf_starts[i+1]]`, the last leaf running
    ///     to the end.
    ///
    /// Over all trees this is the hypergraph a shared-corpus descriptor ordering
    /// is optimized against: each leaf is a set of IDs one query evaluates
    /// together, so an assignment keeping a leaf's members in one descriptor
    /// block turns its reads into one read.
    fn leaf_layout<'py>(&self, py: Python<'py>, tree: usize) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        if tree >= self.inner.num_trees() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tree {tree} out of range; the forest has {}",
                self.inner.num_trees()
            )));
        }
        let (ids, starts) = self.inner.tree_leaves(tree);
        Ok((
            numpy::PyArray1::from_slice(py, ids).into_any().unbind(),
            numpy::PyArray1::from_vec(py, starts).into_any().unbind(),
        ))
    }

    /// Number of trees in the forest.
    #[getter]
    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    /// Approximate k-NN query for a batch of descriptors.
    ///
    /// Args:
    ///     descriptors: (M, D) uint8 array; D must match the forest's dim.
    ///     k: Number of neighbors per query (default 2, for the ratio test).
    ///     max_leaf_checks: Per-query budget; None uses the build-time default.
    ///     max_dist: Optional Euclidean distance cutoff (None = unbounded).
    ///
    /// Returns:
    ///     Tuple (indices, distances):
    ///     - indices: (M, k) uint32 array of neighbor indices, nearest first.
    ///       Unfilled slots (fewer than k found, or beyond max_dist) are 2**32-1.
    ///     - distances: (M, k) float32 array of Euclidean distances; unfilled
    ///       slots are +inf.
    #[pyo3(signature = (descriptors, k=2, max_leaf_checks=None, max_dist=None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        descriptors: &Bound<'py, PyAny>,
        k: usize,
        max_leaf_checks: Option<usize>,
        max_dist: Option<f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let descriptors = extract_u8_2d(descriptors, "query descriptors")?;
        let shape = descriptors.shape();
        let m = shape[0];
        let dim = shape[1];
        if dim != self.inner.dim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query width {dim} does not match forest dim {}",
                self.inner.dim()
            )));
        }

        let budget = max_leaf_checks.unwrap_or_else(|| self.inner.params().max_leaf_checks);
        let data: Cow<[u8]> = to_contiguous!(descriptors);

        let (indices, dist_sq) = py.detach(|| {
            self.inner
                .search_batch_with_distances(&data, m, k, budget, max_dist)
        });
        // Report Euclidean distances; sqrt(inf) stays inf for unfilled slots.
        let distances: Vec<f32> = dist_sq.into_iter().map(|d| d.sqrt()).collect();

        let idx_arr = numpy::PyArray1::from_vec(py, indices).reshape([m, k])?;
        let dist_arr = numpy::PyArray1::from_vec(py, distances).reshape([m, k])?;
        Ok((idx_arr.into_any().unbind(), dist_arr.into_any().unbind()))
    }

    /// Rank the images that contain a constellation of features.
    ///
    /// The resident twin of `LazyKdForest.constellation_query`, and it answers
    /// identically for the same forest. It takes the source tables explicitly
    /// because this forest has none: a forest built here, or loaded back from a
    /// `.kdf`, holds the trees and the corpus and no record of which image any
    /// feature came from.
    ///
    /// Args:
    ///     positions: (N, 2) float32 positions of the constellation's features
    ///         in the query image.
    ///     sources: Per-feature source tables, in corpus feature-ID order, as
    ///         keys `image_indexes`, `image_feature_indexes`, `positions`
    ///         ((N, 2) float32) and `affine_shapes` ((N, 2, 2) float32) -- the
    ///         same mapping `write_kdf` takes, whose other keys are ignored.
    ///     descriptors: (N, D) uint8 descriptors. Pass this or `feature_ids`.
    ///     feature_ids: Their indices in this forest, when the query image is
    ///         itself indexed.
    ///     image_index: The query image's index; candidates from it are dropped.
    ///     Remaining arguments are as `LazyKdForest.constellation_query`.
    ///
    /// Returns:
    ///     The same list of dicts `LazyKdForest.constellation_query` returns.
    #[pyo3(signature = (positions, sources, *, descriptors=None, feature_ids=None,
                        image_index=None, k=DEFAULTS.k,
                        max_leaf_checks=DEFAULTS.max_leaf_checks,
                        threshold_px=DEFAULTS.threshold_px, iterations=DEFAULTS.iterations,
                        min_correspondences=DEFAULTS.min_correspondences,
                        min_inliers=DEFAULTS.min_inliers, max_scale=DEFAULTS.max_scale,
                        seed=DEFAULTS.seed))]
    #[allow(clippy::too_many_arguments)]
    fn constellation_query<'py>(
        &self,
        py: Python<'py>,
        positions: &Bound<'py, PyAny>,
        sources: &Bound<'py, PyAny>,
        descriptors: Option<&Bound<'py, PyAny>>,
        feature_ids: Option<Vec<u32>>,
        image_index: Option<u32>,
        k: usize,
        max_leaf_checks: usize,
        threshold_px: f64,
        iterations: usize,
        min_correspondences: usize,
        min_inliers: usize,
        max_scale: f64,
        seed: u64,
    ) -> PyResult<Py<pyo3::types::PyList>> {
        let sources = super::constellation::parse_resident_sources(sources)?;
        super::constellation::query(
            py,
            &self.inner,
            &sources,
            positions,
            descriptors,
            feature_ids,
            image_index,
            &super::constellation::QueryOptions {
                k,
                max_leaf_checks,
                threshold_px,
                iterations,
                min_correspondences,
                min_inliers,
                max_scale,
                seed,
            },
        )
    }

    /// `constellation_query` for a pixel and a radius in one image.
    ///
    /// The resident twin of `LazyKdForest.constellation_at_pixel`, taking the
    /// same `sources` mapping as `constellation_query` above.
    #[pyo3(signature = (sift_path, center, radius, sources, *, image_index=None,
                        k=DEFAULTS.k, max_leaf_checks=DEFAULTS.max_leaf_checks,
                        threshold_px=DEFAULTS.threshold_px, iterations=DEFAULTS.iterations,
                        min_correspondences=DEFAULTS.min_correspondences,
                        min_inliers=DEFAULTS.min_inliers, max_scale=DEFAULTS.max_scale,
                        seed=DEFAULTS.seed))]
    #[allow(clippy::too_many_arguments)]
    fn constellation_at_pixel<'py>(
        &self,
        py: Python<'py>,
        sift_path: std::path::PathBuf,
        center: (f32, f32),
        radius: f32,
        sources: &Bound<'py, PyAny>,
        image_index: Option<u32>,
        k: usize,
        max_leaf_checks: usize,
        threshold_px: f64,
        iterations: usize,
        min_correspondences: usize,
        min_inliers: usize,
        max_scale: f64,
        seed: u64,
    ) -> PyResult<Py<pyo3::types::PyDict>> {
        let sources = super::constellation::parse_resident_sources(sources)?;
        super::constellation::at_pixel(
            py,
            &self.inner,
            &sources,
            sift_path,
            center,
            radius,
            image_index,
            &super::constellation::QueryOptions {
                k,
                max_leaf_checks,
                threshold_px,
                iterations,
                min_correspondences,
                min_inliers,
                max_scale,
                seed,
            },
        )
    }
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKdForest>()?;
    Ok(())
}
