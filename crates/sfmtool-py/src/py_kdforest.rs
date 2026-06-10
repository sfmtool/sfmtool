// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the randomized kd-tree forest ANN index.
//!
//! [`PyKdForest`] wraps [`sfmtool_core::kdforest::KdForestU8`]: it builds once
//! from an `(N, D)` `uint8` descriptor array and answers batched approximate
//! k-NN queries. The `(indices, distances)` output is exactly the layout the
//! `sfmtool.feature_match` ratio test consumes, so an approximate matcher
//! backend slots in alongside the exact scanner.

use std::borrow::Cow;

use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::kdforest::{KdForestParams, KdForestU8};

/// Extract an `(N, D)` `uint8` array, with a clear error if the dtype is wrong.
///
/// PyO3's own extraction failure for a mismatched dtype is opaque; this mirrors
/// the explicit dtype errors the sibling `py_kdtree` bindings give.
pub(crate) fn extract_u8_2d<'py>(
    arr: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<PyReadonlyArray2<'py, u8>> {
    arr.extract::<PyReadonlyArray2<u8>>().map_err(|_| {
        let dtype = arr
            .getattr("dtype")
            .and_then(|d| d.getattr("name"))
            .and_then(|n| n.extract::<String>())
            .unwrap_or_else(|_| "?".to_string());
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
#[pyclass(name = "KdForest", module = "sfmtool")]
pub struct PyKdForest {
    inner: KdForestU8,
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

        let data: Cow<[u8]> = match descriptors.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(descriptors.as_array().iter().copied().collect()),
        };

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
        let data: Cow<[u8]> = match descriptors.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(descriptors.as_array().iter().copied().collect()),
        };

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
}
