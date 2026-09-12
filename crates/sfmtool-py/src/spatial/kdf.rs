// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for persistent `.kdf` forests and their file-backed queries.
//!
//! These exist to make the layout comparison in
//! `specs/core/features/lazy-kdforest-query.md` runnable from Python: export a
//! forest each way, then measure. That shapes the surface more than a general
//! wrapper would.
//!
//! Two consequences worth knowing before reading further:
//!
//! * **Every counter the benchmark plan asks for is reachable, and resettable.**
//!   [`PyLazyKdForest::io_stats`] returns the cache and read counters, and
//!   [`PyLazyKdForest::reset_io_stats`] zeroes them without reopening — which is
//!   what separates an open from the queries after it, or a cold pass from a
//!   warm one, inside a single measurement run.
//! * **Byte limits are per-instance and explicit.** The Rust defaults are
//!   generous (256 MiB of cache); a benchmark sweeping cache budgets sets them
//!   per open, so every one is a keyword argument rather than a global.
//!
//! `query` returns Euclidean distances, matching the eager `KdForest.query` it
//! is compared against — the underlying search works in squared distance, and
//! both bindings take the square root at the boundary so a caller never has to
//! ask which convention a given object uses.

use std::borrow::Cow;
use std::path::PathBuf;

use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::features::kdforest::{
    kdf_summary, DescriptorStorage, FeatureOrigin, KdfError, KdfSiftSources, KdfWorkspaceContents,
    KdfWorkspaceMetadata, KdfWriteOptions, LazyKdForestOptions, LazyKdForestU8,
};

use super::kdforest::extract_u8_2d;

/// Map a format error onto the closest Python exception.
///
/// The variants split cleanly by who is at fault: a bad argument is a
/// `ValueError`, a budget that cannot hold what was asked for is a
/// `MemoryError`, a damaged or malformed file is an `OSError`, and a missing
/// source file is a `FileNotFoundError`. Collapsing all of them into one
/// exception type would make a benchmark sweep unable to tell "this cache
/// budget is too small" (retry smaller) from "this file is corrupt" (stop).
pub(crate) fn to_py_err(err: KdfError) -> PyErr {
    let message = err.to_string();
    match err {
        KdfError::Io(e) => PyErr::from(e),
        KdfError::MissingSource(path) => pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "SIFT source is missing: {}",
            path.display()
        )),
        KdfError::ResourceLimit(_) => pyo3::exceptions::PyMemoryError::new_err(message),
        KdfError::InvalidQuery(_) | KdfError::ScalarType { .. } => {
            pyo3::exceptions::PyValueError::new_err(message)
        }
        KdfError::InvalidFormat(_)
        | KdfError::ShapeMismatch(_)
        | KdfError::Integrity(_)
        | KdfError::Zip(_)
        | KdfError::Json(_) => pyo3::exceptions::PyOSError::new_err(message),
    }
}

/// Resolve the `layout` argument into a [`DescriptorStorage`].
///
/// `descriptor_block_bytes` is meaningful only for the shared layout, so
/// supplying it with `layout="tree_local"` is rejected rather than ignored: in a
/// sweep that silently-ignored argument would produce two identical runs
/// labelled as different ones.
fn parse_storage(
    layout: &str,
    descriptor_block_bytes: Option<usize>,
) -> PyResult<DescriptorStorage> {
    match layout {
        "tree_local" => {
            if descriptor_block_bytes.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "descriptor_block_bytes applies to layout='shared' only",
                ));
            }
            Ok(DescriptorStorage::TreeLocal)
        }
        "shared" => {
            let bytes = descriptor_block_bytes.unwrap_or(64 << 10);
            if bytes == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "descriptor_block_bytes must be positive",
                ));
            }
            Ok(DescriptorStorage::Shared {
                target_descriptor_block_bytes: bytes,
            })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown layout {other:?}; expected 'tree_local' or 'shared'"
        ))),
    }
}

/// Build [`KdfSiftSources`] from the plain Python mapping the bindings accept.
///
/// The Rust type wants parallel arrays plus a workspace record; expressing that
/// as one dict keeps the binding free of a class whose only job would be to
/// carry six fields from Python to Rust.
fn parse_sources(sources: &Bound<'_, PyAny>) -> PyResult<KdfSiftSources> {
    fn need<'py>(d: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        d.get_item(key).map_err(|_| {
            pyo3::exceptions::PyKeyError::new_err(format!("sources is missing {key:?}"))
        })
    }
    fn hashes(value: Bound<'_, PyAny>, what: &str) -> PyResult<Vec<[u8; 16]>> {
        let raw: Vec<Vec<u8>> = value.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("{what} must be a sequence of bytes"))
        })?;
        raw.into_iter()
            .map(|v| {
                <[u8; 16]>::try_from(v.as_slice()).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{what} entries must be exactly 16 bytes, got {}",
                        v.len()
                    ))
                })
            })
            .collect()
    }

    let workspace = need(sources, "workspace")?;
    let contents = need(&workspace, "contents")?;

    let image_names: Vec<String> = need(sources, "image_names")?.extract()?;
    let feature_tool_hashes = hashes(need(sources, "feature_tool_hashes")?, "feature_tool_hashes")?;
    let sift_content_hashes = hashes(need(sources, "sift_content_hashes")?, "sift_content_hashes")?;

    // Origins arrive as two parallel uint32 columns, the same shape they are
    // stored in and the shape `resolve_origins` hands back, so a round trip
    // through Python needs no reshaping in either direction.
    let image_indexes: Vec<u32> = need(sources, "image_indexes")?.extract()?;
    let image_feature_indexes: Vec<u32> = need(sources, "image_feature_indexes")?.extract()?;
    if image_indexes.len() != image_feature_indexes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "image_indexes has {} entries but image_feature_indexes has {}",
            image_indexes.len(),
            image_feature_indexes.len()
        )));
    }

    // A JSON *string* rather than a Python object: this crate has no
    // Python-to-serde_json bridge, and adding one for a provenance field no
    // benchmark reads would be a dependency for a single call site. Callers
    // already hold this value as workspace JSON, so `json.dumps` is a no-op for
    // them and an explicit parse error for anything else.
    let feature_options: String = need(&contents, "feature_options")?.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "sources['workspace']['contents']['feature_options'] must be a JSON string",
        )
    })?;
    let feature_options: serde_json::Value = serde_json::from_str(&feature_options)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("feature_options: {e}")))?;

    Ok(KdfSiftSources {
        workspace: KdfWorkspaceMetadata {
            absolute_path: need(&workspace, "absolute_path")?.extract()?,
            relative_path: need(&workspace, "relative_path")?.extract()?,
            contents: KdfWorkspaceContents {
                feature_tool: need(&contents, "feature_tool")?.extract()?,
                feature_type: need(&contents, "feature_type")?.extract()?,
                feature_options,
                feature_prefix_dir: need(&contents, "feature_prefix_dir")?.extract()?,
            },
        },
        image_names,
        feature_tool_hashes,
        sift_content_hashes,
        origins: image_indexes
            .into_iter()
            .zip(image_feature_indexes)
            .map(|(image_index, image_feature_index)| FeatureOrigin {
                image_index,
                image_feature_index,
            })
            .collect(),
    })
}

/// A `.kdf` opened for queries that decode only what they reach.
///
/// Construct with a path; every byte limit is a keyword argument so a sweep can
/// vary one per open. `dtype` is always `"uint8"`: the format also carries
/// `float32`, but the eager `KdForest` these are compared against is `uint8`
/// only, and binding a scalar with no counterpart to compare against would add
/// surface no benchmark can use.
#[pyclass(name = "LazyKdForest", module = "sfmtool.spatial")]
pub struct PyLazyKdForest {
    inner: LazyKdForestU8,
    path: PathBuf,
}

#[pymethods]
impl PyLazyKdForest {
    /// Open a `.kdf` without decoding tree or descriptor payloads.
    ///
    /// Args:
    ///     path: The `.kdf` file to open.
    ///     cache_bytes: Decoded-data cache budget (default 256 MiB).
    ///     max_in_flight_bytes: Ceiling on concurrent decode reservations.
    ///     max_compressed_bytes: Ceiling on one entry's compressed buffer.
    ///     max_metadata_bytes: Ceiling on the decoded metadata JSON.
    ///     max_chunk_bytes: Ceiling on one decoded chunk.
    ///     max_address_map_bytes: Ceiling on the shared row map, read at open.
    ///     max_leaf_features: Ceiling on the rows one leaf may own.
    ///     query_workers: Threads used by batch queries (default 1).
    ///
    /// Raises:
    ///     MemoryError: A limit is zero, or cannot hold what the file needs.
    ///     OSError: The file is malformed, or a hash does not match.
    #[new]
    #[pyo3(signature = (path, *, cache_bytes=None, max_in_flight_bytes=None,
                        max_compressed_bytes=None, max_metadata_bytes=None,
                        max_chunk_bytes=None, max_address_map_bytes=None,
                        max_leaf_features=None, query_workers=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        cache_bytes: Option<usize>,
        max_in_flight_bytes: Option<usize>,
        max_compressed_bytes: Option<usize>,
        max_metadata_bytes: Option<usize>,
        max_chunk_bytes: Option<usize>,
        max_address_map_bytes: Option<usize>,
        max_leaf_features: Option<usize>,
        query_workers: Option<usize>,
    ) -> PyResult<Self> {
        let mut options = LazyKdForestOptions::default();
        if let Some(v) = cache_bytes {
            options.cache_bytes = v;
        }
        if let Some(v) = max_in_flight_bytes {
            options.max_in_flight_bytes = v;
        }
        if let Some(v) = max_compressed_bytes {
            options.max_compressed_bytes = v;
        }
        if let Some(v) = max_metadata_bytes {
            options.max_metadata_bytes = v;
        }
        if let Some(v) = max_chunk_bytes {
            options.max_chunk_bytes = v;
        }
        if let Some(v) = max_address_map_bytes {
            options.max_address_map_bytes = v;
        }
        if let Some(v) = max_leaf_features {
            options.max_leaf_features = v;
        }
        if let Some(v) = query_workers {
            options.query_workers = v;
        }
        let opened = path.clone();
        let inner = py
            .detach(|| LazyKdForestU8::open(&opened, options))
            .map_err(to_py_err)?;
        Ok(Self { inner, path })
    }

    /// Number of indexed descriptors.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the file indexes no descriptors.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Descriptor dimensionality.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The numpy dtype of the indexed descriptors (always "uint8").
    #[getter]
    fn dtype(&self) -> &'static str {
        "uint8"
    }

    /// The file this forest was opened from.
    #[getter]
    fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Approximate k-NN query for a batch of descriptors.
    ///
    /// Args:
    ///     descriptors: (M, D) uint8 array; D must match the file's dim.
    ///     k: Neighbors per query (default 2, for the ratio test).
    ///     max_leaf_checks: Per-query leaf budget (default 128). Unlike the
    ///         eager forest, a `.kdf` stores no build-time default to fall back
    ///         on, so this is always the caller's choice.
    ///     max_dist: Optional Euclidean distance cutoff (None = unbounded).
    ///
    /// Returns:
    ///     Tuple (indices, distances) shaped (M, k), with 2**32-1 and +inf in
    ///     unfilled slots, matching `KdForest.query`.
    ///
    /// Raises:
    ///     OSError: A chunk this query reached is damaged. Opening validates
    ///         structure but decodes nothing, so this is the first point a
    ///         corrupt payload can surface.
    #[pyo3(signature = (descriptors, k=2, max_leaf_checks=128, max_dist=None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        descriptors: &Bound<'py, PyAny>,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (indices, distances, _) =
            self.run_batch(py, descriptors, k, max_leaf_checks, max_dist)?;
        Ok((indices, distances))
    }

    /// `query`, plus the traversal counters the batch accumulated.
    ///
    /// Returns:
    ///     Tuple (indices, distances, stats), where stats is a dict with
    ///     `checks`, `pushes` and `pops` summed over the batch. `checks` is the
    ///     denominator of read amplification: pair it with the `decoded_bytes`
    ///     that `io_stats` reports across the same call.
    #[pyo3(signature = (descriptors, k=2, max_leaf_checks=128, max_dist=None))]
    fn query_with_stats<'py>(
        &self,
        py: Python<'py>,
        descriptors: &Bound<'py, PyAny>,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyDict>)> {
        let (indices, distances, stats) =
            self.run_batch(py, descriptors, k, max_leaf_checks, max_dist)?;
        let d = PyDict::new(py);
        d.set_item("checks", stats.checks)?;
        d.set_item("pushes", stats.pushes)?;
        d.set_item("pops", stats.pops)?;
        Ok((indices, distances, d.unbind()))
    }

    /// Cache and read counters for everything this instance has done so far.
    ///
    /// Returns:
    ///     A dict with `read_calls`, `compressed_bytes`, `decoded_bytes`,
    ///     `cache_hits`, `cache_misses`, `evictions`, `duplicate_load_waits`,
    ///     `resident_bytes`, `peak_resident_bytes`, `in_flight_bytes`,
    ///     `peak_in_flight_bytes` and `address_map_bytes`. The counters are
    ///     cumulative; the `*_bytes` gauges without `peak_` are current.
    fn io_stats<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let s = self.inner.io_stats();
        let d = PyDict::new(py);
        d.set_item("read_calls", s.read_calls)?;
        d.set_item("compressed_bytes", s.compressed_bytes)?;
        d.set_item("decoded_bytes", s.decoded_bytes)?;
        d.set_item("cache_hits", s.cache_hits)?;
        d.set_item("cache_misses", s.cache_misses)?;
        d.set_item("evictions", s.evictions)?;
        d.set_item("duplicate_load_waits", s.duplicate_load_waits)?;
        d.set_item("resident_bytes", s.resident_bytes)?;
        d.set_item("peak_resident_bytes", s.peak_resident_bytes)?;
        d.set_item("in_flight_bytes", s.in_flight_bytes)?;
        d.set_item("peak_in_flight_bytes", s.peak_in_flight_bytes)?;
        d.set_item("address_map_bytes", s.address_map_bytes)?;
        Ok(d.unbind())
    }

    /// Zero the cumulative counters, keeping what the cache currently holds.
    ///
    /// The peaks restart from the current gauges rather than from zero, so a
    /// reported peak is never below a byte count already resident. Use this to
    /// separate an open from the queries after it without reopening — reopening
    /// would also drop the cache, which is usually the opposite of the intent.
    fn reset_io_stats(&self) {
        self.inner.reset_io_stats();
    }

    /// Resolve corpus feature IDs to their source image features.
    ///
    /// Args:
    ///     feature_ids: Sequence of uint32 corpus feature IDs.
    ///
    /// Returns:
    ///     Tuple (image_indexes, image_feature_indexes) as uint32 arrays in the
    ///     order asked for, repeats included; or None when the file carries no
    ///     sources. Only the origin blocks covering the requested IDs are read.
    fn resolve_origins<'py>(
        &self,
        py: Python<'py>,
        feature_ids: Vec<u32>,
    ) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let resolved = py
            .detach(|| self.inner.resolve_origins(&feature_ids))
            .map_err(to_py_err)?;
        let Some(origins) = resolved else {
            return Ok(None);
        };
        let images: Vec<u32> = origins.iter().map(|o| o.image_index).collect();
        let features: Vec<u32> = origins.iter().map(|o| o.image_feature_index).collect();
        Ok(Some((
            numpy::PyArray1::from_vec(py, images).into_any().unbind(),
            numpy::PyArray1::from_vec(py, features).into_any().unbind(),
        )))
    }

    /// The image table, read once on first access.
    ///
    /// Returns:
    ///     A dict with `names`, `feature_tool_hashes` and `sift_content_hashes`
    ///     (both hash lists as 16-byte objects), or None when the file carries
    ///     no sources. It never opens a referenced `.sift` file.
    fn image_table<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        let Some(table) = self.inner.image_table().map_err(to_py_err)? else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("names", PyList::new(py, &table.names)?)?;
        d.set_item(
            "feature_tool_hashes",
            PyList::new(
                py,
                table
                    .feature_tool_hashes
                    .iter()
                    .map(|h| pyo3::types::PyBytes::new(py, h)),
            )?,
        )?;
        d.set_item(
            "sift_content_hashes",
            PyList::new(
                py,
                table
                    .sift_content_hashes
                    .iter()
                    .map(|h| pyo3::types::PyBytes::new(py, h)),
            )?,
        )?;
        Ok(Some(d.unbind()))
    }

    fn __repr__(&self) -> String {
        format!(
            "LazyKdForest(path={:?}, len={}, dim={})",
            self.path,
            self.inner.len(),
            self.inner.dim()
        )
    }
}

impl PyLazyKdForest {
    /// Shared body of `query` and `query_with_stats`.
    fn run_batch<'py>(
        &self,
        py: Python<'py>,
        descriptors: &Bound<'py, PyAny>,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> PyResult<(
        Py<PyAny>,
        Py<PyAny>,
        sfmtool_core::features::kdforest::LazyQueryStats,
    )> {
        let descriptors = extract_u8_2d(descriptors, "query descriptors")?;
        let shape = descriptors.shape();
        let (m, dim) = (shape[0], shape[1]);
        if dim != self.inner.dim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query width {dim} does not match file dim {}",
                self.inner.dim()
            )));
        }
        let data: Cow<[u8]> = to_contiguous!(descriptors);
        let (indices, dist_sq, stats) = py
            .detach(|| {
                self.inner
                    .search_batch_with_stats(&data, m, k, max_leaf_checks, max_dist)
            })
            .map_err(to_py_err)?;
        // Euclidean, matching `KdForest.query`; sqrt(inf) stays inf.
        let distances: Vec<f32> = dist_sq.into_iter().map(|d| d.sqrt()).collect();
        let idx = numpy::PyArray1::from_vec(py, indices).reshape([m, k])?;
        let dst = numpy::PyArray1::from_vec(py, distances).reshape([m, k])?;
        Ok((idx.into_any().unbind(), dst.into_any().unbind(), stats))
    }
}

/// Export an in-memory forest to a `.kdf`.
///
/// Implemented as a free function rather than a `KdForest` method because it
/// belongs to the persistence layer, not to the index: the eager forest is
/// usable, and testable, with no `.kdf` support compiled in at all.
///
/// Args:
///     forest: The `KdForest` to write.
///     path: Destination; the call fails if it already exists.
///     layout: "tree_local" (a vector copy per tree) or "shared" (one corpus
///         plus a row map). Explicit because which is smaller or faster is
///         exactly what the benchmark is for; there is no default worth
///         asserting yet.
///     descriptor_block_bytes: Target size of one shared descriptor block
///         (default 64 KiB). Rejected for layout="tree_local".
///     chunk_bytes: Target decoded size of one tree chunk (default 1 MiB).
///     compression_level: zstd level (default 3).
///     origin_block_rows: Rows per origin block (default 131072).
///     sources: Optional dict of SIFT provenance with keys `workspace`,
///         `image_names`, `feature_tool_hashes`, `sift_content_hashes`,
///         `image_indexes` and `image_feature_indexes`.
///     descriptor_order: Order the shared corpus is stored in, as a permutation
///         of 0..N where entry r is the feature at row r. None uses tree 0's
///         leaf order. Any permutation is valid — the stored row map is what a
///         reader follows — so this selects an ordering policy without changing
///         the file format. Ignored for layout="tree_local".
///
/// Raises:
///     FileExistsError: `path` already exists. Writing goes through a sibling
///         temporary file and publishes only a complete archive, so an
///         interrupted export never leaves a half-written `.kdf` in place.
///     ValueError: The options or the sources are inconsistent.
#[pyfunction]
#[pyo3(signature = (forest, path, *, layout, descriptor_block_bytes=None, chunk_bytes=None,
                    compression_level=None, origin_block_rows=None, sources=None,
                    descriptor_order=None))]
#[allow(clippy::too_many_arguments)]
fn write_kdf(
    py: Python<'_>,
    forest: PyRef<'_, super::kdforest::PyKdForest>,
    path: PathBuf,
    layout: &str,
    descriptor_block_bytes: Option<usize>,
    chunk_bytes: Option<usize>,
    compression_level: Option<i32>,
    origin_block_rows: Option<usize>,
    sources: Option<&Bound<'_, PyAny>>,
    descriptor_order: Option<Vec<u32>>,
) -> PyResult<()> {
    let mut options = KdfWriteOptions::tree_local();
    options.descriptor_storage = parse_storage(layout, descriptor_block_bytes)?;
    if let Some(v) = chunk_bytes {
        options.target_chunk_bytes = v;
    }
    if let Some(v) = compression_level {
        options.compression_level = v;
    }
    if let Some(v) = origin_block_rows {
        options.origin_block_rows = v;
    }
    let sources = sources.map(parse_sources).transpose()?;

    // `write_kdf` refuses an existing destination, but the error it raises is a
    // plain io::Error; surfacing it as FileExistsError is what lets a sweep
    // distinguish "already measured this cell" from a real I/O failure.
    if path.exists() {
        return Err(pyo3::exceptions::PyFileExistsError::new_err(format!(
            "{} already exists",
            path.display()
        )));
    }
    let inner = forest.inner();
    py.detach(|| {
        inner.write_kdf_ordered(
            &path,
            sources.as_ref(),
            &options,
            descriptor_order.as_deref(),
        )
    })
    .map_err(to_py_err)
}

/// Load a `.kdf` fully into memory as a `KdForest`.
///
/// The third option between querying a file lazily and rebuilding an index from
/// the descriptor corpus. The file stores the exact topology, leaf order and
/// feature IDs of the forest it was written from, so this is decompression and
/// reassembly rather than a build — no median splits and no randomization to
/// reproduce — and the result answers exactly as the original did.
///
/// Use it when many queries will follow: it pays the whole file's read up front
/// and then queries at in-memory speed, where `LazyKdForest` pays almost nothing
/// up front and more per query.
///
/// Args:
///     path: The `.kdf` to load. Either descriptor layout works.
///     max_chunk_bytes / max_compressed_bytes / max_metadata_bytes: reader
///         limits, as for `LazyKdForest`. The cache only buffers the load here,
///         so its budget bounds working memory during the read, not after.
///
/// Returns:
///     A `KdForest` holding the corpus and every tree.
///
/// Raises:
///     OSError: The file is malformed, or a hash does not match.
///     MemoryError: A limit is too small for the file's chunks.
#[pyfunction]
#[pyo3(signature = (path, *, cache_bytes=None, max_chunk_bytes=None,
                    max_compressed_bytes=None, max_metadata_bytes=None))]
fn read_kdf(
    py: Python<'_>,
    path: PathBuf,
    cache_bytes: Option<usize>,
    max_chunk_bytes: Option<usize>,
    max_compressed_bytes: Option<usize>,
    max_metadata_bytes: Option<usize>,
) -> PyResult<super::kdforest::PyKdForest> {
    let mut options = LazyKdForestOptions::default();
    if let Some(v) = cache_bytes {
        options.cache_bytes = v;
        options.max_in_flight_bytes = v;
    }
    if let Some(v) = max_chunk_bytes {
        options.max_chunk_bytes = v;
    }
    if let Some(v) = max_compressed_bytes {
        options.max_compressed_bytes = v;
    }
    if let Some(v) = max_metadata_bytes {
        options.max_metadata_bytes = v;
    }
    let inner = py
        .detach(|| sfmtool_core::features::kdforest::KdForestU8::read_kdf(&path, options))
        .map_err(to_py_err)?;
    Ok(super::kdforest::PyKdForest::from_inner(inner))
}

/// Account for a `.kdf`'s size without decoding its payloads.
///
/// This is the file half of the layout comparison. It reads the ZIP central
/// directory and the metadata entry only, so it costs the same on a 5 GB file as
/// on a 5 KB one, and it splits the total per role — `tree_vectors` is what the
/// shared layout removes T-1 copies of, `shared_vectors` and `shared_row_map`
/// are what it adds back.
///
/// Args:
///     path: The `.kdf` to inspect.
///     max_metadata_bytes: Ceiling on the decoded metadata JSON (default 64 MiB).
///
/// Returns:
///     A dict describing the file: `feature_count`, `dimension`, `scalar_type`,
///     `tree_count`, `descriptor_storage`, `descriptor_block_rows`,
///     `target_chunk_bytes`, `chunks_per_tree`, `nodes_per_tree`,
///     `has_sources`, `file_bytes`, `payload_compressed_bytes`,
///     `payload_decoded_bytes`, and `sections` — a list of per-role dicts with
///     `section`, `entries`, `compressed_bytes` and `decoded_bytes`.
///
/// `file_bytes` exceeds `payload_compressed_bytes` by the ZIP headers and the
/// central directory, which is a real cost at high entry counts and one no
/// per-section total shows.
#[pyfunction]
#[pyo3(signature = (path, *, max_metadata_bytes=None))]
fn kdf_file_summary<'py>(
    py: Python<'py>,
    path: PathBuf,
    max_metadata_bytes: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let limit = max_metadata_bytes.unwrap_or(64 << 20);
    let summary = py.detach(|| kdf_summary(&path, limit)).map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item("feature_count", summary.feature_count)?;
    d.set_item("dimension", summary.dimension)?;
    d.set_item("scalar_type", &summary.scalar_type)?;
    d.set_item("tree_count", summary.tree_count)?;
    d.set_item("descriptor_storage", &summary.descriptor_storage)?;
    d.set_item("descriptor_block_rows", summary.descriptor_block_rows)?;
    d.set_item("target_chunk_bytes", summary.target_chunk_bytes)?;
    d.set_item("chunks_per_tree", summary.chunks_per_tree.clone())?;
    d.set_item("nodes_per_tree", summary.nodes_per_tree.clone())?;
    d.set_item("has_sources", summary.has_sources)?;
    d.set_item("file_bytes", summary.file_bytes)?;
    d.set_item("payload_compressed_bytes", summary.payload_compressed_bytes)?;
    d.set_item("payload_decoded_bytes", summary.payload_decoded_bytes)?;

    let sections = PyList::empty(py);
    for section in &summary.sections {
        let entry = PyDict::new(py);
        entry.set_item("section", &section.section)?;
        entry.set_item("entries", section.entries)?;
        entry.set_item("compressed_bytes", section.compressed_bytes)?;
        entry.set_item("decoded_bytes", section.decoded_bytes)?;
        sections.append(entry)?;
    }
    d.set_item("sections", sections)?;
    Ok(d.unbind())
}

/// Fully verify a `.kdf`, reading every entry.
///
/// Recomputes every digest and checks the structural constraints that only a
/// whole-file pass can see: reachability, ID permutations, cross-tree vector
/// equality and split constraints. Opening a file checks none of these beyond
/// the parts it reads, so a lazily-queried file can hold corruption in a chunk
/// no query ever reached — this is what settles that.
///
/// Args:
///     path: The `.kdf` to verify.
///
/// Returns:
///     A dict with `features`, `trees`, `chunks`, `descriptor_blocks` and
///     `origin_blocks` on success.
///
/// Raises:
///     OSError: The file is malformed, or a digest does not match.
#[pyfunction]
fn verify_kdf<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Py<PyDict>> {
    let verified = py
        .detach(|| {
            sfmtool_core::features::kdforest::verify_kdf::<u8>(
                &path,
                LazyKdForestOptions::default(),
            )
        })
        .map_err(to_py_err)?;
    verification_dict(py, verified)
}

/// Verify the `.sift` files named by a KDF provenance table.
///
/// This is deliberately separate from [`verify_kdf`]: ordinary queries and KDF
/// verification are self-contained and keep working after their source files
/// move or disappear. Call this only when auditing provenance against a live
/// workspace. It checks each referenced SIFT identity, feature bound, and
/// descriptor byte row.
///
/// Args:
///     path: A `.kdf` carrying SIFT source provenance.
///
/// Returns:
///     The same verification-count dict as `verify_kdf`.
///
/// Raises:
///     FileNotFoundError: A referenced workspace or `.sift` file is absent.
///     OSError: A source identity, feature bound, or descriptor does not match.
#[pyfunction]
fn verify_sift_sources<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Py<PyDict>> {
    let verified = py
        .detach(|| {
            sfmtool_core::features::kdforest::verify_sift_sources(
                &path,
                LazyKdForestOptions::default(),
            )
        })
        .map_err(to_py_err)?;
    verification_dict(py, verified)
}

fn verification_dict<'py>(
    py: Python<'py>,
    verified: sfmtool_core::features::kdforest::Verification,
) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("features", verified.features)?;
    d.set_item("trees", verified.trees)?;
    d.set_item("chunks", verified.chunks)?;
    d.set_item("descriptor_blocks", verified.descriptor_blocks)?;
    d.set_item("origin_blocks", verified.origin_blocks)?;
    Ok(d.unbind())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLazyKdForest>()?;
    m.add_function(wrap_pyfunction!(write_kdf, m)?)?;
    m.add_function(wrap_pyfunction!(read_kdf, m)?)?;
    m.add_function(wrap_pyfunction!(kdf_file_summary, m)?)?;
    m.add_function(wrap_pyfunction!(verify_kdf, m)?)?;
    m.add_function(wrap_pyfunction!(verify_sift_sources, m)?)?;
    Ok(())
}
