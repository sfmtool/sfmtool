// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for frame pair matching: one frame's features with its
//! forest, and the motion statistics of a pair of frames
//! (``sfmtool.analysis.FrameFeatures`` / ``match_frame_pair``; see
//! ``specs/core/analysis/frame-pair-matching.md``).

use std::path::PathBuf;

use numpy::ndarray::Ix2;
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::analysis::pick_images::{
    match_frame_pair as core_match_frame_pair, FrameFeatures, FramePairOptions,
};
use sfmtool_core::features::kdforest::KdForestParams;
use sfmtool_sift_format::{read_sift_features, read_sift_metadata, SiftMetadata, DESCRIPTOR_DIM};

use crate::csr_args::float_array_f32;
use crate::fileio::sift::PySiftFile;
use crate::format_errors::read_err;
use crate::helpers::value_err;
use crate::spatial::kdforest::{extract_u8_2d, resolve_forest_params};

/// One frame's features, with the KD-forest over its descriptors built once,
/// so that a frame matched with several others is indexed once.
///
/// Args:
///     positions_xy: (N, 2) float32 feature positions in pixels, as a
///         ``.sift`` file stores them. A float64 array is accepted and cast.
///     descriptors: (N, 128) uint8 descriptors.
///     width: The image's width in pixels.
///     height: The image's height in pixels.
///     preset: The forest preset, "balanced" (default), "fast" or
///         "accurate", as for ``sfmtool.spatial.KdForest``.
///     num_trees: Override the preset's number of trees.
///     leaf_size: Override the preset's leaf size.
///     max_leaf_checks: Override the preset's per-query budget, which
///         ``match_frame_pair`` uses for the queries against this frame's
///         forest unless it is given its own.
///     seed: Override the preset's seed for building the trees.
///
/// Raises:
///     TypeError: an array of the wrong dtype.
///     ValueError: arrays of the wrong shape or that disagree, a position
///         that is not finite, an image side of 0 or a forest parameter of 0.
///
/// See specs/core/analysis/frame-pair-matching.md.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented `Args:` continuation paragraphs read as Markdown indented code
// blocks, which rustdoc then tries to parse as Rust.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyclass(name = "FrameFeatures", module = "sfmtool.analysis", frozen)]
pub struct PyFrameFeatures {
    inner: FrameFeatures,
}

/// The forest parameters a `FrameFeatures` constructor was given.
fn forest_params(
    preset: Option<&str>,
    num_trees: Option<usize>,
    leaf_size: Option<usize>,
    max_leaf_checks: Option<usize>,
    seed: Option<u64>,
) -> PyResult<KdForestParams> {
    resolve_forest_params(
        preset,
        "balanced",
        num_trees,
        leaf_size,
        max_leaf_checks,
        seed,
    )
}

#[pymethods]
impl PyFrameFeatures {
    #[new]
    #[pyo3(signature = (positions_xy, descriptors, width, height, *, preset=None, num_trees=None,
                        leaf_size=None, max_leaf_checks=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        positions_xy: &Bound<'_, PyAny>,
        descriptors: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        preset: Option<&str>,
        num_trees: Option<usize>,
        leaf_size: Option<usize>,
        max_leaf_checks: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let forest = forest_params(preset, num_trees, leaf_size, max_leaf_checks, seed)?;
        let positions = float_array_f32::<Ix2>(
            positions_xy,
            "positions_xy must be a (N, 2) float32 or float64 array",
        )?;
        if positions.shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "positions_xy must have shape (N, 2), got {:?}",
                positions.shape
            )));
        }
        let descriptors = extract_u8_2d(descriptors, "descriptors")?;
        if descriptors.shape()[1] != DESCRIPTOR_DIM {
            return Err(PyValueError::new_err(format!(
                "descriptors must have shape (N, {DESCRIPTOR_DIM}), got {:?}",
                descriptors.shape()
            )));
        }
        let descriptors = to_contiguous!(descriptors).into_owned();
        let positions: Vec<[f32; 2]> = positions.data.as_chunks::<2>().0.to_vec();
        let inner = py
            .detach(move || FrameFeatures::new(positions, descriptors, width, height, forest))
            .map_err(value_err)?;
        Ok(Self { inner })
    }

    /// The features of a ``.sift`` file, with the image size its metadata
    /// records.
    ///
    /// Args:
    ///     sift: A ``.sift`` path (str or Path) or a ``SiftFile``.
    ///     preset, num_trees, leaf_size, max_leaf_checks, seed: The forest,
    ///         as for the constructor.
    ///
    /// Raises:
    ///     TypeError: ``sift`` is neither a path nor a ``SiftFile``.
    ///     OSError: the file cannot be read, as ``sfmtool.fileio.read_sift``
    ///         raises it.
    ///     ValueError: a forest parameter of 0, or a file whose features and
    ///         metadata disagree.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[staticmethod]
    #[pyo3(signature = (sift, *, preset=None, num_trees=None, leaf_size=None,
                        max_leaf_checks=None, seed=None))]
    fn from_sift(
        py: Python<'_>,
        sift: &Bound<'_, PyAny>,
        preset: Option<&str>,
        num_trees: Option<usize>,
        leaf_size: Option<usize>,
        max_leaf_checks: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let forest = forest_params(preset, num_trees, leaf_size, max_leaf_checks, seed)?;
        let (path, metadata): (PathBuf, Option<SiftMetadata>) =
            if let Ok(file) = sift.cast::<PySiftFile>() {
                let file = file.get();
                (
                    file.sift_path().to_path_buf(),
                    Some(file.sift_metadata().clone()),
                )
            } else if let Ok(path) = sift.extract::<PathBuf>() {
                (path, None)
            } else {
                return Err(PyTypeError::new_err(
                    "sift must be a .sift path (str or Path) or a SiftFile",
                ));
            };
        let features = py
            .detach(|| {
                let metadata = match metadata {
                    Some(metadata) => metadata,
                    None => read_sift_metadata(&path)?.1,
                };
                read_sift_features(&path).map(|features| (features, metadata))
            })
            .map_err(read_err)?;
        let inner = py
            .detach(move || FrameFeatures::from_sift(features.0, &features.1, forest))
            .map_err(value_err)?;
        Ok(Self { inner })
    }

    /// The image's width in pixels.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    /// The image's height in pixels.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    /// The number of features.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameFeatures({} features, {} x {})",
            self.inner.len(),
            self.inner.width(),
            self.inner.height()
        )
    }
}

/// Match two frames' features and measure the motion between them.
///
/// The matches are the mutual best matches of a k = 2 query each way that
/// pass Lowe's ratio test both ways. Options left as ``None`` take the
/// defaults of the Rust ``FramePairOptions``: ``ratio`` 0.8, ``min_matches``
/// 30, the queried forest's own ``max_leaf_checks``,
/// ``fundamental_error_per_960px`` 1.0, ``homography_error_per_960px`` 2.0
/// and ``seed`` 0.
///
/// Args:
///     a: The first frame's ``FrameFeatures``.
///     b: The second frame's, of the same image size.
///     ratio: Lowe's ratio on descriptor distances, in (0, 1].
///     min_matches: The fewest matches the shares are computed from, at
///         least 7.
///     max_leaf_checks: The per-query budget of distance computations.
///     fundamental_error_per_960px: The fundamental matrix's inlier bound,
///         in pixels per 960 pixels of the long side.
///     homography_error_per_960px: The homography's inlier bound, likewise.
///     seed: The RANSAC seed.
///
/// Returns:
///     A dict: ``matches`` (int), ``step_percent`` (float, the median
///     displacement in percent of the long side, NaN with no matches),
///     ``f_inlier_share``, ``h_over_f`` and ``rotation_error`` (float, or
///     None under ``min_matches``).
///
/// Raises:
///     ValueError: an option out of range, or frames of different sizes.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[pyo3(signature = (a, b, *, ratio=None, min_matches=None, max_leaf_checks=None,
                    fundamental_error_per_960px=None, homography_error_per_960px=None,
                    seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn match_frame_pair<'py>(
    py: Python<'py>,
    a: PyRef<'py, PyFrameFeatures>,
    b: PyRef<'py, PyFrameFeatures>,
    ratio: Option<f32>,
    min_matches: Option<usize>,
    max_leaf_checks: Option<usize>,
    fundamental_error_per_960px: Option<f64>,
    homography_error_per_960px: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    let defaults = FramePairOptions::default();
    let options = FramePairOptions {
        ratio: ratio.unwrap_or(defaults.ratio),
        min_matches: min_matches.unwrap_or(defaults.min_matches),
        max_leaf_checks: max_leaf_checks.or(defaults.max_leaf_checks),
        fundamental_error_per_960px: fundamental_error_per_960px
            .unwrap_or(defaults.fundamental_error_per_960px),
        homography_error_per_960px: homography_error_per_960px
            .unwrap_or(defaults.homography_error_per_960px),
        seed: seed.unwrap_or(defaults.seed),
    };
    let (a, b) = (&a.inner, &b.inner);
    let stats = py
        .detach(|| core_match_frame_pair(a, b, &options))
        .map_err(value_err)?;
    let out = PyDict::new(py);
    out.set_item("matches", stats.matches)?;
    out.set_item("step_percent", stats.step_percent)?;
    out.set_item("f_inlier_share", stats.f_inlier_share)?;
    out.set_item("h_over_f", stats.h_over_f)?;
    out.set_item("rotation_error", stats.rotation_error)?;
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrameFeatures>()?;
    m.add_function(wrap_pyfunction!(match_frame_pair, m)?)?;
    Ok(())
}
