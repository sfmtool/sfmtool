// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for [`sfmtool_core::features::sift::build_sift_files`], the
//! Rust writer of `.sift` files for the `sfmtool` feature tool.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::features::sift::{
    build_sift_files as core_build_sift_files, BuildSiftFilesError, BuildSiftFilesOptions,
};
use sfmtool_core::Progress;

use super::extract::{feature_options_error, feature_options_from_dict};

/// Extract SIFT features with the sfmtool backend and write each image's
/// `.sift` file, skipping an image whose `.sift` is at least as new as it.
///
/// The Rust counterpart of `sfm sift --extract` for a workspace whose feature
/// tool is `sfmtool`: each file goes to
/// `<image dir>/<feature_prefix_dir>/<image name>.sift` and holds what the
/// Python `sfmtool` backend writes for the same image and options. Several
/// images are extracted at once; `concurrent_images` sets how many, and
/// `None` takes `SFMTOOL_SIFT_EXTRACT_WORKERS` or a default from the core
/// count and memory. The GIL is released throughout.
///
/// Args:
///     image_paths: the image files.
///     feature_options: the workspace's `feature_options` dict for the
///         sfmtool tool (see `extract_sift` for the keys). It is recorded in
///         each file as given.
///     feature_prefix_dir: the workspace's `feature_prefix_dir`, relative to
///         each image's directory.
///     concurrent_images: how many images to extract at once, or `None`.
///
/// Returns:
///     Tuple of (sift_paths, extracted, skipped): each image's `.sift` path
///     in input order, how many files were written and how many images were
///     up to date.
///
/// Raises:
///     ValueError: if `feature_options` is invalid.
///     OSError: if an image cannot be read or a file cannot be written; the
///         first failing image in input order is named.
#[pyfunction]
#[pyo3(signature = (image_paths, feature_options, feature_prefix_dir, *, concurrent_images=None))]
pub fn build_sift_files(
    py: Python<'_>,
    image_paths: Vec<PathBuf>,
    feature_options: &Bound<'_, PyDict>,
    feature_prefix_dir: PathBuf,
    concurrent_images: Option<usize>,
) -> PyResult<(Vec<PathBuf>, usize, usize)> {
    let options = BuildSiftFilesOptions {
        feature_options: feature_options_from_dict(feature_options)?,
        feature_prefix_dir,
        concurrent_images,
    };
    let report = py
        .detach(|| core_build_sift_files(&image_paths, &options, &Progress::none()))
        .map_err(|e| match e {
            BuildSiftFilesError::FeatureOptions(e) => feature_options_error(e),
            other => pyo3::exceptions::PyOSError::new_err(other.to_string()),
        })?;
    Ok((report.sift_paths, report.extracted, report.skipped))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_sift_files, m)?)?;
    Ok(())
}
