// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for [`sfmtool_core::features::sift::build_sift_files`], the
//! Rust writer of `.sift` files for the `sfmtool` feature tool, and for
//! [`sfmtool_core::features::sift::default_concurrent_images_for_dimensions`], how
//! many images both extractors run at once.

use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::features::sift::{
    build_sift_files as core_build_sift_files,
    default_concurrent_images_for_dimensions as core_default_concurrent_images_for_dimensions,
    BuildSiftFilesError, BuildSiftFilesOptions,
};
use sfmtool_core::progress::{Event, Level};
use sfmtool_core::Progress;

use super::extract::{feature_options_error, feature_options_from_dict, parse_sift_params};

/// Extract SIFT features with the sfmtool backend and write each image's
/// `.sift` file, skipping an image whose `.sift` is at least as new as it.
///
/// The Rust counterpart of `sfm sift --extract` for a workspace whose feature
/// tool is `sfmtool`: each file goes to
/// `<image dir>/<feature_prefix_dir>/<image name>.sift` and holds what the
/// Python `sfmtool` backend writes for the same image and options, a `.sift`
/// version 2 file with the image's readings. When there is something to
/// write, version 1 settings (options with no `image_to_gray`) are refused.
/// Several
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
///     ValueError: if `feature_options` is invalid or holds version 1
///         settings.
///     OSError: if an image cannot be read, its readings are not finite, or
///         a file cannot be written; the first failing image in input order
///         is named.
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
            e @ BuildSiftFilesError::Version1Settings => {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            }
            other => pyo3::exceptions::PyOSError::new_err(other.to_string()),
        })?;
    Ok((report.sift_paths, report.extracted, report.skipped))
}

/// How many images of `width` x `height` pixels to decode and extract at once.
///
/// The `SFMTOOL_SIFT_EXTRACT_WORKERS` environment variable when it holds a
/// whole number, raised to 1 if lower; otherwise the core count, capped so
/// that that many images fit in half of physical memory at 192 bytes per
/// source pixel for an image `params` doubles and 64 for one it does not, or
/// at most 4 when either side is 0. Always at least 1. This is the count
/// `build_sift_files` uses when `concurrent_images` is `None`, and the Python
/// `sfmtool` backend of `sfm sift` sizes its pool with it from the first
/// image.
///
/// Args:
///     width: the image's width in pixels.
///     height: the image's height in pixels.
///     params: the extraction's options, the dict `extract_sift` takes, or
///         `None` for its defaults, which double every image. Only whether
///         they double an image of this size (`double_image` and
///         `double_image_below_long_side`) changes the count, but the whole
///         dict is checked as `extract_sift` checks it.
///
/// Returns:
///     The number of images, at least 1.
///
/// Raises:
///     ValueError: if `params` is invalid.
///
/// Warns:
///     UserWarning: when `SFMTOOL_SIFT_EXTRACT_WORKERS` is set to something
///         other than a whole number, which is then ignored.
#[pyfunction]
#[pyo3(signature = (width, height, params=None))]
pub fn default_concurrent_images(
    py: Python<'_>,
    width: u32,
    height: u32,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<usize> {
    let params = parse_sift_params(params)?;
    let warnings = Mutex::new(Vec::<String>::new());
    let sink = |event: Event<'_>| {
        if let Event::Message {
            level: Level::Warn,
            text,
            ..
        } = event
        {
            warnings
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(text.to_owned());
        }
    };
    let count = core_default_concurrent_images_for_dimensions(
        Some((width, height)),
        &params,
        &Progress::to(&sink),
    );
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    for text in warnings.into_inner().unwrap_or_else(|e| e.into_inner()) {
        let message = std::ffi::CString::new(text)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        PyErr::warn(py, &category, &message, 1)?;
    }
    Ok(count)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_sift_files, m)?)?;
    m.add_function(wrap_pyfunction!(default_concurrent_images, m)?)?;
    Ok(())
}
