// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for [`sfmtool_core::features::sift::Version2SiftFolder`],
//! which converts a version 1 `.sift` file to version 2 without extracting it
//! again.
//!
//! A `.sift` file that cannot be read or written raises through the
//! format-error table in [`crate::format_errors`], as the `sfmtool.fileio`
//! readers and writers do.

use std::path::{Path, PathBuf};

use pyo3::exceptions::{PyFileNotFoundError, PyInterruptedError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::features::sift::{ConvertSiftFileError, Version2SiftFolder};
use sfmtool_core::Progress;

use crate::format_errors::{fault_to_py, Direction, FormatError};
use crate::helpers::serde_to_py;

/// The Python exception for `e`.
fn convert_err(e: ConvertSiftFileError) -> PyErr {
    let message = e.to_string();
    match e {
        ConvertSiftFileError::Read { source, .. } => {
            fault_to_py(source.fault(), message, Direction::Read)
        }
        ConvertSiftFileError::Write { source, .. } => {
            fault_to_py(source.fault(), message, Direction::Write)
        }
        ConvertSiftFileError::NotVersion1 { .. } | ConvertSiftFileError::FeatureOptions { .. } => {
            PyValueError::new_err(message)
        }
        ConvertSiftFileError::ImageMissing { .. } => PyFileNotFoundError::new_err(message),
        ConvertSiftFileError::Cancelled => PyInterruptedError::new_err(message),
        ConvertSiftFileError::Version1Files(_)
        | ConvertSiftFileError::ListFolder { .. }
        | ConvertSiftFileError::CreateFolder { .. }
        | ConvertSiftFileError::ImageUnreadable { .. }
        | ConvertSiftFileError::ImageChanged { .. }
        | ConvertSiftFileError::ImageSize { .. }
        | ConvertSiftFileError::Readings { .. } => PyOSError::new_err(message),
    }
}

/// A folder that version 1 `.sift` files are converted into, checked to hold
/// no version 1 `.sift` file.
///
/// Making one checks the folder with `version_1_sift_files`, the check every
/// writer of version 2 files makes, and makes the folder when it is not
/// there. The check opens every `.sift` file in the folder, so it is made
/// once, here, and not for each file converted into it.
///
/// Args:
///     folder: the folder, a `str` or `pathlib.Path`.
///
/// Raises:
///     OSError: if the folder holds version 1 `.sift` files (the message is
///         `str` of `version_1_sift_files`' result), or is there but cannot
///         be listed, or cannot be made.
#[pyclass(name = "Version2SiftFolder", module = "sfmtool.sift", frozen)]
pub struct PyVersion2SiftFolder {
    inner: Version2SiftFolder,
}

#[pymethods]
impl PyVersion2SiftFolder {
    #[new]
    fn new(py: Python<'_>, folder: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| Version2SiftFolder::create(&folder))
            .map_err(convert_err)?;
        Ok(Self { inner })
    }

    /// The folder, a `pathlib.Path`.
    #[getter]
    fn path(&self) -> &Path {
        self.inner.path()
    }

    /// Convert a version 1 `.sift` file to a version 2 file of the same name
    /// in this folder, without extracting its features again.
    ///
    /// The positions, affine shapes, descriptors and thumbnail are copied byte
    /// for byte, and every metadata field but the version is kept.
    /// `feature_options` becomes `upgrade_feature_options` of the file's own,
    /// and the metadata gains the image's `sharpness` and
    /// `gradient_structure_tensor_xx_yy_xy`, taken from the image file through
    /// the upgraded options' `image_to_gray`, after the image is checked
    /// against the file's `image_file_xxh128`. The source file is not
    /// changed. The GIL is released throughout.
    ///
    /// Args:
    ///     sift_path: the version 1 `.sift` file.
    ///     image_path: the image file it was extracted from.
    ///
    /// Returns:
    ///     A dict with `path`, the version 2 file (a `pathlib.Path`), and
    ///     `source_content_hash` and `content_hash`, the
    ///     `content_hash.json` dicts of the version 1 and the version 2 file
    ///     (`metadata_xxh128`, `feature_tool_xxh128`, `content_xxh128`).
    ///
    /// Raises:
    ///     ValueError: if the file is not version 1, or its feature options
    ///         cannot be upgraded: an `sfmtool-undistort` file, whose message
    ///         says to run `sfm undistort` again, a feature tool other than
    ///         `sfmtool`, `colmap` and `opencv`, or a `gray_formula` that does
    ///         not parse.
    ///     FileNotFoundError: if the `.sift` file or the image is not there.
    ///     OSError: if the `.sift` file cannot be read or written, or the
    ///         image cannot be read or decoded, does not hash to the recorded
    ///         `image_file_xxh128`, decodes to another size than recorded, or
    ///         has readings that are not finite.
    fn convert_version_1_sift_file<'py>(
        &self,
        py: Python<'py>,
        sift_path: PathBuf,
        image_path: PathBuf,
    ) -> PyResult<Bound<'py, PyDict>> {
        let converted = py
            .detach(|| {
                self.inner
                    .convert_version_1_sift_file(&sift_path, &image_path, &Progress::none())
            })
            .map_err(convert_err)?;
        let dict = PyDict::new(py);
        dict.set_item("path", converted.path)?;
        dict.set_item(
            "source_content_hash",
            serde_to_py(py, &converted.source_content_hash)?,
        )?;
        dict.set_item("content_hash", serde_to_py(py, &converted.content_hash)?)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "Version2SiftFolder({:?})",
            self.inner.path().display().to_string()
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVersion2SiftFolder>()?;
    Ok(())
}
