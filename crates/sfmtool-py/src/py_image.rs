// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for lightweight image inspection.

use std::path::PathBuf;

use pyo3::prelude::*;

/// Read an image file's `(width, height)` from its header alone.
///
/// Decodes only the format header — JPEG `SOF`, PNG `IHDR`, and so on — never
/// the pixel data, so it stays cheap on large images. The dimensions are the
/// ones stored in the file; EXIF orientation is not applied.
#[pyfunction]
pub fn image_dimensions(path: PathBuf) -> PyResult<(u32, u32)> {
    image::image_dimensions(&path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
