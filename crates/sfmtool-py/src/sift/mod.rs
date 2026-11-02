// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! sfmtool SIFT detection and extraction bindings, the `.sift` file
//! builder, and an image's grey conversion and sharpness readings.

use pyo3::prelude::*;

pub mod extract;
pub mod files;
pub mod image_readings;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    extract::register(m)?;
    files::register(m)?;
    image_readings::register(m)?;
    Ok(())
}
