// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! sfmtool SIFT detection and extraction bindings, the `.sift` file
//! builder, the conversion of a version 1 `.sift` file to version 2, and an
//! image's grey conversion and sharpness readings.

use pyo3::prelude::*;

pub mod convert;
pub mod extract;
pub mod files;
pub mod image_readings;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    convert::register(m)?;
    extract::register(m)?;
    files::register(m)?;
    image_readings::register(m)?;
    Ok(())
}
