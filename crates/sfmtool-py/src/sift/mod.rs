// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! sfmtool SIFT detection and extraction bindings, and the `.sift` file
//! builder.

use pyo3::prelude::*;

pub mod extract;
pub mod files;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    extract::register(m)?;
    files::register(m)?;
    Ok(())
}
