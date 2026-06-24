// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! sfmtool SIFT detection and extraction bindings.

use pyo3::prelude::*;

pub mod extract;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    extract::register(m)?;
    Ok(())
}
