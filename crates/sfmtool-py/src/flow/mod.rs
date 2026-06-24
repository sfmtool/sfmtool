// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Optical-flow and image-warping bindings.

use pyo3::prelude::*;

pub mod optical;
pub mod warp;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    optical::register(m)?;
    warp::register(m)?;
    Ok(())
}
