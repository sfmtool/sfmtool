// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Feature-matching bindings: descriptor, image-pair, sweep, and
//! background-floor cluster matching.
//!
//! Each child file owns its own `pub fn register`; this module just chains
//! them into the `sfmtool.matching` Python submodule wired up by `lib.rs`.

use pyo3::prelude::*;

pub mod cluster;
pub mod descriptor;
pub mod image;
pub mod sweep;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    descriptor::register(m)?;
    image::register(m)?;
    sweep::register(m)?;
    cluster::register(m)?;
    Ok(())
}
