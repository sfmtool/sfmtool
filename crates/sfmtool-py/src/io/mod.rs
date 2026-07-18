// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! File-format I/O bindings: `.sfmr`, `.sift`, `.matches`, `.camrig`, the
//! COLMAP binary + SQLite database formats, and header-only image inspection.
//!
//! Each child file owns its own `pub fn register`; this module just chains them
//! into the `_sfmtool.io` Python submodule wired up by `lib.rs`.

use pyo3::prelude::*;

pub mod camrig;
pub mod colmap_binary;
pub mod colmap_db;
pub mod image;
pub mod matches;
pub mod sfmr;
pub mod sift;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    sfmr::register(m)?;
    image::register(m)?;
    sift::register(m)?;
    matches::register(m)?;
    camrig::register(m)?;
    colmap_binary::register(m)?;
    colmap_db::register(m)?;
    Ok(())
}
