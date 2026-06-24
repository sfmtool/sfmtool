// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for spherical tile rigs, per-tile source stacks, and uniform
//! sphere-point generation.

use pyo3::prelude::*;

pub mod sphere_points;
pub mod tile_rig;
pub mod tile_source_stack;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    sphere_points::register(m)?;
    tile_rig::register(m)?;
    tile_source_stack::register(m)?;
    Ok(())
}
