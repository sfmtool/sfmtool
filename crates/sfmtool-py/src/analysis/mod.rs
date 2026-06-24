// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction-analysis bindings: pose/track operations, Kabsch + RANSAC
//! alignment, point correspondence, batch triangulation, epipolar curves, and
//! image-pair graph construction.

use pyo3::prelude::*;

pub mod core;
pub mod epipolar;
pub mod image_pair_graph;
pub mod triangulation;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    core::register(m)?;
    triangulation::register(m)?;
    epipolar::register(m)?;
    image_pair_graph::register(m)?;
    Ok(())
}
