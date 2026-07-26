// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction-analysis bindings: pose/track operations, least-squares +
//! RANSAC alignment, point correspondence, batch triangulation, epipolar curves,
//! image-pair graph construction, and the cluster match census.

use pyo3::prelude::*;

pub mod cluster_census;
pub mod core;
pub mod epipolar;
pub mod image_pair_graph;
pub mod triangulation;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    core::register(m)?;
    triangulation::register(m)?;
    epipolar::register(m)?;
    image_pair_graph::register(m)?;
    cluster_census::register(m)?;
    Ok(())
}
