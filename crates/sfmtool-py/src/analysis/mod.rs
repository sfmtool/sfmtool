// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction-analysis bindings: pose/track operations, least-squares +
//! RANSAC alignment, point correspondence, batch triangulation, epipolar curves,
//! image-pair graph construction, image-space observation adjacency and the
//! surfel normals fitted over it, per-image observation coverage grids, and the
//! cluster match census.

use pyo3::prelude::*;

pub mod adjacency_surfel_normals;
pub mod cluster_census;
pub mod core;
pub mod epipolar;
pub mod image_pair_graph;
pub mod observation_adjacency;
pub mod observation_coverage;
pub mod triangulation;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    core::register(m)?;
    triangulation::register(m)?;
    epipolar::register(m)?;
    image_pair_graph::register(m)?;
    observation_adjacency::register(m)?;
    observation_coverage::register(m)?;
    adjacency_surfel_normals::register(m)?;
    cluster_census::register(m)?;
    Ok(())
}
