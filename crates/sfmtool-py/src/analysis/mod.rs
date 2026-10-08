// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction-analysis bindings: pose/track operations, least-squares +
//! RANSAC alignment, point correspondence, batch triangulation and the
//! point-or-bearing test, epipolar curves,
//! image-pair graph construction, image-space observation adjacency and the
//! surfel normals fitted over it, the patch normals a refined cluster's cell
//! displacements give once poses exist, per-image observation coverage grids, the
//! per-image keypoint reach enumeration and the rule that retires a coarse
//! observation a finer one covers, the cluster match census, the per-cluster
//! feature radius and the coarsest-N cut over it, and the join that names the
//! selection clusters a member left behind.

use pyo3::prelude::*;

pub mod adjacency_surfel_normals;
pub mod cell_plane_normals;
pub mod cluster_census;
pub mod cluster_radii;
pub mod core;
pub mod covered_by_finer;
pub mod epipolar;
pub mod image_pair_graph;
pub mod keypoint_reach;
pub mod observation_adjacency;
pub mod observation_coverage;
pub mod point_or_bearing;
pub mod source_clusters;
pub mod triangulation;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    core::register(m)?;
    triangulation::register(m)?;
    point_or_bearing::register(m)?;
    epipolar::register(m)?;
    image_pair_graph::register(m)?;
    keypoint_reach::register(m)?;
    covered_by_finer::register(m)?;
    observation_adjacency::register(m)?;
    observation_coverage::register(m)?;
    source_clusters::register(m)?;
    adjacency_surfel_normals::register(m)?;
    cell_plane_normals::register(m)?;
    cluster_census::register(m)?;
    cluster_radii::register(m)?;
    Ok(())
}
