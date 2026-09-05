// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction analysis: alignment between reconstructions, points-at-infinity
//! discovery, covisibility / frustum-intersection image-pair graphs, image-space
//! observation adjacency between points and the surfel normals fitted over it,
//! per-image occupancy grids over the observations' footprints, per-point
//! triangulation inspection, the cluster match census, the per-cluster feature
//! radius and the coarsest-N cut over it, and the join that names the selection
//! clusters a member left behind.

pub mod adjacency_surfel_normals;
pub mod alignment;
pub mod cluster_census;
pub mod cluster_radii;
pub mod image_pair_graph;
pub mod infinity;
pub mod observation_adjacency;
pub mod observation_coverage;
pub mod point_inspect;
pub mod source_clusters;
