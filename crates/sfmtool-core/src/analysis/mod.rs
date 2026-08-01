// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction analysis: alignment between reconstructions, points-at-infinity
//! discovery, covisibility / frustum-intersection image-pair graphs, image-space
//! observation adjacency between points and the surfel normals fitted over it,
//! per-image occupancy grids over the observations' footprints, per-point
//! triangulation inspection, and the cluster match census.

pub mod adjacency_surfel_normals;
pub mod alignment;
pub mod cluster_census;
pub mod image_pair_graph;
pub mod infinity;
pub mod observation_adjacency;
pub mod observation_coverage;
pub mod point_inspect;
