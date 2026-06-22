// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Spherical-tile rigs, consensus atlases, photometric RANSAC, and the sphere-point
//! sampling and per-tile source stacks the rigs operate on.

pub mod consensus_atlas;
pub mod per_tile_source_stack;
pub mod photometric_ransac;
pub mod sphere_points;
pub mod tile_rig;

pub use consensus_atlas::{
    render_consensus_atlas, ConsensusAtlasBatchError, ConsensusAtlasBatchParams,
    ConsensusAtlasReport,
};
