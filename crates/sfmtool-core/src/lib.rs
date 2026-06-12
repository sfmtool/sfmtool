// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Core data structures and algorithms for sfmtool.
//!
//! This crate provides:
//! - SfM reconstruction data structures
//! - Camera representation
//! - Geometric algorithms (epipolar, rectification, alignment, etc.)
//!
//! File format I/O is provided by the `sift-format` and `sfmr-format` crates.

pub mod alignment;
pub mod camera;
pub mod camera_intrinsics;
pub mod cluster_match;
pub mod consensus_atlas;
pub mod distortion;
pub mod epipolar;
pub mod feature_match;
pub mod filter;
pub mod frustum;
pub mod image_pair_graph;
pub mod infinity;
pub mod kdforest;
pub mod optical_flow;
pub mod patch_cloud;
pub mod patch_normal_refine;
pub mod per_spherical_tile_source_stack;
pub mod photometric_ransac;
pub mod point_correspondence;
pub mod point_inspect;
pub mod reconstruction;
pub mod rectification;
pub mod remap;
pub mod rigid_transform;
pub mod rot_quaternion;
pub mod rotation;
pub mod se3_transform;
pub mod sift;
pub mod spatial;
pub mod sphere_points;
pub mod spherical_tile_rig;
pub mod transform;
pub mod triangulation;
pub mod viewing_angle;
pub mod warp_map;

pub use camera::Camera;
pub use camera_intrinsics::{CameraIntrinsics, CameraModel};
pub use consensus_atlas::{
    render_consensus_atlas, ConsensusAtlasBatchError, ConsensusAtlasBatchParams,
    ConsensusAtlasReport,
};
pub use reconstruction::{
    Point3D, ReconstructionError, SfmrImage, SfmrReconstruction, TrackObservation,
};
pub use rigid_transform::RigidTransform;
pub use rot_quaternion::RotQuaternion;
pub use se3_transform::Se3Transform;
