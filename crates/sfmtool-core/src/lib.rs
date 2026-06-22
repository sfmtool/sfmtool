// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Core data structures and algorithms for sfmtool.
//!
//! This crate is organized into topic groups, each owning a coherent layer
//! of the SfM pipeline:
//!
//! - [`geometry`] — rotations, rigid/SE(3) transforms, ray geometry
//! - [`camera`] — camera model, intrinsics, distortion, projection, image warping
//! - [`reconstruction`] — `.sfmr` data structures and per-point operations
//!   (triangulation, filtering, covisibility, point correspondence/inspection)
//! - [`features`] — SIFT, descriptor / cluster / flow matching, KD-forest
//! - [`analysis`] — alignment between reconstructions and points-at-infinity discovery
//! - [`spherical`] — spherical-tile rigs, consensus atlases, photometric RANSAC
//! - [`patch`] — patch clouds and patch-normal refinement
//! - [`spatial`] — generic KD-tree point-cloud utility used across groups
//!
//! File-format I/O is provided by the sibling crates `sift-format`,
//! `sfmr-format`, `matches-format`, and `camrig-format`.

pub mod analysis;
pub mod camera;
pub mod features;
pub mod geometry;
pub mod patch;
pub mod reconstruction;
pub mod spatial;
pub mod spherical;

// ── Top-level type re-exports ──────────────────────────────────────────
//
// Headline types that have always been reachable as `sfmtool_core::Foo`
// (predating the group structure). Kept unconditionally — these are the
// crate's public API, not a transition concession.
pub use camera::camera_intrinsics::{CameraIntrinsics, CameraModel};
pub use camera::Camera;
pub use geometry::rigid_transform::RigidTransform;
pub use geometry::rot_quaternion::RotQuaternion;
pub use geometry::se3_transform::Se3Transform;
pub use reconstruction::{
    Point3D, ReconstructionError, SfmrImage, SfmrReconstruction, TrackObservation,
};
pub use spherical::consensus_atlas::{
    render_consensus_atlas, ConsensusAtlasBatchError, ConsensusAtlasBatchParams,
    ConsensusAtlasReport,
};

// ── Transition shims ───────────────────────────────────────────────────
//
// Every module that lived flat at the crate root before the regroup is
// re-exported here under its old name. Downstream crates (`sfmtool-py`,
// `sfm-explorer`, `sfmr-colmap`) and intra-crate `use crate::<flat>::*`
// paths keep compiling without edits.
//
// Phase 2 will sweep callers to the new paths and delete this block.
pub use analysis::{alignment, infinity};
pub use camera::{
    camera_intrinsics, distortion, epipolar, frustum, rectification, remap, warp_map,
};
pub use features::{cluster_match, feature_match, kdforest, optical_flow, sift};
pub use geometry::{
    rigid_transform, rot_quaternion, rotation, se3_transform, transform, viewing_angle,
};
pub use patch::{patch_cloud, patch_normal_refine};
pub use reconstruction::{
    filter, image_pair_graph, point_correspondence, point_inspect, triangulation,
};
pub use spherical::{
    consensus_atlas, per_spherical_tile_source_stack, photometric_ransac, sphere_points,
    spherical_tile_rig,
};
