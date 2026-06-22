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
// Headline types reachable as `sfmtool_core::Foo`.
pub use camera::intrinsics::{CameraIntrinsics, CameraModel};
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
