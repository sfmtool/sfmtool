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
//!   (triangulation, filtering, point correspondence)
//! - [`features`] — SIFT, descriptor / cluster / flow matching, KD-forest
//! - [`analysis`] — alignment, points-at-infinity discovery, covisibility /
//!   frustum image-pair graphs, per-point triangulation inspection
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

// Headline types reachable as `sfmtool_core::Foo`, leaning on each group's
// own facade re-exports (`camera::{Camera, CameraIntrinsics, CameraModel}`,
// `geometry::{RigidTransform, RotQuaternion, Se3Transform}`, etc.).
pub use camera::{Camera, CameraIntrinsics, CameraModel};
pub use geometry::{RigidTransform, RotQuaternion, Se3Transform};
pub use reconstruction::{
    ObservationSource, Point3D, ReconstructionError, SfmrImage, SfmrReconstruction,
    TrackObservation,
};
pub use spherical::{
    render_consensus_atlas, ConsensusAtlasBatchError, ConsensusAtlasBatchParams,
    ConsensusAtlasReport,
};
