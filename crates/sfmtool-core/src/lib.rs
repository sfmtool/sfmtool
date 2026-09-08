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
//!   frustum image-pair graphs, per-point triangulation inspection, cross-group
//!   consistency census
//! - [`spherical`] — spherical-tile rigs, consensus atlases, photometric RANSAC
//! - [`patch`] — patch clouds and patch-normal refinement
//! - [`spatial`] — generic KD-tree point-cloud utility used across groups, and
//!   the per-image keypoint reach enumeration in the pixel domain
//!
//! File-format I/O is provided by the sibling crates `sift-format`,
//! `sfmr-format`, `matches-format`, and `camrig-format`.

pub mod analysis;
pub mod camera;
pub mod features;
pub mod geometry;
pub mod numeric;
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
    EditError, EditedReconstruction, ImageTable, ObservationSource, Point3D,
    PointConstraintColumns, PointRecord, PointSet, PointView, ReconstructionError,
    RecordObservation, RowMap, SfmrImage, SfmrReconstruction, TrackObservation,
};
/// Re-exported so consumers of [`ImageTable::thumbnails_y_x_rgb`] can
/// size buffers from the same constant the format pins, without depending on
/// `sfmr-format` directly.
pub use sfmr_format::THUMBNAIL_SIZE;
/// Re-exported for the same reason, one level up: [`ImageTable::rig_frame_data`]
/// is a public field whose type nothing downstream could otherwise name, so a
/// consumer wanting to read — or build — a rig had to depend on `sfmr-format`
/// itself to say what it was holding.
pub use sfmr_format::{FramesMetadata, RigDefinition, RigFrameData, RigsMetadata};
/// Re-exported alongside [`PointConstraintColumns`], whose entries are these
/// codes: a consumer reading or building the constraint columns needs to name
/// them without depending on `sfmr-format` itself.
pub use sfmr_format::{
    NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED,
};

// `.sfmr` thumbnails are copied verbatim out of the per-image `.sift` files, so
// the two formats' thumbnail extents must agree. Neither format crate depends
// on the other — this crate is the first place both are visible, which makes it
// the only place the agreement can be enforced at compile time rather than
// discovered as a shape mismatch at runtime.
const _: () = assert!(sfmr_format::THUMBNAIL_SIZE == sift_format::THUMBNAIL_SIZE);
pub use spherical::{
    render_consensus_atlas, ConsensusAtlasBatchError, ConsensusAtlasBatchParams,
    ConsensusAtlasReport,
};
