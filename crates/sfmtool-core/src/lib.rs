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
//! - [`bench`](mod@bench): the bench beside a reconstruction and the editable
//!   track on it: the item list, its labels, and the steps that work a track up
//!   to a point and commit it
//! - [`machine`]: what the machine offers the code that runs on it: the SIMD
//!   level the kernels dispatch to, the CPU, the thread pool, memory and the
//!   GPU adapter, which `sfm version` reports
//! - [`spatial`] — generic KD-tree point-cloud utility used across groups, and
//!   the per-image keypoint reach enumeration in the pixel domain
//! - [`progress`] — the parameter a long-running call reports its phases,
//!   messages and counts through, and is told to stop by
//! - [`web_export`]: a reconstruction as static web files, the `scene.json`
//!   and JPEG atlas pages `sfm web-export` writes for a browser viewer
//! - [`workspace`]: the files in a workspace root that configure the
//!   workspace as a whole, such as its named image lists
//!
//! File-format I/O is provided by the sibling crates `sfmtool-sift-format`,
//! `sfmtool-sfmr-format`, `sfmtool-matches-format`, and `sfmtool-camrig-format`.

pub mod analysis;
pub mod bench;
pub mod camera;
pub mod features;
pub mod geometry;
pub mod machine;
pub mod numeric;
pub mod patch;
mod profiling;
pub mod readable;
pub mod reconstruction;
pub mod spatial;
pub mod spherical;
pub mod web_export;
pub mod workspace;

// Headline types reachable as `sfmtool_core::Foo`, leaning on each group's
// own facade re-exports (`camera::{Camera, CameraIntrinsics, CameraModel}`,
// `geometry::{RigidTransform, RotQuaternion, Se3Transform}`, etc.).
pub use camera::{Camera, CameraIntrinsics, CameraModel, IntrinsicAdjustment};
pub use geometry::{RigidTransform, RotQuaternion, Se3Transform};
pub use progress::Progress;
/// The reconstruction-level bundle adjustment, which is a different function
/// from the array kernel [`geometry::bundle_adjust()`] it runs: it takes a
/// reconstruction value and hands back another one.
pub use reconstruction::{
    bundle_adjust, BundleAdjustError, BundleAdjustOptions, BundleAdjustReport, CameraAdjustment,
};
pub use reconstruction::{
    move_camera, EditError, EditedReconstruction, ImageSet, MoveCameraError, MoveCameraReport,
    ObservationSource, PerImageAdjustments, Point3D, PointConstraintColumns, PointMap, PointRecord,
    PointSet, PointView, ReconstructionError, RecordObservation, ReprojectionSample, RowMap,
    SfmrImage, SfmrReconstruction, SiftKeypointFill, TrackObservation,
};
/// The progress parameter is its own crate, so the file-format crates can
/// report through the very type this one does without depending on this one.
/// It is re-exported as this crate's `progress` module, and its four macros at
/// this crate's root, so `sfmtool_core::progress::Progress` and
/// `sfmtool_core::progress_info!` name what they have always named.
pub use sfmtool_progress as progress;
pub use sfmtool_progress::{progress_info, progress_note, progress_status, progress_warn};
/// Re-exported so consumers of [`ImageSet::thumbnails_y_x_rgb`] can
/// size buffers from the same constant the format pins, without depending on
/// `sfmtool-sfmr-format` directly.
pub use sfmtool_sfmr_format::THUMBNAIL_SIZE;
/// Re-exported for the same reason, one level up: [`ImageSet::rig_frame_data`]
/// is a public field whose type nothing downstream could otherwise name, so a
/// consumer wanting to read — or build — a rig had to depend on `sfmtool-sfmr-format`
/// itself to say what it was holding.
pub use sfmtool_sfmr_format::{FramesMetadata, RigDefinition, RigFrameData, RigsMetadata};
/// Re-exported alongside [`PointConstraintColumns`], whose entries are these
/// codes: a consumer reading or building the constraint columns needs to name
/// them without depending on `sfmtool-sfmr-format` itself.
pub use sfmtool_sfmr_format::{
    NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED,
};

/// The units a reconstruction's `world_space_unit` may name, with their lengths
/// in metres. Re-exported because the metadata field is reached through
/// [`SfmrReconstruction`], and a consumer converting a length out of it needs
/// the table without depending on `sfmtool-sfmr-format` itself.
pub use sfmtool_sfmr_format::{world_space_unit_in_metres, WORLD_SPACE_UNITS};

pub use spherical::{
    render_consensus_atlas, ConsensusAtlasBatchError, ConsensusAtlasBatchParams,
    ConsensusAtlasReport,
};
