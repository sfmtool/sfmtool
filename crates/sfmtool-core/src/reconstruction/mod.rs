// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! SfM reconstruction: the [`SfmrReconstruction`] data type plus per-point
//! operations (triangulation, filtering, correspondence).

pub mod bundle_adjust;
pub(crate) mod data;
mod edit;
pub mod edited;
mod embed;
pub mod filter;
pub mod move_camera;
pub mod point_correspondence;
pub mod triangulation;

pub use bundle_adjust::{
    bundle_adjust, BundleAdjustError, BundleAdjustOptions, BundleAdjustReport,
};

pub use move_camera::{move_camera, MoveCameraError, MoveCameraReport, ReprojectionSample};

pub use triangulation::retriangulate::{
    retriangulate_points, RetriangulateError, RetriangulateOptions, RetriangulateReport,
    RetriangulateWhich,
};

pub use edited::{
    EditError, EditedReconstruction, PointMap, PointRecord, PointView, RecordObservation, RowMap,
};

pub use data::{
    unit_quaternion_preserving, ImageTable, ObservationSource, Point3D, PointConstraintColumns,
    PointSet, ReconstructionError, SfmrImage, SfmrReconstruction, TrackObservation,
};
