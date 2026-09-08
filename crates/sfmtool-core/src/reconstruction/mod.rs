// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! SfM reconstruction: the [`SfmrReconstruction`] data type plus per-point
//! operations (triangulation, filtering, correspondence).

pub(crate) mod data;
mod edit;
pub mod edited;
mod embed;
pub mod filter;
pub mod point_correspondence;
pub mod point_estimation;
pub mod triangulation;

pub use edited::{
    EditError, EditedReconstruction, PointRecord, PointView, RecordObservation, RowMap,
};

pub use data::{
    unit_quaternion_preserving, ImageTable, ObservationSource, Point3D, PointConstraintColumns,
    PointSet, ReconstructionError, SfmrImage, SfmrReconstruction, TrackObservation,
};
