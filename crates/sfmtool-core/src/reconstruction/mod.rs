// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! SfM reconstruction: the [`SfmrReconstruction`] data type plus per-point
//! operations (triangulation, filtering, correspondence).

pub(crate) mod data;
mod edit;
mod embed;
pub mod filter;
pub mod point_correspondence;
pub mod triangulation;

pub use data::{
    ObservationSource, Point3D, ReconstructionError, SfmrImage, SfmrReconstruction,
    TrackObservation,
};
