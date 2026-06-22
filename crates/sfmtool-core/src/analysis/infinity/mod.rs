// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Points at infinity for [`SfmrReconstruction`].
//!
//! A point at infinity (`w = 0`) is a feature track whose observation rays are
//! parallel to within measurement noise — distant content whose depth the SfM
//! solve cannot pin down. This module has two complementary halves:
//!
//! - [`convert`] *reclassifies* points the solve already triangulated, moving
//!   them across the finite ↔ infinity boundary.
//! - [`discover`] *finds* new infinite tracks the solve's parallax filters threw
//!   away, by clustering world-space keypoint directions on the unit sphere.

mod convert;
mod discover;

pub use convert::{
    Classification, RayClassification, CONDITION_NUMBER_PREFILTER, DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
    DEFAULT_NOISE_FLOOR_PX,
};
pub use discover::{find_infinity_tracks, InfinityParams, InfinityTrack};

pub(crate) use convert::{camera_extents, classify_rays_at_infinity};
