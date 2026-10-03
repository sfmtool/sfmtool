// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Points at infinity for [`crate::SfmrReconstruction`].
//!
//! A point at infinity (`w = 0`) is a feature track whose observation rays are
//! parallel to within measurement noise — distant content whose depth the SfM
//! solve cannot pin down. This module has two complementary halves:
//!
//! - `convert` *reclassifies* the points a reconstruction already has, moving
//!   them across the finite ↔ infinity boundary in either direction.
//! - `discover` *finds* new infinite tracks the solve's parallax filters threw
//!   away, by clustering world-space keypoint directions on the unit sphere.

mod convert;
mod discover;

pub use convert::{camera_extents, InfinityReclassification, DEFAULT_MIN_DEPTH_FRACTION};
pub use discover::{
    decide_candidate_tracks, find_infinity_tracks, CandidateDecision, InfinityDiscovery,
    InfinityParams, InfinityTrack,
};
