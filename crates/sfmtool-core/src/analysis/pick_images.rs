// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Measuring the motion between two frames of a video from their SIFT
//! features, before anything is solved.
//!
//! [`match_frame_pair`] matches the features of two frames through a KD-forest
//! per frame and reports how far the image moved (the **step**, the median
//! displacement of the mutual matches in percent of the image's long side),
//! what share of the matches fit one fundamental matrix, what share of those
//! also fit one homography, and how close that homography is to a pure
//! rotation. [`FrameFeatures`] holds one frame's features with its forest
//! built once, so a frame matched with several others is indexed once.
//!
//! The design is `specs/core/analysis/frame-pair-matching.md`.

mod frame_pair;

pub use frame_pair::{
    match_frame_pair, FrameFeatures, FramePairError, FramePairOptions, FramePairStats,
};
