// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Choosing the frames of a video capture a Structure-from-Motion solve
//! should use, from the SIFT features of the frames, before anything is
//! solved.
//!
//! [`pick_images`] reads a baseline sample of each sequence, judges whether
//! each is a video and whether the capture can be solved, and walks each
//! sequence choosing frames whose **step**, the median displacement of their
//! mutual matches in percent of the image's long side, is near a target, with
//! sharp frames in place of blurred ones and shorter steps where the capture
//! is hard to follow. A [`FrameSource`] makes each frame's `.sift` files
//! exist: an [`ExtractingSource`] extracts them on a pool of workers ahead of
//! the walk, from the frames it knows it will read, and a
//! [`ReadOnlySource`] reads files that already exist. The walk decides only
//! from frames it waited for, so the work done ahead changes how long a run
//! takes and never what it chooses.
//!
//! [`match_frame_pair`] matches the features of two frames through a KD-forest
//! per frame and reports the step, what share of the matches fit one
//! fundamental matrix, what share of those also fit one homography, and how
//! close that homography is to a pure rotation. [`FrameFeatures`] holds one
//! frame's features with its forest built once, so a frame matched with
//! several others is indexed once.
//!
//! The designs are `specs/core/analysis/pick-images.md` and
//! `specs/core/analysis/frame-pair-matching.md`.

mod baseline;
mod capture;
mod frame_pair;
mod frames;
mod pick;
mod report;
mod source;
mod walk;

#[cfg(test)]
mod tests;

pub use capture::{
    CaptureFrame, CaptureImage, CaptureSequence, FrameId, PickImagesError, PickImagesOptions,
};
pub use frame_pair::{
    match_frame_pair, FrameFeatures, FramePairError, FramePairOptions, FramePairStats,
};
pub use pick::pick_images;
pub use report::{
    BaselineReport, Break, FlaggedRegion, FrameReading, Gap, KeptBlurred, MeasuredPair,
    PickImagesReport, SampleRate, Segment, SensorBaseline, SensorReading, SequenceReport,
    SolvabilityCheck, Stage, StepKind, Swap, Verdict, VideoCheck, WalkReport, WalkStep, WeakPoint,
};
pub use source::{ExtractingSource, FrameSource, FrameSourceError, ReadOnlySource};
