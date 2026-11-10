// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Choosing the frames of a video capture a Structure-from-Motion solve
//! should use, from the SIFT features of the frames, before anything is
//! solved.
//!
//! [`pick_images`] walks each sequence from its first frame, choosing frames
//! whose **step**, the median displacement of their mutual matches in percent
//! of the image's long side, is near a target, with no step beyond the next
//! frame keeping fewer mutual matches than the match floor, and a frame
//! blurred against its neighbouring frames swapped for a sharper one. Given a
//! sequence's camera, the local density term divides the target by a gain
//! where the essential splits of the last steps chosen show a near
//! foreground in front of a distant view ([`DensityTerm`]); unless
//! [`PickImagesOptions::pilot_vote`] is off, a sequence with no camera takes
//! the camera of a focal vote on the first frames the walk chooses without
//! the term ([`PilotVote`]). A
//! [`FrameSource`] makes each frame's `.sift` files exist: an
//! [`ExtractingSource`] extracts them on a pool of workers ahead of the walk,
//! from the frames it predicts it will read, and a [`ReadOnlySource`] reads
//! files that already exist. The walk decides only from frames it waited for,
//! so the work done ahead changes how long a run takes and never what it
//! chooses.
//!
//! [`match_frame_pair`] matches the features of two frames through a KD-forest
//! per frame and reports the number of mutual matches and the step and, given
//! a camera, the essential split of the matches ([`measure_essential_split`]):
//! the rotation between the frames and the parallax left once it is taken out.
//! [`FrameFeatures`] holds one frame's features with its forest built once,
//! so a frame matched with several others is indexed once.
//!
//! The designs are `specs/core/analysis/pick-images.md` and
//! `specs/core/analysis/frame-pair-matching.md`.

mod capture;
mod density;
mod essential_split;
mod frame_pair;
mod frames;
mod pick;
mod pilot;
mod report;
mod source;
mod walk;

#[cfg(test)]
mod tests;

pub use capture::{
    CaptureFrame, CaptureImage, CaptureSequence, FrameId, PickImagesError, PickImagesOptions,
};
pub use density::{DensityTerm, DensityTermOff};
pub use essential_split::{
    measure_essential_split, EssentialSplit, EssentialSplitOptions, EssentialSplitOutcome,
    SplitNotMeasured,
};
pub use frame_pair::{
    match_frame_pair, FrameFeatures, FramePairError, FramePairOptions, FramePairStats,
};
pub use pick::pick_images;
pub use pilot::{PilotOutcome, PilotVote};
pub use report::{
    FrameReading, MeasuredPair, PickImagesReport, SensorReading, SequenceReport, StepDensity,
    StepKind, Swap, WalkStep,
};
pub use source::{ExtractingSource, FrameSource, FrameSourceError, ReadOnlySource};
