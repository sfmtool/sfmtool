// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What [`super::pick_images`] found: the verdict, the baseline's readings,
//! and the walk's choices and decisions, per sequence. Every type serializes
//! with `serde`, a non-finite number as `null`, so a caller can write the
//! whole report as JSON.
//!
//! Frames are named by their position in their sequence, as [`FrameId`]
//! does; a caller turns a position into a file number with
//! [`super::CaptureSequence::frames`].

use serde::Serialize;

use super::capture::FrameId;
use super::frame_pair::FramePairStats;

/// What [`super::pick_images`] found.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PickImagesReport {
    /// Whether every sequence is a video, which decides whether the capture
    /// was walked.
    pub verdict: Verdict,
    /// One report per sequence, in the order the sequences were given.
    pub sequences: Vec<SequenceReport>,
    /// Every frame read, in the order it was read, with its readings. The
    /// walk waited on the source for exactly these frames.
    pub frames: Vec<FrameReading>,
    /// Every pair of frames matched, in the order it was matched, each once.
    pub pairs: Vec<MeasuredPair>,
    /// Per entry of `frames`, whether the source wrote its `.sift` files
    /// during the run rather than finding them up to date.
    pub frames_extracted: Vec<bool>,
    /// How many frames the source wrote any `.sift` file for during the run
    /// that the walk never read: frames it predicted it would read and then
    /// did not, a rig frame stopped between its images among them. Unlike
    /// everything else in the report, it depends on how far ahead the walk
    /// predicts and on how fast the source works.
    pub frames_extracted_unread: usize,
}

/// Whether every sequence of the capture is a video, and so was walked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Every sequence is a video; every sequence was walked.
    Video,
    /// A sequence is not a video ([`VideoCheck::is_video`]). Nothing was read
    /// after the baseline, and nothing was walked.
    NotAVideo,
}

/// Which stage of [`super::pick_images`] read a frame or matched a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// The baseline: the sampled frames and the frames 1 and 3 after them.
    Baseline,
    /// The walk.
    Walk,
}

/// One frame read, with the readings of each of its sensors.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrameReading {
    pub id: FrameId,
    /// The stage that read it first.
    pub stage: Stage,
    /// One reading per sensor, the reference sensor's first.
    pub sensors: Vec<SensorReading>,
    /// Whether any sensor's image is blurred.
    pub blurred: bool,
}

/// One image's readings and the blur judgement made from its sharpness
/// against its sensor's baseline ([`SensorBaseline`]).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SensorReading {
    /// The `.sift` file's sharpness.
    pub sharpness: f32,
    /// The `.sift` file's `[Exx, Eyy, Exy]`.
    pub gradient_structure_tensor_xx_yy_xy: [f32; 3],
    /// `ImageReadings::axis_ratio`: `None` for a flat image. Reported as a
    /// reading only; blur is not judged from it.
    pub axis_ratio: Option<f32>,
    /// `ImageReadings::coherence`, likewise reported only.
    pub coherence: f32,
    /// The sharpness over its sensor's median sharpness; NaN when that median
    /// is 0.
    pub sharpness_ratio: f64,
    /// Whether the image is blurred: its sharpness is below `blur_ratio` times
    /// the median.
    pub blurred: bool,
}

/// One pair of frames matched.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MeasuredPair {
    pub a: FrameId,
    pub b: FrameId,
    pub stats: FramePairStats,
    /// The stage that matched it.
    pub stage: Stage,
}

/// One sequence's baseline, the judgement of whether it is a video, and its
/// walk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SequenceReport {
    pub baseline: BaselineReport,
    pub video: VideoCheck,
    /// `None` unless the verdict is [`Verdict::Video`].
    pub walk: Option<WalkReport>,
}

/// One sequence's baseline: what the sampled frames and their pairs read.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BaselineReport {
    /// The positions of the sampled frames, first and last included.
    pub samples: Vec<usize>,
    /// The frames of this sequence the baseline read.
    pub frames_read: usize,
    /// The median sharpness each sensor's frames are judged against.
    pub sensors: Vec<SensorBaseline>,
    /// How many sampled frames are blurred.
    pub blurred_samples: usize,
    /// The rate at each sampled frame that has one, in percent of the long
    /// side per frame: the step to the frame 3 after it divided by 3, or the
    /// step to the next frame when the frame 3 after it has fewer than
    /// `min_matches` matches.
    pub sample_rates: Vec<SampleRate>,
    /// The median of `sample_rates`; NaN when there are none.
    pub rate_median: f64,
    /// The 0.9 quantile of `sample_rates`; NaN when there are none.
    pub rate_p90: f64,
    /// The median rate times the frame count, in percent of the long side.
    pub total_motion_percent: f64,
    /// The stretches the walk goes through in short steps, in order, merged
    /// where they overlap.
    pub flagged_regions: Vec<FlaggedRegion>,
}

/// One sensor's baseline: the median sharpness of its images over the
/// sampled frames.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SensorBaseline {
    pub sharpness_median: f64,
}

/// The rate at one sampled frame.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SampleRate {
    /// The sampled frame's position.
    pub frame: usize,
    /// The step per frame, in percent of the long side.
    pub rate: f64,
}

/// A stretch of frames, from the sampled frame before a flagging sample to
/// the sampled frame after it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FlaggedRegion {
    pub first: usize,
    pub last: usize,
    /// A sampled frame in it is blurred.
    pub blurred: bool,
    /// A sampled frame in it does not match the frame 1 or 3 after it well.
    pub weak_matches: bool,
}

/// The three signs of a video, each the median over the sampled frames with a
/// frame 3 after them. A sample whose 1-frame or 3-frame pair has fewer than
/// `min_matches` matches counts against every sign it enters.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoCheck {
    /// The median step to the next frame, in percent of the long side.
    pub step_1_median: f64,
    /// The median of the step 3 frames apart over the step 1 frame apart.
    pub ratio_3_to_1_median: f64,
    /// The median of `|log(step_i / step_{i+1})|` over neighbouring sampled
    /// frames' 1-frame steps.
    pub step_change_median: f64,
    /// `ratio_3_to_1_median` is between 2 and 4.
    pub motion_adds_up: bool,
    /// `step_1_median` is below 5%.
    pub step_small: bool,
    /// `step_change_median` is below `log 2`.
    pub step_steady: bool,
    /// All three signs hold. False for a sequence of fewer than 4 frames,
    /// which has no 3-frame pair.
    pub is_video: bool,
}

/// One sequence's walk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WalkReport {
    /// The chosen frames' positions, in order.
    pub chosen: Vec<usize>,
    /// The segments between breaks, as runs of `chosen`.
    pub segments: Vec<Segment>,
    /// Every accepted step, in order: one per chosen frame after the first of
    /// each segment.
    pub steps: Vec<WalkStep>,
    /// The blurred candidates swapped for a sharp neighbour.
    pub swaps: Vec<Swap>,
    /// The runs of steps accepted below the band after a step failed.
    pub gaps: Vec<Gap>,
    /// The frames chosen though blurred.
    pub kept_blurred: Vec<KeptBlurred>,
    pub breaks: Vec<Break>,
    pub weak_points: Vec<WeakPoint>,
    /// The frames of this sequence the walk read or reused.
    pub frames_used: usize,
    /// How many of `frames_used` the baseline had already read.
    pub frames_from_baseline: usize,
}

/// A run of [`WalkReport::chosen`]: `chosen[start..start + len]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Segment {
    pub start: usize,
    pub len: usize,
}

/// One accepted step.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WalkStep {
    pub from: usize,
    pub to: usize,
    pub stats: FramePairStats,
    /// The step per frame, `stats.step_percent / (to − from)`.
    pub rate: f64,
    pub kind: StepKind,
    /// The step's band was halved because `from` lies in a region flagged
    /// for weak matches.
    pub flagged: bool,
}

/// Why a step was accepted where it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// The step is within the band.
    InBand,
    /// Even the next frame is above the band: the motion per frame is wider
    /// than the band.
    AboveBand,
    /// Below the band, with the frame after it above: the motion per frame is
    /// wider than the band.
    Bracketed,
    /// Below the band at the last frame of the sequence or the first frame of
    /// a region flagged for weak matches, which a step does not pass.
    AtLimit,
    /// Below the band after a step failed: part of a gap filled in.
    GapFill,
    /// Below the band to land on a sharp frame, or on the sharpest, in a
    /// blurred run.
    BlurredRun,
}

/// A blurred candidate and the sharper neighbour chosen in its place.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Swap {
    pub candidate: usize,
    pub chosen: usize,
    /// What the candidate's sharpness was judged against.
    pub reason: SwapReason,
}

/// Why a candidate was swapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SwapReason {
    /// The candidate is blurred: its sharpness is below `blur_ratio` times its
    /// sensor's median sharpness.
    Median,
    /// The candidate is blurred against its neighbours: its sharpness is
    /// below `neighbour_blur_ratio` times the sharpest neighbouring frame's,
    /// on a sensor whose sharpness is below its median.
    Neighbours,
}

/// A run of [`StepKind::GapFill`] steps: `steps[first_step..end_step]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Gap {
    pub first_step: usize,
    pub end_step: usize,
}

/// A frame chosen though blurred, the sharpest that matched well among the
/// frames searched.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KeptBlurred {
    pub frame: usize,
    /// The first and last positions searched.
    pub searched_first: usize,
    pub searched_last: usize,
    /// The frame's sharpness over its sensor's median, the smallest over its
    /// sensors.
    pub sharpness_ratio: f64,
}

/// Even the step from `after` to the next frame did not match well. The walk
/// resumed at `resumes_at`, the first later frame that matches its next frame
/// well, or not at all.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Break {
    pub after: usize,
    pub resumes_at: Option<usize>,
}

/// Where a solve is most likely to split.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum WeakPoint {
    /// An accepted step with fewer than twice `min_matches` matches.
    Step {
        /// The index in [`WalkReport::steps`].
        step: usize,
    },
    /// A segment of fewer than 3 chosen frames.
    ShortSegment {
        /// The index in [`WalkReport::segments`].
        segment: usize,
    },
}
