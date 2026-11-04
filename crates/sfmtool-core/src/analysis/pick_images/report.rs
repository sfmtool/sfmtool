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
    /// Whether the capture can be solved, and if not, the first check that
    /// stopped it.
    pub verdict: Verdict,
    /// One report per sequence, in the order the sequences were given.
    pub sequences: Vec<SequenceReport>,
    /// Every frame read, in the order it was read, with its readings. The
    /// source was asked for exactly these frames.
    pub frames: Vec<FrameReading>,
    /// Every pair of frames matched, in the order it was matched, each once.
    pub pairs: Vec<MeasuredPair>,
}

/// Whether the capture can be solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Every sequence is a video and passed every check of solvability; every
    /// sequence was walked.
    Solvable,
    /// A sequence is not a video ([`VideoCheck::is_video`]). Nothing was read
    /// after the baseline's first batch, and nothing was walked.
    NotAVideo,
    /// Every sequence is a video, and a sequence failed a check of
    /// solvability ([`SolvabilityCheck::solvable`]). Nothing was read after
    /// the baseline, and nothing was walked.
    NotSolvable,
}

/// Which stage of [`super::pick_images`] read a frame or matched a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// The baseline's first batch: the sampled frames and the frames 1 and 3
    /// after them.
    BaselineNear,
    /// The baseline's second batch: the frames 10 and 30 after the sampled
    /// frames.
    BaselineFar,
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

/// One image's readings and the blur judgement made from them against its
/// sensor's baseline ([`SensorBaseline`]).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SensorReading {
    /// The `.sift` file's sharpness.
    pub sharpness: f32,
    /// The `.sift` file's `[Exx, Eyy, Exy]`.
    pub gradient_structure_tensor_xx_yy_xy: [f32; 3],
    /// `ImageReadings::axis_ratio`: `None` for a flat image.
    pub axis_ratio: Option<f32>,
    /// `ImageReadings::coherence`.
    pub coherence: f32,
    /// The sharpness over its sensor's median sharpness; NaN when that median
    /// is 0.
    pub sharpness_ratio: f64,
    /// The axis ratio's robust z-score against its sensor's baseline:
    /// `(x − median) / (1.4826 · MAD)`. `None` for a flat image; infinite
    /// when the MAD is 0 and `x` is not the median.
    pub axis_ratio_z: Option<f64>,
    /// The coherence's robust z-score, likewise.
    pub coherence_z: f64,
    /// Whether the image is blurred: its sharpness is below `blur_ratio` times
    /// the median, or a z-score is above 2.
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

/// One sequence's baseline, its two judgements and its walk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SequenceReport {
    pub baseline: BaselineReport,
    pub video: VideoCheck,
    /// `None` when some sequence of the capture is not a video, since the
    /// checks read pairs 10 and 30 frames apart, which are then not read.
    pub solvability: Option<SolvabilityCheck>,
    /// `None` unless the verdict is [`Verdict::Solvable`].
    pub walk: Option<WalkReport>,
}

/// One sequence's baseline: what the sampled frames and their pairs read.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BaselineReport {
    /// The positions of the sampled frames, first and last included.
    pub samples: Vec<usize>,
    /// The frames of this sequence the baseline read, in both batches.
    pub frames_read: usize,
    /// The medians and spreads each sensor's frames are judged against.
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

/// One sensor's baseline: the medians and median absolute deviations of its
/// readings over the sampled frames.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SensorBaseline {
    pub sharpness_median: f64,
    /// Over the sampled frames that are not flat.
    pub axis_ratio_median: f64,
    pub axis_ratio_mad: f64,
    pub coherence_median: f64,
    pub coherence_mad: f64,
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

/// The four signs of a video, each the median over the sampled frames with a
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
    /// The median F-inlier share of the 1-frame pairs.
    pub f_inlier_share_1_median: f64,
    /// `ratio_3_to_1_median` is between 2 and 4.
    pub motion_adds_up: bool,
    /// `step_1_median` is below 5%.
    pub step_small: bool,
    /// `step_change_median` is below `log 2`.
    pub step_steady: bool,
    /// `f_inlier_share_1_median` is above 0.8.
    pub neighbours_match: bool,
    /// At least three of the four signs hold. False for a sequence of fewer
    /// than 4 frames, which has no 3-frame pair.
    pub is_video: bool,
}

/// The three checks of solvability, from the pairs 10 and 30 frames apart.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SolvabilityCheck {
    /// The median F-inlier share 10 frames apart, over the samples whose pair
    /// has a share; `None` when none has.
    pub f_inlier_share_10_median: Option<f64>,
    /// The median H/F 30 frames apart, likewise.
    pub h_over_f_30_median: Option<f64>,
    /// The median rotation error 30 frames apart, likewise.
    pub rotation_error_30_median: Option<f64>,
    /// [`BaselineReport::total_motion_percent`].
    pub total_motion_percent: f64,
    /// `f_inlier_share_10_median` is below 0.3.
    pub moving_scene: bool,
    /// `h_over_f_30_median` is above 0.97 and `rotation_error_30_median` is
    /// below 0.01.
    pub panorama: bool,
    /// `total_motion_percent` is below 100.
    pub barely_moves: bool,
    /// None of the three holds.
    pub solvable: bool,
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
    /// The step's band was halved because `from` lies in a flagged region.
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
    /// a flagged region, which a step does not pass.
    AtLimit,
    /// Below the band after a step failed: part of a gap filled in.
    GapFill,
    /// Below the band to land on a sharp frame, or on the sharpest, in a
    /// blurred run.
    BlurredRun,
}

/// A blurred candidate and the sharp neighbour chosen in its place.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Swap {
    pub candidate: usize,
    pub chosen: usize,
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
    /// An accepted step with fewer than twice `min_matches` matches, or an
    /// F-inlier share within 0.1 of `min_inlier_share`.
    Step {
        /// The index in [`WalkReport::steps`].
        step: usize,
        few_matches: bool,
        low_inlier_share: bool,
    },
    /// A segment of fewer than 3 chosen frames.
    ShortSegment {
        /// The index in [`WalkReport::segments`].
        segment: usize,
    },
}
