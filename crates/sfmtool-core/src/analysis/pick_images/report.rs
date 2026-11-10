// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What [`super::pick_images`] found: the readings of every frame read and
//! the walk's choices and decisions, per sequence. Every type serializes
//! with `serde`, a non-finite number as `null`, so a caller can write the
//! whole report as JSON.
//!
//! Frames are named by their position in their sequence, as [`FrameId`]
//! does; a caller turns a position into a file number with
//! [`super::CaptureSequence::frames`].

use serde::Serialize;

use super::capture::FrameId;
use super::density::DensityTerm;
use super::essential_split::EssentialSplitOutcome;
use super::frame_pair::FramePairStats;

/// What [`super::pick_images`] found.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PickImagesReport {
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

/// One frame read, with the readings of each of its sensors.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrameReading {
    pub id: FrameId,
    /// One reading per sensor, the reference sensor's first.
    pub sensors: Vec<SensorReading>,
}

/// One image's readings, from its `.sift` file's metadata.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SensorReading {
    /// The `.sift` file's sharpness, which judges blur.
    pub sharpness: f32,
    /// The `.sift` file's `[Exx, Eyy, Exy]`.
    pub gradient_structure_tensor_xx_yy_xy: [f32; 3],
    /// `ImageReadings::axis_ratio`: `None` for a flat image. Reported as a
    /// reading only; blur is not judged from it.
    pub axis_ratio: Option<f32>,
    /// `ImageReadings::coherence`, likewise reported only.
    pub coherence: f32,
}

/// One pair of frames matched.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MeasuredPair {
    pub a: FrameId,
    pub b: FrameId,
    pub stats: FramePairStats,
}

/// One sequence's walk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SequenceReport {
    /// The chosen frames' positions, in order: the first frame, then one per
    /// step.
    pub chosen: Vec<usize>,
    /// Every accepted step, in order: one per chosen frame after the first.
    pub steps: Vec<WalkStep>,
    /// The candidates blurred against their neighbours that were swapped for
    /// a sharper neighbour.
    pub swaps: Vec<Swap>,
    /// How many frames of this sequence the walk read.
    pub frames_read: usize,
    /// Whether the walk had the local density term, and why not.
    pub density_term: DensityTerm,
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
    /// The target the step was walked with and the local density term's
    /// reading of the step.
    pub density: StepDensity,
}

/// One accepted step's target and the local density term's reading of it.
/// With the term off, the target is the base target, the gain, the median
/// measure and the measure are 1, and no split is measured.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StepDensity {
    /// The target the step was walked with, in percent of the long side: the
    /// base target over `gain`.
    pub target_percent: f64,
    /// The gain the base target was divided by, from `median_measure`.
    pub gain: f64,
    /// The median of the measures of the last three steps before this one,
    /// or of as many as there were; 1 before the first.
    pub median_measure: f64,
    /// The step's essential split, measured once it was accepted; `None`
    /// with the term off.
    pub essential_split: Option<EssentialSplitOutcome>,
    /// The step's measure, which the next steps' median measures take in:
    /// its parallax P90 over P50 when its split counts, and 1 otherwise.
    pub measure: f64,
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
    /// Below the band at the sequence's last frame, which a step does not
    /// pass.
    AtLimit,
    /// Stopped by the match floor: the step to the frame after it keeps fewer
    /// than `min_step_matches` matches, or it is the step to the next frame
    /// and keeps fewer itself, so every step beyond it would too.
    AtFloor,
    /// Below the band after four re-aims did not reach it: the farthest
    /// candidate measured below the band, or the next frame when none was.
    BelowBand,
}

/// A candidate blurred against its neighbours and the sharper neighbour
/// chosen in its place.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Swap {
    pub candidate: usize,
    pub chosen: usize,
}
