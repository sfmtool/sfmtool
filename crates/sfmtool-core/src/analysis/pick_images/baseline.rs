// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The baseline: the sampled frames of each sequence, the median sharpness
//! its frames are judged against, the rate along it, the flagged regions, and
//! the two judgements made from it, whether a sequence is a video and whether
//! its camera moves enough to be solved. See
//! `specs/core/analysis/pick-images.md` § "The baseline".

use super::capture::{FrameId, PickImagesOptions};
use super::frame_pair::FramePairStats;
use super::frames::Frames;
use super::report::{
    FlaggedRegion, SampleRate, SensorBaseline, SensorReading, SolvabilityCheck, VideoCheck,
};
use crate::features::sift::ImageReadings;
use crate::numeric::{median, quantile};

/// The gaps after each sampled frame the baseline reads: enough to tell a
/// video from a set of photographs.
pub(super) const BASELINE_GAPS: [usize; 2] = [1, 3];

/// The range of the 3-frame step over the 1-frame step in which the motion
/// adds up.
const VIDEO_RATIO_RANGE: (f64, f64) = (2.0, 4.0);

/// The 1-frame step, in percent of the long side, below which the step is
/// small.
const VIDEO_MAX_STEP_PERCENT: f64 = 5.0;

/// The factor within which neighbouring samples' 1-frame steps lie when the
/// step changes little.
const VIDEO_MAX_STEP_FACTOR: f64 = 2.0;

/// The total motion, in percent of the long side, below which the camera
/// barely moves.
const MIN_TOTAL_MOTION_PERCENT: f64 = 100.0;

/// How many sampled frames each sequence gets: its share of
/// `baseline_frames` by frame count, at least 2 (the first and last) and at
/// most every frame.
pub(super) fn sample_counts(lengths: &[usize], baseline_frames: usize) -> Vec<usize> {
    let total: usize = lengths.iter().sum();
    lengths
        .iter()
        .map(|&len| {
            let share = (baseline_frames as f64 * len as f64 / total as f64).round() as usize;
            share.max(2).min(len)
        })
        .collect()
}

/// `count` positions spread evenly over `len` frames, the first and last
/// included.
pub(super) fn sample_positions(len: usize, count: usize) -> Vec<usize> {
    if count <= 1 || len <= 1 {
        return vec![0];
    }
    let mut positions: Vec<usize> = (0..count)
        .map(|i| ((i * (len - 1)) as f64 / (count - 1) as f64).round() as usize)
        .collect();
    positions.dedup();
    positions
}

/// Whether a pair matches well: it has at least `min_matches` mutual matches.
pub(super) fn matches_well(stats: &FramePairStats, options: &PickImagesOptions) -> bool {
    stats.matches >= options.min_matches
}

/// One sensor's baseline over the readings of the sampled frames.
pub(super) fn sensor_baseline(readings: &[ImageReadings]) -> SensorBaseline {
    let sharpness: Vec<f64> = readings.iter().map(|r| f64::from(r.sharpness)).collect();
    SensorBaseline {
        sharpness_median: median(&sharpness),
    }
}

/// Judge one image's readings against its sensor's baseline: it is blurred
/// when its sharpness is below `blur_ratio` times the median sharpness.
pub(super) fn judge(
    readings: &ImageReadings,
    baseline: &SensorBaseline,
    blur_ratio: f64,
) -> SensorReading {
    let sharpness = f64::from(readings.sharpness);
    SensorReading {
        sharpness: readings.sharpness,
        gradient_structure_tensor_xx_yy_xy: readings.gradient_structure_tensor_xx_yy_xy,
        axis_ratio: readings.axis_ratio(),
        coherence: readings.coherence(),
        sharpness_ratio: sharpness / baseline.sharpness_median,
        blurred: sharpness < blur_ratio * baseline.sharpness_median,
    }
}

/// One sequence's baseline as the walk reads it.
pub(super) struct SequenceBaseline {
    /// The sequence's frame count.
    pub(super) len: usize,
    pub(super) samples: Vec<usize>,
    pub(super) sensors: Vec<SensorBaseline>,
    pub(super) sample_rates: Vec<SampleRate>,
    pub(super) rate_median: f64,
    pub(super) flagged_regions: Vec<FlaggedRegion>,
}

impl SequenceBaseline {
    /// Whether position `frame` lies in a region flagged for weak matches.
    pub(super) fn is_flagged(&self, frame: usize) -> bool {
        self.flagged_regions
            .iter()
            .any(|r| r.weak_matches && r.first <= frame && frame <= r.last)
    }

    /// The first position after `frame` in a region flagged for weak
    /// matches.
    pub(super) fn next_flagged(&self, frame: usize) -> Option<usize> {
        self.flagged_regions
            .iter()
            .filter(|r| r.weak_matches && r.last > frame)
            .map(|r| r.first.max(frame + 1))
            .min()
    }

    /// The rate of the sampled frame nearest `frame` that has one, and the
    /// sequence's median rate when none has.
    pub(super) fn rate_near(&self, frame: usize) -> f64 {
        self.sample_rates
            .iter()
            .min_by_key(|r| r.frame.abs_diff(frame))
            .map_or(self.rate_median, |r| r.rate)
    }

    /// Judge a frame's readings, one per sensor.
    pub(super) fn judge_frame(
        &self,
        readings: &[ImageReadings],
        blur_ratio: f64,
    ) -> Vec<SensorReading> {
        readings
            .iter()
            .zip(&self.sensors)
            .map(|(r, b)| judge(r, b, blur_ratio))
            .collect()
    }

    /// Whether a frame is blurred, and its sharpness over the median, the
    /// smallest over its sensors (NaN read as the least sharp).
    pub(super) fn blur(&self, readings: &[ImageReadings], blur_ratio: f64) -> (bool, f64) {
        let judged = self.judge_frame(readings, blur_ratio);
        let blurred = judged.iter().any(|j| j.blurred);
        let ratio = judged
            .iter()
            .map(|j| {
                if j.sharpness_ratio.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    j.sharpness_ratio
                }
            })
            .fold(f64::INFINITY, f64::min);
        (blurred, ratio)
    }
}

/// The sequence's baseline from its pairs: the sensors' median
/// sharpness over the sampled frames, the rate at each sample, and the regions
/// a blurred sample or one that does not match the frame 1 or 3 after it
/// well flags.
pub(super) fn sequence_baseline(
    frames: &Frames,
    sequence: usize,
    samples: Vec<usize>,
    len: usize,
    options: &PickImagesOptions,
) -> SequenceBaseline {
    let id = |frame| FrameId { sequence, frame };
    let sensor_count = frames.readings(id(samples[0])).len();
    let sensors: Vec<SensorBaseline> = (0..sensor_count)
        .map(|k| {
            let readings: Vec<ImageReadings> =
                samples.iter().map(|&s| frames.readings(id(s))[k]).collect();
            sensor_baseline(&readings)
        })
        .collect();

    let min_matches = options.min_matches;
    let measured = |s: usize, gap: usize| frames.pair(id(s), id(s + gap));
    let mut sample_rates = Vec::new();
    for &s in &samples {
        let rate = [(3, measured(s, 3)), (1, measured(s, 1))]
            .into_iter()
            .find_map(|(gap, stats)| {
                stats
                    .filter(|st| st.matches >= min_matches && st.step_percent.is_finite())
                    .map(|st| st.step_percent / gap as f64)
            });
        if let Some(rate) = rate {
            sample_rates.push(SampleRate { frame: s, rate });
        }
    }
    let rates: Vec<f64> = sample_rates.iter().map(|r| r.rate).collect();
    let rate_median = median(&rates);

    let mut baseline = SequenceBaseline {
        len,
        samples,
        sensors,
        sample_rates,
        rate_median,
        flagged_regions: Vec::new(),
    };
    let mut regions: Vec<FlaggedRegion> = Vec::new();
    for (i, &s) in baseline.samples.iter().enumerate() {
        let (blurred, _) = baseline.blur(frames.readings(id(s)), options.blur_ratio);
        let weak = BASELINE_GAPS
            .iter()
            .filter_map(|&gap| measured(s, gap))
            .any(|stats| !matches_well(stats, options));
        if !(blurred || weak) {
            continue;
        }
        let first = if i == 0 { s } else { baseline.samples[i - 1] };
        let last = baseline
            .samples
            .get(i + 1)
            .copied()
            .unwrap_or(s)
            .min(len - 1);
        regions.push(FlaggedRegion {
            first,
            last,
            blurred,
            weak_matches: weak,
        });
    }
    let mut merged: Vec<FlaggedRegion> = Vec::new();
    for region in regions {
        match merged.last_mut() {
            Some(previous) if region.first <= previous.last => {
                previous.last = previous.last.max(region.last);
                previous.blurred |= region.blurred;
                previous.weak_matches |= region.weak_matches;
            }
            _ => merged.push(region),
        }
    }
    baseline.flagged_regions = merged;
    baseline
}

/// The 0.9 quantile of the sample rates.
pub(super) fn rate_p90(baseline: &SequenceBaseline) -> f64 {
    let rates: Vec<f64> = baseline.sample_rates.iter().map(|r| r.rate).collect();
    quantile(&rates, 0.9)
}

/// Judge whether a sequence is a video from the baseline's pairs.
pub(super) fn video_check(
    frames: &Frames,
    sequence: usize,
    baseline: &SequenceBaseline,
    options: &PickImagesOptions,
) -> VideoCheck {
    let id = |frame| FrameId { sequence, frame };
    let len = baseline.len;
    let min_matches = options.min_matches;
    // Per sample with a frame 3 after it: the 1-frame step and the 3-to-1
    // ratio, or a value that fails each sign when either pair has fewer than
    // `min_matches` matches.
    let mut steps = Vec::new();
    let mut ratios = Vec::new();
    for &s in baseline.samples.iter().filter(|&&s| s + 3 < len) {
        let one = frames.pair(id(s), id(s + 1));
        let three = frames.pair(id(s), id(s + 3));
        match (one, three) {
            (Some(one), Some(three))
                if one.matches >= min_matches && three.matches >= min_matches =>
            {
                steps.push(Some(one.step_percent));
                ratios.push(three.step_percent / one.step_percent);
            }
            _ => {
                steps.push(None);
                ratios.push(f64::INFINITY);
            }
        }
    }
    let changes: Vec<f64> = steps
        .windows(2)
        .map(|w| match (w[0], w[1]) {
            (Some(a), Some(b)) => (a / b).ln().abs(),
            _ => f64::INFINITY,
        })
        .collect();
    let step_values: Vec<f64> = steps.iter().map(|s| s.unwrap_or(f64::INFINITY)).collect();

    let step_1_median = median(&step_values);
    let ratio_3_to_1_median = median(&ratios);
    let step_change_median = median(&changes);
    let motion_adds_up = (VIDEO_RATIO_RANGE.0..=VIDEO_RATIO_RANGE.1).contains(&ratio_3_to_1_median);
    let step_small = step_1_median < VIDEO_MAX_STEP_PERCENT;
    let step_steady = step_change_median < VIDEO_MAX_STEP_FACTOR.ln();
    VideoCheck {
        step_1_median,
        ratio_3_to_1_median,
        step_change_median,
        motion_adds_up,
        step_small,
        step_steady,
        is_video: len >= 4 && step_small && motion_adds_up && step_steady,
    }
}

/// Judge whether a sequence's camera moves enough to be solved, from its
/// baseline's rate.
pub(super) fn solvability_check(baseline: &SequenceBaseline) -> SolvabilityCheck {
    let total_motion_percent = baseline.rate_median * baseline.len as f64;
    let barely_moves = total_motion_percent < MIN_TOTAL_MOTION_PERCENT;
    SolvabilityCheck {
        total_motion_percent,
        barely_moves,
        solvable: !barely_moves,
    }
}
