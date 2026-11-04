// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The baseline: the sampled frames of each sequence, the medians and spreads
//! its frames are judged against, the rate along it, the flagged regions, and
//! the two judgements made from it, whether a sequence is a video and whether
//! it can be solved. See `specs/core/analysis/pick-images.md` § "The
//! baseline".

use super::capture::{FrameId, PickImagesOptions};
use super::frame_pair::FramePairStats;
use super::frames::Frames;
use super::report::{
    FlaggedRegion, SampleRate, SensorBaseline, SensorReading, SolvabilityCheck, VideoCheck,
};
use crate::features::sift::ImageReadings;
use crate::numeric::{median, quantile};

/// The gaps after each sampled frame the first batch reads: enough to tell a
/// video from a set of photographs.
pub(super) const NEAR_GAPS: [usize; 2] = [1, 3];

/// The gaps the second batch reads, for the checks of solvability.
pub(super) const FAR_GAPS: [usize; 2] = [10, 30];

/// The scale from a median absolute deviation to a standard deviation for
/// normally distributed values.
const MAD_TO_SIGMA: f64 = 1.4826;

/// A reading whose robust z-score is above this is blurred along one
/// direction.
const BLUR_Z: f64 = 2.0;

/// The range of the 3-frame step over the 1-frame step in which the motion
/// adds up.
const VIDEO_RATIO_RANGE: (f64, f64) = (2.0, 4.0);

/// The 1-frame step, in percent of the long side, below which the step is
/// small.
const VIDEO_MAX_STEP_PERCENT: f64 = 5.0;

/// The factor within which neighbouring samples' 1-frame steps lie when the
/// step changes little.
const VIDEO_MAX_STEP_FACTOR: f64 = 2.0;

/// The 1-frame F-inlier share above which neighbours match closely.
const VIDEO_MIN_SHARE: f64 = 0.8;

/// How many of the four signs make a video.
const VIDEO_SIGNS_NEEDED: usize = 3;

/// The F-inlier share 10 frames apart below which the scene moves.
const MOVING_SCENE_MAX_SHARE: f64 = 0.3;

/// The H/F 30 frames apart above which, with a rotation error below
/// [`PANORAMA_MAX_ROTATION_ERROR`], the camera only turned.
const PANORAMA_MIN_H_OVER_F: f64 = 0.97;

const PANORAMA_MAX_ROTATION_ERROR: f64 = 0.01;

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

/// Whether a pair matches well: at least `min_matches` matches, and an
/// F-inlier share of at least `min_inlier_share`.
pub(super) fn matches_well(stats: &FramePairStats, options: &PickImagesOptions) -> bool {
    stats.matches >= options.pair.min_matches
        && stats
            .f_inlier_share
            .is_some_and(|share| share >= options.min_inlier_share)
}

/// The median of `values` and their median absolute deviation from it.
fn centre_and_spread(values: &[f64]) -> (f64, f64) {
    let m = median(values);
    let deviations: Vec<f64> = values.iter().map(|v| (v - m).abs()).collect();
    (m, median(&deviations))
}

/// The robust z-score `(x − median) / (1.4826 · MAD)`, and whether it is
/// above 2. The comparison is made without dividing, so a MAD of 0 counts
/// every value above the median and none at it.
fn robust_z(x: f64, median: f64, mad: f64) -> (f64, bool) {
    let spread = MAD_TO_SIGMA * mad;
    let above = x - median > BLUR_Z * spread;
    let z = if spread > 0.0 {
        (x - median) / spread
    } else if x == median {
        0.0
    } else {
        (x - median).signum() * f64::INFINITY
    };
    (z, above)
}

/// One sensor's baseline over the readings of the sampled frames.
pub(super) fn sensor_baseline(readings: &[ImageReadings]) -> SensorBaseline {
    let sharpness: Vec<f64> = readings.iter().map(|r| f64::from(r.sharpness)).collect();
    let axis: Vec<f64> = readings
        .iter()
        .filter_map(|r| r.axis_ratio().map(f64::from))
        .collect();
    let coherence: Vec<f64> = readings.iter().map(|r| f64::from(r.coherence())).collect();
    let (axis_ratio_median, axis_ratio_mad) = centre_and_spread(&axis);
    let (coherence_median, coherence_mad) = centre_and_spread(&coherence);
    SensorBaseline {
        sharpness_median: median(&sharpness),
        axis_ratio_median,
        axis_ratio_mad,
        coherence_median,
        coherence_mad,
    }
}

/// Judge one image's readings against its sensor's baseline.
pub(super) fn judge(
    readings: &ImageReadings,
    baseline: &SensorBaseline,
    blur_ratio: f64,
) -> SensorReading {
    let sharpness = f64::from(readings.sharpness);
    let soft = sharpness < blur_ratio * baseline.sharpness_median;
    let axis_ratio = readings.axis_ratio();
    let axis = axis_ratio.map(|a| {
        robust_z(
            f64::from(a),
            baseline.axis_ratio_median,
            baseline.axis_ratio_mad,
        )
    });
    let coherence = readings.coherence();
    let (coherence_z, coherence_high) = robust_z(
        f64::from(coherence),
        baseline.coherence_median,
        baseline.coherence_mad,
    );
    SensorReading {
        sharpness: readings.sharpness,
        gradient_structure_tensor_xx_yy_xy: readings.gradient_structure_tensor_xx_yy_xy,
        axis_ratio,
        coherence,
        sharpness_ratio: sharpness / baseline.sharpness_median,
        axis_ratio_z: axis.map(|(z, _)| z),
        coherence_z,
        blurred: soft || axis.is_some_and(|(_, high)| high) || coherence_high,
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
    /// Whether position `frame` lies in a flagged region.
    pub(super) fn is_flagged(&self, frame: usize) -> bool {
        self.flagged_regions
            .iter()
            .any(|r| r.first <= frame && frame <= r.last)
    }

    /// The first flagged position after `frame`.
    pub(super) fn next_flagged(&self, frame: usize) -> Option<usize> {
        self.flagged_regions
            .iter()
            .filter(|r| r.last > frame)
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

/// The sequence's baseline from its first batch: the sensors' medians and
/// spreads over the sampled frames, the rate at each sample, and the regions
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

    let min_matches = options.pair.min_matches;
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
        let weak = NEAR_GAPS
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

/// Judge whether a sequence is a video from its first batch's pairs.
pub(super) fn video_check(
    frames: &Frames,
    sequence: usize,
    baseline: &SequenceBaseline,
    options: &PickImagesOptions,
) -> VideoCheck {
    let id = |frame| FrameId { sequence, frame };
    let len = baseline.len;
    let min_matches = options.pair.min_matches;
    // Per sample with a frame 3 after it: the 1-frame step, the 3-to-1
    // ratio and the 1-frame share, or a value that fails each sign when
    // either pair has fewer than `min_matches` matches.
    let mut steps = Vec::new();
    let mut ratios = Vec::new();
    let mut shares = Vec::new();
    for &s in baseline.samples.iter().filter(|&&s| s + 3 < len) {
        let one = frames.pair(id(s), id(s + 1));
        let three = frames.pair(id(s), id(s + 3));
        match (one, three) {
            (Some(one), Some(three))
                if one.matches >= min_matches && three.matches >= min_matches =>
            {
                steps.push(Some(one.step_percent));
                ratios.push(three.step_percent / one.step_percent);
                shares.push(one.f_inlier_share.unwrap_or(0.0));
            }
            _ => {
                steps.push(None);
                ratios.push(f64::INFINITY);
                shares.push(0.0);
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
    let f_inlier_share_1_median = median(&shares);
    let motion_adds_up = (VIDEO_RATIO_RANGE.0..=VIDEO_RATIO_RANGE.1).contains(&ratio_3_to_1_median);
    let step_small = step_1_median < VIDEO_MAX_STEP_PERCENT;
    let step_steady = step_change_median < VIDEO_MAX_STEP_FACTOR.ln();
    let neighbours_match = f_inlier_share_1_median > VIDEO_MIN_SHARE;
    let signs = [motion_adds_up, step_small, step_steady, neighbours_match]
        .iter()
        .filter(|&&sign| sign)
        .count();
    VideoCheck {
        step_1_median,
        ratio_3_to_1_median,
        step_change_median,
        f_inlier_share_1_median,
        motion_adds_up,
        step_small,
        step_steady,
        neighbours_match,
        is_video: len >= 4 && signs >= VIDEO_SIGNS_NEEDED,
    }
}

/// Judge whether a sequence can be solved from its second batch's pairs.
pub(super) fn solvability_check(
    frames: &Frames,
    sequence: usize,
    baseline: &SequenceBaseline,
) -> SolvabilityCheck {
    let id = |frame| FrameId { sequence, frame };
    let len = baseline.len;
    let medians = |gap: usize, value: fn(&FramePairStats) -> Option<f64>| {
        let values: Vec<f64> = baseline
            .samples
            .iter()
            .filter_map(|&s| frames.pair(id(s), id(s + gap)).and_then(value))
            .collect();
        (!values.is_empty()).then(|| median(&values))
    };
    let f_inlier_share_10_median = medians(10, |s| s.f_inlier_share);
    let h_over_f_30_median = medians(30, |s| s.h_over_f);
    let rotation_error_30_median = medians(30, |s| s.rotation_error);
    let total_motion_percent = baseline.rate_median * len as f64;
    let moving_scene = f_inlier_share_10_median.is_some_and(|s| s < MOVING_SCENE_MAX_SHARE);
    let panorama = h_over_f_30_median.is_some_and(|h| h > PANORAMA_MIN_H_OVER_F)
        && rotation_error_30_median.is_some_and(|e| e < PANORAMA_MAX_ROTATION_ERROR);
    let barely_moves = total_motion_percent < MIN_TOTAL_MOTION_PERCENT;
    SolvabilityCheck {
        f_inlier_share_10_median,
        h_over_f_30_median,
        rotation_error_30_median,
        total_motion_percent,
        moving_scene,
        panorama,
        barely_moves,
        solvable: !(moving_scene || panorama || barely_moves),
    }
}
