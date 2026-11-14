// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The local density term: how the walk shortens its target where the
//! steps it has chosen show a near foreground in front of a distant view,
//! read from each step, or from a reference pair of a shorter step, with a
//! search over the focal.
//! See `specs/core/analysis/pick-images.md` § "The local density term".

use serde::Serialize;

use super::essential_split::{EssentialSplitOptions, EssentialSplitOutcome};
use crate::camera::{CameraIntrinsics, IntrinsicAdjustment};
use crate::numeric::median;

/// How many of the last chosen steps the median measure is taken over.
pub(super) const WINDOW: usize = 3;

/// The median measure at which the gain starts to rise above 1.
pub(super) const UNIT_GAIN_MEASURE: f64 = 2.5;

/// The fewest essential inliers a pair's split has for its measure to count.
pub(super) const MIN_INLIERS: usize = 50;

/// The least parallax P90, in degrees, a pair has for its measure to count.
pub(super) const MIN_PARALLAX_P90_DEG: f64 = 0.2;

/// The step's own parallax P50, in degrees, is below this for its measure to
/// count.
pub(super) const MAX_STEP_PARALLAX_P50_DEG: f64 = 0.5;

/// A reference pair's parallax P50, in degrees, is below this for its
/// measure to count.
pub(super) const MAX_REFERENCE_PARALLAX_P50_DEG: f64 = 0.2;

/// How far a step may be from the reference step, as a fraction of it, and
/// still be measured as the reference pair: a step up to `1 +` this times the
/// reference step is measured itself, and a pair the walk has matched is the
/// reference pair when its step is within `1 ±` this times it.
pub(super) const REFERENCE_BAND: f64 = 0.3;

/// The spacing of the focals the focal search measures at, as a fraction of
/// the camera's focal.
pub(super) const FOCAL_SEARCH_STEP: f64 = 0.03;

/// How many focals the focal search measures at on either side of the
/// camera's: `1 + FOCAL_SEARCH_STEP · i` times its focal, for `i` from
/// `-FOCAL_SEARCH_STEPS` to `FOCAL_SEARCH_STEPS`.
pub(super) const FOCAL_SEARCH_STEPS: i32 = 4;

/// The least rotation, in degrees, a pair's split at the camera's focal has
/// for the focal search to run on it.
pub(super) const MIN_FOCAL_SEARCH_ROTATION_DEG: f64 = 0.5;

/// How far the frames' aspect ratio, width over height, may be from the
/// camera's for the camera to be scaled to them: the bound `sfm solve`
/// holds a `.camrig` camera's images to.
const ASPECT_TOLERANCE: f64 = 1e-3;

/// Whether a sequence's walk has the local density term, and why not.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DensityTerm {
    /// The term is on, measuring with the sequence's camera at the frames'
    /// size, whose focal, the mean of its two focal lengths, is `focal_px`.
    On { focal_px: f64 },
    /// The term is off, and the walk aims every step at the base target.
    Off(DensityTermOff),
}

/// Why a sequence's walk has no local density term.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum DensityTermOff {
    /// `max_density_gain` is 1.
    MaxGainOne,
    /// The sequence has no camera.
    NoCamera,
    /// The camera's aspect ratio is not the frames', so it cannot be scaled
    /// to them.
    CameraShape { camera: [u32; 2], frames: [u32; 2] },
    /// The essential split cannot measure with the camera, scaled to the
    /// frames: `problem` says why, as the split's option check does.
    CameraRefused { problem: String },
}

/// A pair's measure: its parallax P90 over its parallax P50 when its split
/// was measured with at least [`MIN_INLIERS`] inliers, a P90 of at least
/// [`MIN_PARALLAX_P90_DEG`] and a positive P50 below `p50_gate_deg`, and 1
/// otherwise.
pub(super) fn pair_measure(split: &EssentialSplitOutcome, p50_gate_deg: f64) -> f64 {
    match split {
        EssentialSplitOutcome::Measured(s)
            if s.inliers >= MIN_INLIERS
                && s.parallax_p90_deg >= MIN_PARALLAX_P90_DEG
                && s.parallax_p50_deg > 0.0
                && s.parallax_p50_deg < p50_gate_deg =>
        {
            s.parallax_p90_deg / s.parallax_p50_deg
        }
        _ => 1.0,
    }
}

/// The pair a step's measure is read from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum MeasurePlan {
    /// The step itself, its parallax P50 gated at `p50_gate_deg`.
    Step { p50_gate_deg: f64 },
    /// No pair: the step's own parallax P50, scaled to the span of the
    /// reference pair from the step's first frame to `to`, is already at
    /// least [`MAX_REFERENCE_PARALLAX_P50_DEG`], so the measure is 1.
    WideParallax { to: usize },
    /// The reference pair from the step's first frame to `to`: a pair the
    /// walk has matched already when `matched`, and otherwise one it reads
    /// and matches for the measure.
    Reference { to: usize, matched: bool },
}

/// The pair the measure of the accepted step from `from` to `to` is read
/// from, given the step `step_percent`, the reference step
/// `reference_percent` (the base target over the maximum gain), the step's
/// own split `step_split`, and `matched`: the frames whose pair with `from`
/// the walk has matched and kept the positions of, with their steps, in
/// order.
///
/// A step up to `1 + REFERENCE_BAND` times the reference step, or one whose
/// step is not finite, is measured itself, gated at
/// [`MAX_STEP_PARALLAX_P50_DEG`]. A longer step is measured by a reference
/// pair from `from`, gated at [`MAX_REFERENCE_PARALLAX_P50_DEG`]. Its frame
/// is where the step's rate puts the reference step:
/// `k = round(span · reference_percent / step_percent)` positions on, kept
/// within `[1, span]`. When the step's split has at least [`MIN_INLIERS`]
/// inliers and its P50 times `k / span` is at least the reference gate, no
/// pair is measured ([`MeasurePlan::WideParallax`]). Otherwise the reference
/// pair is the matched pair whose step is within `1 ± REFERENCE_BAND` times
/// the reference step and nearest it in ratio, the first on a tie, or, with
/// none, the pair to the frame `k` positions on. That frame is `to` itself
/// only for a step of one or two frames, and the step is then measured
/// itself with the reference gate.
pub(super) fn measure_plan(
    from: usize,
    to: usize,
    step_percent: f64,
    reference_percent: f64,
    step_split: &EssentialSplitOutcome,
    matched: &[(usize, f64)],
) -> MeasurePlan {
    if !(step_percent.is_finite() && step_percent > (1.0 + REFERENCE_BAND) * reference_percent) {
        return MeasurePlan::Step {
            p50_gate_deg: MAX_STEP_PARALLAX_P50_DEG,
        };
    }
    let span = to - from;
    let k = (span as f64 * reference_percent / step_percent)
        .round()
        .clamp(1.0, span as f64) as usize;
    if let EssentialSplitOutcome::Measured(s) = step_split {
        if s.inliers >= MIN_INLIERS
            && s.parallax_p50_deg * k as f64 / span as f64 >= MAX_REFERENCE_PARALLAX_P50_DEG
        {
            return MeasurePlan::WideParallax { to: from + k };
        }
    }
    let low = (1.0 - REFERENCE_BAND) * reference_percent;
    let high = (1.0 + REFERENCE_BAND) * reference_percent;
    let mut nearest: Option<(usize, f64)> = None;
    for &(n, step) in matched {
        if !(low..=high).contains(&step) {
            continue;
        }
        let off = (step / reference_percent).ln().abs();
        if nearest.is_none_or(|(_, best)| off < best) {
            nearest = Some((n, off));
        }
    }
    match nearest {
        Some((n, _)) => MeasurePlan::Reference {
            to: n,
            matched: true,
        },
        None if k == span => MeasurePlan::Step {
            p50_gate_deg: MAX_REFERENCE_PARALLAX_P50_DEG,
        },
        None => MeasurePlan::Reference {
            to: from + k,
            matched: false,
        },
    }
}

/// Whether the focal search runs on a pair whose split at the camera's
/// focal is `split`: when it was measured with a rotation of at least
/// [`MIN_FOCAL_SEARCH_ROTATION_DEG`], and its P50, less the most that a
/// focal error the search reaches adds, `FOCAL_SEARCH_STEP ·
/// FOCAL_SEARCH_STEPS` degrees per degree of rotation, is below
/// [`MAX_STEP_PARALLAX_P50_DEG`]. Elsewhere no focal the search measures at
/// could bring the pair under a gate.
pub(super) fn focal_search_runs(split: &EssentialSplitOutcome) -> bool {
    let reach = FOCAL_SEARCH_STEP * f64::from(FOCAL_SEARCH_STEPS);
    matches!(
        split,
        EssentialSplitOutcome::Measured(s)
            if s.rotation_deg >= MIN_FOCAL_SEARCH_ROTATION_DEG
                && s.parallax_p50_deg - reach * s.rotation_deg < MAX_STEP_PARALLAX_P50_DEG
    )
}

/// A pair's split, with the focal search. `measure` measures the pair's
/// split with the options it is given. The split is measured with `split`
/// and, when [`focal_search_runs`] on that, with `split`'s camera at
/// `1 + FOCAL_SEARCH_STEP · i` times its focal for every other `i` from
/// `-FOCAL_SEARCH_STEPS` to `FOCAL_SEARCH_STEPS`, in that order. The split
/// kept is the one with the smallest parallax P50 among those with at least
/// [`MIN_INLIERS`] inliers, the earliest on a tie, the camera's own first.
/// It is returned with its focal as a factor of the camera's: 1 for the
/// camera's own.
pub(super) fn focal_searched_split<E>(
    split: &EssentialSplitOptions,
    mut measure: impl FnMut(&EssentialSplitOptions) -> Result<EssentialSplitOutcome, E>,
) -> Result<(EssentialSplitOutcome, f64), E> {
    let given = measure(split)?;
    if !focal_search_runs(&given) {
        return Ok((given, 1.0));
    }
    let p50 = |outcome: &EssentialSplitOutcome| match outcome {
        EssentialSplitOutcome::Measured(s) if s.inliers >= MIN_INLIERS => s.parallax_p50_deg,
        _ => f64::INFINITY,
    };
    let mut best_p50 = p50(&given);
    let mut best = (given, 1.0);
    for i in (-FOCAL_SEARCH_STEPS..=FOCAL_SEARCH_STEPS).filter(|&i| i != 0) {
        let scale = FOCAL_SEARCH_STEP * f64::from(i);
        let adjustment = IntrinsicAdjustment {
            focal: scale,
            ..IntrinsicAdjustment::default()
        };
        let options = EssentialSplitOptions {
            camera: split.camera.with_adjustment(&adjustment, None),
            ..split.clone()
        };
        let outcome = measure(&options)?;
        let x = p50(&outcome);
        if x < best_p50 {
            best_p50 = x;
            best = (outcome, 1.0 + scale);
        }
    }
    Ok(best)
}

/// The median of `measures`, 1 when there are none, and the gain it gives:
/// the median over [`UNIT_GAIN_MEASURE`], kept within `[1, max_gain]`.
pub(super) fn gain(measures: &[f64], max_gain: f64) -> (f64, f64) {
    let m = if measures.is_empty() {
        1.0
    } else {
        median(measures)
    };
    (m, (m / UNIT_GAIN_MEASURE).clamp(1.0, max_gain))
}

/// The essential split options the term measures with for frames of
/// `width` × `height`: `camera` scaled to that size, or why the term is off.
pub(super) fn split_options(
    camera: Option<&CameraIntrinsics>,
    width: u32,
    height: u32,
    max_gain: f64,
) -> Result<EssentialSplitOptions, DensityTermOff> {
    if max_gain == 1.0 {
        return Err(DensityTermOff::MaxGainOne);
    }
    let camera = camera.ok_or(DensityTermOff::NoCamera)?;
    let shape = || DensityTermOff::CameraShape {
        camera: [camera.width, camera.height],
        frames: [width, height],
    };
    if camera.width == 0 || camera.height == 0 {
        return Err(shape());
    }
    let aspect = |w: u32, h: u32| f64::from(w) / f64::from(h);
    if (aspect(width, height) - aspect(camera.width, camera.height)).abs() > ASPECT_TOLERANCE {
        return Err(shape());
    }
    let scaled = if (width, height) == (camera.width, camera.height) {
        camera.clone()
    } else {
        camera.scaled(
            f64::from(width.max(height)) / f64::from(camera.width.max(camera.height)),
            width,
            height,
        )
    };
    let options = EssentialSplitOptions::new(scaled);
    options
        .validate()
        .map_err(|e| DensityTermOff::CameraRefused {
            problem: e.to_string(),
        })?;
    Ok(options)
}

#[cfg(test)]
mod tests;
