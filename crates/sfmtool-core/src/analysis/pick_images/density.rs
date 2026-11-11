// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The local density term: how the walk shortens its target where the
//! steps it has chosen show a near foreground in front of a distant view.
//! See `specs/core/analysis/pick-images.md` § "The local density term".

use serde::Serialize;

use super::essential_split::{EssentialSplitOptions, EssentialSplitOutcome};
use crate::camera::CameraIntrinsics;
use crate::numeric::median;

/// How many of the last chosen steps the median measure is taken over.
pub(super) const WINDOW: usize = 3;

/// The median measure at which the gain starts to rise above 1.
pub(super) const UNIT_GAIN_MEASURE: f64 = 2.5;

/// The fewest essential inliers a step's split has for its measure to count.
pub(super) const MIN_INLIERS: usize = 50;

/// The least parallax P90, in degrees, a step has for its measure to count.
pub(super) const MIN_PARALLAX_P90_DEG: f64 = 0.2;

/// A step's parallax P50, in degrees, is below this for its measure to count.
pub(super) const MAX_PARALLAX_P50_DEG: f64 = 0.5;

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

/// A step's measure: its parallax P90 over its parallax P50 when its split
/// was measured with at least [`MIN_INLIERS`] inliers, a P90 of at least
/// [`MIN_PARALLAX_P90_DEG`] and a positive P50 below
/// [`MAX_PARALLAX_P50_DEG`], and 1 otherwise.
pub(super) fn step_measure(split: &EssentialSplitOutcome) -> f64 {
    match split {
        EssentialSplitOutcome::Measured(s)
            if s.inliers >= MIN_INLIERS
                && s.parallax_p90_deg >= MIN_PARALLAX_P90_DEG
                && s.parallax_p50_deg > 0.0
                && s.parallax_p50_deg < MAX_PARALLAX_P50_DEG =>
        {
            s.parallax_p90_deg / s.parallax_p50_deg
        }
        _ => 1.0,
    }
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
