// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared parameter-string parsers and small numeric helpers for the
//! patch-kernel bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sfmtool_core::patch::cloud::{PatchExtent, PatchNormal, ViewReduce};
use sfmtool_core::patch::normal_refine::PatchWindow;

/// `numpy.median` of a non-empty slice: the middle value for an odd count, the
/// mean of the two central values for an even count. Sorts `v` in place.
pub(super) fn np_median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Map a window name + sigma to the shared [`PatchWindow`] kernel.
pub(crate) fn parse_patch_window(window: &str, sigma: f64) -> PyResult<PatchWindow> {
    match window {
        "uniform" => Ok(PatchWindow::Uniform),
        "gaussian" => Ok(PatchWindow::Gaussian { sigma }),
        "gaussian_disk" => Ok(PatchWindow::GaussianDisk { sigma }),
        other => Err(PyValueError::new_err(format!(
            "unknown window: {other:?} (expected uniform|gaussian|gaussian_disk)"
        ))),
    }
}

pub(super) fn parse_reduce(s: &str) -> PyResult<ViewReduce> {
    match s {
        "min" => Ok(ViewReduce::Min),
        "max" => Ok(ViewReduce::Max),
        "median" => Ok(ViewReduce::Median),
        "mean" => Ok(ViewReduce::Mean),
        other => Err(PyValueError::new_err(format!(
            "unknown reduce: {other:?} (expected min|max|median|mean)"
        ))),
    }
}

/// Map the binding's `normal` policy string (+ neighbor count) to [`PatchNormal`].
/// Shared by [`PyPatchCloud::from_reconstruction`] and
/// [`PyPatchCloud::from_tracks`].
pub(super) fn parse_normal(normal: &str, k_neighbors: usize) -> PyResult<PatchNormal> {
    match normal {
        "stored" => Ok(PatchNormal::Stored),
        "mean_viewing" | "mean" => Ok(PatchNormal::MeanViewing),
        "geometric" => Ok(PatchNormal::Geometric { k_neighbors }),
        other => Err(PyValueError::new_err(format!(
            "unknown normal policy: {other:?} (expected stored|mean_viewing|geometric)"
        ))),
    }
}

/// Map the binding's `extent` policy string (+ value and per-axis reduces) to
/// [`PatchExtent`]. Shared by [`PyPatchCloud::from_reconstruction`] and
/// [`PyPatchCloud::from_tracks`].
pub(super) fn parse_extent(
    extent: &str,
    extent_value: f64,
    pixel_reduce: &str,
    feature_reduce: &str,
) -> PyResult<PatchExtent> {
    match extent {
        "fixed" => Ok(PatchExtent::Fixed(extent_value)),
        "relative_spacing" => Ok(PatchExtent::RelativeToSpacing(extent_value)),
        "pixel_radius" => Ok(PatchExtent::PixelRadius {
            radius_px: extent_value,
            across: parse_reduce(pixel_reduce)?,
        }),
        "feature_size" => Ok(PatchExtent::FeatureSize {
            factor: extent_value,
            across: parse_reduce(feature_reduce)?,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown extent policy: {other:?} \
             (expected fixed|relative_spacing|pixel_radius|feature_size)"
        ))),
    }
}
