// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared parameter-string parsers for the patch-kernel bindings.
//!
//! Numeric helpers do not belong here: the bindings take them from
//! `sfmtool_core::numeric`, so a binding and the kernel it wraps compute
//! the same number.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sfmtool_core::patch::cloud::{PatchExtent, PatchNormal, ViewReduce};
use sfmtool_core::patch::normal_refine::{PatchWindow, Sampler, SamplerChoice};
use sfmtool_core::patch::pair_sharpness::PairMatching;

/// The [`PairMatching`] a `matching` string names, `ratio` the factor of
/// `"blur_matched_above_ratio"`.
pub(crate) fn parse_matching(name: &str, ratio: f64) -> PyResult<PairMatching> {
    if !(ratio.is_finite() && ratio >= 1.0) {
        return Err(PyValueError::new_err(format!(
            "min_ellipse_ratio must be a finite number of at least 1, not {ratio}"
        )));
    }
    PairMatching::from_name(name, ratio).ok_or_else(|| {
        PyValueError::new_err(format!(
            "matching must be \"plain\", \"blur_matched\" or \"blur_matched_above_ratio\", \
             not {name:?}"
        ))
    })
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

/// Map a sampler name to the shared [`SamplerChoice`] the patch kernels
/// resample through: `per_view` is the sampler rule at its default threshold,
/// and the three sampler names render every view with that sampler. The
/// companion to [`parse_patch_window`]: every binding that takes a `window`
/// takes a `sampler` beside it.
pub(super) fn parse_sampler(sampler: &str) -> PyResult<SamplerChoice> {
    match sampler {
        "per_view" => Ok(SamplerChoice::per_view()),
        "bilinear" => Ok(Sampler::Bilinear.into()),
        "bilinear_mip" => Ok(Sampler::BilinearMip.into()),
        "anisotropic" => Ok(Sampler::Anisotropic.into()),
        other => Err(PyValueError::new_err(format!(
            "unknown sampler: {other:?} (expected per_view|bilinear|bilinear_mip|anisotropic)"
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
/// Shared by [`crate::PyPatchCloud`]'s `from_reconstruction` and `from_tracks`
/// constructors.
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
/// [`PatchExtent`]. Shared by [`crate::PyPatchCloud`]'s `from_reconstruction`
/// and `from_tracks` constructors.
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

/// Each patch's reference observation as a position in its view set, from the
/// caller's `reference_images` map (`point_index -> image index`). A point
/// absent from the map, mapped to `None` or a negative index, or whose image
/// is not in its view set, gets `None`: the kernel's reference-view rule picks
/// one from the renders at the starting keypoints.
pub(super) fn reference_positions(
    reference_images: Option<&std::collections::HashMap<u32, Option<i64>>>,
    point_indexes: &[u32],
    sets: &[Vec<u32>],
) -> Option<Vec<Option<usize>>> {
    let map = reference_images?;
    Some(
        point_indexes
            .iter()
            .zip(sets)
            .map(|(pid, set)| {
                let image = map.get(pid).copied().flatten()?;
                let image = u32::try_from(image).ok()?;
                set.iter().position(|&v| v == image)
            })
            .collect(),
    )
}
