// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The flow between two crops of a pair, at the pair's pyramid level.
//!
//! The flow is DIS ([`compute_optical_flow`]) at the level's full resolution:
//! `DisFlowParams::high_quality` with `finest_scale` `Some(0)` and
//! `patch_stride` 1. `high_quality` alone fixes `finest_scale` at 1, so on
//! crops this small it solves the whole flow at half resolution and
//! upsamples it, which the prototype measured at 0.093 px median flow error
//! against 0.040 px at full resolution
//! (`specs/drafts/pairwise-flow-normal-measurements.md` § "Flow settings").
//!
//! The flow runs on the CPU. A crop under 224 samples a side is below the
//! presets' `gpu_min_pixels` of 50,000, where the GPU path would run on the
//! CPU anyway; whether the larger crops gain from a `GpuFlowContext` is the
//! draft's open question "GPU flow", left for when the step is built.

use crate::features::optical_flow::{
    compute_optical_flow, DisFlowParams, FlowField, GrayImage, OpticalFlowError,
};

/// The flow settings the step reads its pairs with by default.
pub(crate) fn default_flow() -> DisFlowParams {
    DisFlowParams {
        finest_scale: Some(0),
        patch_stride: 1,
        ..DisFlowParams::high_quality()
    }
}

/// The flow from `from` to `to`, two crops of one pair at the same level;
/// `Ok(None)` where they are narrower or shorter than the flow's
/// `patch_size`, which is a pair with no flow rather than a refusal.
///
/// Any other refusal of the flow is returned: settings the options' check
/// passes can still be refused against a particular size, such as a
/// `finest_scale` deeper than a small crop has levels.
pub(crate) fn pair_flow(
    from: &GrayImage,
    to: &GrayImage,
    params: &DisFlowParams,
) -> Result<Option<FlowField>, OpticalFlowError> {
    let narrowest = from
        .width()
        .min(from.height())
        .min(to.width())
        .min(to.height());
    if narrowest < params.patch_size {
        return Ok(None);
    }
    compute_optical_flow(from, to, params, None).map(Some)
}
