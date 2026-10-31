// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared patch rendering and scoring helpers.
//!
//! The patch kernels (view selection, member coherence, keypoint localization,
//! keypoint sub-pixel refinement, cluster refinement, the stored bitmap and the
//! bench) render a patch from several photographs and correlate the renders.
//! This module holds the pieces they share: the [`ProjectedImage`] a patch is
//! rendered from, the [`PatchWindow`] scoring weight, the support and ZNCC
//! helpers, the frozen common support of several views, the per-channel
//! z-normalization, the robust IRLS view weights, and the [`viewing_angle`] of a
//! view to a patch. See `specs/core/patch/patch-view-selection.md` § "The shared
//! render and score helpers".
//!
//! The module is named for the photometric normal refinement it once held; the
//! grown-plane normal (`bench::grown_plane_normal`) replaced that refinement.

mod consensus;
mod level;
mod obliquity;
mod params;
mod support;
mod znorm;

// Public API.
pub use obliquity::{surface_to_camera, viewing_angle, ViewingAngle, MIN_TILT_ANGLE_DEG};
pub use params::{PatchWindow, ProjectedImage, Sampler, SamplerChoice};

// Patch-internal helpers consumed by sibling `patch` modules (keypoint_localize,
// keypoint_subpixel, view_selection, member_coherence, cluster_refine, …) at
// their `crate::patch::normal_refine::<name>` paths.
pub(in crate::patch) use consensus::{irls_view_weights, ConsensusScratch};
pub(in crate::patch) use level::{build_level_context, LevelContext};
pub(in crate::patch) use params::{FLAT_NORM_SQ_EPS, MIN_MASK_PIXELS};
pub(crate) use support::window_weights;
pub(in crate::patch) use support::{
    build_support, grid_bounds, grid_cell_centres, middle_span, view_render_patch, PartZncc, Parts,
    Support, ViewSamplers,
};
pub(in crate::patch) use znorm::{
    normalized_stack, weighted_moments_pub, znorm_write, znormalize_into_kept,
};

#[cfg(test)]
mod tests;
