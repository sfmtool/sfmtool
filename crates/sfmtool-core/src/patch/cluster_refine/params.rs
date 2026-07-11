// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Configuration and result types for cluster-patch refinement: the per-image
//! [`FeatureGeometry`] views, the [`ClusterRefineParams`] bundle, the
//! per-member [`MemberStatus`], and the member-parallel
//! [`ClusterRefineResult`].

use ndarray::{Array3, ArrayView2, ArrayView3};

use crate::patch::normal_refine::PatchWindow;

/// Per-member refinement status.
///
/// Discriminants MUST match `matches_format::ClusterMemberStatus` — this crate
/// does not depend on `matches-format`, so the PyO3 binding passes the `u8`
/// array straight into the `cluster_patches/` section without translation.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemberStatus {
    /// The cluster's reference member (identity affine, ZNCC 1.0).
    Reference = 0,
    /// Refined and vetted successfully.
    Kept = 1,
    /// Rejected: achieved ZNCC below [`ClusterRefineParams::min_zncc`].
    RejectedLowZncc = 2,
    /// Rejected: translation drifted more than
    /// [`ClusterRefineParams::max_shift_px`] from the SIFT seed.
    RejectedShift = 3,
    /// Outscored by another kept member in the same image, or shares the
    /// reference's image.
    DuplicateImage = 4,
    /// Not evaluated: degenerate shape, template/seed support out of frame,
    /// or the cluster itself was unrefinable.
    NotEvaluated = 5,
    /// Rejected: the member's own patch scored a keypoint position
    /// uncertainty above
    /// [`ClusterRefineParams::max_keypoint_uncertainty`] (excluded before
    /// reference selection and refinement).
    RejectedUnlocalizable = 6,
}

/// Sentinel in [`ClusterRefineResult::reference_members`] for a cluster with
/// no usable reference (mirrors
/// `matches_format::CLUSTER_REFERENCE_UNREFINABLE`).
pub const REFERENCE_UNREFINABLE: u32 = u32::MAX;

/// Tunables for [`refine_cluster_patches`](super::refine_cluster_patches).
#[derive(Clone, Debug)]
pub struct ClusterRefineParams {
    /// Template half-width, keypoint-frame units (the reference's SIFT affine
    /// shape maps one keypoint-frame unit to source pixels).
    pub radius: f64,
    /// Support samples per axis (the template is `resolution²` samples).
    pub resolution: u32,
    /// Per-sample scoring weight over the template grid. `sigma` is in
    /// [`PatchWindow`]'s normalized patch coordinates (the grid spans
    /// `[-1, 1]²`), so the prototype's Gaussian of `radius / 2`
    /// keypoint-frame units is `sigma = 0.5`.
    pub window: PatchWindow,
    /// Member acceptance threshold on the achieved windowed ZNCC.
    pub min_zncc: f64,
    /// Max translation drift from the SIFT seed, source-image pixels.
    pub max_shift_px: f64,
    /// Localizability gate: exclude a member up front (before reference
    /// selection and refinement) when its own patch's noise-normalized
    /// weak-axis positional uncertainty `σ_pos` exceeds this, in
    /// template-grid px (see `specs/core/patch-localizability.md`). `0`
    /// disables the gate. The default value matches `embed-patches`'
    /// `--max-keypoint-uncertainty`, though the score here is measured on
    /// the member's template-grid patch with [`Self::window`] rather than
    /// on the consensus with the scorer's frozen window.
    pub max_keypoint_uncertainty: f64,
    /// Nelder-Mead iterations per cascade stage.
    pub max_iters: u32,
    /// Simplex value-spread stop threshold.
    pub convergence: f64,
}

impl Default for ClusterRefineParams {
    fn default() -> Self {
        Self {
            radius: 4.0,
            resolution: 15,
            // The prototype's window: a Gaussian of sigma = radius/2 in
            // keypoint-frame units = 0.5 in the normalized [-1, 1] patch
            // coordinate `PatchWindow` uses (= resolution/4 in grid px),
            // confined to the inscribed disk per the house default.
            window: PatchWindow::GaussianDisk { sigma: 0.5 },
            min_zncc: 0.85,
            max_shift_px: 3.0,
            max_keypoint_uncertainty: 0.35,
            max_iters: 120,
            convergence: 1e-5,
        }
    }
}

/// One image's SIFT feature geometry (borrowed views of the `.sift` arrays).
pub struct FeatureGeometry<'a> {
    /// `(N, 2)` keypoint positions in source-image pixels (COLMAP pixel
    /// convention, centers at `+0.5`).
    pub positions_xy: ArrayView2<'a, f32>,
    /// `(N, 2, 2)` SIFT affine shapes: keypoint-frame → pixel offsets.
    pub affine_shapes: ArrayView3<'a, f32>,
}

/// Member-parallel result of
/// [`refine_cluster_patches`](super::refine_cluster_patches); the arrays map
/// 1:1 onto the `cluster_patches/` section of the `.matches` format.
pub struct ClusterRefineResult {
    /// `(C,)` global member index of each cluster's reference, or
    /// [`REFERENCE_UNREFINABLE`].
    pub reference_members: Vec<u32>,
    /// `(M,)` per-member statuses.
    pub member_status: Vec<MemberStatus>,
    /// `(M, 2, 3)` absolute affine warps in pixel coordinates:
    /// `x_member = A·x_ref + t` with `A` the leading 2×2 and `t` the last
    /// column. Identity|0 for the reference row; zeros where not evaluated.
    pub member_affines: Array3<f64>,
    /// `(M,)` achieved windowed ZNCC vs the reference (`NaN` if not
    /// evaluated).
    pub member_zncc: Vec<f32>,
    /// `(M,)` translation drift from the SIFT seed, source-image pixels
    /// (`NaN` if not evaluated).
    pub member_shift_px: Vec<f32>,
}
