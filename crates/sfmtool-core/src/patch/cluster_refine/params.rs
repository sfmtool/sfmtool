// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Configuration and result types for cluster-patch refinement: the per-image
//! [`FeatureGeometry`] views, the [`ClusterRefineParams`] bundle, the
//! per-member [`MemberStatus`], and the member-parallel
//! [`ClusterRefineResult`].

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use crate::patch::keypoint_localize::DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS;
use crate::patch::normal_refine::PatchWindow;

use super::piecewise::{CellRefinement, PiecewiseParams};

/// Per-member refinement status.
///
/// Discriminants MUST match `sfmtool_matches_format::ClusterMemberStatus` — this crate
/// does not depend on `sfmtool-matches-format`, so the PyO3 binding passes the `u8`
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
    /// Rejected: the member's own patch does not pin a position, its ZNCC
    /// self-similarity radius is above
    /// [`ClusterRefineParams::max_member_zncc_self_similarity_radius`]
    /// (excluded before reference selection and refinement).
    RejectedUnlocalizable = 6,
}

/// Sentinel in [`ClusterRefineResult::reference_members`] for a cluster with
/// no usable reference (mirrors
/// `sfmtool_matches_format::CLUSTER_REFERENCE_UNREFINABLE`).
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
    /// Exclude a member up front (before reference selection and refinement)
    /// when its own patch does not pin a 2D position: its
    /// [ZNCC self-similarity radius](crate::patch::self_similarity), how far
    /// the patch can slide over itself and still match itself as well as a
    /// true match between two views would, is above this bar, in
    /// template-grid px. The patch is the member's `R×R` grid at its SIFT
    /// seed geometry ([`sample_member_grid`](super::sample_member_grid)), read
    /// the overlap way with the default
    /// [`SelfSimilarityParams`](crate::patch::self_similarity::SelfSimilarityParams),
    /// so no pixel outside the grid enters it
    /// ([`member_zncc_self_similarity_radius`](super::member_zncc_self_similarity_radius)).
    /// A flat or edge-only member matches itself along the edge or
    /// everywhere, so its ZNCC to the reference cannot place it.
    ///
    /// A member passes when its radius is at or below the bar; a `NaN`
    /// radius fails. The radius reads at most `max_radius` (`3`), which
    /// stands for "that far or further", so a bar at or above it turns
    /// nothing out. `0.0` (or a non-finite value) disables the gate exactly.
    /// See [`Self::admits_member_zncc_self_similarity_radius`]. The default is
    /// the keypoint localizer's
    /// [`DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS`], `2.5`, so the two
    /// member gates start from the same bar.
    pub max_member_zncc_self_similarity_radius: f64,
    /// Nelder-Mead iterations per cascade stage.
    pub max_iters: u32,
    /// Simplex value-spread stop threshold for the affine stage (the stored
    /// result).
    pub convergence: f64,
    /// Simplex value-spread stop threshold for the shift and similarity
    /// stages, which only seed the next stage — looser than
    /// [`Self::convergence`] (dino_dog_toy: −30% intermediate-stage
    /// evaluations for a kept-set change under 0.1%).
    pub intermediate_convergence: f64,
    /// Stall exit for every stage: stop when the best simplex value has not
    /// improved by more than [`Self::stall_tol`] for this many consecutive
    /// iterations. Releases the reflect-heavy affine crawl (which otherwise
    /// runs into [`Self::max_iters`] long after the score stopped moving)
    /// without truncating members that are still improving.
    pub stall_iters: u32,
    /// Minimum best-value improvement (ZNCC units) that counts as progress
    /// for the stall exit.
    pub stall_tol: f64,
    /// The piecewise refinement that follows the cascade for every kept
    /// member: the nine cells of the template are registered separately at
    /// the member's cascade shape, and [`ClusterRefineResult::cells`] carries
    /// what each cell read. The member's shape, position and readings stay
    /// the cascade's unless [`PiecewiseParams::move_shape`] lets the cells'
    /// shifts refine them. `None` skips the stage, leaving every output the
    /// cascade's.
    ///
    /// Off (`None`) by default. `sfm cluster-patches --piecewise` turns it on
    /// and stores the cells as the `.matches` per-cell entries (format
    /// version 8 and later, through
    /// [`member_cell_data`](super::member_cell_data)). Whether it becomes the
    /// default is decided once a consumer reads the cells. See
    /// `specs/drafts/cluster-patches-piecewise-refinement.md`.
    pub piecewise: Option<PiecewiseParams>,
}

impl Default for ClusterRefineParams {
    fn default() -> Self {
        Self {
            radius: 6.0,
            resolution: 25,
            // The prototype's window: a Gaussian of sigma = radius/2 in
            // keypoint-frame units = 0.5 in the normalized [-1, 1] patch
            // coordinate `PatchWindow` uses (= resolution/4 in grid px),
            // confined to the inscribed disk per the house default.
            window: PatchWindow::GaussianDisk { sigma: 0.5 },
            min_zncc: 0.85,
            max_shift_px: 3.0,
            max_member_zncc_self_similarity_radius: DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS,
            max_iters: 120,
            convergence: 1e-5,
            // Tuned on dino_dog_toy (85 images, 105K clusters): together
            // these cut objective evaluations ~19% (265 → 213 per member)
            // for a +0.03% kept-set delta and a slightly better
            // warp-consistency profile, and they keep the synthetic
            // warp-recovery suite green. See
            // specs/core/patch/cluster-patch-refinement.md ("Performance").
            intermediate_convergence: 1e-4,
            stall_iters: 20,
            stall_tol: 1e-4,
            piecewise: None,
        }
    }
}

impl ClusterRefineParams {
    /// Whether [`Self::max_member_zncc_self_similarity_radius`] is on: finite
    /// and above `0`.
    pub fn member_self_similarity_gate_is_on(&self) -> bool {
        let bar = self.max_member_zncc_self_similarity_radius;
        bar.is_finite() && bar > 0.0
    }

    /// Whether a member whose own patch reads `radius` passes the member
    /// gate: always when the gate is off, otherwise when `radius` is at or
    /// below [`Self::max_member_zncc_self_similarity_radius`]. A `NaN` radius
    /// fails an active gate.
    pub fn admits_member_zncc_self_similarity_radius(&self, radius: f64) -> bool {
        !self.member_self_similarity_gate_is_on()
            || radius <= self.max_member_zncc_self_similarity_radius
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
/// [`refine_cluster_patches`](super::refine_cluster_patches).
///
/// The geometry arrays are the refinement's answer for the members it
/// measured — the reference, every kept member, and the ZNCC/shift-rejected
/// ones, which keep their measurement so a consumer can re-gate. A member the
/// cascade never fitted (`NotEvaluated`, `RejectedUnlocalizable`, and a
/// `DuplicateImage` that shared the reference's image) has an all-zero row:
/// [`Self::member_status`] is what says which, and a caller writing a
/// `.matches` file leaves those members' detections in place rather than
/// storing a zero.
pub struct ClusterRefineResult {
    /// `(C,)` global member index of each cluster's reference, or
    /// [`REFERENCE_UNREFINABLE`].
    pub reference_members: Vec<u32>,
    /// `(M,)` per-member statuses.
    pub member_status: Vec<MemberStatus>,
    /// `(M, 2)` the member's refined absolute keypoint position `p` in source
    /// image pixels. All-zeros where the member was never fitted; the
    /// reference's own row is its detected position.
    pub member_positions: Array2<f64>,
    /// `(M, 2, 2)` the member's absolute affine SHAPE `S = W·S_ref` — the map
    /// from the detector's canonical unit frame onto that member's image
    /// pixels, so its column norms are the member's image-space extent. The
    /// reference→member warp is `W = S·S_ref⁻¹` and then reads
    /// `x_member = W·(x − x_ref) + p`, with the reference's own row holding
    /// `S_ref` (its detected shape). All-zeros where never fitted.
    pub member_affine_shapes: Array3<f64>,
    /// `(M,)` achieved windowed ZNCC vs the reference (`NaN` if not
    /// evaluated).
    pub member_zncc: Vec<f32>,
    /// `(M,)` the middle ZNCC beside [`Self::member_zncc`]: the member's
    /// samples at the same final map against the reference's, read over only
    /// the middle square of the grid, the rows and columns `R/4 .. R - R/4`.
    /// A whole-patch agreement the middle does not share is carried by the
    /// parts of the patch away from its centre. `NaN` if not evaluated, or
    /// where the reference's middle is flat.
    pub member_zncc_middle: Vec<f32>,
    /// `(M,)` the ZNCC grid beside [`Self::member_zncc`]: the member's samples
    /// at the same final map against the reference's, read over each cell of
    /// a three-by-three split of the grid (rows and columns cut at `R/3` and
    /// `R - R/3`) with every pixel weighted equally, `grid[row][col]` from the
    /// top-left cell. All `NaN` if not evaluated; one cell is `NaN` where the
    /// reference is flat over it.
    pub member_zncc_grid: Vec<[[f32; 3]; 3]>,
    /// `(M,)` translation drift from the SIFT seed, source-image pixels
    /// (`NaN` if not evaluated).
    pub member_shift_px: Vec<f32>,
    /// `(M,)` the piecewise refinement's cells, one entry per member, indexed
    /// like every other per-member array: `Some` for a member whose status is
    /// [`MemberStatus::Kept`] when the stage ran, `None` for every other
    /// member, and `None` throughout when [`ClusterRefineParams::piecewise`]
    /// is `None`.
    ///
    /// Where the stage moved a kept member, which it does only with
    /// [`PiecewiseParams::move_shape`], that member's shape and position
    /// are the stage's, and its ZNCC, middle ZNCC, ZNCC grid and shift are
    /// read again at that shape and position; its status stays the cascade's.
    pub cells: Vec<Option<CellRefinement>>,
}
