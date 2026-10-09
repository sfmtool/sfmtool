// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Configuration and result types for cluster-patch refinement: the per-image
//! [`FeatureGeometry`] views, the [`ClusterRefineParams`] bundle, the
//! per-member [`MemberStatus`], and the member-parallel
//! [`ClusterRefineResult`].

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use sfmtool_matches_format::{ClusterMemberStatus, CLUSTER_REFERENCE_UNREFINABLE};

use crate::patch::keypoint_localize::DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS;
use crate::patch::normal_refine::PatchWindow;

use super::piecewise::{CellRefinement, PiecewiseParams};

/// Per-member refinement status.
///
/// The discriminants are those of [`ClusterMemberStatus`], one for one (the
/// `const` assertion after this enum checks it), so the PyO3 binding passes the
/// `u8` array straight into the `cluster_patches/` section without translation.
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
    /// Rejected: the member passed the ZNCC and shift gates, but its own patch
    /// read at the shape and position the cascade returned does not pin a
    /// position: its ZNCC self-similarity radius
    /// there is above
    /// [`ClusterRefineParams::max_member_zncc_self_similarity_radius`]. Only
    /// with [`ClusterRefineParams::regate_at_refined_shape`].
    RejectedUnlocalizableRefined = 7,
    /// Rejected: the member passed the ZNCC and shift gates, but more of the
    /// nine cells of its own patch at its refined shape and position read the
    /// largest ZNCC self-similarity radius than
    /// [`ClusterRefineParams::max_capped_cells`] allows.
    RejectedUnlocalizableCells = 8,
}

impl MemberStatus {
    /// Whether the refinement fitted the member, so that its position and
    /// shape are the refinement's answer rather than the detection: the
    /// reference (whose answer is its detection), a kept member, and every
    /// status that rejects a member after its fit. A `.matches` writer stores
    /// these members' refined geometry and the detection for the rest.
    pub fn is_measured(self) -> bool {
        ClusterMemberStatus::from_u8(self as u8).is_some_and(|status| status.is_measured())
    }
}

const _: () = {
    use ClusterMemberStatus as F;
    assert!(MemberStatus::Reference as u8 == F::Reference as u8);
    assert!(MemberStatus::Kept as u8 == F::Kept as u8);
    assert!(MemberStatus::RejectedLowZncc as u8 == F::RejectedLowZncc as u8);
    assert!(MemberStatus::RejectedShift as u8 == F::RejectedShift as u8);
    assert!(MemberStatus::DuplicateImage as u8 == F::DuplicateImage as u8);
    assert!(MemberStatus::NotEvaluated as u8 == F::NotEvaluated as u8);
    assert!(MemberStatus::RejectedUnlocalizable as u8 == F::RejectedUnlocalizable as u8);
    assert!(
        MemberStatus::RejectedUnlocalizableRefined as u8 == F::RejectedUnlocalizableRefined as u8
    );
    assert!(MemberStatus::RejectedUnlocalizableCells as u8 == F::RejectedUnlocalizableCells as u8);
    assert!(F::ALL.len() == 9);
};

/// The number of cells in the three-by-three split of a member's patch that
/// [`ClusterRefineParams::max_capped_cells`] counts over; a bar of this many or
/// more turns nothing out.
pub const CELL_COUNT: u8 = 9;

/// The default of [`ClusterRefineParams::regate_at_refined_shape`].
pub const DEFAULT_REGATE_AT_REFINED_SHAPE: bool = true;

/// The default of [`ClusterRefineParams::max_capped_cells`].
pub const DEFAULT_MAX_CAPPED_CELLS: u8 = 8;

/// Sentinel in [`ClusterRefineResult::reference_members`] for a cluster with
/// no usable reference; the same value as [`CLUSTER_REFERENCE_UNREFINABLE`].
pub const REFERENCE_UNREFINABLE: u32 = CLUSTER_REFERENCE_UNREFINABLE;

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
    /// Read the member gate again at the member's refined shape and position,
    /// for every member that passes the ZNCC and shift gates: the member's own
    /// `R×R` grid is sampled at the shape the cascade returned, its ZNCC
    /// self-similarity radius is read as the up-front gate reads it, and a
    /// member whose radius is above
    /// [`Self::max_member_zncc_self_similarity_radius`] becomes
    /// [`MemberStatus::RejectedUnlocalizableRefined`]. Where the piecewise
    /// stage moves a kept member it is read again at the moved shape, and a
    /// moved shape over the bar is reverted to the cascade's rather than
    /// refused. The up-front gate reads
    /// the SIFT detection's shape, which can sample a different stretch of the
    /// photograph than the shape the member is matched at. Off when the bar
    /// itself is off. See [`Self::refined_shape_verdict`].
    pub regate_at_refined_shape: bool,
    /// The most cells of the three-by-three split of a member's own grid, read
    /// at the refined shape and position as [`Self::regate_at_refined_shape`]
    /// reads the whole grid, that may be *capped*: read the largest ZNCC
    /// self-similarity radius, the default
    /// [`SelfSimilarityParams`](crate::patch::self_similarity::SelfSimilarityParams)'
    /// `max_radius`, which reads "that far or further" (a cell with no
    /// reading counts as capped too). A member with more capped cells becomes
    /// [`MemberStatus::RejectedUnlocalizableCells`]: a patch whose texture
    /// sits in a few cells can pin a position as a whole while most of it
    /// matches anything. [`CELL_COUNT`] (`9`) or more turns nothing out. See
    /// [`Self::admits_capped_cells`].
    pub max_capped_cells: u8,
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
    /// what each cell read. By default ([`PiecewiseParams::move_shape`]) the
    /// cells' shifts refine the member's shape and position where the
    /// whole-member ZNCC does not fall; with `move_shape` off they stay the
    /// cascade's. `None` skips the stage, leaving every output the cascade's.
    ///
    /// Off (`None`) by default. `sfm cluster-patches --piecewise` turns it on
    /// and stores the cells as the `.matches` per-cell entries (format
    /// version 8 and later, through
    /// [`member_cell_data`](super::member_cell_data)). Whether it becomes the
    /// default is decided once a consumer reads the cells. See
    /// `specs/core/patch/cluster-patch-refinement.md`.
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
            regate_at_refined_shape: DEFAULT_REGATE_AT_REFINED_SHAPE,
            max_capped_cells: DEFAULT_MAX_CAPPED_CELLS,
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

    /// Whether [`Self::regate_at_refined_shape`] is on: it is set and the
    /// member gate's bar is on.
    pub fn refined_shape_gate_is_on(&self) -> bool {
        self.regate_at_refined_shape && self.member_self_similarity_gate_is_on()
    }

    /// Whether [`Self::max_capped_cells`] is on: below [`CELL_COUNT`].
    pub fn capped_cell_gate_is_on(&self) -> bool {
        self.max_capped_cells < CELL_COUNT
    }

    /// Whether either gate at the refined shape is on, so that the member's
    /// own grid is read there.
    pub fn reads_refined_shape(&self) -> bool {
        self.refined_shape_gate_is_on() || self.capped_cell_gate_is_on()
    }

    /// Whether a member with `capped` capped cells passes the capped-cell
    /// gate: always when it is off, otherwise when `capped` is at most
    /// [`Self::max_capped_cells`].
    pub fn admits_capped_cells(&self, capped: usize) -> bool {
        capped <= self.max_capped_cells as usize
    }

    /// What the two gates at the refined shape make of a member whose own
    /// grid there reads `radius` as a whole and `cells` per cell: `None` when
    /// it passes both, or both are off, and otherwise the status of the first
    /// it fails, the whole grid's gate before the cells'.
    pub fn refined_shape_verdict(
        &self,
        radius: f64,
        cells: &[[f64; 3]; 3],
    ) -> Option<MemberStatus> {
        if self.refined_shape_gate_is_on()
            && !self.admits_member_zncc_self_similarity_radius(radius)
        {
            return Some(MemberStatus::RejectedUnlocalizableRefined);
        }
        if self.capped_cell_gate_is_on() && !self.admits_capped_cells(capped_cell_count(cells)) {
            return Some(MemberStatus::RejectedUnlocalizableCells);
        }
        None
    }
}

/// How many of a reading's nine cell radii are capped: at or above the default
/// [`SelfSimilarityParams`](crate::patch::self_similarity::SelfSimilarityParams)'
/// `max_radius`, the largest radius the reading reports, which reads "that
/// far or further", or `NaN`, no reading.
pub fn capped_cell_count(cells: &[[f64; 3]; 3]) -> usize {
    let cap = crate::patch::self_similarity::SelfSimilarityParams::default().max_radius as f64;
    cells
        .iter()
        .flatten()
        .filter(|&&r| r.is_nan() || r >= cap)
        .count()
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
    /// are the stage's, and its ZNCC, middle ZNCC, ZNCC grid, shift and
    /// readings at the refined shape are read again at that shape and
    /// position. A move whose shape fails a gate the cascade's shape passed is
    /// not stored: the member keeps the cascade's shape and readings and its
    /// cells are all [`CellStatus::NotAttempted`](super::CellStatus::NotAttempted). The
    /// stage never changes a member's status.
    pub cells: Vec<Option<CellRefinement>>,
    /// `(M,)` the ZNCC self-similarity radius of the member's own grid at its
    /// refined shape and position, the reading
    /// [`ClusterRefineParams::regate_at_refined_shape`] judges, template-grid
    /// px. Read for every member that passed the ZNCC and shift gates when
    /// either gate at the refined shape is on
    /// ([`ClusterRefineParams::reads_refined_shape`]), whether or not it
    /// refuses the member; where the piecewise stage moved the member it is
    /// read at the moved shape. `NaN` for every other member, and throughout
    /// when both gates are off.
    pub refined_zncc_self_similarity_radius: Vec<f32>,
    /// `(M,)` the same reading's nine cell radii, `[row][col]` from the
    /// top-left cell, which [`ClusterRefineParams::max_capped_cells`] counts
    /// ([`capped_cell_count`]); `NaN` where
    /// [`Self::refined_zncc_self_similarity_radius`] is.
    pub refined_zncc_self_similarity_radius_grid: Vec<[[f32; 3]; 3]>,
}
