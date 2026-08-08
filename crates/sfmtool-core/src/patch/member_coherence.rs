// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Track member-coherence validation.
//!
//! See `specs/core/member-coherence-validation.md`. A track's members are the
//! observations of one 3D point. When they image two different surfaces, every
//! member still scores well against the fused cross-view consensus — a balanced
//! split makes that consensus a compromise blend that flatters both sides — so
//! the disagreement only shows in the **pairwise** member agreement.
//!
//! [`member_zncc_matrix`] renders each member's patch and returns the `k×k`
//! matrix of pairwise windowed ZNCC between members; [`decide_member_coherence`]
//! reads a verdict off that matrix (the max-support block plus a separation-margin
//! gate); [`validate_member_coherence`] runs both for one point and
//! [`validate_patch_cloud_member_coherence`] batches them over a [`PatchCloud`].
//!
//! The matrix is rendered through exactly the machinery
//! [view selection](super::view_selection) builds its reference appearance with —
//! `build_level_context` for the frozen common support, `normalized_stack` for
//! the renders and `znormalize_into_kept` for the per-channel z-normalization —
//! so a member's pairwise agreement lives in the same photometric *space* as the
//! member-vs-consensus score selection admits views on. It is not the same
//! *estimator*: selection scores one view against the fused consensus, and does it
//! through the affine fast path where that path's gates allow. Same metric family,
//! same render conventions, agreeing to the affine tolerance documented in
//! `specs/core/patch-view-selection.md` — not the identical number.
//!
//! **Members are sampled at their stored keypoints** when the caller supplies
//! them ([`member_keypoints_from_reconstruction`]): each member's render is
//! recentered in-plane so it is anchored where that image's feature actually
//! *is*, not where the current geometry reprojects the point. See
//! [`member_zncc_matrix`].
//!
//! Members that carry no pairwise evidence (nothing rendered them, or nothing
//! could be correlated with them) are **unscored**: they sit outside the whole
//! decision rule and pass through kept. See `decide_member_coherence`.

use rayon::prelude::*;

use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::normal_refine::{
    build_level_context, normalized_stack, weighted_moments_pub, window_weights,
    znormalize_into_kept, NormalRefineParams, PatchWindow, ProjectedImage, Sampler,
    FLAT_NORM_SQ_EPS, MIN_MASK_PIXELS,
};
use crate::reconstruction::SfmrReconstruction;

/// Tunables for [`validate_member_coherence`].
///
/// The render / window / validity knobs mirror
/// [`ViewSelectParams`](super::view_selection::ViewSelectParams) so the pairwise
/// matrix is built on the same conventions as the reference appearance view
/// selection scores against.
#[derive(Debug, Clone)]
pub struct MemberCoherenceParams {
    /// Pairwise ZNCC at or above which two members are taken to agree — the
    /// edge threshold of the agreement graph the block sweep runs on. Calibrated
    /// against the render conventions below; a caller that changes `window`,
    /// `sampler` or `resolution` should re-pick it.
    pub bar: f64,
    /// Separation-margin floor. A split whose margin (weakest link inside the
    /// winning block minus strongest link leaving it) does not exceed this is
    /// refused: the matrix is a drift chain, not two surfaces, and the track is
    /// kept whole.
    pub margin_gate: f64,
    /// The `R×R` patch grid members are rendered and correlated on.
    pub resolution: u32,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// How to sample the source pyramids when rendering patches.
    pub sampler: Sampler,
    /// Per-member floor on the window-weighted valid-pixel fraction; a member
    /// below it does not cover enough of the patch to be correlated and is left
    /// unscored (its row and column stay `NaN`).
    pub min_valid_fraction: f64,
    /// Floor on the **common** support: the number of pixels valid in *every*
    /// scoreable member, which is what all the pairwise ZNCCs are computed over.
    /// A track whose intersected support falls below it is left entirely unscored
    /// (fail-open: no evidence decides `KeepAll`), with the count still reported
    /// as [`MemberMatrix::n_support`].
    ///
    /// The default `8` is the floor `build_level_context` already enforces
    /// (`MIN_MASK_PIXELS`), so it changes nothing on its own; values below it are
    /// inert for the same reason. A caller vetting wide-baseline tracks — where
    /// the intersection can shrink to a sliver of an `R×R` grid and a correlation
    /// over a handful of pixels is noise — wants it higher.
    pub min_support_pixels: u32,
    /// How many units of the track's **own** core scatter the self-normalized
    /// admission bar sits below its core centre — see [`core_coherence`] and
    /// [`decide_member_coherence`]. The effective admission bar becomes
    /// `max(bar, min(centre − self_bar_k · scatter, `[`SELF_BAR_CEILING`]`))` and
    /// the effective separation-margin floor `min(margin_gate, scatter)`, so a
    /// track whose members agree tightly demands tight agreement of a newcomer
    /// while a noisy or drifting track — large scatter — collapses back to the
    /// absolute `bar` / `margin_gate` pair.
    ///
    /// `0` (or negative) **disables** the relative term entirely: every quantity
    /// is decided at the absolute thresholds, bit for bit.
    pub self_bar_k: f64,
    /// Retained-deficit ratio at or below which a member the **relative** term
    /// alone would evict is spared — see
    /// [multi-scale exoneration](decide_member_coherence#multi-scale-exoneration).
    /// A member's agreement deficit is measured again on the **half-scale** grid
    /// [`MemberMatrix::zncc_coarse`] carries; the ratio of that to the deficit at
    /// full scale says whether the disagreement survives the loss of one octave
    /// of detail (structure — an occluder) or is made of it (spectrum — a soft
    /// frame). It runs high — an occluder retains 0.85–1.00 of its deficit across
    /// one halving — because the test is *survival*, not decay.
    ///
    /// `0` (or negative) **disables** exoneration: the self-normalized rule
    /// decides alone. It is inert whenever the relative term is, since a member
    /// the absolute bar rejects is never a candidate.
    pub exoneration_ratio: f64,
}

/// Upper bound on the self-normalized admission bar: a perfect core cannot
/// demand perfection of a newcomer, whatever [`MemberCoherenceParams::self_bar_k`]
/// and the scatter floor would otherwise ask for.
pub const SELF_BAR_CEILING: f64 = 0.99;

/// Floor on the core scatter [`core_coherence`] reports. A block whose intra-pair
/// agreement is *exactly* uniform has zero measured spread, which would put the
/// admission bar on the centre itself and the margin floor at zero; the floor is
/// the smallest dispersion the rule will believe.
pub const SELF_BAR_MIN_SCATTER: f64 = 0.005;

/// Fewest intra-block pairs [`core_coherence`] will estimate a centre and a
/// scatter from. Six is the pair count of a four-member block; below it the
/// median and its upper-half dispersion are being read off two or three numbers,
/// which is not a measurement of anything, so the relative term stays inactive
/// and the absolute thresholds decide.
pub const SELF_BAR_MIN_PAIRS: usize = 6;

/// The box-downsampling factors [`member_zncc_matrix`] measures the coarse
/// agreement at, in order of increasing coarseness. A factor is skipped when the
/// grid does not divide by it, or when the resulting grid would be smaller than
/// [`MIN_COARSE_RESOLUTION`], so at the default `resolution = 24` these are the
/// `12×12` and `6×6` grids.
pub const COARSE_FACTORS: [u32; 2] = [2, 4];

/// Smallest coarse grid [`COARSE_FACTORS`] will build. Below `4×4` the windowed
/// correlation is taken over a handful of cells and reports noise.
pub const MIN_COARSE_RESOLUTION: u32 = 4;

/// Smallest full-scale agreement deficit a retained-deficit ratio is computed
/// from. The ratio is a quotient by that deficit, and below this floor the
/// member is not measurably out of step with its core at all, so the quotient is
/// reading rounding. Such a member is **not** exonerated: exoneration requires
/// positive evidence that a real deficit decays, not the absence of a deficit.
pub const EXONERATION_MIN_DEFICIT: f64 = 0.01;

impl Default for MemberCoherenceParams {
    fn default() -> Self {
        Self {
            bar: 0.65,
            margin_gate: 0.05,
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: Sampler::BilinearMip,
            min_valid_fraction: 0.6,
            min_support_pixels: MIN_MASK_PIXELS as u32,
            self_bar_k: 1.5,
            exoneration_ratio: 0.90,
        }
    }
}

/// Which members of a row-major `k×k` pairwise table carry evidence: a member is
/// **scored** iff it has at least one finite off-diagonal entry.
///
/// The one definition of "scored" in this module — [`MemberMatrix`] reports it and
/// [`decide_member_coherence`] runs its whole rule over exactly these members, so
/// the two layers cannot disagree about who is in play.
pub fn scored_mask(zncc: &[f64], k: usize) -> Vec<bool> {
    (0..k)
        .map(|i| (0..k).any(|j| j != i && zncc[i * k + j].is_finite()))
        .collect()
}

/// One point's pairwise member agreement.
#[derive(Debug, Clone, Default)]
pub struct MemberMatrix {
    /// The point's members as image indices, deduplicated first-seen-wins and in
    /// that order. Every other field is indexed by position in this list.
    pub members: Vec<u32>,
    /// Row-major `k×k` windowed ZNCC between members. The diagonal is `1.0`;
    /// `NaN` marks a pair that could not be correlated (either member unscored).
    pub zncc: Vec<f64>,
    /// The same `k×k` agreement measured again on **box-downsampled** copies of
    /// the very same renders, one table per surviving [`COARSE_FACTORS`] entry
    /// and in that order (coarsest last). Empty when no coarse scale could be
    /// built (the grid does not divide, or nothing was rendered).
    ///
    /// The renders are not re-sampled: each coarse cell is the mean of the fine
    /// pixels of the frozen common support inside it, so every scale reads the
    /// same pixels through a wider aperture. The pair scores are otherwise
    /// computed exactly as [`zncc`](Self::zncc) is — window weights recomputed at
    /// the coarse grid, per-channel z-normalization, unit-norm dot product — so
    /// the tables are directly comparable to it and to each other.
    ///
    /// This is what separates a **structural** disagreement from a **spectral**
    /// one: an occluding member differs from its core at every scale, a soft
    /// frame only at the finest. See
    /// [multi-scale exoneration](decide_member_coherence#multi-scale-exoneration).
    pub zncc_coarse: Vec<Vec<f64>>,
    /// The downsampling factor of each [`zncc_coarse`](Self::zncc_coarse) table,
    /// parallel to it — a subset of [`COARSE_FACTORS`], in the same order.
    pub coarse_factors: Vec<u32>,
    /// Per member: whether it carries pairwise evidence — see [`scored_mask`],
    /// the single definition this and the decision rule share. An unscored member
    /// has a `NaN` row and column but still holds its `1.0` diagonal.
    pub scored: Vec<bool>,
    /// Size of the **common** support every pairwise ZNCC was computed over: the
    /// number of patch-grid pixels valid in every scoreable member, after the
    /// per-member validity gate. One number per point, because the support is
    /// frozen once per point (intersected over its members) rather than per pair.
    /// `0` when no support could be built at all, and for a matrix handed in
    /// through [`from_zncc`](Self::from_zncc) (no render happened).
    pub n_support: u32,
}

impl MemberMatrix {
    /// Build a matrix from an already-computed row-major `k×k` ZNCC table (the
    /// entry point for callers holding their own pairwise scores, and for tests).
    /// The diagonal is forced to `1.0`, `scored` is derived by [`scored_mask`],
    /// `n_support` is `0` (nothing was rendered) and no coarse scale is carried,
    /// which leaves [multi-scale
    /// exoneration](decide_member_coherence#multi-scale-exoneration) inactive.
    ///
    /// # Panics
    ///
    /// Panics if `zncc.len() != members.len() * members.len()`.
    pub fn from_zncc(members: Vec<u32>, zncc: Vec<f64>) -> Self {
        Self::from_zncc_scales(members, zncc, Vec::new(), Vec::new())
    }

    /// [`from_zncc`](Self::from_zncc) with the coarse-scale tables supplied — the
    /// entry point for tests and callers that measure their own multi-scale
    /// agreement. Every table's diagonal is forced to `1.0`; `scored` is derived
    /// from the **full-scale** table alone, which is the one the decision rule
    /// admits members on.
    ///
    /// # Panics
    ///
    /// Panics if any table is not `k×k`, or if `coarse_factors` is not parallel
    /// to `zncc_coarse`.
    pub fn from_zncc_scales(
        members: Vec<u32>,
        mut zncc: Vec<f64>,
        mut zncc_coarse: Vec<Vec<f64>>,
        coarse_factors: Vec<u32>,
    ) -> Self {
        let k = members.len();
        assert_eq!(zncc.len(), k * k, "zncc must be a k*k row-major matrix");
        assert_eq!(
            zncc_coarse.len(),
            coarse_factors.len(),
            "coarse_factors must be parallel to zncc_coarse"
        );
        for i in 0..k {
            zncc[i * k + i] = 1.0;
        }
        for table in &mut zncc_coarse {
            assert_eq!(
                table.len(),
                k * k,
                "every coarse table must be a k*k row-major matrix"
            );
            for i in 0..k {
                table[i * k + i] = 1.0;
            }
        }
        let scored = scored_mask(&zncc, k);
        Self {
            members,
            zncc,
            zncc_coarse,
            coarse_factors,
            scored,
            n_support: 0,
        }
    }

    /// Member count `k`.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the point has no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Pairwise ZNCC between members `i` and `j`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.zncc[i * self.members.len() + j]
    }
}

/// What the pairwise matrix says about a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemberVerdict {
    /// The members are one surface (or one appearance continuum): keep them all.
    #[default]
    KeepAll,
    /// A strict majority of the members agree and the rest are separated from
    /// them by a real gap: keep the block, reject the outsiders.
    Split,
    /// The members support two incompatible surfaces with neither in the
    /// majority: the point should not ship.
    Retire,
}

/// The verdict [`decide_member_coherence`] reads off a [`MemberMatrix`], with the
/// quantities it was decided on.
#[derive(Debug, Clone)]
pub struct MemberDecision {
    /// The verdict. `Default` is [`MemberVerdict::KeepAll`] over an empty track.
    pub verdict: MemberVerdict,
    /// Per member: whether the point keeps it. All `true` for
    /// [`MemberVerdict::KeepAll`], all `false` for [`MemberVerdict::Retire`]
    /// (the point ships nothing at all); for [`MemberVerdict::Split`] the winning
    /// block **plus every unscored member** — an unscored member is missing
    /// evidence, not contrary evidence, so nothing here can evict it.
    pub kept: Vec<bool>,
    /// The winning max-support block, over the *scored* members only: an unscored
    /// member is never in the block (it took no part in the sweep) even when
    /// [`kept`](Self::kept). Equals `kept` on a [`MemberVerdict::Split`] of a
    /// fully-scored track; on a [`MemberVerdict::Retire`] the point ships nothing
    /// and the block is **informational** — on a balanced split the two sides are
    /// interchangeable and only the deterministic tie-break decides which one is
    /// reported.
    pub block: Vec<bool>,
    /// Size of the winning block (its support count, itself included). `0` when
    /// fewer than two members were scored, i.e. there was no sweep.
    pub support: u32,
    /// Separation margin: [`min_intra`](Self::min_intra) −
    /// [`max_cross`](Self::max_cross). `NaN` when the block holds fewer than two
    /// members, spans every scored member, or has no finite link on one of the two
    /// sides — i.e. `NaN` means *no cut was on the table*, which is a different
    /// thing from a cut the gate refused (a finite margin at or below
    /// `margin_gate`).
    pub margin: f64,
    /// Weakest finite link inside the winning block (`NaN` when it has none).
    pub min_intra: f64,
    /// Strongest finite link from the winning block to a *scored* member outside
    /// it (`NaN` when there is none).
    pub max_cross: f64,
    /// The admission bar the winning block was actually swept at:
    /// [`MemberCoherenceParams::bar`] when the relative term is off or inactive,
    /// higher when the track's own core coherence tightened it. `NaN` when no
    /// sweep ran (fewer than two scored members, or an empty track), which is
    /// what distinguishes "no bar was applied" from "the absolute bar was".
    /// `effective_bar > bar` is exactly "the relative term engaged".
    pub effective_bar: f64,
    /// The separation-margin floor the cut was actually gated on:
    /// [`MemberCoherenceParams::margin_gate`], or the core scatter when that is
    /// smaller. `NaN` when no sweep ran.
    pub effective_margin_gate: f64,
    /// Centre of the pass-1 block's intra-pair agreement — the `c` of
    /// [`core_coherence`]. `NaN` when the relative term was inactive (disabled,
    /// or too few intra-block pairs to estimate from).
    pub core_center: f64,
    /// Scatter of that agreement — the `σ` of [`core_coherence`], floored at
    /// [`SELF_BAR_MIN_SCATTER`]. `NaN` when the relative term was inactive.
    pub core_scatter: f64,
    /// Per member: whether the **relative** term alone put it outside the block —
    /// it clears the absolute `bar` against the pass-1 block but not the
    /// tightened `effective_bar`. All `false` when the relative term did not
    /// engage, and all `false` when no cut was on the table (the margin gate
    /// refused, or the block spans every scored member): nothing was flagged
    /// because nothing was being evicted.
    ///
    /// These are the only members [multi-scale
    /// exoneration](decide_member_coherence#multi-scale-exoneration) can spare.
    pub relative_flagged: Vec<bool>,
    /// Per member: whether exoneration spared it — a subset of
    /// [`relative_flagged`](Self::relative_flagged). An exonerated member is
    /// [`kept`](Self::kept) but stays out of [`block`](Self::block), which is the
    /// block the sweep actually produced.
    pub exonerated: Vec<bool>,
    /// Per member: the **retained-deficit ratio** — the member's agreement
    /// deficit against the block on the coarsest available grid scale, over the
    /// same deficit at full scale. Near `1` the disagreement survives the loss of
    /// the fine detail (structure); near `0` it is made of it (spectrum).
    ///
    /// `NaN` where it was not computed or is undefined: an unflagged member, a
    /// matrix carrying no coarse scale, a block with fewer than two other
    /// members, or a full-scale deficit at or below [`EXONERATION_MIN_DEFICIT`].
    pub retained_deficit: Vec<f64>,
    /// Per member: **photometric sharpness relative to the track consensus** —
    /// the part of the member's agreement deficit that exists only at fine scale,
    /// `deficit_full − deficit_coarsest`, against the block excluding the member
    /// itself. `0` for a member whose disagreement (if any) is scale-free;
    /// positive and growing for a member that agrees with the block coarsely and
    /// not finely, which is what defocus and motion blur do.
    ///
    /// Unlike [`retained_deficit`](Self::retained_deficit) this is computed for
    /// **every** scored member, flagged or not, so it describes the observations
    /// the point ships rather than the ones it was thinking about evicting.
    /// `NaN` where the two deficits are not both defined.
    pub sharpness_deficit: Vec<f64>,
}

impl Default for MemberDecision {
    /// [`MemberVerdict::KeepAll`] over an empty track: nothing kept, nothing in
    /// the block, and every quantity undefined (`NaN`) rather than zero — an empty
    /// track has no margin, and `0.0` would read as a measured one.
    fn default() -> Self {
        Self {
            verdict: MemberVerdict::KeepAll,
            kept: Vec::new(),
            block: Vec::new(),
            support: 0,
            margin: f64::NAN,
            min_intra: f64::NAN,
            max_cross: f64::NAN,
            effective_bar: f64::NAN,
            effective_margin_gate: f64::NAN,
            core_center: f64::NAN,
            core_scatter: f64::NAN,
            relative_flagged: Vec::new(),
            exonerated: Vec::new(),
            retained_deficit: Vec::new(),
            sharpness_deficit: Vec::new(),
        }
    }
}

/// One point's matrix and the verdict read off it.
#[derive(Debug, Clone, Default)]
pub struct MemberCoherence {
    /// The pairwise member agreement.
    pub matrix: MemberMatrix,
    /// The verdict.
    pub decision: MemberDecision,
}

/// A `NormalRefineParams` shim carrying just the gating knobs
/// [`build_level_context`] and [`normalized_stack`] read, so the matrix drives the
/// shared support / render machinery without re-deriving it. The `min_views`
/// floor is 2 (its minimum): every member that passes the per-member validity
/// gate is kept, and a track that cannot reach two members yields no matrix.
fn normal_refine_shim(params: &MemberCoherenceParams) -> NormalRefineParams {
    NormalRefineParams {
        window: params.window,
        sampler: params.sampler,
        min_valid_fraction: params.min_valid_fraction,
        min_views: 2,
        ..NormalRefineParams::default()
    }
}

/// The pairwise windowed-ZNCC matrix of one point's members.
///
/// `views` is one [`ProjectedImage`] per reconstruction image (indexed by image
/// index); `members` lists the image indices observing the point. Members are
/// deduplicated first-seen-wins (a rig or a retriangulated track can observe the
/// same image twice, which would otherwise enter the matrix twice and let one
/// image vote for its own block).
///
/// Every member is rendered at the patch's own normal over the **common frozen
/// support** — the intersection of the members' validity masks, gated per member
/// on `min_valid_fraction` — and z-normalized per colour channel, exactly as view
/// selection builds its reference. A pair's score is the mean over surviving
/// channels of the dot product of the two members' z-normalized columns, i.e. the
/// windowed per-channel ZNCC. Members the validity gate drops, and members with no
/// texture at all, are left unscored (`NaN` row and column); when no support
/// survives, or it is smaller than `min_support_pixels`, the matrix carries only
/// its `1.0` diagonal.
///
/// Because the support is the intersection **over the members supplied**, entry
/// `(i, j)` depends on the whole member list: the same two members correlated
/// inside a different track (or after a member is dropped) can score differently.
/// Matrices built from different member subsets are not comparable.
///
/// # Anchoring
///
/// `member_keypoints`, when given, is parallel to the **input** `members` slice
/// (one entry per listed member, deduplicated alongside it) and carries that
/// member's stored source-pixel keypoint. Each member's patch is then recentered
/// in-plane so it renders **anchored at that keypoint** — the appearance the
/// matcher actually matched — instead of at the point's reprojection. The
/// per-member validity mask is built through the same recentered render
/// (`build_level_context` takes the same anchors), so the frozen common support
/// is the intersection of where the members are *sampled*, not of where the
/// geometry predicts them.
///
/// This matters because the reprojection residual is a **geometric** quantity: a
/// member carrying a sub-pixel-to-pixel residual is sampled that far off its own
/// content, and the resulting misalignment deflates every pairwise ZNCC it takes
/// part in — punishing it inside a measure that is supposed to read content
/// agreement alone. A residual that is a large fraction of the patch half-width
/// can cost several tenths of ZNCC on a member whose content is perfectly
/// correct.
///
/// Passing `None` (for the slice, or for an individual member inside it) falls
/// back to projection anchoring for that member — the behaviour a caller with no
/// keypoints (a hand-built member list, a `CameraViews` scene) necessarily gets.
/// Because anchoring changes what is sampled, `bar` is calibrated **per
/// anchoring**: keypoint-anchored scores run higher for exactly the members whose
/// residual was deflating them, so a caller switching anchoring should re-check
/// its threshold rather than assume it transfers.
///
/// # Panics
///
/// Panics if `member_keypoints` is given and is not parallel to `members`.
pub fn member_zncc_matrix(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    member_keypoints: Option<&[Option<[f64; 2]>]>,
    params: &MemberCoherenceParams,
) -> MemberMatrix {
    let resolution = params.resolution.max(2);

    if let Some(kps) = member_keypoints {
        assert_eq!(
            kps.len(),
            members.len(),
            "member_keypoints must be parallel to members"
        );
    }

    // Dedup first-seen-wins, carrying each survivor's keypoint with it.
    let mut seen = std::collections::HashSet::new();
    let keep: Vec<usize> = (0..members.len())
        .filter(|&i| seen.insert(members[i]))
        .collect();
    let members: Vec<u32> = keep.iter().map(|&i| members[i]).collect();
    let member_kps: Option<Vec<Option<[f64; 2]>>> =
        member_keypoints.map(|kps| keep.iter().map(|&i| kps[i]).collect());
    let k = members.len();

    let mut zncc = vec![f64::NAN; k * k];
    for i in 0..k {
        zncc[i * k + i] = 1.0;
    }
    let coarse_factors = coarse_factors_for(resolution);
    let mut zncc_coarse: Vec<Vec<f64>> = coarse_factors
        .iter()
        .map(|_| {
            let mut t = vec![f64::NAN; k * k];
            for i in 0..k {
                t[i * k + i] = 1.0;
            }
            t
        })
        .collect();
    let n_support = if k >= 2 {
        fill_member_zncc(
            patch,
            views,
            &members,
            member_kps.as_deref(),
            params,
            resolution,
            &mut zncc,
            &coarse_factors,
            &mut zncc_coarse,
        )
    } else {
        0
    };
    // Derived from the filled table, so "scored" means the same thing here as it
    // does in the decision rule.
    let scored = scored_mask(&zncc, k);
    MemberMatrix {
        members,
        zncc,
        zncc_coarse,
        coarse_factors,
        scored,
        n_support,
    }
}

/// The [`COARSE_FACTORS`] that divide `resolution` and leave a grid of at least
/// [`MIN_COARSE_RESOLUTION`], in order of increasing coarseness.
pub fn coarse_factors_for(resolution: u32) -> Vec<u32> {
    COARSE_FACTORS
        .iter()
        .copied()
        .filter(|&f| resolution.is_multiple_of(f) && resolution / f >= MIN_COARSE_RESOLUTION)
        .collect()
}

/// Render the members over one frozen common support and fill the off-diagonal
/// pairwise ZNCC, at full scale and at each coarse scale in `coarse_factors`.
/// Returns the size of that common support (`0` when none could be built). Leaves
/// every table untouched for members (or whole tracks) the support / validity /
/// texture gates drop.
///
/// The coarse tables are built from the **same** rendered stack, box-averaged: no
/// second render happens, and a member unscored at full scale is unscored at every
/// scale.
#[allow(clippy::too_many_arguments)]
fn fill_member_zncc(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    member_keypoints: Option<&[Option<[f64; 2]>]>,
    params: &MemberCoherenceParams,
    resolution: u32,
    zncc: &mut [f64],
    coarse_factors: &[u32],
    zncc_coarse: &mut [Vec<f64>],
) -> u32 {
    let k = members.len();
    let w_full = window_weights(params.window, resolution);
    let member_proj: Vec<ProjectedImage<'_>> = members.iter().map(|&i| views[i as usize]).collect();
    let shim = normal_refine_shim(params);

    // Frozen common support at the patch's own normal — `build_reference`'s first
    // step, unchanged except for the anchoring: with keypoints the mask is built
    // through the same recentered render the stack below samples, so the frozen
    // support intersects where the members are actually read.
    let Some(ctx) = build_level_context(
        patch,
        &patch.normal(),
        &member_proj,
        resolution,
        &w_full,
        &shim,
        member_keypoints,
    ) else {
        return 0;
    };
    let n = ctx.pixels.len();
    let n_support = n as u32;
    // Too little common support to correlate anything over: report the count and
    // leave the whole track unscored.
    if n_support < params.min_support_pixels {
        return n_support;
    }
    let Some((raw, channels)) = normalized_stack(
        patch,
        &ctx,
        &member_proj,
        resolution,
        params.sampler,
        member_keypoints,
    ) else {
        return n_support;
    };
    let total_weight: f64 = ctx.weights.iter().sum();
    if total_weight <= 0.0 {
        return n_support;
    }

    // Drop members with no texture at all before z-normalizing. The shared
    // `znormalize_into_kept` gate is per *channel* across *all* members — a
    // channel flat in any member is dropped for every member — which is right for
    // a consensus over one surface, but here one blown-out or sky member would
    // flatten every channel and silently leave the whole track unscored. Treat it
    // as this module's own coverage failure instead: exclude it from the stack
    // (so it ends up unscored, like a member the validity gate drops) and let the
    // rest score. The shared helper keeps its behaviour for its other callers.
    let alive: Vec<usize> = (0..ctx.kept.len())
        .filter(|&v| member_has_texture(&raw, v, channels, n, &ctx.weights, total_weight))
        .collect();
    if alive.len() < 2 {
        return n_support;
    }
    let compacted: Option<Vec<f32>> = (alive.len() < ctx.kept.len()).then(|| {
        let mut out = Vec::with_capacity(alive.len() * channels * n);
        for &v in &alive {
            out.extend_from_slice(&raw[v * channels * n..(v + 1) * channels * n]);
        }
        out
    });
    let stack: &[f32] = compacted.as_deref().unwrap_or(&raw);

    // Full scale, over the frozen support exactly as it stands.
    let rows: Vec<usize> = alive.iter().map(|&v| ctx.kept[v]).collect();
    if !fill_scale(
        stack,
        alive.len(),
        channels,
        n,
        &ctx.weights,
        &rows,
        k,
        zncc,
    ) {
        return n_support;
    }

    // Coarse scales, from the same stack: box-average the support's pixels into
    // the coarse cells, recompute the window on the coarse grid, and correlate by
    // the identical estimator. A scale whose stack has no surviving channel simply
    // leaves its table unfilled; the full-scale verdict does not depend on it.
    for (level, &factor) in coarse_factors.iter().enumerate() {
        let Some((coarse_stack, coarse_weights, cn)) = box_downsample(
            stack,
            alive.len(),
            channels,
            &ctx.pixels,
            resolution,
            factor,
            params.window,
        ) else {
            continue;
        };
        fill_scale(
            &coarse_stack,
            alive.len(),
            channels,
            cn,
            &coarse_weights,
            &rows,
            k,
            &mut zncc_coarse[level],
        );
    }
    n_support
}

/// Z-normalize one scale's stack and write its pairwise ZNCC into `table`.
/// `rows[a]` is the member index of stack row `a`. Returns `false` when the
/// shared flat-channel gate leaves nothing to correlate, which leaves `table`
/// untouched.
#[allow(clippy::too_many_arguments)]
fn fill_scale(
    stack: &[f32],
    n_members: usize,
    channels: usize,
    n: usize,
    weights: &[f64],
    rows: &[usize],
    k: usize,
    table: &mut [f64],
) -> bool {
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return false;
    }
    let sqrt_weights: Vec<f32> = weights.iter().map(|&w| w.sqrt() as f32).collect();
    let mut xs = Vec::new();
    let Some((kept_channels, _)) = znormalize_into_kept(
        stack,
        n_members,
        channels,
        n,
        weights,
        total_weight,
        &sqrt_weights,
        &mut xs,
    ) else {
        return false;
    };
    // Each kept member's z-normalized column is unit-norm per channel, so a plain
    // dot is the windowed ZNCC; average over the channels that survived the shared
    // flat-channel gate, matching the reference's own channel convention.
    for a in 0..n_members {
        let ia = rows[a];
        for b in (a + 1)..n_members {
            let ib = rows[b];
            let mut s = 0.0;
            for c in 0..kept_channels {
                let ca = &xs[(a * kept_channels + c) * n..][..n];
                let cb = &xs[(b * kept_channels + c) * n..][..n];
                s += ca
                    .iter()
                    .zip(cb)
                    .map(|(&x, &y)| (x as f64) * (y as f64))
                    .sum::<f64>();
            }
            let z = s / kept_channels as f64;
            table[ia * k + ib] = z;
            table[ib * k + ia] = z;
        }
    }
    true
}

/// Box-average a `[(member*channels + channel)*n + pixel]` stack over the frozen
/// support onto the `factor`-times coarser grid.
///
/// `pixels` are the support's linear `row * resolution + col` indices, parallel to
/// the stack's pixel axis. A coarse cell exists when at least one support pixel
/// falls in it **and** the window weight recomputed on the coarse grid is
/// positive; its value is the plain mean of the support pixels it contains, per
/// member and per channel. Because the support is common to every member, the
/// surviving cells are the same for all of them — the coarse stack is as
/// rectangular as the fine one.
///
/// Returns the coarse stack, its per-cell window weights and the cell count;
/// `None` when no cell survives.
fn box_downsample(
    stack: &[f32],
    n_members: usize,
    channels: usize,
    pixels: &[usize],
    resolution: u32,
    factor: u32,
    window: PatchWindow,
) -> Option<(Vec<f32>, Vec<f64>, usize)> {
    let r = resolution as usize;
    let f = factor as usize;
    let cr = r / f;
    let n = pixels.len();
    let w_coarse = window_weights(window, (r / f) as u32);

    // Which coarse cell each support pixel lands in, and how many land in each.
    let mut counts = vec![0u32; cr * cr];
    let cell_of: Vec<usize> = pixels
        .iter()
        .map(|&p| {
            let cell = (p / r / f) * cr + (p % r) / f;
            counts[cell] += 1;
            cell
        })
        .collect();
    let cells: Vec<usize> = (0..cr * cr)
        .filter(|&c| counts[c] > 0 && w_coarse[c] > 0.0)
        .collect();
    if cells.is_empty() {
        return None;
    }
    let mut slot = vec![usize::MAX; cr * cr];
    for (s, &c) in cells.iter().enumerate() {
        slot[c] = s;
    }
    let cn = cells.len();

    let mut out = vec![0.0f32; n_members * channels * cn];
    for m in 0..n_members {
        for c in 0..channels {
            let src = &stack[(m * channels + c) * n..][..n];
            let dst = &mut out[(m * channels + c) * cn..][..cn];
            for (p, &v) in src.iter().enumerate() {
                let s = slot[cell_of[p]];
                if s != usize::MAX {
                    dst[s] += v;
                }
            }
            for (s, &cell) in cells.iter().enumerate() {
                dst[s] /= counts[cell] as f32;
            }
        }
    }
    let weights: Vec<f64> = cells.iter().map(|&c| w_coarse[c]).collect();
    Some((out, weights, cn))
}

/// Whether member `v` of a raw `[(view*channels + channel)*n + pixel]` stack has
/// any channel with windowed texture, by the same `FLAT_NORM_SQ_EPS` criterion
/// [`znormalize_into_kept`] drops flat channels on.
fn member_has_texture(
    raw: &[f32],
    v: usize,
    channels: usize,
    n: usize,
    weights: &[f64],
    total_weight: f64,
) -> bool {
    (0..channels).any(|c| {
        let col = &raw[(v * channels + c) * n..][..n];
        let (s1, s2) = weighted_moments_pub(col, weights);
        s2 - s1 * (s1 / total_weight) >= FLAT_NORM_SQ_EPS
    })
}

/// The winning max-support block of the agreement graph, as a member mask.
///
/// `zncc` is the `k×k` table of the members in play — [`decide_member_coherence`]
/// passes the *scored* sub-matrix, so an unscored member is never a hypothesis and
/// never a tie-break candidate.
///
/// Every member is a hypothesis; its support is the set of members whose pairwise
/// ZNCC to it reaches `bar` (itself always included, so a member with no partner
/// supports a block of one). The largest support wins.
///
/// Ties are broken **deterministically**: first on the block's own mean coherence
/// (the mean of its finite intra-block links, `-1` when it has none), then on the
/// lowest member index. Nothing in the rule consults iteration or thread order.
fn max_support_block(zncc: &[f64], k: usize, bar: f64) -> Vec<bool> {
    // Agreement graph, self-loops forced on.
    let mut adj = vec![false; k * k];
    for i in 0..k {
        for j in 0..k {
            let z = zncc[i * k + j];
            adj[i * k + j] = i == j || (z.is_finite() && z >= bar);
        }
    }
    let support = |i: usize| (0..k).filter(|&j| adj[i * k + j]).count();
    let best_support = (0..k).map(support).max().unwrap_or(0);
    let ties: Vec<usize> = (0..k).filter(|&i| support(i) == best_support).collect();

    let best = if ties.len() > 1 {
        // Mean coherence of the block each tied hypothesis induces.
        let mean_intra = |t: usize| {
            let block: Vec<usize> = (0..k).filter(|&j| adj[t * k + j]).collect();
            let mut sum = 0.0;
            let mut count = 0usize;
            for (bi, &a) in block.iter().enumerate() {
                for &b in block.iter().skip(bi + 1) {
                    for (x, y) in [(a, b), (b, a)] {
                        let z = zncc[x * k + y];
                        if z.is_finite() {
                            sum += z;
                            count += 1;
                        }
                    }
                }
            }
            if count == 0 {
                -1.0
            } else {
                sum / count as f64
            }
        };
        let mut best = ties[0];
        let mut best_score = mean_intra(ties[0]);
        for &t in &ties[1..] {
            let s = mean_intra(t);
            if s > best_score {
                best_score = s;
                best = t;
            }
        }
        best
    } else {
        ties.first().copied().unwrap_or(0)
    };

    (0..k).map(|j| adj[best * k + j]).collect()
}

/// The `q`-quantile of an already-sorted non-empty slice, by linear
/// interpolation between the bracketing order statistics — a pure function of
/// the multiset, with no dependence on how it arrived.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if lo + 1 >= n {
        sorted[n - 1]
    } else {
        sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    }
}

/// The **centre and scatter of one block's own agreement**: the statistics the
/// self-normalized admission bar is measured in.
///
/// `zncc` is a `k×k` table and `block` a mask over it (in practice the *scored*
/// sub-matrix and its pass-1 max-support block). The sample is every finite
/// pairwise link **inside** the block, each counted once. Returns `None` — the
/// relative term stays inactive — when that sample holds fewer than
/// [`SELF_BAR_MIN_PAIRS`] links.
///
/// - **Centre** `c` is the median of those links.
/// - **Scatter** `σ` is the **upper** semi-interquartile distance, made
///   normal-consistent the same way a MAD is: `1.4826 · (Q₇₅ − median)`, floored
///   at [`SELF_BAR_MIN_SCATTER`].
///
/// The one-sidedness is the point. The block admitted at the absolute bar is the
/// very thing under suspicion: on a track with an occluding member the block
/// still contains it, and its links sit in the **lower** tail. A two-sided MAD
/// reads that tail as spread and inflates σ — letting the contamination loosen
/// the bar that is supposed to exclude it. The half above the median is the part
/// of the sample the contamination cannot reach (it is a minority, or the block
/// would not be the core), so it measures the core's own tightness. For a
/// symmetric sample the two coincide, which is what the `1.4826` is for.
///
/// It is read as an order statistic (`Q₇₅ − Q₅₀`) rather than as the median of
/// the members above the centre, because those two differ exactly when the
/// sample has a **mass at the median** — a two-population matrix whose median
/// lands on the lower mode. Counting the ties would report that matrix as
/// tightly coherent; the quartile distance reports the spread that is really
/// there and the relative term collapses, which is the intended behaviour for a
/// track with no single core.
pub fn core_coherence(zncc: &[f64], k: usize, block: &[bool]) -> Option<(f64, f64)> {
    let mut v = Vec::new();
    for a in 0..k {
        if !block[a] {
            continue;
        }
        for b in (a + 1)..k {
            if !block[b] {
                continue;
            }
            let z = zncc[a * k + b];
            if z.is_finite() {
                v.push(z);
            }
        }
    }
    if v.len() < SELF_BAR_MIN_PAIRS {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let center = quantile_sorted(&v, 0.5);
    let scatter = (1.4826 * (quantile_sorted(&v, 0.75) - center)).max(SELF_BAR_MIN_SCATTER);
    Some((center, scatter))
}

/// The absolute thresholds tightened by one block's own coherence: the pair
/// `(effective_bar, effective_margin_gate)` plus the `(centre, scatter)` they
/// were derived from.
///
/// One tighten pass, never iterated to a fixed point — see
/// [`decide_member_coherence`].
fn self_normalized_thresholds(
    sub: &[f64],
    s: usize,
    block: &[bool],
    params: &MemberCoherenceParams,
) -> (f64, f64, Option<(f64, f64)>) {
    let k_self = params.self_bar_k;
    if k_self.is_nan() || k_self <= 0.0 {
        return (params.bar, params.margin_gate, None);
    }
    match core_coherence(sub, s, block) {
        None => (params.bar, params.margin_gate, None),
        Some((c, sigma)) => {
            let relative = (c - k_self * sigma).min(SELF_BAR_CEILING);
            (
                params.bar.max(relative),
                params.margin_gate.min(sigma),
                Some((c, sigma)),
            )
        }
    }
}

/// Read a verdict off a pairwise member matrix.
///
/// Takes the max-support block (`max_support_block`), then gates the cut on its
/// **separation margin** — the weakest link inside the block minus the strongest
/// link leaving it. A margin at or below `margin_gate` (or an undefined one) means
/// the block boundary runs through a continuum rather than between two surfaces,
/// and the track is kept whole. Past the gate, a block holding a strict majority
/// of the members splits the track; a block that does not is a track whose
/// evidence supports two incompatible surfaces with neither prevailing, and the
/// point is retired.
///
/// # The self-normalized admission bar
///
/// `bar` and `margin_gate` are absolute, and an absolute threshold can only be
/// calibrated against one kind of disagreement. A member imaging a *different*
/// surface scores 0.2–0.5 against the core and 0.65 catches it. A member imaging
/// an *occluder in front of the same repeating texture* shares the core's
/// dominant structure and scores 0.85–0.95 — against a core that agrees with
/// itself at 0.99. The block structure is real and entirely above the bar.
///
/// So the thresholds are re-derived **per track, from the track's own
/// coherence**, in two passes over the same matrix:
///
/// 1. Sweep the max-support block (`max_support_block`) at the absolute `bar`.
/// 2. Measure that block's own [`core_coherence`] — centre `c`, scatter `σ`.
/// 3. `effective_bar = max(bar, min(c − self_bar_k · σ, `[`SELF_BAR_CEILING`]`))`
///    and `effective_margin_gate = min(margin_gate, σ)`.
/// 4. Re-sweep the block at `effective_bar` (only when it actually rose), and run
///    the margin and majority tests below on *that* block, against
///    `effective_margin_gate`.
///
/// The margin floor moves with the bar because it is the same problem one level
/// down: a margin is a difference of two ZNCCs, and the noise on that difference
/// is the core's own pair-to-pair scatter. A tight core separates from an
/// occluder by 0.02–0.04 — a real gap in its own units, refused outright by an
/// absolute 0.05. Both terms therefore relax back to the absolute pair exactly
/// when σ is large, which is what a drift chain and a noisy track have in common.
///
/// The circularity is real and is **cut, not solved**: admission defines the
/// block whose statistics set the admission bar. Pass 1 is deliberately run at
/// the loose absolute bar so the block is the widest defensible one, the scatter
/// estimator is one-sided so the members under suspicion cannot inflate the
/// scale that is meant to exclude them, and the tightening runs **once**. It is
/// not iterated to a fixed point: each pass would shrink the block, tighten the
/// bar off the survivors, and shrink it again, converging on the tightest
/// sub-clique of every track regardless of whether anything is wrong with it.
///
/// `self_bar_k = 0` disables all of it, reproducing the absolute rule exactly.
/// Below [`SELF_BAR_MIN_PAIRS`] intra-block pairs — a block of three or fewer —
/// there is nothing to estimate a scatter from and the relative term stays
/// inactive for the same reason.
///
/// `self_bar_k` **trades occlusion recall against collateral**: every member that
/// trails a tight core for an innocent reason — motion blur, an exposure step, a
/// grazing view — is a member the tightened bar is also more willing to evict,
/// and nothing in the *full-scale* matrix distinguishes the two. Multi-scale
/// exoneration, below, is what does.
///
/// # Multi-scale exoneration
///
/// A member that trails a tight core does so for one of two reasons, and the
/// pairwise agreement at a single scale cannot tell them apart because the two
/// produce the same number. They stop looking alike as soon as the fine detail is
/// taken away:
///
/// - **Structural** disagreement — an occluder, a different surface — is present
///   at every scale. The member's low-frequency content is already the wrong
///   content, so blurring both sides changes nothing about how badly they agree.
/// - **Spectral** disagreement — a defocused or motion-blurred frame of the *same*
///   surface — lives entirely in the detail. Coarsen both sides and it evaporates:
///   the member's low frequencies are the core's low frequencies.
///
/// So for each member the **relative** term alone would evict, the agreement
/// deficit is measured twice:
///
/// `deficit(scale) = mean(core↔core at scale) − mean(member↔core at scale)`
///
/// where the core is the winning block minus the member itself, over the tables
/// [`MemberMatrix::zncc_coarse`] already carries. Their quotient — the **retained
/// deficit** — is compared to [`MemberCoherenceParams::exoneration_ratio`], and a
/// member at or below it is **exonerated**: it stays in `kept`, out of `block`,
/// and marked in [`MemberDecision::exonerated`].
///
/// The comparison scale is the **first** coarse table — one halving — and not the
/// coarsest, because the test is whether the disagreement *survives* removing
/// detail, and a grid coarse enough washes out structure too. Measured against
/// the two labelled populations, one halving separates them (occluders retain
/// 0.85–1.00 of their deficit, soft frames a median 0.85 with a long lower tail);
/// two halvings collapses the occluders into the same range as the soft frames
/// and the separation is gone. That is why the threshold sits high: it is
/// asking "did *anything* decay", not "did most of it".
///
/// **Only the relative term's evictions are exonerable, and this asymmetry is
/// deliberate.** A member the *absolute* bar rejects is not a soft frame of the
/// track's surface at all — it correlates 0.2–0.5, the cross-surface chimera the
/// absolute rule was calibrated on — and how its disagreement is distributed
/// across scales says nothing about whether it belongs. Blur is not a defence
/// against imaging a different thing. Exoneration therefore never loosens the
/// validated absolute rule; it only refunds what tightening the bar per track
/// took.
///
/// Exoneration runs **after** the margin gate and before the majority test, on
/// the block the sweep produced. It re-admits individual members from the rejected
/// side rather than re-running the sweep: `margin`, `min_intra`, `max_cross`,
/// `support` and `block` all keep describing the cut that was proposed, and
/// exoneration is recorded as what it is — a spared member, not a different
/// block. When every rejected member is spared the verdict falls back to
/// `KeepAll`; when enough are spared to restore a majority, a `Retire` becomes a
/// `Split`.
///
/// `exoneration_ratio = 0` disables it. It is also inert whenever the relative
/// term is (there is nothing it may spare), and whenever the matrix carries no
/// coarse scale.
///
/// **The whole rule runs over the [scored](scored_mask) members only** — the block
/// sweep, both margin sides, and the majority denominator. An unscored member
/// carries no pairwise evidence at all, so it can neither be evicted by a cut it
/// took no part in nor dilute a majority among members that did: it passes through
/// `kept` (a `Retire` still ships nothing, the point itself is refused) and stays
/// out of `block`. With every member scored this is the plain rule, unchanged.
///
/// Fewer than two scored members means no evidence: `KeepAll`, empty block,
/// `support = 0`, undefined margin.
pub fn decide_member_coherence(
    matrix: &MemberMatrix,
    params: &MemberCoherenceParams,
) -> MemberDecision {
    let k = matrix.len();
    if k == 0 {
        return MemberDecision::default();
    }
    let zncc = &matrix.zncc;
    let scored = scored_mask(zncc, k);
    let idx: Vec<usize> = (0..k).filter(|&i| scored[i]).collect();
    let s = idx.len();
    if s < 2 {
        return MemberDecision {
            verdict: MemberVerdict::KeepAll,
            kept: vec![true; k],
            block: vec![false; k],
            support: 0,
            margin: f64::NAN,
            min_intra: f64::NAN,
            max_cross: f64::NAN,
            effective_bar: f64::NAN,
            effective_margin_gate: f64::NAN,
            core_center: f64::NAN,
            core_scatter: f64::NAN,
            relative_flagged: vec![false; k],
            exonerated: vec![false; k],
            retained_deficit: vec![f64::NAN; k],
            sharpness_deficit: vec![f64::NAN; k],
        };
    }

    // The scored sub-matrix, in member order. Every quantity below is computed on
    // it, so unscored members are structurally outside the rule rather than
    // half-counted by it. The coarse tables are sliced to the same members, so a
    // scale index means the same thing at every scale.
    let mut sub = vec![f64::NAN; s * s];
    for (a, &ia) in idx.iter().enumerate() {
        for (b, &ib) in idx.iter().enumerate() {
            sub[a * s + b] = zncc[ia * k + ib];
        }
    }
    let slice_scale = |table: &Vec<f64>| {
        let mut out = vec![f64::NAN; s * s];
        for (a, &ia) in idx.iter().enumerate() {
            for (b, &ib) in idx.iter().enumerate() {
                out[a * s + b] = table[ia * k + ib];
            }
        }
        out
    };
    // Two coarse scales, two different questions — see the module docs on
    // `zncc_coarse`. Exoneration asks whether the disagreement SURVIVES the loss
    // of one octave, so it reads the finest coarse table; sharpness measures how
    // much of the deficit is detail, so it reads the coarsest for the widest span.
    let exon_scale: Option<Vec<f64>> = matrix.zncc_coarse.first().map(&slice_scale);
    let sharp_scale: Option<Vec<f64>> = matrix.zncc_coarse.last().map(&slice_scale);
    // Pass 1 at the absolute bar, then the one tighten pass off that block's own
    // coherence. `effective_bar == params.bar` short-circuits the re-sweep, which
    // is what makes `self_bar_k = 0` bit-for-bit the absolute rule.
    let pass1 = max_support_block(&sub, s, params.bar);
    let (effective_bar, effective_margin_gate, core) =
        self_normalized_thresholds(&sub, s, &pass1, params);
    let relative_engaged = effective_bar > params.bar;
    let sub_block = if relative_engaged {
        max_support_block(&sub, s, effective_bar)
    } else {
        pass1.clone()
    };
    let (core_center, core_scatter) = core.unwrap_or((f64::NAN, f64::NAN));
    let support = sub_block.iter().filter(|&&b| b).count();

    // Margin components. Undefined (NaN) for a block of one, a block spanning
    // every scored member, or a side with no finite link.
    let mut min_intra = f64::INFINITY;
    let mut max_cross = f64::NEG_INFINITY;
    for a in 0..s {
        if !sub_block[a] {
            continue;
        }
        for b in 0..s {
            if a == b {
                continue;
            }
            let z = sub[a * s + b];
            if !z.is_finite() {
                continue;
            }
            if sub_block[b] {
                min_intra = min_intra.min(z);
            } else {
                max_cross = max_cross.max(z);
            }
        }
    }
    let min_intra = if min_intra.is_finite() {
        min_intra
    } else {
        f64::NAN
    };
    let max_cross = if max_cross.is_finite() {
        max_cross
    } else {
        f64::NAN
    };
    let whole = support == s;
    let margin = if whole || support < 2 {
        f64::NAN
    } else {
        min_intra - max_cross
    };

    // Scatter the block back over the full member list; unscored members are not
    // in it.
    let mut block = vec![false; k];
    for (a, &ia) in idx.iter().enumerate() {
        block[ia] = sub_block[a];
    }

    // Per-member sharpness, for every scored member and independently of any
    // verdict: the part of the member's deficit that only exists at fine scale.
    // Reported so a consumer can read the observations the point ships, which is
    // why it is not confined to the members under suspicion.
    let mut sharpness_deficit = vec![f64::NAN; k];
    if let Some(coarse) = sharp_scale.as_ref() {
        for (a, &ia) in idx.iter().enumerate() {
            let df = core_deficit(&sub, s, &sub_block, a);
            let dc = core_deficit(coarse, s, &sub_block, a);
            if df.is_finite() && dc.is_finite() {
                sharpness_deficit[ia] = df - dc;
            }
        }
    }

    let base = |verdict, kept: Vec<bool>, block: Vec<bool>, relative_flagged, exonerated, rd| {
        MemberDecision {
            verdict,
            kept,
            block,
            support: support as u32,
            margin,
            min_intra,
            max_cross,
            effective_bar,
            effective_margin_gate,
            core_center,
            core_scatter,
            relative_flagged,
            exonerated,
            retained_deficit: rd,
            sharpness_deficit: sharpness_deficit.clone(),
        }
    };
    // Nothing was being evicted, so nothing was flagged and nothing was spared.
    let keep_all = || {
        base(
            MemberVerdict::KeepAll,
            vec![true; k],
            block.clone(),
            vec![false; k],
            vec![false; k],
            vec![f64::NAN; k],
        )
    };

    if whole {
        return keep_all();
    }
    // Refuse to cut a continuum: no gap between the block and its outside, in the
    // track's own units.
    if margin.is_nan() || margin <= effective_margin_gate {
        return keep_all();
    }

    // A cut is on the table. Which of its evictions does the RELATIVE term own?
    // Only those: a member the absolute bar already rejected is not exonerable,
    // whatever its deficit does across scales.
    let mut relative_flagged = vec![false; k];
    let mut exonerated = vec![false; k];
    let mut retained_deficit = vec![f64::NAN; k];
    let mut spared = 0usize;
    if relative_engaged {
        for (a, &ia) in idx.iter().enumerate() {
            if sub_block[a] || !pass1[a] {
                continue;
            }
            relative_flagged[ia] = true;
            let Some(coarse) = exon_scale.as_ref() else {
                continue;
            };
            let df = core_deficit(&sub, s, &sub_block, a);
            if !(df.is_finite() && df > EXONERATION_MIN_DEFICIT) {
                continue;
            }
            let dc = core_deficit(coarse, s, &sub_block, a);
            if !dc.is_finite() {
                continue;
            }
            let ratio = dc / df;
            retained_deficit[ia] = ratio;
            if params.exoneration_ratio > 0.0 && ratio <= params.exoneration_ratio {
                exonerated[ia] = true;
                spared += 1;
            }
        }
    }

    // Everything the cut would have taken was spared: there is no cut left.
    let kept_scored = support + spared;
    if kept_scored == s {
        return base(
            MemberVerdict::KeepAll,
            vec![true; k],
            block,
            relative_flagged,
            exonerated,
            retained_deficit,
        );
    }
    // No strict majority among the members that carry evidence: the two sides are
    // equally supported, so neither can be called the point's surface. The spared
    // members count here — they ship, so they are part of the side that would.
    if 2 * kept_scored <= s {
        return base(
            MemberVerdict::Retire,
            vec![false; k],
            block,
            relative_flagged,
            exonerated,
            retained_deficit,
        );
    }
    // The cut evicts the scored members outside the block that were not spared.
    let kept = (0..k)
        .map(|i| block[i] || exonerated[i] || !scored[i])
        .collect();
    base(
        MemberVerdict::Split,
        kept,
        block,
        relative_flagged,
        exonerated,
        retained_deficit,
    )
}

/// One member's **agreement deficit** against a block, on one scale's `s×s`
/// table: how much worse the member agrees with the block than the block agrees
/// with itself.
///
/// `member` is excluded from the core on both sides, so the quantity means the
/// same thing for a member inside the block and one outside it. Both means are
/// over finite links only. `NaN` when the core holds fewer than two members, or
/// when either side has no finite link to average.
pub fn core_deficit(zncc: &[f64], s: usize, block: &[bool], member: usize) -> f64 {
    let core: Vec<usize> = (0..s).filter(|&i| block[i] && i != member).collect();
    if core.len() < 2 {
        return f64::NAN;
    }
    let (mut intra_sum, mut intra_n) = (0.0, 0usize);
    for (a, &ia) in core.iter().enumerate() {
        for &ib in core.iter().skip(a + 1) {
            let z = zncc[ia * s + ib];
            if z.is_finite() {
                intra_sum += z;
                intra_n += 1;
            }
        }
    }
    let (mut cross_sum, mut cross_n) = (0.0, 0usize);
    for &ic in &core {
        let z = zncc[member * s + ic];
        if z.is_finite() {
            cross_sum += z;
            cross_n += 1;
        }
    }
    if intra_n == 0 || cross_n == 0 {
        return f64::NAN;
    }
    intra_sum / intra_n as f64 - cross_sum / cross_n as f64
}

/// Validate one point's track: build its pairwise member matrix and read the
/// verdict off it. `member_keypoints` anchors the members' renders — see
/// [`member_zncc_matrix`].
pub fn validate_member_coherence(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    member_keypoints: Option<&[Option<[f64; 2]>]>,
    params: &MemberCoherenceParams,
) -> MemberCoherence {
    let matrix = member_zncc_matrix(patch, views, members, member_keypoints, params);
    let decision = decide_member_coherence(&matrix, params);
    MemberCoherence { matrix, decision }
}

/// Batch [`validate_member_coherence`] over a [`PatchCloud`], parallel across
/// patches (rayon). `member_views[i]` lists, for patch `i`, the track image
/// indices of its source point (see [`member_views_from_reconstruction`]);
/// `member_keypoints`, when given, is parallel to it in both dimensions and
/// carries each member's stored keypoint (see
/// [`member_keypoints_from_reconstruction`]). Results are returned in cloud
/// order — the kernel is per point, so nothing depends on thread scheduling.
///
/// # Panics
///
/// Panics if `member_views.len() != cloud.len()`, if `member_keypoints` is given
/// and not parallel to `member_views`, or if an index is out of range.
pub fn validate_patch_cloud_member_coherence(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    member_views: &[Vec<u32>],
    member_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    params: &MemberCoherenceParams,
    progress: Option<&std::sync::atomic::AtomicUsize>,
) -> Vec<MemberCoherence> {
    assert_eq!(
        member_views.len(),
        cloud.len(),
        "member_views must be parallel to the cloud"
    );
    if let Some(kps) = member_keypoints {
        assert_eq!(
            kps.len(),
            member_views.len(),
            "member_keypoints must be parallel to member_views"
        );
    }
    cloud
        .patches
        .par_iter()
        .enumerate()
        .zip(member_views.par_iter())
        .map(|((i, patch), mv)| {
            let kps = member_keypoints.map(|k| k[i].as_slice());
            let out = validate_member_coherence(patch, views, mv, kps, params);
            if let Some(c) = progress {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            out
        })
        .collect()
}

/// For each patch of `cloud` (linked to `recon` via `point_indexes`), the track
/// image indices observing its source 3D point — ready to use as the
/// `member_views` of [`validate_patch_cloud_member_coherence`].
///
/// # Panics
///
/// Panics if `cloud.point_indexes` is not parallel to its patches.
pub fn member_views_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<u32>> {
    super::normal_refine::view_indices_from_reconstruction(recon, cloud)
}

/// The stored keypoint of each member returned by
/// [`member_views_from_reconstruction`], in the same order — the
/// `member_keypoints` of [`validate_patch_cloud_member_coherence`].
///
/// `None` for every member of a `sift_files` reconstruction (it carries feature
/// indexes rather than inline keypoints), which anchors those members at their
/// projections.
///
/// # Panics
///
/// Panics if `cloud.point_indexes` is not parallel to its patches.
pub fn member_keypoints_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<Option<[f64; 2]>>> {
    assert_eq!(
        cloud.point_indexes.len(),
        cloud.len(),
        "cloud must carry a point_index per patch"
    );
    let kxy = recon.keypoints_xy();
    cloud
        .point_indexes
        .iter()
        .map(|&p| {
            let p = p as usize;
            let range = recon.observation_offsets[p]..recon.observation_offsets[p + 1];
            match kxy {
                Some(k) => range
                    .map(|r| Some([k[[r, 0]] as f64, k[[r, 1]] as f64]))
                    .collect(),
                None => range.map(|_| None).collect(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
