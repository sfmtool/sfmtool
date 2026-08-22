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
use crate::patch::normal_refine::{PatchWindow, ProjectedImage, Sampler, MIN_MASK_PIXELS};
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

mod decide;
mod matrix;

pub use decide::{core_coherence, core_deficit, decide_member_coherence};
pub use matrix::{coarse_factors_for, member_zncc_matrix};

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
