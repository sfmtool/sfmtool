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
//! [`build_level_context`] for the frozen common support, [`normalized_stack`] for
//! the renders and [`znormalize_into_kept`] for the per-channel z-normalization —
//! so a member's pairwise agreement lives in the same photometric space as the
//! member-vs-consensus score selection admits views on.

use rayon::prelude::*;

use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::normal_refine::{
    build_level_context, normalized_stack, window_weights, znormalize_into_kept,
    NormalRefineParams, PatchWindow, ProjectedImage, Sampler,
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
}

impl Default for MemberCoherenceParams {
    fn default() -> Self {
        Self {
            bar: 0.65,
            margin_gate: 0.05,
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: Sampler::BilinearMip,
            min_valid_fraction: 0.6,
        }
    }
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
    /// Per member: whether it was rendered and correlated at all. An unscored
    /// member has a `NaN` row and column but still holds its `1.0` diagonal.
    pub scored: Vec<bool>,
}

impl MemberMatrix {
    /// Build a matrix from an already-computed row-major `k×k` ZNCC table (the
    /// entry point for callers holding their own pairwise scores, and for tests).
    /// The diagonal is forced to `1.0` and `scored` is derived as "this member
    /// has at least one finite off-diagonal entry".
    ///
    /// # Panics
    ///
    /// Panics if `zncc.len() != members.len() * members.len()`.
    pub fn from_zncc(members: Vec<u32>, mut zncc: Vec<f64>) -> Self {
        let k = members.len();
        assert_eq!(zncc.len(), k * k, "zncc must be a k*k row-major matrix");
        for i in 0..k {
            zncc[i * k + i] = 1.0;
        }
        let scored = (0..k)
            .map(|i| (0..k).any(|j| j != i && zncc[i * k + j].is_finite()))
            .collect();
        Self {
            members,
            zncc,
            scored,
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
#[derive(Debug, Clone, Default)]
pub struct MemberDecision {
    /// The verdict. `Default` is [`MemberVerdict::KeepAll`] over an empty track.
    pub verdict: MemberVerdict,
    /// Per member: whether the point keeps it. All `true` for
    /// [`MemberVerdict::KeepAll`], the winning block for
    /// [`MemberVerdict::Split`], all `false` for [`MemberVerdict::Retire`].
    pub kept: Vec<bool>,
    /// The winning max-support block. Equals [`kept`](Self::kept) except on
    /// [`MemberVerdict::Retire`], where the point ships nothing and the block is
    /// **informational** — on a balanced split the two sides are interchangeable
    /// and only the deterministic tie-break decides which one is reported.
    pub block: Vec<bool>,
    /// Size of the winning block (its support count, itself included).
    pub support: u32,
    /// Separation margin: [`min_intra`](Self::min_intra) −
    /// [`max_cross`](Self::max_cross). `NaN` when the block holds fewer than two
    /// members, spans the whole track, or has no finite link on one of the two
    /// sides.
    pub margin: f64,
    /// Weakest finite link inside the winning block (`NaN` when it has none).
    pub min_intra: f64,
    /// Strongest finite link from the winning block to a member outside it
    /// (`NaN` when there is none).
    pub max_cross: f64,
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
/// windowed per-channel ZNCC. Members the validity gate drops are left unscored
/// (`NaN` row and column); when no support survives at all the matrix carries only
/// its `1.0` diagonal.
pub fn member_zncc_matrix(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    params: &MemberCoherenceParams,
) -> MemberMatrix {
    let resolution = params.resolution.max(2);

    let mut seen = std::collections::HashSet::new();
    let members: Vec<u32> = members
        .iter()
        .copied()
        .filter(|i| seen.insert(*i))
        .collect();
    let k = members.len();

    let mut zncc = vec![f64::NAN; k * k];
    for i in 0..k {
        zncc[i * k + i] = 1.0;
    }
    let mut scored = vec![false; k];
    if k >= 2 {
        fill_member_zncc(
            patch,
            views,
            &members,
            params,
            resolution,
            &mut zncc,
            &mut scored,
        );
    }
    MemberMatrix {
        members,
        zncc,
        scored,
    }
}

/// Render the members over one frozen common support and fill the off-diagonal
/// pairwise ZNCC. Leaves `zncc` / `scored` untouched for members (or whole
/// tracks) the support / validity gates drop.
fn fill_member_zncc(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    params: &MemberCoherenceParams,
    resolution: u32,
    zncc: &mut [f64],
    scored: &mut [bool],
) {
    let k = members.len();
    let w_full = window_weights(params.window, resolution);
    let member_proj: Vec<ProjectedImage<'_>> = members.iter().map(|&i| views[i as usize]).collect();
    let shim = normal_refine_shim(params);

    // Frozen common support at the patch's own normal — `build_reference`'s first
    // step, unchanged.
    let Some(ctx) = build_level_context(
        patch,
        &patch.normal(),
        &member_proj,
        resolution,
        &w_full,
        &shim,
        None,
    ) else {
        return;
    };
    let Some((raw, channels)) =
        normalized_stack(patch, &ctx, &member_proj, resolution, params.sampler, None)
    else {
        return;
    };
    let n = ctx.pixels.len();
    let total_weight: f64 = ctx.weights.iter().sum();
    if total_weight <= 0.0 {
        return;
    }
    let sqrt_weights: Vec<f32> = ctx.weights.iter().map(|&w| w.sqrt() as f32).collect();
    let mut xs = Vec::new();
    let Some((kept_channels, _)) = znormalize_into_kept(
        &raw,
        ctx.kept.len(),
        channels,
        n,
        &ctx.weights,
        total_weight,
        &sqrt_weights,
        &mut xs,
    ) else {
        return;
    };

    // Each kept member's z-normalized column is unit-norm per channel, so a plain
    // dot is the windowed ZNCC; average over the channels that survived the shared
    // flat-channel gate, matching the reference's own channel convention.
    for (a, &ia) in ctx.kept.iter().enumerate() {
        scored[ia] = true;
        for (b, &ib) in ctx.kept.iter().enumerate().skip(a + 1) {
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
            zncc[ia * k + ib] = z;
            zncc[ib * k + ia] = z;
        }
    }
}

/// The winning max-support block of the agreement graph, as a member mask.
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

/// Read a verdict off a pairwise member matrix.
///
/// Takes the [max-support block](max_support_block), then gates the cut on its
/// **separation margin** — the weakest link inside the block minus the strongest
/// link leaving it. A margin at or below `margin_gate` (or an undefined one) means
/// the block boundary runs through a continuum rather than between two surfaces,
/// and the track is kept whole. Past the gate, a block holding a strict majority
/// of the members splits the track; a block that does not is a track whose
/// evidence supports two incompatible surfaces with neither prevailing, and the
/// point is retired.
pub fn decide_member_coherence(
    matrix: &MemberMatrix,
    params: &MemberCoherenceParams,
) -> MemberDecision {
    let k = matrix.len();
    if k == 0 {
        return MemberDecision::default();
    }
    let zncc = &matrix.zncc;
    let block = max_support_block(zncc, k, params.bar);
    let support = block.iter().filter(|&&b| b).count();

    // Margin components. Undefined (NaN) for a block of one, a block spanning the
    // whole track, or a side with no finite link.
    let mut min_intra = f64::INFINITY;
    let mut max_cross = f64::NEG_INFINITY;
    for i in 0..k {
        if !block[i] {
            continue;
        }
        for j in 0..k {
            if i == j {
                continue;
            }
            let z = zncc[i * k + j];
            if !z.is_finite() {
                continue;
            }
            if block[j] {
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
    let whole = support == k;
    let margin = if whole || support < 2 {
        f64::NAN
    } else {
        min_intra - max_cross
    };

    let keep_all = || MemberDecision {
        verdict: MemberVerdict::KeepAll,
        kept: vec![true; k],
        block: block.clone(),
        support: support as u32,
        margin,
        min_intra,
        max_cross,
    };

    if whole {
        return keep_all();
    }
    // Refuse to cut a continuum: no gap between the block and its outside.
    if margin.is_nan() || margin <= params.margin_gate {
        return keep_all();
    }
    // No strict majority: the two sides are equally supported, so neither can be
    // called the point's surface.
    if 2 * support <= k {
        return MemberDecision {
            verdict: MemberVerdict::Retire,
            kept: vec![false; k],
            block,
            support: support as u32,
            margin,
            min_intra,
            max_cross,
        };
    }
    MemberDecision {
        verdict: MemberVerdict::Split,
        kept: block.clone(),
        block,
        support: support as u32,
        margin,
        min_intra,
        max_cross,
    }
}

/// Validate one point's track: build its pairwise member matrix and read the
/// verdict off it.
pub fn validate_member_coherence(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    params: &MemberCoherenceParams,
) -> MemberCoherence {
    let matrix = member_zncc_matrix(patch, views, members, params);
    let decision = decide_member_coherence(&matrix, params);
    MemberCoherence { matrix, decision }
}

/// Batch [`validate_member_coherence`] over a [`PatchCloud`], parallel across
/// patches (rayon). `member_views[i]` lists, for patch `i`, the track image
/// indices of its source point (see [`member_views_from_reconstruction`]).
/// Results are returned in cloud order — the kernel is per point, so nothing
/// depends on thread scheduling.
///
/// # Panics
///
/// Panics if `member_views.len() != cloud.len()` or an index is out of range.
pub fn validate_patch_cloud_member_coherence(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    member_views: &[Vec<u32>],
    params: &MemberCoherenceParams,
    progress: Option<&std::sync::atomic::AtomicUsize>,
) -> Vec<MemberCoherence> {
    assert_eq!(
        member_views.len(),
        cloud.len(),
        "member_views must be parallel to the cloud"
    );
    cloud
        .patches
        .par_iter()
        .zip(member_views.par_iter())
        .map(|(patch, mv)| {
            let out = validate_member_coherence(patch, views, mv, params);
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

#[cfg(test)]
mod tests;
