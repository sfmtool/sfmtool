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
    /// The default `8` is the floor [`build_level_context`] already enforces
    /// (`MIN_MASK_PIXELS`), so it changes nothing on its own; values below it are
    /// inert for the same reason. A caller vetting wide-baseline tracks — where
    /// the intersection can shrink to a sliver of an `R×R` grid and a correlation
    /// over a handful of pixels is noise — wants it higher.
    pub min_support_pixels: u32,
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
            min_support_pixels: MIN_MASK_PIXELS as u32,
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
    /// and `n_support` is `0` (nothing was rendered).
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
        let scored = scored_mask(&zncc, k);
        Self {
            members,
            zncc,
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
/// ([`build_level_context`] takes the same anchors), so the frozen common support
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
    let n_support = if k >= 2 {
        fill_member_zncc(
            patch,
            views,
            &members,
            member_kps.as_deref(),
            params,
            resolution,
            &mut zncc,
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
        scored,
        n_support,
    }
}

/// Render the members over one frozen common support and fill the off-diagonal
/// pairwise ZNCC. Returns the size of that common support (`0` when none could be
/// built). Leaves `zncc` untouched for members (or whole tracks) the support /
/// validity / texture gates drop.
fn fill_member_zncc(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    member_keypoints: Option<&[Option<[f64; 2]>]>,
    params: &MemberCoherenceParams,
    resolution: u32,
    zncc: &mut [f64],
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

    let sqrt_weights: Vec<f32> = ctx.weights.iter().map(|&w| w.sqrt() as f32).collect();
    let mut xs = Vec::new();
    let Some((kept_channels, _)) = znormalize_into_kept(
        stack,
        alive.len(),
        channels,
        n,
        &ctx.weights,
        total_weight,
        &sqrt_weights,
        &mut xs,
    ) else {
        return n_support;
    };

    // Each kept member's z-normalized column is unit-norm per channel, so a plain
    // dot is the windowed ZNCC; average over the channels that survived the shared
    // flat-channel gate, matching the reference's own channel convention.
    for (a, &va) in alive.iter().enumerate() {
        let ia = ctx.kept[va];
        for (b, &vb) in alive.iter().enumerate().skip(a + 1) {
            let ib = ctx.kept[vb];
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
    n_support
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
        };
    }

    // The scored sub-matrix, in member order. Every quantity below is computed on
    // it, so unscored members are structurally outside the rule rather than
    // half-counted by it.
    let mut sub = vec![f64::NAN; s * s];
    for (a, &ia) in idx.iter().enumerate() {
        for (b, &ib) in idx.iter().enumerate() {
            sub[a * s + b] = zncc[ia * k + ib];
        }
    }
    let sub_block = max_support_block(&sub, s, params.bar);
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
    // No strict majority among the members that carry evidence: the two sides are
    // equally supported, so neither can be called the point's surface.
    if 2 * support <= s {
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
    // The cut evicts the scored members outside the block, and only those.
    let kept = (0..k).map(|i| block[i] || !scored[i]).collect();
    MemberDecision {
        verdict: MemberVerdict::Split,
        kept,
        block,
        support: support as u32,
        margin,
        min_intra,
        max_cross,
    }
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
