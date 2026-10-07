// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Construction of a point's pairwise member-agreement matrix.
//!
//! [`member_zncc_matrix`] renders every member of one track over the frozen
//! common support and fills the `k×k` windowed-ZNCC table, plus the coarse-scale
//! copies [multi-scale
//! exoneration](super::decide_member_coherence#multi-scale-exoneration) reads.
//! Everything here is rendering and image plumbing; the verdict read off the
//! matrix lives in [`super::decide`].

use super::{
    scored_mask, MemberCoherenceParams, MemberMatrix, COARSE_FACTORS, MIN_COARSE_RESOLUTION,
};
use crate::patch::blur_matched::{blur_tile, pair_blur, BlurCovariance, BlurScratch};
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::{
    build_level_context, normalized_stack, weighted_moments_pub, window_weights,
    znormalize_into_kept, NormalRefineParams, PatchWindow, ProjectedImage, ViewSamplers,
    FLAT_NORM_SQ_EPS,
};
use crate::patch::self_similarity::{zncc_self_similarity_radius, PatchTile, SelfSimilarityParams};
use crate::progress::Progress;
use crate::progress_note;

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
    member_zncc_matrix_reporting(
        patch,
        views,
        members,
        member_keypoints,
        params,
        &Progress::none(),
    )
}

/// [`member_zncc_matrix`] reporting to `progress`: a detailed `progress` times
/// the members' renders under each sampler in its own detail phase.
///
/// # Panics
///
/// As [`member_zncc_matrix`].
pub fn member_zncc_matrix_reporting(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    member_keypoints: Option<&[Option<[f64; 2]>]>,
    params: &MemberCoherenceParams,
    progress: &Progress<'_>,
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
    let blur_matched = params.matching.is_blur_matched();
    let mut blur_matched_zncc = if blur_matched {
        let mut t = vec![f64::NAN; k * k];
        for i in 0..k {
            t[i * k + i] = 1.0;
        }
        t
    } else {
        Vec::new()
    };
    let mut pairs_blurred = 0u32;
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
            blur_matched.then_some((&mut blur_matched_zncc, &mut pairs_blurred)),
            progress,
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
        blur_matched_zncc,
        pairs_blurred,
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
    blur_matched: Option<(&mut Vec<f64>, &mut u32)>,
    progress: &Progress<'_>,
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
        ViewSamplers::Each(params.sampler),
        member_keypoints,
        progress,
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
    let Some(keep) = fill_scale(
        stack,
        alive.len(),
        channels,
        n,
        &ctx.weights,
        &rows,
        k,
        zncc,
    ) else {
        return n_support;
    };

    // Blur-matched, from the same renders, over the same support, with the
    // channels the plain table kept.
    if let Some((table, pairs_blurred)) = blur_matched {
        let members = MemberStack {
            stack,
            n_members: alive.len(),
            channels,
            pixels: &ctx.pixels,
            weights: &ctx.weights,
            resolution,
        };
        *pairs_blurred =
            fill_blur_matched(&members, &keep, &rows, k, zncc, params, table, progress);
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
        // A scale with no channel left keeps its table unfilled.
        let _ = fill_scale(
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
/// `rows[a]` is the member index of stack row `a`. Returns which channels the
/// shared flat-channel gate kept, or `None` when it leaves nothing to
/// correlate, which leaves `table` untouched.
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
) -> Option<Vec<bool>> {
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return None;
    }
    let sqrt_weights: Vec<f32> = weights.iter().map(|&w| w.sqrt() as f32).collect();
    let mut xs = Vec::new();
    let (kept_channels, keep) = znormalize_into_kept(
        stack,
        n_members,
        channels,
        n,
        weights,
        total_weight,
        &sqrt_weights,
        &mut xs,
    )?;
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
    Some(keep)
}

/// The renders of the members that scored, as [`fill_scale`] reads them: a
/// `[(member*channels + channel)*n + pixel]` stack over the common support
/// `pixels` (grid positions `row * resolution + col`), with its window weights.
struct MemberStack<'a> {
    stack: &'a [f32],
    n_members: usize,
    channels: usize,
    pixels: &'a [usize],
    weights: &'a [f64],
    resolution: u32,
}

impl MemberStack<'_> {
    /// Member `m`'s render on the `R×R` grid, planar, zero off the support.
    fn planes(&self, m: usize) -> Vec<f32> {
        let rr = (self.resolution as usize).pow(2);
        let n = self.pixels.len();
        let mut out = vec![0.0f32; self.channels * rr];
        for c in 0..self.channels {
            let col = &self.stack[(m * self.channels + c) * n..][..n];
            for (&p, &v) in self.pixels.iter().zip(col) {
                out[c * rr + p] = v;
            }
        }
        out
    }
}

/// Fill `table` with the blur-matched agreement of the members that scored
/// ([`MemberMatrix::blur_matched_zncc`]), copying `plain` for a pair left
/// plain, and return how many pairs were blurred.
///
/// Each member's self-similarity ellipse is read on its render over the
/// support, the support as its data, in the `blur-matched ellipses` detail
/// phase of `progress`. A pair is blurred where [`pair_blur`] asks, each
/// render by normalized convolution over the support, then gathered back,
/// z-normalized over the support with the same window and channels the plain
/// table kept, and correlated as [`fill_scale`] correlates, in the
/// `blur-matched pairs` detail phase.
#[allow(clippy::too_many_arguments)]
fn fill_blur_matched(
    members: &MemberStack<'_>,
    keep: &[bool],
    rows: &[usize],
    k: usize,
    plain: &[f64],
    params: &MemberCoherenceParams,
    table: &mut [f64],
    progress: &Progress<'_>,
) -> u32 {
    let r = members.resolution as usize;
    let rr = r * r;
    let n = members.pixels.len();
    let channels = members.channels;
    let Some(ratio) = params.matching.min_ratio() else {
        return 0;
    };
    let mut support = vec![false; rr];
    for &p in members.pixels {
        support[p] = true;
    }
    let planes: Vec<Vec<f32>> = (0..members.n_members).map(|m| members.planes(m)).collect();
    let ellipses: Vec<Option<[[f64; 2]; 2]>> = {
        let mut phase = progress.detail_phase("blur-matched ellipses");
        let colour = channels.min(3);
        let read: Vec<Option<[[f64; 2]; 2]>> = planes
            .iter()
            .map(|values| {
                let reading = zncc_self_similarity_radius(
                    &PatchTile {
                        values: &values[..colour * rr],
                        channels: colour,
                        width: r,
                        height: r,
                    },
                    Some(&support),
                    [0, 0, r, r],
                    &SelfSimilarityParams::default(),
                );
                let e = reading.ellipse;
                e.axes.iter().all(|a| a.is_finite()).then_some(e.matrix)
            })
            .collect();
        progress_note!(phase, "{} members", read.len());
        read
    };

    let mut phase = progress.detail_phase("blur-matched pairs");
    let kept: Vec<usize> = (0..channels).filter(|&c| keep[c]).collect();
    let total_weight: f64 = members.weights.iter().sum();
    let sqrt_weights: Vec<f32> = members.weights.iter().map(|&w| w.sqrt() as f32).collect();
    let mut scratch = BlurScratch::default();
    let mut blurred = vec![0.0f32; channels * rr];
    let mut raw = vec![0.0f32; 2 * kept.len() * n];
    let mut xs = Vec::new();
    let (mut pairs, mut pairs_blurred) = (0u32, 0u32);
    for a in 0..members.n_members {
        for b in (a + 1)..members.n_members {
            let (ia, ib) = (rows[a], rows[b]);
            pairs += 1;
            let blur = match (ellipses[a], ellipses[b]) {
                (Some(ea), Some(eb)) => pair_blur(&ea, &eb, ratio),
                _ => Default::default(),
            };
            if blur.is_none() {
                table[ia * k + ib] = plain[ia * k + ib];
                table[ib * k + ia] = plain[ib * k + ia];
                continue;
            }
            pairs_blurred += 1;
            for (slot, (m, cov)) in [(a, blur.a), (b, blur.b)].into_iter().enumerate() {
                let source: &[f32] = if cov == BlurCovariance::ZERO || cov.is_zero() {
                    &planes[m]
                } else {
                    blur_tile(
                        &planes[m],
                        channels,
                        r,
                        &support,
                        cov,
                        &mut blurred,
                        &mut scratch,
                    );
                    &blurred
                };
                for (kc, &c) in kept.iter().enumerate() {
                    let dst = &mut raw[(slot * kept.len() + kc) * n..][..n];
                    for (d, &p) in dst.iter_mut().zip(members.pixels) {
                        *d = source[c * rr + p];
                    }
                }
            }
            let z = match znormalize_into_kept(
                &raw,
                2,
                kept.len(),
                n,
                members.weights,
                total_weight,
                &sqrt_weights,
                &mut xs,
            ) {
                Some((kc, _)) => {
                    let s: f64 = (0..kc)
                        .map(|c| {
                            let x = &xs[c * n..][..n];
                            let y = &xs[(kc + c) * n..][..n];
                            x.iter()
                                .zip(y)
                                .map(|(&u, &v)| f64::from(u) * f64::from(v))
                                .sum::<f64>()
                        })
                        .sum();
                    s / kc as f64
                }
                None => f64::NAN,
            };
            table[ia * k + ib] = z;
            table[ib * k + ia] = z;
        }
    }
    progress_note!(phase, "{pairs_blurred} of {pairs} pairs blurred");
    pairs_blurred
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
