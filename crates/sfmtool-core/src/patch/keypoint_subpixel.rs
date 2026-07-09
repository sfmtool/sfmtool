// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Photometric subpixel keypoint refinement (the high-accuracy reference).
//!
//! See `specs/core/keypoint-subpixel-refinement.md`. Given a keypoint that is
//! **already close** to correct (the caller's precondition), this refines it to
//! sub-pixel by a **local** continuous optimization of cross-view
//! photoconsistency: per view, a 2-DOF in-plane translation offset `δ` solved by
//! a few **forward-additive ECC (Enhanced Correlation Coefficient) Gauss–Newton**
//! steps against the robust (IRLS) cross-view consensus `T`. It does no grid
//! search; the only membership change is the projection gate (a view in which
//! `project_i(X_p)` fails has no projection-anchored offset to report, so it is
//! dropped). Each accepted step raises the ECC score against the current `T` and
//! stays in frame — the never-worse-than-seed guarantee — see
//! [`KeypointSubpixelParams::max_outer_sweeps`] for how this composes when `T` is
//! refreshed across sweeps.
//!
//! The number of **outer sweeps** is a tunable
//! ([`KeypointSubpixelParams::max_outer_sweeps`]): each sweep re-renders the
//! views at their current offsets, rebuilds the robust (IRLS) consensus from
//! those, and refines every view against it. With `max_outer_sweeps = 1` (the
//! default) this is the spec's cheapest **single-pass frozen** variant — build
//! `T` once at the seed, hold it fixed, move every view; with
//! `max_outer_sweeps > 1` it is the spec's **per-sweep refresh** variant. Whether refreshing earns
//! its keep at sub-pixel scale is a measurable question, not a settled one (the
//! prototype observed `T` sharpening as views aligned), so this is a knob, not a
//! constant.
//!
//! [`KeypointSubpixelParams::consensus_refresh`] additionally selects the
//! within-sweep granularity: [`ConsensusRefresh::PerSweep`] (default — `T` is
//! rebuilt only at sweep boundaries) or [`ConsensusRefresh::PerMove`] (the
//! spec's Gauss–Seidel **incremental consensus**: after each view moves, its
//! contribution to the running weighted sum `S = Σ_v w_v · ẑ_v` is
//! delta-updated and `T = normalize(S)` is re-derived for the next view). The
//! IRLS weights are refreshed at the lower per-sweep frequency (the spec's
//! two-frequency design). The per-move path uses the **shared** consensus
//! (not leave-one-out): measurement on real data (dino, mean view count 3–5)
//! found LOO regressed mean ECC (0.82 vs 0.87 shared at 5 sweeps); see
//! [`RunningConsensus::write_shared_template`] for the reasoning and the
//! commit message that landed this for the per-sweep / per-move comparison
//! numbers.
//!
//! Points at **infinity** (`w = 0`) are refined exactly like finite ones — the
//! warp + projection already handle `w = 0`, so the same objective, sampling, and
//! Jacobian apply. They are *not* skipped (the opposite of normal refinement).
//!
//! The render → z-normalize → robust-consensus machinery is shared with
//! [keypoint localization](super::keypoint_localize) (the typical producer of the
//! seed) and [normal refinement](super::normal_refine); this module reuses those
//! helpers rather than re-deriving the math, and adds only the continuous
//! ECC/Gauss–Newton inner solve.
//!
//! ## Render-once context tile
//!
//! Within one refinement the patch frame is fixed and only each view's 2-DOF
//! in-plane offset moves, so every render of a (point, view) pair is the same
//! patch→image map at a slightly different sub-pixel shift — and the solver
//! evaluates ~10 of them per pair (GN steps + line-search probes). Instead of
//! a full projective render per evaluation, the pair's map is prerendered
//! **once** into a [`RefineTile`] (patch-grid-aligned, centred at the view's
//! seed, sized to cover the `max_offset_px` drift, storing the sampler's
//! unquantized values plus the pre-composed patch-grid gradient planes
//! `∇_src I · J`); every evaluation is then a continuous prefiltered
//! cubic-B-spline read of that tile (exact at integer shifts). See the
//! [`RefineTile`] doc for the exactness/coverage contract and
//! the accepted double-interpolation loss, and
//! `specs/core/keypoint-subpixel-refinement.md` for the design discussion.
//!
//! ## ECC Gauss–Newton, derived
//!
//! Per view the ECC criterion is `S(δ) = (1/C) Σ_c ⟨ẑ_c(δ), T_c⟩`, the
//! channel-averaged windowed ZNCC of the view's z-normalized core `ẑ` against the
//! consensus `T` (each `T_c` zero-(weighted-)mean, unit-norm, with `√w` folded in,
//! exactly as the consensus is built for the discrete search). Maximizing `S` is
//! equivalent to least-squares minimizing `½ Σ_c ‖ẑ_c − T_c‖²` (since `‖ẑ_c‖ =
//! ‖T_c‖ = 1`), whose forward-additive Gauss–Newton step is `H δ = b` with
//! `H = Σ_c Σ_k (∂ẑ_c[k]/∂δ)(∂ẑ_c[k]/∂δ)ᵀ` and `b = Σ_c Σ_k (∂ẑ_c[k]/∂δ) T_c[k]`
//! — and `b = C·∇S` because `Σ_k (∂ẑ_c[k]/∂δ)·ẑ_c[k] = ½∂‖ẑ_c‖²/∂δ = 0`. So the
//! step rises along the score gradient with the natural GN Hessian. The
//! z-normalization derivative `∂ẑ` is taken analytically (see [`view_jacobian`]);
//! the raw image Jacobian `∂g/∂δ` is now also analytic, via the sampler's
//! value+gradient interface (`remap_bilinear_with_grad` / `remap_aniso_with_grad`,
//! returning `(I, ∂I/∂x, ∂I/∂y)` in source-pixel coords per support pixel and
//! channel) composed pixel-wise with the warp Jacobian
//! (`WarpMap::get_jacobian`): `∂I/∂δ = ∇_src I · J`. The previous finite-difference
//! path took five renders per GN step; the analytic path takes one.

use crate::camera::remap::ImageF32WithGrad;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::keypoint_localize::{project, seed_offset, shifted_center};
use crate::patch::normal_refine::{
    build_support, irls_view_weights, weighted_unit_template_into, ConsensusScratch,
    PatchViewStack, ProjectedImage, Support, AGREEMENT_SIGMA,
};
use rayon::prelude::*;

pub mod prof;

mod kernels;
mod params;

// Public API, re-exported at the historical `keypoint_subpixel::` paths.
pub use params::{ConsensusRefresh, KeypointRefinement, KeypointSubpixelParams};

// Rendering + scoring kernels consumed by the Gauss–Newton orchestration below.
use kernels::{
    core_value, core_value_with_jg, ecc_score, solve_2x2, try_render_refine_tile, view_jacobian,
    znorm_core, RefineTile,
};

// Render entry points + the coarse-grid gate re-exported into this module's
// namespace only for the sibling test module's `use super::*`; production reaches
// them through `kernels` (or the wrappers above), so these are test-gated to stay
// warning-clean in release. `WarpMap` / `PatchWindow` likewise moved to `kernels`
// / `params` but are still named directly by tests.
#[cfg(test)]
use crate::camera::remap::remap_bilinear;
#[cfg(test)]
use crate::camera::WarpMap;
#[cfg(test)]
use crate::patch::normal_refine::PatchWindow;
#[cfg(test)]
use kernels::{
    grid_to_source_scale, render_core, render_core_with_jg, render_refine_tile,
    TILE_MAX_GRID_TO_SOURCE,
};

/// One view's mutable refinement state.
struct ViewState {
    /// Image index into the caller's `views` slice.
    idx: u32,
    /// Seed offset `(au, av)` in patch-grid px (the keypoint at refine start).
    seed: [f64; 2],
    /// Current offset `(au, av)` in patch-grid px.
    off: [f64; 2],
    /// The view's projection of the point `project_i(X_p)`, source px.
    proj: [f64; 2],
    /// Final ECC score (NaN until scored).
    score: f64,
}

/// Refine the per-view keypoints of one oriented patch by forward-additive ECC
/// Gauss–Newton against a single frozen cross-view consensus.
///
/// `views` is one [`ProjectedImage`] per reconstruction image (indexed by image
/// index); `view_set` lists the views to refine. `starting_keypoints`, when given,
/// is one seed per `view_set` entry (source-image px); `None` seeds every view at
/// the point's own projection `project_i(X_p)`. Returns the views (input order,
/// deduplicated) with their refined keypoints. The only membership change is the
/// projection gate (a view in which `project_i(X_p)` fails — behind the camera or
/// out of frame — is dropped, as the offset has nothing to be measured from);
/// otherwise the set is preserved, and a guard-failed view keeps its seed. See
/// `specs/core/keypoint-subpixel-refinement.md`.
pub fn refine_patch_keypoints(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    starting_keypoints: Option<&[Option<[f64; 2]>]>,
    params: &KeypointSubpixelParams,
) -> KeypointRefinement {
    prof::TOTAL
        .time(|| refine_patch_keypoints_impl(patch, views, view_set, starting_keypoints, params))
}

/// Untimed body of [`refine_patch_keypoints`] (split so the enclosing
/// [`prof::TOTAL`] phase is a single wrap covering both batch entries — the
/// Rust [`refine_patch_cloud_keypoints`] and the PyO3 binding's inlined loop).
fn refine_patch_keypoints_impl(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    starting_keypoints: Option<&[Option<[f64; 2]>]>,
    params: &KeypointSubpixelParams,
) -> KeypointRefinement {
    let resolution = params.resolution.max(2);
    let wpp_u = 2.0 * patch.half_extent[0] / resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / resolution as f64;

    // Window support over the R×R core.
    let support = build_support(params.window, resolution);
    let n = support.pixels.len();

    // Build the deduplicated view states (input order). A view that can't project
    // the point in-frame can't be refined; it is dropped from the set entirely
    // (it has no seed to keep), matching the localizer's projection gate.
    let mut seen = std::collections::HashSet::new();
    let mut states: Vec<ViewState> = Vec::new();
    for (k, &i) in view_set.iter().enumerate() {
        if !seen.insert(i) {
            continue;
        }
        let view = &views[i as usize];
        let Some(proj) = project(view, &patch.center, patch.w) else {
            continue;
        };
        let off = match starting_keypoints.and_then(|seeds| seeds[k]) {
            Some(kp) => seed_offset(patch, view, kp, wpp_u, wpp_v).unwrap_or([0.0, 0.0]),
            None => [0.0, 0.0],
        };
        states.push(ViewState {
            idx: i,
            seed: off,
            off,
            proj: [proj.0, proj.1],
            score: f64::NAN,
        });
    }

    if states.len() < 2 {
        // No cross-view consensus to refine against: keep every seed.
        return finalize(patch, views, &states, wpp_u, wpp_v);
    }

    // Channel count is the first view's image channels; the warp renders at that
    // count, and all reconstruction images share it.
    let channels = views[states[0].idx as usize].pyramid.level(0).channels() as usize;
    let mut raw = vec![0f32; channels * n];
    let mut znorm = vec![0f32; channels * n];
    let mut sc = ConsensusScratch::default();
    let mut tmpl = Vec::new();
    let mut xs: Vec<f32> = Vec::new();
    let mut scratch = GnScratch::new(channels * n);

    // Render-once context tiles, one per (point, view) pair, centred at each
    // view's seed and sized to cover the whole `max_offset_px` drift (plus the
    // cubic read's tap margin). Every value / GN-gradient evaluation below
    // reads its view's tile; a coarse-grid view gets no tile (`None` — it
    // keeps the exact direct-render path, see `try_render_refine_tile`), and
    // an out-of-coverage offset (expected never, given the line-search bound)
    // falls back to a direct render.
    let pad = (params.max_offset_px.max(0.0).ceil() as u32).max(1) + 2;
    let tiles: Vec<Option<RefineTile>> = states
        .iter()
        .map(|st| {
            try_render_refine_tile(
                patch,
                &views[st.idx as usize],
                st.seed,
                wpp_u,
                wpp_v,
                resolution,
                pad,
                params.sampler,
                &mut scratch.img,
            )
        })
        .collect();

    // Outer sweeps: each sweep re-renders the views at their current offsets,
    // rebuilds the robust consensus from them, and refines every view against it.
    // With `max_outer_sweeps = 1` (default) only the seed-aligned cores ever build
    // `T` (the single-pass-frozen variant). The sweep loop early-exits when the
    // mean per-view move falls below `outer_convergence_px`; on sweeps after the
    // first, a view whose core has left the frame at its current offset doesn't
    // contribute to the rebuilt `T` (it stays out of `live`) but other views still
    // refine against the `T` the in-frame ones build.
    //
    // Per-sweep refresh (the default) holds `T` fixed for the duration of a
    // sweep — every view sees the same template. Per-move (the
    // `ConsensusRefresh::PerMove` variant) maintains a [`RunningConsensus`]:
    // after each view's GN solve, its (refined) z-normalized core delta-updates
    // the running sum `S` so the next view aligns to a freshly-incrementalized
    // `T`. IRLS weights stay fixed within a sweep either way — for PerMove
    // that's the spec's two-frequency design (fixed weights → delta-update is
    // exact); the per-sweep boundary rebuilds both.
    let max_sweeps = params.max_outer_sweeps.max(1);
    let mut live: Vec<usize> = Vec::new();
    let mut running = RunningConsensus::default();
    for _ in 0..max_sweeps {
        prof::count(&prof::N_SWEEPS, 1);
        // 1. Render every view's core at its current offset and z-normalize the
        //    live ones into the per-sweep template-build buffer `xs`.
        live.clear();
        xs.clear();
        for (si, st) in states.iter().enumerate() {
            if core_value(
                patch,
                &views[st.idx as usize],
                tiles[si].as_ref(),
                st.off[0],
                st.off[1],
                wpp_u,
                wpp_v,
                resolution,
                params.sampler,
                &support,
                channels,
                &mut raw,
            ) {
                znorm_core(&raw, &support, channels, &mut znorm);
                live.push(si);
                xs.extend_from_slice(&znorm);
            }
        }
        if live.len() < 2 {
            // Lost cross-view consensus mid-sweep: keep current offsets.
            break;
        }

        // 2. (Re)build the robust consensus from the current-offset cores: the
        //    IRLS view weights, then the weighted unit-norm template. On sweep 0
        //    this is the spec's frozen `T`; on later sweeps it is the per-sweep
        //    refresh. Per-move additionally rebuilds the running sum `S` so it
        //    can be delta-updated as views move within the sweep.
        prof::CONSENSUS.time(|| {
            irls_view_weights(
                &xs,
                live.len(),
                channels,
                n,
                params.robust_iters,
                None,
                &mut sc,
            );
            weighted_unit_template_into(&xs, &sc.w, live.len(), channels, n, &mut tmpl);
            if matches!(params.consensus_refresh, ConsensusRefresh::PerMove) {
                running.rebuild(&xs, &sc.w, live.len(), channels, n);
            }
        });

        // 3. Move every live view against the current consensus, tracking the
        //    sweep's mean per-view move for the outer convergence check. For
        //    PerSweep all views see the same `tmpl` (frozen for the sweep). For
        //    PerMove each view sees the **shared** running consensus
        //    `normalize(S)`, then its refined ẑ is folded back into `S` for the
        //    next view. (The spec's leave-one-out alternative was measured-and-
        //    rejected — see `RunningConsensus::write_shared_template`.)
        let mut sweep_move_sum = 0.0;
        for (slot, &si) in live.iter().enumerate() {
            let before = states[si].off;
            let view_tmpl: &[f32] = match params.consensus_refresh {
                ConsensusRefresh::PerSweep => &tmpl,
                ConsensusRefresh::PerMove => {
                    prof::CONSENSUS_UPDATE.time(|| running.write_shared_template(&mut tmpl));
                    &tmpl
                }
            };
            refine_one_view(
                patch,
                &views[states[si].idx as usize],
                tiles[si].as_ref(),
                &mut states[si],
                &support,
                view_tmpl,
                channels,
                resolution,
                wpp_u,
                wpp_v,
                params,
                &mut scratch,
            );
            let after = states[si].off;
            sweep_move_sum += (after[0] - before[0]).hypot(after[1] - before[1]);

            // PerMove: fold the refined ẑ back into the running sum so the next
            // view sees an updated consensus. `scratch.zbuf` reflects the kept δ
            // — `refine_one_view` re-renders at the kept offset before returning
            // (the GN loop's last `score_at` may have been a rejected line-search
            // candidate, so the explicit re-render is what makes this safe).
            // Skip when the seed core was OOF and the view was never scored.
            if matches!(params.consensus_refresh, ConsensusRefresh::PerMove)
                && states[si].score.is_finite()
            {
                prof::CONSENSUS_UPDATE.time(|| running.update_view(slot, &scratch.zbuf));
            }
        }
        let mean_move = sweep_move_sum / live.len() as f64;

        // Single-pass variant exits after one sweep regardless of convergence;
        // multi-sweep exits as soon as a completed sweep stops moving views.
        if max_sweeps == 1 || mean_move < params.outer_convergence_px {
            break;
        }
    }

    let mut out = finalize(patch, views, &states, wpp_u, wpp_v);
    if params.render_bitmaps {
        out.representative = render_representative(
            patch, views, &states, &tiles, &support, wpp_u, wpp_v, params,
        );
    }
    out
}

/// Fuse the point's representative RGBA texture at the **final** per-view
/// keypoints (the [`KeypointSubpixelParams::render_bitmaps`] path). The views are
/// re-rendered (support-only) at their final offsets to rebuild the final IRLS
/// view weights — the sweep loop's weights predate the last moves — then the live
/// views are rendered full-grid ([`PatchViewStack`]) at the finalize-identical
/// keypoints and fused with those weights ([`AGREEMENT_SIGMA`]). Returns `None`
/// when fewer than two views render in frame at their final offsets — no
/// cross-view consensus exists, so the point has no valid representative (the
/// caller's culled-point signal). Infinity patches (`w = 0`) take the same path.
#[allow(clippy::too_many_arguments)]
fn render_representative(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: &[ViewState],
    tiles: &[Option<RefineTile>],
    support: &Support,
    wpp_u: f64,
    wpp_v: f64,
    params: &KeypointSubpixelParams,
) -> Option<Vec<u8>> {
    if states.len() < 2 {
        return None;
    }
    let resolution = params.resolution.max(2);
    let channels = views[states[0].idx as usize].pyramid.level(0).channels() as usize;
    let n = support.pixels.len();
    let mut raw = vec![0f32; channels * n];
    let mut znorm = vec![0f32; channels * n];
    let mut xs: Vec<f32> = Vec::new();
    let mut live: Vec<usize> = Vec::new();
    for (si, st) in states.iter().enumerate() {
        if core_value(
            patch,
            &views[st.idx as usize],
            tiles[si].as_ref(),
            st.off[0],
            st.off[1],
            wpp_u,
            wpp_v,
            resolution,
            params.sampler,
            support,
            channels,
            &mut raw,
        ) {
            znorm_core(&raw, support, channels, &mut znorm);
            live.push(si);
            xs.extend_from_slice(&znorm);
        }
    }
    if live.len() < 2 {
        return None;
    }

    // Final IRLS view weights over the final-offset cores (parallel to `live`).
    let mut sc = ConsensusScratch::default();
    prof::CONSENSUS.time(|| {
        irls_view_weights(
            &xs,
            live.len(),
            channels,
            n,
            params.robust_iters,
            None,
            &mut sc,
        )
    });
    let weights: Vec<f64> = sc.w[..live.len()].to_vec();

    // Anchor each live view's full-grid render at its final keypoint — the same
    // `shifted_center → project` (with projection fallback) `finalize` reports, so
    // the stored bitmap matches the keypoints the caller writes out.
    let mut view_keypoints: Vec<Option<[f64; 2]>> = vec![None; views.len()];
    let mut kept: Vec<usize> = Vec::with_capacity(live.len());
    for &si in &live {
        let st = &states[si];
        let center = shifted_center(patch, st.off[0], st.off[1], wpp_u, wpp_v);
        let (kx, ky) =
            project(&views[st.idx as usize], &center, patch.w).unwrap_or((st.proj[0], st.proj[1]));
        view_keypoints[st.idx as usize] = Some([kx, ky]);
        kept.push(st.idx as usize);
    }
    Some(prof::REPR_FUSE.time(|| {
        let stack = PatchViewStack::render(
            patch,
            views,
            &kept,
            resolution,
            params.sampler,
            Some(&view_keypoints),
        );
        stack.fuse(&weights, AGREEMENT_SIGMA)
    }))
}

/// The per-move (Gauss–Seidel) incremental consensus state — the running
/// weighted sum `S = Σ_v w_v · ẑ_v` plus the per-view z-normalized cores `ẑ_v`
/// the spec describes. At a sweep boundary `rebuild` resets both from the
/// current `xs` stack and IRLS weights (the spec's two-frequency design: weights
/// are held fixed within a sweep so the delta update is exact). Inside a sweep,
/// after view `v` is refined, `update_view(v, ẑ_v')` does
/// `S += w_v · (ẑ_v' − ẑ_v)` and stores the new `ẑ_v` — O(`channels · n`) per
/// move, the same cost as the move's z-normalization and negligible next to the
/// sampling/gradient render of a GN step. `write_shared_template(out)` realizes
/// the **shared** consensus `normalize(S)`; the spec's leave-one-out alternative
/// (`normalize(S − w_v · ẑ_v)`, the "free with running sum" bonus) was measured
/// against shared T on real data and regressed — see `write_shared_template`'s
/// doc for the numbers and reasoning.
#[derive(Default)]
struct RunningConsensus {
    /// Per-view z-normalized cores `ẑ_v`, stacked `[(v*channels + c)*n + k]`.
    xs: Vec<f32>,
    /// IRLS view weights (per-sweep frequency), `[v]`. `Σw_v = 1`.
    weights: Vec<f64>,
    /// Running weighted sum `S` per (channel, pixel), `[c*n + k]`. f64 to keep
    /// the subtract → renormalize path of a future LOO variant precise (and
    /// harmless for the shared-T variant currently in use).
    s_sum: Vec<f64>,
    /// Number of views currently tracked.
    views: usize,
    channels: usize,
    n: usize,
}

impl RunningConsensus {
    /// Rebuild from the current z-normalized stack and per-sweep IRLS weights.
    /// Called once per outer sweep (the spec's lower-frequency weight refresh).
    fn rebuild(&mut self, xs: &[f32], weights: &[f64], views: usize, channels: usize, n: usize) {
        self.xs.clear();
        self.xs.extend_from_slice(&xs[..views * channels * n]);
        self.weights.clear();
        self.weights.extend_from_slice(&weights[..views]);
        self.views = views;
        self.channels = channels;
        self.n = n;
        self.s_sum.clear();
        self.s_sum.resize(channels * n, 0.0);
        for v in 0..views {
            let wv = weights[v];
            for c in 0..channels {
                let row = &xs[(v * channels + c) * n..][..n];
                let dst = &mut self.s_sum[c * n..][..n];
                for (d, &s) in dst.iter_mut().zip(row) {
                    *d += wv * s as f64;
                }
            }
        }
    }

    /// Delta-update: `S += w_v · (ẑ_v_new − ẑ_v_old)`, then store the new
    /// `ẑ_v`. The view's weight stays fixed (per-sweep refresh frequency).
    fn update_view(&mut self, v: usize, z_new: &[f32]) {
        debug_assert_eq!(z_new.len(), self.channels * self.n);
        let wv = self.weights[v];
        for c in 0..self.channels {
            let old = &mut self.xs[(v * self.channels + c) * self.n..][..self.n];
            let new = &z_new[c * self.n..][..self.n];
            let s = &mut self.s_sum[c * self.n..][..self.n];
            #[allow(clippy::manual_memcpy)] // the loop fuses delta + copy.
            for k in 0..self.n {
                let delta = new[k] as f64 - old[k] as f64;
                s[k] += wv * delta;
                old[k] = new[k];
            }
        }
    }

    /// Write the **shared** per-channel unit-norm template `normalize(S)` into
    /// `out`. Resized in place. A channel whose norm collapses is written as
    /// zeros (matches the [`znorm_core`] convention so the ECC dot product
    /// against it is zero — no gradient pull from a degenerate channel).
    ///
    /// Note on shared vs LOO: the spec offers leave-one-out
    /// (`normalize(S − w_v · ẑ_v)`) as the "free with the running sum" bonus,
    /// avoiding the self-pollution where a view aligns to a `T` that includes
    /// itself. We measured both on the `dino_dog_toy` reconstruction: LOO
    /// consistently produced a lower mean ECC than shared T (mean 0.82 vs 0.87
    /// at 5 sweeps) — at the small view counts of real tracks (3–5 views) LOO
    /// drops effective averaging enough that the per-view template is noisier
    /// than the shared one, and the within-sweep chain of LOO updates amplifies
    /// the drift. The self-pollution turns out to be a damping term that helps,
    /// not a bias worth removing at this scale. So per-move uses **shared T**;
    /// LOO is recorded as the measured-and-rejected alternative.
    fn write_shared_template(&self, out: &mut Vec<f32>) {
        out.clear();
        out.resize(self.channels * self.n, 0.0);
        for c in 0..self.channels {
            let s = &self.s_sum[c * self.n..][..self.n];
            let dst = &mut out[c * self.n..][..self.n];
            let mut norm_sq = 0.0f64;
            for k in 0..self.n {
                let val = s[k];
                norm_sq += val * val;
                dst[k] = val as f32;
            }
            if norm_sq > 1e-24 {
                let inv = (1.0 / norm_sq.sqrt()) as f32;
                for x in dst.iter_mut() {
                    *x *= inv;
                }
            } else {
                dst.fill(0.0);
            }
        }
    }
}

/// Reused per-view scratch for [`refine_one_view`]: the value core `g`, the two
/// per-axis image-Jacobian buffers `Jg_u`/`Jg_v`, a z-normalize buffer (all
/// `channels · n`), and a reused [`ImageF32WithGrad`] for the value+gradient
/// renders (the per-view tile prerenders, and any direct-render fallback).
/// All buffers are allocated once per patch and shared across its views.
///
/// With the render-once [`RefineTile`], the steady-state GN/line-search loop
/// allocates nothing: every evaluation is a cardinal-spline read of the tile
/// into these buffers. The remaining per-patch allocations are the tiles themselves
/// (one value + two gradient planes + validity per view) and the per-tile
/// `WarpMap` inside [`render_refine_tile`].
struct GnScratch {
    g: Vec<f32>,
    jg_u: Vec<f32>,
    jg_v: Vec<f32>,
    zbuf: Vec<f32>,
    img: ImageF32WithGrad,
}

impl GnScratch {
    fn new(len: usize) -> Self {
        Self {
            g: vec![0.0; len],
            jg_u: vec![0.0; len],
            jg_v: vec![0.0; len],
            zbuf: vec![0.0; len],
            img: ImageF32WithGrad::empty(),
        }
    }
}

/// Solve one view's offset by forward-additive ECC Gauss–Newton against the frozen
/// `tmpl`, with the never-worse guard. `tile` is the view's render-once context
/// tile (every evaluation reads it; `None` for a coarse-grid view, which
/// renders directly); `scratch` holds the reused render / z-norm buffers
/// (value plus the per-axis pre-composed image Jacobian).
///
/// **`g` invariant.** At the top of every GN iteration `scratch.g` holds the
/// value core at the current offset `cur`: the seed score fills it, and a step
/// is only accepted when its (last, successful) `score_at` call — which fills
/// `g` — was at the accepted candidate. The tile GN path relies on this: its
/// gradient read ([`RefineTile::read_jg`]) fills only the Jacobian planes and
/// reuses `g` as the value core for the normal equations.
#[allow(clippy::too_many_arguments)]
fn refine_one_view(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    tile: Option<&RefineTile>,
    st: &mut ViewState,
    support: &Support,
    tmpl: &[f32],
    channels: usize,
    resolution: u32,
    wpp_u: f64,
    wpp_v: f64,
    params: &KeypointSubpixelParams,
    scratch: &mut GnScratch,
) {
    let GnScratch {
        g,
        jg_u,
        jg_v,
        zbuf,
        img,
    } = scratch;
    let n = support.pixels.len();

    // Score at a candidate offset; `None` if the core left the frame.
    let score_at = |off: [f64; 2], g: &mut [f32], zbuf: &mut [f32]| -> Option<f64> {
        if !core_value(
            patch,
            view,
            tile,
            off[0],
            off[1],
            wpp_u,
            wpp_v,
            resolution,
            params.sampler,
            support,
            channels,
            g,
        ) {
            return None;
        }
        znorm_core(g, support, channels, zbuf);
        Some(ecc_score(zbuf, tmpl, channels, n))
    };

    // Seed score (the floor the guard never drops below).
    let Some(mut best_score) = score_at(st.off, g, zbuf) else {
        // Seed core out of frame: nothing to refine against; keep the seed.
        st.score = f64::NAN;
        return;
    };
    st.score = best_score;
    let mut cur = st.off;

    for _ in 0..params.max_gn_steps {
        prof::count(&prof::N_GN_STEPS, 1);
        // The per-pixel ∂I/∂δ in patch-grid coords: a tile read of the
        // pre-composed ∇_src I · J planes (`g` already holds the value core at
        // `cur` — the invariant above), or a direct value+gradient render on
        // the no-tile / out-of-coverage path. If any support pixel is out of
        // frame the local Jacobian is ill-defined here: stop.
        if !core_value_with_jg(
            patch,
            view,
            tile,
            cur[0],
            cur[1],
            wpp_u,
            wpp_v,
            resolution,
            params.sampler,
            support,
            channels,
            g,
            jg_u,
            jg_v,
            img,
        ) {
            break;
        }

        let Some((hess, b)) = view_jacobian(g, jg_u, jg_v, tmpl, support, channels) else {
            break; // no textured channel — aperture/low-texture, keep current δ
        };
        let Some(step) = solve_2x2(hess, b) else {
            break; // near-singular system, keep current δ
        };

        // Backtracking line search: accept the largest `α·step` that raises the
        // score, stays within `max_offset_px` of the seed, and stays in frame.
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..params.line_search_max.max(1) {
            let cand = [cur[0] + alpha * step[0], cur[1] + alpha * step[1]];
            let du = cand[0] - st.seed[0];
            let dv = cand[1] - st.seed[1];
            if (du * du + dv * dv).sqrt() <= params.max_offset_px {
                prof::count(&prof::N_LINE_SEARCH, 1);
                if let Some(s) = score_at(cand, g, zbuf) {
                    if s > best_score {
                        let mv = (alpha * step[0]).hypot(alpha * step[1]);
                        cur = cand;
                        best_score = s;
                        accepted = true;
                        if mv < params.convergence_px {
                            // Tiny accepted step → converged.
                            alpha = 0.0;
                        }
                        break;
                    }
                }
            }
            alpha *= params.line_search_shrink;
        }
        if !accepted || alpha == 0.0 {
            break;
        }
    }

    st.off = cur;
    st.score = best_score;
    // PerMove consumes `scratch.zbuf` as the kept-offset z-normalized core (to
    // delta-update the running consensus). The GN loop's last `score_at` may
    // have left `zbuf` at a rejected candidate (line-search exhausted); refresh
    // it to the kept `cur` so the caller can use it unconditionally. PerSweep
    // ignores `zbuf` so this is a small one-extra-render-per-view cost only on
    // the path that needs the value.
    if matches!(params.consensus_refresh, ConsensusRefresh::PerMove)
        && core_value(
            patch,
            view,
            tile,
            cur[0],
            cur[1],
            wpp_u,
            wpp_v,
            resolution,
            params.sampler,
            support,
            channels,
            g,
        )
    {
        znorm_core(g, support, channels, zbuf);
    }
}

/// Build the result from the final view states: the refined keypoint
/// `project_i(center_v)`, its offset from the projection (source px), and the
/// final ECC score, per view (input order preserved).
fn finalize(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: &[ViewState],
    wpp_u: f64,
    wpp_v: f64,
) -> KeypointRefinement {
    let mut out = KeypointRefinement::default();
    for st in states {
        let view = &views[st.idx as usize];
        let center = shifted_center(patch, st.off[0], st.off[1], wpp_u, wpp_v);
        let (kx, ky) = project(view, &center, patch.w).unwrap_or((st.proj[0], st.proj[1]));
        out.views.push(st.idx);
        out.keypoints.push([kx, ky]);
        out.offsets_px
            .push((kx - st.proj[0]).hypot(ky - st.proj[1]));
        out.scores.push(st.score);
    }
    out
}

/// Batch [`refine_patch_keypoints`] over a [`PatchCloud`], parallel across patches
/// (rayon). `view_sets[i]` lists, for patch `i`, the views to refine.
/// `starting_keypoints`, when given, is parallel to `view_sets` (one seed per
/// view); `None` seeds every view at the point's projection. Results are returned
/// in cloud order.
///
/// Note: the PyO3 binding for `PatchCloud.refine_keypoints` does NOT call this
/// wrapper — it inlines its own `par_iter` so it can build per-patch seed slices
/// **lazily** (most patches typically have no caller-provided seed, and the
/// recon-default path picks per-view from a shared `(pid, image_index)` map).
/// Building the full-cloud `Vec<Vec<Option<[f64;2]>>>` up front would force an
/// allocation the binding can avoid. This entry stays as the cloud-level API
/// for Rust callers that already have a parallel-to-cloud seed slice in hand.
///
/// # Panics
///
/// Panics if `view_sets.len() != cloud.len()` (or `starting_keypoints` is given
/// and not parallel), or an index is out of range.
pub fn refine_patch_cloud_keypoints(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    view_sets: &[Vec<u32>],
    starting_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    params: &KeypointSubpixelParams,
) -> Vec<KeypointRefinement> {
    assert_eq!(
        view_sets.len(),
        cloud.len(),
        "view_sets must be parallel to the cloud"
    );
    if let Some(seeds) = starting_keypoints {
        assert_eq!(
            seeds.len(),
            cloud.len(),
            "starting_keypoints must be parallel to the cloud"
        );
    }
    prof::reset();
    let wall_start = std::time::Instant::now();
    let out = cloud
        .patches
        .par_iter()
        .enumerate()
        .map(|(i, patch)| {
            let seeds = starting_keypoints.map(|s| s[i].as_slice());
            refine_patch_keypoints(patch, views, &view_sets[i], seeds, params)
        })
        .collect();
    prof::report(cloud.len(), wall_start.elapsed().as_secs_f64());
    out
}

#[cfg(test)]
mod tests;
