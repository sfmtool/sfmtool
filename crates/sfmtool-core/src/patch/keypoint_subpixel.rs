// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Photometric subpixel keypoint refinement (the high-accuracy reference).
//!
//! See `specs/core/patch/keypoint-subpixel-refinement.md`. Given a keypoint that is
//! **already close** to correct (the caller's precondition), this refines it to
//! sub-pixel by a **local** continuous optimization: per view, a 2-DOF in-plane
//! translation offset `δ` solved by a few **forward-additive ECC (Enhanced
//! Correlation Coefficient) Gauss–Newton** steps against the point's reference
//! render `T`, the template the localizer aligns views to
//! ([`keypoint_localize`](super::keypoint_localize)): the reference
//! observation's render at its own keypoint, which is never blurred and never
//! moved. Where the point has no reference observation and the reference-view
//! rule finds no eligible view, there is no template and no view moves. It does
//! no grid search; the only membership change
//! is the projection gate (a view in which `project_i(X_p)` fails has no
//! projection-anchored offset to report, so it is dropped). Each accepted step
//! raises the ECC score against `T` and stays in frame, so no view ends worse
//! than its seed. `T` does not change while the views move, so each view is
//! refined once.
//!
//! Points at **infinity** (`w = 0`) are refined exactly like finite ones — the
//! warp + projection already handle `w = 0`, so the same objective, sampling, and
//! Jacobian apply. They are *not* skipped (the opposite of normal refinement).
//!
//! The render → z-normalize machinery is shared with
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
//! **once** into a `RefineTile` (patch-grid-aligned, centred at the view's
//! seed, sized to cover the `max_offset_px` drift, storing the sampler's
//! unquantized values plus the pre-composed patch-grid gradient planes
//! `∇_src I · J`); every evaluation is then a continuous prefiltered
//! cubic-B-spline read of that tile (exact at integer shifts). See the
//! `RefineTile` doc for the exactness/coverage contract and
//! the accepted double-interpolation loss, and
//! `specs/core/patch/keypoint-subpixel-refinement.md` for the design discussion.
//!
//! ## ECC Gauss–Newton, derived
//!
//! Per view the ECC criterion is `S(δ) = (1/C) Σ_c ⟨ẑ_c(δ), T_c⟩`, the
//! channel-averaged windowed ZNCC of the view's z-normalized core `ẑ` against the
//! template `T` (each `T_c` zero-(weighted-)mean, unit-norm, with `√w` folded in,
//! exactly as the template is built for the discrete search). Maximizing `S` is
//! equivalent to least-squares minimizing `½ Σ_c ‖ẑ_c − T_c‖²` (since `‖ẑ_c‖ =
//! ‖T_c‖ = 1`), whose forward-additive Gauss–Newton step is `H δ = b` with
//! `H = Σ_c Σ_k (∂ẑ_c[k]/∂δ)(∂ẑ_c[k]/∂δ)ᵀ` and `b = Σ_c Σ_k (∂ẑ_c[k]/∂δ) T_c[k]`
//! — and `b = C·∇S` because `Σ_k (∂ẑ_c[k]/∂δ)·ẑ_c[k] = ½∂‖ẑ_c‖²/∂δ = 0`. So the
//! step rises along the score gradient with the natural GN Hessian. The
//! z-normalization derivative `∂ẑ` is taken analytically (see `view_jacobian`);
//! the raw image Jacobian `∂g/∂δ` is now also analytic, via the sampler's
//! value+gradient interface (`remap_bilinear_with_grad` / `remap_aniso_with_grad`,
//! returning `(I, ∂I/∂x, ∂I/∂y)` in source-pixel coords per support pixel and
//! channel) composed pixel-wise with the warp Jacobian
//! (`WarpMap::get_jacobian`): `∂I/∂δ = ∇_src I · J`. The previous finite-difference
//! path took five renders per GN step; the analytic path takes one.

use crate::camera::image::ImageF32WithGrad;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::keypoint_localize::{project, resolve_reference, seed_frame, shifted_center};
use crate::patch::normal_refine::{build_support, ProjectedImage, Support};
use crate::progress::{Cancelled, Progress};
use rayon::prelude::*;

pub mod prof;

mod kernels;
mod params;

// Public API, re-exported at the historical `keypoint_subpixel::` paths.
pub use params::{KeypointRefinement, KeypointSubpixelParams};

// Rendering + scoring kernels consumed by the Gauss–Newton orchestration below.
use crate::camera::sampler::render_phase;
use crate::patch::normal_refine::Sampler;
use crate::patch::reference_view::render_view_tile;
use crate::patch::stored_bitmap::bitmap_from_tile;
use crate::patch::PatchCounter;
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
    /// The view's starting keypoint as given, source px (`None` starts at the
    /// projection).
    keypoint: Option<[f64; 2]>,
    /// The patch slid onto `keypoint` (`seed_frame`): the frame every render
    /// of the view is cut through and every offset is read on, from its centre.
    patch_frame: OrientedPatch,
    /// Whether `patch_frame` is at `keypoint`: no keypoint was given, or the slide
    /// followed its ray. When it did not, `patch_frame` is the patch, at the
    /// projection.
    seed_is_keypoint: bool,
    /// Current offset `(au, av)` from `patch_frame`'s centre in patch-grid px; the
    /// seed is `(0, 0)`.
    off: [f64; 2],
    /// The view's projection of the point `project_i(X_p)`, source px.
    proj: [f64; 2],
    /// Final ECC score (NaN until scored).
    score: f64,
    /// The sampler every render of this view uses, chosen once by the rule for
    /// the observation at its seed keypoint
    /// ([`SamplerChoice::for_observation`](crate::camera::sampler::SamplerChoice::for_observation)),
    /// so the GN steps, the template and the stored bitmap all read the view
    /// through one sampler.
    sampler: Sampler,
}

/// Refine the per-view keypoints of one oriented patch by forward-additive ECC
/// Gauss–Newton against the point's reference render.
///
/// `views` is one [`ProjectedImage`] per reconstruction image (indexed by image
/// index); `patch_views` lists the views to refine. `patch_keypoints`, when given,
/// is one seed per `patch_views` entry (source-image px); `None` seeds every view at
/// the point's own projection `project_i(X_p)`. `reference` is the position in
/// `patch_views` of the point's reference observation, whose render at its
/// starting keypoint is the template and whose keypoint is returned unmoved;
/// `None` has the reference-view rule pick one from the renders at the starting
/// keypoints, as the localizer does. Returns the views (input order,
/// deduplicated) with their refined keypoints. The only membership change is the
/// projection gate (a view in which `project_i(X_p)` fails — behind the camera or
/// out of frame — is dropped, as the offset has nothing to be measured from);
/// otherwise the set is preserved, and a guard-failed view keeps its seed. See
/// `specs/core/patch/keypoint-subpixel-refinement.md`.
pub fn refine_patch_keypoints(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    patch_views: &[u32],
    patch_keypoints: Option<&[Option<[f64; 2]>]>,
    reference: Option<usize>,
    params: &KeypointSubpixelParams,
) -> KeypointRefinement {
    refine_patch_keypoints_reporting(
        patch,
        views,
        patch_views,
        patch_keypoints,
        reference,
        params,
        &Progress::none(),
    )
}

/// [`refine_patch_keypoints`] reporting to `progress`: a detailed `progress`
/// times the renders of each sampler in its own detail phase
/// ([`crate::camera::sampler::render_phase`]). Nothing here polls for
/// cancellation; the batch callers do, between patches.
pub fn refine_patch_keypoints_reporting(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    patch_views: &[u32],
    patch_keypoints: Option<&[Option<[f64; 2]>]>,
    reference: Option<usize>,
    params: &KeypointSubpixelParams,
    progress: &Progress<'_>,
) -> KeypointRefinement {
    prof::TOTAL.time(|| {
        refine_patch_keypoints_impl(
            patch,
            views,
            patch_views,
            patch_keypoints,
            reference,
            params,
            progress,
        )
    })
}

/// Untimed body of [`refine_patch_keypoints`] (split so the enclosing
/// [`prof::TOTAL`] phase is a single wrap covering both batch entries — the
/// Rust [`refine_patch_cloud_keypoints`] and the PyO3 binding's inlined loop).
fn refine_patch_keypoints_impl(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    patch_views: &[u32],
    patch_keypoints: Option<&[Option<[f64; 2]>]>,
    reference: Option<usize>,
    params: &KeypointSubpixelParams,
    progress: &Progress<'_>,
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
    for (k, &i) in patch_views.iter().enumerate() {
        if !seen.insert(i) {
            continue;
        }
        let view = &views[i as usize];
        let Some(proj) = project(view, &patch.center, patch.w) else {
            continue;
        };
        let seed = patch_keypoints.and_then(|seeds| seeds[k]);
        let (patch_frame, seed_is_keypoint) = seed_frame(patch, view, seed);
        let sampler = params.sampler.for_observation(
            patch,
            view.camera,
            view.cam_from_world,
            seed,
            resolution,
        );
        states.push(ViewState {
            idx: i,
            keypoint: seed,
            patch_frame,
            seed_is_keypoint,
            off: [0.0; 2],
            proj: [proj.0, proj.1],
            score: f64::NAN,
            sampler,
        });
    }

    if states.len() < 2 {
        // No second view to align: keep every seed.
        return finalize(patch, views, &states, None, wpp_u, wpp_v);
    }

    // Channel count is the first view's image channels; the warp renders at that
    // count, and all reconstruction images share it.
    let channels = views[states[0].idx as usize].pyramid.level(0).channels() as usize;
    let mut raw = vec![0f32; channels * n];
    let mut znorm = vec![0f32; channels * n];
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
            let _phase = render_phase(progress, st.sampler, 1);
            try_render_refine_tile(
                &st.patch_frame,
                &views[st.idx as usize],
                [0.0; 2],
                wpp_u,
                wpp_v,
                resolution,
                pad,
                st.sampler,
                &mut scratch.img,
            )
        })
        .collect();

    // What the views are aligned to: the caller's reference, else the
    // reference-view rule's pick at the starting keypoints, resolved by the
    // function the localizer uses, with its grazing pre-filter. Where the rule
    // finds no eligible view there is none.
    let mut anchor = prof::REFERENCE.time(|| {
        // Matched by image, so a reference given at a repeated image's
        // dropped slot is kept.
        let given = reference
            .and_then(|k| patch_views.get(k))
            .and_then(|&img| states.iter().position(|st| st.idx == img));
        let set: Vec<u32> = states.iter().map(|st| st.idx).collect();
        let seeds: Vec<Option<[f64; 2]>> = states.iter().map(|st| st.keypoint).collect();
        resolve_reference(
            patch,
            views,
            &set,
            &seeds,
            given,
            params.min_grazing_cos,
            resolution,
            params.sampler,
            &params.reference_view,
            progress,
        )
    });

    // The template `T`: the reference's core at its own keypoint,
    // z-normalized with `√w` folded in. Never blurred. A reference whose
    // keypoint the slide refuses has no core there.
    let template: Option<Vec<f32>> = match anchor {
        Some(a) if !states[a].seed_is_keypoint => None,
        Some(a) => {
            let st = &states[a];
            core_value(
                &st.patch_frame,
                &views[st.idx as usize],
                tiles[a].as_ref(),
                st.off[0],
                st.off[1],
                wpp_u,
                wpp_v,
                resolution,
                st.sampler,
                &support,
                channels,
                &mut raw,
            )
            .then(|| {
                znorm_core(&raw, &support, channels, &mut znorm);
                znorm.clone()
            })
        }
        None => None,
    };
    // As in the localizer, a reference whose core cannot be rendered at its
    // keypoint (out of frame, flat, or refused by the slide) is no reference:
    // nothing is aligned, every view keeps its seed unscored, and no stored
    // bitmap is rendered.
    if template.is_none() {
        anchor = None;
    }

    // Move every view but the reference against `T`, once.
    if let Some(tmpl) = &template {
        if let Some(a) = anchor {
            states[a].score = 1.0;
        }
        for si in 0..states.len() {
            if Some(si) == anchor {
                continue;
            }
            refine_one_view(
                &views[states[si].idx as usize],
                tiles[si].as_ref(),
                &mut states[si],
                &support,
                tmpl,
                channels,
                resolution,
                wpp_u,
                wpp_v,
                params,
                &mut scratch,
            );
            reanchor(
                patch,
                &views[states[si].idx as usize],
                &mut states[si],
                &support,
                tmpl,
                channels,
                resolution,
                wpp_u,
                wpp_v,
                pad,
                params,
                &mut scratch,
            );
        }
    }

    let mut out = finalize(patch, views, &states, anchor, wpp_u, wpp_v);
    // The reference is reported whether or not the bitmap is rendered: callers
    // that refine without it still store which observation the keypoints were
    // aligned to.
    out.reference = anchor;
    if params.render_bitmaps {
        // The stored bitmap is the reference's tile at its keypoint, which the
        // refinement did not move; where there is no reference there is none.
        if let Some(a) = anchor {
            let st = &states[a];
            let tile = render_view_tile(
                patch,
                &views[st.idx as usize],
                Some(out.keypoints[a]),
                resolution as usize,
                params.sampler,
                progress,
            );
            out.representative = Some(bitmap_from_tile(&tile));
        }
    }
    out
}

/// The raw core of an `R×R` RGBA `bitmap` over `support`, `channels` channels
/// planar `[c · n + k]`: channel `c` reads the bitmap's colour channel `c`, the
/// last of its three for a wider view, so an alpha (a confidence, not a
/// colour) is never read.
fn bitmap_core(
    bitmap: &[u8],
    resolution: usize,
    support: &Support,
    channels: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(bitmap.len(), resolution * resolution * 4);
    let n = support.pixels.len();
    for (k, &p) in support.pixels.iter().enumerate() {
        for c in 0..channels {
            out[c * n + k] = f32::from(bitmap[p * 4 + c.min(2)]);
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
/// `WarpMap` inside `kernels::render_refine_tile`.
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
    let sampler = st.sampler;
    // Every offset is read on the view's own frame, slid onto its seed.
    let patch = &st.patch_frame.clone();

    // Score at a candidate offset; `None` if the core left the frame.
    let score_at = |off: [f64; 2], g: &mut [f32], zbuf: &mut [f32]| -> Option<f64> {
        if !core_value(
            patch, view, tile, off[0], off[1], wpp_u, wpp_v, resolution, sampler, support,
            channels, g,
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
            patch, view, tile, cur[0], cur[1], wpp_u, wpp_v, resolution, sampler, support,
            channels, g, jg_u, jg_v, img,
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
            let (du, dv) = (cand[0], cand[1]);
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
}

/// An offset, in grid px, past which the refiner renders the view again through
/// the patch slid onto the keypoint it reached and refines once more from
/// there.
///
/// The view's tile is rendered once and read at every offset the solve tries,
/// each read being the frame moved within its own plane; the patch slid onto
/// that keypoint sits on the same ray at another depth, and the tile's corners
/// part from it by a measured p95 of 0.003 to 0.010 px for offsets under 1 grid
/// px and 0.025 to 0.10 px from 1 to 2, on the three ground truths.
const REANCHOR_MIN_OFFSET: f64 = 0.5;

/// Where `st`'s refined offset is past [`REANCHOR_MIN_OFFSET`], render its tile
/// again through the patch slid onto the keypoint it reached and solve again
/// from there, within what is left of `max_offset_px` of its seed. The second
/// solve's result is kept only where it scores at least what the first
/// reached, so the refinement still never ends below its seed.
#[allow(clippy::too_many_arguments)]
fn reanchor(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    st: &mut ViewState,
    support: &Support,
    tmpl: &[f32],
    channels: usize,
    resolution: u32,
    wpp_u: f64,
    wpp_v: f64,
    pad: u32,
    params: &KeypointSubpixelParams,
    scratch: &mut GnScratch,
) {
    let moved = st.off[0].hypot(st.off[1]);
    if !(st.score.is_finite() && moved > REANCHOR_MIN_OFFSET) {
        return;
    }
    let centre = shifted_center(&st.patch_frame, st.off[0], st.off[1], wpp_u, wpp_v);
    let Some((x, y)) = project(view, &centre, patch.w) else {
        return;
    };
    let (frame, followed) = seed_frame(patch, view, Some([x, y]));
    if !followed {
        return;
    }
    let tile = try_render_refine_tile(
        &frame,
        view,
        [0.0; 2],
        wpp_u,
        wpp_v,
        resolution,
        pad,
        st.sampler,
        &mut scratch.img,
    );
    let kept = (st.patch_frame.clone(), st.off, st.score, st.keypoint);
    // The frame slid onto the reached keypoint stands as the view's start: a
    // zero offset from it is that keypoint.
    st.patch_frame = frame;
    st.off = [0.0; 2];
    st.keypoint = Some([x, y]);
    // The second solve moves from the reached keypoint, so what it may move is
    // what is left of the bound: the two moves together stay within
    // `max_offset_px` of the seed. With nothing left it only scores the new
    // render at the reached keypoint.
    let left = KeypointSubpixelParams {
        max_offset_px: (params.max_offset_px - moved).max(0.0),
        ..params.clone()
    };
    refine_one_view(
        view,
        tile.as_ref(),
        st,
        support,
        tmpl,
        channels,
        resolution,
        wpp_u,
        wpp_v,
        &left,
        scratch,
    );
    // The never-worse guard across the two solves: where the second does not
    // score at least what the first reached, the first stands.
    if !(st.score.is_finite() && st.score >= kept.2) {
        (st.patch_frame, st.off, st.score, st.keypoint) = kept;
    }
}

/// Build the result from the final view states: the refined keypoint
/// `project_i(center_v)`, its offset from the projection (source px), and the
/// final ECC score, per view (input order preserved).
fn finalize(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: &[ViewState],
    anchor: Option<usize>,
    wpp_u: f64,
    wpp_v: f64,
) -> KeypointRefinement {
    let mut out = KeypointRefinement::default();
    for (si, st) in states.iter().enumerate() {
        let view = &views[st.idx as usize];
        let center = shifted_center(&st.patch_frame, st.off[0], st.off[1], wpp_u, wpp_v);
        let projected = project(view, &center, patch.w).unwrap_or((st.proj[0], st.proj[1]));
        // The reference's keypoint, and any view's that did not move (no
        // template, or the guard kept its seed), is returned exactly as it was
        // given rather than through a round trip onto the plane.
        let unmoved = Some(si) == anchor || (st.off == [0.0; 2] && st.seed_is_keypoint);
        let (kx, ky) = match (unmoved, st.keypoint) {
            (true, Some([x, y])) => (x, y),
            _ => projected,
        };
        out.views.push(st.idx);
        out.keypoints.push([kx, ky]);
        out.offsets_px
            .push((kx - st.proj[0]).hypot(ky - st.proj[1]));
        out.scores.push(st.score);
    }
    out
}

/// What one view is refined against by [`refine_view_against_reference`]: the
/// point's stored bitmap, or its reference observation rendered at its
/// keypoint.
#[derive(Debug, Clone, Copy)]
pub enum ReferenceTemplate<'a> {
    /// An `R×R` RGBA patch bitmap on the refinement's grid, row-major, as a
    /// `.sfmr` stores it. Its alpha is not read.
    Bitmap(&'a [u8]),
    /// The reference observation: its image index and keypoint, source px.
    Observation {
        /// The reference observation's image index into `views`.
        image: u32,
        /// The reference observation's keypoint, source px.
        keypoint: [f64; 2],
    },
}

/// Refine one view's keypoint against the point's reference render, moving
/// nothing else.
///
/// The template is `template`: the stored bitmap, or the reference
/// observation's render at its keypoint. `target` is refined from
/// `target_keypoint` by the same ECC Gauss-Newton solve every view gets in
/// [`refine_patch_keypoints`], with the never-worse guard, and its keypoint is
/// returned with its final ECC score against the template. This is the
/// sub-pixel step of adding an observation to an existing track
/// (`specs/core/reconstruction/add-image-to-tracks.md`).
///
/// `None` when the template is not on the refinement's grid or its reference
/// does not render in frame, the reference's channel count differs from the
/// target's, the target does not project into its frame, or the target's core
/// is out of frame at its seed.
///
/// # Panics
///
/// Panics if an image index is out of range for `views`.
pub fn refine_view_against_reference(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    template: ReferenceTemplate<'_>,
    target: u32,
    target_keypoint: [f64; 2],
    params: &KeypointSubpixelParams,
) -> Option<([f64; 2], f64)> {
    let resolution = params.resolution.max(2);
    let wpp_u = 2.0 * patch.half_extent[0] / resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / resolution as f64;
    let support = build_support(params.window, resolution);
    let n = support.pixels.len();
    let target_view = &views[target as usize];
    let channels = target_view.pyramid.level(0).channels() as usize;

    let mut raw = vec![0f32; channels * n];
    let mut tmpl = vec![0f32; channels * n];
    match template {
        ReferenceTemplate::Bitmap(bitmap) => {
            let r = resolution as usize;
            if bitmap.len() != r * r * 4 {
                return None;
            }
            bitmap_core(bitmap, r, &support, channels, &mut raw);
        }
        ReferenceTemplate::Observation { image, keypoint } => {
            let view = &views[image as usize];
            if view.pyramid.level(0).channels() as usize != channels {
                return None;
            }
            let (patch_frame, followed) = seed_frame(patch, view, Some(keypoint));
            if !followed {
                return None;
            }
            let sampler = params.sampler.for_observation(
                patch,
                view.camera,
                view.cam_from_world,
                Some(keypoint),
                resolution,
            );
            if !core_value(
                &patch_frame,
                view,
                None,
                0.0,
                0.0,
                wpp_u,
                wpp_v,
                resolution,
                sampler,
                &support,
                channels,
                &mut raw,
            ) {
                return None;
            }
        }
    }
    znorm_core(&raw, &support, channels, &mut tmpl);

    let proj = project(target_view, &patch.center, patch.w)?;
    let (patch_frame, followed) = seed_frame(patch, target_view, Some(target_keypoint));
    if !followed {
        return None;
    }
    let mut state = ViewState {
        idx: target,
        seed_is_keypoint: true,
        keypoint: Some(target_keypoint),
        patch_frame,
        off: [0.0; 2],
        proj: [proj.0, proj.1],
        score: f64::NAN,
        sampler: params.sampler.for_observation(
            patch,
            target_view.camera,
            target_view.cam_from_world,
            Some(target_keypoint),
            resolution,
        ),
    };
    let mut scratch = GnScratch::new(channels * n);
    let pad = (params.max_offset_px.max(0.0).ceil() as u32).max(1) + 2;
    let tile = try_render_refine_tile(
        &state.patch_frame,
        target_view,
        [0.0; 2],
        wpp_u,
        wpp_v,
        resolution,
        pad,
        state.sampler,
        &mut scratch.img,
    );
    refine_one_view(
        target_view,
        tile.as_ref(),
        &mut state,
        &support,
        &tmpl,
        channels,
        resolution,
        wpp_u,
        wpp_v,
        params,
        &mut scratch,
    );
    reanchor(
        patch,
        target_view,
        &mut state,
        &support,
        &tmpl,
        channels,
        resolution,
        wpp_u,
        wpp_v,
        pad,
        params,
        &mut scratch,
    );
    if !state.score.is_finite() {
        return None;
    }
    let center = shifted_center(&state.patch_frame, state.off[0], state.off[1], wpp_u, wpp_v);
    project(target_view, &center, patch.w).map(|(x, y)| ([x, y], state.score))
}

/// Batch [`refine_patch_keypoints`] over a [`PatchCloud`], parallel across patches
/// (rayon). `patch_views[i]` lists, for patch `i`, the views to refine.
/// `patch_keypoints`, when given, is parallel to `patch_views` (one seed per
/// view); `None` seeds every view at the point's projection. `references`,
/// when given, is parallel to the cloud: per patch, the position in its patch views
/// of its reference observation, or `None` to have the reference-view rule pick
/// one. Results are returned in cloud order.
///
/// Note: the PyO3 binding for `PatchCloud.refine_keypoints` does NOT call this
/// wrapper — it inlines its own `par_iter` so it can build per-patch seed slices
/// **lazily** (most patches typically have no caller-provided seed, and the
/// recon-default path picks per-view from a shared `(pid, image_index)` map).
/// Building the full-cloud `Vec<Vec<Option<[f64;2]>>>` up front would force an
/// allocation the binding can avoid. This entry stays as the cloud-level API
/// for Rust callers that already have a parallel-to-cloud seed slice in hand.
///
/// `progress` receives a `patches` count about every hundredth of the way
/// through, is polled for cancellation before each patch, and, when detailed,
/// times the renders of each sampler in its own detail phase.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled before every patch was refined.
///
/// # Panics
///
/// Panics if `patch_views.len() != cloud.len()` (or `patch_keypoints` or
/// `references` is given and not parallel), or an index is out of range.
pub fn refine_patch_cloud_keypoints(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    patch_views: &[Vec<u32>],
    patch_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    references: Option<&[Option<usize>]>,
    params: &KeypointSubpixelParams,
    progress: &Progress<'_>,
) -> Result<Vec<KeypointRefinement>, Cancelled> {
    assert_eq!(
        patch_views.len(),
        cloud.len(),
        "patch_views must be parallel to the cloud"
    );
    if let Some(seeds) = patch_keypoints {
        assert_eq!(
            seeds.len(),
            cloud.len(),
            "patch_keypoints must be parallel to the cloud"
        );
    }
    if let Some(refs) = references {
        assert_eq!(
            refs.len(),
            cloud.len(),
            "references must be parallel to the cloud"
        );
    }
    prof::reset();
    let wall_start = std::time::Instant::now();
    let counter = PatchCounter::new(cloud.len(), None, progress);
    let out: Vec<Option<KeypointRefinement>> = cloud
        .patches
        .par_iter()
        .enumerate()
        .map(|(i, patch)| {
            if progress.is_cancelled() {
                return None;
            }
            let seeds = patch_keypoints.map(|s| s[i].as_slice());
            let out = refine_patch_keypoints_reporting(
                patch,
                views,
                &patch_views[i],
                seeds,
                references.and_then(|r| r[i]),
                params,
                progress,
            );
            counter.finished();
            Some(out)
        })
        .collect();
    prof::report(cloud.len(), wall_start.elapsed().as_secs_f64());
    progress.check_cancel()?;
    Ok(out
        .into_iter()
        .map(|o| o.expect("every patch ran when nothing was cancelled"))
        .collect())
}

#[cfg(test)]
mod tests;
