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

use crate::camera::remap::{
    remap_aniso_with_grad_into, remap_aniso_with_pyramid, remap_bilinear,
    remap_bilinear_with_grad_into, ImageF32WithGrad,
};
use crate::camera::WarpMap;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::keypoint_localize::{project, seed_offset, shifted_center};
use crate::patch::normal_refine::{
    build_support, irls_view_weights, weighted_moments_pub, weighted_unit_template_into,
    ConsensusScratch, PatchViewStack, PatchWindow, ProjectedImage, Sampler, Support,
    AGREEMENT_SIGMA, FLAT_NORM_SQ_EPS,
};
use rayon::prelude::*;

pub mod prof;

/// `remap_aniso` sample cap along the major axis (mirrors `normal_refine`).
const MAX_ANISOTROPY: u32 = 16;

/// Within-sweep granularity of consensus refresh — the spec's "Consensus
/// refresh granularity" axis. Across sweeps the consensus is always rebuilt
/// from scratch (the [`KeypointSubpixelParams::max_outer_sweeps`] loop); this
/// enum chooses what happens **inside** a sweep as views move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsensusRefresh {
    /// `T` is held fixed for the duration of a sweep. Every view refines
    /// against the same consensus the sweep was built with; the next sweep
    /// rebuilds `T` from the moved views. With `max_outer_sweeps = 1` this is
    /// the spec's **single-pass frozen** variant; with `> 1` it is the
    /// **per-sweep refresh** variant. The default — preserves existing
    /// behavior.
    #[default]
    PerSweep,
    /// The spec's **per-move (Gauss–Seidel) incremental** variant. The
    /// consensus is maintained as a running weighted sum
    /// `S = Σ_v w_v · ẑ_v` of the z-normalized view cores; when view `v` moves
    /// from `δ` to `δ'`, `S += w_v · (ẑ_v' − ẑ_v)` and `T = normalize(S)`. The
    /// next view's GN solve aligns to a consensus that already reflects the
    /// previous view's move. The IRLS view weights are refreshed at the lower
    /// **per-sweep** frequency (the spec's two-frequency design); within a
    /// sweep the weights are held fixed so the delta update is exact.
    ///
    /// **Shared (not LOO) consensus.** The spec lists leave-one-out as the
    /// "free with the running sum" bonus, avoiding the self-pollution where a
    /// view aligns to a `T` that includes itself. Direct measurement on
    /// `dino_dog_toy` (see `RunningConsensus::write_shared_template`) found
    /// LOO regressed mean ECC (0.82 vs 0.87 for shared) at the small view
    /// counts of real tracks: the chain of LOO updates amplifies drift more
    /// than the self-pollution it removes. PerMove therefore uses shared `T`.
    ///
    /// **Limitation at `N = 2` views.** A direct consequence of the shared-`T`
    /// choice is that with only two views one view's own contribution dominates
    /// the consensus the other view aligns to. On the minimal 2-view planted-
    /// offset fixture PerMove underestimates the relative offset by ~3%
    /// (`per_move_two_views_known_underestimate_at_n2` pins the actual bias);
    /// PerSweep at `N = 2` is unaffected. **`N ≥ 3` is recommended** when
    /// opting into PerMove.
    PerMove,
}

/// Tunables for [`refine_patch_keypoints`].
///
/// The render/window knobs mirror
/// [`KeypointLocalizeParams`](super::keypoint_localize::KeypointLocalizeParams) so
/// the consensus is built on the same conventions as the discrete search that
/// typically seeds this refiner.
#[derive(Debug, Clone)]
pub struct KeypointSubpixelParams {
    /// The `R×R` patch grid the consensus and per-view ECC are scored on.
    pub resolution: u32,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// How to sample the source pyramids when rendering patch tiles. The GN inner
    /// step uses the **value+gradient** variant of the chosen sampler — one render
    /// returns `(value, ∂I/∂x, ∂I/∂y)` per support pixel and channel — composed
    /// per-pixel with the warp Jacobian `J = WarpMap::get_jacobian(col, row)` to
    /// give the analytic `∂I/∂δ` the GN normal equations need. The
    /// [`Sampler::Anisotropic`] gradient is computed at the same LOD(s) /
    /// footprint as the value (per-level bilinear gradient **divided** by the
    /// level's `2^level` to convert from level-pixel to level-0 source-pixel
    /// coords, blended with the same `frac` the value uses), so value and
    /// gradient stay LOD-consistent.
    pub sampler: Sampler,
    /// IRLS reweighting passes for the robust consensus.
    pub robust_iters: u32,
    /// Maximum **outer sweeps** of the alternating loop (refresh consensus → move
    /// every view). `1` is the spec's single-pass-frozen variant (build `T` once at
    /// the seed, hold it fixed) — the cheapest, and the default. `> 1` is the
    /// per-sweep-refresh variant: each subsequent sweep re-renders the views at
    /// their current offsets and rebuilds `T` from those. The outer loop early-exits
    /// when the mean per-view move of a sweep falls below
    /// [`outer_convergence_px`](Self::outer_convergence_px).
    ///
    /// The "never worse than the seed" guard is **per sweep**: within a sweep each
    /// accepted step is non-decreasing in the ECC score against THAT sweep's `T`.
    /// Across sweeps `T` changes, so the final score against the final `T` is not
    /// bit-bounded below by the seed score against the seed `T`. With
    /// `max_outer_sweeps = 1` (the default) the two coincide, so the guarantee is
    /// the spec's strict form.
    pub max_outer_sweeps: u32,
    /// Stop the outer (consensus-refresh) loop once the mean per-view move across a
    /// completed sweep falls below this many patch-grid px. Ignored when
    /// `max_outer_sweeps == 1`.
    pub outer_convergence_px: f64,
    /// Maximum forward-additive Gauss–Newton steps per view per outer sweep.
    pub max_gn_steps: u32,
    /// Stop a view's GN solve once the accepted step magnitude falls below this
    /// many patch-grid px.
    pub convergence_px: f64,
    /// Maximum total per-view drift from the seed, in patch-grid px. A step that
    /// would carry `|δ − δ_seed|` past this is rejected (keeps the local-refiner
    /// contract; the seed must already be in the basin).
    pub max_offset_px: f64,
    /// Backtracking line-search shrink factor (`0 < γ < 1`) and attempt cap: a
    /// rejected step is retried at `γ·step`, up to [`line_search_max`].
    pub line_search_shrink: f64,
    /// Maximum backtracking attempts before a GN step is abandoned (the seed/δ is
    /// kept for that step).
    pub line_search_max: u32,
    /// Within-sweep consensus refresh granularity (the spec's "Consensus refresh
    /// granularity" choice). [`ConsensusRefresh::PerSweep`] (default) holds `T`
    /// fixed for the sweep — preserves existing behavior. [`ConsensusRefresh::PerMove`]
    /// is the spec's Gauss–Seidel incremental variant: after each view moves,
    /// the consensus is delta-updated from the running weighted sum
    /// `S = Σ_v w_v · ẑ_v` so the next view aligns to a `T` that already
    /// reflects the previous move. IRLS weights are still refreshed only at the
    /// per-sweep boundary (the spec's two-frequency design — fixed weights make
    /// the delta exact). Per-move uses the **shared** consensus `normalize(S)`;
    /// the spec's leave-one-out alternative was measured-and-rejected (regressed
    /// mean ECC on real tracks — see [`ConsensusRefresh::PerMove`]).
    /// **Limitation:** at `N = 2` views PerMove underestimates the relative
    /// offset by ~3% (the moved view's own contribution dominates the shared
    /// `T`); `N ≥ 3` is recommended.
    pub consensus_refresh: ConsensusRefresh,
    /// Also fuse each point's **representative RGBA texture** at the FINAL
    /// per-view keypoints (see [`KeypointRefinement::representative`]): after the
    /// last sweep the views are re-rendered at their final offsets, the final IRLS
    /// view weights are rebuilt from those cores, and the kept views are rendered
    /// full-grid ([`PatchViewStack`]) at the final keypoints and fused
    /// (weighted-mean RGB + agreement·coverage alpha, exactly the normal-refine
    /// representative). Points at infinity go through the same path (`w = 0`
    /// rendering is first-class here). Costs one extra full-grid source render per
    /// live view per point, so it is off by default.
    pub render_bitmaps: bool,
    /// Sampler for the representative render ([`render_bitmaps`](Self::render_bitmaps)
    /// path only). Defaults to [`Sampler::Anisotropic`] — the representative is a
    /// stored texture (quality matters more than the per-step refine cost), and the
    /// aniso footprint avoids the bilinear shimmer on foreshortened views. The
    /// refine loop itself keeps using [`sampler`](Self::sampler).
    pub representative_sampler: Sampler,
}

impl Default for KeypointSubpixelParams {
    fn default() -> Self {
        Self {
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: Sampler::Bilinear,
            robust_iters: 3,
            max_outer_sweeps: 1,
            outer_convergence_px: 0.005,
            max_gn_steps: 10,
            convergence_px: 0.01,
            max_offset_px: 2.0,
            line_search_shrink: 0.5,
            line_search_max: 8,
            consensus_refresh: ConsensusRefresh::PerSweep,
            render_bitmaps: false,
            representative_sampler: Sampler::Anisotropic,
        }
    }
}

/// The refined keypoints for one point — parallel arrays over the views, in the
/// **input order**. The one membership change is the projection gate: a view in
/// which the patch centre fails to project (behind the camera or outside the
/// frame) has no projection-anchored offset to report, so it is dropped from the
/// returned set (matching the sibling localizer). Otherwise the set is preserved
/// — a view whose GN solve fails the guard stays at its seed.
#[derive(Debug, Clone, Default)]
pub struct KeypointRefinement {
    /// The image indices, in the input `view_set` order (deduplicated; a repeated
    /// image is refined once).
    pub views: Vec<u32>,
    /// The refined keypoint `project_i(X_p) + δ_v` per view, in source-image
    /// pixels (`[x, y]`), parallel to [`views`](Self::views).
    pub keypoints: Vec<[f64; 2]>,
    /// Per view, the keypoint's offset from the point's projection
    /// `project_i(X_p)` in source-image pixels, parallel to [`views`](Self::views).
    pub offsets_px: Vec<f64>,
    /// Per view, the final ECC score (channel-averaged windowed ZNCC of the
    /// refined core against the frozen consensus). `NaN` when the view could not be
    /// scored (e.g. fewer than two views, so no consensus was built).
    pub scores: Vec<f64>,
    /// The point's fused representative RGBA texture (`R·R·4`, row-major), rendered
    /// at the **final** per-view keypoints and fused with the final IRLS view
    /// weights — only when [`KeypointSubpixelParams::render_bitmaps`] is set.
    /// `None` when the point produced no valid cross-view consensus (fewer than
    /// two views survive the projection gate / render at their final offsets) —
    /// the uniform "culled point" signal, for finite and infinity points alike.
    pub representative: Option<Vec<u8>>,
}

/// Render one view's `R×R` core at in-plane offset `(au, av)` (patch-grid px) into
/// `out` (flat `[channel * n + support_index]`), reading only the window-support
/// pixels. Returns `false` (leaving `out` untouched) when any support pixel is out
/// of frame — a δ whose core left the frame is invalid and can't be scored.
///
/// Value-only path used during scoring (initial seed, line-search candidates).
/// The GN normal-equations build uses [`render_core_with_jg`] for the value plus
/// the per-pixel analytic image Jacobian in one render.
#[allow(clippy::too_many_arguments)]
fn render_core(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    sampler: Sampler,
    support: &Support,
    channels: usize,
    out: &mut [f32],
) -> bool {
    let center = shifted_center(patch, au, av, wpp_u, wpp_v);
    let mut core_patch =
        OrientedPatch::from_center_normal(center, patch.normal(), patch.u_axis, patch.half_extent);
    // Preserve the homogeneous weight so a point at infinity renders as a
    // direction patch (corners are directions), not a finite surfel.
    core_patch.w = patch.w;
    let mut map = WarpMap::from_patch(&core_patch, view.camera, view.cam_from_world, resolution);
    let img = match sampler {
        Sampler::Anisotropic => {
            map.compute_svd();
            remap_aniso_with_pyramid(view.pyramid, &map, MAX_ANISOTROPY)
        }
        Sampler::Bilinear => remap_bilinear(view.pyramid.level(0), &map),
    };
    let n = support.pixels.len();
    for (k, &p) in support.pixels.iter().enumerate() {
        let col = (p % resolution as usize) as u32;
        let row = (p / resolution as usize) as u32;
        if !map.is_valid(col, row) {
            return false;
        }
        for c in 0..channels {
            out[c * n + k] = img.get_pixel(col, row, c as u32) as f32;
        }
    }
    true
}

/// Render one view's `R×R` core at offset `(au, av)` and also fill the analytic
/// image Jacobian `∂I/∂δ` in patch-grid coords per support pixel and channel
/// — one render that returns value + gradient (instead of the previous 5×
/// finite-difference pattern). Returns `false` (leaving outputs untouched) when
/// any support pixel is out of frame.
///
/// Per pixel the sampler returns `(I, ∂I/∂x, ∂I/∂y)` in **source-pixel** coords;
/// composing with the warp Jacobian `J = ∂(source)/∂(grid)` gives `∂I/∂δ` in
/// **patch-grid** coords (`δ = (δ_col, δ_row)`):
///
/// ```text
/// Jg_u = J[0][0] · dI_dx + J[1][0] · dI_dy   (column = u axis)
/// Jg_v = J[0][1] · dI_dx + J[1][1] · dI_dy   (row    = v axis)
/// ```
///
/// where `J[0][0] = dx/dcol`, `J[0][1] = dx/drow`, `J[1][0] = dy/dcol`,
/// `J[1][1] = dy/drow` (the convention `WarpMap::get_jacobian` stores).
#[allow(clippy::too_many_arguments)]
fn render_core_with_jg(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    sampler: Sampler,
    support: &Support,
    channels: usize,
    g: &mut [f32],
    jg_u: &mut [f32],
    jg_v: &mut [f32],
    img_scratch: &mut ImageF32WithGrad,
) -> bool {
    let center = shifted_center(patch, au, av, wpp_u, wpp_v);
    let mut core_patch =
        OrientedPatch::from_center_normal(center, patch.normal(), patch.u_axis, patch.half_extent);
    core_patch.w = patch.w;
    let mut map = WarpMap::from_patch(&core_patch, view.camera, view.cam_from_world, resolution);
    match sampler {
        Sampler::Anisotropic => {
            map.compute_svd(); // also populates jacobians as a by-product
            remap_aniso_with_grad_into(view.pyramid, &map, MAX_ANISOTROPY, img_scratch);
        }
        Sampler::Bilinear => {
            map.compute_jacobians();
            remap_bilinear_with_grad_into(view.pyramid.level(0), &map, img_scratch);
        }
    };
    let n = support.pixels.len();
    let stride = img_scratch.width() as usize * channels;
    let value = img_scratch.value();
    let grad_x = img_scratch.grad_x();
    let grad_y = img_scratch.grad_y();
    for (k, &p) in support.pixels.iter().enumerate() {
        let col = (p % resolution as usize) as u32;
        let row = (p / resolution as usize) as u32;
        if !map.is_valid(col, row) {
            return false;
        }
        let j = map.get_jacobian(col, row);
        let row_off = row as usize * stride + col as usize * channels;
        for c in 0..channels {
            let idx = row_off + c;
            let v = value[idx];
            let gx = grad_x[idx];
            let gy = grad_y[idx];
            g[c * n + k] = v;
            jg_u[c * n + k] = j[0][0] * gx + j[1][0] * gy;
            jg_v[c * n + k] = j[0][1] * gx + j[1][1] * gy;
        }
    }
    true
}

/// z-normalize a raw core (`raw[channel * n + k]`, all channels) over the windowed
/// support into `out` (`out[channel * n + k]`), folding `√w` in so a plain dot
/// realizes the windowed inner product. A channel flat in this core (windowed
/// norm² below [`FLAT_NORM_SQ_EPS`]) is written as zeros. Mirrors
/// `keypoint_localize::znorm_core` / `normal_refine::znormalize_into`.
fn znorm_core(raw: &[f32], support: &Support, channels: usize, out: &mut [f32]) {
    let n = support.pixels.len();
    for c in 0..channels {
        let col = &raw[c * n..][..n];
        let (s1, s2) = weighted_moments_pub(col, &support.weights);
        let mean = (s1 / support.total_weight) as f32;
        let norm_sq = s2 - s1 * (mean as f64);
        let dst = &mut out[c * n..][..n];
        if norm_sq < FLAT_NORM_SQ_EPS {
            dst.fill(0.0);
        } else {
            let inv = (1.0 / norm_sq.sqrt()) as f32;
            for (d, (&x, &sw)) in dst.iter_mut().zip(col.iter().zip(&support.sqrt_weights)) {
                *d = sw * (x - mean) * inv;
            }
        }
    }
}

/// Channel-averaged windowed ZNCC of a z-normalized core against the unit-norm
/// consensus template (both `[c * n + k]`): the ECC score `S(δ)`.
fn ecc_score(znorm: &[f32], tmpl: &[f32], channels: usize, n: usize) -> f64 {
    let mut s = 0.0;
    for c in 0..channels {
        let a = &znorm[c * n..][..n];
        let b = &tmpl[c * n..][..n];
        s += a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum::<f64>();
    }
    s / channels as f64
}

/// The analytic ECC Gauss–Newton normal equations at the current offset. Given
/// the raw core `g` at `δ` and the **pre-composed** raw image Jacobian
/// `Jg = (Jg_u, Jg_v) = ∇_src I · J` (one render of the value+gradient sampler
/// composed per-pixel with the warp Jacobian — see [`render_core_with_jg`]),
/// this composes the z-normalization derivative
/// `∂ẑ_c[k]/∂δ = (∂a/∂δ)/N − a·(a·∂a/∂δ)/N³` (with `a = √w(g − μ)`, `N = ‖a‖`)
/// and accumulates `H = Σ(∂ẑ)(∂ẑ)ᵀ` and `b = Σ(∂ẑ)·T`. Returns `(H, b)` as
/// `([Hxx, Hxy, Hyy], [bx, by])`, or `None` if every channel is flat (no
/// texture to localize on — the aperture/low-texture case the guard keeps the
/// seed for).
#[allow(clippy::too_many_arguments)]
fn view_jacobian(
    g: &[f32],
    jg_u: &[f32],
    jg_v: &[f32],
    tmpl: &[f32],
    support: &Support,
    channels: usize,
) -> Option<([f64; 3], [f64; 2])> {
    let n = support.pixels.len();
    let mut hxx = 0.0;
    let mut hxy = 0.0;
    let mut hyy = 0.0;
    let mut bx = 0.0;
    let mut by = 0.0;
    let mut any_textured = false;

    // Per-pixel ∂ẑ/∂δ, reused per channel.
    let mut dzu = vec![0.0f64; n];
    let mut dzv = vec![0.0f64; n];
    for c in 0..channels {
        let gc = &g[c * n..][..n];
        let (s1, s2) = weighted_moments_pub(gc, &support.weights);
        let mean = s1 / support.total_weight;
        let norm_sq = s2 - s1 * mean;
        if norm_sq < FLAT_NORM_SQ_EPS {
            continue; // flat channel: zeros into ẑ, no gradient contribution
        }
        any_textured = true;
        let nrm = norm_sq.sqrt();
        let inv_n = 1.0 / nrm;
        let inv_n3 = inv_n / norm_sq;

        // a = √w (g − μ); raw image Jacobian Jg = ∂g/∂δ supplied analytically.
        // ∂a/∂δ = √w (Jg − μ'), where μ' = Σ_k w_k·Jg_k / W (∂(weighted mean)/∂δ).
        let jgu_c = &jg_u[c * n..][..n];
        let jgv_c = &jg_v[c * n..][..n];

        // ∂(weighted mean)/∂δ (the centering's mean term).
        let mut mu_du = 0.0;
        let mut mu_dv = 0.0;
        for k in 0..n {
            let w = support.weights[k];
            mu_du += w * jgu_c[k] as f64;
            mu_dv += w * jgv_c[k] as f64;
        }
        mu_du /= support.total_weight;
        mu_dv /= support.total_weight;

        // a·(∂a/∂δ) for the norm-derivative term (Σ_k a_k · ∂a_k/∂δ).
        let mut a_dau = 0.0;
        let mut a_dav = 0.0;
        for k in 0..n {
            let sw = support.sqrt_weights[k] as f64;
            let a = sw * (gc[k] as f64 - mean);
            let dau = sw * (jgu_c[k] as f64 - mu_du);
            let dav = sw * (jgv_c[k] as f64 - mu_dv);
            a_dau += a * dau;
            a_dav += a * dav;
        }

        // ∂ẑ/∂δ per pixel, then accumulate H and b against the template.
        let tc = &tmpl[c * n..][..n];
        for k in 0..n {
            let sw = support.sqrt_weights[k] as f64;
            let a = sw * (gc[k] as f64 - mean);
            let dau = sw * (jgu_c[k] as f64 - mu_du);
            let dav = sw * (jgv_c[k] as f64 - mu_dv);
            dzu[k] = dau * inv_n - a * a_dau * inv_n3;
            dzv[k] = dav * inv_n - a * a_dav * inv_n3;
        }
        for k in 0..n {
            let zu = dzu[k];
            let zv = dzv[k];
            hxx += zu * zu;
            hxy += zu * zv;
            hyy += zv * zv;
            let t = tc[k] as f64;
            bx += zu * t;
            by += zv * t;
        }
    }
    if !any_textured {
        return None;
    }
    Some(([hxx, hxy, hyy], [bx, by]))
}

/// Solve the 2×2 SPD system `H δ = b` (`H = [[Hxx, Hxy], [Hxy, Hyy]]`), with a
/// small Levenberg damping for conditioning. Returns `None` when the (damped)
/// system is near-singular — the aperture problem / low-texture case, where the
/// guard keeps the seed.
fn solve_2x2(h: [f64; 3], b: [f64; 2]) -> Option<[f64; 2]> {
    let [hxx, hxy, hyy] = h;
    // Levenberg damping relative to the trace keeps a degenerate (rank-1) Hessian
    // from producing a huge step along the unconstrained direction.
    let lambda = 1e-3 * (hxx + hyy).max(1e-12);
    let a = hxx + lambda;
    let d = hyy + lambda;
    let det = a * d - hxy * hxy;
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (d * b[0] - hxy * b[1]) * inv_det,
        (a * b[1] - hxy * b[0]) * inv_det,
    ])
}

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
        // 1. Render every view's core at its current offset and z-normalize the
        //    live ones into the per-sweep template-build buffer `xs`.
        live.clear();
        xs.clear();
        for (si, st) in states.iter().enumerate() {
            if render_core(
                patch,
                &views[st.idx as usize],
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
                    running.write_shared_template(&mut tmpl);
                    &tmpl
                }
            };
            refine_one_view(
                patch,
                &views[states[si].idx as usize],
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
                running.update_view(slot, &scratch.zbuf);
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
        out.representative =
            render_representative(patch, views, &states, &support, wpp_u, wpp_v, params);
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
fn render_representative(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: &[ViewState],
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
        if render_core(
            patch,
            &views[st.idx as usize],
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
    irls_view_weights(
        &xs,
        live.len(),
        channels,
        n,
        params.robust_iters,
        None,
        &mut sc,
    );
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
    let stack = PatchViewStack::render(
        patch,
        views,
        &kept,
        resolution,
        params.representative_sampler,
        Some(&view_keypoints),
    );
    Some(stack.fuse(&weights, AGREEMENT_SIGMA))
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
/// pre-composed per-axis image-Jacobian buffers `Jg_u`/`Jg_v` produced by
/// [`render_core_with_jg`], a z-normalize buffer (all `channels · n`), and a
/// reused [`ImageF32WithGrad`] for the value+gradient render
/// (`~3·R²·channels·4 B`, resized in place by the `_into` samplers — after
/// warm-up the **gradient render's own buffer** no longer reallocates per call).
/// All buffers are allocated once per patch and shared across its views.
///
/// Still-allocating-per-call (not scratchified here, follow-up): the
/// per-`render_core_with_jg` `WarpMap` (`data` + `jacobians` + optional `svd`),
/// and the value-only `score_at` render path (which goes through
/// [`render_core`] and allocates an `ImageU8` + `WarpMap` per line-search
/// candidate). These are smaller than the `ImageF32WithGrad` allocation that
/// was just removed, but the per-step budget is not "zero allocations."
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
/// `tmpl`, with the never-worse guard. `scratch` holds the reused render / z-norm
/// buffers (value plus the per-axis pre-composed image Jacobian).
#[allow(clippy::too_many_arguments)]
fn refine_one_view(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
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
        if !render_core(
            patch,
            view,
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
        // One render: value core plus per-pixel ∂I/∂δ in patch-grid coords
        // (∇_src I composed with the warp Jacobian). If any support pixel is out
        // of frame the local Jacobian is ill-defined here: stop.
        if !render_core_with_jg(
            patch,
            view,
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
        && render_core(
            patch,
            view,
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
