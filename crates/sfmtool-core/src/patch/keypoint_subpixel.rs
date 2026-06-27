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
//! constant. Per-move (Gauss–Seidel) incremental refresh and leave-one-out
//! consensus are still deferred (spec "Consensus refresh granularity").
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
//! only the raw sampled core's image Jacobian `∂g/∂δ` is finite-differenced — on
//! the warp/sample coords, the simplest correct route the MVP spec permits (no new
//! value+gradient sampler interface, which is deferred to a later phase).

use crate::camera::remap::{remap_aniso_with_pyramid, remap_bilinear};
use crate::camera::WarpMap;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::keypoint_localize::{project, seed_offset, shifted_center};
use crate::patch::normal_refine::{
    build_support, irls_view_weights, weighted_moments_pub, weighted_unit_template_into,
    ConsensusScratch, PatchWindow, ProjectedImage, Sampler, Support, FLAT_NORM_SQ_EPS,
};
use rayon::prelude::*;

/// `remap_aniso` sample cap along the major axis (mirrors `normal_refine`).
const MAX_ANISOTROPY: u32 = 16;

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
    /// How to sample the source pyramids when rendering patch tiles. Both
    /// [`Sampler::Bilinear`] and [`Sampler::Anisotropic`] use the same sampler for
    /// every render in the GN step (value + four `±h` finite-difference renders),
    /// so the per-axis FD `(I(δ + h) − I(δ − h)) / (2h)` is consistent within a
    /// single LOD. The MVP simplification is that this FD can straddle an
    /// anisotropic mip-level boundary, which then mixes gradients from two LODs;
    /// the LOD-consistent analytic Jacobian (the spec's "Design details") is
    /// deferred to a later phase.
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
    /// Finite-difference step (patch-grid px) for the raw image Jacobian
    /// `∂g/∂δ`. Small enough to be local, large enough to clear sampling noise.
    pub fd_step_px: f64,
    /// Backtracking line-search shrink factor (`0 < γ < 1`) and attempt cap: a
    /// rejected step is retried at `γ·step`, up to [`line_search_max`].
    pub line_search_shrink: f64,
    /// Maximum backtracking attempts before a GN step is abandoned (the seed/δ is
    /// kept for that step).
    pub line_search_max: u32,
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
            fd_step_px: 0.5,
            line_search_shrink: 0.5,
            line_search_max: 8,
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
}

/// Render one view's `R×R` core at in-plane offset `(au, av)` (patch-grid px) into
/// `out` (flat `[channel * n + support_index]`), reading only the window-support
/// pixels. Returns `false` (leaving `out` untouched) when any support pixel is out
/// of frame — a δ whose core left the frame is invalid and can't be scored.
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

/// The analytic ECC Gauss–Newton normal equations at the current offset. Given the
/// raw core `g` at `δ` and the raw cores `g±` finite-differenced `±h` along each
/// patch-grid axis, this composes the z-normalization derivative
/// `∂ẑ_c[k]/∂δ = (∂a/∂δ)/N − a·(a·∂a/∂δ)/N³` (with `a = √w(g − μ)`, `N = ‖a‖`) and
/// accumulates `H = Σ(∂ẑ)(∂ẑ)ᵀ` and `b = Σ(∂ẑ)·T`. Returns `(H, b)` as
/// `([Hxx, Hxy, Hyy], [bx, by])`, or `None` if every channel is flat (no texture
/// to localize on — the aperture/low-texture case the guard keeps the seed for).
#[allow(clippy::too_many_arguments)]
fn view_jacobian(
    g: &[f32],
    g_um: &[f32],
    g_up: &[f32],
    g_vm: &[f32],
    g_vp: &[f32],
    tmpl: &[f32],
    support: &Support,
    channels: usize,
    h: f64,
) -> Option<([f64; 3], [f64; 2])> {
    let n = support.pixels.len();
    let inv_2h = 1.0 / (2.0 * h);
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

        // a = √w (g − μ); raw image Jacobian Jg = ∂g/∂δ by central differences.
        // ∂a/∂δ = √w (Jg − μ'), where μ' = Σ_k w_k·Jg_k / W (∂(weighted mean)/∂δ).
        let gum = &g_um[c * n..][..n];
        let gup = &g_up[c * n..][..n];
        let gvm = &g_vm[c * n..][..n];
        let gvp = &g_vp[c * n..][..n];

        // ∂(weighted mean)/∂δ (the centering's mean term).
        let mut mu_du = 0.0;
        let mut mu_dv = 0.0;
        for k in 0..n {
            let jgu = (gup[k] as f64 - gum[k] as f64) * inv_2h;
            let jgv = (gvp[k] as f64 - gvm[k] as f64) * inv_2h;
            let w = support.weights[k];
            mu_du += w * jgu;
            mu_dv += w * jgv;
        }
        mu_du /= support.total_weight;
        mu_dv /= support.total_weight;

        // a·(∂a/∂δ) for the norm-derivative term (Σ_k a_k · ∂a_k/∂δ).
        let mut a_dau = 0.0;
        let mut a_dav = 0.0;
        for k in 0..n {
            let sw = support.sqrt_weights[k] as f64;
            let a = sw * (gc[k] as f64 - mean);
            let jgu = (gup[k] as f64 - gum[k] as f64) * inv_2h;
            let jgv = (gvp[k] as f64 - gvm[k] as f64) * inv_2h;
            let dau = sw * (jgu - mu_du);
            let dav = sw * (jgv - mu_dv);
            a_dau += a * dau;
            a_dav += a * dav;
        }

        // ∂ẑ/∂δ per pixel, then accumulate H and b against the template.
        let tc = &tmpl[c * n..][..n];
        for k in 0..n {
            let sw = support.sqrt_weights[k] as f64;
            let a = sw * (gc[k] as f64 - mean);
            let jgu = (gup[k] as f64 - gum[k] as f64) * inv_2h;
            let jgv = (gvp[k] as f64 - gvm[k] as f64) * inv_2h;
            let dau = sw * (jgu - mu_du);
            let dav = sw * (jgv - mu_dv);
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
    starting_keypoints: Option<&[[f64; 2]]>,
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
        let off = match starting_keypoints {
            Some(seeds) => seed_offset(patch, view, seeds[k], wpp_u, wpp_v).unwrap_or([0.0, 0.0]),
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
    let h = params.fd_step_px.max(1e-3);

    // Outer sweeps: each sweep re-renders the views at their current offsets,
    // rebuilds the robust consensus from them, and refines every view against it.
    // With `max_outer_sweeps = 1` (default) only the seed-aligned cores ever build
    // `T` (the single-pass-frozen variant). The sweep loop early-exits when the
    // mean per-view move falls below `outer_convergence_px`; on sweeps after the
    // first, a view whose core has left the frame at its current offset doesn't
    // contribute to the rebuilt `T` (it stays out of `live`) but other views still
    // refine against the `T` the in-frame ones build.
    let max_sweeps = params.max_outer_sweeps.max(1);
    let mut live: Vec<usize> = Vec::new();
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
        //    refresh.
        irls_view_weights(&xs, live.len(), channels, n, params.robust_iters, &mut sc);
        weighted_unit_template_into(&xs, &sc.w, live.len(), channels, n, &mut tmpl);

        // 3. Move every live view against this sweep's consensus, tracking the
        //    sweep's mean per-view move for the outer convergence check.
        let mut sweep_move_sum = 0.0;
        for &si in &live {
            let before = states[si].off;
            refine_one_view(
                patch,
                &views[states[si].idx as usize],
                &mut states[si],
                &support,
                &tmpl,
                channels,
                resolution,
                wpp_u,
                wpp_v,
                params,
                h,
                &mut scratch,
            );
            let after = states[si].off;
            sweep_move_sum += (after[0] - before[0]).hypot(after[1] - before[1]);
        }
        let mean_move = sweep_move_sum / live.len() as f64;

        // Single-pass variant exits after one sweep regardless of convergence;
        // multi-sweep exits as soon as a completed sweep stops moving views.
        if max_sweeps == 1 || mean_move < params.outer_convergence_px {
            break;
        }
    }

    finalize(patch, views, &states, wpp_u, wpp_v)
}

/// Reused per-view scratch for [`refine_one_view`]: the value core `g`, the four
/// finite-difference renders, and a z-normalize buffer — all `channels · n`,
/// allocated once per patch and shared across its views.
struct GnScratch {
    g: Vec<f32>,
    g_um: Vec<f32>,
    g_up: Vec<f32>,
    g_vm: Vec<f32>,
    g_vp: Vec<f32>,
    zbuf: Vec<f32>,
}

impl GnScratch {
    fn new(len: usize) -> Self {
        Self {
            g: vec![0.0; len],
            g_um: vec![0.0; len],
            g_up: vec![0.0; len],
            g_vm: vec![0.0; len],
            g_vp: vec![0.0; len],
            zbuf: vec![0.0; len],
        }
    }
}

/// Solve one view's offset by forward-additive ECC Gauss–Newton against the frozen
/// `tmpl`, with the never-worse guard. `scratch` holds the reused render / z-norm
/// buffers (value plus the four finite-difference renders).
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
    h: f64,
    scratch: &mut GnScratch,
) {
    let GnScratch {
        g,
        g_um,
        g_up,
        g_vm,
        g_vp,
        zbuf,
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
        // Render the value core and the four finite-difference cores at `cur`. If
        // any leaves the frame, the local Jacobian is ill-defined here: stop.
        if !render_core(
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
        ) {
            break;
        }
        let ok = render_core(
            patch,
            view,
            cur[0] - h,
            cur[1],
            wpp_u,
            wpp_v,
            resolution,
            params.sampler,
            support,
            channels,
            g_um,
        ) && render_core(
            patch,
            view,
            cur[0] + h,
            cur[1],
            wpp_u,
            wpp_v,
            resolution,
            params.sampler,
            support,
            channels,
            g_up,
        ) && render_core(
            patch,
            view,
            cur[0],
            cur[1] - h,
            wpp_u,
            wpp_v,
            resolution,
            params.sampler,
            support,
            channels,
            g_vm,
        ) && render_core(
            patch,
            view,
            cur[0],
            cur[1] + h,
            wpp_u,
            wpp_v,
            resolution,
            params.sampler,
            support,
            channels,
            g_vp,
        );
        if !ok {
            break;
        }

        let Some((hess, b)) = view_jacobian(g, g_um, g_up, g_vm, g_vp, tmpl, support, channels, h)
        else {
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
/// # Panics
///
/// Panics if `view_sets.len() != cloud.len()` (or `starting_keypoints` is given
/// and not parallel), or an index is out of range.
pub fn refine_patch_cloud_keypoints(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    view_sets: &[Vec<u32>],
    starting_keypoints: Option<&[Vec<[f64; 2]>]>,
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
    cloud
        .patches
        .par_iter()
        .enumerate()
        .map(|(i, patch)| {
            let seeds = starting_keypoints.map(|s| s[i].as_slice());
            refine_patch_keypoints(patch, views, &view_sets[i], seeds, params)
        })
        .collect()
}

#[cfg(test)]
mod tests;
