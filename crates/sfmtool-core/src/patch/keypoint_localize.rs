// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch-keypoint localization by group-wise translation registration
//! (congealing).
//!
//! See `specs/core/patch-keypoint-localization.md`. Given one 3D point with its
//! oriented patch, a view set `G`, and a starting keypoint per view,
//! [`localize_patch_keypoints`] refines each keypoint to sub-pixel and reports
//! which views it kept. Each round it renders every view's patch tile at its
//! accumulated in-plane offset (a *single* resample of the source, never
//! re-warping an already-warped tile, so blur cannot compound), builds the
//! robust (IRLS) consensus, and searches each view's residual in-plane shift
//! against the **leave-one-out** consensus of the *others* (so a view is never
//! aligned to a template its own pixels polluted). Views that drift too far,
//! leave the frame, or stop agreeing are dropped in-loop, so the survivors
//! register against a cleaner template.
//!
//! The render → z-normalize → robust-consensus machinery is the same as
//! [normal refinement](super::normal_refine) and
//! [view selection](super::view_selection); the new kernel here is the per-view
//! windowed-ZNCC translation search against the leave-one-out consensus.

pub mod prof;

use crate::camera::remap::{remap_aniso_with_pyramid, remap_bilinear};
use crate::camera::WarpMap;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::normal_refine::{
    irls_view_weights, window_weights, znormalize_into_kept, ConsensusScratch, PatchWindow,
    ProjectedImage, Sampler, FLAT_NORM_SQ_EPS,
};
// Only the reference scorer (`znorm_core`, test-only) needs the moment helper.
#[cfg(test)]
use crate::patch::normal_refine::weighted_moments_pub;
use crate::reconstruction::SfmrReconstruction;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// `remap_aniso` sample cap along the major axis (mirrors `normal_refine`).
const MAX_ANISOTROPY: u32 = 16;

/// Tunables for [`localize_patch_keypoints`].
///
/// The render/window knobs mirror
/// [`NormalRefineParams`](super::normal_refine::NormalRefineParams) so the
/// consensus is built on the same conventions as refinement and selection.
#[derive(Debug, Clone)]
pub struct KeypointLocalizeParams {
    /// Maximum congealing rounds; the loop stops early once the mean per-view
    /// residual shift falls below [`convergence_px`](Self::convergence_px).
    pub max_iters: u32,
    /// Max total per-view drift from the point's projection (patch-grid px). The
    /// accumulated offset is clipped to `±search` each round, bounding runaway, and
    /// the context tile is rendered this much larger than the scored core so the
    /// shift search can slide without running off the edge. (The drift, the
    /// `max_shift_px` gate, and the reported offset are all anchored at the
    /// projection `project_i(X_p)`, i.e. `acc = 0`.)
    pub search: f64,
    /// Drop a view whose refined keypoint sits more than this many *source-image*
    /// pixels from the point's projection `project_i(X_p)` (an absolute distance,
    /// not the per-round move).
    pub max_shift_px: f64,
    /// Drop a view whose leave-one-out ZNCC falls below this fraction of the
    /// views' *median* leave-one-out ZNCC (relative, so a uniformly low-texture
    /// patch is not over-dropped).
    pub min_relative_zncc: f64,
    /// Grazing cutoff: drop a view whose viewing ray is near-parallel to the
    /// patch plane (`|d̂ · n̂|` below this), where the in-plane anchor is
    /// ill-conditioned and the view would only contaminate the consensus.
    pub min_grazing_cos: f64,
    /// The `R×R` patch grid the consensus and per-view ZNCC are scored on.
    pub resolution: u32,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// How to sample the source pyramids when rendering patch tiles.
    pub sampler: Sampler,
    /// IRLS reweighting passes for the robust consensus.
    pub robust_iters: u32,
    /// Convergence threshold: stop once the mean per-view residual shift of a
    /// round is below this many patch-grid px.
    pub convergence_px: f64,
}

impl Default for KeypointLocalizeParams {
    fn default() -> Self {
        Self {
            max_iters: 5,
            search: 6.0,
            max_shift_px: 3.0,
            min_relative_zncc: 0.7,
            min_grazing_cos: 0.1,
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: Sampler::Bilinear,
            robust_iters: 3,
            convergence_px: 0.05,
        }
    }
}

/// The localized keypoints for one point — parallel arrays over the **kept**
/// views (a subset of the input view set, in the input's order; grazing /
/// out-of-frame / large-shift / low-agreement views are dropped in-loop).
#[derive(Debug, Clone, Default)]
pub struct KeypointLocalization {
    /// The kept image indices (into the `views` slice), a subset of the input
    /// view set preserving its order.
    pub views: Vec<u32>,
    /// The refined keypoint `project_i(X_p) + δ_j` per kept view, in source-image
    /// pixels (`[x, y]`), parallel to [`views`](Self::views).
    pub keypoints: Vec<[f64; 2]>,
    /// Per kept view, the keypoint's offset from the point's projection
    /// `project_i(X_p)` in source-image pixels (`|δ_j|`), parallel to
    /// [`views`](Self::views).
    pub offsets_px: Vec<f64>,
    /// Per kept view, the leave-one-out ZNCC against the other views' consensus
    /// from the last round that scored it (the integer-peak value of that round's
    /// shift search), parallel to [`views`](Self::views). `NaN` for a view no round
    /// ever scored — e.g. a lone input view, or a view kept by the early
    /// "fewer than two views remain" exit before any consensus was built.
    pub loo_zncc: Vec<f64>,
}

/// The frozen window support over the `R×R` core: the linear `row * R + col`
/// indices of the positive-weight pixels, their window weights, and `√weight`.
struct Support {
    /// Linear `row * R + col` indices of the window-support pixels.
    pixels: Vec<usize>,
    /// Window weight per support pixel (parallel to `pixels`).
    weights: Vec<f64>,
    /// `√weight` per support pixel (folded into the z-normalized space so a plain
    /// dot product realizes the windowed inner product).
    sqrt_w: Vec<f32>,
    /// Sum of `weights` (the windowed mean's denominator).
    total_w: f64,
}

/// A rendered context tile for one view: source colour over a
/// `context_res × context_res` grid (larger than the scored `R×R` core so the
/// shift search can slide), plus per-pixel validity from the warp map (an
/// invalid pixel is out of frame, rendered black, and must not be scored).
struct ContextTile {
    /// Colour, flat `[(row * context_res + col) * channels + channel]` (f32).
    px: Vec<f32>,
    /// Per-pixel validity, `[row * context_res + col]`.
    valid: Vec<bool>,
    /// Channel count.
    channels: usize,
}

/// Project a homogeneous world point `(p, w)` into a view; `None` when it falls
/// behind the camera or outside the frame. `w = 1` is a finite point; `w = 0` is
/// a direction (a point at infinity), rotated into the camera frame without
/// translation and projected as a ray.
fn project(view: &ProjectedImage<'_>, p: &Point3<f64>, w: f64) -> Option<(f64, f64)> {
    let pc = view.cam_from_world.transform_point_homogeneous(p.coords, w);
    if pc.z <= 0.0 {
        return None;
    }
    let (px, py) = view.camera.ray_to_pixel([pc.x, pc.y, pc.z])?;
    let (iw, ih) = (view.camera.width as f64, view.camera.height as f64);
    (px >= 0.0 && py >= 0.0 && px < iw && py < ih).then_some((px, py))
}

/// The patch centre re-anchored on the plane by an in-plane offset `(au, av)` in
/// patch-grid px: `X_p + au·wpp_u·û + av·wpp_v·v̂`.
pub(super) fn shifted_center(
    patch: &OrientedPatch,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
) -> Point3<f64> {
    patch.center + patch.u_axis * (au * wpp_u) + patch.v_axis * (av * wpp_v)
}

/// Render one view's context tile with the patch centre at in-plane offset
/// `(au, av)` (patch-grid px). The context patch spans `context_res / R` times
/// the core extent, rendered at `context_res`, so each context pixel equals one
/// core pixel in world units and the scored core sits centred at offset
/// `margin = (context_res − R) / 2`.
#[allow(clippy::too_many_arguments)]
fn render_context(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    context_res: u32,
    sampler: Sampler,
) -> ContextTile {
    let center = shifted_center(patch, au, av, wpp_u, wpp_v);
    let scale = context_res as f64 / resolution as f64;
    let mut ctx_patch = OrientedPatch::from_center_normal(
        center,
        patch.normal(),
        patch.u_axis,
        [patch.half_extent[0] * scale, patch.half_extent[1] * scale],
    );
    // Preserve the anchor's homogeneous weight so a point at infinity renders as
    // a direction patch (corners are directions), not a finite surfel.
    ctx_patch.w = patch.w;
    let mut map = WarpMap::from_patch(&ctx_patch, view.camera, view.cam_from_world, context_res);
    let img = match sampler {
        Sampler::Anisotropic => {
            map.compute_svd();
            remap_aniso_with_pyramid(view.pyramid, &map, MAX_ANISOTROPY)
        }
        Sampler::Bilinear => remap_bilinear(view.pyramid.level(0), &map),
    };
    let cr = context_res as usize;
    let channels = img.channels() as usize;
    let mut px = vec![0f32; cr * cr * channels];
    let mut valid = vec![false; cr * cr];
    for row in 0..context_res {
        for col in 0..context_res {
            let p = row as usize * cr + col as usize;
            valid[p] = map.is_valid(col, row);
            let base = p * channels;
            for ch in 0..channels {
                px[base + ch] = img.get_pixel(col, row, ch as u32) as f32;
            }
        }
    }
    ContextTile {
        px,
        valid,
        channels,
    }
}

/// Extract the raw (un-normalized) core of `tile` at window offset `(oy, ox)`
/// into `out`, flat `[channel * n + support_index]`. Returns `false` (leaving
/// `out` untouched) when any support pixel is invalid (out of frame) — the slid
/// core then can't be scored.
fn extract_core(
    tile: &ContextTile,
    support: &Support,
    context_res: usize,
    resolution: usize,
    oy: usize,
    ox: usize,
    out: &mut [f32],
) -> bool {
    let ch = tile.channels;
    let n = support.pixels.len();
    for (k, &p) in support.pixels.iter().enumerate() {
        let (r, c) = (p / resolution, p % resolution);
        let cp = (oy + r) * context_res + (ox + c);
        if !tile.valid[cp] {
            return false;
        }
        let base = cp * ch;
        for cc in 0..ch {
            out[cc * n + k] = tile.px[base + cc];
        }
    }
    true
}

/// z-normalize a raw core (`raw[channel * n + k]`) over the kept original
/// channels into `out` (compacted, `out[kept_c * n + k]`), folding `√w` in so a
/// plain dot is a windowed ZNCC. A kept channel that is flat in this core
/// (windowed norm² below [`FLAT_NORM_SQ_EPS`]) is written as zeros (contributes
/// `0` to the ZNCC rather than a misaligned dot), matching view selection.
///
/// Production [`search_shift`] folds this z-normalization into its correlation
/// maps; this remains the reference the equivalence test scores against.
#[cfg(test)]
fn znorm_core(raw: &[f32], support: &Support, keep_mask: &[bool], out: &mut [f32]) {
    let n = support.pixels.len();
    let mut kc = 0;
    for (c, &keep) in keep_mask.iter().enumerate() {
        if !keep {
            continue;
        }
        let col = &raw[c * n..][..n];
        let (s1, s2) = weighted_moments_pub(col, &support.weights);
        let mean = (s1 / support.total_w) as f32;
        let norm_sq = s2 - s1 * (mean as f64);
        let dst = &mut out[kc * n..][..n];
        if norm_sq < FLAT_NORM_SQ_EPS {
            dst.fill(0.0);
        } else {
            let inv = (1.0 / norm_sq.sqrt()) as f32;
            for (d, (&x, &sw)) in dst.iter_mut().zip(col.iter().zip(&support.sqrt_w)) {
                *d = sw * (x - mean) * inv;
            }
        }
        kc += 1;
    }
}

/// Build the unit-norm-per-channel template of a z-normalized stack `xs`
/// (`xs[(v*channels + c)*n + k]`) weighted by `weights` into `out` (resized and
/// overwritten). The result is directly dot-able against another z-normalized core
/// to yield a per-channel ZNCC. `out` is a reused scratch buffer (this runs once
/// per view per round, inside a per-point rayon batch), mirroring the
/// scratch-reuse discipline of [`ConsensusScratch`].
fn weighted_unit_template_into(
    xs: &[f32],
    weights: &[f64],
    views: usize,
    channels: usize,
    n: usize,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(channels * n, 0.0);
    for (v, &w) in weights.iter().enumerate().take(views) {
        let wv = w as f32;
        for c in 0..channels {
            let src = &xs[(v * channels + c) * n..][..n];
            let dst = &mut out[c * n..][..n];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += wv * s;
            }
        }
    }
    for c in 0..channels {
        let col = &mut out[c * n..][..n];
        let norm = col
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        if norm > 1e-12 {
            let inv = (1.0 / norm) as f32;
            for x in col.iter_mut() {
                *x *= inv;
            }
        }
    }
}

/// Per-channel-averaged ZNCC of a z-normalized core against a unit-norm template
/// (both laid out `[c * n + k]`). Reference scoring for the equivalence test;
/// production [`search_shift`] computes the same value by accumulation.
#[cfg(test)]
fn template_zncc(core: &[f32], tmpl: &[f32], channels: usize, n: usize) -> f64 {
    let mut s = 0.0;
    for c in 0..channels {
        let a = &core[c * n..][..n];
        let b = &tmpl[c * n..][..n];
        s += a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum::<f64>();
    }
    s / channels as f64
}

/// Sub-sample peak offset in `[-1, 1]` from a 3-point parabola (scores at `-1`,
/// `0`, `+1` around an integer maximum).
fn parabolic(mid: f64, left: f64, right: f64) -> f64 {
    let denom = left - 2.0 * mid + right;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (0.5 * (left - right) / denom).clamp(-1.0, 1.0)
}

/// Median of `values` (sorted in place). `NaN` for an empty slice.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = values.len() / 2;
    if values.len() % 2 == 1 {
        values[m]
    } else {
        0.5 * (values[m - 1] + values[m])
    }
}

/// Reused per-call scratch for [`search_shift`], created once per
/// [`localize_patch_keypoints`] and shared across every view and round (the
/// search then allocates nothing after warm-up), mirroring [`ConsensusScratch`].
#[derive(Default)]
struct SearchScratch {
    /// The leave-one-out template the candidates score against (`kept_ch · n`),
    /// laid out `[c · n + k]`; the caller writes it each round.
    tmpl: Vec<f32>,
    /// Kept tile-channel indices (the `c` where `keep_mask[c]`), len `channels`.
    kept_ch_idx: Vec<usize>,
    /// The context tile deinterleaved into planar kept channels —
    /// `[kc · (cr·cr) + row·cr + col]` — the SIMD-friendly base (rendered once
    /// per round) the whole shift grid is scored against, mirroring the fronto
    /// cache's planar layout.
    planes: Vec<f32>,
    /// Per-tile-pixel invalidity (`1.0` out of frame, else `0.0`), `[row·cr+col]`.
    invalid: Vec<f32>,
    /// Per-support-pixel kernel `√w · tmpl` for the channel being accumulated.
    kern: Vec<f64>,
    /// Per-channel correlation maps over the shift grid (`(2·margin+1)²`): the
    /// numerator `Σ kern·I` and the window moments `Σ w·I`, `Σ w·I²`.
    g_n: Vec<f64>,
    g_s1: Vec<f64>,
    g_s2: Vec<f64>,
    /// Per-shift count of out-of-frame support pixels (a shift is scorable iff 0).
    ginv: Vec<f64>,
    /// Combined ZNCC grid over the `±margin` window (`(2·margin+1)²`).
    grid: Vec<f64>,
}

/// Integer windowed-ZNCC translation search of view `v`'s context tile against
/// `sc.tmpl`, refined to sub-pixel by a separable parabolic fit. Returns
/// `(δx, δy, peak_zncc)` — the residual shift in patch-grid px and the ZNCC at
/// the integer peak — or `None` if no in-frame window position could be scored.
/// `margin` is the base window offset (`(context_res − R) / 2`); the search slides
/// the window over `±margin`.
///
/// Rather than re-extract + z-normalize + dot each of the `(2·margin+1)²`
/// candidates (a strided gather and a horizontal reduction per candidate), the
/// whole grid is scored by accumulation. Because the template carries a fixed
/// weighted mean, the windowed ZNCC of every shift factors into three
/// correlation maps per channel — `Σ kern·I`, `Σ w·I`, `Σ w·I²` — whose inner
/// loop is a contiguous fused SAXPY across the shift row (the vectorizable core).
/// `zncc = (Σkern·I − mean·Σ√w·tmpl) / √(Σw·I² − mean·Σw·I)`, averaged over
/// channels — algebraically identical to the per-candidate z-normalize-then-dot
/// path (see [`search_shift_ref`]).
#[allow(clippy::too_many_arguments)]
fn search_shift(
    tile: &ContextTile,
    sc: &mut SearchScratch,
    support: &Support,
    keep_mask: &[bool],
    channels: usize,
    context_res: usize,
    resolution: usize,
    margin: i64,
) -> Option<(f64, f64, f64)> {
    let n = support.pixels.len();
    let cr = context_res;
    let span = (2 * margin + 1) as usize;
    let gsz = span * span;
    let tc = tile.channels;
    let plane_len = cr * cr;

    // The kept tile-channel indices (where `keep_mask` is set).
    sc.kept_ch_idx.clear();
    sc.kept_ch_idx
        .extend((0..keep_mask.len()).filter(|&c| keep_mask[c]));
    debug_assert_eq!(sc.kept_ch_idx.len(), channels);

    // Deinterleave the kept channels into planar f32, plus the per-pixel
    // invalidity plane — the base the shift grid scores against.
    sc.planes.clear();
    sc.planes.resize(channels * plane_len, 0.0);
    sc.invalid.clear();
    sc.invalid.resize(plane_len, 0.0);
    for i in 0..plane_len {
        if !tile.valid[i] {
            sc.invalid[i] = 1.0;
        }
        let base = i * tc;
        for (kc, &c) in sc.kept_ch_idx.iter().enumerate() {
            sc.planes[kc * plane_len + i] = tile.px[base + c];
        }
    }

    // Validity: count out-of-frame support pixels per shift (channel-independent);
    // a shift with any is unscorable, matching `extract_core`'s all-valid gate.
    sc.ginv.clear();
    sc.ginv.resize(gsz, 0.0);
    accumulate_count(&sc.invalid, support, resolution, cr, span, &mut sc.ginv);

    // Per kept channel: accumulate the three correlation maps, then fold the
    // channel's ZNCC into the combined grid. The numerator's mean term is carried
    // explicitly (`mean·Σ√w·tmpl`) so this matches the per-candidate path even
    // when the template is not exactly zero-weighted-mean.
    sc.grid.clear();
    sc.grid.resize(gsz, 0.0);
    sc.g_n.resize(gsz, 0.0);
    sc.g_s1.resize(gsz, 0.0);
    sc.g_s2.resize(gsz, 0.0);
    let inv_total_w = 1.0 / support.total_w;
    for kc in 0..channels {
        let tmpl_c = &sc.tmpl[kc * n..][..n];
        sc.kern.clear();
        let mut tsum = 0.0;
        for (&sw, &t) in support.sqrt_w.iter().zip(tmpl_c) {
            let kk = sw as f64 * t as f64;
            sc.kern.push(kk);
            tsum += kk;
        }
        sc.g_n.fill(0.0);
        sc.g_s1.fill(0.0);
        sc.g_s2.fill(0.0);
        accumulate_channel(
            &sc.planes[kc * plane_len..][..plane_len],
            support,
            &sc.kern,
            resolution,
            cr,
            span,
            &mut sc.g_n,
            &mut sc.g_s1,
            &mut sc.g_s2,
        );
        for s in 0..gsz {
            let s1 = sc.g_s1[s];
            let mean = s1 * inv_total_w;
            let norm_sq = sc.g_s2[s] - s1 * mean;
            // A channel flat in this window contributes 0 (matches `znorm_core`).
            if norm_sq >= FLAT_NORM_SQ_EPS {
                sc.grid[s] += (sc.g_n[s] - mean * tsum) / norm_sq.sqrt();
            }
        }
    }

    // Average over channels; out-of-frame shifts score −∞ (never chosen).
    let chf = channels as f64;
    let at =
        |dy: i64, dx: i64| -> usize { ((dy + margin) as usize) * span + (dx + margin) as usize };
    let mut best = (f64::NEG_INFINITY, 0i64, 0i64);
    for dy in -margin..=margin {
        for dx in -margin..=margin {
            let s = at(dy, dx);
            let z = if sc.ginv[s] > 0.5 {
                f64::NEG_INFINITY
            } else {
                sc.grid[s] / chf
            };
            sc.grid[s] = z;
            if z > best.0 {
                best = (z, dy, dx);
            }
        }
    }
    if !best.0.is_finite() {
        return None;
    }
    let (peak, py, px) = best;
    let grid = &sc.grid;
    // Separable parabolic sub-pixel using the integer neighbours, when both are
    // finite (fall back to the integer peak at a frame edge).
    let nb = |dy: i64, dx: i64| -> Option<f64> {
        if dy.abs() <= margin && dx.abs() <= margin {
            let g = grid[at(dy, dx)];
            g.is_finite().then_some(g)
        } else {
            None
        }
    };
    let sy = match (nb(py - 1, px), nb(py + 1, px)) {
        (Some(l), Some(r)) => parabolic(peak, l, r),
        _ => 0.0,
    };
    let sx = match (nb(py, px - 1), nb(py, px + 1)) {
        (Some(l), Some(r)) => parabolic(peak, l, r),
        _ => 0.0,
    };
    Some((px as f64 + sx, py as f64 + sy, peak))
}

/// Accumulate, over the support pixels, one channel's per-shift correlation maps:
/// numerator `g_n[s] = Σ_k kern[k]·I[s+k]` and window moments
/// `g_s1[s] = Σ_k w[k]·I[s+k]`, `g_s2[s] = Σ_k w[k]·I[s+k]²`, where `I` is the
/// channel's planar context tile and `s` runs over the `span×span` shift grid.
/// The inner `gx` loop walks a contiguous tile row and a contiguous grid row (a
/// fused SAXPY) — the vectorizable core of the search.
#[allow(clippy::too_many_arguments)]
fn accumulate_channel(
    plane: &[f32],
    support: &Support,
    kern: &[f64],
    resolution: usize,
    cr: usize,
    span: usize,
    g_n: &mut [f64],
    g_s1: &mut [f64],
    g_s2: &mut [f64],
) {
    for (k, &p) in support.pixels.iter().enumerate() {
        let r_k = p / resolution;
        let c_k = p % resolution;
        let w_k = support.weights[k];
        let kern_k = kern[k];
        for gy in 0..span {
            let src = &plane[(gy + r_k) * cr + c_k..][..span];
            let gbase = gy * span;
            let on = &mut g_n[gbase..][..span];
            let o1 = &mut g_s1[gbase..][..span];
            let o2 = &mut g_s2[gbase..][..span];
            for gx in 0..span {
                let v = src[gx] as f64;
                on[gx] += kern_k * v;
                o1[gx] += w_k * v;
                o2[gx] += w_k * v * v;
            }
        }
    }
}

/// Accumulate the per-shift count of out-of-frame support pixels (`invalid` is
/// `1.0` where the tile pixel is invalid, else `0.0`). A shift is scorable iff its
/// count is `0` (every support pixel in frame), the grid analogue of
/// [`extract_core`] returning `false` for any invalid support pixel.
fn accumulate_count(
    invalid: &[f32],
    support: &Support,
    resolution: usize,
    cr: usize,
    span: usize,
    g: &mut [f64],
) {
    for &p in &support.pixels {
        let r_k = p / resolution;
        let c_k = p % resolution;
        for gy in 0..span {
            let src = &invalid[(gy + r_k) * cr + c_k..][..span];
            let gr = &mut g[gy * span..][..span];
            for gx in 0..span {
                gr[gx] += src[gx] as f64;
            }
        }
    }
}

/// One view's mutable congealing state.
struct ViewState {
    /// Image index into the caller's `views` slice.
    idx: u32,
    /// Accumulated in-plane offset `(au, av)` of the patch centre, patch-grid px.
    acc: [f64; 2],
    /// The view's projection of the point `project_i(X_p)`, source-image px — the
    /// anchor the keypoint and its `max_shift_px` gate are measured from.
    proj: [f64; 2],
    /// The latest leave-one-out ZNCC (peak from the round's search); `NaN` until
    /// a round scores it.
    loo: f64,
}

/// Localize the keypoints of one oriented patch over a view set by congealing.
///
/// `views` is one [`ProjectedImage`] per reconstruction image (indexed by image
/// index); `view_set` lists the views to refine (the output of
/// [view selection](super::view_selection)). `starting_keypoints`, when given, is
/// one seed per `view_set` entry (source-image px); `None` seeds every view at the
/// point's own projection `project_i(X_p)`. Returns the kept views and their
/// refined keypoints; see [`KeypointLocalization`] and
/// `specs/core/patch-keypoint-localization.md`.
pub fn localize_patch_keypoints(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    starting_keypoints: Option<&[[f64; 2]]>,
    params: &KeypointLocalizeParams,
) -> KeypointLocalization {
    let resolution = params.resolution.max(2);
    let margin = params.search.ceil().max(1.0) as i64;
    let context_res = resolution + 2 * margin as u32;
    let r = resolution as usize;
    let cr = context_res as usize;

    let wpp_u = 2.0 * patch.half_extent[0] / resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / resolution as f64;

    // Window support over the R×R core.
    let w_full = window_weights(params.window, resolution);
    let mut pixels = Vec::new();
    let mut weights = Vec::new();
    for (p, &w) in w_full.iter().enumerate() {
        if w > 0.0 {
            pixels.push(p);
            weights.push(w);
        }
    }
    let total_w: f64 = weights.iter().sum();
    let sqrt_w: Vec<f32> = weights.iter().map(|&w| w.sqrt() as f32).collect();
    let support = Support {
        pixels,
        weights,
        sqrt_w,
        total_w,
    };

    // Dedup the view set order-preserving (a point can carry two observations in
    // one image; refining it twice double-weights that view in the consensus).
    let mut seen = std::collections::HashSet::new();
    let mut states: Vec<ViewState> = Vec::new();
    let normal = patch.normal();
    for (k, &i) in view_set.iter().enumerate() {
        if !seen.insert(i) {
            continue;
        }
        let view = &views[i as usize];
        // Grazing pre-filter: drop a view whose ray is near-parallel to the plane.
        // The viewing direction is camera→point: `center − cam_c` for a finite
        // point, or the direction `d` itself for a point at infinity (every ray to
        // it is parallel to `d`, so it is always fully frontal — cos = 1).
        let d = if patch.w == 0.0 {
            patch.center.coords
        } else {
            patch.center - view.cam_from_world.inverse_translation_origin()
        };
        let dn = d.norm();
        if dn <= 1e-12 || (d.dot(&normal) / dn).abs() < params.min_grazing_cos {
            continue;
        }
        // The point's own projection (the keypoint anchor). A view that can't
        // project the point in-frame can't be refined; skip it.
        let Some(proj) = project(view, &patch.center, patch.w) else {
            continue;
        };
        // Seed offset: unproject the starting keypoint onto the plane, else zero.
        let acc = match starting_keypoints {
            Some(seeds) => seed_offset(patch, view, seeds[k], wpp_u, wpp_v).unwrap_or([0.0, 0.0]),
            None => [0.0, 0.0],
        };
        states.push(ViewState {
            idx: i,
            acc,
            proj: [proj.0, proj.1],
            loo: f64::NAN,
        });
    }

    if states.len() < 2 {
        return finalize(patch, views, &states, wpp_u, wpp_v);
    }

    let mut sc = ConsensusScratch::default();
    let mut search = SearchScratch::default();
    for _round in 0..params.max_iters.max(1) {
        prof::count(&prof::N_ROUNDS, 1);
        // 1. Render every view's context tile at its accumulated offset; a view
        //    whose core has left the frame (any support pixel invalid) is dropped.
        let mut tiles: Vec<ContextTile> = Vec::with_capacity(states.len());
        let mut live: Vec<usize> = Vec::with_capacity(states.len());
        let mut base_raw = Vec::new();
        prof::count(&prof::N_RENDER, states.len() as u64);
        prof::RENDER.time(|| {
            for (si, st) in states.iter().enumerate() {
                let tile = render_context(
                    patch,
                    &views[st.idx as usize],
                    st.acc[0],
                    st.acc[1],
                    wpp_u,
                    wpp_v,
                    resolution,
                    context_res,
                    params.sampler,
                );
                let mut raw = vec![0f32; tile.channels * support.pixels.len()];
                if extract_core(
                    &tile,
                    &support,
                    cr,
                    r,
                    margin as usize,
                    margin as usize,
                    &mut raw,
                ) {
                    live.push(si);
                    base_raw.push(raw);
                    tiles.push(tile);
                }
            }
        });
        if live.len() < 2 {
            // Too few views still see the patch to congeal; keep the in-frame ones
            // (with their current offsets) and finalize.
            let live_set: std::collections::HashSet<u32> =
                live.iter().map(|&si| states[si].idx).collect();
            states.retain(|st| live_set.contains(&st.idx));
            break;
        }

        // 2. z-normalize the live cores into a shared compacted channel space.
        let channels0 = tiles.iter().map(|t| t.channels).min().unwrap();
        let n = support.pixels.len();
        let mut flat = vec![0f32; live.len() * channels0 * n];
        for (vk, raw) in base_raw.iter().enumerate() {
            flat[vk * channels0 * n..][..channels0 * n].copy_from_slice(&raw[..channels0 * n]);
        }
        let mut xs = Vec::new();
        let znorm = prof::ZNORM.time(|| {
            znormalize_into_kept(
                &flat,
                live.len(),
                channels0,
                n,
                &support.weights,
                support.total_w,
                &support.sqrt_w,
                &mut xs,
            )
        });
        let Some((kept_ch, keep_mask)) = znorm else {
            break; // no textured channel — leave the seeds in place
        };

        // 3-4. Per live view: search its residual shift against the leave-one-out
        //      consensus of the others, then accumulate (clipped to ±search).
        let mut deltas = vec![[0.0f64; 2]; live.len()];
        let mut loos = vec![f64::NAN; live.len()];
        let mut loo_xs = vec![0f32; (live.len() - 1) * kept_ch * n];
        for v in 0..live.len() {
            // Copy the other views' rows contiguously and build their robust
            // consensus template (the leave-one-out reference for view v).
            prof::TEMPLATE.time(|| {
                let mut w = 0;
                for u in 0..live.len() {
                    if u == v {
                        continue;
                    }
                    loo_xs[w * kept_ch * n..][..kept_ch * n]
                        .copy_from_slice(&xs[u * kept_ch * n..][..kept_ch * n]);
                    w += 1;
                }
                irls_view_weights(&loo_xs, w, kept_ch, n, params.robust_iters, &mut sc);
                weighted_unit_template_into(&loo_xs, &sc.w, w, kept_ch, n, &mut search.tmpl);
            });
            prof::count(&prof::N_SEARCH, 1);
            let shift = prof::SEARCH.time(|| {
                search_shift(
                    &tiles[v],
                    &mut search,
                    &support,
                    &keep_mask,
                    kept_ch,
                    cr,
                    r,
                    margin,
                )
            });
            if let Some((dx, dy, peak)) = shift {
                deltas[v] = [dx, dy];
                loos[v] = peak;
            }
        }

        // Accumulate onto the (still full) `states`, clipping the total drift.
        let mut shift_sum = 0.0;
        for (v, &si) in live.iter().enumerate() {
            let st = &mut states[si];
            st.acc[0] = (st.acc[0] + deltas[v][0]).clamp(-params.search, params.search);
            st.acc[1] = (st.acc[1] + deltas[v][1]).clamp(-params.search, params.search);
            st.loo = loos[v];
            shift_sum += deltas[v][0].hypot(deltas[v][1]);
        }

        // 5. Drop failing views (out-of-frame already removed above): keypoint too
        //    far from the projection, or leave-one-out ZNCC below the relative bar.
        //    Stop dropping once only two views (the leave-one-out floor) remain.
        let mut live_loo: Vec<f64> = live
            .iter()
            .map(|&si| states[si].loo)
            .filter(|z| z.is_finite())
            .collect();
        let med = median(&mut live_loo);
        let bar = if med.is_finite() {
            params.min_relative_zncc * med
        } else {
            f64::NEG_INFINITY
        };
        let live_idx: std::collections::HashSet<u32> =
            live.iter().map(|&si| states[si].idx).collect();
        // Keep only live views (drops out-of-frame from this round) that also pass
        // the shift / agreement gates; guarantee at least the top-two by ZNCC.
        let mut kept: Vec<usize> = Vec::new();
        let mut fallback: Vec<(f64, usize)> = Vec::new();
        for (si, st) in states.iter().enumerate() {
            if !live_idx.contains(&st.idx) {
                continue;
            }
            let center = shifted_center(patch, st.acc[0], st.acc[1], wpp_u, wpp_v);
            let shift_px = match project(&views[st.idx as usize], &center, patch.w) {
                Some((x, y)) => (x - st.proj[0]).hypot(y - st.proj[1]),
                None => f64::INFINITY, // keypoint left the frame
            };
            // `shift_px <= max_shift_px` already rejects the out-of-frame INFINITY.
            let ok = shift_px <= params.max_shift_px && st.loo.is_finite() && st.loo >= bar;
            if ok {
                kept.push(si);
            }
            let rank = if st.loo.is_finite() { st.loo } else { -1.0 };
            fallback.push((rank, si));
        }
        if kept.len() < 2 {
            // Honor the leave-one-out floor: retain the two best-agreeing views.
            fallback.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            kept = fallback.iter().take(2).map(|&(_, si)| si).collect();
            kept.sort_unstable();
        }
        let keep_set: std::collections::HashSet<usize> = kept.into_iter().collect();
        let mut si = 0;
        states.retain(|_| {
            let keep = keep_set.contains(&si);
            si += 1;
            keep
        });

        // 6. Converge.
        let mean_shift = shift_sum / live.len() as f64;
        if states.len() < 2 || mean_shift < params.convergence_px {
            break;
        }
    }

    finalize(patch, views, &states, wpp_u, wpp_v)
}

/// Unproject a starting keypoint onto the patch plane and express the in-plane
/// offset of its hit point (from the patch centre) in patch-grid px. `None` when
/// the ray is parallel to the plane.
pub(super) fn seed_offset(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: [f64; 2],
    wpp_u: f64,
    wpp_v: f64,
) -> Option<[f64; 2]> {
    let ray_cam = view.camera.pixel_to_ray(keypoint[0], keypoint[1]);
    let r = view.cam_from_world.to_rotation_matrix();
    // World ray direction: R^T · ray_cam (camera-to-world rotation).
    let dir = r.transpose() * Vector3::new(ray_cam[0], ray_cam[1], ray_cam[2]);
    // A zero patch extent would make `wpp` zero; guard so a degenerate patch seeds
    // at the projection (`acc = 0`) rather than propagating a NaN/inf offset.
    if wpp_u <= 0.0 || wpp_v <= 0.0 {
        return None;
    }
    let off = if patch.w == 0.0 {
        // Point at infinity: `center` is the unit direction `d`, the patch corner
        // `d + a·û + b·v̂` is a direction, and the observed ray is parallel to it:
        // `dir ∝ d + a·û + b·v̂`. With `û, v̂ ⊥ d`, `a = (dir·û)/(dir·d)` and
        // `b = (dir·v̂)/(dir·d)`. `dir·d ≤ 0` means the ray points away from `d`.
        let d = patch.center.coords;
        let denom = dir.dot(&d);
        if denom <= 1e-12 {
            return None;
        }
        patch.u_axis * (dir.dot(&patch.u_axis) / denom)
            + patch.v_axis * (dir.dot(&patch.v_axis) / denom)
    } else {
        // Finite point: intersect the ray with the patch plane and offset from the
        // centre.
        let cam_c = view.cam_from_world.inverse_translation_origin();
        let n = patch.normal();
        let denom = dir.dot(&n);
        if denom.abs() < 1e-12 {
            return None;
        }
        let s = (patch.center - cam_c).dot(&n) / denom;
        let hit = cam_c + dir * s;
        hit - patch.center
    };
    Some([
        off.dot(&patch.u_axis) / wpp_u,
        off.dot(&patch.v_axis) / wpp_v,
    ])
}

/// Build the result from the final view states: the refined keypoint
/// `project_i(center_v)`, its offset from the projection, and the last
/// leave-one-out ZNCC, per surviving view.
fn finalize(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: &[ViewState],
    wpp_u: f64,
    wpp_v: f64,
) -> KeypointLocalization {
    let mut out = KeypointLocalization::default();
    for st in states {
        let view = &views[st.idx as usize];
        let center = shifted_center(patch, st.acc[0], st.acc[1], wpp_u, wpp_v);
        // The refined keypoint is the projection of the re-anchored centre. Fall
        // back to the point's own projection if the shifted centre fails to project.
        let (kx, ky) = project(view, &center, patch.w).unwrap_or((st.proj[0], st.proj[1]));
        out.views.push(st.idx);
        out.keypoints.push([kx, ky]);
        out.offsets_px
            .push((kx - st.proj[0]).hypot(ky - st.proj[1]));
        out.loo_zncc.push(st.loo);
    }
    out
}

/// Batch [`localize_patch_keypoints`] over a [`PatchCloud`], parallel across
/// patches (rayon). `view_sets[i]` lists, for patch `i`, the views to refine
/// (typically the output of view selection). `starting_keypoints`, when given, is
/// parallel to `view_sets` (one seed per view); `None` seeds every view at the
/// point's projection. Results are returned in cloud order.
///
/// # Panics
///
/// Panics if `view_sets.len() != cloud.len()` (or `starting_keypoints` is given
/// and not parallel), or an index is out of range.
pub fn localize_patch_cloud_keypoints(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    view_sets: &[Vec<u32>],
    starting_keypoints: Option<&[Vec<[f64; 2]>]>,
    params: &KeypointLocalizeParams,
) -> Vec<KeypointLocalization> {
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
    if prof::enabled() {
        prof::reset();
    }
    let wall_start = std::time::Instant::now();
    let out: Vec<KeypointLocalization> = cloud
        .patches
        .par_iter()
        .enumerate()
        .map(|(i, patch)| {
            let seeds = starting_keypoints.map(|s| s[i].as_slice());
            prof::TOTAL
                .time(|| localize_patch_keypoints(patch, views, &view_sets[i], seeds, params))
        })
        .collect();
    if prof::enabled() {
        prof::report(cloud.len(), wall_start.elapsed().as_secs_f64());
    }
    out
}

/// For each patch of `cloud` (linked to `recon` via `point_ids`), the track image
/// indices observing its source 3D point — a convenience default `view_sets` for
/// [`localize_patch_cloud_keypoints`] when no view selection has been run.
/// Identical to
/// [`view_indices_from_reconstruction`](super::normal_refine::view_indices_from_reconstruction).
///
/// # Panics
///
/// Panics if `cloud.point_ids` is not parallel to its patches.
pub fn track_views_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<u32>> {
    super::normal_refine::view_indices_from_reconstruction(recon, cloud)
}

/// Reference (pre-optimization) translation search: score each candidate by
/// extract → z-normalize → template dot. Kept as the oracle the accumulation
/// [`search_shift`] is checked against in [`tests`]; not used in production.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn search_shift_ref(
    tile: &ContextTile,
    tmpl: &[f32],
    support: &Support,
    keep_mask: &[bool],
    channels: usize,
    context_res: usize,
    resolution: usize,
    margin: i64,
) -> Option<(f64, f64, f64)> {
    let n = support.pixels.len();
    let mut raw = vec![0f32; tile.channels * n];
    let mut core = vec![0f32; channels * n];
    let span = (2 * margin + 1) as usize;
    let mut grid = vec![f64::NEG_INFINITY; span * span];
    let at =
        |dy: i64, dx: i64| -> usize { ((dy + margin) as usize) * span + (dx + margin) as usize };
    let mut best = (f64::NEG_INFINITY, 0i64, 0i64);
    for dy in -margin..=margin {
        for dx in -margin..=margin {
            let oy = (margin + dy) as usize;
            let ox = (margin + dx) as usize;
            if !extract_core(tile, support, context_res, resolution, oy, ox, &mut raw) {
                continue;
            }
            znorm_core(&raw, support, keep_mask, &mut core);
            let z = template_zncc(&core, tmpl, channels, n);
            grid[at(dy, dx)] = z;
            if z > best.0 {
                best = (z, dy, dx);
            }
        }
    }
    if !best.0.is_finite() {
        return None;
    }
    let (peak, py, px) = best;
    let nb = |dy: i64, dx: i64| -> Option<f64> {
        if dy.abs() <= margin && dx.abs() <= margin {
            let g = grid[at(dy, dx)];
            g.is_finite().then_some(g)
        } else {
            None
        }
    };
    let sy = match (nb(py - 1, px), nb(py + 1, px)) {
        (Some(l), Some(r)) => parabolic(peak, l, r),
        _ => 0.0,
    };
    let sx = match (nb(py, px - 1), nb(py, px + 1)) {
        (Some(l), Some(r)) => parabolic(peak, l, r),
        _ => 0.0,
    };
    Some((px as f64 + sx, py as f64 + sy, peak))
}

#[cfg(test)]
mod tests;
