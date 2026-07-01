// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch-keypoint localization by group-wise translation registration
//! (congealing).
//!
//! See `specs/core/patch-keypoint-localization.md` and
//! `specs/core/keypoint-localization-search-cache.md`. Given one 3D point with
//! its oriented patch, a view set `G`, and a starting keypoint per view,
//! [`localize_patch_keypoints`] refines each keypoint to sub-pixel and reports
//! which views it kept. The patch frame is fixed during localization, so each
//! view's source is resampled into an expanded, frame-oriented **cache exactly
//! once** (sized to cover the whole search drift); since an integer in-plane
//! shift is an integer cache-index shift, reading the cache at an integer offset
//! is bit-identical to re-warping the patch there. Each round then reads every
//! view's core from its cache at the view's current **integer** offset (no
//! render), builds the robust (IRLS) consensus, and searches each view's residual
//! in-plane shift against the **leave-one-out** consensus of the *others* (so a
//! view is never aligned to a template its own pixels polluted). The integer
//! argmax moves the cache-read accumulator while the parabolic sub-pixel residual
//! rides alongside (it gates convergence and seeds the final keypoint but never
//! moves the read position — keeping every read exact). Views that drift too far,
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
    build_support, irls_view_weights, weighted_unit_template_into, znormalize_into_kept,
    ConsensusScratch, PatchWindow, ProjectedImage, Sampler, Support, FLAT_NORM_SQ_EPS,
};
// Only the reference scorer (`znorm_core`, test-only) needs the moment helper.
#[cfg(test)]
use crate::patch::normal_refine::weighted_moments_pub;
use crate::reconstruction::SfmrReconstruction;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// `remap_aniso` sample cap along the major axis (mirrors `normal_refine`).
const MAX_ANISOTROPY: u32 = 16;

/// How the per-(view, round) shift grid is traversed inside `search_shift`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchStrategy {
    /// "+"-descent on the integer shift grid: starts at `(dy, dx) = (0, 0)`,
    /// evaluates the 4 axis neighbors per step, moves to the best improver,
    /// and stops when no neighbor beats the current cell. Each cell is scored
    /// at most once via a per-cell ZNCC kernel
    /// ([`score_cell_one_channel`] — AVX2-gather when available, scalar
    /// otherwise); the visited cache stores `(n, s1, s2, ginv)` per cell. The
    /// final-cell separable parabolic sub-pixel fit reuses the 4
    /// cardinal-neighbor cells already in the cache (each was evaluated to
    /// discover the STOP condition).
    ///
    /// **The default.** Late-round congealing typically leaves a view's
    /// argmax at `(0, 0)`, so the descent stops after 5 cell scores; on the
    /// dino dataset this drives per-`search_shift` cost from ~145 µs
    /// ([`Exhaustive`](Self::Exhaustive)) to ~32 µs and total localize wall
    /// down ~1.9× at comparable accuracy (median per-observation keypoint
    /// shift vs `Exhaustive` is ~0.05 px, 91 % of observations within 1 px on
    /// dino). The accuracy tail is the local-optima failure mode of any
    /// descent on a multi-modal ZNCC landscape; pick
    /// [`Exhaustive`](Self::Exhaustive) when the tail matters or for
    /// bit-equivalent comparisons.
    #[default]
    PlusDescent,
    /// Score every cell of the `(2·margin+1) × (2·margin+1)` shift grid via
    /// the hand-rolled SIMD SAXPY accumulator (`compute_channel_grids`), then
    /// argmax + separable parabolic. The original whole-grid path; retained
    /// as the global-argmax fallback (no local-optima risk) and as the
    /// per-cell reference both equivalence tests and the per-cell ZNCC kernel
    /// (`score_cell_one_channel`) check against.
    Exhaustive,
}

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
    /// Search-resolution multiplier `m` (default `1.0`, a no-op). The discrete
    /// cross-view search is run at resolution `R_s = round(m·R)`: the per-view
    /// cache, window support, and shift grid are all built at `R_s`, so one
    /// integer step in the search grid is `1/m` patch-grid px. The found shift is
    /// scaled by `1/m` back to patch-grid px before it moves the accumulator and
    /// is reported. `m < 1` smooths the correlation surface for a speed fallback;
    /// `m > 1` resolves sub-pixel offsets directly on a finer grid. See
    /// `specs/core/keypoint-localization-search-cache.md`.
    pub search_resolution_multiplier: f32,
    /// Per-(view, round) shift-grid traversal — see [`SearchStrategy`].
    /// Defaults to [`SearchStrategy::PlusDescent`].
    pub search_strategy: SearchStrategy,
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
            search_resolution_multiplier: 1.0,
            search_strategy: SearchStrategy::PlusDescent,
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

/// A rendered context tile for one view: source colour over a
/// `cache_res × cache_res` grid (larger than the scored `R×R` core so the shift
/// search can slide), plus per-pixel validity from the warp map (an invalid pixel
/// is out of frame, rendered black, and must not be scored).
///
/// In the cached congealing loop this is the **per-view render-once cache**: it
/// is rendered a single time per view, frame-oriented at the seed (`acc = 0`,
/// centred on `project_i(X_p)`), sized to cover the full search drift, and every
/// round reads its core / scores its shift grid from it at the view's current
/// **integer** offset. Because the patch frame is fixed during localization, an
/// integer in-plane shift is an integer cache-index shift, so a read at an
/// integer offset is bit-identical to re-warping the patch at that offset (see
/// `specs/core/keypoint-localization-search-cache.md`).
///
/// **Layout (stage 1 of the SIMD search kernel).** The cache is **planar per
/// channel** in **centered `f32`**: `planes[c][row · istride + col]` holds
/// `I − means[c]` (the channel mean over the cache). Centering is load-bearing —
/// the windowed-ZNCC denominator `S2 − S1²/W` is a catastrophic-cancellation trap
/// in `f32` when `I ~ 10²` (`S2 ~ 10⁷`); centering makes `S1 ≈ 0` and `S2 ≈
/// variance · W`, so `f32` accumulation is accurate. The numerator is recovered
/// exactly by `Ncross = Ncross' + mean · Σ kern` — algebraically identical to
/// z-normalize-then-dot for any template. Rows are padded to `istride =
/// align_up(cacheW − 1 + 16, 8)` so a 16-wide aligned `f32` load from any support
/// column stays in bounds; the pad columns hold `0` (= the mean after centering,
/// harmless — they only feed discarded grid cells). The per-pixel invalidity
/// plane (`1.0` out of frame, else `0.0`) lives alongside in the same `istride`
/// row layout for the SIMD validity pass; the `bool` `valid` map is kept too for
/// the integer-tracked consensus core read.
struct ContextTile {
    /// Side length of the (square) tile, in patch-grid px.
    res: usize,
    /// Row stride of each plane in `f32` lanes: `align_up(res − 1 + 16, 8)` so a
    /// 16-wide aligned load from any support column stays in bounds.
    istride: usize,
    /// Channel count.
    channels: usize,
    /// Per-channel mean over the cache (the value subtracted to produce
    /// [`planes`](Self::planes)). Used to recover original-scale values on the
    /// consensus-core read (`extract_core`) and to fold back into the numerator
    /// (`Ncross = Ncross' + mean · Σ kern`).
    means: Vec<f32>,
    /// Centered per-channel planes: `planes[c][row · istride + col] = I_c − means[c]`.
    /// Length `channels`, each plane length `istride · res`.
    planes: Vec<Vec<f32>>,
    /// Per-pixel invalidity plane in the same `istride` row layout (`1.0` invalid,
    /// `0.0` valid). Drives the SIMD validity count pass that gates `−∞` shifts.
    invalid_plane: Vec<f32>,
    /// Per-pixel validity (`true` in frame), `[row · res + col]`. The `bool` form
    /// is what the round-loop consensus-core read (`extract_core`) checks.
    valid: Vec<bool>,
}

/// Compute the row stride for the centered planar cache: enough lanes to admit a
/// 16-wide aligned `f32` load starting at any column in `[0, res)`. `align_up`
/// to 8 lanes (one `__m256`) keeps row starts naturally aligned for AVX2.
#[inline]
fn cache_istride(res: usize) -> usize {
    // The widest load needed is 16 f32s (two `__m256`s) starting at column
    // `res − 1`, which reads through `res − 1 + 15`; the buffer must hold
    // through `res − 1 + 16` (cap). Round up to a multiple of 8.
    let need = res - 1 + 16;
    need.div_ceil(8) * 8
}

/// Project a homogeneous world point `(p, w)` into a view; `None` when it falls
/// behind the camera or outside the frame. `w = 1` is a finite point; `w = 0` is
/// a direction (a point at infinity), rotated into the camera frame without
/// translation and projected as a ray.
pub(super) fn project(view: &ProjectedImage<'_>, p: &Point3<f64>, w: f64) -> Option<(f64, f64)> {
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

/// Render one view's context tile / cache with the patch centre at in-plane
/// offset `(au, av)` (patch-grid px). The context patch spans `context_res / R`
/// times the core extent, rendered at `context_res`, so each context pixel equals
/// one core pixel in world units. In the cached loop this is called **once per
/// view** with `(au, av) = (0, 0)` to build the per-view cache: the scored core
/// at the view's accumulated integer offset `iacc` then sits at cache offset
/// `(context_res − R) / 2 + iacc`.
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
    let mut map = prof::RENDER_PROJECT
        .time(|| WarpMap::from_patch(&ctx_patch, view.camera, view.cam_from_world, context_res));
    if matches!(sampler, Sampler::Anisotropic) {
        prof::RENDER_SVD.time(|| map.compute_svd());
    }
    let img = prof::RENDER_REMAP.time(|| match sampler {
        Sampler::Anisotropic => remap_aniso_with_pyramid(view.pyramid, &map, MAX_ANISOTROPY),
        Sampler::Bilinear => remap_bilinear(view.pyramid.level(0), &map),
    });
    let cr = context_res as usize;
    let channels = img.channels() as usize;
    let istride = cache_istride(cr);

    // Per-channel sum → mean over the cache. We accumulate in `f64` to keep the
    // centering exact to the last `f32` ulp (one mean per channel; cheap).
    let means: Vec<f32> = prof::RENDER_MEAN.time(|| {
        let mut sums = vec![0.0f64; channels];
        let total = (cr * cr) as f64;
        for row in 0..context_res {
            for col in 0..context_res {
                for (ch, slot) in sums.iter_mut().enumerate() {
                    *slot += img.get_pixel(col, row, ch as u32) as f64;
                }
            }
        }
        sums.iter().map(|&s| (s / total) as f32).collect()
    });

    // Centered planar planes. Pad columns past `cr` stay at `0.0` (= the mean
    // after centering, harmless — those columns only feed discarded grid cells
    // past the search window).
    let (planes, invalid_plane, valid) = prof::RENDER_CENTER.time(|| {
        let mut planes: Vec<Vec<f32>> = (0..channels).map(|_| vec![0.0f32; istride * cr]).collect();
        let mut invalid_plane = vec![0.0f32; istride * cr];
        let mut valid = vec![false; cr * cr];
        for row in 0..context_res {
            for col in 0..context_res {
                let r = row as usize;
                let c = col as usize;
                let row_off = r * istride + c;
                let v = map.is_valid(col, row);
                valid[r * cr + c] = v;
                invalid_plane[row_off] = if v { 0.0 } else { 1.0 };
                for ch in 0..channels {
                    let p = img.get_pixel(col, row, ch as u32) as f32;
                    planes[ch][row_off] = p - means[ch];
                }
            }
        }
        (planes, invalid_plane, valid)
    });
    ContextTile {
        res: cr,
        istride,
        channels,
        means,
        planes,
        invalid_plane,
        valid,
    }
}

/// Extract the raw (un-normalized) core of `tile` at window offset `(oy, ox)`
/// into `out`, flat `[channel * n + support_index]`. Returns `false` (leaving
/// `out` untouched) when any support pixel is invalid (out of frame) — the slid
/// core then can't be scored.
fn extract_core(
    tile: &ContextTile,
    support: &Support,
    resolution: usize,
    oy: usize,
    ox: usize,
    out: &mut [f32],
) -> bool {
    let ch = tile.channels;
    let n = support.pixels.len();
    let tile_res = tile.res;
    let istride = tile.istride;
    for (k, &p) in support.pixels.iter().enumerate() {
        let (r, c) = (p / resolution, p % resolution);
        // The `valid` plane stays in the tight `tile_res`-stride layout (it's a
        // per-pixel mask for the consensus core read, not part of the SIMD hot
        // loop); the centered planes live in the padded `istride` layout.
        if !tile.valid[(oy + r) * tile_res + (ox + c)] {
            return false;
        }
        let cp = (oy + r) * istride + (ox + c);
        // Add back the per-channel mean so a core read recovers the original
        // source value (the centered representation is an internal optimization
        // for the SIMD search; `znorm_core` and the rest of the pipeline see
        // the same values as the old interleaved cache).
        for cc in 0..ch {
            out[cc * n + k] = tile.planes[cc][cp] + tile.means[cc];
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
        let mean = (s1 / support.total_weight) as f32;
        let norm_sq = s2 - s1 * (mean as f64);
        let dst = &mut out[kc * n..][..n];
        if norm_sq < FLAT_NORM_SQ_EPS {
            dst.fill(0.0);
        } else {
            let inv = (1.0 / norm_sq.sqrt()) as f32;
            for (d, (&x, &sw)) in dst.iter_mut().zip(col.iter().zip(&support.sqrt_weights)) {
                *d = sw * (x - mean) * inv;
            }
        }
        kc += 1;
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
///
/// The cache itself (centered planar `f32` planes + invalidity plane) lives on
/// the [`ContextTile`] now, so the per-call scratch holds only the per-channel
/// kernel, the running correlation maps, and the combined grid.
#[derive(Default)]
struct SearchScratch {
    /// The leave-one-out template the candidates score against (`kept_ch · n`),
    /// laid out `[c · n + k]`; the caller writes it each round.
    tmpl: Vec<f32>,
    /// Per-support-pixel kernel `√w · tmpl` for the channel being accumulated
    /// (`f32` — the AVX2 kernel broadcasts these as `f32` lanes).
    kern: Vec<f32>,
    /// Per-support-pixel window weight `w` as `f32` (one-time conversion from the
    /// `f64` `support.weights`; broadcast each k-step).
    w_f32: Vec<f32>,
    /// Per-channel correlation maps over the shift grid (`(2·margin+1)²`): the
    /// numerator `Σ kern·I_c` and the centered window moments `Σ w·I_c`,
    /// `Σ w·I_c²` (`I_c = I − cache_mean`, so the centering algebra absorbs the
    /// mean: the windowed ZNCC formula on centered S1/S2 is identical to the
    /// raw-value form — see the [`ContextTile`] doc and `search_shift_scalar`).
    g_n: Vec<f32>,
    g_s1: Vec<f32>,
    g_s2: Vec<f32>,
    /// Per-shift count of out-of-frame support pixels (a shift is scorable iff 0).
    /// `f32` — values are small integers (≤ `n`).
    ginv: Vec<f32>,
    /// Combined ZNCC grid over the `±margin` window (`(2·margin+1)²`).
    grid: Vec<f64>,
    /// [`SearchStrategy::PlusDescent`]-only: flat per-(kept channel, support
    /// pixel) kernel buffer, `[c · n + k]`, built once per
    /// `search_shift_plus_descent` call and reused across every cell scored.
    /// The exhaustive path's `kern` rebuilds per channel inside its loop; the
    /// descent visits ~10 cells per call, so amortising the kern build over all
    /// of them takes the per-cell rebuild off the hot path. Reused here so the
    /// descent allocates nothing per cell in the steady state.
    pd_kerns: Vec<f32>,
    /// [`SearchStrategy::PlusDescent`]-only: per-kept-channel `tsum_c =
    /// Σ kern[c · n + k]`, parallel to the channel dimension of [`pd_kerns`].
    pd_tsums: Vec<f64>,
    /// [`SearchStrategy::PlusDescent`]-only: per-kept-channel `(n, s1, s2)`
    /// scratch, overwritten by every cell the descent scores. The combine
    /// pass reads it back to fold into the cell's ZNCC. Sized once per
    /// `search_shift_plus_descent` call (`resize(channels, ..)`); the
    /// per-cell scoring writes by index, so no Vec bookkeeping happens
    /// inside the timed `SEARCH_ACC` block.
    pd_per_channel: Vec<(f32, f32, f32)>,
    /// [`SearchStrategy::PlusDescent`]-only: kept-channel index → tile-channel
    /// index lookup, parallel to the channel dimension of [`pd_kerns`].
    /// Walked once at the top of every `search_shift_plus_descent` call so
    /// per-cell scoring can index this rather than re-walking `keep_mask`.
    pd_kept_channels: Vec<usize>,
    /// [`SearchStrategy::PlusDescent`]-only: dense visited cache sized
    /// `(2·margin+1)²`, indexed `((dy + margin) · span + (dx + margin))`.
    /// Sentinel-encoded to avoid carrying a separate "visited" bitmap:
    /// `f64::NAN` = unvisited (skip the score, evaluate it),
    /// `f64::NEG_INFINITY` = visited and unscorable (oob / oof / disk —
    /// don't re-evaluate, don't admit as a neighbor),
    /// any other finite value = visited and that cell's combined ZNCC.
    /// Reset to all-NaN at the start of every `search_shift_plus_descent`
    /// call. Replaces a per-call `HashMap<(i64,i64), Option<f64>>`.
    pd_visited: Vec<f64>,
}

/// The result of a [`search_shift`] — the residual shift of one view relative to
/// its current integer base offset, in `R_s`-grid steps.
#[derive(Debug, Clone, Copy)]
struct ShiftResult {
    /// Sub-pixel-refined shift in the grid's x (column) axis: integer argmax plus
    /// the separable parabolic residual.
    dx: f64,
    /// Sub-pixel-refined shift in the grid's y (row) axis.
    dy: f64,
    /// The integer argmax shift in x — the part that moves the integer read
    /// accumulator `iacc` (every cache read stays at an integer index).
    ix: i64,
    /// The integer argmax shift in y.
    iy: i64,
    /// ZNCC at the integer peak.
    peak: f64,
}

/// Integer windowed-ZNCC translation search of view `v`'s cache against
/// `sc.tmpl`, refined to sub-pixel by a separable parabolic fit. Returns a
/// [`ShiftResult`] — the integer argmax shift `(ix, iy)`, its sub-pixel-refined
/// counterpart `(dx, dy)`, and the ZNCC `peak` at the integer peak — or `None` if
/// no in-frame window position could be scored. The search centres on
/// the view's current integer offset `iacc`: the `(dy, dx) = (0, 0)` shift scores
/// the `R×R` core window whose top-left support pixel sits at cache index
/// `(base_y, base_x) = ((cache_res − R) / 2 + iacc_y, … + iacc_x)`, and the search
/// slides the window over `±margin` around it. Both `base ± margin` are guaranteed
/// in-bounds by the cache sizing (`cache_res = R + 4·margin`, `|iacc| ≤ search`).
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
    resolution: usize,
    margin: i64,
    base_y: usize,
    base_x: usize,
) -> Option<ShiftResult> {
    let n = support.pixels.len();
    let istride = tile.istride;
    // The search grid's origin (`gy = gx = 0`) reads the window at `base − margin`.
    let win_oy = base_y - margin as usize;
    let win_ox = base_x - margin as usize;
    let span = (2 * margin + 1) as usize;
    let gsz = span * span;

    debug_assert_eq!(keep_mask.iter().filter(|&&k| k).count(), channels);

    // Per-support `w` as f32 (one-time conversion the AVX2 kernel can broadcast).
    sc.w_f32.clear();
    sc.w_f32.extend(support.weights.iter().map(|&w| w as f32));

    // Validity: count out-of-frame support pixels per shift (channel-independent);
    // a shift with any is unscorable, matching `extract_core`'s all-valid gate.
    sc.ginv.clear();
    sc.ginv.resize(gsz, 0.0);
    accumulate_count(
        &tile.invalid_plane,
        support,
        resolution,
        istride,
        span,
        win_oy,
        win_ox,
        &mut sc.ginv,
    );

    // Per kept channel: accumulate the three correlation maps over the centered
    // plane, then fold the channel's ZNCC into the combined grid. With centered
    // values, `S2 − S1²/W` is the same algebra as raw and the mean offset in the
    // numerator cancels (`Σkern = tsum` exactly by construction), so the combine
    // step is identical to the raw-value formula — see the [`ContextTile`] doc.
    // The combined `grid` accumulates across channels and must start at zero;
    // `clear()` + `resize()` so a reused scratch is reliably zeroed up front,
    // not only on grow. `g_n / g_s1 / g_s2` are sized here (overwritten per
    // channel by `compute_channel_grids`, no pre-zero needed).
    sc.grid.clear();
    sc.grid.resize(gsz, 0.0);
    sc.g_n.resize(gsz, 0.0);
    sc.g_s1.resize(gsz, 0.0);
    sc.g_s2.resize(gsz, 0.0);
    let inv_total_weight = 1.0 / support.total_weight;
    let mut kc_out = 0usize;
    for (c, &keep) in keep_mask.iter().enumerate() {
        if !keep {
            continue;
        }
        let tmpl_c = &sc.tmpl[kc_out * n..][..n];
        sc.kern.clear();
        let mut tsum = 0.0f64;
        for (&sw, &t) in support.sqrt_weights.iter().zip(tmpl_c) {
            let kk = sw * t;
            sc.kern.push(kk);
            tsum += kk as f64;
        }
        // `compute_channel_grids` overwrites; no pre-zero needed.
        prof::SEARCH_ACC.time(|| {
            compute_channel_grids(
                &tile.planes[c],
                support,
                &sc.kern,
                &sc.w_f32,
                resolution,
                istride,
                span,
                win_oy,
                win_ox,
                &mut sc.g_n,
                &mut sc.g_s1,
                &mut sc.g_s2,
            );
        });
        prof::SEARCH_COMBINE.time(|| {
            for s in 0..gsz {
                let s1 = sc.g_s1[s] as f64;
                let s2 = sc.g_s2[s] as f64;
                let nval = sc.g_n[s] as f64;
                let mean = s1 * inv_total_weight;
                let norm_sq = s2 - s1 * mean;
                // A channel flat in this window contributes 0 (matches `znorm_core`).
                if norm_sq >= FLAT_NORM_SQ_EPS {
                    sc.grid[s] += (nval - mean * tsum) / norm_sq.sqrt();
                }
            }
        });
        kc_out += 1;
    }

    // Average over channels, find the integer argmax, and refine sub-pixel by a
    // separable parabolic fit. Out-of-frame shifts score −∞ (never chosen).
    prof::SEARCH_ARGMAX.time(|| {
        let chf = channels as f64;
        let at = |dy: i64, dx: i64| -> usize {
            ((dy + margin) as usize) * span + (dx + margin) as usize
        };
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
        Some(ShiftResult {
            dx: px as f64 + sx,
            dy: py as f64 + sy,
            ix: px,
            iy: py,
            peak,
        })
    })
}

/// [`SearchStrategy::PlusDescent`] counterpart to [`search_shift`]: starts at
/// `(dy, dx) = (0, 0)` (the view's current integer base offset), evaluates the
/// 4 axis neighbors per step, moves to the best improver, and stops when no
/// neighbor beats the current cell. Each cell is scored at most once via
/// [`score_cell_one_channel`]; the visited cache stores the combined ZNCC per
/// cell. The final separable parabolic sub-pixel fit reuses the 4 cardinal
/// neighbors already in the cache (each was evaluated to discover the STOP
/// condition).
///
/// Same `ShiftResult` contract as `search_shift` — the integer argmax `(ix, iy)`
/// drives the read accumulator, `(dx, dy)` carry the sub-pixel residual, and
/// `peak` is the combined ZNCC at the integer cell. Bounded to `|dy|, |dx| ≤
/// margin`; neighbors past the bound or with any out-of-frame support pixel
/// score `None` (skipped, never chosen).
///
/// Cells visited per call: `5 + 3 · walk_steps` (1 seed + 4 neighbors per step,
/// with 1 cache hit per move). On dino the average is ~6 cells per call — vs
/// the 169 cells of the default ±6 grid `search_shift` processes — at ~32 µs
/// per call (the per-cell `vgatherdps` kernel) vs ~145 µs (the SAXPY). The
/// crossover with `search_shift`'s whole-grid SIMD is around 50–80 cells
/// visited, so the descent loses on pathological long walks; `Exhaustive`
/// remains the right pick for the global-argmax fallback. See
/// `specs/core/keypoint-localization-search-cache.md` for the strategy
/// trade-off discussion.
///
/// **Profile attribution** (`SFMTOOL_PROFILE=1`): the descent reports per-cell
/// `SEARCH_ACC` (invalidity-count + per-channel scoring), per-cell
/// `SEARCH_COMBINE` (mean / ZNCC fold + cross-channel sum), and per-call
/// `SEARCH_ARGMAX` (the final parabolic). The `N_CELLS` event counter bumps
/// once per cell scored (visited-cache hits and oob/oof/disk skips do not
/// count), so `N_CELLS / N_SEARCH` is the average cells-per-call directly out
/// of the profile output. `N_CELLS` is `0` under `Exhaustive` — its whole-
/// grid SAXPY has no per-cell event.
#[allow(clippy::too_many_arguments)]
fn search_shift_plus_descent(
    tile: &ContextTile,
    sc: &mut SearchScratch,
    support: &Support,
    keep_mask: &[bool],
    channels: usize,
    resolution: usize,
    margin: i64,
    base_y: usize,
    base_x: usize,
) -> Option<ShiftResult> {
    let n = support.pixels.len();
    let istride = tile.istride;
    debug_assert_eq!(keep_mask.iter().filter(|&&k| k).count(), channels);

    // Per-support `w` as f32 — mirrors `search_shift`'s one-time conversion.
    sc.w_f32.clear();
    sc.w_f32.extend(support.weights.iter().map(|&w| w as f32));

    // Build the flat per-(kept channel, support pixel) kern + per-channel
    // tsum into the reused `SearchScratch` slots. Layout: `pd_kerns[c · n + k]
    // = √w[k] · tmpl[c · n + k]` for `c in 0..channels` (kept-channel index).
    // The SAXPY path rebuilds these per channel inside its loop; the descent
    // visits ~10 cells per call, so amortising the rebuild across all of them
    // takes the kern build off the per-cell hot path. Reusing the
    // `SearchScratch` buffers means no allocation per `search_shift` call
    // after the first.
    sc.pd_kerns.clear();
    sc.pd_kerns.resize(channels * n, 0.0);
    sc.pd_tsums.clear();
    sc.pd_tsums.resize(channels, 0.0);
    // Precompute the kept-channel-index → tile-channel-index lookup once per
    // call; the per-cell scoring loop indexes into this rather than walking
    // `keep_mask` linearly each time. Drops the per-cell scoring's keep_mask
    // dispatch from O(channels) (one walk per kept channel) to O(1).
    sc.pd_kept_channels.clear();
    sc.pd_kept_channels.extend(
        keep_mask
            .iter()
            .enumerate()
            .filter_map(|(c, &k)| k.then_some(c)),
    );
    debug_assert_eq!(sc.pd_kept_channels.len(), channels);
    for kc_out in 0..channels {
        let tmpl_c = &sc.tmpl[kc_out * n..][..n];
        let kern_c = &mut sc.pd_kerns[kc_out * n..][..n];
        let mut tsum = 0.0f64;
        for ((&sw, &t), kk) in support
            .sqrt_weights
            .iter()
            .zip(tmpl_c)
            .zip(kern_c.iter_mut())
        {
            let v = sw * t;
            *kk = v;
            tsum += v as f64;
        }
        sc.pd_tsums[kc_out] = tsum;
    }

    // Dense visited cache, sentinel-encoded in `pd_visited[idx(dy, dx)]`:
    // `f64::NAN` = unvisited, `f64::NEG_INFINITY` = visited+unscorable,
    // any other finite value = visited+scored. Replaces the previous
    // `HashMap<(i64, i64), Option<f64>>` — the dense Vec is sized
    // `(2·margin+1)²` (169 for the production default), fits in one cache
    // line of pointers, and the index→slot lookup is a couple of arithmetic
    // ops vs a hash + bucket walk.
    let span_axis = (2 * margin + 1) as usize;
    let gsz = span_axis * span_axis;
    sc.pd_visited.clear();
    sc.pd_visited.resize(gsz, f64::NAN);
    let idx_of = |dy: i64, dx: i64| -> usize {
        ((dy + margin) as usize) * span_axis + ((dx + margin) as usize)
    };

    // Pre-size the per-cell `(n, s1, s2)` scratch to exactly `channels` slots
    // so the per-cell SEARCH_ACC block writes by index — no Vec bookkeeping
    // (clear/push) inside the timed kernel block. Initial value is overwritten
    // by every scoreable cell before COMBINE reads it.
    sc.pd_per_channel.clear();
    sc.pd_per_channel.resize(channels, (0.0, 0.0, 0.0));

    let inv_total_weight = 1.0 / support.total_weight;
    let chf = channels as f64;

    // Score one (dy, dx) cell across all kept channels and combine into the
    // mean-over-channels ZNCC. Returns `None` for out-of-bounds, out-of-disk,
    // or any-invalid-support-pixel cells; cached either way so neighbors
    // revisited by the descent walk hit the slot. Profile attribution:
    // `SEARCH_ACC` wraps the data-touching kernel (invalidity-count +
    // per-channel scoring), `SEARCH_COMBINE` wraps the per-channel ZNCC
    // algebra + cross-channel sum, and `N_CELLS` is bumped once per cell that
    // gets actually scored (visited-cache hits and oob/oof/disk skips do not
    // count).
    macro_rules! score_cell {
        ($dy:expr, $dx:expr) => {{
            let dy_: i64 = $dy;
            let dx_: i64 = $dx;
            if dy_.abs() > margin || dx_.abs() > margin {
                None
            } else {
                let slot = sc.pd_visited[idx_of(dy_, dx_)];
                if slot.is_nan() {
                    // First visit: compute the score (or detect unscorable).
                    let win_y = (base_y as i64 + dy_) as usize;
                    let win_x = (base_x as i64 + dx_) as usize;
                    let ginv = prof::SEARCH_ACC.time(|| {
                        let ginv = count_invalid_at_cell(
                            &tile.invalid_plane,
                            support,
                            resolution,
                            istride,
                            win_y,
                            win_x,
                        );
                        if ginv <= 0.5 {
                            for k in 0..channels {
                                let tile_c = sc.pd_kept_channels[k];
                                let kern_k = &sc.pd_kerns[k * n..][..n];
                                sc.pd_per_channel[k] = score_cell_one_channel(
                                    &tile.planes[tile_c],
                                    support,
                                    kern_k,
                                    &sc.w_f32,
                                    resolution,
                                    istride,
                                    win_y,
                                    win_x,
                                );
                            }
                        }
                        ginv
                    });
                    let result = if ginv > 0.5 {
                        sc.pd_visited[idx_of(dy_, dx_)] = f64::NEG_INFINITY;
                        None
                    } else {
                        let zncc = prof::SEARCH_COMBINE.time(|| {
                            let mut combined = 0.0_f64;
                            for (k, &(n_acc, s1_acc, s2_acc)) in
                                sc.pd_per_channel.iter().enumerate()
                            {
                                let s1 = s1_acc as f64;
                                let s2 = s2_acc as f64;
                                let nval = n_acc as f64;
                                let mean = s1 * inv_total_weight;
                                let norm_sq = s2 - s1 * mean;
                                if norm_sq >= FLAT_NORM_SQ_EPS {
                                    combined += (nval - mean * sc.pd_tsums[k]) / norm_sq.sqrt();
                                }
                            }
                            combined / chf
                        });
                        prof::count(&prof::N_CELLS, 1);
                        sc.pd_visited[idx_of(dy_, dx_)] = zncc;
                        Some(zncc)
                    };
                    result
                } else if slot == f64::NEG_INFINITY {
                    None
                } else {
                    Some(slot)
                }
            }
        }};
    }

    // Seed at the cache's centre (current integer base offset).
    let mut current_phi = score_cell!(0_i64, 0_i64)?;
    let mut current = (0_i64, 0_i64);

    // Steepest-descent walk: pick the best improving neighbor, stop when none.
    // The per-neighbor best-of-4 pick is a handful of `f64` comparisons —
    // small enough relative to the score_cell calls driving it that it is not
    // separately timed.
    loop {
        let mut best_move: Option<((i64, i64), f64)> = None;
        for (dy_step, dx_step) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let next = (current.0 + dy_step, current.1 + dx_step);
            if let Some(phi) = score_cell!(next.0, next.1) {
                if phi > current_phi && best_move.is_none_or(|(_, bs)| phi > bs) {
                    best_move = Some((next, phi));
                }
            }
        }
        match best_move {
            None => break,
            Some((next, phi)) => {
                current = next;
                current_phi = phi;
            }
        }
    }

    // SEARCH_ARGMAX: separable parabolic sub-pixel refinement from the 4
    // cardinal-neighbor cells — all already in the visited cache (each was
    // evaluated by the STOP-check loop above). A neighbor that scored `None`
    // (out-of-grid / out-of-disk / out-of-frame) drops its axis to the
    // integer offset. Matches `search_shift`'s SEARCH_ARGMAX wrap, which
    // similarly times the argmax + parabolic.
    prof::SEARCH_ARGMAX.time(|| {
        let py = current.0;
        let px = current.1;
        let nb = |dy: i64, dx: i64| -> Option<f64> {
            if dy.abs() > margin || dx.abs() > margin {
                return None;
            }
            let v = sc.pd_visited[idx_of(dy, dx)];
            if v.is_finite() {
                Some(v)
            } else {
                None
            }
        };
        let sy = match (nb(py - 1, px), nb(py + 1, px)) {
            (Some(l), Some(r)) => parabolic(current_phi, l, r),
            _ => 0.0,
        };
        let sx = match (nb(py, px - 1), nb(py, px + 1)) {
            (Some(l), Some(r)) => parabolic(current_phi, l, r),
            _ => 0.0,
        };
        Some(ShiftResult {
            dx: px as f64 + sx,
            dy: py as f64 + sy,
            ix: px,
            iy: py,
            peak: current_phi,
        })
    })
}

/// **Compute and overwrite** one channel's per-shift correlation grids over the
/// `span × span` shift window: numerator `g_n[s] = Σ_k kern[k]·I[s+k]` and the
/// centered window moments `g_s1[s] = Σ_k w[k]·I[s+k]`,
/// `g_s2[s] = Σ_k w[k]·I[s+k]²`. `I` is the channel's **centered** planar plane
/// (`I = I_raw − cache_mean`, stored in the `istride`-row-stride
/// [`ContextTile::planes`]). `(win_oy, win_ox)` is the cache index of the search
/// grid's `(gy, gx) = (0, 0)` cell (the window's top-left, at the view's integer
/// base offset minus `margin`). On return, `g_n / g_s1 / g_s2` hold the per-shift
/// grids — they are overwritten, not accumulated; the caller does **not** need to
/// pre-zero them.
///
/// Runtime-dispatched to a hand-rolled AVX2 kernel where available (mirrors
/// [`super::normal_refine::fronto_cache::resample_support_avx2`]); the scalar
/// form is the reference, the non-x86 / non-AVX2 fallback, and the path for spans
/// larger than the AVX2 kernel's 16-lane row.
#[allow(clippy::too_many_arguments)]
fn compute_channel_grids(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    span: usize,
    win_oy: usize,
    win_ox: usize,
    g_n: &mut [f32],
    g_s1: &mut [f32],
    g_s2: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        // The AVX2 kernel handles spans up to 16 (one pair of `__m256` lanes per
        // gy row). The default `span = 13` fits; larger spans fall back to scalar.
        if span <= 16 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: see `compute_channel_grids_avx2`'s SAFETY block — the runtime
            // feature check, `span ≤ 16`, and the cache's `istride` padding plus the
            // `win_o*`/`r_k`/`c_k` invariants ensure every 16-wide load stays in
            // bounds of `plane[..istride * cache_rows]`.
            unsafe {
                return compute_channel_grids_avx2(
                    plane, support, kern, w, resolution, istride, span, win_oy, win_ox, g_n, g_s1,
                    g_s2,
                );
            }
        }
    }
    compute_channel_grids_scalar(
        plane, support, kern, w, resolution, istride, span, win_oy, win_ox, g_n, g_s1, g_s2,
    );
}

/// Scalar reference for [`compute_channel_grids`]: the algebra the AVX2 kernel
/// must match, the non-x86 / non-AVX2 fallback, and the equivalence test's oracle.
/// Internally uses a k-outer in-memory accumulator over the `(N, S1, S2)` grids;
/// the function owns its outputs (zeroes them before accumulating) so the external
/// contract is "overwrite", matching the AVX2 path.
#[allow(clippy::too_many_arguments)]
fn compute_channel_grids_scalar(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    span: usize,
    win_oy: usize,
    win_ox: usize,
    g_n: &mut [f32],
    g_s1: &mut [f32],
    g_s2: &mut [f32],
) {
    let gsz = span * span;
    g_n[..gsz].fill(0.0);
    g_s1[..gsz].fill(0.0);
    g_s2[..gsz].fill(0.0);
    for (k, &p) in support.pixels.iter().enumerate() {
        let r_k = p / resolution;
        let c_k = p % resolution;
        let w_k = w[k];
        let kern_k = kern[k];
        for gy in 0..span {
            let src = &plane[(gy + win_oy + r_k) * istride + (win_ox + c_k)..][..span];
            let gbase = gy * span;
            let on = &mut g_n[gbase..][..span];
            let o1 = &mut g_s1[gbase..][..span];
            let o2 = &mut g_s2[gbase..][..span];
            for gx in 0..span {
                let v = src[gx];
                on[gx] += kern_k * v;
                o1[gx] += w_k * v;
                o2[gx] += w_k * v * v;
            }
        }
    }
}

/// Hand-rolled AVX2 implementation of [`compute_channel_grids`]. Per `gy` row,
/// holds the row's `span ≤ 16` cells of `(N, S1, S2)` in 6 YMM accumulators
/// (two `__m256` each) across the `k`-loop; the centered plane streams through
/// once. Inner loop: **2 loads, 2 mul, 6 FMA, 2 broadcasts** for 16 lanes — the
/// whole plane is in L1, so every load hits L1. Overwrites the output grids
/// (per [`compute_channel_grids`]); the lanes past `span` are spilled to a stack
/// scratch and discarded by the `copy_from_slice` that follows.
///
/// The `span → 16` lane padding wastes `(16 − span) / 16` of the FMA throughput
/// (~19% at `span = 13`), per the spec — acceptable for the gain of holding the
/// row's accumulators in registers. Combine (per-row, after the k-loop) is scalar
/// over `span` cells — small relative to the inner loop (AVX2 `rsqrt` is an open
/// question called out in the spec; skip for stage 1).
///
/// # Safety
///
/// The caller must uphold:
///
/// 1. **CPU features:** `is_x86_feature_detected!("avx2") &&
///    is_x86_feature_detected!("fma")` (the `#[target_feature]` enable is
///    sound only when the runtime has these).
/// 2. **Span fits in 16 lanes:** `span ≤ 16` (one pair of `__m256` per `gy`
///    row); larger spans must take the scalar fallback.
/// 3. **Plane buffer covers every 16-wide load:** for every support pixel `k`
///    with `r_k = pixel/resolution`, `c_k = pixel%resolution`, and every
///    `gy ∈ [0, span)`,
///    `(gy + win_oy + r_k) * istride + (win_ox + c_k) + 16 ≤ plane.len()`.
///    This is what the cache's `istride = align_up(cacheW − 1 + 16, 8)`
///    padding plus the `win_o*`/`r_k`/`c_k` bounds in `search_shift` guarantee.
/// 4. **Output buffers cover the span × span grid:**
///    `g_n.len()`, `g_s1.len()`, `g_s2.len() ≥ span * span`.
/// 5. **`support.pixels` is laid out as `row * resolution + col`** so `r_k <
///    resolution` and `c_k < resolution` — both contribute to invariant 3.
///
/// `debug_assert!`s spot-check (2), (4), and the tight per-call worst case of
/// (3); release builds rely on the caller's contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn compute_channel_grids_avx2(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    span: usize,
    win_oy: usize,
    win_ox: usize,
    g_n: &mut [f32],
    g_s1: &mut [f32],
    g_s2: &mut [f32],
) {
    use std::arch::x86_64::*;
    debug_assert!(span <= 16, "AVX2 kernel handles spans up to 16, got {span}");
    let gsz = span * span;
    debug_assert!(
        g_n.len() >= gsz && g_s1.len() >= gsz && g_s2.len() >= gsz,
        "output grids must cover span * span"
    );
    // Worst-case 16-wide load index: largest `r_k`, `c_k`, and `gy`. The support
    // is `row * resolution + col` with both `< resolution`, so `r_k, c_k ≤
    // resolution − 1`; `gy ≤ span − 1`. (Spot-check the bound at the per-call
    // worst case; the caller is responsible per the SAFETY contract.)
    if let Some(&p_max) = support.pixels.iter().max() {
        let r_max = p_max / resolution;
        let c_max = p_max % resolution;
        let worst_base = (span - 1 + win_oy + r_max) * istride + (win_ox + c_max);
        debug_assert!(
            worst_base + 16 <= plane.len(),
            "AVX2 load would read past the centered plane: \
             worst_base+16={} > plane.len()={} (span={span}, istride={istride}, \
             win_oy={win_oy}, win_ox={win_ox}, r_max={r_max}, c_max={c_max})",
            worst_base + 16,
            plane.len(),
        );
    }
    let plane_ptr = plane.as_ptr();
    let pixels = support.pixels.as_ptr();
    let n = support.pixels.len();
    let kern_ptr = kern.as_ptr();
    let w_ptr = w.as_ptr();
    // Per-row temporaries that mirror the YMM accumulators when we store them.
    let mut tmp = [0.0f32; 16];
    for gy in 0..span {
        let row_y = gy + win_oy;
        let mut n_lo = _mm256_setzero_ps();
        let mut n_hi = _mm256_setzero_ps();
        let mut s1_lo = _mm256_setzero_ps();
        let mut s1_hi = _mm256_setzero_ps();
        let mut s2_lo = _mm256_setzero_ps();
        let mut s2_hi = _mm256_setzero_ps();
        for k in 0..n {
            let p = *pixels.add(k);
            let r_k = p / resolution;
            let c_k = p % resolution;
            let base = (row_y + r_k) * istride + (win_ox + c_k);
            let src_lo = _mm256_loadu_ps(plane_ptr.add(base));
            let src_hi = _mm256_loadu_ps(plane_ptr.add(base + 8));
            let kb = _mm256_set1_ps(*kern_ptr.add(k));
            let wb = _mm256_set1_ps(*w_ptr.add(k));
            n_lo = _mm256_fmadd_ps(kb, src_lo, n_lo);
            n_hi = _mm256_fmadd_ps(kb, src_hi, n_hi);
            s1_lo = _mm256_fmadd_ps(wb, src_lo, s1_lo);
            s1_hi = _mm256_fmadd_ps(wb, src_hi, s1_hi);
            let sq_lo = _mm256_mul_ps(src_lo, src_lo);
            let sq_hi = _mm256_mul_ps(src_hi, src_hi);
            s2_lo = _mm256_fmadd_ps(wb, sq_lo, s2_lo);
            s2_hi = _mm256_fmadd_ps(wb, sq_hi, s2_hi);
        }
        // Combine: spill the row's 16-lane accumulators and copy the first `span`
        // lanes into the channel-shared grid maps. The spilled buffer is `[lo|hi]`.
        let gbase = gy * span;
        let mut store_first = |acc_lo: __m256, acc_hi: __m256, dst: &mut [f32]| {
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc_lo);
            _mm256_storeu_ps(tmp.as_mut_ptr().add(8), acc_hi);
            dst[..span].copy_from_slice(&tmp[..span]);
        };
        store_first(n_lo, n_hi, &mut g_n[gbase..][..span]);
        store_first(s1_lo, s1_hi, &mut g_s1[gbase..][..span]);
        store_first(s2_lo, s2_hi, &mut g_s2[gbase..][..span]);
    }
}

/// Accumulate the per-shift count of out-of-frame support pixels (`invalid` is
/// `1.0` where the tile pixel is invalid, else `0.0`). A shift is scorable iff its
/// count is `0` (every support pixel in frame), the grid analogue of
/// [`extract_core`] returning `false` for any invalid support pixel. The invalidity
/// plane lives in the same `istride` row layout as the centered planes.
#[allow(clippy::too_many_arguments)]
fn accumulate_count(
    invalid: &[f32],
    support: &Support,
    resolution: usize,
    istride: usize,
    span: usize,
    win_oy: usize,
    win_ox: usize,
    g: &mut [f32],
) {
    for &p in &support.pixels {
        let r_k = p / resolution;
        let c_k = p % resolution;
        for gy in 0..span {
            let src = &invalid[(gy + win_oy + r_k) * istride + (win_ox + c_k)..][..span];
            let gr = &mut g[gy * span..][..span];
            for gx in 0..span {
                gr[gx] += src[gx];
            }
        }
    }
}

/// Per-cell counterpart to [`compute_channel_grids`]: score one (and only one)
/// shift `(win_y, win_x)` in the cache plane, returning the three correlation
/// sums needed to assemble the windowed ZNCC at that single cell:
///
///   `n  = Σ_k kern[k] · I[(win_y + r_k, win_x + c_k)]`
///   `s1 = Σ_k w[k]    · I[(win_y + r_k, win_x + c_k)]`
///   `s2 = Σ_k w[k]    · I[(win_y + r_k, win_x + c_k)]²`
///
/// Algebraically a single shift's slice of the grid form, but evaluated by a
/// direct loop over support pixels (no SAXPY across the grid row) — the
/// fundamental primitive for the "+"-descent search strategy, where each call
/// scores one cell and the descent visits ≲ 10 cells per search instead of
/// 169 = `(2·6+1)²`. Mirrors [`compute_channel_grids_scalar`]'s algebra
/// pixel-for-pixel; the AVX2 form below uses `vgatherdps`.
///
/// Runtime-dispatched to the AVX2 kernel where available; otherwise scalar.
#[allow(clippy::too_many_arguments)]
fn score_cell_one_channel(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    win_y: usize,
    win_x: usize,
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the runtime feature check is the `#[target_feature]`
            // pre-condition; the per-pixel cache index is the same expression
            // `compute_channel_grids_avx2` validates, so its in-bounds-load
            // contract carries over for the gather lanes here.
            unsafe {
                return score_cell_one_channel_avx2(
                    plane, support, kern, w, resolution, istride, win_y, win_x,
                );
            }
        }
    }
    score_cell_one_channel_scalar(plane, support, kern, w, resolution, istride, win_y, win_x)
}

/// Scalar reference for [`score_cell_one_channel`]: the algebra the AVX2 kernel
/// must match, and the non-x86 / non-AVX2 fallback.
#[allow(clippy::too_many_arguments)]
fn score_cell_one_channel_scalar(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    win_y: usize,
    win_x: usize,
) -> (f32, f32, f32) {
    let mut n = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    for (k, &p) in support.pixels.iter().enumerate() {
        let r_k = p / resolution;
        let c_k = p % resolution;
        let v = plane[(win_y + r_k) * istride + (win_x + c_k)];
        n += kern[k] * v;
        s1 += w[k] * v;
        s2 += w[k] * v * v;
    }
    (n, s1, s2)
}

/// Hand-rolled AVX2 implementation of [`score_cell_one_channel`]: 8 support
/// pixels per iteration via `_mm256_i32gather_ps` against a freshly-built
/// per-batch index vector, three FMAs per iteration into 8-lane accumulators
/// (`n`, `s1`, `s2`), with a horizontal reduce at the end and a scalar tail for
/// the support's remainder.
///
/// The gather is the path's bottleneck (Intel `vgatherdps` is roughly
/// 10-12 cycles per lane on this generation, so 8 lanes ≈ 80 cycles); even so,
/// the per-call total stays well under the SAXPY's per-cell amortized cost
/// because we only visit a handful of cells per descent. The lane-fill of the
/// index vector is done from a stack buffer, so the gather sees a fresh
/// 8-element index array each iteration.
///
/// # Safety
///
/// 1. **CPU features:** `is_x86_feature_detected!("avx2") &&
///    is_x86_feature_detected!("fma")`.
/// 2. **Plane covers every gathered index:** for each support pixel `k` with
///    `r_k = pixel/resolution`, `c_k = pixel%resolution`, the byte offset
///    `(win_y + r_k) * istride + (win_x + c_k)` must lie within
///    `plane[..]` — exactly the contract `compute_channel_grids_avx2` already
///    relies on for one shift cell, and the same `cache_res = R + 4·margin`
///    plus `|win_y - cache_c0|, |win_x - cache_c0| ≤ margin` invariants guard
///    it.
/// 3. **Kern / weight slices cover the support:** `kern.len() >=
///    support.pixels.len()`, `w.len() >= support.pixels.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_cell_one_channel_avx2(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    win_y: usize,
    win_x: usize,
) -> (f32, f32, f32) {
    use std::arch::x86_64::*;
    let n = support.pixels.len();
    let pixels = support.pixels.as_ptr();
    let kern_ptr = kern.as_ptr();
    let w_ptr = w.as_ptr();
    let plane_ptr = plane.as_ptr();

    let mut acc_n = _mm256_setzero_ps();
    let mut acc_s1 = _mm256_setzero_ps();
    let mut acc_s2 = _mm256_setzero_ps();

    let chunks = n / 8;
    let mut idx_buf = [0_i32; 8];
    for chunk in 0..chunks {
        let k0 = chunk * 8;
        // Build the 8 cache indices for this batch (per-pixel `(win_y + r_k) *
        // istride + (win_x + c_k)`; the `* 4` byte-scaling lives in the gather's
        // scale argument). `usize` → `i32` is sound here because the cache size
        // is bounded by `cache_res² < i32::MAX` for any realistic patch.
        for (lane, slot) in idx_buf.iter_mut().enumerate() {
            let p = *pixels.add(k0 + lane);
            let r_k = p / resolution;
            let c_k = p % resolution;
            *slot = ((win_y + r_k) * istride + (win_x + c_k)) as i32;
        }
        let indices = _mm256_loadu_si256(idx_buf.as_ptr().cast::<__m256i>());
        // `vgatherdps` with scale = 4 (f32 stride).
        let v8 = _mm256_i32gather_ps::<4>(plane_ptr, indices);
        let kern8 = _mm256_loadu_ps(kern_ptr.add(k0));
        let w8 = _mm256_loadu_ps(w_ptr.add(k0));
        acc_n = _mm256_fmadd_ps(kern8, v8, acc_n);
        acc_s1 = _mm256_fmadd_ps(w8, v8, acc_s1);
        let v_sq = _mm256_mul_ps(v8, v8);
        acc_s2 = _mm256_fmadd_ps(w8, v_sq, acc_s2);
    }

    // Horizontal-reduce the three accumulators into scalar sums.
    let mut tmp = [0.0_f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc_n);
    let mut n_acc = tmp.iter().sum::<f32>();
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc_s1);
    let mut s1_acc = tmp.iter().sum::<f32>();
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc_s2);
    let mut s2_acc = tmp.iter().sum::<f32>();

    // Scalar tail for the trailing support pixels (`n % 8`).
    for k in (chunks * 8)..n {
        let p = *pixels.add(k);
        let r_k = p / resolution;
        let c_k = p % resolution;
        let v = *plane_ptr.add((win_y + r_k) * istride + (win_x + c_k));
        n_acc += *kern_ptr.add(k) * v;
        s1_acc += *w_ptr.add(k) * v;
        s2_acc += *w_ptr.add(k) * v * v;
    }
    (n_acc, s1_acc, s2_acc)
}

/// Per-cell counterpart to [`accumulate_count`]: count out-of-frame support
/// pixels at one shift `(win_y, win_x)`. A shift is scorable iff this returns
/// `0` (every support pixel in frame).
fn count_invalid_at_cell(
    invalid: &[f32],
    support: &Support,
    resolution: usize,
    istride: usize,
    win_y: usize,
    win_x: usize,
) -> f32 {
    let mut acc = 0.0_f32;
    for &p in &support.pixels {
        let r_k = p / resolution;
        let c_k = p % resolution;
        acc += invalid[(win_y + r_k) * istride + (win_x + c_k)];
    }
    acc
}

/// One view's mutable congealing state.
///
/// The accumulated in-plane offset is split into an **integer** read accumulator
/// `iacc` and a **sub-pixel residual** (both in `R_s`-grid steps, where `R_s` is
/// the search resolution): `iacc` is the only thing that indexes the per-view
/// cache, so every cache read stays exact (an integer cache-index shift is a
/// bit-exact re-warp); the parabolic residual rides alongside for convergence
/// detection and the final reported keypoint but is **never folded back into the
/// read position**. The patch-grid in-plane offset is `(iacc + residual) / m` —
/// see [`shifted_center`] / [`finalize`] and
/// `specs/core/keypoint-localization-search-cache.md`.
struct ViewState {
    /// Image index into the caller's `views` slice.
    idx: u32,
    /// Integer read accumulator `(iau, iav)` in `R_s`-grid steps: the cache index
    /// (relative to the cache centre) the core and search candidates are read at.
    iacc: [i64; 2],
    /// The latest parabolic sub-pixel residual `(ru, rv)` in `R_s`-grid steps from
    /// the last round that scored this view; used for convergence and the final
    /// keypoint only, never for cache reads.
    residual: [f64; 2],
    /// The view's projection of the point `project_i(X_p)`, source-image px — the
    /// anchor the keypoint and its `max_shift_px` gate are measured from.
    proj: [f64; 2],
    /// The latest leave-one-out ZNCC (peak from the round's search); `NaN` until
    /// a round scores it.
    loo: f64,
}

impl ViewState {
    /// The total in-plane offset `(au, av)` in **`R_s`-grid steps**: the integer
    /// read accumulator plus the sub-pixel residual. This is the unit
    /// [`shifted_center`] expects when paired with the `R_s`-resolution
    /// world-per-grid-px (`wpp`), so the `1/m` scaling back to patch-grid px is
    /// absorbed into `wpp` rather than applied here.
    fn offset_steps(&self) -> [f64; 2] {
        [
            self.iacc[0] as f64 + self.residual[0],
            self.iacc[1] as f64 + self.residual[1],
        ]
    }
}

/// Drop the `(state, cache)` pairs for which `keep` returns `false`, keeping the
/// two vectors parallel. `keep` is evaluated once per state in order; the cache at
/// the same index is dropped with it. Cold path (runs only on view drops).
fn retain_states_and_caches(
    states: &mut Vec<ViewState>,
    caches: &mut Vec<ContextTile>,
    mut keep: impl FnMut(&ViewState) -> bool,
) {
    debug_assert_eq!(states.len(), caches.len());
    let mask: Vec<bool> = states.iter().map(&mut keep).collect();
    let mut i = 0;
    states.retain(|_| {
        let k = mask[i];
        i += 1;
        k
    });
    let mut i = 0;
    caches.retain(|_| {
        let k = mask[i];
        i += 1;
        k
    });
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
    // Search resolution `R_s = round(m·R)`: the cache, support, and shift grid all
    // build at `R_s`. An integer step in this grid is `1/m` patch-grid px, so the
    // found shift is scaled by `inv_m = 1/m` back to patch-grid px (`m = 1` — the
    // default — is a no-op). See `specs/core/keypoint-localization-search-cache.md`.
    let m = (params.search_resolution_multiplier as f64).max(1e-3);
    let base_res = params.resolution.max(2);
    let resolution = ((m * base_res as f64).round() as u32).max(2);
    // In-round search radius, in `R_s`-grid steps (`search` is patch-grid px, an
    // `R_s` step is `1/m` patch-grid px). At `m = 1` this is the old `margin`.
    let margin = (params.search * m).ceil().max(1.0) as i64;
    // The accumulated integer drift is clipped to `±search_steps`; `search_steps =
    // margin` keeps the larger `R_s + 4·margin` cache exactly covering every round.
    let search_steps = margin;
    // Render-once cache size: `R_s + 4·search`, the unconditionally-correct option
    // (covers a window centre at `iacc + d` with `|iacc| ≤ search`, `|d| ≤ margin`).
    let context_res = resolution + 4 * margin as u32;
    // Cache index of the `R×R` core at zero offset (`iacc = 0`).
    let cache_c0 = (2 * margin) as usize;
    let r = resolution as usize;

    let wpp_u = 2.0 * patch.half_extent[0] / resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / resolution as f64;

    // Window support over the R_s×R_s core.
    let support = build_support(params.window, resolution);

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
        // Seed offset (in `R_s`-grid steps, since `wpp` is at `R_s`): unproject the
        // starting keypoint onto the plane, else zero. Split into the integer read
        // accumulator (clipped to the cache's drift bound) and a sub-pixel residual
        // — the residual keeps a lone-view seed exact through `finalize` while the
        // congealing read position stays integer.
        let off = match starting_keypoints {
            Some(seeds) => seed_offset(patch, view, seeds[k], wpp_u, wpp_v).unwrap_or([0.0, 0.0]),
            None => [0.0, 0.0],
        };
        // `off = [u, v]` (u-axis, v-axis components, in `R_s`-grid steps). Split each
        // axis into the clamped integer read accumulator and a pure sub-pixel
        // residual — the residual keeps a lone-view seed exact through `finalize`; a
        // seed beyond `±search` is clamped on the integer part, as the round drift is.
        let iu = (off[0].round() as i64).clamp(-search_steps, search_steps);
        let iv = (off[1].round() as i64).clamp(-search_steps, search_steps);
        states.push(ViewState {
            idx: i,
            iacc: [iu, iv],
            residual: [off[0] - off[0].round(), off[1] - off[1].round()],
            proj: [proj.0, proj.1],
            loo: f64::NAN,
        });
    }

    if states.len() < 2 {
        return finalize(patch, views, &states, wpp_u, wpp_v);
    }

    // Render each view's expanded cache **once** (frame-oriented at the seed,
    // `acc = 0`), sized `R_s + 4·margin` to cover the full search drift. Every
    // round reads its core and scores its shift grid from this cache at the view's
    // current integer offset `iacc` — no per-round render. `caches` stays parallel
    // to `states`; the view-dropping retain below filters both together.
    let mut caches: Vec<ContextTile> = Vec::with_capacity(states.len());
    prof::count(&prof::N_RENDER, states.len() as u64);
    prof::RENDER.time(|| {
        for st in &states {
            caches.push(render_context(
                patch,
                &views[st.idx as usize],
                0.0,
                0.0,
                wpp_u,
                wpp_v,
                resolution,
                context_res,
                params.sampler,
            ));
        }
    });

    let mut sc = ConsensusScratch::default();
    let mut search = SearchScratch::default();
    for _round in 0..params.max_iters.max(1) {
        prof::count(&prof::N_ROUNDS, 1);
        // 1. Read every view's R_s×R_s core from its cache at the integer offset
        //    `iacc` (no render — exact). A view whose core has left the frame (any
        //    support pixel invalid) is dropped for this round.
        let mut live: Vec<usize> = Vec::with_capacity(states.len());
        let mut base_raw = Vec::new();
        for (si, st) in states.iter().enumerate() {
            let cache = &caches[si];
            let mut raw = vec![0f32; cache.channels * support.pixels.len()];
            // `iacc[0]` is the u-axis (column/x) accumulator, `iacc[1]` the v-axis
            // (row/y) — matching the search grid's `(dx, dy)` and `shifted_center`.
            let ox = (cache_c0 as i64 + st.iacc[0]) as usize;
            let oy = (cache_c0 as i64 + st.iacc[1]) as usize;
            if extract_core(cache, &support, r, oy, ox, &mut raw) {
                live.push(si);
                base_raw.push(raw);
            }
        }
        if live.len() < 2 {
            // Too few views still see the patch to congeal; keep the in-frame ones
            // (with their current offsets) and finalize.
            let live_set: std::collections::HashSet<u32> =
                live.iter().map(|&si| states[si].idx).collect();
            retain_states_and_caches(&mut states, &mut caches, |st| live_set.contains(&st.idx));
            break;
        }

        // 2. z-normalize the live cores into a shared compacted channel space.
        let channels0 = live.iter().map(|&si| caches[si].channels).min().unwrap();
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
                support.total_weight,
                &support.sqrt_weights,
                &mut xs,
            )
        });
        let Some((kept_ch, keep_mask)) = znorm else {
            break; // no textured channel — leave the seeds in place
        };

        // 3-4. Per live view: search its residual shift against the leave-one-out
        //      consensus of the others, then accumulate (clipped to ±search).
        let mut shifts: Vec<Option<ShiftResult>> = vec![None; live.len()];
        let mut loo_xs = vec![0f32; (live.len() - 1) * kept_ch * n];
        for (v, &si) in live.iter().enumerate() {
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
                irls_view_weights(&loo_xs, w, kept_ch, n, params.robust_iters, None, &mut sc);
                weighted_unit_template_into(&loo_xs, &sc.w, w, kept_ch, n, &mut search.tmpl);
            });
            prof::count(&prof::N_SEARCH, 1);
            // Score the shift grid in view `si`'s cache around its current integer
            // base offset `cache_c0 + iacc`. The returned shift is in `R_s`-grid
            // steps relative to that base.
            let st = &states[si];
            let base_x = (cache_c0 as i64 + st.iacc[0]) as usize;
            let base_y = (cache_c0 as i64 + st.iacc[1]) as usize;
            shifts[v] = prof::SEARCH.time(|| match params.search_strategy {
                SearchStrategy::Exhaustive => search_shift(
                    &caches[si],
                    &mut search,
                    &support,
                    &keep_mask,
                    kept_ch,
                    r,
                    margin,
                    base_y,
                    base_x,
                ),
                SearchStrategy::PlusDescent => search_shift_plus_descent(
                    &caches[si],
                    &mut search,
                    &support,
                    &keep_mask,
                    kept_ch,
                    r,
                    margin,
                    base_y,
                    base_x,
                ),
            });
        }

        // Accumulate onto the (still full) `states`. The integer argmax moves the
        // read accumulator `iacc` (clipped to the cache drift bound); the sub-pixel
        // parabolic residual is stored separately and never fed back into the read
        // position — keeping every cache read exact.
        let mut shift_sum = 0.0;
        for (v, &si) in live.iter().enumerate() {
            let st = &mut states[si];
            match shifts[v] {
                Some(sh) => {
                    st.iacc[0] = (st.iacc[0] + sh.ix).clamp(-search_steps, search_steps);
                    st.iacc[1] = (st.iacc[1] + sh.iy).clamp(-search_steps, search_steps);
                    st.residual = [sh.dx - sh.ix as f64, sh.dy - sh.iy as f64];
                    st.loo = sh.peak;
                    shift_sum += sh.dx.hypot(sh.dy);
                }
                // No scorable window this round: leave `iacc`/`residual` in place and
                // mark the LOO unknown (matches the pre-cache None handling).
                None => st.loo = f64::NAN,
            }
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
            let off = st.offset_steps();
            let center = shifted_center(patch, off[0], off[1], wpp_u, wpp_v);
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
        let mut idx = 0;
        retain_states_and_caches(&mut states, &mut caches, |_| {
            let keep = keep_set.contains(&idx);
            idx += 1;
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
        // `offset_steps` is in `R_s`-grid steps (integer read accumulator + the
        // sub-pixel residual); `wpp` is the matching `R_s`-resolution world-per-step.
        let off = st.offset_steps();
        let center = shifted_center(patch, off[0], off[1], wpp_u, wpp_v);
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

/// For each patch of `cloud` (linked to `recon` via `point_indexes`), the track image
/// indices observing its source 3D point — a convenience default `view_sets` for
/// [`localize_patch_cloud_keypoints`] when no view selection has been run.
/// Identical to
/// [`view_indices_from_reconstruction`](super::normal_refine::view_indices_from_reconstruction).
///
/// # Panics
///
/// Panics if `cloud.point_indexes` is not parallel to its patches.
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
    resolution: usize,
    margin: i64,
    base_y: usize,
    base_x: usize,
) -> Option<ShiftResult> {
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
            // The shift `(dy, dx)` window's top-left support pixel sits at the base
            // offset plus the shift — matching `search_shift`'s grid (whose `gy =
            // margin` row is `dy = 0`, reading at `win_oy + margin = base_y`).
            let oy = (base_y as i64 + dy) as usize;
            let ox = (base_x as i64 + dx) as usize;
            if !extract_core(tile, support, resolution, oy, ox, &mut raw) {
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
    Some(ShiftResult {
        dx: px as f64 + sx,
        dy: py as f64 + sy,
        ix: px,
        iy: py,
        peak,
    })
}

#[cfg(test)]
mod tests;
