// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch-keypoint localization by group-wise translation registration
//! (congealing).
//!
//! See `specs/core/patch/patch-keypoint-localization.md` and
//! `specs/core/patch/keypoint-localization-search-cache.md`. Given one 3D point with
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

mod basis;
mod kernels;
mod params;
mod search;

use crate::camera::remap::{remap_aniso_with_pyramid, remap_bilinear, remap_bilinear_mip};
use crate::camera::WarpMap;
use crate::numeric::median_in_place;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::normal_refine::{
    build_support, irls_view_weights, weighted_unit_template_into, znormalize_into_kept,
    ConsensusScratch, ProjectedImage, Sampler, Support,
};
// Only the reference scorer (`znorm_core`, test-only) needs the moment helper and
// the flat-norm floor; the reference LOO-template test also needs the window enum.
#[cfg(test)]
use crate::patch::normal_refine::{weighted_moments_pub, PatchWindow, FLAT_NORM_SQ_EPS};
use crate::reconstruction::SfmrReconstruction;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

// Public API, re-exported at the historical `keypoint_localize::` paths.
pub use params::{BasisPick, KeypointLocalization, KeypointLocalizeParams, SearchStrategy};

use basis::select_basis;

// Search machinery consumed by the congealing orchestration below.
use search::{
    build_loo_gram, loo_consensus_template, search_shift, search_shift_plus_descent, LooScratch,
    SearchScratch, ShiftResult,
};

// Correlation kernels re-exported into this module's namespace only for the
// sibling test module's `use super::*`; production callers reach them through
// `search`, so the re-export is test-gated to stay warning-clean in release.
#[cfg(test)]
use kernels::{
    compute_channel_grids, compute_channel_grids_scalar, score_cell_one_channel,
    score_cell_one_channel_scalar,
};

/// `remap_aniso` sample cap along the major axis (mirrors `normal_refine`).
const MAX_ANISOTROPY: u32 = 16;

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
/// `specs/core/patch/keypoint-localization-search-cache.md`).
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

/// Project a homogeneous world point `(p, w)` into a view **without** the
/// frame-bounds test: `None` only when the point falls behind the camera or
/// outside the camera model's valid domain. `w = 1` is a finite point; `w = 0` is
/// a direction (a point at infinity), rotated into the camera frame without
/// translation and projected as a ray.
///
/// This is the form a *residual* wants: a reprojection a pixel outside the frame
/// is a small error, not a missing measurement, so [candidate
/// spawning](super::spawn) measures against this while the visibility-gated
/// [`project`] decides which views a patch can be rendered in at all.
pub(super) fn project_unclipped(
    view: &ProjectedImage<'_>,
    p: &Point3<f64>,
    w: f64,
) -> Option<(f64, f64)> {
    let pc = view.cam_from_world.transform_point_homogeneous(p.coords, w);
    // Cheirality: a point in front of a *perspective* camera has z < 0. A
    // ray-path model (fisheye / equirectangular) images past 90° off axis,
    // where a legitimate observation has `z >= 0`; there the model's own
    // domain — whatever `ray_to_pixel` accepts — is the only oracle, and this
    // test would discard the whole periphery a >180° capture exists to carry.
    if !view.camera.model.needs_ray_path() && pc.z >= 0.0 {
        return None;
    }
    view.camera.ray_to_pixel([pc.x, pc.y, pc.z])
}

/// Project a homogeneous world point `(p, w)` into a view; `None` when it falls
/// behind the camera or outside the frame. See [`project_unclipped`] for the
/// variant that skips the frame test.
pub(super) fn project(view: &ProjectedImage<'_>, p: &Point3<f64>, w: f64) -> Option<(f64, f64)> {
    let (px, py) = project_unclipped(view, p, w)?;
    let (iw, ih) = (view.camera.width as f64, view.camera.height as f64);
    (px >= 0.0 && py >= 0.0 && px < iw && py < ih).then_some((px, py))
}

/// The patch centre re-anchored on the plane by an in-plane offset `(au, av)` in
/// patch-grid px: `X_p + au·wpp_u·û − av·wpp_v·v̂`. Grid rows count *downward*
/// from `+v̂` (they map to `−v_axis`, matching `WarpMap::from_patch`), so a
/// positive `av` steps along `−v̂`.
pub(super) fn shifted_center(
    patch: &OrientedPatch,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
) -> Point3<f64> {
    patch.center + patch.u_axis * (au * wpp_u) - patch.v_axis * (av * wpp_v)
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
        patch.v_axis,
        [patch.half_extent[0] * scale, patch.half_extent[1] * scale],
    );
    // Preserve the anchor's homogeneous weight so a point at infinity renders as
    // a direction patch (corners are directions), not a finite surfel.
    ctx_patch.w = patch.w;
    let mut map = prof::RENDER_PROJECT
        .time(|| WarpMap::from_patch(&ctx_patch, view.camera, view.cam_from_world, context_res));
    if matches!(sampler, Sampler::Anisotropic | Sampler::BilinearMip) {
        prof::RENDER_SVD.time(|| map.compute_svd());
    }
    let img = prof::RENDER_REMAP.time(|| match sampler {
        Sampler::Anisotropic => remap_aniso_with_pyramid(view.pyramid, &map, MAX_ANISOTROPY),
        Sampler::BilinearMip => remap_bilinear_mip(view.pyramid, &map),
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
/// `specs/core/patch/keypoint-localization-search-cache.md`.
struct ViewState {
    /// Image index into the caller's `views` slice.
    idx: u32,
    /// Position in the deduped/filtered candidate list — the input view-set
    /// order the result contract reports in. Basis and tail members are merged
    /// back on this key after phase B.
    order: u32,
    /// Whether this view congealed as a consensus-basis member (rather than
    /// being registered once against the finished basis template in phase B).
    /// Always `true` when the cap is off.
    is_basis: bool,
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

/// Per-point ranking evidence for the consensus-basis pick.
///
/// Only consulted when [`KeypointLocalizeParams::basis_max_views`] caps the
/// point's view set; the default (all fields empty) leaves the uncapped path
/// untouched. See `specs/core/patch/keypoint-localization-consensus-basis.md`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasisEvidence<'a> {
    /// Per-view score to the point's starting appearance, parallel to the
    /// `view_set` (higher is a better match; `NaN` = unscored, which ranks
    /// below every scored view). `sfm embed-patches` passes the `select_views`
    /// per-admitted-view ZNCC through here. `None` — and, equivalently, an
    /// **empty** slice, which is how a batch caller says "no scores for *this*
    /// point" — falls back to ranking by grazing angle (`|d̂·n̂|`, most frontal
    /// first).
    pub view_scores: Option<&'a [f64]>,
    /// The number of **leading** `view_set` entries that are the point's track
    /// views — the `select_views` output contract, whose `admitted` lists the
    /// track views first. `0` means no track membership is known, which makes
    /// [`KeypointLocalizeParams::basis_force_track_views`] a no-op.
    pub track_view_count: usize,
}

/// Localize the keypoints of one oriented patch over a view set by congealing.
///
/// `views` is one [`ProjectedImage`] per reconstruction image (indexed by image
/// index); `view_set` lists the views to refine (the output of
/// [view selection](super::view_selection)). `starting_keypoints`, when given, is
/// one **optional** seed per `view_set` entry (source-image px), parallel to it:
/// `Some([x, y])` seeds that view by unprojecting the keypoint onto the patch
/// plane, `None` seeds it at the point's own projection `project_i(X_p)` — and so
/// does `None` for the whole slice. A view set that mixes observed views (which
/// have a stored keypoint) with expansion candidates (which do not) is therefore
/// seeded per view rather than all-or-nothing. Returns the kept views and their
/// refined keypoints; see [`KeypointLocalization`] and
/// `specs/core/patch/patch-keypoint-localization.md`.
///
/// Equivalent to [`localize_patch_keypoints_with_basis`] with no basis
/// evidence — which is what the uncapped path (the default
/// `basis_max_views = 0`) needs.
pub fn localize_patch_keypoints(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    starting_keypoints: Option<&[Option<[f64; 2]>]>,
    params: &KeypointLocalizeParams,
) -> KeypointLocalization {
    localize_patch_keypoints_with_basis(
        patch,
        views,
        view_set,
        starting_keypoints,
        BasisEvidence::default(),
        params,
    )
}

/// [`localize_patch_keypoints`] with the caller's per-view ranking evidence for
/// the **consensus-basis cap** (`specs/core/patch/keypoint-localization-consensus-basis.md`).
///
/// With [`KeypointLocalizeParams::basis_max_views`] at `0` — or a
/// view set no larger than the cap (the default `8`) — this is bit-identical to
/// [`localize_patch_keypoints`] and `evidence` is unread. Otherwise `K` views
/// are picked as the consensus basis and congeal exactly as before, and every
/// remaining view registers once against the finished basis template.
pub fn localize_patch_keypoints_with_basis(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    starting_keypoints: Option<&[Option<[f64; 2]>]>,
    evidence: BasisEvidence<'_>,
    params: &KeypointLocalizeParams,
) -> KeypointLocalization {
    // Search resolution `R_s = round(m·R)`: the cache, support, and shift grid all
    // build at `R_s`. An integer step in this grid is `1/m` patch-grid px, so the
    // found shift is scaled by `inv_m = 1/m` back to patch-grid px (`m = 1` — the
    // default — is a no-op). See `specs/core/patch/keypoint-localization-search-cache.md`.
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
    // Parallel to `states`, consumed only by the consensus-basis pick: each
    // candidate's rank score (the caller's, else the grazing cosine computed
    // just below for free) and whether it is one of the point's track views.
    // Only built when the cap can bite, so the (default) uncapped path stays
    // allocation-free.
    let ranking = params.basis_max_views > 0;
    let mut cand_scores: Vec<f64> = Vec::new();
    let mut cand_is_track: Vec<bool> = Vec::new();
    // An empty caller slice means "this point is unscored" — the batch entry
    // point has no per-point `Option`, so it says so with an empty list.
    let view_scores = evidence.view_scores.filter(|s| !s.is_empty());
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
        let grazing_cos = if dn > 1e-12 {
            (d.dot(&normal) / dn).abs()
        } else {
            0.0
        };
        if dn <= 1e-12 || grazing_cos < params.min_grazing_cos {
            continue;
        }
        // The point's own projection (the keypoint anchor). A view that can't
        // project the point in-frame can't be refined; skip it.
        let Some(proj) = project(view, &patch.center, patch.w) else {
            continue;
        };
        // Seed offset (in `R_s`-grid steps, since `wpp` is at `R_s`): unproject the
        // starting keypoint onto the plane, else zero. A view with no seed of its
        // own (`None` — the expansion candidate that has no observation to take a
        // keypoint from) falls back to that same zero, i.e. to its projection, so
        // an all-`None` table is bit-identical to passing no seeds at all. Split
        // into the integer read accumulator (clipped to the cache's drift bound)
        // and a sub-pixel residual — the residual keeps a lone-view seed exact
        // through `finalize` while the congealing read position stays integer.
        let off = starting_keypoints
            .and_then(|seeds| seeds[k])
            .and_then(|kp| seed_offset(patch, view, kp, wpp_u, wpp_v))
            .unwrap_or([0.0, 0.0]);
        // `off = [u, v]` (u-axis, v-axis components, in `R_s`-grid steps). Split each
        // axis into the clamped integer read accumulator and a pure sub-pixel
        // residual — the residual keeps a lone-view seed exact through `finalize`; a
        // seed beyond `±search` is clamped on the integer part, as the round drift is.
        let iu = (off[0].round() as i64).clamp(-search_steps, search_steps);
        let iv = (off[1].round() as i64).clamp(-search_steps, search_steps);
        if ranking {
            cand_scores.push(match view_scores {
                // A caller score shorter than the view set leaves the tail
                // unscored, which ranks it last rather than panicking.
                Some(s) => s.get(k).copied().unwrap_or(f64::NAN),
                None => grazing_cos,
            });
            cand_is_track.push(k < evidence.track_view_count);
        }
        states.push(ViewState {
            idx: i,
            order: states.len() as u32,
            is_basis: true,
            iacc: [iu, iv],
            residual: [off[0] - off[0].round(), off[1] - off[1].round()],
            proj: [proj.0, proj.1],
            loo: f64::NAN,
        });
    }

    if states.len() < 2 {
        return finalize(patch, views, &states, wpp_u, wpp_v);
    }

    // Consensus-basis pick: hold every view past the cap out of the congealing
    // loop; phase B registers them once against the finished basis template.
    // `K < 2` is raised to 2 — a leave-one-out consensus needs two members.
    let mut tail: Vec<ViewState> = Vec::new();
    let k_cap = params.basis_max_views as usize;
    if k_cap > 0 && states.len() > k_cap {
        let mask = prof::BASIS_PICK.time(|| {
            select_basis(
                &cand_scores,
                &cand_is_track,
                k_cap.max(2),
                params.basis_force_track_views,
                params.basis_pick,
            )
        });
        let mut kept = Vec::with_capacity(k_cap.max(2));
        for (i, mut st) in std::mem::take(&mut states).into_iter().enumerate() {
            if mask[i] {
                kept.push(st);
            } else {
                st.is_basis = false;
                tail.push(st);
            }
        }
        states = kept;
    }
    prof::count(&prof::N_BASIS, states.len() as u64);
    prof::count(&prof::N_TAIL, tail.len() as u64);

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

    let mut loo = LooScratch::default();
    let mut search = SearchScratch::default();
    let mut rounds_run = 0u32;
    for _round in 0..params.max_iters.max(1) {
        prof::count(&prof::N_ROUNDS, 1);
        rounds_run += 1;
        // View count entering this round: convergence below requires the view
        // set to have survived the round unchanged (see step 6).
        let n_entering = states.len();
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
        //
        // The LOO consensus is built **incrementally**: one shared per-round
        // accumulation — the Gram matrix over the live views' z-normalized
        // cores — replaces the per-(view, round) holdout-stack copy + IRLS
        // rebuild; each view's template then needs only a Gram-space IRLS
        // (O(V²) scalars per iteration) plus one pixel-space materialization.
        // See `loo_consensus_template`.
        let nv = live.len();
        prof::TEMPLATE_GRAM.time(|| build_loo_gram(&xs, nv, kept_ch * n, &mut loo));
        let mut shifts: Vec<Option<ShiftResult>> = vec![None; nv];
        for (v, &si) in live.iter().enumerate() {
            // Build the other views' robust consensus template (the
            // leave-one-out reference for view v) from the shared Gram.
            prof::TEMPLATE.time(|| {
                loo_consensus_template(
                    &xs,
                    nv,
                    v,
                    kept_ch,
                    n,
                    params.robust_iters,
                    &mut loo,
                    &mut search.tmpl,
                );
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
        //
        // The convergence metric is the round-over-round CHANGE of each live
        // view's refined position (`iacc + residual` before vs after this round's
        // update). The raw search output `|sh.dx, sh.dy|` is NOT that change: it
        // bundles the freshly recomputed parabolic residual — which never moves
        // the read position — so its magnitude has a ~0.1–0.5-step floor even
        // once the argmax stops moving, and summing it kept `mean_shift` above
        // `convergence_px` forever (every point ran all `max_iters` rounds).
        let mut shift_sum = 0.0;
        for (v, &si) in live.iter().enumerate() {
            let st = &mut states[si];
            match shifts[v] {
                Some(sh) => {
                    let prev = st.offset_steps();
                    st.iacc[0] = (st.iacc[0] + sh.ix).clamp(-search_steps, search_steps);
                    st.iacc[1] = (st.iacc[1] + sh.iy).clamp(-search_steps, search_steps);
                    st.residual = [sh.dx - sh.ix as f64, sh.dy - sh.iy as f64];
                    st.loo = sh.peak;
                    let now = st.offset_steps();
                    shift_sum += (now[0] - prev[0]).hypot(now[1] - prev[1]);
                }
                // No scorable window this round: leave `iacc`/`residual` in place and
                // mark the LOO unknown (matches the pre-cache None handling). The
                // position did not move, so it contributes 0 to the round's shift.
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
        let med = median_in_place(&mut live_loo);
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

        // 6. Converge. `shift_sum` is in `R_s`-grid steps; one step is `1/m`
        //    patch-grid px, so scale before comparing to `convergence_px`.
        //    Positional stability alone is not enough: a view dropped THIS round
        //    (step 5, or gone out-of-frame in step 1) changes the consensus the
        //    survivors registered against, so they get at least one more round to
        //    re-equilibrate against the survivor-only template before the
        //    stationarity test can fire. The `< 2` floor exit stays unconditional
        //    (no leave-one-out consensus exists to re-register against).
        let mean_shift = shift_sum / (live.len() as f64 * m);
        if states.len() < 2 || (states.len() == n_entering && mean_shift < params.convergence_px) {
            break;
        }
    }

    // Phase B — register the tail. One final all-basis consensus template (no
    // holdout: a tail view never contributed to it, so leave-one-out is
    // unnecessary by construction), then one shift search per tail view against
    // it. See `specs/core/patch/keypoint-localization-consensus-basis.md`.
    if !tail.is_empty() {
        prof::TAIL_REGISTER.time(|| {
            register_tail(
                patch,
                views,
                &states,
                &caches,
                &mut tail,
                &support,
                &mut search,
                TailGeometry {
                    resolution,
                    margin,
                    search_steps,
                    cache_c0,
                    wpp_u,
                    wpp_v,
                },
                params,
            );
            states.append(&mut tail);
            // Restore the input view-set order the result contract reports in
            // (basis and tail were each already ascending in `order`).
            states.sort_by_key(|st| st.order);
        });
    }

    let mut out = finalize(patch, views, &states, wpp_u, wpp_v);
    out.rounds = rounds_run;
    out
}

/// The grid geometry phase-B registration needs, bundled so
/// [`register_tail`] keeps a readable signature.
struct TailGeometry {
    /// Search resolution `R_s`.
    resolution: u32,
    /// In-round search radius in `R_s`-grid steps.
    margin: i64,
    /// Bound on the accumulated integer drift.
    search_steps: i64,
    /// Cache index of the `R_s×R_s` core at zero offset in a **basis** cache.
    cache_c0: usize,
    wpp_u: f64,
    wpp_v: f64,
}

/// Build the final all-basis consensus template into `out` from the surviving
/// basis members' cores, read at their final integer offsets. Returns the kept
/// channel count and mask, or `None` when fewer than two basis cores are still
/// in frame or no channel carries texture (no template to register against).
fn basis_template(
    states: &[ViewState],
    caches: &[ContextTile],
    support: &Support,
    geom: &TailGeometry,
    robust_iters: u32,
    out: &mut Vec<f32>,
) -> Option<(usize, Vec<bool>)> {
    let r = geom.resolution as usize;
    let n = support.pixels.len();
    let mut raws: Vec<Vec<f32>> = Vec::with_capacity(states.len());
    let mut live_channels: Vec<usize> = Vec::with_capacity(states.len());
    for (si, st) in states.iter().enumerate() {
        let cache = &caches[si];
        let mut raw = vec![0f32; cache.channels * n];
        let ox = (geom.cache_c0 as i64 + st.iacc[0]) as usize;
        let oy = (geom.cache_c0 as i64 + st.iacc[1]) as usize;
        if extract_core(cache, support, r, oy, ox, &mut raw) {
            raws.push(raw);
            live_channels.push(cache.channels);
        }
    }
    if raws.len() < 2 {
        return None;
    }
    let channels0 = *live_channels.iter().min().unwrap();
    let mut flat = vec![0f32; raws.len() * channels0 * n];
    for (vk, raw) in raws.iter().enumerate() {
        flat[vk * channels0 * n..][..channels0 * n].copy_from_slice(&raw[..channels0 * n]);
    }
    let mut xs = Vec::new();
    let (kept_ch, keep_mask) = prof::ZNORM.time(|| {
        znormalize_into_kept(
            &flat,
            raws.len(),
            channels0,
            n,
            &support.weights,
            support.total_weight,
            &support.sqrt_weights,
            &mut xs,
        )
    })?;
    // Same robust consensus as a congealing round, without a holdout.
    prof::TEMPLATE.time(|| {
        let mut sc = ConsensusScratch::default();
        irls_view_weights(&xs, raws.len(), kept_ch, n, robust_iters, None, &mut sc);
        weighted_unit_template_into(&xs, &sc.w, raws.len(), kept_ch, n, out);
    });
    Some((kept_ch, keep_mask))
}

/// Whether a view's refined keypoint is close enough to the point's projection
/// to keep — the `max_shift_px` half of the drop gates, shared by the scored
/// path and the no-template early exit so a tail view is never emitted without
/// at least this check.
fn within_max_shift(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    st: &ViewState,
    wpp_u: f64,
    wpp_v: f64,
    max_shift_px: f64,
) -> bool {
    let off = st.offset_steps();
    let center = shifted_center(patch, off[0], off[1], wpp_u, wpp_v);
    let shift_px = match project(view, &center, patch.w) {
        Some((x, y)) => (x - st.proj[0]).hypot(y - st.proj[1]),
        None => f64::INFINITY, // keypoint left the frame
    };
    shift_px <= max_shift_px
}

/// Phase B: register every `tail` view once against the basis consensus and
/// apply the per-view drop gates, leaving `tail` holding only the kept views.
///
/// Each tail view renders a cache centred on its **own** seed offset, sized
/// `R_s + 2·margin` — it searches one `±margin` window around that seed and so
/// needs no drift headroom (basis caches keep the `R_s + 4·margin` sizing that
/// covers a whole round loop). The gates are the loop's verbatim: drop a view
/// whose refined keypoint sits more than `max_shift_px` from the projection, or
/// whose ZNCC falls below `min_relative_zncc ×` the **basis members'** median
/// final ZNCC (the same threshold rule as the round loop, measured against a
/// different reference — the no-holdout basis template).
///
/// **Mixed channel counts.** The template's channel space is the one
/// [`basis_template`] built, i.e. the minimum over the *basis* caches. A tail
/// view can be narrower than that (a grayscale frame among colour ones), and
/// its tile then has no plane for the template's trailing channels. The round
/// loop's rule is "score in the channel space common to the views taking part",
/// so the tail applies the same rule pairwise: the mask is truncated to the
/// tail tile's own width. Only trailing *original* channels drop out, so the
/// surviving kept channels are a prefix of the template's rows and the template
/// needs no rebuild. A tail view with no kept channel left is unscorable and is
/// dropped by the gates below, exactly like one whose window is out of frame.
#[allow(clippy::too_many_arguments)]
fn register_tail(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: &[ViewState],
    caches: &[ContextTile],
    tail: &mut Vec<ViewState>,
    support: &Support,
    search: &mut SearchScratch,
    geom: TailGeometry,
    params: &KeypointLocalizeParams,
) {
    let Some((kept_ch, keep_mask)) = basis_template(
        states,
        caches,
        support,
        &geom,
        params.robust_iters,
        &mut search.tmpl,
    ) else {
        // No usable basis consensus (the loop collapsed below two in-frame
        // views, or no channel carries texture): there is nothing to register
        // against, so the tail keeps its seed offsets with an unknown ZNCC. The
        // agreement gate cannot be evaluated without a template, but the
        // positional one can and still must be — a seed can already sit further
        // than `max_shift_px` from the projection, and nothing downstream would
        // catch it.
        prof::count(&prof::N_TAIL_NO_BASIS, tail.len() as u64);
        tail.retain(|st| {
            within_max_shift(
                patch,
                &views[st.idx as usize],
                st,
                geom.wpp_u,
                geom.wpp_v,
                params.max_shift_px,
            )
        });
        return;
    };

    // The tail's relative-agreement bar, from the basis members' final ZNCCs.
    let mut basis_loo: Vec<f64> = states
        .iter()
        .map(|st| st.loo)
        .filter(|z| z.is_finite())
        .collect();
    let med = median_in_place(&mut basis_loo);
    let bar = if med.is_finite() {
        params.min_relative_zncc * med
    } else {
        f64::NEG_INFINITY
    };

    let r = geom.resolution as usize;
    let tail_res = geom.resolution + 2 * geom.margin as u32;
    // The tail cache is centred on the view's seed, so the `(0, 0)` shift reads
    // its core at `margin`.
    let tail_c0 = geom.margin as usize;
    prof::count(&prof::N_RENDER, tail.len() as u64);
    for st in tail.iter_mut() {
        let view = &views[st.idx as usize];
        let cache = prof::RENDER.time(|| {
            render_context(
                patch,
                view,
                st.iacc[0] as f64,
                st.iacc[1] as f64,
                geom.wpp_u,
                geom.wpp_v,
                geom.resolution,
                tail_res,
                params.sampler,
            )
        });
        // Score in the channel space this tail tile actually has (see the
        // "Mixed channel counts" note above); `sub_mask` is a prefix of the
        // template's mask, so `search.tmpl`'s leading rows still line up.
        let sub_mask = &keep_mask[..keep_mask.len().min(cache.channels)];
        let sub_kept = sub_mask.iter().filter(|&&k| k).count();
        // Truncating the mask can only remove kept channels, and only trailing
        // ones — so the template's leading `sub_kept` rows are the right ones.
        debug_assert!(sub_kept <= kept_ch);
        if sub_kept == 0 {
            // No channel the template scores on survives in this view.
            st.loo = f64::NAN;
            continue;
        }
        prof::count(&prof::N_SEARCH, 1);
        let sh = prof::SEARCH.time(|| match params.search_strategy {
            SearchStrategy::Exhaustive => search_shift(
                &cache,
                search,
                support,
                sub_mask,
                sub_kept,
                r,
                geom.margin,
                tail_c0,
                tail_c0,
            ),
            SearchStrategy::PlusDescent => search_shift_plus_descent(
                &cache,
                search,
                support,
                sub_mask,
                sub_kept,
                r,
                geom.margin,
                tail_c0,
                tail_c0,
            ),
        });
        match sh {
            Some(sh) => {
                st.iacc[0] = (st.iacc[0] + sh.ix).clamp(-geom.search_steps, geom.search_steps);
                st.iacc[1] = (st.iacc[1] + sh.iy).clamp(-geom.search_steps, geom.search_steps);
                st.residual = [sh.dx - sh.ix as f64, sh.dy - sh.iy as f64];
                st.loo = sh.peak;
            }
            // No scorable window: the view's core is out of frame at its seed.
            None => st.loo = f64::NAN,
        }
    }

    tail.retain(|st| {
        within_max_shift(
            patch,
            &views[st.idx as usize],
            st,
            geom.wpp_u,
            geom.wpp_v,
            params.max_shift_px,
        ) && st.loo.is_finite()
            && st.loo >= bar
    });
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
        // The plane must be hit FORWARD along the bearing. Under a perspective
        // camera the upstream cheirality gates already guarantee `s > 0`, so
        // this only ever bites on a ray-path model, where a periphery bearing
        // can meet the plane behind the camera centre — a mirrored "hit" that
        // is not an observation of this patch at all.
        if view.camera.model.needs_ray_path() && s <= 0.0 {
            return None;
        }
        let hit = cam_c + dir * s;
        hit - patch.center
    };
    // Grid rows count downward from `+v̂` (they map to `−v_axis`), so the
    // v-grid coordinate negates the in-plane `v̂` component — the inverse of
    // `shifted_center`.
    Some([
        off.dot(&patch.u_axis) / wpp_u,
        -off.dot(&patch.v_axis) / wpp_v,
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
        out.is_basis.push(st.is_basis);
    }
    out
}

/// Per-patch ranking evidence for the consensus-basis pick, parallel to the
/// cloud's patches — the batch form of [`BasisEvidence`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BasisInputs<'a> {
    /// Per patch, the per-view scores parallel to that patch's view set. An
    /// **empty** entry means that patch is unscored — its basis pick falls back
    /// to the grazing rank, exactly as if no scores had been supplied at all.
    pub view_scores: Option<&'a [Vec<f64>]>,
    /// Per patch, how many leading view-set entries are track views.
    pub track_view_counts: Option<&'a [u32]>,
}

/// Batch [`localize_patch_keypoints`] over a [`PatchCloud`], parallel across
/// patches (rayon). `view_sets[i]` lists, for patch `i`, the views to refine
/// (typically the output of view selection). `starting_keypoints`, when given, is
/// parallel to `view_sets` in both dimensions (one **optional** seed per view);
/// `None` seeds every view at the point's projection, and so does an **empty**
/// per-patch entry — the batch form has no per-patch `Option`, so it says "this
/// patch is unseeded" with an empty list, exactly as [`BasisInputs::view_scores`]
/// says "unscored" — and so does a `None` for one view inside a seeded patch's
/// list. `basis`, when
/// given, carries the per-patch ranking
/// evidence the [consensus-basis cap](KeypointLocalizeParams::basis_max_views)
/// consumes (unread when the cap is off). Results are returned in cloud order.
///
/// # Panics
///
/// Panics if `view_sets.len() != cloud.len()` (or `starting_keypoints` / the
/// `basis` arrays are given and not parallel), or an index is out of range.
pub fn localize_patch_cloud_keypoints(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    view_sets: &[Vec<u32>],
    starting_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    basis: Option<&BasisInputs<'_>>,
    params: &KeypointLocalizeParams,
    progress: Option<&std::sync::atomic::AtomicUsize>,
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
    if let Some(b) = basis {
        if let Some(s) = b.view_scores {
            assert_eq!(
                s.len(),
                cloud.len(),
                "view_scores must be parallel to the cloud"
            );
        }
        if let Some(t) = b.track_view_counts {
            assert_eq!(
                t.len(),
                cloud.len(),
                "track_view_counts must be parallel to the cloud"
            );
        }
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
            let seeds = starting_keypoints
                .map(|s| s[i].as_slice())
                .filter(|s| !s.is_empty());
            let evidence = BasisEvidence {
                view_scores: basis.and_then(|b| b.view_scores).map(|s| s[i].as_slice()),
                track_view_count: basis
                    .and_then(|b| b.track_view_counts)
                    .map_or(0, |t| t[i] as usize),
            };
            let out = prof::TOTAL.time(|| {
                localize_patch_keypoints_with_basis(
                    patch,
                    views,
                    &view_sets[i],
                    seeds,
                    evidence,
                    params,
                )
            });
            // Bump the shared work counter per patch for a Python progress poller.
            if let Some(c) = progress {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            out
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
