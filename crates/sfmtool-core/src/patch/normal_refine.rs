// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Photometric patch-normal refinement.
//!
//! See `specs/core/patch-normal-refinement.md`. Given an [`OrientedPatch`] and
//! the views that observe it, [`refine_patch_normal`] searches the 2-DOF
//! normal (the only thing that affects cross-view consistency) for the plane
//! whose rendered patches agree the most, scored by the consensus all-pairs
//! mean ZNCC. [`refine_patch_cloud`] batches the routine over a [`PatchCloud`]
//! in parallel.

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::camera::remap::{remap_aniso_with_pyramid, remap_bilinear, ImageU8Pyramid};
use crate::camera::CameraIntrinsics;
use crate::camera::WarpMap;
use crate::geometry::RigidTransform;
use crate::patch::cloud::{mean_viewing_normal, OrientedPatch, PatchCloud};
use crate::reconstruction::SfmrReconstruction;

mod fronto_cache;
pub mod prof;

/// A source image together with the projection that maps world points into it:
/// the camera intrinsics, its world-to-camera pose, and a prebuilt pyramid for
/// sampling colour. Everything a patch needs to be rendered from one view. The
/// pyramid is built once and borrowed for every candidate render, so a refinement
/// allocates no per-candidate source image data.
#[derive(Clone, Copy)]
pub struct ProjectedImage<'a> {
    pub camera: &'a CameraIntrinsics,
    pub cam_from_world: &'a RigidTransform,
    pub pyramid: &'a ImageU8Pyramid,
}

/// Photoconsistency `Φ`: the consensus all-pairs mean ZNCC (see the spec's
/// "Objective" for the closed form and why there is no reference-view
/// variant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Unweighted consensus `ρ̄ = (V‖x̄‖² − 1)/(V − 1)`.
    MeanPairwise,
    /// IRLS-weighted consensus that down-weights outlier (occluded /
    /// wrong-surface) views by a Tukey weight on each view's residual
    /// `‖xᵢ − x̄_w‖`, re-weighting `iters` times. Recommended default.
    RobustWeighted { iters: u32 },
}

/// Per-pixel weight applied to the `R×R` patch when scoring (the NCC window).
///
/// `sigma` is expressed in normalized patch coordinates (the patch spans
/// `(s, t) ∈ [-1, 1]²`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchWindow {
    /// Uniform weight over the whole square grid (rotation-leaky; mainly a
    /// baseline).
    Uniform,
    /// Gaussian center weight over the square grid.
    Gaussian { sigma: f64 },
    /// Gaussian weight confined to the inscribed disk — radial, so in-plane
    /// rotation is exactly free and grazing corners don't leak in.
    /// Recommended default.
    GaussianDisk { sigma: f64 },
    // Beyond-v1: `Alpha` — an explicit per-(s, t) mask carried by the patch
    // (spec item 6); deferred until a producer exists.
}

/// How to sample a [`ProjectedImage`]'s pyramid when rendering a patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// Plain bilinear from the full-resolution level. The **default**: it barely
    /// moves the found normal (within ~1° of anisotropic on pinhole cameras) at a
    /// fraction of the cost.
    Bilinear,
    /// Anisotropic sampling over the pyramid — the patch warp's Jacobian SVD picks
    /// the level, de-aliasing oblique / grazing views. Costs ~1.6–3× more but keeps
    /// the reported `Φ`/confidence unbiased (bilinear depresses `Φ` on oblique
    /// views) and helps slightly on distorted/fisheye rigs.
    Anisotropic,
}

/// How candidate normals are scored: re-rendered from the source images, or
/// resampled from a cached base patch (see
/// `specs/core/fronto-parallel-patch-cache.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Re-render every candidate from the source pyramid — the exact path; opt
    /// in for the tightest tail when wall time doesn't matter.
    Off,
    /// Render one supersampled fronto-parallel base per view up front and
    /// affine-resample each candidate from it. **The default** (with
    /// `cache_supersample = 2`): ~2× faster at Φ-equivalent *median* accuracy;
    /// the affine approximation costs a little in the tail on flat-`Φ` data. Note
    /// the returned `photoconsistency` is always source-scored in the final pass,
    /// and `≥ init_photoconsistency` by construction, regardless of the cache.
    /// Falls back to `Off` per-patch when the base can't be built (e.g. non-RGB
    /// imagery).
    FrontoParallel,
}

/// Tunables for [`refine_patch_normal`].
#[derive(Debug, Clone)]
pub struct NormalRefineParams {
    /// Half-extent of the level-0 search cone, degrees.
    pub angular_range_deg: f64,
    /// Grid resolution per tangent axis at each level.
    pub init_steps: u32,
    /// Coarse-to-fine passes; each recenters on the level's best normal and
    /// shrinks the cone to one previous grid spacing.
    pub refine_levels: u32,
    /// Consensus objective.
    pub objective: Objective,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// Per-view floor on the window-weighted valid-pixel fraction; views below
    /// it are dropped for the level.
    pub min_valid_fraction: f64,
    /// Minimum number of (effective) views; below it the candidate / patch is
    /// not refined.
    pub min_views: u32,
    /// How to sample the source pyramids when rendering candidates.
    pub sampler: Sampler,
    /// Candidate-scoring strategy (source re-render vs. fronto-parallel cache).
    pub cache: CacheMode,
    /// Base-patch supersample factor for the cache (≥ 1; `1.0` = candidate
    /// density). Denser bases sharpen the resample and tighten accuracy at the
    /// cost of a bigger up-front render. Ignored when `cache == Off`.
    pub cache_supersample: f64,
    /// Whether to compute the per-patch confidence (the `Φ`-peakedness stencil).
    /// Off by default: it is an un-cached extra source-render pass per patch
    /// (~⅙ of the cached runtime) and is purely informational — when `false`,
    /// [`NormalRefineResult::confidence`] is `NaN`.
    pub compute_confidence: bool,
    /// Cheaper consensus objective for the coarse-to-fine *search* ranking only
    /// (the bulk of the consensus evaluations). The final pass that picks the
    /// winner and reports `Φ` always uses [`objective`](Self::objective), so the
    /// returned `photoconsistency` stays honest regardless of this knob. `None`
    /// (the default) uses `objective` for the search too; `Some(0)` searches
    /// with [`Objective::MeanPairwise`]; `Some(k)` searches with
    /// `RobustWeighted { iters: k }`. Lowering it trades a little tail accuracy
    /// in the found normal for a faster search. It only *reduces* search cost
    /// relative to a robust [`objective`](Self::objective); under
    /// `objective = MeanPairwise` the final pass is already the cheapest, so the
    /// knob has no benefit (and `Some(k ≥ 1)` would make the search dearer than
    /// the final pass).
    pub search_robust_iters: Option<u32>,
    /// Whether to render the per-patch RGBA representative texture
    /// ([`NormalRefineResult::representative`]) at the found normal. Off by
    /// default: it is one extra full-grid source render per kept view per patch
    /// (the search and scoring only touch the masked common support), so it is
    /// computed only when a caller wants to persist the patch bitmaps. When
    /// `false`, [`NormalRefineResult::representative`] is `None`.
    pub render_bitmap: bool,
}

impl NormalRefineParams {
    /// The consensus objective used to rank candidates during the
    /// coarse-to-fine search. Derived from
    /// [`search_robust_iters`](Self::search_robust_iters); the final winner /
    /// `Φ` pass always uses [`objective`](Self::objective).
    fn search_objective(&self) -> Objective {
        match self.search_robust_iters {
            None => self.objective,
            Some(0) => Objective::MeanPairwise,
            Some(iters) => Objective::RobustWeighted { iters },
        }
    }
}

impl Default for NormalRefineParams {
    fn default() -> Self {
        Self {
            angular_range_deg: 25.0,
            init_steps: 7,
            refine_levels: 3,
            objective: Objective::RobustWeighted { iters: 3 },
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            min_valid_fraction: 0.6,
            min_views: 3,
            sampler: Sampler::Bilinear,
            cache: CacheMode::FrontoParallel,
            cache_supersample: 2.0,
            compute_confidence: false,
            search_robust_iters: None,
            render_bitmap: false,
        }
    }
}

/// Result of [`refine_patch_normal`].
#[derive(Debug, Clone)]
pub struct NormalRefineResult {
    /// The input patch with its normal replaced by the optimum; `center`,
    /// `half_extent`, and the in-plane convention are preserved (the input
    /// `u_axis` is reprojected onto the new plane, `v = n × u`).
    pub patch: OrientedPatch,
    /// Consensus photoconsistency `Φ` at the returned normal (NaN if it could
    /// not be evaluated).
    pub photoconsistency: f64,
    /// `Φ` at the input normal under the same frozen support (NaN if it could
    /// not be evaluated).
    pub init_photoconsistency: f64,
    /// Number of views kept by the validity gates at the final evaluation.
    pub valid_view_count: u32,
    /// Peakedness of `Φ` at the optimum: the smaller eigenvalue of the
    /// finite-difference curvature of `Φ(δ)`, normalized against the larger
    /// one (see [`grid_confidence`] internals). `0` when the patch was not
    /// refined or `Φ` is flat (e.g. the narrow-baseline degeneracy). `NaN` when
    /// not requested ([`NormalRefineParams::compute_confidence`] is `false`).
    pub confidence: f64,
    /// The canonical robust-template appearance at the found normal: a fused
    /// `R×R` RGBA texture, flat in row-major `(row, col, channel)` order (length
    /// `R·R·4`). RGB is the cross-view fused colour (a robust IRLS-weighted mean
    /// under [`Objective::RobustWeighted`], an unweighted mean under
    /// [`Objective::MeanPairwise`]); the alpha channel is a per-pixel cross-view
    /// agreement confidence (`0` where no kept view covers the pixel). `None`
    /// when [`NormalRefineParams::render_bitmap`] is `false`, or the patch was
    /// not refined (too few valid views).
    pub representative: Option<Vec<u8>>,
}

/// Windowed norm² below which a colour channel counts as flat (no texture
/// signal); flat channels are dropped from the consensus so the z-normalized
/// identity stays well-defined.
pub(super) const FLAT_NORM_SQ_EPS: f64 = 1e-6;

/// Minimum number of commonly-valid pixels for a support to be scoreable.
const MIN_MASK_PIXELS: usize = 8;

/// Minimum robust *effective* view count `1/Σwᵢ²` for the weighted consensus to
/// be meaningful (a pair) and avoid the `0/0` at `Σwᵢ² → 1`. This is a degeneracy
/// floor only — the *count* of kept views is gated separately by `min_views`, so
/// a clean `V == min_views` track (weights near- but not exactly uniform, hence
/// `1/Σwᵢ² < V`) still scores.
const MIN_EFFECTIVE_VIEWS: f64 = 2.0;

/// `remap_aniso` sample cap along the major axis.
const MAX_ANISOTROPY: u32 = 16;

// ---------------------------------------------------------------------------
// Parameterization: deterministic tangent basis + exponential map
// ---------------------------------------------------------------------------

/// Deterministic orthonormal tangent basis of the unit normal `n` — a pure
/// function of `n` (least-aligned world axis + Gram-Schmidt), so refinements
/// are reproducible.
pub fn tangent_basis(n: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let n = n.normalize();
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    let a = if ax <= ay && ax <= az {
        Vector3::x()
    } else if ay <= az {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let u = (a - n * a.dot(&n)).normalize();
    let v = n.cross(&u);
    (u, v)
}

/// Exponential map on the sphere: tilt the unit normal `n0` by angle `‖δ‖`
/// toward the tangent direction `δ` (expressed in [`tangent_basis`]`(n0)`):
/// `n(δ) = cos‖δ‖·n₀ + sin‖δ‖·δ̂`. Angle-uniform — equal steps in `δ` are
/// equal angles.
pub fn exp_map_normal(n0: &Vector3<f64>, delta: [f64; 2]) -> Vector3<f64> {
    let n = n0.normalize();
    let (u, v) = tangent_basis(&n);
    exp_map_in_basis(&n, &u, &v, delta)
}

fn exp_map_in_basis(
    n0: &Vector3<f64>,
    u: &Vector3<f64>,
    v: &Vector3<f64>,
    delta: [f64; 2],
) -> Vector3<f64> {
    let theta = delta[0].hypot(delta[1]);
    if theta < 1e-12 {
        return *n0;
    }
    let dir = (u * delta[0] + v * delta[1]) / theta;
    n0 * theta.cos() + dir * theta.sin()
}

// ---------------------------------------------------------------------------
// Window / support
// ---------------------------------------------------------------------------

/// Per-pixel window weight over the `R×R` patch grid, in row-major order.
pub(super) fn window_weights(window: PatchWindow, resolution: u32) -> Vec<f64> {
    let r = resolution as usize;
    let step = 2.0 / r as f64;
    let mut w = Vec::with_capacity(r * r);
    for row in 0..r {
        let t = (row as f64 + 0.5) * step - 1.0;
        for col in 0..r {
            let s = (col as f64 + 0.5) * step - 1.0;
            let r2 = s * s + t * t;
            let weight = match window {
                PatchWindow::Uniform => 1.0,
                PatchWindow::Gaussian { sigma } => (-r2 / (2.0 * sigma * sigma)).exp(),
                PatchWindow::GaussianDisk { sigma } => {
                    if r2 > 1.0 {
                        0.0
                    } else {
                        (-r2 / (2.0 * sigma * sigma)).exp()
                    }
                }
            };
            w.push(weight);
        }
    }
    w
}

/// Rebuild the patch on a new plane: same `center` / `half_extent`, the input
/// `u_axis` reprojected onto the plane of `n` (`v = n × u`).
pub(super) fn repose_patch(base: &OrientedPatch, n: &Vector3<f64>) -> OrientedPatch {
    let mut p = OrientedPatch::from_center_normal(base.center, *n, base.u_axis, base.half_extent);
    p.w = base.w;
    p
}

// ---------------------------------------------------------------------------
// Validity / common support
// ---------------------------------------------------------------------------

/// The frozen scoring support of one grid level: the kept views and the
/// commonly-valid masked pixels (computed at the level's *center* normal and
/// held fixed across that level's candidates, so `Φ` stays continuous and
/// tilts can't shrink the support onto an easy region).
pub(super) struct LevelContext {
    /// Indices into the caller's `views` slice.
    pub(super) kept: Vec<usize>,
    /// Linear `row * R + col` indices of the masked pixels.
    pub(super) pixels: Vec<usize>,
    /// Window weight per masked pixel (parallel to `pixels`).
    pub(super) weights: Vec<f64>,
}

/// Per-pixel validity of `patch` in `view` (window support only).
fn view_valid_mask(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    resolution: u32,
    w_full: &[f64],
) -> Vec<bool> {
    let map = prof::MASK
        .time(|| WarpMap::from_patch(patch, view.camera, view.cam_from_world, resolution));
    let r = resolution;
    let mut mask = vec![false; (r as usize) * (r as usize)];
    for row in 0..r {
        for col in 0..r {
            let p = (row * r + col) as usize;
            if w_full[p] > 0.0 && map.is_valid(col, row) {
                mask[p] = true;
            }
        }
    }
    mask
}

/// Build the frozen support at `center_n`: cull back-facing views, gate each
/// view on its window-weighted valid fraction, intersect the survivors'
/// validity. `None` when fewer than `min_views` views (or too few pixels)
/// survive.
pub(super) fn build_level_context(
    base: &OrientedPatch,
    center_n: &Vector3<f64>,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    w_full: &[f64],
    params: &NormalRefineParams,
) -> Option<LevelContext> {
    let patch = repose_patch(base, center_n);
    let support_mass: f64 = w_full.iter().filter(|&&w| w > 0.0).sum();
    if support_mass <= 0.0 {
        return None;
    }

    let mut kept = Vec::new();
    let mut masks: Vec<Vec<bool>> = Vec::new();
    for (i, view) in views.iter().enumerate() {
        // Back-face cull before building any warp map.
        if !patch.is_front_facing(view.cam_from_world) {
            continue;
        }
        let mask = view_valid_mask(&patch, view, resolution, w_full);
        let mass: f64 = mask
            .iter()
            .zip(w_full)
            .filter(|(&m, _)| m)
            .map(|(_, &w)| w)
            .sum();
        if mass / support_mass >= params.min_valid_fraction {
            kept.push(i);
            masks.push(mask);
        }
    }

    if kept.len() < params.min_views.max(2) as usize {
        return None;
    }

    let mut pixels = Vec::new();
    let mut weights = Vec::new();
    for p in 0..w_full.len() {
        if w_full[p] > 0.0 && masks.iter().all(|m| m[p]) {
            pixels.push(p);
            weights.push(w_full[p]);
        }
    }
    if pixels.len() < MIN_MASK_PIXELS {
        return None;
    }
    Some(LevelContext {
        kept,
        pixels,
        weights,
    })
}

// ---------------------------------------------------------------------------
// Objective
// ---------------------------------------------------------------------------

/// Render the kept views of `patch` and z-normalize each colour channel over
/// the windowed common support. Returns `xs[view][channel][pixel]`, each
/// channel vector unit-norm and zero weighted-mean (window weights are folded
/// in as `√w`, so plain dot products realize the windowed inner product).
/// Channels that are flat (windowed norm ≈ 0) in *any* kept view are dropped
/// for every view, keeping all inner products in one space; `None` when no
/// channel survives.
pub(super) fn normalized_stack(
    patch: &OrientedPatch,
    ctx: &LevelContext,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    sampler: Sampler,
) -> Option<(Vec<f32>, usize)> {
    let n_views = ctx.kept.len();
    let channels = ctx
        .kept
        .iter()
        .map(|&i| views[i].pyramid.level(0).channels() as usize)
        .min()?;
    if channels == 0 {
        return None;
    }
    let n = ctx.pixels.len();
    let r = resolution as usize;

    prof::count(&prof::N_RENDER, n_views as u64);

    // Raw masked pixel values, flat `[(view*channels + channel)*n + pixel]` in
    // `ctx.pixels` order — the same layout the cache fills and the consensus reads.
    let mut raw = vec![0f32; n_views * channels * n];
    for (vk, &vi) in ctx.kept.iter().enumerate() {
        let view = &views[vi];
        let mut map = prof::WARP
            .time(|| WarpMap::from_patch(patch, view.camera, view.cam_from_world, resolution));
        // Reject a candidate that pushes any frozen-support pixel out of frame in
        // any kept view: such a pixel renders as 0 (zero-fill on a NaN warp), and
        // several views going black at the same pixels fake cross-view agreement,
        // biasing the level's argmax toward frame-edge tilts. Rejecting keeps Φ
        // comparable across the level over the exact frozen mask; the final pass
        // re-intersects each winner's validity separately.
        if ctx
            .pixels
            .iter()
            .any(|&p| !map.is_valid((p % r) as u32, (p / r) as u32))
        {
            prof::count(&prof::N_REJECT, 1);
            return None;
        }
        let img = match sampler {
            Sampler::Anisotropic => {
                prof::SVD.time(|| map.compute_svd());
                prof::REMAP.time(|| remap_aniso_with_pyramid(view.pyramid, &map, MAX_ANISOTROPY))
            }
            Sampler::Bilinear => prof::REMAP.time(|| remap_bilinear(view.pyramid.level(0), &map)),
        };
        prof::ZNORM.time(|| {
            for c in 0..channels {
                let base = (vk * channels + c) * n;
                for (ki, &p) in ctx.pixels.iter().enumerate() {
                    raw[base + ki] = img.get_pixel((p % r) as u32, (p / r) as u32, c as u32) as f32;
                }
            }
        });
    }

    Some((raw, channels))
}

/// z-normalize the flat raw stack `raw[(view*channels + channel)*n + pixel]`
/// (f32) over the windowed support into `out` (flat, channels compacted to the
/// kept ones), reusing `out`'s capacity. Subtract the weighted mean, scale to
/// unit windowed-norm, and fold `√w` in (so plain dot products realize the
/// windowed inner product). A channel flat (windowed norm² < [`FLAT_NORM_SQ_EPS`])
/// in *any* view is dropped for *every* view. The per-(view, channel) moments
/// are reduced in **f64** SIMD: the variance `norm_sq = s2 − s1·mean` is
/// cancellation-sensitive and feeds Φ directly (the stack is the consensus
/// input), so f32 accumulation there moves normals. The element-wise normalize
/// is f32 and the stack is stored f32 to halve the memory traffic the consensus
/// re-reads. Returns the kept-channel count, or `None` when none survive. Shared
/// by the source-render path ([`normalized_stack`]) and the fronto cache
/// ([`fronto_cache::eval_phi`]) so the two cannot drift.
#[allow(clippy::too_many_arguments)]
pub(super) fn znormalize_into(
    raw: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    weights: &[f64],
    total_w: f64,
    sqrt_w: &[f32],
    out: &mut Vec<f32>,
) -> Option<usize> {
    znormalize_into_kept(raw, views, channels, n, weights, total_w, sqrt_w, out)
        .map(|(kept, _)| kept)
}

/// Like [`znormalize_into`], but also returns the per-*original*-channel keep
/// mask (length `channels`, parallel to the input channels): `true` where the
/// channel was textured in every view and survives into the compacted output,
/// `false` where it was dropped as flat. View selection needs this to anchor a
/// candidate's score to the *reference's* surviving original channels (so it
/// never correlates the reference's channel A against a candidate's channel B);
/// refinement ignores it. Numerically identical to [`znormalize_into`].
#[allow(clippy::too_many_arguments)]
pub(super) fn znormalize_into_kept(
    raw: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    weights: &[f64],
    total_w: f64,
    sqrt_w: &[f32],
    out: &mut Vec<f32>,
) -> Option<(usize, Vec<bool>)> {
    let mut keep = vec![true; channels];
    // (mean, inv_norm) per (view, channel), as f32 for the normalize kernel.
    let mut stats = vec![(0.0f32, 0.0f32); views * channels];
    for v in 0..views {
        for c in 0..channels {
            let col = &raw[(v * channels + c) * n..][..n];
            // Moments reduced in f64 (vectorized): the variance difference
            // `s2 − s1·mean` is cancellation-sensitive and feeds Φ directly (it is
            // the consensus input, unlike the IRLS residual), so f32 accumulation
            // here moves normals. A (near-)constant channel cancels below
            // `FLAT_NORM_SQ_EPS` and is dropped by the flat gate (shared across
            // views) before `1/√norm_sq` is taken, so it never reaches a NaN.
            let (s1, s2) = weighted_moments(col, weights);
            let mean = s1 / total_w;
            let norm_sq = s2 - s1 * mean;
            if norm_sq < FLAT_NORM_SQ_EPS {
                keep[c] = false;
            }
            let inv_norm = if norm_sq > 0.0 {
                1.0 / norm_sq.sqrt()
            } else {
                0.0
            };
            stats[v * channels + c] = (mean as f32, inv_norm as f32);
        }
    }
    let kept = keep.iter().filter(|&&k| k).count();
    if kept == 0 {
        out.clear();
        return None;
    }
    out.resize(views * kept * n, 0.0);
    for v in 0..views {
        let mut kc = 0;
        for c in 0..channels {
            if !keep[c] {
                continue;
            }
            let (mean, inv_norm) = stats[v * channels + c];
            let src = &raw[(v * channels + c) * n..][..n];
            let base = (v * kept + kc) * n;
            znorm_write(src, sqrt_w, mean, inv_norm, &mut out[base..base + n]);
            kc += 1;
        }
    }
    Some((kept, keep))
}

/// Horizontal sum of a 4-lane f64 vector.
///
/// # Safety
/// Requires `avx` (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum256_pd(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let s2 = _mm_add_pd(lo, hi);
    let s1 = _mm_add_sd(s2, _mm_unpackhi_pd(s2, s2));
    _mm_cvtsd_f64(s1)
}

/// Weighted moments `(Σ w·x, Σ w·x²)` of `col` over the windowed support,
/// accumulated in **f64**: the caller forms the cancellation-sensitive variance
/// `s2 − s1·mean` from these and feeds it (via `1/√norm_sq`) straight into Φ, so
/// f32 accumulation here moves normals (measured). The baseline target limits the
/// autovectorized fallback to SSE, so an AVX2+FMA kernel (4-wide f64, widening the
/// f32 `col`) is dispatched at runtime; same pattern as [`sum_sq_diff`].
/// `pub(super)` re-export of [`weighted_moments`] for view selection, which
/// computes a candidate channel's windowed mean / norm directly (to score it on
/// the *reference's* surviving channels) rather than going through the
/// compacting [`znormalize_into`]. Same `(Σw·x, Σw·x²)` convention.
#[inline]
pub(super) fn weighted_moments_pub(col: &[f32], w: &[f64]) -> (f64, f64) {
    weighted_moments(col, w)
}

#[inline]
fn weighted_moments(col: &[f32], w: &[f64]) -> (f64, f64) {
    debug_assert_eq!(col.len(), w.len());
    let n = col.len().min(w.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check above.
            return unsafe { weighted_moments_avx2(col, w, n) };
        }
    }
    weighted_moments_scalar(col, w, 0, n)
}

/// Scalar reference for [`weighted_moments`] over `[i0, i1)` (also the AVX2 tail).
/// Four independent f64 lane sums keep the fallback SSE-vectorizable.
fn weighted_moments_scalar(col: &[f32], w: &[f64], i0: usize, i1: usize) -> (f64, f64) {
    const LANES: usize = 4;
    let mut s1 = [0f64; LANES];
    let mut s2 = [0f64; LANES];
    let body = i0 + (i1 - i0) / LANES * LANES;
    let mut i = i0;
    while i < body {
        for l in 0..LANES {
            let f = col[i + l] as f64;
            let wx = w[i + l] * f;
            s1[l] += wx;
            s2[l] += wx * f;
        }
        i += LANES;
    }
    let mut a1: f64 = s1.iter().sum();
    let mut a2: f64 = s2.iter().sum();
    for k in i..i1 {
        let f = col[k] as f64;
        let wx = w[k] * f;
        a1 += wx;
        a2 += wx * f;
    }
    (a1, a2)
}

/// AVX2+FMA reduction of the weighted moments over the first `n` elements (4
/// f64/iteration; the 4 f32 `col` lanes are widened with one `vcvtps2pd`). The
/// `n % 4` tail falls back to [`weighted_moments_scalar`].
///
/// # Safety
/// Requires the `avx2` + `fma` target features (guarded by the dispatcher).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn weighted_moments_avx2(col: &[f32], w: &[f64], n: usize) -> (f64, f64) {
    use std::arch::x86_64::*;
    let mut a1 = _mm256_setzero_pd();
    let mut a2 = _mm256_setzero_pd();
    let mut i = 0usize;
    while i + 4 <= n {
        let f = _mm256_cvtps_pd(_mm_loadu_ps(col.as_ptr().add(i)));
        let wv = _mm256_loadu_pd(w.as_ptr().add(i));
        let wx = _mm256_mul_pd(wv, f);
        a1 = _mm256_add_pd(a1, wx);
        a2 = _mm256_fmadd_pd(wx, f, a2);
        i += 4;
    }
    let s1 = hsum256_pd(a1);
    let s2 = hsum256_pd(a2);
    let (t1, t2) = weighted_moments_scalar(col, w, i, n);
    (s1 + t1, s2 + t2)
}

/// `out[k] = sqrt_w[k] · (src[k] − mean) · inv_norm`, all f32 — the per-pixel
/// z-normalize write. AVX2+FMA (8-wide) dispatched at runtime; same pattern as
/// [`sum_sq_diff`].
#[inline]
fn znorm_write(src: &[f32], sqrt_w: &[f32], mean: f32, inv_norm: f32, out: &mut [f32]) {
    let n = src.len().min(sqrt_w.len()).min(out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check above.
            unsafe {
                znorm_write_avx2(src, sqrt_w, mean, inv_norm, out, n);
            }
            return;
        }
    }
    znorm_write_scalar(src, sqrt_w, mean, inv_norm, out, 0, n);
}

/// Scalar reference for [`znorm_write`] over `[i0, i1)` (also the AVX2 tail).
fn znorm_write_scalar(
    src: &[f32],
    sqrt_w: &[f32],
    mean: f32,
    inv_norm: f32,
    out: &mut [f32],
    i0: usize,
    i1: usize,
) {
    for k in i0..i1 {
        out[k] = sqrt_w[k] * (src[k] - mean) * inv_norm;
    }
}

/// AVX2+FMA `out[k] = sqrt_w[k]·(src[k] − mean)·inv_norm` over the first `n`
/// elements (8 f32/iteration). The `n % 8` tail falls back to
/// [`znorm_write_scalar`].
///
/// # Safety
/// Requires the `avx2` + `fma` target features (guarded by the dispatcher).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn znorm_write_avx2(
    src: &[f32],
    sqrt_w: &[f32],
    mean: f32,
    inv_norm: f32,
    out: &mut [f32],
    n: usize,
) {
    use std::arch::x86_64::*;
    let m = _mm256_set1_ps(mean);
    let s = _mm256_set1_ps(inv_norm);
    let mut i = 0usize;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(src.as_ptr().add(i));
        let sw = _mm256_loadu_ps(sqrt_w.as_ptr().add(i));
        let d = _mm256_sub_ps(x, m);
        let scale = _mm256_mul_ps(sw, s);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(scale, d));
        i += 8;
    }
    znorm_write_scalar(src, sqrt_w, mean, inv_norm, out, i, n);
}

/// Reused buffers for the consensus / IRLS reductions, so scoring a candidate
/// allocates nothing after warm-up. The cache path holds one in its `Scratch`;
/// the source path makes a fresh (default-empty) one per call.
#[derive(Default)]
pub(super) struct ConsensusScratch {
    /// View weights (`views`).
    pub(super) w: Vec<f64>,
    /// Weighted per-(channel, pixel) consensus (`channels·n`). f32: it is only
    /// consumed by the IRLS residual that feeds the robust reweight, so the
    /// accumulation precision is immaterial, and f32 keeps both the SAXPY and the
    /// residual a single 8-wide AVX2 path with no f64↔f32 narrowing.
    xbar: Vec<f32>,
    /// Per-pixel accumulator for a channel's weighted sum (`n`).
    s: Vec<f64>,
    /// Per-view residual to the consensus (`views`).
    resid: Vec<f64>,
    /// Scratch for the median/MAD sort (`views`), avoids per-iter clones.
    sorted: Vec<f64>,
    /// New (Tukey) weights before normalization (`views`).
    wt: Vec<f64>,
}

/// Unweighted consensus over one channel, `ρ̄ = (V‖x̄‖² − 1)/(V − 1)`. Accumulates
/// the per-pixel cross-view sum `s[k] = Σ_v x[v,c,k]` by SAXPY over contiguous
/// per-view rows (`sc.s` reused), then `‖x̄‖² = Σ_k (s[k]/V)²`.
fn mean_pairwise_channel(
    xs: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    c: usize,
    sc: &mut ConsensusScratch,
) -> f64 {
    sc.s.clear();
    sc.s.resize(n, 0.0);
    for v in 0..views {
        let row = &xs[(v * channels + c) * n..][..n];
        sc.s.iter_mut().zip(row).for_each(|(s, &r)| *s += r as f64);
    }
    let inv_v = 1.0 / views as f64;
    let mut norm_sq = 0.0;
    for &s in &sc.s {
        let m = s * inv_v;
        norm_sq += m * m;
    }
    (views as f64 * norm_sq - 1.0) / (views as f64 - 1.0)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// `Σ_k (row[k] − xbar[k])²`, all f32 — the IRLS per-view residual sum. An f64
/// `.sum()` cannot vectorize (non-associative → a serial dependency chain), and
/// the crate builds for a baseline target (no `-C target-feature=+avx2`), so the
/// scalar fallback only reaches SSE; the dispatched AVX2+FMA kernel below is
/// 8-wide. With `xbar` also f32 there is no per-element narrowing. The residual
/// only feeds the Tukey reweight (median/MAD/cutoff), so f32 precision is ample
/// and the found weights — hence normals — are unaffected in practice. Mirrors
/// the runtime-dispatch pattern of [`fronto_cache`] and [`crate::features::sift::simd`].
#[inline]
fn sum_sq_diff(row: &[f32], xbar: &[f32]) -> f32 {
    debug_assert_eq!(row.len(), xbar.len());
    let n = row.len().min(xbar.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check above.
            return unsafe { sum_sq_diff_avx2(row, xbar, n) };
        }
    }
    sum_sq_diff_scalar(row, xbar, 0, n)
}

/// Scalar reference for [`sum_sq_diff`] over `[i0, i1)` (also the AVX2 tail). Eight
/// independent accumulators keep the fallback SSE-vectorizable on a baseline build.
fn sum_sq_diff_scalar(row: &[f32], xbar: &[f32], i0: usize, i1: usize) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0f32; LANES];
    let body = i0 + (i1 - i0) / LANES * LANES;
    let mut i = i0;
    while i < body {
        for (l, a) in acc.iter_mut().enumerate() {
            let d = row[i + l] - xbar[i + l];
            *a += d * d;
        }
        i += LANES;
    }
    let mut s: f32 = acc.iter().sum();
    for k in i..i1 {
        let d = row[k] - xbar[k];
        s += d * d;
    }
    s
}

/// AVX2+FMA reduction of `Σ_k (row[k] − xbar[k])²` over the first `n` elements
/// (8 lanes/iteration; both operands f32, so one `vmovups` each — no narrowing).
/// The `n % 8` tail falls back to [`sum_sq_diff_scalar`]. Result matches the
/// scalar path up to f32 summation order.
///
/// # Safety
/// Requires the `avx2` + `fma` target features (guarded by the dispatcher).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_sq_diff_avx2(row: &[f32], xbar: &[f32], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let r = _mm256_loadu_ps(row.as_ptr().add(i));
        let xb = _mm256_loadu_ps(xbar.as_ptr().add(i));
        let d = _mm256_sub_ps(r, xb);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }
    // Horizontal sum of the 8 lanes.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x1));
    _mm_cvtss_f32(s1) + sum_sq_diff_scalar(row, xbar, i, n)
}

/// `xb[k] += w · row[k]` — the IRLS weighted-consensus SAXPY, all f32. The
/// baseline target limits the autovectorized fallback to SSE, so an AVX2+FMA
/// kernel (8-wide f32) is dispatched at runtime. Same runtime-dispatch pattern as
/// [`sum_sq_diff`].
#[inline]
fn axpy_f32(xb: &mut [f32], row: &[f32], w: f32) {
    debug_assert_eq!(xb.len(), row.len());
    let n = xb.len().min(row.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check above.
            unsafe {
                axpy_f32_avx2(xb, row, w, n);
            }
            return;
        }
    }
    axpy_f32_scalar(xb, row, w, 0, n);
}

/// Scalar reference for [`axpy_f32`] over `[i0, i1)` (also the AVX2 tail).
fn axpy_f32_scalar(xb: &mut [f32], row: &[f32], w: f32, i0: usize, i1: usize) {
    for k in i0..i1 {
        xb[k] += w * row[k];
    }
}

/// AVX2+FMA `xb[k] += w · row[k]` over the first `n` elements (8 f32/iteration;
/// both operands f32, no narrowing). The `n % 8` tail falls back to
/// [`axpy_f32_scalar`].
///
/// # Safety
/// Requires the `avx2` + `fma` target features (guarded by the dispatcher).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f32_avx2(xb: &mut [f32], row: &[f32], w: f32, n: usize) {
    use std::arch::x86_64::*;
    let wv = _mm256_set1_ps(w);
    let mut i = 0usize;
    while i + 8 <= n {
        let r = _mm256_loadu_ps(row.as_ptr().add(i));
        let acc = _mm256_loadu_ps(xb.as_ptr().add(i));
        _mm256_storeu_ps(xb.as_mut_ptr().add(i), _mm256_fmadd_ps(wv, r, acc));
        i += 8;
    }
    axpy_f32_scalar(xb, row, w, i, n);
}

/// IRLS view weights into `sc.w` (`Σwᵢ = 1`): a Tukey weight on each view's
/// residual `‖xᵢ − x̄_w‖` (stacked over channels), scaled by the residual MAD
/// (floored against the median so a clean, outlier-free stack keeps near-uniform
/// weights), re-formed `iters` times. All buffers live in `sc` (no per-candidate
/// / per-iteration allocation); the weighted consensus `xbar` is accumulated by
/// SAXPY over contiguous per-view rows.
pub(super) fn irls_view_weights(
    xs: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    iters: u32,
    sc: &mut ConsensusScratch,
) {
    sc.w.clear();
    sc.w.resize(views, 1.0 / views as f64);
    sc.xbar.resize(channels * n, 0.0);
    sc.resid.resize(views, 0.0);
    sc.sorted.resize(views, 0.0);
    sc.wt.resize(views, 0.0);

    for _ in 0..iters {
        // Weighted consensus per (channel, pixel): SAXPY each view's row.
        prof::IRLS_XBAR.time(|| {
            sc.xbar.iter_mut().for_each(|x| *x = 0.0);
            for v in 0..views {
                let wv = sc.w[v] as f32;
                for c in 0..channels {
                    let row = &xs[(v * channels + c) * n..][..n];
                    let xb = &mut sc.xbar[c * n..][..n];
                    axpy_f32(xb, row, wv);
                }
            }
        });
        // Per-view residual to the consensus (contiguous per row), all f32: only
        // the Tukey reweight consumes it, so the precision is immaterial, and the
        // dispatched AVX2 reduction needs no narrowing now that xbar is f32 too.
        prof::IRLS_RESID.time(|| {
            for v in 0..views {
                let mut r2 = 0f32;
                for c in 0..channels {
                    let row = &xs[(v * channels + c) * n..][..n];
                    let xb = &sc.xbar[c * n..][..n];
                    r2 += sum_sq_diff(row, xb);
                }
                sc.resid[v] = (r2 as f64).sqrt();
            }
        });

        let should_break = prof::IRLS_REWEIGHT.time(|| {
            sc.sorted.copy_from_slice(&sc.resid);
            let med = median(&mut sc.sorted);
            for (s, &r) in sc.sorted.iter_mut().zip(&sc.resid) {
                *s = (r - med).abs();
            }
            let mad = median(&mut sc.sorted);
            let scale = (1.4826 * mad).max(0.5 * med).max(1e-12);
            let cutoff = 4.685 * scale;

            let mut sum = 0.0;
            for v in 0..views {
                let r = sc.resid[v];
                let wt = if r >= cutoff {
                    0.0
                } else {
                    let t = 1.0 - (r / cutoff) * (r / cutoff);
                    t * t
                };
                sc.wt[v] = wt;
                sum += wt;
            }
            if sum <= 1e-12 {
                return true; // Degenerate re-weight; keep the previous weights.
            }
            for v in 0..views {
                sc.w[v] = sc.wt[v] / sum;
            }
            false
        });
        if should_break {
            break;
        }
    }
}

/// Consensus photoconsistency `Φ` over the normalized stack, per the
/// objective. `None` when fewer than 2 views, or when the robust effective
/// view count `1/Σwᵢ²` drops below [`MIN_EFFECTIVE_VIEWS`] (weights collapsed
/// onto essentially one view).
fn consensus_phi(
    xs: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    objective: Objective,
    sc: &mut ConsensusScratch,
) -> Option<f64> {
    consensus_phi_with_weights(xs, views, channels, n, objective, sc).map(|(phi, _)| phi)
}

/// [`consensus_phi`] plus the per-view consensus weights that produced it,
/// normalized to `Σwᵢ = 1` (uniform `1/views` under [`Objective::MeanPairwise`],
/// the IRLS weights under [`Objective::RobustWeighted`]). Callers that re-use the
/// weights — the representative fusion, which down-weights the same outlier views
/// in the blended colour as the consensus does in `Φ` — take this variant so the
/// IRLS pass is not run twice.
fn consensus_phi_with_weights(
    xs: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    objective: Objective,
    sc: &mut ConsensusScratch,
) -> Option<(f64, Vec<f64>)> {
    if views < 2 {
        return None;
    }
    match objective {
        Objective::MeanPairwise => {
            let sum: f64 = (0..channels)
                .map(|c| mean_pairwise_channel(xs, views, channels, n, c, sc))
                .sum();
            Some((sum / channels as f64, vec![1.0 / views as f64; views]))
        }
        Objective::RobustWeighted { iters } => {
            irls_view_weights(xs, views, channels, n, iters, sc);
            let sum_w2: f64 = sc.w.iter().map(|&x| x * x).sum();
            // Degeneracy gate only (the view *count* is gated by `min_views` per
            // level): as weight concentrates on one view, Σwᵢ² → 1 and ρ̄_w → 0/0.
            if 1.0 / sum_w2 < MIN_EFFECTIVE_VIEWS || 1.0 - sum_w2 < 1e-9 {
                return None;
            }
            let phi = prof::CONS_FINAL.time(|| {
                let mut sum = 0.0;
                sc.s.clear();
                sc.s.resize(n, 0.0);
                for c in 0..channels {
                    // s[k] = Σ_v w[v]·x[v,c,k], SAXPY over contiguous per-view rows.
                    sc.s.iter_mut().for_each(|x| *x = 0.0);
                    for v in 0..views {
                        let wv = sc.w[v];
                        let row = &xs[(v * channels + c) * n..][..n];
                        sc.s.iter_mut()
                            .zip(row)
                            .for_each(|(s, &r)| *s += wv * r as f64);
                    }
                    let norm_sq: f64 = sc.s.iter().map(|&s| s * s).sum();
                    sum += (norm_sq - sum_w2) / (1.0 - sum_w2);
                }
                sum / channels as f64
            });
            Some((phi, sc.w.clone()))
        }
    }
}

/// Evaluate `Φ` for the candidate normal `n` over the frozen support `ctx`,
/// scoring with `objective` (the search path passes a possibly-cheaper
/// objective than `params.objective`; the final / confidence passes pass
/// `params.objective`).
fn eval_phi(
    base: &OrientedPatch,
    n: &Vector3<f64>,
    ctx: &LevelContext,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
    objective: Objective,
) -> Option<f64> {
    prof::count(&prof::N_EVAL, 1);
    let patch = repose_patch(base, n);
    let (raw, channels) = normalized_stack(&patch, ctx, views, resolution, params.sampler)?;
    let n = ctx.pixels.len();
    let total_w: f64 = ctx.weights.iter().sum();
    if total_w <= 0.0 {
        return None;
    }
    let sqrt_w: Vec<f32> = ctx.weights.iter().map(|&w| w.sqrt() as f32).collect();
    let mut xs = Vec::new();
    let kept = prof::ZNORM.time(|| {
        znormalize_into(
            &raw,
            ctx.kept.len(),
            channels,
            n,
            &ctx.weights,
            total_w,
            &sqrt_w,
            &mut xs,
        )
    })?;
    let mut cons = ConsensusScratch::default();
    prof::CONSENSUS.time(|| consensus_phi(&xs, ctx.kept.len(), kept, n, objective, &mut cons))
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Coarse-to-fine exp-map grid search from one seed normal. Each level
/// freezes the common-valid mask at its center normal, scans the δ-disk
/// (`‖δ‖ ≤ range`), recenters on the best candidate, and shrinks the cone to
/// one previous grid spacing. Returns the seed's winning normal, or `None`
/// when nothing could be evaluated from this seed.
fn coarse_to_fine(
    base: &OrientedPatch,
    seed: Vector3<f64>,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    w_full: &[f64],
    params: &NormalRefineParams,
) -> Option<Vector3<f64>> {
    // At least 3 grid samples per axis: with 2 the only nonzero candidates land
    // on the disk corners and are clamped away, leaving the center as the sole
    // sample (a no-op search).
    let steps = params.init_steps.max(3) as usize;
    let mut center = seed.normalize();
    let mut range = params.angular_range_deg.to_radians().max(1e-4);
    let mut any_eval = false;
    // The search ranks candidates with a possibly-cheaper objective; the final
    // pass re-scores the survivors with `params.objective`.
    let search_obj = params.search_objective();

    // Fronto-parallel cache: render one supersampled base per view here, reused
    // across all levels. `None` (cache off, or the base could not be built for
    // this patch) means score every candidate from the source images instead.
    let cache = match params.cache {
        CacheMode::FrontoParallel => prof::PRERENDER.time(|| {
            fronto_cache::prerender(
                base,
                &center,
                views,
                resolution,
                params.cache_supersample,
                params,
            )
        }),
        CacheMode::Off => None,
    };
    // Reused across every candidate of every level (cache path only).
    let mut scratch = fronto_cache::Scratch::default();

    for _ in 0..params.refine_levels.max(1) {
        let Some(ctx) = build_level_context(base, &center, views, resolution, w_full, params)
        else {
            break;
        };
        let (u, v) = tangent_basis(&center);
        // The cache resamples the level's frozen support; its grid coords and
        // window weights are candidate-independent, so compute them once per level.
        let (cols, rows): (Vec<i32>, Vec<i32>) = if cache.is_some() {
            ctx.pixels
                .iter()
                .map(|&p| {
                    (
                        (p % resolution as usize) as i32,
                        (p / resolution as usize) as i32,
                    )
                })
                .unzip()
        } else {
            (Vec::new(), Vec::new())
        };
        let (sqrt_w, total_w): (Vec<f32>, f64) = if cache.is_some() {
            (
                ctx.weights.iter().map(|&w| w.sqrt() as f32).collect(),
                ctx.weights.iter().sum(),
            )
        } else {
            (Vec::new(), 0.0)
        };
        let mut eval = |n: &Vector3<f64>| -> Option<f64> {
            match &cache {
                Some(c) => fronto_cache::eval_phi(
                    base,
                    n,
                    c,
                    &ctx,
                    views,
                    resolution,
                    &cols,
                    &rows,
                    &sqrt_w,
                    total_w,
                    &mut scratch,
                    search_obj,
                ),
                None => eval_phi(base, n, &ctx, views, resolution, params, search_obj),
            }
        };
        let mut best_n = center;
        let mut best_phi = f64::NEG_INFINITY;
        if let Some(phi) = eval(&center) {
            best_phi = phi;
        }
        for i in 0..steps {
            let a = -range + 2.0 * range * i as f64 / (steps - 1) as f64;
            for j in 0..steps {
                let b = -range + 2.0 * range * j as f64 / (steps - 1) as f64;
                if a == 0.0 && b == 0.0 {
                    continue; // Center already evaluated.
                }
                // Clamp the square grid to the δ-disk (circular cone).
                if a.hypot(b) > range * (1.0 + 1e-9) {
                    continue;
                }
                let n = exp_map_in_basis(&center, &u, &v, [a, b]);
                if let Some(phi) = eval(&n) {
                    if phi > best_phi {
                        best_phi = phi;
                        best_n = n;
                    }
                }
            }
        }
        if !best_phi.is_finite() {
            break;
        }
        any_eval = true;
        center = best_n;
        // Next cone: ± one grid spacing of this level.
        range = (2.0 * range / (steps - 1) as f64).max(1e-4);
    }
    any_eval.then_some(center)
}

/// Final-pass support: kept views and mask frozen at the *init* normal, with
/// the mask further intersected with each surviving winner's validity, so the
/// init and every winner are scored over the same pixel set. Winners that are
/// back-facing in a kept view are discarded. Returns the context and the
/// surviving winners.
fn build_final_context(
    base: &OrientedPatch,
    init_n: &Vector3<f64>,
    winners: &[Vector3<f64>],
    views: &[ProjectedImage<'_>],
    resolution: u32,
    w_full: &[f64],
    params: &NormalRefineParams,
) -> Option<(LevelContext, Vec<Vector3<f64>>)> {
    let mut ctx = build_level_context(base, init_n, views, resolution, w_full, params)?;
    let mut survivors = Vec::new();
    for n in winners {
        let patch = repose_patch(base, n);
        if !ctx
            .kept
            .iter()
            .all(|&i| patch.is_front_facing(views[i].cam_from_world))
        {
            continue;
        }
        let mut pixels = Vec::with_capacity(ctx.pixels.len());
        let mut weights = Vec::with_capacity(ctx.pixels.len());
        let masks: Vec<Vec<bool>> = ctx
            .kept
            .iter()
            .map(|&i| view_valid_mask(&patch, &views[i], resolution, w_full))
            .collect();
        for (k, &p) in ctx.pixels.iter().enumerate() {
            if masks.iter().all(|m| m[p]) {
                pixels.push(p);
                weights.push(ctx.weights[k]);
            }
        }
        if pixels.len() < MIN_MASK_PIXELS {
            continue; // This winner would leave too little common support.
        }
        ctx.pixels = pixels;
        ctx.weights = weights;
        survivors.push(*n);
    }
    Some((ctx, survivors))
}

/// Finite-difference confidence: the curvature of `Φ(δ)` at the optimum,
/// estimated from a 3×3 stencil of grid samples (spacing `h`). The smaller
/// eigenvalue of the negated 2×2 Hessian measures how tightly the normal is
/// constrained; it is normalized against the larger eigenvalue plus an
/// absolute floor, so a flat `Φ` (the narrow-baseline degeneracy, where
/// tilting the plane shifts all patches identically) reports ≈ 0 and an
/// isotropically peaked `Φ` reports ≈ 1.
///
/// The spec's analytic *centered Gauss-Newton Hessian*
/// (`H̃ = Σ wᵢ JᵢᵀJᵢ − J̄ᵀJ̄`) is the fast-follow; this grid estimate is
/// GN-free and captures the same degeneracy because `Φ` itself is genuinely
/// flat there.
fn grid_confidence(
    base: &OrientedPatch,
    n: &Vector3<f64>,
    ctx: &LevelContext,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
    h: f64,
) -> f64 {
    let (u, v) = tangent_basis(n);
    let phi = |a: f64, b: f64| -> Option<f64> {
        eval_phi(
            base,
            &exp_map_in_basis(n, &u, &v, [a, b]),
            ctx,
            views,
            resolution,
            params,
            params.objective,
        )
    };
    let stencil = [
        phi(0.0, 0.0),
        phi(h, 0.0),
        phi(-h, 0.0),
        phi(0.0, h),
        phi(0.0, -h),
        phi(h, h),
        phi(h, -h),
        phi(-h, h),
        phi(-h, -h),
    ];
    let Some(vals) = stencil.into_iter().collect::<Option<Vec<f64>>>() else {
        return 0.0;
    };
    let [f00, fp0, fm0, f0p, f0m, fpp, fpm, fmp, fmm] = vals[..] else {
        return 0.0;
    };
    let h2 = h * h;
    // Negated Hessian of Φ (positive curvature at a maximum).
    let haa = -(fp0 - 2.0 * f00 + fm0) / h2;
    let hbb = -(f0p - 2.0 * f00 + f0m) / h2;
    let hab = -(fpp - fpm - fmp + fmm) / (4.0 * h2);
    let mean = 0.5 * (haa + hbb);
    let disc = (0.25 * (haa - hbb) * (haa - hbb) + hab * hab).sqrt();
    let lmax = (mean + disc).max(0.0);
    let lmin = (mean - disc).max(0.0);
    if !lmin.is_finite() || !lmax.is_finite() {
        return 0.0;
    }
    // Scale normalization: dimensionless in texture contrast (Φ is already a
    // correlation); the +1 floor (Φ per radian²) keeps weakly-curved optima
    // from reporting full confidence.
    lmin / (lmax + 1.0)
}

/// Refine one patch's normal photometrically. Takes the patch and returns an
/// updated copy plus diagnostics; see `specs/core/patch-normal-refinement.md`.
///
/// In-plane rotation can't affect photoconsistency, so the routine searches
/// only the 2-DOF normal; it reprojects the input `u_axis` onto each candidate
/// plane (`v = n × u`) and keeps the input's `center`/`half_extent`, so the
/// frame moves as little as the new plane forces and no `up` hint is needed.
///
/// The search is seeded from the patch's current normal and the mean-viewing
/// normal of the supplied views, runs a coarse-to-fine exp-map grid per seed,
/// then scores every seed winner *and* the init normal over one final frozen
/// support — so the returned `photoconsistency` is never below
/// `init_photoconsistency` when both are finite. Patches whose validity gates
/// fail outright are returned unrefined (NaN scores, zero confidence).
///
/// Not idempotent by design: feeding a refined normal back in can improve it
/// further (each pass re-seeds and re-explores), so running to convergence is the
/// thorough setting — this is intentional, not a fixed-point operation.
pub fn refine_patch_normal(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
) -> NormalRefineResult {
    prof::TOTAL.time(|| refine_patch_normal_impl(patch, views, resolution, params))
}

fn refine_patch_normal_impl(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
) -> NormalRefineResult {
    let resolution = resolution.max(2);
    let init_n = patch.normal();
    let w_full = window_weights(params.window, resolution);

    let unrefined = |valid_view_count: u32| NormalRefineResult {
        patch: patch.clone(),
        photoconsistency: f64::NAN,
        init_photoconsistency: f64::NAN,
        valid_view_count,
        confidence: 0.0,
        representative: None,
    };

    // A point at infinity has a fixed outward normal (`normalize(-d)`, set by its
    // direction) — there is nothing to refine, and the finite-depth render path
    // is invalid for it. Leave its frame untouched.
    if patch.w == 0.0 {
        return unrefined(0);
    }

    // Degenerate points skip the search outright.
    if views.len() < params.min_views.max(2) as usize {
        return unrefined(0);
    }

    // Seeds: the patch's current normal, plus the mean-viewing normal of the
    // supplied views when it differs.
    let mut seeds = vec![init_n];
    let centers: Vec<Point3<f64>> = views
        .iter()
        .map(|v| v.cam_from_world.inverse_translation_origin())
        .collect();
    let mean_view = mean_viewing_normal(&patch.center, &centers);
    if mean_view.dot(&init_n) < (0.5f64).to_radians().cos() {
        seeds.push(mean_view);
    }

    // Stage 1: coarse-to-fine grid per seed; keep each seed's winner.
    let mut winners: Vec<Vector3<f64>> = Vec::new();
    for seed in &seeds {
        if let Some(n) = coarse_to_fine(patch, *seed, views, resolution, &w_full, params) {
            winners.push(n);
        }
    }

    // Final pass: one frozen support for the init and all winners.
    let Some((ctx, survivors)) =
        build_final_context(patch, &init_n, &winners, views, resolution, &w_full, params)
    else {
        return unrefined(0);
    };
    let valid_view_count = ctx.kept.len() as u32;

    // Final scoring: score the init and every survivor over the same frozen
    // support `ctx`, so Φ is never below the init's. Only the bitmap path needs
    // the per-view render kept around, so it scores through a `PatchViewStack`
    // (retaining the *winner's* stack + consensus weights, which the
    // representative then fuses with no extra render or IRLS pass); the default
    // path stays on the lean masked-only `eval_phi`, paying nothing for a feature
    // it doesn't use. `extra` carries the winner's stack only in the bitmap path.
    type Winner = Option<(Vec<f64>, PatchViewStack)>;
    let score = |n: &Vector3<f64>| -> Option<(f64, Winner)> {
        prof::count(&prof::N_EVAL, 1);
        if params.render_bitmap {
            let stack = PatchViewStack::render(
                &repose_patch(patch, n),
                views,
                &ctx.kept,
                resolution,
                params.sampler,
            );
            let (phi, weights) = stack.score(&ctx, params)?;
            Some((phi, Some((weights, stack))))
        } else {
            let phi = eval_phi(patch, n, &ctx, views, resolution, params, params.objective)?;
            Some((phi, None))
        }
    };

    let init = score(&init_n);
    let phi_init = init.as_ref().map(|(phi, _)| *phi);
    let mut best_n = init_n;
    let mut best_phi = phi_init.unwrap_or(f64::NEG_INFINITY);
    // The winner's rendered stack + consensus weights (bitmap path only).
    let mut best: Winner = init.and_then(|(_, extra)| extra);
    let mut improved = false;
    for n in &survivors {
        if let Some((phi, extra)) = score(n) {
            if phi > best_phi {
                best_phi = phi;
                best_n = *n;
                best = extra;
                improved = true;
            }
        }
    }
    if !best_phi.is_finite() {
        return unrefined(valid_view_count);
    }

    // Confidence (optional): an extra source-render stencil around the optimum,
    // un-cached and purely informational, so it is computed only on request.
    let confidence = if params.compute_confidence {
        // Stencil spacing: the final grid spacing of the coarse-to-fine schedule,
        // clamped to a sane angular band.
        let steps = params.init_steps.max(3) as f64;
        let shrink = 2.0 / (steps - 1.0);
        let h = (params.angular_range_deg.to_radians()
            * shrink.powi(params.refine_levels.max(1) as i32))
        .clamp(0.2f64.to_radians(), 5.0f64.to_radians());
        prof::CONFIDENCE
            .time(|| grid_confidence(patch, &best_n, &ctx, views, resolution, params, h))
    } else {
        f64::NAN
    };

    // Representative RGBA texture (optional): fuse the winner's already-rendered
    // view stack — no extra render or IRLS pass (the scoring pass produced both).
    // Unlike scoring (which reads only the masked common support), fusion spans
    // the full R×R grid, filling every pixel a kept view covers. `best` is `Some`
    // exactly in the bitmap path (and then non-empty, since `best_phi` is finite).
    let representative = best
        .as_ref()
        .map(|(weights, stack)| stack.fuse(weights, AGREEMENT_SIGMA));

    NormalRefineResult {
        patch: if improved {
            repose_patch(patch, &best_n)
        } else {
            patch.clone()
        },
        photoconsistency: best_phi,
        init_photoconsistency: phi_init.unwrap_or(f64::NAN),
        valid_view_count,
        confidence,
        representative,
    }
}

// ---------------------------------------------------------------------------
// Multi-view patch render (shared substrate: scoring, weighting, fusion)
// ---------------------------------------------------------------------------

/// Contrast scale (in 0–255 colour units) of the per-pixel agreement confidence:
/// the weighted cross-view colour RMS deviation `σ` at which agreement decays to
/// `e^{-1/2}`. Larger tolerates more cross-view colour spread before the alpha
/// drops.
const AGREEMENT_SIGMA: f64 = 24.0;

/// Map a rendered source pixel to RGB, replicating a single channel to grey and
/// taking the first three of a multi-channel image (matching the thumbnail RGB
/// convention).
fn sample_rgb(img: &crate::camera::remap::ImageU8, col: u32, row: u32, channels: u32) -> [f64; 3] {
    match channels {
        0 => [0.0; 3],
        1 | 2 => {
            let g = img.get_pixel(col, row, 0) as f64;
            [g, g, g]
        }
        _ => [
            img.get_pixel(col, row, 0) as f64,
            img.get_pixel(col, row, 1) as f64,
            img.get_pixel(col, row, 2) as f64,
        ],
    }
}

/// The appearance of one oriented patch as each observing camera sees it: every
/// kept view warped into the patch's `R×R` `(s, t)` grid, with per-pixel
/// validity. Rendering is the dominant cost of refinement, so this is the single
/// substrate the operations that all need "the patch from each view" read from,
/// instead of re-rendering:
///
/// - photoconsistency scoring and the robust IRLS view weights ([`Self::score`],
///   over a frozen masked support);
/// - the fused representative texture ([`Self::fuse`], over the full grid).
///
/// It is the natural input for future per-view operations too — cross-validation
/// strips, the per-pixel robust template (`patch-normal-refinement.md` item 7),
/// per-view patch export, and GPU textured-surfel rendering.
struct PatchViewStack {
    resolution: u32,
    /// One full `R×R` render per kept view.
    images: Vec<crate::camera::remap::ImageU8>,
    /// Full-grid per-pixel validity per kept view (row-major, length `R²`),
    /// parallel to `images`.
    valid: Vec<Vec<bool>>,
    /// Channels shared across the consensus: the min over kept views.
    channels: usize,
}

impl PatchViewStack {
    /// Render `patch` into every view in `kept` (full grid + per-pixel validity).
    fn render(
        patch: &OrientedPatch,
        views: &[ProjectedImage<'_>],
        kept: &[usize],
        resolution: u32,
        sampler: Sampler,
    ) -> Self {
        let r = resolution as usize;
        let npix = r * r;
        prof::count(&prof::N_RENDER, kept.len() as u64);
        // The consensus space is the channel count shared by every kept view (its
        // min), exactly as `normalized_stack`; `0` for an empty stack.
        let channels = kept
            .iter()
            .map(|&vi| views[vi].pyramid.level(0).channels() as usize)
            .min()
            .unwrap_or(0);
        let mut images = Vec::with_capacity(kept.len());
        let mut valid = Vec::with_capacity(kept.len());
        for &vi in kept {
            let view = &views[vi];
            let mut map = prof::WARP
                .time(|| WarpMap::from_patch(patch, view.camera, view.cam_from_world, resolution));
            let mut vmask = vec![false; npix];
            for row in 0..resolution {
                for col in 0..resolution {
                    vmask[(row * resolution + col) as usize] = map.is_valid(col, row);
                }
            }
            let img = match sampler {
                Sampler::Anisotropic => {
                    prof::SVD.time(|| map.compute_svd());
                    prof::REMAP
                        .time(|| remap_aniso_with_pyramid(view.pyramid, &map, MAX_ANISOTROPY))
                }
                Sampler::Bilinear => {
                    prof::REMAP.time(|| remap_bilinear(view.pyramid.level(0), &map))
                }
            };
            images.push(img);
            valid.push(vmask);
        }
        Self {
            resolution,
            images,
            valid,
            channels,
        }
    }

    /// Gather the masked `pixels` into the flat consensus layout
    /// `[(view*channels + channel)*n + pixel]` (`n = pixels.len()`). `None` when a
    /// masked pixel is invalid in any kept view (the frame-edge rejection that
    /// keeps `Φ` comparable over the frozen mask) or no channel is present —
    /// matching [`normalized_stack`].
    fn gather(&self, pixels: &[usize]) -> Option<Vec<f32>> {
        let n_views = self.images.len();
        let channels = self.channels;
        if channels == 0 || n_views == 0 {
            return None;
        }
        let r = self.resolution as usize;
        let n = pixels.len();
        // Reject if any masked pixel falls out of frame in any kept view.
        for vmask in &self.valid {
            if pixels.iter().any(|&p| !vmask[p]) {
                prof::count(&prof::N_REJECT, 1);
                return None;
            }
        }
        let mut raw = vec![0f32; n_views * channels * n];
        prof::ZNORM.time(|| {
            for (vk, img) in self.images.iter().enumerate() {
                for c in 0..channels {
                    let base = (vk * channels + c) * n;
                    for (ki, &p) in pixels.iter().enumerate() {
                        raw[base + ki] =
                            img.get_pixel((p % r) as u32, (p / r) as u32, c as u32) as f32;
                    }
                }
            }
        });
        Some(raw)
    }

    /// Consensus `Φ` over the frozen support `ctx` and the per-view weights that
    /// produced it. Mirrors [`eval_phi`] (the same render → z-normalize →
    /// consensus), but reads the already-rendered images and also returns the
    /// weights, so the representative fusion reuses them. `None` when the support
    /// can't be scored.
    fn score(&self, ctx: &LevelContext, params: &NormalRefineParams) -> Option<(f64, Vec<f64>)> {
        let raw = self.gather(&ctx.pixels)?;
        let n = ctx.pixels.len();
        let total_w: f64 = ctx.weights.iter().sum();
        if total_w <= 0.0 {
            return None;
        }
        let sqrt_w: Vec<f32> = ctx.weights.iter().map(|&w| w.sqrt() as f32).collect();
        let mut xs = Vec::new();
        let kept = prof::ZNORM.time(|| {
            znormalize_into(
                &raw,
                self.images.len(),
                self.channels,
                n,
                &ctx.weights,
                total_w,
                &sqrt_w,
                &mut xs,
            )
        })?;
        let mut sc = ConsensusScratch::default();
        prof::CONSENSUS.time(|| {
            consensus_phi_with_weights(&xs, self.images.len(), kept, n, params.objective, &mut sc)
        })
    }

    /// Fuse the kept views into an `R×R` RGBA representative: per-pixel cross-view
    /// weighted-mean colour (RGB) and a cross-view *agreement* confidence (alpha).
    /// `weights` are the per-view consensus weights (parallel to the stack's
    /// views); only views that cover a pixel contribute, renormalized there. Alpha
    /// is `0` where no kept view covers a pixel and for a pixel seen by a single
    /// view (no cross-view evidence). Returns the flat `R·R·4` row-major texture.
    fn fuse(&self, weights: &[f64], sigma: f64) -> Vec<u8> {
        let r = self.resolution as usize;
        let npix = r * r;
        let mut out = vec![0u8; npix * 4];
        for p in 0..npix {
            let col = (p % r) as u32;
            let row = (p / r) as u32;
            // One pass: weighted colour sum and sum-of-squares over covering views.
            let mut wsum = 0.0;
            let mut s = [0.0f64; 3];
            let mut s2 = [0.0f64; 3];
            let mut n_cover = 0u32;
            for (vk, img) in self.images.iter().enumerate() {
                let w = weights.get(vk).copied().unwrap_or(1.0);
                if !self.valid[vk][p] || w <= 0.0 {
                    continue;
                }
                n_cover += 1;
                wsum += w;
                let rgb = sample_rgb(img, col, row, img.channels());
                for c in 0..3 {
                    s[c] += w * rgb[c];
                    s2[c] += w * rgb[c] * rgb[c];
                }
            }
            if n_cover == 0 || wsum <= 0.0 {
                continue; // no coverage: leave RGB and alpha zero
            }
            let mean = [s[0] / wsum, s[1] / wsum, s[2] / wsum];
            // Weighted per-channel variance E[x²]−E[x]², averaged over RGB; the
            // fp difference can dip slightly negative, so floor at 0.
            let var = (0..3)
                .map(|c| (s2[c] / wsum - mean[c] * mean[c]).max(0.0))
                .sum::<f64>()
                / 3.0;
            let rms = var.sqrt();
            let agreement = (-(rms * rms) / (2.0 * sigma * sigma)).exp();
            let coverage = 1.0 - 1.0 / n_cover as f64; // 1 view -> 0, 2 -> 0.5, ...
            let alpha = (255.0 * agreement * coverage).round().clamp(0.0, 255.0) as u8;
            let base_idx = p * 4;
            out[base_idx] = mean[0].round().clamp(0.0, 255.0) as u8;
            out[base_idx + 1] = mean[1].round().clamp(0.0, 255.0) as u8;
            out[base_idx + 2] = mean[2].round().clamp(0.0, 255.0) as u8;
            out[base_idx + 3] = alpha;
        }
        out
    }
}

/// Batch [`refine_patch_normal`] over a [`PatchCloud`], parallel across
/// patches (rayon). `patch_views[i]` lists, for patch `i`, the indices into
/// `views` of the cameras observing it (see
/// [`view_indices_from_reconstruction`]). Each patch is replaced with
/// its refined copy; the per-patch results are returned in order.
///
/// # Panics
///
/// Panics if `patch_views.len() != cloud.len()` or an index is out of range.
pub fn refine_patch_cloud(
    cloud: &mut PatchCloud,
    views: &[ProjectedImage<'_>],
    patch_views: &[Vec<u32>],
    resolution: u32,
    params: &NormalRefineParams,
) -> Vec<NormalRefineResult> {
    assert_eq!(
        patch_views.len(),
        cloud.len(),
        "patch_views must be parallel to the cloud"
    );
    if prof::enabled() {
        prof::reset();
    }
    let wall_start = std::time::Instant::now();
    let results: Vec<NormalRefineResult> = cloud
        .patches
        .par_iter()
        .zip(patch_views.par_iter())
        .map(|(patch, vidx)| {
            let pv: Vec<ProjectedImage<'_>> = vidx.iter().map(|&i| views[i as usize]).collect();
            refine_patch_normal(patch, &pv, resolution, params)
        })
        .collect();
    if prof::enabled() {
        prof::report(cloud.len(), wall_start.elapsed().as_secs_f64());
    }
    for (p, r) in cloud.patches.iter_mut().zip(&results) {
        *p = r.patch.clone();
    }
    results
}

/// For each patch of `cloud` (linked to `recon` via `point_ids`), the image
/// indices observing its source 3D point — ready to use as the `patch_views`
/// of [`refine_patch_cloud`] with one [`ProjectedImage`] per reconstruction image.
///
/// # Panics
///
/// Panics if `cloud.point_ids` is not parallel to its patches.
pub fn view_indices_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<u32>> {
    assert_eq!(
        cloud.point_ids.len(),
        cloud.len(),
        "cloud must carry a point_id per patch"
    );
    cloud
        .point_ids
        .iter()
        .map(|&p| {
            let p = p as usize;
            recon.tracks[recon.observation_offsets[p]..recon.observation_offsets[p + 1]]
                .iter()
                .map(|o| o.image_index)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests;
