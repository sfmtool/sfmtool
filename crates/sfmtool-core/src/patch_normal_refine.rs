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

use crate::camera_intrinsics::CameraIntrinsics;
use crate::patch_cloud::{mean_viewing_normal, OrientedPatch, PatchCloud};
use crate::reconstruction::SfmrReconstruction;
use crate::remap::{remap_aniso_with_pyramid, remap_bilinear, ImageU8Pyramid};
use crate::rigid_transform::RigidTransform;
use crate::warp_map::WarpMap;

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
        }
    }
}

/// Result of [`refine_patch_normal`].
///
/// The spec also sketches a `representative: Option<PatchTexture>` output (the
/// canonical robust-template appearance); that belongs to the joint
/// normal + robust representative variant which is beyond v1, so the field is
/// omitted until a producer exists. TODO(beyond-v1): add
/// `representative: Option<PatchTexture>` per spec item 7.
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
}

/// Windowed norm² below which a colour channel counts as flat (no texture
/// signal); flat channels are dropped from the consensus so the z-normalized
/// identity stays well-defined.
const FLAT_NORM_SQ_EPS: f64 = 1e-6;

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
fn window_weights(window: PatchWindow, resolution: u32) -> Vec<f64> {
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
fn repose_patch(base: &OrientedPatch, n: &Vector3<f64>) -> OrientedPatch {
    OrientedPatch::from_center_normal(base.center, *n, base.u_axis, base.half_extent)
}

// ---------------------------------------------------------------------------
// Validity / common support
// ---------------------------------------------------------------------------

/// The frozen scoring support of one grid level: the kept views and the
/// commonly-valid masked pixels (computed at the level's *center* normal and
/// held fixed across that level's candidates, so `Φ` stays continuous and
/// tilts can't shrink the support onto an easy region).
struct LevelContext {
    /// Indices into the caller's `views` slice.
    kept: Vec<usize>,
    /// Linear `row * R + col` indices of the masked pixels.
    pixels: Vec<usize>,
    /// Window weight per masked pixel (parallel to `pixels`).
    weights: Vec<f64>,
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
fn build_level_context(
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
fn normalized_stack(
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
/// in *any* view is dropped for *every* view. Reductions accumulate in f64; the
/// stack is stored f32 to halve the memory traffic the consensus re-reads.
/// Returns the kept-channel count, or `None` when none survive. Shared by the
/// source-render path ([`normalized_stack`]) and the fronto cache
/// ([`fronto_cache::eval_phi`]) so the two cannot drift.
#[allow(clippy::too_many_arguments)]
fn znormalize_into(
    raw: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    weights: &[f64],
    total_w: f64,
    sqrt_w: &[f64],
    out: &mut Vec<f32>,
) -> Option<usize> {
    let mut keep = vec![true; channels];
    let mut stats = vec![(0.0f64, 0.0f64); views * channels];
    for v in 0..views {
        for c in 0..channels {
            let col = &raw[(v * channels + c) * n..][..n];
            let mean = col
                .iter()
                .zip(weights)
                .map(|(&f, &w)| w * f as f64)
                .sum::<f64>()
                / total_w;
            let norm_sq = col
                .iter()
                .zip(weights)
                .map(|(&f, &w)| {
                    let d = f as f64 - mean;
                    w * d * d
                })
                .sum::<f64>();
            if norm_sq < FLAT_NORM_SQ_EPS {
                keep[c] = false;
            }
            stats[v * channels + c] = (mean, norm_sq);
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
            let (mean, norm_sq) = stats[v * channels + c];
            let inv_norm = 1.0 / norm_sq.sqrt();
            let src = &raw[(v * channels + c) * n..][..n];
            let base = (v * kept + kc) * n;
            for k in 0..n {
                out[base + k] = (sqrt_w[k] * (src[k] as f64 - mean) * inv_norm) as f32;
            }
            kc += 1;
        }
    }
    Some(kept)
}

/// Unweighted consensus over one channel: `ρ̄ = (V‖x̄‖² − 1)/(V − 1)` with
/// `x̄ = (1/V) Σ xᵢ` — the closed form of the all-pairs mean ZNCC.
fn mean_pairwise_channel(xs: &[f32], views: usize, channels: usize, n: usize, c: usize) -> f64 {
    let mut norm_sq = 0.0;
    for k in 0..n {
        let s: f64 = (0..views)
            .map(|v| xs[(v * channels + c) * n + k] as f64)
            .sum();
        norm_sq += (s / views as f64) * (s / views as f64);
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

/// IRLS view weights (`Σwᵢ = 1`): Tukey weight on each view's residual
/// `‖xᵢ − x̄_w‖` (stacked over channels), with a scale from the residual MAD
/// (floored against the median so a clean, outlier-free stack keeps
/// near-uniform weights), re-formed `iters` times.
fn irls_view_weights(xs: &[f32], views: usize, channels: usize, n: usize, iters: u32) -> Vec<f64> {
    let mut w = vec![1.0 / views as f64; views];
    let mut xbar = vec![0.0f64; channels * n];
    let mut resid = vec![0.0f64; views];

    for _ in 0..iters {
        // Weighted consensus per channel.
        for c in 0..channels {
            for k in 0..n {
                xbar[c * n + k] = (0..views)
                    .map(|v| w[v] * xs[(v * channels + c) * n + k] as f64)
                    .sum();
            }
        }
        // Per-view residual to the consensus.
        for v in 0..views {
            let mut r2 = 0.0;
            for c in 0..channels {
                for k in 0..n {
                    let d = xs[(v * channels + c) * n + k] as f64 - xbar[c * n + k];
                    r2 += d * d;
                }
            }
            resid[v] = r2.sqrt();
        }

        let med = median(&mut resid.clone());
        let mad = median(&mut resid.iter().map(|r| (r - med).abs()).collect::<Vec<_>>());
        let scale = (1.4826 * mad).max(0.5 * med).max(1e-12);
        let cutoff = 4.685 * scale;

        let wt: Vec<f64> = resid
            .iter()
            .map(|&r| {
                if r >= cutoff {
                    0.0
                } else {
                    let t = 1.0 - (r / cutoff) * (r / cutoff);
                    t * t
                }
            })
            .collect();
        let sum: f64 = wt.iter().sum();
        if sum <= 1e-12 {
            break; // Degenerate re-weight; keep the previous weights.
        }
        w = wt.iter().map(|&x| x / sum).collect();
    }
    w
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
) -> Option<f64> {
    if views < 2 {
        return None;
    }
    match objective {
        Objective::MeanPairwise => {
            let sum: f64 = (0..channels)
                .map(|c| mean_pairwise_channel(xs, views, channels, n, c))
                .sum();
            Some(sum / channels as f64)
        }
        Objective::RobustWeighted { iters } => {
            let w = irls_view_weights(xs, views, channels, n, iters);
            let sum_w2: f64 = w.iter().map(|&x| x * x).sum();
            // Degeneracy gate only (the view *count* is gated by `min_views` per
            // level): as weight concentrates on one view, Σwᵢ² → 1 and ρ̄_w → 0/0.
            if 1.0 / sum_w2 < MIN_EFFECTIVE_VIEWS || 1.0 - sum_w2 < 1e-9 {
                return None;
            }
            let mut sum = 0.0;
            for c in 0..channels {
                let mut norm_sq = 0.0;
                for k in 0..n {
                    let s: f64 = (0..views)
                        .map(|v| w[v] * xs[(v * channels + c) * n + k] as f64)
                        .sum();
                    norm_sq += s * s;
                }
                sum += (norm_sq - sum_w2) / (1.0 - sum_w2);
            }
            Some(sum / channels as f64)
        }
    }
}

/// Evaluate `Φ` for the candidate normal `n` over the frozen support `ctx`.
fn eval_phi(
    base: &OrientedPatch,
    n: &Vector3<f64>,
    ctx: &LevelContext,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
) -> Option<f64> {
    prof::count(&prof::N_EVAL, 1);
    let patch = repose_patch(base, n);
    let (raw, channels) = normalized_stack(&patch, ctx, views, resolution, params.sampler)?;
    let n = ctx.pixels.len();
    let total_w: f64 = ctx.weights.iter().sum();
    if total_w <= 0.0 {
        return None;
    }
    let sqrt_w: Vec<f64> = ctx.weights.iter().map(|&w| w.sqrt()).collect();
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
    prof::CONSENSUS.time(|| consensus_phi(&xs, ctx.kept.len(), kept, n, params.objective))
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

    // Fronto-parallel cache: render one supersampled base per view here, reused
    // across all levels. `None` (cache off, or the base could not be built for
    // this patch) means score every candidate from the source images instead.
    let cache = match params.cache {
        CacheMode::FrontoParallel => fronto_cache::prerender(
            base,
            &center,
            views,
            resolution,
            params.cache_supersample,
            params,
        ),
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
        let (sqrt_w, total_w): (Vec<f64>, f64) = if cache.is_some() {
            (
                ctx.weights.iter().map(|&w| w.sqrt()).collect(),
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
                    params,
                ),
                None => eval_phi(base, n, &ctx, views, resolution, params),
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
    };

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

    let phi_init = eval_phi(patch, &init_n, &ctx, views, resolution, params);
    let mut best_n = init_n;
    let mut best_phi = phi_init.unwrap_or(f64::NEG_INFINITY);
    let mut improved = false;
    for n in &survivors {
        if let Some(phi) = eval_phi(patch, n, &ctx, views, resolution, params) {
            if phi > best_phi {
                best_phi = phi;
                best_n = *n;
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
    }
}

/// Batch [`refine_patch_normal`] over a [`PatchCloud`], parallel across
/// patches (rayon). `patch_views[i]` lists, for patch `i`, the indices into
/// `views` of the cameras observing it (see
/// [`patch_view_indices_from_reconstruction`]). Each patch is replaced with
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
pub fn patch_view_indices_from_reconstruction(
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
