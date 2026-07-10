// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Render-and-z-normalize numerics: project each kept view's patch into the
//! frozen support ([`normalized_stack`]), z-normalize the raw stack
//! ([`znormalize_into`] / [`znormalize_into_kept`]), and the AVX2 moment /
//! write kernels they dispatch to.

use crate::camera::remap::{remap_aniso_with_pyramid, remap_bilinear};
use crate::camera::WarpMap;
use crate::patch::cloud::OrientedPatch;

use super::level::LevelContext;
use super::params::{ProjectedImage, Sampler, FLAT_NORM_SQ_EPS, MAX_ANISOTROPY};
use super::prof;
use super::support::view_render_patch;

/// Render the kept views of `patch` and z-normalize each colour channel over
/// the windowed common support. Returns `xs[view][channel][pixel]`, each
/// channel vector unit-norm and zero weighted-mean (window weights are folded
/// in as `√w`, so plain dot products realize the windowed inner product).
/// Channels that are flat (windowed norm ≈ 0) in *any* kept view are dropped
/// for every view, keeping all inner products in one space; `None` when no
/// channel survives.
pub(in crate::patch) fn normalized_stack(
    patch: &OrientedPatch,
    ctx: &LevelContext,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    sampler: Sampler,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
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
        let rpatch = view_render_patch(patch, view, view_keypoints.and_then(|k| k[vi]));
        let mut map = prof::WARP
            .time(|| WarpMap::from_patch(&rpatch, view.camera, view.cam_from_world, resolution));
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
    total_weight: f64,
    sqrt_weights: &[f32],
    out: &mut Vec<f32>,
) -> Option<usize> {
    znormalize_into_kept(
        raw,
        views,
        channels,
        n,
        weights,
        total_weight,
        sqrt_weights,
        out,
    )
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
pub(in crate::patch) fn znormalize_into_kept(
    raw: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    weights: &[f64],
    total_weight: f64,
    sqrt_weights: &[f32],
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
            let mean = s1 / total_weight;
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
            znorm_write(src, sqrt_weights, mean, inv_norm, &mut out[base..base + n]);
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
pub(in crate::patch) fn weighted_moments_pub(col: &[f32], w: &[f64]) -> (f64, f64) {
    weighted_moments(col, w)
}

#[inline]
pub(super) fn weighted_moments(col: &[f32], w: &[f64]) -> (f64, f64) {
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
pub(super) fn weighted_moments_scalar(col: &[f32], w: &[f64], i0: usize, i1: usize) -> (f64, f64) {
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

/// `out[k] = sqrt_weights[k] · (src[k] − mean) · inv_norm`, all f32 — the per-pixel
/// z-normalize write. AVX2+FMA (8-wide) dispatched at runtime; same pattern as
/// [`sum_sq_diff`]. `pub(in crate::patch)` so cluster refinement's template
/// build z-normalizes with the exact same kernel.
#[inline]
pub(in crate::patch) fn znorm_write(
    src: &[f32],
    sqrt_weights: &[f32],
    mean: f32,
    inv_norm: f32,
    out: &mut [f32],
) {
    let n = src.len().min(sqrt_weights.len()).min(out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check above.
            unsafe {
                znorm_write_avx2(src, sqrt_weights, mean, inv_norm, out, n);
            }
            return;
        }
    }
    znorm_write_scalar(src, sqrt_weights, mean, inv_norm, out, 0, n);
}

/// Scalar reference for [`znorm_write`] over `[i0, i1)` (also the AVX2 tail).
pub(super) fn znorm_write_scalar(
    src: &[f32],
    sqrt_weights: &[f32],
    mean: f32,
    inv_norm: f32,
    out: &mut [f32],
    i0: usize,
    i1: usize,
) {
    for k in i0..i1 {
        out[k] = sqrt_weights[k] * (src[k] - mean) * inv_norm;
    }
}

/// AVX2+FMA `out[k] = sqrt_weights[k]·(src[k] − mean)·inv_norm` over the first `n`
/// elements (8 f32/iteration). The `n % 8` tail falls back to
/// [`znorm_write_scalar`].
///
/// # Safety
/// Requires the `avx2` + `fma` target features (guarded by the dispatcher).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn znorm_write_avx2(
    src: &[f32],
    sqrt_weights: &[f32],
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
        let sw = _mm256_loadu_ps(sqrt_weights.as_ptr().add(i));
        let d = _mm256_sub_ps(x, m);
        let scale = _mm256_mul_ps(sw, s);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(scale, d));
        i += 8;
    }
    znorm_write_scalar(src, sqrt_weights, mean, inv_norm, out, i, n);
}
