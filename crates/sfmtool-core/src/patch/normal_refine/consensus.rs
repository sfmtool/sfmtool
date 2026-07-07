// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Consensus photoconsistency `Φ`: the reusable [`ConsensusScratch`], the robust
//! IRLS view weights, the AVX2 residual / SAXPY kernels, and the
//! [`consensus_phi`] objective evaluations over a z-normalized stack.

use super::params::{Objective, MIN_EFFECTIVE_VIEWS};
use super::prof;

/// Reused buffers for the consensus / IRLS reductions, so scoring a candidate
/// allocates nothing after warm-up. The cache path holds one in its `Scratch`;
/// the source path makes a fresh (default-empty) one per call.
#[derive(Default)]
pub(in crate::patch) struct ConsensusScratch {
    /// View weights (`views`).
    pub(in crate::patch) w: Vec<f64>,
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
pub(super) fn mean_pairwise_channel(
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
pub(super) fn sum_sq_diff(row: &[f32], xbar: &[f32]) -> f32 {
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
pub(super) fn sum_sq_diff_scalar(row: &[f32], xbar: &[f32], i0: usize, i1: usize) -> f32 {
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
pub(super) fn axpy_f32(xb: &mut [f32], row: &[f32], w: f32) {
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
pub(super) fn axpy_f32_scalar(xb: &mut [f32], row: &[f32], w: f32, i0: usize, i1: usize) {
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
///
/// `view_priors` (length `views`, if given) is a multiplicative per-view prior on
/// the weights — the obliquity view-weight `|v̂·n|^power` (see
/// [`obliquity`](super::obliquity), use A). It seeds the initial weights and is
/// re-multiplied into every Tukey reweight, so an oblique view stays down-weighted
/// throughout. `None` is the plain uniform-init IRLS (byte-for-byte the prior-free
/// behavior).
pub(in crate::patch) fn irls_view_weights(
    xs: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    iters: u32,
    view_priors: Option<&[f64]>,
    sc: &mut ConsensusScratch,
) {
    sc.w.clear();
    match view_priors {
        Some(pr) => {
            debug_assert_eq!(pr.len(), views, "view_priors must be parallel to views");
            let sum: f64 = pr.iter().take(views).sum();
            if sum > 0.0 {
                sc.w.extend(pr.iter().take(views).map(|&p| p / sum));
            } else {
                sc.w.resize(views, 1.0 / views as f64);
            }
        }
        None => sc.w.resize(views, 1.0 / views as f64),
    }
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
            tukey_reweight_from_residuals(
                &sc.resid,
                view_priors,
                &mut sc.sorted,
                &mut sc.wt,
                &mut sc.w,
            )
        });
        if should_break {
            break;
        }
    }
}

/// One Tukey/MAD IRLS reweight step: from the per-view residuals, form the
/// MAD-scaled Tukey weights (times the optional multiplicative `view_priors`)
/// and write them, normalized to `Σw = 1`, into `w`. Returns `true` when the
/// re-weight is degenerate (`Σ` of the raw weights ≈ 0) — the caller should
/// keep the previous weights and stop iterating. `sorted` / `wt` are reused
/// scratch. Extracted from [`irls_view_weights`] so the keypoint localizer's
/// Gram-space leave-one-out IRLS shares the exact reweight semantics.
pub(in crate::patch) fn tukey_reweight_from_residuals(
    resid: &[f64],
    view_priors: Option<&[f64]>,
    sorted: &mut Vec<f64>,
    wt: &mut Vec<f64>,
    w: &mut [f64],
) -> bool {
    let views = resid.len();
    sorted.clear();
    sorted.extend_from_slice(resid);
    let med = median(sorted);
    for (s, &r) in sorted.iter_mut().zip(resid) {
        *s = (r - med).abs();
    }
    let mad = median(sorted);
    let scale = (1.4826 * mad).max(0.5 * med).max(1e-12);
    let cutoff = 4.685 * scale;

    wt.clear();
    wt.resize(views, 0.0);
    let mut sum = 0.0;
    for v in 0..views {
        let r = resid[v];
        let tukey = if r >= cutoff {
            0.0
        } else {
            let t = 1.0 - (r / cutoff) * (r / cutoff);
            t * t
        };
        // Fold the obliquity prior back in each iteration so the
        // down-weighting persists (the Tukey factor alone would let a
        // grazing but self-consistent view regain full weight).
        let wv = tukey * view_priors.map_or(1.0, |pr| pr[v]);
        wt[v] = wv;
        sum += wv;
    }
    if sum <= 1e-12 {
        return true; // Degenerate re-weight; keep the previous weights.
    }
    for v in 0..views {
        w[v] = wt[v] / sum;
    }
    false
}

/// Build the unit-norm-per-channel template of a z-normalized stack `xs`
/// (`xs[(v*channels + c)*n + k]`) weighted by `weights` into `out` (resized and
/// overwritten). The result is directly dot-able against another z-normalized core
/// to yield a per-channel ZNCC. `out` is a reused scratch buffer, mirroring the
/// scratch-reuse discipline of [`ConsensusScratch`]. The natural follow-on to
/// [`irls_view_weights`] (which fills the per-view weights this consumes).
pub(in crate::patch) fn weighted_unit_template_into(
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

/// [`weighted_unit_template_into`] over a **leave-one-out** subset of the
/// stack: sum every view row except `skip`, weighted by the full-stack-indexed
/// `weights` (the skipped view's entry is ignored), then unit-normalize per
/// channel. Iteration order matches copying the hold-out rows into a compacted
/// stack and calling [`weighted_unit_template_into`] on it, so the result is
/// identical — without materializing the hold-out copy.
#[allow(clippy::too_many_arguments)]
pub(in crate::patch) fn weighted_unit_template_skip_into(
    xs: &[f32],
    weights: &[f64],
    skip: usize,
    views: usize,
    channels: usize,
    n: usize,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(channels * n, 0.0);
    for (v, &w) in weights.iter().enumerate().take(views) {
        if v == skip {
            continue;
        }
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

/// Consensus photoconsistency `Φ` over the normalized stack, per the
/// objective. `None` when fewer than 2 views, or when the robust effective
/// view count `1/Σwᵢ²` drops below [`MIN_EFFECTIVE_VIEWS`] (weights collapsed
/// onto essentially one view).
pub(super) fn consensus_phi(
    xs: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    objective: Objective,
    view_priors: Option<&[f64]>,
    sc: &mut ConsensusScratch,
) -> Option<f64> {
    consensus_phi_with_weights(xs, views, channels, n, objective, view_priors, sc)
        .map(|(phi, _)| phi)
}

/// [`consensus_phi`] plus the per-view consensus weights that produced it,
/// normalized to `Σwᵢ = 1` (uniform `1/views` under [`Objective::MeanPairwise`],
/// the IRLS weights under [`Objective::RobustWeighted`]). Callers that re-use the
/// weights — the representative fusion, which down-weights the same outlier views
/// in the blended colour as the consensus does in `Φ` — take this variant so the
/// IRLS pass is not run twice.
///
/// `view_priors` (if given, length `views`) is the multiplicative obliquity prior
/// on the robust weights (use A); it is applied only under
/// [`Objective::RobustWeighted`] — [`Objective::MeanPairwise`] is unweighted by
/// definition and ignores it.
pub(super) fn consensus_phi_with_weights(
    xs: &[f32],
    views: usize,
    channels: usize,
    n: usize,
    objective: Objective,
    view_priors: Option<&[f64]>,
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
            irls_view_weights(xs, views, channels, n, iters, view_priors, sc);
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
