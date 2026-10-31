// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Robust consensus weights of a z-normalized stack: the reusable
//! [`ConsensusScratch`], the IRLS view weights ([`irls_view_weights`]) and the
//! AVX2 residual / SAXPY kernels they run on.

use crate::numeric::median_in_place;

/// Reused buffers for the IRLS reductions, so repeated weighting allocates
/// nothing after warm-up.
#[derive(Default)]
pub(in crate::patch) struct ConsensusScratch {
    /// View weights (`views`).
    pub(in crate::patch) w: Vec<f64>,
    /// Weighted per-(channel, pixel) consensus (`channels·n`). f32: it is only
    /// consumed by the IRLS residual that feeds the robust reweight, so the
    /// accumulation precision is immaterial, and f32 keeps both the SAXPY and the
    /// residual a single 8-wide AVX2 path with no f64↔f32 narrowing.
    xbar: Vec<f32>,
    /// Per-view residual to the consensus (`views`).
    resid: Vec<f64>,
    /// Scratch for the median/MAD sort (`views`), avoids per-iter clones.
    sorted: Vec<f64>,
    /// New (Tukey) weights before normalization (`views`).
    wt: Vec<f64>,
}

/// `Σ_k (row[k] − xbar[k])²`, all f32 — the IRLS per-view residual sum. An f64
/// `.sum()` cannot vectorize (non-associative → a serial dependency chain), and
/// the crate builds for a baseline target (no `-C target-feature=+avx2`), so the
/// scalar fallback only reaches SSE; the dispatched AVX2+FMA kernel below is
/// 8-wide. With `xbar` also f32 there is no per-element narrowing. The residual
/// only feeds the Tukey reweight (median/MAD/cutoff), so f32 precision is ample
/// and the found weights are unaffected in practice. Mirrors the
/// runtime-dispatch pattern of `features::sift::simd`.
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
/// weights), re-formed `iters` times from uniform initial weights. All buffers
/// live in `sc` (no per-call / per-iteration allocation); the weighted consensus
/// `xbar` is accumulated by SAXPY over contiguous per-view rows.
pub(in crate::patch) fn irls_view_weights(
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
        sc.xbar.iter_mut().for_each(|x| *x = 0.0);
        for v in 0..views {
            let wv = sc.w[v] as f32;
            for c in 0..channels {
                let row = &xs[(v * channels + c) * n..][..n];
                let xb = &mut sc.xbar[c * n..][..n];
                axpy_f32(xb, row, wv);
            }
        }
        // Per-view residual to the consensus (contiguous per row), all f32: only
        // the Tukey reweight consumes it, so the precision is immaterial, and the
        // dispatched AVX2 reduction needs no narrowing now that xbar is f32 too.
        for v in 0..views {
            let mut r2 = 0f32;
            for c in 0..channels {
                let row = &xs[(v * channels + c) * n..][..n];
                let xb = &sc.xbar[c * n..][..n];
                r2 += sum_sq_diff(row, xb);
            }
            sc.resid[v] = (r2 as f64).sqrt();
        }

        let degenerate =
            tukey_reweight_from_residuals(&sc.resid, &mut sc.sorted, &mut sc.wt, &mut sc.w);
        if degenerate {
            break;
        }
    }
}

/// One Tukey/MAD IRLS reweight step: from the per-view residuals, form the
/// MAD-scaled Tukey weights and write them, normalized to `Σw = 1`, into `w`.
/// Returns `true` when the re-weight is degenerate (`Σ` of the raw weights ≈
/// 0) — the caller should keep the previous weights and stop iterating.
/// `sorted` / `wt` are reused scratch. The reweight step of
/// [`irls_view_weights`].
fn tukey_reweight_from_residuals(
    resid: &[f64],
    sorted: &mut Vec<f64>,
    wt: &mut Vec<f64>,
    w: &mut [f64],
) -> bool {
    let views = resid.len();
    sorted.clear();
    sorted.extend_from_slice(resid);
    let med = median_in_place(sorted);
    for (s, &r) in sorted.iter_mut().zip(resid) {
        *s = (r - med).abs();
    }
    let mad = median_in_place(sorted);
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
        wt[v] = tukey;
        sum += tukey;
    }
    if sum <= 1e-12 {
        return true; // Degenerate re-weight; keep the previous weights.
    }
    for v in 0..views {
        w[v] = wt[v] / sum;
    }
    false
}
